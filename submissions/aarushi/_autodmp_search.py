#!/usr/bin/env python3
"""
AutoDMP: Optuna TPE search over DREAMPlace knobs.

Per benchmark:
  - 50-trial TPE search over 8 DREAMPlace knobs
  - First trial warm-starts from Phase 1 best knobs
  - Resumable via SQLite study DB
  - Saves best placement to cache_autodmp/<bench>.pt

Usage:
    python3 _autodmp_search.py --benches all --trials 30 \
        --study-out submissions/aarushi/autodmp_study.db

    python3 _autodmp_search.py --benches ibm01,ibm06 --trials 20 \
        --study-out submissions/aarushi/autodmp_study.db
"""

import sys, os, json, time, subprocess, argparse
from pathlib import Path

REPO = Path('/home/ubuntu/macro-place-challenge-2026')
sys.path.insert(0, str(REPO))
SUB = REPO / 'submissions' / 'aarushi'
sys.path.insert(0, str(SUB))

import torch
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from macro_place.objective import compute_proxy_cost
from macro_place.loader    import load_benchmark_from_dir
from dp_converter import (write_bookshelf, write_dreamplace_config,
                           read_dreamplace_pl, legalize_overlaps_strict)

# ── Paths ──────────────────────────────────────────────────────────────────────

BENCH_ROOT = REPO / 'external' / 'MacroPlacement' / 'Testcases' / 'ICCAD04'
WORK_DIR   = Path('/tmp/autodmp_work')
OUT_DIR    = SUB / 'cache_autodmp'

DREAMPLACE_PLACER = os.environ.get(
    'DREAMPLACE_PLACER',
    '/opt/DREAMPlace/install/dreamplace/Placer.py'
)

ALL_BENCHES = [
    'ibm01','ibm02','ibm03','ibm04','ibm06','ibm07','ibm08',
    'ibm09','ibm10','ibm11','ibm12','ibm13','ibm14','ibm15',
    'ibm16','ibm17','ibm18',
]

BIG_BENCHES = {'ibm14','ibm15','ibm16','ibm17','ibm18'}

# Phase 1 best knobs — warm-start first trial
WARM_KNOBS = {
    'target_density':                   0.7,
    'learning_rate':                    0.01,
    'density_weight':                   8e-5,
    'gamma':                            4.0,
    'num_bins':                         256,
    'iteration':                        1000,
    'stop_overflow':                    0.07,
    'Llambda_density_weight_iteration': 1,
    'macro_halo_frac':                  0.01,
    'routability_opt':                  0,
}
WARM_KNOBS_BIG = dict(WARM_KNOBS, target_density=0.5, num_bins=512)


# ── Single DREAMPlace trial ────────────────────────────────────────────────────

def run_trial(bench_name: str, trial_id: int, knobs: dict,
              bench, plc) -> float:
    """Run one DREAMPlace trial; return proxy cost (inf on failure)."""
    trial_dir = WORK_DIR / bench_name / f't{trial_id:04d}'
    trial_dir.mkdir(parents=True, exist_ok=True)

    nb = knobs['num_bins']
    # Compute macro halo from canvas size
    from macro_place.benchmark import Benchmark as _B
    _bm = _B.load(str(REPO / 'benchmarks' / 'processed' / 'public' / f'{bench_name}.pt'))
    macro_halo = knobs.get('macro_halo_frac', 0.01) * max(
        float(_bm.canvas_width), float(_bm.canvas_height))

    aux = write_bookshelf(bench, trial_dir / 'bench', bench_name=bench_name)
    cfg = write_dreamplace_config(
        aux, trial_dir / 'results',
        gpu=True,
        extra={
            'random_seed':     42,
            'macro_halo_x':    macro_halo,
            'macro_halo_y':    macro_halo,
            'routability_opt': knobs.get('routability_opt', 0),
        },
        target_density=knobs['target_density'],
        density_weight=knobs['density_weight'],
        gamma=knobs['gamma'],
        num_bins_x=nb,
        num_bins_y=nb,
        iteration=knobs['iteration'],
        learning_rate=knobs['learning_rate'],
        legalize_flag=0,
        detailed_place_flag=0,
    )

    # Patch stop_overflow and Llambda into the config JSON
    with open(cfg) as f:
        cfg_dict = json.load(f)
    stage = cfg_dict['global_place_stages'][0]
    stage['stop_overflow']                    = knobs['stop_overflow']
    stage['Llambda_density_weight_iteration'] = knobs['Llambda_density_weight_iteration']
    with open(cfg, 'w') as f:
        json.dump(cfg_dict, f, indent=2)

    result = subprocess.run(
        ['python3', DREAMPLACE_PLACER, str(cfg)],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        return float('inf')

    pl_files = sorted((trial_dir / 'results').glob('**/*.pl'))
    if not pl_files:
        return float('inf')

    placement = read_dreamplace_pl(pl_files[-1], bench)
    placement[bench.macro_fixed] = bench.macro_positions[bench.macro_fixed]
    legal, _ = legalize_overlaps_strict(placement, bench, max_iters=100)
    costs = compute_proxy_cost(legal, bench, plc)
    return float(costs['proxy_cost']), legal, costs


# ── Optuna objective ───────────────────────────────────────────────────────────

def make_objective(bench_name: str, bench, plc, best_holder: dict):
    is_big = bench_name in BIG_BENCHES

    def objective(trial: optuna.Trial) -> float:
        nb_choices = [512, 1024] if bench_name in {'ibm17','ibm18'} else \
                     [512]       if is_big else \
                     [128, 256, 512]

        knobs = {
            'target_density':   trial.suggest_float('target_density', 0.35, 0.85),
            'learning_rate':    trial.suggest_float('learning_rate',  1e-3, 0.05, log=True),
            'density_weight':   trial.suggest_float('density_weight', 1e-5, 1e-2, log=True),
            'gamma':            trial.suggest_float('gamma',          1.0,  8.0),
            'num_bins':         trial.suggest_categorical('num_bins', nb_choices),
            'iteration':        trial.suggest_int('iteration',        500,  1500, step=100),
            'stop_overflow':    trial.suggest_float('stop_overflow',  0.04, 0.15),
            'Llambda_density_weight_iteration':
                                trial.suggest_int('Llambda_density_weight_iteration', 1, 5),
            'macro_halo_frac':  trial.suggest_float('macro_halo_frac', 0.001, 0.05, log=True),
            'routability_opt':  trial.suggest_categorical('routability_opt', [0, 1]),
        }

        t0 = time.time()
        result = run_trial(bench_name, trial.number, knobs, bench, plc)
        elapsed = time.time() - t0

        if isinstance(result, tuple):
            proxy, positions, costs = result
            print(f'    trial={trial.number:3d}  proxy={proxy:.4f}  '
                  f'density={knobs["target_density"]:.2f}  '
                  f'bins={knobs["num_bins"]}  [{elapsed:.0f}s]')
            if proxy < best_holder['proxy']:
                best_holder['proxy']     = proxy
                best_holder['positions'] = positions
                best_holder['costs']     = costs
                best_holder['knobs']     = knobs
            return proxy
        else:
            print(f'    trial={trial.number:3d}  FAILED  [{elapsed:.0f}s]')
            return float('inf')

    return objective


# ── Per-benchmark search ───────────────────────────────────────────────────────

def search_benchmark(bench_name: str, n_trials: int, study_url: str):
    print(f"\n{'='*60}\n  {bench_name}  ({n_trials} trials)\n{'='*60}")

    bench, plc = load_benchmark_from_dir(str(BENCH_ROOT / bench_name))

    best_holder = {'proxy': float('inf'), 'positions': None,
                   'costs': None, 'knobs': None}

    study = optuna.create_study(
        study_name=bench_name,
        storage=study_url,
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )

    # Warm-start: enqueue Phase 1 best knobs as trial 0 (skip if already done)
    if len(study.trials) == 0:
        warm = WARM_KNOBS_BIG if bench_name in BIG_BENCHES else WARM_KNOBS
        nb_choices = [512, 1024] if bench_name in {'ibm17','ibm18'} else \
                     [512]       if bench_name in BIG_BENCHES else \
                     [128, 256, 512]
        nb = min(warm['num_bins'], max(nb_choices))
        study.enqueue_trial({
            'target_density':   warm['target_density'],
            'learning_rate':    warm['learning_rate'],
            'density_weight':   warm['density_weight'],
            'gamma':            warm['gamma'],
            'num_bins':         nb,
            'iteration':        warm['iteration'],
            'stop_overflow':    warm['stop_overflow'],
            'Llambda_density_weight_iteration': warm['Llambda_density_weight_iteration'],
        })

    remaining = n_trials - len([t for t in study.trials
                                 if t.state == optuna.trial.TrialState.COMPLETE])
    if remaining <= 0:
        print(f'  already have {n_trials} complete trials, skipping')
    else:
        print(f'  resuming: {len(study.trials)} done, {remaining} remaining')
        study.optimize(
            make_objective(bench_name, bench, plc, best_holder),
            n_trials=remaining,
            catch=(Exception,),
        )

    # Pick overall best from study if we didn't capture it in this run
    if best_holder['positions'] is None:
        best_trial = study.best_trial
        # Re-run the best knobs to get positions
        print(f'  re-running best trial (value={best_trial.value:.4f}) to get positions')
        result = run_trial(bench_name, 9999, best_trial.params, bench, plc)
        if isinstance(result, tuple):
            best_holder['proxy'], best_holder['positions'], best_holder['costs'] = result
            best_holder['knobs'] = best_trial.params

    if best_holder['positions'] is None:
        print(f'  !! no valid placement found for {bench_name}')
        return None

    # Save best
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'{bench_name}.pt'
    torch.save({
        'positions': best_holder['positions'],
        'score':     best_holder['proxy'],
        'costs':     best_holder['costs'],
        'knobs':     best_holder['knobs'],
        'bench':     bench_name,
    }, str(out_path))
    print(f'  → BEST  proxy={best_holder["proxy"]:.4f}  saved → {out_path}')
    return best_holder['proxy']


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benches',   default='all',
                        help='comma-separated list or "all"')
    parser.add_argument('--trials',    type=int, default=30)
    parser.add_argument('--study-out', default=str(SUB / 'autodmp_study.db'))
    args = parser.parse_args()

    benches = ALL_BENCHES if args.benches == 'all' \
              else [b.strip() for b in args.benches.split(',')]

    study_url = f'sqlite:///{args.study_out}'
    print(f'Study DB: {study_url}')
    print(f'Benches:  {benches}')
    print(f'Trials:   {args.trials} per bench')

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    t0 = time.time()

    for bench_name in benches:
        try:
            proxy = search_benchmark(bench_name, args.trials, study_url)
            if proxy is not None:
                results[bench_name] = proxy
        except Exception as e:
            print(f'  ERROR {bench_name}: {e}')
            import traceback; traceback.print_exc()

    scores  = list(results.values())
    avg     = sum(scores) / len(scores) if scores else float('nan')
    elapsed = (time.time() - t0) / 3600
    print(f"\n{'='*60}")
    print(f'  DONE  avg proxy={avg:.4f}  ({len(scores)}/{len(benches)} benches)')
    print(f'  Elapsed: {elapsed:.1f} h')
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
