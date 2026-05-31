#!/usr/bin/env python3
"""
Phase 1: DREAMPlace global placement, all 17 IBM ICCAD04 benchmarks.

Per benchmark:
  - 5-seed DREAMPlace sweep (legalize_flag=0, detailed_place_flag=0)
  - Legalization via legalize_overlaps_strict post-hoc
  - Save/reload drift check
  - Fallback to legalized initial.plc if all DP runs fail
  - Keep best proxy per bench

Coverage:
  ibm01-13 use DEFAULT_KNOBS (target_density=0.7, 256x256 bins, ~30 sec each)
  ibm14-18 use tuned knobs   (target_density=0.5, 512/1024 bins, ~5-15 min each)
"""

import sys, os, json, time, subprocess
from pathlib import Path

REPO = Path('/home/ubuntu/macro-place-challenge-2026')
sys.path.insert(0, str(REPO))
SUB = REPO / 'submissions' / 'aarushi'
sys.path.insert(0, str(SUB))

import torch
from macro_place.objective  import compute_proxy_cost
from macro_place.loader     import load_benchmark_from_dir
from dp_converter import (write_bookshelf, write_dreamplace_config,
                           read_dreamplace_pl, legalize_overlaps_strict)

# ── Paths ─────────────────────────────────────────────────────────────────────

BENCH_ROOT = REPO / 'external' / 'MacroPlacement' / 'Testcases' / 'ICCAD04'
WORK_DIR   = Path('/tmp/dp_work')
OUT_DIR    = Path('/tmp/dp_overnight')

DREAMPLACE_PLACER = os.environ.get(
    'DREAMPLACE_PLACER',
    '/opt/DREAMPlace/install/dreamplace/Placer.py'
)

# ── Config ────────────────────────────────────────────────────────────────────

BENCHES = [
    'ibm01','ibm02','ibm03','ibm04','ibm06','ibm07','ibm08',
    'ibm09','ibm10','ibm11','ibm12','ibm13','ibm14','ibm15',
    'ibm16','ibm17','ibm18',
]

SEEDS = [1, 7, 42, 1234, 9999]

DEFAULT_KNOBS = dict(
    target_density=0.7,
    density_weight=8e-5,
    gamma=4.0,
    num_bins_x=256,
    num_bins_y=256,
)

DP_KNOBS = {
    'ibm14': dict(target_density=0.5, density_weight=8e-5, gamma=4.0,
                  num_bins_x=512, num_bins_y=512),
    'ibm15': dict(target_density=0.5, density_weight=8e-5, gamma=4.0,
                  num_bins_x=512, num_bins_y=512),
    'ibm16': dict(target_density=0.5, density_weight=8e-5, gamma=4.0,
                  num_bins_x=512, num_bins_y=512),
    'ibm17': dict(target_density=0.5, density_weight=8e-5, gamma=4.0,
                  num_bins_x=1024, num_bins_y=1024),
    'ibm18': dict(target_density=0.5, density_weight=8e-5, gamma=4.0,
                  num_bins_x=1024, num_bins_y=1024),
}

WORK_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Per-seed DREAMPlace run ───────────────────────────────────────────────────

def run_dp(bench_name: str, seed: int, bench, plc):
    """
    Run DREAMPlace for one (bench, seed).
    Returns (save_path, proxy_cost) or None on failure.
    """
    trial_dir = WORK_DIR / bench_name / f'seed{seed}'
    trial_dir.mkdir(parents=True, exist_ok=True)
    save_path = OUT_DIR / f'{bench_name}_s{seed}.pt'

    aux  = write_bookshelf(bench, trial_dir / 'bench', bench_name=bench_name)
    knobs = dict(DP_KNOBS.get(bench_name, DEFAULT_KNOBS))
    knobs['iteration'] = 1000
    cfg  = write_dreamplace_config(
        aux, trial_dir / 'results',
        gpu=True, extra={'random_seed': seed}, **knobs,
    )

    result = subprocess.run(
        ['python3', DREAMPLACE_PLACER, str(cfg)],
        capture_output=True, text=True, timeout=1800,
    )
    if result.returncode != 0:
        print(f'    DP failed seed={seed}: {result.stderr[-300:].strip()}')
        return None

    pl_files = sorted((trial_dir / 'results').glob('**/*.pl'))
    if not pl_files:
        print(f'    no .pl output for {bench_name} seed={seed}')
        return None

    placement = read_dreamplace_pl(pl_files[-1], bench)
    placement[bench.macro_fixed] = bench.macro_positions[bench.macro_fixed]

    legal, n_ov = legalize_overlaps_strict(placement, bench, max_iters=100)
    costs  = compute_proxy_cost(legal, bench, plc)
    proxy  = float(costs['proxy_cost'])

    torch.save({'positions': legal, 'score': proxy, 'costs': dict(costs),
                'bench': bench_name, 'seed': seed},
               str(save_path))

    # Save/reload drift check
    reloaded     = torch.load(str(save_path), map_location='cpu', weights_only=False)
    reload_costs = compute_proxy_cost(reloaded['positions'], bench, plc)
    reload_proxy = float(reload_costs['proxy_cost'])
    if abs(reload_proxy - proxy) > 1e-3:
        print(f'    drift {bench_name} s{seed}: {proxy:.4f} vs {reload_proxy:.4f}')
        return None

    return save_path, proxy


# ── Per-benchmark sweep ───────────────────────────────────────────────────────

def run_benchmark(bench_name: str):
    print(f"\n{'='*60}\n  {bench_name}\n{'='*60}")

    bench, plc = load_benchmark_from_dir(str(BENCH_ROOT / bench_name))

    # Fallback: legalized initial.plc
    legal_init, _ = legalize_overlaps_strict(bench.macro_positions.clone(), bench)
    init_costs    = compute_proxy_cost(legal_init, bench, plc)
    init_proxy    = float(init_costs['proxy_cost'])
    candidates    = [(init_proxy, legal_init, 'init')]
    print(f'  initial: proxy={init_proxy:.4f}  overlaps={init_costs["overlap_count"]}')

    for seed in SEEDS:
        t0 = time.time()
        print(f'  seed={seed} ... ', end='', flush=True)
        try:
            res = run_dp(bench_name, seed, bench, plc)
            if res:
                save_path, proxy = res
                data = torch.load(str(save_path), weights_only=False)
                candidates.append((proxy, data['positions'], f'seed{seed}'))
                print(f'proxy={proxy:.4f}  [{time.time()-t0:.1f}s]')
            else:
                print(f'failed  [{time.time()-t0:.1f}s]')
        except Exception as e:
            print(f'error: {e}  [{time.time()-t0:.1f}s]')

    candidates.sort(key=lambda x: x[0])
    best_proxy, best_pos, best_trial = candidates[0]

    final_costs = compute_proxy_cost(best_pos, bench, plc)
    torch.save({'positions': best_pos, 'score': best_proxy,
                'costs': dict(final_costs), 'bench': bench_name,
                'trial': best_trial},
               str(OUT_DIR / f'{bench_name}_best.pt'))

    print(f'  → BEST  proxy={best_proxy:.4f}  trial={best_trial}')
    return {
        'bench':           bench_name,
        'trial':           best_trial,
        'proxy_cost':      best_proxy,
        'wirelength_cost': float(final_costs['wirelength_cost']),
        'density_cost':    float(final_costs['density_cost']),
        'congestion_cost': float(final_costs['congestion_cost']),
        'overlap_count':   int(final_costs['overlap_count']),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    best_per_bench = {}
    t0 = time.time()

    for name in BENCHES:
        try:
            best_per_bench[name] = run_benchmark(name)
        except Exception as e:
            print(f'  ERROR {name}: {e}')
            import traceback; traceback.print_exc()

    out = OUT_DIR / 'best_per_bench.json'
    with open(str(out), 'w') as f:
        json.dump(best_per_bench, f, indent=2)

    scores  = [v['proxy_cost'] for v in best_per_bench.values()]
    avg     = sum(scores) / len(scores) if scores else float('nan')
    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f'  DONE  avg proxy={avg:.4f}  total={elapsed:.1f} min')
    print(f'  results → {out}')
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
