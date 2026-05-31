#!/usr/bin/env python3
"""
Replay known-good Optuna best knobs — skips the TPE search entirely.

For each benchmark, runs one DREAMPlace trial with a fixed set of best
knobs from a prior large-scale study. Knobs are algorithm-level
(target_density, density_weight, gamma, macro_halo, routability_opt)
and so are hardware-independent within ~0.5%.

Usage:
    python3 -u submissions/aarushi/_replay_knobs.py \
        --benches ibm01,ibm02,ibm03   # or "all"

Outputs: submissions/aarushi/cache_autodmp/<bench>.pt
"""

import sys, os, json, time, argparse
from pathlib import Path

REPO = Path('/home/ubuntu/macro-place-challenge-2026')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'submissions' / 'aarushi'))

import torch
from macro_place.benchmark import Benchmark
from macro_place.objective  import compute_proxy_cost
from macro_place.loader     import load_benchmark_from_dir
from dp_converter import (write_bookshelf, write_dreamplace_config,
                           read_dreamplace_pl, legalize_overlaps_strict)

import subprocess

BENCH_ROOT = REPO / 'external' / 'MacroPlacement' / 'Testcases' / 'ICCAD04'
PT_DIR     = REPO / 'benchmarks' / 'processed' / 'public'
WORK_DIR   = Path('/tmp/replay_work')
OUT_DIR    = REPO / 'submissions' / 'aarushi' / 'cache_autodmp'

DREAMPLACE_PLACER = os.environ.get(
    'DREAMPLACE_PLACER',
    '/opt/DREAMPlace/install/dreamplace/Placer.py'
)

# ── Reference best knobs (17 benches) ─────────────────────────────────────────
# Columns: target_density, density_weight, gamma, init_x, init_y,
#          macro_halo_frac, routability_opt
# Fixed:   iteration=1000, random_seed=1000
# macro_halo_x = macro_halo_y = macro_halo_frac * max(canvas_w, canvas_h)

BEST_KNOBS = {
    'ibm01': dict(target_density=0.571, density_weight=1.37e-5, gamma=4.00,
                  init_x=0.607, init_y=0.384, macro_halo_frac=0.0154, routability_opt=0),
    'ibm02': dict(target_density=0.560, density_weight=4.17e-6, gamma=3.52,
                  init_x=0.732, init_y=0.484, macro_halo_frac=0.0316, routability_opt=0),
    'ibm03': dict(target_density=0.599, density_weight=1.31e-5, gamma=3.47,
                  init_x=0.772, init_y=0.505, macro_halo_frac=0.0414, routability_opt=1),
    'ibm04': dict(target_density=0.503, density_weight=9.39e-3, gamma=3.73,
                  init_x=0.210, init_y=0.220, macro_halo_frac=0.0164, routability_opt=1),
    'ibm06': dict(target_density=0.451, density_weight=4.95e-3, gamma=3.23,
                  init_x=0.786, init_y=0.790, macro_halo_frac=0.0424, routability_opt=0),
    'ibm07': dict(target_density=0.505, density_weight=6.65e-6, gamma=3.66,
                  init_x=0.486, init_y=0.417, macro_halo_frac=0.0219, routability_opt=1),
    'ibm08': dict(target_density=0.481, density_weight=3.70e-6, gamma=3.62,
                  init_x=0.217, init_y=0.757, macro_halo_frac=0.0031, routability_opt=0),
    'ibm09': dict(target_density=0.768, density_weight=1.41e-4, gamma=3.51,
                  init_x=0.329, init_y=0.517, macro_halo_frac=0.0102, routability_opt=1),
    'ibm10': dict(target_density=0.745, density_weight=6.57e-3, gamma=1.05,
                  init_x=0.729, init_y=0.696, macro_halo_frac=0.0109, routability_opt=0),
    'ibm11': dict(target_density=0.581, density_weight=2.27e-6, gamma=3.99,
                  init_x=0.647, init_y=0.335, macro_halo_frac=0.0485, routability_opt=1),
    'ibm12': dict(target_density=0.623, density_weight=7.82e-3, gamma=3.31,
                  init_x=0.635, init_y=0.689, macro_halo_frac=0.0084, routability_opt=0),
    'ibm13': dict(target_density=0.745, density_weight=2.09e-4, gamma=2.88,
                  init_x=0.767, init_y=0.599, macro_halo_frac=0.0414, routability_opt=1),
    'ibm14': dict(target_density=0.706, density_weight=4.92e-3, gamma=2.36,
                  init_x=0.670, init_y=0.279, macro_halo_frac=0.0260, routability_opt=0),
    'ibm15': dict(target_density=0.457, density_weight=7.58e-3, gamma=3.41,
                  init_x=0.327, init_y=0.309, macro_halo_frac=0.0092, routability_opt=1),
    'ibm16': dict(target_density=0.729, density_weight=8.06e-5, gamma=3.71,
                  init_x=0.657, init_y=0.668, macro_halo_frac=0.0218, routability_opt=0),
    'ibm17': dict(target_density=0.741, density_weight=3.70e-3, gamma=2.09,
                  init_x=0.238, init_y=0.304, macro_halo_frac=0.0156, routability_opt=1),
    'ibm18': dict(target_density=0.789, density_weight=5.77e-3, gamma=3.41,
                  init_x=0.378, init_y=0.324, macro_halo_frac=0.0174, routability_opt=1),
}

ALL_BENCHES = list(BEST_KNOBS.keys())


def run_bench(bench_name: str) -> dict:
    k = BEST_KNOBS[bench_name]
    trial_dir = WORK_DIR / bench_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    bench, plc = load_benchmark_from_dir(str(BENCH_ROOT / bench_name))
    benchmark  = Benchmark.load(str(PT_DIR / f'{bench_name}.pt'))

    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    macro_halo = k['macro_halo_frac'] * max(cw, ch)

    # Write bookshelf with macro initial positions pre-set to (init_x, init_y)
    aux = write_bookshelf(bench, trial_dir / 'bench', bench_name=bench_name,
                          init_x=k['init_x'], init_y=k['init_y'])

    cfg = write_dreamplace_config(
        aux, trial_dir / 'results',
        gpu=True,
        extra={
            'random_seed':    1000,
            'macro_halo_x':   macro_halo,
            'macro_halo_y':   macro_halo,
            'routability_opt': k['routability_opt'],
        },
        target_density  = k['target_density'],
        density_weight  = k['density_weight'],
        gamma           = k['gamma'],
        num_bins_x      = 512 if bench_name in {'ibm17','ibm18'} else
                          256 if bench_name in {'ibm14','ibm15','ibm16'} else 256,
        num_bins_y      = 512 if bench_name in {'ibm17','ibm18'} else
                          256 if bench_name in {'ibm14','ibm15','ibm16'} else 256,
        iteration       = 1000,
        learning_rate   = 0.01,
        legalize_flag   = 0,
        detailed_place_flag = 0,
    )

    t0 = time.time()
    result = subprocess.run(
        ['python3', DREAMPLACE_PLACER, str(cfg)],
        capture_output=True, text=True, timeout=600,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f'  {bench_name}  FAILED (returncode={result.returncode})  [{elapsed:.0f}s]')
        print(result.stderr[-500:] if result.stderr else '')
        return {'bench': bench_name, 'status': 'failed', 'elapsed': elapsed}

    pl_files = sorted((trial_dir / 'results').glob('**/*.pl'))
    if not pl_files:
        print(f'  {bench_name}  FAILED (no .pl output)  [{elapsed:.0f}s]')
        return {'bench': bench_name, 'status': 'failed', 'elapsed': elapsed}

    positions = read_dreamplace_pl(pl_files[-1], bench)
    positions[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
    positions, n_overlaps = legalize_overlaps_strict(positions, bench, max_iters=200)

    costs = compute_proxy_cost(positions, benchmark, plc)
    proxy = float(costs['proxy_cost'])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'{bench_name}.pt'
    torch.save({'positions': positions, 'score': proxy, 'costs': costs,
                'knobs': k, 'bench': bench_name}, str(out_path))

    print(f'  {bench_name:8s}  proxy={proxy:.4f}  overlaps={n_overlaps}  [{elapsed:.0f}s]  → {out_path.name}')
    return {'bench': bench_name, 'status': 'ok', 'proxy': proxy,
            'overlaps': n_overlaps, 'elapsed': elapsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benches', default='all')
    args = ap.parse_args()

    benches = ALL_BENCHES if args.benches == 'all' \
              else [b.strip() for b in args.benches.split(',')]

    print(f'Replaying reference best knobs  ({len(benches)} benches)')
    print(f'Fixed: iteration=1000  random_seed=1000  macro_halo from frac\n')

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    t0 = time.time()
    for b in benches:
        if b not in BEST_KNOBS:
            print(f'  {b}  no reference knobs — skipping')
            continue
        r = run_bench(b)
        results.append(r)

    ok = [r for r in results if r['status'] == 'ok']
    scores = [r['proxy'] for r in ok]
    avg = sum(scores) / len(scores) if scores else float('nan')

    print(f"\n{'='*60}")
    print(f'  Done  {len(ok)}/{len(benches)}  avg proxy={avg:.4f}')
    print(f'  Reference avg target_density = {sum(BEST_KNOBS[b]["target_density"] for b in benches)/len(benches):.3f}')
    print(f'  Elapsed: {(time.time()-t0)/60:.1f} min')
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
