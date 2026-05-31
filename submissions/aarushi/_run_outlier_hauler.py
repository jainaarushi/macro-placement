#!/usr/bin/env python3
"""
Phase: Outlier Hauler on soft macros → cache_hauler/

Reads best available input (cache_sa_12 → cache_autodmp → cache_legal → cache).
Applies outlier_hauler to each bench. Writes to cache_hauler/.

Usage:
    python3 -u submissions/aarushi/_run_outlier_hauler.py \
        --benches ibm01,ibm02,ibm03   # or "all"
"""

import sys, argparse, time, json
from pathlib import Path

REPO = Path('/home/ubuntu/macro-place-challenge-2026')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'submissions' / 'aarushi'))

import torch
from macro_place.benchmark import Benchmark
from macro_place.objective  import compute_proxy_cost
from macro_place.loader     import load_benchmark_from_dir
from outlier_hauler         import outlier_hauler

SUB           = REPO / 'submissions' / 'aarushi'
TESTCASE_ROOT = REPO / 'external' / 'MacroPlacement' / 'Testcases' / 'ICCAD04'
PT_DIR        = REPO / 'benchmarks' / 'processed' / 'public'

IN_DIRS = [
    SUB / 'cache_sa_12',
    SUB / 'cache_autodmp',
    SUB / 'cache_legal',
    SUB / 'cache',
]
OUT_DIR = SUB / 'cache_hauler'

BENCHMARKS = [
    'ibm01','ibm02','ibm03','ibm04','ibm06','ibm07','ibm08',
    'ibm09','ibm10','ibm11','ibm12','ibm13','ibm14','ibm15',
    'ibm16','ibm17','ibm18',
]


def load_best_input(bench: str):
    for d in IN_DIRS:
        p = d / f'{bench}.pt'
        if p.exists():
            data = torch.load(str(p), weights_only=False)
            return data['positions'], str(p)
    return None, None


def process(bench: str, manifest: dict):
    print(f"\n{'─'*60}\n  {bench}")
    pos, src = load_best_input(bench)
    if pos is None:
        print('  no input — skipping'); return
    print(f"  input: {src}")

    # Use load_benchmark_from_dir — Benchmark.load(.pt) returns empty net_nodes
    benchmark, plc = load_benchmark_from_dir(str(TESTCASE_ROOT / bench))

    before = compute_proxy_cost(pos, benchmark, plc)
    proxy_b = float(before['proxy_cost'])
    print(f"  before: proxy={proxy_b:.4f}  cong={before['congestion_cost']:.3f}  "
          f"soft={benchmark.num_soft_macros}")

    t0  = time.time()
    pos = outlier_hauler(pos, benchmark, plc)
    elapsed = time.time() - t0

    after = compute_proxy_cost(pos, benchmark, plc)
    proxy_a = float(after['proxy_cost'])
    out = OUT_DIR / f'{bench}.pt'
    torch.save({'positions': pos, 'score': proxy_a, 'costs': after}, str(out))

    print(f"  after:  proxy={proxy_a:.4f}  (Δ{proxy_a - proxy_b:+.4f})  "
          f"overlaps={after['overlap_count']}  [{elapsed:.0f}s]")
    print(f"  ✓ saved → {out}")

    manifest[bench] = {
        'proxy_before': proxy_b,
        'proxy_after':  proxy_a,
        'delta':        proxy_a - proxy_b,
        'overlap_count': after['overlap_count'],
        'elapsed_s':    elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benches', default='ibm01,ibm02,ibm03')
    args = ap.parse_args()
    benches = BENCHMARKS if args.benches == 'all' else \
              [b.strip() for b in args.benches.split(',')]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}; t0 = time.time()

    for b in benches:
        try:
            process(b, manifest)
        except Exception as e:
            print(f'  ERROR {b}: {e}')
            import traceback; traceback.print_exc()

    valid = [v for v in manifest.values() if 'proxy_after' in v]
    scores = [v['proxy_after'] for v in valid]
    before = [v['proxy_before'] for v in valid]
    avg_a = sum(scores) / len(scores) if scores else float('nan')
    avg_b = sum(before) / len(before) if before else float('nan')

    print(f"\n{'='*60}")
    print(f'  outlier_hauler  {len(valid)}/{len(benches)} benches  '
          f'avg {avg_b:.4f} → {avg_a:.4f}  (Δ{avg_a - avg_b:+.4f})')
    print(f'  Output: {OUT_DIR}')
    print(f'  Elapsed: {(time.time()-t0)/60:.1f} min')
    print(f"{'='*60}")
    with open(str(OUT_DIR / 'hauler_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)


if __name__ == '__main__':
    main()
