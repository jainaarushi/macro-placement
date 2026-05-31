#!/usr/bin/env python3
"""
Phase: Soft-macro diffusion v2 → cache_sa_12/

Reads from cache_autodmp/ (falls back to cache_legal/, then cache/).
Applies soft_macro_diffusion_v2 to each bench.
Writes improved placements to cache_sa_12/.

Usage:
    python3 -u submissions/aarushi/_run_sa_diffusion.py \
        --benches ibm01,ibm02,ibm03   # or "all"
    python3 -u submissions/aarushi/_run_sa_diffusion.py --all
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
from soft_macro_diffusion_v2 import soft_macro_diffusion_v2

SUB           = REPO / 'submissions' / 'aarushi'
TESTCASE_ROOT = REPO / 'external' / 'MacroPlacement' / 'Testcases' / 'ICCAD04'
PT_DIR        = REPO / 'benchmarks' / 'processed' / 'public'

IN_DIRS = [
    SUB / 'cache_autodmp',
    SUB / 'cache_legal',
    SUB / 'cache',
]
OUT_DIR = SUB / 'cache_sa_12'

BENCHMARKS = [
    'ibm01','ibm02','ibm03','ibm04','ibm06','ibm07','ibm08',
    'ibm09','ibm10','ibm11','ibm12','ibm13','ibm14','ibm15',
    'ibm16','ibm17','ibm18',
]


def load_best_input(bench: str) -> tuple:
    """Load positions from best available input dir."""
    for d in IN_DIRS:
        p = d / f'{bench}.pt'
        if p.exists():
            data = torch.load(str(p), weights_only=False)
            return data['positions'], str(p)
    return None, None


def process_bench(bench: str, manifest: dict):
    print(f"\n{'─'*60}\n  {bench}")

    pos, src = load_best_input(bench)
    if pos is None:
        print(f"  !! no input found — skipping")
        return

    print(f"  input: {src}")

    benchmark = Benchmark.load(str(PT_DIR / f'{bench}.pt'))
    _, plc    = load_benchmark_from_dir(str(TESTCASE_ROOT / bench))

    costs_before = compute_proxy_cost(pos, benchmark, plc)
    proxy_before = float(costs_before['proxy_cost'])
    print(f"  before: proxy={proxy_before:.4f}  "
          f"wl={costs_before['wirelength_cost']:.3f}  "
          f"cong={costs_before['congestion_cost']:.3f}  "
          f"soft_macros={benchmark.num_soft_macros}")

    t0  = time.time()
    pos = soft_macro_diffusion_v2(pos, benchmark, plc)
    elapsed = time.time() - t0

    costs_after = compute_proxy_cost(pos, benchmark, plc)
    proxy_after = float(costs_after['proxy_cost'])
    delta = proxy_after - proxy_before

    out_path = OUT_DIR / f'{bench}.pt'
    torch.save({'positions': pos, 'score': proxy_after, 'costs': costs_after}, str(out_path))

    print(f"  after:  proxy={proxy_after:.4f}  "
          f"(Δ{delta:+.4f})  overlaps={costs_after['overlap_count']}  "
          f"[{elapsed:.1f}s]")
    print(f"  ✓ saved → {out_path}")

    manifest[bench] = {
        'proxy_before': proxy_before,
        'proxy_after':  proxy_after,
        'delta':        delta,
        'overlap_count': costs_after['overlap_count'],
        'elapsed_s':    elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benches', default='', help='comma-separated or "all"')
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()

    if args.all or args.benches == 'all':
        benches = BENCHMARKS
    elif args.benches:
        benches = [b.strip() for b in args.benches.split(',')]
    else:
        benches = BENCHMARKS

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}
    t0 = time.time()

    for bench in benches:
        try:
            process_bench(bench, manifest)
        except Exception as e:
            print(f"  ERROR {bench}: {e}")
            import traceback; traceback.print_exc()

    valid  = [v for v in manifest.values() if 'proxy_after' in v]
    scores = [v['proxy_after'] for v in valid]
    avg    = sum(scores) / len(scores) if scores else float('nan')

    scores_before = [v['proxy_before'] for v in valid]
    avg_before    = sum(scores_before) / len(scores_before) if scores_before else float('nan')

    print(f"\n{'='*60}")
    print(f"  SA diffusion v2 complete  {len(valid)}/{len(benches)} benches")
    print(f"  avg proxy:  {avg_before:.4f} → {avg:.4f}  (Δ{avg - avg_before:+.4f})")
    print(f"  Output: {OUT_DIR}")
    print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")
    print(f"{'='*60}")

    with open(str(OUT_DIR / 'sa_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)


if __name__ == '__main__':
    main()
