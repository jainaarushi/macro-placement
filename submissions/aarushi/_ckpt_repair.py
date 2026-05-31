#!/usr/bin/env python3
"""
Phase 2.1 — Legalize all cached placements to zero overlaps.

Usage:
  python -u submissions/aarushi/_ckpt_repair.py --in cache --out cache_legal

For each <bench>.pt in --in:
  1. Runs legalize_v2.legalize_overlaps_strict to push overlapping macros apart.
  2. Writes <bench>.pt to --out with zero overlaps guaranteed.
  3. Verifies via verify_zero_overlaps.

Runs in parallel across all benches (one worker per bench).
"""

import sys, argparse, json, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = Path('/home/ubuntu/macro-place-challenge-2026')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'submissions' / 'aarushi'))

import torch
from macro_place.benchmark import Benchmark
from macro_place.objective  import compute_proxy_cost
from macro_place.loader     import load_benchmark_from_dir
from legalize_v2            import legalize_overlaps_strict, verify_zero_overlaps

TESTCASE_ROOT = REPO / 'external' / 'MacroPlacement' / 'Testcases' / 'ICCAD04'
PT_DIR        = REPO / 'benchmarks' / 'processed' / 'public'

BENCHMARKS = [
    'ibm01','ibm02','ibm03','ibm04','ibm06','ibm07','ibm08',
    'ibm09','ibm10','ibm11','ibm12','ibm13','ibm14','ibm15',
    'ibm16','ibm17','ibm18',
]


def _process_one(bench: str, in_dir: Path, out_dir: Path) -> dict:
    src = in_dir / f'{bench}.pt'
    if not src.exists():
        return {'bench': bench, 'status': 'missing'}

    t0 = time.time()
    data      = torch.load(str(src), weights_only=False)
    pos       = data['positions']
    benchmark = Benchmark.load(str(PT_DIR / f'{bench}.pt'))
    _, plc    = load_benchmark_from_dir(str(TESTCASE_ROOT / bench))

    before    = compute_proxy_cost(pos, benchmark, plc)
    n_before  = verify_zero_overlaps(pos, benchmark)

    if n_before == 0:
        # Already clean — just copy
        torch.save(data, str(out_dir / f'{bench}.pt'))
        return {
            'bench': bench, 'status': 'clean',
            'proxy': before['proxy_cost'], 'overlaps': 0,
            'elapsed': time.time() - t0,
        }

    try:
        pos_fixed = legalize_overlaps_strict(pos, benchmark)
    except RuntimeError as e:
        return {'bench': bench, 'status': 'failed', 'error': str(e)}

    after = compute_proxy_cost(pos_fixed, benchmark, plc)
    torch.save({'positions': pos_fixed, 'score': after['proxy_cost'], 'costs': after},
               str(out_dir / f'{bench}.pt'))

    return {
        'bench':   bench,
        'status':  'fixed',
        'proxy_before': before['proxy_cost'],
        'proxy_after':  after['proxy_cost'],
        'overlaps_before': n_before,
        'overlaps_after':  verify_zero_overlaps(pos_fixed, benchmark),
        'elapsed': time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in',  dest='in_dir',  default='cache',       help='input dir relative to repo root')
    ap.add_argument('--out', dest='out_dir', default='cache_legal', help='output dir relative to repo root')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    in_dir  = REPO / 'submissions' / 'aarushi' / args.in_dir
    out_dir = REPO / 'submissions' / 'aarushi' / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    benches = [b for b in BENCHMARKS if (in_dir / f'{b}.pt').exists()]
    print(f"Legalizing {len(benches)} benches: {in_dir} → {out_dir}")
    print(f"Workers: {args.workers}\n")

    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_one, b, in_dir, out_dir): b for b in benches}
        for fut in as_completed(futures):
            r = fut.result()
            b = r['bench']
            results[b] = r
            if r['status'] == 'clean':
                print(f"  {b:8s}  clean      proxy={r['proxy']:.4f}")
            elif r['status'] == 'fixed':
                delta = r['proxy_after'] - r['proxy_before']
                print(f"  {b:8s}  fixed      "
                      f"overlaps {r['overlaps_before']}→{r['overlaps_after']}  "
                      f"proxy={r['proxy_after']:.4f}  (Δ{delta:+.4f})  "
                      f"{r['elapsed']:.1f}s")
            elif r['status'] == 'failed':
                print(f"  {b:8s}  FAILED     {r['error']}")
            else:
                print(f"  {b:8s}  missing")

    valid   = [r for r in results.values() if r['status'] in ('clean', 'fixed')]
    scores  = [r.get('proxy_after', r.get('proxy', 0)) for r in valid]
    avg     = sum(scores) / len(scores) if scores else float('nan')
    failed  = [r['bench'] for r in results.values() if r['status'] == 'failed']

    print(f"\n{'='*60}")
    print(f"  Done  {len(valid)}/{len(benches)} benches  avg proxy={avg:.4f}")
    if failed:
        print(f"  FAILED: {failed}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    manifest = {r['bench']: r for r in results.values()}
    with open(str(out_dir / 'repair_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)


if __name__ == '__main__':
    main()
