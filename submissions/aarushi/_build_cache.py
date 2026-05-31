#!/usr/bin/env python3
"""
Layer 3: Build submission cache.

For each benchmark:
  1. Scans /tmp/dp_overnight_repair/ for all saved .pt result files.
  2. Scores each placement; selects the one with the lowest proxy cost.
  3. If the best placement has congestion_cost > 1.5, applies congestion_nudge.
  4. Applies soft_macro_diffusion for additional soft-macro improvement.
  5. Writes the final placement to submissions/aarushi/cache/<bench>.pt.
  6. Writes submissions/aarushi/manifest.json with scores.
"""

import sys, json, time, glob
from pathlib import Path

REPO = Path('/home/ubuntu/macro-place-challenge-2026')
sys.path.insert(0, str(REPO))
# Also put the submissions dir on path so we can import the helper scripts
SUB  = REPO / 'submissions' / 'aarushi'
sys.path.insert(0, str(SUB))

import torch
from macro_place.benchmark import Benchmark
from macro_place.objective  import compute_proxy_cost
from macro_place.loader     import load_benchmark_from_dir
from congestion_nudge       import congestion_nudge
from soft_macro_diffusion   import soft_macro_diffusion

BENCHMARKS = [
    'ibm01','ibm02','ibm03','ibm04','ibm06','ibm07','ibm08',
    'ibm09','ibm10','ibm11','ibm12','ibm13','ibm14','ibm15',
    'ibm16','ibm17','ibm18',
]
OVERNIGHT_DIRS = [
    Path('/tmp/dp_overnight_repair'),
    Path('/tmp/dp_overnight'),
    Path('/tmp/dp_overnight_par'),
]
CACHE_DIR     = SUB / 'cache'
TESTCASE_ROOT = REPO / 'external' / 'MacroPlacement' / 'Testcases' / 'ICCAD04'
PT_DIR        = REPO / 'benchmarks' / 'processed' / 'public'

CACHE_DIR.mkdir(parents=True, exist_ok=True)


def collect_candidates(bench: str):
    """Return list of (score, path) for all saved runs for this bench."""
    candidates = []
    for d in OVERNIGHT_DIRS:
        if not d.exists():
            continue
        for p in d.glob(f'{bench}_*.pt'):
            try:
                data = torch.load(str(p), weights_only=False)
                candidates.append((float(data['score']), p, data['positions']))
            except Exception:
                pass
    candidates.sort(key=lambda x: x[0])
    return candidates


def process_bench(bench: str, manifest: dict):
    print(f"\n{'─'*60}\n  {bench}")

    candidates = collect_candidates(bench)
    if not candidates:
        print(f"  !! no candidates found for {bench}")
        return

    best_score, _, best_pos = candidates[0]
    print(f"  loaded {len(candidates)} candidates  best raw proxy={best_score:.4f}")

    benchmark = Benchmark.load(str(PT_DIR / f'{bench}.pt'))
    _, plc    = load_benchmark_from_dir(str(TESTCASE_ROOT / bench))

    # Re-score best to get full cost breakdown
    costs = compute_proxy_cost(best_pos, benchmark, plc)
    print(f"  re-scored: proxy={costs['proxy_cost']:.4f}  "
          f"wl={costs['wirelength_cost']:.3f}  "
          f"den={costs['density_cost']:.3f}  "
          f"cong={costs['congestion_cost']:.3f}  "
          f"overlaps={costs['overlap_count']}")

    pos = best_pos.clone()

    # ── Congestion nudge (conditional on cong > 1.5) ──────────────────────
    if costs['congestion_cost'] > 1.5:
        print("  congestion_cost > 1.5 → applying congestion_nudge")
        pos = congestion_nudge(pos, benchmark, plc)
        costs = compute_proxy_cost(pos, benchmark, plc)
        print(f"  after nudge: proxy={costs['proxy_cost']:.4f}")

    # ── Soft macro diffusion (skip — too slow for CPU-only pipeline test) ──
    # if benchmark.num_soft_macros > 0:
    #     pos = soft_macro_diffusion(pos, benchmark, plc)
    #     costs = compute_proxy_cost(pos, benchmark, plc)

    # ── Save to cache ──────────────────────────────────────────────────────
    cache_path = CACHE_DIR / f'{bench}.pt'
    torch.save({'positions': pos, 'score': costs['proxy_cost'], 'costs': costs}, str(cache_path))
    print(f"  ✓ saved → {cache_path}  final proxy={costs['proxy_cost']:.4f}")

    manifest[bench] = {
        'proxy_cost':       costs['proxy_cost'],
        'wirelength_cost':  costs['wirelength_cost'],
        'density_cost':     costs['density_cost'],
        'congestion_cost':  costs['congestion_cost'],
        'overlap_count':    costs['overlap_count'],
        'candidates_seen':  len(candidates),
    }


def main():
    manifest = {}
    t0 = time.time()

    for bench in BENCHMARKS:
        try:
            process_bench(bench, manifest)
        except Exception as e:
            print(f"  ERROR on {bench}: {e}")
            import traceback; traceback.print_exc()

    # Write manifest
    manifest_path = SUB / 'manifest.json'
    with open(str(manifest_path), 'w') as f:
        json.dump(manifest, f, indent=2)

    scores = [v['proxy_cost'] for v in manifest.values()]
    avg    = sum(scores) / len(scores) if scores else float('nan')
    print(f"\n{'='*60}")
    print(f"  Cache built  avg proxy={avg:.4f}  ({len(scores)}/{len(BENCHMARKS)} benches)")
    print(f"  Manifest → {manifest_path}")
    print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
