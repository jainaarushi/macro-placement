#!/usr/bin/env python3
"""
Layer 4: Final polish.

For each benchmark in the cache:
  1. Re-legalize hard macros (removes any residual overlaps).
  2. Conditional congestion nudge: only if nudge improves proxy by >= min_delta.
  3. Overwrites the cache entry with the improved placement.
  4. Updates manifest.json with final scores.
"""

import sys, json, time
from pathlib import Path

REPO = Path('/home/ubuntu/macro-place-challenge-2026')
sys.path.insert(0, str(REPO))
SUB  = REPO / 'submissions' / 'aarushi'
sys.path.insert(0, str(SUB))

import torch
import numpy as np
from macro_place.benchmark import Benchmark
from macro_place.objective  import compute_proxy_cost
from macro_place.loader     import load_benchmark_from_dir
from congestion_nudge       import congestion_nudge

BENCHMARKS = [
    'ibm01','ibm02','ibm03','ibm04','ibm06','ibm07','ibm08',
    'ibm09','ibm10','ibm11','ibm12','ibm13','ibm14','ibm15',
    'ibm16','ibm17','ibm18',
]
CACHE_DIR     = SUB / 'cache'
TESTCASE_ROOT = REPO / 'external' / 'MacroPlacement' / 'Testcases' / 'ICCAD04'
PT_DIR        = REPO / 'benchmarks' / 'processed' / 'public'
MIN_DELTA     = 0.005   # minimum proxy improvement to accept nudge


def _re_legalize(pos_t: torch.Tensor, benchmark: Benchmark,
                 max_iters: int = 2000, margin: float = 1.0) -> torch.Tensor:
    """Remove residual overlaps between hard macros."""
    pos   = pos_t.numpy().copy()
    nh    = benchmark.num_hard_macros
    sz    = benchmark.macro_sizes.numpy()
    fixed = benchmark.macro_fixed.numpy()
    cw, ch = benchmark.canvas_width, benchmark.canvas_height

    for _ in range(max_iters):
        moved = False
        for i in range(nh):
            if fixed[i]: continue
            for j in range(i + 1, nh):
                dx = abs(pos[i,0] - pos[j,0])
                dy = abs(pos[i,1] - pos[j,1])
                sx = (sz[i,0] + sz[j,0]) / 2 + margin
                sy = (sz[i,1] + sz[j,1]) / 2 + margin
                if dx < sx and dy < sy:
                    ox, oy = sx - dx, sy - dy
                    if ox <= oy:
                        # resolve fully in X
                        s = 1 if pos[i,0] >= pos[j,0] else -1
                        if not fixed[i] and not fixed[j]:
                            pos[i,0] += s * (ox / 2 + margin)
                            pos[j,0] -= s * (ox / 2 + margin)
                        elif not fixed[i]:
                            pos[i,0] += s * (ox + margin)
                        else:
                            pos[j,0] -= s * (ox + margin)
                    else:
                        # resolve fully in Y
                        s = 1 if pos[i,1] >= pos[j,1] else -1
                        if not fixed[i] and not fixed[j]:
                            pos[i,1] += s * (oy / 2 + margin)
                            pos[j,1] -= s * (oy / 2 + margin)
                        elif not fixed[i]:
                            pos[i,1] += s * (oy + margin)
                        else:
                            pos[j,1] -= s * (oy + margin)
                    moved = True

        for i in range(nh):
            if fixed[i]: continue
            pos[i,0] = np.clip(pos[i,0], sz[i,0]/2, cw - sz[i,0]/2)
            pos[i,1] = np.clip(pos[i,1], sz[i,1]/2, ch - sz[i,1]/2)

        if not moved:
            break

    return torch.tensor(pos, dtype=torch.float32)


def polish_bench(bench: str, manifest: dict):
    print(f"\n  {bench}")
    cache_path = CACHE_DIR / f'{bench}.pt'
    if not cache_path.exists():
        print(f"    !! cache missing, skipping")
        return

    data      = torch.load(str(cache_path), weights_only=False)
    pos       = data['positions']
    benchmark = Benchmark.load(str(PT_DIR / f'{bench}.pt'))
    _, plc    = load_benchmark_from_dir(str(TESTCASE_ROOT / bench))

    # 1. Re-legalize
    pos = _re_legalize(pos, benchmark)
    costs = compute_proxy_cost(pos, benchmark, plc)
    cur  = costs['proxy_cost']
    print(f"    after re-legalize: proxy={cur:.4f}  overlaps={costs['overlap_count']}")

    # 2. Conditional nudge
    nudged = congestion_nudge(pos, benchmark, plc, rounds=2)
    nudge_costs = compute_proxy_cost(nudged, benchmark, plc)
    if nudge_costs['proxy_cost'] < cur - MIN_DELTA:
        pos    = nudged
        costs  = nudge_costs
        cur    = costs['proxy_cost']
        print(f"    nudge accepted: proxy={cur:.4f}")
    else:
        print(f"    nudge skipped (Δ={nudge_costs['proxy_cost'] - cur:+.4f} < {MIN_DELTA})")

    # 3. Overwrite cache
    torch.save({'positions': pos, 'score': cur, 'costs': costs}, str(cache_path))

    manifest[bench] = {
        'proxy_cost':      costs['proxy_cost'],
        'wirelength_cost': costs['wirelength_cost'],
        'density_cost':    costs['density_cost'],
        'congestion_cost': costs['congestion_cost'],
        'overlap_count':   costs['overlap_count'],
    }
    print(f"    ✓ final proxy={cur:.4f}")


def main():
    manifest = {}
    t0 = time.time()
    print("Layer 4: Final polish")

    for bench in BENCHMARKS:
        try:
            polish_bench(bench, manifest)
        except Exception as e:
            print(f"  ERROR {bench}: {e}")
            import traceback; traceback.print_exc()

    manifest_path = SUB / 'manifest.json'
    with open(str(manifest_path), 'w') as f:
        json.dump(manifest, f, indent=2)

    scores = [v['proxy_cost'] for v in manifest.values()]
    avg    = sum(scores) / len(scores) if scores else float('nan')
    print(f"\n{'='*60}")
    print(f"  FINAL avg proxy={avg:.4f}  ({len(scores)}/{len(BENCHMARKS)} benches)")
    print(f"  Manifest → {manifest_path}")
    print(f"  Elapsed: {(time.time()-t0)/60:.1f} min")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
