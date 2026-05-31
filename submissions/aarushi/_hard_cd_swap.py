#!/usr/bin/env python3
"""
Hard-macro CD + pairwise swap → cache_hard2/

After CD-on-soft, hard macros are stale but the layout is now super-tight
(~5% of pairs have < 1 grid-cell gap on ibm04). Random SA shifts hit overlaps
60-70% of the time. Systematic 8-direction probes at decaying step sizes find
tiny safe shifts that SA misses. Same-size pairwise swap is the only
non-local lever that's overlap-safe by construction.

Pipeline per benchmark:
  1. Multi-scale CD on hard macros (steps [4, 2, 1, 0.5, 0.25] cells × 3 passes)
     For each macro, try 8 directions (4 cardinal + 4 diagonal unit-norm).
     Accept the best Δproxy<0 move that does not introduce a new overlap.
  2. Pairwise swap among same-size hard macros (1 pass).
     For each pair (i, j) with sizes_i == sizes_j: swap positions, eval, accept iff Δproxy<0.

Inputs preference: cache_sa_12/ → cache_autodmp/ → cache_legal/ → cache/
Output: submissions/aarushi/cache_hard2/<bench>.pt

Usage:
    python3 -u submissions/aarushi/_hard_cd_swap.py --benches ibm01,ibm02,ibm03
    python3 -u submissions/aarushi/_hard_cd_swap.py --benches all
"""

import sys, math, random, argparse, time, json
from pathlib import Path

REPO = Path('/home/ubuntu/macro-place-challenge-2026')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'submissions' / 'aarushi'))

import torch
import numpy as np

from macro_place.benchmark import Benchmark
from macro_place.objective  import compute_proxy_cost, _set_placement
from macro_place._plc       import PlacementCost
from macro_place.loader     import load_benchmark_from_dir


SUB           = REPO / 'submissions' / 'aarushi'
TESTCASE_ROOT = REPO / 'external' / 'MacroPlacement' / 'Testcases' / 'ICCAD04'

IN_DIRS = [
    SUB / 'cache_sa_12',
    SUB / 'cache_autodmp',
    SUB / 'cache_legal',
    SUB / 'cache',
]
OUT_DIR = SUB / 'cache_hard2'

BENCHMARKS = [
    'ibm01','ibm02','ibm03','ibm04','ibm06','ibm07','ibm08',
    'ibm09','ibm10','ibm11','ibm12','ibm13','ibm14','ibm15',
    'ibm16','ibm17','ibm18',
]

# 8 unit-norm directions (4 cardinal + 4 diagonal)
_inv_sqrt2 = 1.0 / math.sqrt(2.0)
DIRECTIONS = [
    (+1.0,  0.0), (-1.0,  0.0), ( 0.0, +1.0), ( 0.0, -1.0),
    (+_inv_sqrt2, +_inv_sqrt2), (+_inv_sqrt2, -_inv_sqrt2),
    (-_inv_sqrt2, +_inv_sqrt2), (-_inv_sqrt2, -_inv_sqrt2),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _proxy(pos_np: np.ndarray, benchmark: Benchmark, plc: PlacementCost) -> float:
    """Full proxy via PlacementCost."""
    pos_t = torch.from_numpy(pos_np)
    _set_placement(plc, pos_t, benchmark)
    plc.get_congestion_cost()
    return float(compute_proxy_cost(pos_t, benchmark, plc)['proxy_cost'])


def _rect_overlaps(ax, ay, aw, ah, bx, by, bw, bh, tol: float = 0.0) -> bool:
    """Center-based AABB overlap test with tolerance margin."""
    dx = abs(ax - bx); dy = abs(ay - by)
    return (dx < (aw + bw) / 2.0 - tol) and (dy < (ah + bh) / 2.0 - tol)


def _move_creates_overlap(pos_np: np.ndarray, sizes: np.ndarray, nh: int,
                          m: int, nx: float, ny: float, tol: float = 1e-3) -> bool:
    """Check whether moving macro m to (nx, ny) overlaps any other hard macro."""
    mw, mh = sizes[m]
    for j in range(nh):
        if j == m:
            continue
        if _rect_overlaps(nx, ny, mw, mh,
                          pos_np[j, 0], pos_np[j, 1], sizes[j, 0], sizes[j, 1], tol):
            return True
    return False


# ── CD pass on hard macros ────────────────────────────────────────────────────

def cd_pass_hard(pos_np: np.ndarray,
                 benchmark: Benchmark,
                 plc: PlacementCost,
                 step_cells: float,
                 grid_w: float, grid_h: float,
                 cur_score: float,
                 rng: random.Random) -> tuple:
    """One CD pass: for each hard macro try 8 directions, take best Δ<0 overlap-free."""
    nh = benchmark.num_hard_macros
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    sizes = benchmark.macro_sizes.numpy()
    fixed = benchmark.macro_fixed.numpy().astype(bool)

    order = list(range(nh))
    rng.shuffle(order)

    n_accepts = 0
    n_evals   = 0
    for m in order:
        if fixed[m]:
            continue
        ox, oy = pos_np[m, 0], pos_np[m, 1]
        mw, mh = sizes[m, 0], sizes[m, 1]

        best_delta = 0.0
        best_xy    = None
        for ux, uy in DIRECTIONS:
            nx = ox + ux * step_cells * grid_w
            ny = oy + uy * step_cells * grid_h
            # canvas clip
            nx = max(mw / 2.0, min(cw - mw / 2.0, nx))
            ny = max(mh / 2.0, min(ch - mh / 2.0, ny))
            if nx == ox and ny == oy:
                continue
            if _move_creates_overlap(pos_np, sizes, nh, m, nx, ny):
                continue

            saved = pos_np[m].copy()
            pos_np[m, 0] = nx
            pos_np[m, 1] = ny
            new_score = _proxy(pos_np, benchmark, plc)
            n_evals += 1
            delta = new_score - cur_score
            pos_np[m] = saved
            if delta < best_delta:
                best_delta = delta
                best_xy    = (nx, ny)

        if best_xy is not None:
            pos_np[m, 0] = best_xy[0]
            pos_np[m, 1] = best_xy[1]
            cur_score += best_delta
            n_accepts += 1

    return cur_score, n_accepts, n_evals


# ── pairwise swap on same-size hard macros ────────────────────────────────────

def _build_size_groups(sizes: np.ndarray, nh: int, fixed: np.ndarray) -> dict:
    """idx by rounded (w, h) → list of movable hard-macro indices."""
    groups = {}
    for i in range(nh):
        if fixed[i]:
            continue
        key = (round(float(sizes[i, 0]), 4), round(float(sizes[i, 1]), 4))
        groups.setdefault(key, []).append(i)
    return groups


def pairwise_swap_pass(pos_np: np.ndarray,
                       benchmark: Benchmark,
                       plc: PlacementCost,
                       cur_score: float,
                       rng: random.Random) -> tuple:
    """For each same-size pair (i, j): swap and accept iff Δproxy < 0."""
    nh = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes.numpy()
    fixed = benchmark.macro_fixed.numpy().astype(bool)
    groups = _build_size_groups(sizes, nh, fixed)

    pairs = []
    for key, ids in groups.items():
        if len(ids) < 2:
            continue
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                pairs.append((ids[a], ids[b]))
    rng.shuffle(pairs)

    n_accepts = 0
    n_evals   = 0
    for i, j in pairs:
        # swap
        pi = pos_np[i].copy()
        pj = pos_np[j].copy()
        pos_np[i] = pj
        pos_np[j] = pi
        new_score = _proxy(pos_np, benchmark, plc)
        n_evals += 1
        delta = new_score - cur_score
        if delta < -1e-6:
            cur_score = new_score
            n_accepts += 1
        else:
            # revert
            pos_np[i] = pi
            pos_np[j] = pj

    return cur_score, n_accepts, n_evals, len(pairs)


# ── per-bench driver ──────────────────────────────────────────────────────────

def process(bench: str, manifest: dict, step_cells, n_passes: int, seed: int):
    print(f"\n{'─'*60}\n  {bench}")

    # Locate input
    pos = None; src = None
    for d in IN_DIRS:
        p = d / f'{bench}.pt'
        if p.exists():
            data = torch.load(str(p), weights_only=False)
            pos = data['positions']; src = str(p); break
    if pos is None:
        print('  no input — skipping'); return
    print(f"  input: {src}")

    benchmark, plc = load_benchmark_from_dir(str(TESTCASE_ROOT / bench))

    pos_np = pos.numpy().copy()
    rng = random.Random(seed)

    rows, cols = plc.grid_row, plc.grid_col
    grid_w = float(benchmark.canvas_width)  / cols
    grid_h = float(benchmark.canvas_height) / rows

    before = compute_proxy_cost(torch.from_numpy(pos_np), benchmark, plc)
    proxy_b = float(before['proxy_cost'])
    print(f"  before: proxy={proxy_b:.4f}  cong={before['congestion_cost']:.3f}  "
          f"wl={before['wirelength_cost']:.3f}  overlaps={before['overlap_count']}  "
          f"hard={benchmark.num_hard_macros}")

    cur_score = proxy_b
    t0 = time.time()
    total_cd_accepts = 0
    total_cd_evals   = 0

    # ── CD cascade on hard macros ────────────────────────────────────────────
    for step in step_cells:
        for pass_idx in range(n_passes):
            cur_score, n_acc, n_eval = cd_pass_hard(
                pos_np, benchmark, plc, step, grid_w, grid_h, cur_score, rng)
            total_cd_accepts += n_acc
            total_cd_evals   += n_eval
            print(f"    CD step={step:5.2f} pass {pass_idx+1}  "
                  f"evals={n_eval}  accepts={n_acc}  proxy={cur_score:.4f}")
            if n_acc == 0:
                break

    # ── Pairwise swap among same-size hard macros ────────────────────────────
    cur_score, n_swap_acc, n_swap_eval, n_pairs = pairwise_swap_pass(
        pos_np, benchmark, plc, cur_score, rng)
    print(f"    SWAP  pairs={n_pairs}  evals={n_swap_eval}  "
          f"accepts={n_swap_acc}  proxy={cur_score:.4f}")

    elapsed = time.time() - t0

    after = compute_proxy_cost(torch.from_numpy(pos_np), benchmark, plc)
    proxy_a = float(after['proxy_cost'])

    out = OUT_DIR / f'{bench}.pt'
    torch.save({'positions': torch.from_numpy(pos_np),
                'score': proxy_a, 'costs': after}, str(out))

    delta = proxy_a - proxy_b
    pct = (delta / proxy_b * 100.0) if proxy_b else 0.0
    print(f"  after:  proxy={proxy_a:.4f}  cong={after['congestion_cost']:.3f}  "
          f"wl={after['wirelength_cost']:.3f}  "
          f"(Δ{delta:+.4f}, {pct:+.1f}%)  overlaps={after['overlap_count']}  "
          f"CD acc={total_cd_accepts}/{total_cd_evals}  "
          f"swap acc={n_swap_acc}/{n_swap_eval}  [{elapsed:.0f}s]")
    print(f"  ✓ saved → {out}")

    manifest[bench] = {
        'proxy_before':  proxy_b,
        'proxy_after':   proxy_a,
        'delta':         delta,
        'pct':           pct,
        'overlap_count': after['overlap_count'],
        'cd_accepts':    total_cd_accepts,
        'cd_evals':      total_cd_evals,
        'swap_accepts':  n_swap_acc,
        'swap_evals':    n_swap_eval,
        'elapsed_s':     elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benches', default='ibm01,ibm02,ibm03')
    ap.add_argument('--steps',   default='4,2,1,0.5,0.25',
                    help='Comma-separated step sizes (in grid cells)')
    ap.add_argument('--passes',  type=int, default=3,
                    help='Max passes per step (early-out on zero accepts)')
    ap.add_argument('--seed',    type=int, default=42)
    args = ap.parse_args()

    step_cells = [float(s) for s in args.steps.split(',')]
    benches = BENCHMARKS if args.benches == 'all' else \
              [b.strip() for b in args.benches.split(',')]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}; t0 = time.time()

    print(f"hard_cd_swap  steps={step_cells}  passes={args.passes}  "
          f"benches={benches}")

    for b in benches:
        try:
            process(b, manifest, step_cells, args.passes, args.seed)
        except Exception as e:
            print(f'  ERROR {b}: {e}')
            import traceback; traceback.print_exc()

    valid = [v for v in manifest.values() if 'proxy_after' in v]
    if valid:
        scores = [v['proxy_after']  for v in valid]
        before = [v['proxy_before'] for v in valid]
        avg_a, avg_b = sum(scores)/len(scores), sum(before)/len(before)
        avg_pct = (avg_a - avg_b) / avg_b * 100.0
        print(f"\n{'='*60}")
        print(f'  hard_cd_swap  {len(valid)}/{len(benches)} benches  '
              f'avg {avg_b:.4f} → {avg_a:.4f}  '
              f'(Δ{avg_a-avg_b:+.4f}, {avg_pct:+.1f}%)')
        print(f'  Output: {OUT_DIR}')
        print(f'  Elapsed: {(time.time()-t0)/60:.1f} min')
        print(f"{'='*60}")
    with open(str(OUT_DIR / 'hard_cd_swap_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)


if __name__ == '__main__':
    main()
