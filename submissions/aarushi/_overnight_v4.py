#!/usr/bin/env python3
"""
Layer 0 v4: ePlace-style density-spreading from initial.plc.

Core insight vs v3: start from initial.plc (already good HPWL), add a bin-density
penalty that creates a spreading force.  Both hard and soft macros move.
Hard macros also have a pairwise overlap penalty.  Density weight warms up
from near-zero so HPWL first holds macros near initial positions, then
density gradually pushes them into less-crowded regions.

Runtime estimate on A100: ~20-40 min for all 17 benches.
"""

import sys, os, json, time, math
import torch
import numpy as np
from pathlib import Path

REPO = Path('/home/ubuntu/macro-place-challenge-2026')
sys.path.insert(0, str(REPO))

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from macro_place.loader import load_benchmark_from_dir

# ── Config ────────────────────────────────────────────────────────────────────

BENCHMARKS = [
    'ibm01','ibm02','ibm03','ibm04','ibm06','ibm07','ibm08',
    'ibm09','ibm10','ibm11','ibm12','ibm13','ibm14','ibm15',
    'ibm16','ibm17','ibm18',
]

GAMMA          = 4.0      # LSE smoothing
NUM_ITERS      = 2000
LR             = 0.003
BOUNDARY_W     = 1e3      # out-of-canvas penalty
DENSITY_W_MAX  = 1.5      # peak density weight
OVERLAP_W      = 2.0      # hard-macro pairwise overlap weight
WARMUP_FRAC    = 0.35     # ramp density weight over first 35% of iters

# Multiple trials: vary density weight ceiling
DENSITY_CEILINGS = [0.8, 1.5, 3.0]

TESTCASE_ROOT = REPO / 'external' / 'MacroPlacement' / 'Testcases' / 'ICCAD04'
PT_DIR        = REPO / 'benchmarks' / 'processed' / 'public'
OUTPUT_DIR    = Path('/tmp/dp_overnight_v4')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[overnight_v4] device={DEVICE}")


# ── Net preprocessing ─────────────────────────────────────────────────────────

def build_net_tensors(benchmark, device):
    valid = [(n, w.item()) for n, w in zip(benchmark.net_nodes, benchmark.net_weights)
             if len(n) >= 2]
    if not valid:
        return None, None, None
    nets, wts = zip(*valid)
    K = max(len(n) for n in nets)
    M = len(nets)
    padded = torch.zeros((M, K), dtype=torch.long,  device=device)
    mask   = torch.zeros((M, K), dtype=torch.bool,  device=device)
    for i, n in enumerate(nets):
        padded[i, :len(n)] = n.to(device)
        mask  [i, :len(n)] = True
    return padded, mask, torch.tensor(wts, dtype=torch.float32, device=device)


# ── LSE-HPWL ─────────────────────────────────────────────────────────────────

def hpwl_loss(pos, net_idx, net_mask, net_w, gamma):
    if net_idx is None:
        return pos.new_zeros(1).squeeze()
    BIG = 1e6
    xy  = pos[net_idx]
    x, y = xy[:, :, 0], xy[:, :, 1]
    x_fwd = x.masked_fill(~net_mask, -BIG)
    x_rev = x.masked_fill(~net_mask,  BIG)
    y_fwd = y.masked_fill(~net_mask, -BIG)
    y_rev = y.masked_fill(~net_mask,  BIG)
    g = gamma
    lse_x = g*torch.logsumexp(x_fwd/g,1) - g*torch.logsumexp(-x_rev/g,1)
    lse_y = g*torch.logsumexp(y_fwd/g,1) - g*torch.logsumexp(-y_rev/g,1)
    return (net_w * (lse_x + lse_y)).sum()


# ── Bin density (differentiable bilinear scatter) ────────────────────────────

def bin_density_loss(pos, sizes, grid_cols, grid_rows, cw, ch):
    """
    Differentiable bin-density penalty via bilinear scatter.
    pos   : [N, 2] (requires_grad where applicable)
    sizes : [N, 2] (no grad)
    Returns scalar penalty = sum((density_bin - 1)^2) where density > 1.
    Gradient flows through the bilinear weights (not through bin indices).
    """
    N   = pos.shape[0]
    bw  = cw / grid_cols
    bh  = ch / grid_rows
    B   = grid_rows * grid_cols

    x   = pos[:, 0].clamp(bw * 0.01, cw - bw * 0.01)
    y   = pos[:, 1].clamp(bh * 0.01, ch - bh * 0.01)
    area = (sizes[:, 0] * sizes[:, 1]) / (bw * bh)  # normalised by bin area

    # Bin float coordinates
    bxf = x / bw        # [N]  in [0, grid_cols]
    byf = y / bh        # [N]  in [0, grid_rows]

    # Integer bin indices (no grad)
    bx0 = bxf.detach().floor().long().clamp(0, grid_cols - 1)
    by0 = byf.detach().floor().long().clamp(0, grid_rows - 1)
    bx1 = (bx0 + 1).clamp(max=grid_cols - 1)
    by1 = (by0 + 1).clamp(max=grid_rows - 1)

    # Bilinear weights (carry gradient w.r.t. pos)
    wx1 = bxf - bx0.float()   # [N]
    wy1 = byf - by0.float()
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    # Scatter density contributions into bins
    density = pos.new_zeros(B)
    for bxi, byi, wx, wy in (
        (bx0, by0, wx0, wy0),
        (bx0, by1, wx0, wy1),
        (bx1, by0, wx1, wy0),
        (bx1, by1, wx1, wy1),
    ):
        idx = (byi * grid_cols + bxi).clamp(0, B - 1)
        density = density + torch.zeros(B, device=pos.device,
                                        dtype=pos.dtype).scatter_add(
            0, idx, area * wx * wy)

    # Penalty: excess above capacity 1.0
    excess = (density - 1.0).clamp(min=0.0)
    return (excess ** 2).sum()


# ── Pairwise hard-macro overlap penalty ───────────────────────────────────────

def hard_overlap_penalty(hard_pos, hard_sizes):
    N = hard_pos.shape[0]
    if N < 2:
        return hard_pos.new_zeros(1).squeeze()
    hw = hard_sizes[:, 0] / 2
    hh = hard_sizes[:, 1] / 2
    px, py = hard_pos[:, 0], hard_pos[:, 1]
    dx  = (px.unsqueeze(0) - px.unsqueeze(1)).abs()
    dy  = (py.unsqueeze(0) - py.unsqueeze(1)).abs()
    sx  = hw.unsqueeze(0) + hw.unsqueeze(1)
    sy  = hh.unsqueeze(0) + hh.unsqueeze(1)
    ov_x = (sx - dx).clamp(min=0)
    ov_y = (sy - dy).clamp(min=0)
    triu = torch.triu(torch.ones(N, N, dtype=torch.bool, device=hard_pos.device), diagonal=1)
    return (ov_x * ov_y)[triu].sum()


# ── Boundary penalty ──────────────────────────────────────────────────────────

def boundary_loss(pos, sizes, cw, ch):
    hw = sizes[:, 0] / 2
    hh = sizes[:, 1] / 2
    x, y = pos[:, 0], pos[:, 1]
    return ((hw - x).clamp(min=0)**2 + (x + hw - cw).clamp(min=0)**2
          + (hh - y).clamp(min=0)**2 + (y + hh - ch).clamp(min=0)**2).sum()


# ── Legalize hard macros ───────────────────────────────────────────────────────

def legalize(hard_pos_t, benchmark, max_passes=400):
    pos   = hard_pos_t.numpy().copy()
    nh    = benchmark.num_hard_macros
    sz    = benchmark.macro_sizes.numpy()[:nh]
    fixed = benchmark.macro_fixed.numpy()[:nh]
    cw, ch = benchmark.canvas_width, benchmark.canvas_height
    for _ in range(max_passes):
        moved = False
        for i in range(nh):
            if fixed[i]: continue
            for j in range(i + 1, nh):
                dx = abs(pos[i,0] - pos[j,0])
                dy = abs(pos[i,1] - pos[j,1])
                sx = (sz[i,0] + sz[j,0]) / 2 + 1e-3
                sy = (sz[i,1] + sz[j,1]) / 2 + 1e-3
                if dx < sx and dy < sy:
                    ox, oy = sx - dx, sy - dy
                    if ox <= oy:
                        d = ox/2 + 1e-3; s = 1 if pos[i,0] > pos[j,0] else -1
                        if not fixed[i]: pos[i,0] += s * d
                        if not fixed[j]: pos[j,0] -= s * d
                    else:
                        d = oy/2 + 1e-3; s = 1 if pos[i,1] > pos[j,1] else -1
                        if not fixed[i]: pos[i,1] += s * d
                        if not fixed[j]: pos[j,1] -= s * d
                    moved = True
        for i in range(nh):
            if fixed[i]: continue
            pos[i,0] = np.clip(pos[i,0], sz[i,0]/2, cw - sz[i,0]/2)
            pos[i,1] = np.clip(pos[i,1], sz[i,1]/2, ch - sz[i,1]/2)
        if not moved:
            break
    return torch.tensor(pos, dtype=torch.float32)


# ── Nesterov optimizer ────────────────────────────────────────────────────────

def nesterov_place(benchmark, init_pos, density_w_ceiling, device):
    """
    Nesterov gradient descent optimizing all macros (hard + soft).

    init_pos         : [Nh+Ns, 2]  starting positions (from initial.plc)
    density_w_ceiling: float        peak density weight
    """
    cw, ch = benchmark.canvas_width, benchmark.canvas_height
    nh     = benchmark.num_hard_macros
    total  = benchmark.num_hard_macros + benchmark.num_soft_macros
    sizes  = benchmark.macro_sizes[:total].to(device)
    fixed  = benchmark.macro_fixed[:total].to(device)  # bool [total]
    scale  = math.hypot(cw, ch)
    gcols  = benchmark.grid_cols
    grows  = benchmark.grid_rows

    net_idx, net_mask, net_w = build_net_tensors(benchmark, device)

    warmup_end = int(NUM_ITERS * WARMUP_FRAC)

    x = init_pos.clone().to(device)   # [total, 2]
    v = x.clone()
    a = 1.0

    for k in range(NUM_ITERS):
        # Density weight warmup
        t_frac = min(k / max(warmup_end, 1), 1.0)
        dw_k = density_w_ceiling * (0.002 + 0.998 * t_frac)

        y = x.detach().requires_grad_(True)   # [total, 2]

        wl  = hpwl_loss(y, net_idx, net_mask, net_w, GAMMA) / scale
        den = bin_density_loss(y, sizes, gcols, grows, cw, ch)
        ovl = hard_overlap_penalty(y[:nh], sizes[:nh])
        bnd = boundary_loss(y, sizes, cw, ch)

        loss = wl + dw_k * den + OVERLAP_W * ovl + BOUNDARY_W * bnd
        loss.backward()

        with torch.no_grad():
            g = y.grad.clone()
            g[fixed] = 0.0   # do not move fixed nodes

            a_new = (1.0 + math.sqrt(1.0 + 4.0*a*a)) / 2.0
            beta  = (a - 1.0) / a_new
            a     = a_new

            v_new = y - LR * g
            x     = v_new + beta * (v_new - v)
            v     = v_new

            # Clamp to canvas
            hw = sizes[:, 0] / 2; hh_ = sizes[:, 1] / 2
            x[:, 0] = x[:, 0].clamp(hw, cw - hw)
            x[:, 1] = x[:, 1].clamp(hh_, ch - hh_)

    return x.detach().cpu()


# ── Per-benchmark run ─────────────────────────────────────────────────────────

def run_benchmark(name):
    print(f"\n{'='*64}\n  {name}\n{'='*64}")

    pt_path   = PT_DIR / f'{name}.pt'
    benchmark = Benchmark.load(str(pt_path))
    _, plc    = load_benchmark_from_dir(str(TESTCASE_ROOT / name))

    nh    = benchmark.num_hard_macros
    total = nh + benchmark.num_soft_macros

    # ── Baseline: legalized initial.plc ──────────────────────────────────
    init_pos    = benchmark.macro_positions[:total].clone()
    leg_hard    = legalize(init_pos[:nh].clone(), benchmark)
    full_init   = torch.cat([leg_hard, init_pos[nh:]], dim=0)
    init_costs  = compute_proxy_cost(full_init, benchmark, plc)
    best_score  = float(init_costs['proxy_cost'])
    best_pos    = full_init.clone()
    best_cfg    = {
        'bench': name, 'trial': 'init', 'density_ceiling': 0.0,
        'proxy_cost': best_score,
        'wirelength_cost': float(init_costs['wirelength_cost']),
        'density_cost':    float(init_costs['density_cost']),
        'congestion_cost': float(init_costs['congestion_cost']),
        'overlap_count':   int(init_costs['overlap_count']),
    }
    print(f"  init: proxy={best_score:.4f}  wl={init_costs['wirelength_cost']:.4f}  "
          f"den={init_costs['density_cost']:.4f}  cong={init_costs['congestion_cost']:.4f}")

    torch.save({'positions': full_init, 'score': best_score, 'costs': dict(init_costs),
                'bench': name, 'trial': 'init', 'density_ceiling': 0.0},
               str(OUTPUT_DIR / f'{name}_init.pt'))

    # ── Nesterov trials with density spreading ───────────────────────────
    for dc in DENSITY_CEILINGS:
        tag = f'{name}_dc{dc:.1f}'
        print(f"  {tag} ... ", end='', flush=True)
        t0 = time.time()

        opt_pos = nesterov_place(benchmark, full_init, dc, DEVICE)

        # Legalize only hard macros
        leg_hard_opt = legalize(opt_pos[:nh].clone(), benchmark)
        full_opt = torch.cat([leg_hard_opt, opt_pos[nh:]], dim=0)

        costs = compute_proxy_cost(full_opt, benchmark, plc)
        score = float(costs['proxy_cost'])
        print(f"proxy={score:.4f}  wl={costs['wirelength_cost']:.4f}  "
              f"den={costs['density_cost']:.4f}  cong={costs['congestion_cost']:.4f}  "
              f"[{time.time()-t0:.1f}s]")

        torch.save({'positions': full_opt, 'score': score, 'costs': dict(costs),
                    'bench': name, 'trial': tag, 'density_ceiling': float(dc)},
                   str(OUTPUT_DIR / f'{tag}.pt'))

        if score < best_score:
            best_score = score
            best_pos   = full_opt.clone()
            best_cfg = {
                'bench': name, 'trial': tag, 'density_ceiling': float(dc),
                'proxy_cost':      score,
                'wirelength_cost': float(costs['wirelength_cost']),
                'density_cost':    float(costs['density_cost']),
                'congestion_cost': float(costs['congestion_cost']),
                'overlap_count':   int(costs['overlap_count']),
            }

    print(f"  → BEST  proxy={best_score:.4f}  trial={best_cfg['trial']}")
    return best_cfg, best_pos


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    best_per_bench = {}
    total_t = time.time()

    for name in BENCHMARKS:
        try:
            cfg, _ = run_benchmark(name)
            best_per_bench[name] = cfg
        except Exception as e:
            print(f"  ERROR {name}: {e}")
            import traceback; traceback.print_exc()

    out = OUTPUT_DIR / 'best_per_bench_v4.json'
    with open(str(out), 'w') as f:
        json.dump(best_per_bench, f, indent=2)

    avg     = sum(v['proxy_cost'] for v in best_per_bench.values()) / len(best_per_bench)
    elapsed = (time.time() - total_t) / 60
    print(f"\n{'='*64}")
    print(f"  DONE  avg proxy={avg:.4f}  total={elapsed:.1f} min")
    print(f"  best_per_bench → {out}")
    print(f"{'='*64}")


if __name__ == '__main__':
    main()
