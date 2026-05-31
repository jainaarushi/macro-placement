#!/usr/bin/env python3
"""
Layer 0: GPU-accelerated hard-macro global placement via Nesterov optimizer.

Key design decisions vs naive Nesterov:
  1. Only HARD macros get gradient — soft macros are fixed at initial.plc
     (initial.plc already has good soft-macro positions; they should not move).
  2. Pairwise overlap penalty for hard macros (no bin-density for hard macros
     because their area utilization is far below any reasonable target density).
  3. Density warmup: overlap weight ramps 0.001→1.0 over first 40% of iters
     so HPWL first clusters macros near their nets, then overlap is resolved.
  4. Start from random positions for hard macros (4 seeds × 3 overlap weights).
     Also try the legalized initial.plc as a no-perturbation candidate.
  5. Keep the best result per bench across all trials.

Runtime estimate on A100: ~30-50 min for all 17 benches.
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
SEEDS           = [1, 7, 42, 1234]
OVERLAP_WEIGHTS = [0.5, 1.0, 2.0]   # hard-macro pairwise overlap penalty weight
GAMMA           = 4.0                # LSE smoothing
NUM_ITERS       = 400
STOP_OVERLAP    = 1e-4               # stop when total overlap area < this (um²)
LR              = 0.005
BOUNDARY_W      = 5e2                # boundary out-of-canvas penalty

TESTCASE_ROOT = REPO / 'external' / 'MacroPlacement' / 'Testcases' / 'ICCAD04'
PT_DIR        = REPO / 'benchmarks' / 'processed' / 'public'
OUTPUT_DIR    = Path('/tmp/dp_overnight_repair')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[overnight] device={DEVICE}  gpu={'A100' if torch.cuda.is_available() else 'CPU'}")


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
    """
    All macros contribute to the wirelength, but only those with
    requires_grad=True receive gradient. We concatenate hard (grad)
    and soft (no-grad) positions before calling this function.
    """
    if net_idx is None:
        return pos.new_zeros(1).squeeze()
    BIG = 1e6
    xy = pos[net_idx]
    x, y = xy[:, :, 0], xy[:, :, 1]
    x_fwd = x.masked_fill(~net_mask, -BIG)
    x_rev = x.masked_fill(~net_mask,  BIG)
    y_fwd = y.masked_fill(~net_mask, -BIG)
    y_rev = y.masked_fill(~net_mask,  BIG)
    g = gamma
    lse_x = g*torch.logsumexp(x_fwd/g,1) - g*torch.logsumexp(-x_rev/g,1)
    lse_y = g*torch.logsumexp(y_fwd/g,1) - g*torch.logsumexp(-y_rev/g,1)
    return (net_w * (lse_x + lse_y)).sum()


# ── Pairwise hard-macro overlap penalty ───────────────────────────────────────

def hard_overlap_penalty(hard_pos, hard_sizes):
    """
    Smooth pairwise overlap area sum for hard macros.
    hard_pos   : [Nh, 2]  requires_grad=True
    hard_sizes : [Nh, 2]  no grad
    O(Nh²) — fine for Nh ≤ 800.
    """
    N  = hard_pos.shape[0]
    if N < 2:
        return hard_pos.new_zeros(1).squeeze()

    hw = hard_sizes[:, 0] / 2   # [N]
    hh = hard_sizes[:, 1] / 2

    px = hard_pos[:, 0]          # [N]
    py = hard_pos[:, 1]

    # All pairs
    dx  = (px.unsqueeze(0) - px.unsqueeze(1)).abs()   # [N,N]
    dy  = (py.unsqueeze(0) - py.unsqueeze(1)).abs()

    sx  = hw.unsqueeze(0) + hw.unsqueeze(1)            # [N,N]
    sy  = hh.unsqueeze(0) + hh.unsqueeze(1)

    ov_x = (sx - dx).clamp(min=0)
    ov_y = (sy - dy).clamp(min=0)
    overlap = ov_x * ov_y                              # [N,N]

    # Upper-triangular only (avoid double-counting)
    triu = torch.triu(torch.ones(N, N, dtype=torch.bool, device=hard_pos.device), diagonal=1)
    return overlap[triu].sum()


# ── Boundary penalty ──────────────────────────────────────────────────────────

def boundary_loss(hard_pos, hard_sizes, cw, ch):
    hw, hh = hard_sizes[:, 0] / 2, hard_sizes[:, 1] / 2
    x, y   = hard_pos[:, 0], hard_pos[:, 1]
    return ((hw - x).clamp(min=0)**2 + (x + hw - cw).clamp(min=0)**2
          + (hh - y).clamp(min=0)**2 + (y + hh - ch).clamp(min=0)**2).sum()


# ── Nesterov optimizer ────────────────────────────────────────────────────────

def nesterov_place(benchmark, hard_init, soft_fixed, overlap_weight, device):
    """
    Optimize only hard macro positions; soft macros are fixed.

    hard_init   : [Nh, 2]  initial hard macro positions
    soft_fixed  : [Ns, 2]  fixed soft macro positions (no grad)
    """
    cw, ch  = benchmark.canvas_width, benchmark.canvas_height
    nh      = benchmark.num_hard_macros
    h_sizes = benchmark.macro_sizes[:nh].to(device)
    fixed   = benchmark.macro_fixed[:nh].to(device)  # fixed hard macros
    scale   = math.hypot(cw, ch)

    net_idx, net_mask, net_w = build_net_tensors(benchmark, device)
    soft_t = soft_fixed.to(device)  # [Ns, 2] no grad

    warmup_end = int(NUM_ITERS * 0.4)

    x_h = hard_init.clone().to(device)   # hard macro positions (will update)
    v_h = x_h.clone()
    a   = 1.0

    for k in range(NUM_ITERS):
        # Overlap weight warmup
        ow_k = overlap_weight * (0.001 + 0.999 * min(k / warmup_end, 1.0))

        y_h = x_h.detach().requires_grad_(True)  # [Nh, 2]

        # Concatenate hard + soft into full position tensor for HPWL
        # Only hard positions have grad; soft positions are detached
        full_pos = torch.cat([y_h, soft_t], dim=0)  # [Nh+Ns, 2]

        wl  = hpwl_loss(full_pos, net_idx, net_mask, net_w, GAMMA) / scale
        ovl = hard_overlap_penalty(y_h, h_sizes)
        bnd = boundary_loss(y_h, h_sizes, cw, ch)

        loss = wl + ow_k * ovl + BOUNDARY_W * bnd
        loss.backward()

        with torch.no_grad():
            g = y_h.grad.clone()
            g[fixed] = 0.0

            a_new = (1.0 + math.sqrt(1.0 + 4.0*a*a)) / 2.0
            beta  = (a - 1.0) / a_new
            a     = a_new

            v_new = y_h - LR * g
            x_h   = v_new + beta * (v_new - v_h)
            v_h   = v_new

            hw = h_sizes[:, 0] / 2; hh = h_sizes[:, 1] / 2
            x_h[:, 0] = x_h[:, 0].clamp(hw, cw - hw)
            x_h[:, 1] = x_h[:, 1].clamp(hh, ch - hh)

        if float(ovl.detach()) < STOP_OVERLAP and k > 80:
            break

    return x_h.detach().cpu()


# ── Legalizer ─────────────────────────────────────────────────────────────────

def legalize(hard_pos_t, benchmark, max_passes=400):
    """Remove overlaps from hard macros only."""
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


# ── Per-benchmark sweep ───────────────────────────────────────────────────────

def run_benchmark(name):
    print(f"\n{'='*64}\n  {name}\n{'='*64}")

    pt_path   = PT_DIR / f'{name}.pt'
    benchmark = Benchmark.load(str(pt_path))
    _, plc    = load_benchmark_from_dir(str(TESTCASE_ROOT / name))

    nh   = benchmark.num_hard_macros
    cw, ch = benchmark.canvas_width, benchmark.canvas_height
    h_sz = benchmark.macro_sizes[:nh]
    hw   = h_sz[:, 0] / 2
    hh_  = h_sz[:, 1] / 2
    fixed = benchmark.macro_fixed[:nh]

    # Fixed soft macro positions from initial.plc
    soft_fixed = benchmark.macro_positions[nh:].clone()  # [Ns, 2]

    # --- Baseline: legalized initial.plc ---
    initial_pos = benchmark.macro_positions.clone()
    leg_init_h  = legalize(initial_pos[:nh].clone(), benchmark)
    full_leg_init = torch.cat([leg_init_h, soft_fixed], dim=0)
    init_costs = compute_proxy_cost(full_leg_init, benchmark, plc)
    print(f"  legalized initial: proxy={init_costs['proxy_cost']:.4f}  "
          f"overlaps={init_costs['overlap_count']}")

    best_score     = float(init_costs['proxy_cost'])
    best_positions = full_leg_init.clone()
    best_cfg = {
        'bench': name, 'seed': 0, 'overlap_weight': 0.0,
        'proxy_cost':      float(init_costs['proxy_cost']),
        'wirelength_cost': float(init_costs['wirelength_cost']),
        'density_cost':    float(init_costs['density_cost']),
        'congestion_cost': float(init_costs['congestion_cost']),
        'overlap_count':   int(init_costs['overlap_count']),
    }
    torch.save({'positions': full_leg_init, 'score': best_score, 'costs': init_costs,
                'bench': name, 'seed': 0, 'density_weight': 0.0},
               str(OUTPUT_DIR / f'{name}_s0_init.pt'))

    # --- Nesterov trials: 4 seeds × 3 overlap weights ---
    for seed in SEEDS:
        for ow in OVERLAP_WEIGHTS:
            tag = f'{name}_s{seed}_ow{ow:.1f}'
            print(f"  {tag} ... ", end='', flush=True)

            torch.manual_seed(seed)
            np.random.seed(seed)

            # Random initialization for hard macros
            h_init = torch.zeros(nh, 2)
            mov = ~fixed
            n_mov = int(mov.sum())
            h_init[mov, 0] = torch.rand(n_mov) * (cw - 2*hw[mov]) + hw[mov]
            h_init[mov, 1] = torch.rand(n_mov) * (ch - 2*hh_[mov]) + hh_[mov]
            h_init[~mov]   = benchmark.macro_positions[:nh][~mov]  # keep fixed

            t0  = time.time()
            opt_h = nesterov_place(benchmark, h_init, soft_fixed, ow, DEVICE)
            leg_h = legalize(opt_h, benchmark)
            full  = torch.cat([leg_h, soft_fixed], dim=0)
            costs = compute_proxy_cost(full, benchmark, plc)
            score = float(costs['proxy_cost'])
            print(f"proxy={score:.4f}  [{time.time()-t0:.1f}s]")

            torch.save({'positions': full, 'score': score, 'costs': costs,
                        'bench': name, 'seed': int(seed), 'density_weight': float(ow)},
                       str(OUTPUT_DIR / f'{tag}.pt'))

            if score < best_score:
                best_score     = score
                best_positions = full.clone()
                best_cfg = {
                    'bench': name, 'seed': int(seed), 'overlap_weight': float(ow),
                    'proxy_cost':      float(costs['proxy_cost']),
                    'wirelength_cost': float(costs['wirelength_cost']),
                    'density_cost':    float(costs['density_cost']),
                    'congestion_cost': float(costs['congestion_cost']),
                    'overlap_count':   int(costs['overlap_count']),
                }

    print(f"  → BEST  proxy={best_score:.4f}  "
          f"seed={best_cfg['seed']}  ow={best_cfg.get('overlap_weight',0):.1f}")
    return best_cfg, best_positions


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    best_per_bench = {}
    total_t = time.time()

    for name in BENCHMARKS:
        cfg, _ = run_benchmark(name)
        best_per_bench[name] = cfg

    out = OUTPUT_DIR / 'best_per_bench.json'
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
