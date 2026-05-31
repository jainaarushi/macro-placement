#!/usr/bin/env python3
"""
Layer 2: Soft-macro diffusion — Metropolis trust-region optimizer (CPU only).

Each iteration:
  1. Identify routing-hot grid cells.
  2. For each movable soft macro, compute a force vector:
       F = alpha * F_repulsive  +  beta * F_attractive
     F_repulsive: push away from hot-cell centres (weighted by congestion excess)
     F_attractive: pull toward net-centroid of connected nodes
  3. Propose new position = current + clipped_force
  4. Evaluate proxy cost (real PlacementCost scorer).
  5. Metropolis accept/reject.

Knobs:
  alpha=1.0, beta=0.3, t_init_frac=3 (of canvas diagonal), max_iters=15
"""

import math, random
import numpy as np
import torch
from collections import defaultdict
from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost, _set_placement
from macro_place._plc import PlacementCost


def _get_congestion(plc: PlacementCost, rows: int, cols: int) -> np.ndarray:
    _set_placement.__module__   # ensure plc is updated externally
    h = np.array(plc.H_routing_cong, dtype=np.float32).reshape(rows, cols)
    v = np.array(plc.V_routing_cong, dtype=np.float32).reshape(rows, cols)
    return np.maximum(h, v)


def _build_adjacency(benchmark: Benchmark) -> dict:
    """Precompute node -> list of connected node indices (excluding self)."""
    adj = defaultdict(list)
    for net in benchmark.net_nodes:
        net_list = net.tolist()
        for i, a in enumerate(net_list):
            for b in net_list:
                if b != a:
                    adj[a].append(b)
    return adj


def soft_macro_diffusion(
    positions: torch.Tensor,
    benchmark: Benchmark,
    plc: PlacementCost,
    alpha: float = 1.0,
    beta: float = 0.3,
    t_init_frac: float = 3.0,
    max_iters: int = 15,
) -> torch.Tensor:
    """
    Apply Metropolis trust-region diffusion to soft macros.
    Returns improved positions tensor.
    """
    pos = positions.clone()
    cw, ch = benchmark.canvas_width, benchmark.canvas_height
    rows, cols = plc.grid_row, plc.grid_col
    bw, bh = cw / cols, ch / rows
    ns  = benchmark.num_soft_macros
    nh  = benchmark.num_hard_macros
    szs = benchmark.macro_sizes.numpy()

    if ns == 0:
        return pos

    # Precompute adjacency once (avoids O(n_nets) scan per macro per iter)
    adj = _build_adjacency(benchmark)

    diag  = math.hypot(cw, ch)
    T     = t_init_frac * diag / 100.0
    T_min = diag / 1000.0
    cool  = (T_min / T) ** (1.0 / max(max_iters - 1, 1))

    _set_placement(plc, pos, benchmark)
    plc.get_congestion_cost()
    cur_costs = compute_proxy_cost(pos, benchmark, plc)
    cur_score = cur_costs['proxy_cost']
    best_score = cur_score
    best_pos   = pos.clone()

    pos_np = pos.numpy()  # work in numpy, sync to tensor only on accept

    for it in range(max_iters):
        _set_placement(plc, pos, benchmark)
        plc.get_congestion_cost()
        cong = _get_congestion(plc, rows, cols)
        cong_mean = cong.mean()

        soft_indices = list(range(ns))
        random.shuffle(soft_indices)
        batch = soft_indices[:32]

        for si in batch:
            gi = nh + si
            if benchmark.macro_fixed[gi]:
                continue

            cx, cy = float(pos_np[gi, 0]), float(pos_np[gi, 1])

            # Repulsive force from hot cells
            r0 = int(np.clip(cy / bh, 0, rows - 1))
            c0 = int(np.clip(cx / bw, 0, cols - 1))
            fr = np.zeros(2, dtype=np.float64)
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    r1 = int(np.clip(r0 + dr, 0, rows - 1))
                    c1 = int(np.clip(c0 + dc, 0, cols - 1))
                    excess = max(0.0, float(cong[r1, c1]) - float(cong_mean))
                    if excess <= 0:
                        continue
                    hcx, hcy = (c1 + 0.5) * bw, (r1 + 0.5) * bh
                    dx, dy   = cx - hcx, cy - hcy
                    dist2    = dx*dx + dy*dy + 1e-6
                    fr += (excess / dist2) * np.array([dx, dy])

            # Attractive force toward net centroid (O(1) via precomputed adj)
            nbrs = adj.get(gi, [])
            if nbrs:
                centroid = pos_np[nbrs].mean(axis=0)
            else:
                centroid = pos_np[gi]
            fa = centroid - np.array([cx, cy])

            force = alpha * fr + beta * fa
            fn = np.linalg.norm(force)
            if fn > T:
                force = force * T / fn

            nx = float(np.clip(cx + force[0], szs[gi,0]/2, cw - szs[gi,0]/2))
            ny = float(np.clip(cy + force[1], szs[gi,1]/2, ch - szs[gi,1]/2))

            old_xy = pos_np[gi].copy()
            pos_np[gi] = [nx, ny]
            pos = torch.from_numpy(pos_np.copy())
            trial_costs = compute_proxy_cost(pos, benchmark, plc)
            delta = trial_costs['proxy_cost'] - cur_score

            if delta < 0 or (T > 1e-9 and random.random() < math.exp(-delta / T)):
                cur_score = trial_costs['proxy_cost']
                if cur_score < best_score:
                    best_score = cur_score
                    best_pos   = pos.clone()
            else:
                pos_np[gi] = old_xy
                pos = torch.from_numpy(pos_np.copy())

        T = max(T * cool, T_min)

        if (it + 1) % 5 == 0:
            print(f"    diffusion iter {it+1}/{max_iters}  "
                  f"proxy={cur_score:.4f}  best={best_score:.4f}  T={T:.4f}")

    return best_pos
