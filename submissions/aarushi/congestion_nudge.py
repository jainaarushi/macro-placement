#!/usr/bin/env python3
"""
Layer 1: Congestion nudge — greedy hill-climb post-pass (CPU only).

Moves hard macros out of the top-5% routing-hot grid cells toward cold cells.
Uses a strict-improve gate: a move is accepted only if the proxy cost strictly
decreases. Batches non-conflicting moves within each round.

Knobs:
  rounds=4, hot_frac=0.05, search_radius=4, max_macros_per_round=64
"""

import numpy as np
import torch
from typing import Tuple, List

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost, _set_placement
from macro_place._plc import PlacementCost


def _congestion_grid(plc: PlacementCost) -> np.ndarray:
    """Return per-cell routing congestion as (grid_row, grid_col) array."""
    rows, cols = plc.grid_row, plc.grid_col
    h = np.array(plc.H_routing_cong, dtype=np.float32).reshape(rows, cols)
    v = np.array(plc.V_routing_cong, dtype=np.float32).reshape(rows, cols)
    return np.maximum(h, v)


def _cell_of(pos, cw, ch, rows, cols) -> Tuple[int, int]:
    """Return (row, col) grid cell for a position (x, y)."""
    r = int(np.clip(pos[1] / (ch / rows), 0, rows - 1))
    c = int(np.clip(pos[0] / (cw / cols), 0, cols - 1))
    return r, c


def _cell_centre(r, c, cw, ch, rows, cols) -> Tuple[float, float]:
    bw = cw / cols
    bh = ch / rows
    return (c + 0.5) * bw, (r + 0.5) * bh


def congestion_nudge(
    positions: torch.Tensor,
    benchmark: Benchmark,
    plc: PlacementCost,
    rounds: int = 4,
    hot_frac: float = 0.05,
    search_radius: int = 4,
    max_macros_per_round: int = 64,
) -> torch.Tensor:
    """
    Apply congestion nudge to *positions* and return improved positions.
    Modifies a copy; the original is unchanged.
    """
    pos = positions.clone()
    cw, ch = benchmark.canvas_width, benchmark.canvas_height
    nh = benchmark.num_hard_macros
    fixed = benchmark.macro_fixed.numpy()
    sizes = benchmark.macro_sizes.numpy()
    rows, cols = plc.grid_row, plc.grid_col

    # Initial score
    cur_costs = compute_proxy_cost(pos, benchmark, plc)
    cur_score = cur_costs['proxy_cost']

    for rnd in range(rounds):
        # Refresh congestion map
        _set_placement(plc, pos, benchmark)
        plc.get_congestion_cost()
        cong = _congestion_grid(plc)

        # Identify hot threshold
        hot_thresh = float(np.percentile(cong, (1 - hot_frac) * 100))
        hot_cells  = set(zip(*np.where(cong >= hot_thresh)))

        # Find hard macros overlapping hot cells, rank by overlap heat
        macro_heat: List[Tuple[float, int]] = []
        for i in range(nh):
            if fixed[i]:
                continue
            r, c = _cell_of(pos[i].numpy(), cw, ch, rows, cols)
            heat = cong[r, c]
            if (r, c) in hot_cells:
                macro_heat.append((heat, i))

        macro_heat.sort(reverse=True)
        candidates = [idx for _, idx in macro_heat[:max_macros_per_round]]

        moves: List[Tuple[int, float, float]] = []  # (macro_idx, new_x, new_y)
        occupied = set()  # cells claimed this round

        for i in candidates:
            r0, c0 = _cell_of(pos[i].numpy(), cw, ch, rows, cols)
            best_move = None
            best_delta = 0.0  # must strictly improve

            for dr in range(-search_radius, search_radius + 1):
                for dc in range(-search_radius, search_radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    r1 = int(np.clip(r0 + dr, 0, rows - 1))
                    c1 = int(np.clip(c0 + dc, 0, cols - 1))
                    if (r1, c1) in hot_cells or (r1, c1) in occupied:
                        continue
                    if cong[r1, c1] >= cong[r0, c0]:
                        continue

                    nx, ny = _cell_centre(r1, c1, cw, ch, rows, cols)
                    # Clamp to valid macro bounds
                    nx = np.clip(nx, sizes[i,0]/2, cw - sizes[i,0]/2)
                    ny = np.clip(ny, sizes[i,1]/2, ch - sizes[i,1]/2)

                    trial = pos.clone()
                    trial[i] = torch.tensor([nx, ny])
                    trial_costs = compute_proxy_cost(trial, benchmark, plc)
                    delta = cur_score - trial_costs['proxy_cost']

                    if delta > best_delta:
                        best_delta = delta
                        best_move  = (i, nx, ny, r1, c1)

            if best_move is not None:
                mi, nx, ny, r1, c1 = best_move
                moves.append((mi, nx, ny))
                occupied.add((r1, c1))

        if not moves:
            break  # no improving moves in this round

        # Apply batch of non-conflicting moves
        for mi, nx, ny in moves:
            pos[mi] = torch.tensor([nx, ny])

        new_costs = compute_proxy_cost(pos, benchmark, plc)
        new_score = new_costs['proxy_cost']
        gain = cur_score - new_score
        print(f"    nudge round {rnd+1}: {len(moves)} moves  "
              f"Δproxy={gain:+.4f}  proxy={new_score:.4f}")
        cur_score = new_score

    return pos
