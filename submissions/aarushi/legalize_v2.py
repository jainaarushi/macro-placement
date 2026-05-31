#!/usr/bin/env python3
"""
legalize_v2 — strict overlap removal for hard macros.

legalize_overlaps_strict: iterative force-based push with random perturbation
  to escape local minima. Raises RuntimeError if overlaps remain.
verify_zero_overlaps: count overlapping macro pairs.
"""

import numpy as np
import torch
from macro_place.benchmark import Benchmark

_MARGIN = 0.01   # tiny gap — just enough to clear the overlap check


def verify_zero_overlaps(pos_t: torch.Tensor, benchmark: Benchmark) -> int:
    """Return number of overlapping hard-macro pairs."""
    pos = pos_t.numpy()
    nh  = benchmark.num_hard_macros
    sz  = benchmark.macro_sizes.numpy()
    count = 0
    for i in range(nh):
        for j in range(i + 1, nh):
            dx = abs(pos[i, 0] - pos[j, 0])
            dy = abs(pos[i, 1] - pos[j, 1])
            if dx < (sz[i, 0] + sz[j, 0]) / 2 and dy < (sz[i, 1] + sz[j, 1]) / 2:
                count += 1
    return count


def legalize_overlaps_strict(
    pos_t: torch.Tensor,
    benchmark: Benchmark,
    max_iters: int = 8000,
    margin: float = _MARGIN,
    noise_period: int = 400,
    noise_scale: float = 2.0,
    rng_seed: int = 0,
) -> torch.Tensor:
    """
    Push overlapping hard macros apart until zero overlaps remain.

    Uses half-push (split evenly between both free macros) with a tiny margin
    to avoid cascading on tight-canvas designs. Periodic random kicks to
    escape local minima when stuck.
    """
    rng = np.random.default_rng(rng_seed)
    pos   = pos_t.numpy().copy().astype(np.float64)
    nh    = benchmark.num_hard_macros
    sz    = benchmark.macro_sizes.numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.numpy().astype(bool)
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)

    def _clamp():
        for i in range(nh):
            if not fixed[i]:
                pos[i, 0] = np.clip(pos[i, 0], sz[i, 0] / 2, cw - sz[i, 0] / 2)
                pos[i, 1] = np.clip(pos[i, 1], sz[i, 1] / 2, ch - sz[i, 1] / 2)

    def _collect_overlaps():
        pairs = []
        for i in range(nh):
            for j in range(i + 1, nh):
                dx = abs(pos[i, 0] - pos[j, 0])
                dy = abs(pos[i, 1] - pos[j, 1])
                sx = (sz[i, 0] + sz[j, 0]) / 2
                sy = (sz[i, 1] + sz[j, 1]) / 2
                if dx < sx and dy < sy:
                    pairs.append((i, j, sx - dx, sy - dy))
        return pairs

    stuck_count = 0
    prev_n = None

    for iteration in range(max_iters):
        pairs = _collect_overlaps()
        n = len(pairs)
        if n == 0:
            break

        # Detect stuck (no progress for noise_period iters) → kick
        if n == prev_n:
            stuck_count += 1
        else:
            stuck_count = 0
        prev_n = n

        if stuck_count > 0 and stuck_count % noise_period == 0:
            # Scale kick by how stuck we are
            kick = noise_scale * margin * (1 + stuck_count // noise_period)
            for i in range(nh):
                if not fixed[i]:
                    pos[i, 0] += rng.uniform(-kick, kick)
                    pos[i, 1] += rng.uniform(-kick, kick)
            _clamp()
            continue

        for i, j, ox, oy in pairs:
            if ox <= oy:
                s = 1.0 if pos[i, 0] >= pos[j, 0] else -1.0
                if s == 0.0:
                    s = 1.0
                d = ox / 2 + margin
                if not fixed[i] and not fixed[j]:
                    pos[i, 0] += s * d
                    pos[j, 0] -= s * d
                elif not fixed[i]:
                    pos[i, 0] += s * (ox + margin)
                elif not fixed[j]:
                    pos[j, 0] -= s * (ox + margin)
            else:
                s = 1.0 if pos[i, 1] >= pos[j, 1] else -1.0
                if s == 0.0:
                    s = 1.0
                d = oy / 2 + margin
                if not fixed[i] and not fixed[j]:
                    pos[i, 1] += s * d
                    pos[j, 1] -= s * d
                elif not fixed[i]:
                    pos[i, 1] += s * (oy + margin)
                elif not fixed[j]:
                    pos[j, 1] -= s * (oy + margin)

        _clamp()

    result = torch.tensor(pos, dtype=torch.float32)
    n_left = verify_zero_overlaps(result, benchmark)
    if n_left > 0:
        raise RuntimeError(
            f"legalize_overlaps_strict: {n_left} overlaps remain after {max_iters} iters"
        )
    return result
