#!/usr/bin/env python3
"""
Hotness-Weighted Spectral Placement (HWSP).

Solves for the joint optimal positions of all soft macros via a constrained
quadratic minimization with closed-form linear solve at each iteration.

Math:
  Build hypergraph H = (V, N) where V = macros + ports, N = nets.
  Clique-expand each net of k pins into k(k-1)/2 weighted edges with
    w_ij = Σ_{n: i,j∈pins(n)} hotness(n) / (k_n - 1)
  Construct sparse Laplacian L = D - A.
  Partition V = V_free (soft macros) ∪ V_fix (hard macros + ports).
  Block-partition L:
    L = [ L_ff  L_fb ]   →   min over p_free of (1/2) p^T L p
        [ L_bf  L_bb ]       with p_fix held constant
  Solution is the linear system:
    L_ff p_free = -L_fb p_fix
  Solve with conjugate gradient + Jacobi preconditioner.

Iterate with hot-cell feedback (weights change as positions change).
Each iteration also clips solved positions to canvas and applies damping.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective  import compute_proxy_cost, _set_placement
from macro_place._plc       import PlacementCost


def _hot_mask(plc: PlacementCost, rows: int, cols: int,
              top_pct: float = 0.05) -> np.ndarray:
    h    = np.array(plc.H_routing_cong, dtype=np.float32).reshape(rows, cols)
    v    = np.array(plc.V_routing_cong, dtype=np.float32).reshape(rows, cols)
    comb = np.concatenate([h.ravel(), v.ravel()])
    thr  = np.percentile(comb, (1.0 - top_pct) * 100.0)
    return np.maximum(h, v) >= thr


def _proxy(pos: torch.Tensor, benchmark: Benchmark, plc: PlacementCost) -> float:
    _set_placement(plc, pos, benchmark)
    plc.get_congestion_cost()
    return float(compute_proxy_cost(pos, benchmark, plc)['proxy_cost'])


def _build_extended_pos(pos_np: np.ndarray, benchmark: Benchmark) -> np.ndarray:
    if benchmark.port_positions.shape[0] == 0:
        return pos_np
    return np.concatenate([pos_np, benchmark.port_positions.numpy()], axis=0)


def _build_hot_laplacian(benchmark: Benchmark,
                         pos_ext: np.ndarray,
                         hot_mask: np.ndarray,
                         bw: float, bh: float,
                         rows: int, cols: int,
                         baseline_w: float = 1.0):
    """
    Build sparse Laplacian L = D - A.
    Edge weight from net n with k pins:  w_n = hotness(n) + baseline_w  (per edge: w_n / (k-1))
    """
    n_macros = benchmark.num_hard_macros + benchmark.num_soft_macros
    n_ports  = benchmark.port_positions.shape[0]
    n_v      = n_macros + n_ports

    rows_l = []
    cols_l = []
    vals_l = []

    for net in benchmark.net_nodes:
        pins = net.tolist() if hasattr(net, 'tolist') else list(net)
        k = len(pins)
        if k < 2:
            continue

        coords = pos_ext[pins]
        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        cmin_c = int(max(0, xmin // bw))
        cmax_c = int(min(cols - 1, xmax // bw))
        cmin_r = int(max(0, ymin // bh))
        cmax_r = int(min(rows - 1, ymax // bh))
        hotness = int(hot_mask[cmin_r:cmax_r+1, cmin_c:cmax_c+1].sum())

        edge_w = (hotness + baseline_w) / (k - 1)

        # Add all pairs in clique
        for i in range(k):
            pi = pins[i]
            for j in range(i + 1, k):
                pj = pins[j]
                rows_l.append(pi); cols_l.append(pj); vals_l.append(edge_w)
                rows_l.append(pj); cols_l.append(pi); vals_l.append(edge_w)

    A = sp.csr_matrix((vals_l, (rows_l, cols_l)), shape=(n_v, n_v))
    degree = np.asarray(A.sum(axis=1)).flatten()
    L = sp.diags(degree) - A
    return L.tocsr()


def hwsp(positions: torch.Tensor,
         benchmark:  Benchmark,
         plc:        PlacementCost,
         top_pct:    float = 0.05,
         damping:    float = 0.30,
         max_iters:  int   = 25,
         min_damping: float = 0.02,
         cg_tol:     float = 1e-5,
         cg_maxiter: int   = 500,
         baseline_w: float = 1.0,
         patience:   int   = 4) -> torch.Tensor:
    """
    Iterative hotness-weighted spectral placement of soft macros.
    Hard macros and ports are held fixed (Dirichlet boundary).
    """
    nh, ns = benchmark.num_hard_macros, benchmark.num_soft_macros
    n_macros = nh + ns
    n_ports  = benchmark.port_positions.shape[0]
    n_v      = n_macros + n_ports

    cw, ch     = float(benchmark.canvas_width), float(benchmark.canvas_height)
    rows, cols = plc.grid_row, plc.grid_col
    bw, bh     = cw / cols, ch / rows
    szs        = benchmark.macro_sizes.numpy()
    fixed      = benchmark.macro_fixed.numpy().astype(bool)

    # Only soft macros are free — all hard macros are pinned (overlap constraints)
    # and ports are pinned (fixed I/O locations). Restrict free set to soft macros.
    free_idx = np.array([i for i in range(nh, n_macros) if not fixed[i]], dtype=np.int64)
    fix_idx  = np.array(list(range(0, nh))                       # all hard macros
                        + [i for i in range(nh, n_macros) if fixed[i]]  # fixed soft (rare)
                        + list(range(n_macros, n_v)),            # ports
                        dtype=np.int64)
    print(f"    HWSP setup: n_v={n_v}  free={len(free_idx)}  fix={len(fix_idx)}  "
          f"nets={len(benchmark.net_nodes)}")

    pos_np      = positions.numpy().copy()
    cur_score   = _proxy(torch.from_numpy(pos_np), benchmark, plc)
    best_score  = cur_score
    best_np     = pos_np.copy()
    no_improve  = 0

    print(f"    HWSP init proxy={cur_score:.4f}  damping={damping}")

    for it in range(max_iters):
        # Refresh hot mask
        _set_placement(plc, torch.from_numpy(pos_np), benchmark)
        plc.get_congestion_cost()
        hot = _hot_mask(plc, rows, cols, top_pct)
        n_hot = int(hot.sum())

        pos_ext = _build_extended_pos(pos_np, benchmark)

        # Build hot-weighted Laplacian
        L = _build_hot_laplacian(benchmark, pos_ext, hot, bw, bh, rows, cols,
                                  baseline_w=baseline_w)

        # Block partition
        L_ff = L[free_idx][:, free_idx]
        L_fb = L[free_idx][:, fix_idx]

        # Fixed positions
        p_fix_x = pos_ext[fix_idx, 0]
        p_fix_y = pos_ext[fix_idx, 1]

        # RHS
        b_x = -np.asarray(L_fb @ p_fix_x).flatten()
        b_y = -np.asarray(L_fb @ p_fix_y).flatten()

        # Jacobi preconditioner
        diag = L_ff.diagonal()
        diag = np.where(diag > 1e-12, diag, 1.0)
        M    = sp.diags(1.0 / diag)

        x0_x = pos_np[free_idx, 0]
        x0_y = pos_np[free_idx, 1]

        sol_x, info_x = cg(L_ff, b_x, x0=x0_x, M=M, atol=cg_tol, maxiter=cg_maxiter)
        sol_y, info_y = cg(L_ff, b_y, x0=x0_y, M=M, atol=cg_tol, maxiter=cg_maxiter)

        # Damped update + canvas clipping
        new_x = (1 - damping) * x0_x + damping * sol_x
        new_y = (1 - damping) * x0_y + damping * sol_y

        sz_x_half = szs[free_idx, 0] / 2.0
        sz_y_half = szs[free_idx, 1] / 2.0
        new_x = np.clip(new_x, sz_x_half, cw - sz_x_half)
        new_y = np.clip(new_y, sz_y_half, ch - sz_y_half)

        trial_np = pos_np.copy()
        trial_np[free_idx, 0] = new_x
        trial_np[free_idx, 1] = new_y

        trial_score = _proxy(torch.from_numpy(trial_np), benchmark, plc)
        delta = trial_score - cur_score

        max_shift_x = float(np.max(np.abs(new_x - x0_x)))
        max_shift_y = float(np.max(np.abs(new_y - x0_y)))
        print(f"      iter {it+1:2d}  proxy={trial_score:.4f}  Δ={delta:+.4f}  "
              f"hot_cells={n_hot}  damping={damping:.3f}  "
              f"max_shift=({max_shift_x:.2f},{max_shift_y:.2f})  "
              f"cg_info=({info_x},{info_y})")

        if trial_score < cur_score - 1e-6:
            pos_np    = trial_np
            cur_score = trial_score
            if cur_score < best_score:
                best_score = cur_score
                best_np    = pos_np.copy()
            no_improve = 0
        else:
            no_improve += 1
            damping *= 0.5
            if damping < min_damping or no_improve >= patience:
                break

    print(f"    HWSP done: best={best_score:.4f}")
    return torch.from_numpy(best_np)
