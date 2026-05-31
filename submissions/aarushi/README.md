# aarushi — Macro Placement Submission

placer.py is the submission entry point. It is a CachedPlacer that
serves pre-computed positions from cache/<bench>.pt. Run the evaluator
against it:

```bash
uv run evaluate submissions/aarushi/placer.py --all
```

## Pipeline

The cache files in cache/ are the end result of a multi-phase pipeline.
Each phase reads from the best available preceding cache and writes its own
output directory, so any phase can be re-run independently.

| # | Phase | Script | Output |
|---|---|---|---|
| 1 | Global placement (DREAMPlace) | _overnight.py, _dp_all_benches.py | /tmp/dp_overnight/ |
| 2 | Build cache + legalize | _build_cache.py, _ckpt_repair.py | cache/, cache_legal/ |
| 3 | AutoDMP TPE knob search | _autodmp_search.py, _replay_knobs.py | cache_autodmp/ |
| 4a | Soft-macro diffusion v2 | _run_sa_diffusion.py | cache_sa_12/ |
| 4b | Hot-net outlier hauler | _run_outlier_hauler.py | cache_hauler/ |
| 4c | Hotness-weighted spectral (HWSP) | _run_hwsp.py | cache_hwsp/ |
| 5 | Hard-macro CD + pairwise swap | _hard_cd_swap.py | cache_hard2/ |

## Modules

### Global placement (Phase 1)

- **_overnight.py** — early version of the DREAMPlace sweep.
- **_overnight_v4.py** — refined sweep with per-bench knob overrides.
- **_dp_all_benches.py** — fan-out driver: 5 seeds × 17 benches.
- **dp_converter.py** — ISPD05 bookshelf writer (write_bookshelf),
  DREAMPlace config writer (write_dreamplace_config), placement reader
  (read_dreamplace_pl), and legalize_overlaps_strict.
- **docker/Dockerfile.cuda** — mpc:cuda image. **Must** use
  pytorch:2.1.0-cuda11.8-cudnn8-devel; CUDA 12.x breaks DREAMPlace.

### Cache + legalization (Phase 2)

- **_build_cache.py** — picks the lowest-proxy run per bench from the
  Phase-1 sweep, optionally applies congestion nudge, writes cache/.
- **legalize_v2.py** — legalize_overlaps_strict (iterative force-push
  + periodic noise kick to escape local minima) and
  verify_zero_overlaps.
- **_ckpt_repair.py** — parallel runner: cache/ → cache_legal/ via
  legalize_overlaps_strict. Falls back gracefully on unfixable benches.
- **congestion_nudge.py** — small displacement of macros sitting in
  the densest cells.
- **_final_polish.py** — second-pass legalize + conditional nudge.

### AutoDMP (Phase 3)

- **_autodmp_search.py** — Optuna TPE study (SQLite-backed, resumable)
  over 10 DREAMPlace knobs:
  target_density, learning_rate, density_weight, gamma, num_bins,
  iteration, stop_overflow, Llambda_density_weight_iteration,
  macro_halo_frac, routability_opt.
  Warm-starts trial 0 from Phase-1 best.
- **_replay_knobs.py** — skip the search entirely. Replays a fixed table
  of per-bench best knobs from a prior large-scale study in ~30 s/bench.

### Soft-macro post-processing (Phase 4)

- **soft_macro_diffusion.py** — original Metropolis trust-region
  diffusion (force = repulsive from hot cell + attractive toward net
  centroid).
- **soft_macro_diffusion_v2.py** — congestion-targeted batch greedy.
  Hot cells via concat(H_smoothed, V_smoothed) (matches PlacementCost
  smooth_range=2). Per iteration: identify candidates, batch-apply moves,
  one proxy evaluation, strict accept-if-Δ<0. Adaptive trust region.
- **_run_sa_diffusion.py** — driver → cache_sa_12/.

- **outlier_hauler.py** — directed haul of hot-net extreme pins.
  Profiling shows ~98 % of hot-net bbox extremes are movable soft macros
  and ~55-60 % of hot nets have a single outlier with gap ≥ 30 % of bbox.
  Per pass: compute hot nets, identify extreme-pin soft macros with large
  gaps, propose move = 0.5 × gap toward the 2nd-extreme on that axis,
  sort by hotness × |gap|, accept if Δproxy < 0.
- **_run_outlier_hauler.py** — driver → cache_hauler/.

- **hwsp.py** — hotness-weighted spectral placement. Build the
  clique-expansion Laplacian L with edge weights
  w_ij = (hotness(n) + baseline) / (k_n − 1). Partition into free (soft
  macros) and fixed (hard macros + ports). Solve L_ff p_free = -L_fb p_fix
  via conjugate gradient with Jacobi preconditioner. Damped update with
  hot-cell feedback iteration.
- **_run_hwsp.py** — driver → cache_hwsp/.

### Hard-macro post-processing (Phase 5)

- **_hard_cd_swap.py** — two stages:
  1. **CD cascade**: for each hard macro, try 8 unit-norm directions at
     decaying step sizes [4, 2, 1, 0.5, 0.25] grid cells, 3 passes per
     step. Per probe: AABB-check against every other hard macro, evaluate
     full proxy if no overlap. Accept best Δ<0 move.
  2. **Pairwise swap**: group hard macros by rounded (w, h). For each
     same-size pair, swap positions, evaluate, accept if Δ<0. Overlap-safe
     by construction.

### Helpers and exploratory work

- **score_initial.py** — score a benchmark's raw initial positions
  against the proxy cost; useful for spot-checks.
- **placer.py** — CachedPlacer (the submission interface).
- **manifest.json** — last-known per-bench scores written by
  _build_cache.py / _final_polish.py.

## Caches

Each cache directory holds one <bench>.pt per benchmark with
{positions, score, costs}. The placer reads from cache/; the other
directories are intermediate outputs used by downstream phases or kept
for comparison.

| Directory | Source phase | What it contains |
|---|---|---|
| cache/ | Phase 2 (overwritten by 4/5) | Default — consumed by placer.py |
| cache_legal/ | Phase 2 | Strict overlap-free |
| cache_autodmp/ | Phase 3 | Best of 30-trial TPE search |
| cache_sa_12/ | Phase 4a | Soft-macro diffusion v2 output |
| cache_hauler/ | Phase 4b | Outlier-hauler output |
| cache_hwsp/ | Phase 4c | Spectral placement output |
| cache_hard2/ | Phase 5 | Hard-macro CD + swap output |

cache/ is updated to point at whichever downstream pass produced the
lowest-proxy placement for each bench (so placer.py always serves the
current best).

## Engineering footnotes

- **Benchmark.load(.pt) returns empty net_nodes.** Anything that
  iterates over net connectivity must call load_benchmark_from_dir(<dir>)
  instead. The compute-from-dir variant parses netlist.pb.txt and
  populates net_nodes.
- **Port nodes have indices ≥ n_macros.** When indexing positions for
  net pins, concatenate benchmark.port_positions after the macro
  positions:

  ```python
  pos_ext = np.concatenate([pos_np, benchmark.port_positions.numpy()], axis=0)
  ```

- **compute_proxy_cost ≈ 100 ms** at ibm02 scale. CD / SA loops that
  call it per probe are bound at ~10 evals/s on CPU. Lawnmower's
  IncrementalEval (0.5 ms / eval) was attempted (fast_proxy.py) but
  drifted from ground truth on the congestion term (top-5 % of
  concat(H_smoothed, V_smoothed) is hard to maintain incrementally),
  so it was reverted.
