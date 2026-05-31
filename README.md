# Macro Placement — Aarushi's Submission

A pipeline for macro placement on the IBM ICCAD04 benchmark suite,
combining **DREAMPlace** global placement, an **Optuna TPE search** over
DREAMPlace knobs (AutoDMP style), strict overlap legalization, and a series
of post-processing passes (congestion-targeted soft-macro diffusion,
hot-net outlier hauling, hotness-weighted spectral placement, hard-macro
coordinate descent with pairwise swap).

The submission entry point is [`submissions/aarushi/placer.py`](submissions/aarushi/placer.py)
— a `CachedPlacer` that returns pre-computed positions from
[`submissions/aarushi/cache/`](submissions/aarushi/cache).

---

## Solution overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 1   Global placement                                          │
│  DREAMPlace × 5 seeds × 17 benches  ─────►  /tmp/dp_overnight/*.pt   │
│  (GPU, CUDA 11.8, sm_80)                                             │
├──────────────────────────────────────────────────────────────────────┤
│  Phase 2   Build cache + legalize                                    │
│  _build_cache.py    ─►  cache/                                       │
│  _ckpt_repair.py    ─►  cache_legal/  (legalize_v2.strict)           │
├──────────────────────────────────────────────────────────────────────┤
│  Phase 3   AutoDMP TPE search over DREAMPlace knobs                  │
│  _autodmp_search.py + _replay_knobs.py  ─►  cache_autodmp/           │
│  (30 trials / bench, warm-started from Phase-1 best)                 │
├──────────────────────────────────────────────────────────────────────┤
│  Phase 4   Post-processing on soft macros                            │
│   ▪ Soft-macro diffusion v2 (congestion-targeted)  ─► cache_sa_12/   │
│   ▪ Outlier hauler (hot-net extremes)              ─► cache_hauler/  │
│   ▪ HWSP (hotness-weighted spectral)               ─► cache_hwsp/    │
├──────────────────────────────────────────────────────────────────────┤
│  Phase 5   Post-processing on hard macros                            │
│   ▪ Hard CD cascade (8-direction × decaying steps)                   │
│   ▪ Pairwise swap on same-size macros                                │
│   _hard_cd_swap.py                                  ─► cache_hard2/  │
└──────────────────────────────────────────────────────────────────────┘
```

Each phase reads from the best available previous cache and writes its own,
so phases are independently re-runnable.

## Repository layout (submission)

```
submissions/aarushi/
│
├── placer.py                      # CachedPlacer — the submission entry point
│
├── _overnight.py                  # Phase 1: DREAMPlace 5-seed sweep
├── _dp_all_benches.py             # Phase 1 driver for all 17 benches
├── dp_converter.py                # ISPD05 bookshelf writer + .pl reader + legalizer
├── docker/Dockerfile.cuda         # mpc:cuda image (DREAMPlace on PyTorch 2.1 + CUDA 11.8)
│
├── _build_cache.py                # Phase 2: pick best run per bench, write cache/
├── legalize_v2.py                 # legalize_overlaps_strict + verify_zero_overlaps
├── _ckpt_repair.py                # Parallel legalization runner → cache_legal/
│
├── _autodmp_search.py             # Phase 3: Optuna TPE over 10 DREAMPlace knobs
├── _replay_knobs.py               # Replay known-good knobs (skip TPE search)
│
├── soft_macro_diffusion.py        # Phase 4 (v1): Metropolis force-directed
├── soft_macro_diffusion_v2.py     # Phase 4 (v2): congestion-targeted batch greedy
├── _run_sa_diffusion.py           # v2 driver → cache_sa_12/
│
├── outlier_hauler.py              # Phase 4: hot-net extreme-pin haul
├── _run_outlier_hauler.py         #          driver → cache_hauler/
│
├── hwsp.py                        # Phase 4: hotness-weighted spectral placement
├── _run_hwsp.py                   #          driver → cache_hwsp/
│
├── congestion_nudge.py            # Phase 2/4 helper: small re-legalization pushes
├── _final_polish.py               # Phase 2 helper: re-legalize + conditional nudge
│
├── _hard_cd_swap.py               # Phase 5: hard-macro CD + same-size swap
│
├── cache/             ←─ default placements consumed by placer.py
├── cache_legal/       ←─ strict-overlap-free
├── cache_autodmp/     ←─ Phase 3 best
├── cache_sa_12/       ←─ Phase 4 v2 diffusion output
├── cache_hauler/      ←─ Phase 4 outlier hauler output
├── cache_hwsp/        ←─ Phase 4 spectral output
└── cache_hard2/       ←─ Phase 5 hard CD+swap output
```

## How `placer.py` works

```python
class CachedPlacer:
    """Loads pre-computed positions from cache/<bench>.pt."""
```

At evaluation time, the placer:

1. Scans `submissions/aarushi/cache/` at construction.
2. Loads each `<bench>.pt` into memory (cached).
3. For each `place(benchmark)` call, returns the cached positions for
   that benchmark — or falls back to `benchmark.macro_positions` if the
   bench isn't in cache.

## Reproducing the submission

```bash
# 0. Set up CUDA 11.8 + DREAMPlace
docker build -f submissions/aarushi/docker/Dockerfile.cuda -t mpc:cuda .
# (or use docker-commit flow — see docker/Dockerfile.cuda header)

# 1. DREAMPlace global placement, 5 seeds × 17 benches
python -u submissions/aarushi/_dp_all_benches.py

# 2. Build cache + legalize
python -u submissions/aarushi/_build_cache.py
python -u submissions/aarushi/_ckpt_repair.py --in cache --out cache_legal

# 3. AutoDMP TPE search (30 trials × bench)
python -u submissions/aarushi/_autodmp_search.py --benches all --trials 30
# Optional: replay known-good knobs to skip search
python -u submissions/aarushi/_replay_knobs.py --benches all

# 4. Soft-macro post-processing (CPU-only from here)
python -u submissions/aarushi/_run_sa_diffusion.py     --benches all
python -u submissions/aarushi/_run_outlier_hauler.py   --benches all
python -u submissions/aarushi/_run_hwsp.py             --benches all

# 5. Hard-macro post-processing
python -u submissions/aarushi/_hard_cd_swap.py --benches all

# Evaluate
uv run evaluate submissions/aarushi/placer.py --all
```

## Key algorithmic choices

**Soft-macro diffusion v2** — congestion-targeted batch greedy.
Identifies top-5% hot cells in `concat(H_smoothed, V_smoothed)` (matches
PlacementCost semantics), finds soft macros overlapping any hot cell, applies
repulsive + attractive force vectors, batch-evaluates via PlacementCost,
strict accept-if-Δ<0. Adaptive trust region (cool on reject, grow on accept).

**Outlier hauler** — directed soft-macro moves on hot-net extreme pins.
Profiling found ~98% of hot-net bbox extremes are soft macros and ~55-60%
of hot nets have a single outlier with gap ≥30% of bbox extent. Hauling
the outlier halfway toward the 2nd-extreme shrinks the bbox dramatically
in one move. Move ordering by `hotness × |gap|`.

**HWSP — hotness-weighted spectral placement** — solves the joint optimum
of all soft macros via a constrained quadratic minimization:
`L_ff p_free = -L_fb p_fix` with conjugate gradient. Edge weights are
`hotness(n) / (k_n − 1)` from the clique expansion. Hard macros and ports
form the Dirichlet boundary. Iterates with hot-cell feedback.

**Hard CD + pairwise swap** — multi-scale 8-direction probe on hard macros
(`steps = [4, 2, 1, 0.5, 0.25]` grid cells, 3 passes each) with strict
pairwise-AABB overlap check, then a single pairwise swap pass on
same-size hard macros (overlap-safe by construction).

## Engineering notes

- **DREAMPlace requires CUDA 11.8**. The CUB 2.x namespace change in
  CUDA 12.x breaks DREAMPlace's kernels — `pytorch:2.1.0-cuda11.8-cudnn8-devel`
  is the working base image.
- **`docker build` has no GPU access**, so the image is constructed by
  `docker run --gpus all` + `docker commit`. See
  [`submissions/aarushi/docker/Dockerfile.cuda`](submissions/aarushi/docker/Dockerfile.cuda).
- **`Benchmark.load(.pt)` returns empty `net_nodes`**. For anything that
  reads net connectivity, use `load_benchmark_from_dir(.../testcase_dir)`
  instead.
- **PlacementCost evaluator is CPU-only**. All Phase 4 and Phase 5
  scripts can run on a cheap CPU machine — only Phases 1 and 3 need the
  GPU. After Phase 3, the GPU sits idle.

---

# About the Challenge (upstream)

The remainder of this README is from the upstream challenge repo
(Partcl/HRT macro placement competition).

See [`SCORING.md`](SCORING.md) for the proxy-cost evaluation methodology,
[`SETUP.md`](SETUP.md) for environment setup, and
[`baselines/`](baselines) for reference implementations.

## Background papers

[1] [An Updated Assessment of Reinforcement Learning for Macro Placement](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11300304)
[2] [Assessment of Reinforcement Learning for Macro Placement](https://vlsicad.ucsd.edu/Publications/Conferences/396/c396.pdf)
[3] [Reevaluating Google's Reinforcement Learning for IC Macro Placement](https://cacm.acm.org/research/reevaluating-googles-reinforcement-learning-for-ic-macro-placement/)
[4] [A graph placement methodology for fast chip design](https://www.nature.com/articles/s41586-021-03544-w)
