"""
Bookshelf adapter for DREAMPlace (ISPD05 format).

Public API:
    write_bookshelf(bench, out_dir, bench_name, write_routability=False) -> Path
    write_dreamplace_config(aux, results_dir, ...)                        -> Path
    read_dreamplace_pl(pl_path, bench)                                    -> Tensor
    legalize_overlaps_strict(positions, bench, max_iters, verbose)        -> (Tensor, int)
"""

import json
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark

SCALE = 1000  # microns → integer bookshelf units


# ── write_bookshelf ───────────────────────────────────────────────────────────

def write_bookshelf(bench: Benchmark, out_dir, bench_name=None,
                    write_routability=False,
                    init_x: float = None, init_y: float = None):
    """
    Write ISPD05 bookshelf bundle (.aux/.nodes/.nets/.pl/.scl[/.wts][/.route]).

    Node order: hard macros, then soft macros, then ports.
    Soft macros are left movable inside DREAMPlace — the challenge evaluator
    re-pins them via PlacementCost, so this is a no-op on the judged metric.
    Hard macros with macro_fixed=True are written as terminals.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name     = bench_name or getattr(bench, 'name', None) or 'design'
    sc       = SCALE
    n_hard   = bench.num_hard_macros
    n_soft   = bench.num_soft_macros
    n_macros = n_hard + n_soft
    cw, ch   = bench.canvas_width, bench.canvas_height
    w_int    = int(round(cw * sc))
    h_int    = int(round(ch * sc))

    sizes    = bench.macro_sizes.numpy()           # [N, 2]
    positions= bench.macro_positions.numpy().copy() # [N, 2]
    fixed    = bench.macro_fixed.numpy()           # [N] bool

    # Override initial positions for movable macros if init_x/init_y supplied
    if init_x is not None and init_y is not None:
        for i in range(n_macros):
            if not fixed[i]:
                positions[i, 0] = float(init_x) * cw
                positions[i, 1] = float(init_y) * ch

    port_pos  = bench.port_positions.numpy() if bench.port_positions.shape[0] > 0 else np.zeros((0, 2))
    n_ports   = port_pos.shape[0]

    n_terminals = int(fixed[:n_macros].sum()) + n_ports  # all ports are fixed terminals

    # ── .nodes ───────────────────────────────────────────────────────────────
    nodes_path = out_dir / f'{name}.nodes'
    with open(nodes_path, 'w') as f:
        f.write('UCLA nodes 1.0\n\n')
        f.write(f'NumNodes : {n_macros + n_ports}\n')
        f.write(f'NumTerminals : {n_terminals}\n\n')
        for i in range(n_macros):
            w = max(1, int(round(sizes[i, 0] * sc)))
            h = max(1, int(round(sizes[i, 1] * sc)))
            term = ' terminal' if bool(fixed[i]) else ''
            f.write(f'm{i}\t{w}\t{h}{term}\n')
        for i in range(n_ports):
            f.write(f'p{i}\t1\t1\tterminal\n')

    # ── .pl ──────────────────────────────────────────────────────────────────
    pl_path = out_dir / f'{name}.pl'
    with open(pl_path, 'w') as f:
        f.write('UCLA pl 1.0\n\n')
        for i in range(n_macros):
            # lower-left corner in integer units
            x_ll = int(round(positions[i, 0] * sc - sizes[i, 0] * sc / 2))
            y_ll = int(round(positions[i, 1] * sc - sizes[i, 1] * sc / 2))
            x_ll = max(0, min(x_ll, w_int - 1))
            y_ll = max(0, min(y_ll, h_int - 1))
            fix_str = ' FIXED' if bool(fixed[i]) else ''
            f.write(f'm{i}\t{x_ll}\t{y_ll}\t: N{fix_str}\n')
        for i in range(n_ports):
            x_ll = int(round(port_pos[i, 0] * sc))
            y_ll = int(round(port_pos[i, 1] * sc))
            x_ll = max(0, min(x_ll, w_int - 1))
            y_ll = max(0, min(y_ll, h_int - 1))
            f.write(f'p{i}\t{x_ll}\t{y_ll}\t: N FIXED\n')

    # ── .nets ─────────────────────────────────────────────────────────────────
    nets_path = out_dir / f'{name}.nets'
    net_nodes   = bench.net_nodes    # list of tensors
    total_pins  = sum(len(n) for n in net_nodes)

    has_pin_info = (
        hasattr(bench, 'net_pin_nodes')
        and bench.net_pin_nodes is not None
        and len(bench.net_pin_nodes) > 0
        and hasattr(bench, 'macro_pin_offsets')
        and bench.macro_pin_offsets is not None
    )

    with open(nets_path, 'w') as f:
        f.write('UCLA nets 1.0\n\n')
        f.write(f'NumNets : {len(net_nodes)}\n')
        f.write(f'NumPins : {total_pins}\n\n')
        for net_i, nodes in enumerate(net_nodes):
            f.write(f'NetDegree : {len(nodes)} net{net_i}\n')
            for j, nid in enumerate(nodes.tolist()):
                nid = int(nid)
                node_name = f'm{nid}' if nid < n_macros else f'p{nid - n_macros}'
                ox = oy = 0.0
                if has_pin_info and net_i < len(bench.net_pin_nodes):
                    pins = bench.net_pin_nodes[net_i]
                    if j < pins.shape[0]:
                        owner     = int(pins[j, 0].item())
                        pin_local = int(pins[j, 1].item())
                        if owner < n_hard:
                            offs = bench.macro_pin_offsets[owner]
                            if pin_local < offs.shape[0]:
                                ox = offs[pin_local, 0].item() * sc
                                oy = offs[pin_local, 1].item() * sc
                io = 'I' if j == 0 else 'O'
                f.write(f'\t{node_name}\t{io} : {ox:.6f}\t{oy:.6f}\n')

    # ── .scl ─────────────────────────────────────────────────────────────────
    # Synthesise a minimal row grid — DREAMPlace needs it to parse but we
    # disable its row-aware legaliser, so exact values don't matter much.
    scl_path = out_dir / f'{name}.scl'
    min_h_sc   = max(1, int(sizes[:n_macros, 1].min() * sc))
    row_height = max(50, min(min_h_sc, 200))
    num_rows   = max(1, h_int // row_height)
    with open(scl_path, 'w') as f:
        f.write('UCLA scl 1.0\n\n')
        f.write(f'Numrows : {num_rows}\n\n')
        for r in range(num_rows):
            f.write('CoreRow Horizontal\n')
            f.write(f'  Coordinate    : {r * row_height}\n')
            f.write(f'  Height        : {row_height}\n')
            f.write( '  Sitewidth     : 1\n')
            f.write( '  Sitespacing   : 1\n')
            f.write( '  Siteorient    : N\n')
            f.write( '  Sitesymmetry  : Y\n')
            f.write(f'  SubrowOrigin  : 0   NumSites : {w_int}\n')
            f.write('End\n')

    # ── .wts ─────────────────────────────────────────────────────────────────
    wts_path = out_dir / f'{name}.wts'
    with open(wts_path, 'w') as f:
        f.write('UCLA wts 1.0\n\n')
        for i, w in enumerate(bench.net_weights.tolist()):
            f.write(f'net{i}\t{w:.6f}\n')

    # ── .route (optional) ────────────────────────────────────────────────────
    files = [f'{name}.nodes', f'{name}.nets', f'{name}.pl',
             f'{name}.scl', f'{name}.wts']

    if write_routability:
        n_tiles = 128
        tile_x  = max(1, w_int // n_tiles)
        tile_y  = max(1, h_int // n_tiles)
        cap_h   = max(1, int(round(bench.hroutes_per_micron * (tile_y / sc))))
        cap_v   = max(1, int(round(bench.vroutes_per_micron * (tile_x / sc))))
        route_path = out_dir / f'{name}.route'
        with open(route_path, 'w') as f:
            f.write('UCLA route 1.0\n\n')
            f.write(f'Grid : {n_tiles} {n_tiles} 2\n')
            f.write(f'VerticalCapacity : {cap_v}\n')
            f.write(f'HorizontalCapacity : {cap_h}\n')
            f.write( 'MinWireWidth : 1\n')
            f.write( 'MinWireSpacing : 0\n')
            f.write( 'ViaSpacing : 0\n')
            f.write( 'GridOrigin : 0 0\n')
            f.write(f'TileSize : {tile_x} {tile_y}\n')
            f.write( 'BlockageLayer : 2\n')
        files.append(f'{name}.route')

    # ── .aux ─────────────────────────────────────────────────────────────────
    aux_path = out_dir / f'{name}.aux'
    with open(aux_path, 'w') as f:
        f.write(f'RowBasedPlacement :  {" ".join(files)}\n')

    return aux_path


# ── write_dreamplace_config ───────────────────────────────────────────────────

def write_dreamplace_config(
    aux_path,
    results_dir,
    gpu=True,
    extra=None,
    target_density=0.7,
    density_weight=8e-5,
    gamma=4.0,
    num_bins_x=256,
    num_bins_y=256,
    iteration=1000,
    learning_rate=0.01,
    legalize_flag=0,
    detailed_place_flag=0,
):
    """Write DREAMPlace JSON config. Returns path to .json file."""
    aux_path    = Path(aux_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    seed = (extra or {}).get('random_seed', 1000)

    cfg = {
        'aux_input':           str(aux_path),
        'gpu':                 1 if gpu else 0,
        'result_dir':          str(results_dir),
        'global_place_flag':   1,
        'legalize_flag':       legalize_flag,       # 0 — use legalize_v2 post-hoc
        'detailed_place_flag': detailed_place_flag, # 0 — DP's DP doesn't help proxy
        'scale_factor':        0.0,
        'random_seed':         seed,
        'global_place_stages': [
            {
                'num_bins_x':                    num_bins_x,
                'num_bins_y':                    num_bins_y,
                'iteration':                     iteration,
                'learning_rate':                 learning_rate,
                'wirelength':                    'weighted_average',
                'optimizer':                     'nesterov',
                'Llambda_density_weight_iteration': 1,
                'Lsub_iteration':                1,
                'target_density':                target_density,
                'density_weight':                density_weight,
                'gamma':                         gamma,
                'stop_overflow':                 0.07,
                'legalize_flag':                 legalize_flag,
                'detailed_place_flag':           detailed_place_flag,
            }
        ],
    }

    cfg_path = aux_path.parent / 'dreamplace_config.json'
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    return cfg_path


# ── read_dreamplace_pl ────────────────────────────────────────────────────────

def read_dreamplace_pl(pl_path, bench: Benchmark):
    """
    Parse DREAMPlace output .pl file → [N, 2] positions tensor (microns, centres).
    """
    sc       = SCALE
    n_macros = bench.num_hard_macros + bench.num_soft_macros
    sizes    = bench.macro_sizes.numpy()
    pos      = bench.macro_positions.numpy().copy()  # fallback

    with open(pl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in ('#', 'U'):
                continue
            parts = line.split()
            if len(parts) < 3 or not parts[0].startswith('m'):
                continue
            try:
                i    = int(parts[0][1:])
                x_ll = float(parts[1])
                y_ll = float(parts[2])
            except (ValueError, IndexError):
                continue
            if i < n_macros:
                pos[i, 0] = (x_ll + sizes[i, 0] * sc / 2) / sc
                pos[i, 1] = (y_ll + sizes[i, 1] * sc / 2) / sc

    return torch.tensor(pos, dtype=torch.float32)


# ── legalize_overlaps_strict ──────────────────────────────────────────────────

def legalize_overlaps_strict(positions, bench: Benchmark,
                              max_iters=100, verbose=False):
    """
    Iterative pairwise hard-macro overlap resolution.
    Returns (legalized_tensor, n_remaining_overlaps).
    Soft macros are never moved.
    """
    pos   = positions.numpy().copy()
    nh    = bench.num_hard_macros
    sz    = bench.macro_sizes.numpy()
    fixed = bench.macro_fixed.numpy()
    cw, ch = bench.canvas_width, bench.canvas_height

    for it in range(max_iters):
        moved = False
        n_ov  = 0
        for i in range(nh):
            if fixed[i]:
                continue
            for j in range(i + 1, nh):
                dx = abs(pos[i, 0] - pos[j, 0])
                dy = abs(pos[i, 1] - pos[j, 1])
                sx = (sz[i, 0] + sz[j, 0]) / 2 + 1e-4
                sy = (sz[i, 1] + sz[j, 1]) / 2 + 1e-4
                if dx < sx and dy < sy:
                    n_ov += 1
                    if sx - dx <= sy - dy:
                        d = (sx - dx) / 2 + 1e-4
                        s = 1 if pos[i, 0] >= pos[j, 0] else -1
                        if not fixed[i]: pos[i, 0] += s * d
                        if not fixed[j]: pos[j, 0] -= s * d
                    else:
                        d = (sy - dy) / 2 + 1e-4
                        s = 1 if pos[i, 1] >= pos[j, 1] else -1
                        if not fixed[i]: pos[i, 1] += s * d
                        if not fixed[j]: pos[j, 1] -= s * d
                    moved = True

        for i in range(nh):
            if fixed[i]:
                continue
            pos[i, 0] = np.clip(pos[i, 0], sz[i, 0] / 2, cw - sz[i, 0] / 2)
            pos[i, 1] = np.clip(pos[i, 1], sz[i, 1] / 2, ch - sz[i, 1] / 2)

        if verbose:
            print(f'  legalize iter {it}: {n_ov} overlaps')
        if not moved:
            break

    # Count remaining
    final_ov = 0
    for i in range(nh):
        for j in range(i + 1, nh):
            dx = abs(pos[i, 0] - pos[j, 0])
            dy = abs(pos[i, 1] - pos[j, 1])
            if dx < (sz[i, 0] + sz[j, 0]) / 2 and dy < (sz[i, 1] + sz[j, 1]) / 2:
                final_ov += 1

    return torch.tensor(pos, dtype=torch.float32), final_ov
