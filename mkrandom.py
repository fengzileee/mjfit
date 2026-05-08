#!/usr/bin/env python3
"""Generate random peg-in-hole configs with random hole, island, and physical
parameters.

Samples a hole shape, optional islands inside it, and physical scene
parameters (plate size/thickness, peg length, initial peg height, optional
grip handle and its dimensions). Shape primitives come from mkpeg.py
(circle, rect, stadium, regular_ngon, star). Validates that islands fit
inside the hole with the requested clearance and don't overlap each other.

Any physical parameter can be fixed via its CLI flag; omitted flags are
randomized within sensible ranges (see RANGE_* constants below). Output is
a YAML config compatible with mkpeg.py's --config flag, so random scenes
are reproducible (via --seed) and inspectable.

CLI:
    python mkrandom.py -o random.yaml
    python mkrandom.py --seed 42 --n-islands 2 -o random.yaml
    python mkrandom.py --shape-pool circle,star,regular_ngon -o random.yaml
    python mkrandom.py --with-handle yes --handle-height 0.03 -o random.yaml
    python mkrandom.py --seed 7 --xml-output scene.xml -o scene.yaml
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import yaml

from mkpeg import (
    _ensure_ccw,
    _offset_polygon_inward,
    _offset_polygon_outward,
    _resolve_shape,
    _segments_proper_intersect,
    generate_from_config,
)


SHAPE_POOL_DEFAULT = ['circle', 'rect', 'stadium', 'regular_ngon', 'star']
# Stars excluded from islands by default — sharp inner corners blow up under
# the outward miter offset for any non-trivial clearance.
ISLAND_SHAPE_POOL_DEFAULT = ['circle', 'rect', 'stadium', 'regular_ngon']


def _sample_shape(rng: random.Random, pool: list[str], size: float,
                  *, allow_rotation: bool = True,
                  center: tuple[float, float] = (0.0, 0.0)) -> dict:
    """Sample a random shape config dict whose extent is bounded by `size`."""
    t = rng.choice(pool)
    cfg: dict = {'type': t}
    if t == 'circle':
        cfg['radius'] = float(size * rng.uniform(0.5, 1.0))
        cfg['n'] = 32
    elif t == 'rect':
        cfg['width']  = float(size * rng.uniform(0.6, 1.0))
        cfg['height'] = float(size * rng.uniform(0.4, 1.0))
    elif t == 'stadium':
        cfg['width']  = float(size * rng.uniform(0.7, 1.0))
        cfg['height'] = float(size * rng.uniform(0.4, 0.9))
    elif t == 'regular_ngon':
        cfg['n'] = int(rng.randint(3, 8))
        cfg['circumradius'] = float(size * rng.uniform(0.6, 1.0))
    elif t == 'star':
        cfg['n_points'] = int(rng.randint(4, 8))
        outer = float(size * rng.uniform(0.7, 1.0))
        cfg['outer_radius'] = outer
        cfg['inner_radius'] = float(outer * rng.uniform(0.4, 0.7))
    else:
        raise ValueError(f"Unsupported shape type in pool: {t!r}")
    if allow_rotation and t != 'circle':
        cfg['rotate'] = float(rng.uniform(0, 2 * np.pi))
    if center != (0.0, 0.0):
        cfg['center'] = [float(center[0]), float(center[1])]
    return cfg


def _point_in_polygon(point, poly: np.ndarray) -> bool:
    """Standard ray casting; works for any simple polygon (CW or CCW)."""
    x, y = float(point[0]), float(point[1])
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i, 0], poly[i, 1]
        xj, yj = poly[j, 0], poly[j, 1]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside


def _polygons_have_edge_crossing(p1: np.ndarray, p2: np.ndarray) -> bool:
    n1, n2 = len(p1), len(p2)
    for i in range(n1):
        a, b = p1[i], p1[(i + 1) % n1]
        for j in range(n2):
            c, d = p2[j], p2[(j + 1) % n2]
            if _segments_proper_intersect(a, b, c, d):
                return True
    return False


def _polygon_strictly_inside(inner: np.ndarray, outer: np.ndarray) -> bool:
    """True iff `inner` is fully contained in `outer` with no edge crossings."""
    if _polygons_have_edge_crossing(inner, outer):
        return False
    return all(_point_in_polygon(v, outer) for v in inner)


def _polygons_disjoint(p1: np.ndarray, p2: np.ndarray) -> bool:
    """True iff p1 and p2 share no interior points (edge-disjoint, neither
    contains the other)."""
    if _polygons_have_edge_crossing(p1, p2):
        return False
    if _point_in_polygon(p1[0], p2):
        return False
    if _point_in_polygon(p2[0], p1):
        return False
    return True


# Default sampling ranges for physical scene parameters. Each is `(low, high)`.
# Plate size scales with hole size (multiplicative); everything else absolute.
RANGE_PLATE_THICKNESS  = (0.010, 0.030)
RANGE_PLATE_SIZE_RATIO = (5.0, 10.0)         # × hole_size
RANGE_PEG_LENGTH_EXTRA = (0.020, 0.060)      # added on top of plate_thickness
RANGE_PEG_HEIGHT       = (0.05, 0.15)
RANGE_HANDLE_HEIGHT    = (0.010, 0.040)
RANGE_HANDLE_MARGIN    = (0.002, 0.006)
HANDLE_PROBABILITY     = 0.5


def sample_random_config(
    rng: random.Random,
    *,
    hole_size: float = 0.025,
    plate_size: float | None = None,
    plate_thickness: float | None = None,
    peg_length: float | None = None,
    peg_height: float | None = None,
    clearance: float = 0.0005,
    with_handle: bool | None = None,
    handle_height: float | None = None,
    handle_margin: float | None = None,
    n_islands: int | None = None,
    max_islands: int = 3,
    shape_pool: list[str] | None = None,
    island_shape_pool: list[str] | None = None,
    max_attempts: int = 200,
) -> tuple[dict, int]:
    """Sample a random scene config compatible with mkpeg.generate_from_config.

    Any physical parameter passed as None gets randomized; pass a value to fix
    it. `n_islands=None` means random in [0, max_islands]. `with_handle=None`
    means random with `HANDLE_PROBABILITY`.

    Returns (config_dict, actual_n_islands_placed). The placed count can be
    less than requested if rejection sampling failed to fit them all.
    """
    shape_pool = shape_pool or SHAPE_POOL_DEFAULT
    island_shape_pool = island_shape_pool or ISLAND_SHAPE_POOL_DEFAULT

    # --- physical params (sample anything left as None) -------------------
    if plate_thickness is None:
        plate_thickness = rng.uniform(*RANGE_PLATE_THICKNESS)
    if plate_size is None:
        # Scale relative to hole_size; mkpeg requires plate_half > hole_max + 1mm.
        plate_size = hole_size * rng.uniform(*RANGE_PLATE_SIZE_RATIO)
    if peg_length is None:
        # Anchor to plate_thickness so mkpeg's `peg_length > plate_thickness +
        # clearance` check (when islands are present) is always satisfied.
        peg_length = plate_thickness + rng.uniform(*RANGE_PEG_LENGTH_EXTRA)
    if peg_height is None:
        peg_height = rng.uniform(*RANGE_PEG_HEIGHT)
    if with_handle is None:
        with_handle = rng.random() < HANDLE_PROBABILITY
    if with_handle:
        if handle_height is None:
            handle_height = rng.uniform(*RANGE_HANDLE_HEIGHT)
        if handle_margin is None:
            # Must exceed clearance so the handle can't pass through the hole.
            lo = max(RANGE_HANDLE_MARGIN[0], clearance * 2)
            hi = max(RANGE_HANDLE_MARGIN[1], lo + 0.001)
            handle_margin = rng.uniform(lo, hi)
    else:
        handle_height = 0.0  # mkpeg treats 0 as "no handle"

    # --- shape sampling ---------------------------------------------------
    hole_cfg = _sample_shape(rng, shape_pool, hole_size)
    hole_poly = _ensure_ccw(_resolve_shape(hole_cfg))

    # Available area for islands = hole shrunk by clearance (the peg's outer).
    peg_outer = _offset_polygon_inward(hole_poly, clearance)

    if n_islands is None:
        n_islands = rng.randint(0, max_islands)

    islands: list[dict] = []
    placed_polys: list[np.ndarray] = []
    for _ in range(n_islands):
        for _attempt in range(max_attempts):
            isl_size = hole_size * rng.uniform(0.10, 0.35)
            cx = rng.uniform(-hole_size * 0.6, hole_size * 0.6)
            cy = rng.uniform(-hole_size * 0.6, hole_size * 0.6)
            cand_cfg = _sample_shape(rng, island_shape_pool, isl_size,
                                     center=(cx, cy))
            cand_poly = _ensure_ccw(_resolve_shape(cand_cfg))
            cand_outer = _offset_polygon_outward(cand_poly, clearance)

            if not _polygon_strictly_inside(cand_outer, peg_outer):
                continue
            ok = True
            for prev in placed_polys:
                prev_outer = _offset_polygon_outward(prev, clearance)
                if not _polygons_disjoint(cand_outer, prev_outer):
                    ok = False
                    break
            if not ok:
                continue
            islands.append(cand_cfg)
            placed_polys.append(cand_poly)
            break
        else:
            break  # give up on remaining islands rather than failing the run

    config: dict = {
        'clearance': float(clearance),
        'plate_size': float(plate_size),
        'plate_thickness': float(plate_thickness),
        'peg_length': float(peg_length),
        'peg_height': float(peg_height),
    }
    if with_handle:
        config['handle_height'] = float(handle_height)
        config['handle_margin'] = float(handle_margin)
    config['hole'] = hole_cfg
    if islands:
        config['islands'] = islands
    return config, len(islands)


def _parse_n_islands(s: str) -> int | None:
    if s == 'random':
        return None
    return int(s)


def _parse_with_handle(s: str) -> bool | None:
    s = s.lower()
    if s == 'random':
        return None
    if s in ('yes', 'true', '1'):
        return True
    if s in ('no', 'false', '0'):
        return False
    raise argparse.ArgumentTypeError(
        f"--with-handle must be one of: random, yes, no (got {s!r})")


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seed', type=int, default=None,
                   help='RNG seed for reproducibility (default: nondeterministic)')
    p.add_argument('--n-islands', type=str, default='random',
                   help='number of islands ("random" or an integer; default random)')
    p.add_argument('--max-islands', type=int, default=3,
                   help='upper bound when --n-islands is "random" (default 3)')
    p.add_argument('--shape-pool', type=str,
                   default=','.join(SHAPE_POOL_DEFAULT),
                   help='comma-separated shape types for the hole '
                        f"(default: {','.join(SHAPE_POOL_DEFAULT)})")
    p.add_argument('--island-shape-pool', type=str,
                   default=','.join(ISLAND_SHAPE_POOL_DEFAULT),
                   help='comma-separated shape types for islands '
                        f"(default: {','.join(ISLAND_SHAPE_POOL_DEFAULT)})")
    p.add_argument('--hole-size', type=float, default=0.025,
                   help='hole circumradius / bounding-box size (default 0.025)')
    p.add_argument('--clearance', type=float, default=0.0005)
    # Physical params: omit (or pass `None`) to randomize. Pass a value to fix.
    p.add_argument('--plate-size', type=float, default=None,
                   help=f'fixed plate side length; default randomizes in '
                        f'[{RANGE_PLATE_SIZE_RATIO[0]}, {RANGE_PLATE_SIZE_RATIO[1]}] × hole_size')
    p.add_argument('--plate-thickness', type=float, default=None,
                   help=f'fixed plate thickness; default randomizes in '
                        f'[{RANGE_PLATE_THICKNESS[0]}, {RANGE_PLATE_THICKNESS[1]}]')
    p.add_argument('--peg-length', type=float, default=None,
                   help=f'fixed peg length; default randomizes as '
                        f'plate_thickness + uniform[{RANGE_PEG_LENGTH_EXTRA[0]}, '
                        f'{RANGE_PEG_LENGTH_EXTRA[1]}]')
    p.add_argument('--peg-height', type=float, default=None,
                   help=f'fixed initial peg height above plate; default '
                        f'randomizes in [{RANGE_PEG_HEIGHT[0]}, {RANGE_PEG_HEIGHT[1]}]')
    p.add_argument('--with-handle', type=str, default='random',
                   help='whether the peg has a grip handle: random, yes, no '
                        f'(default random with p={HANDLE_PROBABILITY})')
    p.add_argument('--handle-height', type=float, default=None,
                   help=f'fixed handle height when present; default randomizes '
                        f'in [{RANGE_HANDLE_HEIGHT[0]}, {RANGE_HANDLE_HEIGHT[1]}]')
    p.add_argument('--handle-margin', type=float, default=None,
                   help=f'fixed handle radial margin when present; default '
                        f'randomizes in [{RANGE_HANDLE_MARGIN[0]}, '
                        f'{RANGE_HANDLE_MARGIN[1]}] (clamped above clearance)')
    p.add_argument('-o', '--output', type=Path,
                   help='write YAML config here (default: stdout)')
    p.add_argument('--xml-output', type=Path,
                   help='also build the MJCF XML for the sampled config and '
                        'write it here. Honors --export-meshes.')
    p.add_argument('--export-meshes', action='store_true',
                   help='when --xml-output is given, write each mesh as .obj '
                        'next to the XML. Required for USD export.')
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    n_islands = _parse_n_islands(args.n_islands)
    with_handle = _parse_with_handle(args.with_handle)
    shape_pool = [s.strip() for s in args.shape_pool.split(',') if s.strip()]
    island_shape_pool = [s.strip() for s in args.island_shape_pool.split(',') if s.strip()]

    cfg, placed = sample_random_config(
        rng,
        hole_size=args.hole_size,
        plate_size=args.plate_size,
        plate_thickness=args.plate_thickness,
        peg_length=args.peg_length,
        peg_height=args.peg_height,
        clearance=args.clearance,
        with_handle=with_handle,
        handle_height=args.handle_height,
        handle_margin=args.handle_margin,
        n_islands=n_islands,
        max_islands=args.max_islands,
        shape_pool=shape_pool,
        island_shape_pool=island_shape_pool,
    )
    if n_islands is not None and placed < n_islands:
        print(f"warning: placed {placed}/{n_islands} islands "
              "(rejection sampling gave up)", file=sys.stderr)

    yaml_text = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None)
    if args.output:
        args.output.write_text(yaml_text)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(yaml_text)

    if args.xml_output:
        mesh_dir = args.xml_output.parent.resolve() if args.export_meshes else None
        xml = generate_from_config(cfg, mesh_dir=mesh_dir)
        args.xml_output.write_text(xml)
        print(f"wrote {args.xml_output}", file=sys.stderr)


if __name__ == '__main__':
    main()
