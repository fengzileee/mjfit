#!/usr/bin/env python3
"""Generate MuJoCo MJCF for peg-in-hole tasks with arbitrary hole shapes.

The plate-with-hole is built by radial wedge decomposition around the hole
centroid: every wedge is a convex prism, so all collision pairs end up
convex-vs-convex. No V-HACD, no CoACD, no SDF. Works for any hole shape that
is star-shaped from its centroid (all convex polygons + the usual stars/gears).

CLI:
    python mkpeg.py --shape circle --output circle.xml
    python mkpeg.py --shape star --hole-size 0.03 -o star.xml
    python mkpeg.py --shape triangle -o tri.xml

Library:
    from mkpeg import generate_xml
    xml = generate_xml(shape='hexagon', hole_size=0.02)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import mapbox_earcut as _earcut
    _HAS_EARCUT = True
except ImportError:  # pragma: no cover
    _HAS_EARCUT = False


# ----- shape library (2D hole cross-sections, CCW) -------------------------

def _circle(size: float, n: int = 32) -> np.ndarray:
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([size * np.cos(a), size * np.sin(a)])


def _regular_ngon(size: float, n: int) -> np.ndarray:
    """Regular n-gon with circumradius = size."""
    a = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    return np.column_stack([size * np.cos(a), size * np.sin(a)])


def _star(size: float, n_points: int = 5, inner_ratio: float = 0.45) -> np.ndarray:
    a = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 2 * n_points, endpoint=False)
    r = np.where(np.arange(2 * n_points) % 2 == 0, size, size * inner_ratio)
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def _slot(size: float, aspect: float = 0.35) -> np.ndarray:
    """Rounded-ended slot (capsule cross-section). size = half-length."""
    h = size * aspect
    n_cap = 8
    a_r = np.linspace(-np.pi / 2, np.pi / 2, n_cap)
    right = np.column_stack([size - h + h * np.cos(a_r), h * np.sin(a_r)])
    a_l = np.linspace(np.pi / 2, 3 * np.pi / 2, n_cap)
    left = np.column_stack([-size + h + h * np.cos(a_l), h * np.sin(a_l)])
    return np.vstack([right, left])


SHAPES = {
    'circle':   lambda size, n=32: _circle(size, n),
    'triangle': lambda size, n=None: _regular_ngon(size, 3),
    'square':   lambda size, n=None: _regular_ngon(size, 4),
    'pentagon': lambda size, n=None: _regular_ngon(size, 5),
    'hexagon':  lambda size, n=None: _regular_ngon(size, 6),
    'octagon':  lambda size, n=None: _regular_ngon(size, 8),
    'star':     lambda size, n=None: _star(size),
    'slot':     lambda size, n=None: _slot(size),
}


# ----- geometry helpers ----------------------------------------------------

def _ray_to_square(centroid: np.ndarray, direction: np.ndarray, half: float) -> np.ndarray:
    """First exit of ray (centroid + t*direction, t>0) from square [-half, half]^2."""
    t_min = np.inf
    best = None
    for axis in range(2):
        d = direction[axis]
        if abs(d) < 1e-12:
            continue
        for bound in (-half, half):
            t = (bound - centroid[axis]) / d
            if t <= 1e-9:
                continue
            pt = centroid + t * direction
            other = pt[1 - axis]
            if -half - 1e-9 <= other <= half + 1e-9 and t < t_min:
                t_min = t
                best = pt
    if best is None:
        raise ValueError("Ray from centroid failed to intersect plate square.")
    return best


def _is_convex_poly(poly: np.ndarray) -> bool:
    n = len(poly)
    signs = []
    for i in range(n):
        a, b, c = poly[i], poly[(i + 1) % n], poly[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-12:
            signs.append(1 if cross > 0 else -1)
    return len(set(signs)) <= 1


def _offset_polygon_inward(poly: np.ndarray, offset: float) -> np.ndarray:
    """Offset a CCW polygon inward by `offset` along each edge's normal.

    Uses miter joins at corners, so the output has the same vertex count and
    topology as the input. Every edge of the output lies at normal distance
    `offset` from the corresponding input edge — i.e. the gap between input
    and output edges is exactly `offset` regardless of shape. This is the
    right semantics for peg-in-hole clearance.

    Assumes CCW polygon (inward normal = +90° CCW rotation of edge direction).
    For very sharp convex corners, miter extension is offset / sin(alpha/2);
    caller must keep `offset` small relative to local feature sizes.
    """
    n = len(poly)
    edges = np.roll(poly, -1, axis=0) - poly
    edge_lens = np.linalg.norm(edges, axis=1, keepdims=True)
    edge_dirs = edges / edge_lens
    # CCW polygon: interior on left of edge → inward normal = (-dy, dx)
    inward = np.column_stack([-edge_dirs[:, 1], edge_dirs[:, 0]])

    out = np.zeros_like(poly)
    for i in range(n):
        d_prev, d_curr = edge_dirs[i - 1], edge_dirs[i]
        n_prev, n_curr = inward[i - 1], inward[i]
        # Intersect the two translated edges incident to vertex i:
        #   line_prev: poly[i] + offset*n_prev + t*d_prev
        #   line_curr: poly[i] + offset*n_curr + s*d_curr
        # Equate and solve 2x2 system for t.
        A = np.column_stack([d_prev, -d_curr])
        b = offset * (n_curr - n_prev)
        if abs(np.linalg.det(A)) < 1e-12:
            # Parallel edges (straight line through vertex) — no corner offset needed.
            out[i] = poly[i] + offset * n_prev
        else:
            t, _ = np.linalg.solve(A, b)
            out[i] = poly[i] + offset * n_prev + t * d_prev
    return out


def _offset_polygon_outward(poly: np.ndarray, offset: float) -> np.ndarray:
    """Outward offset: same miter math, opposite sign."""
    return _offset_polygon_inward(poly, -offset)


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly[::-1].copy() if _signed_area(poly) < 0 else poly


def _is_star_shaped_from(poly: np.ndarray, point: np.ndarray, tol: float = 1e-9) -> bool:
    """True iff `point` lies in the kernel of CCW polygon `poly` — i.e. sees
    every edge from the interior side. Sufficient for fan-triangulation and
    radial-wedge decomposition to produce a valid convex decomposition."""
    n = len(poly)
    for i in range(n):
        v0, v1 = poly[i], poly[(i + 1) % n]
        d = v1 - v0
        cross = d[0] * (point[1] - v0[1]) - d[1] * (point[0] - v0[0])
        if cross <= tol:
            return False
    return True


# ----- general (non-star-shaped) decomposition: ear clipping + bridging ----

def _points_equal(a, b, tol: float = 1e-9) -> bool:
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def _cross2(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _segments_proper_intersect(p1, p2, p3, p4) -> bool:
    """True iff open segments p1p2 and p3p4 cross in their interiors.
    Endpoint contacts and collinear overlaps return False (the bridge-finder
    skips adjacent edges separately, so that's fine here)."""
    d1 = _cross2(p4[0] - p3[0], p4[1] - p3[1], p1[0] - p3[0], p1[1] - p3[1])
    d2 = _cross2(p4[0] - p3[0], p4[1] - p3[1], p2[0] - p3[0], p2[1] - p3[1])
    d3 = _cross2(p2[0] - p1[0], p2[1] - p1[1], p3[0] - p1[0], p3[1] - p1[1])
    d4 = _cross2(p2[0] - p1[0], p2[1] - p1[1], p4[0] - p1[0], p4[1] - p1[1])
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def _point_in_triangle(p, a, b, c, tol: float = 1e-12) -> bool:
    """True if p is strictly inside triangle abc."""
    def s(q1, q2, q3):
        return (q1[0] - q3[0]) * (q2[1] - q3[1]) - (q2[0] - q3[0]) * (q1[1] - q3[1])
    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    has_neg = d1 < -tol or d2 < -tol or d3 < -tol
    has_pos = d1 > tol or d2 > tol or d3 > tol
    return not (has_neg and has_pos)


def _ear_clip_simple(poly: np.ndarray) -> list[np.ndarray]:
    """Triangulate a simple CCW polygon by ear clipping. Filters degenerate
    triangles (which come from bridge seams in polygon-with-hole inputs)."""
    if _signed_area(poly) < 0:
        poly = poly[::-1]
    verts = [tuple(v) for v in poly]
    triangles = []
    safety = len(verts) ** 2 + 20
    while len(verts) > 3:
        safety -= 1
        if safety <= 0:
            raise RuntimeError("Ear clipping failed to terminate.")
        n = len(verts)
        found = False
        for i in range(n):
            a, b, c = verts[(i - 1) % n], verts[i], verts[(i + 1) % n]
            # Must be a convex corner (CCW left turn) with nonzero area.
            if _cross2(b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1]) <= 1e-12:
                continue
            # No other polygon vertex may lie inside the candidate ear. Skip
            # vertices that are geometrically coincident with a triangle corner
            # (happens at bridge seams, where outer[j] and inner[i] are each
            # duplicated as the start/end of their loops).
            blocked = False
            for j in range(n):
                if j == (i - 1) % n or j == i or j == (i + 1) % n:
                    continue
                vj = verts[j]
                if _points_equal(vj, a) or _points_equal(vj, b) or _points_equal(vj, c):
                    continue
                if _point_in_triangle(vj, a, b, c):
                    blocked = True
                    break
            if blocked:
                continue
            triangles.append(np.array([a, b, c]))
            verts.pop(i)
            found = True
            break
        if not found:
            raise RuntimeError("No valid ear found — polygon may be self-intersecting.")
    triangles.append(np.array(verts))
    return [t for t in triangles if abs(_signed_area(t)) > 1e-14]


def _decompose_annular(outer_poly: np.ndarray, inner_poly: np.ndarray) -> list[np.ndarray]:
    """Triangulate the region bounded by outer (CCW) and inner (CCW) polygons.

    Used for both plate-minus-hole and for a hollow peg (plug shell). Both
    polygons are taken as CCW around their own interiors; this function
    reverses the inner so traversal puts annular material on the left, bridges
    the loops into a single simple polygon, and ear-clips it.
    """
    return _decompose_polygon_with_holes(outer_poly, [inner_poly])


def _find_bridge_against(p1, p2, *polygons) -> bool:
    """True iff segment p1–p2 doesn't properly cross any polygon edge."""
    for poly in polygons:
        n = len(poly)
        for k in range(n):
            a, b = poly[k], poly[(k + 1) % n]
            if (_points_equal(a, p1) or _points_equal(a, p2) or
                    _points_equal(b, p1) or _points_equal(b, p2)):
                continue
            if _segments_proper_intersect(p1, p2, a, b):
                return False
    return True


def _decompose_polygon_with_holes(outer_poly: np.ndarray,
                                   inner_polys: list) -> list[np.ndarray]:
    """Triangulate the region bounded by an outer CCW polygon with zero or
    more CCW inner polygons (holes).

    For 0 holes: just ear-clip the outer.
    For 1 hole: bridge + ear-clip (my custom code, used when mapbox_earcut
      isn't installed — works reliably for single-hole cases like USB-A).
    For 2+ holes: mapbox_earcut (required). Multi-hole via naive bridging
      creates coincident vertices at shared bridge targets which my simple
      ear-clipper can't handle; mapbox_earcut's O'Rourke-style ray-casting
      produces a robust simple polygon.
    """
    if not inner_polys:
        return _ear_clip_simple(outer_poly)

    if not _HAS_EARCUT and len(inner_polys) == 1:
        # Fall back to custom bridging for the single-hole case.
        inner = np.asarray(inner_polys[0])[::-1].copy()
        # Find bridge via brute-force visibility search (subset of earcut's job).
        bridge = None
        for ii in range(len(inner)):
            for jj in range(len(outer_poly)):
                if _find_bridge_against(inner[ii], outer_poly[jj],
                                         outer_poly, inner):
                    bridge = (ii, jj)
                    break
            if bridge is not None:
                break
        if bridge is None:
            raise ValueError("Could not find bridge between outer and hole.")
        ii, jj = bridge
        bridged = []
        for k in range(len(outer_poly) + 1):
            bridged.append(outer_poly[(jj + k) % len(outer_poly)])
        for k in range(len(inner) + 1):
            bridged.append(inner[(ii + k) % len(inner)])
        return _ear_clip_simple(np.array(bridged))

    if not _HAS_EARCUT:
        raise RuntimeError(
            "Multi-hole (>=2) polygon decomposition needs mapbox_earcut. "
            "Install with: pip install mapbox_earcut")

    # Use mapbox_earcut for the general case. Convention: outer CCW, holes CW.
    outer = np.asarray(outer_poly, dtype=np.float64)
    holes_cw = [np.asarray(ip, dtype=np.float64)[::-1] for ip in inner_polys]
    rings = [outer] + holes_cw
    vertices = np.concatenate(rings, axis=0)
    # `rings_end_indices` marks where each ring ENDS (exclusive) in `vertices`.
    ring_end = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tri_idx = _earcut.triangulate_float64(vertices, ring_end)
    triangles = []
    for i in range(0, len(tri_idx), 3):
        triangles.append(np.array([vertices[tri_idx[i]],
                                   vertices[tri_idx[i + 1]],
                                   vertices[tri_idx[i + 2]]]))
    return triangles


def _decompose_plate_general(hole_poly: np.ndarray, plate_half: float) -> list[np.ndarray]:
    """Plate-minus-hole decomposition for arbitrary hole polygons."""
    outer = np.array([
        [-plate_half, -plate_half],
        [ plate_half, -plate_half],
        [ plate_half,  plate_half],
        [-plate_half,  plate_half],
    ])
    return _decompose_annular(outer, hole_poly)


def _decompose_plate(hole_poly: np.ndarray, plate_half: float) -> list[np.ndarray]:
    """Radial decomposition of (plate square) minus (hole polygon) into convex wedges."""
    centroid = hole_poly.mean(axis=0)
    plate_corners = np.array([
        [ plate_half,  plate_half],
        [-plate_half,  plate_half],
        [-plate_half, -plate_half],
        [ plate_half, -plate_half],
    ])

    def ang(p):
        return np.arctan2(p[1] - centroid[1], p[0] - centroid[0])

    def in_ccw_arc(a, start, end):
        span = (end - start) % (2 * np.pi)
        off = (a - start) % (2 * np.pi)
        return 1e-9 < off < span - 1e-9

    n = len(hole_poly)
    wedges = []
    for i in range(n):
        v0 = hole_poly[i]
        v1 = hole_poly[(i + 1) % n]
        o0 = _ray_to_square(centroid, v0 - centroid, plate_half)
        o1 = _ray_to_square(centroid, v1 - centroid, plate_half)
        a0, a1 = ang(v0), ang(v1)
        verts = [v0, v1, o1]
        # Any plate corner in the CCW arc from v0 to v1 needs to be included so
        # the wedge stays convex (otherwise it'd cut across the plate corner).
        corners_between = sorted(
            [(ang(c), c) for c in plate_corners if in_ccw_arc(ang(c), a0, a1)],
            key=lambda x: x[0], reverse=True,
        )
        for _, c in corners_between:
            verts.append(c)
        verts.append(o0)
        wedges.append(np.asarray(verts))
    return wedges


def _fan_triangulate(poly: np.ndarray) -> list[np.ndarray]:
    """Fan triangulation from centroid. For star-shaped polygons this yields
    a valid convex decomposition (each triangle is convex)."""
    centroid = poly.mean(axis=0)
    n = len(poly)
    return [np.array([centroid, poly[i], poly[(i + 1) % n]]) for i in range(n)]


# ----- config-driven scene builders ---------------------------------------

def _shape_stadium(w: float, h: float, n_cap: int = 12) -> np.ndarray:
    """Rounded-rectangle (stadium) polygon centered at origin."""
    r = min(w, h) / 2
    if w >= h:
        straight = w - 2 * r
        a = np.linspace(-np.pi / 2, np.pi / 2, n_cap)
        right = np.column_stack([straight / 2 + r * np.cos(a), r * np.sin(a)])
        a = np.linspace(np.pi / 2, 3 * np.pi / 2, n_cap)
        left = np.column_stack([-straight / 2 + r * np.cos(a), r * np.sin(a)])
        return np.vstack([right, left])
    straight = h - 2 * r
    a = np.linspace(0, np.pi, n_cap)
    top = np.column_stack([r * np.cos(a), straight / 2 + r * np.sin(a)])
    a = np.linspace(np.pi, 2 * np.pi, n_cap)
    bot = np.column_stack([r * np.cos(a), -straight / 2 + r * np.sin(a)])
    return np.vstack([top, bot])


def _resolve_shape(cfg: dict, config_dir: Path | None = None) -> np.ndarray:
    """Config dict -> 2D polygon (CCW, centered on origin by construction).
    Supports `center: [x,y]` and `rotate: radians` transformations.
    """
    t = cfg['type']
    if t == 'circle':
        r, n = cfg['radius'], cfg.get('n', 32)
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        verts = np.column_stack([r * np.cos(a), r * np.sin(a)])
    elif t == 'rect':
        w, h = cfg['width'], cfg['height']
        verts = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    elif t == 'stadium':
        verts = _shape_stadium(cfg['width'], cfg['height'], cfg.get('n_cap', 12))
    elif t == 'trapezoid':
        bw, tw, h = cfg['bottom_width'], cfg['top_width'], cfg['height']
        verts = np.array([[-bw/2, -h/2], [bw/2, -h/2], [tw/2, h/2], [-tw/2, h/2]])
    elif t == 'regular_ngon':
        n, r = cfg['n'], cfg['circumradius']
        a = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
        verts = np.column_stack([r * np.cos(a), r * np.sin(a)])
    elif t == 'star':
        n_pts = cfg['n_points']
        outer_r, inner_r = cfg['outer_radius'], cfg['inner_radius']
        a = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 2 * n_pts, endpoint=False)
        rr = np.where(np.arange(2 * n_pts) % 2 == 0, outer_r, inner_r)
        verts = np.column_stack([rr * np.cos(a), rr * np.sin(a)])
    elif t == 'polygon':
        verts = np.asarray(cfg['vertices'], dtype=float)
    elif t == 'from_csv':
        p = Path(cfg['path'])
        if not p.is_absolute() and config_dir is not None:
            p = config_dir / p
        verts = np.loadtxt(p, delimiter=',')
    else:
        raise ValueError(f"Unknown shape type: {t!r}")

    angle = float(cfg.get('rotate', 0))
    if abs(angle) > 1e-12:
        c, s = np.cos(angle), np.sin(angle)
        verts = verts @ np.array([[c, -s], [s, c]])

    center = cfg.get('center', [0.0, 0.0])
    verts = verts + np.array(center, dtype=float)
    return verts


def generate_from_config(config: dict, config_dir: Path | None = None) -> str:
    """Build MJCF from a config dict. See myscripts/examples/*.yaml for the
    expected schema. Islands list can mix shape types freely."""
    hole = _resolve_shape(config['hole'], config_dir)
    island_cfgs = config.get('islands', []) or []
    islands = [_resolve_shape(c, config_dir) for c in island_cfgs]
    return generate_xml(
        hole_poly=hole,
        hole_island_polys=islands or None,
        peg_clearance=config['clearance'],
        plate_size=config['plate_size'],
        plate_thickness=config['plate_thickness'],
        peg_length=config['peg_length'],
        peg_height=config.get('peg_height', 0.1),
        include_peg=config.get('include_peg', True),
        include_floor=config.get('include_floor', True),
    )


def _load_config(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix in ('.yaml', '.yml'):
        import yaml
        return yaml.safe_load(path.read_text())
    if suffix == '.json':
        import json
        return json.loads(path.read_text())
    if suffix == '.toml':
        import tomllib
        return tomllib.loads(path.read_text())
    raise ValueError(f"Unsupported config extension {suffix!r} "
                     "(use .yaml, .yml, .json, or .toml)")


# ----- MJCF assembly -------------------------------------------------------

def _extrude_mesh_xml(name: str, poly2d: np.ndarray, thickness: float) -> str:
    verts = []
    for v in poly2d:
        verts.extend([v[0], v[1], -thickness / 2])
    for v in poly2d:
        verts.extend([v[0], v[1],  thickness / 2])
    vstr = ' '.join(f'{x:.6f}' for x in verts)
    return f'    <mesh name="{name}" vertex="{vstr}"/>'


def generate_xml(
    shape: str | None = 'circle',
    hole_size: float = 0.025,
    plate_size: float = 0.2,
    plate_thickness: float = 0.02,
    peg_clearance: float = 0.002,
    peg_length: float = 0.06,
    peg_height: float = 0.10,
    n_circle: int = 32,
    include_peg: bool = True,
    include_floor: bool = True,
    peg_poly: np.ndarray | None = None,
    hole_poly: np.ndarray | None = None,
    hole_island_polys: list | np.ndarray | None = None,
) -> str:
    """Build MJCF for peg-in-hole.

    Shape source — pass exactly one (in priority order):
      peg_poly  : 2D peg polygon; hole = outward offset by peg_clearance.
      hole_poly : 2D hole polygon; peg = inward offset by peg_clearance.
      shape     : name from SHAPES.

    hole_island_polys (optional, only with hole_poly): a list of island
    polygons sitting inside the hole — e.g. the tongue of a USB-A receptacle
    (1 island), or the 9 pins of a D-sub connector. When provided, the plate
    gets island geoms AND the peg is auto-generated as a hollow shell with
    one pocket per island, each pocket surrounding its island with
    peg_clearance. A single ndarray is accepted as syntactic sugar for a
    one-island list.

    Custom polygons are auto-centered on their hole's vertex centroid and
    auto-reoriented to CCW. Islands shift with the parent hole to preserve
    relative positioning.
    """
    islands: list[np.ndarray] = []
    peg_outer: np.ndarray | None = None
    peg_inners: list[np.ndarray] = []

    # Normalize islands input to a list of polygons.
    if hole_island_polys is None:
        island_inputs = []
    elif isinstance(hole_island_polys, np.ndarray) and hole_island_polys.ndim == 2:
        island_inputs = [hole_island_polys]
    else:
        island_inputs = list(hole_island_polys)

    if peg_poly is not None:
        p = _ensure_ccw(np.asarray(peg_poly, dtype=float))
        peg = p - p.mean(axis=0)
        hole = _offset_polygon_outward(peg, peg_clearance)
        shape = None
    elif hole_poly is not None:
        h = _ensure_ccw(np.asarray(hole_poly, dtype=float))
        shift = h.mean(axis=0)
        hole = h - shift
        if island_inputs:
            islands = [_ensure_ccw(np.asarray(isl, dtype=float)) - shift
                       for isl in island_inputs]
            peg_outer = _offset_polygon_inward(hole, peg_clearance)
            peg_inners = [_offset_polygon_outward(isl, peg_clearance)
                          for isl in islands]
            peg = peg_outer  # placeholder for plate_half bound check
        else:
            peg = _offset_polygon_inward(hole, peg_clearance)
        shape = None
    else:
        if shape not in SHAPES:
            raise ValueError(f"Unknown shape {shape!r}. Known: {sorted(SHAPES)}")
        hole = np.asarray(SHAPES[shape](hole_size, n_circle) if shape == 'circle'
                          else SHAPES[shape](hole_size))
        peg = _offset_polygon_inward(hole, peg_clearance)

    plate_half = plate_size / 2
    if plate_half <= np.abs(hole).max() + 0.001:
        raise ValueError("plate_size must comfortably exceed the hole's bounding box.")

    # --- plate decomposition (always a polygon-with-hole around the opening)
    if _is_star_shaped_from(hole, hole.mean(axis=0)):
        wedges = _decompose_plate(hole, plate_half)
    else:
        wedges = _decompose_plate_general(hole, plate_half)

    # --- island decomposition: each island becomes a list of convex pieces
    # attached to the plate body as extra material (e.g. USB-A tongue, D-sub
    # pin posts). Flattened into one list of meshes.
    all_island_pieces: list[np.ndarray] = []
    for isl in islands:
        all_island_pieces.extend(
            [isl] if _is_convex_poly(isl) else _ear_clip_simple(isl))

    # --- peg decomposition
    if peg_outer is not None and peg_inners:
        # Hollow-shell peg with one or more internal hollows (USB-A: 1 pocket,
        # D-sub: 9 pockets, circular MIL connectors: N pockets).
        peg_pieces = _decompose_polygon_with_holes(peg_outer, peg_inners)
        peg_convex = False
    else:
        peg_convex = _is_convex_poly(peg)
        if peg_convex:
            peg_pieces = None
        elif _is_star_shaped_from(peg, peg.mean(axis=0)):
            peg_pieces = _fan_triangulate(peg)
        else:
            peg_pieces = _ear_clip_simple(peg)

    # --- collect peg geom specs: (mesh_name, poly2d, extrusion_thickness, z_offset).
    # For a hollow-shell peg with islands, hollow(s) only extend as deep as the
    # islands, so split the peg into ring (bottom) + solid back (top).
    peg_geom_specs: list[tuple[str, np.ndarray, float, float]] = []
    if include_peg and shape != 'circle':
        if peg_outer is not None and peg_inners:
            pocket_depth = plate_thickness + peg_clearance
            if pocket_depth >= peg_length:
                raise ValueError(
                    "peg_length must exceed plate_thickness + peg_clearance "
                    "(otherwise there's no solid back to close the plug).")
            back_thick = peg_length - pocket_depth
            ring_z = -peg_length / 2 + pocket_depth / 2
            back_z = pocket_depth / 2
            for i, p in enumerate(peg_pieces):
                peg_geom_specs.append((f'peg_ring_{i}', p, pocket_depth, ring_z))
            peg_geom_specs.append(('peg_back', peg_outer, back_thick, back_z))
        elif peg_convex:
            peg_geom_specs.append(('peg_0', peg, peg_length, 0.0))
        else:
            for i, p in enumerate(peg_pieces):
                peg_geom_specs.append((f'peg_{i}', p, peg_length, 0.0))

    # ---- assets
    assets = ['  <asset>']
    for i, w in enumerate(wedges):
        assets.append(_extrude_mesh_xml(f'plate_{i}', w, plate_thickness))
    for i, p in enumerate(all_island_pieces):
        assets.append(_extrude_mesh_xml(f'island_{i}', p, plate_thickness))
    for name, poly, thick, _ in peg_geom_specs:
        assets.append(_extrude_mesh_xml(name, poly, thick))
    assets.append('  </asset>')

    # ---- worldbody
    wb = [
        '  <worldbody>',
        '    <light pos="0.3 0.3 1.2" dir="-0.3 -0.3 -1" diffuse="0.8 0.8 0.8"/>',
    ]
    if include_floor:
        wb.append('    <geom name="floor" type="plane" size="1 1 0.01" rgba="0.92 0.92 0.92 1"/>')
    wb.append(f'    <body name="plate" pos="0 0 {plate_thickness / 2:.6f}">')
    for i in range(len(wedges)):
        wb.append(f'      <geom type="mesh" mesh="plate_{i}" rgba="0.70 0.75 0.82 1"/>')
    for i in range(len(all_island_pieces)):
        wb.append(f'      <geom type="mesh" mesh="island_{i}" rgba="0.70 0.75 0.82 1"/>')
    wb.append('    </body>')

    if include_peg:
        peg_z = plate_thickness + peg_height
        lateral = peg_clearance * 0.3
        wb.append(f'    <body name="peg" pos="{lateral:.6f} 0 {peg_z:.6f}">')
        wb.append('      <freejoint/>')
        if shape == 'circle':
            wb.append(f'      <geom type="cylinder" '
                      f'size="{hole_size - peg_clearance:.6f} {peg_length / 2:.6f}" '
                      f'rgba="0.90 0.30 0.30 1"/>')
        else:
            for name, _, _, pos_z in peg_geom_specs:
                pos_attr = f' pos="0 0 {pos_z:.6f}"' if abs(pos_z) > 1e-9 else ''
                wb.append(f'      <geom type="mesh" mesh="{name}"{pos_attr} '
                          f'rgba="0.90 0.30 0.30 1"/>')
        wb.append('    </body>')
    wb.append('  </worldbody>')

    header = [
        '<mujoco model="peg_in_hole">',
        '  <option timestep="0.001" integrator="implicitfast" gravity="0 0 -9.81"/>',
        '  <default>',
        '    <geom solref="0.002 1" solimp="0.99 0.999 0.001" '
        'condim="4" friction="1.0 0.01 0.001"/>',
        '  </default>',
    ]
    return '\n'.join(header + assets + wb + ['</mujoco>']) + '\n'


# ----- CLI -----------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--shape', choices=sorted(SHAPES), default='circle',
                   help='named shape (ignored if --peg-csv or --hole-csv is given)')
    p.add_argument('--peg-csv', type=Path,
                   help='load peg polygon from CSV (x,y per line); '
                        'hole is generated as outward offset by --peg-clearance')
    p.add_argument('--hole-csv', type=Path,
                   help='load hole polygon from CSV (x,y per line); '
                        'peg is generated as inward offset by --peg-clearance')
    p.add_argument('--hole-island-csv', action='append', dest='hole_island_csvs',
                   type=Path, default=[],
                   help='optional island of material inside the hole '
                        '(e.g. USB-A tongue, D-sub pin post). May be passed '
                        'multiple times for connectors with multiple islands. '
                        'When present, peg is auto-generated as a hollow shell '
                        'with one pocket per island.')
    p.add_argument('--hole-size', type=float, default=0.025,
                   help='circumradius of hole (outer radius for star)')
    p.add_argument('--plate-size', type=float, default=0.2,
                   help='plate side length (square plate)')
    p.add_argument('--plate-thickness', type=float, default=0.02)
    p.add_argument('--peg-clearance', type=float, default=0.0005,
                   help='radial clearance (hole minus peg)')
    p.add_argument('--peg-length', type=float, default=0.06)
    p.add_argument('--peg-height', type=float, default=0.10,
                   help='initial height of peg above plate top surface')
    p.add_argument('--n-circle', type=int, default=128,
                   help='polygonal sampling of the circle shape')
    p.add_argument('--no-peg', action='store_true')
    p.add_argument('--no-floor', action='store_true',
                   help='omit the ground plane; output contains only the plate '
                        '(with hole) and the peg')
    p.add_argument('--config', type=Path,
                   help='load full scene from YAML/JSON/TOML config file. '
                        'When given, most other flags are ignored.')
    p.add_argument('--output', '-o', type=Path,
                   help='write MJCF here (default: stdout)')
    args = p.parse_args(argv)

    # Config-file path: bypass other flags entirely.
    if args.config:
        try:
            cfg = _load_config(args.config)
            xml = generate_from_config(cfg, config_dir=args.config.parent.resolve())
        except (ValueError, KeyError) as e:
            p.error(f"config error: {e}")
        if args.output:
            args.output.write_text(xml)
            print(f"wrote {args.output}", file=sys.stderr)
        else:
            sys.stdout.write(xml)
        return

    if args.peg_csv and args.hole_csv:
        p.error("--peg-csv and --hole-csv are mutually exclusive")
    if args.hole_island_csvs and not args.hole_csv:
        p.error("--hole-island-csv requires --hole-csv")
    peg_poly = np.loadtxt(args.peg_csv, delimiter=',') if args.peg_csv else None
    hole_poly = np.loadtxt(args.hole_csv, delimiter=',') if args.hole_csv else None
    hole_island_polys = [np.loadtxt(pth, delimiter=',')
                         for pth in args.hole_island_csvs]

    try:
        xml = generate_xml(
            shape=None if (peg_poly is not None or hole_poly is not None) else args.shape,
            hole_size=args.hole_size,
            plate_size=args.plate_size,
            plate_thickness=args.plate_thickness,
            peg_clearance=args.peg_clearance,
            peg_length=args.peg_length,
            peg_height=args.peg_height,
            n_circle=args.n_circle,
            include_peg=not args.no_peg,
            include_floor=not args.no_floor,
            peg_poly=peg_poly,
            hole_poly=hole_poly,
            hole_island_polys=hole_island_polys or None,
        )
    except ValueError as e:
        p.error(str(e))
    if args.output:
        args.output.write_text(xml)
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(xml)


if __name__ == '__main__':
    main()
