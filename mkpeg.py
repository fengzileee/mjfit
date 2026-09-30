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
import math
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


def _convex_polygon(vertices, name: str) -> np.ndarray:
    """Validate a finite convex outline, preserving its supplied coordinates."""
    poly = np.asarray(vertices, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or not np.isfinite(poly).all():
        raise ValueError(f"{name} needs at least three finite [x,y] vertices.")
    poly = _ensure_ccw(poly)
    edges = np.roll(poly, -1, axis=0) - poly
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 1e-12) or _signed_area(poly) <= 1e-12:
        raise ValueError(f"{name} must have positive area and nonzero edges.")
    normals = np.column_stack((-edges[:, 1], edges[:, 0])) / lengths[:, None]
    distances = np.einsum('ej,evj->ev', normals, poly[None] - poly[:, None])
    if np.any(distances < -1e-10):
        raise ValueError(f"{name} must be a simple convex polygon.")
    return poly


def _require_containment(outer: np.ndarray, inner: np.ndarray,
                         margin: float, name: str) -> None:
    edges = np.roll(outer, -1, axis=0) - outer
    normals = np.column_stack((-edges[:, 1], edges[:, 0]))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    distances = np.einsum('ej,evj->ev', normals, inner[None] - outer[:, None])
    if np.min(distances) < margin - 1e-10:
        raise ValueError(f"{name} does not contain its inner outline with {margin:g} m rim.")


def bezier_polygon(vertices, corner_rounding=0.5,
                   max_chord_error: float = 0.0001) -> np.ndarray:
    """Round convex control corners with quadratic Beziers, then tessellate.

    ``corner_rounding`` is one fraction or one per vertex in [0, 0.5]. Zero
    retains a sharp corner; 0.5 joins neighboring curves at edge midpoints.
    The Euclidean curve-to-chord error is at most ``max_chord_error`` meters.
    Control points and output share a frame; no centering/scaling is performed.
    """
    original = np.asarray(vertices, dtype=float)
    control = _convex_polygon(original, 'Bezier control polygon')
    alphas = np.asarray(corner_rounding, dtype=float)
    if alphas.ndim == 0:
        alphas = np.full(len(control), float(alphas))
    if alphas.shape != (len(control),) or not np.isfinite(alphas).all() or np.any((alphas < 0) | (alphas > 0.5)):
        raise ValueError("corner_rounding needs fractions in [0, 0.5], one per vertex.")
    if _signed_area(original) < 0:
        alphas = alphas[::-1]
    if not math.isfinite(max_chord_error) or max_chord_error <= 0:
        raise ValueError("max_chord_error must be finite and positive.")
    points = []
    for index, (vertex, alpha) in enumerate(zip(control, alphas)):
        if alpha == 0:
            candidates = [vertex]
        else:
            entry = vertex + alpha * (control[index - 1] - vertex)
            exit = vertex + alpha * (control[(index + 1) % len(control)] - vertex)
            curvature = np.linalg.norm(entry - 2 * vertex + exit)
            segments = max(1, math.ceil(math.sqrt(curvature / (4 * max_chord_error))))
            candidates = [
                (1 - t)**2 * entry + 2 * t * (1 - t) * vertex + t**2 * exit
                for t in np.linspace(0.0, 1.0, segments + 1)
            ]
        for point in candidates:
            if not points or np.linalg.norm(point - points[-1]) > 1e-12:
                points.append(point)
    if np.linalg.norm(points[0] - points[-1]) <= 1e-12:
        points.pop()
    return _convex_polygon(points, 'Tessellated Bezier polygon')


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


def _resolve_shape(cfg: dict, config_dir: Path | None = None,
                   scale: float = 1.0) -> np.ndarray:
    """Config dict -> 2D polygon (CCW, centered on origin by construction).
    Supports `center: [x,y]` and `rotate: radians` transformations.
    `scale` uniformly scales the resulting polygon (size + center offset).
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
    elif t == 'bezier_polygon':
        verts = bezier_polygon(cfg['vertices'], cfg.get('corner_rounding', 0.5),
                               cfg.get('max_chord_error', 0.0001))
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

    if scale != 1.0:
        verts = verts * scale
    return verts


def generate_from_config(config: dict, config_dir: Path | None = None,
                         mesh_dir: Path | None = None,
                         scale: float | None = None) -> str:
    """Build MJCF from a config dict. See myscripts/examples/*.yaml for the
    expected schema. Islands list can mix shape types freely.

    `scale` (if not None) overrides `config['scale']`. The effective scale is
    applied uniformly to every length: plate/peg dimensions, clearance, margin,
    and all shape sizes/centers inside the hole and islands.
    """
    if scale is None:
        scale = float(config.get('scale', 1.0))
    if ('hole' in config) == ('peg' in config):
        raise ValueError("Config requires exactly one of hole or peg.")
    hole = _resolve_shape(config['hole'], config_dir, scale=scale) if 'hole' in config else None
    peg = _resolve_shape(config['peg'], config_dir, scale=scale) if 'peg' in config else None
    handle = _resolve_shape(config['handle'], config_dir, scale=scale) if 'handle' in config else None
    plate = _resolve_shape(config['plate'], config_dir, scale=scale) if 'plate' in config else None
    island_cfgs = config.get('islands', []) or []
    islands = [_resolve_shape(c, config_dir, scale=scale) for c in island_cfgs]
    if islands and hole is None:
        raise ValueError("Islands require a hole shape, not a peg shape.")
    return generate_xml(
        hole_poly=hole,
        peg_poly=peg,
        handle_poly=handle,
        plate_poly=plate,
        hole_island_polys=islands or None,
        peg_clearance=config['clearance'] * scale,
        plate_size=config.get('plate_size', 0.2) * scale,
        plate_thickness=config['plate_thickness'] * scale,
        peg_length=config['peg_length'] * scale,
        peg_height=config.get('peg_height', 0.1) * scale,
        peg_density=config.get('peg_density', 1000.0),
        peg_mass_kg=config.get('peg_mass_kg'),
        plate_density=config.get('plate_density', 1000.0),
        handle_height=config.get('handle_height', 0.0) * scale,
        handle_margin=config.get('handle_margin', 0.003) * scale,
        plate_min_rim_m=config.get('plate_min_rim_m', 0.0) * scale,
        bottom_stop_thickness=config.get('bottom_stop_thickness', 0.0) * scale,
        center_shapes=config.get('center_shapes', True),
        include_peg=config.get('include_peg', True),
        include_floor=config.get('include_floor', True),
        mesh_only=config.get('mesh_only', False),
        free_plate=config.get('free_plate', False),
        mesh_dir=mesh_dir,
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

def _write_prism_obj(
    path: Path,
    solids: list[tuple[np.ndarray, list, float, float]],
    *, precise: bool = False,
) -> None:
    """Write one or more thickness-extruded polygons as a watertight OBJ.

    Each solid is `(outer, holes, thickness, z_offset)` — `outer` is a CCW
    polygon, `holes` is a (possibly empty) list of CCW interior polygons
    (by this codebase's convention; the triangulator flips to CW internally).
    Produces for each solid: triangulated top cap (+z normal), mirrored
    bottom cap (-z normal), walls around the outer boundary (outward), and
    walls around each hole (outward = pointing into the hole).

    Crucially, no interior walls between adjacent convex pieces — the
    triangulation handles the annular top/bottom as one big region, so there
    are no z-fighting shared faces (which is what makes the wedge-concat
    approach look glitchy in a renderer).
    """
    lines = ["# Generated by mkpeg.py"]
    face_lines: list[str] = []
    n_verts = 0

    def add_vert(x: float, y: float, z: float) -> int:
        nonlocal n_verts
        lines.append(f"v {x:.17g} {y:.17g} {z:.17g}" if precise
                     else f"v {x:.6f} {y:.6f} {z:.6f}")
        n_verts += 1
        return n_verts

    for outer, holes, thickness, z_offset in solids:
        half = thickness / 2
        z_bot = z_offset - half
        z_top = z_offset + half

        # Top + bottom caps from one annular triangulation.
        tris = _decompose_polygon_with_holes(outer, holes)
        for tri in tris:
            idx = [add_vert(p[0], p[1], z_top) for p in tri]
            face_lines.append(f"f {idx[0]} {idx[1]} {idx[2]}")
            idx = [add_vert(p[0], p[1], z_bot) for p in tri]
            face_lines.append(f"f {idx[0]} {idx[2]} {idx[1]}")

        # Outer walls: CCW polygon → outward normal points right of walk.
        n = len(outer)
        for i in range(n):
            v0, v1 = outer[i], outer[(i + 1) % n]
            b0 = add_vert(v0[0], v0[1], z_bot)
            b1 = add_vert(v1[0], v1[1], z_bot)
            t1 = add_vert(v1[0], v1[1], z_top)
            t0 = add_vert(v0[0], v0[1], z_top)
            face_lines.append(f"f {b0} {b1} {t1}")
            face_lines.append(f"f {b0} {t1} {t0}")

        # Hole walls: reversed — outward normal points into the hole (away
        # from the surrounding material).
        for hole in holes:
            nh = len(hole)
            for i in range(nh):
                v0, v1 = hole[i], hole[(i + 1) % nh]
                b0 = add_vert(v0[0], v0[1], z_bot)
                b1 = add_vert(v1[0], v1[1], z_bot)
                t1 = add_vert(v1[0], v1[1], z_top)
                t0 = add_vert(v0[0], v0[1], z_top)
                face_lines.append(f"f {b0} {t1} {b1}")
                face_lines.append(f"f {b0} {t0} {t1}")

    lines.extend(face_lines)
    path.write_text("\n".join(lines) + "\n")


def _extrude_mesh_xml(name: str, poly2d: np.ndarray, thickness: float,
                      mesh_dir: Path | None = None, *, precise: bool = False) -> str:
    """Return MJCF `<mesh>` element for an extruded prism.

    With `mesh_dir`: writes `{name}.obj` and emits a `file="..."` reference
    (required for USD conversion — inline vertex meshes aren't portable).
    Without: emits MuJoCo's inline `vertex="..."` form.
    """
    if mesh_dir is not None:
        _write_prism_obj(mesh_dir / f'{name}.obj',
                         [(poly2d, [], thickness, 0.0)], precise=precise)
        return f'    <mesh name="{name}" file="{name}.obj"/>'
    verts = []
    for v in poly2d:
        verts.extend([v[0], v[1], -thickness / 2])
    for v in poly2d:
        verts.extend([v[0], v[1],  thickness / 2])
    vstr = ' '.join(f'{x:.17g}' if precise else f'{x:.6f}' for x in verts)
    return f'    <mesh name="{name}" vertex="{vstr}"/>'


def generate_xml(
    shape: str | None = 'circle',
    hole_size: float = 0.025,
    plate_size: float = 0.2,
    plate_thickness: float = 0.02,
    peg_clearance: float = 0.002,
    peg_length: float = 0.06,
    peg_height: float = 0.10,
    peg_density: float = 1000.0,
    plate_density: float = 1000.0,
    handle_height: float = 0.0,
    handle_margin: float = 0.003,
    n_circle: int = 32,
    include_peg: bool = True,
    include_floor: bool = True,
    mesh_only: bool = False,
    free_plate: bool = False,
    mesh_dir: Path | None = None,
    peg_poly: np.ndarray | None = None,
    hole_poly: np.ndarray | None = None,
    hole_island_polys: list | np.ndarray | None = None,
    handle_poly: np.ndarray | None = None,
    plate_poly: np.ndarray | None = None,
    peg_mass_kg: float | None = None,
    plate_min_rim_m: float = 0.0,
    bottom_stop_thickness: float = 0.0,
    center_shapes: bool = True,
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

    Custom polygons are auto-centered on the supplied source's vertex mean and
    auto-reoriented to CCW. Islands shift with the parent hole to preserve
    relative positioning. Explicit handle_poly and convex plate_poly use that
    same translation. center_shapes=False preserves all input coordinates.
    The explicit handle must contain the insertion outline; the plate must
    contain the hole and handle with plate_min_rim_m perpendicular edge margin.
    bottom_stop_thickness adds a floor following the plate's outer outline.
    peg_mass_kg sets the TOTAL insertion-plus-handle mass, split by volume;
    otherwise the historical peg_density behavior is retained.
    """
    islands: list[np.ndarray] = []
    peg_outer: np.ndarray | None = None
    peg_inners: list[np.ndarray] = []
    precise = (handle_poly is not None or plate_poly is not None
               or peg_mass_kg is not None or not center_shapes
               or bottom_stop_thickness > 0)
    if precise and (not math.isfinite(peg_clearance) or peg_clearance < 0):
        raise ValueError("peg_clearance must be finite and nonnegative.")
    if precise:
        for name, value in (('peg_length', peg_length), ('plate_thickness', plate_thickness)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(handle_height) or handle_height < 0:
            raise ValueError("handle_height must be finite and nonnegative.")
        if handle_poly is None and handle_height > 0 and (not math.isfinite(handle_margin) or handle_margin < 0):
            raise ValueError("Generated handle_margin must be finite and nonnegative.")
    if peg_mass_kg is not None and (not math.isfinite(peg_mass_kg) or peg_mass_kg <= 0 or not include_peg):
        raise ValueError("peg_mass_kg needs a finite positive total mass and include_peg=True.")
    if not math.isfinite(plate_min_rim_m) or plate_min_rim_m < 0:
        raise ValueError("plate_min_rim_m must be finite and nonnegative.")
    if plate_min_rim_m > 0 and plate_poly is None:
        raise ValueError("plate_min_rim_m requires an explicit plate_poly.")
    if not math.isfinite(bottom_stop_thickness) or bottom_stop_thickness < 0:
        raise ValueError("bottom_stop_thickness must be finite and nonnegative.")
    if handle_poly is not None and (handle_height <= 0 or not include_peg):
        raise ValueError("handle_poly requires positive handle_height and include_peg=True.")

    # Normalize islands input to a list of polygons.
    if hole_island_polys is None:
        island_inputs = []
    elif isinstance(hole_island_polys, np.ndarray) and hole_island_polys.ndim == 2:
        island_inputs = [hole_island_polys]
    else:
        island_inputs = list(hole_island_polys)

    if peg_poly is not None:
        p = _ensure_ccw(np.asarray(peg_poly, dtype=float))
        shift = p.mean(axis=0) if center_shapes else np.zeros(2)
        peg = p - shift
        hole = _offset_polygon_outward(peg, peg_clearance)
        shape = None
    elif hole_poly is not None:
        h = _ensure_ccw(np.asarray(hole_poly, dtype=float))
        shift = h.mean(axis=0) if center_shapes else np.zeros(2)
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
        shift = np.zeros(2)
        if shape not in SHAPES:
            raise ValueError(f"Unknown shape {shape!r}. Known: {sorted(SHAPES)}")
        if shape == 'circle':
            # The peg is an analytic cylinder of radius hole_size - clearance,
            # so the n-gon must circumscribe radius hole_size: its flats then
            # sit exactly at hole_size (gap = clearance) instead of cutting a
            # chord of depth hole_size * (1 - cos(pi/n)) into the clearance.
            hole = _circle(hole_size / np.cos(np.pi / n_circle), n_circle)
        else:
            hole = np.asarray(SHAPES[shape](hole_size))
        peg = _offset_polygon_inward(hole, peg_clearance)

    if precise and _is_convex_poly(hole):
        hole = _convex_polygon(hole, 'Hole outline')
        peg = _convex_polygon(peg, 'Offset insertion outline')
        _require_containment(hole, peg, peg_clearance, 'Hole clearance')

    # Custom handles on the built-in circle use its sampled convex prism so
    # both supplied outlines retain their exact common insertion-plane frame.
    if handle_poly is not None:
        shape = None
    plate_half = plate_size / 2
    plate_outline = np.array([[-plate_half, -plate_half], [plate_half, -plate_half],
                              [plate_half, plate_half], [-plate_half, plate_half]])
    if plate_poly is not None:
        plate_outline = _convex_polygon(plate_poly, 'Plate outline') - shift
        _require_containment(plate_outline, hole, max(plate_min_rim_m, 1e-9), 'Plate')
    elif plate_half <= np.abs(hole).max() + 0.001:
        raise ValueError("plate_size must comfortably exceed the hole's bounding box.")

    # --- plate decomposition (always a polygon-with-hole around the opening)
    if plate_poly is not None:
        wedges = _decompose_annular(plate_outline, hole)
    elif _is_star_shaped_from(hole, hole.mean(axis=0)):
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

    # --- optional grip handle on top of the peg (plastic connector housing
    # pattern: a wider section the user / robot can grasp without fitting
    # through the hole). Handle outline = peg outline offset outward.
    handle_outline: np.ndarray | None = None
    if handle_height > 0 and include_peg and shape != 'circle':
        base = peg_outer if peg_outer is not None else peg
        if handle_poly is None:
            handle_outline = _offset_polygon_outward(base, handle_margin)
        else:
            handle_outline = _convex_polygon(handle_poly, 'Handle outline') - shift
            _require_containment(handle_outline, base, 0.0, 'Handle')
        if precise and _is_convex_poly(base):
            handle_outline = _convex_polygon(handle_outline, 'Handle outline')
            _require_containment(handle_outline, base, 0.0, 'Handle')
        handle_z = (peg_length + handle_height) / 2
        if _is_convex_poly(handle_outline):
            handle_pieces = [handle_outline]
        elif _is_star_shaped_from(handle_outline, handle_outline.mean(axis=0)):
            handle_pieces = _fan_triangulate(handle_outline)
        else:
            handle_pieces = _ear_clip_simple(handle_outline)
        for i, piece in enumerate(handle_pieces):
            peg_geom_specs.append((f'handle_{i}', piece, handle_height, handle_z))

    if plate_poly is not None and handle_height > 0 and include_peg:
        checked_handle = handle_outline
        if shape == 'circle':
            # The built-in handle is an analytic cylinder, so test its exact
            # support radius rather than an inscribed polygon approximation.
            radius = hole_size - peg_clearance + handle_margin
            _require_containment(plate_outline, np.zeros((1, 2)),
                                 radius + plate_min_rim_m, 'Plate around handle')
        else:
            _require_containment(plate_outline, checked_handle, plate_min_rim_m, 'Plate around handle')

    masses = {}
    if peg_mass_kg is not None:
        if shape == 'circle':
            volumes = {'cylinder': np.pi * (hole_size - peg_clearance)**2 * peg_length}
            if handle_height > 0:
                volumes['handle_cylinder'] = np.pi * (hole_size - peg_clearance + handle_margin)**2 * handle_height
        else:
            volumes = {name: abs(_signed_area(poly)) * thick
                       for name, poly, thick, _ in peg_geom_specs}
        if any(volume <= 0 for volume in volumes.values()):
            raise ValueError("Total mass requires positive insertion/handle volumes.")
        total_volume = sum(volumes.values())
        masses = {name: peg_mass_kg * volume / total_volume for name, volume in volumes.items()}

    export_visuals = mesh_dir is not None
    # Peg visual mesh applies only when peg is actually a mesh (not the
    # built-in cylinder primitive used by shape='circle').
    peg_has_visual = export_visuals and shape != 'circle' and include_peg

    # ---- assets
    if mesh_dir is not None:
        mesh_dir.mkdir(parents=True, exist_ok=True)
    assets = ['  <asset>']
    if export_visuals:
        # Build the plate visual from the ORIGINAL outer/hole boundaries so
        # the OBJ has no interior seams (wedge-concat produced z-fighting).
        plate_solids: list = [(plate_outline, [hole], plate_thickness, 0.0)]
        for isl in islands:
            plate_solids.append((isl, [], plate_thickness, 0.0))
        _write_prism_obj(mesh_dir / 'plate_visual.obj', plate_solids, precise=precise)
        assets.append('    <mesh name="plate_visual" file="plate_visual.obj"/>')
        if peg_has_visual:
            if peg_outer is not None and peg_inners:
                # Hollow-shell peg: annular ring (at pocket depth) + solid back.
                pocket_depth = plate_thickness + peg_clearance
                back_thick = peg_length - pocket_depth
                ring_z = -peg_length / 2 + pocket_depth / 2
                back_z = pocket_depth / 2
                peg_solids: list = [
                    (peg_outer, list(peg_inners), pocket_depth, ring_z),
                    (peg_outer, [], back_thick, back_z),
                ]
            else:
                peg_solids = [(peg, [], peg_length, 0.0)]
            _write_prism_obj(mesh_dir / 'peg_visual.obj', peg_solids, precise=precise)
            assets.append('    <mesh name="peg_visual" file="peg_visual.obj"/>')
            if handle_outline is not None:
                handle_z = (peg_length + handle_height) / 2
                _write_prism_obj(mesh_dir / 'handle_visual.obj',
                                 [(handle_outline, [], handle_height, handle_z)],
                                 precise=precise)
                assets.append(
                    '    <mesh name="handle_visual" file="handle_visual.obj"/>')
    for i, w in enumerate(wedges):
        assets.append(_extrude_mesh_xml(f'plate_{i}', w, plate_thickness, mesh_dir, precise=precise))
    for i, p in enumerate(all_island_pieces):
        assets.append(_extrude_mesh_xml(f'island_{i}', p, plate_thickness, mesh_dir, precise=precise))
    for name, poly, thick, _ in peg_geom_specs:
        assets.append(_extrude_mesh_xml(name, poly, thick, mesh_dir, precise=precise))
    if bottom_stop_thickness > 0:
        assets.append(_extrude_mesh_xml('bottom_stop', plate_outline,
                                        bottom_stop_thickness, mesh_dir, precise=True))
    assets.append('  </asset>')

    def _geom_line(mesh: str, rgba: str, density: float, *, pos_z: float = 0.0,
                   mass: float | None = None) -> str:
        """Assemble one collision-or-combined mesh geom. When export_visuals
        is on, omit rgba (visual mesh carries color) and tag class="collision";
        otherwise emit rgba inline and no class."""
        parts = [f'type="mesh"', f'mesh="{mesh}"']
        if abs(pos_z) > 1e-9:
            parts.append(f'pos="0 0 {pos_z:.17g}"' if precise else f'pos="0 0 {pos_z:.6f}"')
        if export_visuals:
            parts.append('class="collision"')
        else:
            parts.append(f'rgba="{rgba}"')
        if mass is not None:
            parts.append(f'mass="{mass:.17g}"')
        elif abs(density - 1000.0) > 1e-9:
            parts.append(f'density="{density:.3f}"')
        return f'      <geom {" ".join(parts)}/>'

    # ---- worldbody
    wb = ['  <worldbody>']
    if not mesh_only:
        wb.append('    <light pos="0.3 0.3 1.2" dir="-0.3 -0.3 -1" diffuse="0.8 0.8 0.8"/>')
    if include_floor and not mesh_only:
        wb.append('    <geom name="floor" type="plane" size="1 1 0.01" rgba="0.92 0.92 0.92 1"/>')
    plate_z = format(plate_thickness / 2, '.17g' if precise else '.6f')
    wb.append(f'    <body name="plate" pos="0 0 {plate_z}">')
    if free_plate:
        wb.append('      <freejoint/>')
    if export_visuals:
        wb.append('      <geom type="mesh" mesh="plate_visual" '
                  'class="visual" rgba="0.70 0.75 0.82 1"/>')
    for i in range(len(wedges)):
        wb.append(_geom_line(f'plate_{i}', '0.70 0.75 0.82 1', plate_density))
    for i in range(len(all_island_pieces)):
        wb.append(_geom_line(f'island_{i}', '0.70 0.75 0.82 1', plate_density))
    if bottom_stop_thickness > 0:
        wb.append(_geom_line('bottom_stop', '0.25 0.25 0.25 1', plate_density,
                             pos_z=-(plate_thickness + bottom_stop_thickness) / 2))
    wb.append('    </body>')

    if include_peg:
        peg_z = plate_thickness + peg_height
        lateral = peg_clearance * 0.3 if center_shapes else 0.0
        peg_pos = (f'{lateral:.17g} 0 {peg_z:.17g}' if precise
                   else f'{lateral:.6f} 0 {peg_z:.6f}')
        wb.append(f'    <body name="peg" pos="{peg_pos}">')
        wb.append('      <freejoint/>')
        if peg_has_visual:
            wb.append('      <geom type="mesh" mesh="peg_visual" '
                      'class="visual" rgba="0.90 0.30 0.30 1"/>')
            if handle_outline is not None:
                wb.append('      <geom type="mesh" mesh="handle_visual" '
                          'class="visual" rgba="0.55 0.30 0.30 1"/>')
        if shape == 'circle':
            # Cylinder primitive — one geom does both render and collide
            # even in export mode (no class; class="visual" would zero mass).
            density_attr = (f' density="{peg_density:.3f}"'
                            if abs(peg_density - 1000.0) > 1e-9 else '')
            if peg_mass_kg is not None:
                density_attr = f' mass="{masses["cylinder"]:.17g}"'
            wb.append(f'      <geom type="cylinder" '
                      f'size="{hole_size - peg_clearance:.6f} {peg_length / 2:.6f}" '
                      f'rgba="0.90 0.30 0.30 1"{density_attr}/>')
            if handle_height > 0:
                handle_radius = (hole_size - peg_clearance) + handle_margin
                handle_z = (peg_length + handle_height) / 2
                if peg_mass_kg is not None:
                    density_attr = f' mass="{masses["handle_cylinder"]:.17g}"'
                wb.append(f'      <geom type="cylinder" '
                          f'size="{handle_radius:.6f} {handle_height / 2:.6f}" '
                          f'pos="0 0 {handle_z:.6f}" '
                          f'rgba="0.55 0.30 0.30 1"{density_attr}/>')
        else:
            for name, _, _, pos_z in peg_geom_specs:
                rgba = ('0.55 0.30 0.30 1' if name.startswith('handle_')
                        else '0.90 0.30 0.30 1')
                wb.append(_geom_line(name, rgba, peg_density, pos_z=pos_z,
                                     mass=masses.get(name)))
        wb.append('    </body>')
    wb.append('  </worldbody>')

    header = ['<mujoco model="peg_in_hole">']
    need_default = (not mesh_only) or export_visuals
    if not mesh_only:
        header.append(
            '  <option timestep="0.001" integrator="implicitfast" gravity="0 0 -9.81"/>')
    if need_default:
        header.append('  <default>')
        if not mesh_only:
            header.append('    <geom solref="0.002 1" solimp="0.99 0.999 0.001" '
                          'condim="4" friction="1.0 0.01 0.001"/>')
        if export_visuals:
            header.append('    <default class="visual">')
            header.append('      <geom group="2" contype="0" conaffinity="0" density="0"/>')
            header.append('    </default>')
            header.append('    <default class="collision">')
            header.append('      <geom group="3"/>')
            header.append('    </default>')
        header.append('  </default>')
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
    p.add_argument('--handle-csv', type=Path,
                   help='explicit convex handle outline containing the peg; '
                        'shares the input hole/peg coordinate frame')
    p.add_argument('--plate-csv', type=Path,
                   help='explicit convex outer plate outline, containing hole and handle')
    p.add_argument('--plate-min-rim-m', type=float, default=0.0,
                   help='minimum plate rim beyond hole and handle (requires --plate-csv)')
    p.add_argument('--bottom-stop-thickness', type=float, default=0.0,
                   help='optional floor thickness below the opening, following the plate outline')
    p.add_argument('--preserve-origin', action='store_true',
                   help='retain input coordinates and omit the default lateral spawn offset')
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
    p.add_argument('--peg-density', type=float, default=1000.0,
                   help='peg material density in kg/m^3 (default 1000 = water). '
                        'MuJoCo derives mass and inertia from density * volume.')
    p.add_argument('--peg-mass-kg', type=float,
                   help='TOTAL insertion-plus-handle mass, distributed by volume; '
                        'overrides --peg-density and is not scaled by --scale')
    p.add_argument('--plate-density', type=float, default=1000.0,
                   help='plate material density in kg/m^3 (default 1000). '
                        'Only matters when --free-plate is set.')
    p.add_argument('--handle-height', type=float, default=0.0,
                   help='vertical extent of an optional grip handle stacked '
                        'on top of the peg (default 0 = no handle). Shares '
                        'the peg body, so it moves rigidly with the peg.')
    p.add_argument('--handle-margin', type=float, default=0.003,
                   help='radial outward offset of the handle from the peg '
                        'outline (default 0.003 m). Make it larger than '
                        'peg_clearance so the handle cannot fit through '
                        'the hole.')
    p.add_argument('--free-plate', action='store_true',
                   help='give the plate a freejoint so it is dynamic. By '
                        'default the plate is welded to the world.')
    p.add_argument('--n-circle', type=int, default=128,
                   help='polygonal sampling of the circle shape')
    p.add_argument('--no-peg', action='store_true')
    p.add_argument('--no-floor', action='store_true',
                   help='omit the ground plane; output contains only the plate '
                        '(with hole) and the peg')
    p.add_argument('--mesh-only', action='store_true',
                   help='emit only the asset block and the plate/peg bodies — '
                        'no <option>, <default>, light, or floor. Useful when '
                        'the output will be <include>d into a larger scene.')
    p.add_argument('--export-meshes', action='store_true',
                   help='write each mesh as a .obj file next to the XML and '
                        'reference it with <mesh file="..."/>. Required for '
                        'USD and other external conversion pipelines — they '
                        "cannot read MuJoCo's inline vertex-only meshes. "
                        'Requires --output.')
    p.add_argument('--config', type=Path,
                   help='load full scene from YAML/JSON/TOML config file. '
                        'When given, most other flags are ignored.')
    p.add_argument('--scale', type=float, default=None,
                   help='uniform scale factor applied to every length '
                        '(plate/peg/clearance/handle dimensions and all shape '
                        'sizes + centers). With --config, overrides the '
                        'config\'s "scale" key if present.')
    p.add_argument('--output', '-o', type=Path,
                   help='write MJCF here (default: stdout)')
    args = p.parse_args(argv)

    if args.export_meshes and not args.output:
        p.error("--export-meshes requires --output (needs a directory for .obj files)")
    mesh_dir = args.output.parent.resolve() if args.export_meshes else None

    # Config-file path: bypass other flags entirely (except --scale, which
    # overrides the config's scale key when provided).
    if args.config:
        try:
            cfg = _load_config(args.config)
            xml = generate_from_config(cfg, config_dir=args.config.parent.resolve(),
                                       mesh_dir=mesh_dir, scale=args.scale)
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
    scale = args.scale if args.scale is not None else 1.0
    peg_poly = np.loadtxt(args.peg_csv, delimiter=',') * scale if args.peg_csv else None
    hole_poly = np.loadtxt(args.hole_csv, delimiter=',') * scale if args.hole_csv else None
    handle_poly = np.loadtxt(args.handle_csv, delimiter=',') * scale if args.handle_csv else None
    plate_poly = np.loadtxt(args.plate_csv, delimiter=',') * scale if args.plate_csv else None
    hole_island_polys = [np.loadtxt(pth, delimiter=',') * scale
                         for pth in args.hole_island_csvs]

    try:
        xml = generate_xml(
            shape=None if (peg_poly is not None or hole_poly is not None) else args.shape,
            hole_size=args.hole_size * scale,
            plate_size=args.plate_size * scale,
            plate_thickness=args.plate_thickness * scale,
            peg_clearance=args.peg_clearance * scale,
            peg_length=args.peg_length * scale,
            peg_height=args.peg_height * scale,
            peg_density=args.peg_density,
            peg_mass_kg=args.peg_mass_kg,
            plate_density=args.plate_density,
            handle_height=args.handle_height * scale,
            handle_margin=args.handle_margin * scale,
            plate_min_rim_m=args.plate_min_rim_m * scale,
            bottom_stop_thickness=args.bottom_stop_thickness * scale,
            center_shapes=not args.preserve_origin,
            n_circle=args.n_circle,
            include_peg=not args.no_peg,
            include_floor=not args.no_floor,
            mesh_only=args.mesh_only,
            free_plate=args.free_plate,
            mesh_dir=mesh_dir,
            peg_poly=peg_poly,
            hole_poly=hole_poly,
            handle_poly=handle_poly,
            plate_poly=plate_poly,
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
