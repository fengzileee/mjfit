#!/usr/bin/env python3
"""Generate MuJoCo MJCF for threaded bolt/nut pairs.

Applies the same convex-decomposition philosophy as mkpeg.py, but discretizes
angularly along the helix instead of radially across a 2D hole. Each short
angular slice of a thread is a twisted triangular prism — still convex. The
bolt = core cylinder + N thread wedges; the nut = annular-shell decomposition
+ N inward-facing thread wedges. All contact pairs stay convex-vs-convex, so
no V-HACD, CoACD, or SDF needed.

CLI:
    python mkscrew.py --major-diameter 0.010 --pitch 0.0015 \
        --thread-length 0.020 -o m10.xml
    python mkscrew.py --config examples/screws/m10.yaml -o m10.xml
    python mkscrew.py --config examples/screws/m6.yaml --export-meshes -o m6.xml

Library:
    from mkscrew import generate_screw_xml
    xml = generate_screw_xml(major_diameter=0.010, pitch=0.0015,
                             thread_length=0.020)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ----- config loader (matches mkpeg.py) ------------------------------------

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


# ----- 2D polygon helpers (subset of mkpeg.py, for hex-nut shell path) -----

def _f(x: float) -> str:
    """Geometry number for MJCF / OBJ: full precision (1e-6 m rounding would
    be a sizeable fraction of a 20 um thread clearance)."""
    return f'{x:.10g}'


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly[::-1].copy() if _signed_area(poly) < 0 else poly


def _is_convex_poly(poly: np.ndarray) -> bool:
    n = len(poly)
    signs = []
    for i in range(n):
        a, b, c = poly[i], poly[(i + 1) % n], poly[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-12:
            signs.append(1 if cross > 0 else -1)
    return len(set(signs)) <= 1


def _points_equal(a, b, tol: float = 1e-9) -> bool:
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def _cross2(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _segments_proper_intersect(p1, p2, p3, p4) -> bool:
    d1 = _cross2(p4[0] - p3[0], p4[1] - p3[1], p1[0] - p3[0], p1[1] - p3[1])
    d2 = _cross2(p4[0] - p3[0], p4[1] - p3[1], p2[0] - p3[0], p2[1] - p3[1])
    d3 = _cross2(p2[0] - p1[0], p2[1] - p1[1], p3[0] - p1[0], p3[1] - p1[1])
    d4 = _cross2(p2[0] - p1[0], p2[1] - p1[1], p4[0] - p1[0], p4[1] - p1[1])
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def _point_in_triangle(p, a, b, c, tol: float = 1e-12) -> bool:
    def s(q1, q2, q3):
        return (q1[0] - q3[0]) * (q2[1] - q3[1]) - (q2[0] - q3[0]) * (q1[1] - q3[1])
    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    has_neg = d1 < -tol or d2 < -tol or d3 < -tol
    has_pos = d1 > tol or d2 > tol or d3 > tol
    return not (has_neg and has_pos)


def _ear_clip_simple(poly: np.ndarray) -> list[np.ndarray]:
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
            if _cross2(b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1]) <= 1e-12:
                continue
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


def _find_bridge_against(p1, p2, *polygons) -> bool:
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


def _decompose_annular(outer_poly: np.ndarray, inner_poly: np.ndarray) -> list[np.ndarray]:
    """Triangulate a polygon with one hole via bridging + ear-clip."""
    inner = np.asarray(inner_poly)[::-1].copy()
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


# ----- shape builders ------------------------------------------------------

def _hex_polygon(across_flats: float) -> np.ndarray:
    """Regular hexagon CCW, with two flats parallel to the x-axis.
    `across_flats` is the flat-to-flat distance (standard hex-nut sizing)."""
    # circumradius = across_flats / cos(30°) = across_flats * 2/√3
    R = across_flats / np.sqrt(3)
    a = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    return np.column_stack([R * np.cos(a), R * np.sin(a)])


def _circle_polygon(radius: float, n: int = 32) -> np.ndarray:
    """CCW circle polygon."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(a), radius * np.sin(a)])


# ----- thread profile (ISO 60° metric) -------------------------------------

def _iso_thread_profile_external(pitch: float, clearance: float) -> np.ndarray:
    """Bolt thread tooth in the meridian (dr, z) plane.

    Returns a 4-vertex trapezoid (CCW in the (z, dr) convention used by the
    sweep generator — see `_thread_wedge_vertices`). `dr` is the radial offset
    outward from the minor radius (the bolt core surface). `z` is axial
    position within one pitch (0 ≤ z ≤ pitch).

    ISO 68-1 basic profile, external thread:
      - sharp-V height      H = P·√3/2
      - thread depth        h = 5H/8 = 0.5413·P
      - root flat (axial)   P/4   (at the minor radius)
      - flank span (axial)  5P/16 (60° included, 30° off the radial plane)
      - crest flat (axial)  P/8   (at the major radius)

    Clearance c moves each flank back into the tooth by c/2 along its normal
    (an axial shift of c/√3 at every radius) and lowers the crest by c/2. The
    nut tooth gets the same treatment, so when the nut sits on the pitch grid
    facing flanks are exactly c apart on both sides, and the crest clears the
    nut root by c. Axial play is then ±c/cos 30° = ±2c/√3.
    """
    P = pitch
    h = 5 * P * np.sqrt(3) / 16          # thread depth
    s = clearance / np.sqrt(3)           # axial shift of a flank moved c/2 along its normal
    dr_crest = h - clearance / 2
    crest_start = 9 * P / 16 + s / 2     # flank u = P/4 + s + dr/√3 at dr = h - c/2
    crest_end = 11 * P / 16 - s / 2
    if dr_crest <= 0 or crest_end <= crest_start:
        raise ValueError(
            f"clearance ({clearance}) leaves no bolt crest for pitch {P} "
            f"(needs clearance < √3·P/8 = {np.sqrt(3) * P / 8:.6g}).")
    # Order: two base vertices, then the two crest vertices (the sweep
    # generator's face topology relies on it).
    return np.array([
        [0.0, P / 4 + s],          # base-left
        [0.0, P - s],              # base-right
        [dr_crest, crest_end],     # crest-right
        [dr_crest, crest_start],   # crest-left
    ])


def _iso_thread_profile_internal(pitch: float, clearance: float) -> np.ndarray:
    """Nut thread tooth, dr measured from the major radius (negative = toward
    the axis), z in the nut's thread phase (see `nut_thread_offset`).

    The ISO internal tooth is the space between basic external teeth, not a
    mirror image of one: its tip (at the minor radius) is P/4 wide and its
    base (at the major radius) 7P/8, leaving a P/8 root. Clearance moves both
    flanks back c/2 along their normals and lifts the tip by c/2, as for the
    bolt.
    """
    P = pitch
    h = 5 * P * np.sqrt(3) / 16
    s = clearance / np.sqrt(3)
    dr_tip = -(h - clearance / 2)
    tip_start = 7 * P / 16 + s / 2       # flank u = P/8 + s - dr/√3 at dr = -(h - c/2)
    tip_end = 11 * P / 16 - s / 2
    if dr_tip >= 0 or tip_end <= tip_start:
        raise ValueError(
            f"clearance ({clearance}) leaves no nut tooth tip for pitch {P}.")
    return np.array([
        [0.0, P / 8 + s],          # base-left
        [0.0, P - s],              # base-right
        [dr_tip, tip_end],         # tip-right
        [dr_tip, tip_start],       # tip-left
    ])


# ----- thread forms -----------------------------------------------------------
#
# A thread form is a pair of convex teeth in the meridian half-plane, in
# coordinates (r, u): radius, and axial position u = z - (the helix's advance
# at that angle) taken mod P. The bolt's tooth stands on its core (radius
# r_core) and the nut's hangs from its root (radius r_root). Sweeping each
# tooth along the helix gives the part. Both are given at zero clearance in
# one shared phase, so with the two part frames a whole number of pitches
# apart the teeth fill each other's gaps. Clearance c then moves every exposed
# edge of both teeth back c/2 along its normal (the base on the core or root
# stays put), and the nut's root out by c/2. For ISO that is exactly the
# standard's rule; for any tooth it keeps facing surfaces c apart.
#
# Teeth need not be convex (hooks, T-slots, barbs): a non-convex tooth is
# offset as a whole (each edge back c/2, mitred corners, which at reflex
# corners only keeps more clear) and then cut into convex pieces, each swept
# as its own wedges. The cuts lie inside the tooth, so they are never moved.

THREAD_FORMS = ('iso', 'trapezoidal', 'acme', 'square', 'buttress', 'wire', 'custom')

# The trapezoid family. flank_angles: (lower, upper) flank of the bolt tooth,
# degrees from the radial plane; the upper flank faces the bolt head (+z).
# depth and crest_flat (the bolt crest) are in pitches.
TRAPEZOID_FORMS = {
    'iso':         {'flank_angles': (30.0, 30.0), 'depth': 5 * np.sqrt(3) / 16,
                    'crest_flat': 1 / 8},                                   # ISO 68-1
    'trapezoidal': {'flank_angles': (15.0, 15.0), 'depth': 0.5, 'crest_flat': 0.366},  # ISO 2904
    'acme':        {'flank_angles': (14.5, 14.5), 'depth': 0.5, 'crest_flat': 0.3707},
    'square':      {'flank_angles': (0.0, 0.0), 'depth': 0.5, 'crest_flat': 0.5},
    # ANSI B1.9 7°/45°: the steep flank carries the load of a nut pressed
    # toward the bolt tip (a bolt in tension).
    'buttress':    {'flank_angles': (45.0, 7.0), 'depth': 0.6, 'crest_flat': 0.1631},
}
# Length-valued thread_params (scaled with the part in configs).
THREAD_LENGTH_PARAMS = ('wire_diameter', 'nut_wire_offset', 'bore_diameter',
                        'core_diameter', 'root_diameter', 'bolt_tooth', 'nut_tooth')


class ThreadProfile:
    """Zero-clearance teeth of a thread form (see the section comment)."""

    def __init__(self, bolt_tooth, nut_tooth, r_core: float, r_root: float,
                 pitch: float):
        self.pitch = pitch
        self.r_core, self.r_root = r_core, r_root
        # Each tooth ends at its base line, so the base is one edge on it.
        self.bolt_tooth = _clip_radius(_ensure_tooth(bolt_tooth, 'bolt'), r_core, above=True)
        self.nut_tooth = _clip_radius(_ensure_tooth(nut_tooth, 'nut'), r_root, above=False)
        if self.bolt_tooth is None or self.nut_tooth is None:
            raise ValueError("a tooth does not reach past its base line "
                             "(bolt tooth above r_core, nut tooth below r_root)")

    @property
    def r_crest(self) -> float:
        """Bolt major radius (tip of the bolt tooth)."""
        return float(self.bolt_tooth[:, 0].max())

    def symmetric(self, tol: float = 1e-12) -> bool:
        """Is the bolt tooth mirror-symmetric in u about its centre?"""
        t = self.bolt_tooth
        m = t.copy()
        m[:, 1] = t[:, 1].min() + t[:, 1].max() - t[:, 1]
        a = t[np.lexsort((t[:, 1], t[:, 0]))]
        b = m[np.lexsort((m[:, 1], m[:, 0]))]
        return len(a) == len(b) and np.allclose(a, b, atol=tol * self.pitch)

    def teeth(self, clearance: float):
        """(bolt, nut) teeth at this clearance, as (r, u) arrays in the order
        base-left, base-right, then around (see _canonical). Raises when the
        clearance leaves no tooth."""
        out = []
        for name, tooth, line in (('bolt', self.bolt_tooth, self.r_core),
                                  ('nut', self.nut_tooth, self.r_root)):
            base = _base_edges(tooth, line)
            deltas = np.where(base, 0.0, clearance / 2)
            if _is_convex_poly(tooth):
                t = _offset_polygon(tooth, deltas)
            else:
                t = _offset_mitre(tooth, deltas)
            if t is None:
                raise ValueError(f"clearance ({clearance}) leaves no {name} tooth "
                                 f"for pitch {self.pitch}"
                                 + ("" if _is_convex_poly(tooth) else
                                    " (or is too large for its smallest feature)"))
            out.append(_canonical(t, line))
        return tuple(out)


def _ensure_tooth(tooth, name: str) -> np.ndarray:
    t = _ensure_ccw(np.asarray(tooth, dtype=float).reshape(-1, 2))
    if len(t) < 3 or not _is_simple_polygon(t):
        raise ValueError(f"the {name} tooth must be a simple polygon of (r, u) points")
    return t


def _is_simple_polygon(poly: np.ndarray) -> bool:
    """No two non-adjacent edges cross."""
    n = len(poly)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _segments_proper_intersect(poly[i], poly[(i + 1) % n], poly[j],
                                          poly[(j + 1) % n]):
                return False
    return True


def _offset_mitre(poly: np.ndarray, deltas) -> np.ndarray | None:
    """Move each edge i -> i+1 of a simple polygon (either winding) inward by
    deltas[i], each new vertex where its two edges' moved lines meet. Keeps
    the vertex order. Returns None when an edge would vanish or flip, or the
    result would cross itself (the offset is too large for the polygon)."""
    n = len(poly)
    deltas = np.broadcast_to(np.asarray(deltas, dtype=float), (n,))
    e = np.roll(poly, -1, axis=0) - poly
    normal = np.column_stack([e[:, 1], -e[:, 0]]) / np.linalg.norm(e, axis=1)[:, None]
    if _signed_area(poly) < 0:
        normal = -normal                                  # outward for either winding
    out = np.empty_like(poly)
    for i in range(n):
        j = (i - 1) % n                                   # edges j = i-1 and i meet at i
        # points x with n_j·x = n_j·p_j - d_j and n_i·x = n_i·p_i - d_i
        A = np.array([normal[j], normal[i]])
        b = np.array([normal[j] @ poly[j] - deltas[j], normal[i] @ poly[i] - deltas[i]])
        if abs(np.linalg.det(A)) < 1e-12:                 # collinear edges
            out[i] = poly[i] - deltas[i] * normal[i]
        else:
            out[i] = np.linalg.solve(A, b)
    e2 = np.roll(out, -1, axis=0) - out
    if (np.sum(e * e2, axis=1) <= 0).any() or not _is_simple_polygon(out) or \
            np.sign(_signed_area(out)) != np.sign(_signed_area(poly)):
        return None
    return out


def _convex_partition(poly: np.ndarray) -> list[list[int]]:
    """Convex pieces of a simple CCW polygon, as vertex index lists (CCW):
    ear-clip into triangles, then merge neighbours while the union stays
    convex (Hertel-Mehlhorn, at most four times the fewest pieces)."""
    n = len(poly)
    idx = list(range(n))
    pieces = []
    while len(idx) > 3:
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if _cross2(b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1]) <= 1e-18:
                continue                                  # reflex or flat: not an ear
            if any(_point_in_triangle(poly[j], a, b, c) for j in idx
                   if j not in (i0, i1, i2) and not any(_points_equal(poly[j], q, 1e-15)
                                                          for q in (a, b, c))):
                continue
            pieces.append([i0, i1, i2])
            idx.pop(k)
            break
        else:
            raise ValueError("could not cut the tooth into convex pieces "
                             "(is it a simple polygon?)")
    pieces.append(idx)

    def convex(ix):
        q = poly[ix]
        e = np.roll(q, -1, axis=0) - q
        return (e[:, 0] * np.roll(e[:, 1], -1) - e[:, 1] * np.roll(e[:, 0], -1) >= -1e-18).all()

    merged = True
    while merged:
        merged = False
        for a in range(len(pieces)):
            for b in range(a + 1, len(pieces)):
                A, B = pieces[a], pieces[b]
                for k in range(len(A)):
                    u, v = A[k], A[(k + 1) % len(A)]          # edge u -> v in A, v -> u in B
                    if v in B and B[(B.index(v) + 1) % len(B)] == u:
                        ra = A[k + 1:] + A[:k + 1]            # v ... u
                        rb = B[B.index(u):] + B[:B.index(u)]  # u ... v
                        union = ra + rb[1:-1]
                        if convex(union):
                            pieces[a] = union
                            pieces.pop(b)
                            merged = True
                        break
                if merged:
                    break
            if merged:
                break
    return pieces


def _clip_halfplane(poly: np.ndarray, n: np.ndarray, d: float) -> np.ndarray | None:
    """Convex polygon (CCW) intersected with n·x <= d."""
    out = []
    k = len(poly)
    for i in range(k):
        a, b = poly[i], poly[(i + 1) % k]
        fa, fb = n @ a - d, n @ b - d
        if fa <= 0:
            out.append(a)
        if fa * fb < 0:
            out.append(a + (b - a) * (fa / (fa - fb)))
    if len(out) < 3:
        return None
    out = np.array(out)
    keep = np.linalg.norm(out - np.roll(out, 1, axis=0), axis=1) > 1e-15
    out = out[keep]
    return out if len(out) >= 3 and _signed_area(out) > 1e-20 else None


def _offset_polygon(poly: np.ndarray, deltas) -> np.ndarray | None:
    """Move each edge i -> i+1 of a convex polygon (either winding) inward
    by deltas[i] along its normal. Returns a CCW polygon, or None when
    nothing is left."""
    n = len(poly)
    deltas = np.broadcast_to(np.asarray(deltas, dtype=float), (n,))
    if _signed_area(poly) < 0:
        # Reversed, edge k runs between the original vertices n-1-k and n-2-k.
        poly = poly[::-1]
        deltas = deltas[(n - 2 - np.arange(n)) % n]
    out = poly
    for i in range(len(poly)):
        if deltas[i] == 0:
            continue
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = b - a
        n = np.array([e[1], -e[0]]) / np.linalg.norm(e)      # outward for CCW
        out = _clip_halfplane(out, n, n @ a - deltas[i])
        if out is None:
            return None
    return out


def _edge_normals(poly: np.ndarray) -> np.ndarray:
    """Outward unit normals (n_r, n_u) of a convex polygon's edges i -> i+1,
    for either winding."""
    e = np.roll(poly, -1, axis=0) - poly
    n = np.column_stack([e[:, 1], -e[:, 0]]) / np.linalg.norm(e, axis=1)[:, None]
    return n if _signed_area(poly) > 0 else -n


def _base_edges(poly: np.ndarray, line: float) -> np.ndarray:
    """Edges i -> i+1 lying on the radius `line`."""
    tol = 1e-9 * max(abs(line), 1e-3)
    on = np.abs(poly[:, 0] - line) < tol
    return on & np.roll(on, -1)


def _clip_radius(poly: np.ndarray, line: float, above: bool) -> np.ndarray | None:
    """Keep r >= line (above) or r <= line."""
    n = np.array([-1.0, 0.0]) if above else np.array([1.0, 0.0])
    return _clip_halfplane(_ensure_ccw(poly), n, -line if above else line)


def _canonical(poly: np.ndarray, line: float) -> np.ndarray:
    """Order a tooth as base-left, base-right, then on around, the order the
    wedge writer's face table assumes (the ISO profiles' order)."""
    base = _base_edges(poly, line)
    if not base.any():
        return poly
    i = int(np.flatnonzero(base)[0])
    j = (i + 1) % len(poly)
    if poly[i, 1] > poly[j, 1]:          # walk the other way round
        poly = poly[::-1]
        i, j = len(poly) - 1 - j, len(poly) - 1 - i
    return np.roll(poly, -i, axis=0)


def _trapezoid_profile(major_radius: float, pitch: float, flank_angles=(30.0, 30.0),
                       depth: float = 5 * np.sqrt(3) / 16,
                       crest_flat: float = 1 / 8) -> ThreadProfile:
    """Trapezoid family (angles in degrees, depth and crest_flat in pitches).
    The bolt tooth starts after its root gap, so ISO lands on the ISO
    profiles' phase (bolt base from P/4 to P). Negative angles undercut the
    flank: a dovetail, whose teeth widen away from their base (both teeth
    stay convex)."""
    P, R = pitch, major_radius
    h, fc = depth * P, crest_flat * P
    t1, t2 = (np.tan(np.radians(a)) for a in flank_angles)
    fr = P - fc - h * (t1 + t2)                      # root gap of the bolt
    if h <= 0 or fc < 0 or fr < -1e-15 * P or P - fr <= 0 or P - fc <= 0 \
            or max(abs(a) for a in flank_angles) >= 90:
        raise ValueError(f"trapezoid thread does not fit one pitch: depth {depth}, "
                         f"crest_flat {crest_flat}, flank_angles {flank_angles} "
                         f"(root gap {fr / P:.4g} P)")
    fr = max(fr, 0.0)
    bolt = [(R - h, fr), (R - h, P), (R, fr + h * t1 + fc), (R, fr + h * t1)]
    # The nut tooth is the gap between bolt teeth: its tip is the bolt's root
    # gap, its base spans the bolt crests' gap.
    nut = [(R - h, 0.0), (R - h, fr), (R, fr + h * t1), (R, fr + h * t1 + fc - P)]
    if fr == 0:          # sharp bolt root: the nut tooth is a triangle
        nut = [nut[0], nut[2], nut[3]]
    if fc == 0:
        bolt = bolt[:3]
    return ThreadProfile(bolt, nut, R - h, R, P)


def _circle_tooth(r_c: float, u_c: float, radius: float, n: int) -> np.ndarray:
    """Inscribed n-gon of a circle in (r, u), with vertices at the circle's
    radial and axial extremes (n a multiple of 4)."""
    a = 2 * np.pi * np.arange(n) / n
    return np.column_stack([r_c + radius * np.cos(a), u_c + radius * np.sin(a)])


def _wire_profile(major_radius: float, pitch: float, wire_diameter: float | None = None,
                  nut_wire_offset: float | None = None, bore_diameter: float | None = None,
                  wire_segments: int = 16) -> ThreadProfile:
    """Round-wire thread: a wire of circular section wound on each part (the
    FurnitureBench legs). The bolt wire is centred on the core (radius
    major - wire/2); the nut wire half a pitch along, `nut_wire_offset`
    further out, on a bore of `bore_diameter` (default: through the nut wire's
    centre). The defaults are the tight form: wire_diameter P/√3 and the
    offset at which the two wires touch at zero clearance. The circles are
    inscribed polygons, so the teeth never exceed the round wire."""
    P = pitch
    d_w = P / np.sqrt(3) if wire_diameter is None else float(wire_diameter)
    rho = d_w / 2
    if nut_wire_offset is None:
        if d_w < P / 2:
            raise ValueError("wires thinner than P/2 never touch; give nut_wire_offset")
        nut_wire_offset = np.sqrt(d_w ** 2 - P ** 2 / 4)
    if wire_segments < 8 or wire_segments % 4:
        raise ValueError("wire_segments must be a multiple of 4, at least 8")
    r_b = major_radius - rho
    r_n = r_b + float(nut_wire_offset)
    r_root = r_n if bore_diameter is None else float(bore_diameter) / 2
    bolt = _circle_tooth(r_b, rho, rho, wire_segments)
    nut = _circle_tooth(r_n, rho + P / 2, rho, wire_segments)
    return ThreadProfile(bolt, nut, r_b, r_root, P)


def thread_profile(form: str, major_diameter: float, pitch: float,
                   params: dict | None = None) -> ThreadProfile:
    """The zero-clearance teeth of a named thread form.

    form: one of THREAD_FORMS. params override the form's defaults:
      trapezoid family (iso, trapezoidal, acme, square, buttress):
        flank_angles (deg, lower and upper), depth, crest_flat (pitches);
      wire: wire_diameter, nut_wire_offset, bore_diameter (m), wire_segments;
      custom: bolt_tooth and nut_tooth, lists of (r, u) points (m, one
        shared phase), core_diameter and root_diameter (m).
    major_diameter is the bolt's outer diameter (ignored by custom).
    """
    params = dict(params or {})
    if form in TRAPEZOID_FORMS:
        return _trapezoid_profile(major_diameter / 2, pitch,
                                  **{**TRAPEZOID_FORMS[form], **params})
    if form == 'wire':
        return _wire_profile(major_diameter / 2, pitch, **params)
    if form == 'custom':
        need = {'bolt_tooth', 'nut_tooth', 'core_diameter', 'root_diameter'}
        if need - params.keys():
            raise ValueError(f"custom thread form needs {sorted(need)}")
        return ThreadProfile(params['bolt_tooth'], params['nut_tooth'],
                             params['core_diameter'] / 2, params['root_diameter'] / 2, pitch)
    raise ValueError(f"thread_form must be one of {THREAD_FORMS}, got {form!r}")


def _thread_wedge_vertices(profile: np.ndarray, theta: float, dtheta: float,
                           z_advance: float, base_radius: float,
                           z_offset: float) -> np.ndarray:
    """Sweep a 2D meridian profile through (θ, θ+Δθ) with axial advance.

    profile: (N, 2) array of (dr, z_local) points. Radius at vertex = base_radius + dr.
    theta, dtheta: angular range (radians).
    z_advance: axial shift between the θ face and the θ+Δθ face (signed; sign
        = handedness × lead × dtheta / (2π)).
    base_radius: reference radius (minor for bolt, major for nut).
    z_offset: global z offset applied to z_local at θ.

    Returns (2N, 3) vertices, first N at θ, next N at θ+Δθ, same profile order.
    """
    verts = np.zeros((2 * len(profile), 3))
    for j, (ang, dz) in enumerate([(theta, 0.0), (theta + dtheta, z_advance)]):
        c, s = np.cos(ang), np.sin(ang)
        for i, (dr, z_local) in enumerate(profile):
            r = base_radius + dr
            verts[j * len(profile) + i] = [r * c, r * s, z_offset + z_local + dz]
    return verts


def _profile_ring(profile: np.ndarray, angle: float, base_radius: float,
                  z: float) -> np.ndarray:
    """The (dr, z_local) profile placed in the meridian plane at `angle`,
    with its z_local measured from `z`."""
    c, s = np.cos(angle), np.sin(angle)
    r = base_radius + profile[:, 0]
    return np.column_stack([r * c, r * s, z + profile[:, 1]])


def _wedge_has_volume(verts: np.ndarray, n_ring: int = 4, eps: float = 1e-15) -> bool:
    """Conservative check that a wedge (rings of n_ring profile vertices)
    spans a nonzero-volume hull: the tetrahedron of a first-ring triangle
    and the second ring's first vertex has volume. The triangle is the
    first-ring vertices 0, k, j that span the most area (a piece of a
    non-convex tooth can start with collinear vertices)."""
    ring = verts[:n_ring]
    v0 = ring[0]
    d = np.linalg.norm(ring - v0, axis=1)
    k = int(np.argmax(d))
    area = np.linalg.norm(np.cross(ring[k] - v0, ring - v0), axis=1)
    j = int(np.argmax(area))
    m = np.column_stack([ring[k] - v0, ring[j] - v0, verts[n_ring] - v0])
    return abs(np.linalg.det(m)) > eps


def _clip_z(verts: np.ndarray, z0: float, z1: float,
            min_volume: float = 1e-13) -> np.ndarray | None:
    """Vertices of the convex hull of `verts` clipped to z0 <= z <= z1.

    The clipped hull is the hull of the vertices inside the slab plus every
    crossing of the planes by a segment between two vertices (all such
    segments lie in the hull, and its clipped edges are among them). Returns
    `verts` unchanged when already inside, and None when what remains is a
    nub too small for MuJoCo to hull reliably (its volume, estimated from the
    principal extents, under `min_volume`); dropping one only removes a sliver
    of tooth at the part's end face.
    """
    z = verts[:, 2]
    if z.min() >= z0 and z.max() <= z1:
        return verts
    if z.max() <= z0 or z.min() >= z1:
        return None
    pts = [verts[(z >= z0) & (z <= z1)]]
    for a in range(len(verts)):
        for b in range(a + 1, len(verts)):
            for zc in (z0, z1):
                if (z[a] - zc) * (z[b] - zc) < 0:
                    t = (zc - z[a]) / (z[b] - z[a])
                    pts.append((verts[a] + t * (verts[b] - verts[a]))[None])
    pts = np.unique(np.concatenate(pts), axis=0)
    if len(pts) < 4:
        return None
    centered = pts - pts.mean(axis=0)
    extents = np.ptp(centered @ np.linalg.svd(centered, full_matrices=False)[2].T, axis=0)
    return pts if np.prod(extents) / 6 > min_volume else None


def _clip_half(verts: np.ndarray, n: np.ndarray, d: float,
               min_volume: float = 1e-13) -> np.ndarray | None:
    """Vertices of the convex hull of `verts` clipped to n·x <= d: the hull
    of the vertices inside plus every crossing of the plane by a segment
    between two vertices. None when nothing, or a nub too small to hull, is
    left (see _clip_z)."""
    side = verts @ n - d
    if (side <= 0).all():
        return verts
    if (side >= 0).all():
        return None
    i, j = np.nonzero((side[:, None] < 0) & (side[None, :] > 0))
    t = side[i] / (side[i] - side[j])
    pts = np.unique(np.concatenate([verts[side <= 0],
                                    verts[i] + t[:, None] * (verts[j] - verts[i])]), axis=0)
    if len(pts) < 4:
        return None
    centered = pts - pts.mean(axis=0)
    extents = np.ptp(centered @ np.linalg.svd(centered, full_matrices=False)[2].T, axis=0)
    return pts if np.prod(extents) / 6 > min_volume else None


def _cone_chord(r0: float, slope: float, z0: float, towards: int,
                theta0: float, theta1: float):
    """(n, d): a half-space n·x <= d whose part in the sector [theta0,
    theta1] lies inside the cone r <= r0 + slope·towards·(z - z0) (a bolt
    chamfer at the end z0, the part on the side `towards`). Its plane holds
    the cone's generators at theta0 and theta1, which lie on the cone: the
    chord of a convex surface lies inside it."""
    a = slope * towards
    n = np.array([a * (np.sin(theta0) - np.sin(theta1)), a * (np.cos(theta1) - np.cos(theta0)),
                  a * a * np.sin(theta1 - theta0)])
    apex = np.array([0.0, 0.0, z0 - r0 / a])
    if n @ (np.array([0.0, 0.0, z0]) - apex) > 0:     # keep the axis side
        n = -n
    return n, float(n @ apex)


def _cone_tangent(r0: float, slope: float, z0: float, towards: int, theta: float):
    """(n, d): the half-space n·x <= d outside the tangent plane at angle
    theta of the cone r = r0 - slope·towards·(z - z0) (a countersink at the
    face z0, narrowing towards the part). A cone is convex, so everything
    beyond a tangent plane lies outside it, whatever the angle."""
    n = -np.array([np.cos(theta), np.sin(theta), slope * towards])
    return n, float(-r0 - slope * towards * z0)


# ----- nut shell decomposition ---------------------------------------------

def _decompose_nut_shell_cylinder(
    r_inner: float,
    r_outer: float,
    n_sectors: int,
) -> list[np.ndarray]:
    """Return a list of 2D (x, y) trapezoidal polygons tiling the annular
    cylinder (r_inner, r_outer). Inner arc is chord-approximated so the
    *midpoint* of each chord sits at r_inner — i.e. we inflate the polygon
    inner radius so the flat chord never crosses the nominal bore.

    n_sectors trapezoidal prisms; each is a 4-vertex convex polygon.
    """
    if n_sectors < 3:
        raise ValueError("n_sectors must be >= 3 for a cylinder shell.")
    # Inflate so chord midpoint sits at r_inner.
    # chord midpoint distance = r_inflated * cos(π/n_sectors). Set = r_inner.
    r_inner_inflated = r_inner / np.cos(np.pi / n_sectors)
    # Outer: chord endpoints touch nominal outer; the midpoint then sits
    # slightly inside. Acceptable (doesn't affect contact with the bolt).
    sectors = []
    for k in range(n_sectors):
        a0 = 2 * np.pi * k / n_sectors
        a1 = 2 * np.pi * (k + 1) / n_sectors
        # CCW: outer(a0) -> outer(a1) -> inner(a1) -> inner(a0)
        poly = np.array([
            [r_outer * np.cos(a0), r_outer * np.sin(a0)],
            [r_outer * np.cos(a1), r_outer * np.sin(a1)],
            [r_inner_inflated * np.cos(a1), r_inner_inflated * np.sin(a1)],
            [r_inner_inflated * np.cos(a0), r_inner_inflated * np.sin(a0)],
        ])
        sectors.append(poly)
    return sectors


def _decompose_nut_shell_hex(
    r_inner: float,
    across_flats: float,
    n_bore: int = 32,
) -> list[np.ndarray]:
    """Hex-outline nut: (hexagon) minus (cylindrical bore). Decompose via
    bridge + ear-clip (handles any simple polygon-with-hole). Returns triangle
    soup — many small convex pieces — which MuJoCo handles cheaply."""
    hex_outer = _hex_polygon(across_flats)
    # Circumscribe the bore polygon so its chords never cut inside r_inner.
    bore = _circle_polygon(r_inner / np.cos(np.pi / n_bore), n_bore)
    # Make sure bore fits well inside hexagon inscribed radius (= across_flats/2).
    if r_inner >= across_flats / 2 * 0.98:
        raise ValueError(
            f"bore radius {r_inner} too large for hex across_flats {across_flats} "
            f"(min flat-to-center distance = {across_flats/2}).")
    return _decompose_annular(hex_outer, bore)


# ----- OBJ / mesh-asset emission -------------------------------------------

def _write_prism_obj(
    path: Path,
    solids: list[tuple[np.ndarray, list, float, float]],
) -> None:
    """Watertight OBJ for one-or-more extruded 2D polygons with optional holes.
    Exactly the same format as mkpeg.py's `_write_prism_obj` — used for the
    nut shell and bolt head/shank meshes."""
    lines = ["# Generated by mkscrew.py"]
    face_lines: list[str] = []
    n_verts = 0

    def add_vert(x: float, y: float, z: float) -> int:
        nonlocal n_verts
        lines.append(f"v {_f(x)} {_f(y)} {_f(z)}")
        n_verts += 1
        return n_verts

    for outer, holes, thickness, z_offset in solids:
        half = thickness / 2
        z_bot = z_offset - half
        z_top = z_offset + half

        if holes:
            tris = _decompose_annular(outer, holes[0])
        elif _is_convex_poly(outer):
            tris = _ear_clip_simple(outer)
        else:
            tris = _ear_clip_simple(outer)
        for tri in tris:
            idx = [add_vert(p[0], p[1], z_top) for p in tri]
            face_lines.append(f"f {idx[0]} {idx[1]} {idx[2]}")
            idx = [add_vert(p[0], p[1], z_bot) for p in tri]
            face_lines.append(f"f {idx[0]} {idx[2]} {idx[1]}")

        n = len(outer)
        for i in range(n):
            v0, v1 = outer[i], outer[(i + 1) % n]
            b0 = add_vert(v0[0], v0[1], z_bot)
            b1 = add_vert(v1[0], v1[1], z_bot)
            t1 = add_vert(v1[0], v1[1], z_top)
            t0 = add_vert(v0[0], v0[1], z_top)
            face_lines.append(f"f {b0} {b1} {t1}")
            face_lines.append(f"f {b0} {t1} {t0}")

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


# Face topology for a trapezoid-swept thread wedge (8 verts).
# Profile order from _iso_thread_profile_*: [base_left, base_right,
# crest_right, crest_left]. First 4 at θ, next 4 at θ+Δθ. Indices 0..7.
#
#     4 (base_left,  θ+Δθ)         5 (base_right, θ+Δθ)
#     0 (base_left,  θ)            1 (base_right, θ)
#     7 (crest_left, θ+Δθ)         6 (crest_right, θ+Δθ)
#     3 (crest_left, θ)            2 (crest_right, θ)
#
# Each face is a triangulated quad; orientation chosen for outward normals
# on a positive-dr (bolt) tooth. For internal (nut) profile the dr's are
# negated so normals naturally flip — we patch that by reversing the
# triangle winding via `flip` flag below.
_WEDGE_FACES_QUAD = [
    # end cap at θ (vertices 0,1,2,3 — trapezoid in meridian plane)
    (0, 1, 2, 3),
    # end cap at θ+Δθ (4,5,6,7 — reversed to face the other way)
    (7, 6, 5, 4),
    # base barrel (0,1,5,4): at core cylinder, outward normal = into the core
    # (interior of wedge is above, i.e. +dr). This face is internal when core
    # cylinder is also present, so orientation doesn't matter for contact.
    (4, 5, 1, 0),
    # crest barrel (3,2,6,7): outward = +dr (bolt apex)
    (3, 2, 6, 7),
    # rising flank (1,2,6,5): connects base-right to crest-right
    (1, 2, 6, 5),
    # falling flank (0,4,7,3): connects base-left to crest-left
    (0, 4, 7, 3),
]


def _write_wedge_obj(path: Path, wedges: list[np.ndarray], flip: bool = False,
                     faces: bool = True) -> None:
    """Write a batch of 8-vertex thread wedges as a single OBJ. Face topology
    is the trapezoid-swept prism connectivity above. `flip` reverses triangle
    winding (used for nut threads, which have -dr profile)."""
    lines = ["# Generated by mkscrew.py"]
    face_lines: list[str] = []
    n_verts = 0
    if not faces or any(len(v) != 8 for v in wedges):
        # A wedge clipped at a thread end: vertices only, MuJoCo hulls them.
        if len(wedges) != 1:
            raise ValueError("clipped wedges must be written one per OBJ")
        lines += [f"v {_f(x)} {_f(y)} {_f(z)}" for x, y, z in wedges[0]]
        path.write_text("\n".join(lines) + "\n")
        return

    for verts in wedges:
        base_idx = n_verts + 1
        for (x, y, z) in verts:
            lines.append(f"v {_f(x)} {_f(y)} {_f(z)}")
            n_verts += 1
        for quad in _WEDGE_FACES_QUAD:
            a, b, c, d = (base_idx + i for i in quad)
            if flip:
                face_lines.append(f"f {a} {c} {b}")
                face_lines.append(f"f {a} {d} {c}")
            else:
                face_lines.append(f"f {a} {b} {c}")
                face_lines.append(f"f {a} {c} {d}")

    lines.extend(face_lines)
    path.write_text("\n".join(lines) + "\n")


def _extrude_mesh_xml(name: str, poly2d: np.ndarray, thickness: float,
                      z_offset: float = 0.0,
                      mesh_dir: Path | None = None) -> str:
    """Return MJCF `<mesh>` for an extruded prism. Same structure as mkpeg.py
    but supports a z_offset (for nut shell pieces, whose z-center is not 0)."""
    if mesh_dir is not None:
        _write_prism_obj(mesh_dir / f'{name}.obj',
                         [(poly2d, [], thickness, z_offset)])
        return f'    <mesh name="{name}" file="{name}.obj"/>'
    verts = []
    for v in poly2d:
        verts.extend([v[0], v[1], z_offset - thickness / 2])
    for v in poly2d:
        verts.extend([v[0], v[1], z_offset + thickness / 2])
    vstr = ' '.join(_f(x) for x in verts)
    return f'    <mesh name="{name}" vertex="{vstr}"/>'


def _wedge_mesh_xml(name: str, verts: np.ndarray,
                    mesh_dir: Path | None = None,
                    flip: bool = False, faces: bool = True) -> str:
    """Return MJCF `<mesh>` for a single 8-vertex thread wedge.

    When mesh_dir is None: inline `vertex="..."` form — MuJoCo convex-hulls.
    When mesh_dir is set: write a small OBJ with explicit face topology.
    """
    if mesh_dir is not None:
        _write_wedge_obj(mesh_dir / f'{name}.obj', [verts], flip=flip, faces=faces)
        return f'    <mesh name="{name}" file="{name}.obj"/>'
    flat = verts.flatten()
    vstr = ' '.join(_f(x) for x in flat)
    return f'    <mesh name="{name}" vertex="{vstr}"/>'


# ----- top-level MJCF assembly ---------------------------------------------

def generate_screw_xml(
    major_diameter: float = 0.010,
    pitch: float = 0.0015,
    thread_length: float = 0.020,
    n_starts: int = 1,
    segments_per_turn: int = 24,
    thread_form: str = 'iso',
    thread_params: dict | None = None,
    handedness: str = 'right',
    clearance: float = 0.00005,
    seam_relief: float = 0.0,
    thread_chamfer: float = 0.0,
    # bolt
    include_bolt: bool = True,
    bolt_shank_length: float = 0.0,
    bolt_head_height: float = 0.005,
    bolt_head_diameter: float = 0.015,
    bolt_head_across_flats: float = 0.013,
    bolt_head_shape: str = 'cylinder',
    bolt_density: float = 7800.0,
    # nut
    include_nut: bool = True,
    nut_height: float = 0.008,
    nut_outer_diameter: float = 0.016,
    nut_across_flats: float = 0.016,
    nut_shell_shape: str = 'cylinder',
    nut_shell_sectors: int = 16,
    nut_density: float = 7800.0,
    # scene
    include_floor: bool = True,
    free_bolt: bool = True,
    free_nut: bool = False,
    nut_driven: bool = False,
    mesh_only: bool = False,
    mesh_dir: Path | None = None,
    # contact tuning
    friction: tuple[float, float, float] = (1.0, 0.01, 0.001),
    condim: int = 4,
    timestep: float = 0.001,
) -> str:
    """nut_driven=True sets up the reverse scenario: bolt welded with head
    at the floor and tip pointing up, nut free and positioned at the top
    of the bolt ready to thread down under rotation. Overrides `free_bolt`
    and `free_nut` (bolt welded, nut free).

    thread_form picks the tooth shape (see THREAD_FORMS and thread_profile):
    the ISO 60° metric thread, trapezoidal / ACME / square / buttress power
    threads, a round-wire thread, or custom teeth; thread_params overrides the
    form's defaults. major_diameter is the bolt's outer diameter for all of
    them, and clearance is the gap between facing surfaces of the two teeth.

    friction is the MuJoCo geom friction triple (tangential, torsional,
    rolling). Drop tangential toward 0.1–0.2 to make threading less sticky.
    condim is the contact dimension: 4 includes torsional friction (resists
    rotation), 3 drops it for freer-spinning threads. 6 is rarely useful.

    seam_relief (m, default 0 = off) cuts a shallow V-groove into the flanks
    and crest at every seam between neighbouring wedges: each wedge's teeth
    dip by this depth at both ends and ramp back up (slope about 1:20). MuJoCo
    collides every pair of convex pieces separately, so a tooth sliding
    across a seam first overlaps the next wedge in a thin sliver at its end
    face. The shortest way out of that sliver is back through the end face,
    which gives a contact normal around the axis that stops a spinning nut
    dead (an "internal edge"). Relief deeper than the contact penetration
    under load (about 1 µm at MuJoCo's default contact softness) turns each
    seam into a gentle ramp instead. It only removes material.

    thread_chamfer (m, default 0 = off) chamfers the thread ends the way a
    machined bolt's are: a cone cuts the bolt's thread back from its core
    radius at the end face to its crest `thread_chamfer` in (45° when that is
    the thread depth), at the tip, and at the far end too when the bolt is a
    bare threaded rod (no shank or head). The nut is countersunk the same way
    at both faces, from its bore to its tooth tips. It gives a lead-in, and
    it removes the thin blade an undercut tooth (a dovetail) leaves where a
    thread is cut off flat. Each wedge is cut by a plane that stays inside
    the cone, so this too only removes material."""
    if handedness not in ('right', 'left'):
        raise ValueError(f"handedness must be 'right' or 'left', got {handedness!r}")
    if n_starts < 1:
        raise ValueError("n_starts must be >= 1")
    if segments_per_turn < 6:
        raise ValueError("segments_per_turn must be >= 6")

    D = major_diameter
    P = pitch
    lead = n_starts * P
    r_major = D / 2
    form = thread_profile(thread_form, D, P, thread_params)
    r_minor = form.r_core            # bolt core
    r_root = form.r_root             # nut root at zero clearance

    if clearance < 0:
        raise ValueError("clearance must be >= 0")
    if seam_relief < 0:
        raise ValueError("seam_relief must be >= 0")
    if thread_chamfer < 0:
        raise ValueError("thread_chamfer must be >= 0")
    # form.teeth rejects a clearance that leaves no tooth (for ISO,
    # clearance >= √3·P/8, about 0.22·P).

    sign = 1 if handedness == 'right' else -1
    dtheta = 2 * np.pi / segments_per_turn

    # Bolt threads span [0, thread_length] in the bolt local frame.
    # Nut threads span only [0, nut_height] (nut's bore is shorter than the
    # bolt shaft in general). Interleave the two with a half-pitch axial
    # offset so teeth of one fit into the root flats of the other.
    # Teeth as (r, u) in the bolt's phase, both parts' frames on the pitch
    # grid (the nut tooth already sits in the bolt's gaps).
    bolt_profile, nut_profile = form.teeth(clearance)
    # Nut-driven scenes flip the bolt (below), which mirrors its tooth in u.
    # A symmetric tooth only moves the nut's phase; an asymmetric one needs
    # the nut's tooth mirrored too, as a nut turned over with its bolt.
    mirror_nut = nut_driven and not form.symmetric()
    if mirror_nut:
        nut_profile = _canonical(nut_profile * [1, -1], r_root)
    # Seam relief: the same teeth with flanks and crest moved in by
    # `seam_relief` (i.e. clearance + 2 * seam_relief), used at wedge ends.
    bolt_relief = nut_relief = None
    relief_angle = 0.0
    if seam_relief > 0:
        bolt_relief, nut_relief = form.teeth(clearance + 2 * seam_relief)
        if mirror_nut:
            nut_relief = _canonical(nut_relief * [1, -1], r_root)
        # Ramp about 20x as long as it is deep, measured at mid-flank; at most
        # a third of the wedge at each end so every wedge keeps full teeth.
        relief_angle = min(20 * seam_relief / ((r_minor + form.r_crest) / 2),
                           2 * np.pi / segments_per_turn / 3)

    def _gen_wedges(profile, base_radius, axial_extent, internal=False, relief=None,
                    pieces=None):
        """Generate wedge vertex arrays for all starts × turns × segments.

        The helix covers [0, axial_extent] axially: the turns run from the
        lowest one that still rises above z = 0 (the partial first turn at
        the z = 0 end face) to past the other end. The profile's u is its phase:
        the nut's is already in the bolt's, so the teeth interlock.

        Multi-start threads are interleaved AXIALLY. Each start k has its
        first tooth at z = k·P; all N helices share the same angular range
        [0, 2π·n_turns] — that produces the "tooth every P" axial density
        regardless of n_starts.

        Each wedge is the hull of the profile at θ and θ+Δθ, so its faces
        are chords of the helical surfaces. On an external thread they fall
        inside the true tooth (or inside the core). On an internal thread
        they bulge toward the bolt two ways, and `internal` removes both so
        that discretization only ever adds play:
          - the tip and flank arcs sag by r(1 - cos(Δθ/2)) (36 µm at M10 with
            24 segments): the wedge is pushed out radially by 1/cos(Δθ/2);
          - each flank is a twisted quad of the helicoid, and its hull
            diagonal sits off the helicoid axially by
            lead·Δθ·(r1 - r2) / (4π(r1 + r2)) (2.6 µm at M10 with 24
            segments): both flanks are moved back into the tooth by that.
        Wedges are clipped to [0, axial_extent] so no tooth sticks out past
        the part's end faces.

        With `relief` (the tooth shrunk by the seam relief) each wedge is the
        hull of four copies instead of two: `relief` at θ and θ+Δθ, the full
        tooth `relief_angle` in from each end (see seam_relief).

        `pieces`: for a non-convex tooth (then CCW), the vertex index lists of
        its convex pieces. The corrections move the tooth's own edges (mitred,
        so reflex corners come out right), and each piece is swept as its own
        wedges; the cuts between pieces lie inside the tooth.
        """
        n_turns = int(np.ceil(axial_extent / lead)) + 2
        # A turn's tooth rises lead per turn from u in [u_min, u_max], so the
        # turn before turn 0 still pokes above z = 0 over most of the first
        # turn. It is clipped at the face like the rest.
        first_turn = -int(np.ceil(max(profile[:, 1].max(), 0.0) / lead)) - 1
        wedges = []
        r = base_radius + profile[:, 0]
        normals = _edge_normals(profile)
        base = _base_edges(profile, r_root if internal else r_minor)   # never moved
        # Moving a flank back along u by t moves edge i in by |n_u| t.
        n_u = np.where(base, 0.0, np.abs(normals[:, 1]))
        # The internal-thread corrections must cover every chord of the hull.
        # Without relief that is the one chord across the wedge. With relief
        # it is the larger of the full-height middle chord (span
        # dtheta - 2 * relief_angle) and the chord between the two relieved
        # ends (across the whole wedge, but starting `seam_relief` further
        # back, which is an axial setback of seam_relief / |n_u| on a flank).
        span = dtheta if relief is None else dtheta - 2 * relief_angle
        radial_scale = 1 / np.cos(span / 2)
        offsets = np.zeros(len(profile))
        if internal:
            twist = lead * span * (r.max() - r.min()) / (4 * np.pi * (r.max() + r.min()))
            if relief is not None:
                r_tip = r.min()
                radial_scale = max(radial_scale,
                                   r_tip / ((r_tip + seam_relief) * np.cos(dtheta / 2)))
                full = lead * dtheta * (r.max() - r.min()) / (4 * np.pi * (r.max() + r.min()))
                setback = seam_relief / max(n_u.max(), 0.1)
                twist = max(twist, full - setback)
            offsets = n_u * twist
        else:
            # External thread. A flank quad's hull diagonal sits off the
            # helicoid axially by lead·Δθ·(r1 - r2) / (4π(r1 + r2)), but the
            # chord also sags inward by r(1 - cos(Δθ/2)), which on a flank
            # that widens toward the core (ISO, trapezoidal) more than makes
            # up for it. On steep flanks (square, the sides of a round wire)
            # nothing makes up for it, so those edges are moved back.
            r1, r2 = r, np.roll(r, -1)
            bulge = lead * dtheta * np.abs(r1 - r2) / (4 * np.pi * (r1 + r2))
            sag = np.minimum(r1, r2) * (1 - np.cos(dtheta / 2))
            offsets = np.where(base, 0.0, np.maximum(0.0, n_u * bulge - normals[:, 0] * sag))
            if relief is not None:
                # With relief, the short ramp quads between a relieved and a
                # full copy can bulge out by their own twist. Those quads join
                # two different profiles, so the first-order twist bound is
                # doubled (measured bulge stays below it).
                twist = 2 * lead * relief_angle * (r.max() - r.min()) / (4 * np.pi * (r.max() + r.min()))
                offsets = np.maximum(offsets, n_u * twist)
        line = r_root if internal else r_minor
        full, full_relief = profile, relief

        def shrink(offsets):
            out = []
            for prof in (full, full_relief):
                if prof is not None and offsets.any():
                    # relief shares the profile's edge directions, so the
                    # same per-edge offsets apply edge by edge.
                    if pieces is not None:       # keep the vertex order the pieces index
                        prof = _offset_mitre(prof, offsets)
                    else:
                        prof = _offset_polygon(prof, offsets if len(prof) == len(full)
                                               else offsets.max())
                    if prof is None:
                        raise ValueError("segments_per_turn too low for this thread: "
                                         "the flank corrections leave no tooth")
                    if pieces is None:
                        prof = _canonical(prof, line)
                out.append(prof)
            return out

        n_full = len(full)
        parts = pieces if pieces is not None else [list(range(n_full))]

        def excess(prof, rel):
            """How far each edge of the full tooth is overshot by the hull of
            one wedge per piece (all wedges are congruent), measured in each
            hull point's own meridian plane. Samples every facet the hull can
            have between two rings: both triangle pairs of each side quad
            (either diagonal can split it), interiors included, since the
            curved flank can bow out past a facet's middle (2 nm at 12
            segments) as well as along its edges."""
            req = np.zeros(n_full)
            if len(prof) != n_full or (rel is not None and len(rel) != n_full):
                return req
            ii, jj = np.meshgrid(np.arange(33), np.arange(33), indexing='ij')
            on = ii + jj <= 32
            w = np.column_stack([ii[on], jj[on], 32 - ii[on] - jj[on]]) / 32

            def overshoot(tri, k):
                """Largest overshoot of the full tooth's edge k over a
                triangle of hull points: a barycentric grid, then zooming in
                on its peak (the overshoot is curved; a grid alone misses the
                peak by ~10 nm)."""
                tri, peak, at, ww = np.array(tri), -np.inf, None, w
                for level in range(4):
                    x = ww @ tri
                    u = x[:, 2] - sign * lead * np.arctan2(x[:, 1], x[:, 0]) / (2 * np.pi)
                    rr = np.hypot(x[:, 0], x[:, 1]) - base_radius
                    dev = (np.column_stack([rr, u]) - full[k]) @ normals[k]
                    i = int(np.argmax(dev))
                    if dev[i] > peak:
                        peak, at = dev[i], ww[i]
                    ww = np.clip(at + (w - 1 / 3) / 8 ** (level + 1), 0.0, None)
                    ww /= ww.sum(axis=1, keepdims=True)
                return peak
            for ix in parts:
                rings = [(prof[ix], 0.0), (prof[ix], dtheta)] if rel is None else \
                    [(rel[ix], 0.0), (prof[ix], relief_angle),
                     (prof[ix], dtheta - relief_angle), (rel[ix], dtheta)]
                pts = []
                for pr, a in rings:
                    v = _profile_ring(pr, a, base_radius, sign * a * lead / (2 * np.pi))
                    if internal:
                        v[:, :2] *= radial_scale
                    pts.append(v)
                m = len(ix)
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        a, b = pts[i], pts[j]
                        for e in range(m):
                            k = ix[e]
                            if (ix[(e + 1) % m] - k) % n_full != 1 or base[k]:
                                continue            # a cut between pieces, or the base
                            e1 = (e + 1) % m
                            for tri in ((a[e], a[e1], b[e1]), (a[e], b[e1], b[e]),
                                        (a[e], a[e1], b[e]), (a[e1], b[e1], b[e])):
                                req[k] = max(req[k], overshoot(tri, k))
            return req

        # The formulas above are first order. Top them up from the geometry:
        # measure one wedge against the full tooth and move back any edge
        # that still sticks out (a few nm at most), until none does.
        for _ in range(5):
            profile, relief = shrink(offsets)
            over = excess(profile, relief)
            if over.max() <= 1e-13:
                break
            offsets = offsets + 1.1 * over
        z_adv = sign * dtheta * lead / (2 * np.pi)
        for ix in parts:
            prof_p = profile[ix]
            rel_p = None if relief is None else relief[ix]
            if pieces is not None and not (_is_convex_poly(prof_p) and
                                           (rel_p is None or _is_convex_poly(rel_p))):
                raise ValueError("clearance or seam_relief too large for this tooth's "
                                 "convex pieces")
            for start_idx in range(n_starts):
                for turn in range(first_turn, n_turns):
                    z_turn_base = start_idx * P + turn * lead
                    for seg in range(segments_per_turn):
                        theta_i = seg * dtheta
                        z_at_theta = z_turn_base + sign * theta_i * lead / (2 * np.pi)
                        if rel_p is None:
                            v = _thread_wedge_vertices(
                                prof_p, theta_i, dtheta, z_adv,
                                base_radius, z_at_theta)
                        else:
                            rings = [(rel_p, 0.0), (prof_p, relief_angle),
                                     (prof_p, dtheta - relief_angle), (rel_p, dtheta)]
                            v = np.concatenate([
                                _profile_ring(prof, theta_i + a, base_radius,
                                              z_at_theta + sign * a * lead / (2 * np.pi))
                                for prof, a in rings])
                        if internal:
                            v[:, :2] *= radial_scale
                        if _wedge_has_volume(v, len(prof_p)):
                            v = _clip_z(v, 0.0, axial_extent)
                            if v is not None and thread_chamfer > 0:
                                v = _chamfer(v, internal, theta_i, theta_i + dtheta)
                            if v is not None:
                                wedges.append(v)
        return wedges

    # The teeth carry absolute radii (base_radius 0). The nut tooth's u is in
    # the bolt's phase; normalize it to end within the first pitch so its
    # first turn starts at or below the nut's base (the wedges are clipped to
    # the part), as the bolt's does.
    def _first_pitch(prof, rel, shift):
        return prof + [0, shift], (None if rel is None else rel + [0, shift])

    bolt_profile, bolt_relief = _first_pitch(
        bolt_profile, bolt_relief, -P * np.floor(bolt_profile[:, 1].min() / P))
    nut_profile, nut_relief = _first_pitch(
        nut_profile, nut_relief, -P * np.ceil(nut_profile[:, 1].max() / P - 1))

    # Chamfers: (radius at the end face, slope, face z, direction into the
    # part) for each chamfered end of each part.
    bolt_ends = nut_faces = ()
    if thread_chamfer > 0:
        r_crest = bolt_profile[:, 0].max()
        r_tip, r_bore = nut_profile[:, 0].min(), r_root + clearance / 2
        k_bolt = (r_crest - r_minor) / thread_chamfer
        k_nut = (r_bore - r_tip) / thread_chamfer
        bare = bolt_shank_length == 0 and bolt_head_height == 0
        bolt_ends = ((0.0, 1),) + (((thread_length, -1),) if bare else ())
        nut_faces = ((0.0, 1), (nut_height, -1))

    def _chamfer(v, internal, theta0, theta1):
        """Cut a wedge by the chamfer cones near its part's ends."""
        z = v[:, 2]
        for z0, towards in (nut_faces if internal else bolt_ends):
            if towards * (z.min() - z0) > 1.05 * thread_chamfer and \
                    towards * (z.max() - z0) > 1.05 * thread_chamfer:
                continue                        # wholly past this chamfer
            if internal:
                n, d = _cone_tangent(r_bore, k_nut, z0, towards, (theta0 + theta1) / 2)
            else:
                n, d = _cone_chord(r_minor, k_bolt, z0, towards, theta0, theta1)
            v = _clip_half(v, n, d)
            if v is None:
                return None
            z = v[:, 2]
        return v

    def _split(tooth, rel):
        """A non-convex tooth made CCW (with its relief), and its convex
        pieces; a convex tooth as it is, with no pieces."""
        if _is_convex_poly(tooth):
            return tooth, rel, None
        if _signed_area(tooth) < 0:
            tooth, rel = tooth[::-1], (None if rel is None else rel[::-1])
        return tooth, rel, _convex_partition(tooth)

    bolt_wedges = []
    if include_bolt:
        tooth, rel, pieces = _split(bolt_profile, bolt_relief)
        bolt_wedges = _gen_wedges(tooth, 0.0, thread_length, relief=rel, pieces=pieces)
    nut_wedges = []
    if include_nut:
        tooth, rel, pieces = _split(nut_profile, nut_relief)
        nut_wedges = _gen_wedges(tooth, 0.0, nut_height, internal=True, relief=rel,
                                 pieces=pieces)
    # The OBJ writer's face table needs a plain four-sided tooth in canonical
    # order; pieces of a non-convex tooth are written as bare vertices.
    faces = _is_convex_poly(bolt_profile) and _is_convex_poly(nut_profile)

    # ---- nut shell -----------------------------------------------------
    shell_pieces: list[tuple[str, np.ndarray, float, float]] = []
    if include_nut:
        # Inner radius: the nut root, moved out c/2 like the teeth (for ISO,
        # the major radius + c/2: the bolt crest, lowered c/2, clears it by c).
        r_bore_inner = r_root + clearance / 2
        if nut_shell_shape == 'cylinder':
            r_outer = nut_outer_diameter / 2
            shell_polys = _decompose_nut_shell_cylinder(
                r_bore_inner, r_outer, nut_shell_sectors)
        elif nut_shell_shape == 'hex':
            shell_polys = _decompose_nut_shell_hex(
                r_bore_inner, nut_across_flats)
        else:
            raise ValueError(f"nut_shell_shape must be 'cylinder' or 'hex', "
                             f"got {nut_shell_shape!r}")
        # Shell is centered axially on the nut threads (which span [0, nut_height]).
        z_nut_center = nut_height / 2
        for i, poly in enumerate(shell_polys):
            shell_pieces.append(
                (f'nut_shell_{i}', poly, nut_height, z_nut_center))

    # ---- bolt head outline (polygon for extruded mesh) ------------------
    bolt_head_poly = None
    if include_bolt and bolt_head_height > 0:
        if bolt_head_shape == 'cylinder':
            bolt_head_poly = _circle_polygon(bolt_head_diameter / 2, n=32)
        elif bolt_head_shape == 'hex':
            bolt_head_poly = _hex_polygon(bolt_head_across_flats)
        else:
            raise ValueError(f"bolt_head_shape must be 'cylinder' or 'hex', "
                             f"got {bolt_head_shape!r}")

    # ---- assets --------------------------------------------------------
    if mesh_dir is not None:
        mesh_dir.mkdir(parents=True, exist_ok=True)
    assets = ['  <asset>']

    for name, poly, thick, zc in shell_pieces:
        assets.append(_extrude_mesh_xml(name, poly, thick, zc, mesh_dir))

    # Bolt thread wedges: one mesh per wedge (MuJoCo hulls the 8 vertices).
    for i, v in enumerate(bolt_wedges):
        assets.append(_wedge_mesh_xml(f'bolt_thread_{i}', v, mesh_dir,
                                      flip=False, faces=faces))
    # Nut thread wedges: flip=True because the profile is reflected (dr < 0).
    for i, v in enumerate(nut_wedges):
        assets.append(_wedge_mesh_xml(f'nut_thread_{i}', v, mesh_dir,
                                      flip=True, faces=faces))

    # Bolt head mesh (if hex) — cylinder head uses the MuJoCo primitive.
    if bolt_head_poly is not None and bolt_head_shape == 'hex':
        z_head_center = (thread_length + bolt_shank_length +
                         bolt_head_height / 2)
        assets.append(
            _extrude_mesh_xml('bolt_head', bolt_head_poly,
                              bolt_head_height, z_head_center, mesh_dir))

    assets.append('  </asset>')

    # ---- worldbody -----------------------------------------------------
    wb = ['  <worldbody>']
    if not mesh_only:
        wb.append('    <light pos="0.3 0.3 0.3" dir="-0.3 -0.3 -1" '
                  'diffuse="0.8 0.8 0.8"/>')
    if include_floor and not mesh_only:
        wb.append('    <geom name="floor" type="plane" size="0.2 0.2 0.01" '
                  'rgba="0.92 0.92 0.92 1"/>')

    # Two scene layouts:
    #   bolt-driven (default): nut welded to the world, bolt above it spinning down.
    #   nut-driven: bolt welded head-at-floor tip-up, nut above spinning down.
    #
    # In nut-driven mode the bolt body is rotated 180° about the y-axis so the
    # local +z (head direction) flips to world −z. Threads in world still wind
    # as a right-handed helix (rotation preserves handedness).
    floor_clearance = 0.001
    bolt_full_length = thread_length + bolt_shank_length + bolt_head_height

    if nut_driven:
        eff_free_bolt = False
        eff_free_nut = True
        bolt_flipped = True
        # Bolt body origin at z_body so local z=0 (thread TIP) is at world
        # top of bolt. After the 180° y-flip, head bottom sits at floor_clearance.
        z_bolt_body = floor_clearance + bolt_full_length
        z_bolt_tip_world = z_bolt_body  # thread tip (local z=0) — now topmost.
        # Nut: bottom one pitch below bolt tip so first nut tooth engages first
        # bolt tooth, snapped to the thread phase of the flipped bolt. The flip
        # (a half turn about y, φ -> π - φ) maps the bolt's u to
        # z_bolt_body - lead/2 - u (mod P), so a mirrored nut tooth interlocks
        # at z_bolt_body - lead/2. A symmetric tooth is its own mirror about
        # its centre u_c, which puts the unmirrored nut at
        # z_bolt_body - lead/2 - 2 u_c (ISO single-start: u_c = 5P/8, phase
        # +P/4).
        u_c = (bolt_profile[:, 1].min() + bolt_profile[:, 1].max()) / 2
        phase = (-lead / 2 - (0.0 if mirror_nut else 2 * u_c)) % P
        z_nut_base = z_bolt_tip_world - nut_height + P
        z_nut_base = z_bolt_body + phase + P * round((z_nut_base - z_bolt_body - phase) / P)
    else:
        eff_free_bolt = free_bolt
        eff_free_nut = free_nut
        bolt_flipped = False
        z_nut_base = floor_clearance
        z_nut_top = z_nut_base + nut_height
        # Tip about one pitch into the nut, on the pitch grid relative to the
        # nut: frames a whole number of pitches apart are interlocked with
        # the flanks c apart on both sides.
        z_bolt_body = z_nut_base + P * max(0, round((nut_height - P) / P))
        z_bolt_tip_world = z_bolt_body

    if include_nut:
        wb.append(f'    <body name="nut" pos="0 0 {_f(z_nut_base)}">')
        if eff_free_nut:
            wb.append('      <freejoint/>')
        for name, _, _, _ in shell_pieces:
            wb.append(f'      <geom type="mesh" mesh="{name}" '
                      f'rgba="0.75 0.72 0.68 1" '
                      f'density="{nut_density:.3f}"/>')
        for i in range(len(nut_wedges)):
            wb.append(f'      <geom type="mesh" mesh="nut_thread_{i}" '
                      f'rgba="0.75 0.72 0.68 1" '
                      f'density="{nut_density:.3f}"/>')
        wb.append('    </body>')

    if include_bolt:
        # Body tag — include quat only when flipped (180° about y).
        if bolt_flipped:
            wb.append(f'    <body name="bolt" pos="0 0 {_f(z_bolt_body)}" '
                      f'quat="0 0 1 0">')
        else:
            wb.append(f'    <body name="bolt" pos="0 0 {_f(z_bolt_body)}">')
        if eff_free_bolt:
            wb.append('      <freejoint/>')
        # Geoms below are in LOCAL bolt coords. The body's quat (if flipped)
        # rotates them into world. Local layout: core at [0, thread_length],
        # shank above, head above shank. Same for both modes — the flip handles
        # orientation.
        z_core_center = thread_length / 2
        wb.append(f'      <geom name="bolt_core" type="cylinder" '
                  f'size="{_f(r_minor)} {_f(thread_length / 2)}" '
                  f'pos="0 0 {_f(z_core_center)}" '
                  f'rgba="0.55 0.58 0.62 1" '
                  f'density="{bolt_density:.3f}"/>')
        if bolt_shank_length > 0:
            z_shank_center = thread_length + bolt_shank_length / 2
            wb.append(f'      <geom name="bolt_shank" type="cylinder" '
                      f'size="{_f(r_major)} {_f(bolt_shank_length / 2)}" '
                      f'pos="0 0 {_f(z_shank_center)}" '
                      f'rgba="0.55 0.58 0.62 1" '
                      f'density="{bolt_density:.3f}"/>')
        if bolt_head_height > 0:
            z_head_center = thread_length + bolt_shank_length + bolt_head_height / 2
            if bolt_head_shape == 'cylinder':
                wb.append(f'      <geom name="bolt_head" type="cylinder" '
                          f'size="{_f(bolt_head_diameter / 2)} '
                          f'{_f(bolt_head_height / 2)}" '
                          f'pos="0 0 {_f(z_head_center)}" '
                          f'rgba="0.35 0.38 0.42 1" '
                          f'density="{bolt_density:.3f}"/>')
            else:  # hex
                wb.append(f'      <geom name="bolt_head" type="mesh" '
                          f'mesh="bolt_head" '
                          f'rgba="0.35 0.38 0.42 1" '
                          f'density="{bolt_density:.3f}"/>')
        for i in range(len(bolt_wedges)):
            wb.append(f'      <geom type="mesh" mesh="bolt_thread_{i}" '
                      f'rgba="0.55 0.58 0.62 1" '
                      f'density="{bolt_density:.3f}"/>')
        wb.append('    </body>')

    wb.append('  </worldbody>')

    # ---- header --------------------------------------------------------
    header = ['<mujoco model="screw">']
    if not mesh_only:
        header.append(
            f'  <option timestep="{timestep:g}" integrator="implicitfast" '
            f'gravity="0 0 -9.81"/>')
        header.append('  <default>')
        fric_str = ' '.join(f'{x:g}' for x in friction)
        header.append(
            f'    <geom solref="0.002 1" solimp="0.99 0.999 0.001" '
            f'condim="{condim}" friction="{fric_str}"/>')
        header.append('  </default>')
    return '\n'.join(header + assets + wb + ['</mujoco>']) + '\n'


# ----- config-driven entry -------------------------------------------------

def _parse_friction(val) -> tuple[float, float, float]:
    """Accept either a scalar (treated as tangential, with defaults for the
    torsional and rolling components) or an iterable of 3 floats."""
    if isinstance(val, (int, float)):
        return (float(val), 0.01, 0.001)
    t = tuple(float(x) for x in val)
    if len(t) != 3:
        raise ValueError(f"friction must be scalar or 3-tuple, got {val!r}")
    return t


def _scale_thread_params(params: dict | None, scale: float) -> dict:
    """thread_params with its lengths (THREAD_LENGTH_PARAMS) scaled."""
    out = {}
    for key, val in (params or {}).items():
        if key in THREAD_LENGTH_PARAMS:
            val = (np.asarray(val, dtype=float) * scale).tolist() \
                if isinstance(val, (list, tuple)) else float(val) * scale
        out[key] = val
    return out


def _parse_thread_param(text: str) -> tuple[str, object]:
    """KEY=VALUE, VALUE a Python literal (number, list, ...)."""
    import ast
    key, sep, val = text.partition('=')
    if not sep:
        raise argparse.ArgumentTypeError(f"--thread-param expects KEY=VALUE, got {text!r}")
    try:
        return key.strip(), ast.literal_eval(val.strip())
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"--thread-param {key}: not a literal: {val!r}")


def generate_screw_from_config(
    config: dict,
    config_dir: Path | None = None,
    mesh_dir: Path | None = None,
    scale: float | None = None,
) -> str:
    """Build MJCF from a config dict. All lengths are multiplied by `scale`
    (from the config's `scale` key, or overridden by this function's `scale`
    argument if provided — matches mkpeg.py behavior)."""
    if scale is None:
        scale = float(config.get('scale', 1.0))

    def L(v):
        """Scale a length scalar from the config."""
        return float(v) * scale

    bolt_cfg = config.get('bolt', {}) or {}
    nut_cfg = config.get('nut', {}) or {}

    return generate_screw_xml(
        major_diameter=L(config['major_diameter']),
        pitch=L(config['pitch']),
        thread_length=L(config['thread_length']),
        n_starts=int(config.get('n_starts', 1)),
        segments_per_turn=int(config.get('segments_per_turn', 24)),
        thread_form=config.get('thread_form', 'iso'),
        thread_params=_scale_thread_params(config.get('thread_params'), scale),
        handedness=config.get('handedness', 'right'),
        clearance=L(config.get('clearance', 0.00005)),
        seam_relief=L(config.get('seam_relief', 0.0)),
        thread_chamfer=L(config.get('thread_chamfer', 0.0)),
        include_bolt=bool(bolt_cfg.get('include', True)),
        bolt_shank_length=L(bolt_cfg.get('shank_length', 0.0)),
        bolt_head_height=L(bolt_cfg.get('head_height', 0.005)),
        bolt_head_diameter=L(bolt_cfg.get('head_diameter', 0.015)),
        bolt_head_across_flats=L(bolt_cfg.get('head_across_flats', 0.013)),
        bolt_head_shape=bolt_cfg.get('head_shape', 'cylinder'),
        bolt_density=float(bolt_cfg.get('density', 7800.0)),
        include_nut=bool(nut_cfg.get('include', True)),
        nut_height=L(nut_cfg.get('height', 0.008)),
        nut_outer_diameter=L(nut_cfg.get('outer_diameter', 0.016)),
        nut_across_flats=L(nut_cfg.get('across_flats', 0.016)),
        nut_shell_shape=nut_cfg.get('shell_shape', 'cylinder'),
        nut_shell_sectors=int(nut_cfg.get('shell_sectors', 16)),
        nut_density=float(nut_cfg.get('density', 7800.0)),
        include_floor=bool(config.get('include_floor', True)),
        free_bolt=bool(config.get('free_bolt', True)),
        free_nut=bool(config.get('free_nut', False)),
        nut_driven=bool(config.get('nut_driven', False)),
        mesh_only=bool(config.get('mesh_only', False)),
        friction=_parse_friction(config.get('friction', (1.0, 0.01, 0.001))),
        condim=int(config.get('condim', 4)),
        timestep=float(config.get('timestep', 0.001)),
        mesh_dir=mesh_dir,
    )


# ----- CLI -----------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # thread geometry
    p.add_argument('--major-diameter', type=float, default=0.010,
                   help='thread nominal diameter D, meters')
    p.add_argument('--pitch', type=float, default=0.0015,
                   help='thread pitch P (axial per revolution per start), meters')
    p.add_argument('--thread-length', type=float, default=0.020,
                   help='axial length of threaded region, meters')
    p.add_argument('--n-starts', type=int, default=1,
                   help='number of parallel thread starts (lead = n_starts * pitch)')
    p.add_argument('--segments-per-turn', type=int, default=24,
                   help='angular segments per helix revolution')
    p.add_argument('--thread-form', default='iso', choices=list(THREAD_FORMS),
                   help='tooth shape (see thread_profile): iso, trapezoidal, acme, '
                        'square, buttress, wire, custom')
    p.add_argument('--thread-param', type=_parse_thread_param, action='append', default=[],
                   metavar='KEY=VALUE',
                   help='override a thread-form default, e.g. flank_angles=[45,7] or '
                        'wire_diameter=0.004 (lengths in m, scaled by --scale); repeatable')
    p.add_argument('--handedness', default='right', choices=['right', 'left'])
    p.add_argument('--seam-relief', type=float, default=0.0,
                   help='seam relief depth (m), see generate_screw_xml')
    p.add_argument('--thread-chamfer', type=float, default=0.0,
                   help='chamfer the thread ends over this axial length (m); 0 = off')
    p.add_argument('--clearance', type=float, default=0.00005,
                   help='flank-normal clearance (reduces tooth size on both '
                        'bolt and nut by clearance/2)')
    # bolt
    p.add_argument('--no-bolt', action='store_true')
    p.add_argument('--bolt-shank-length', type=float, default=0.0)
    p.add_argument('--bolt-head-height', type=float, default=0.005)
    p.add_argument('--bolt-head-diameter', type=float, default=0.015)
    p.add_argument('--bolt-head-across-flats', type=float, default=0.013)
    p.add_argument('--bolt-head-shape', default='cylinder',
                   choices=['cylinder', 'hex'])
    p.add_argument('--bolt-density', type=float, default=7800.0)
    # nut
    p.add_argument('--no-nut', action='store_true')
    p.add_argument('--nut-height', type=float, default=0.008)
    p.add_argument('--nut-outer-diameter', type=float, default=0.016)
    p.add_argument('--nut-across-flats', type=float, default=0.016)
    p.add_argument('--nut-shell-shape', default='cylinder',
                   choices=['cylinder', 'hex'])
    p.add_argument('--nut-shell-sectors', type=int, default=16)
    p.add_argument('--nut-density', type=float, default=7800.0)
    # scene
    p.add_argument('--no-floor', action='store_true')
    p.add_argument('--free-bolt', action='store_true', default=True)
    p.add_argument('--free-nut', action='store_true')
    p.add_argument('--nut-driven', action='store_true',
                   help='flip scene: bolt welded head-at-floor tip-up, nut '
                        'free and positioned at top of bolt to thread down')
    p.add_argument('--friction', type=float, default=1.0,
                   help='tangential friction coefficient (default 1.0). '
                        'Drop to 0.1–0.3 for easier-spinning threads.')
    p.add_argument('--condim', type=int, default=4, choices=[3, 4, 6],
                   help='contact dimension. 3 = no torsional friction '
                        '(easiest spinning). 4 (default) adds torsion. '
                        '6 adds rolling.')
    p.add_argument('--timestep', type=float, default=0.001,
                   help='MuJoCo simulation timestep in seconds (default 0.001). '
                        'Smaller = stabler at the cost of speed.')
    p.add_argument('--mesh-only', action='store_true')
    p.add_argument('--export-meshes', action='store_true',
                   help='write each mesh as .obj next to the XML')
    # common
    p.add_argument('--config', type=Path,
                   help='load full scene from YAML/JSON/TOML config')
    p.add_argument('--scale', type=float, default=None,
                   help='uniform scale factor applied to every length')
    p.add_argument('--output', '-o', type=Path,
                   help='write MJCF here (default: stdout)')

    args = p.parse_args(argv)

    if args.export_meshes and not args.output:
        p.error("--export-meshes requires --output")
    mesh_dir = args.output.parent.resolve() if args.export_meshes else None

    if args.config:
        try:
            cfg = _load_config(args.config)
            # CLI overrides for a few run-time-flippable options:
            if args.nut_driven:
                cfg['nut_driven'] = True
            if args.thread_param:     # in the config's units, like its own
                cfg['thread_params'] = {**(cfg.get('thread_params') or {}),
                                        **dict(args.thread_param)}
            # Honor explicit CLI --timestep / --friction / --condim only when
            # they differ from the function defaults (otherwise they'd shadow
            # the config's values).
            if '--timestep' in (sys.argv or []):
                cfg['timestep'] = args.timestep
            if '--friction' in (sys.argv or []):
                cfg['friction'] = args.friction
            if '--condim' in (sys.argv or []):
                cfg['condim'] = args.condim
            xml = generate_screw_from_config(
                cfg, config_dir=args.config.parent.resolve(),
                mesh_dir=mesh_dir, scale=args.scale)
        except (ValueError, KeyError) as e:
            p.error(f"config error: {e}")
        if args.output:
            args.output.write_text(xml)
            print(f"wrote {args.output}", file=sys.stderr)
        else:
            sys.stdout.write(xml)
        return

    scale = args.scale if args.scale is not None else 1.0
    try:
        xml = generate_screw_xml(
            major_diameter=args.major_diameter * scale,
            pitch=args.pitch * scale,
            thread_length=args.thread_length * scale,
            n_starts=args.n_starts,
            segments_per_turn=args.segments_per_turn,
            thread_form=args.thread_form,
            thread_params=_scale_thread_params(dict(args.thread_param), scale),
            handedness=args.handedness,
            clearance=args.clearance * scale,
            seam_relief=args.seam_relief * scale,
            thread_chamfer=args.thread_chamfer * scale,
            include_bolt=not args.no_bolt,
            bolt_shank_length=args.bolt_shank_length * scale,
            bolt_head_height=args.bolt_head_height * scale,
            bolt_head_diameter=args.bolt_head_diameter * scale,
            bolt_head_across_flats=args.bolt_head_across_flats * scale,
            bolt_head_shape=args.bolt_head_shape,
            bolt_density=args.bolt_density,
            include_nut=not args.no_nut,
            nut_height=args.nut_height * scale,
            nut_outer_diameter=args.nut_outer_diameter * scale,
            nut_across_flats=args.nut_across_flats * scale,
            nut_shell_shape=args.nut_shell_shape,
            nut_shell_sectors=args.nut_shell_sectors,
            nut_density=args.nut_density,
            include_floor=not args.no_floor,
            free_bolt=args.free_bolt,
            free_nut=args.free_nut,
            nut_driven=args.nut_driven,
            friction=(args.friction, 0.01, 0.001),
            condim=args.condim,
            timestep=args.timestep,
            mesh_only=args.mesh_only,
            mesh_dir=mesh_dir,
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
