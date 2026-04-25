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

    ISO 68-1 basic profile:
      - sharp-V height      H = P·√3/2
      - thread depth        h = 5H/8 = 0.5413·P
      - root flat (axial)   P/4
      - flank span (axial)  5P/16  (30° from axis, 60° included)
      - crest flat (axial)  P/8

    Clearance reduces the tooth: crest radius shifts in by clearance/2,
    flanks shift along their normals by clearance/2 (which moves the crest-
    start / crest-end axial positions inward by clearance/(2·sin(60°))).
    """
    P = pitch
    H = P * np.sqrt(3) / 2
    h = 5 * H / 8  # thread depth, 0.5413·P

    flank_axial = 5 * P / 16
    crest_axial = P / 8
    root_axial = P / 4

    z_base_start = root_axial
    z_crest_start = z_base_start + flank_axial
    z_crest_end = z_crest_start + crest_axial
    z_base_end = z_crest_end + flank_axial  # == P

    dr_crest = h - clearance / 2
    # Keep the flank slope at √3 (60° from axis) while reducing crest r by c/2.
    # The flank still goes from (P/4, r_min) with slope √3, but now it reaches
    # (r_min + h - c/2) earlier in z, so the crest starts earlier and ends
    # later:   new_crest_start = 9P/16 − c/(2√3)
    #          new_crest_end   = 11P/16 + c/(2√3)
    # i.e. the crest WIDENS as we reduce its radius (not narrows — easy to
    # get this sign wrong; getting it wrong gives the wrong flank slope).
    dz_flank_adj = (clearance / 2) * flank_axial / h  # = c/(2√3)
    z_crest_start_adj = z_crest_start - dz_flank_adj
    z_crest_end_adj = z_crest_end + dz_flank_adj

    if dr_crest <= 0:
        raise ValueError(
            f"clearance ({clearance}) exceeds thread depth ({h}); "
            f"no tooth remains.")
    if z_crest_end_adj >= P:
        raise ValueError(
            f"clearance ({clearance}) too large for pitch ({P}); "
            f"crest runs into next pitch period.")

    # Vertices: CCW when viewed with (z on x-axis, dr on y-axis, external tooth
    # pointing up). Ordered so that the first two are base vertices and the
    # last two are crest vertices — the sweep generator relies on this order
    # for consistent face topology.
    return np.array([
        [0.0, z_base_start],       # base-left
        [0.0, z_base_end],         # base-right
        [dr_crest, z_crest_end_adj],   # crest-right
        [dr_crest, z_crest_start_adj], # crest-left
    ])


def _iso_thread_profile_internal(pitch: float, clearance: float) -> np.ndarray:
    """Nut thread tooth — same shape as external but tooth points inward
    (negative dr, i.e. smaller radius than the bore reference). The reference
    radius for an internal profile is the major radius (bore). Tooth tip is
    at dr = -(h - clearance/2)."""
    ext = _iso_thread_profile_external(pitch, clearance)
    out = ext.copy()
    out[:, 0] = -out[:, 0]
    return out


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


def _wedge_has_volume(verts: np.ndarray, eps: float = 1e-15) -> bool:
    """Conservative check that 8 vertices span a nonzero-volume hull.
    Uses the signed volume of the tetrahedron spanned by (v0, v1, v2, v4):
    if it's nonzero the 8-point hull is nondegenerate."""
    v0, v1, v2, v4 = verts[0], verts[1], verts[2], verts[4]
    m = np.column_stack([v1 - v0, v2 - v0, v4 - v0])
    return abs(np.linalg.det(m)) > eps


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
    bore = _circle_polygon(r_inner, n_bore)
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
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
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


def _write_wedge_obj(path: Path, wedges: list[np.ndarray], flip: bool = False) -> None:
    """Write a batch of 8-vertex thread wedges as a single OBJ. Face topology
    is the trapezoid-swept prism connectivity above. `flip` reverses triangle
    winding (used for nut threads, which have -dr profile)."""
    lines = ["# Generated by mkscrew.py"]
    face_lines: list[str] = []
    n_verts = 0

    for verts in wedges:
        base_idx = n_verts + 1
        for (x, y, z) in verts:
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
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
    vstr = ' '.join(f'{x:.6f}' for x in verts)
    return f'    <mesh name="{name}" vertex="{vstr}"/>'


def _wedge_mesh_xml(name: str, verts: np.ndarray,
                    mesh_dir: Path | None = None,
                    flip: bool = False) -> str:
    """Return MJCF `<mesh>` for a single 8-vertex thread wedge.

    When mesh_dir is None: inline `vertex="..."` form — MuJoCo convex-hulls.
    When mesh_dir is set: write a small OBJ with explicit face topology.
    """
    if mesh_dir is not None:
        _write_wedge_obj(mesh_dir / f'{name}.obj', [verts], flip=flip)
        return f'    <mesh name="{name}" file="{name}.obj"/>'
    flat = verts.flatten()
    vstr = ' '.join(f'{x:.6f}' for x in flat)
    return f'    <mesh name="{name}" vertex="{vstr}"/>'


# ----- top-level MJCF assembly ---------------------------------------------

def generate_screw_xml(
    major_diameter: float = 0.010,
    pitch: float = 0.0015,
    thread_length: float = 0.020,
    n_starts: int = 1,
    segments_per_turn: int = 24,
    thread_form: str = 'iso',
    handedness: str = 'right',
    clearance: float = 0.00005,
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

    friction is the MuJoCo geom friction triple (tangential, torsional,
    rolling). Drop tangential toward 0.1–0.2 to make threading less sticky.
    condim is the contact dimension: 4 includes torsional friction (resists
    rotation), 3 drops it for freer-spinning threads. 6 is rarely useful."""
    if thread_form != 'iso':
        raise ValueError(f"thread_form {thread_form!r} not supported (only 'iso')")
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
    # ISO minor radius:
    H = P * np.sqrt(3) / 2
    h = 5 * H / 8
    r_minor = r_major - h

    # Geometric limit: at clearance = 2h both the crest radius and crest-axial
    # width degenerate simultaneously (dr_crest → 0 and z_crest_end → P). We
    # reject at 2h (tooth-gone) and warn from h onward (tooth < half depth).
    if clearance >= 2 * h:
        raise ValueError(
            f"clearance ({clearance}) >= 2 * thread depth ({2*h:.6f}); "
            f"no tooth remains. For looser threading reduce friction "
            f"(default 'friction=\"1.0 ...\"' → '0.2') rather than pushing "
            f"clearance this high.")
    if clearance > h:
        import warnings
        warnings.warn(
            f"clearance ({clearance}) > thread depth ({h:.6f}); tooth is "
            f"shallower than half of full height — thread may not engage "
            f"reliably. Consider dropping friction instead for 'easier spin'.")

    sign = 1 if handedness == 'right' else -1
    dtheta = 2 * np.pi / segments_per_turn

    # Bolt threads span [0, thread_length] in the bolt local frame.
    # Nut threads span only [0, nut_height] (nut's bore is shorter than the
    # bolt shaft in general). Interleave the two with a half-pitch axial
    # offset so teeth of one fit into the root flats of the other.
    bolt_profile = _iso_thread_profile_external(P, clearance)
    nut_profile = _iso_thread_profile_internal(P, clearance)

    def _gen_wedges(profile, base_radius, axial_extent, extra_z_offset=0.0):
        """Generate wedge vertex arrays for all starts × turns × segments.

        The helix covers [0, axial_extent] axially; n_turns is chosen so the
        helix wraps fully across that range. `extra_z_offset` shifts every
        wedge by that amount (used to put the nut's teeth at half-pitch
        offset from the bolt's so they interlock).

        Multi-start threads are interleaved AXIALLY. Each start k has its
        first tooth at z = k·P; all N helices share the same angular range
        [0, 2π·n_turns] — that produces the "tooth every P" axial density
        regardless of n_starts.
        """
        n_turns = int(np.ceil(axial_extent / lead)) + 2
        wedges = []
        z_adv = sign * dtheta * lead / (2 * np.pi)
        for start_idx in range(n_starts):
            for turn in range(n_turns):
                z_turn_base = start_idx * P + turn * lead + extra_z_offset
                for seg in range(segments_per_turn):
                    theta_i = seg * dtheta
                    z_at_theta = z_turn_base + sign * theta_i * lead / (2 * np.pi)
                    v = _thread_wedge_vertices(
                        profile, theta_i, dtheta, z_adv,
                        base_radius, z_at_theta)
                    if _wedge_has_volume(v):
                        # Clip: keep only wedges overlapping [0, axial_extent].
                        if v[:, 2].max() >= -1e-9 and v[:, 2].min() <= axial_extent + 1e-9:
                            wedges.append(v)
        return wedges

    bolt_wedges = (_gen_wedges(bolt_profile, r_minor, thread_length)
                   if include_bolt else [])
    # Nut threads span [0, nut_height] with an axial offset that aligns the
    # nut's falling flank exactly onto the bolt's rising flank (zero-clearance
    # case). Derivation: bolt rising flank starts at z=P/4 on the line
    # r−r_min = √3·(z − P/4). The nut falling flank (unshifted profile) sits
    # on r−r_min = √3·(z − 11P/16). Matching intercepts:
    #     Δz_nut = P/4 − 11P/16 = −7P/16
    # That's the offset the nut needs relative to the bolt for tight interlock.
    # The small clearance-dependent correction (c/√3) is absorbed into dz_flank_adj
    # inside the profile itself, so no adjustment needed here.
    nut_thread_offset = -7 * P / 16
    nut_wedges = (_gen_wedges(nut_profile, r_major, nut_height,
                              extra_z_offset=nut_thread_offset)
                  if include_nut else [])

    # ---- nut shell -----------------------------------------------------
    shell_pieces: list[tuple[str, np.ndarray, float, float]] = []
    if include_nut:
        # Inner radius: outer-facing surface of the nut bore.
        # Must be slightly inside r_major so the bolt thread crest doesn't
        # scrape the shell. Set at r_major + clearance/2 (so the shell chord
        # sits exactly at the nominal bolt crest clearance plane after the
        # mid-arc inflation).
        r_bore_inner = r_major + clearance / 2
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
                                      flip=False))
    # Nut thread wedges: flip=True because the profile is reflected (dr < 0).
    for i, v in enumerate(nut_wedges):
        assets.append(_wedge_mesh_xml(f'nut_thread_{i}', v, mesh_dir,
                                      flip=True))

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
        # bolt tooth.
        z_nut_base = z_bolt_tip_world - nut_height + P
    else:
        eff_free_bolt = free_bolt
        eff_free_nut = free_nut
        bolt_flipped = False
        z_nut_base = floor_clearance
        z_nut_top = z_nut_base + nut_height
        z_bolt_body = z_nut_top - P  # tip engaged one pitch into the nut.
        z_bolt_tip_world = z_bolt_body

    if include_nut:
        wb.append(f'    <body name="nut" pos="0 0 {z_nut_base:.6f}">')
        if eff_free_nut:
            wb.append('      <freejoint/>')
        for name, _, _, _ in shell_pieces:
            wb.append(f'      <geom type="mesh" mesh="{name}" '
                      f'rgba="0.75 0.72 0.68 1" '
                      f'density="{nut_density:.3f}"/>')
        for i in range(len(nut_wedges)):
            wb.append(f'      <geom type="mesh" mesh="nut_thread_{i}" '
                      f'rgba="0.75 0.72 0.68 1" '
                      f'density="{nut_density:.3f}" '
                      f'contype="2" conaffinity="2"/>')
        wb.append('    </body>')

    if include_bolt:
        # Body tag — include quat only when flipped (180° about y).
        if bolt_flipped:
            wb.append(f'    <body name="bolt" pos="0 0 {z_bolt_body:.6f}" '
                      f'quat="0 0 1 0">')
        else:
            wb.append(f'    <body name="bolt" pos="0 0 {z_bolt_body:.6f}">')
        if eff_free_bolt:
            wb.append('      <freejoint/>')
        # Geoms below are in LOCAL bolt coords. The body's quat (if flipped)
        # rotates them into world. Local layout: core at [0, thread_length],
        # shank above, head above shank. Same for both modes — the flip handles
        # orientation.
        z_core_center = thread_length / 2
        wb.append(f'      <geom name="bolt_core" type="cylinder" '
                  f'size="{r_minor:.6f} {thread_length / 2:.6f}" '
                  f'pos="0 0 {z_core_center:.6f}" '
                  f'rgba="0.55 0.58 0.62 1" '
                  f'density="{bolt_density:.3f}"/>')
        if bolt_shank_length > 0:
            z_shank_center = thread_length + bolt_shank_length / 2
            wb.append(f'      <geom name="bolt_shank" type="cylinder" '
                      f'size="{r_major:.6f} {bolt_shank_length / 2:.6f}" '
                      f'pos="0 0 {z_shank_center:.6f}" '
                      f'rgba="0.55 0.58 0.62 1" '
                      f'density="{bolt_density:.3f}"/>')
        if bolt_head_height > 0:
            z_head_center = thread_length + bolt_shank_length + bolt_head_height / 2
            if bolt_head_shape == 'cylinder':
                wb.append(f'      <geom name="bolt_head" type="cylinder" '
                          f'size="{bolt_head_diameter / 2:.6f} '
                          f'{bolt_head_height / 2:.6f}" '
                          f'pos="0 0 {z_head_center:.6f}" '
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
                      f'density="{bolt_density:.3f}" '
                      f'contype="2" conaffinity="2"/>')
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
        handedness=config.get('handedness', 'right'),
        clearance=L(config.get('clearance', 0.00005)),
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
    p.add_argument('--thread-form', default='iso', choices=['iso'],
                   help='thread profile form (only ISO 60° in v1)')
    p.add_argument('--handedness', default='right', choices=['right', 'left'])
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
            handedness=args.handedness,
            clearance=args.clearance * scale,
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
