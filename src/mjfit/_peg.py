"""Peg-in-hole assemblies from planar outlines."""
from __future__ import annotations

import math
import numbers
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from . import _mjcf as mj
from . import _polygon as pg
from . import shapes

PLATE_RGBA = '0.70 0.75 0.82 1'
PEG_RGBA = '0.90 0.30 0.30 1'
HANDLE_RGBA = '0.55 0.30 0.30 1'
BOTTOM_RGBA = '0.25 0.25 0.25 1'


def peg(
    *,
    hole=None,
    peg=None,
    islands=(),
    clearance: float = 1e-4,
    peg_length: float = 0.03,
    chamfer: float = 0.0,
    handle=None,
    handle_height: float = 0.0,
    handle_margin: float = 0.003,
    plate=None,
    plate_size: float | None = None,
    plate_thickness: float = 0.01,
    min_rim: float = 0.0,
    bottom_thickness: float = 0.0,
    peg_mass: float | None = None,
    peg_density: float = 1000.0,
    plate_density: float = 1000.0,
    spawn_height: float = 0.005,
    center: bool = True,
    include_peg: bool = True,
    free_plate: bool = False,
    friction=1.0,
    condim: int = 4,
    timestep: float = 0.001,
    standalone: bool = True,
    floor: bool = True,
    mesh_dir: str | Path | None = None,
    scale: float = 1.0,
) -> str:
    """A plate with a hole and the peg that fits it, as an MJCF string.

    Give the cross-section of either the ``hole`` or the ``peg``; the other is
    its exact offset by ``clearance``, so every peg edge sits exactly
    ``clearance`` from its hole edge along the straight section. Every
    collision geom is convex.

    Args:
        hole: Outline of the hole: an (n, 2) vertex array, or a shape spec
            such as ``{'type': 'circle', 'radius': 0.008}`` (see
            :mod:`mjfit.shapes`).
        peg: Outline of the peg, instead of ``hole``.
        islands: Outlines of material standing inside the hole, such as a USB
            tongue or connector pins. The peg becomes a hollow shell with a
            pocket around each island. Needs ``hole``.
        clearance: Gap between every peg edge and its hole edge, in meters.
        peg_length: Length of the part of the peg that fits the hole.
        chamfer: Axial length and inward edge offset of a 45-degree bevel at
            the peg's bottom tip, in meters; 0 leaves a flat-ended prism.
            Hollow pegs also bevel the pocket mouths outward. Must be less
            than peg_length (and the pocket depth for hollow pegs), and small
            enough to leave a valid tip outline and positive wall thickness.
            The total peg length and straight-section clearance stay unchanged.
        handle: Convex outline of a wider grip section above the peg. It must
            contain the peg outline.
        handle_height: Height of the grip section; 0 for none. Without a
            ``handle`` outline it is the peg's convex hull grown by
            ``handle_margin``.
        handle_margin: How far the generated handle extends past the peg's
            convex hull.
        plate: Convex outer outline of the plate. Default: a square of side
            ``plate_size``.
        plate_size: Side of the square plate. Default: the hole's width plus a
            rim of the hole's radius, at least 5 mm.
        plate_thickness: Thickness of the plate (the depth of the hole).
        min_rim: Least distance from the outline of an explicit ``plate`` to the
            hole and handle.
        bottom_thickness: Closes the hole from below with a bottom this thick;
            0 leaves a through hole.
        peg_mass: Total mass of the peg and handle in kg, spread by volume.
            Overrides ``peg_density``.
        peg_density: Density of the peg, kg/m^3.
        plate_density: Density of the plate, kg/m^3 (matters with ``free_plate``).
        spawn_height: Gap between the peg's bottom face and the plate's top.
        center: Move all outlines together so the centroid of the given
            ``hole`` or ``peg`` is at the origin. False keeps the coordinates
            as given.
        include_peg: False emits only the plate.
        free_plate: Give the plate a free joint instead of fixing it.
        friction: Sliding friction, or the (sliding, torsional, rolling) triple.
        condim: MuJoCo contact dimensionality (1, 3, 4 or 6).
        timestep: Simulation time step in seconds (standalone files).
        standalone: Add a floor, a light and solver options so the file runs on
            its own. False emits only the parts, for including in a scene.
        floor: Add a ground plane (standalone files).
        mesh_dir: Write meshes as OBJ files into this folder, to be saved next
            to the XML file, instead of inline vertex lists.
        scale: Multiplies every length, including the outlines.

    Returns:
        The MJCF model as a string. The plate's bottom rests on z = 0; bodies
        are named ``plate`` and ``peg``.
    """
    lengths = dict(clearance=clearance, peg_length=peg_length, chamfer=chamfer,
                   handle_height=handle_height,
                   handle_margin=handle_margin, plate_thickness=plate_thickness, min_rim=min_rim,
                   bottom_thickness=bottom_thickness, spawn_height=spawn_height)
    for name, value in lengths.items():
        if not (isinstance(value, numbers.Real) and math.isfinite(value)):
            raise ValueError(f"{name} must be a finite number, got {value!r}.")
    _check(scale > 0, "scale must be positive.")
    for name in ('peg_length', 'plate_thickness'):
        _check(lengths[name] > 0, f"{name} must be positive.")
    for name in ('clearance', 'chamfer', 'handle_height', 'handle_margin', 'min_rim',
                 'bottom_thickness'):
        _check(lengths[name] >= 0, f"{name} must be nonnegative.")
    _check(peg_density > 0 and plate_density > 0, "Densities must be positive.")
    _check(peg_mass is None or (math.isfinite(peg_mass) and peg_mass > 0),
           "peg_mass must be positive.")
    _check((hole is None) != (peg is None), "Give exactly one of hole and peg.")
    _check(not (islands and hole is None), "islands need a hole outline, not a peg outline.")
    _check(not (handle is not None and handle_height <= 0),
           "A handle outline needs handle_height > 0.")
    _check(plate is None or plate_size is None, "Give plate or plate_size, not both.")
    _check(not (min_rim > 0 and plate is None), "min_rim applies to an explicit plate outline.")
    _check(not (peg_mass is not None and not include_peg), "peg_mass needs include_peg.")
    _check(chamfer < peg_length and not math.isclose(chamfer, peg_length, rel_tol=1e-12),
           "chamfer must be less than peg_length.")
    _check(not (chamfer > 0 and not include_peg), "chamfer needs include_peg.")

    c = clearance * scale
    bevel = chamfer * scale
    length, thick = peg_length * scale, plate_thickness * scale
    handle_h, margin = handle_height * scale, handle_margin * scale
    rim, bottom = min_rim * scale, bottom_thickness * scale

    def resolve(spec, name):
        return shapes.from_spec(spec, name=name) * scale

    # ---- outlines --------------------------------------------------------
    if peg is not None:
        source = resolve(peg, 'peg')
    else:
        source = resolve(hole, 'hole')
    shift = _centroid(source) if center else np.zeros(2)
    source = source - shift
    isl = [resolve(s, f'island {k}') - shift for k, s in enumerate(islands)]
    pockets: list[np.ndarray] = []
    if peg is not None:
        insert = source
        hole_poly = pg.offset_outward(insert, c)
        _check(pg.offset_is_valid(insert, hole_poly),
               f"The peg outline has features too small for a {clearance:g} m clearance.")
    else:
        hole_poly = source
        insert = pg.offset_inward(hole_poly, c)
        _check(pg.offset_is_valid(hole_poly, insert),
               f"clearance {clearance:g} m is too large for this hole: offsetting it inward "
               "by the clearance leaves no valid peg.")
        for k, island in enumerate(isl):
            _check(pg.contains(hole_poly, island), f"Island {k} is not inside the hole.")
            pocket = pg.offset_outward(island, c)
            _check(pg.offset_is_valid(island, pocket), f"Island {k} has features too small "
                   f"for a {clearance:g} m clearance.")
            _check(pg.contains(insert, pocket), f"The pocket around island {k} (the island plus "
                   "the clearance) does not fit inside the peg (the hole less the clearance).")
            for j, other in enumerate(pockets):
                _check(pg.disjoint(pocket, other),
                       f"Islands {j} and {k} are closer than twice the clearance.")
            pockets.append(pocket)

    tip, tip_pockets = insert, pockets
    if bevel > 0:
        tip = pg.offset_inward(insert, bevel)
        _check(pg.offset_is_valid(insert, tip) and pg.contains(insert, tip),
               "chamfer is too large for the peg outline: it collapses or folds the tip.")
        tip_pockets = []
        for k, pocket in enumerate(pockets):
            mouth = pg.offset_outward(pocket, bevel)
            _check(pg.offset_is_valid(pocket, mouth),
                   f"chamfer is too large for pocket {k}: its mouth folds over itself.")
            _check(pg.contains(tip, mouth) and _boundary_gap(tip, mouth) > pg.TOL,
                   f"chamfer is too large: the wall around pocket {k} disappears at the tip.")
            for j, other in enumerate(tip_pockets):
                _check(pg.disjoint(mouth, other) and _boundary_gap(mouth, other) > pg.TOL,
                       f"chamfer is too large: pocket mouths {j} and {k} meet at the tip.")
            tip_pockets.append(mouth)

    # ---- plate -----------------------------------------------------------
    if plate is not None:
        outer = pg.convex_outline(resolve(plate, 'plate') - shift, 'Plate outline')
        _check(pg.min_rim(outer, hole_poly) >= max(rim, 1e-9) - 1e-10,
               f"The plate outline must contain the hole with a rim of at least {min_rim:g} m.")
        plate_pieces = pg.triangulate(outer, [hole_poly])
    else:
        reach = float(np.abs(hole_poly).max())
        half = (reach + max(0.005 * scale, reach)) if plate_size is None else plate_size * scale / 2
        _check(half > reach, "plate_size leaves no rim around the hole.")
        outer = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
        if pg.is_star_shaped_from(hole_poly, hole_poly.mean(axis=0)):
            plate_pieces = pg.radial_wedges(hole_poly, half)
        else:
            plate_pieces = pg.triangulate(outer, [hole_poly])
    island_pieces = [p for island in isl
                     for p in ([island] if pg.is_convex(island) else pg.ear_clip(island))]

    # ---- peg -------------------------------------------------------------
    # Each entry: (geom name, polygon, z0, z1, rgba), z in the peg frame whose
    # origin is the middle of the insertion section.
    parts: list[tuple[str, np.ndarray, float, float, str]] = []
    # Chamfer meshes carry outward-facing triangles for export and exact mass.
    polyhedra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    chamfer_surface = None
    handle_poly = None
    if include_peg:
        tip_z, shoulder_z = -length / 2, -length / 2 + bevel
        if pockets:
            depth = thick + c                      # islands are as tall as the plate
            _check(depth < length, "peg_length must exceed plate_thickness + clearance so "
                   "the hollow peg has a solid back above its pockets.")
            _check(bevel < depth and not math.isclose(bevel, depth, rel_tol=1e-12),
                   "chamfer must be less than the pocket depth "
                   "(plate_thickness + clearance).")
            for k, p in enumerate(pg.triangulate(insert, pockets)):
                parts.append((f'peg_ring_{k}', p, shoulder_z, tip_z + depth, PEG_RGBA))
            for k, p in enumerate(pg.convex_pieces(insert)):
                parts.append((f'peg_back_{k}', p, -length / 2 + depth, length / 2, PEG_RGBA))
        elif bevel == 0 or not pg.is_convex(insert):
            for k, p in enumerate(pg.convex_pieces(insert)):
                parts.append((f'peg_{k}', p, shoulder_z, length / 2, PEG_RGBA))
        if bevel > 0:
            convex = not pockets and pg.is_convex(insert)
            if convex or mesh_dir is not None:
                chamfer_surface = _chamfer_surface(
                    insert, pockets, tip, tip_pockets, tip_z, length / 2, bevel, thick + c)
            if convex:
                polyhedra['peg_0'] = chamfer_surface
            else:
                # The tip core plus one wedge per boundary edge tiles each
                # intermediate offset exactly, including at concave corners.
                core = (pg.triangulate(tip, tip_pockets) if pockets
                        else pg.convex_pieces(tip))
                for k, p in enumerate(core):
                    parts.append((f'peg_tip_{k}', p, tip_z, shoulder_z, PEG_RGBA))
                rings = [(insert, tip)]
                rings += [(p[::-1], t[::-1]) for p, t in zip(pockets, tip_pockets)]
                for full, small in rings:
                    for i in range(len(full)):
                        j = (i + 1) % len(full)
                        polyhedra[f'peg_chamfer_{len(polyhedra)}'] = _chamfer_wedge(
                            small[[i, j]], full[[i, j]], tip_z, shoulder_z)
        if handle_h > 0:
            if handle is not None:
                handle_poly = pg.convex_outline(resolve(handle, 'handle') - shift, 'Handle outline')
                _check(pg.min_rim(handle_poly, insert) >= -1e-10,
                       "The handle outline must contain the peg outline.")
            else:
                handle_poly = pg.offset_outward(pg.convex_hull(insert), margin)
            if plate is not None:
                _check(pg.min_rim(outer, handle_poly) >= rim - 1e-10, "The plate outline must "
                       f"contain the handle with a rim of at least {min_rim:g} m.")
            for k, p in enumerate(pg.convex_pieces(handle_poly)):
                parts.append((f'handle_{k}', p, length / 2, length / 2 + handle_h, HANDLE_RGBA))

    masses = {}
    if peg_mass is not None:
        volumes = {name: abs(pg.signed_area(p)) * (z1 - z0) for name, p, z0, z1, _ in parts}
        volumes.update({name: _mesh_volume(v, f) for name, (v, f) in polyhedra.items()})
        total = sum(volumes.values())
        masses = {name: peg_mass * v / total for name, v in volumes.items()}

    # ---- MJCF ------------------------------------------------------------
    model = mj.Model('peg_in_hole')
    model.contact_defaults(friction=friction, condim=condim, visual_classes=mesh_dir is not None)
    meshes = mj.Meshes(model, mesh_dir)
    collision = {'class': mj.COLLISION} if meshes.exporting else {}
    if standalone:
        model.standalone(timestep=timestep, floor=floor, light_pos=(0.3, 0.3, 1.2),
                         floor_size=max(1.0, 4 * float(np.abs(outer).max())))
    wb = model.section('worldbody')

    plate_body = ET.SubElement(wb, 'body', name='plate', childclass=mj.CLASS,
                               pos=mj.vec((0, 0, bottom + thick / 2)))
    if free_plate:
        ET.SubElement(plate_body, 'freejoint', name='plate_joint')
    if meshes.exporting:
        solids = [(outer, [hole_poly], -thick / 2, thick / 2)]
        solids += [(island, [], -thick / 2, thick / 2) for island in isl]
        ET.SubElement(plate_body, 'geom', {'type': 'mesh', 'class': mj.VISUAL, 'rgba': PLATE_RGBA,
                                           'mesh': meshes.solid('plate_visual', solids)})
    pieces = [(f'plate_{k}', p) for k, p in enumerate(plate_pieces)]
    pieces += [(f'island_{k}', p) for k, p in enumerate(island_pieces)]
    for name, p in pieces:
        _geom(plate_body, name, meshes.prism(name, p, -thick / 2, thick / 2), PLATE_RGBA,
              collision, density=plate_density)
    if bottom > 0:
        _geom(plate_body, 'bottom', meshes.prism('bottom', outer, -thick / 2 - bottom, -thick / 2),
              BOTTOM_RGBA, collision, density=plate_density)

    if include_peg:
        z = bottom + thick + spawn_height * scale + length / 2
        peg_body = ET.SubElement(wb, 'body', name='peg', childclass=mj.CLASS, pos=mj.vec((0, 0, z)))
        ET.SubElement(peg_body, 'freejoint', name='peg_joint')
        if meshes.exporting:
            if bevel > 0:
                visual = meshes.points('peg_visual', *chamfer_surface)
            elif pockets:
                solids = [(insert, list(pockets), -length / 2, -length / 2 + thick + c),
                          (insert, [], -length / 2 + thick + c, length / 2)]
                visual = meshes.solid('peg_visual', solids)
            else:
                visual = meshes.solid('peg_visual', [(insert, [], -length / 2, length / 2)])
            ET.SubElement(peg_body, 'geom', {'type': 'mesh', 'class': mj.VISUAL, 'rgba': PEG_RGBA,
                                             'mesh': visual})
            if handle_poly is not None:
                ET.SubElement(peg_body, 'geom', {
                    'type': 'mesh', 'class': mj.VISUAL, 'rgba': HANDLE_RGBA,
                    'mesh': meshes.solid('handle_visual',
                                         [(handle_poly, [], length / 2, length / 2 + handle_h)])})
        for name, p, z0, z1, rgba in parts:
            _geom(peg_body, name, meshes.prism(name, p, z0, z1), rgba, collision,
                  density=peg_density, mass=masses.get(name))
        for name, (vertices, faces) in polyhedra.items():
            _geom(peg_body, name, meshes.points(name, vertices, faces), PEG_RGBA, collision,
                  density=peg_density, mass=masses.get(name))
    return model.to_xml()


def _chamfer_wedge(tip, full, z0, z1):
    """One convex edge wedge. Boundaries run with material on their left."""
    vertices = np.vstack([
        np.column_stack((tip, np.full(2, z0))),
        np.column_stack((tip, np.full(2, z1))),
        np.column_stack((full, np.full(2, z1))),
    ])
    faces = np.array([
        [2, 4, 5], [2, 5, 3],       # top
        [0, 2, 3], [0, 3, 1],       # core wall
        [0, 1, 5], [0, 5, 4],       # sloping outer face
        [0, 4, 2], [1, 3, 5],       # ends
    ])
    return vertices, faces


def _chamfer_surface(outer, holes, tip, mouths, z0, z1, height, depth):
    """Closed outer surface, without internal decomposition walls.

    For a solid convex peg this is also a single exact collision mesh.
    Pocket boundaries run clockwise so all side faces point out of material.
    """
    vertices, faces = [], []

    def cap(outline, pockets, z, up):
        for triangle in pg.triangulate(outline, pockets):
            start = len(vertices)
            vertices.extend((x, y, z) for x, y in triangle)
            faces.append((start, start + 1, start + 2) if up
                         else (start, start + 2, start + 1))

    def wall(rings):
        start = len(vertices)
        n = len(rings[0][0])
        for ring, z in rings:
            vertices.extend((x, y, z) for x, y in ring)
        for layer in range(len(rings) - 1):
            for i in range(n):
                j = (i + 1) % n
                a, b = start + layer * n + i, start + layer * n + j
                faces.extend(((a, b, b + n), (a, b + n, a + n)))

    cap(tip, mouths, z0, False)
    cap(outer, [], z1, True)
    wall([(tip, z0), (outer, z0 + height), (outer, z1)])
    for hole, mouth in zip(holes, mouths):
        wall([(mouth[::-1], z0), (hole[::-1], z0 + height), (hole[::-1], z0 + depth)])
        cap(hole, [], z0 + depth, False)
    return np.asarray(vertices), np.asarray(faces)


def _mesh_volume(vertices, faces):
    """Signed tetrahedra about a nearby origin, for outward-facing triangles."""
    a, b, c = (vertices - vertices.mean(axis=0))[faces].transpose(1, 0, 2)
    return abs(float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum())) / 6


def _boundary_gap(a, b):
    """Minimum separation of noncrossing boundaries, including touching edges."""
    gap = math.inf
    for points, ring in ((a, b), (b, a)):
        edges = np.roll(ring, -1, axis=0) - ring
        delta = points[:, None] - ring
        t = np.clip(np.einsum('ijk,jk->ij', delta, edges) / (edges * edges).sum(axis=1), 0, 1)
        gap = min(gap, float(np.linalg.norm(delta - t[:, :, None] * edges, axis=2).min()))
    return gap


def _geom(body, name, mesh, rgba, extra, *, density, mass=None):
    attrs = {'name': name, 'type': 'mesh', 'mesh': mesh, **extra}
    if not extra:
        attrs['rgba'] = rgba
    if mass is not None:
        attrs['mass'] = mj.num(mass)
    else:
        attrs['density'] = mj.num(density)
    ET.SubElement(body, 'geom', attrs)


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _centroid(poly: np.ndarray) -> np.ndarray:
    following = np.roll(poly, -1, axis=0)
    cross = poly[:, 0] * following[:, 1] - following[:, 0] * poly[:, 1]
    return ((poly + following) * cross[:, None]).sum(axis=0) / (3.0 * cross.sum())
