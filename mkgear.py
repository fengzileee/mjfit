#!/usr/bin/env python3
"""Generate MuJoCo MJCF for spur gears with convex primitive collision geoms.

Each gear is generated as a real gear-like 2D outline, rendered as one visual
mesh and decomposed into hidden convex triangular-prism collision geoms. There
is no colliding root disk filling the tooth valleys and no rectangular tooth
blocks. Contact stays MuJoCo-native convex-vs-convex, matching the rest of
this repository's generators.

The default scene is a pair of external spur gears mounted on hinge joints,
with tooth phases chosen so one tooth starts in the opposite gear's gap.

CLI:
    python mkgear.py -o gear_pair.xml
    python mkgear.py --teeth 16 --gear-b-teeth 32 --module 0.003 -o gears.xml
    python mkgear.py --single --teeth 24 -o gear.xml
    python mkgear.py --config configs/gears/spur_pair.yaml -o gear_pair.xml

Library:
    from mkgear import generate_gear_xml
    xml = generate_gear_xml(teeth=16, gear_b_teeth=24, module=0.003)
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


# ----- config loader -------------------------------------------------------

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


def _parse_friction(val) -> tuple[float, float, float]:
    """Accept either a scalar or a 3-tuple friction triple."""
    if isinstance(val, (int, float)):
        return (float(val), 0.01, 0.001)
    t = tuple(float(x) for x in val)
    if len(t) != 3:
        raise ValueError(f"friction must be scalar or 3-tuple, got {val!r}")
    return t


# ----- gear geometry -------------------------------------------------------

def _gear_dimensions(
    teeth: int,
    module: float,
    pressure_angle: float,
    addendum_coeff: float,
    dedendum_coeff: float,
    tooth_width_ratio: float,
) -> dict[str, float]:
    """Return derived standard-ish spur-gear dimensions.

    module m gives pitch radius r_p = m*z/2. Addendum/dedendum are kept as
    configurable coefficients because coarse primitive teeth often benefit
    from being a little shallower than a true involute profile.
    """
    if teeth < 3:
        raise ValueError("teeth must be >= 3")
    if module <= 0:
        raise ValueError("module must be positive")
    if not (0 < pressure_angle < math.radians(45)):
        raise ValueError("pressure_angle must be in (0, 45 degrees)")
    if addendum_coeff <= 0 or dedendum_coeff <= 0:
        raise ValueError("addendum_coeff and dedendum_coeff must be positive")
    if not (0.1 < tooth_width_ratio < 0.8):
        raise ValueError("tooth_width_ratio must be in (0.1, 0.8)")

    pitch_radius = module * teeth / 2
    base_radius = pitch_radius * math.cos(pressure_angle)
    addendum = addendum_coeff * module
    dedendum = dedendum_coeff * module
    root_radius = pitch_radius - dedendum
    outer_radius = pitch_radius + addendum
    if root_radius <= module * 0.25:
        raise ValueError(
            f"root radius {root_radius:.6g} is too small; increase teeth "
            f"or reduce dedendum_coeff.")

    half_pitch_angle = math.pi / teeth
    half_tooth_angle_pitch = half_pitch_angle * tooth_width_ratio
    if half_tooth_angle_pitch >= half_pitch_angle:
        raise ValueError(
            "tooth_width_ratio leaves no gap at the pitch circle; reduce it.")

    return {
        'teeth': float(teeth),
        'module': module,
        'pressure_angle': pressure_angle,
        'pitch_radius': pitch_radius,
        'base_radius': base_radius,
        'root_radius': root_radius,
        'outer_radius': outer_radius,
        'half_pitch_angle': half_pitch_angle,
        'half_tooth_angle_pitch': half_tooth_angle_pitch,
    }


def _fmt(x: float) -> str:
    return f"{x:.6f}"


def _polar(radius: float, angle: float) -> tuple[float, float]:
    return radius * math.cos(angle), radius * math.sin(angle)


def _involute(a: float) -> float:
    return math.tan(a) - a


def _tooth_half_angle(radius: float, dims: dict[str, float]) -> float:
    """Half tooth angular thickness at a given radius.

    Uses the standard involute relation:
      theta(r) = theta_pitch + inv(alpha) - inv(alpha_r)
    where alpha_r is the pressure angle at radius r.
    """
    base_radius = dims['base_radius']
    pitch_radius = dims['pitch_radius']
    pressure_angle = dims['pressure_angle']
    theta_pitch = dims['half_tooth_angle_pitch']
    if radius <= base_radius:
        return theta_pitch + _involute(pressure_angle)
    alpha_r = math.acos(max(-1.0, min(1.0, base_radius / radius)))
    return theta_pitch + _involute(pressure_angle) - _involute(alpha_r)


def _gear_outline(
    dims: dict[str, float],
    phase: float = 0.0,
    flank_samples: int = 4,
    root_arc_samples: int = 2,
    tip_arc_samples: int = 2,
) -> list[tuple[float, float]]:
    """Build a CCW 2D gear outline with involute flanks.

    The root below the base circle is connected by radial flank segments,
    which is a simple but useful approximation of an unfilleted root.
    """
    teeth = int(dims['teeth'])
    root_radius = dims['root_radius']
    base_radius = dims['base_radius']
    outer_radius = dims['outer_radius']
    pitch = 2 * math.pi / teeth
    flank_start_radius = max(root_radius, base_radius)
    theta_root = _tooth_half_angle(flank_start_radius, dims)
    if theta_root >= pitch * 0.48:
        raise ValueError(
            "tooth base consumes almost the full tooth pitch; reduce "
            "tooth_width_ratio, pressure_angle, or dedendum_coeff.")

    outline: list[tuple[float, float]] = []

    def append_point(radius: float, angle: float) -> None:
        p = _polar(radius, angle)
        if outline:
            q = outline[-1]
            if abs(p[0] - q[0]) < 1e-12 and abs(p[1] - q[1]) < 1e-12:
                return
        outline.append(p)

    for tooth in range(teeth):
        center = phase + tooth * pitch
        prev_right_root = center - pitch + theta_root
        left_root = center - theta_root
        for j in range(1, root_arc_samples + 1):
            a = prev_right_root + (left_root - prev_right_root) * j / root_arc_samples
            append_point(root_radius, a)

        if root_radius < base_radius:
            append_point(base_radius, left_root)

        for j in range(flank_samples + 1):
            r = flank_start_radius + (outer_radius - flank_start_radius) * j / flank_samples
            append_point(r, center - _tooth_half_angle(r, dims))

        theta_outer = _tooth_half_angle(outer_radius, dims)
        for j in range(1, tip_arc_samples + 1):
            a = center - theta_outer + 2 * theta_outer * j / tip_arc_samples
            append_point(outer_radius, a)

        for j in range(1, flank_samples + 1):
            r = outer_radius - (outer_radius - flank_start_radius) * j / flank_samples
            append_point(r, center + _tooth_half_angle(r, dims))

        if root_radius < base_radius:
            append_point(root_radius, center + theta_root)

    return outline


def _triangle_area(a, b, c) -> float:
    return 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) -
                  (b[1] - a[1]) * (c[0] - a[0]))


def _mesh_xml(name: str, vertices: list[tuple[float, float, float]],
              faces: list[tuple[int, int, int]] | None = None) -> str:
    vstr = ' '.join(f'{coord:.6f}' for v in vertices for coord in v)
    if faces is None:
        return f'    <mesh name="{name}" vertex="{vstr}"/>'
    fstr = ' '.join(str(idx) for face in faces for idx in face)
    return f'    <mesh name="{name}" vertex="{vstr}" face="{fstr}"/>'


def _gear_visual_mesh_xml(name: str, outline: list[tuple[float, float]],
                          thickness: float) -> str:
    half = thickness / 2
    n = len(outline)
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, half),
                                                  (0.0, 0.0, -half)]
    vertices.extend((x, y, half) for x, y in outline)
    vertices.extend((x, y, -half) for x, y in outline)

    faces: list[tuple[int, int, int]] = []
    top0 = 2
    bot0 = 2 + n
    for i in range(n):
        j = (i + 1) % n
        faces.append((0, top0 + i, top0 + j))
        faces.append((1, bot0 + j, bot0 + i))
        faces.append((bot0 + i, bot0 + j, top0 + j))
        faces.append((bot0 + i, top0 + j, top0 + i))
    return _mesh_xml(name, vertices, faces)


def _gear_collision_mesh_xmls(name: str, outline: list[tuple[float, float]],
                              thickness: float) -> list[str]:
    half = thickness / 2
    center = (0.0, 0.0)
    meshes = []
    out_i = 0
    for i, a in enumerate(outline):
        b = outline[(i + 1) % len(outline)]
        if abs(_triangle_area(center, a, b)) < 1e-14:
            continue
        vertices = [
            (0.0, 0.0, -half),
            (a[0], a[1], -half),
            (b[0], b[1], -half),
            (0.0, 0.0, half),
            (a[0], a[1], half),
            (b[0], b[1], half),
        ]
        meshes.append(_mesh_xml(f'{name}_col_{out_i}', vertices))
        out_i += 1
    return meshes


def _gear_body_xml(
    name: str,
    x: float,
    z: float,
    collision_count: int,
    density: float,
    visual_rgba: str,
    hub_radius: float,
    hub_width: float,
    joint_damping: float,
    joint_armature: float,
) -> list[str]:
    lines = [f'    <body name="{name}" pos="{_fmt(x)} 0 {_fmt(z)}">']
    lines.append(
        f'      <joint name="{name}_hinge" type="hinge" axis="0 0 1" '
        f'damping="{joint_damping:g}" armature="{joint_armature:g}"/>')
    lines.append(
        f'      <geom name="{name}_visual" type="mesh" mesh="{name}_visual" '
        f'class="visual" rgba="{visual_rgba}"/>')

    if hub_radius > 0 and hub_width > 0:
        lines.append(
            f'      <geom name="{name}_hub" type="cylinder" '
            f'size="{_fmt(hub_radius)} {_fmt(hub_width / 2)}" '
            f'rgba="0.42 0.44 0.48 1" density="0" '
            f'contype="0" conaffinity="0"/>')

    for i in range(collision_count):
        lines.append(
            f'      <geom name="{name}_col_{i}" type="mesh" '
            f'mesh="{name}_col_{i}" class="collision" '
            f'density="{density:.3f}"/>')
    lines.append('    </body>')
    return lines


# ----- top-level MJCF assembly --------------------------------------------

def generate_gear_xml(
    teeth: int = 18,
    gear_b_teeth: int = 24,
    module: float = 0.003,
    face_width: float = 0.008,
    pressure_angle: float = math.radians(20),
    addendum_coeff: float = 1.00,
    dedendum_coeff: float = 1.25,
    tooth_width_ratio: float = 0.48,
    backlash: float = 0.00015,
    center_distance: float | None = None,
    include_gear_b: bool = True,
    gear_b_phase: float | None = None,
    # mass / appearance
    density: float = 7800.0,
    gear_b_density: float | None = None,
    hub_radius: float | None = None,
    hub_width: float | None = None,
    # scene / supports
    include_floor: bool = True,
    include_axles: bool = True,
    floor_clearance: float = 0.001,
    axle_radius: float | None = None,
    axle_margin: float = 0.004,
    mesh_only: bool = False,
    include_actuator: bool = True,
    motor_gear: float = 0.02,
    motor_ctrlrange: tuple[float, float] = (-1.0, 1.0),
    # solver
    friction: tuple[float, float, float] = (0.8, 0.01, 0.001),
    condim: int = 3,
    timestep: float = 0.001,
    joint_damping: float = 0.001,
    joint_armature: float = 1e-5,
) -> str:
    """Build a convex-decomposed spur-gear MJCF scene.

    The default pair uses one hinge joint per gear. `module` is shared by both
    gears, as it must be for external spur gears to mesh. Collision is an
    exact fan decomposition of the generated 2D outline into convex prisms.
    """
    if face_width <= 0:
        raise ValueError("face_width must be positive")
    if backlash < 0:
        raise ValueError("backlash must be non-negative")
    if condim not in (3, 4, 6):
        raise ValueError("condim must be 3, 4, or 6")

    dims_a = _gear_dimensions(
        teeth, module, pressure_angle, addendum_coeff, dedendum_coeff,
        tooth_width_ratio)
    dims_b = _gear_dimensions(
        gear_b_teeth, module, pressure_angle, addendum_coeff, dedendum_coeff,
        tooth_width_ratio)

    if center_distance is None:
        center_distance = (dims_a['pitch_radius'] +
                           dims_b['pitch_radius'] + backlash)
    elif center_distance <= 0:
        raise ValueError("center_distance must be positive")

    if not include_gear_b:
        center_distance = 0.0

    gear_b_density = density if gear_b_density is None else gear_b_density
    hub_radius = dims_a['root_radius'] * 0.35 if hub_radius is None else hub_radius
    hub_width = face_width * 1.25 if hub_width is None else hub_width
    axle_radius = module * 0.45 if axle_radius is None else axle_radius
    max_geom_width = max(face_width, hub_width if hub_radius > 0 else face_width)
    gear_z = max_geom_width / 2 + floor_clearance
    axle_height = gear_z + max_geom_width / 2 + axle_margin

    x_a = -center_distance / 2 if include_gear_b else 0.0
    x_b = center_distance / 2
    phase_a = 0.0
    # Put a gear-B gap on the line of centers so gear-A's phase-zero tooth
    # starts in that gap. Users can still override for experiments.
    phase_b = (math.pi + math.pi / gear_b_teeth
               if gear_b_phase is None else gear_b_phase)

    outline_a = _gear_outline(dims_a, phase_a)
    outline_b = _gear_outline(dims_b, phase_b)
    collision_count_a = len(_gear_collision_mesh_xmls(
        'gear_a', outline_a, face_width))
    collision_count_b = (len(_gear_collision_mesh_xmls(
        'gear_b', outline_b, face_width)) if include_gear_b else 0)

    header = ['<mujoco model="gear_pair">']
    if not mesh_only:
        header.append(
            f'  <option timestep="{timestep:g}" integrator="implicitfast" '
            f'gravity="0 0 -9.81"/>')
    header.append('  <default>')
    fric_str = ' '.join(f'{x:g}' for x in friction)
    if not mesh_only:
        header.append(
            f'    <geom solref="0.002 1" solimp="0.99 0.999 0.001" '
            f'condim="{condim}" friction="{fric_str}"/>')
        header.append(
            f'    <joint damping="{joint_damping:g}" '
            f'armature="{joint_armature:g}"/>')
    header.append('    <default class="visual">')
    header.append('      <geom group="2" contype="0" conaffinity="0" density="0"/>')
    header.append('    </default>')
    header.append('    <default class="collision">')
    header.append('      <geom group="3" rgba="0.05 0.75 1.0 0.35"/>')
    header.append('    </default>')
    header.append('  </default>')

    assets = ['  <asset>']
    assets.append(_gear_visual_mesh_xml('gear_a_visual', outline_a, face_width))
    assets.extend(_gear_collision_mesh_xmls('gear_a', outline_a, face_width))
    if include_gear_b:
        assets.append(_gear_visual_mesh_xml('gear_b_visual', outline_b, face_width))
        assets.extend(_gear_collision_mesh_xmls('gear_b', outline_b, face_width))
    assets.append('  </asset>')

    wb = ['  <worldbody>']
    if not mesh_only:
        wb.append('    <light pos="0.2 -0.35 0.35" dir="-0.2 0.35 -1" '
                  'diffuse="0.8 0.8 0.8"/>')
    if include_floor and not mesh_only:
        wb.append('    <geom name="floor" type="plane" size="0.4 0.4 0.01" '
                  'rgba="0.92 0.92 0.92 1"/>')

    if include_axles:
        for name, x in [('gear_a_axle', x_a),
                        ('gear_b_axle', x_b) if include_gear_b else (None, None)]:
            if name is None:
                continue
            wb.append(
                f'    <geom name="{name}" type="cylinder" '
                f'size="{_fmt(axle_radius)} {_fmt(axle_height / 2)}" '
                f'pos="{_fmt(x)} 0 {_fmt(axle_height / 2)}" '
                f'rgba="0.18 0.20 0.23 1" density="0" '
                f'contype="0" conaffinity="0"/>')

    wb.extend(_gear_body_xml(
        'gear_a', x_a, gear_z, collision_count_a, density,
        '0.55 0.58 0.62 1', min(hub_radius, dims_a['root_radius'] * 0.8), hub_width,
        joint_damping, joint_armature))
    if include_gear_b:
        wb.extend(_gear_body_xml(
            'gear_b', x_b, gear_z, collision_count_b, gear_b_density,
            '0.78 0.58 0.26 1', min(hub_radius, dims_b['root_radius'] * 0.8), hub_width,
            joint_damping, joint_armature))
    wb.append('  </worldbody>')

    actuator: list[str] = []
    if include_actuator and not mesh_only:
        lo, hi = motor_ctrlrange
        actuator = [
            '  <actuator>',
            f'    <motor name="drive_gear_a" joint="gear_a_hinge" '
            f'gear="{motor_gear:g}" ctrllimited="true" '
            f'ctrlrange="{lo:g} {hi:g}"/>',
            '  </actuator>',
        ]

    return '\n'.join(header + assets + wb + actuator + ['</mujoco>']) + '\n'


# ----- config-driven entry -------------------------------------------------

def generate_gear_from_config(
    config: dict,
    config_dir: Path | None = None,
    scale: float | None = None,
) -> str:
    """Build MJCF from a config dict. All lengths are multiplied by `scale`.

    `config_dir` is accepted for API symmetry with the other generators; the
    primitive gear schema currently has no external file references.
    """
    del config_dir
    if scale is None:
        scale = float(config.get('scale', 1.0))

    def L(v):
        return float(v) * scale

    gear_a = config.get('gear_a', {}) or {}
    gear_b = config.get('gear_b', {}) or {}
    support = config.get('support', {}) or {}
    actuator = config.get('actuator', {}) or {}

    include_gear_b = bool(gear_b.get('include',
                                     config.get('include_gear_b', True)))
    hub_radius = config.get('hub_radius', None)
    hub_width = config.get('hub_width', None)
    axle_radius = support.get('axle_radius', config.get('axle_radius', None))
    center_distance = config.get('center_distance', None)

    return generate_gear_xml(
        teeth=int(gear_a.get('teeth', config.get('teeth', 18))),
        gear_b_teeth=int(gear_b.get('teeth', config.get('gear_b_teeth', 24))),
        module=L(config.get('module', 0.003)),
        face_width=L(config.get('face_width', 0.008)),
        pressure_angle=math.radians(float(config.get('pressure_angle_degrees', 20.0))),
        addendum_coeff=float(config.get('addendum_coeff', 1.00)),
        dedendum_coeff=float(config.get('dedendum_coeff', 1.25)),
        tooth_width_ratio=float(config.get('tooth_width_ratio', 0.48)),
        backlash=L(config.get('backlash', 0.00015)),
        center_distance=(L(center_distance) if center_distance is not None else None),
        include_gear_b=include_gear_b,
        gear_b_phase=(float(gear_b['phase']) if 'phase' in gear_b else None),
        density=float(gear_a.get('density', config.get('density', 7800.0))),
        gear_b_density=float(gear_b.get('density',
                                        gear_a.get('density',
                                                   config.get('density', 7800.0)))),
        hub_radius=(L(hub_radius) if hub_radius is not None else None),
        hub_width=(L(hub_width) if hub_width is not None else None),
        include_floor=bool(config.get('include_floor', True)),
        include_axles=bool(support.get('include_axles',
                                       config.get('include_axles', True))),
        floor_clearance=L(config.get('floor_clearance', 0.001)),
        axle_radius=(L(axle_radius) if axle_radius is not None else None),
        axle_margin=L(support.get('axle_margin',
                                  config.get('axle_margin', 0.004))),
        mesh_only=bool(config.get('mesh_only', False)),
        include_actuator=bool(actuator.get('include',
                                           config.get('include_actuator', True))),
        motor_gear=float(actuator.get('gear', config.get('motor_gear', 0.02))),
        motor_ctrlrange=tuple(actuator.get(
            'ctrlrange', config.get('motor_ctrlrange', (-1.0, 1.0)))),
        friction=_parse_friction(config.get('friction', (0.8, 0.01, 0.001))),
        condim=int(config.get('condim', 3)),
        timestep=float(config.get('timestep', 0.001)),
        joint_damping=float(config.get('joint_damping', 0.001)),
        joint_armature=float(config.get('joint_armature', 1e-5)),
    )


# ----- CLI -----------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # gear geometry
    p.add_argument('--teeth', type=int, default=18,
                   help='number of teeth on gear A')
    p.add_argument('--gear-b-teeth', type=int, default=24,
                   help='number of teeth on gear B')
    p.add_argument('--module', type=float, default=0.003,
                   help='gear module m, meters per tooth')
    p.add_argument('--face-width', type=float, default=0.008,
                   help='gear thickness along z, meters')
    p.add_argument('--pressure-angle-degrees', type=float, default=20.0,
                   help='involute pressure angle in degrees')
    p.add_argument('--addendum-coeff', type=float, default=1.0,
                   help='tooth addendum as a multiple of module')
    p.add_argument('--dedendum-coeff', type=float, default=1.25,
                   help='root dedendum as a multiple of module')
    p.add_argument('--tooth-width-ratio', type=float, default=0.48,
                   help='tooth thickness at pitch circle / circular pitch')
    p.add_argument('--backlash', type=float, default=0.00015,
                   help='extra center spacing for the default gear pair')
    p.add_argument('--center-distance', type=float,
                   help='override center distance for a gear pair')
    p.add_argument('--single', action='store_true',
                   help='emit only gear A, centered at the origin')
    p.add_argument('--gear-b-phase', type=float,
                   help='override gear B tooth phase in radians')
    # body / supports
    p.add_argument('--density', type=float, default=7800.0)
    p.add_argument('--gear-b-density', type=float)
    p.add_argument('--hub-radius', type=float,
                   help='optional visual hub radius; default = 35%% root radius')
    p.add_argument('--hub-width', type=float,
                   help='optional visual hub height; default = 1.25 * face_width')
    p.add_argument('--no-floor', action='store_true')
    p.add_argument('--no-axles', action='store_true',
                   help='omit non-colliding visual axle posts')
    p.add_argument('--floor-clearance', type=float, default=0.001)
    p.add_argument('--axle-radius', type=float,
                   help='visual axle radius; default = 0.45 * module')
    p.add_argument('--axle-margin', type=float, default=0.004,
                   help='height above the gear top reached by visual axles')
    # solver / runtime
    p.add_argument('--friction', type=float, default=0.8,
                   help='tangential friction coefficient')
    p.add_argument('--condim', type=int, default=3, choices=[3, 4, 6])
    p.add_argument('--timestep', type=float, default=0.001)
    p.add_argument('--joint-damping', type=float, default=0.001)
    p.add_argument('--joint-armature', type=float, default=1e-5)
    p.add_argument('--no-actuator', action='store_true',
                   help='omit the default motor on gear A')
    p.add_argument('--motor-gear', type=float, default=0.02,
                   help='torque scale for the optional motor on gear A')
    p.add_argument('--mesh-only', action='store_true',
                   help='emit only primitive gear bodies, no option/default/floor')
    # common
    p.add_argument('--config', type=Path,
                   help='load full scene from YAML/JSON/TOML config')
    p.add_argument('--scale', type=float, default=None,
                   help='uniform scale factor applied to every length')
    p.add_argument('--output', '-o', type=Path,
                   help='write MJCF here (default: stdout)')

    args = p.parse_args(argv)

    if args.config:
        try:
            cfg = _load_config(args.config)
            xml = generate_gear_from_config(
                cfg, config_dir=args.config.parent.resolve(), scale=args.scale)
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
        xml = generate_gear_xml(
            teeth=args.teeth,
            gear_b_teeth=args.gear_b_teeth,
            module=args.module * scale,
            face_width=args.face_width * scale,
            pressure_angle=math.radians(args.pressure_angle_degrees),
            addendum_coeff=args.addendum_coeff,
            dedendum_coeff=args.dedendum_coeff,
            tooth_width_ratio=args.tooth_width_ratio,
            backlash=args.backlash * scale,
            center_distance=(args.center_distance * scale
                             if args.center_distance is not None else None),
            include_gear_b=not args.single,
            gear_b_phase=args.gear_b_phase,
            density=args.density,
            gear_b_density=args.gear_b_density,
            hub_radius=(args.hub_radius * scale
                        if args.hub_radius is not None else None),
            hub_width=(args.hub_width * scale
                       if args.hub_width is not None else None),
            include_floor=not args.no_floor,
            include_axles=not args.no_axles,
            floor_clearance=args.floor_clearance * scale,
            axle_radius=(args.axle_radius * scale
                         if args.axle_radius is not None else None),
            axle_margin=args.axle_margin * scale,
            mesh_only=args.mesh_only,
            include_actuator=not args.no_actuator,
            motor_gear=args.motor_gear,
            friction=(args.friction, 0.01, 0.001),
            condim=args.condim,
            timestep=args.timestep,
            joint_damping=args.joint_damping,
            joint_armature=args.joint_armature,
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
