# peg-in-hole + threading + gear MJCF generators

Three tools here, sharing the same idea:

1. **`mkpeg.py`** — peg-in-hole scenes with arbitrary 2D hole shapes (plus
   island-bearing connectors like USB-A, D-sub).
2. **`mkscrew.py`** — threaded bolt/nut pairs (ISO metric, single- or
   multi-start, cylindrical or hex exteriors).
3. **`mkgear.py`** — spur gears and gear pairs built from generated gear
   outlines with convex collision-prism geoms.

All three avoid V-HACD / CoACD / SDF: every contact surface is either a
MuJoCo primitive, a convex prism, or a convex polyhedron, so contact pairs are
handled natively by MuJoCo.

## Why

MuJoCo's narrow-phase collision is convex-only. For peg-in-hole tasks, the
hole is the concave side, so the usual options are:

1. **Convex decomposition (V-HACD, CoACD)** — often fills in exactly the
   concavity that defines the hole.
2. **SDF** — slow in practice, fiddly to tune, known issues.
3. **Hand-author the plate** as a ring of convex primitives. Works, but
   nobody wants to do it by hand for every new shape.

This utility automates option 3 for arbitrary 2D cross-sections: you describe
the insertion-plane shape (hole outer boundary + any islands) and it emits a
full MJCF with the plate decomposed into convex pieces, a matching peg (solid
or hollow with a pocket), sensible contact parameters, and a drop-in
`freejoint`. The outputs are ~dozens of mesh geoms; MuJoCo handles hundreds
of small convex meshes easily.

## Install

```bash
pip install numpy mujoco mapbox_earcut pyyaml
```

- `numpy`, `mujoco`: core.
- `mapbox_earcut`: required only for holes with **2+ islands** (polygon-with-
  N-holes triangulation). Single-island cases use a hand-rolled ear-clipper
  with no extra dependency.
- `pyyaml`: required only if you use `.yaml` config files. JSON and TOML
  configs work without it.

## Quick start

Drop the peg into a star-shaped hole:

```bash
python3 mkpeg.py --shape star -o star.xml
python3 -m mujoco.viewer --mjcf=star.xml
```

Drive from a declarative config:

```bash
python3 mkpeg.py --config configs/connectors/usbc.yaml -o usbc.xml
```

Drive from custom CSV polygons:

```bash
python3 mkpeg.py --hole-csv my_hole.csv -o out.xml
python3 mkpeg.py \
    --hole-csv my_hole.csv \
    --hole-island-csv my_island.csv \
    -o out.xml                      # hollow-shell peg
```

Generate a primitive-geometry spur gear pair:

```bash
python3 mkgear.py --config configs/gears/spur_pair.yaml -o gear_pair.xml
python3 -m mujoco.viewer --mjcf=gear_pair.xml
```

## Config schema (YAML / JSON / TOML)

```yaml
clearance:       0.0001            # radial clearance (hole - peg), meters
plate_size:      0.05              # square plate side, meters
plate_thickness: 0.005             # plate z-thickness, meters
peg_length:      0.012
peg_height:      0.015             # initial spawn height above plate top
include_peg:     true              # default true

hole:
  type: <primitive>
  ...primitive-specific params...
  center: [0.0, 0.0]               # optional, default origin
  rotate: 0                        # optional, radians

islands:                            # optional list
  - type: <primitive>
    ...
```

### Primitives

| Type           | Parameters                                       |
|----------------|--------------------------------------------------|
| `circle`       | `radius`, `n` (vertices, default 32)             |
| `rect`         | `width`, `height`                                |
| `stadium`      | `width`, `height`, `n_cap` (cap vertices, default 12) |
| `trapezoid`    | `bottom_width`, `top_width`, `height`            |
| `regular_ngon` | `n`, `circumradius`                              |
| `star`         | `n_points`, `outer_radius`, `inner_radius`       |
| `polygon`      | `vertices: [[x, y], ...]`                        |
| `from_csv`     | `path` (relative to the config file, or absolute) |

Every primitive accepts `center` and `rotate`. Islands use the same schema.

### Example: USB-A

```yaml
clearance:       0.0001
plate_size:      0.05
plate_thickness: 0.005
peg_length:      0.012
peg_height:      0.015

hole:
  type:   rect
  width:  0.012
  height: 0.0045

islands:
  - type:   rect           # the tongue
    width:  0.008
    height: 0.0015
    center: [0.0, 0.001]   # offset up for keying
```

The generator auto-produces a plug shell with a matching pocket (pocket
depth = `plate_thickness + clearance`; solid back above).

## Library API

```python
from mkpeg import generate_xml, generate_from_config

# Built-in named shape
xml = generate_xml(shape='star', hole_size=0.02)

# Custom polygon hole, auto-generate peg
xml = generate_xml(hole_poly=my_hole_np_array, peg_clearance=0.0005)

# Hole with N islands → hollow plug with N pockets
xml = generate_xml(
    hole_poly=hole_outer,
    hole_island_polys=[island_a, island_b, ...],
    peg_clearance=0.0001,
    plate_size=0.05,
    plate_thickness=0.005,
    peg_length=0.012,
)

# From a config dict (same schema as the YAML files)
xml = generate_from_config(cfg_dict)
```

## How it works

1. **Plate**: the square-plate-minus-hole annular region is decomposed into
   convex wedges. Star-shaped-from-centroid holes use a clean radial
   decomposition (few 4–5-sided pieces). Anything else falls back to
   bridging + ear-clipping (triangles; works for any simple polygon).
2. **Islands**: each island polygon is attached to the plate body as extra
   material (convex meshes if convex, else fan- or ear-clipped).
3. **Peg geometry**:
   - No islands → peg is a solid extrusion of the inward-offset hole
     polygon (`peg = hole - clearance`), decomposed as above.
   - With islands → peg is a **hollow shell**: a ring section on the entry
     side (thickness matches the island depth) plus a solid back block. The
     ring section is an annular polygon triangulated via mapbox_earcut;
     pockets surround each island with `clearance` on every edge.
4. **Clearance** is true inward polygon offset (miter joins). Every edge of
   the peg sits at exactly `clearance` perpendicular distance from the
   corresponding hole edge — constant around the perimeter regardless of
   shape. The built-in `--shape circle` is the exception: its peg is an
   analytic cylinder of radius `hole_size - clearance`, so the hole n-gon
   circumscribes `hole_size`. The gap is exactly `clearance` at the flats and
   grows to `clearance + hole_size * (1/cos(π/n) - 1)` at the n-gon corners.

## Contact parameters

The generated MJCF defaults to:

```xml
<option timestep="0.001" integrator="implicitfast" gravity="0 0 -9.81"/>
<default>
  <geom solref="0.002 1" solimp="0.99 0.999 0.001"
        condim="4" friction="1.0 0.01 0.001"/>
</default>
```

These work for most peg-in-hole scenarios. For stiffer contact (precision
insertion studies), switch to spring-damper form: `solref="-10000 -100"`.
For free-spinning torsional contact issues, bump to `condim=6`.

## Limitations

- **Star-shaped kernel requirement for the fast path**. Star-shaped-from-
  centroid holes and pegs use radial wedges / fan triangulation. Non-star-
  shaped shapes (L-slot, crescent, Pac-Man) automatically fall back to
  ear-clipping, which handles anything but produces more (smaller) pieces.
- **2+ islands require mapbox_earcut**. The naïve bridge-per-island approach
  creates coincident vertices at shared bridge targets; mapbox_earcut's
  ray-cast bridging doesn't. Without it, only single-island cases work.
- **Miter offset at very sharp corners**. Miter length at a convex corner is
  `clearance / sin(α/2)`. For a 36° star tip and 1 mm clearance, the peg tip
  moves ~3.24 mm inward. If that's unacceptable, use a smaller clearance.
- **Islands must be non-overlapping**. The generator doesn't check;
  overlapping islands produce undefined geometry.
- **Plate is a square**. Non-square plates are a 10-line edit (just change
  the outer polygon in `_decompose_plate_general` to your shape).

## Directory layout

```
mkpeg.py          # peg-in-hole generator (library + CLI)
mkscrew.py        # threaded bolt/nut generator (library + CLI)
mkgear.py         # primitive-geometry spur gear generator (library + CLI)
configs/
  connectors/               # declarative YAML configs for peg-in-hole
  screws/                   # declarative YAML configs for bolt/nut
  gears/                    # declarative YAML configs for primitive gears
examples/
  build_connectors.py       # Python driver, same content as YAML but programmatic
  build_dsub.py             # example of scripted pin placement
  *.csv                     # raw polygon data for CSV-driven inputs
  peg_*.xml                 # generated MJCF outputs (regenerate any time)
```

---

# Threading: `mkscrew.py`

Same idea as `mkpeg.py`, but the discretization axis is the thread helix
instead of the 2D hole boundary. Each short angular slice of a thread (Δθ
of a few degrees) is a twisted triangular prism — still convex — so the
bolt = **core cylinder + N helical thread wedges** and the nut = **annular
shell + N inward-facing thread wedges**. All contact pairs stay convex-vs-
convex; no SDF, no V-HACD.

## Quick start

Two scene modes are supported:

**Bolt-driven** (default): nut welded to world, bolt free and positioned above,
tip-down. Spin the bolt CW about z — it threads down into the fixed nut.
```bash
python3 mkscrew.py --config configs/screws/m10.yaml -o m10.xml
python3 -m mujoco.viewer --mjcf=m10.xml
```

**Nut-driven**: bolt welded with head at the floor and tip pointing up, nut
free and positioned at the top of the bolt. Spin the nut CW about z — it
threads down onto the fixed bolt (natural "tightening a nut onto a stud"
setup). Enable with `nut_driven: true` in the YAML or `--nut-driven` on CLI.
```bash
python3 mkscrew.py --config configs/screws/m10.yaml --nut-driven -o m10.xml
```
In this mode the bolt's body carries `quat="0 0 1 0"` (180° about y) so its
local +z (head direction) flips to world −z; the helix handedness is
preserved by the rotation, so it's still a right-handed thread in world.

## Config schema

```yaml
major_diameter:  0.010      # D (e.g., M10 → 10 mm)
pitch:           0.0015     # P (axial distance per turn per start)
thread_length:   0.020      # axial length of bolt threads
n_starts:        1          # multi-start threads: lead = n_starts × pitch
segments_per_turn: 24       # angular segments per helix revolution
thread_form:     iso        # ISO 60° (only form in v1)
handedness:      right      # 'right' or 'left'
clearance:       0.00010    # flank-normal gap between bolt and nut threads

bolt:
  shank_length:  0.008      # unthreaded shank above the thread
  head_height:   0.0064
  head_across_flats: 0.016  # for head_shape: hex
  head_shape:    hex        # 'cylinder' or 'hex'

nut:
  height:        0.008
  across_flats:  0.016      # for shell_shape: hex
  shell_shape:   hex        # 'cylinder' or 'hex'
  shell_sectors: 16         # chord count for cylindrical nut bore
```

Every length scales uniformly with `scale` (and `--scale` on the CLI).

## Library API

```python
from mkscrew import generate_screw_xml, generate_screw_from_config

# Parametric call
xml = generate_screw_xml(
    major_diameter=0.010, pitch=0.0015, thread_length=0.020,
    clearance=1e-4, bolt_head_shape='hex', nut_shell_shape='hex',
)

# Config-driven
import yaml
cfg = yaml.safe_load(open('configs/screws/m10.yaml'))
xml = generate_screw_from_config(cfg)
```

## How it works

1. **Thread profile** (meridian plane): ISO 60°-angle truncated trapezoid,
   parameterized by pitch P. External thread: base at minor radius r_min,
   crest at major radius r_max. Internal thread: same shape, inverted
   radially (base at r_max, tip at r_min).
2. **Clearance**: reduces thread crest radius by `clearance/2` on both bolt
   and nut, preserving the 60° flank angle. Resulting flank-to-flank
   perpendicular gap ≈ clearance.
3. **Axial offset for interlock**: the nut's thread is axially shifted by
   `−7P/16` from the bolt's so the nut's falling flank coincides with the
   bolt's rising flank (zero-clearance case). This is what makes the two
   threads mesh rather than collide.
4. **Wedge construction**: for each angular segment [θᵢ, θᵢ₊₁] of each
   turn of each start, the thread profile is swept through Δθ with the
   appropriate helical axial advance (`Δθ · lead / 2π`). Result: an 8-vertex
   convex polyhedron, emitted as `<mesh vertex="..."/>` so MuJoCo convex-
   hulls it.
5. **Bolt assembly**: one analytical `<geom type="cylinder">` at r_min (the
   core) plus the thread wedges. Hex head is a 6-gon convex prism; cylinder
   head is another primitive cylinder.
6. **Nut assembly**: annular shell decomposed into sectors (cylindrical
   nut) or via ear-clipping (hex nut — hex-minus-bore). Thread wedges point
   inward. Shell inner radius is inflated so chord midpoints sit exactly
   at `r_max + clearance/2` — ensures the flat chord doesn't intrude into
   the thread engagement zone.

## Clearance and friction tuning

Threading efficiency (Δz per revolution relative to nominal P) depends on
clearance, spin rate, and friction:

- **Tight clearance** (c / P < 5%) produces many simultaneous contacts and
  can cause the contact solver to under-resolve, dragging efficiency toward
  ~50%. Loosen to 7–10% of pitch for clean threading.
- **Slow rotation** makes efficiency dip (solver play at low rates — the
  bolt "slips through" the flank-normal clearance). Faster spin pins the
  bolt to a consistent flank contact.
- **Segments per turn**: too coarse (≤12) produces visible chord error and
  catching at segment seams. 24 is a good default; bump to 32–48 for tight
  simulations.

For M10 at clearance=100 µm, segments_per_turn=24, spin=−15 rad/s: the
generator reliably produces ≥95% threading efficiency (Δz/rev vs. nominal
pitch) on the provided example configs.

## Verification

```python
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path('m10.xml')
d = mujoco.MjData(m)
bolt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'bolt')
qvel0 = m.jnt_dofadr[m.body_jntadr[bolt]]
z0 = d.qpos[m.jnt_qposadr[m.body_jntadr[bolt]] + 2]
for _ in range(2000):
    d.qvel[qvel0 + 5] = -15.0      # CW about z → right-handed threads in
    d.qvel[qvel0 + 0:qvel0 + 2] = 0
    d.qvel[qvel0 + 3:qvel0 + 5] = 0
    mujoco.mj_step(m, d)
print('advance:', d.qpos[m.jnt_qposadr[m.body_jntadr[bolt]] + 2] - z0)
```

## Limitations

- ISO metric only in v1. Acme/trapezoidal, UTS imperial, buttress are
  profile-function plug-ins (edit `_iso_thread_profile_external` to add).
- Mesh count scales with (thread_length / pitch) × segments_per_turn. M10
  with 20 mm thread and 24 segs/turn → ~500 meshes; MuJoCo handles it fine
  but loading time grows linearly.
- No support yet for left-hand + right-hand mixed assemblies, tapered
  threads (NPT), or multi-material bolts (the whole bolt is one body).

---

# Gears: `mkgear.py`

`mkgear.py` makes a contact-driven spur gear scene from a generated involute-
style tooth outline. Each gear has one clean visual mesh, while collision is
decomposed into hidden convex triangular-prism mesh geoms that match the same
outline. There is no colliding root disk filling the valleys and no rectangular
tooth blocks.

## Quick start

```bash
python3 mkgear.py -o gear_pair.xml
python3 mkgear.py --single --teeth 24 -o gear.xml
python3 mkgear.py --config configs/gears/spur_pair.yaml -o gear_pair.xml
```

Included configs:

- `configs/gears/spur_pair.yaml` — default 18:24 pair
- `configs/gears/equal_20t.yaml` — equal 20:20 pair
- `configs/gears/reduction_12_36.yaml` — 3:1 reduction
- `configs/gears/overdrive_32_16.yaml` — 2:1 overdrive
- `configs/gears/fine_pitch_32_48.yaml` — fine-pitch higher tooth counts
- `configs/gears/compact_coarse_10_20.yaml` — compact coarse pair
- `configs/gears/single_24t.yaml` — single centered gear

Viewer groups:

- Group `2`: clean visual gear meshes.
- Group `3`: translucent cyan collision wedges. This group is hidden by
  MuJoCo's default viewer options, so press `3` to show/hide collision.

## Config schema

```yaml
module:                 0.003     # pitch diameter / teeth, meters
face_width:             0.008
pressure_angle_degrees: 20.0
addendum_coeff:         1.00
dedendum_coeff:         1.25
tooth_width_ratio:      0.48      # tooth thickness at pitch circle / circular pitch
backlash:               0.00015   # extra center spacing

gear_a:
  teeth: 18

gear_b:
  include: true
  teeth: 24

support:
  include_axles: true

actuator:
  include: true              # motor on gear_a_hinge
  gear: 0.02
```

## Library API

```python
from mkgear import generate_gear_xml, generate_gear_from_config

xml = generate_gear_xml(teeth=18, gear_b_teeth=24, module=0.003)
```

## Limitations

- The tooth outline is an involute-style approximation with an unfilleted
  root, not a manufacturing-grade CAD gear.
- Gear centers are hinge-constrained in the generated scene. Torque transfer
  comes from contact between convex tooth/profile wedges; no equality gear
  constraint is inserted.
- Collision mesh count scales with tooth count and outline sampling. The
  default example is still small enough to load and step quickly.
