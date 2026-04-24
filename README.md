# peg-in-hole MJCF generator

Generate MuJoCo peg-in-hole scenes with arbitrary 2D hole shapes — including
concave holes, holes with one or more solid "islands" suspended inside them
(USB-A tongue, D-sub pins, XLR contacts), and plugs that wrap those islands
as true hollow shells.

The point: you can simulate connector insertion in MuJoCo without
V-HACD / CoACD / SDF. Every geom the generator emits is a convex prism, so
all contact pairs are convex-vs-convex, handled natively.

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
python3 gen_peg_in_hole.py --shape star -o star.xml
python3 -m mujoco.viewer --mjcf=star.xml
```

Drive from a declarative config:

```bash
python3 gen_peg_in_hole.py --config examples/connectors/usbc.yaml -o usbc.xml
```

Drive from custom CSV polygons:

```bash
python3 gen_peg_in_hole.py --hole-csv my_hole.csv -o out.xml
python3 gen_peg_in_hole.py \
    --hole-csv my_hole.csv \
    --hole-island-csv my_island.csv \
    -o out.xml                      # hollow-shell peg
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
from gen_peg_in_hole import generate_xml, generate_from_config

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
   shape.

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
gen_peg_in_hole.py          # the generator (library + CLI)
examples/
  connectors/               # declarative YAML configs
    circle.yaml  triangle.yaml  square.yaml
    pentagon.yaml  hexagon.yaml  octagon.yaml
    star.yaml  slot.yaml
    usb_a.yaml  usbc.yaml  hdmi.yaml
    xlr3.yaml  rj45.yaml  db9.yaml
  build_connectors.py       # Python driver, same content as YAML but programmatic
  build_dsub.py             # example of scripted pin placement
  *.csv                     # raw polygon data for CSV-driven inputs
  peg_*.xml                 # generated MJCF outputs (regenerate any time)
```
