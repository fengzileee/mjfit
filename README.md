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
| `bezier_polygon` | convex control `vertices`, `corner_rounding` (scalar or one per corner, 0–0.5), `max_chord_error` (meters, default 0.0001) |
| `from_csv`     | `path` (relative to the config file, or absolute) |

Every primitive accepts `center` and `rotate`. Islands use the same schema.

### Convex insertion, independent handle, and outer plate

`bezier_polygon` replaces each rounded control corner with a quadratic Bezier
curve. A rounding fraction of `0` leaves that corner sharp; `0.5` joins adjacent
curves at their common edge midpoint. Mixed fractions combine straight and curved
sections. The curves are tessellated into convex native MuJoCo prism meshes with
a conservative Euclidean chord-error bound. Fractions correspond to the supplied
vertex order, including clockwise input; output is counterclockwise.

Configs accept either `hole` (derive the peg by inward clearance) or `peg`
(derive the aperture by outward clearance). Optional `handle` and `plate` use
the same primitive schema. The handle must contain the insertion outline; an
explicit convex outer plate must contain the aperture and handle. No outline
is automatically resized to make these constraints pass.

```bash
python3 mkpeg.py --config configs/connectors/convex_bezier_handle.yaml -o curved.xml
```

Additional settings for this example are:

- `peg_length` is the lower insertion-section height A; `handle_height` is B.
- `peg_mass_kg` is the **total** mass of A plus B. Collision-geom masses share
  one material density, weighted by each section's physical volume. This overrides
  `peg_density` and stays fixed under `scale`. Omit it for the legacy density rule.
- `plate_min_rim_m: 0.005` requires at least 5 mm of perpendicular clearance
  from every outer-plate edge to both the aperture and handle. It has no maximum
  rim and requires an explicit `plate`. The default is zero.
- `bottom_stop_thickness: 0.002` adds a 2 mm floor below the hole following
  the exact outer-plate outline. Default zero retains the historical open hole.
  `include_floor: false` omits the unrelated infinite world ground plane.
- `center_shapes: false` preserves every supplied coordinate and disables the
  default lateral peg spawn offset. The default `true` applies one shared source
  translation to the hole, peg, handle, and explicit plate, preserving their
  relative placement. It never independently recenters Y or Z.

`scale` multiplies all lengths, including clearance, minimum rim, and chord-error
tolerance. Pass already-resolved physical polygons with `scale=1` to preserve
an experiment's physical settings. The new explicit-geometry paths serialize
coordinates with 17 significant digits; legacy calls retain their output format.

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

# Export an experiment's exact resolved aperture, containing Y, and outer Z.
# X is derived from the aperture with the same physical inward clearance.
xml = generate_xml(
    hole_poly=aperture, handle_poly=Y, plate_poly=Z,
    peg_clearance=0.001, peg_length=A, handle_height=B,
    peg_mass_kg=0.55, plate_min_rim_m=0.005,
    bottom_stop_thickness=0.002, center_shapes=False, include_floor=False,
)
```

CSV equivalents are `--handle-csv`, `--plate-csv`, `--peg-mass-kg`,
`--plate-min-rim-m`, `--bottom-stop-thickness`, and `--preserve-origin`.
The `bezier_polygon(vertices, corner_rounding, max_chord_error)` library helper
also returns the tessellated outline directly, without translating or scaling it.

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
- **Explicit outer plates must be convex**. Rotated/asymmetric rectangles and
  convex Bezier outlines are supported; nonconvex outer plates are rejected.

## Directory layout

The convex/Bezier regression tests use `pytest` in addition to the runtime
dependencies. Run `python -m pytest tests/test_mkpeg_convex.py` to check native
mesh loading, containment/clearance, shared frames, total mass, OBJ export,
clockwise corner parameters, and unchanged legacy output.

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
thread_form:     iso        # iso, trapezoidal, acme, square, buttress, wire, custom
thread_params: {}           # overrides of the form's defaults (see Thread forms)
handedness:      right      # 'right' or 'left'
clearance:       0.00010    # gap between facing flanks (along their normal)
seam_relief:     0.0        # dip teeth at wedge seams (m); see How it works
thread_chamfer:  0.0        # chamfer the thread ends over this axial length (m)

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

## Thread forms

A thread is a tooth profile swept along a helix. Each part's tooth is any
simple polygon in the meridian plane:
- **Convex teeth** (V, trapezoid, square, buttress, round wire, dovetail)
  are swept as they are.
- **Non-convex teeth** (hooks, T-slots, barbs) are first cut into convex
  pieces, and each piece is swept as its own wedges.

`thread_form` picks one, and `thread_params` overrides its defaults. On the CLI, repeat `--thread-param KEY=VALUE`, where
VALUE is a Python literal (`--thread-param "flank_angles=[30,3]"`).

| `thread_form` | Profile | Defaults (P = pitch) |
|---|---|---|
| `iso` | ISO 68-1 metric (and UTS): 60° V with flats | depth 5√3/16·P, crest flat P/8 |
| `trapezoidal` | ISO 2904 Tr: 30° | depth P/2, crest flat 0.366 P |
| `acme` | ACME: 29° | depth P/2, crest flat 0.3707 P |
| `square` | Square: 0° flanks | depth P/2, crest flat P/2 |
| `buttress` | ANSI 7°/45° | depth 0.6 P, crest flat 0.1631 P |
| `wire` | Round wire wound on each part (FurnitureBench) | wire Ø P/√3, tight |
| `custom` | Your own teeth, convex or hooked | — |

- **Trapezoid family** (`iso` to `buttress`). `thread_params` can set:
  - `flank_angles`: the bolt tooth's lower and upper flank, in degrees from
    the radial plane. The upper flank faces the bolt head, so a buttress's
    steep 7° flank carries a bolt in tension. Negative angles undercut the
    flanks, giving a dovetail: both teeth widen away from their base and stay
    convex.
  - `depth` and `crest_flat`, both in pitches. The root gap is whatever is
    left of the pitch.
- **`wire`**. `thread_params` can set:
  - `wire_diameter`;
  - `nut_wire_offset`: how far out the nut's wire centre sits from the
    bolt's;
  - `bore_diameter`: the nut root;
  - `wire_segments`: sides of the polygon inscribed in the wire's section,
    16 by default.

  The bolt's wire is centred on its core. The nut's wire sits half a pitch
  along. The default is the tight form: wire Ø P/√3, at the offset where the
  wires touch at zero clearance.
- **`custom`**: give `bolt_tooth` and `nut_tooth` as lists of (r, u) points
  in metres, in one shared phase: u is the axial position along the helix,
  mod P. Also give `core_diameter` and `root_diameter`. Each tooth is
  clipped at its base line and must be a simple polygon.
  - **Non-convex teeth, clearance:** every edge moves back c/2 at once, with
    mitred corners. At a reflex corner that only leaves more clear. The
    offset is rejected if it would make the tooth cross itself.
  - **Non-convex teeth, pieces:** the tooth is then cut into convex pieces,
    by ear-clipping and merging triangles back while they stay convex
    (Hertel–Mehlhorn).
  - **Chord and twist corrections** move the whole tooth's edges before the
    cut, so they come out right at the reflex corners. The cuts lie inside
    the tooth and never move.

Clearance `c` works the same way for every form: every exposed edge of both
teeth moves back c/2 along its normal, and the nut root moves out by c/2.
Facing surfaces are then c apart; for `iso` that is exactly the standard's
rule. The tests check that the wedges of every form stay inside its exact
swept teeth, for multi-start and left-hand threads too, and that every form
starts interlocked in both scene layouts.

How far this stretches: `benchmark/thread/wacky.py` has nine designs, and
`thread_wacky.mp4` shows each screwed down under a nut runner:
- a sawtooth, a knife-edge V and an auger;
- a six-start thread steep enough to back-drive;
- left-hand double- and triple-start threads;
- a loose coil spring, and custom pentagon-and-dart teeth;
- a fine M10×0.5.

`benchmark/thread/wacky.py` also has four undercut designs in
`thread_undercut.mp4`:
- a 15° dovetail;
- interlocking L-hooks;
- T's in inverted-T slots;
- barbed arrowheads, where the nut fills the pockets under the barbs.

They are chamfered 45° at the ends (`thread_chamfer`, item 10 of How it
works). The tests check that their wedges stay inside the exact teeth and cover
them.

Not expressible yet:
- tapered threads;
- variable pitch;
- non-round cores.

`configs/screws/furniture_bench_leg.yaml` is the FurnitureBench square-table
leg in its tabletop hole, measured from the FurnitureBench meshes:
- **Thread:** a round-wire thread, pitch 9.375 mm, single start,
  right-handed, with a Ø4.36 mm wire on both parts.
- **Leg:** a Ø20 mm stud with the wire centred on its surface.
- **Hole:** a Ø25 mm bore, with the wire centred 3.1 mm further out than the
  stud's.
- **Fit:** loose, as 3D-printed parts are: about ±1.6 mm of axial play at
  zero clearance.

## Library API

```python
from mkscrew import generate_screw_xml, generate_screw_from_config

# Parametric call
xml = generate_screw_xml(
    major_diameter=0.010, pitch=0.0015, thread_length=0.020,
    clearance=1e-4, bolt_head_shape='hex', nut_shell_shape='hex',
)
# Another form, with overrides
xml = generate_screw_xml(thread_form='buttress',
                         thread_params={'flank_angles': (30, 3), 'depth': 0.75})

# Config-driven
import yaml
cfg = yaml.safe_load(open('configs/screws/m10.yaml'))
xml = generate_screw_from_config(cfg)
```

## How it works

1. **Thread profiles** (meridian plane): a convex tooth per part, in (r, u)
   with u the axial position along the helix (see Thread forms). For ISO
   68-1 the bolt tooth has a P/8 crest at the major radius and leaves a P/4
   root at the minor radius. The nut tooth is the space between basic bolt
   teeth, not a mirror of the bolt tooth: its tip at the minor radius is P/4
   wide and its base at the major radius 7P/8.
2. **Clearance** `c`: every exposed edge of both teeth moves back c/2 along
   its normal (the flanks, the bolt crest and the nut tip), and the nut root
   moves out c/2. With the nut on the pitch grid (see 3), facing flanks are
   exactly `c` apart on both sides, the bolt crest clears the nut root by
   `c`, and the nut tip clears the bolt core by c/2. For ISO, axial play is
   ±c/cos 30° = ±2c/√3 and radial play is c/2.
3. **Thread phase**: both teeth are defined in one phase, the bolt's, with
   the nut tooth in the bolt's gaps. For ISO that shifts the nut's thread
   `−7P/16`. Frames a whole number of pitches apart then interlock with the
   clearance centred, and the bolt-driven layout puts the bolt on that grid.
   The nut-driven layout flips the bolt (a half turn about y), which mirrors
   its tooth in u:
   - a symmetric tooth only moves the nut's phase to bolt origin −
     lead/2 − 2u_c, mod P, where u_c is the tooth centre (for single-start
     ISO, +P/4);
   - an asymmetric tooth (buttress) gets a mirrored nut tooth, as a nut
     turned over with its bolt, at bolt origin − lead/2.
4. **Wedge construction**: for each angular segment [θᵢ, θᵢ₊₁] of each
   turn of each start, the thread profile is swept through Δθ with the
   appropriate helical axial advance (`Δθ · lead / 2π`). Result: an 8-vertex
   convex polyhedron, emitted as `<mesh vertex="..."/>` so MuJoCo convex-
   hulls it. Each wedge is clipped exactly to the thread length, so no tooth
   sticks out past the bolt tip or the nut faces. The turns start from the
   one below turn 0, so the partial first turn at each z = 0 face (the
   bolt's tip) is there too.
5. **Discretization is one-sided**: a wedge's faces are chords of the
   helical surfaces. On the bolt they fall inside the true thread. On the
   nut they would bulge toward the bolt in two ways, and both are removed:
   - the arc sag r(1 − cos(Δθ/2)) (36 µm at M10, 24 segments), by pushing
     the wedge out radially by 1/cos(Δθ/2);
   - the helicoid twist of each flank quad, lead·Δθ·(r₁ − r₂)/(4π(r₁ + r₂))
     axially (2.6 µm), by moving both flanks back by that amount.

   On a bolt flank that widens toward the core (ISO, trapezoidal) the chord
   sag outweighs the twist. On steep flanks (square, the sides of a round
   wire) it doesn't, and those edges move back too. These formulas are
   first order, so the generator then measures one wedge's hull facets
   (edges and interiors, zooming in on each facet's peak) against the exact
   tooth and moves back any edge that still sticks out (by a few nm at
   most). All wedges are congruent, so one is enough.

   Fewer segments per turn therefore only add play; they never add
   interference. `tests/test_mkscrew.py` checks the wedges against the exact
   helical parts, for every thread form.
6. **Bolt assembly**: one analytical `<geom type="cylinder">` at r_min (the
   core) plus the thread wedges. Hex head is a 6-gon convex prism; cylinder
   head is another primitive cylinder.
7. **Nut assembly**: annular shell decomposed into sectors (cylindrical
   nut) or via ear-clipping (hex nut — hex-minus-bore). Both bore polygons
   are circumscribed so their chords sit at or outside `r_max + c/2`. Thread
   wedges point inward.
8. **Collisions**: every bolt geom can touch every nut geom, so a nut tooth
   tip under radial load stops on the bolt core as it should.
9. **Seam relief** (`seam_relief`, metres; off by default): MuJoCo collides
   every pair of convex pieces separately. A tooth sliding across the seam
   into the next wedge first overlaps it in a thin sliver at its end face.
   The shortest way out is back through that face, so the contact pushes
   around the axis and stops a spinning nut dead (an "internal edge").
   `seam_relief` dips the teeth by that depth at every seam, ramping back up
   at about 1:20, so seams become gentle ramps. It only removes material (the
   internal-thread corrections cover the new chords), and it slightly reduces
   play error, because full-height chords get shorter. About 5 µm suits
   MuJoCo's default contact softness. It raises a frictionless M10 nut's
   spin-down from 69 / 41 / 30 % of the exact descent to 92 / 87 / 71 % at
   24 / 48 / 96 segments per turn (see `benchmark/thread/README.md`).
10. **Chamfer** (`thread_chamfer`, metres; off by default): machined bolts
    have chamfered ends, and so can these.
    - **Bolt:** a cone cuts the thread back from the core radius at the end
      face to the crest `thread_chamfer` in. That is 45° when
      `thread_chamfer` is the thread depth. It applies at the tip, and at
      the far end too on a bare threaded rod (no shank or head).
    - **Nut:** countersunk the same way at both faces, from its bore to its
      tooth tips.
    - **What it's for:** a lead-in for the nut. It also removes the thin
      blade that an undercut tooth (a dovetail) leaves when the thread is
      cut off flat, because the crest runs on past the root.
    - **Only removes material:** a wedge can't be cut by a curved cone
      exactly. Each bolt wedge is cut by the plane through the cone's surface
      lines at the wedge's two edges, which lies inside the cone. Each nut
      wedge is cut by the plane tangent to its countersink, outside which
      the countersink never reaches. On the CLI: `--thread-chamfer`
      (and `--seam-relief`).

## Clearance and friction tuning

Threading efficiency (Δz per revolution relative to P) for the verification
snippet below: bolt spun at −15 rad/s for 2 s, M10, 24 segments per turn.

| friction, condim | c = 100 µm | c = 20 µm |
|---|---|---|
| 1.0, 4 (generator default) | 75 % | 34 % |
| 0.3, 3 (example configs) | 94 % | 89 % |
| 0.1, 3 | 98 % | 97 % |
| 0 (frictionless), 1 | 98 % | 98 % |

The loss comes from friction, not from the geometry. Driving the rotation by
velocity against a self-locking thread (μ above tan of the 3° lead angle)
pushes the soft contacts about 40 µm deep, until the flanks slip. For
spun-in threading, use friction 0.1–0.3 with condim 3.

- **Slow rotation** makes efficiency dip (solver play at low rates — the
  bolt "slips through" the flank-normal clearance). Faster spin pins the
  bolt to a consistent flank contact.
- **Segments per turn** trade speed for play: the discretization only
  loosens the nut, by up to r(1/cos(π/n) − 1) radially plus the twist
  allowance above. 24 is a good default; use 48–96 for clearances of tens
  of µm.

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

- Teeth must be simple polygons in the meridian plane. Non-convex ones are
  cut into convex pieces, one set of wedges each, so a hooked tooth costs
  two or three times the geoms. A round wire's section is an inscribed
  polygon: it only rounds the wire down, by up to ρ(1 − cos(π/n)).
- The benchmark (`benchmark/thread/`) takes any form, and so does its
  analytic SDF plugin. `benchmark/thread/forms.py` has five M10-sized forms
  and the FurnitureBench leg; see "Thread forms" in its README.
- Mesh count scales with (thread_length / pitch) × segments_per_turn. M10
  with 20 mm thread and 24 segs/turn → ~500 meshes; MuJoCo handles it fine
  but loading time grows linearly.
- No support yet for tapered threads (NPT), variable pitch, left-hand +
  right-hand mixed assemblies, or multi-material bolts (the whole bolt is
  one body).

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
