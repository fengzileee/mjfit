# Collision-representation benchmark: peg-in-hole

Compares **exact convex-prism decomposition** (`mkpeg.py`, "ours") against the
usual ways of giving MuJoCo a concave hole — **convex decomposition** (CoACD,
V-HACD) and **mesh SDF** — on:

1. **Physical accuracy against exact geometry**: passive probes whose correct
   answer is computed from the analytic shapes, not from any simulator.
2. **Speed**: CPU (one environment, one thread) and GPU (MuJoCo Warp, 256
   worlds).

Everything here is **passive**: no actuators or controllers. The peg moves only
under gravity, a constant applied wrench (`xfrc_applied`), or kinematic
placement, and is constrained only by plain joints. The actuated, randomized
insertion task comes later.

Figures and summary tables: [`report/`](report/). Raw CSVs are written to
`results/` (gitignored); see *Running*.

**Bolt/nut (M10×1.5 thread):** the same comparison on a threaded fastener,
with an analytic-SDF baseline. It has its own README:
[`thread/README.md`](thread/README.md).

## Findings (Ø16 mm hole, 15 mm plate, MuJoCo 3.14)

Clearance is radial: c = hole radius − peg radius. Whitney's clearance ratio
(D − d)/D is given alongside: c = 100 µm is 0.0125, and c = 20 µm is 0.0025,
close to a precision sliding fit (H7/g6 at Ø16 is ~6–35 µm diametral).

**Headline, at c = 20 µm**
([`report/pareto_c20.png`](report/pareto_c20.png)):
- every CoACD and V-HACD setting fills the clearance;
- SDF is either filled or off by 23–79 µm;
- ours stays within 3 µm on the circle even as a heptagon (≤ 0.8 µm from
  n = 20 sides), and within 0.2 µm on the square.

The metrics, with their provenance:

| metric | what it measures | follows |
|---|---|---|
| **wall offset** | where each representation's hole wall is, by ray casting, with no solver | CoACD / V-HACD surface-distance concavity |
| **S0 contact onset** | the true gap at which the collision detector first reports contact | LOCC collision-detection accuracy |
| **S1 drop** | pass or block outcome vs the exact geometry | — |
| **S2 / S3 / S4 play** | settled lateral, tilt, and yaw limits vs analytic oracles | analytic-solution tests of Le Lidec et al. |
| **contact sd** | true signed distance whenever the sim reports contact: false-contact gap (> 0) and penetration (< 0) | `min-|sd|` of Son & Kim (LOCC); max interpenetration of IndustReal |
| **cost** | CPU µs per step (1 env, 1 thread), GPU µs per env-step (256 worlds), compiled model memory (MB) | speed-accuracy Pareto fronts over each method's settings, after Erez, Tassa & Todorov |

### CPU, circle: full Pareto fronts

Figures: [`report/pareto_cpu_circle.png`](report/pareto_cpu_circle.png) (time)
and [`report/pareto_memory_circle.png`](report/pareto_memory_circle.png)
(memory). x is cost, y is the worst settled error over S2–S3, both on log
axes; lower-left is better. Each line is one method's Pareto front over all
its settings, i.e. its granularity knob × MULTICCD. The baseline knobs are
pushed until they saturate ([`configs/circle_cpu.yaml`](configs/circle_cpu.yaml)).

| method | c = 100 µm (ratio 0.0125) | c = 20 µm (ratio 0.0025) |
|---|---|---|
| **ours** (n = 7 … 281 sides) | 11.3 µm at 12 µs / 0.04 MB → 0.29 µm at 169 µs / 0.73 MB | 2.6 µm at 12 µs → 0.17 µm at 223 µs |
| **CoACD** (threshold 0.2 … 0.005) | fills at 14 µs → 40 µm at 572 µs (0.01) → 20 µm at 1.1 ms (0.0075) → **9.7 µm at 2.0 ms / 18 MB (0.005: 4168 hulls, 8 min build)** | fills at every threshold except 0.005 (10 µm at 2.0 ms) |
| **V-HACD** (up to 4096 hulls, 120M voxels) | fills at 34 µs → 68 µm at 0.56 ms → **27 µm at 1.8 ms / 8.5 MB**; the floor is set by voxel size | fills at every setting |
| **SDF** (octree depth 5 … 10) | **52 µm at 0.41 ms (depth 6)**; depths 7–10 (up to 2.9 GB) are dominated | fills, or 23–60 µm off |

- **Ours' front lies below every baseline's at every cost.**
  - **At c = 100 µm:** its 10-gon (5.5 µm at 14 µs, 0.05 MB) already beats
    the best any baseline reaches (CoACD 0.005: 9.7 µm at 2.0 ms), for 145×
    less time. Its most accurate setting (0.29 µm at ≤ 223 µs) is 33×
    (CoACD), 93× (V-HACD) and 180× (SDF) more accurate than the best of
    each, which all cost more time.
  - **At c = 20 µm:** even the heptagon (2.6 µm) beats every baseline.
- **SDF's remaining error is not geometry.** At depth 10 the octree wall is
  within 7 µm (wall offset), yet the tilt error stays at 50–100 µm. That is
  the sampled gradient-descent contact search. More initial points and
  iterations (100 / 30) cost 3× the time without helping.

### GPU (MuJoCo Warp), circle: Pareto fronts

Figure: [`report/pareto_gpu_circle.png`](report/pareto_gpu_circle.png). The
same settings and probes (S2, S3) run on MuJoCo Warp, so both axes are
Warp's: x is GPU µs per env-step at 256 worlds (wall scenario), y is the
error Warp itself produces.

| method | c = 100 µm | c = 20 µm |
|---|---|---|
| **ours** | 16 µm at 1.6 µs → 5.6 µm at 2.4 µs (n = 89) → **3.2 µm at 6.9 µs** (n = 281) | 7.6 µm at 1.6 µs → **2.2 µm at 2.4 µs** (n = 89) |
| **CoACD** | fills at 1.3 µs → 35 µm at 6.3 µs (0.01) → **9.3 µm at 16 µs** (0.005) | fills, except 0.005: 9.5 µm at 16 µs |
| **V-HACD** | fills at 1.4 µs → 68 µm at 4.0 µs → **27 µm at 17 µs** (4096 hulls, 120M voxels) | fills at every setting |
| **SDF** | **181–344 µm at 2.2–2.5 µs**, at every depth 5–10 | fills (depth 5), or 130–450 µm |

- **Ours' front is again below every baseline's.** Even ours' coarsest
  polygon (n = 7, 18 µm) beats every V-HACD and SDF setting at c = 100 µm.
  The best CoACD (9.3 µm) needs 16 µs, 10× the time of ours at the same
  error.
- **Batching flattens the cost axis.** The fronts span only 1.3–17 µs per
  env-step (CPU fronts: 12 µs–2 ms). Ours' finest polygon costs 2.5–4× its
  coarsest. Hull-heavy CoACD and V-HACD pay 12×, because each hull near the
  peg is a broadphase pair in every world.
- **Warp adds its own error floor of about 3–5 µm.** Ours' finest settings
  reach 3–5 µm on Warp, against 0.2–1 µm on CPU on identical scenes. Warp's
  tilt probe also does not settle: median 2.9 µm jitter, where CPU's is
  usually zero.
  - The model is the same: margin, solref/solimp and solver settings match
    after `put_model`.
  - Warp floors the solver tolerance at 1e-6 (CPU: 1e-8), but CPU MuJoCo at
    1e-6 gives the same answer as at 1e-8, so the floor comes from Warp's
    float32 solver and contact generation, not the tolerance.
  - Where geometry error dominates, Warp tracks CPU. With MULTICCD off,
    CoACD and V-HACD errors agree within 6 µm at every setting that keeps the
    clearance open. With MULTICCD on, two CoACD settings that go wrong on CPU
    (0.01 and 0.005 at c = 100 µm: 191 and 100 µm) stay at 108 and 10 µm on
    Warp.
- **Warp's SDF collision misses contacts.** At c = 100 µm the tilted peg
  sinks 180–310 µm past the exact limit at every depth; on CPU, the same
  scenes stay within 120 µm. Finer octrees do not help; depth 10 is the worst
  (344 µm). At Warp's settled tilt pose, Warp reports no contact, while CPU
  MuJoCo at the same pose finds about 110 µm of penetration.

### Renderings: Pareto points under passive disturbances

`MUJOCO_GL=egl python -m benchmark.render` writes six videos to `results/render/`:
{drop, push, tilt} × c = {100, 20} µm. Each video tiles eight settings, a
coarse and a fine point from each method's CPU front at c = 100 µm:

- ours: n = 10 and n = 45;
- CoACD: 0.01 and 0.005;
- V-HACD: 1024 hulls at 20M voxels, and 4096 hulls at 120M voxels;
- SDF: depth 5 and depth 6.

Each tile has two parts:

- **Render:** the plate is cut at y = 0, so it shows the collision geometry
  the simulator actually uses (every hull in its own shade, or the SDF
  octree's zero level set by marching cubes). The peg is checkered and
  translucent. Black arrows are the applied load: the weight, the push at
  the centre, or the torque drawn as its equivalent couple at the two plate
  faces. Magenta arrows are the contact forces the plate applies to the peg,
  drawn at their contact points.
- **Magnified trace:** the simulated motion against the exact answer, with
  live readouts of the true penetration.

The behaviors are passive versions of S1–S3 with a free peg and friction on:

- **drop:** S1 as is.
- **push:** a 0.1 N lateral force at the peg centre, turning once every 2 s.
  Exact: the centre rides the circle of radius c.
- **tilt:** a 1.5 mN·m horizontal torque, turning once every 2 s. Exact:
  the axis sits at the maximum tilt.

Each cell below gives the coarse / fine setting. Push and tilt errors are
medians after the first 0.5 s; ranges are in parentheses. Push is in µm
(|centre| − c); tilt is in mrad (tilt − max tilt; × 7.5 gives µm at the
plate faces). Negative means stopped early by phantom material.

| behavior | ours (n = 10 / 45) | CoACD (0.01 / 0.005) | V-HACD (1024 / 4096 hulls) | SDF (depth 5 / 6) |
|---|---|---|---|---|
| drop, c = 100 | free fall, no wall contact | same | same | same |
| drop, c = 20 | free fall, no wall contact | stops on the rim, 79 µm true pen. / passes | stops on the rim / jams 2.1 mm deep, 147 µm pen. | stops on the rim, 130 µm pen. / jams 1.9 mm deep, 185 µm pen. |
| push, c = 100 | +7.1 / +0.6 | −6.1 (−36 … −3.6) / −6.3 | −46 / −11 | −38 (−58 … −21) / +5.6 (−12 … +24); 8 mrad spurious tilt, 51–125 µm axial drift |
| tilt, c = 100 | +0.29 / +0.06 | −0.99 / −0.88 | −5.6 / −1.4 | +0.7 (−3.3 … +4.9) / +8.3 (+3.0 … +12.4); true pen. 53 / 105 µm |
| push, c = 20 | +2.8 / +0.4 | −9.6 / −8.0 | −15.9 / −15.9 (pinned) | −16.3 (pinned) / +3.9 (−8.9 … +9.7) |
| tilt, c = 20 (max 2.7 mrad) | +0.13 / +0.03 | −1.6 / −1.1 | −2.65 / −2.4 (no tilt) | −1.9 / +2.8 (−1.1 … +5.3) |

- **Filled clearance:** no setting explodes when the clearance is filled.
  Instead the peg is held rigidly by phantom material. In the drop it sits
  on the rim or jams millimetres deep. Under the push it moves only 4–10 µm
  of its 20 µm play and hardly tilts. The real part would fall straight
  through and rattle freely.
- **SDF errs in both directions.** Its traces are jittery. Depth 5 mostly
  stops early; depth 6 overshoots the exact limit, with up to 105 µm of true
  penetration. A purely lateral push tilts it by up to 8 mrad (ours: ≤ 0.7).
- **Axial drift appears for every method, SDF most.** A purely lateral push
  should not move the peg axially at all. Over 4 s, SDF drifts 51–125 µm,
  ours 4–46 µm, CoACD up to 15 µm and V-HACD up to 6 µm.
- **Ours' free-peg penetration is small but not zero.**
  - **45-gon:** the median true penetration is under 1 µm, with brief spikes
    to 7–11 µm while the peg rolls.
  - **10-gon:** it sits at about 8 µm, its 5 µm of corner play plus contact
    softness (solref 2 ms). With friction and a rolling peg, that is a little
    above S2's frictionless 5.5 µm.

### Accuracy (CPU, c = 100 µm unless noted)

- **ours reproduces the exact geometry.**
  - **Square:** the wall offset is exactly 0. S0–S4 are within 0.5 µm, and
    contact sd ≤ 0.4 µm, which is pure contact softness. Every drop outcome
    is correct.
  - **Circle:** the hole is a *circumscribed* n-gon, so the wall is never
    phantom. It is recessed by at most ε at the corners (wall offset
    0 … +ε exactly), and an oversize peg is always blocked.
  - **Its only error is extra play toward the corners,** c·(1/cos(π/n) − 1).
    That scales with the clearance, not with ε:

    | n (sides), ε = 1000 … 0.5 µm | 7 | 10 | 15 | 20 | ≥ 32 |
    |---|---|---|---|---|---|
    | predicted at c = 100 µm (µm) | 11.0 | 5.2 | 2.2 | 1.3 | ≤ 0.5 |
    | measured, S2 (µm) | 11.2 | 5.5 | 2.7 | 1.7 | 0.3–1.1 (softness floor) |
    | contact sd, penetration (µm, MULTICCD off) | 11.4 | 5.6 | 2.7 | 11.8* | 0.3–1.5 |

    \* a single chatter event.
  - **False contacts:** none at any n.
  - **Drop test:** every S1 outcome is correct, even for the heptagon.
- **CoACD fills the clearance at every threshold except the finest.**
  - **Circle, default threshold (0.05):** the wall offset is −864 µm, i.e.
    phantom wall; contact is reported through 200–700 µm of empty space, and
    pegs with up to +500 µm clearance are blocked.
  - **Circle, finest threshold (0.01:** 839 hulls, 17 min to build): still
    −46 µm of phantom wall, with false contacts through 46 µm gaps and 49 µm
    of penetration. It blocks at c ≤ 20 µm.
  - **Square:** thresholds ≤ 0.05 all give the same 7 hulls, with −22 µm
    phantom walls and false contacts through 16 µm gaps.
- **V-HACD fills the clearance at every setting.** Its walls are both
  phantom (−197 … −1227 µm) and recessed (up to +320 µm) in places.
- **SDF has two separate errors, and the geometry metric separates them:**
  - **Octree depth:** at depth ≥ 7 the interior wall is nearly exact (−2 …
    −5 µm mid-wall), but the rim edges are rounded by +76 … +215 µm. Depth 5
    and 6 are phantom mid-wall (−27 … −87 µm) and round the edges by
    0.5–1.2 mm.
  - **Collision detection:** the sampled gradient descent misses overlap.
    Contact onset comes 2–50 µm late even where the wall itself is exact.
  - **Result:** contact sd shows 46–129 µm of real penetration (tilting pegs
    ride over the rounded rims). On the square there are also false contacts
    through 12–29 µm gaps, since the corners are filled.
- **MULTICCD (on by default in 3.14)** adds cylinder-vs-prism chatter
  spikes of 10–28 µm to ours on the circle. It's a setting like any other:
  off is both more accurate and faster for ours, so the Pareto fronts include
  both settings.
- **Solver floor for S1 (not geometry).**
  - **What happens:** at mkpeg's `solref` (2 ms), frictionless drops let pegs
    5–20 µm *oversize* slip in, for every method including exact geometry.
  - **Why:** a peg dropped from 1 mm moves several µm per step, so its edge
    gets past the rim before the contact holds.
  - **Stiffer contacts** (0.5 / 0.2 ms with a smaller timestep) halve the
    band.
  - **Consequence:** S1 cannot resolve geometric errors below ~20 µm. The
    other probes can.

### Speed

Real-time factor at the 1 ms timestep, peg pressed against the wall
([`report/speed.png`](report/speed.png)):

| | CPU, 1 env, 1 thread | GPU, 256 Warp worlds (summed) |
|---|---|---|
| ours, square | 93× | not rerun at 256 (970× at 512) |
| ours, circle, n = 7 … 32 (ε = 1000 … 40 µm) | 82 … 29× | 600 … 570× |
| ours, circle, n = 63 … 281 (ε = 10 … 0.5 µm) | 16 … 4.9× | 470 … 245× |
| SDF, depth 5 … 10 | 2.9 … 2.2× (0.9× with 100 pts / 30 iterations) | 400 … 460× |
| CoACD 0.01 … 0.005 (circle, finest) | 1.8 … 0.5× | 160 … 64× |
| V-HACD 1024 … 4096 hulls (circle, finest) | 1.8 … 0.5× | 340 … 57× |
| other CoACD / V-HACD settings (all fill the clearance at c = 20 µm) | 3.3 … 70× | 450 … 770× |

Each cell is the best of MULTICCD on and off.

- **CPU:** ours is 10–40× faster than SDF at matching or better accuracy.
- **GPU at 256 worlds:** batching hides most of the per-contact cost. Ours
  at n ≤ 32 runs 1.2–1.6× faster than SDF at any depth, and its finest
  polygon (n = 281) at about 0.6× SDF's speed. Plates with many hulls near the peg
  fall behind: every hull becomes a broadphase candidate pair, which costs
  Warp memory and time (CoACD 0.005: 64×).
- **Cost scaling for ours:** cost grows with the wedge count, because each
  radial wedge's bounding box overlaps the peg's. That's the obvious
  decomposition follow-up.

GPU accuracy (MuJoCo Warp vs CPU on the same scenes):
[`report/cpu_vs_warp.png`](report/cpu_vs_warp.png), and the Warp Pareto
fronts above.

---

## What varies: the plate representation

Only the plate/hole geometry changes. The peg is always an exact MuJoCo
primitive (cylinder for the circle, box for the square), so every difference
comes from the hole. The plate depends only on the hole, never on the
clearance, so one decomposition serves a whole clearance sweep.

| method  | plate collision geometry | sweep ([`configs/probes.yaml`](configs/probes.yaml)) |
|---------|--------------------------|------------------------------------------------------|
| `ours`  | mkpeg convex prisms, via `mkpeg.generate_xml` | circle: chord error ε = 1000 … 0.5 µm (n = 7 … 281 sides); exact for polygons |
| `coacd` | CoACD hulls of the exact mesh | `threshold` ∈ {0.2, 0.1, 0.05 (default), 0.02, 0.01}; circle adds 0.0075 and 0.005 |
| `vhacd` | V-HACD hulls (`vhacdx`) | `max_hulls` ∈ {16, 64 (default), 256 @ 4M voxels}; circle adds up to 4096 hulls @ 120M voxels, recursion depth 14 |
| `sdf`   | MuJoCo native mesh SDF (octree) | octree depth ∈ {5, 6 (default), 7, 8}, 8 with 100 init points / 30 iterations; circle adds depths 9 and 10, each also with 100 / 30 |

Baselines receive an exact, watertight plate mesh; the circle is tessellated at
1024 segments (0.04 µm sagitta). Ground truth uses the *analytic* circle, so
`ours` is also charged for its own chord error.

**Task:** Ø16 mm hole (circle, or a 16 mm square), plate 48 × 48 × 15 mm,
through-hole, peg 40 mm long.

**Held fixed:** timestep 1 ms, `implicitfast`, Newton solver, mkpeg's contact
parameters (`solref 0.002 1`, `solimp 0.99 0.999 0.001`, µ = 1, condim 4, or
condim 1 in the frictionless probes), `margin 0`. MULTICCD is run both **on**
(MuJoCo 3.14 default) and **off** for every mesh method; SDF ignores it.

## Probes (all passive, all with exact answers)

| id | probe | setup | exact answer |
|----|-------|-------|--------------|
| — | **wall offset** (no solver) | rays from the hole axis at 720 angles × 11 heights. Exact slab intersection for hulls; MuJoCo's own octree (same lookup and trilinear interpolation as `oct_distance`) for SDF | wall exactly on the analytic outline; reported for the full wall and mid-wall (10–90 % of T), so rim-edge rounding is separated out |
| S0 | **static contact onset** | aligned peg placed at poses of known true gap (+1 mm … −200 µm, 24 directions), `mj_fwdPosition` only | contact appears exactly at gap 0; reported depth = true per-wall depth |
| S1 | **drop** | centred, aligned free peg dropped from 1 mm; c from +1 mm to −500 µm; friction on and off | c > 0 passes through with **zero** wall contact; c < 0 stays on top with **zero** sink |
| S2 | **lateral play** | x/y slides only, frictionless, gravity off, 0.1 N push in 24 directions, 1 s | reach along u = support of the feasible-translation set (c for the circle; c(\|uₓ\|+\|u_y\|) for the square) |
| S3 | **tilt play** | x/y slides + one tilt hinge, 1.5 mN·m torque (φ = 0°, 22.5°, 45° for the square) | max tilt at which both plate-face slices fit (closed form for the circle, exact 2D check for the square) |
| S4 | **yaw play** (square) | x/y slides + z hinge, 0.8 mN·m torque | asin(a / (s√2)) − π/4 |

All errors are in **µm of wall position** (S3 is converted at the plate faces;
S4 at the peg corner). **Negative** means the peg was stopped early by phantom
material. **Positive** means it went where solid material is. S2–S4 report the
median, the worst spike, and the jitter over the settled second half of the run.
S1–S4 also record the **true signed distance whenever the simulator reports
contact**. This is Son & Kim's `min-|sd|`: a false-contact gap if positive,
penetration if negative, from sampled exact SDFs. It is computed on both CPU
and Warp. S1 also records wall impulse, sink depth, and the IndustReal-style
maximum interpenetration over the trajectory.

The oracles were cross-checked against the independent penetration calculation
(zero penetration at the predicted limit, positive 2 % beyond it).

## Speed

Three passive, contact-rich states, with gravity and friction on and a free
peg: `wall` (peg pushed sideways inside the hole, c = +100 µm), `rim` (peg
200 µm too wide, resting on the rim), `drop` (the S1 drop).

- **CPU:** one environment on one thread. Best-of-5 `mujoco.rollout`, 1000
  steps, plus `mj_collision` time replayed on sampled states.
- **GPU:** MuJoCo Warp with **256 worlds**, one CUDA graph per 100 steps,
  reported as µs per env-step.
  - **Buffers:** per-world contact and constraint buffers are calibrated per
    representation (1.5 × the observed peak, including broadphase pairs).
  - **Overflow checks:** Warp's sticky overflow bitmask is checked on every
    run, since contact, constraint, CCD, or EPA overflow would silently drop
    contacts. An overflowed run is retried with doubled buffers.
  - **Isolation:** each configuration runs in a fresh process, because
    captured CUDA graphs keep device memory after a batch is freed.

CPU timings come from `improbable381` (i9-13900KS, RTX 5090), and the 256-world
GPU timings from `improbable382` (Core Ultra 9 285K, RTX 5090). Each ran with
nothing else on the machine.

## Layout

```
benchmark/
  geometry.py         Task, exact SDFs, penetration / contact-gap metrics, oracles
  representations.py  plate builders (ours / coacd / vhacd / sdf) + on-disk cache
  scene.py            common MjSpec scene (options, contact params, peg, joints)
  probes.py           S0-S4 on CPU MuJoCo
  warp_backend.py     batched scenes on MuJoCo Warp; S1-S3 on GPU
  run_probes.py       CPU accuracy sweep     -> results/probes/*.csv
  run_warp_probes.py  GPU accuracy sweep     -> results/probes/warp_*.csv
  run_speed.py        CPU + GPU timing       -> results/speed/*.csv
  report.py           tables + figures       -> report/
  render.py           behavior videos        -> results/render/*.mp4
  thread/             bolt/nut benchmark (same layout; see thread/README.md)
    plugin/           analytic ISO thread SDF (MuJoCo plugin, C++)
  configs/probes.yaml the sweep grid (configs/thread.yaml for bolt/nut)
  cache/  results/    gitignored
```

## Running

```bash
conda create -n mjbench python=3.12 && conda activate mjbench
pip install -r benchmark/requirements.txt
python -m benchmark.run_probes             # CPU accuracy, all cores
python -m benchmark.run_warp_probes        # GPU accuracy
python -m benchmark.run_speed              # timing: run on an idle machine
python -m benchmark.report
MUJOCO_GL=egl python -m benchmark.render   # videos of Pareto points
```

The same pinned requirements install with `uv pip install -r` where conda is
not available (as on `improbable381`). Copy `benchmark/cache/` between machines
so every machine simulates byte-identical plates.

## Platform notes (MuJoCo 3.14 / Warp 3.14)

- **SDF octree depth** exists only on the spec (`mjsMesh.octree_maxdepth`,
  not XML), and MuJoCo's mesh cache is keyed by file. A file-loaded mesh
  therefore silently reuses the octree from its first load. `scene.py` passes
  meshes as arrays to bypass the cache.
- **MULTICCD is on by default** in 3.14 (`mjDSBL_MULTICCD` to disable). Warp
  rejects `margin > 0` on mesh pairs while it is on, and CPU SDF collisions
  ignore margin, so all probes use `margin = 0`.
- In **Warp 3.14**, `nconmax` / `njmax` are **per world**; `naconmax` is the
  total. The broadphase stores candidate pairs in the same contact buffer. So
  `nconmax` must cover about one pair per plate geom near the peg, not just
  real contacts; otherwise it raises BROADPHASE overflow and drops contacts.
  High-hull plates therefore cost GPU memory per world: CoACD 0.01 on the
  circle needs about 490 pairs per world, and 16k worlds would need 43 GB.
- `mjw.put_data` copies the CPU `MjData`'s contacts, so `nconmax` must also
  cover the contacts at the initial pose. A plate that fills the clearance
  starts with hundreds; otherwise `put_data` raises "nconmax overflow".
- Warp floors the solver `tolerance` at 1e-6 (float32); CPU uses 1e-8. This
  does not change the probe results on CPU.
- `vhacdx` expects VTK-style faces (`[3, a, b, c, …]`, flattened). Plain
  `(n, 3)` faces silently produce flat, meaningless hulls.
- CoACD at fine thresholds takes minutes per plate and runs single-threaded
  when BLAS/OpenMP threads are capped; `run_probes` builds all plates in
  parallel first.
