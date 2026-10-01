# Bolt/nut benchmark: M10×1.5 thread

The same comparison as the peg benchmark (`../README.md`), run on a threaded
fastener:

- **ours:** `mkscrew.py`'s convex wedges;
- **CoACD** and **V-HACD:** convex decomposition of exact watertight meshes;
- **SDF:** mesh SDF with an octree;
- **SDF, analytic:** the exact shape as a MuJoCo SDF plugin.

Every probe is passive and has an answer computed from the exact helical
geometry.

Figures: [`../report/pareto_thread_cpu.png`](../report/pareto_thread_cpu.png)
(time) and [`../report/pareto_thread_memory.png`](../report/pareto_thread_memory.png)
(memory). Full tables: [`../report/thread_summary.md`](../report/thread_summary.md).

## The task

- **Parts:** an ISO 68-1 M10×1.5 thread, a 16 mm threaded rod (fixed) and an
  8 mm nut (Ø16, steel, about five engaged turns).
- **Clearance c:** the gap between facing flanks when the nut is centred.
  - Axial play is ±2c/√3 (±115 µm at c = 100 µm, ±23 µm at c = 20 µm).
  - Radial play is c/2 (the nut tip over the bolt core).
- **Nut mass:** every representation's nut gets the exact nut's mass and
  inertia, so representations differ only in contact.

## Fixes to `mkscrew.py` this benchmark needed

The exact geometry exposed several problems in the generator. All are fixed
and covered by `tests/test_mkscrew.py`.

1. **Nut tooth P/8 too thin.** The nut used the bolt tooth mirrored. The ISO
   internal tooth has a P/4 tip and a 7P/8 base, so the old nut had a fixed
   188 µm of axial play at M10 whatever `clearance` was.
2. **`clearance` did not open the flanks.** It only lowered the crests, so
   one flank pair touched at the nominal pose. Now both parts' flanks move
   back c/2 along their normals.
3. **Nut wedges bulged toward the bolt.** A wedge is the hull of the profile
   at θ and θ+Δθ, so its faces are chords of curved surfaces. On the nut they
   stuck out two ways, and both are now removed, so coarse wedges only add
   play:
   - arc sag, 36 µm at 24 segments per turn, removed by pushing the wedge
     out radially by 1/cos(Δθ/2);
   - helicoid twist, 2.6 µm, removed by moving the flanks back by
     lead·Δθ·(r₁ − r₂)/(4π(r₁ + r₂)).
4. **Wedges overhung the part ends** by up to about 1 mm. They are now
   clipped exactly to the thread length.
5. **Thread wedges only collided with thread wedges,** so tooth tips passed
   through the opposite root. All bolt/nut pairs now collide.
6. **The default layouts started with overlapping threads.** Placement is now
   snapped to the thread phase.
7. **Vertices were rounded to 1 µm.** They are now written at full precision.
8. **The partial first turn at each part's z = 0 end was missing.** The
   turn loop started at turn 0. The turn before it still rises above z = 0
   over most of a turn: 171–360° on the M10 nut and 14–360° on the bolt,
   material up to 0.7 mm inside the exact parts. In mkscrew's own scenes
   those are the ends that meet first: the bolt's tip, and the nut face that
   goes onto it in the nut-driven layout. The probes on the middle of the
   bolt could not see it. The engagement probe (T6) did: resting on the tip,
   the nut sank 556 µm, 90° before the exact click, and jammed. Now the loop
   starts at the lowest turn that reaches the face, and
   `test_every_form_fills_its_teeth_to_both_ends` checks every form. On the
   old code, every form but a left-hand two-start one fails it.
9. **The nut's flank correction checked only the facet edges.** A flank is
   curved, so it could bow out past the middle of a flat facet: 2.3 nm at 12
   segments and c = 20 µm. It showed up once item 8 changed which points the
   containment test sampled. The correction now zooms in on each facet's
   peak, and wedges stay inside the exact parts to 0.001 nm.

Behaviour changed as a result. With friction 0.3 and condim 3 (the example
configs), a spun bolt advances 94 % of the pitch per turn at c = 100 µm
(89 % at 20 µm). At friction 0.1 it is 97–98 %. At the generator's default
friction of 1.0 it is only 75 % (34 % at 20 µm), because velocity-driven
rotation against a self-locking thread pushes the soft contacts about 40 µm
deep. The top-level README has the table.

## Findings

Pareto fronts: CPU µs per step (1 thread, nut screwed up the thread under
load) against the worst settled error over T1–T4.

| method (knob) | c = 100 µm | c = 20 µm |
|---|---|---|
| **ours** (segments per turn 12 … 192) | 87 µm at 0.46 ms → 20 µm at 0.93 ms (n = 24) → 4.5 µm at 1.9 ms (n = 48) → 2.2 µm at 2.6 ms (n = 64) → **0.97 µm at 4.1 ms, 6 MB (n = 96)** | 86 µm at 0.49 ms → 20 µm at 1.0 ms (n = 24) → 4.3 µm at 2.1 ms (n = 48) → 2.5 µm at 2.8 ms (n = 64) → **1.5 µm at 7.9 ms (n = 128)** |
| **CoACD** (threshold 0.1 … 0.02) | fills at every threshold (0.95–9.6 ms, up to 52 MB and 586 hulls) | fills (2.0–17 ms) |
| **V-HACD** (64 … 4096 hulls) | fills at every setting (0.9–30 ms, up to 55 MB and 8180 hulls) | fills (1.0–48 ms) |
| **SDF, mesh** (octree depth 6 … 9) | fills at every depth; the nut slides through the thread (0.74–0.98 ms, 46 MB–1.7 GB) | fills |
| **SDF, analytic** (points × iterations) | 91 µm at 1.8 ms (MuJoCo's default 40 × 10; the screw jams) → 52 µm at 4.3 ms → 22 µm at 12.7 ms (300 × 60); 48 MB | 53 µm at 1.6 ms → 44 µm at 4.1 ms → 21 µm at 12.2 ms |

- **Only ours keeps the clearance open, and it converges.** Its error is the
  discretization, one-sided by construction (extra play, never interference),
  and it falls as 1/n². Axial error goes from 87 µm at 12 segments to 0.03 µm
  at 192. With MULTICCD off, radial and tilt errors are under 2.1 µm. (The
  ours rows were re-run after mkscrew fix 8. The worst settled T3 error fell
  from 8.1 to 2.1 µm; single settings moved by up to about 1 µm either way,
  which is the probe's settling noise at this level. The worst T1–T4 error
  per setting moved by 0.25 µm at most.) From 96 segments up, the worst
  error sits at about 1–2 µm, set by the loaded screwing probe (T4), where
  the nut briefly penetrates up to 10 µm as it slides across facet seams.
- **At equal cost, ours is 20–50× more accurate than the analytic SDF.** At
  about 4 ms per step, ours at 96 segments has a 1 µm error and the analytic
  SDF at 100 × 30 has 52 µm. Ours at 24 segments (0.93 ms, 20 µm) already
  beats the analytic SDF's best setting (12.7 ms, 22 µm).
- **Threads cost far more than the peg.** Every wedge on all five engaged
  turns touches the bolt, which is 200–1800 contacts for ours. Ours at 24
  segments costs 0.93 ms per step, against 14 µs for the circle peg's
  10-gon.
- **Thread geometry defeats generic decomposition.** At c = 100 µm every
  CoACD and V-HACD setting, up to 586 and 8180 hulls, overlaps the parts at
  the centred pose. The nut is held by phantom material, with errors of the
  order of the clearance or larger, and jams when screwed.
- **Mesh SDFs fail at every depth.** The octree's trilinear interpolation
  rounds the sharp 60° tooth. Even at depth 9 (1.7 GB) the parts overlap at
  the start, and under torque the nut slides through the thread.
- **With the exact shape, the SDF's remaining error is its contact search.**
  The analytic plugin evaluates the true distance, and its contact normals
  are right (0.500 radial, 0.865 axial, 0.045 from the 3° helix). Yet
  MuJoCo's SDF collision finds only a few contacts along five turns of
  thread, and reports depths about 3.5× too shallow. At the default search
  effort (40 × 10) the torqued nut jams. More effort helps, but at 300 × 60
  the radial and tilt errors are still 14–22 µm. That is the same finding as
  on the peg, where a depth-10 octree matched the wall within 7 µm but tilt
  was still 50–100 µm off.
- **Frictionless spin-down (T5) is ours' weak spot.** The analytic SDF
  descends exactly as the closed form predicts (100 %). Ours does not:
  - 12 and 16 segments barely move (2–5 %);
  - 24–64 segments reach 30–74 % with MULTICCD off;
  - finer threads fall back to 14–33 %.

  CoACD and V-HACD mostly jam (≤ 6 %). The mesh SDF's nut slips down through
  the thread instead of following it. With friction 0.3 (a self-locking
  thread), ours holds, and its only error is the extra discretization play.

  **Cause: internal-edge ("ghost") contacts at the seams between wedges.**
  MuJoCo collides a nut wedge with each bolt wedge separately. Under load the
  soft contacts let the nut sink about 0.5–1 µm into the bolt. So when a nut
  wedge slides across the seam into the next bolt wedge, the first overlap
  with that wedge is a sliver at its end face. The penetration solver takes
  the shortest way out, back through that end face. The contact normal then
  points around the axis (0.84 tangential, against 0.09 radial and 0.15
  axial), a wall in front of the nut. The nut stops dead in one step, and its
  entire rotational energy is lost (24.9 µJ ≈ ½Iω² at 96 segments). Then it
  re-accelerates and snags again every 0.15–0.25 s. At 96 segments, the 5 %
  of steps holding such contacts carry 98 % of the loss.

  - **Finer threads are worse** because they have more seams to snag on.
  - **The analytic SDF has no seams,** so it matches exact.
  - **Less penetration removes most of it:**

    | ours | step, contact time constant | descent (% of exact) |
    |---|---|---|
    | 96 segments | 1 ms, 2 ms (as benchmarked) | 22 % |
    | 96 segments | 0.2 ms, 0.5 ms | 95 % |
    | 24 segments | 0.1 ms, 0.2 ms | 77 % |

    The 24-segment plateau is geometric: its 15° facets make a washboard that
    throws the nut up, and it loses energy on landing.
  - **Geometry fix: `mkscrew`'s `seam_relief`.** It dips the teeth a few
    µm at every seam, with a 1:20 ramp, so a sliding tooth meets a ramp
    instead of an end face. It only removes material: the nut corrections
    cover every chord of the new hulls, and the protrusion test holds to
    0.35 nm. Results over 6 start angles each, with default contacts
    (1 ms step, 2 ms time constant):

    | ours | no relief | 3 µm | 5 µm | static axial error, none → 5 µm |
    |---|---|---|---|---|
    | 24 segments | 69 % (43–94) | 89 % | **92 % (91–93)** | 20.1 → 16.3 µm |
    | 48 segments | 41 % (32–48) | 86 % | **87 % (87–88)** | 4.4 → 2.1 µm |
    | 96 segments | 30 % (25–36) | 66 % | **71 % (65–75)** | 0.77 → 0.38 µm |

    - **Dead stops are gone.** At 96 segments they drop from 3 per 0.5 s to
      none.
    - **The static error improves** (radial and tilt are unchanged), because
      each full-height chord is shorter than the whole wedge.
    - **The cost:** the loaded-screwing error at 96 segments rises from 1.0
      to 2.6 µm.
    - **What remains at fine resolution:** the grooves leave less area to
      carry the nut, so contacts sink deeper (median 3.4 µm, 90th
      percentile about 7 µm). Some sink past the relief.
  - **This probe is chaotic.** Snags hinge on tiny differences, and the same
    run gave 44 % on one machine and 59 % on another. Compare settings over
    several start angles (`probe_spin(start_angle=...)`), not single runs.
- **Run-down with friction** (the nut runner in `thread_rundown_*.mp4`, not
  a probe).
  - **Setup:** a free nut, gravity, sliding friction 0.3, 1 ms step. A speed
    servo about the nut axis turns it 4 turns down at 1 rev/s, with a torque
    limit of 5 mN·m.
  - **Reference:** the power-screw lowering torque for the nut's weight,
    0.106 mN·m (`geometry.rundown_torque`). It is a textbook formula at the
    pitch radius, not an exact answer.
  - **Results:** mean runner torque as a multiple of that formula, over 4
    start angles each. In brackets, the share of 40 ms windows above 3×.

    | setting | c = 100 µm | c = 20 µm |
    |---|---|---|
    | ours 24, no relief | 1.06–1.08 (0 %) | 1.07–1.11 (0 %) |
    | ours 24, 5 µm relief | 1.06 (0 %) | 1.06 (0 %) |
    | ours 48, no relief | 1.10–1.14 (0 %) | 1.28–1.48 (4–8 %) |
    | ours 48, 5 µm relief | 1.07 (0 %) | 1.06–1.16 (0–2 %) |
    | ours 96, no relief | 1.67–1.96 (6–13 %) | 1.21–1.52 (1–7 %) |
    | ours 96, 5 µm relief | 1.10–1.21 (0–1 %) | 2.54–2.96 (25–38 %) |
    | ours 128, no relief | 2.08–2.30 (13–21 %) | 2.15–2.48 (13–18 %) |
    | ours 128, 5 µm relief | 1.27–1.54 (1–7 %) | 3.89–4.58 (40–59 %) |
    | SDF, analytic, 100 × 30 | 0.93 (0 %) | 1.10–1.13 (0 %) |
    | SDF, mesh, depth 7–9 | jams: 0.05–1.2 of 4 turns, tilted 10–40° | jams |

  - **Seam relief is not a universal fix.** It clears the catching at
    c = 100 µm at every resolution, and at c = 20 µm up to 48 segments. At
    96–128 segments and c = 20 µm it catches more than no relief. Shallower
    relief is worse there: 2 µm pins the runner at its limit. The wedges still
    stay inside the exact parts at every depth (0.35 nm), so this is contact
    dynamics, not a geometry error. `seam_relief` stays off by default.
  - **Use condim 3 on threads.** `build_scene` defaults to condim 4 when
    friction is on. That carries the peg benchmark's 0.01 m torsional
    friction, which resists the spin at each flank contact about 6× harder
    than the sliding friction does. With it, the nut chatters and binds: at
    c = 20 µm, ours at 48 segments made 2.3 of 4 turns.
- **Turn MULTICCD off for thread wedges.** With it on, MuJoCo sometimes
  reports a spurious contact about 0.7 mm deep between thin wedges. The
  repulsion launched the nut through the thread (ours, 32 segments: 11.6 mm
  axial travel, before mkscrew fix 8) and jams the screw probe (64–128
  segments at c = 100 µm). On the tip it holds the nut up on the top face,
  so it never clicks (T6, 48–128 segments at c = 100 µm). Where it doesn't
  misfire, MULTICCD on is sometimes marginally more accurate, but it costs
  15–40 % more time. Off is the safe setting.

CoACD at threshold 0.01 (in `configs/thread.yaml`) took about 1.9 hours per
clearance to decompose the bolt and nut, into about 3,800 hulls. It
finished after the sweep, so the probe tables and Pareto plots stop at 0.02.
The 0.02 setting already has 586 hulls and fills the clearance.
`run --resume --only-cached` adds 0.01 now that its cache files exist (on
improbable381).

Timings are the best of three 500-step runs on one thread, on improbable381.
Two CoACD decompositions (the 0.01 threshold, still running) occupied 2 of its
32 cores during the timing pass. A filled setting sometimes ejects the nut in
the timing run (the mesh SDF at depth 9, c = 20 µm). With nothing left to
collide, its time is not a cost, and it is dropped from the time plot.
Ours' timings predate mkscrew fix 8, which adds 9 % more geoms and 11 %
more model memory (the memory plot has the new figures). Interleaved on one
core against the old parts, it costs 0 % at 24 segments and 5 % at 96.

## Initial engagement: the nut on the bolt's tip (T6, T7)

T1–T5 hold the nut about five turns onto the middle of the bolt. Starting a
nut is different: it sits on the bolt's top face, the partial threads at
both parts' ends meet, and at most a turn or two is engaged. T6 and T7 test
that, against exact answers (see Probes). Full tables are in
[`../report/thread_summary.md`](../report/thread_summary.md). Figures:
[`../report/thread_start.png`](../report/thread_start.png) (T6 traces) and
[`../report/thread_partial.png`](../report/thread_partial.png) (T7). CoACD
and V-HACD are not run on these probes. The ends are flat-cut, as in the
rest of the benchmark; `geometry.Tip` also handles chamfers.

**What the exact geometry says:**
- **The back-spin click is the axial play.** A coaxial nut that is out of
  phase sits flat on the top face. Turned the unscrewing way, it drops into
  the thread where its phase enters the play window, by exactly 4c/√3:
  231 µm at c = 100 µm, 46 µm at 20 µm. It then rides the flank back up to
  the face over 55° (11° at 20 µm). With 1 mm chamfers the out-of-phase nut
  rests on the cones instead, 1.07 mm lower, and the click is the same.
- **Screwing on** starts where the window's other edge reaches the face, and
  follows the helix with no step.
- **A just-started nut is loose.**
  - **Radial:** a quarter turn on, the play is up to 2c in one direction, the
    full-engagement c/2 only in others. By one turn it is c/2 all round.
    Pushed along 90°, the nut meets the core's rim where its tip arc ends,
    40° off the push, and slides around it: 93 µm of reach, against 63 µm
    straight along the push.
  - **Tilt:** with the hinge half-way along the engaged length, the limit is
    20–25 mrad from half a turn on, and up to 44 mrad at a quarter turn
    (c = 100 µm). T3's fully engaged nut, hinged at its centre, stops at
    12.4 mrad. With a short engagement, only the axial play at the thread
    radius stops the tilt, not the radial play at the faces.

**T6** (ours with MULTICCD off; torque 5× the exact back-off torque, T4's
ratio, unless noted). Click and entry errors are in µm of axial phase along
the helix (angle × lead/2π); negative is early. The height error is the
worst over the run, leaving out the falls.

| setting | back-spin (c = 100 / 20 µm) | click (µm) | screw-on | entry (µm) | height, back-spin / screw-on (µm, c = 100) |
|---|---|---|---|---|---|
| ours, 12 segments | clicks; jams at 2× | −33 / −14 | engages | −51 / −30 | 87 / 95 |
| ours, 24 | clicks; jams at 2× | −14 / −0.7 | engages | −0.1 / −12 | 21 / 29 |
| ours, 32 | clicks, also at 2× | −2.6 / −3.1 | engages | −0.7 / −1.3 | 11 / 20 |
| ours, 64 | clicks, also at 2× | 1.3 / 0.3 | engages | −6.8 / −0.5 | 9 / 12 |
| ours, 96 | clicks, also at 2× | 1.7 / 0.8 | engages | −0.7 / −5.6 | 9 / 10 |
| ours, 192 | clicks; at 2× and 20 µm, no click | 1.7 / 1.0 | engages | −6.7 / −5.5 | 10 / 10 |
| SDF, analytic, 40 × 10 | clicks, also at 2× | −153 / −23 | engages | −8.5 / −4.0 | 63 / 23 |
| SDF, analytic, 300 × 60 | clicks, also at 2× | −13 / −3.7 | engages | −8.4 / −9.7 | 14 / 20 |
| SDF, mesh, depth 6–9 | overlaps at the start; never clicks | — | never enters | — | 1–12 mm |

- **Ours clicks where the thread says, and screws on, at every
  resolution.** From 64 segments up the click is within 2 µm of axial phase.
  Coarser, it comes early by the extra axial play of the discretization (T1).
  The height error off the drops is about 9 µm at fine resolutions: brief
  catches at facet seams. On the helix its median is 0–0.5 µm. The one
  exception is chaotic: 128 segments at c = 20 µm jammed after the click at
  5×, yet clicked at 2×.
- **At 2× the exact torque, 12–24 segments jam right after the click.** The
  few contacts all sit at one phase of the seams, on facets whose
  tangential/axial normal ratio is about 0.12. That is 2.3× the helix's
  0.056, so they need 2.3× the exact torque. From 32 segments the nut rides
  back up. The analytic SDF, which has no facets, clicks at 2× too.
- **The click overshoots by 23–26 µm at every fine resolution.** That is
  the falling nut's impact into the soft contacts, not geometry, whose error
  there is under 2 µm. The analytic SDF sinks further: 10–74 µm at 100 × 30
  and 300 × 60, and 57–169 µm at MuJoCo's default 40 × 10, where it also
  clicks up to 153 µm early.
- **MULTICCD on stops the click.** At c = 100 µm, ours at 48–128 segments
  never clicks: the nut is held up on the top face, face on face. Off is the
  safe setting here too.
- **The mesh SDF overlaps the parts at the start pose at every depth.** Its
  octree rounds the teeth and the end faces, so the nut is held millimetres
  off the path and never engages.
- **This probe found mkscrew fix 8.** Before it, ours at 24 segments sank
  556 µm, 90° before the exact click, and jammed there.

**T7:** worst settled error (µm; tilt at the thread crest), c = 100 µm. The
exact rows give the range over directions.

| | ¼ turn | ½ turn | 1 turn | 2 turns |
|---|---|---|---|---|
| exact radial play (µm) | 50 … 100 | 50 … 80 | 50 | 50 |
| exact tilt limit (mrad) | 23 … 44 | 22 … 25 | 22 … 23 | 20 … 21 |
| ours 24, radial / tilt | 13 / 40 | 1.8 / 29 | 1.0 / 26 | 0.6 / 24 |
| ours 64, radial / tilt | 4.7 / 7.0 | 2.5 / 1.7 | 0.5 / 1.4 | 0.3 / 1.2 |
| ours 128, radial / tilt | 2.7 / 0.9 | 3.5 / 0.6 | 0.6 / 0.6 | 0.3 / 0.5 |
| SDF, analytic 300 × 60, radial / tilt | 36 / 31 | 56 / 24 | 20 / 20 | 12 / 22 |
| SDF, mesh, depth 6–9 | 1–4 mm | | | |

- **Ours' radial error is under 1 µm from one turn on**, as in T2. At ¼–½
  turn, up to 13 µm remains at 16–48 segments. In those directions the bolt
  crest reaches the nut's bore where the tip arc has ended. mkscrew's bore
  is a 16-sector polygon circumscribing the circle, up to
  r(1/cos(π/16) − 1) = 99 µm outside it at the sector joins. With 128
  sectors (`nut_shell_sectors`) the error falls below 2 µm (checked at 24
  segments). At full engagement a whole ring of crests touches the bore, so
  T2 never sees this.
- **Ours' tilt error is the discretization's extra axial play.** A short
  engagement's tilt is limited by the axial play at the thread radius. At
  24 segments, whose seams line up with the bolt's at these poses, that is
  23–46 µm at the crest. From 64 segments it is 1–7 µm. At 128 it is under
  1 µm from half a turn on (6.6 µm a quarter turn on, at c = 20 µm).
  At 96 and 192 segments, 7–11 µm remain at 1–2 turns: on two of the four
  axes the nut chatters and never settles (jitter 8–12 µm). With MULTICCD
  on, 128–192 segments misfire by 100–200 µm.
- **The analytic SDF is 11–74 µm off** at its finest setting, and worse at
  MuJoCo's default. From 32 segments, ours is more accurate.

## Turning torque against friction (T8)

How hard is it to turn an engaged nut? The nut sits at home, fully engaged,
on an axial slide and spin hinge with no joint damping, pressed down by an
axial load. Friction is μ on both parts. The torque ramps slowly, and the
breakaway torque is the one at which the nut has turned 5°, either against
the load or with it. The reference is the power-screw formula
(`geometry.screw_torque`), at the pitch radius with the flank tilt folded in
(μ′ = μ/cos 30°). It is textbook, not exact. Figure:
[`../report/thread_torque.png`](../report/thread_torque.png). Full table:
[`../report/thread_summary.md`](../report/thread_summary.md).

Breakaway over the formula, against / with the load, c = 100 µm, MULTICCD
off. "> 3": never turned within 3× the formula. "0": back-drove, so the
thread did not self-lock, where at this friction it should.

| setup (ours, 48 segments) | μ = 0.1 | μ = 0.3 | μ = 1 |
|---|---|---|---|
| 0.1 N, condim 3 | > 3 / 1.49 | 1.22 / 1.28 | 1.20 / 0.97 |
| 0.1 N, condim 3, elliptic cone | 2.79 / 1.48 | 1.20 / 1.22 | 1.16 / 1.18 |
| 0.1 N, condim 4 (mkscrew's default) | > 3 / > 3 | 2.41 / 3.40 | 1.28 / 1.53 |
| 1 N, condim 3 | > 3 / 0 | 1.63 / 1.81 | 1.28 / 1.32 |
| 1 N, condim 3, elliptic cone | 2.54 / 5.71 | 1.60 / 1.92 | 1.28 / 1.35 |
| 1 N, condim 3, elliptic, 0.25 ms step | 1.48 / 1.50 | 1.24 / 1.27 | 1.14 / 1.22 |
| analytic SDF 100 × 30, 0.1 N, condim 3 | 1.08 / 1.30 | 1.08 / 1.03 | 1.15 / 1.16 |

- **At a light load, the torque follows the formula, a little high.** With
  condim 3 and μ ≥ 0.3, ours needs 1.0–1.7× the formula at every resolution
  (mostly 1.1–1.4×). The analytic SDF needs 1.0–1.2×. With MULTICCD on, 96–192
  segments misfire, up to more than 3× at μ = 0.3.
- **At low friction, ours' facets dominate.** At μ = 0.1 the formula torque
  is small, and ours needs 1.5–5× it, or never turns within 3×. The analytic
  SDF, which has no seams, still needs only 1.1–1.3×. This is the same seam
  resistance that jams coarse wedges after the T6 click.
- **condim 4 adds a large spurious term.** mkscrew's default torsional
  coefficient (0.01 m) resists the spin at every flank contact: 1.3–3.4× at
  μ = 0.3–1, and no turn within 3× at μ = 0.1. Use condim 3 on threads, as
  the example configs do.
- **A heavy load for the nut's mass breaks it at a 1 ms step.** 1 N is
  120 m/s² on the 8 g nut, so a step that starts without contact lets the
  nut fall 120 µm (a·dt²), about the whole axial play. It digs into the flanks
  and chatters. The torque rises to 1.3–1.9× at μ ≥ 0.3 and 2.5–5.7× at
  μ = 0.1. With MuJoCo's default pyramidal cone at μ = 0.1 the thread
  back-drives, and the analytic SDF drops to 0.7–1.0×. A 0.25 ms step brings
  it back to 1.1–1.5×. Keep the load over the nut's mass, times the step
  squared, well under the axial play.
- **Bolt and nut friction combine as the larger.** MuJoCo gives a contact
  the larger of the two geoms' friction. Measured: a bolt at 0.3 against a
  nut at 1.0 breaks away at exactly the torque of both at 1.0.

## Renderings

`MUJOCO_GL=egl python -m benchmark.thread.render` writes three videos to
`../results/render/`:

- `thread_screw_c100.mp4` (T4);
- `thread_spin_c100.mp4` (T5);
- `thread_rundown_c100.mp4`, the nut-runner run-down. Add `--behaviors
  rundown --clearance 20` for `thread_rundown_c20.mp4`.
- `thread_wacky.mp4` (`--behaviors wacky`): nine wacky screws (see Wacky
  screws).
- `thread_undercut.mp4` (`--behaviors undercut`): dovetail and hooked teeth
  (see Undercut teeth).
- `thread_forms.mp4` (`--behaviors forms`) and `thread_compare_forms.mp4`
  (`--behaviors compare --forms all`): the run-down on other thread forms
  (see Thread forms).
- `thread_compare_c{100,20}.mp4` (`--behaviors compare --workers 1`), the
  same run-down as plain side-by-side renders, each labelled with its
  real-time factor. The three settings are each method's pick:
  - ours at 24 segments with 5 µm relief, which runs down cleanly at both
    clearances;
  - CoACD at threshold 0.01, its finest (every threshold jams);
  - the analytic SDF at 100 × 30 (the mesh SDF jams).

  The videos were rendered on improbable382, a Core Ultra 9 285K, pinned to
  one performance core (`taskset -c 1`, 5.8 GHz). The table gives real-time
  factors for the c = 100 µm run-down, over 1 s of steady run-down, pinned
  to one performance core on each machine. improbable381 is an i9-13900KS
  (`taskset -c 8`).

  | setting | 382: µs/step | 382: real time | 381: real time |
  |---|---|---|---|
  | ours 24 | 508 | 1.97× | 1.0× |
  | ours 24, 5 µm relief | 529 | 1.89× | 0.92× |
  | ours 48 | 985 | 1.01× | 0.50× |
  | ours 48, 5 µm relief | 1,114 | 0.90× | 0.43× |
  | ours 96, 5 µm relief | 2,188 | 0.46× | 0.22× |
  | SDF, analytic, 40 × 10 | 896 | 1.12× | 0.57× |
  | SDF, analytic, 100 × 30 | 2,208 | 0.45× | 0.23× |
  | SDF, analytic, 300 × 60 | 6,542 | 0.15× | 0.08× |
  | SDF, mesh, depth 9 (jammed) | 614 | 1.63× | 0.95× |
  | CoACD 0.02 (jammed) | 5,322 | 0.19× | 0.10× |
  | CoACD 0.01 (jammed) | 11,011 | 0.09× | 0.04× |

The screw and spin videos tile a coarse and a fine front setting per method.
The run-down video puts ours, at 24 and 96 segments with 5 µm seam relief,
above the mesh SDF (depth 9) and the analytic SDF (100 × 30). Each run-down
tile prints the µs per step that its simulation took. The render is the
simulation model itself:

- hulls for ours, CoACD and V-HACD;
- the mesh an SDF was built from;
- MuJoCo's surface mesh for the analytic SDF.

A stripe on the nut shows it turning, and arrows show the applied load and the
net contact force. Beside each render, a trace plots the motion against the
exact answer:

- **screwing:** the nut's offset from the helix against its exact seat;
- **spin-down:** the descent against the exact parabola;
- **run-down:** three traces:
  - turns against the runner's command;
  - the nut centre's offset from the helix against its seat;
  - the runner torque (40 ms means, log scale) against the screw formula
    and the runner's limit.

What the videos show:

- **Screwing:**
  - ours turns and follows the helix on its exact seat;
  - CoACD and V-HACD never turn;
  - the mesh SDF's nut slides 4–5 mm down through the thread;
  - the analytic SDF barely creeps at the default search effort and follows
    at 300 × 60.
- **Spin-down:**
  - the analytic SDF traces the exact parabola;
  - ours falls behind (44 % of the exact descent at 24 segments, 26 % at 96);
  - CoACD and V-HACD stay put;
  - the mesh SDF lurches up and down as it skips threads.
- **Run-down:**
  - At c = 100 µm, ours at both resolutions and the analytic SDF follow the
    runner. They ride their seat at the formula torque.
  - At c = 20 µm, ours at 96 segments catches: torque spikes of 10–20×,
    though it still makes the 4 turns.
  - The mesh SDF jams, cocks and climbs at both clearances.

## Thread forms

The geometry, oracles and representations take any of `mkscrew`'s thread
forms (see the root README), not just ISO. A `ThreadTask` carries these
fields:
- `form` and `form_params` (mkscrew's `thread_params`);
- `starts`, where the lead is starts × pitch;
- `handedness`;
- `chamfer`: mkscrew's `thread_chamfer`, at both rod ends and both nut
  faces. The exact SDFs, meshes and analytic SDF include the cones. The
  surface samplers drop thread the cones cut away, but don't sample the
  cones themselves.

Everything is built from mkscrew's teeth for that form:
- **Exact geometry:** the SDFs, sampled surfaces and watertight meshes.
- **Analytic SDF plugin:** it takes the tooth polygon as attributes.
- **ours:** mkscrew with that form.

ISO tasks keep their cache keys and give the same numbers as before.

[`forms.py`](forms.py) defines the tasks:
- **Five M10-sized forms** (D 10 mm, P 1.5 mm, c = 100 µm): ISO, trapezoidal
  30°, square, buttress 7°/45°, and the tight round wire.
- **The FurnitureBench leg** in its tabletop hole, as measured: round wire,
  P 9.375 mm, about ±1.6 mm of axial play, PLA. Its rod is lengthened to
  56 mm so the nut can make the same four turns.

The references are exact in the meridian plane, for any tooth:
- **`axial_gap`:** how far the centred nut slides before the teeth touch.
  That is c / cos a on a flank at angle a: c for square, c/cos 7° down and
  c/cos 45° up for buttress.
- **`seat_contact`:** where the nut then rests, and the tilt of the contact
  normal.
- **`rundown_torque`:** the screw formula with that tilt, at the pitch radius,
  or at the contact for a wire never half a pitch wide.

The nut runner's gains scale with each form's formula torque, so ISO keeps
its 5 mN·m limit.

`--behaviors forms` renders `thread_forms.mp4`: ours on every form, in the
run-down. Over 4 start angles each, ours at 24 segments per turn
(`benchmark/thread/forms.py` tasks) gives:

| form | frictionless spin-down, % of exact: no relief → 5 µm | run-down torque / formula, 5 µm relief |
|---|---|---|
| ISO | 57 % → 92 % | 1.06 |
| trapezoidal | 57 % → 98 % | 1.16 |
| square | 84 % → 97 % | 1.04 |
| buttress | 100 % → 99 % | 1.06 |
| round wire | 23 % → 87 % | 1.03 |
| FurnitureBench | 55 % → 69 % | 0.94 |

- **Seam relief helps the spin-down on every form but buttress.** Buttress
  is already exact without it. The internal-edge snag is the same wherever
  flanks slide across wedge seams.
- **The run-down stays within 6 % of the formula on most forms.**
  Trapezoidal runs 16 % over and FurnitureBench 6 % under. No setting
  spends more than 0.5 % of its 40 ms windows above 3×.
- **48 segments catch more than 24** on the flanked forms: trapezoidal 1.52×
  with 3.8 % of windows above 3×, square and buttress about 1.25×. That is
  the same pattern as ISO at 20 µm.

`--behaviors compare --forms all --workers 1` renders the run-down side by
side for each form, as `thread_compare_<form>.mp4`, and joins them into
`thread_compare_forms.mp4`. Each shows ours (24 segments, 5 µm relief),
CoACD at threshold 0.02 and the analytic SDF (100 × 30). The runs are pinned
to one performance core of improbable382 (Core Ultra 9 285K). All three are
given 4.22 turns by the runner.

| form | ours | CoACD 0.02 | SDF, analytic |
|---|---|---|---|
| ISO | 1.9× real time, 4.18 turns | 0.18×, jams (0 turns) | 0.45×, 4.19 turns |
| trapezoidal | 2.0×, 4.18 | 0.19×, jams | 0.44×, 4.18 |
| square | 2.6×, 4.18 | 0.18×, jams | 0.47×, 4.18 |
| buttress | 1.6×, 4.18 | 0.16×, jams | 0.43×, 4.18 |
| round wire | 2.2×, 4.18 | 0.27×, jams | 0.27×, 4.18 |
| FurnitureBench | 10×, 4.19 | 0.88×, 2.58 turns, then sticks | 0.21×, 4.19 |

- **Ours and the analytic SDF follow the runner on every form.** Ours runs
  4–10× faster than the SDF.
- **CoACD jams on all five M10 forms, and also at 0.05.** Its hulls bridge
  the helical groove, so the nut is held by phantom material.
- **CoACD half-works on the loose FurnitureBench thread.** Its 1.6 mm of
  play hides most of the phantom material, but the nut still sticks after
  2.6 turns.

CoACD at 0.05 and 0.02 took 5–8 and 20–40 minutes per form on improbable381.

### Wacky screws

[`wacky.py`](wacky.py) stretches the generator. `--behaviors wacky` renders
the nine designs in `thread_wacky.mp4`, 3 × 3. Each tile has the render, the
two teeth in the meridian plane, and a live readout. Ours runs every design
at 24 segments per turn with 5 µm relief, and each gets its own runner speed.

| design | what's wacky | turns (runner) | seat, µm (exact) |
|---|---|---|---|
| sawtooth 0°/60° | a flat ledge carries the load | 4.18 (4.22) | −99 (−100) |
| knife-edge 90° V | no flats at all | 4.19 (4.22) | −125 (−141) |
| six starts, 18 mm lead | back-drivable: the runner brakes | 1.28 (1.27) | −113 (−104) |
| left-hand double-start square | turns the other way | 4.19 (4.22) | −95 (−100) |
| auger | 5° blades 4.8 mm deep on a 6.4 mm core | 4.18 (4.22) | −108 (−100) |
| coil spring | loose 1.6 mm wire, 6 mm pitch | 2.48 (2.53) | −1818 (−1781) |
| house and dart | custom pentagon and triangle teeth | 4.18 (4.22) | −489 (−489) |
| fine pitch M10×0.5 | sixteen engaged turns | 4.18 (4.22) | −61 (−58) |
| left-hand triple-start wire | three wires, 6 mm lead | 2.51 (2.53) | −150 (−144) |

- **Every nut follows its runner** and rests within about 20 µm of its exact
  seat (37 µm on the loose coil, whose play is 1.8 mm).
- **No design needs anything beyond `thread_form`, `thread_params`,
  `n_starts` and `handedness`.**
- **The torque formula is only textbook.** The auger and the coil need 2×
  the screw-formula torque, because their deep blades and loose wire carry
  the load well outside the pitch radius the formula assumes. So the wacky
  tiles show whether the runner drives or brakes, not a formula ratio.

### Undercut teeth

`wacky.py`'s `UNDERCUT` designs are teeth that hook under each other. Each
pair exactly fills the 3 mm band between the core and the nut root at zero
clearance, on a 16 mm bolt with a 4 mm pitch and c = 100 µm. The dovetail's
teeth are convex (negative flank angles). The other three are non-convex,
and mkscrew cuts them into convex pieces. All four are chamfered 45° at
both rod ends and both nut faces (`chamfer` = the thread depth).

| design | pieces (bolt / nut tooth) | turns (runner) | seat, µm (exact) | µs/step |
|---|---|---|---|---|
| dovetail, 15° | 1 / 1 | 4.19 (4.22) | −107 (−104) | 385 |
| L-hooks | 2 / 2 | 4.18 (4.22) | −105 (−100) | 699 |
| T-slot | 2 / 2 | 4.19 (4.22) | −108 (−100) | 587 |
| barbed arrowheads | 3 / 2 | 4.19 (4.22) | −106 (−100) | 1,085 |

- **Why the chamfer.** Cut off flat, the dovetail left a thin blade at each
  rod end: its crest runs on about 40° further around than its root. The
  blade is part of the exact flat-cut part too, only a few tenths of a mm
  thick, and attached along the helix. The chamfer removes it
  (`test_chamfer_removes_the_dovetail_blade`).

- **The exact geometry takes any simple tooth.** The polygon distance is
  general, separation is tested piece by piece, and tube end caps are
  triangulated piece by piece. The analytic SDF plugin's polygon path
  computes the same general distance.
- **`test_wedges_cover_the_teeth`** checks that points well inside each
  tooth lie inside some wedge. Protrusion tests cannot see a missing piece;
  a wedge-volume check that dropped every T-bar wedge showed up only here
  (and as a 1.1 mm seat error).

## Probes (all passive, exact answers)

| probe | setup | exact answer |
|---|---|---|
| T1 axial | axial slide only, frictionless, ±0.1 N | play 2c/√3 each way |
| T2 radial | lateral slides only, frictionless, 0.1 N in 12 directions | play c/2 |
| T3 tilt | hinge through the nut centre, frictionless, 1.5 mN·m, two axes | exact maximum tilt (computed), error reported at the nut faces |
| T4 screw | axial slide + spin hinge, frictionless, 0.12 mN·m torque against a 0.1 N load | follows the helix z = P·θ/2π, seated 2c/√3 below it; outcome follows / jams / skips |
| T5 spin | free nut, gravity, frictionless (and friction 0.3) | descends at g/(1 + I(2π/P)²/m) (0.013 m/s²); at friction 0.3, holds |
| T6 start | nut resting on the bolt's top face; axial slide + spin hinge, frictionless, T4's 0.1 N load; a constant torque of 5× (and 2×) the exact back-off torque, 0.1 N × lead/2π. `back` unscrews, `on` screws | the quasi-static path `geometry.Tip.path`: `back` clicks (drops by the axial play 4c/√3) and rides the flank back up; `on` starts down the thread and follows the helix for a turn. Scored: event angle (as axial phase along the helix), click depth, height off the drops; outcome clicks / jams / no click / falls, or engages / jams / skips |
| T7 partial | T2 and T3 with the nut threaded ¼, ½, 1 and 2 turns onto the tip; 12 directions, 4 hinge axes, hinge half-way along the engaged length | on the sampled exact surfaces, end faces included: radial, the support along the push of the free sideways translations (a frictionless nut slides around a contact whose normal is off the push, as in the peg's S2), from first-contact rays (`partial_radial_play`); tilt, the first contact about the hinge (`partial_max_tilt`), its error at the thread crest |
| T8 torque | engaged nut, axial slide + spin hinge, no joint damping, axial load 0.1 or 1 N, friction μ = 0.1, 0.3, 1 (condim 3 or 4, pyramidal or elliptic cone, 1 or 0.25 ms step); the torque ramps until the nut turns 5°, against the load or with it | the power-screw formula (`geometry.screw_torque`), textbook rather than exact; outcome turns / holds / back-drives |

The exact geometry is in `geometry.py`. A point is inside the thread exactly
when its meridian coordinates (r, z − P·φ/2π mod P) lie in the tooth profile.
Distances are measured in that plane, which is within 0.1 % of the 3D
distance at a 3° helix angle.

- **Oracles:** `tests/test_benchmark_thread.py` checks them against the
  closed forms: the centred gap is c/2, axial play 2c/√3 and radial play c/2.
- **The nut on the tip, coaxial (T6), is exact without sampling.** At one
  angle φ each part is its meridian section, and turning φ slides both tooth
  patterns along z together while the end faces stay put. So the parts meet
  at some φ exactly when the teeth, at the nut's helical phase σ, overlap at
  a point (r, u) that both chamfers leave. With the bolt's cone cutting its
  thread (r − r_core)/k_b below the top face, and the nut's countersink
  (r_bore − r)/k_n above its base, the faces must overlap by more than their
  sum (`tip_overlap`). Without chamfers the sum is 0, so a misaligned nut
  sits on the top face. Collision at (angle, height) is then a 1-D test,
  and `Tip.path` follows the lowest free height as the nut turns. The tests
  check it against the sampled exact surfaces: they touch at the rest
  height, overlap 3 µm below and part 3 µm above, with and without
  chamfers.
- **Penetration and contact signed distance:** measured on sampled exact
  surfaces, as in the peg benchmark.

## Representations

- **ours:** `mkscrew.generate_screw_xml` with a cylindrical nut shell. The
  bolt is a primitive core cylinder plus wedges. The knob is segments per
  turn.
- **CoACD and V-HACD:** decompose watertight meshes of the exact bolt and
  nut. The meshes use `manifold3d` booleans with 360 steps per turn, and the
  nut's steps are circumscribed. CoACD runs with `preprocess_mode='off'` (see
  platform notes).
- **SDF, mesh:** the same meshes, each part a MuJoCo SDF geom with an octree
  of the given depth.
- **SDF, analytic:** `plugin/iso_thread_sdf.cc`, a MuJoCo SDF plugin that
  evaluates the exact distance (a C++ port of `geometry.py`), so its shape
  error is zero. It is compiled against the installed wheel on first use
  (`plugin/__init__.py`, needs g++). The knob is MuJoCo's SDF contact search
  (`sdf_initpoints` × `sdf_iterations`).

## Running

```bash
python -m benchmark.thread.run --parts probes                 # accuracy, all cores
python -m benchmark.thread.run --parts speed                  # timing: idle machine
python -m benchmark.thread.run --probes start partial        # T6, T7 only
python -m benchmark.thread.run --probes torque --methods ours sdf_analytic --clearance 100  # T8
python -m benchmark.thread.report                             # tables + figures -> ../report/
python -m benchmark.thread.report --traces                    # + thread_start.png (re-simulates)
MUJOCO_GL=egl python -m benchmark.thread.render               # videos -> ../results/render/
MUJOCO_GL=egl python -m benchmark.thread.render --behaviors rundown --clearance 20 --workers 1
```

`--workers 1` runs the simulations one at a time, which keeps the run-down's
µs/step labels clean.

- `--resume` runs only the jobs missing from `--out`.
- `--only-cached` skips CoACD and V-HACD settings that are not built yet.
- Each setting's probes start as soon as its decomposition is ready.

## Platform notes

- **CoACD's `preprocess_mode='auto'` aborts on the threaded bolt**
  (`std::logic_error: unexpected code path`), taking the process down. `'on'`
  re-voxelizes the part far coarser than the thread. The input meshes are
  watertight manifolds, so preprocessing is off.
- **`manifold3d` booleans are not reproducible** from run to run: the vertex
  counts of the same part differ. The exact meshes are built once per
  clearance and cached (`exact_meshes`), so every method starts from the same
  file.
- **The contact memory must be raised.** Fine wedges with MULTICCD produce
  5,000–22,000 contacts, and MuJoCo's default arena (sized from the model)
  overflows. The scenes set `spec.memory = 512 MB`.
- **The analytic plugin's memory is mostly its surface mesh.** MuJoCo builds a
  marching-cubes mesh for every plugin SDF (about 48 MB for this pair); the
  collision itself uses the analytic distance.
