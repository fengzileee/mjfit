"""Build a DB9 D-sub connector peg-in-hole scene.

Run from the repo root:
    python3 myscripts/examples/build_dsub.py

Writes peg_dsub.xml alongside this script. Drag into MuJoCo viewer to test.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gen_peg_in_hole import generate_xml

# D-shape outer: rectangle with top corners chamfered (keyed).
W, H, c = 0.020, 0.010, 0.0015
hole_outer = np.array([
    [-W/2, -H/2], [W/2, -H/2],
    [W/2, H/2 - c], [W/2 - c, H/2],
    [-W/2 + c, H/2], [-W/2, H/2 - c],
])

def circle(cx, cy, r, n=12):
    a = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([cx + r*np.cos(a), cy + r*np.sin(a)])

# 9 pin posts: 5 on bottom row, 4 staggered on top row.
pin_r, dx = 0.001, 0.0028
islands = ([circle(-2*dx + i*dx, -0.0015, pin_r) for i in range(5)] +
           [circle(-1.5*dx + i*dx,  0.0015, pin_r) for i in range(4)])

xml = generate_xml(
    hole_poly=hole_outer,
    hole_island_polys=islands,
    peg_clearance=0.0001,
    plate_size=0.05,
    plate_thickness=0.004,
    peg_length=0.015,
    peg_height=0.02,
)
out = Path(__file__).parent / 'peg_dsub.xml'
out.write_text(xml)
print(f'wrote {out}')
