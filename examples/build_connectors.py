"""Generate MuJoCo MJCF scenes for a handful of popular connectors.

Dimensions are taken from manufacturer/Wikipedia specifications (documented
inline per connector). Real connectors are complex — these are simulation-
appropriate simplifications that capture the insertion-plane cross-section
and the keying/island features that make each connector distinctive.

Run from repo root:
    python3 myscripts/examples/build_connectors.py

Generates peg_usbc.xml, peg_hdmi.xml, peg_xlr3.xml, peg_rj45.xml alongside
this file.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mkpeg import generate_xml


HERE = Path(__file__).parent


def stadium(w, h, n_cap=16):
    """Rounded-rectangle / stadium polygon. w, h are full width/height; the
    short dimension becomes the diameter of the end caps."""
    r = min(w, h) / 2
    # straight length along the long axis
    if w >= h:
        straight = w - 2 * r
        # right cap (angles -90..+90)
        a = np.linspace(-np.pi / 2, np.pi / 2, n_cap)
        right = np.column_stack([straight / 2 + r * np.cos(a), r * np.sin(a)])
        a = np.linspace(np.pi / 2, 3 * np.pi / 2, n_cap)
        left = np.column_stack([-straight / 2 + r * np.cos(a), r * np.sin(a)])
        return np.vstack([right, left])
    else:
        straight = h - 2 * r
        a = np.linspace(0, np.pi, n_cap)
        top = np.column_stack([r * np.cos(a), straight / 2 + r * np.sin(a)])
        a = np.linspace(np.pi, 2 * np.pi, n_cap)
        bot = np.column_stack([r * np.cos(a), -straight / 2 + r * np.sin(a)])
        return np.vstack([top, bot])


def rect(w, h, cx=0.0, cy=0.0):
    return np.array([[cx - w/2, cy - h/2], [cx + w/2, cy - h/2],
                     [cx + w/2, cy + h/2], [cx - w/2, cy + h/2]])


def circle(cx, cy, r, n=20):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


# ----- USB-C -----------------------------------------------------------
# Receptacle opening: 8.34 × 2.56 mm stadium.
# Internal tongue is ~0.7 mm thick with 24 contacts; for insertion
# simulation we model it as a centered thin oval island.
def build_usbc():
    W, H = 0.00834, 0.00256
    opening = stadium(W, H, n_cap=12)
    # Thin tongue: 6.5 × 0.7 mm centered
    tongue = stadium(0.0065, 0.00070, n_cap=8)
    xml = generate_xml(
        hole_poly=opening,
        hole_island_polys=[tongue],
        peg_clearance=0.00005,   # 50 µm (tight)
        plate_size=0.04,
        plate_thickness=0.0062,  # 6.2 mm USB-C receptacle depth
        peg_length=0.012,
        peg_height=0.015,
    )
    (HERE / 'peg_usbc.xml').write_text(xml)
    print('wrote peg_usbc.xml')


# ----- HDMI Type A ----------------------------------------------------
# Receptacle 14 × 4.55 mm. Slightly trapezoidal (top narrower than bottom
# by a few tenths of a mm, used for plug orientation). 19-pin internal
# tongue roughly 13 × 1 mm offset toward one side.
def build_hdmi():
    # Trapezoid: wider at bottom (15 × 4.55), top edge 13 × 4.55
    bot_w, top_w, H = 0.014, 0.0128, 0.00455
    opening = np.array([
        [-bot_w/2, -H/2],
        [ bot_w/2, -H/2],
        [ top_w/2,  H/2],
        [-top_w/2,  H/2],
    ])
    tongue = rect(0.0126, 0.00100, cx=0.0, cy=0.00100)  # offset upward
    xml = generate_xml(
        hole_poly=opening,
        hole_island_polys=[tongue],
        peg_clearance=0.00008,
        plate_size=0.05,
        plate_thickness=0.007,
        peg_length=0.014,
        peg_height=0.018,
    )
    (HERE / 'peg_hdmi.xml').write_text(xml)
    print('wrote peg_hdmi.xml')


# ----- XLR 3-pin -----------------------------------------------------
# 14 mm diameter cylindrical shell. 3 pins arranged in an equilateral
# triangle, ~6.35 mm pitch; pin diameter ~2.0 mm. Iconic pro-audio
# connector — shows circular-outer + N circular islands.
def build_xlr3():
    shell_r = 0.007  # 14 mm diameter
    opening = circle(0, 0, shell_r, n=32)
    pin_r = 0.001    # 2 mm diameter
    pitch = 0.00345  # center-to-center from shell center to pin
    # 3 pins equilateral triangle
    angles = [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]
    islands = [circle(pitch*np.cos(a), pitch*np.sin(a), pin_r, n=16)
               for a in angles]
    xml = generate_xml(
        hole_poly=opening,
        hole_island_polys=islands,
        peg_clearance=0.0001,
        plate_size=0.06,
        plate_thickness=0.008,
        peg_length=0.016,
        peg_height=0.02,
    )
    (HERE / 'peg_xlr3.xml').write_text(xml)
    print('wrote peg_xlr3.xml')


# ----- RJ45 / 8P8C ----------------------------------------------------
# Plug body ~11.68 × 6.50 mm. The keying is a plastic latch tab on one
# side — in the insertion cross-section we approximate this as a
# trapezoidal/tab-cut outer profile so the plug is one-way.
def build_rj45():
    W, H = 0.01168, 0.0065
    # Latch-tab cutout on the top edge: narrow notch, 3 mm wide × 1.0 mm deep
    tab_w, tab_d = 0.0030, 0.0010
    opening = np.array([
        [-W/2, -H/2],
        [ W/2, -H/2],
        [ W/2,  H/2],
        [ tab_w/2,  H/2],
        [ tab_w/2,  H/2 - tab_d],
        [-tab_w/2,  H/2 - tab_d],
        [-tab_w/2,  H/2],
        [-W/2,  H/2],
    ])
    xml = generate_xml(
        hole_poly=opening,
        peg_clearance=0.0001,
        plate_size=0.05,
        plate_thickness=0.008,
        peg_length=0.016,
        peg_height=0.02,
    )
    (HERE / 'peg_rj45.xml').write_text(xml)
    print('wrote peg_rj45.xml')


if __name__ == '__main__':
    build_usbc()
    build_hdmi()
    build_xlr3()
    build_rj45()
