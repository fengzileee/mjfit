"""Compare a flat square peg with a chamfered peg in a stationary scene.

    pixi run chamfer-demo
    pixi run python -m mujoco.viewer --mjcf=build/chamfer_comparison.xml

Gravity is disabled so the insertion tips stay visible above their plates.
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

import mjfit

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument('--chamfer', type=float, default=0.001, help='45-degree chamfer length, m')
parser.add_argument('-o', '--output', type=Path, default=Path('build/chamfer_comparison.xml'))
args = parser.parse_args()

parts = [
    mjfit.peg(
        peg=mjfit.shapes.rect(0.01, 0.01),
        peg_length=0.02,
        clearance=0.0001,
        chamfer=bevel,
        spawn_height=0.008,
        standalone=False,
    )
    for bevel in (0.0, args.chamfer)
]
scene = mjfit.grid(parts, spacing=0.035)
scene.option.gravity[:] = 0
scene.compile()
root = ET.fromstring(scene.to_xml())
visual = root.find('visual')
if visual is None:
    visual = ET.SubElement(root, 'visual')
camera = visual.find('global')
if camera is None:
    camera = ET.SubElement(visual, 'global')
camera.set('azimuth', '90')
camera.set('elevation', '-15')
stats = root.find('statistic')
if stats is None:
    stats = ET.SubElement(root, 'statistic')
stats.set('center', '0 0.0175 0.02')
stats.set('extent', '0.07')
xml = ET.tostring(root, encoding='unicode')
mujoco.MjModel.from_xml_string(xml)  # Validate the complete inspection scene.
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(xml + '\n')
print(f'wrote {args.output}')
print(f'Left: flat tip. Right: {args.chamfer * 1000:g} mm chamfer. Both shafts are 10 mm wide.')
