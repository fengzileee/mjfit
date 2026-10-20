"""Render flat/chamfered comparisons for all bundled peg presets.

    pixi run chamfer-gallery
    pixi run python -m mujoco.viewer --mjcf=build/chamfer_gallery/usb_c.xml

The PNGs show the assembly and a view of the insertion face from below.
Each XML is a stationary, interactive pair: flat on the left, bevel on the right.
All dimensions come from the presets; only the viewing gap and chamfer change.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mjfit
from mjfit import _polygon as pg
from mjfit import shapes

# Sizes fit the original presets, including both sides of hollow peg walls.
CHAMFERS = {
    'circle': 0.001,
    'star': 0.001,
    'usb_a': 0.0001,
    'usb_c': 0.00025,
    'hdmi': 0.00001,
    'rj45': 0.0005,
    'db9': 0.0005,
    'xlr3': 0.00075,
    'rounded_convex': 0.001,
}
CARD_W, CARD_H = 1040, 820
PANEL_W, PANEL_H = 476, 300
BACKGROUND = '#f1f5f9'
INK = '#172b40'
MUTED = '#53677c'


def font(size, bold=False):
    names = (['/System/Library/Fonts/Supplemental/Arial Bold.ttf', 'DejaVuSans-Bold.ttf']
             if bold else ['/System/Library/Fonts/Supplemental/Arial.ttf', 'DejaVuSans.ttf'])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def studio(xml):
    """Light the real meshes from above and below, without adding a floor."""
    root = ET.fromstring(xml)
    ET.SubElement(root, 'option', gravity='0 0 0')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth=str(PANEL_W * 2), offheight=str(PANEL_H * 2))
    ET.SubElement(visual, 'quality', offsamples='4', shadowsize='2048')
    ET.SubElement(visual, 'headlight', ambient='0.3 0.3 0.3', diffuse='0.55 0.55 0.55',
                  specular='0.15 0.15 0.15')
    ET.SubElement(visual, 'map', znear='0.01')
    asset = root.find('asset')
    ET.SubElement(asset, 'texture', name='studio_sky', type='skybox', builtin='gradient',
                  rgb1='0.945 0.961 0.980', rgb2='0.945 0.961 0.980',
                  width='256', height='1536')
    world = root.find('worldbody')
    ET.SubElement(world, 'light', directional='true', pos='-1 -2 3', dir='1 2 -3',
                  diffuse='0.4 0.4 0.4', castshadow='false')
    ET.SubElement(world, 'light', directional='true', pos='1 -2 -3', dir='-1 2 3',
                  diffuse='0.65 0.65 0.65', castshadow='false')
    return ET.tostring(root, encoding='unicode')


def vertices(model, data, ids):
    points = []
    for geom in ids:
        mesh = model.geom_dataid[geom]
        if mesh < 0:
            continue
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        points.append(model.mesh_vert[start:start + count]
                      @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
    return np.vstack(points)


def camera_for(model, data, bounds, *, elevation, azimuth=105):
    """Fit the same physical bounds in each pair, with perspective padding."""
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.azimuth, camera.elevation = azimuth, elevation
    camera.lookat[:] = bounds.mean(axis=0)
    camera.distance = 1
    scene = mujoco.MjvScene(model, maxgeom=1)
    mujoco.mjv_updateCamera(model, data, camera, scene)
    forward = np.asarray(scene.camera[0].forward, float)
    up = np.asarray(scene.camera[0].up, float)
    right = np.cross(forward, up)
    relative = bounds - camera.lookat
    vertical = math.tan(math.radians(model.vis.global_.fovy / 2))
    horizontal = vertical * PANEL_W / PANEL_H
    distance = max(np.max(np.abs(relative @ right) / horizontal - relative @ forward),
                   np.max(np.abs(relative @ up) / vertical - relative @ forward))
    camera.distance = 1.16 * distance
    return camera


def box(low, high):
    return np.array(list(product(*zip(low, high))))


def frames(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    all_points = vertices(model, data, range(model.ngeom))
    peg_geoms = [g for g in range(model.ngeom)
                if model.geom(g).name.startswith('peg_')]
    peg_points = vertices(model, data, peg_geoms)
    low, high = peg_points.min(axis=0), peg_points.max(axis=0)
    depth = min((high[2] - low[2]) * 0.25, max(high[:2] - low[:2]) * 0.18)
    tip_high = high.copy()
    tip_high[2] = low[2] + depth
    return data, box(all_points.min(axis=0), all_points.max(axis=0)), box(low, tip_high)


def render_views(model, assembly_bounds, tip_bounds, macro_bounds=None):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    model.geom_rgba[:] = [0.48, 0.55, 0.65, 1]
    for g in range(model.ngeom):
        name = model.geom(g).name
        if name.startswith('peg_'):
            model.geom_rgba[g] = [0.84, 0.34, 0.16, 1]
            model.geom_group[g] = 0
        else:
            model.geom_group[g] = 1
        if name.startswith('handle_'):
            model.geom_rgba[g] = [0.27, 0.33, 0.41, 1]
    views = {}
    with mujoco.Renderer(model, height=PANEL_H * 2, width=PANEL_W * 2) as renderer:
        entries = [('assembly', assembly_bounds, -25), ('tip', tip_bounds, 65)]
        if macro_bounds is not None:
            entries.append(('detail', macro_bounds, 50))
        for name, bounds, elevation in entries:
            option = mujoco.MjvOption()
            if name != 'assembly':
                option.geomgroup[:] = 0
                option.geomgroup[0] = 1
            camera = camera_for(model, data, bounds, elevation=elevation)
            renderer.update_scene(data, camera=camera, scene_option=option)
            views[name] = Image.fromarray(renderer.render()).resize(
                (PANEL_W, PANEL_H), Image.Resampling.LANCZOS)
    return views


def hdmi_detail(config, tip_z, chamfer):
    """A true-scale crop at HDMI's thinnest wall; do not enlarge the chamfer."""
    outer = pg.offset_inward(shapes.from_spec(config['hole']), config['clearance'])
    pocket = pg.offset_outward(shapes.from_spec(config['islands'][0]), config['clearance'])
    # Centering is the area centroid of the supplied hole, not its bounding box.
    hole = shapes.from_spec(config['hole'])
    following = np.roll(hole, -1, axis=0)
    cross = hole[:, 0] * following[:, 1] - following[:, 0] * hole[:, 1]
    shift = ((hole + following) * cross[:, None]).sum(axis=0) / (3 * cross.sum())
    edges = np.roll(outer, -1, axis=0) - outer
    delta = pocket[:, None] - outer
    t = np.clip(np.einsum('ijk,jk->ij', delta, edges) / (edges * edges).sum(axis=1), 0, 1)
    nearest = outer + t[:, :, None] * edges
    gaps = np.linalg.norm(pocket[:, None] - nearest, axis=2)
    i, j = np.unravel_index(np.argmin(gaps), gaps.shape)
    center = np.r_[(pocket[i] + nearest[i, j]) / 2 - shift, tip_z + chamfer / 2]
    radius = chamfer * 6
    return box(center - [radius, radius, chamfer * 2],
               center + [radius, radius, chamfer * 3])


def write_pair(parts, output, spacing):
    scene = mjfit.grid(parts, spacing=spacing, floor=False)
    scene.option.gravity[:] = 0
    scene.compile()
    root = ET.fromstring(scene.to_xml())
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', azimuth='105', elevation='-20')
    xml = ET.tostring(root, encoding='unicode')
    mujoco.MjModel.from_xml_string(xml)
    output.write_text(xml + '\n')


def card(name, chamfer, flat, bevel):
    image = Image.new('RGB', (CARD_W, CARD_H), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((32, 20), name, fill=INK, font=font(36, True))
    label = f'45° chamfer · {chamfer * 1000:g} mm'
    draw.text((CARD_W - 32, 28), label, fill='#a54020', font=font(24), anchor='ra')
    for x, title, views in [(32, 'FLAT', flat), (532, 'CHAMFERED', bevel)]:
        draw.text((x, 78), title, fill=MUTED, font=font(20, True))
        image.paste(views['assembly'], (x, 108))
        draw.text((x, 426), 'INSERTION FACE · VIEW FROM BELOW', fill=MUTED, font=font(17, True))
        image.paste(views['tip'], (x, 456))
    draw.line((32, 778, CARD_W - 32, 778), fill='#cbd5e1', width=1)
    note = ('HDMI: very thin wall; see hdmi_detail.png for the 10 µm bevel.'
            if name == 'hdmi' else 'Matched camera and scale within each pair. Geometry is not exaggerated.')
    draw.text((32, 790), note, fill=MUTED, font=font(18))
    return image


def detail_card(flat, bevel):
    image = Image.new('RGB', (CARD_W, 450), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((32, 20), 'hdmi · thin-wall detail', fill=INK, font=font(32, True))
    draw.text((CARD_W - 32, 27), '10 µm chamfer · true proportions',
              fill='#a54020', font=font(22), anchor='ra')
    for x, title, views in [(32, 'FLAT', flat), (532, 'CHAMFERED', bevel)]:
        draw.text((x, 78), title, fill=MUTED, font=font(20, True))
        image.paste(views['detail'], (x, 108))
    draw.text((32, 422), 'Same small region at the outer wall / pocket corner, enlarged equally.',
              fill=MUTED, font=font(18))
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--configs', nargs='+', choices=CHAMFERS, default=list(CHAMFERS))
    parser.add_argument('-o', '--output', type=Path, default=Path('build/chamfer_gallery'))
    parser.add_argument('--no-render', action='store_true', help='only write interactive XML scenes')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cards, metadata = [], []
    for name in args.configs:
        chamfer, config = CHAMFERS[name], mjfit.load(name)
        spawn = max(0.005, config.get('peg_length', 0.03) * 0.4)
        parts = [mjfit.build(config, chamfer=c, spawn_height=spawn, standalone=False)
                 for c in (0, chamfer)]
        models = [mujoco.MjModel.from_xml_string(studio(p)) for p in parts]
        _, assembly_bounds, tip_bounds = frames(models[0])
        spacing = float(np.ptp(assembly_bounds[:, 0])) * 1.4
        write_pair(parts, args.output / f'{name}.xml', spacing)
        if not args.no_render:
            flat, bevel = [render_views(m, assembly_bounds, tip_bounds) for m in models]
            image = card(name, chamfer, flat, bevel)
            image.save(args.output / f'{name}.png')
            cards.append(image)
            if name == 'hdmi':
                macro = hdmi_detail(config, tip_bounds[:, 2].min(), chamfer)
                # Render the microscopic crop in millimetres so its near plane
                # stays above the renderer's numerical limit. Scale the whole
                # model and camera equally; the chamfer is never exaggerated.
                for view, c in zip((flat, bevel), (0, chamfer)):
                    detail_model = mujoco.MjModel.from_xml_string(studio(mjfit.build(
                        config, chamfer=c, spawn_height=spawn, standalone=False, scale=1000)))
                    detail_model.vis.map.znear = 0.0001
                    view['detail'] = render_views(
                        detail_model, assembly_bounds * 1000, tip_bounds * 1000,
                        macro * 1000)['detail']
                detail_card(flat, bevel).save(args.output / 'hdmi_detail.png')
        metadata.append(dict(config=name, chamfer_m=chamfer, spawn_height_m=spawn,
                             scene=f'{name}.xml', image=f'{name}.png'))
        print(f'{name}: {chamfer * 1000:g} mm chamfer; compiled both variants', flush=True)
    if cards:
        cols = min(3, len(cards))
        rows = math.ceil(len(cards) / cols)
        thumb_w, thumb_h, margin, header = 780, 615, 20, 100
        overview = Image.new('RGB', (cols * (thumb_w + margin) + margin,
                                    rows * (thumb_h + margin) + margin + header), '#ffffff')
        draw = ImageDraw.Draw(overview)
        draw.text((margin, 18), 'PEG TIP CHAMFERS', fill=INK, font=font(42, True))
        draw.text((margin, 67),
                  f'Flat / bevelled pairs · assembly + underside detail · {len(cards)} original presets',
                  fill=MUTED, font=font(24))
        for index, image in enumerate(cards):
            col, row = index % cols, index // cols
            overview.paste(image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS),
                           (margin + col * (thumb_w + margin),
                            header + margin + row * (thumb_h + margin)))
        overview.save(args.output / 'overview.png')
    (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Outputs: {args.output.resolve()}')


if __name__ == '__main__':
    main()
