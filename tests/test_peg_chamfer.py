"""Chamfered peg geometry, mass, export, and insertion contacts."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from conftest import hull_volume, inline_meshes
from scipy.spatial import ConvexHull

import mjfit
from mjfit import _polygon as pg
from mjfit import shapes
from mjfit.cli import main

L_SHAPE = np.array([
    [-0.01, -0.01], [0.01, -0.01], [0.01, 0], [0, 0], [0, 0.01], [-0.01, 0.01],
])
U_SHAPE = shapes.polygon([
    [-0.01, -0.01], [0.01, -0.01], [0.01, 0.01], [0.004, 0.01],
    [0.004, -0.004], [-0.004, -0.004], [-0.004, 0.01], [-0.01, 0.01],
], rotate=23)
CASES = [
    (shapes.rect(0.02, 0.016), []),
    (shapes.circle(0.01), []),
    (shapes.star(5, 0.014, 0.008), []),
    (L_SHAPE, []),
    (U_SHAPE, []),
    (shapes.rect(0.03, 0.02), [shapes.rect(0.01, 0.006)]),
    (shapes.rect(0.04, 0.04), [L_SHAPE]),
    (shapes.star(5, 0.02, 0.012), [
        shapes.rect(0.003, 0.004, center=(-0.004, 0)),
        shapes.circle(0.002, center=(0.004, 0)),
    ]),
]
CASE_NAMES = ['rectangle', 'circle', 'star', 'concave', 'u_shape', 'hollow',
              'concave_pocket', 'multiple_pockets']


def peg_meshes(xml):
    return {name: v for name, v in inline_meshes(xml).items() if name.startswith('peg_')}


@pytest.mark.parametrize('outline,islands', CASES, ids=CASE_NAMES)
def test_chamfer_sections_tile_the_exact_offset(outline, islands):
    """Probe the actual 3D hulls: no filled concavities, gaps, or overlaps."""
    clearance, bevel, length = 1e-4, 8e-4, 0.03
    xml = mjfit.peg(hole=outline, islands=islands, clearance=clearance,
                    chamfer=bevel, peg_length=length, center=False)
    meshes = peg_meshes(xml)
    hulls = [ConvexHull(v) for v in meshes.values()]
    model = mujoco.MjModel.from_xml_string(xml)
    for (name, _), hull in zip(meshes.items(), hulls):
        assert hull_volume(model, model.mesh(name).id) == pytest.approx(hull.volume, rel=2e-6)

    rng = np.random.default_rng(42)
    xy = rng.uniform(outline.min(axis=0) - 0.001, outline.max(axis=0) + 0.001, (2000, 2))
    for fraction in (0.01, 0.25, 0.5, 0.9, 1.1):
        distance = clearance + max(0, 1 - fraction) * bevel
        expected = pg.points_inside(pg.offset_inward(outline, distance), xy)
        for island in islands:
            expected &= ~pg.points_inside(pg.offset_outward(island, distance), xy)
        points = np.column_stack((xy, np.full(len(xy), -length / 2 + fraction * bevel)))
        inside_count = np.zeros(len(xy), dtype=int)
        for hull in hulls:
            planes = hull.equations
            inside_count += np.all(points @ planes[:, :3].T + planes[:, 3] < -1e-12, axis=1)
        np.testing.assert_array_equal(inside_count, expected.astype(int))

    # Integrating the quadratic area of parallel offsets is exact by Simpson's rule.
    def area(inset):
        return (pg.signed_area(pg.offset_inward(outline, clearance + inset))
                - sum(pg.signed_area(pg.offset_outward(p, clearance + inset)) for p in islands))

    expected_volume = bevel * (area(bevel) + 4 * area(bevel / 2) + area(0)) / 6
    depth = 0.01 + clearance
    expected_volume += (depth - bevel) * area(0)
    expected_volume += (length - depth) * pg.signed_area(pg.offset_inward(outline, clearance))
    if not islands and pg.is_convex(outline):
        assert len(hulls) == 1
    assert sum(h.volume for h in hulls) == pytest.approx(expected_volume, rel=1e-10)


@pytest.mark.parametrize('source', ['hole', 'peg'])
def test_square_bevel_dimensions_angle_and_volume(source):
    half, clearance, bevel, length = 0.005, 1e-4, 0.001, 0.03
    width = 2 * (half + clearance if source == 'hole' else half)
    xml = mjfit.peg(**{source: shapes.rect(width, width)}, clearance=clearance,
                    chamfer=bevel, peg_length=length)
    vertices = peg_meshes(xml)['peg_0']
    tip = vertices[np.isclose(vertices[:, 2], -length / 2)]
    np.testing.assert_allclose(np.abs(tip[:, :2]), half - bevel, atol=1e-15)
    assert vertices[:, 2].min() == -length / 2
    assert vertices[:, 2].max() == length / 2
    hull = ConvexHull(vertices)
    normals = np.unique(np.round(hull.equations[:, :3], 12), axis=0)
    slopes = normals[(normals[:, 2] < -0.1) & (normals[:, 2] > -0.9)]
    assert len(slopes) == 4
    np.testing.assert_allclose(np.linalg.norm(slopes[:, :2], axis=1), -slopes[:, 2])
    tip_half = half - bevel
    expected = 4 * half**2 * (length - bevel)
    expected += 4 * bevel * (half**2 + half * tip_half + tip_half**2) / 3
    assert hull.volume == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize('outline,islands', CASES, ids=CASE_NAMES)
def test_chamfer_export_mass_and_scaling(outline, islands, tmp_path):
    kwargs = dict(hole=outline + [0.004, -0.003],
                  islands=[p + [0.004, -0.003] for p in islands], center=False,
                  chamfer=8e-4, peg_mass=0.25, handle_height=0.01, bottom_thickness=0.002)
    xml = mjfit.peg(**kwargs)
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.body('peg').mass[0] == pytest.approx(0.25, abs=1e-12)
    np.testing.assert_allclose(model.body('peg').pos, [0, 0, 0.002 + 0.01 + 0.005 + 0.015])
    meshes = inline_meshes(xml)
    geoms = ET.fromstring(xml).findall(".//body[@name='peg']/geom")
    density = [float(g.get('mass')) / ConvexHull(meshes[g.get('mesh')]).volume for g in geoms]
    np.testing.assert_allclose(density, density[0], rtol=1e-10)

    scaled_xml = mjfit.peg(**kwargs, scale=3)
    for name, vertices in inline_meshes(scaled_xml).items():
        np.testing.assert_allclose(vertices, 3 * meshes[name], atol=1e-15)

    path = tmp_path / 'peg.xml'
    path.write_text(mjfit.peg(**kwargs, mesh_dir=tmp_path / 'peg_meshes'))
    exported = mujoco.MjModel.from_xml_path(str(path))
    np.testing.assert_allclose(exported.body('peg').mass, model.body('peg').mass, rtol=1e-7)
    np.testing.assert_allclose(exported.body('peg').ipos, model.body('peg').ipos, atol=1e-8)
    np.testing.assert_allclose(exported.body('peg').inertia, model.body('peg').inertia, rtol=1e-6)

    # The exported visual surface encloses exactly the collision pieces' volume.
    lines = (tmp_path / 'peg_meshes' / 'peg_visual.obj').read_text().splitlines()
    verts = np.array([line.split()[1:] for line in lines if line.startswith('v ')], float)
    faces = np.array([line.split()[1:] for line in lines if line.startswith('f ')], int) - 1
    a, b, c = verts[faces].transpose(1, 0, 2)
    visual_volume = np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6
    assert visual_volume == pytest.approx(
        sum(ConvexHull(v).volume for v in peg_meshes(xml).values()), rel=1e-10)


def test_chamfer_provides_clearance_at_first_entry():
    """An offset flat tip hits the rim; the bevel clears it until deeper entry."""
    def contacts(bevel, penetration):
        xml = mjfit.peg(peg=shapes.rect(0.01, 0.01), chamfer=bevel,
                        spawn_height=-penetration, floor=False)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        data.qpos[0] = 0.0005
        mujoco.mj_forward(model, data)
        return data.ncon

    assert contacts(0, 0.00025) > 0
    assert contacts(0.001, 0.00025) == 0
    assert contacts(0.001, 0.0008) > 0


@pytest.mark.parametrize('value', [-1e-4, float('nan'), float('inf'), '0.001', 0.03, 0.1])
def test_invalid_chamfer_values(value):
    with pytest.raises(ValueError, match='chamfer'):
        mjfit.peg(peg=shapes.rect(0.02, 0.02), chamfer=value)


@pytest.mark.parametrize('kwargs,match', [
    (dict(peg=shapes.rect(0.002, 0.004), chamfer=0.001), 'tip'),
    (dict(peg=L_SHAPE, chamfer=0.006), 'tip'),
    (dict(peg=shapes.rect(0.02, 0.02), include_peg=False, chamfer=0.001), 'include_peg'),
    (dict(hole=shapes.rect(0.02, 0.02), islands=[shapes.rect(0.01, 0.01)],
          clearance=0, chamfer=0.0025), 'wall'),
    (dict(hole=shapes.rect(0.03, 0.02), clearance=0, chamfer=0.001,
          islands=[shapes.rect(0.004, 0.004, center=(x, 0)) for x in (-0.003, 0.003)]),
     'mouths'),
    (dict(hole=shapes.rect(0.03, 0.02), islands=[shapes.rect(0.01, 0.004)],
          plate_thickness=0.0005, chamfer=0.0006), 'pocket depth'),
])
def test_chamfers_that_destroy_features_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        mjfit.peg(**kwargs)


def test_zero_chamfer_preserves_existing_output():
    for name, kind in mjfit.configs().items():
        if kind == 'peg':
            assert mjfit.build(name, chamfer=0) == mjfit.build(name)


def test_chamfer_config_and_cli_override(tmp_path):
    config = tmp_path / 'peg.yaml'
    config.write_text('kind: peg\nhole: {type: rect, width: 0.02, height: 0.02}\nchamfer: 0.0005\n')
    before = mjfit.build(config)
    output = tmp_path / 'peg.xml'
    main(['peg', '--config', str(config), '--chamfer', '0.001', '--export-meshes',
          '-o', str(output)])
    assert before != mjfit.build(config, chamfer=0.001)
    model = mujoco.MjModel.from_xml_path(str(output))
    expected = mujoco.MjModel.from_xml_string(mjfit.build(config, chamfer=0.001))
    np.testing.assert_allclose(model.body('peg').mass, expected.body('peg').mass, rtol=1e-6)
