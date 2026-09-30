"""Native geometry, physical mass, and compatibility checks for convex assemblies."""

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

import mkpeg

CONTROL = np.array([[-.022, -.013], [.024, -.010], [.019, .017], [-.017, .022]])


def outlines():
    shift = np.array([.013, -.007])
    hole = mkpeg.bezier_polygon(CONTROL, [.5, 0, .25, .5]) + shift
    handle = mkpeg.bezier_polygon(CONTROL * 1.8, [.25, .5, .5, .4]) + shift
    plate = mkpeg._offset_polygon_outward(handle, .006)
    return hole, handle, plate


def mesh_vertices(xml, name):
    mesh = ET.fromstring(xml).find(f"./asset/mesh[@name='{name}']")
    return np.fromstring(mesh.attrib['vertex'], sep=' ').reshape(-1, 3)


def compiled_vertices(model, data, name):
    mesh = model.mesh(name).id
    geom = np.flatnonzero(model.geom_dataid == mesh)[0]
    vertices = model.mesh_vert[model.mesh_vertadr[mesh]:
                               model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
    return vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def test_bezier_corner_order_and_chord_error():
    alphas = np.array([.5, 0, .25, .4])
    tolerance = 0.0001
    polygon = mkpeg.bezier_polygon(CONTROL, alphas, tolerance)
    np.testing.assert_array_equal(
        mkpeg.bezier_polygon(CONTROL[::-1], alphas[::-1], tolerance), polygon)
    np.testing.assert_array_equal(mkpeg.bezier_polygon(CONTROL, 0), CONTROL)
    starts, ends = polygon, np.roll(polygon, -1, axis=0)
    edges = ends - starts
    for index, alpha in enumerate(alphas):
        vertex = CONTROL[index]
        entry = vertex + alpha * (CONTROL[index - 1] - vertex)
        exit = vertex + alpha * (CONTROL[(index + 1) % len(CONTROL)] - vertex)
        t = np.linspace(0, 1, 501)[:, None]
        curve = (1-t)**2 * entry + 2*t*(1-t)*vertex + t**2 * exit
        fraction = np.clip(np.einsum('pei,ei->pe', curve[:, None] - starts, edges)
                           / np.sum(edges**2, axis=1), 0, 1)
        closest = starts + fraction[..., None] * edges
        error = np.linalg.norm(curve[:, None] - closest, axis=2).min(axis=1)
        assert error.max() <= tolerance * (1 + 1e-10)


@pytest.mark.parametrize('rounding,tolerance', [(-.1, .0001), (.51, .0001),
                                               ([.5, 0], .0001), (.5, 0)])
def test_bad_bezier_settings_rejected(rounding, tolerance):
    with pytest.raises(ValueError):
        mkpeg.bezier_polygon(CONTROL, rounding, tolerance)


@pytest.mark.parametrize('mass', [.35, .675])
@pytest.mark.parametrize('center', [False, True])
def test_curved_nested_geometry_mass_and_common_frame(mass, center):
    hole, handle, plate = outlines()
    clearance = .00123456789
    peg = mkpeg._offset_polygon_inward(hole, clearance)
    shift = hole.mean(axis=0) if center else np.zeros(2)
    xml = mkpeg.generate_xml(
        hole_poly=hole, handle_poly=handle, plate_poly=plate,
        peg_clearance=clearance, peg_length=.045, handle_height=.025,
        peg_mass_kg=mass, plate_min_rim_m=.005, bottom_stop_thickness=.002,
        center_shapes=center, include_floor=False, peg_height=-.02,
    )
    for name, expected in [('peg_0', peg), ('handle_0', handle), ('bottom_stop', plate)]:
        actual = mesh_vertices(xml, name)
        np.testing.assert_allclose(actual[:len(expected), :2], expected - shift, atol=1e-16)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.body_mass[model.body('peg').id] == pytest.approx(mass, abs=1e-14)
    native = compiled_vertices(model, data, 'peg_0')
    lateral = clearance * .3 if center else 0.0
    expected = mesh_vertices(xml, 'peg_0') + (lateral, 0, 0)
    error = np.linalg.norm(native[:, None] - expected[None], axis=2).min(axis=1)
    assert error.max() < 1e-8
    floor = compiled_vertices(model, data, 'bottom_stop')
    assert floor[:, 2].min() == pytest.approx(-.002, abs=1e-9)
    assert floor[:, 2].max() == pytest.approx(0, abs=1e-9)
    # Per-geom masses share one material density over X and Y.
    geoms = ET.fromstring(xml).findall("./worldbody/body[@name='peg']/geom")
    masses = {geom.attrib['mesh']: float(geom.attrib['mass']) for geom in geoms}
    density_x = masses['peg_0'] / (mkpeg._signed_area(peg) * .045)
    density_y = masses['handle_0'] / (mkpeg._signed_area(handle) * .025)
    assert density_x == pytest.approx(density_y, rel=1e-12)
    # Decomposed plate area equals Z minus the aperture; no filled-in hole.
    plate_area = sum(mkpeg._signed_area(mesh_vertices(xml, mesh.attrib['name'])[:
                         len(mesh_vertices(xml, mesh.attrib['name'])) // 2, :2])
                     for mesh in ET.fromstring(xml).findall('./asset/mesh')
                     if mesh.attrib['name'].startswith('plate_'))
    assert plate_area == pytest.approx(mkpeg._signed_area(plate) - mkpeg._signed_area(hole))


def test_rotated_asymmetric_plate_and_rim_rejection():
    hole, handle, _ = outlines()
    angle = .63
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    plate = np.array([[-.11, -.09], [.14, -.09], [.14, .12], [-.11, .12]]) @ rotation.T
    xml = mkpeg.generate_xml(hole_poly=hole, handle_poly=handle, plate_poly=plate,
                             handle_height=.03, plate_min_rim_m=.005,
                             center_shapes=False, bottom_stop_thickness=.002)
    mujoco.MjModel.from_xml_string(xml)
    np.testing.assert_allclose(mesh_vertices(xml, 'bottom_stop')[:4, :2], plate, atol=1e-16)
    with pytest.raises(ValueError, match='rim'):
        mkpeg.generate_xml(hole_poly=hole, handle_poly=handle,
                            plate_poly=mkpeg._offset_polygon_outward(handle, .004),
                            handle_height=.03, plate_min_rim_m=.005)


def test_analytic_cylinder_handle_rim_and_total_mass():
    # A diamond at 45 degrees must clear the entire cylinder, including the
    # arc between the deliberately sparse four aperture sample points.
    plate = np.array([[.05, 0], [0, .05], [-.05, 0], [0, -.05]])
    xml = mkpeg.generate_xml(shape='circle', hole_size=.025, peg_clearance=.002,
                             handle_height=.03, handle_margin=.006,
                             plate_poly=plate, plate_min_rim_m=.005,
                             peg_mass_kg=.35, n_circle=4)
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.body_mass[model.body('peg').id] == pytest.approx(.35)
    with pytest.raises(ValueError, match='Plate around handle'):
        mkpeg.generate_xml(shape='circle', hole_size=.025, peg_clearance=.002,
                            handle_height=.03, handle_margin=.006,
                            plate_poly=plate, plate_min_rim_m=.010, n_circle=4)


def test_collapsed_offset_and_noncontaining_handle_rejected():
    hole = np.array([[-.01, -.005], [.01, -.005], [.01, .005], [-.01, .005]])
    with pytest.raises(ValueError):
        mkpeg.generate_xml(hole_poly=hole, peg_clearance=.006, center_shapes=False)
    with pytest.raises(ValueError):
        mkpeg.generate_xml(hole_poly=hole, peg_clearance=.001, handle_height=.03,
                            handle_margin=-.020, center_shapes=False)
    with pytest.raises(ValueError, match='Handle'):
        mkpeg.generate_xml(hole_poly=hole, handle_poly=hole * .1, handle_height=.03)


def test_config_cli_and_export(tmp_path):
    config = {
        'clearance': .001, 'plate_thickness': .02, 'peg_length': .045,
        'handle_height': .025, 'peg_mass_kg': .55, 'center_shapes': False,
        'bottom_stop_thickness': .002, 'plate_min_rim_m': .005,
        'include_floor': False,
        'peg': {'type': 'bezier_polygon', 'vertices': CONTROL.tolist(),
                'corner_rounding': [.5, 0, .25, .5]},
        'handle': {'type': 'bezier_polygon', 'vertices': (CONTROL * 1.8).tolist(),
                   'corner_rounding': .5},
        'plate': {'type': 'bezier_polygon', 'vertices': (CONTROL * 3.0).tolist(),
                  'corner_rounding': [.5, .3, 0, .4]},
    }
    inline = mkpeg.generate_from_config(config)
    model = mujoco.MjModel.from_xml_string(inline)
    assert model.body_mass[model.body('peg').id] == pytest.approx(.55)
    xml_path = tmp_path / 'scene.xml'
    xml_path.write_text(mkpeg.generate_from_config(config, mesh_dir=tmp_path, scale=1.25))
    exported = mujoco.MjModel.from_xml_path(str(xml_path))
    assert exported.body_mass[exported.body('peg').id] == pytest.approx(.55)
    hole, handle, plate = outlines()
    for name, values in [('hole', hole), ('handle', handle), ('plate', plate)]:
        np.savetxt(tmp_path / f'{name}.csv', values, delimiter=',')
    mkpeg.main(['--hole-csv', str(tmp_path/'hole.csv'), '--handle-csv', str(tmp_path/'handle.csv'),
                '--plate-csv', str(tmp_path/'plate.csv'), '--handle-height', '.025',
                '--peg-mass-kg', '.675', '--plate-min-rim-m', '.005',
                '--bottom-stop-thickness', '.002', '--preserve-origin', '-o', str(xml_path)])
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert model.body_mass[model.body('peg').id] == pytest.approx(.675)


def test_legacy_serialization_unchanged():
    # Default circle: hole n-gon circumscribes hole_size (flats at nominal radius).
    cases = [({}, 'db18fc3f48952291aeed43d483544f027793a65323c815b9e13753bc5d82f846'),
             ({'shape': 'star'}, '58d91cd12ff1f46d3cc664c7c5ce2256bf1c1149880b0edcb9231c23adf0bc81'),
             ({'shape': 'slot', 'handle_height': .025, 'peg_density': 350.0},
              '2b2449539ee395bee90d897bcd1e169db8c62ee752dc9a4b1bc600694c381581')]
    for kwargs, expected in cases:
        assert hashlib.sha256(mkpeg.generate_xml(**kwargs).encode()).hexdigest() == expected
    path = Path(__file__).resolve().parents[1] / 'configs/connectors/usb_a.yaml'
    xml = mkpeg.generate_from_config(mkpeg._load_config(path), path.parent)
    assert hashlib.sha256(xml.encode()).hexdigest() == 'fbd774abcf198dcaebdc27fcd830f33203999c129924321bef50d402bd503216'
