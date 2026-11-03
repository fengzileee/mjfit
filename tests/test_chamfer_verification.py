"""Adversarial checks of full-solid verification, independent of peg construction."""
import numpy as np
import pytest
from scipy.spatial import ConvexHull

from mjfit import _mjcf as mj
from mjfit import shapes
from mjfit._chamfer import verified_hulls


def box(x0, x1):
    outline = np.array([[x0, 0], [x1, 0], [x1, 1], [x0, 1]], float)
    return mj.extrusion_triangles([(outline, [], 0, 1)])


@pytest.mark.parametrize('scale', [1e-4, 1, 1e4])
def test_shared_faces_are_valid(scale):
    vertices, faces = box(0, 1)
    shift = np.array([2, -3, 4]) * scale
    pieces = [box(0, 0.5)[0], box(0.5, 1)[0]]
    result = verified_hulls([p * scale + shift for p in pieces],
                            (vertices * scale + shift, faces))
    assert len(result) == 2
    assert sum(ConvexHull(v).volume for v, _ in result) == pytest.approx(scale**3)


@pytest.mark.parametrize('intervals,reason', [
    ([(0.1, 1.1)], 'add material'),  # Same volume as reference, wrong location.
    ([(0, 0.5), (0.50001, 1)], 'leave gaps'),  # Thin gap sampling could miss.
    ([(0, 0.6), (0.5, 0.9)], 'leave gaps'),  # Gap and overlap cancel in summed volume.
])
@pytest.mark.parametrize('scale', [1e-4, 1, 1e4])
def test_invalid_decompositions(intervals, reason, scale):
    vertices, faces = box(0, 1)
    with pytest.raises(ValueError, match=f'failed to create chamfer: collision hulls {reason}'):
        verified_hulls([box(a, b)[0] * scale for a, b in intervals],
                       (vertices * scale, faces))


@pytest.mark.parametrize('intervals', [[(0, 0.6), (0.5, 1)], [(0, 1), (0, 1)]])
def test_overlapping_hulls_are_valid(intervals):
    result = verified_hulls([box(a, b)[0] for a, b in intervals], box(0, 1))
    assert len(result) == len(intervals)
    assert sum(ConvexHull(v).volume for v, _ in result) > 1


def test_hulling_a_pocket_is_rejected():
    surface = mj.extrusion_triangles([
        (shapes.rect(2, 2), [shapes.rect(1, 1)], 0, 1)])
    with pytest.raises(ValueError, match='failed to create chamfer: collision hulls add material'):
        verified_hulls([surface[0]], surface)


def test_open_reference_is_rejected():
    vertices, faces = box(0, 1)
    with pytest.raises(ValueError, match='failed to create chamfer: geometry verification failed'):
        verified_hulls([vertices], (vertices, faces[:-1]))


def test_degenerate_hull_is_rejected():
    vertices, faces = box(0, 1)
    with pytest.raises(ValueError, match='failed to create chamfer: invalid collision hull'):
        verified_hulls([vertices[vertices[:, 2] == 0]], (vertices, faces))


def test_repair_merges_seeds_within_original_count():
    result = verified_hulls([box(0, 1.1)[0]], box(0, 1),
                            repair=lambda: ([box(0, 0.5)[0], box(0.5, 1)[0]], []))
    assert len(result) == 1
    assert ConvexHull(result[0][0]).volume == pytest.approx(1)


def test_repair_cannot_exceed_original_count():
    outline = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)
    surface = mj.extrusion_triangles([(outline, [], 0, 1)])
    seeds = [box(0, 2)[0], box(0, 1)[0] + [0, 1, 0]]
    with pytest.raises(ValueError, match='failed to create chamfer: could not repair geometry'):
        verified_hulls([surface[0]], surface, repair=lambda: (seeds, []))


def test_repair_is_verified_after_merging():
    with pytest.raises(ValueError, match='failed to create chamfer: collision hulls leave gaps'):
        verified_hulls([box(0, 1.1)[0]], box(0, 1),
                       repair=lambda: ([box(0, 0.9)[0]], []))
