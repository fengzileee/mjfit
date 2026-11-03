"""Convex hulls and solid Boolean verification for same-count peg chamfers."""
from __future__ import annotations

import math

import manifold3d as manifold
import numpy as np


def verified_hulls(pieces, surface, *, repair=None):
    """Hull each point cloud and verify its union against the intended solid.

    The reference is the independently constructed, closed chamfer surface.
    Boolean differences check coverage and containment. Overlap is allowed.
    No point or cross-section sampling is used. Normalize coordinates so the
    numerical tolerances do not depend on the model's units or translation.
    If supplied, repair returns exact seed pieces and unchanged back pieces.
    Seeds are merged into convex hulls within the original geom budget.
    """
    vertices, faces = surface
    origin = vertices.min(axis=0)
    size = float(np.ptp(vertices, axis=0).max())
    if not np.isfinite(vertices).all() or not math.isfinite(size) or size <= 0:
        _fail('invalid reference surface')

    # The surface writer duplicates vertices at face boundaries. Weld identical
    # positions before asking Manifold to validate the closed, oriented mesh.
    unique, indices = np.unique((vertices - origin) / size, axis=0, return_inverse=True)
    reference = manifold.Manifold(manifold.Mesh64(
        np.ascontiguousarray(unique), np.ascontiguousarray(indices[faces], dtype=np.uint64)))
    _check_solid(reference, 'reference surface')
    tolerance = max(64 * np.finfo(float).eps, 1e-10 * reference.volume())

    def hull(points):
        if not np.isfinite(points).all():
            _fail('non-finite collision vertices')
        result = manifold.Manifold.hull_points(np.ascontiguousarray((points - origin) / size))
        _check_solid(result, 'collision hull')
        return result

    hulls = [hull(points) for points in pieces]
    reason = _union_error(hulls, reference, tolerance)
    if reason is not None and repair is not None:
        seeds, fixed = repair()
        hulls = _merge_cover([hull(p) for p in seeds], len(pieces) - len(fixed),
                            reference, tolerance)
        hulls += [hull(p) for p in fixed]
        if len(hulls) != len(pieces):
            _fail('repair changed the collision geom count')
        reason = _union_error(hulls, reference, tolerance)
    if reason is not None:
        _fail(reason)
    result = []
    for hull in hulls:
        mesh = hull.to_mesh64()
        result.append((np.asarray(mesh.vert_properties)[:, :3] * size + origin,
                       np.asarray(mesh.tri_verts, dtype=int)))
    return result


def _union_error(hulls, reference, tolerance):
    union = manifold.Manifold.batch_boolean(hulls, manifold.OpType.Add)
    _check_solid(union, 'collision union')
    for difference, reason in ((union - reference, 'collision hulls add material'),
                               (reference - union, 'collision hulls leave gaps')):
        _check_status(difference)
        if not math.isfinite(difference.volume()) or difference.volume() > tolerance:
            return reason
    return None


def _merge_cover(hulls, budget, reference, tolerance):
    """Merge exact pieces without introducing material outside the target.

    Unlike a convex partition, the merged hull may overlap other pieces or
    cover material between its inputs. That is safe when it stays inside the
    target: taking hulls cannot remove any of the seeds' original coverage.
    """
    hulls.sort(key=lambda h: -h.volume())

    def bounds(hull):
        vertices = np.asarray(hull.to_mesh64().vert_properties)[:, :3]
        return vertices.min(axis=0), vertices.max(axis=0)

    boxes = [bounds(hull) for hull in hulls]
    while len(hulls) > budget:
        before = len(hulls)
        for i in range(len(hulls)):
            if i >= len(hulls) or len(hulls) <= budget:
                break
            j = i + 1
            while j < len(hulls) and len(hulls) > budget:
                lo = np.maximum(boxes[i][0], boxes[j][0])
                hi = np.minimum(boxes[i][1], boxes[j][1])
                if np.any(lo > hi + 1e-12):
                    j += 1
                    continue
                merged = manifold.Manifold.batch_hull([hulls[i], hulls[j]])
                _check_solid(merged, 'merged collision hull')
                extra = merged - reference
                _check_status(extra)
                if not math.isfinite(extra.volume()) or extra.volume() > tolerance:
                    j += 1
                    continue
                hulls[i] = merged
                boxes[i] = bounds(merged)
                hulls.pop(j)
                boxes.pop(j)
                # The enlarged hull can now touch an earlier skipped piece.
                j = i + 1
        if len(hulls) == before:
            _fail('could not repair geometry within the original collision geom count')
    return hulls


def _check_status(solid):
    if solid.status() != manifold.Error.NoError:
        _fail(f'geometry verification failed ({solid.status().name})')


def _check_solid(solid, name):
    _check_status(solid)
    if solid.is_empty() or not math.isfinite(solid.volume()) or solid.volume() <= 0:
        _fail(f'invalid {name}')


def _fail(reason):
    raise ValueError(f'failed to create chamfer: {reason}')
