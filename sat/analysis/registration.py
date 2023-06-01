"""Provide algorithms to align and assign two meshes to each other"""

import trimesh
import numpy as np
from typing import Tuple, Generator
from scipy.spatial import cKDTree
from tqdm import tqdm

from . import progress

def invT(T: np.ndarray[(4, 4), float]):
    """Inverse 4x4 transformation matrix"""

    U = T[:3, :3].T  # reverse rotation
    inv = np.eye(4)
    inv[:3, :3] = U
    inv[:3,  3] = - U @ T[:3, 3]
    return inv


def align(
    nom: trimesh.Trimesh,
    act: trimesh.Trimesh
) -> np.ndarray:
    """Align two meshes via ICP.

    Parameters
    ----------
    nom, act : trimesh.Trimesh
        Triangulated meshes to align

    Return
    ------
    H : (4, 4) float
        Estimated transformation matrix
    
    Notes
    -----
    trimesh allows rotation matrices with det(R) = -1, which are actually
    mirror matrices
    """
    H, _ = trimesh.registration.mesh_other(nom, act)
    return H


def nearby_faces(
    mesh: trimesh.Trimesh, 
    points: np.ndarray
) -> Generator[list, None, None]:
    """For each point find nearby faces relatively quickly.

    Algorithm from trimesh.proximity.nearby_faces()
    Yield results instead of returning the entire list!

    The closest point on the mesh to the queried point is guaranteed to be
    on one of the faces listed.
    Does this by finding the nearest vertex on the mesh to each point, and
    then returns all the faces that intersect the axis aligned bounding box
    centered at the queried point and extending to the nearest vertex.


    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to query.
    points : (n, 3) float
        Points in space

    Yields
    ------
    candidates : (n, ) int
        Indices for candidates in mesh.faces
    """
    points = np.asanyarray(points, dtype=float)
    if not trimesh.util.is_shape(points, (-1, 3)):
        raise ValueError('points must be (n, 3)!')

    # an r-tree containing the axis aligned bounding box for every triangle
    rtree = mesh.triangles_tree
    # a kd-tree containing every vertex of the mesh
    kdtree = cKDTree(mesh.vertices[mesh.referenced_vertices])

    # query the distance to the nearest vertex to get AABB of a sphere
    distance_vertex = kdtree.query(points)[0].reshape((-1, 1))
    distance_vertex += trimesh.constants.tol.merge

    # axis aligned bounds
    bounds = np.column_stack((points - distance_vertex,
                              points + distance_vertex))

    # yield as generator instead of creating a python list as opposed to the
    # trimesh implementatin
    for b in bounds:
        # faces that intersect axis aligned bounding box
        yield list(rtree.intersection(b))


def assign(
    nom: trimesh.Trimesh,
    act: trimesh.Trimesh
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each triangle center of act to the closest triangle of nom

    Parameters
    ----------
    nom : trimesh.Trimesh
        Triangle mesh to assign act to
    act : trimesh.Trimesh
        Triangle mesh to assign (has n faces)
    progressbar: bool
        Print a progress bar if True

    Return
    ------
    corr : ndarray, (n, ) int
        Index of the closest correspinding triangle
    dist : (n, ) float
        Distance between corresponding triangles and points

    Notes
    -----
    Distance is calculated from face centers of act to the triangles in nom
    The algorithm follows the implementation of
    trimesh.proximity.closest_point(), but avoids huge arrays
    """
    # view triangles, points and normals as an ndarray so we don't have to
    # recompute the MD5 during all of the subsequent advanced indexing
    triangles = nom.triangles.view(np.ndarray)
    normals = nom.face_normals.view(np.ndarray)
    points = act.triangles_center.view(np.ndarray)

    # do a tree- based query for faces near each point
    all_candidates = nearby_faces(nom, points)

    # prepare output arrays
    corr = np.empty(len(points), dtype=int)
    dist = np.empty(len(points), dtype=float)
    # prepare progress bar
    iterator = progress.report(
        enumerate(zip(points, all_candidates)),
        desc="Assign triangles",
        ascii=True,
        total=len(points),
        unit="triangles")

    for i, (point, candidates) in iterator:
        candidates = np.asarray(candidates)

        # candidating triangles
        tri_cand = triangles[candidates]
        # point repeated for each candidate
        point_rep = np.repeat(point, len(tri_cand)).reshape(3, -1).T

        # closest point on the triangles for each point
        closest_on_tri = trimesh.triangles.closest_point(tri_cand, point_rep)
        # distance to minimize
        query_vector = closest_on_tri - point_rep
        dist_cand = trimesh.util.diagonal_dot(query_vector, query_vector)**.5

        # Triangles can have the same distance
        # Chose the triangle with the most normal query_vector as
        # correct triangle
        naive_best = np.min(dist_cand)
        # can only be 1D, so additional dimensions are discarded
        similar_best = np.nonzero(
            dist_cand <= (naive_best + trimesh.constants.tol.merge))[0]

        if len(similar_best) == 1:
            # no similar matches have been found
            best = similar_best[0]
        else:
            # cos() of the angles in question
            # Cannot avoid DivisionByZeroError error
            cos = np.einsum(
                "ij,ij->i",
                (query_vector[similar_best].T / dist_cand[similar_best]).T,
                normals[similar_best]
            )

            # the most rectangular query_vector
            best_of_similar = np.argmax(np.abs(cos))
            # index of the best candidate
            best = similar_best[best_of_similar]

        # desired values of the best candidate
        corr[i] = candidates[best]
        dist[i] = dist_cand[best]

    return corr, dist


def fitnom(
        act: trimesh.Trimesh
    ) -> Tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """Estimate a nominal surface mesh for a given actual surface mesh

    Parameters
    ----------
    act: trimesh.Trimesh
        Actual surface mesh

    Return
    ------
    nom: trimesh.Trimesh
        Nominal surface mesh
    corr: int(n, )
        Index of the closest correspinding triangle
    dist: float(n, )
        Distance to the closest corresponding triangle
    """
    raise NotImplementedError()


if __name__ == "__main__":
    # prepare random transformation matrix
    # with translation anywhere between [0, 0, 0] and [3, 3, 3]
    t1, t2, t3 = 2*np.pi*np.random.rand(3)
    Rx = np.array([[ 1,          0,           0          ],
                   [ 0,          np.cos(t1), -np.sin(t1)],
                   [ 0,          np.sin(t1),  np.cos(t1)]])

    Ry = np.array([[ np.cos(t2), 0,           np.sin(t2)],
                   [ 0,          1,           0         ],
                   [-np.sin(t2), 0,           np.cos(t2)]])

    Rz = np.array([[ np.cos(t3), -np.sin(t3), 0         ],
                   [ np.sin(t3),  np.cos(t3), 0         ],
                   [ 0,           0,          1         ]])
    
    T = np.eye(4)
    T[:3, :3] = Rx @ Ry @ Rz
    T[:3,  3] = 3*np.random.rand(3)
    
    # inverse transformation test
    # ---------------------------
    H = invT(invT(T))
    assert np.allclose(T, H), "Inverse transformation test failed"
    print("Inverse transformation test successful")



    # create a mesh without rotational symmtery
    def mesh():
        a, b, c, d, e, h, g = 1, 3, 5, 1, 1, 2, 1
        m = trimesh.Trimesh(vertices=[
                [0,   0,   0], # 0
                [0,   a,   0], # 1
                [0,   a, b-d], # 2
                [0,   c, b-d], # 3
                [0,   c,   b], # 4
                [0,   0,   b], # 5
                [g,   0,   0],  # 6
                [g,   a,   0],  # 7
                [g,   a, b-d], # 8
                [g, c-e, b-d], # 9
                [g, c-e,   b], # 10
                [g,   0,   b], # 11
                [h, c-e, b-d], # 12
                [h,   c, b-d], # 13
                [h,   c,   b], # 14
                [h, c-e,   b]  # 15
            ], faces=[
                # bottom
                [0, 2, 1],
                [0, 5, 2],
                [2, 5, 4],
                [2, 4, 3],
                # mid
                [6, 7, 8],
                [6, 8, 11],
                [8, 10, 11],
                [8, 9, 10],
                # top
                [12, 13, 14],
                [12, 14, 15],
                # sides
                [0, 1, 6],
                [1, 7, 6],
                [1, 2, 7],
                [2, 8, 7],
                [3, 8, 2],
                [3, 9, 8],
                [3, 12, 9],
                [3, 13, 12],
                [3, 4, 13],
                [4, 14, 13],
                [4, 15, 14],
                [4, 10, 15 ],
                [4, 11, 10],
                [4, 5, 11],
                [0, 11, 5],
                [0, 6, 11],
                [9, 12, 10],
                [10, 12, 15]
            ])
        return m
    act = mesh()
    nom = act.copy()
    
    act.apply_transform(T)
    H = align(nom, act)
    nom.apply_transform(H)

    l = np.linalg.norm(nom.vertices - act.vertices, axis=1)
    assert np.mean(l) < 5e-2, "Align test failed" +\
        "; This is statistically possible due to poor ICP estimation"
    print("Align test successful")

    import os
    with open(os.devnull, "w") as out:
        corr, dist = assign(nom, act, tqdmout=out)
    assert np.array_equal(corr, np.arange(len(act.faces))), \
        "Calculation of corresponding faces failed"
    assert np.mean(dist) < 5e-2, \
        "Calculation of distances failed"
    print("Assign test successful")