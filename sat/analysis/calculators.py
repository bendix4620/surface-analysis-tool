"""Provide Calculator classes to calculate surface and vertex parameters
of trimesh triangle meshes"""

__all__ = ["SPC", "VPC"]

import numpy as np
import trimesh

from trimesh import caching
from scipy.optimize import least_squares

from . import progress


class Calculator:
    """Base class for parameter calculators"""
    def __init__(
            self,
            nom: trimesh.Trimesh,
            act: trimesh.Trimesh,
            corr: np.ndarray,
            tol: float = 1e-8
    ):
        """
        Parameters
        ----------
        act : trimesh.Trimesh
            Actual surface whose height will be calculated (has n faces)
        nom : trimesh.Trimesh
            Nominal surface defining the base line of heights
        corr : int (n, )
            Indices associating each triangle in act with the closest in nom
        tol : float
            Tolerance to mark values as equal
        """

        self._data = caching.DataStore()
        self._data["corr"] = np.asarray(corr)
        self._act = act
        self._nom = nom
        self.tol = tol

        self._cache = caching.Cache(
            id_function=self.__hash__, force_immutable=True)

    def __hash__(self) -> int:
        """Combined hash from _data, act._data and nom._data"""
        return hash((self.act._data, self.nom._data, self._data))

    @property
    def nom(self) -> trimesh.Trimesh:
        return self._nom

    @property
    def act(self) -> trimesh.Trimesh:
        return self._act

    @property
    def corr(self) -> np.ndarray:
        return self._data["corr"]


def abs_tri_heights_01sim(z) -> np.ndarray:
    """Return average absolute heights of a triangles.
    Assumes that (only) z[0] and z[1] are equal.

    Parameters
    ----------
    z: (3, n) float
        Heights of triangle corners

    Return
    ------
    h: (n, ) float
        average absolute height

    Example
    -------
    abs_tri_heights_01sim([[-1, -1, 1]])
    array([0.5])
    """
    z0, _, z2 = np.transpose(z)
    absz0 = np.abs(z0)
    z2_min_z0 = z2 - z0
    return ((np.abs(z2*z2*z2) - absz0*absz0*absz0) / (z2_min_z0)
            - 3 * absz0 * z0
            ) / (3 * z2_min_z0)


def abs_tri_heights_12sim(z) -> np.ndarray:
    """Return average absolute heights of a triangles.
    Assumes that (only) z[1] and z[2] are equal.

    Parameters
    ----------
    z: (3, n) float
        Heights of triangle corners

    Return
    ------
    h: (n, ) float
        average absolute height

    Example
    -------
    abs_tri_heights_12sim([[1, -1, -1]])
    array([0.5])
    """
    z0, z1, _ = np.transpose(z)
    absz1 = np.abs(z1)
    z1_min_z0 = z1 - z0
    return (3 * absz1 * z1
            - (absz1*absz1*absz1 - np.abs(z0*z0*z0)) / (z1_min_z0)
            ) / (3 * z1_min_z0)


def abs_tri_heights_20sim(z) -> np.ndarray:
    """Return average absolute heights of a triangles.
    Assumes that (only) z[2] and z[0] are equal.

    Parameters
    ----------
    z: (3, n) float
        Heights of triangle corners

    Return
    ------
    h: (n, ) float
        average absolute height

    Example
    -------
    abs_tri_heights_20sim([[-1, 1, -1]])
    array([0.5])
    """
    z0, z1, _ = np.transpose(z)
    absz0 = np.abs(z0)
    z1_min_z0 = z1 - z0
    return ((np.abs(z1*z1*z1) - absz0*absz0*absz0) / (z1_min_z0)
            - 3 * absz0 * z0
            ) / (3 * z1_min_z0)


def abs_tri_heights_general(z) -> np.ndarray:
    """Return average absolute heights of a triangles.
    Assumes that each two values are never equal.

    Parameters
    ----------
    z: (3, n) float
        Heights of triangle corners

    Return
    ------
    h: (n, ) float
        average absolute height

    Example
    -------
    abs_tri_heights_general([[-1, 0, 1]])
    array([0.33333333])
    """
    z0, z1, z2 = np.transpose(z)
    absz2_3 = np.abs(z2*z2*z2)
    return ((absz2_3 - np.abs(z1*z1*z1)) / (z2 - z1)
            - (absz2_3 - np.abs(z0*z0*z0)) / (z2 - z0)
            ) / (3 * (z1 - z0))


class SPC(Calculator):
    """Handle calculations of surface parameters (scalar) with caching
    
    Attributes
    ----------
    params: dict
        Function name and display name pairs of calculable parameters
    """

    # function name: display name
    params = {
        'Sa':    'Sa',
        'Sp':    'Sp',
        'Sv':    'Sv',
        'Sz':    'Sz',
        'Sdr':   'Sdr',
        'Sdr_p': "Sdr'",
        'Srf':   'Srf',
        'Srr':   'Srr'
    }

    @caching.cache_decorator
    def corr_normals(self) -> np.ndarray:
        """Normals of nom triangles corresponding to act triangles"""
        return self.nom.face_normals[self.corr]

    # area values

    @caching.cache_decorator
    def proj_act_tri_areas_signed(self) -> np.ndarray:
        """Signed projection of act's triangle areas onto corresponding
        triangles in nom
        """
        return .5 * np.einsum(
            "ij,ij->i", self.corr_normals, self.act.triangles_cross)

    @caching.cache_decorator
    def proj_act_tri_areas(self) -> np.ndarray:
        """Sign-free projection of act's triangle areas onto corresponding
        triangles in nom
        """
        return np.abs(self.proj_act_tri_areas_signed)

    @caching.cache_decorator
    def proj_act_area(self) -> float:
        """Total area of act's projection onto nom"""
        return np.sum(self.proj_act_tri_areas)

    @caching.cache_decorator
    def shadow_area(self) -> float:
        """Area of the shadow casted onto nom by act"""
        return np.sum(self.proj_act_tri_areas_signed)

    # height values

    @caching.cache_decorator
    def heights(self) -> np.ndarray:
        """Heights of triangle corners in act over the corresponding
        triangles in nom
        """
        heights = np.einsum(
            "ik,ijk->ij",
            self.corr_normals,
            self.act.triangles - self.nom.triangles[self.corr])
        return heights

    @caching.cache_decorator
    def _sp(self) -> float:
        """Signed maximum height"""
        return np.max(self.heights)

    @caching.cache_decorator
    def _sv(self) -> float:
        """Signed minimal valley"""
        return np.min(self.heights)

    @caching.cache_decorator
    def absolute_triangle_heights(self) -> np.ndarray:
        """Average absolute heights of act's triangles over the corresponding
        triangles in nom
        """
        # do not check cache within the function
        heights = self.heights.view(np.ndarray)
        res = np.empty(heights.shape[0])

        # all values have the same sign
        smaller = heights <= 0
        mask_samesign = np.all(smaller, axis=1) | ~np.any(smaller, axis=1)
        res[mask_samesign] = np.abs(heights[mask_samesign].sum(axis=1)) / 3

        # two values are similar
        # use 3 different functions to avoid re-ordering the array
        mask_01sim = ~mask_samesign & np.isclose(
            heights[:, 0], heights[:, 1], atol=self.tol)
        res[mask_01sim] = abs_tri_heights_01sim(heights[mask_01sim])

        mask_12sim = ~mask_samesign & np.isclose(
            heights[:, 1], heights[:, 2], atol=self.tol)
        res[mask_12sim] = abs_tri_heights_12sim(heights[mask_12sim])

        mask_20sim = ~mask_samesign & np.isclose(
            heights[:, 2], heights[:, 0], atol=self.tol)
        res[mask_20sim] = abs_tri_heights_20sim(heights[mask_20sim])

        # remaining general case
        mask_general = ~(mask_samesign | mask_01sim | mask_12sim | mask_20sim)
        res[mask_general] = abs_tri_heights_general(heights[mask_general])

        return res

    # parameters
    @caching.cache_decorator
    def Sa(self) -> float:
        """Average absolute height
        (of the actual surface over its projection onto the nominal surface)
        """
        res = np.sum(self.proj_act_tri_areas * self.absolute_triangle_heights)
        return res / self.proj_act_area

    @caching.cache_decorator
    def Sp(self) -> float:
        """Maximum peak height"""
        return np.abs(self._sp)

    @caching.cache_decorator
    def Sv(self) -> float:
        """Minimum valley depth"""
        return np.abs(self._sv)

    @caching.cache_decorator
    def Sz(self) -> float:
        """Maximum peak-valley difference"""
        return self._sp - self._sv

    @caching.cache_decorator
    def Sdr(self) -> float:
        """Developed interfacial area ratio"""
        return self.act.area / self.proj_act_area - 1

    @caching.cache_decorator
    def Sdr_p(self) -> float:
        """Alternative (not identical) definition of Sdr"""
        return self.act.area / self.shadow_area - 1

    @caching.cache_decorator
    def Srf(self) -> float:
        """Re-entrant feature ratio"""
        return (self.proj_act_area - self.shadow_area) / (2*self.shadow_area)

    @caching.cache_decorator
    def Srr(self) -> float:
        """Re-entrant feature ratio calculated without projections"""
        inwards = self.proj_act_tri_areas_signed < 0
        return np.sum(self.act.area_faces[inwards]) / self.act.area

    # utility
    def tojson(self, _) -> dict:
        """Return the json serializable representation of the exportable
        cached values
        """
        out = {}
        for fname, dname in self.params.items():
            if fname in self._cache:
                out[dname] = self._cache[fname]
        return out


class VPC:
    """Handle calculations of vertex parameters with caching.
    Parameters have shape (len(vertices), -1)

    Attributes
    ----------
    params: dict
        Function name and display name pairs of calculable parameters
    """

    # function name: display name
    params = {
        "Curv": "Principle Curvature",
        "Anglediff": "Angle Difference"
    }

    def __init__(self, act):
        self._act = act
        self._cache = caching.Cache(id_function=self.act._data.__hash__)
    
    @property
    def act(self) -> trimesh.Trimesh:
        return self._act

    @caching.cache_decorator
    def Anglediff(self):
        """Angle Difference of neighboring vertex normal vectors (in degree)
        Retrieve unique edges from vertex neighbors, then calculate curvature
        from distance between vertices and angle between their normals
        """
        neighbors = self.act.vertex_neighbors
        repeats = np.fromiter(map(len, neighbors), dtype=int)
        iv1 = np.repeat(
            np.arange(len(neighbors)),
            repeats=repeats
        )
        # concatenate returns float if empty lists are present
        iv2 = np.concatenate(neighbors).astype(int)

        # edge index list
        iedges = np.vstack((iv1, iv2)).T
        iedges_sorted = np.sort(iedges, axis=1)

        # unique edge index list
        uiedges, unique_inv = np.unique(
            iedges_sorted, return_inverse=True, axis=0)
        uedges = self.act.vertices.view(np.ndarray)[uiedges.T]

        # reciprocal length
        vec = np.subtract(*uedges)
        urec_length = 1/np.sqrt(np.dot(vec ** 2, [1] * vec.shape[1]))
        # split into vertex neighbor groups
        split_rec_length = np.split(
            urec_length[unique_inv],
            indices_or_sections=np.cumsum(repeats[:-1])
        )

        # calculate angle between vertex normals
        products = np.einsum("ij,ij->i", *self.act.vertex_normals[uiedges.T])
        uangles = np.rad2deg(np.arccos(np.clip(products, -1, 1)))
        # avoid parallel normals
        uangles = np.clip(uangles, 1, None)
        # split into vertex neighbor groups
        split_angles = np.split(
            uangles[unique_inv],
            indices_or_sections=np.cumsum(repeats[:-1])
        )

        # obtain curvature
        curvature = np.empty(len(self.act.vertices))
        for i in range(len(curvature)):
            srl = split_rec_length[i]
            curvature[i] = np.sum(srl*split_angles[i]) / np.sum(srl) / len(srl)
        return curvature.reshape(-1, 1)

    @caching.cache_decorator
    def Curv(self):
        """Principle curvatures for each vertex

        Algorithm
        ---------
        Szymon Rusinkiewicz.
        "Estimating Curvatures and Their Derivatives on Triangle Meshes."
        Symposium on 3D Data Processing, Visualization, and Transmission, 
        September 2004. 
        Link: https://doi.org/10.1109/TDPVT.2004.1335277

        Notes
        -----
        Will return np.nan for vertices that are not present in any face
        """

        def residuals(x, *args):
            """linear contraints as residuals"""
            a, b, c, d = x
            u, v, e0, e1, e2, n0, n1, n2 = args

            e0u = np.dot(e0, u)
            e0v = np.dot(e0, v)
            e1u = np.dot(e1, u)
            e1v = np.dot(e1, v)
            e2u = np.dot(e2, u)
            e2v = np.dot(e2, v)

            n0u = np.dot(n0, u)
            n0v = np.dot(n0, v)
            n1u = np.dot(n1, u)
            n1v = np.dot(n1, v)
            n2u = np.dot(n2, u)
            n2v = np.dot(n2, v)

            res = np.empty(6)
            res[0] = (a*e0u + b*e0v) - (n2u - n1u)
            res[1] = (c*e0u + d*e0v) - (n2v - n1v)
            res[2] = (a*e1u + b*e1v) - (n0u - n2u)
            res[3] = (c*e1u + d*e1v) - (n0v - n2v)
            res[4] = (a*e2u + b*e2v) - (n1u - n0u)
            res[5] = (c*e2u + d*e2v) - (n1v - n0v)
            return res

        def normalize(x): return x / np.linalg.norm(x)

        # bypass cache checks for the following calculations
        vertices = self.act.vertices.view(np.ndarray)
        faces = self.act.faces.view(np.ndarray)
        vertex_normals = self.act.vertex_normals.view(np.ndarray)
        area_faces = self.act.area_faces.view(np.ndarray)

        iterator = progress.report(
            enumerate(faces),
            desc="Approximate curvatures", 
            ascii=True, 
            total=len(self.act.faces), 
            unit="triangles")

        # calculate Second Fundamental Tensor for each face
        vtensor = np.zeros((len(vertices), 2, 2))
        weights = np.zeros(len(vertices))
        for i, face in iterator:
            p = vertices[face]
            # edges
            e0 = p[2] - p[1]
            e1 = p[0] - p[2]
            e2 = p[1] - p[0]
            # gram-schmidt orthonormalisation
            uf = normalize(e0)
            vf = normalize(e1 - (uf @ e1)*uf)
            # vertex normals
            n0, n1, n2 = vertex_normals[face]

            # tensor in face's system
            args = (uf, vf, e0, e1, e2, n0, n1, n2)
            tensor = least_squares(residuals, x0=[1, 0, 0, 1], args=args)["x"].reshape(2, 2)

            # barycentric weights instead of voronoi weights
            weight = area_faces[i] / 3
            if weight <= 0:
                raise RuntimeError("Weird weights")

            # add tensors to corresponding vertices's tensors
            # (transform tensor to vertex system)
            for j in face:
                p = vertices[j]

                # guess 1st normal vector, use another if it turns out to be 0
                try:
                    with np.errstate(all="raise"):
                        up = normalize(np.array([0, -p[2], p[1]]))
                except (ZeroDivisionError, FloatingPointError):
                    up = normalize(np.array([-p[1], p[0], 0]))
                # 2nd normal vector
                vp = normalize(np.cross(p, up))

                # basis change
                m0 = np.array([up @ uf, up @ vf])
                m1 = np.array([vp @ uf, vp @ vf])

                vtensor[j, 0, 0] += (m0.T @ tensor @ m0) * weight
                vtensor[j, 0, 1] += (m0.T @ tensor @ m1) * weight
                vtensor[j, 1, 1] += (m1.T @ tensor @ m1) * weight
                weights[j] += weight

        # divide by weights
        # a vertex that is not in any face will result in 0/0=nan
        with np.errstate(invalid="ignore"):
            vtensor[:, 0, 0] /= weights
            vtensor[:, 0, 1] /= weights
            vtensor[:, 1, 0] = vtensor[:, 0, 1]  # is symmetric
            vtensor[:, 1, 1] /= weights

        return np.linalg.eigvalsh(vtensor)
