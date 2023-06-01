"""Provide geometries (selection bodies) for slicing.py"""

import trimesh
import numpy as np
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod
from typing import Callable

# all geometries that exist
__all__ = ["Box", "Sphere", "TrianglePrism"]

normalize = lambda x: x/np.linalg.norm(x)

class Body(ABC):
    """Abstract baseclass for Bodies"""

    @classmethod
    def validate_points(cls, points, fixpoint) -> np.ndarray:
        """Check if points has a suitable shape"""

        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape((1, -1))
        elif points.ndim >= 3:
            raise TypeError(
                f"Cannot understand point cloud of shape {points.shape}"
            )

        if points.shape[1] != fixpoint.shape[0]:
            raise TypeError(
                f"Dimesnions of points = {points.shape[1]} does not match "
                f"dimension of {cls.__name__} = {fixpoint.shape[0]}"
            )
        return points

    @abstractmethod
    def asdict(self):
        """Return selection parameters as dict.
        Has to enable creation from dict: Body(**mybody.asdict())
        """
        # default example
        out = {"name": self.__class__.__name__}
        out.update(self.__dict__)
        return out

    @abstractmethod
    def __init__(self, center=np.zeros(3), **kwargs):
        """Constructor of the Body.

        Notes
        -----
        - Can only accept keyword arguments
        - Needs 'center' as input parameter (has to be the real center)
        - Internal parameters are arbitrary
        """

    @abstractmethod
    def contains(self, points: ArrayLike):
        """Return a boolean mask for points, True if the point is inside"""

    @abstractmethod
    def transform(self, T: np.ndarray):
        """Transform body in-place"""

    @abstractmethod
    def to_mesh(self) -> trimesh.Trimesh:
        """Return trimesh representation"""

    @abstractmethod
    def make_widgets(self, selector):
        """Create widgets that control the body"""


class Space(Body):
    """The entire available space"""

    def __init__(self, center=np.zeros(3)):
        self.center = center
        return

    def asdict(self) -> dict:
        """Return selection parameters as dict"""

        out = {"name": self.__class__.__name__,
               "center": self.center}
        return out

    def contains(self, points):
        """Return that all points are inside"""
        points = self.validate_points(points, self.center)
        return np.ones(len(points), dtype=bool)
    
    def transform(self, T: np.ndarray):
        """Nothing to transform"""
        return

    def to_mesh(self):
        """Return an empty mesh"""
        return trimesh.Trimesh(vertices=np.zeros((1, 3)))

    def make_widgets(self, selector):
        """Do not create any widgets"""
        return

class NoSpace(Space):
    """Counter part to Space, nothing is inside"""

    def contains(self, points):
        """Return that no points are inside"""
        points = self.validate_points(points, self.center)
        return np.zeros(len(points), dtype=bool)


class Box(Body):
    """Box object described by a corner and edge extents along a basis"""

    def __init__(
        self,
        center: ArrayLike=np.zeros(3),
        edges: ArrayLike=np.ones(3),
        basis: ArrayLike=np.eye(3)
    ):
        """Constructor for box object described by a corner and edge extents
        along a basis

        Parameters
        ----------
        center : (3, ) float, optional(=coordinate origin)
            Center of the Box 
        edges : (3, ) float, optional(=[1, 1, 1])
            Lengths of the box's edges
        basis : (3, 3) float, optional(=euclidean standard basis))
            Basis for the box edges
            Column vectors
        """
        self.edges = np.asarray(edges)
        self.basis = np.asarray(basis)
        for i in range(3):
            self.basis[:, i] = normalize(self.basis[:, i])
        self.corner = np.asarray(center) - self.corner2center
    
    @property
    def corner2center(self):
        return np.sum(self.basis * self.edges, axis=1) / 2

    def asdict(self) -> dict:
        """Return selection parameters as dict"""

        out = {"name": self.__class__.__name__,
               "center": self.corner + self.corner2center,
               "edges": self.edges,
               "basis": self.basis}
        return out

    def contains(self, points: ArrayLike) -> np.ndarray:
        """Return a boolean mask for points, True if inside"""

        points = self.validate_points(points, self.corner)
        # mask for all `points` that are inside
        inside = np.ones(len(points), dtype=bool)

        for length, normal in zip(self.edges, self.basis.T):
            # dot product of all `points` that are currently inside
            points_ = points[inside]
            if len(points_) == 0:  # stop if nothing is inside
                return inside

            dots = np.dot(points_, normal)

            # The `offset` results from `corner` not being in the origin
            # Adding this onto the threshold values later is faster than
            # subtracting it from `points` or `dots` initially
            offset = np.dot(self.corner, normal)

            # `dots` were only calculated of all points inside, so we only
            # update points that are inside
            inside[inside] = (dots >= (offset)) & (dots <= (length+offset))
        return inside

    def transform(self, T: np.ndarray):
        """Apply (4x4) transformation T inplace"""

        R = T[:3, :3]
        self.corner = R @ self.corner + T[:3, 3]
        self.basis = R @ self.basis

    def to_mesh(self) -> trimesh.Trimesh:
        """Return mesh representation"""

        # Is a speedup possible by unpacking instead of indexing?
        corners = np.empty((8, 3))
        # "bottom"
        corners[0] = self.corner
        corners[1] = corners[0] + self.basis[:, 0] * self.edges[0]
        corners[2] = corners[1] + self.basis[:, 1] * self.edges[1]
        corners[3] = corners[0] + self.basis[:, 1] * self.edges[1]
        # "top"
        corners[4] = corners[0] + self.basis[:, 2] * self.edges[2]
        corners[5] = corners[4] + self.basis[:, 0] * self.edges[0]
        corners[6] = corners[5] + self.basis[:, 1] * self.edges[1]
        corners[7] = corners[4] + self.basis[:, 1] * self.edges[1]

        # indices rotating counter clock-wise from the outside
        faces = np.array([
            [0, 3, 2],
            [0, 2, 1],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [4, 5, 6],
            [4, 6, 7]
        ])
        return trimesh.Trimesh(vertices=corners, faces=faces)

    def make_widgets(self, selector):
        """Add widgets to the selector instance to control the box size
        
        Parameters
        ----------
        selector : slicing.Selector
            The selector in which the body is used
        """
        def _create_callback(self, selector, axis: int) -> Callable:
            """Create a callback(float) function for a slider"""
            def callback(value: float):
                self.edges[axis] = value
                selector.update_body()
            return callback

        for axis, axis_name in enumerate("xyz"):
            # create callback and ensure that default values are displayed
            callback = _create_callback(self, selector, axis)

            selector.plotter.add_slider_widget(
                callback,
                [0, selector.maxextent * 1.1],
                value=self.edges[axis],
                title=f"{axis_name} extent",
                event_type="always",  # use "end" or "always"
                style="modern",
                pointa=(0.8,  (2.5-axis) / 3),
                pointb=(0.98, (2.5-axis) / 3)
            )


def sqnorm(x: ArrayLike) -> float:
    """Return squared norm of x"""
    return np.einsum("ij,ij->i", x, x)


class Sphere(Body):
    """Sphere object described by center and radius"""

    def __init__(
        self,
        center: ArrayLike=np.zeros(3),
        radius: float=1,
        latitudes: float=32,
        longitudes: float=32
    ):
        """Constructor for Sphere object described by center and radius

        Parameters
        ----------
        center : (3, ) float, optional(=coordinate origin)
            Center of the sphere
        radius : float, optional(=1)
            Radius of the sphere
        latitudes : int, optional(=32)
            Number of latitude steps
        longitudes : int, optional(=32)
            Number of longitude steps
        """
        self.center = np.asarray(center)
        self.radius = radius
        self.latitudes = latitudes
        self.longitudes = longitudes

    def asdict(self) -> dict:
        """Return selection parameters as dict"""

        out = {"name": self.__class__.__name__,
               "center": self.center,
               "radius": self.radius}
        return out

    def contains(self, points: ArrayLike) -> np.ndarray:
        """Return a boolean mask for points, True if inside"""

        points = self.validate_points(points, self.center)
        # mask for all `points` that are inside
        inside = sqnorm(points - self.center) <= self.radius*self.radius

        return inside

    def transform(self, T: np.ndarray):
        """Apply (4x4) transformation T inplace"""

        self.center = T[:3, :3] @ self.center + T[:3, 3]
        # scaling = np.linalg.det(T[:3, :3])**(1/3)
        # self.radius *= scaling
        # print("sphere scaling =", scaling)

    def to_mesh(self) -> trimesh.Trimesh:
        """Return mesh representation"""

        mesh = trimesh.creation.uv_sphere(
            radius=self.radius, count=(self.latitudes, self.longitudes))
        mesh.vertices += self.center
        return mesh

    def make_widgets(self, selector):
        """Add widgets to the selector instance to control the box size
        
        Parameters
        ----------
        selector : slicing.Selector
            The selector in which the body is used
        """
        def callback(value: float):
            self.radius = value
            selector.update_body()

        selector.plotter.add_slider_widget(
            callback,
            [0, selector.maxextent * 0.55],
            value=self.radius,
            title="radius",
            event_type="always",  # use "end" or "always"
            style="modern",
            pointa=(0.8,  0.333),
            pointb=(0.98, 0.333)
        )


def rodrigues(axis, angle):
    """Return rotation matrix that rotates with angle around axis"""
    x, y, z = axis
    K = np.array([[ 0, -z,  y],
                  [ z,  0, -x],
                  [-y,  x,  0]])
    return np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)


class TrianglePrism(Body):
    """Prism object with triangles as bases"""

    def __init__(
        self,
        center: ArrayLike=np.zeros(3),
        edges: ArrayLike=np.ones(3),
        nh: ArrayLike=[0, 0, 1],
        na: ArrayLike=[1, 0, 0],
        nb: ArrayLike=[0, 1, 0],
        gamma: float=None
    ):
        """Constructor for prism object with triangles as bases

        Parameters
        ----------
        center : (3, ) float, optional(=coordinate origin)
            Lower corner of the triangle 
        edges : (3, ) float, optional(=[1, 1, 1])
            Tuple of prism height h, edge length a, edge length b
        nh : (3, ) float, optional(=z-axis)
            Normal vector along the prism height
        na : (3, ) float, optional(=x-axis)
            Normal vector along edge a of the base triangle
        nb : (3, ) float, optional(=y-axis)
            Normal vector along edge b of the base triangle
        gamma : float, optional
            Angle between triangle edges a and b
            Will overwrite nb!

        Notes
        -----
        Base triangle with mathematically positive angle gamma (y):
            _________
            \       /
             \<---./
            b \ y / a
               \ /
                *
        """
        self.edges = np.asarray(edges)
        self.nh = normalize(np.asarray(nh))
        self.na = normalize(np.asarray(na))
        self.nb = normalize(np.asarray(nb))
        if gamma is not None:
            self.gamma = gamma
        if not np.isclose(np.dot(self.nh, self.na), 0):
            raise ValueError("nh and na need to be orthogonal")
        if not np.isclose(np.dot(self.nh, self.nb), 0):
            raise ValueError("nh and nb need to be orthogonal")
        self.corner = np.asarray(center) - self.corner2center

    @property
    def corner2center(self):
        out =  self.nh*self.edges[0] \
             + self.na*self.edges[1] \
             + self.nb*self.edges[2]
        return out/2

    @property
    def gamma(self) -> float:
        """Angle gamma between edges a and b in radiant"""
        return np.arccos(np.dot(self.na, self.nb))
    
    @gamma.setter
    def gamma(self, angle: float):
        R = rodrigues(self.nh, angle)
        self.nb = R @ self.na
        

    def asdict(self) -> dict:
        """Return selection parameters as dict"""

        out = {"name": self.__class__.__name__,
               "center": self.corner + self.corner2center,
               "edges": self.edges,
               "nh": self.nh,
               "na": self.na,
               "nb": self.nb}
        return out

    def contains(self, points: ArrayLike) -> np.ndarray:
        """Return a boolean mask for points, True if inside"""

        points = self.validate_points(points, self.corner)
        h, a, b = self.edges

        corners = np.empty((3, 3)) # row vectors
        corners[0] = self.corner
        corners[1] = self.corner + a*self.na
        corners[2] = self.corner + b*self.nb

        # get rotation matrix that transforms n_z to self.normal
        axis = normalize(np.array([0, 0, 1]) + self.nh)
        R = rodrigues(axis, np.pi)

        # rotate points and triangle corners
        points = R.dot(points.T).T
        corners = R.dot(corners.T).T

        # discard points that are too low or too high and project them 
        # into the 2d plane
        mask = np.logical_and(points[:, 2] >= corners[0, 2],
                              points[:, 2] <= (corners[0, 2] + h))
        p2d = points[:, :2][mask] # points 2d
        c2d = corners[:, :2] # corners 2d

        # calculate barycentric coordinates for remaining 2d points
        x1, y1 = c2d[0]
        x2, y2 = c2d[1]
        x3, y3 = c2d[2]
        denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
        l1 = ((y2 - y3)*(p2d[:, 0] - x3) + (x3 - x2)*(p2d[:, 1] - y3))/denom
        l2 = ((y3 - y1)*(p2d[:, 0] - x3) + (x1 - x3)*(p2d[:, 1] - y3))/denom
        l3 = 1 - l1 - l2

        # points are inside if all 3 barycentric coordinates are within [0, 1]
        mask[mask] = (  (0 <= l1) & (l1 <= 1)
                      & (0 <= l2) & (l2 <= 1)
                      & (0 <= l3) & (l3 <= 1))
        return mask

    def transform(self, T: np.ndarray):
        """Transform prism inplace"""

        R = T[:3, :3]
        self.corner = R @ self.corner + T[:3, 3]
        self.nh = R @ self.nh
        self.na = R @ self.na
        self.nb = R @ self.nb

    def to_mesh(self) -> trimesh.Trimesh:
        """Return mesh representation"""
        h, a, b = self.edges
        toother = h*self.nh
        
        corners = np.empty((6, 3))
        corners[0] = self.corner
        corners[1] = self.corner + a*self.na
        corners[2] = self.corner + b*self.nb
        corners[3, :] = corners[0, :] + toother
        corners[4, :] = corners[1, :] + toother
        corners[5, :] = corners[2, :] + toother

        faces = np.array([
            [0, 1, 2],  # base
            [3, 4, 5],  # base
            [0, 1, 3],
            [3, 1, 4],
            [0, 3, 2],
            [3, 5, 2],
            [2, 5, 1],
            [5, 4, 1]
        ])
        return trimesh.Trimesh(vertices=corners, faces=faces)

    def make_widgets(self, selector):
        """Add widgets to the selector instance to control the box size
        
        Parameters
        ----------
        selector : slicing.Selector
            The selector in which the body is used
        """

        def callback_gamma(value: float):
            """Callback updating the angle gamma"""
            self.gamma = value
            selector.update_body()

        selector.plotter.add_slider_widget(
            callback_gamma,
            [0, np.pi],
            value=self.gamma,
            title=f"Angle ∠a,b",
            event_type="always",  # use "end" or "always"
            style="modern",
            pointa=(0.8,  3.5 / 4),
            pointb=(0.98, 3.5 / 4)
        )

        def _create_callback(self, selector, axis: int) -> Callable:
            """Create a callback(float) function for a slider"""

            def callback(value: float):
                """Callback updating the edges"""
                self.edges[axis] = value
                selector.update_body()
            return callback

        # triangle sides a, b and prism height h
        for axis, axis_name in enumerate("hab"):
            # create callback and ensure that default values are displayed
            callback = _create_callback(self, selector, axis)

            selector.plotter.add_slider_widget(
                callback,
                [0, selector.maxextent * 1.1],
                value=self.edges[axis],
                title=f"{axis_name} extent",
                event_type="always",  # use "end" or "always"
                style="modern",
                pointa=(0.8,  (2.5-axis) / 4),
                pointb=(0.98, (2.5-axis) / 4)
            )
