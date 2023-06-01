"""Provide Data class to manage evaluation data easily"""
from __future__ import annotations

from functools import wraps
from typing import Callable, Any, Type, Union
from dataclasses import dataclass

import trimesh
import numpy as np

from .analysis import registration, calculators
from .visual import geometries, slicing
from .cache import Cache, cached, EMPTY


@dataclass(frozen=True)
class Source:
    """Store information to re-create meshes
    Note: Immutability of attributes should not be violated!
    """
    file: str = None
    fitted: bool = False
    transf: np.ndarray[(4, 4), Any] = np.eye(4)
    subdiv: int = 0
    body: geometries.Body = None

    def recreate(self, master: Mesh):
        """Re-create the described full mesh"""
        if self.file is None:
            raise ValueError("No mesh specified")
        mesh = trimesh.load(self.file)

        if self.fitted is True:
            mesh, corr, dist = registration.nom_from_fit(mesh)
            master.corr = corr
            master.dist = dist
        
        if self.transf is not None:
            mesh.apply_transform(self.transf)
        
        if self.subdiv is not None:
            mesh = mesh.subdivide_loop(self.subdiv)
        
        if self.body is not None:
            mesh = slicing.crop(self.body, mesh)
        return mesh


class Mesh:
    def __init__(self, master: Data):
        self.master = master
        self.cache = Cache(self)
        source = self.cache.newEntry(key="source", name="source")
        mesh = self.cache.newEntry(key="mesh", name="mesh", src=[source])
        self.cache.newEntry(key="transf", name="transformation", src=[mesh])
        self.cache.newEntry(key="subdivs", name="subdivisions", src=[mesh])
        return

    @cached
    def source(self) -> Source:
        return Source()

    @source.setter
    def source(self, value):
        if not isinstance(value, Source):
            raise TypeError(f"Expected type Source, got {type(value)}")
        self.cache["source"] = value
    
    @cached
    def mesh(self) -> trimesh.Trimesh:
        return self.source.recreate(self.master)

    def set_source_and_mesh(self, source: Source, mesh: trimesh.Trimesh):
        """Set source and mesh that was generated externally"""
        self.source = source
        self.cache["mesh"] = mesh

    @cached
    def transf(self) -> np.ndarray[(4, 4), Any]:
        return np.eye(4)

    @transf.setter
    def transf(self, value: np.ndarray[(4, 4), Any]):
        value = np.asarray(value)
        if value.shape != (4, 4):
            raise ValueError(f"Expected shape (4, 4), got {value.shape}")

        inv = registration.invT(self.transf)
        self.mesh.apply_transform(value @ inv)
        self.cache["transf"] = value

    @transf.deleter
    def transf(self):
        self.transf = np.eye(4)
        del self.cache["transf"]
    
    @cached
    def subdivs(self) -> int:
        return 0

    @subdivs.setter
    def subdivs(self, value):
        n = value - self.subdivs
        if n < 0:
            raise ValueError("Subdivisions cannot be reverted")
        elif n == 0:
            return
        mesh = self.mesh.subdivide_loop(n)
        self.cache["mesh"] = mesh
        self.cache["subdivs"] = n
    
    def assource(self) -> dict:
        out = self.source.__dict__
        if out["body"] is not None:
            raise ValueError("Cropped meshes cannot be the source for new meshes")
        out["subdiv"] += self.subdivs
        out["transf"] = self.transf @ out["transf"]
        return out

    def update(self, dct: dict):
        self.source = Source(**dct.get("source", {}))
        self.transf = dct.get("transformation", np.eye(4))
        self.subdivs = dct.get("subdivisions", 0)


class Data:
    def __init__(self):
        self.cache = Cache()
        # permanently "cached" entries
        self.cache.newEntry(key="nom", name="nominal")
        self.cache.newEntry(key="act", name="actual")
        # cached entries
        corr = self.cache.newEntry(key="corr", name="corresponding faces",
            src=[self.nom.cache, self.act.cache])
        self.cache.newEntry(key="dist", name="distance", src=[corr])
        self.cache.newEntry(key="SPC", name="surface parameters", src=[corr])
        self.cache.newEntry(key="VPC", name="vertex parameters", 
            src=[self.act.cache])

    @cached
    def nom(self) -> Mesh:
        return Mesh(self)

    @cached
    def act(self) -> Mesh:
        return Mesh(self)

    def align(self):
        T = registration.align(self.nom.mesh, self.act.mesh)
        # keep the current transformation
        self.nom.transf = T @ self.nom.transf

    @cached
    def corr(self) -> np.ndarray:
        corr, dist = registration.assign(self.nom.mesh, self.act.mesh)
        self.cache["dist"] = dist
        return corr

    @corr.setter
    def corr(self, value: np.ndarray):
        value = np.asarray(value)
        if len(value) != len(self.nom.mesh.faces) or value.ndim != 1:
            raise ValueError(
                f"Corresponding faces of shape {value.shape} cannot be applied"
                f" to nominal mesh with {len(self.nom.mesh.faces)} faces")
        self.cache["corr"] = value

    @cached
    def dist(self) -> np.ndarray:
        self.corr # also calculates dist
        # PROBLEM: will raise an error if corr was set manually
        return self.cache["dist"]

    @dist.setter
    def dist(self, value: np.ndarray):
        value = np.asarray(value)
        if len(value) != len(self.nom.mesh.faces) or value.ndim != 1:
            raise ValueError(
                f"Distances of shape {value.shape} cannot be applied"
                f" to nominal mesh with {len(self.nom.mesh.faces)} faces")
        self.cache["dist"] = value

    @cached
    def SPC(self) -> calculators.SPC:
        return calculators.SPC(self.nom.mesh, self.act.mesh, self.corr)

    @cached
    def VPC(self) -> calculators.VPC:
        return calculators.VPC(self.act.mesh)

    def update(self, dct: dict):
        self.nom.update(dct.get("nominal", {}))
        self.act.update(dct.get("actual", {}))
        self.corr = dct.get("corresponding faces", EMPTY)
        self.dist = dct.get("distance", EMPTY)

if __name__ == "__main__":
    full = Data()
    full.nom.source = Source(full.nom, file="data/ellipsoid.stl")
    full.act.source = Source(full.act, file="data/ellipsoid_rough.stl")
    full.align()
    full.dist
    print(full.cache)
