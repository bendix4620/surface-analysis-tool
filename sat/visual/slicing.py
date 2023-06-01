"""Provide Selector class to interactively select regions to crop"""

import sys
import trimesh
import numpy as np
import vtk
import pyvista as pv
from typing import Type, Tuple
from qtpy.QtWidgets import QMessageBox, QPushButton, QLabel

from . import geometries
from . import plotter


class CroppingTool(plotter.RenderWindow):
    """Graphical selection tool based on simple geometrical bodies to
    manage loose cropping of meshes
    """

    def __init__(
        self,
        nom: trimesh.Trimesh,
        act: trimesh.Trimesh
    ):
        """Graphical selection tool based on simple geometrical bodies to
        manage loose cropping of meshes

        Parameters
        ----------
        nom : trimesh.Trimesh
            Mesh of the nominal surface
        act : trimesh.Trimesh
            Mesh of the actual surface
        """
        # output variable
        self.cropped: Tuple = None
        
        super().__init__(title="Cropping Tool")
        self.add_toolbar_crop()

        # body visual settings
        self.settings = dict(color=[255, 0, 0], opacity=.5)

        # add the body
        self._act_body = None
        self.maxextent = np.max([act.extents, nom.extents])
        self._set_body("NoSpace")

        # add the meshes
        self.act = act
        self.nom = nom
        self._plotter.add_mesh(self.act, pickable=False)
        self._plotter.add_mesh(self.nom, pickable=False, style="wireframe")

    def add_toolbar_crop(self):
        self.crop_toolbar = self.addToolBar("Selection Bodies")
        self.crop_toolbar.setMovable(False)
        
        self._add_action(self.crop_toolbar, "crop", self._crop)
        self.crop_toolbar.addSeparator()
        
        def wrapper(name):
            return lambda: self._set_body(name)
        self._add_action(self.crop_toolbar, "No Body", wrapper("NoSpace"))
        for name in geometries.__all__:
            self._add_action(self.crop_toolbar, name, wrapper(name))
        
        self.crop_toolbar.addSeparator()
        self._add_action(self.crop_toolbar, "center at actual surf.", 
                         lambda: self._center_body(self.act))
        self._add_action(self.crop_toolbar, "center at nominal surf.", 
                         lambda: self._center_body(self.nom))

    @property
    def body(self) -> Type[geometries.Body]:
        return self._body

    def update_body(self):
        """Update body inplace, delete body actor and create new body actor"""
        # Retrieve transformation matrix from body actor
        v = vtk.vtkMatrix4x4()
        self._act_body.GetMatrix(v)
        T = np.empty((4, 4))
        for i in range(4):
            for j in range(4):
                T[i, j] = v.GetElement(i, j)
        # alternative method: deepcopy

        # transform the body and create a new actor
        self._body.transform(T)
        self._recreate_body()

    def _recreate_body(self):
        """delete and re-create body actor"""
        self._plotter.remove_actor(self._act_body)
        self._act_body = self._plotter.add_mesh(
            self._body.to_mesh(),
            pickable=True,
            **self.settings)
    
    def _set_body(self, name: str):
        """set a body actor"""
        self._body = getattr(geometries, name)()
        self._recreate_body()
        self._plotter.clear_slider_widgets()
        self._body.make_widgets(self)
    
    def _center_body(self, mesh: trimesh.Trimesh):
        """center the body actor at a given mesh"""
        self.update_body()
        description = self._body.asdict()
        description["center"] = mesh.center_mass
        name = description.pop("name")
        self._body = getattr(geometries, name)(**description)
        self._recreate_body()
    
    def _crop(self):
        """Crop meshes, render preview and save results"""
        self.update_body()
        try:
            act = crop(self.body, self.act)
            nom = crop(self.body, self.nom)
        except ValueError:
            d = QMessageBox()
            d.setWindowTitle("Cropping Error")
            d.setIcon(QMessageBox.Critical)
            d.setText("Selection must not be empty!")
            d.setDefaultButton(QMessageBox.Ok)
            details = "Selection Body:"
            for key, val in self.body.asdict().items():
                details += f"\n{key}: {val}"
            d.setDetailedText(details.replace("\n", "\n    "))
            d.setModal(True)
            d.exec_()
            return
            
        
        # start new window
        p = pv.Plotter(title="Cropped Meshes")
        p.add_mesh(act)
        p.add_mesh(nom, style="wireframe")
        p.show()
        
        self.cropped = (act, nom)


def crop(
    body: Type[geometries.Body],
    mesh: trimesh.Trimesh
) -> trimesh.Trimesh:
    """Crop mesh with a body
    Each triangle that has at least one vertex inside the body remains in
    the output mesh
    """
    vmask = body.contains(mesh.vertices)
    fmask = np.any(vmask[mesh.faces], axis=1)  # any == at least one inside
    ind = np.argwhere(fmask)

    if len(ind) == 0:
        raise ValueError("Selection must not be empty")
    new = mesh.submesh(ind, append=True)
    new.process()
    return new
