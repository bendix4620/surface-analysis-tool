"""Provide visualization of meshes based on pyvista (vtk)"""

import sys
from functools import wraps
from typing import Any
from pyvistaqt import QtInteractor
from qtpy.QtWidgets import QApplication, QAction, QMainWindow, QToolBar 

class RenderWindow(QMainWindow):
    """PyQt window that renders a pyvista plotter
    See https://github.com/pyvista/pyvista-support/issues/139 and
    https://github.com/pyvista/pyvistaqt for reference
    
    Usage:
    -------
    app = QApplication(sys.argv)
    window = RenderWindow(pyvista.Sphere(), pyvista.Box())
    window.show()
    app.exec_()
    """

    def __init__(self, *meshes, title="Renderer"):
        """ Generate window with a dock """
        QMainWindow.__init__(self)
        self.setWindowTitle(title)

        # initialize the plotter
        self._plotter = QtInteractor(self) # idential to pyvista.Plotter
        self.setCentralWidget(self._plotter.interactor)

        # add a mesh
        self._plotter.add_axes()
        self._plotter.enable_trackball_style()
        self._plotter.camera.zoom(1.0)
        self._actor_style = False
        for mesh in meshes:
            self._plotter.add_mesh(mesh, pickable=True)

        self.add_toolbar_base()


    def _add_action(self, tool_bar: QToolBar, key: str, method: Any) -> None:
        """Register a Qt action"""
        action = QAction(key, self)
        action.triggered.connect(method)
        tool_bar.addAction(action)
        return action

    def view_vector(self, *args, **kwarg):
        """Wrap ``Renderer.view_vector``."""
        self.renderer.view_vector(*args, **kwarg)

    @property
    def plotter(self):
        return self._plotter
    
    def add_toolbar_base(self):
        self.base_toolbar = self.addToolBar("Camera Position")
        self.base_toolbar.setMovable(False)
        
        cvec_setters = {
            # Viewing vector then view up vector
            "Top (-Z)": lambda: self.view_vector((0, 0, 1), (0, 1, 0)),
            "Bottom (+Z)": lambda: self.view_vector((0, 0, -1), (0, 1, 0)),
            "Front (-Y)": lambda: self.view_vector((0, 1, 0), (0, 0, 1)),
            "Back (+Y)": lambda: self.view_vector((0, -1, 0), (0, 0, 1)),
            "Left (-X)": lambda: self.view_vector((1, 0, 0), (0, 0, 1)),
            "Right (+X)": lambda: self.view_vector((-1, 0, 0), (0, 0, 1)),
            "Isometric": lambda: self.view_vector((1, 1, 1), (0, 0, 1)),
        }
        for key, method in cvec_setters.items():
            self._add_action(self.base_toolbar, key, method)

        self.base_toolbar.addSeparator()
        self._add_action(self.base_toolbar, 
                         "Toggle Relative Panning", 
                         self.toggle_actor_style)
        
        self.addToolBarBreak()
        return
    
    def toggle_actor_style(self):
        """Change how camera movements interact with actors"""
        if self._actor_style:
            self._plotter.enable_trackball_style()
            self._actor_style = False
        else:
            self._plotter.enable_trackball_actor_style()
            self._actor_style = True
        return
    
    @property
    def renderer(self):
        """The active renderer of the plotter"""
        return self._plotter.renderers.active_renderer


def render(act, nom):
    """Render two meshes
    
    Parameters
    ----------
    act: trimesh.Trimesh
        Actual surface
    nom: trimesh.Trimesh
        Nominal surface
    """
    app = QApplication(sys.argv)
    window = RenderWindow(title="Meshes")
    window.plotter.add_mesh(act)
    window.plotter.add_mesh(nom, style="wireframe")
    window.show()
    app.exec_()

def distmap(act, nom, dist):
    """Render a distance map
    
    Parameters
    ----------
    act: trimesh.Trimesh
        Actual surface
    nom: trimesh.Trimesh
        Nominal surface
    dist: float(n, )
        Shortest absolute distance between the triangle centers of act and 
        the mesh nom
    """
    app = QApplication(sys.argv)
    window = RenderWindow(title="Meshes")
    window.plotter.add_mesh(act, scalars=dist, show_scalar_bar=False)
    window.plotter.add_scalar_bar(title="Absolute distance", interactive=False)
    window.plotter.add_mesh(nom, style="wireframe", color=[44, 1, 54]) # viridis
    window.show()
    app.exec_()
