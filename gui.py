from __future__ import annotations
import json
import numpy as np
import tkinter as tk
import tkinter.filedialog, tkinter.messagebox, tkinter.ttk
import sys, os, logging, webbrowser, trimesh
from contextlib import suppress
from typing import Any, Type, Iterable, Dict
from sat import jsonio
from qtpy.QtWidgets import QApplication

from sat.data import Data, Mesh, Source
from sat.jsonio import data2json
from sat.visual import plotter, slicing
from sat.analysis import calculators, progress

full = Data()
crop = Data()
logger = logging.getLogger("gui")
# progress.use_tk()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Surface Analysis Tool")
        self.resizable(False, False)
        global full, crop

        self.menubar = Menubar(self)
        self.config(menu=self.menubar)

        self.parF = ParameterFrame(self)

        self.fullF = FullFrame(self)
        self.fullF.pack(fill="x", anchor="nw")

        sep = tk.Frame(self, height=12)
        sep.pack(fill="x", anchor="nw")
        
        self.cropF = CropFrame(self)
        self.cropF.pack(fill="x", anchor="nw")

        sep = tk.Frame(self, height=12)
        sep.pack(fill="x", anchor="nw")
        
        self.parF.pack(fill="x", anchor="nw")

    def report_callback_exception(self, exc, val, tb):
        """Redirect the default tkinter error handling to the logger and 
        also create a pop-up window for it
        """
        logger.error("Exception", exc_info=(exc, val, tb))
        tk.messagebox.showerror(exc.__name__, message=str(val))


class Menubar(tk.Menu):
    def __init__(self, master):
        super().__init__(master)

        fileM = tk.Menu(self)
        fileM.add_command(label="New", command=self.new)
        self.add_cascade(label="File", menu=fileM)

        helpM = tk.Menu(self)
        helpM.add_command(label="About", command=self.help)
        self.add_cascade(label="Help", menu=helpM)

    def new(self):
        # TODO: Fix weird root window interaction
        import gui
        gui.App().mainloop()

    def help(self):
        pass

class AlignSubframe(tk.Frame):
    """Alignment and rendering buttons"""

    def __init__(self, master, data):
        super().__init__(master)
        tk.Grid.columnconfigure(self, 0, weight=1)
        tk.Grid.columnconfigure(self, 1, weight=1)
        tk.Grid.columnconfigure(self, 2, weight=1)
        self.data = data

        alignB = tk.Button(self, text="Align", command=self.align)
        alignB.grid(row=0, column=0, sticky="NSEW")
        renderB = tk.Button(self, text="Render", command=self.render)
        renderB.grid(row=0, column=1, sticky="NSEW")
        distmapB = tk.Button(self, text="Distance Map", command=self.distmap)
        distmapB.grid(row=0, column=2, sticky="NSEW")

    def align(self):
        self.data.align()

    def render(self):
        plotter.render(self.data.act.mesh, self.data.nom.mesh)


    def distmap(self):
        plotter.distmap(self.data.act.mesh, self.data.nom.mesh, self.data.dist)


class ManipulationSubframe(tk.Frame):
    def __init__(self, master, meshdata: Mesh, name: str):
        super().__init__(master)
        
        self.meshdata = meshdata
        subdB = tk.Button(self, text="Subdivide " + name, command=self.subd)
        subdB.pack(fill="x", anchor="nw")
        transfB = tk.Button(self, text="Transform " + name, command=self.transf)
        transfB.pack(fill="x", anchor="nw")

    def subd(self):
        self.meshdata.subdivs += 1

    def transf(self):
        raise NotImplementedError("Manual transformation matrix input is not yet implemented")


class FullFrame(tk.LabelFrame):
    """Frame to select mesh files"""

    def __init__(self, master):
        super().__init__(master, text="Full Meshes")
        tk.Grid.columnconfigure(self, 0, weight=1)
        tk.Grid.columnconfigure(self, 1, weight=1)
        global full

        # buttons
        self.nomB = tk.Button(self, text="Open Nominal Surface",
            command=lambda: self.open("nominal surface", self.nomL, full.nom))
        self.nomB.grid(row=0, column=0, sticky="NSEW")
        self.nomL = tk.Label(self, text="")
        self.nomL.grid(row=0, column=1, sticky="NSEW")

        self.actB = tk.Button(self, text="Open Actual Surface",
            command=lambda: self.open("actual surface", self.actL, full.act))
        self.actB.grid(row=1, column=0, sticky="NSEW")
        self.actL = tk.Label(self, text="")
        self.actL.grid(row=1, column=1, sticky="NSEW")

        # subframes
        frame = tk.Frame(self)
        frame.grid(row=2, column=0, columnspan=2, sticky="NSEW")
        tk.Grid.columnconfigure(frame, 0, weight=1)
        tk.Grid.columnconfigure(frame, 1, weight=1)
        
        sep = tk.Frame(frame, height=4)
        sep.grid(row=0, column=0, columnspan=2, sticky="EW")
        
        alignSF = AlignSubframe(frame, full)
        alignSF.grid(row=1, column=0, columnspan=2, sticky="EW")
        
        sep = tk.Frame(frame, height=4)
        sep.grid(row=2, column=0, columnspan=2, sticky="EW")
        
        maniASF = ManipulationSubframe(frame, full.act, "Actual")
        maniASF.grid(row=3, column=0, sticky="EW")
        maniNSF = ManipulationSubframe(frame, full.nom, "Nominal")
        maniNSF.grid(row=3, column=1, sticky="EW")
        
        sep = tk.Frame(frame, height=4)
        sep.grid(row=4, column=0, columnspan=2, sticky="EW")
        
        self.ioF = IOFrame(frame, self.master.parF, data=full, 
                           labels=(self.nomL, self.actL))
        self.ioF.grid(row=5, column=0, columnspan=2, sticky="EW")

    def open(self, objname: str, label: tk.Label, mesh: Mesh):
        filename = tk.filedialog.askopenfilename(title=f"Select {objname}",
            initialdir="data", filetypes=[("STL", ".stl"), ("ALL", ".*")])
        if filename == "":
            return
        elif not os.path.exists(filename):
            raise FileNotFoundError(f"Could not open {objname} {filename}")
        mesh.source = Source(file=filename)
        label["text"] = os.path.basename(filename)


class CropFrame(tk.LabelFrame):
    """Frame to align meshes"""

    def __init__(self, master):
        super().__init__(master, text="Cropped Meshes")
        global crop

        self.cropB = tk.Button(self, text="Crop Meshes", command=self.crop)
        self.cropB.pack(fill="x", anchor="nw")

        # subframes
        frame = tk.Frame(self)
        frame.pack(fill="x", anchor="nw")
        tk.Grid.columnconfigure(frame, 0, weight=1)
        tk.Grid.columnconfigure(frame, 1, weight=1)
        
        sep = tk.Frame(frame, height=4)
        sep.grid(row=0, column=0, columnspan=2, sticky="EW")
        
        alignSF = AlignSubframe(frame, crop)
        alignSF.grid(row=1, column=0, columnspan=2, sticky="EW")
        
        sep = tk.Frame(frame, height=4)
        sep.grid(row=2, column=0, columnspan=2, sticky="EW")
        
        maniASF = ManipulationSubframe(frame, crop.act, "Actual")
        maniASF.grid(row=3, column=0, sticky="EW")
        maniNSF = ManipulationSubframe(frame, crop.nom, "Nominal")
        maniNSF.grid(row=3, column=1, sticky="EW")
        
        sep = tk.Frame(frame, height=4)
        sep.grid(row=4, column=0, columnspan=2, sticky="EW")
        
        self.ioF = IOFrame(frame, self.master.parF, data=crop)
        self.ioF.grid(row=5, column=0, columnspan=2, sticky="EW")

    def crop(self):
        global crop, full, app
        app.grab_set()
        plotapp = QApplication(sys.argv)
        window = slicing.CroppingTool(full.nom.mesh, full.act.mesh)
        window.show()
        plotapp.exec_()

        # set meshes
        if window.cropped is not None:
            act, nom = window.cropped

            temp = full.act.assource()
            temp["body"] = window.body
            source = Source(**temp)
            crop.act.set_source_and_mesh(source, act)

            temp = full.nom.assource()
            temp["body"] = window.body
            source = Source(**temp)
            crop.nom.set_source_and_mesh(source, nom)
        app.grab_release()


class CheckboxList(tk.Frame):
    """Frame to list many options"""
    padding = 10

    def __init__(self, master: Type[tk.Frame], title: str = ""):
        super().__init__(master)

        self.boxes = {}
        self.leadV = tk.BooleanVar(self, value=True)
        self.leadC = tk.Checkbutton(self, text=title, variable=self.leadV,
            onvalue=True, offvalue=False, command=self.setall)
        self.leadC.pack(anchor="nw")

    def addbox(self, key: str, label: str):
        if key in self.boxes:
            raise KeyError(f"Checkbox with key {key} is alredy present")
        var = tk.BooleanVar(self, value=True)
        box = tk.Checkbutton(self, text=label, variable=var, 
            onvalue=True, offvalue=False, command=self.callback)
        box.pack(padx=self.padding, anchor="nw")
        self.boxes[key] = var

    def setall(self):
        for val in self.boxes.values():
            val.set(self.leadV.get())

    def callback(self):
        self.leadV.set(all(self.values()))


    # forward some dictionary interaction to boxes
    def __getitem__(self, key: str) -> bool:
        return self.boxes[key].get()

    def __setitem__(self, key: str, value: bool) -> None:
        return self.boxes[key].set(value)

    def items(self):
        for key, val in self.boxes.items():
            yield key, val.get()

    def keys(self):
        return self.boxes.keys()

    def values(self):
        for val in self.boxes.values():
            yield val.get()


class ParameterFrame(tk.LabelFrame):
    """Frame to select which parameters to calculate"""

    def __init__(self, master):
        super().__init__(master, text="Parameters")
        tk.Grid.columnconfigure(self, 0, weight=1)
        tk.Grid.columnconfigure(self, 1, weight=0)
        tk.Grid.columnconfigure(self, 2, weight=1)

        self.spF = CheckboxList(self, title="Surface Parameters")
        for key, dname in calculators.SPC.params.items():
            self.spF.addbox(key, dname)
        self.spF.grid(row=0, column=0, sticky="NSEW")

        sep = tk.ttk.Separator(self, orient="vertical")
        sep.grid(row=0, column=1, sticky="NS")

        self.vpF = CheckboxList(self, title="Vertex Parameters")
        for key, dname in calculators.VPC.params.items():
            self.vpF.addbox(key, dname)
        self.vpF.grid(row=0, column=2, sticky="NSEW")


class IOFrame(tk.Frame):
    """Frame to import and export calculated data"""

    def __init__(self, master, parF: ParameterFrame, data: Data, labels=None):
        super().__init__(master)
        tk.Grid.columnconfigure(self, 0, weight=6)
        tk.Grid.columnconfigure(self, 1, weight=1)

        self.masterlabels = labels
        self.parF = parF
        self.data = data
        self.calcB = tk.Button(self, text="Calculate", command=self.calc)
        self.calcB.grid(row=0, column=0, columnspan=2, sticky="NSEW")
        self.saveB = tk.Button(self, text="Save", command=self.save)
        self.saveB.grid(row=1, column=0, sticky="NSEW")
        self.openB = tk.Button(self, text="Import", command=self.open)
        self.openB.grid(row=1, column=1, sticky="NSEW")

    def save(self):
        filename = tk.filedialog.asksaveasfilename(initialdir="out",
            filetypes=[("JSON", ".json"), ("ALL", ".*")])
        if filename != "":
            data2json(filename, self.data)

    def open(self):
        filename = tk.filedialog.askopenfilename(title=f"Select JSON",
            initialdir="out", filetypes=[("JSON", ".json"), ("ALL", ".*")])
        jsonio.json2data(filename, self.data)
        
        if self.masterlabels is not None:
            nom, act = self.masterlabels
            nom["text"] = os.path.basename(self.data.nom.source.file)
            act["text"] = os.path.basename(self.data.act.source.file)

    def calc(self):
        for key, val in self.parF.spF.items():
            if val:
                getattr(self.data.SPC, key)
        for key, val in self.parF.vpF.items():
            if val:
                getattr(self.data.VPC, key)


if __name__ == "__main__":
    app = App()
    app.mainloop()
