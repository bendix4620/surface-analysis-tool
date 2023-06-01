# Surface Analysis Tool for Additively Manufactured Parts
A program for surface characterization of arbitrarily shaped triangulated surface meshes, commonly encountered in analying addtively manufactured parts.

## Installation
Close the repository (or download it's contents manually):
```shell
git clone https://github.com/bendix4620/surface-analysis-tool.git
```
Install dependencies:
```shell
python -m pip install -r requirements.txt
```
Verify the installation:
```shell
cd path/to/repository
python test.py
```

## Usage
The program is usable with a graphical user interface (GUI) or a command line interface (CLI)
```shell
python gui.py
python cli.py --help
```


## Parameter Definitions
| Parameter | Dimension | Definition | Description | Source |
|-----------|-----------|------------|-------------|--------|
| Sa  | Length | Average absolute height | [1] |
| Sp  | Length | Maximum peak height | [1] |
| Sv  | Length | Maximum valley depth | [1] |
| Sz  | Length | Maximum height of the scale limited surface | [1] |
| Sdr | - | Developed interfacial ratio | [1] |
| Sdr'| - | Alternative definition of Sdr | [1] |
| Srf | - | Re-entrant feature ratio | [1] |
| Srr | - | Alternative re-entrant feature ratio | [2] |
| Angle Difference | Angle | Average angle between a vertex normal and all neighbor's vertex normals in degree | [3] |
| Principle Curvature | Length | The two principle axes of curvature for each vertex | [4] |


## Notes
| Term | Explanation |
|------|-------------|
| Nominal Surface | The ideal surface |
| Actual Surface  | The real/measured surface |
| Form Surface | The form of the actual surface. Acquired by per-triangle projection of the actual surface onto the closest triangle of the nominal surface (as opposed to the method used in [1]) |
| Subdivide | Split each triangle of a triangle mesh into 4 triangles of equal size |


## Units
The program does not deal with units. The imported meshes are expected to have the same unit, which will be the unit of all parameters with the dimension 'length'

## Sources
[1]: Pagani, L., “Towards a new definition of areal surface texture parameters on freeform surface: Re-entrant features and functional parameters”, <i>Measurements</i>, vol. 141, pp. 442–459, 2019. https://doi.org/10.1016/j.measurement.2019.04.027.

[2]: Fritsch, Tobias. (2019): A multiscale analysis of additively manufactured lattice structures. Potsdam, Universität Potsdam. https://doi.org/10.25932/publishup-47041.

[3]: Wilke, Wilhelm. (2002): Segmentierung und Approximation großer Punktwolken. Darmstadt, Technische Universität. https://tuprints.ulb.tu-darmstadt.de/255/

[4]: S. Rusinkiewicz, "Estimating curvatures and their derivatives on triangle meshes," Proceedings. 2nd International Symposium on 3D Data Processing, Visualization and Transmission, 2004. 3DPVT 2004., Thessaloniki, Greece, 2004, pp. 486-493, https://doi.org/10.1109/TDPVT.2004.1335277.
