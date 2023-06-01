"""Manage the file IO of Data"""

from _ctypes import PyObj_FromPtr
from collections import OrderedDict
import numpy as np
import json, os, trimesh, re

from .analysis import calculators
from .visual.geometries import Body
from .cache import EMPTY
from .data import Data, Mesh, Source


class FixedFormat(object):
    """ Value wrapper. """
    def __init__(self, value):
        self.value = value


class FixedFormatEncoder(json.JSONEncoder):
    """Encoder objects that get formatted to strings externally and are 
    wrapped with FixedFormat
    
    Adapted from: https://stackoverflow.com/a/42721412
    """
    FORMAT_SPEC = '@@{}@@'  # Unique string pattern of FixedFormat object ids.
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))  # compile(r'@@(\d+)@@')

    def __init__(self, **kwargs):
        # Keyword arguments to ignore when encoding FixedFormat wrapped values.
        ignore = {'cls', 'indent'}

        # Save copy of any keyword argument values needed for use here.
        self._kwargs = {k: v for k, v in kwargs.items() if k not in ignore}
        super().__init__(**kwargs)

    def default(self, obj):
        if isinstance(obj, FixedFormat):
            return self.FORMAT_SPEC.format(id(obj))
        return super().default(obj)

    def iterencode(self, obj, **kwargs):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.

        # Replace any marked-up FixedFormat wrapped values in the JSON repr
        # with the json.dumps() of the corresponding wrapped Python object.
        for encoded in super().iterencode(obj, **kwargs):
            match = self.regex.search(encoded)
            if match:
                id = int(match.group(1))
                no_indent = PyObj_FromPtr(id)
                # Replace the matched id string with the formatted value
                encoded = encoded.replace(
                    '"{}"'.format(format_spec.format(id)), no_indent.value)
            yield encoded


class ToSeperateFile:
    """Wrap arrays into this to save them in a serperate files when creating
    the JSON output
    """

    def __init__(self, array: np.array, file: str, header: str):
        self.array = array
        self.file = file
        self.header = header

    def save(self):
        np.savetxt(self.file, self.array, header=self.header)


class DataEncoder(FixedFormatEncoder):
    def default(self, obj):
        # reduce array dimensions recursively, then fully print last dimension
        if isinstance(obj, np.ndarray):
            if obj.ndim > 1:
                return list(obj)
            return FixedFormat(np.array2string(obj, separator=", ", sign="-",
                                    threshold=np.inf, max_line_width=np.inf,
                                    formatter={'float_kind': '{:e}'.format}))
        elif obj is EMPTY:
            return None
        elif isinstance(obj, trimesh.Trimesh):
            # save Trimesh string representation for simple troubleshooting
            return str(obj)
        elif isinstance(obj, Body):
            # save bodies as dict
            return obj.asdict()
        return super().default(obj)


def prepare(obj, filename: str, savelist: list) -> dict:
    """Return JSON serializable representation of obj.
    Essentially like JSONEncoder.default(), but has access to
    the target filename
    """
    if isinstance(obj, (Data, Mesh)):
        # create nested dictionary from cache entries
        # needs to be exported here to access the filename later down the
        # line (in VPC export)
        out = OrderedDict()
        for key in obj.cache.keys():
            entry = obj.cache.getEntry(key)
            val = entry.get()
            if val is not EMPTY:
                out[entry.name] = prepare(val, filename, savelist)
        # reorder to place dist and corr last
        if "corresponding faces" in out.keys():
            out.move_to_end("corresponding faces", last=True)
        if "distance" in out.keys():
            out.move_to_end("distance", last=True)
        return out

    elif isinstance(obj, Source):
        # return simple dictionary form of source
        return obj.__dict__

    elif isinstance(obj, calculators.SPC):
        # return calculated parameters as displayname:value dict pairs
        out = {}
        for fn, dn in obj.params.items():
            if fn in obj._cache:
                out[dn] = obj._cache[fn]
        return out

    elif isinstance(obj, calculators.VPC):
        # stack all parameters behind vertex coordinates
        # return contents as a ToSeperateFile object
        stack = [obj.act.vertices]
        header = "x, y, z"
        for fn, dn in obj.params.items():
            if fn in obj._cache:
                arr = obj._cache[fn]
                stack.append(arr)
                header += f", {dn}" * arr.shape[1]
        name, ext = os.path.splitext(filename)
        name = name + "_vertex.txt"
        if len(stack) == 1:
            return {}
        savelist.append(ToSeperateFile(np.hstack(stack), name, header))
        return name
    return obj


def data2json(filename, data: Data):
    """Save data as json under filename"""
    savelist = []
    out = prepare(data, filename, savelist)
    with open(filename, "w") as fp:
        json.dump(out, fp, ensure_ascii=True, cls=DataEncoder, indent=4)
    for val in savelist:
        val.save()

def json2data(filename, data: Data):
    with open(filename) as fp:
        dct = json.load(fp)
    data.update(dct)
