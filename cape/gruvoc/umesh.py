r"""
:mod:`gruvoc.umesh`: Unstructured volume mesh class
====================================================

This module provides the :class:`Umesh` class that interacts with
traditional generic mixed-element volume grids. In particular, the type
of meshes considered are those whose surface elements consist of

*   triangles (3 points) and
*   quads (4 points)

and whose volume elements consist of

*   tetrahedra (4 tri faces and 4 nodes),
*   pyramids (4 tri faces, 1 quad, and 5 nodes),
*   prisms (2 tri faces, 3 quads, and 6 nodes), and
*   hexahedra (6 quad faces and 8 nodes).

"""

# Standard library
import os
import re
from collections import namedtuple
from io import IOBase
from typing import Optional, Union

# Third-party

# Local imports
from .umeshbase import UmeshBase
from .avmfile import read_avm, write_avm
from .fileutils import openfile
from .flowfile import read_fun3d_flow, read_fun3d_tavg
from .pltfile import write_plt
from .surfconfig import SurfConfig
from .surf3dfile import read_surf3d, write_surf3d
from .trifile import read_tri, write_tri, write_triq
from .ufuncfile import read_ufunc, write_ufunc
from .ugridfile import read_ugrid, write_ugrid
from .uh3dfile import read_uh3d, write_uh3d


# Regular expression to identify file name
REGEX_FILENAME = re.compile(
    r"(?P<base>[\w+_=-]+)(\.(?P<infix>\w+))?" +
    r"\.(?P<ext>[\w_]+)$")

# File types based on extension
GRID_FORMATS = (
    "avm",
    "plt",
    "tri",
    "ugrid",
    "surf",
)
# Non-obvious file types by extension
GRID_FORMAT_EXTS = {}
# Class for file types
GridFileFormat = namedtuple("GridFileFormat", ("basename", "infix", "format"))

# Descriptions of each slot
FIELD_DESCRIPTIONS = {
    "nodes": "node coordinates",
    "tris": "tri face node indices",
    "quads": "quad face node indices",
    "surf_ids": "surface face ID numbers",
    "tets": "tetrahedral cell node indices",
    "pyrs": "pyramid (5-node, 5-face) node indices",
    "pris": "prism (6-node, 5-face) node indices",
    "hexs": "hexahedral node indices",
    "ntet_bl": "number of tets in boundary layer",
    "vol_ids": "volume cell ID numbers",
    "surf_flags": "surface reconnection flags",
    "surf_bcs": "surface boundary condition flags",
    "blds": "initial normal grid spacing",
    "bldel": "total BL height for each node",
}
SLOT_GROUPS = {
    "surf_ids": ("tri_ids", "quad_ids"),
    "vol_ids": ("tet_ids", "pyr_ids", "pri_ids", "hex_ids"),
    "surf_flags": ("tri_flags", "quad_flags"),
    "surf_bcs": ("tri_bcs", "quad_bcs"),
}


def name2format(fname: str, **kw):
    r"""Infer file format based on file name and/or kwargs

    :Call:
        >>> gridfmt = name2format(fname, **kw)
    :Inputs:
        *fname*: :class:`str`
            Base name of file (name of file w/o path)
        *fmt*, *format*: {``None``} | :class:`str`
            Directly-specified file format
        *ext*, *infix*: {``None``} | :class:`str`
            Directly-specified file name infix
        *ugrid*: ``True`` | {``False``}
            Use UGRID file format
    :Outputs:
        *gridfmt*: :class:`GridFileFormat`
            Information on file format
        *gridfmt.basename*: :class:`str`
            Name of file w/o extension
        *gridfmt.infix*: ``None`` | :class:`str`
            Infix from penultimate part of file name
        *gridfmt.format*: ``None`` | :class:`str`
            Grid format based on file name or user overrides
    """
    # Identify file extension
    re_match = REGEX_FILENAME.search(fname)
    # Check for match of regular expression
    if re_match is None:
        # Could not process file name
        basename = fname
        file_infix = None
        file_ext = None
    else:
        # Get groups
        basename = re_match.group("base")
        file_infix = re_match.group("infix")
        file_ext = re_match.group("ext")
    # Get file format based on file extension
    format_ext = GRID_FORMAT_EXTS.get(file_ext, file_ext)
    # Check for override by looping through known formats
    for fmt in GRID_FORMATS:
        # Check kwargs for ``ugrid=True`` or similar
        if kw.get(fmt):
            # User override
            format_ext = fmt
            break
    # Check for user override using *fmt* kwarg
    format_ext = kw.get("format", kw.get("fmt", format_ext))
    # Check for infix
    file_infix = kw.get("infix", kw.get("ext", file_infix))
    # Output
    return GridFileFormat(basename, file_infix, format_ext)


# Base class
class Umesh(UmeshBase):
  # === Class attributes ===
    __slots__ = ()

  # === __dunder__ ===
    def __init__(
            self,
            fname_or_fp: Optional[Union[str, IOBase]] = None,
            meta: bool = False,
            **kw):
        # Intialize every slot
        for slot in UmeshBase.__slots__:
            setattr(self, slot, None)
        # Read if fname given
        if fname_or_fp is not None:
            self.read(fname_or_fp, meta, **kw)
        # Common kwargs
        fmt = kw.get("fmt")
        # Check for file names/handles by kwarg
        fugrid = kw.get("ugrid")
        favm = kw.get("avm")
        fsurf = kw.get("surf")
        ftri = kw.get("tri")
        fuh3d = kw.get("uh3d")
        # Unspecified-type config file
        fcfg = kw.get("c", kw.get("config"))
        novol = kw.get("novol", not kw.get("vol", True))
        # Check for kwarg-based formats
        if fugrid:
            # UGRID (main format)
            self.read_ugrid(fugrid, meta=meta, fmt=fmt, novol=novol)
        elif favm:
            # AVM (Kestrel) format
            self.read_avm(favm, meta=meta)
        elif fsurf:
            # SURF3D (input to AFLR3)
            self.read_surf3d(fsurf, meta=meta, fmt=fmt)
        elif ftri:
            # Cart3D TRI format
            self.read_tri(ftri, meta=meta, fmt=fmt)
        elif fuh3d:
            # UH3D mystery format
            self.read_uh3d(fuh3d, meta=meta)
        # Read config
        if fcfg:
            # Generic config
            self.config = SurfConfig(fcfg)
        elif kw.get("mapbc"):
            # MapBC surface configuration
            self.config = SurfConfig(mapbc=kw["mapbc"])
        else:
            # Create empty config
            self.config = SurfConfig()

  # === Readers ===
    def read(
            self,
            fname_or_fp: Union[str, IOBase],
            meta: bool = False,
            **kw):
        # Option to read surface only
        novol = kw.get("novol", not kw.get("vol", True))
        # Open file if able
        with openfile(fname_or_fp) as fp:
            # Get file name
            fdir, fname = os.path.split(os.path.abspath(fp.name))
            # Save
            self.fname = fname
            self.fdir = fdir
            # Read format from file name
            gridfmt = name2format(fname)
            # Status update
            if kw.get('v', False):
                print(
                    f"Reading file '{fname}' using format '{gridfmt.format}'")
            # Save base name
            self.basename = gridfmt.basename
            # Default format
            fmt = kw.get("fmt", gridfmt.infix)
            # Read data
            if gridfmt.format == "ugrid":
                # UGRID format
                self.read_ugrid(fp, meta, fmt=fmt, novol=novol)
            elif gridfmt.format == "avm":
                # AVM format
                self.read_avm(fp, meta)
            elif gridfmt.format == "surf":
                # SURF3D format
                self.read_surf3d(fp, meta, fmt=fmt)
            elif gridfmt.format == "tri":
                # TRI format
                self.read_tri(fp, meta, fmt=fmt)
            elif gridfmt.format == "uh3d":
                # UH3D format
                self.read_uh3d(fp, meta)

    def read_avm(
            self,
            fname_or_fp: Union[str, IOBase],
            meta: bool = False):
        # Read DoD CREATE/AV mesh
        read_avm(self, fname_or_fp, meta)

    def read_fun3d_flow(
            self,
            fname_or_fp: Union[str, IOBase],
            meta: bool = False):
        # Read FUN3D flow state
        read_fun3d_flow(self, fname_or_fp, meta)

    def read_fun3d_tavg(
            self,
            fname_or_fp: Union[str, IOBase],
            meta: bool = False):
        # Read FUN3D averaged flow
        read_fun3d_tavg(self, fname_or_fp, meta)

    def read_surf3d(
            self,
            fname_or_fp: Union[str, IOBase],
            meta: bool = False,
            fmt: Optional[str] = None):
        read_surf3d(self, fname_or_fp, meta, fmt)

    def read_tri(
            self,
            fname_or_fp: Union[str, IOBase],
            meta: bool = False,
            fmt: Optional[str] = None):
        read_tri(self, fname_or_fp, meta, fmt)

    def read_ugrid(
            self,
            fname_or_fp: Union[str, IOBase],
            meta: bool = False,
            fmt: Optional[str] = None,
            novol: bool = False):
        read_ugrid(self, fname_or_fp, meta, fmt, novol=novol)

    def read_uh3d(
            self,
            fname_or_fp: Union[str, IOBase],
            meta: bool = False):
        # Read nodes and tris
        read_uh3d(self, fname_or_fp, meta)
        # Read config info
        with openfile(fname_or_fp) as fp:
            self.config = SurfConfig(uh3d=fp.name)

  # === Writers ===
    def write(
            self,
            fname_or_fp: Union[str, IOBase],
            **kw):
        # Open file if able
        with openfile(fname_or_fp, 'wb') as fp:
            # Get file name
            fdir, fname = os.path.split(os.path.abspath(fp.name))
            # Save
            self.fname = fname
            self.fdir = fdir
            # Read format from file name
            gridfmt = name2format(fname)
            # Default format
            fmt = kw.get("fmt", gridfmt.infix)
            # Status update
            if kw.get('v', False):
                print(
                    f"Writing file '{fname}' using format '{gridfmt.format}'")
            # Read data
            if gridfmt.format == "ugrid":
                # UGRID format
                self.write_ugrid(fp, fmt=fmt)
            elif gridfmt.format == "avm":
                # AVM format
                self.write_avm(fp)
            elif gridfmt.format == "plt":
                # Tecplot PLT format
                self.write_plt(fp, v=kw.get("v"))
            elif gridfmt.format == "surf":
                # SURF3D format
                self.write_surf3d(fp, fmt=fmt)
            elif gridfmt.format == "tri":
                # TRI format
                self.write_tri(fp, fmt=fmt)
            elif gridfmt.format == "uh3d":
                # UH3D format
                self.write_uh3d(fp)
            elif gridfmt.format == "ufunc":
                # UFUNC format
                self.write_ufunc(fp, fmt=fmt)

    def write_plt(
            self,
            fname_or_fp: Union[str, IOBase],
            v: bool = False):
        # Write mesh
        write_plt(self, fname_or_fp, v=v)

    def write_surf3d(
            self,
            fname_or_fp: Union[str, IOBase],
            fmt: Optional[str] = None):
        write_surf3d(self, fname_or_fp, fmt)

    def write_tri(
            self,
            fname_or_fp: Union[str, IOBase],
            fmt: Optional[str] = None):
        write_tri(self, fname_or_fp, fmt)

    def write_triq(
            self,
            fname_or_fp: Union[str, IOBase],
            fmt: Optional[str] = None):
        write_triq(self, fname_or_fp, fmt)

    def write_ufunc(
            self,
            fname_or_fp: Union[str, IOBase],
            fmt: Optional[str] = None):
        # Write mesh
        write_ufunc(self, fname_or_fp, fmt)

    def write_ugrid(
            self,
            fname_or_fp: Union[str, IOBase],
            fmt: Optional[str] = None):
        # Write mesh
        write_ugrid(self, fname_or_fp, fmt)

    def write_uh3d(
            self,
            fname_or_fp: Union[str, IOBase]):
        # Write mesh
        write_uh3d(self, fname_or_fp)

    def write_avm(
            self,
            fname_or_fp: Union[str, IOBase]):
        # Write mesh
        write_avm(self, fname_or_fp)

