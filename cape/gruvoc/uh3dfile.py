r"""
:mod:`gruvoc.trifile`: Tools for reading Cart3D ``tri`` files
===================================================================

One key functionality of this module is to determine the file type of
``.ugrid`` files or determine that they are not recognizable files of
that format.

"""

# Standard library
import os
import re
from collections import namedtuple
from io import IOBase
from typing import Callable, Optional, Union

# Third-party
import numpy as np

# Local imports
from .umeshbase import UmeshBase
from .errors import GruvocValueError, assert_isinstance
from .fileutils import openfile


# Special file type
UH3DFileType = namedtuple(
    "UH3DFileType", [
        "fmt",
        "byteorder",
        "filetype",
        "precision",
    ])

# Known file formats
DATA_FORMATS = (
    None,
)
# Regular expression to identify file fmtension
REGEX_UH3D = re.compile(r"\.uh3d$")


# Read UH3D file
def read_uh3d(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        meta: bool = False,
        fmt: Optional[str] = None):
    r"""Read data to a mesh object from ``.tri`` file

    :Call:
        >>> read_uh3d(mesh, fname, meta=False, fmt=None)
        >>> read_uh3d(mesh, fp, meta=False, fmt=None)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fname*: :class:`str`
            Name of file
        *fp*: :class:`IOBase`
            File object
        *meta*: ``True`` | {``False``}
            Read only metadata (number of nodes, tris, etc.)
        *fmt*: {``None``} | :class:`str`
            Manual data format, ``"l?[br][48]l?"``
    """
    # Check type
    assert_isinstance(mesh, UmeshBase, "mesh object to store data in")
    assert_isinstance(fname_or_fp, (str, IOBase), "mesh file")
    # Open file
    with openfile(fname_or_fp, 'rb') as fp:
        # Determine mode
        mode = get_uh3d_mode(fp, fmt)
        # Check if unidentified
        if mode is None:
            raise GruvocValueError(
                f"Unable to recognize UH3D file format for '{fp.name}'")
        # Get format
        fmt = mode.fmt
        # Get appropriate int and float readers for this format
        iread = read_ascii_i
        fread = read_ascii_f
        # Read file
        _read_uh3d(
            mesh,
            fp, iread, fread, meta=meta)


# Write UH3D file
def write_uh3d(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        fmt: Optional[str] = None):
    r"""Write data from a mesh object to ``.tri`` file

    :Call:
        >>> write_uh3d(mesh, fname, fmt=None)
        >>> write_uh3d(mesh, fp, fmt=None)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fname*: :class:`str`
            Name of file
        *fp*: :class:`IOBase`
            File object
        *fmt*: {``None``} | :class:`str`
            Manual data format, ``"l?[br][48]l?"``
    """
    # Check type
    assert_isinstance(mesh, UmeshBase, "mesh object to write to")
    assert_isinstance(fname_or_fp, (str, IOBase), "mesh file")
    # Open file
    with openfile(fname_or_fp, 'wb') as fp:
        # Determine file type
        mode = get_uh3d_mode_fname(fp.name, fmt)
        # Check if unidentified
        if mode is None:
            raise GruvocValueError(
                f"Unable to recognize UH3D file format for '{fp.name}'")
        # Write file
        _write_uh3d(mesh, fp)


def read_ascii_i(fp: IOBase, n: int):
    return np.fromfile(fp, sep=", ", count=n, dtype="int")


def read_ascii_f(fp: IOBase, n: int):
    return np.fromfile(fp, sep=", ", count=n, dtype="float")


def _read_uh3d(
        mesh: UmeshBase,
        fp: IOBase,
        iread: Callable,
        fread: Callable,
        meta: bool = False):
    # Resulting settings induced from ugrid
    mesh.mesh_type = "unstruc"
    # Save location
    mesh.path = os.path.dirname(os.path.abspath(fp.name))
    mesh.name = os.path.basename(fp.name).split(".")[0]
    # Skip first line
    fp.readline()
    # Read sizing parameters
    ns = iread(fp, 6)
    # Check size
    if ns.size == 6:
        # Unpack all parameters
        nnode, ntri, nface = ns[[0, 2, 4]]
    else:
        # Error
        raise GruvocValueError(
            "Failed to read expected 6 integers from uh3d header")
    # Save parameters
    mesh.nnode = nnode
    mesh.ntri = ntri
    mesh.nface = nface
    # Exit if given "meta" option
    if meta:
        return
    # Initialize arrays
    nodes = np.zeros((nnode, 3), dtype="float64")
    tris = np.zeros((ntri, 3), dtype="int32")
    tri_ids = np.zeros(ntri, dtype="int32")
    # Loop through nodes
    for i in np.arange(nnode):
        # Read row
        nodei = fread(fp, 4)
        # Save result
        nodes[i, :] = nodei[1:]
    # Loop through tris
    for k in np.arange(ntri):
        # Read row
        trii = iread(fp, 5)
        # Save result
        tris[k, :] = trii[1:4]
        tri_ids[k] = trii[4]
    # Save arrays
    mesh.nodes = nodes
    mesh.tris = tris
    mesh.tri_ids = tri_ids
    # Create empty quads
    mesh.quads = np.zeros((0, 4), dtype="int32")
    mesh.quad_ids = np.zeros(0, dtype="int32")
    mesh.nquad = 0

# Write tri file
def _write_uh3d(
        mesh: UmeshBase,
        fp: IOBase):
    # Write blank line
    fp.write(b" file created by **gruvoc**\n")
    # Unique triangle IDs
    ids = np.unique(mesh.tri_ids)
    # Number of components
    ncomp = ids.size
    # Write counts
    ns = np.array([mesh.nnode, mesh.nnode, mesh.ntri, mesh.ntri, ncomp, ncomp])
    ns.tofile(fp, sep=", ")
    fp.write(b'\n')
    # Loop through nodes
    for i in np.arange(mesh.nnode):
        # Write node index
        fp.write(b"%i, " % (i + 1))
        # Write node coordinates
        mesh.nodes[i].tofile(fp, sep=", ")
        # End line
        fp.write(b'\n')
    # Loop through tris
    for k in np.arange(mesh.ntri):
        # Create array of tri index, node indices, and BC index
        tk = np.hstack((k + 1, mesh.tris[k], mesh.tri_ids[k]))
        # Write to file
        tk.tofile(fp, sep=", ")
    # Write faces
    for j in ids:
        # Get name
        face = mesh.config.get_name(j + 1)
        # Write comp and name (with quotes)
        fp.write(b"%i, %r\n" % (j, face))
    # Final line
    fp.write(b'99,99,99,99,99\n')


# Check ASCII mode
def check_uh3d_ascii(fp):
    # Remember position
    pos = fp.tell()
    # Go to beginning of file
    fp.seek(0)
    # Safely move around file
    try:
        # Ignore first line
        fp.readline()
        # Read second line
        line = fp.readline()
        # Convert to two integers
        ns = np.array([int(part) for part in line.split(b',')])
        # If that had two ints, we're in good shape (probably)
        if ns.size != 6:
            return
        return UH3DFileType("ascii", None, None, None)
    except ValueError:
        return
    finally:
        # Reset *fp* position
        fp.seek(pos)


#: Dictionary of mode checkers
_UH3D_MODE_CHECKERS = {
    None: check_uh3d_ascii,
}


def get_uh3d_mode(
        fname_or_fp: Union[IOBase, str],
        fmt: Optional[str] = None) -> Optional[UH3DFileType]:
    r"""Identify UH3D file format if possible

    :Call:
        >>> mode = get_uh3d_mode(fname_or_fp, fmt=None)
    :Outputs:
        *mode*: ``None`` | :class:`UH33DFileType`
            File type, always ASCII for UH3D
    """
    # Check for name of nonexistent file
    if isinstance(fname_or_fp, str) and not os.path.isfile(fname_or_fp):
        # Get file mode from kwarg and file name alone
        return get_uh3d_mode_fname(fname_or_fp, fmt)
    # Get file
    fp = openfile(fname_or_fp, 'rb')
    # Get name of file
    fname = fp.name
    # Get initial format from file name
    fmt_fname = get_uh3d_mode_fname(fname, fmt)
    # Copy list of modes
    modelist = list(DATA_FORMATS)
    # Check for valid mode
    if fmt_fname in DATA_FORMATS:
        # Move preferred mode to the beginning
        modelist.remove(fmt_fname)
        modelist.insert(0, fmt_fname)
    # Check for explicit input by type flag
    for fmtj in modelist:
        # Get checker
        func = _UH3D_MODE_CHECKERS[fmtj]
        # Apply suggested checker
        mode = func(fp)
        # Check if it worked
        if isinstance(mode, UH3DFileType):
            # Check if unexpected result
            if mode.fmt != fmt_fname and fmt_fname != "ascii":
                UserWarning(
                    f"Expected format '{fmt_fname}' based on file name " +
                    f"but found '{mode.fmt}'")
            # Output
            return mode


def get_uh3d_mode_fname(
        fname: str,
        fmt: Optional[str] = None) -> UH3DFileType:
    # Only on format
    return UH3DFileType("ascii", None, None, None)

