r"""
:mod:`gruvoc.frofile`: Tools FUN3D ``.flow`` files
===================================================================

This function reads FUN3D solution files, which have the extension
``.flow``. It reads files with the goal of writing other formats, e.g.
Tecplot ``.plt`` files.

"""

# Standard library
from io import IOBase
from typing import Union

# Third-party
import numpy as np

# Local imports
from .umeshbase import UmeshBase
from .errors import (
    assert_isinstance,
)
from .fileutils import openfile


def write_fro(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        meta: bool = False):
    r"""Write data from a mesh object to a FRO file

    :Call:
        >>> write_fro(mesh, fname)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fname*: :class:`str`
            Name of file
        *fp*: :class:`IOBase`
            File object
    :Versions:
        * 2025-07-14 ``@aburkhea``: v1.0; copied from trifile
    """
    # Check type
    assert_isinstance(mesh, UmeshBase, "mesh object to write to")
    assert_isinstance(fname_or_fp, (str, IOBase), "mesh file")
    # Open file
    with openfile(fname_or_fp, 'wb') as fp:
        # Write file
        _write_fro(mesh, fp)


def _write_fro(
        mesh: UmeshBase,
        fp: IOBase):
    # Number of components
    ncompid = np.unique(mesh.CompID).size
    # Write header line
    fp.write(f"{mesh.nTri:8d}{mesh.nNode:8d}")
    fp.write(f"{0:8d}{0:8d}{0:8d}{ncompid:8d}\n")
    # Loop through nodes
    for j, node in enumerate(mesh.nodes):
        fp.write(f"{j+1:8d}{node[0]:16g}{node[1]:16g}{node[2]:16g}\n")
    # Loop through tris
    for k, tri in enumerate(mesh.tris):
        # Get component ID for this tri
        c = mesh.CompID[k]
        # Write line
        fp.write(f"{k+1:8d}{tri[0]:8d}{tri[1]:8d}{tri[2]:8d}{c:8d}\n")


# Read from a .fro file.
def read_fro(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        meta: bool = False):
    r"""Read a triangulation (from ``*.fro``) to mesh object

    :Call:
        >>> read_fro(mesh, fname)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fname*: :class:`str`
            Name of file
        *fp*: :class:`IOBase`
            File object
    :Versions:
        * 2025-07-14 ``@aburkhea``: v1.0; copied from trifile
    """
    with open(fname_or_fp, 'r') as fp:
        _read_fro(mesh, fp, meta=meta)


def _read_fro(
        mesh: UmeshBase,
        fp: IOBase,
        meta: bool = False):
    # Read the first line
    line = fp.readline()
    # Split into parts; should have 6 comps, but we only need two
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(
            f"FRO file {fp.name} header line must have >= 2 parts")
    # Get number of tris and number of nodes
    ntri = int(parts[0])
    nnode = int(parts[1])
    # Save the statistics.
    mesh.nNode = nnode
    mesh.nTri = ntri
    mesh.nQuad = 0
    # Exit if given "meta" option
    if meta:
        return
    # Read the data for the nodes
    nodes = np.fromfile(fp, count=4*nnode, sep=' ')
    # Save
    mesh.Nodes = nodes.reshape((nnode, 4))[:, 1:]
    # Read tri node indices and tri comp ids
    tris = np.fromfile(fp, dtype="i4", count=5*ntri, sep=' ')
    tris = tris.reshape((ntri, 5))
    # Save tris and CompIDs
    mesh.Tris = tris[:, 1:4]
    mesh.CompID = tris[:, -1]
