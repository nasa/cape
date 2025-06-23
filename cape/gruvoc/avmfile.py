r"""
:mod:`gruvoc.avmfile`: Tools for reading CREATE-AV ``avm`` files
===================================================================

This module provides functions to read and write ``.avm`` files, which
are the main unstructured grid format for Kestrel.

"""

# Standard library
from io import IOBase
from typing import Optional, Union

# Third-party
import numpy as np

# Local imports
from .errors import (
    GruvocNotImplementedError,
    GruvocValueError,
    assert_isinstance)
from .fileutils import openfile
from .surfconfig import SurfConfig
from .umeshbase import UmeshBase
from ..capeio import (
    fromfile_lb4_i,
    fromfile_lb8_f,
    tofile_lb4_i,
    tofile_lb8_f)


# Constants
AVMESH_VERSION = np.array([1, 2, 1], dtype="int32")
DEFAULT_BC = "unspecified"
DEFAULT_COORD_SYSTEM = "xByRzU"
DEFAULT_FARFIELD = "uniform"
DEFAULT_MESH_TYPE = "unstruc"
DEFAULT_STRLEN = 32
DEFAULT_UNITS = "in"


# Read AVM file
def read_avm(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        meta: bool = False):
    r"""Read data to a mesh object from ``.avm`` file

    :Call:
        >>> read_avm(mesh, fname, meta=False)
        >>> read_avm(mesh, fp, meta=False)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fname*: :class:`str`
            Name of file
        *fp*: :class:`IOBase`
            File object
        *meta*: ``True`` | {``False``}
            Read only metadata (number of nodes, tris, etc.)
    """
    # Check type
    assert_isinstance(mesh, UmeshBase, "mesh object to store data in")
    assert_isinstance(fname_or_fp, (str, IOBase), "mesh file")
    # Open file
    with openfile(fname_or_fp, 'rb') as fp:
        # Read file
        _read_avm(mesh, fp, meta=meta)


# Write AVM file
def write_avm(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase]):
    r"""Write data from a mesh object to ``.avm`` file

    :Call:
        >>> write_avm(mesh, fname, fmt=None)
        >>> write_avm(mesh, fp, fmt=None)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fname*: :class:`str`
            Name of file
        *fp*: :class:`IOBase`
            File object
    """
    # Check type
    assert_isinstance(mesh, UmeshBase, "mesh object to write to")
    assert_isinstance(fname_or_fp, (str, IOBase), "mesh file")
    # Open file
    with openfile(fname_or_fp, 'wb') as fp:
        # Write file
        _write_avm(mesh, fp)


# Read AVM file
def _read_avm(
        mesh: UmeshBase,
        fp: IOBase,
        meta: bool = False):
    # Read header
    header = fp.read(6)
    # Check
    if header != b"AVMESH":
        raise GruvocValueError(
            b"Expected header 'AVMESH' but got '%s'" % header)
    # Read version number (ignore)
    fromfile_lb4_i(fp, 3)
    # Read 4 comments
    mesh.comment1 = _read_strn(fp)
    mesh.comment2 = _read_strn(fp)
    mesh.comment3 = _read_strn(fp)
    mesh.comment4 = _read_strn(fp)
    # Three more integers whose meaning is unknown
    fromfile_lb4_i(fp, 3)
    # Read the "name"
    mesh.name = _read_strn(fp)
    # First set of zeros
    _read_zeros(fp, 24)
    # Read mesh type
    mesh.mesh_type = _read_strn(fp)
    # Many zeros, probably for the motion section
    _read_zeros(fp, 56)
    # Read coordinate system
    mesh.coord_system = _read_strn(fp)
    _read_zeros(fp, 24)
    # Read a float for scale factor
    scale, = fromfile_lb8_f(fp, 1)
    # Check if it's unity
    if np.abs(scale - 1.0) > 1e-4:
        raise GruvocValueError(
            "Expected scaling constant 1.0; got %.2e" % scale)
    # Read units
    mesh.units = _read_strn(fp)
    _read_zeros(fp, 24)
    # Read more scale factors
    r = fromfile_lb8_f(fp, 4)
    if np.max(np.abs(r - 1.0)) > 1e-4:
        print(
            "WARNING: unrecognized values in bytes " +
            f"{fp.tell() - 32}-{fp.tell()}")
    # Initialize config
    mesh.config = SurfConfig()
    # Unusual number of zeros
    _read_zeros(fp, 71)
    # Read basic parameters
    mesh.nnode, nface, nvol = fromfile_lb4_i(fp, 3)
    # These number seem to be [4, 8, 6] or [4, 6, 5], but I don't know hy
    fromfile_lb4_i(fp, 3)
    # Boundary condition type
    bc = _read_strn(fp)
    # check if implemented
    if bc != "uniform":
        raise GruvocNotImplementedError(f"Unrecognized BC type '{bc}'")
    # Again unknown, [1, 1]
    fromfile_lb4_i(fp, 2)
    # Number of patches
    mesh.nface, = fromfile_lb4_i(fp, 1)
    # Volume cell counts
    nhex, ntet, npri, npyr = fromfile_lb4_i(fp, 4)
    # Save counts
    mesh.ntet = ntet
    mesh.npyr = npyr
    mesh.npri = npri
    mesh.nhex = nhex
    # Surface counts
    ntri, _, nquad, _ = fromfile_lb4_i(fp, 4)
    # Save counts
    mesh.ntri = ntri
    mesh.nquad = nquad
    # Five more zeros
    _read_zeros(fp, 5)
    # Read the face information
    for i in range(mesh.nface):
        # Read name of face
        name = _read_strn(fp)
        # Boundary condition
        bc = _read_strn(fp, 16)
        # ID number
        face_id = -fromfile_lb4_i(fp, 1)[0]
        # Save it
        mesh.config.add_face(name, face_id)
        # Save boundary condition
        mesh.config.props[name]["kestrel_bc"] = bc
    # Exit if *meta* option specified
    if meta:
        return
    # Read node coordinates
    nodes = fromfile_lb8_f(fp, mesh.nnode*3)
    mesh.nodes = nodes.reshape((mesh.nnode, 3))
    # Read and save triangle elements
    elems = fromfile_lb4_i(fp, ntri*4).reshape((ntri, 4))
    mesh.tris = elems[:, :3]
    mesh.tri_ids = -elems[:, 3]
    # Read and save quad elements
    elems = fromfile_lb4_i(fp, nquad*5).reshape((nquad, 5))
    mesh.quads = elems[:, :4]
    mesh.quad_ids = -elems[:, 4]
    # Read and save hex volume elements
    mesh.hexs = fromfile_lb4_i(fp, nhex*8).reshape((nhex, 8))
    mesh.tets = fromfile_lb4_i(fp, ntet*4).reshape((ntet, 4))
    mesh.pris = fromfile_lb4_i(fp, npri*6).reshape((npri, 6))
    # Read pyramids in alternate node ordering
    pyrs = fromfile_lb4_i(fp, npyr*5).reshape((npyr, 5))
    # Reorder and save
    mesh.pyrs = pyrs[:, [0, 3, 4, 1, 2]]


# Write AVM file
def _write_avm(
        mesh: UmeshBase,
        fp: IOBase):
    # Write header
    fp.write(b"AVMESH")
    # Write version integers
    tofile_lb4_i(fp, AVMESH_VERSION)
    # Write comments
    _write_strn(fp, mesh.comment1)
    _write_strn(fp, mesh.comment2)
    _write_strn(fp, mesh.comment3)
    _write_strn(fp, mesh.comment4)
    # Mystery array
    tofile_lb4_i(fp, np.array([2, 3, 0], dtype="int32"))
    # Write name
    _write_strn(fp, mesh.name)
    # Add a bunch of zeros
    tofile_lb4_i(fp, np.zeros(24, dtype="int32"))
    # Write mesh type ("unstruc")
    _write_strn(fp, _getattr_str(mesh, "mesh_type", DEFAULT_MESH_TYPE))
    # More zeros related to mesh motion
    tofile_lb4_i(fp, np.zeros(56, dtype="int32"))
    # Write coordinate system
    _write_strn(fp, _getattr_str(mesh, "coord_system", DEFAULT_COORD_SYSTEM))
    # More zeros
    tofile_lb4_i(fp, np.zeros(24, dtype="int32"))
    # Units
    tofile_lb8_f(fp, 1.0)
    _write_strn(fp, _getattr_str(mesh, "units", DEFAULT_UNITS))
    # Again with the zeros
    tofile_lb4_i(fp, np.zeros(24, dtype="int32"))
    # Not sure what these float ones mean
    tofile_lb8_f(fp, np.ones(4))
    # An unusual number of zeros
    tofile_lb4_i(fp, np.zeros(39, dtype="int32"))
    # Write the mesh "number"
    _write_strn(fp, "Mesh 1", 8)
    tofile_lb4_i(fp, np.zeros(30, dtype="int32"))
    # Extract sizes
    ntria = _getattr_int(mesh, "ntri")
    nquad = _getattr_int(mesh, "nquad")
    ntet = _getattr_int(mesh, "ntet")
    npyr = _getattr_int(mesh, "npyr")
    npri = _getattr_int(mesh, "npri")
    nhex = _getattr_int(mesh, "nhex")
    # Total elements for surface and vol
    nvol = ntet + npyr + npri + nhex
    nface = (ntria + nquad + 4*ntet + 5*npyr + 5*npri + 6*nhex) // 2
    # Write sizes
    tofile_lb4_i(fp, mesh.nnode)
    tofile_lb4_i(fp, nface)
    tofile_lb4_i(fp, nvol)
    # Mystery information
    tofile_lb4_i(fp, np.array([4, 6, 5]))
    # Farfield type
    _write_strn(fp, _getattr_str(mesh, "farfield_type", DEFAULT_FARFIELD))
    # Two more ones for some reason
    tofile_lb4_i(fp, np.ones(2, dtype="int32"))
    # Write number of faces
    tofile_lb4_i(fp, mesh.get_nface())
    # Other counts
    tofile_lb4_i(fp, mesh.nhex)
    tofile_lb4_i(fp, mesh.ntet)
    tofile_lb4_i(fp, mesh.npri)
    tofile_lb4_i(fp, mesh.npyr)
    tofile_lb4_i(fp, mesh.ntri)
    tofile_lb4_i(fp, mesh.ntri)
    tofile_lb4_i(fp, mesh.nquad)
    tofile_lb4_i(fp, mesh.nquad)
    # Not sure about these remaining counts
    tofile_lb4_i(fp, np.zeros(5, dtype="int32"))
    # Write patch names
    for surf_id in mesh.get_surfzone_ids():
        # Get name
        name = mesh.config.get_name(surf_id)
        # Get boundary condition
        bc = DEFAULT_BC
        # Write the name and boundary condition
        _write_strn(fp, name)
        _write_strn(fp, bc, n=16)
        # Write the index
        tofile_lb4_i(fp, -np.abs(surf_id))
    # Write nodes
    tofile_lb8_f(fp, mesh.nodes)
    # Triangular faces (w/ IDs as fourth col)
    faces = np.hstack(
        (mesh.tris, np.array(-mesh.tri_ids, ndmin=2).T))
    tofile_lb4_i(fp, faces)
    # Quadrilateral faces (w/ IDs as fifth col)
    faces = np.hstack(
        (mesh.quads, np.array(-mesh.quad_ids, ndmin=2).T))
    tofile_lb4_i(fp, faces)
    # Write volumes
    tofile_lb4_i(fp, mesh.hexs)
    tofile_lb4_i(fp, mesh.tets)
    tofile_lb4_i(fp, mesh.pris)
    tofile_lb4_i(fp, mesh.pyrs[:, [0, 3, 4, 1, 2]])


# Default value
def _getattr_str(mesh: UmeshBase, attr: str, vdef: str = '') -> str:
    # Get value
    v = getattr(mesh, attr)
    # Replace ``None``
    if v is None:
        return vdef
    else:
        return v


# Default attribute
def _getattr_int(mesh: UmeshBase, attr: str, vdef: int = 0) -> int:
    # Get value
    v = getattr(mesh, attr)
    # Replace ``None``
    if v is None:
        return vdef
    else:
        return v


# Weird AVM strings
def _read_strn(fp: IOBase, n: int = DEFAULT_STRLEN):
    r"""Read string from next *n* bytes

    :Call:
        >>> txt = _read_str(fp, n=32)
    :Inputs:
        *fp*: :class:`file`
            File handle open "rb"
        *n*: {``32``} | :class:`int` > 0
            Number of bytes to read
    :Outputs:
        *txt*: :class:`str`
            String decoded from *n* bytes w/ null chars trimmed
    """
    # Read *n* bytes
    buf = fp.read(n)
    # Convert to list of bytes
    bx = tuple(buf)
    # Check if ``0`` (null char) is present
    if 0 in bx:
        # Find the *first* one
        i = bx.index(0)
        # Trim buffer
        txt = buf[:i]
    else:
        # Use whole buffer
        txt = buf
    # Encode
    return txt.decode("utf-8")


def _write_strn(
        fp: IOBase,
        txt: Optional[Union[bytes, str]],
        n: int = DEFAULT_STRLEN):
    # Encode and buffer
    buf = _encode_strn(txt, n)
    # Write it
    fp.write(buf)


def _encode_strn(txt: Optional[Union[bytes, str]], n: int = DEFAULT_STRLEN):
    # Check for ``None``
    if txt is None:
        return b"\x00" * n
    # Encode if needed
    if isinstance(txt, bytes):
        # Already encoded
        buf = txt
    else:
        buf = txt.encode("utf-8")
    # Pad
    return buf[:n] + b"\x00" * max(0, n - len(buf))


def _read_zeros(fp: IOBase, n: int):
    # Skip specified number of int32s
    r = fromfile_lb4_i(fp, n)
    # Check for nonzero
    if np.any(r):
        print(
            "WARNING: found nonzero bytes in section %i-%i"
            % (fp.tell() - n*4, fp.tell()))
