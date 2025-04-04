r"""
:mod:`gruvoc.flowfile`: Tools FUN3D ``.flow`` files
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
    assert_value
)
from .fileutils import openfile
from ._vendor.capeio import (
    fromfile_lb4_i,
    fromfile_lb8_i,
    fromfile_lb8_f)


# Constants
DEFAULT_STRLEN = 256


# Read FUN3D .flow file
def read_fun3d_flow(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        meta: bool = False):
    r"""Read data to a mesh object from FUN3D ``.flow`` file

    :Call:
        >>> read_fun3d_flow(mesh, fname, meta=False)
        >>> read_fun3d_flow(mesh, fp, meta=False)
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
        _read_fun3d_flow(mesh, fp, meta=meta)


# Read FUN3D TAVG.1 file
def read_fun3d_tavg(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        meta: bool = False):
    r"""Read data to a mesh object from FUN3D ``.flow`` file

    :Call:
        >>> read_fun3d_tavg(mesh, fname, meta=False)
        >>> read_fun3d_tavg(mesh, fp, meta=False)
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
        _read_fun3d_tavg(mesh, fp, meta=meta)


# Read flow file
def _read_fun3d_flow(
        mesh: UmeshBase,
        fp: IOBase,
        meta: bool = False):
    # Skip first bytes
    fp.seek(288)
    # Read number of nodes
    nnode, = fromfile_lb4_i(fp, 1)
    mesh.nnode = nnode
    # Skip to byte 580 to read freestream
    fp.seek(580)
    # Fixed freestream in perfect-gas FUN3D
    mesh.qinfvars = ["mach", "Rey", "alpha", "beta", "Tinf"]
    mesh.qinf = fromfile_lb8_f(fp, 5)
    # Skip to byte 900
    fp.seek(900)
    # Read number of iterations
    niter, = fromfile_lb4_i(fp, 1)
    ncol, = fromfile_lb4_i(fp, 1)
    # Skip over iterative history
    fp.seek(niter*ncol*8, 1)
    # Skip to node-reordering
    fp.seek(272, 1)
    # Read node ordering
    if meta:
        fp.seek(nnode*8, 1)
    else:
        # Raw node orders
        jnode = fromfile_lb8_i(fp, nnode)
        # Inversion
        bnode = np.argsort(jnode)
    # Skip to CLI portion
    fp.seek(1004, 1)
    # Check next tag
    _assert_tag(fp, "n_command_lines")
    # Read number of CLI args
    _, ncmd = fromfile_lb4_i(fp, 2)
    # Loop through args
    for _ in range(ncmd):
        # Skip header
        fp.seek(600, 1)
        # Read number of int vars
        n_int_var, = fromfile_lb4_i(fp, 1)
        # Skip int vars ahd real-var header
        fp.seek(528 + n_int_var*4, 1)
        # Read number of real vars
        n_real_var, = fromfile_lb4_i(fp, 1)
        # Skip real vars and char-var header
        fp.seek(528 + n_real_var*4, 1)
        # Read number of char vars
        n_char_var, = fromfile_lb4_i(fp, 1)
        # Skip char_var
        fp.seek(268 + n_char_var*4, 1)
    # Check next string, should be "qnode"
    _assert_tag(fp, "qnode")
    # Read number of turbulence and stat vars
    _, nq = fromfile_lb4_i(fp, 2)
    # Save variable names
    mesh.nq = nq
    if nq == 5:
        mesh.qvars = ["rho", "u", "v", "w", "p"]
    # Exit if *meta* mode
    if meta:
        return
    # Skip 3 more ints
    fp.seek(12, 1)
    # Read state
    q = fromfile_lb8_f(fp, nnode*nq).reshape((nnode, nq))
    # Unpack
    rho, ru, rv, rw, e0 = q.T
    # Calculate normalized velocity magnitude times density (squared)
    rhov2 = ru*ru + rv*rv + rw*rw
    # Calculate pressure
    p = 0.4*(e0 - 0.5/rho*rhov2)
    # Reset to "usual" normalized velocity
    q[:, 1] = ru / rho
    q[:, 2] = rv / rho
    q[:, 3] = rw / rho
    # Save pressure
    q[:, 4] = p
    # Reorder and save it
    mesh.q = q[bnode, :]


# Read flow file
def _read_fun3d_tavg(
        mesh: UmeshBase,
        fp: IOBase,
        meta: bool = False):
    # Skip first bytes
    fp.seek(8)
    # Read number of iterations in average
    mesh.niter, = fromfile_lb4_i(fp, 1)
    # Skip 8 bytes; these may be important?
    fp.seek(8, 1)
    # Read number of states
    nq, = fromfile_lb4_i(fp, 1)
    mesh.nq = nq
    # Skip byte, should be ``1``?
    i, = fromfile_lb4_i(fp, 1)
    assert_value(i, 1, "TAVG file type")
    # Use number of nodes, needed
    nnode = mesh.nnode
    # Fixed freestream in perfect-gas FUN3D
    mesh.qinfvars = ["mach", "alpha"]
    mesh.qinf = fromfile_lb8_f(fp, len(mesh.qinfvars))
    # Exit if *meta*
    if meta:
        return
    # Skip two more bytes
    fp.seek(52)
    # Read number of zones
    nzone, = fromfile_lb4_i(fp, 1)
    # Skip number of nodes in each zone
    fp.seek(nzone*4, 1)
    # Raw node orders
    jnode = fromfile_lb4_i(fp, nnode)
    # Inversion
    bnode = np.argsort(jnode)
    # Save variable names
    if nq == 5:
        mesh.qvars = [
            "rho", "u", "v", "w", "p",
        ]
    # Read RMS and then average state
    q = fromfile_lb8_f(fp, 2*nnode*nq).reshape((nnode, 2*nq))
    qavg = q[:, :nq]
    # Reorder and save it
    mesh.q = qavg[bnode, :]


# Check tag
def _assert_tag(fp: IOBase, targ: str, n: int = DEFAULT_STRLEN):
    # Read next string
    tag = _read_strn(fp, n)
    # Check value
    assert_value(tag, targ, f"{fp.name} at {fp.tell() - n}")


# Fixed-length strings
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
    # Encode
    return buf.decode("utf-8").rstrip()
