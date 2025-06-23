r"""
:mod:`gruvoc.pltfile`: Tools for Tecplot (R) ``.plt`` files
===================================================================

This function reads and writes surface/volume grids to and from Tecplot
files to unstructured mesh objects. These files have a fixed data
format, so there is no need to identify little/big endian, etc.

"""

# Standard library
import sys
import shutil
from io import IOBase
from typing import Union

# Third-party
import numpy as np

# Local imports
from .umeshbase import UmeshBase
from .errors import assert_isinstance
from .fileutils import openfile
from ..capeio import (
    tofile_lb4_i,
    tofile_lb4_f,
    tofile_lb4_s,
    tofile_lb8_f)


# Zone types
ORDERED = 0
FELINESEG = 1
FETRIANGLE = 2
FEQUADRILATERAL = 3
FETETRAHEDRON = 4
FEBRICK = 5
FEPOLYGON = 6
FEPOLYHEDRON = 7
ZONETYPE_NAMES = {
    ORDERED: "ORDERED",
    FELINESEG: "FELINESEG",
    FETRIANGLE: "FETRIANGLE",
    FEQUADRILATERAL: "FEQUADRILATERAL",
    FETETRAHEDRON: "FETETRAHEDRON",
    FEBRICK: "FEBRICK",
    FEPOLYGON: "FEPOLYGON",
    FEPOLYHEDRON: "FEPOLYHEDRON",
}
ZONETYPE_CODES = {
    "ORDERED": ORDERED,
    "FELINESEG": FELINESEG,
    "FETRIANGLE": FETRIANGLE,
    "FEQUADRILATERAL": FEQUADRILATERAL,
    "FETETRAHEDRON": FETETRAHEDRON,
    "FEBRICK": FEBRICK,
    "FEPOLYGON": FEPOLYGON,
    "FEPOLYHEDRON": FEPOLYHEDRON,
}


def write_plt(
        mesh: UmeshBase,
        fname_or_fp: Union[str, IOBase],
        v: bool = False):
    r"""Write data from a mesh object to Tecplot ``.plt`` file

    :Call:
        >>> write_plt(mesh, fname, v=False)
        >>> write_plt(mesh, fp, v=False)
    :Inputs:
        *mesh*: :class:`Umesh`
            Unstructured mesh object
        *fname*: :class:`str`
            Name of file
        *fp*: :class:`IOBase`
            File object
        *v*: ``True`` | {``False``}
            Verbose option
    """
    # Check type
    assert_isinstance(mesh, UmeshBase, "mesh object to write to")
    assert_isinstance(fname_or_fp, (str, IOBase), "mesh file")
    # Open file
    with openfile(fname_or_fp, 'wb') as fp:
        # Write file
        _write_plt(mesh, fp, v=v)


def _write_plt(
        mesh: UmeshBase,
        fp: IOBase,
        v: bool = False):
    # Write header
    fp.write(b"#!TDV112")
    # Universal specifier
    tofile_lb4_i(fp, np.array([1, 0]))
    # Write title
    tofile_lb4_s(fp, mesh.get_title())
    # Get variable list
    varlist = mesh.get_varlist()
    # Number of variables
    nvar = len(varlist)
    # Write number of variables
    tofile_lb4_i(fp, nvar)
    # Write each variable
    for var in varlist:
        tofile_lb4_s(fp, var)
    # Get zones by type
    surf_zones = mesh.get_surf_zones()
    vol_zones = mesh.get_vol_zones()
    # Get zone IDs
    surf_ids = mesh.get_surfzone_ids()
    vol_ids = mesh.get_volzone_ids()
    # Number of surface zones
    nsurf = len(surf_zones)
    nvol = len(vol_zones)
    nzone = nsurf + nvol
    # Keep track of zones so we don't have to create them twice
    zone_elems = {}
    zone_nodes = {}
    zone_jnode = {}
    # Write zones
    for j, zone in enumerate(surf_zones + vol_zones):
        # Status update
        if v:
            _printf(f"  Zone {j+1}/{nzone} '{zone}' metadata\r")
        # Write fixed zone type identifier
        tofile_lb4_f(fp, 299.0)
        # Write zone name
        tofile_lb4_s(fp, zone)
        # Write "ParentZone" (ignored here)
        tofile_lb4_i(fp, -1)
        # Write "StrandID" (basically zone groups for Tecplot)
        tofile_lb4_i(fp, mesh.get_strand_id(j))
        # Write "time" for this zone
        tofile_lb8_f(fp, mesh.get_time(j))
        # Write -1 for some other marker
        tofile_lb4_i(fp, -1)
        # Get surface/volume zone
        if j < nsurf:
            # Surface zone; get surface ID
            surf_id = surf_ids[j]
            # Generate a "zone"
            zone = mesh.genr8_surf_zone(surf_id)
            # Create an all-quad zone
            zone_type = FEQUADRILATERAL
            # Repeat third index of each tri
            tris = np.hstack((zone.tris, zone.tris[:, [2]]))
            # Combine tris/quads into single array
            elems = np.vstack((tris, zone.quads)) - 1
        else:
            # Volume zone; get volume ID
            vol_id = vol_ids[j - nsurf]
            # Generate a "zone"
            zone = mesh.genr8_vol_zone(vol_id)
            # Check for cells other than tets
            n1 = zone.pyrs.size + zone.pris.size + zone.hexs.size
            if n1:
                # Create an all-hex zone
                zone_type = FEBRICK
                # Repeat indices as needed to get to 8 indices
                tets = zone.tets[:, [0, 0, 1, 2, 3, 3, 3, 3]]
                pyrs = zone.pyrs[:, [0, 3, 4, 1, 2, 2, 2, 2]]
                pris = zone.pris[:, [0, 0, 1, 2, 3, 3, 4, 5]]
                elems = np.vstack((tets, pyrs, pris, zone.hexs)) - 1
            else:
                # Create an all-tet zone
                zone_type = FETETRAHEDRON
                # Use tets alone
                elems = zone.tets - 1
        # Save zone for later
        zone_elems[j] = elems
        zone_nodes[j] = zone.nodes
        zone_jnode[j] = zone.jnode
        # Write zone type
        tofile_lb4_i(fp, zone_type)
        # All variables node-centered
        tofile_lb4_i(fp, 0)
        # Two options related to "neighbors"
        tofile_lb4_i(fp, np.zeros(2))
        # Write number of points, elements in zone
        tofile_lb4_i(fp, zone.nodes.shape[0])
        tofile_lb4_i(fp, elems.shape[0])
        # Four more unused parameters
        tofile_lb4_i(fp, np.zeros(4))
    # Write end-of-header marker
    tofile_lb4_f(fp, 357.0)
    # Loop through zones again
    for j, zone in enumerate(surf_zones + vol_zones):
        # Status update
        if v:
            # Determine zone type
            if j < nsurf:
                # Surface zone
                ztype = "surface"
                # Surface zone index
                kj = j
                # Number of surfaces
                nj = nsurf
            else:
                # Volume zone
                ztype = "volume"
                # Volume zone index
                kj = j - nsurf
                # Number of volumes
                nj = nvol
            # Write message
            _printf(f"  Writing {ztype} zone {kj + 1}/{nj} '{zone}'\r")
        # Write marker
        tofile_lb4_f(fp, 299.0)
        # Write variable types (``1`` for "float")
        tofile_lb4_i(fp, np.ones(nvar, dtype="int32"))
        # Set passive variables
        tofile_lb4_i(fp, 1)
        tofile_lb4_i(fp, np.zeros(nvar, dtype="int32"))
        # This is something about sharing coordinates
        tofile_lb4_i(fp, 1)
        tofile_lb4_i(fp, np.full(nvar, -1))
        # This is the *zshare* value
        tofile_lb4_i(fp, -1)
        # Get nodes
        nodes = zone_nodes[j]
        jnode = zone_jnode[j]
        elems = zone_elems[j]
        # Subset remaining variables
        if mesh.q is None:
            # Generate empty array
            qj = np.zeros((jnode.size, 0), dtype=nodes.dtype)
        else:
            # Use actual states
            qj = mesh.q[jnode, :]
        # Combine node and other states
        xj = np.hstack((nodes, qj))
        # Get remaining *q* min/max
        qmin = np.min(xj, axis=0)
        qmax = np.max(xj, axis=0)
        # Combine them; order is qmin[0], qmax[0], qmin[1], ...
        tofile_lb8_f(fp, np.vstack((qmin, qmax)).T)
        # Save the actual data
        tofile_lb4_f(fp, xj.T)
        # Write the element information
        tofile_lb4_i(fp, elems)
    # Status update
    if v:
        print("")


def _printf(txt: str):
    # Get terminal width
    wsize = shutil.get_terminal_size().columns
    # Clear prompt
    sys.stdout.write("%*s\r" % (wsize - 1, ''))
    # Write output
    sys.stdout.write(txt)
    sys.stdout.flush()
