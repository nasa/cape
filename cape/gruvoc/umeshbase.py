
# Standard library
from abc import ABC
from collections import namedtuple
from io import IOBase
from typing import Any, Callable, Optional, Union

# Third party
import numpy as np

# Local imports
from . import volcomp
from .errors import (
    assert_size,
    GruvocKeyError)
from .geom import rotate_points, translate_points
from .surfconfig import INT_TYPES


# Class for Tecplot zones; nodes and indices
SurfZone = namedtuple("SurfZone", ("nodes", "jnode", "tris", "quads"))
VolumeZone = namedtuple(
    "VolumeZone", ("nodes", "jnode", "tets", "pyrs", "pris", "hexs"))

# Other defaults
DEFAULT_STRAND_ID = np.int32(1000)
DEFAULT_TIME = np.float64(0.0)
DEFAULT_TITLE = "untitled"
NROWS = 25

# Types
NUMERIC_TYPES = (int, float, np.int32, np.int64, np.float32, np.float64)
ARRAY_TYPES = (list, tuple, np.ndarray)

# Suffixes
SI_SUFFIXES = {
    0: "",
    1: "k",
    2: "M",
    3: "G",
    4: "T",
}

# Cell edge connectivity
EDGECON = {
    "tris": np.array((
        (0, 1), (1, 2), (2, 0))
    ),
    "quads": np.array((
        (0, 1), (1, 2), (2, 3), (3, 0))
    ),
    "tets": np.array((
        (0, 1), (1, 2), (0, 2),
        (0, 3), (1, 3), (2, 3))
    ),
    "pyrs": np.array((
        (0, 1), (1, 2), (2, 3), (0, 3),
        (0, 4), (1, 4), (2, 4), (3, 4))
    ),
    "pris": np.array((
        (0, 1), (0, 2), (1, 2),
        (0, 3), (1, 4), (2, 5),
        (3, 4), (3, 5), (4, 5))
    ),
    "hexs": np.array((
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5),
        (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),
        (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (6, 11))
    )
}


# Base class
class UmeshBase(ABC):
    # Instance attributes
    __slots__ = (
        "nodes",
        "tris",
        "quads",
        "tets",
        "pyrs",
        "pris",
        "hexs",
        "niter",
        "nnode",
        "ntri",
        "nquad",
        "ntet",
        "npyr",
        "npri",
        "nhex",
        "nface",
        "basename",
        "blds",
        "bldel",
        "comment1",
        "comment2",
        "comment3",
        "comment4",
        "config",
        "coord_system",
        "edges",
        "edge_ids",
        "farfield_type",
        "fdir",
        "fname",
        "hex_ids",
        "mesh_type",
        "name",
        "ndim",
        "ntet_bl",
        "nzone",
        "parentzone",
        "path",
        "pri_ids",
        "pyr_ids",
        "strand_ids",
        "surfzones",
        "surfzone_ids",
        "t",
        "target_config",
        "tet_ids",
        "title",
        "tri_bcs",
        "tri_flags",
        "tri_ids",
        "nq",
        "nq_scalar",
        "nq_vector",
        "nq_matrix",
        "nq_metric",
        "q",
        "q_scalar",
        "q_vector",
        "q_matrix",
        "q_metric",
        "q_type",
        "qvars",
        "qvars_scalar",
        "qvars_vector",
        "qvars_matrix",
        "qinf",
        "qinfvars",
        "quad_bcs",
        "quad_flags",
        "quad_ids",
        "units",
        "volzones",
        "volzone_ids",
        "zone_types",
        "zones",
        "_bbox_cache",
    )

  # === __dunder__ ===
    # String
    def __str__(self) -> str:
        # Initialize string
        txt = f"<{self.__class__.__name__}"
        # Number of attributes printed
        n = 0
        # Check for file name
        if self.fname:
            # Add it
            txt += f" '{self.fname}'"
            # Counter
            n += 1
        # Loop through key attributes
        for attr in ("nnode", "ntri", "nquad", "ntet", "npyr", "npri", "nhex"):
            # Get value
            v = getattr(self, attr)
            # Skip if none
            if not v:
                continue
            # Convert to text
            vtxt = self._str_repr(v)
            # Append counter
            n += 1
            # Separator
            sep = ", " if n else " "
            # Add value
            txt += f"{sep}{attr}={vtxt}"
        # Finalize string
        return txt + ">"

    def __repr__(self) -> str:
        return self.__str__()

    def _str_repr(self, v: Any) -> str:
        # Check for string
        if isinstance(v, (str, bytes)):
            return repr(v)
        elif isinstance(v, NUMERIC_TYPES):
            return pprint_n(v)
        else:
            return str(v)

  # === I/O ===
   # --- Read ---
    def _read_to_slot(
            self,
            fp: IOBase,
            slot_or_slots: Union[str, tuple],
            fn: Callable,
            shape: tuple,
            desc: Optional[str] = None,
            breaks: Optional[tuple] = None):
        # Get total read count
        n = np.prod(shape)
        # Read data
        x = fn(fp, n)
        # Check size
        assert_size(x, n, desc)
        # Reshape data
        y = x.reshape(shape)
        # Check for single slot or multiple
        if isinstance(slot_or_slots, str):
            # Save entire array
            setattr(self, slot_or_slots, y)
        else:
            # Get cumulative ranges
            cuts = np.cumsum(breaks)
            # Split ranges into start and end indices
            ia = np.hstack((0, cuts))
            ib = np.hstack((cuts, n))
            # Save each range
            for slotj, iaj, ibj in zip(slot_or_slots, ia, ib):
                setattr(self, slotj, y[iaj:ibj])

    def _readn_to_slot(
            self,
            fp: IOBase,
            slot: str,
            fn: Callable,
            shape: tuple,
            N: int,
            desc: Optional[str] = None):
        y = None
        for i in range(N):
            # Get total read count
            n = np.prod(shape)
            # Read data
            x = fn(fp, n)
            # Check size
            assert_size(x, n, desc)
            # # Reshape data for stacking
            _y = x.reshape(len(x), 1)
            if y is None:
                y = _y
            else:
                y = np.hstack([y, _y])
        # Save entire array
        setattr(self, slot, y)

   # --- Write ---
    def _combine_slots(
            self,
            slot2d: str,
            slots1d: tuple,
            rc: dict) -> np.ndarray:
        # Get initial value; transposed for appending rows i/o cols
        a = getattr(self, slot2d).T
        # Dimensions
        _, m = a.shape
        # Loop through slots
        for slot1d in slots1d:
            # Get value
            bj = getattr(self, slot1d)
            # Default values
            if bj is None:
                # Get default
                vj = rc.get(slot1d, 0)
                # Initialize w/ fixed value
                bj = np.full(m, vj, dtype=a.dtype)
            # Append
            a = np.vstack((a, bj))
        # Output
        return a.T

    def _write_from_slot(
            self,
            fp: IOBase,
            slot_or_slots: Union[str, tuple],
            fn: Callable) -> bool:
        # Check for single slot or multiple
        if isinstance(slot_or_slots, str):
            # Get single attribute
            x = getattr(self, slot_or_slots)
            # Exit if not found
            if x is None:
                return False
        else:
            # Loop through slots
            for j, slot in enumerate(slot_or_slots):
                # Get value
                xj = getattr(self, slot)
                # Check if empty
                if xj is None:
                    return False
                # Initialize or stack
                if j == 0:
                    # Initialize combined array
                    x = xj
                else:
                    # Stack
                    x = np.hstack((x, xj))
        # Write combined array
        fn(fp, x)
        # Successful operation
        return True

    def _writen_from_slot(
            self,
            fp: IOBase,
            slot: str,
            fn: Callable,
            N: int) -> bool:
        # Get single attribute
        x = getattr(self, slot)
        # Exit if not found
        if x is None:
            return False
        for j in range(N):
            # Write each array
            fn(fp, x[:, j])
        # Successful operation
        return True

  # === Data ===
   # --- Basic info ---
    def get_summary(self, h=False) -> str:
        r"""Create summary of mesh

        :Call:
            >>> txt = mesh.get_summary(h=False)
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *h*: ``True`` | {``False``}
                Human-readable, e.g. "12345678" -> "12.3M"
        :Outputs:
            *txt*: :class:`str`
                Summary of *mesh* properties and counts
        """
        # Initialize string
        txt = f"<{self.__class__.__name__}"
        # Check for file name
        if self.fname:
            # Add it
            txt += f" '{self.fname}'"
        # Total element counts
        nsurf = 0
        nvol = 0
        # Loop through key attributes
        for attr in ("nnode", "ntri", "nquad", "ntet", "npyr", "npri", "nhex"):
            # Get value
            v = getattr(self, attr)
            # Skip if none
            if not v:
                continue
            # Add to totalts
            if attr in ("ntri", "nquad"):
                nsurf += v
            elif attr in ("ntet", "npyr", "npri", "nhes"):
                nvol += v
            # Convert to text, human-readable or full
            vtxt = self._str_repr(v) if h else str(v)
            # Add value
            txt += f"\n  {attr}={vtxt}"
        # Print totals
        if nsurf:
            vtxt = self._str_repr(nsurf) if h else str(nsurf)
            txt += f"\n  TOTAL surf elems: {vtxt}"
        if nvol:
            vtxt = self._str_repr(nvol) if h else str(nvol)
            txt += f"\n  TOTAL volume elems: {vtxt}"
        # Finalize string
        return txt + "\n>"

    def print_summary(self, h=False):
        r"""Print summary of mesh

        :Call:
            >>> txt = mesh.get_summary(h=False)
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *h*: ``True`` | {``False``}
                Human-readable, e.g. "12345678" -> "12.3M"
        """
        # Get summary
        txt = self.get_summary(h=h)
        # Print it
        print(txt)

    def get_title(self) -> str:
        r"""Get "title" of mesh object

        Tries *mesh.title*, then *mesh.fname*

        :Call:
            >>> title = mesh.get_title()
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
        :Outputs:
            *title*: :class:`str`
                Title of *mesh*
        """
        # Get title
        if self.title is not None:
            return self.title
        # Use file name
        if self.fname is not None:
            return self.fname
        # Default title
        return DEFAULT_TITLE

    def get_varlist(self) -> list:
        r"""Get list of variables in the mesh object

        :Call:
            >>> varlist = mesh.get_varlist()
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
        :Outputs:
            *varlist*: :class:`list`\ [:class:`str`]
                List of names of variables, default ``["x", "y", "z"]``
        """
        # Add nodal coordinates to *q* var list
        return ["x", "y", "z"] + self.get_qvars()

    def get_qvars(self) -> list:
        r"""Get list of state variables in the mesh object

        :Call:
            >>> varlist = mesh.get_qvars()
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
        :Outputs:
            *varlist*: :class:`list`\ [:class:`str`]
                List of names of variables, default ``["x", "y", "z"]``
        """
        # Get var list
        varlist = self.qvars
        # Default to "x", "y", "z"
        varlist = [] if varlist is None else varlist
        # Output
        return varlist

    def get_nface(self) -> int:
        r"""Get number of unique surface IDs

        :Call:
            >>> nface = mesh.get_nface()
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
        :Outputs:
            *nface*: :class:`int`
                Number of unique entries in *tri_ids* and *quad_ids*
        """
        # Get value
        nface = self.nface
        # Check if integer
        if nface is not None:
            return nface
        # Get list of faces
        face_ids = self.get_surfzone_ids()
        # Count them
        return face_ids.size

   # --- Calculations (other states) ---
    def add_mach(self, gam: float = 1.4):
        r"""Add Mach number as *mach* to state matrix

        :Call:
            >>> mesh.add_mach(gam=1.4)
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *gam*: {``1.4``} | :class:`float`
                Ratio of specific heats
        """
        # Unpack variable list
        qvars = self.qvars
        # Check if already present
        if "mach" in qvars:
            return
        # Check key vars
        for name in ("rho", "u", "v", "w", "p"):
            if name not in qvars:
                raise GruvocKeyError(f"Missing qvar '{name}'")
        # Get velocity times density
        rho = self.q[:, qvars.index('rho')]
        u = self.q[:, qvars.index('u')]
        v = self.q[:, qvars.index('v')]
        w = self.q[:, qvars.index('w')]
        p = self.q[:, qvars.index('p')]
        # Calculate local sound speed (squared)
        a2 = gam * p / rho
        # Calculate mach number
        mach = np.sqrt((u*u + v*v + w*w) / a2)
        # Save it
        self.qvars.append("mach")
        self.q = np.hstack((self.q, np.array([mach]).T))
        # Increment nq for new var
        self.nq += 1

    def add_cp(self, gam: float = 1.4):
        r"""Add pressure coefficient *cp* to state matrix

        :Call:
            >>> mesh.add_cp(gam=1.4)
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *gam*: {``1.4``} | :class:`float`
                Ratio of specific heats
        """
        # Unpack variable list
        qvars = self.qvars
        qinfvars = self.qinfvars
        # Check if already present
        if "cp" in qvars:
            return
        # Check key vars
        for name in ("p",):
            if name not in qvars:
                raise GruvocKeyError(f"Missing qvar '{name}'")
        for name in ("mach",):
            if name not in qinfvars:
                raise GruvocKeyError(f"Missing freestream qinfvar '{name}'")
        # Get values
        p = self.q[:, qvars.index('p')]
        # Get freestream
        minf = self.qinf[qinfvars.index("mach")]
        # Calculate pressure coefficient
        cp = (p - 1/gam)*2/(minf*minf)
        # Save it
        self.qvars.append("cp")
        self.q = np.hstack((self.q, np.array([cp]).T))
        # Increment nq for new var
        self.nq += 1

   # --- Geometry (manipulation) ---
    def rotate(
            self,
            v1: np.ndarray,
            v2: np.ndarray,
            theta: float,
            comp: Optional[str] = None):
        r"""Rotate subset of points about a vector defined by two points

        :Call:
            >>> Y = mesh.rotate(v1, v2, theta, comp=None)
        :Inputs:
            *X*: :class:`np.ndarray`\ [:class:`float`]
                Coordinates of *N* points to rotate; shape is (*N*, 3)
            *v1*: :class:`np.ndarray`\ [:class:`float`]
                Coordinates of start point of rotation vector; shape (3,)
            *v2*: :class:`np.ndarray`\ [:class:`float`]
                Coordinates of end point of rotation vector; shape (3,)
            *theta*: :class:`float`
                Angle by which to rotate *X*, in degrees
            *comp*: {``None``} | :class:`str` | :class:`int`
                Optional component name/ID (else rotate all points)
        """
        # Get node IDs
        mask = self.get_nodes_by_id(comp) - 1
        # Get said nodes
        nodes = self.nodes[mask, :]
        # Apply rotation
        Y = rotate_points(nodes, v1, v2, theta)
        # Save rotated points
        self.nodes[mask, :] = Y

    def translate(
            self,
            v: np.ndarray,
            comp: Optional[str] = None):
        r"""Translate subset of points

        :Call:
            >>> mesh.translate(v, comp=None)
        :Inputs:
            *X*: :class:`np.ndarray`\ [:class:`float`]
                Coordinates of *N* points to translate; shape is (*N*, 3)
            *v*: :class:`np.ndarray`\ [:class:`float`]
                Vector by which to translate each point; shape (3,)
        :Outputs:
            *Y*: :class:`np.ndarray`\ [:class:`float`]
                Translated points
        """
        # Get node IDs
        mask = self.get_nodes_by_id(comp) - 1
        # Extract coordinates
        nodes = self.nodes[mask, :]
        # Apply translation
        Y = translate_points(nodes, v)
        # Save transformed points
        self.nodes[mask, :] = Y

   # --- Config ---
    def get_surf_ids(self, comp=None) -> np.ndarray:
        r"""Expand integer, str, or list to list of surface ID ints

        :Call:
            >>> surf_ids = mesh.get_surf_ids(comp=None)
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *comp*: {``None``} | :class:`str` | :class:`int`
                Name or index of a component, or list thereof
        """
        # Check for null input
        if comp is None:
            # Return all IDs
            return self.get_surfzone_ids()
        # Initialize output
        surf_ids = []
        # Ensure array
        if isinstance(comp, ARRAY_TYPES):
            # Already array
            comps = comp
        else:
            # Convert
            comps = [comp]
        # Loop through components
        for compj in comps:
            # Check type
            if isinstance(compj, INT_TYPES):
                # Add to list
                surf_ids.append(compj)
            elif isinstance(compj, str):
                # Get IDs in list
                if self.config is None:
                    continue
                # Process
                surf_ids.extend(self.config.get_surf_ids(compj))
        # Output
        return np.unique(np.array(surf_ids, dtype="int32"))

   # --- Zones ---
    def get_zones(self) -> list:
        r"""Get list of zones

        :Call:
            >>> zones = mesh.get_zones()
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
        :Outputs:
            *zones*: :class:`list`\ [:class:`str`]
                List of zone names, defaults to ``"boundary 1"``,
                ``"boundary 2"``, ..., [``"volume 1"``]
        """
        # Combine two sets of zones
        srfzones = self.get_surf_zones()
        volzones = self.get_vol_zones()
        # Output
        return srfzones + volzones

    def get_surf_zones(self) -> list:
        r"""Get list of zones

        :Call:
            >>> zones = mesh.get_zones()
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
        :Outputs:
            *zones*: :class:`list`\ [:class:`str`]
                List of zone names, defaults to ``"boundary 1"``,
                ``"boundary 2"``, ..., [``"tets"``, ``"pyramids"``,
                ``"prisms"``, ``"hexs"``]
        """
        # Get zones
        zones = self.surfzones
        # Check if empty
        if zones is not None:
            return zones
        # Get zone IDs
        zone_ids = self.get_surfzone_ids()
        # Generate list
        zones = [self.config.get_name(k) for k in zone_ids]
        # Output
        return zones

    def get_vol_zones(self) -> list:
        r"""Get list of volume zone names

        :Call:
            >>> zones = mesh.get_volzones()
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
        :Outputs:
            *zones*: :class:`list`\ [:class:`str`]
                List of volume zone names, defaults to ``"volume 1"``,
                [``"volume 2"``, [...]]
        """
        # Get zones
        zones = self.volzones
        # Check if empty
        if zones is not None:
            return zones
        # Get zone IDs
        zone_ids = self.get_volzone_ids()
        # Generate list
        zones = [f"volume {k}" for k in zone_ids]
        # Output
        return zones

    def get_surfzone_ids(self) -> np.ndarray:
        r"""Get list of ID numbers for surface zones

        :Call:
            >>> surf_ids = mesh.get_surfzone_ids()
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
        :Outputs:
            *surf_ids*: :class:`np.ndarray`\ [:class:`int`]
                List of unique surface zone IDs
        """
        # Get zone IDs if able
        zone_ids = self.surfzone_ids
        # Return if defined
        if zone_ids is not None:
            return np.asarray(zone_ids)
        # Get ids for both tris and quads
        tria_ids = np.unique(self.get_tri_ids())
        quad_ids = np.unique(self.get_quad_ids())
        # Combine
        surf_ids = np.hstack((tria_ids, quad_ids))
        # Return unique values
        return np.unique(surf_ids)

    def get_volzone_ids(self) -> np.ndarray:
        r"""Get list of ID numbers for each volume cell

        :Call:
            >>> vol_ids = mesh.get_volzone_ids()
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
        :Outputs:
            *zone_ids*: :class:`np.ndarray`\ [:class:`int`]
                List of unique surface zone IDs
        """
        # Get zone IDs if able
        zone_ids = self.volzone_ids
        # Return if defined
        if zone_ids is not None:
            return np.asarray(zone_ids)
        # Get ids for both tris and quads
        tet_ids = self.get_tet_ids()
        pyr_ids = self.get_pyr_ids()
        pri_ids = self.get_pri_ids()
        hex_ids = self.get_hex_ids()
        # Combine
        vol_ids = np.hstack((tet_ids, pyr_ids, pri_ids, hex_ids))
        # Return unique values
        return np.unique(vol_ids)

    def get_tri_ids(self) -> np.ndarray:
        r"""Get ID numbers for each tri face

        :Call:
            >>> tri_ids = mesh.get_tri_ids()
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
        :Outputs:
            *tri_ids*: :class:`np.ndarray`\ [:class:`int`]
                ID numbers for each tri face
        """
        # Get tri IDs if able
        tri_ids = self.tri_ids
        # Default
        if tri_ids is None:
            tri_ids = np.ones(self.ntri, dtype="int32")
        # Output
        return tri_ids

    def get_quad_ids(self) -> np.ndarray:
        r"""Get ID numbers for each quad face

        :Call:
            >>> quad_ids = mesh.get_quad_ids()
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
        :Outputs:
            *quad_ids*: :class:`np.ndarray`\ [:class:`int`]
                ID numbers for each quad face
        """
        # Get quad IDs if able
        elem_ids = self.quad_ids
        # Default
        if elem_ids is None:
            # Number of faces
            nelem = self.nquad
            nelem = 0 if nelem is None else nelem
            # Set all IDs to one
            elem_ids = np.ones(nelem, dtype="int32")
        # Output
        return elem_ids

    def get_tet_ids(self) -> np.ndarray:
        r"""Get ID numbers for each tetrahedral cell

        :Call:
            >>> tet_ids = mesh.get_tet_ids()
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
        :Outputs:
            *tet_ids*: :class:`np.ndarray`\ [:class:`int`]
                ID numbers for each tet cell
        """
        # Get IDs if able
        ids = self.tet_ids
        # Number of elementds
        n = self.ntet
        n = 0 if n is None else n
        # Default
        if ids is None:
            ids = np.ones(n, dtype="int32")
        # Output
        return ids

    def get_pyr_ids(self) -> np.ndarray:
        r"""Get ID numbers for each pyramid (penta_5) cell

        :Call:
            >>> tet_ids = mesh.get_pyr_ids()
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
        :Outputs:
            *pyr_ids*: :class:`np.ndarray`\ [:class:`int`]
                ID numbers for each pyr cell
        """
        # Get IDs if able
        ids = self.pyr_ids
        # Number of elementds
        n = self.npyr
        n = 0 if n is None else n
        # Default
        if ids is None:
            ids = np.ones(n, dtype="int32")
        # Output
        return ids

    def get_pri_ids(self) -> np.ndarray:
        r"""Get ID numbers for each triangular-prism cell

        :Call:
            >>> pri_ids = mesh.get_pri_ids()
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
        :Outputs:
            *pri_ids*: :class:`np.ndarray`\ [:class:`int`]
                ID numbers for each pri cell
        """
        # Get IDs if able
        ids = self.pri_ids
        # Number of elementds
        n = self.npri
        n = 0 if n is None else n
        # Default
        if ids is None:
            ids = np.ones(n, dtype="int32")
        # Output
        return ids

    def get_hex_ids(self) -> np.ndarray:
        r"""Get ID numbers for each hexahedral cell

        :Call:
            >>> gex_ids = mesh.get_hex_ids()
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
        :Outputs:
            *gex_ids*: :class:`np.ndarray`\ [:class:`int`]
                ID numbers for each hex cell
        """
        # Get IDs if able
        ids = self.hex_ids
        # Number of elementds
        n = self.nhex
        n = 0 if n is None else n
        # Default
        if ids is None:
            ids = np.ones(n, dtype="int32")
        # Output
        return ids

   # --- Bounding Box ---
    # Function to add a bounding box based on a component and buffer
    def get_bbox(self, comp=None, **kwargs):
        r"""Find bounding box of specified component

        Find a bounding box based on the coordinates of a specified
        component or list of components, with an optional buffer or
        buffers in each direction

        :Call:
            >>> xlim = mesh.GetCompBBox(compID, **kwargs)
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *compID*: {``None``} | :class:`int` | :class:`str`
                Component or list of components to use for bounding box;
                if ``None``; return bounding box for entire surface
            *pad*: :class:`float`
                Buffer to add in each dimension to min and max coords
            *xpad*: :class:`float`
                Buffer to minimum and maximum *x*-coordinates
            *ypad*: :class:`float`
                Buffer to minimum and maximum *y*-coordinates
            *zpad*: :class:`float`
                Buffer to minimum and maximum *z*-coordinates
            *xp*: :class:`float`
                Buffer for the maximum *x*-coordinate
            *xm*: :class:`float`
                Buffer for the minimum *x*-coordinate
            *yp*: :class:`float`
                Buffer for the maximum *y*-coordinate
            *ym*: :class:`float`
                Buffer for the minimum *y*-coordinate
            *zp*: :class:`float`
                Buffer for the maximum *z*-coordinate
            *zm*: :class:`float`
                Buffer for the minimum *z*-coordinate
        :Outputs:
            *xlim*: :class:`numpy.ndarray` (:class:`float`), shape=(6,)
                List of *xmin*, *xmax*, *ymin*, *ymax*, *zmin*, *zmax*
        :Versions:
            * 2014-06-16 ``@ddalle``: v1.0
            * 2014-08-03 ``@ddalle``: v1.1; "buff" --> "pad"
            * 2017-02-08 ``@ddalle``: v1.2; CompID=None behavior
        """
        # Don't cache if kwargs provided, or if given list
        if len(kwargs) > 0 or isinstance(comp, ARRAY_TYPES):
            return self.get_bbox_uncached(comp, **kwargs)
        # Initialize cache
        if self._bbox_cache is None:
            self._bbox_cache = {}
        # Retrieve cached bounding box
        cached_val = self._bbox_cache.get(comp)
        # Check if found
        if cached_val is None:
            # Get new bounding box
            calculated_val = self.get_bbox_uncached(comp)
            # Cache that value
            self._bbox_cache[comp] = calculated_val
            # Return calculated value
            return calculated_val
        else:
            # Use cache
            return cached_val

    def get_bbox_uncached(self, comp=None, **kwargs):
        # Get the overall buffer.
        pad = kwargs.get('pad', 0.0)
        # Get the other buffers.
        xpad = kwargs.get('xpad', pad)
        ypad = kwargs.get('ypad', pad)
        zpad = kwargs.get('zpad', pad)
        # Get the directional buffers.
        xp = kwargs.get('xp', xpad)
        xm = kwargs.get('xm', xpad)
        yp = kwargs.get('yp', ypad)
        ym = kwargs.get('ym', ypad)
        zp = kwargs.get('zp', zpad)
        zm = kwargs.get('zm', zpad)
        # Special case: None -> every surf
        if comp is None:
            # Just use all nodes
            xmin = np.nanmin(self.nodes[:, 0])
            ymin = np.nanmin(self.nodes[:, 1])
            zmin = np.nanmin(self.nodes[:, 2])
            xmax = np.nanmax(self.nodes[:, 0])
            ymax = np.nanmax(self.nodes[:, 1])
            zmax = np.nanmax(self.nodes[:, 2])
        else:
            # Get list of surf IDs from input
            surf_ids = self.get_surf_ids(comp)
            # Initialize lists of tris and quads
            tris = np.zeros(0, dtype="int")
            quads = np.zeros(0, dtype="int")
            # Loop through zones
            for surf_id in surf_ids:
                # Get tris and quads in that face
                trisj = self.get_tris_by_id(surf_id)
                quadsj = self.get_quads_by_id(surf_id)
                # Append
                tris = np.hstack((tris, trisj))
                quads = np.hstack((quads, quadsj))
            # Counts for each type
            ntri = tris.size
            nquad = quads.size
            # Check for null component
            if ntri + nquad == 0:
                return
            # Get the coordinates of each vertex of included tris
            xt = self.nodes[self.tris[tris, :] - 1, 0]
            yt = self.nodes[self.tris[tris, :] - 1, 1]
            zt = self.nodes[self.tris[tris, :] - 1, 2]
            # Get the coordinates of each vertex of included quadstris
            xq = self.nodes[self.quads[quads, :] - 1, 0]
            yq = self.nodes[self.quads[quads, :] - 1, 1]
            zq = self.nodes[self.quads[quads, :] - 1, 2]
            # Calculate individual extrema
            xmint = np.nan if ntri == 0 else np.min(xt)
            ymint = np.nan if ntri == 0 else np.min(yt)
            zmint = np.nan if ntri == 0 else np.min(zt)
            xmaxt = np.nan if ntri == 0 else np.max(xt)
            ymaxt = np.nan if ntri == 0 else np.max(yt)
            zmaxt = np.nan if ntri == 0 else np.max(zt)
            xminq = np.nan if nquad == 0 else np.min(xq)
            yminq = np.nan if nquad == 0 else np.min(yq)
            zminq = np.nan if nquad == 0 else np.min(zq)
            xmaxq = np.nan if nquad == 0 else np.max(xq)
            ymaxq = np.nan if nquad == 0 else np.max(yq)
            zmaxq = np.nan if nquad == 0 else np.max(zq)
            # Get the extrema
            xmin = np.nanmin([xmint, xminq])
            ymin = np.nanmin([ymint, yminq])
            zmin = np.nanmin([zmint, zminq])
            xmax = np.nanmax([xmaxt, xmaxq])
            ymax = np.nanmax([ymaxt, ymaxq])
            zmax = np.nanmax([zmaxt, zmaxq])
        # Apply padding
        xmin -= xm
        ymin -= ym
        zmin -= zm
        xmax += xp
        ymax += yp
        zmax += zp
        # Return the 6 elements
        return np.array([xmin, xmax, ymin, ymax, zmin, zmax])

   # --- Other Tecplot ---
    def get_strand_id(self, j: int) -> int:
        # Get attribute if able
        strand_ids = self.strand_ids
        # Get value
        if strand_ids is None:
            # No "StrandID" attribute; use fixed value
            return DEFAULT_STRAND_ID + j
        else:
            # Use specified value
            return strand_ids[j]

    def get_time(self, j: int) -> np.float64:
        # Get attribute if able
        t = self.t
        # Get value
        if t is None:
            # No "time" values set
            return DEFAULT_TIME
        else:
            # Use specified value
            return t[j]

   # --- Surface subsets ---
    def get_nodes_by_id(self, surf_id: int) -> np.ndarray:
        r"""Get indices of all nodes contained on a given surface ID

        :Call:
            >>> mask = mesh.get_nodes_by_id(surf_id)
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *surf_id*: :class:`int`
                Surface ID number
        :Outputs:
            *mask*: :class:`np.ndarray`\ [:class:`int`]
                Indices of nodes in at least one tri/quad w/ matching ID
        """
        # Get tris and quads
        ktria = self.get_tris_by_id(surf_id)
        kquad = self.get_quads_by_id(surf_id)
        # Get the node indices of those quads and tris
        tris = self.tris[ktria]
        quads = self.quads[kquad]
        # Get overall non-repeating list of indices
        jnodes = np.unique(np.hstack((tris.flatten(), quads.flatten()))) - 1
        # Output
        return jnodes

    def get_nodes_by_vol_id(self, vol_id: int) -> np.ndarray:
        r"""Get indices of all nodes contained on a given volume ID

        :Call:
            >>> mask = mesh.get_nodes_by_vol_id(vol_id)
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *vol_id*: :class:`int`
                Volume ID number
        :Outputs:
            *mask*: :class:`np.ndarray`\ [:class:`int`]
                Indices of nodes part of >0 matching volume elements
        """
        # Get element node indices
        k1 = self.get_tets_by_id(vol_id)
        k2 = self.get_pyrs_by_id(vol_id)
        k3 = self.get_pris_by_id(vol_id)
        k4 = self.get_hexs_by_id(vol_id)
        # Get the unique node indices in each element type
        j1 = np.unique(self.tets[k1])
        j2 = np.unique(self.pyrs[k2])
        j3 = np.unique(self.pris[k3])
        j4 = np.unique(self.hexs[k4])
        # Get overall non-repeating list of indices
        jnodes = np.unique(np.hstack((j1, j2, j3, j4)))
        # Output
        return jnodes

    def get_tris_by_id(self, surf_id: int) -> np.ndarray:
        r"""Get indices (0-based) of tris with specified ID

        :Call:
            >>> mask = mesh.get_tris_by_id(surf_id)
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *surf_id*: :class:`int`
                Surface ID number
        :Outputs:
            *mask*: :class:`np.ndarray`\ [:class:`int`]
                Indices of tris whose ID matches *surf_id*
        """
        # Get tri ID list
        ids = self.get_tri_ids()
        # Return those which have the correct compID
        return np.where(ids == surf_id)[0]

    def get_quads_by_id(self, surf_id: int) -> np.ndarray:
        r"""Get indices (0-based) of quads with specified ID

        :Call:
            >>> mask = mesh.get_quads_by_id(surf_id)
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *surf_id*: :class:`int`
                Surface ID number
        :Outputs:
            *mask*: :class:`np.ndarray`\ [:class:`int`]
                Indices of quads whose ID matches *surf_id*
        """
        # Get face ID list
        ids = self.get_quad_ids()
        # Return those which have the correct compID
        return np.where(ids == surf_id)[0]

    def get_tets_by_id(self, vol_id: int) -> np.ndarray:
        r"""Get indices (0-based) of tetrahedra matching volume ID

        :Call:
            >>> mask = mesh.get_tets_by_id(vol_id)
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *vol_id*: :class:`int`
                Volume ID number
        :Outputs:
            *mask*: :class:`np.ndarray`\ [:class:`int`]
                Indices of elements whose ID matches *vol_id*
        """
        # Get ID list
        ids = self.get_tet_ids()
        # Return those which have the correct compID
        return np.where(ids == vol_id)[0]

    def get_pyrs_by_id(self, vol_id: int) -> np.ndarray:
        r"""Get indices (0-based) of pyramids matching volume ID

        :Call:
            >>> mask = mesh.get_pyrs_by_id(vol_id)
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *vol_id*: :class:`int`
                Volume ID number
        :Outputs:
            *mask*: :class:`np.ndarray`\ [:class:`int`]
                Indices of elements whose ID matches *vol_id*
        """
        # Get ID list
        ids = self.get_pyr_ids()
        # Return those which have the correct compID
        return np.where(ids == vol_id)[0]

    def get_pris_by_id(self, vol_id: int) -> np.ndarray:
        r"""Get indices (0-based) of prism cells matching volume ID

        :Call:
            >>> mask = mesh.get_pris_by_id(vol_id)
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *vol_id*: :class:`int`
                Volume ID number
        :Outputs:
            *mask*: :class:`np.ndarray`\ [:class:`int`]
                Indices of elements whose ID matches *vol_id*
        """
        # Get ID list
        ids = self.get_pri_ids()
        # Return those which have the correct compID
        return np.where(ids == vol_id)[0]

    def get_hexs_by_id(self, vol_id: int) -> np.ndarray:
        r"""Get indices (0-based) of hexahedra matching volume ID

        :Call:
            >>> mask = mesh.get_hexs_by_id(vol_id)
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *vol_id*: :class:`int`
                Volume ID number
        :Outputs:
            *mask*: :class:`np.ndarray`\ [:class:`int`]
                Indices of elements whose ID matches *vol_id*
        """
        # Get tri ID list
        ids = self.get_hex_ids()
        # Return those which have the correct compID
        return np.where(ids == vol_id)[0]

   # --- Subzone ---
    def genr8_surf_zone(self, surf_id: int) -> SurfZone:
        r"""Get nodes, tris, and quads for a given surface ID

        This compresses the node indices to only those that are nodes
        of tris or quads that have the correct ID.

        :Call:
            >>> zone = mesh.genr8_surf_zone(surf_id)
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *surf_id*: :class:`int`
                Surface ID number
        :Outputs:
            *zone*: :class:`SurfZone`
                Surface zone information
            *zone.nodes*: :class:`np.ndarray`\ [:class:`float`]
                Coordinates of nodes of relevant tris/quads
            *zone.jnode*: :class:`np.ndarray`\ [:class:`int`]
                Indices of nodes included (0-based)
            *zone.tris*: :class:`np.ndarray`\ [:class:`int`]
                Compressed node indices of matching tris (1-based)
            *zone.quads*: :class:`np.ndarray`\ [:class:`int`]
                Compressed node indices of matching quads (1-based)
        """
        # Get tris and quads
        ktris = self.get_tris_by_id(surf_id)
        kquads = self.get_quads_by_id(surf_id)
        # Get the node indices of those quads and tris
        tris = self.tris[ktris]
        quads = self.quads[kquads]
        # Get overall non-repeating list of indices
        j = np.unique(np.hstack((tris.flatten(), quads.flatten())))
        # Compress node indices
        tris = compress_indices(tris, j)
        quads = compress_indices(quads, j)
        # Select the nodes
        nodes = self.nodes[j - 1]
        # Output
        return SurfZone(nodes, j - 1, tris, quads)

    def genr8_vol_zone(self, vol_id: int) -> VolumeZone:
        r"""Get nodes, tets, pyrs, pris, and hexs for a given volume ID

        This compresses the node indices to only those that are nodes
        of tris or quads that have the correct ID. In most cases, there
        is only a single volume, so it returns the entire volume.

        :Call:
            >>> zone = mesh.genr8_vol_zone(vol_id)
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *vol_id*: :class:`int`
                volume ID number
        :Outputs:
            *zone*: :class:`SurfZone`
                Surface zone information
            *zone.nodes*: :class:`np.ndarray`\ [:class:`float`]
                Coordinates of nodes of requested volume
            *zone.jnode*: :class:`np.ndarray`\ [:class:`int`]
                Indices of nodes included (0-based)
            *zone.tets*: :class:`np.ndarray`\ [:class:`int`]
                Compressed node indices of matching tets (1-based)
            *zone.pyrs*: :class:`np.ndarray`\ [:class:`int`]
                Compressed node indices of matching pyramids (1-based)
            *zone.pris*: :class:`np.ndarray`\ [:class:`int`]
                Compressed node indices of matching tri prisms (1-based)
            *zone.hexs*: :class:`np.ndarray`\ [:class:`int`]
                Compressed node indices of matching hexs (1-based)
        """
        # Get element node indices
        k1 = self.get_tets_by_id(vol_id)
        k2 = self.get_pyrs_by_id(vol_id)
        k3 = self.get_pris_by_id(vol_id)
        k4 = self.get_hexs_by_id(vol_id)
        # Get the node indices in each element type
        tets = self.tets[k1]
        pyrs = self.pyrs[k2]
        pris = self.pris[k3]
        hexs = self.hexs[k4]
        # Unique node indices
        j1 = np.unique(tets)
        j2 = np.unique(pyrs)
        j3 = np.unique(pris)
        j4 = np.unique(hexs)
        # Get overall non-repeating list of indices
        j = np.unique(np.hstack((j1, j2, j3, j4)))
        # Compress node indices
        tets = compress_indices(tets, j)
        pyrs = compress_indices(pyrs, j)
        pris = compress_indices(pris, j)
        hexs = compress_indices(hexs, j)
        # Select the nodes
        nodes = self.nodes[j - 1]
        # Output
        return VolumeZone(nodes, j - 1, tets, pyrs, pris, hexs)

  # === Analysis ===
   # --- Volume removal ---
    def remove_volume(self):
        r"""Remove all volume cells and nodes not used on surface

        :Call:
            >>> mesh.remove_volume()
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
        """
        # Remove cells
        self.tets = np.zeros((0, 4), dtype="int32")
        self.pyrs = np.zeros((0, 5), dtype="int32")
        self.pris = np.zeros((0, 6), dtype="int32")
        self.hexs = np.zeros((0, 8), dtype="int32")
        # Set counts
        self.ntet = 0
        self.npyr = 0
        self.npri = 0
        self.nhex = 0
        # Get the node indices of those quads and tris
        tris = self.tris
        quads = self.quads
        # Get overall non-repeating list of node indices
        j = np.unique(np.hstack((tris.flatten(), quads.flatten())))
        # Compress node indices
        self.tris = compress_indices(tris, j)
        self.quads = compress_indices(quads, j)
        # Select the nodes
        self.nodes = self.nodes[j - 1]
        # Select the states (if present)
        if self.q is not None and self.q.size > 0:
            self.q = self.q[j - 1, :]
        # Reset node count
        self.nnode = self.nodes.shape[0]

   # --- Small volume cells ---
    # Check all four element types for "small" cells
    def report_small_cells(
            self,
            smallvol: Optional[float] = None,
            nrows: int = NROWS):
        # Calculate small volume if necessary
        if smallvol is None:
            # Get bounding box
            bbox = self.get_bbox()
            # Get widths in three axes
            widths = bbox[1::2] - bbox[::2]
            # Use fraction of max width
            smallvol = 2e-13 * np.max(widths)
        # Analyze each volume element type
        n1, m1 = self.report_small_tets(smallvol, nrows)
        n2, m2 = self.report_small_pyrs(smallvol, nrows)
        n3, m3 = self.report_small_pris(smallvol, nrows)
        n4, m4 = self.report_small_hexs(smallvol, nrows)
        # Total results
        nsmall = n1 + n2 + n3 + n4
        ntotal = m1 + m2 + m3 + m4
        # Print results
        print(f"TOTAL: {nsmall}/{pprint_n(ntotal)} small elements")

    # Analyze tetrahedra
    def report_small_tets(self, smallvol: float, nrows: int = NROWS):
        # Tetrahedra verts
        t = self.tets - 1
        # Get coordinates of tets
        x = self.nodes[t, 0]
        y = self.nodes[t, 1]
        z = self.nodes[t, 2]
        # Calculate volumes
        v = volcomp.tetvol(
            x[:, 1], y[:, 1], z[:, 1],
            x[:, 2], y[:, 2], z[:, 2],
            x[:, 3], y[:, 3], z[:, 3],
            x[:, 0], y[:, 0], z[:, 0])
        # Mask small volumes
        mask = v <= smallvol
        v = v[mask]
        x = x[mask]
        y = y[mask]
        z = z[mask]
        # Find smallest tris
        j = np.argsort(v)[:nrows]
        # Print results
        nsmall = np.sum(mask)
        ntotal = t.shape[0]
        print(f"TETS: {nsmall}/{pprint_n(ntotal)} small elements")
        # Print table
        if j.size:
            print_smallvol_table(v, j, x, y, z)
        # Output
        return nsmall, ntotal

    # Analyze pyramids
    def report_small_pyrs(self, smallvol: float, nrows: int = NROWS):
        # Element verts
        t = self.pyrs - 1
        # Get coordinates of tets
        x = self.nodes[t, 0]
        y = self.nodes[t, 1]
        z = self.nodes[t, 2]
        # Calculate volumes
        v = volcomp.pyrvol(
            x[:, 0], y[:, 0], z[:, 0],
            x[:, 3], y[:, 3], z[:, 3],
            x[:, 4], y[:, 4], z[:, 4],
            x[:, 1], y[:, 1], z[:, 1],
            x[:, 2], y[:, 2], z[:, 2])
        # Mask small volumes
        mask = v <= smallvol
        v = v[mask]
        x = x[mask]
        y = y[mask]
        z = z[mask]
        # Find smallest tris
        j = np.argsort(v)[:nrows]
        # Print results
        nsmall = np.sum(mask)
        ntotal = t.shape[0]
        print(f"PYRS: {nsmall}/{pprint_n(ntotal)} small elements")
        # Print table
        if j.size:
            print_smallvol_table(v, j, x, y, z)
        # Output
        return nsmall, ntotal

    # Analyze prisms
    def report_small_pris(self, smallvol: float, nrows: int = NROWS):
        # Element vert indices
        t = self.pris - 1
        # Get coordinates of verts
        x = self.nodes[t, 0]
        y = self.nodes[t, 1]
        z = self.nodes[t, 2]
        # Calculate volumes
        v = volcomp.privol(
            x[:, 0], y[:, 0], z[:, 0],
            x[:, 1], y[:, 1], z[:, 1],
            x[:, 2], y[:, 2], z[:, 2],
            x[:, 3], y[:, 3], z[:, 3],
            x[:, 4], y[:, 4], z[:, 4],
            x[:, 5], y[:, 5], z[:, 5])[-1]
        # Mask small volumes
        mask = v <= smallvol
        v = v[mask]
        x = x[mask]
        y = y[mask]
        z = z[mask]
        # Find smallest tris
        j = np.argsort(v)[:nrows]
        # Print results
        nsmall = np.sum(mask)
        ntotal = t.shape[0]
        print(f"PRIS: {nsmall}/{pprint_n(ntotal)} small elements")
        # Print table
        if j.size:
            print_smallvol_table(v, j, x, y, z)
        # Output
        return nsmall, ntotal

    # Analyze hexs
    def report_small_hexs(self, smallvol: float, nrows: int = NROWS):
        # Element vert indices
        t = self.hexs - 1
        # Get coordinates of verts
        x = self.nodes[t, 0]
        y = self.nodes[t, 1]
        z = self.nodes[t, 2]
        # Calculate volumes
        v = volcomp.hexvol(
            x[:, 0], y[:, 0], z[:, 0],
            x[:, 1], y[:, 1], z[:, 1],
            x[:, 2], y[:, 2], z[:, 2],
            x[:, 3], y[:, 3], z[:, 3],
            x[:, 4], y[:, 4], z[:, 4],
            x[:, 5], y[:, 5], z[:, 5],
            x[:, 6], y[:, 6], z[:, 6],
            x[:, 7], y[:, 7], z[:, 7])
        # Mask small volumes
        mask = v <= smallvol
        v = v[mask]
        x = x[mask]
        y = y[mask]
        z = z[mask]
        # Find smallest tris
        j = np.argsort(v)[:NROWS]
        # Print results
        nsmall = np.sum(mask)
        ntotal = t.shape[0]
        print(f"HEXS: {nsmall}/{pprint_n(ntotal)} small elements")
        # Print table
        if j.size:
            print_smallvol_table(v, j, x, y, z)
        # Output
        return nsmall, ntotal

  # === Slicer ===
    # Function to cut elements on plane
    def _cut_elements_vect(
            self, element_vertices, ele_types, cutinds,
            plane_normal, plane_center, Ieles=None, closed=True,
            edgetol=1e-8):
        r"""Cut element edges with given cut plane, generate
            interpolated nodes, values, and edge connectivity.

        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *element_vertices*: :class:`np.array`
                Array of elements (defined by vertex indices) to be cut
            *ele_types*: :class:`list`
                List of element types to cut
            *cutinds*: :class:`np.array`
                Index of each element to cut
            *plane_normal* :class:`np.array`
                Definition of cut plane normal (3,)
            *plane_center* :class:`np.array`
                Definition of cut plane center (3,)
            *Ieles*: {`None`} | :class:`np.array`
                Array labeling each element to cut as tri/quad (0/1)
            *closed*: {`False`} | `True`
                Attempt to close edges list
            *edgetol*: {`1e-4`} | :class:`float`
                Tolerance for unique point gen (min. parametric dist.)
        :Outputs:
            *intersection_points*: :class:`np.array`
                Points resulting from cut
            *intersection_values*: :class:`np.array`
                Interpolated values at points
            *intersection_edges*: :class:`np.array`
                Edges created from points
            *intersection_ids*: :class:`np.array`
                Uncut index of element that was cut to make edge
        :Versions:
            * 2025-05-13 `@aburkhea`: v1.0
        """
        nodes = self.nodes
        values = self.q
        element_vertices = np.array(element_vertices)
        if not isinstance(ele_types, list):
            ele_type = [ele_types]
        plane_normal = np.asarray(plane_normal)
        plane_center = np.asarray(plane_center)
        # Go through each element and get vertices
        for ie, ele_type in enumerate(ele_types):
            # Get edge connectivity for this element
            _edges = EDGECON[ele_type]
            i_indices, j_indices = _edges[:, 0], _edges[:, 1]
            # If no element types given, take all
            if Ieles is None:
                _ele_vertices1 = element_vertices[:, i_indices]
                _ele_vertices2 = element_vertices[:, j_indices]
            else:
                Iele = Ieles == ie
                # Calculate vertices and edges for all pairs at once
                _ele_vertices1 = element_vertices[Iele][:, i_indices]
                _ele_vertices2 = element_vertices[Iele][:, j_indices]
            # Pad tris
            if ele_type == "tris":
                # Need to pad col
                _ele_vertices1 = np.insert(_ele_vertices1, 3, -1, axis=1)
                _ele_vertices2 = np.insert(_ele_vertices2, 3, -1, axis=1)
            # Start cumulative vertices
            if ie == 0:
                ele_vertices1 = _ele_vertices1
                ele_vertices2 = _ele_vertices2
            # Append cumulative vertices
            else:
                ele_vertices1 = np.vstack([ele_vertices1, _ele_vertices1])
                ele_vertices2 = np.vstack([ele_vertices2, _ele_vertices2])
        # Get mask of -1s (padded indices)
        Ipad = ele_vertices1 == -1
        # Get values a vertexs
        vertices1_values = values[ele_vertices1.flatten()]
        vertices2_values = values[ele_vertices2.flatten()]
        # NAN padded values
        vertices1_values[Ipad.flatten()] = np.nan
        vertices2_values[Ipad.flatten()] = np.nan
        # Get nodes at vertexs
        vertices1_nodes = nodes[ele_vertices1.flatten()]
        vertices2_nodes = nodes[ele_vertices2.flatten()]
        # NAN padded values
        vertices1_nodes[Ipad.flatten()] = np.nan
        vertices2_nodes[Ipad.flatten()] = np.nan
        # Get edge nodes and edge values
        edges = vertices2_nodes - vertices1_nodes
        edges_val = vertices2_values - vertices1_values
        # Vector from plane point to all first vertices
        w_vectors = vertices1_nodes - plane_center
        # Calculate dot products in bulk
        dot_n_values = np.dot(edges, plane_normal)
        dot_w_values = np.dot(w_vectors, plane_normal)
        # Calculate s values safely
        with np.errstate(divide='ignore', invalid='ignore'):
            s_values = -dot_w_values / dot_n_values
        # Mask out nans and non intersecting
        Inan = ~np.isnan(s_values)
        # Filter out 0<s<1
        I1 = s_values <= 1.0
        I0 = s_values >= 0.0
        I2 = np.logical_and(I0, I1)
        # Combine into final mask
        II = np.logical_and(I2, Inan)
        # Supress warnings when multiplying with nans
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            # Vector intersection pts and values
            _intersection_points = vertices1_nodes + \
                edges*s_values.reshape(-1, 1)
            _intersection_values = vertices1_values + \
                edges_val*s_values.reshape(-1, 1)
        # Get valid points/vals
        intersection_points = _intersection_points[II]
        intersection_values = _intersection_values[II]
        intersection_edges = []
        intersection_ids = []
        # If we are cutting a surface element, assemble edges
        if ele_type in ("tris", "quads"):
            """Kind of a mess but for each surf ele, check the svalues
            from interp. calc, apply various logic to reduce duplicate
            interpolated nodes and connect edges to two points only.
            """
            filtered_intersection_pts = []
            filtered_values = []
            point_dict = {}
            # Nan masked points from interp calc (preserve abs. order)
            s_values[~II] = np.nan
            count = 0
            # Flatten these once at the start
            ev1f = ele_vertices1.flatten()
            ev2f = ele_vertices2.flatten()
            for ele in range(ele_vertices1.shape[0] - 1):
                # Each ele has 4 nodes (tris padded)
                i0 = ele*4
                # Get 4 values for this element
                svalele = s_values[i0:(i0 + 4)]
                edge_connect = []
                # Check each sval in element
                for ival, sval in enumerate(svalele):
                    # If svalue is nan we skip, (not valid intersection)
                    if np.isnan(sval):
                        continue
                    # If sval == 0, pt1 on cut plane, keep it w/o duplicating
                    elif np.isclose(sval, 0.0, atol=edgetol):
                        # Need to save key for future ref
                        key = tuple(sorted([ev2f[i0 + ival], ev1f[i0 + ival]]))
                        # Either find this edge cut or save if 1st time seeing
                        cnt = point_dict.setdefault(key, count)
                        # If its 1st time seeing it add to filtered pts list
                        if cnt == count:
                            filtered_intersection_pts.append(
                                _intersection_points[i0 + ival, :]
                            )
                            filtered_values.append(
                                _intersection_values[i0 + ival, :]
                            )
                            count += 1
                        # Append pt as possible connection
                        edge_connect.append(cnt)
                    # If sval == 1, pt2 on cut plane, keep it w/o duplicating
                    elif np.isclose(sval, 1.0, atol=edgetol):
                        # Need to save key for future ref
                        key = tuple(sorted([ev2f[i0 + ival], ev1f[i0 + ival]]))
                        # Either find this edge cut or save if 1st time seeing
                        cnt = point_dict.setdefault(key, count)
                        # If its 1st time seeing it add to filtered pts list
                        if cnt == count:
                            filtered_intersection_pts.append(
                                _intersection_points[i0 + ival, :]
                            )
                            filtered_values.append(
                                _intersection_values[i0 + ival, :]
                            )
                            count += 1
                        # Append pt as possible connection
                        edge_connect.append(cnt)
                    # Else new point made from intersected edge
                    elif II[i0 + ival]:
                        # Need to save key for future ref
                        key = tuple(sorted([ev2f[i0 + ival], ev1f[i0 + ival]]))
                        # Need to save these too to remove duplication
                        cnt = point_dict.setdefault(key, count)
                        # If its 1st time seeing it add to filtered pts list
                        if cnt == count:
                            filtered_intersection_pts.append(
                                _intersection_points[i0 + ival, :]
                            )
                            filtered_values.append(
                                _intersection_values[i0 + ival, :]
                            )
                            count += 1
                        # Should both be the same but just check
                        edge_connect.append(cnt)
                # Try to dedup
                dedup_edgecon = np.unique(np.array(edge_connect))
                # Ensure its length 2
                if len(dedup_edgecon) == 2:
                    intersection_edges.append(dedup_edgecon)
                    # Preserve the element index from all elements
                    intersection_ids.append(cutinds[ele])
            if closed:
                # Add in the final connection to make it a closed manifold
                pts, pt_count = np.unique(
                    intersection_edges,
                    return_counts=True
                )
                # Where pts are only referenced once (discontinuous edge)
                mask = pt_count == 1
                # If theres two discontinuous edges, connect them
                if np.count_nonzero(mask) == 2:
                    intersection_edges.append(pts[mask])
            intersection_points = np.array(filtered_intersection_pts)
            intersection_values = np.array(filtered_values)
        # return unique_pts, unique_vals
        return (intersection_points, intersection_values,
                intersection_edges, intersection_ids)

    # Interface to slice a surface
    def slice_surf(
            self,
            x: np.array,
            n: np.array,
            comp: Optional[Union[str, int, list]] = None,
            fname: Optional[str] = None,
            closed: Optional[bool] = False,
            **kw):
        r"""
        :Call:
            >>> slice = mesh.slice_surf(x, n, comp=None, fname=None)
        :Inputs:
            *x*: :class:`np.ndarray[float]`
                Coordinates of the cut plane center
            *n*: :class:`np.ndarray[float]`
                Vector normal to desired cut plane
            *comp*: :class:`{None}` | `str` | `list`
                Optional name of component(s) to intersect
            *fname*: :class:`{None}` | `str`
                Name of output file
            *closed*: :class: `{False}` | `True`
                Attempt to close returned edges
        :Outputs:
            *slice*: :class:`Umesh`
                Unstructured mesh instance for intersect
            *slice.nodes*: :class:`np.ndarray[float]`
                Nx3 array of node coordinates
            *slice.edges*: :class:`np.ndarray[int]`
                Nx2 array of edge nodes, sorted into contiguous sections
            *slice.edge_ids*: class:`np.ndarray[int]`
                The *tri_id* or *quad_id* from which this edge came
            *slice.q*: :class:`np.ndarray[float]`
                Values of the *mesh.q* at the *slice.nodes*
        :Versions:
            * 2025-05-12 ``@aburkhea``: v1.0
        """
        # Normalize the plane normal vector
        n = n / np.linalg.norm(n)
        # Get rotated points if not given
        rpts = rotate_points_to_plane(self.nodes, x, n)
        surfids = []
        # If string comp, given process to surfids
        if isinstance(comp, str):
            # Some comp -> surfid function here eventually
            raise NotImplementedError("String comps not supported yet")
        elif isinstance(comp, list):
            # Support mixed type comp (str and ints both given)
            for _comp in comp:
                if isinstance(comp, str):
                    # Some comp -> surfid function here eventually
                    raise NotImplementedError("String comps not supported yet")
                else:
                    # Assuming list of surfids already given
                    surfids.extend([_comp])
        # Default to cut everything
        elif comp is None:
            # Add up both tri and quad ids
            surfids = [np.unique(
                np.concatenate(
                    [self.get_tri_ids(),
                     self.get_quad_ids()
                     ]))
            ]
        # Prep surface element cut outputs
        edgelist = []
        surfpts = None
        surfvals = None
        for _surfids in surfids:
            # Calculate boundary bounds
            allseles = np.array([])
            celltype = []
            Itris = None
            Iquads = None
            # Get List of tris by surfid
            for surfid in _surfids:
                _Itris = self.get_tris_by_id(surfid)
                if Itris is None:
                    Itris = _Itris.reshape(-1, 1)
                else:
                    Itris = np.vstack((Itris, _Itris.reshape(-1, 1)))
            # Get List of quads by surfid
            for surfid in _surfids:
                _Iquads = self.get_quads_by_id(surfid)
                if Iquads is None:
                    Iquads = _Iquads.reshape(-1, 1)
                else:
                    Iquads = np.vstack((Iquads, _Iquads.reshape(-1, 1)))
            # Keep seperate surf element lists
            Isurfs = {"tris": Itris, "quads": Iquads}
            Ieles = []
            cutinds = []
            # For each element type
            for ie, ename in enumerate(("tris", "quads")):
                # Try to get element from mesh
                _eles = self.__getattribute__(ename)
                Isurf = Isurfs.get(ename)
                # If no elements of this type (or none in these sids)
                if _eles.size == 0 or Isurf.size == 0:
                    continue
                # Fix base 1 -> base 0 indexing for eles
                eles = _eles - 1
                # Pad tri element inds (N,3) to stack with quads later
                if ename == "tris":
                    eles = np.insert(eles, 3, -1, axis=1)
                surfeles = eles[Isurf[:, 0], :]
                # Get intersected elements
                if ename == "tris":
                    # Get mask for elements cut by plane
                    Icut = get_intersect_elems(surfeles[:, :3], rpts[:, 2])
                else:
                    Icut = get_intersect_elems(surfeles, rpts[:, 2])
                # Add mask to "global" list
                if allseles.size > 0:
                    # Stack surface elements
                    allseles = np.vstack([allseles, surfeles[Icut]])
                    _cutinds = np.where(Icut == 1)[0]
                    cutinds = np.concatenate([cutinds, _cutinds])
                    # Keep track of which surf elements are which type
                    _Ieles = np.ones(len(surfeles[Icut]), dtype=np.int8)*ie
                    # Append list of surf element types
                    Ieles = np.concatenate([Ieles, _Ieles])
                else:
                    allseles = surfeles[Icut]
                    Ieles = np.ones(len(allseles), dtype=np.int8)*ie
                    cutinds = np.where(Icut == 1)[0]
                celltype.extend((ename,))
            # If no elements to cut from these surfs, move on
            if allseles.size == 0:
                continue
            # Prep output plane and values
            csurfpts = None
            csurfvals = None
            edgetol = kw.get("edgetol", 1e-4)
            # Get vertex intersection and values for vol. element
            _results = self._cut_elements_vect(
                allseles, celltype, cutinds,
                n, x,
                Ieles=Ieles,
                closed=closed,
                edgetol=edgetol)
            # Get nodes of surface element cut plane
            csurfpts = _results[0]
            # Get vals of surface element cut plane
            csurfvals = _results[1]
            # Get edges of surface element cut plane
            edges = np.array(_results[2])
            # Get edges of surface element cut plane
            cedgeids = np.array(_results[3])
            # Assemble results for multiple surf id groups
            if surfpts is None:
                nsurfs = 0
                surfpts = csurfpts
                surfvals = csurfvals
                edgeids = cedgeids
            else:
                nsurfs = surfpts.shape[0]
                surfpts = np.vstack((surfpts, csurfpts))
                surfvals = np.vstack((surfvals, csurfvals))
                edgeids = np.concatenate((edgeids, cedgeids))
            # Add cumulative edge index
            edges = np.array(edges) + nsurfs
            edgelist.extend(edges)
        # Initialize output umesh of same class as self
        omesh = self.__class__()
        # Save outputs
        omesh.nodes = surfpts
        omesh.edges = np.array(edgelist)
        omesh.edge_ids = np.array(edgeids)
        omesh.q = surfvals
        return omesh


def get_intersect_elems(eles, ptsz):
    r"""Get mask of eles that are intersected by the z plane"""
    # Check if z > 0
    zbool = ptsz > 0
    # Get status of z node check for each element vertex
    dzM = zbool[eles]
    # Check if there are any sign changes within each ele (True = int.)
    return np.any(dzM != dzM[:, 0:1], axis=1)


# Rotate pts to arbitrary plane (align plane with z axis)
def rotate_points_to_plane(points, plane_center, plane_normal):
    r"""Rotates a numpy array of 3D Cartesian nodes such that the plane
    defined by the given normal vector becomes parallel to the xy-plane

    Inputs:
        *points*: class:`np.array`
            Array representing the node coordinates (x, y, z)
        *plane_center*: class:`np.array`
            Array of the center point of the cut plane (a, b, c)
        *plane_normal*: class:`np.array`
            Array of the normal vector to the cut plane (a, b, c)
    Outputs:
        *rotated_points*: class:`np.array`
            A numpy array of the rotated coordinates.
    r"""
    # Center the points according to plane
    points = points - plane_center
    # Calculate a rotation axis that rotates the normal vector to the z-axis
    # Cross product of normal and z-axis
    raxis = np.cross(plane_normal, np.array([0, 0, 1]))
    # Angle between normal and z-axis
    rotation_angle = np.arccos(plane_normal[2])
    # If the normal is already aligned with the z-axis, no rotation is needed
    if np.allclose(raxis, 0):
        return points
    # Normalize the rotation axis
    raxis = raxis / np.linalg.norm(raxis)
    # Create the rotation matrix using Rodrigues' rotation formula
    cth = np.cos(rotation_angle)
    sth = np.sin(rotation_angle)
    R = np.array([
        [cth + raxis[0]**2 * (1 - cth),
         raxis[0] * raxis[1] * (1 - cth) - raxis[2] * sth,
         raxis[0] * raxis[2] * (1 - cth) + raxis[1] * sth],

        [raxis[1] * raxis[0] * (1 - cth) + raxis[2] * sth,
         cth + raxis[1]**2 * (1 - cth),
         raxis[1] * raxis[2] * (1 - cth) - raxis[0] * sth],

        [raxis[2] * raxis[0] * (1 - cth) - raxis[1] * sth,
         raxis[2] * raxis[1] * (1 - cth) + raxis[0] * sth,
         cth + raxis[2]**2 * (1 - cth)]
    ])
    # Apply the rotation to each point
    rotated_points = np.dot(points, R.T)
    return rotated_points


def compress_indices(
        node_indices: np.ndarray,
        keep_indices: Optional[np.ndarray] = None) -> np.ndarray:
    # Check for trivial case
    if keep_indices.size / np.max(keep_indices) >= 0.999:
        # Keep all nodes
        return node_indices
    elif node_indices.size == 0:
        # Empty array
        return node_indices
    # Default list of prior indices
    if keep_indices is None:
        iold = np.unique(node_indices)
    else:
        iold = keep_indices
    # Number of nodes
    nnode = iold.size
    # Initilialize array of new indices
    inew = np.zeros(np.max(iold), dtype=iold.dtype)
    # Create map
    inew[iold - 1] = np.arange(1, nnode + 1, dtype=iold.dtype)
    # Renumber nodes
    T = inew[node_indices - 1]
    # Output
    return T


def _o_of_thousands(x: Union[float, int]) -> int:
    r"""Get the floor of the order of thousands of a given number

    If ``0``, then *x* is between ``0`` and ``1000``. If the number is
    at least ``1000`` but less than one million, then the output is
    ``1``. This is to correspond roughly to English-language names for
    large numbers.
    """
    if np.abs(x) < 1.0:
        return 0
    return int(np.log10(x) / 3)


def pprint_n(x: Union[float, int], nchar: int = 3) -> str:
    r"""Format a string as number thousands, millions, etc.

    :Call:
        >>> txt = pprint_n(x, nchar=3)
    :Inputs:
        *x*: :class:`int` | :class:`float`
            Number to format
        *nchar*: {``3``} | :class:`int`
            Number of digits to print
    :Outputs:
        *txt*: :class:`str`
            Formatted text
    """
    # Get number of tens
    nten = 0 if abs(x) < 1 else int(np.log10(x))
    # Get number of thousands
    nth = _o_of_thousands(x)
    # Extra orders of magnitude; 0 for 1.23k, 1 for 12.3k, 2 for 123k
    nround = nten - 3*nth
    # Number of digits after decimal we need to retain
    decimals = max(0, nchar - 1 - nround)
    # Divide *xround* by this to get to multiples of k, M, G, etc.
    exp2 = 3*nth
    div2 = 10 ** exp2
    # Final float b/w 0 and 1000, not including exponent
    y = x / div2
    # Get suffix
    suf = SI_SUFFIXES.get(nth)
    # Use appropriate flag
    if decimals and nth > 0:
        return "%.*f%s" % (decimals, y, suf)
    else:
        return "%i%s" % (int(y), suf)


# Generate table
def print_smallvol_table(v, j, x, y, z):
    # Header
    header = "%7s  %9s  %9s  %9s" % ("VOL", "X", "Y", "Z")
    # Dividing line
    hline = "  " + "  ".join(["========="] * 4)
    # Format
    fmt = "  %9.2e  " + "  ".join(["%9.2f"] * 3)
    # Print header
    print(hline)
    print(header)
    print(hline)
    # Calculate means
    xm = np.mean(x, axis=1)
    ym = np.mean(y, axis=1)
    zm = np.mean(z, axis=1)
    # Loop through smallest elements
    for i in j:
        print(fmt % (v[i], xm[i], ym[i], zm[i]))
    # Final line
    print(hline)
