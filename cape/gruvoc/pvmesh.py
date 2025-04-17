r"""
:mod:`gruvoc.pvmesh`: PyVista unstructured mesh class
====================================================

This module provides the :class:`Pvmesh` class that allows for mesh
and solution manipulation with :mod:`pyVista` and :mod:`vtk`. Particularly,
this is it to be used with meshes with the following elements as defined
by its pyVista CellType:

*   triangles[pv.CellType.TRIANGLE] (3 points) and
*   quads[pv.CellType.QUAD] (4 points)
*   tetrahedra[pv.CellType.TETRA] (4 tri faces and 4 nodes)
*   pyramids[pv.CellType.PYRAMID] (4 tri faces, 1 quad, and 5 nodes)
*   prisms[pv.CellType.WEDGE] (2 tri faces, 3 quads, and 6 nodes)

Do to the similar nature of the mesh structure, this class inherits
methods from :class:`UmeshBase`.

"""
# Standard library
import os
from io import IOBase
from typing import Optional, Union

# Third party
import numpy as np

# Local imports
from .umeshbase import UmeshBase
from .umesh import name2format
from .ugridfile import read_ugrid
from .surfconfig import SurfConfig
from .pltfile import write_plt
from .trifile import write_triq
from .fileutils import openfile
from .flowfile import read_fun3d_flow, read_fun3d_tavg

# Optional imports
try:
    import pyvista as pv
    from pyvista.core.filters import _get_output, _update_alg
    from vtkmodules.vtkCommonDataModel import vtkPlane
    from vtkmodules.vtkFiltersCore import vtk3DLinearGridPlaneCutter
except ModuleNotFoundError:
    pass


class Pvmesh(UmeshBase):
    # === Class attributes ===
    __slots__ = (
        "pvmesh",
        "pvslice",
    )

    def __init__(
        self,
        fname_or_fp: Optional[Union[str, IOBase]] = None,
        sname_or_sp: Optional[Union[str, IOBase]] = None,
        meta: bool = False,
        **kw):
        # Intialize every slot
        for slot in UmeshBase.__slots__:
            setattr(self, slot, None)
        # Read if fname given
        if fname_or_fp is not None:
            self.read(fname_or_fp, meta, **kw)
        # Read if sname given (solution: flow or tavg)
        if sname_or_sp is not None:
            self.readsol(sname_or_sp, **kw)
        # Common kwargs
        fmt = kw.get("fmt")
        # Check for file names/handles by kwarg
        fugrid = kw.get("ugrid")
        fflow = kw.get("flow")
        ftavg = kw.get("tavg")
        # Unspecified-type config file
        fcfg = kw.get("c", kw.get("config"))
        novol = kw.get("novol", not kw.get("vol", True))
        # Check for kwarg-based formats
        if fugrid:
            # UGRID (main format)
            self.read_ugrid(fugrid, meta=meta, fmt=fmt, novol=novol)
            # Read Fun3D solutions
            if fflow:
                # Read flow file
                self.read_fun3d_flow(fflow)
                # Add CP and Mach
                self.add_cp()
                self.add_mach()
            elif ftavg:
                # Read tavg file
                self.read_fun3d_tavg(ftavg)
                # Add CP and Mach
                self.add_cp()
                self.add_mach()
        # Read config
        if fcfg:
            # Generic config
            self.config = SurfConfig(fcfg)
        else:
            # Create empty config
            self.config = SurfConfig()

        # Make pyvista mesh
        if self.mesh_type == 'unstruc':
            self._make_pv_unstructuredmesh()
        # Initialize slices
        self.pvslice = {}

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
            else:
                print("Grid format not supported")

    def readsol(
            self,
            sname_or_sp: Union[str, IOBase],
            **kw):
        # Open file if able
        with openfile(sname_or_sp) as sp:
            # Get file name
            sdir, sname = os.path.split(os.path.abspath(sp.name))
            # Read format from file name
            solfmt = sname.split('.')[-1]
            # Status update
            if kw.get('v', False):
                print(
                    f"Reading file '{sname}' using format '{solfmt}'")
            # Read data
            if solfmt == "flow":
                # Flow file
                self.read_fun3d_flow(sp)
            elif solfmt == "tavg":
                # Tavg file
                self.read_fun3d_tavg(sp)
            else:
                print("Solution format not supported")

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

    def read_ugrid(
            self,
            fname_or_fp: Union[str, IOBase],
            meta: bool = False,
            fmt: Optional[str] = None,
            novol: bool = False):
        read_ugrid(self, fname_or_fp, meta, fmt, novol=novol)

   # === pyVista tools ===
    def _make_pv_unstructuredmesh(self):
        r"""Make a pyVista unstructed mesh with solutions vars if present

        :Call:
            >>> mesh.make_pv_unstructuredmesh()
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
        """
        # Set cell types
        celltype = np.concatenate((
            np.repeat(pv.CellType.TRIANGLE, self.ntri),
            np.repeat(pv.CellType.QUAD, self.nquad),
            np.repeat(pv.CellType.TETRA, self.ntet),
            np.repeat(pv.CellType.PYRAMID, self.npyr),
            np.repeat(pv.CellType.WEDGE, self.npri)
        ), dtype=np.int8)
        # Generate array of cells
        pt_arrays = (
            np.hstack((3*np.ones((self.ntri, 1), dtype=int), self.tris-1)),
            np.hstack((4*np.ones((self.nquad, 1), dtype=int), self.quads-1)),
            np.hstack((4*np.ones((self.ntet, 1), dtype=int), self.tets-1)),
            np.hstack((5*np.ones((self.npyr, 1), dtype=int), self.pyrs-1)),
            np.hstack((6*np.ones((self.npri, 1), dtype=int), self.pris-1))
        )
        # Ravel each array and combine
        cells = np.concatenate([pt_array.ravel() for pt_array in pt_arrays])
        # Generate mesh
        self.pvmesh = pv.UnstructuredGrid(
            cells,
            celltype,
            self.nodes
        )
        # Add solution files if present
        if self.qvars:
            for i, var in enumerate(self.qvars):
                self.pvmesh.point_data[var] = self.q[:, i]

    def slice(self,
                      name: str = 'plane-y0',
                      origin: list = (0.0, 0.0, 0.0),
                      normal: list = (0.0, 0.0, 1.0)
                      ):
        r"""Make a pyVista slice

        :Call:
            >>> mesh.make_pv_slice()
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *origin: :class:'list'
                Origin point for plane slice
            *normal: :class:'list'
                Normal for plane slice
        """
        # Check if pyvista unstructured grid present if not make one
        if not self.pvmesh:
            self.make_pv_unstructuredmesh()
        # Make a vtk plane object
        plane = vtkPlane()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)
        # Make the cutter object
        alg = vtk3DLinearGridPlaneCutter()
        # Add mesh data to cutter object
        alg.SetInputDataObject(self.pvmesh)
        # Set plane for slicing
        alg.SetPlane(plane)
        # Make Slice
        _update_alg(alg)
        # Instance the slice dict
        if not self.pvslice:
            self.pvslice = {}
        # Get output slice and asdd to slice dict
        self.pvslice[name] = _get_output(alg)

   # ==== Writers ===
    def write_slicetriq(self,
                        name: str = 'plane-y0',
                        fname: str = "out.triq"
                        ):
        r"""Make a triq from a slice

        :Call:
            >>> mesh.make_slice_triq()
        :Inputs:
            *mesh*: :class:`Pvmesh`
                Unstructured mesh instance
         """
        # Will need to split slice into tris and quads
        sl_tri = self.pvslice[name].extract_cells_by_type(pv.CellType(5))
        sl_quad = self.pvslice[name].extract_cells_by_type(pv.CellType(9))
        # Will focus on the tris for now
        # Instance empty Umesh
        mesh = Umesh()
        # Save tris
        mesh.tris = sl_tri.faces.reshape(-1, 4)[:, 1:] + 1
        mesh.ntri = np.shape(mesh.tris)[0]
        # Save nodes
        mesh.nodes = sl_tri.points
        mesh.nnode = np.shape(mesh.nodes)[0]
        # Save solution
        mesh.q = np.stack(
            [sl_tri.point_data[I].transpose() for I in sl_tri.point_data],
            axis=1
        )
        mesh.nq = np.size(sl_tri.point_data.keys())
        # Make trids, just a single one
        mesh.tri_ids = np.ones(mesh.ntri, dtype=int)
        # Write the triq
        mesh.write_triq(f'{fname}.lr4.triq')

    def write_sliceplt(self,
                        name: str = 'plane-y0',
                        fname: str = "out"
                        ):
        r"""Make a triq from a slice

        :Call:
            >>> mesh.make_slice_triq()
        :Inputs:
            *mesh*: :class:`Pvmesh`
                Unstructured mesh instance
         """
        # Will need to split slice into tris and quads
        sl_tri = self.pvslice[name].extract_cells_by_type(pv.CellType(5))
        sl_quad = self.pvslice[name].extract_cells_by_type(pv.CellType(9))
        # Will focus on the tris for now
        # Instance empty Umesh
        mesh = Umesh()
        # Save tris
        mesh.tris = sl_tri.faces.reshape(-1, 4)[:, 1:] + 1
        mesh.ntri = np.shape(mesh.tris)[0]
        # Save nodes
        mesh.nodes = sl_tri.points
        mesh.nnode = np.shape(mesh.nodes)[0]
        # Save solution
        mesh.q = np.hstack(
            [sl_tri.point_data[I].reshape(-1, 1) for I in sl_tri.point_data]
        )
        mesh.nq = np.size(sl_tri.point_data.keys())
        mesh.qvars = self.qvars
        # Make trids, just a single one
        mesh.tri_ids = np.ones(mesh.ntri, dtype=int)
        # Have to initialize quads if it hasn't been set
        if mesh.quads is None:
            mesh.quads = np.empty((0, 4), dtype=int)
        # Write the triq
        mesh.write_plt(f'{fname}.plt')

    def write_vtp(self,
                  name: str = 'plane-y0'
                 ):
        r"""Write a VTK polyhedral mesh file for slices

        :Call:
            >>> mesh.write_vtp(name)
        :Inputs:
            *mesh*: :class:`Umesh`
                Unstructured mesh instance
            *name: :class:'string'
                String idetifier for slice
        """
        # Return if no pvmesh
        if not self.pvmesh:
            return
        # Return if no slices
        if not self.pvslice:
            return
        # Check to see if plane name exists in slice dict
        if name in self.pvslice.keys():
            # Make output file name
            ofile = f"{name}.vtp"
            # Save vtp file using pyVista
            self.pvslice[name].save(ofile)
        else:
            # Do nothing
            return

    def write_plt(
            self,
            fname_or_fp: Union[str, IOBase],
            v: bool = False):
        # Write mesh
        write_plt(self, fname_or_fp, v=v)

    def write_triq(
            self,
            fname_or_fp: Union[str, IOBase],
            fmt: Optional[str] = None):
        write_triq(self, fname_or_fp, fmt)