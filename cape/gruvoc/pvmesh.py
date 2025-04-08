
# Standard library

# Third party
import numpy as np

# Local imports
from .umesh import Umesh

# Optional imports
try:
    import pyvista as pv
    from pyvista.core.filters import _get_output, _update_alg
    from vtkmodules.vtkCommonDataModel import vtkPlane
    from vtkmodules.vtkFiltersCore import vtk3DLinearGridPlaneCutter
except ModuleNotFoundError:
    pass


class Pvmesh(Umesh):

    def make_pv_unstructuredmesh(self):
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

    def make_pv_slice(self,
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
        # Initialiaze dict of slices
        if not self.pvslice:
            self.pvslice = {}
        # Get output slice and asdd to slice dict
        self.pvslice[name] = _get_output(alg)

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
        mesh.q = np.stack(
            [sl_tri.point_data[I].transpose() for I in sl_tri.point_data]
        )
        mesh.nq = np.size(sl_tri.point_data.keys())
        # Make trids, just a single one
        mesh.tri_ids = np.ones(mesh.ntri, dtype=int)
        # Something else needs to happen here to use gruvocs plt writer
        # cape-tri2plt works on tris made from here, but write_plt fails
        return
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