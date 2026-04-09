r"""
:mod:`cape.pylava.casecntl`: LAVA case control module
=========================================================

This module contains LAVACURV-specific versions of some of the generic
methods from :mod:`cape.cfdx.case`.

All of the functions from :mod:`cape.case` are imported here.  Thus
they are available unless specifically overwritten by specific
:mod:`cape.pylava` versions.
"""

# Standard library modules
import os
import re
from typing import Optional

# Third-party modules
import numpy as np
try:
    import pyvista as pv
    from scipy.spatial import Delaunay
    from vtkmodules.vtkFiltersPoints import vtkPointInterpolator2D
    from vtkmodules.vtkFiltersPoints import vtkLinearKernel
except ImportError:
    Delaunay = None

# Local imports
from . import cmdgen
from .. import fileutils
from .databook import CaseFM, CasePointProbe, CaseResid
from .dataiterfile import DataIterFile
from .runinpfile import CartInputFile
from .yamlfile import RunYAMLFile
from .options.runctlopts import RunControlOpts
from ..cfdx import casecntl
from ..dkit import capefile
from ..dkit.rdb import DataKit
from ..fileutils import tail
from ..gruvoc.umesh import Umesh
from ..capeio import (
    fromfile_lb4_i,
    fromfile_lb8_i,
    tofile_lb4_i,
    tofile_lb4_f,
    tofile_lb8_i,
    tofile_lb8_f)

# Constants
ITER_FILE = "data.iter"
ITER_FILE_CART = os.path.join("monitor", "Cart.data.iter")
BATCHSIZE = 100


# Function to complete final setup and call the appropriate LAVA commands
def run_lavacurv():
    r"""Setup and run the appropriate LAVACURV command

    :Call:
        >>> run_lavacurv()
    :Versions:
        * 2024-09-30 ``@sneuhoff``: v1.0;
    """
    # Get a case reader
    runner = CaseRunner()
    # Run it
    return runner.run()


# Class for running a case
class CaseRunner(casecntl.CaseRunner):
   # --- Class attributes ---
    # Additional atributes
    __slots__ = (
        "data_iter",
        "runinpfile",
        "runinpfile_j",
        "yamlfile",
        "yamlfile_j",
    )

    # Names
    _modname = "pylava"
    _progname = "lava"

    # Specific classes
    _rc_cls = RunControlOpts
    _resid_cls = CaseResid
    _dex_cls = {
        "fm": CaseFM,
        "pointprobe": CasePointProbe,
    }

   # --- Config ---
    def init_post(self):
        r"""Custom initialization for pyfun

        :Call:
            >>> runner.init_post()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Versions:
            * 2023-06-28 ``@ddalle``: v1.0
        """
        self.data_iter = None
        self.runinpfile = None
        self.runinpfile_j = None
        self.yamlfile = None
        self.yamlfile_j = None

   # --- Case control/runners ---
    # Run one phase appropriately
    @casecntl.run_rootdir
    def run_phase(self, j: int):
        r"""Run one phase using appropriate commands

        :Call:
            >>> runner.run_phase(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2024-08-02 ``@sneuhoff``: v1.0
            * 2024-10-11 ``@ddalle``: v1.1; split run_superlava()
        """
        # Run main executable
        self.run_superlava(j)

    # Run superlava one time
    @casecntl.run_rootdir
    def run_superlava(self, j: int):
        r"""Run one phase of the ``superlava`` executable

        :Call:
            >>> runner.run_superlava(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2024-10-11 ``@ddalle``: v1.0
        """
        # Read case settings
        rc = self.read_case_json()
        # Get solver type
        solver = rc.get_LAVASolver()
        # Check which command to generate
        if solver == "curvilinear":
            # LAVA-Curvilinear
            cmdi = cmdgen.superlava(rc, j)
            execname = "superlava"
        elif solver == "cartesian":
            # LAVA-Cartesian
            cmdi = cmdgen.lavacart(rc, j)
            execname = "lava"
        # Run the command
        self.callf(cmdi, f=f"{execname}.out", e=f"{execname}.err")

    # Clean up files afterwrad
    def finalize_files(self, j: int):
        r"""Clean up files after running one cycle of phase *j*

        :Call:
            >>> runner.finalize_files(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2024-10-11 ``@ddalle``: v1.0
        """
        # Get the current iteration number
        n = self.get_iter()
        # Genrate name of STDOUT log, "run.{phase}.{n}"
        fhist = "run.%02i.%i" % (j, n)
        # Get solver
        rc = self.read_case_json()
        solver = rc.get_LAVASolver()
        # Get STDOUT file name
        stdoutbase = "superlava" if solver == "curvilinear" else "lava"
        # Rename the STDOUT file
        if os.path.isfile(f"{stdoutbase}.out"):
            # Move the file
            os.rename(f"{stdoutbase}.out", fhist)
        else:
            # Create an empty file
            fileutils.touch(fhist)

   # --- Status ---
    # Function to get total iteration number
    def getx_restart_iter(self) -> int:
        r"""Get total iteration number of most recent flow file

        :Call:
            >>> n = runner.getx_restart_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *n*: :class:`int`
                Index of most recent check file
        :Versions:
            * 2024-09-16 ``@sneuhoff``: v1.0
        """
        # Read options
        rc = self.read_case_json()
        # Get solver type
        solver = rc.get_LAVASolver()
        # Check which command to generate
        if solver == "cartesian":
            # Search for a restart file
            pat = self.genr8_restart_regex()
            mtch = self.match_regex(pat)
            # Check for a search result
            if mtch is None:
                return 0
            # Infer iteration
            n = int(mtch.group(1))
            return n
        # Fallback to current iter
        return self.getx_iter()

    def get_restart_ctu(self) -> float:
        # Read options
        rc = self.read_case_json()
        # Get solver type
        solver = rc.get_LAVASolver()
        # Check which command to generate
        if solver == "cartesian":
            # Get iteartion
            n = self.get_restart_iter()
            # Read data.iter
            dat = self.read_data_iter(meta=False)
            # Locate *n* in history
            mask, = np.where(dat["nt"] == n)
            # Check for match
            if mask.size == 0:
                return 0.0
            # Convert to CTU
            return dat["ctu"][mask[0]]
        # Fallback
        return 0.0

    # Get current iteration
    def getx_iter(self):
        r"""Get the most recent iteration number for a LAVA case

        :Call:
            >>> n = runner.getx_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *n*: :class:`int` | ``None``
                Last iteration number
        :Versions:
            * 2024-08-02 ``@sneuhoff``: v1.0
            * 2024-10-11 ``@ddalle``: v2.0; use DataIterFile(meta=True)
        """
        # Read it, but only metadata
        db = self.read_data_iter(meta=True)
        # Return the last iteration
        return db.n

    # Get current iteration
    def get_ctu(self) -> float:
        r"""Get the most recent iteration Char. Time Unit value

        :Call:
            >>> t = runner.get_ctu()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *t*: :class:`float`
                Last time step (characteristic units)
        :Versions:
            * 2025-08-14 ``@sneuhoff``: v1.0
        """
        # Read it, but only metadata
        db = self.read_data_iter(meta=True)
        # Return the last iteration
        return db.t

    # Get CTU cutoff
    def get_ctu_max(self, j: Optional[int] = None) -> float:
        r"""Get the characteristic time units cutoff for phase *j*

        :Call:
            >>> t = runner.get_ctu_max()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *t*: :class:`float`
                Last time step (characteristic units)
        :Versions:
            * 2025-08-15 ``@sneuhoff``: v1.0
        """
        # Get solver
        rc = self.read_case_json()
        solver = rc.get_LAVASolver()
        # Filter
        if solver == "cartesian":
            # Default phase
            j = j if (j is not None) else self.get_phase()
            # Read settings
            opts = self.read_runinputs(j)
            # Get option
            ctumax = opts.get_opt("time", "finish ctu")
            ctumax = 0.0 if ctumax is None else ctumax
            # Use it
            return ctumax
        # Fallback
        return 0.0

    # Check for exit criteria
    def check_alt_exit(self, j: Optional[int] = None) -> bool:
        # Get solver
        rc = self.read_case_json()
        solver = rc.get_LAVASolver()
        # Default phase
        j = j if (j is not None) else self.get_phase()
        # Filter
        if solver == "cartesian":
            # Check for CTU criteria
            ctumax = self.get_ctu_max(j)
            if (not ctumax):
                return False
            # Check current value
            ctu = self.get_ctu()
            return bool(ctu + 0.5 >= ctumax)
        elif solver == "curvilinear":
            # Read YAML file
            yamlfile = self.read_runyaml(j)
            # Section with convergence stuff
            sec = "nonlinearsolver"
            # Maximum iterations
            maxiters = yamlfile.get_lava_subopt(sec, "iterations")
            # Read data
            db = self.read_data_iter(meta=False)
            if db.n >= maxiters:
                return True
            # Target convergence
            l2conv_target = yamlfile.get_lava_subopt(sec, "l2conv")
            # Apply it
            if l2conv_target:
                # Check reported convergence
                return db.l2conv <= l2conv_target
        # Fallback
        return False

   # --- DEx ---
    # Create tuple of args after *comp*
    def get_dex_args_post_pointprobe(self) -> tuple:
        r"""Get list of args prior to component name in :class:`CaseFM`

        :Call:
            >>> args = runner.get_dex_args_post_iterfm()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *args*: :class:`tuple`\ [:class:`str`]
                Tuple of one arg, *runner*
        :Versions:
            * 2025-09-25 ``@ddalle``: v1.0
        """
        return (self,)

    # Get dex iter for point probe
    @casecntl.run_rootdir
    def get_dex_iter_pointprobe(self, comp: str) -> int:
        r"""Get number of iterations available for a LAVA point probe

        :Call:
            >>> n = runner.get_dex_iter_pointprobe(comp)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *comp*: :class:`str`
                Name of DataBook component
        :Outputs:
            *n*: :class:`int`
                Iteration count
        :Versions:
            * 2026-03-20 ``@ddalle``: v1.0
        """
        # Get probe index
        j = self.get_dex_opt(comp, "Index")
        # Pattern for file names
        pat = os.path.join("point_probe", rf"Cart\.[0-9]+\.pnt{j:04d}.dat")
        # Find those files
        filelist_raw = self.search_regex(pat)
        # Use latest
        if len(filelist_raw) == 0:
            return 0
        # Read last line of newest file
        line = tail(filelist_raw[-1], n=1)
        # Get integer
        return int(line.split()[0])

   # --- Isosurface data ---
    @casecntl.run_rootdir
    def triangulate_cutplane(
            self,
            nsurf: int,
            clean: bool = False,
            nmax: Optional[int] = None):
        r"""Convert cut plane VTK files to triangulated cut planes

        :Call:
            >>> runner.triangulate_cutplane(nsurf, clean, nmax=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *nsurf*: :class:`int`
                Surface index
            *clean*: ``True`` | {``False``}
                Option to delete original cut plane file
            *nmax*: {``None``} | :class:`int`
                Optional maximum number of files to triangulate
        """
        # Search pattern for these cutplane files
        prefix = self._genr8_cutplane_prefix(nsurf)
        vtkpat = f"{prefix}\\.[0-9]+\\.vtk"
        # Get any current VTK files
        print(
            f"  Searching for cutplanes matching {os.path.basename(prefix)}")
        vtkfiles = sorted(self.search_regex(vtkpat))
        # Get integers from these file names
        iters = [int(v.rsplit('.', 2)[-2]) for v in vtkfiles]
        # Current count of conversions
        ntri = 0
        # Loop through those
        for n in iters:
            # Convert
            q = self.triangulate_cutplane_n(nsurf, n, clean)
            # Update count
            if not q:
                continue
            # Update counter
            ntri += 1
            # Check for nmax
            if (nmax is not None) and (ntri >= nmax):
                break

    def triangulate_cutplane_n(
            self,
            surf: int,
            n: int,
            clean: bool = False) -> bool:
        r"""Create a triangulation so that each point is in a unique tri

        This will convert a file such as

            ``surf00_cutplane_Colorfullq_Cart.000045000.vtk``

        and create from it

            ``surf00_cutplane_Colorfullq_Cart.000045000.tri.vtk``

        :Call:
            >>> q = runner.triangulate_cutplane_n(surf, n, clean=False)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *surf*: :class:`int`
                Surface index
            *n*: :class:`int`
                Iteration number
            *clean*: ``True`` | {``False``}
                Option to delete original cut plane file
        :Outputs:
            *q*: :class:`bool`
                Whether a new file was created
        :Versions:
            * 2026-04-08 ``@ddalle``: v1.0
        """
        # Read cut plane definition
        defn = self.read_cutplane_defn(surf)
        # Check for valid cut plane
        if defn is None:
            return
        # Get name of file
        prefix = self._genr8_cutplane_prefix(surf, defn)
        basename = f"{prefix}.{n:09}"
        fvtk = f"{basename}.vtk"
        ftri = f"{basename}.tri.vtk"
        # Abbreviated file names
        vtk1 = f"surf{surf-1:02d}_cutplane_...{n:09d}.vtk"
        vtk2 = f"surf{surf-1:02d}_cutplane_...{n:09d}.tri.vtk"
        # Check for files
        if os.path.isfile(ftri):
            print(f"  File exists: '{vtk2}'")
            # Cleanup
            if clean and os.path.isfile(fvtk):
                print(f"    rm '{fvtk}'")
                self.remove_file(fvtk)
            return False
        elif not os.path.isfile(fvtk):
            return False
        # Status update
        msg = f"'{vtk1}' => '{vtk2}'"
        print(f"  {msg}")
        self.log_verbose(msg)
        # Read the tri file
        mesh = Umesh(fvtk)
        # Project the nodes to the cut plane
        triangulate_mesh(mesh, defn["normal"], defn["point"])
        # Write it
        mesh.write_vtk(ftri)
        # Cleanup
        if clean:
            print(f"    rm '{fvtk}'")
            self.remove_file(fvtk)
        # Output
        return True

    def read_cutplane_defn(self, surf: int) -> Optional[dict]:
        r"""Read definition for a cut plane isosurface

        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *surf*: :class:`int`
                Isosurface number (1-based)
        :Versions:
            * 2026-04-08 ``@ddalle``: v1.0
        """
        # Read input file
        inp = self.read_runinputs()
        # Name of the isosurface
        name = f"isosurfaces_{surf}"
        # Get section
        sec = inp.get_opt("cartesian", "output")
        if sec is None:
            return
        # Get definition
        defn = sec.get(name)
        if not isinstance(defn, dict):
            return
        # Get normal and point
        p = defn.get("point")
        n = defn.get("normal")
        # Check both
        if not (isinstance(p, list) and isinstance(n, list)):
            return
        # Normalize vectors
        defn["point"] = np.array(p)
        defn["normal"] = np.array(n) / np.linalg.norm(n)
        # Output
        return defn

    @casecntl.run_rootdir
    def read_cutplane_fixed(
            self,
            nsurf: int,
            n: int,
            nref: int = 0) -> Optional[Umesh]:
        # Read cut plane definition
        defn = self.read_cutplane_defn(nsurf)
        # Get file name prefix for this cut plane
        prefix = self._genr8_cutplane_prefix(nsurf, defn)
        # Base file name for this iteration
        basename = f"{prefix}.{n:09d}"
        # Potential file names
        ftri = f"{basename}.tri.vtk"
        ffix = f"{basename}.fixed.vtk"
        # Check for fixed-mesh file
        if os.path.isfile(ffix):
            return Umesh(ffix)
        # Read triangulated data on the reference iteration
        refmesh = self.read_cutplane_tri(nsurf, nref)
        # Read triangulated data on this iteration
        mesh = self.read_cutplane_tri(nsurf, n)
        # Create interpolator
        interp = vtkPointInterpolator2D()
        interp.SetNullPointsStrategyToClosestPoint()
        interp.SetKernel(vtkLinearKernel())
        # Set input and output
        interp.SetInputData(refmesh.pvmesh)
        interp.SetSourceData(mesh.pvmesh)
        # Define the plane
        interp.SetOrigin(defn["point"])
        interp.SetNormal(defn["normal"])
        # Interpolate
        print(f"  Interpolating '{ftri}' based on iter {nref}")
        interp.Update()
        # Get result
        result = pv.wrap(interp.GetOutput())
        # Create Umesh
        fixmesh = Umesh.from_pvmesh(result)
        # Output
        return fixmesh

    @casecntl.run_rootdir
    def read_cutplane_tri(self, nsurf: int, n: int) -> Optional[Umesh]:
        r"""Read triangulated cut-plane file

        :Call:
            >>> mesh = runner.read_cutplane_tri(nsurf, n)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *nsurf*: :class:`int`
                Surface index
            *n*: :class:`int`
                Iteration number
        :Outputs:
            *mesh*: ``None`` | :class:`cape.gruvoc.umesh.Umesh`
                Triangulated cut plane instance
        :Versions:
            * 2026-04-09 ``@ddalle``: v1.0
        """
        # Read cut plane definition
        defn = self.read_cutplane_defn(nsurf)
        # Check for valid cut plane
        if defn is None:
            return
        # Get name of file
        prefix = self._genr8_cutplane_prefix(nsurf, defn)
        basename = f"{prefix}.{n:09d}"
        # Potential file names
        fvtk = f"{basename}.vtk"
        ftri = f"{basename}.tri.vtk"
        # Check for file
        if os.path.isfile(ftri):
            return Umesh(ftri)
        # Check for raw cut plane file
        if not os.path.isfile(fvtk):
            return
        # Read the tri file
        mesh = Umesh(fvtk)
        # Project the nodes to the cut plane
        triangulate_mesh(mesh, defn["normal"], defn["point"])
        # Return that
        return mesh

    @casecntl.run_rootdir
    def read_cutplane_meta(self, nsurf: int) -> DataKit:
        r"""Read database of metadata for collected cutplane data

        :Call:
            >>> db = runner.read_surfdata_meta(nsurf)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *nsurf*: :class:`int`
                Surface index (1-based)
        :Outputs:
            *db*: :class:`cape.dkit.rdb.DataKit`
                DataKit of which batch each iteration is located in
        :Versions:
            * 2026-04-05 ``@ddalle``: v1.0
        """
        # Create file name
        fname = self._genr8_cutplane_metafile(nsurf)
        # Check for file
        if not os.path.isfile(fname):
            # Initialize datakit
            db = DataKit()
            # Save parameters
            db.save_col("nt", 0)
            db.save_col("i", np.zeros(0, dtype="i4"))
            db.save_col("batch", np.zeros(0, dtype="i4"))
            # Output
            return db
        # Otherwise read it
        return DataKit(fname)

    def _genr8_cutplane_prefix(
            self,
            surf: int, defn: Optional[dict] = None) -> str:
        # Read definition if necessary
        if defn is None:
            defn = self.read_cutplane_defn(surf)
        # Isosurface type
        colorfield = defn["colorfield"].lower().replace(' ', '')
        # Suffix based on contents
        if colorfield == "densitygradmag":
            suf = "DensityGradMag"
        else:
            suf = "fullq"
        # Get name of file
        return os.path.join(
            "isosurface", f"surf{surf-1:02d}_cutplane_Color{suf}_Cart")

   # --- Surface data ---
    def collect_surfdata(
            self,
            nsurf: int = 0,
            nbatch: int = BATCHSIZE,
            clean: bool = False,
            nmax: Optional[int] = None):
        r"""Combine data from LAVA surface VTK files into batches

        :Call:
            >>> runner.collect_surfdata(nsurf, nbatch, clean, nmax=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *nsurf*: {``0``} | :class:`int`
                LAVA surface number
            *nbatch*: {``100``} | :class:`int`
                Number of surface snapshots to collect into each file
            *clean*: ``True`` | {``False``}
                Option to delete ``.vtk`` files after processing
            *nmax*: {``None``} | :class:`int`
                Maximum number of snapshots to collect
        :Versions:
            * 2026-04-06 ``@ddalle``: v1.0
        """
        # First read metadata
        db = self.read_surfdata_meta(nsurf)
        # Number of time steps saved
        nt = db["nt"]
        # Get current batch info
        if nt == 0:
            # Starting fresh
            iref = 0
            imax = 0
        else:
            # Get latest
            iref = db["i"][0]
            imax = db["i"][-1]
        # Get any current VTK files
        vtkpat = self._genr8_surfdata_regex(nsurf)
        vtkfiles = sorted(self.search_regex(vtkpat))
        # Get integers from these file names
        iters = [int(v.rsplit('.', 2)[-2]) for v in vtkfiles]
        # Number of saved files
        n = 0
        # List of files to remove (this batch)
        rmfiles = []
        # Loop through files
        for i in iters:
            # Name of VTK file
            fvtk = self._genr8_surfdata_reffile(nsurf, i)
            # Check if already covered
            if i <= imax:
                # Check for clean option
                if clean and (i != iref) and (i > 0):
                    # Delete it
                    print(f"  Already processed '{fvtk}'")
                    rmfiles.append(fvtk)
                continue
            # Increase counter
            nt += 1
            # Get batch
            batchj = nt // nbatch
            batchk = nt % nbatch
            # Check if new batch
            newbatch = db["batch"].size and (db["batch"][-1] != batchj)
            # Append to vectors
            db["nt"] = nt
            db["i"] = np.hstack((db["i"], i))
            db["batch"] = np.hstack((db["batch"], batchj))
            # Status update
            msg = (
                f"  Collecting '{fvtk}' " +
                f"-> batch {batchj} ({batchk}/{nbatch})")
            print(msg)
            self.log_verbose(msg)
            # Write data
            self._write_surfdata(i, nsurf, batchj)
            # Update the batch data
            self.write_surfdata_meta(nsurf, db)
            # Check for clean
            if clean and (i != iref) and (i > 0):
                rmfiles.append(fvtk)
            # Remove files
            if newbatch:
                # Loop through files to delete for this batch
                for fvtk in rmfiles:
                    print(f"  Removing '{fvtk}'")
                    self.remove_file(fvtk)
                # Reset list of files to delete
                rmfiles = []
            # Update
            n += 1
            # Check for exit flag
            if (nmax is not None) and (n >= nmax):
                break
        # Loop through files to delete that didn't line up with a batch
        for fvtk in rmfiles:
            print(f"  Removing '{fvtk}'")
            self.remove_file(fvtk)
        # Update metadata
        self.write_surfdata_meta(nsurf, db)

    @casecntl.run_rootdir
    def read_surfdata_meta(self, nsurf: int = 0) -> DataKit:
        r"""Read database of metadata for collected surface data

        :Call:
            >>> db = runner.read_surfdata_meta(nsurf=0)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *nsurf*: {``0``} | :class:`int`
                Surface index (0-based)
        :Outputs:
            *db*: :class:`cape.dkit.rdb.DataKit`
                DataKit of which batch each iteration is located in
        :Versions:
            * 2026-04-05 ``@ddalle``: v1.0
        """
        # Create file name
        fname = self._genr8_surfdata_metafile(nsurf)
        # Check for file
        if not os.path.isfile(fname):
            # Initialize datakit
            db = DataKit()
            # Save parameters
            db.save_col("nt", 0)
            db.save_col("i", np.zeros(0, dtype="i4"))
            db.save_col("batch", np.zeros(0, dtype="i4"))
            # Output
            return db
        # Otherwise read it
        return DataKit(fname)

    @casecntl.run_rootdir
    def write_surfdata_meta(self, nsurf: int, db: DataKit):
        r"""Write updated surface data collection metadata

        :Call:
            >>> db.write_surfdata_meta(nsurf, db)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *nsurf*: {``0``} | :class:`int`
                Surface index (0-based)
            *db*: :class:`cape.dkit.rdb.DataKit`
                DataKit of which batch each iteration is located in
        :Verions:
            * 2026-04-05 ``@ddalle``: v1.0
        """
        # Create file name
        fname = self._genr8_surfdata_metafile(nsurf)
        # Write it
        db.write(fname)

    @casecntl.run_rootdir
    def _write_surfdata(self, i: int, nsurf: int, batch: int):
        # Name of file to read; create if necessary
        fcdb = self._init_surfdata_batch(nsurf, batch)
        # Name of VTK file
        fvtk = self._genr8_surfdata_reffile(nsurf, i)
        # Check for file
        if not os.path.isfile(fvtk):
            self.log_verbose(f"File not found: {fvtk}")
            return
        if not os.path.isfile(fcdb):
            self.log_verbose(f"File not found: {fcdb}")
            raise FileNotFoundError(
                f"Surfdata collection file not found: {fcdb}")
        # Open batch file
        dat = capefile.CapeFile(fcdb, meta=True)
        # Read surface data
        surf = Umesh(fvtk)
        # Read small fields
        dat.read_record("nt")
        dat.read_record("nnode")
        dat.read_record("nq")
        # Get counts from batch file
        nt = dat["nt"] + 1
        nq = dat["nq"]
        nnode = dat["nnode"]
        # Check counts
        if surf.nq != nq:
            raise ValueError(f"In '{fvtk}', expected nq={nq}; got {surf.nq}")
        if surf.nnode != nnode:
            raise ValueError(
                f"In '{fvtk}', expected nnode={nnode}; got {surf.nnode}")
        # Open the batch file for editing
        with open(fcdb, 'r+b') as fp:
            # Go to *nt* position
            fp.seek(dat.pos['nt'])
            # Read record type and size
            fromfile_lb4_i(fp, 2)
            # Read length of name
            l1, = fromfile_lb4_i(fp, 1)
            # Skip name
            fp.read(l1)
            # Now overwrite number of time steps in file
            tofile_lb4_i(fp, nt)
            # Now go to end of file
            fp.seek(dat.pos['q'])
            # Read record type
            rtype_code, = fromfile_lb4_i(fp, 1)
            # Parse record type details
            rt = capefile.RecordType(rtype_code)
            # Calculate length (in bytes)
            l2 = 2 ** (rt.element_bits - 3)
            l3 = 33 + nt*nq*nnode*l2
            # Position for updated record size
            pos3 = fp.tell()
            fromfile_lb8_i(fp, 1)
            # Get name
            l4, = fromfile_lb4_i(fp, 1)
            fp.read(l4)
            # Skip dimensions
            nd, = fromfile_lb4_i(fp, 1)
            # Position for updated size
            pos4 = fp.tell()
            # Read node count and q count
            nt2, nn2, nq2 = fromfile_lb8_i(fp, 3)
            # Check
            if nt2 != nt - 1:
                raise ValueError(
                    f"In {fcdb}; report time step count {nt - 1} "
                    f"does not match q.shape ({nt2})")
            if nn2 != nnode:
                raise ValueError(
                    f"In {fvtk}: expected {nnode} nodes; got {nn2}")
            if nq2 != nq:
                raise ValueError(
                    f"In {fvtk}: expected {nq} states; got {nq2}")
            # Write size
            fp.seek(pos3)
            tofile_lb8_i(fp, l3)
            # Write updated shape of *q*
            fp.seek(pos4)
            tofile_lb4_i(fp, nt)
            # Go to end of file to write new data
            fp.seek(0, 2)
            # Write state
            if l2 == 8:
                # Write as double-precision data
                tofile_lb8_f(fp, surf.q.astype("f8"))
            else:
                # Write as single-precision data
                tofile_lb4_f(fp, surf.q.astype("f4"))

    @casecntl.run_rootdir
    def _read_surfdata_ref(self, nsurf: int, i: Optional[int] = None) -> Umesh:
        # Name of reference file
        fname = self._genr8_surfdata_reffile(nsurf, i)
        # Read it
        return Umesh(fname)

    def _genr8_surfdata_reffile(
            self,
            nsurf: int,
            i: Optional[int] = None) -> str:
        # Check for explicit iteration
        if i is None:
            # Try iteration zero
            vtkfile = os.path.join(
                "surface", f"surf{nsurf:03d}.Cart.{0:09d}.vtk")
            # Check for it
            if os.path.isfile(vtkfile):
                return vtkfile
            # Get any current VTK files
            vtkpat = self._genr8_surfdata_regex(nsurf)
            vtkfiles = sorted(self.search_regex(vtkpat))
            # Get integers from these file names
            i = int(vtkfiles[0].rsplit('.', 2)[-2])
        # Name of file
        return os.path.join("surface", f"surf{nsurf:03d}.Cart.{i:09d}.vtk")

    def _init_surfdata_batch(self, nsurf: int, batch: int) -> str:
        # Name of file
        fname = self._genr8_surfdata_batchfile(nsurf, batch)
        # Check if file exists
        if not os.path.isfile(fname):
            # Read reference VTK file
            surf = self._read_surfdata_ref(nsurf)
            # Get number of states and nodes
            nnode = surf.nnode
            nq = surf.nq
            # Create a DataKit
            db = DataKit()
            # Initialize
            db.save_col("nt", 0)
            db.save_col("nnode", nnode)
            db.save_col("nq", nq)
            db.save_col("q", np.zeros((0, nnode, nq), dtype="f4"))
            # Get/create CAPEDB file interface
            cdb = db.genr8_cdb()
            # Set special data type (long record) for *q*
            rtyp = capefile.RecordType.from_value(db["q"], "q")
            rt = rtyp.rt | capefile.RT_XLONGREC
            cdb.rt["q"] = rt
            # Write it
            cdb.write(fname)
        # Return name of file
        return fname

    def _genr8_surfdata_regex(self, nsurf: int = 0) -> str:
        return os.path.join(
            "surface",
            f"surf{nsurf:03d}\\.Cart\\.[0-9]+\\.vtk")

    def _genr8_surfdata_metafile(self, nsurf: int = 0) -> str:
        return os.path.join("surface", f"surf{nsurf:03d}.Cart.cdb")

    def _genr8_surfdata_batchfile(self, nsurf: int, batch: int) -> str:
        return os.path.join(
            "surface",
            f"surf{nsurf:03d}.Cart.batch{batch:04d}.cdb")

   # --- File manipulation ---
    # Prepare any input files as needed
    def prepare_files(self, j: int):
        r"""Prepare files for phase *j*, LAVA-specific

        :Call:
            >>> runner.prepare_files(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase index
        :Versions:
            * 2025-07-28 ``@ddalle``: v1.0
        """
        # Create post-processing and log folder to ensure permissions
        self.mkdir("isosurface")
        self.mkdir("monitor")
        self.mkdir("point_probe")
        self.mkdir("restart")
        self.mkdir("surface")
        self.mkdir("volume")
        # Automatically configure restart settings
        self.prepare_restart(j)

    # Set restart option if appropriate
    def prepare_restart(self, j: int):
        r"""Automatically configure a case to restart if appropriate

        :Call:
            >>> runner.prepare_restart(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2025-08-14 ``@ddalle``: v1.0
        """
        # Get settings
        rc = self.read_case_json()
        # Get solver type
        solver = rc.get_LAVASolver()
        # Create function name
        funcname = f"prepare_restart_{solver}"
        # Get function, if any
        func = getattr(self, funcname)
        # Call it if possible
        if callable(func):
            func(j)

    # Set restart option for Cart
    def prepare_restart_cartesian(self, j: int):
        r"""Automatically configure LAVA-Cartesian for restart

        :Call:
            >>> runner.prepare_restart_cartesian(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2025-08-14 ``@ddalle``: v1.0
        """
        # Read input file
        opts = self.read_runinputs(j)
        # Search for a restart file
        restartfile = self.get_restart_file()
        # Set it
        opts.set_opt("solver defaults", "restart.file", restartfile)
        # Remove it if not a restart
        if restartfile is None:
            # Remove restart file if previously set
            opts["solver defaults"].pop("restart", None)
        # Write
        opts.write()

    # Link best Output files
    @casecntl.run_rootdir
    def link_viz(self):
        r"""Link the most recent visualization files

        :Call:
            >>> runner.link_viz()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Versions:
            * 2025-07-25 ``@jmeeroff``: v1.0
            * 2025-07-28 ``@ddalle``: v1.1; bug for subfolder links
        """
        # Visualization subfolders
        vizdirs = ('volume', 'isosurface', 'surface')
        # Call the archivist for grouping
        a = self.get_archivist()
        # Loop through viz directories
        for vizdir in vizdirs:
            # Go to that viz folder
            os.chdir(self.root_dir)
            os.chdir(vizdir)
            # Form search pattern from within that folder
            pat = os.path.join(vizdir, r"(.+)\.[0-9]+\.([a-z0-9]+)")
            # Get groups using archivist
            vgrp = a.search_regex(pat)
            # Loop through keys:
            for fnstr in vgrp.keys():
                # Parse the filename to link to
                parse = re.findall(r"'(.*?)'", fnstr)
                # Append the output name and last file
                fname = f'{parse[0]}.{parse[1]}'
                # Don't include relative paths in link
                fsrc = os.path.basename(vgrp[fnstr][-1])
                # Link the files
                self.link_file(fsrc, fname, f=True)

   # --- Search ---
    def get_restart_file(self, j: Optional[int] = None) -> Optional[str]:
        # Get search pattern
        pat = self.genr8_restart_regex()
        # Search
        mtch = self.match_regex(pat)
        # Return it if possible
        if mtch:
            return mtch.group()

    def genr8_restart_regex(self) -> str:
        r"""Return a regular expression that matches all restart files

        :Call:
            >>> pat = runner.genr8_restart_regex()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *pat*: :class:`str`
                Regular expression pattern
        :Versions:
            * 2025-08-14 ``@ddalle``: v1.0
        """
        return os.path.join("restart", "Cart_restart.([0-9]+).hdf5")

   # --- Input files ---
    # Read YAML inputs
    @casecntl.run_rootdir
    def read_runyaml(self, j: Optional[int] = None) -> RunYAMLFile:
        r"""Read case's LAVA-Curvilinear input file

        :Call:
            >>> yamlfile = runner.read_runyaml(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *yamlfile*: :class:`RunYAMLFile`
                LAVA YAML input file interface
        :Versions:
            * 2024-10-11 ``@ddalle``: v1.0
        """
        # Read ``case.json`` if necessary
        rc = self.read_case_json()
        # Process phase number
        if j is None and rc is not None:
            # Default to most recent phase number
            j = self.get_phase()
        # Get phase of namelist previously read
        yamlj = self.yamlfile_j
        # Check if already read
        if isinstance(self.yamlfile, RunYAMLFile):
            if yamlj == j and j is not None:
                # Return it!
                return self.yamlfile
        # Get name of file to read
        fbase = rc.get_lava_yamlfile()
        fname = cmdgen.infix_phase(fbase, j)
        # Read it
        self.yamlfile = RunYAMLFile(fname)
        # Return it
        return self.yamlfile

    # Read Cart inputs
    @casecntl.run_rootdir
    def read_runinputs(self, j: Optional[int] = None) -> CartInputFile:
        r"""Read case's LAVA-Cartesian input file

        :Call:
            >>> yamlfile = runner.read_runinputs(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *yamlfile*: :class:`CartInputFile`
                LAVA YAML input file interface
        :Versions:
            * 2025-08-14 ``@ddalle``: v1.0
            * 2025-09-03 ``@ddalle``: v1.1; check for file
        """
        # Read ``case.json`` if necessary
        rc = self.read_case_json()
        # Process phase number
        if j is None and rc is not None:
            # Default to most recent phase number
            j = self.get_phase()
        # Get phase of namelist previously read
        runinpj = self.runinpfile_j
        # Check if already read
        if isinstance(self.runinpfile, CartInputFile):
            if runinpj == j and j is not None:
                # Return it!
                return self.runinpfile
        # Get name of file to read
        fname = cmdgen.infix_phase("run.inputs", j)
        # Check for file
        if os.path.isfile(fname):
            # Read it
            self.runinpfile = CartInputFile(fname)
        else:
            # Read from *cntl*
            cntl = self.read_cntl()
            return cntl.CartInputs
        # Return it
        return self.runinpfile

   # --- Special readers ---
    # Check if case is complete
    @casecntl.run_rootdir
    def check_complete(self) -> bool:
        r"""Check if a case is complete (DONE)

        In addition to the standard CAPE checks, this version checks
        residuals convergence udner certain conditions.

        :Call:
            >>> q = runner.check_complete()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Versions:
            * 2024-09-16 ``@sneuhoff``: v1.0
            * 2024-10-11 ``@ddalle``: v2.0; use parent method directly
        """
        # Read it, but only metadata
        db = self.read_data_iter(meta=True)
        # Check history
        if db.n == 0:
            return False
        # Read options
        rc = self.read_case_json()
        # Get solver type
        solver = rc.get_LAVASolver()
        # Check which command to generate
        if solver == "curvilinear":
            # Read YAML file
            yamlfile = self.read_runyaml()
            # Maximum iterations
            maxiters = yamlfile.get_lava_subopt(
                "nonlinearsolver", "iterations")
            if db.n >= maxiters:
                return True
            # Target convergence
            l2conv_target = yamlfile.get_lava_subopt(
                "nonlinearsolver", "l2conv")
            # Apply it
            if l2conv_target:
                # Check reported convergence
                return db.l2conv <= l2conv_target
            else:
                # No convergence test
                return False
        # Perform parent check
        q = casecntl.CaseRunner.check_complete(self)
        # Quit if not complete
        return q

    @casecntl.run_rootdir
    def read_data_iter(
            self,
            fname: str = ITER_FILE,
            meta: bool = False,
            force: bool = False) -> DataIterFile:
        r"""Read ``data.iter``, if present

        :Call:
            >>> db = runner.read_data_iter(fname, meta=False)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *fname*: {``"data.iter"``} | :class:`str`
                Name of file to read
            *meta*: {``True``} | ``False``
                Option to only read basic info such as last iter
            *force*: ``True`` | {``False``}
                Reread even if cached
        :Versions:
            * 2024-08-02 ``@sneuhoff``; v1.0
            * 2024-10-11 ``@ddalle``: v2.0
            * 2025-08-14 ``@ddalle``: v2.1; cache, *force*
        """
        # Check cache
        if (not force) and (self.data_iter is not None):
            return self.data_iter
        # Default file names for convenience
        fname = fname if os.path.isfile(fname) else ITER_FILE
        fname = fname if os.path.isfile(fname) else ITER_FILE_CART
        # Check if file exists
        if os.path.isfile(fname):
            # Read existing file
            dat = DataIterFile(fname, meta=meta)
            # Cache it (not *meta*)
            if not meta:
                self.data_iter = dat
            # Output
            return dat
        else:
            # Empty instance
            return DataIterFile(None)


# Triangulate a mesh
def triangulate_mesh(mesh: Umesh, n: np.ndarray, p: np.ndarray):
    r"""Map a mesh onto a unique triangulation on a plane

    That is, each point on that plane will be in exactly one triangle or
    on the edge of two triangles or on a vertex.

    :Call:
        >>> triangulate_mesh(mesh, n, p)
    :Inputs:
        *mesh*: :class:`cape.gruvoc.umesh.Umesh`
            Mesh instance
        *n*: :class:`np.ndarray`\ [:class:`float`]
            Vector normal to the plane
        *p*: :class:`np.ndarray`\ [:class:`float`]
            Point in the plane
    """
    # Project the nodes to the cut plane
    x = mesh.project_to_plane(n, p)
    # Calcualte Delaunay triangulation
    tri = Delaunay(x[:, :2])
    # Reset the triangles
    mesh.tris = tri.simplices + 1
    mesh.ntri = mesh.tris.shape[0]
    # Resize CompIDs
    mesh.tri_ids = mesh.tri_ids[:mesh.ntri]
    # Remove no-longer-used nodes
    mesh.remove_unused_nodes()


# Link best viz files
def LinkViz():
    r"""Link the most recent viz files to fixed file names

    :Call:
        >>> LinkPLT()
    :Versions:
        * 2025-07-28 ``@jmeeroff``: v1.0
    """
    # Instantiate
    runner = CaseRunner()
    # Call link method
    runner.link_viz()
