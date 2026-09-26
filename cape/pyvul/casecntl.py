r"""
:mod:`cape.pyfun.casecntl`: FUN3D case control module
======================================================

This module contains the important function :func:`casecntl.run_fun3d`,
which actually runs ``nodet`` or ``nodet_mpi``, along with the utilities
that support it.

It also contains FUN3D-specific versions of some of the generic methods
from :mod:`cape.case`.  For instance the function :func:`GetCurrentIter`
determines how many FUN3D iterations have been run in the current
folder, which is obviously a solver-specific task.  It also contains the
function :func:`LinkPLT`, which creates links to fixed Tecplot file
names from the most recent output created by FUN3D.

All of the functions from :mod:`cape.case` are imported here.  Thus they
are available unless specifically overwritten by specific
:mod:`cape.pyfun` versions.

"""

# Standard library modules
import glob
import os
import re
import shutil
import time
from typing import Any, Optional, Tuple, Union

# Third-party modules
import numpy as np

# Local imports
from .. import fileutils
from . import cmdgen
from .inpfile import VulcanInpFile
from .options.runctlopts import RunControlOpts
from ..cfdx import casecntl
from ..errors import CapeFileError
from ..gruvoc import umesh


# Regular expression to find a line with an iteration
_regex_dict = {
    b"time": b"(?P<time>[1-9][0-9]*)",
    b"iter": b"(?P<iter>[1-9][0-9]*)",
}
# Combine them; different format for steady and time-accurate modes
REGEX_F3DOUT = re.compile(
    rb"\s*(%(time)s\s+)?%(iter)s\s{2,}[-0-9]" % _regex_dict)

# Help message for CLI
HELP_RUN_FUN3D = r"""
``run_fun3d.py``: Run FUN3D for one phase
================================================

This script determines the appropriate phase to run for an individual
case (e.g. if a restart is appropriate, etc.), sets that case up, and
runs it.

:Call:

    .. code-block:: console

        $ run_fun3d.py [OPTIONS]
        $ python -m cape.pyfun run [OPTIONS]

:Options:

    -h, --help
        Display this help message and quit

:Versions:
    * 2014-10-02 ``@ddalle``: v1.0 (pycart)
    * 2015-10-19 ``@ddalle``: v1.0
    * 2021-10-01 ``@ddalle``: v2.0; part of :mod:`case`
"""

# Maximum number of calls to run_phase()
NSTART_MAX = 80


# Function to complete final setup and call the appropriate FUN3D commands
def run_vulcan():
    r"""Setup and run the appropriate VUCLAN-CFD command

    :Call:
        >>> run_vulcan()
    """
    # Get a case reader
    runner = CaseRunner()
    # Run it
    return runner.run()


# Initialize class
class CaseRunner(casecntl.CaseRunner):
   # --- Class attributes ---
    # Additional attributes
    __slots__ = (
        "inp",
        "inp_j",
    )

    # Help message
    _help_msg = HELP_RUN_FUN3D

    # Names
    _modname = "pyvul"
    _progname = "vulcan"
    _logprefix = "run"

    # Specific classes
    _rc_cls = RunControlOpts
    # _resid_cls = CaseResid
    # _dex_cls = {
    #     "fm": CaseFM,
    #     "iterfm": CaseFM,
    #     "surfcp": CaseSurfCp,
    # }

   # --- Config ---
    def init_post(self):
        r"""Custom initialization for pyvul

        :Call:
            >>> runner.init_post()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        """
        self.inp = None
        self.inp_j = None

   # --- Main runner methods ---
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
            * 2016-04-13 ``@ddalle``: v1.0 (``RunPhase()``)
            * 2023-06-02 ``@ddalle``: v2.0
            * 2023-06-27 ``@ddalle``: v3.0, instance method
            * 2024-08-23 ``@ddalle``: v3.1; toward simple run_phase()
        """
        # Run mesh prep if indicated: intersect, verify, aflr3
        self.run_intersect_fun3d(j)
        self.run_verify_fun3d(j)
        self.run_aflr3_fun3d(j)
        # Run main solver
        self.run_vulcan(j)

    @casecntl.run_rootdir
    def run_vulcan(self, j: int):
        r"""Run ``nodet``, the main FUN3D executable

        :Call:
            >>> runner.run_nodet(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2024-08-23 ``@ddalle``: v1.0
            * 2024-04-07 ``@ddalle``: v1.1; fork `run_nodet_primal()`
        """
        # Working folder
        fdir = self.get_working_folder()
        # Enter working folder (if necessary)
        os.chdir(fdir)
        # Read settings
        rc = self.read_case_json()
        # Read namelist
        nml = self.read_namelist(j)
        # Get the project name
        fproj = self.get_project_rootname(j)
        # Get the last iteration number
        n = self.get_iter()
        n0 = 0 if n is None else n
        # Number of requested iters for the end of this phase
        nj = rc.get_PhaseIters(j)
        # Number of iterations to run this phase
        ni = rc.get_nIter(j)
        # Check for mesh-only phase
        if nj is None or ni is None or ni <= 0 or nj < 0:
            # Name of next phase
            fproj_adapt = self.get_project_rootname(j+1)
            # AFLR3 output format
            fmt = nml.GetGridFormat()
            # Check for renamed file
            if fproj_adapt != fproj:
                # Copy mesh
                self.link_file(f"{fproj}.{fmt}", f"{fproj_adapt}.{fmt}")
            # Make sure *n* is not ``None``
            if n is None:
                n = 0
            # Exit appropriately
            if rc.get_Dual():
                os.chdir('..')
            # Create an output file to make phase number programs work
            self.finalize_stdoutfile(j)
            return
        # Prepare for restart if that's appropriate
        self.set_restart_read()
        # Prepare for adapt
        self.prep_adapt(j)
        # Run primal solver
        self.run_nodet_primal(j)
        # Get new iteration number
        n1 = self.get_iter()
        n1 = 0 if (n1 is None) else n1
        # Go back up a folder if we're in the "Flow" folder
        os.chdir(self.root_dir)
        # Check current iteration/phase count
        jmax = self.get_last_phase()
        nmax = self.get_last_iter()
        if (j >= jmax) and (n0 >= nmax):
            return
        # Check for adaptive solves
        if n1 < nj:
            return
        # Check for adjoint solver
        if rc.get_Dual() and rc.get_DualPhase(j):
            # Copy the correct namelist
            os.chdir(fdir)
            # Copy the correct one into place
            self.link_file(f'fun3d.dual.{j:02d}.nml' 'fun3d.nml', f=True)
            # Enter the 'Adjoint/' folder
            os.chdir('..')
            os.chdir('Adjoint')
            # Create the command to calculate the adjoint
            cmdi = cmdgen.dual(rc, i=j, rad=False, adapt=False)
            # Run the adjoint analysis
            self.callf(cmdi, f='dual.out')
            # Create the command to adapt
            cmdi = cmdgen.dual(rc, i=j, adapt=True)
            # Estimate error and adapt
            self.callf(cmdi, f='dual.out')
            # Rename output file after completing that command
            os.rename('dual.out', 'dual.%02i.out' % j)
            # Return
            os.chdir('..')
        elif rc.get_Adaptive() and rc.get_AdaptPhase(j):
            # Check if this is a weird mixed case with Dual and Adaptive
            os.chdir(fdir)
            # Check the adapataion method
            self.run_nodet_adapt(j)
            # Run refine translate
            self.run_refine_translate(j)
            # Run refine loop
            self.run_refine_loop(j)
            # Run post adapt procedures
            self.run_post_adapt(j)

    # Run ``nodet``
    def run_nodet_primal(self, j: int):
        r"""Run ``nodet`` (the primal solver)

        :Call:
            >>> runner.run_nodet_primal(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2025-04-07 ``@ddalle``: v1.0
        """
        # Check recently run phase
        jprev = self.get_phase_recent()
        # Get the last iteration number
        n = self.get_iter()
        n0 = 0 if n is None else n
        # Read case settings
        rc = self.read_case_json()
        # Number of requested iters for the end of this phase
        nj = rc.get_PhaseIters(j)
        # Number of iterations to run ``nodet`` for this phase
        nrun = rc.get_nIter(j)
        # Check if run is necessary
        if (not nrun) or (jprev == j and n0 >= nj):
            # Created "run.{j}.{n}
            self.finalize_stdoutfile(j)
            # Exit
            return
        # Get the `nodet` or `nodet_mpi` command
        cmdi = cmdgen.nodet(rc, j=j)
        # STDOUT/STDERR file names
        stdout = self.get_stdout_filename()
        stderr = self.get_stderr_filename()
        # Call the command
        self.callf(cmdi, f=stdout, e=stderr)
        # Get new iteration number
        n1 = self.get_iter()
        n1 = 0 if (n1 is None) else n1
        # Check for NaNs found
        if len(glob.glob("nan_locations*.dat")):
            # Mark failure
            self.mark_failure("Found NaN location files")
            raise SystemError("Found NaN location files")
        # Check for lack of progress
        if n1 <= n0:
            # Mark failure
            self.mark_failure(f"No advance from iter {n0} in phase {j}")
            # Raise an exception for run()
            raise SystemError(
                f"Cycle of phase {j} did not advance iteration count.")
        # Rename "fun3d.out"
        self.finalize_stdoutfile(j)

   # --- File manipulation ---
    # Rename/move files prior to running phase
    def prepare_files(self, j: int):
        r"""Prepare file names appropriate to run phase *i* of FUN3D

        :Call:
            >>> runner.prepare_files(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2016-04-14 ``@ddalle``: v1.0
            * 2023-07-06 ``@ddalle``: v1.1; instance method
        """
        # Read settings
        rc = self.read_case_json()
        # Delete any input file (primary namelist)
        if os.path.isfile('vulcan.inp') or os.path.islink('vulcan.inp'):
            os.remove('vulcan.inp')
        # Create the correct namelist
        os.symlink('vulcan.%02i.nml' % j, 'vulcan.nml')

    # Process the STDOUT file
    def finalize_stdoutfile(self, j: int):
        r"""Move the ``fun3d.out`` file to ``run.{j}.{n}``

        :Call:
            >>> runner.finalize_stdoutfile(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2025-04-07 ``@ddalle``: v1.0
        """
        # Get the last iteration number
        nc = self.get_iter_completed()
        na = self.get_iter_restart_active()
        n = nc + na
        # Get working folder
        fdir = self.get_working_folder_()
        # STDOUT file
        fout = os.path.join(fdir, self.get_stdout_filename())
        # History remains in present folder
        fhist = f"{self._logprefix}.{j:02d}.{n}"
        # Assuming that worked, move the temp output file.
        if os.path.isfile(fout):
            # Check if it's valid
            if not os.path.isfile(fhist):
                # Move the file
                os.rename(fout, fhist)
        else:
            # Create an empty file
            fileutils.touch(fhist)

    # Clean up immediately after running
    def finalize_files(self, j: int):
        r"""Clean up files after running one cycle of phase *j*

        :Call:
            >>> runner.finalize_files(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        """
        pass

    # Function to set the most recent file as restart file.
    def set_restart_read(self, n: Optional[int] = None):
        r"""Set a given check file as the restart point

        :Call:
            >>> runner.set_restart_read(n=None)
        :Inputs:
            *rc*: :class:`RunControlOpts`
                Run control options
            *n*: {``None``} | :class:`int`
                Restart iteration number, defaults to latest available
        """
        pass

   # --- Case options ---
    # Get project root name
    def get_project_rootname(self, j: Optional[int] = None) -> str:
        r"""Read namelist and return project namelist

        :Call:
            >>> rname = runner.get_project_rootname(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *rname*: :class:`str`
                Project rootname
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0
            * 2023-07-05 ``@ddalle``: v1.1; instance method
        """
        # Read a namelist
        nml = self.read_namelist(j)
        # Read the project root name
        return nml.GetRootname()

    # Get project root name but "pyfun", not "pyfun02"
    def get_project_baserootname(self) -> str:
        r"""Read namelist and return base project name w/o adapt counter

        This would be ``"pyfun"`` instead of ``"pyfun03"``, for example.

        :Call:
            >>> rname = runner.get_project_baserootname()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *rname*: :class:`str`
                Project rootname
        :Versions:
            * 2024-03-22 ``@ddalle``: v1.0
        """
        # Read the options
        rc = self.read_case_json()
        # Get the project root name
        proj = self.get_namelist_opt('project', 'project_rootname')
        # Strip suffix
        if rc.get_Dual() or rc.get_Adaptive():
            # Strip adaptive section
            proj = proj[:-2]
        # Output
        return proj

    # Get generic option from namelist
    def get_namelist_opt(
            self, sec: str, opt: str,
            j: Optional[int] = None,
            i=None, vdef=None) -> Any:
        r"""Get option from current ``fun3d.nml``

        :Call:
            >>> v = runner.get_namelist_opt(sec, opt)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *sec*: :class:`str`
                Name of namelist section
            *opt*: :class:`str`
                Option name
            *j*: {``None``} | :class:`int`
                Phase number
            *i*: {``None``} | :class:`int` | :class:`slice` | ``tuple``
                Index or indices of *val* to return ``nml[sec][opt]``
            *vdef*: {``None``} | :class:`object`
                Default value if *opt* not present in ``nml[sec]``
        :Outputs:
            *v*: :class:`object`
                Option value from ``fun3d.nml``
        :Versions:
            * 2025-09-25 ``@ddalle``: v1.0
        """
        # Need the namelist to figure out planes, etc.
        nml = self.read_namelist(j=j)
        # Get the option
        return nml.get_opt(sec, opt, j=i, vdef=vdef)

   # --- Special readers ---
    # Read namelist
    @casecntl.run_rootdir
    def read_inp(self, j: Optional[int] = None) -> VulcanInpFile:
        r"""Read case namelist file

        :Call:
            >>> inp = read_inp.read_namelist(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *inp*: :class:`cape.pyvul.inpfile.VulcanInpFile`
                VULCAN input file interface
        """
        # Read ``case.json`` if necessary
        rc = self.read_case_json()
        # Process phase number
        if j is None and rc is not None:
            # Default to most recent phase number
            j = self.get_phase_next()
        # Get phase of namelist previously read
        inpj = self.inp_j
        # Check if already read
        if isinstance(self.inp, VulcanInpFile) and inpj == j and j is not None:
            # Return it!
            return self.inp
        # Check for folder with no working ``case.json``
        if rc is None:
            # Check for simplest namelist file
            if os.path.isfile('vulcan.inp'):
                # Read the currently linked namelist.
                inp = VulcanInpFile('vulcan.inp')
            else:
                # Look for namelist files
                fglob = glob.glob('vulcan.??.nml')
                # Sort it
                fglob.sort()
                # Read one of them.
                inp = VulcanInpFile(fglob[-1])
            return inp
        # Get the specified namelist
        inp = VulcanInpFile('vulcan.%02i.nml' % j)
        # Cache it
        self.inp = inp
        self.inp_j = j
        # Output
        return inp

   # --- DataBook ---

   # --- File search ---
    # Function to get restart file
    def get_restart_file(self, j: Optional[int] = None) -> str:
        r"""Get the most recent ``.flow`` file for phase *j*

        :Call:
            >>> restartfile = runner.get_restart_file(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *restartfile*: :class:`str`
                Name of restart file, ending with ``.flow``
        """
        # Project name
        fproj = self.get_project_rootname(j)
        # Use the project name with ".flow"
        return f"{fproj}.flow"

    # Function to find grid file
    def get_grid_file(
            self,
            j: Optional[int] = None,
            check: bool = False) -> Optional[str]:
        pass

    # Get mesh format
    def get_grid_format(self, j: Optional[int] = None) -> str:
        r"""Get the grid format option in use for this case

        :Call:
            >>> grid_format = runner.get_grid_format(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase index (or current)
        :Outputs:
            *grid_format*: :class:`str`
                Grid format, ``"fast"``, ``"vgrid"``, ``"aflr3"``
        :Versions:
            * 2025-04-04 ``@ddalle``: v1.0
        """
        # Read namelist
        nml = self.read_namelist(j=j)
        # Get option
        grid_format = nml.get_opt("raw_grid", "grid_format", vdef="vgrid")
        # Lower-case
        return grid_format.lower()

    # Get mesh file extension
    def get_grid_extension(self, j: Optional[int] = None) -> str:
        r"""Get the file extension for the selected grid format

        File extensions taken from the FUN3D manual:

        ===============  ==================  ===============
        Format           Grid files          BC File
        ===============  ==================  ===============
        ``"aflr3"``      ``.ugrid``          ``.mapbc``
        ``"fast"``       ``.fgrid``          ``.mapbc``
        ``"fieldview"``  ``.fvgrid_fmt``     ``.mapbc``
        ``"fieldview"``  ``.fvgrid_unf``     ``.mapbc``
        ``"vgrid"``      ``.cogsg, .bc``     ``.mapbc``
        ``"felisa"``     ``.gri, .fro``      ``.bco``
        ===============  ==================  ===============

        :Call:
            >>> ext = runner.get_grid_extension(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase index (or current)
        :Outputs:
            *ext*: :class:`str`
                Grid file extension, ``"ugrid"``, ``"fgrid"``, etc.
        :Versions:
            * 2025-04-04 ``@ddalle``: v1.0
        """
        # Get option for grid format
        grid_format = self.get_grid_format(j)
        # Filter extension
        if grid_format == "aflr3":
            return "ugrid"
        elif grid_format == "fast":
            return "fgrid"
        elif grid_format == "vgrid":
            return "cogsg"
        else:
            return grid_format

    # Get mesh file extension
    def get_bc_extension(self, j: Optional[int] = None) -> str:
        r"""Get the file extension for the boundary condition files

        File extensions taken from the FUN3D manual:

        ===============  ==================  ===============
        Format           Grid files          BC File
        ===============  ==================  ===============
        ``"aflr3"``      ``.ugrid``          ``.mapbc``
        ``"fast"``       ``.fgrid``          ``.mapbc``
        ``"fieldview"``  ``.fvgrid_fmt``     ``.mapbc``
        ``"fieldview"``  ``.fvgrid_unf``     ``.mapbc``
        ``"vgrid"``      ``.cogsg, .bc``     ``.mapbc``
        ``"felisa"``     ``.gri, .fro``      ``.bco``
        ===============  ==================  ===============

        :Call:
            >>> ext = runner.get_grid_extension(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase index (or current)
        :Outputs:
            *ext*: :class:`str`
                Grid file extension, ``"mapbc"``, ``".bco"``
        :Versions:
            * 2025-04-04 ``@ddalle``: v1.0
            * 2025-05-16 ``@ddalle``: v1.1; typo: ma{bp->pb}c
        """
        # Get option for grid format
        grid_format = self.get_grid_format()
        # Filter extension
        if grid_format == "felisa":
            return "bco"
        else:
            return "mapbc"

   # --- Status ---

   # --- Conditions ---



