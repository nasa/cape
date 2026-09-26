r"""
:mod:`cape.pyvul.casecntl`: VULCAN-CFD case control module
============================================================

This module contains the important function :func:`run_vulcan`, which
actually runs the ``vulcan`` executable, along with the utilities that
support it.

It also contains VULCAN-specific versions of some of the generic
methods from :mod:`cape.cfdx.casecntl`. For instance,
:func:`get_iter_active` determines how many iterations appear in the
current screen output file, which is a solver-specific task.

:Call:

    .. code-block:: console

        $ run_vulcan.py [OPTIONS]
        $ python -m cape.pyvul run [OPTIONS]

:Options:

    -h, --help
        Display this help message and quit

:Versions:
    * 2014-10-02 ``@ddalle``: v1.0 (pycart)
    * 2015-10-19 ``@ddalle``: v1.0
    * 2021-10-01 ``@ddalle``: v2.0; part of :mod:`case`
"""

# Standard library modules
import glob
import os
import re
from typing import Optional

# Local imports
from .. import fileutils
from . import cmdgen
from .inpfile import VulcanInpFile
from .options.runctlopts import RunControlOpts
from ..cfdx import casecntl


# Regular expression to find a line in the VULCAN iteration history
REGEX_VULOUT = re.compile(rb"^\s*(?P<iter>[1-9][0-9]*)\s{2,}[-0-9.]")

# Help message for CLI
HELP_RUN_VULCAN = r"""
``run_vulcan.py``: Run VULCAN-CFD for one phase
=====================================================

This script determines the appropriate phase to run for an individual
case (e.g. if a restart is appropriate, etc.), sets that case up, and
runs it.

:Call:

    .. code-block:: console

        $ run_vulcan.py [OPTIONS]
        $ python -m cape.pyvul run [OPTIONS]

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


# Function to complete final setup and call the appropriate commands
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
    _help_msg = HELP_RUN_VULCAN

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
        self.run_intersect(j)
        self.run_verify(j)
        self.run_aflr3(j)
        # Run main solver
        self.run_vulcan(j)

    @casecntl.run_rootdir
    def run_vulcan(self, j: int):
        r"""Run ``vulcan``, the main VULCAN-CFD executable

        :Call:
            >>> runner.run_vulcan(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2024-08-23 ``@ddalle``: v1.0
            * 2026-09-26 ``@ddalle``: v2.0; native VULCAN case loop
        """
        # Working folder
        fdir = self.get_working_folder()
        # Enter working folder (if necessary)
        os.chdir(fdir)
        # Read settings
        rc = self.read_case_json()
        # Check recently run phase
        jprev = self.get_phase_recent()
        # Get the last iteration number
        n = self.get_iter()
        n0 = 0 if n is None else n
        # Number of requested iters for the end of this phase
        nj = rc.get_PhaseIters(j)
        # Number of iterations to run this phase
        ni = rc.get_nIter(j)
        # Check for mesh-only phase or completed phase
        if (nj is None) or (ni is None) or (ni <= 0) or (nj < 0) or (
                (jprev == j) and (n0 >= nj)):
            # Create "run.{j}.{n}" to make phase number programs work
            if not self.dry_run:
                self.finalize_stdoutfile(j)
            return
        # Get the ``vulcan`` command
        cmdi = cmdgen.vulcan(rc, j=j)
        # STDOUT/STDERR file names
        stdout = self.get_stdout_filename()
        stderr = self.get_stderr_filename()
        # Call the command (only prints in dry-run mode)
        self.callf(cmdi, f=stdout, e=stderr)
        # Exit in dry-run mode (no output files to post-process)
        if self.dry_run:
            return
        # Get new iteration number
        n1 = self.get_iter()
        n1 = 0 if (n1 is None) else n1
        # Check for lack of progress
        if n1 <= n0:
            # Mark failure
            self.mark_failure(f"No advance from iter {n0} in phase {j}")
            # Raise an exception for run()
            raise SystemError(
                f"Cycle of phase {j} did not advance iteration count.")
        # Rename the STDOUT file to "run.{j}.{n}"
        self.finalize_stdoutfile(j)

   # --- File manipulation ---
    # Rename/move files prior to running phase
    def prepare_files(self, j: int):
        r"""Prepare file names appropriate to run phase *j* of VULCAN

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
            * 2026-09-26 ``@ddalle``: v1.2; link ``vulcan.inp``
        """
        # Delete any existing input file link
        if os.path.isfile('vulcan.inp') or os.path.islink('vulcan.inp'):
            os.remove('vulcan.inp')
        # Link the correct phase input file
        os.symlink('vulcan.%02i.inp' % j, 'vulcan.inp')

    # Process the STDOUT file
    def finalize_stdoutfile(self, j: int):
        r"""Move the ``vulcan.out`` file to ``run.{j}.{n}``

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
        n = self.get_iter()
        n = 0 if n is None else n
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
        r"""Get the project root name from ``case.json``

        VULCAN has no rootname setting in the input file, so this
        comes from the CAPE ``"ProjectRootname"`` run-control option,
        which defaults to ``"vulcan"``.

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
            * 2026-09-26 ``@ddalle``: v2.0; RunControl-based
        """
        # Read settings
        rc = self.read_case_json()
        # Check for usable settings
        if rc is None:
            return "vulcan"
        # Get the name
        name = rc.get_ProjectRootname(j)
        # Output
        return "vulcan" if name is None else name

    # Get project root name without any suffix
    def get_project_baserootname(self) -> str:
        r"""Return the project rootname without any adaptation suffix

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
            * 2026-09-26 ``@ddalle``: v1.1; no adaptation numbers
        """
        # No adaptation-number suffixes in pyvul
        return self.get_project_rootname()

    # Get generic option from input file
    def get_inp_opt(
            self, opt: str,
            j: Optional[int] = None,
            vdef=None) -> any:
        r"""Get option from current ``vulcan.inp``

        :Call:
            >>> v = runner.get_inp_opt(opt)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *opt*: :class:`str`
                Option name
            *j*: {``None``} | :class:`int`
                Phase number
            *vdef*: {``None``} | :class:`object`
                Default value if *opt* not present in input file
        :Outputs:
            *v*: :class:`object`
                Option value from ``vulcan.inp``
        :Versions:
            * 2026-09-26 ``@ddalle``: v1.0
        """
        # Need the input file for this
        inp = self.read_inp(j=j)
        # Get the option
        return inp.get_opt(opt, vdef=vdef)

   # --- Special readers ---
    # Read input file
    @casecntl.run_rootdir
    def read_inp(self, j: Optional[int] = None) -> VulcanInpFile:
        r"""Read case input file

        :Call:
            >>> inp = runner.read_inp(j=None)
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
        # Get phase of input file previously read
        inpj = self.inp_j
        # Check if already read
        if isinstance(self.inp, VulcanInpFile) and inpj == j and j is not None:
            # Return it!
            return self.inp
        # Check for folder with no working ``case.json``
        if rc is None:
            # Check for simplest input file
            if os.path.isfile('vulcan.inp'):
                # Read the currently linked input file.
                inp = VulcanInpFile('vulcan.inp')
            else:
                # Look for input files
                fglob = glob.glob('vulcan.??.inp')
                # Sort it
                fglob.sort()
                # Read one of them.
                inp = VulcanInpFile(fglob[-1])
            return inp
        # Get the specified input file
        inp = VulcanInpFile('vulcan.%02i.inp' % j)
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
        r"""Get the grid format in use for this case

        :Call:
            >>> grid_format = runner.get_grid_format(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase index (or current)
        :Outputs:
            *grid_format*: :class:`str`
                Grid format, ``"aflr3"`` for unstructured, else
                ``"vgrid"``
        :Versions:
            * 2026-09-26 ``@ddalle``: v1.0; VULCAN-based
        """
        # Read input file
        inp = self.read_inp(j=j)
        # Unstructured grids use AFLR3-format files
        if inp.get_opt("UNS GRID") is not None:
            return "aflr3"
        # Otherwise structured
        return "vgrid"

    # Get mesh file extension
    def get_grid_extension(self, j: Optional[int] = None) -> str:
        r"""Get the file extension for the grid format in use

        VULCAN reads AFLR3-format (``.ugrid``/``.b8.ugrid``) files for
        unstructured grids and GridPro ``.vgrid`` files otherwise.

        :Call:
            >>> ext = runner.get_grid_extension(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase index (or current)
        :Outputs:
            *ext*: :class:`str`
                Grid file extension, ``"b8.ugrid"`` or ``"vgrid"``
        :Versions:
            * 2026-09-26 ``@ddalle``: v1.0; VULCAN-based
        """
        # Get option for grid format
        grid_format = self.get_grid_format(j)
        # Filter extension
        if grid_format == "aflr3":
            return "ugrid"
        return "vgrid"

    # Get mesh file extension
    def get_bc_extension(self, j: Optional[int] = None) -> str:
        r"""Get the file extension for the boundary condition files

        CAPE uses ``.mapbc`` files to track boundary condition names
        for VULCAN grids.

        :Call:
            >>> ext = runner.get_bc_extension(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase index (or current)
        :Outputs:
            *ext*: :class:`str`
                Boundary condition file extension, ``"mapbc"``
        :Versions:
            * 2026-09-26 ``@ddalle``: v1.0; VULCAN-based
        """
        return "mapbc"

   # --- Status ---
    # Get iterations run since last completed phase run
    @casecntl.run_rootdir
    def get_iter_active(self) -> int:
        r"""Detect the latest iteration in the active screen output file

        VULCAN writes its iteration history table to the screen output
        file, and the iteration counter continues across restarts.

        :Call:
            >>> n = runner.get_iter_active()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *n*: :class:`int`
                Latest iteration number in ``vulcan.out``
        :Versions:
            * 2026-09-26 ``@ddalle``: v1.0
        """
        # Name of the active STDOUT file
        fdir = self.get_working_folder()
        fout = os.path.join(fdir, self.get_stdout_filename())
        # Check for the file
        if not os.path.isfile(fout):
            return 0
        # Scan for the last iteration-history line
        n = 0
        with open(fout, 'rb') as f:
            for line in f:
                mtch = REGEX_VULOUT.match(line)
                if mtch:
                    n = int(mtch.group('iter'))
        # Output
        return n

    # Calculate most recent iteration
    def getx_iter(self, f: bool = False) -> int:
        r"""Calculate most recent iteration

        The VULCAN iteration counter in the screen output is absolute
        (it continues through restarts), so it is compared with the
        count recorded in the last completed ``run.{j}.{n}`` file.

        :Call:
            >>> n = runner.getx_iter(f=False)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *f*: ``True`` | {``False``}
                Force reread; ignore cache
        :Outputs:
            *n*: :class:`int`
                Iteration number
        :Versions:
            * 2026-09-26 ``@ddalle``: v1.0
        """
        # Latest from the active output file
        n = self.get_iter_active()
        # Latest from completed ``run.{j}.{n}`` files
        nc = self.get_iter_completed()
        # Cache and output
        self.n = max(n, nc)
        return self.n

   # --- Conditions ---



