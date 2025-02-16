r"""
:mod:`cape.pyfun.case`: FUN3D case control module
==================================================

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
from typing import Optional

# Third-party modules
import numpy as np

# Local imports
from . import cmdrun
from . import cmdgen
from . import pltfile
from .. import fileutils
from .databook import CaseResid
from .options.runctlopts import RunControlOpts
from .namelist import Namelist
from ..cfdx import casecntl
from ..filecntl.tecfile import convert_szplt


# Regular expression to find a line with an iteration
_regex_dict = {
    b"time": b"(?P<time>[1-9][0-9]*)",
    b"iter": b"(?P<iter>[1-9][0-9]*)",
}
# Combine them; different format for steady and time-accurate modes
REGEX_F3DOUT = re.compile(
    rb"\s*%(time)s?\s+%(iter)s\s{2,}[-0-9]" % _regex_dict)

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
def run_fun3d():
    r"""Setup and run the appropriate FUN3D command

    :Call:
        >>> run_fun3d()
    :Versions:
        * 2015-10-19 ``@ddalle``: v1.0
        * 2016-04-05 ``@ddalle``: v1.1; add AFLR3 hook
        * 2023-07-06 ``@ddalle``: v2.0; use ``CaseRunner``
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
        "nml",
        "nml_j",
    )

    # Help message
    _help_msg = HELP_RUN_FUN3D

    # Names
    _modname = "pyfun"
    _progname = "fun3d"
    _logprefix = "run"

    # Specific classes
    _rc_cls = RunControlOpts

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
        self.nml = None
        self.nml_j = None

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
        self.run_nodet(j)

    @casecntl.run_rootdir
    def run_nodet(self, j: int):
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
            self.touch_file("run.%02i.%i" % (j, n))
            return
        # Prepare for restart if that's appropriate
        self.set_restart_iter()
        # Prepare for adapt
        self.prep_adapt(j)
        # Get *n* but ``0`` instead of ``None``
        n0 = 0 if (n is None) else n
        # Count number of times this phase has been run previously.
        nprev = len(glob.glob('run.%02i.*' % j))
        # Check if the primal solution has already been run
        if nprev == 0 or n0 < nj:
            # Get the `nodet` or `nodet_mpi` command
            cmdi = cmdgen.nodet(rc, j=j)
            # Call the command
            self.callf(cmdi, f='fun3d.out', e='fun3d.err')
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
            # Check for NaNs found
            if len(glob.glob("nan_locations*.dat")):
                # Mark failure
                self.mark_failure("Found NaN location files")
                raise SystemError("Found NaN location files")
        else:
            # No new iterations
            n1 = n
        # Go back up a folder if we're in the "Flow" folder
        if rc.get_Dual():
            os.chdir('..')
        # Check current iteration count.
        if (j >= rc.get_PhaseSequence(-1)) and (n0 >= rc.get_LastIter()):
            return
        # Check for adaptive solves
        if n1 < nj:
            return
        # Check for adjoint solver
        if rc.get_Dual() and rc.get_DualPhase(j):
            # Copy the correct namelist
            os.chdir('Flow')
            # Delete ``fun3d.nml`` if appropriate
            if os.path.isfile('fun3d.nml') or os.path.islink('fun3d.nml'):
                os.remove('fun3d.nml')
            # Copy the correct one into place
            os.symlink('fun3d.dual.%02i.nml' % j, 'fun3d.nml')
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
            if rc.get_Dual():
                os.chdir('Flow')
            # Check the adapataion method
            self.run_nodet_adapt(j)
            # Run refine translate
            self.run_refine_translate(j)
            # Run refine loop
            self.run_refine_loop(j)
            # Return home if appropriate
            if rc.get_Dual():
                os.chdir('..')

    # Prepare for adapt (with refine/three)
    def prep_adapt(self, j: int):
        r"""Prepare required nml options for 'refine/three' adapt

        :Call:
            >>> runner.prep_adapt(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2024-09-04 ``@aburkhea``: v1.0
        """
        rc = self.read_case_json()
        nml = self.read_namelist()
        adpt_opt = rc.get_AdaptPhase(j)
        # Only needed for "refine/three"
        if rc.get_AdaptMethod() != 'refine/three':
            return
        # Only overwrite given nml if the phase adapts
        if not adpt_opt:
            return
        # Required settings
        vov_req = {
            "export_to": 'solb',
            "primitive_variables": True,
            "x": False,
            "y": False,
            "z": False,
            "turb1": True,
            "turb2": True
        }
        # Overwrite volume output variables
        _ = nml.pop("volume_output_variables")
        # Save vol output options to nml
        nml.set_sec("volume_output_variables", vov_req)
        # Ensure volume freq is set
        globl = nml.get("global")
        # Set vol freq
        globl["volume_animation_freq"] = -1
        # Save section update
        nml.set_sec("global", globl)
        nml.write()

    # Run refine translate if needed
    def run_refine_translate(self, j: int):
        r"""Run refine transalte to create input meshb file for
        adaptation

        :Call:
            >>> runner.prepare_files(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2023-07-17 ``@jmeeroff``: v1.0; from ``run_phase``
        """
        # Read settings
        rc = self.read_case_json()
        # Check if adaptive
        if not (rc.get_Adaptive() and rc.get_AdaptPhase(j)):
            return
        # Check the adaption method
        if rc.get_AdaptMethod() != "refine/three":
            return
        # Check if meshb file already exists for this phase
        if os.path.isfile('pyfun%02i.meshb' % j):
            return
        # Get project name
        fproj = self.get_project_rootname(j)
        # TODO: determine grid format
        # Set command line default required args & kws
        rc.set_RefineTranslateOpt("input_grid", f'{fproj}.lb8.ugrid')
        rc.set_RefineTranslateOpt("output_grid", f'{fproj}.meshb')
        rc.set_RefineTranslateOpt("run", True)
        # Run the refine translate command
        cmdi = cmdgen.refine(rc, i=j, function="translate")
        # Call the command
        self.callf(cmdi, f="refine-translate.out")

    # Run refine distance if needed
    def run_refine_distance(self, j: int):
        r"""Run refine distance to create distances reqd for adaptation

        :Call:
            >>> runner.prepare_files(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2024-06-07 ``@aburkhea``: v1.0; from ``run_phase``
        """
        # Read settings
        rc = self.read_case_json()
        # Check if adaptive
        if not (rc.get_Adaptive() and rc.get_AdaptPhase(j)):
            return
        # Check the adaption method
        if rc.get_AdaptMethod() != "refine/three":
            return
        # Check if meshb file already exists for this phase
        if os.path.isfile('pyfun%02i-distance.solb' % j):
            return
        # Formulate kw inputs for command line
        # Get project name
        fproj = self.get_project_rootname(j)
        # Set command line default required args & kws
        rc.set_RefineDistanceOpt("input_grid", f'{fproj}.lb8.ugrid')
        rc.set_RefineDistanceOpt("dist_solb", f'{fproj}-distance.solb')
        rc.set_RefineDistanceOpt("mapbc", f'{fproj}.mapbc')
        rc.set_RefineDistanceOpt("run", True)
        # Run the refine distance command
        cmdi = cmdgen.refine(rc, i=j, function="distance")
        # Call the command
        cmdrun.callf(cmdi, f="refine-distance.out")

    # Run refine distance if needed
    def run_refine_loop(self, j: int):
        r"""Run refine loop to adapt grid to target complexity

        :Call:
            >>> runner.prepare_files(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2024-06-07 ``@aburkhea``: v1.0; from ``run_phase``
        """
        # Read settings
        rc = self.read_case_json()
        # Check if adaptive
        if not (rc.get_Adaptive() and rc.get_AdaptPhase(j)):
            return
        # Check the adaption method
        if rc.get_AdaptMethod() != "refine/three":
            return
        if os.path.isfile("refine-loop.%02i.out" % j):
            return
        # Get project name
        fproj = self.get_project_rootname(j)
        fproj1 = self.get_project_rootname(j+1)
        # Set command line default required args & kws
        rc.set_RefineOpt("input", f"{fproj}")
        rc.set_RefineOpt("output", f"{fproj1}")
        rc.set_RefineOpt("interpolant", "mach")
        rc.set_RefineOpt("mapbc", f'{fproj}.mapbc')
        rc.set_RefineOpt("run", True)
        # Run the refine loop command
        cmdi = cmdgen.refine(rc, i=j)
        # Call the command
        cmdrun.callf(cmdi, f="adapt.%02i.out" % j)
        # Set next phase to initialize from the output
        nml = self.read_namelist(j+1)
        # Set import_from opt
        nml["flow_initialization"]["import_from"] = f"{fproj1}-restart.solb"
        nml.write(nml.fname)
        # Copy over previous mapbc
        fproj1 = self.get_project_rootname(j+1)
        shutil.copyfile(f"{fproj}.mapbc", f"{fproj1}.mapbc")

    # Run nodet with refine/one adaptation
    def run_nodet_adapt(self, j: int):
        r"""Run Fun3D nodet with adaptation for refine/one

        :Call:
            >>> runner.prepare_files(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2023-07-12 ``@jmeeroff``: v1.0; from ``run_phase``
        """
        # Read settings
        rc = self.read_case_json()
        if not (rc.get_Adaptive() and rc.get_AdaptPhase(j)):
            return
        # Check the adapataion method
        # For Refine/one just use the feature based adaptaion in namelist
        if rc.get_AdaptMethod() != "refine/one":
            return
        # Read namelist
        nml = self.read_namelist(j)
        # Run the feature-based adaptive mesher
        cmdi = cmdgen.nodet(rc, adapt=True, j=j)
        # Get current restart option
        restart_opt, nohist_opt = nml.GetRestart()
        # Make sure "restart_read" is set to .true.
        if (not restart_opt) or nohist_opt:
            nml.SetRestart(True)
            nml.write('fun3d.%02i.nml' % j)
        # Call the command.
        cmdrun.callf(cmdi, f='adapt.out')
        # Rename output file after completing that command
        os.rename('adapt.out', 'adapt.%02i.out' % j)

    # Function to intersect geometry if appropriate
    def run_intersect_fun3d(self, j: int):
        r"""Run ``intersect`` to combine surface triangulations

        This version is customized for FUN3D in order to take a single
        argument.

        :Call:
            >>> runner.run_intersect_fun3d(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :See also:
            * :class:`cape.trifile.Tri`
            * :func:`cape.cfdx.cmdgen.intersect`
        :Versions:
            * 2024-08-22 ``@ddalle``: v1.0
        """
        # Get the project name
        fproj = self.get_project_rootname(j)
        # Run intersect
        self.run_intersect(j, fproj)

    def run_verify_fun3d(self, j: int):
        r"""Run ``verify`` to check triangulation if appropriate

        This version is customized for FUN3D in order to take a single
        argument.

        :Call:
            >>> runner.run_verify_fun3d(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2024-08-22 ``@ddalle``: v1.0
        """
        # Get the project name
        fproj = self.get_project_rootname(j)
        # Run verify
        self.run_verify(j, fproj)

    def run_aflr3_fun3d(self, j: int):
        r"""Create volume mesh using ``aflr3``

        This version is customized for FUN3D in order to take a single
        argument.

        :Call:
            >>> runner.run_aflr3_fun3d(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2024-08-22 ``@ddalle``: v1.0
        """
        # Read namelist
        nml = self.read_namelist(j)
        # Get the project name
        fproj = self.get_project_rootname(j)
        # Create volume mesh if necessary
        self.run_aflr3(j, fproj, fmt=nml.GetGridFormat())

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
        # Check for dual phase
        if rc.get_Dual():
            os.chdir('Flow')
        # Move subiterations if present
        self._copy_subhist(j)
        # Delete any input file (primary namelist)
        if os.path.isfile('fun3d.nml') or os.path.islink('fun3d.nml'):
            os.remove('fun3d.nml')
        # Create the correct namelist
        os.symlink('fun3d.%02i.nml' % j, 'fun3d.nml')
        # Delete any moving_body.input namelist link
        fmove = 'moving_body.input'
        if os.path.isfile(fmove) or os.path.islink(fmove):
            os.remove(fmove)
        # Target moving_body.[0-9][0-9].input file
        ftarg = 'moving_body.%02i.input' % j
        # Create the correct namelist
        if os.path.isfile(ftarg):
            os.symlink(ftarg, fmove)
        # Return to original folder
        if rc.get_Dual():
            os.chdir('..')

    # Copy sub-iteration histories
    def _copy_subhist(self, j: int):
        r"""Copy subiteration histories before FUN3D overwrites them

        :Call:
            >>> runner._copy_subhist(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2024-01-24 ``@ddalle``: v1.0
        """
        # Get the project name
        proj = self.get_project_rootname(j)
        # Generate expected file name
        fname = f"{proj}_subhist.dat"
        # No action if file does not exist
        if not os.path.isfile(fname):
            return
        # Check for previous copies
        pat1 = f"{proj}_subhist.old[0-9][0-9].dat"
        glob1 = glob.glob(pat1)
        # Create output file name
        fcopy = f"{proj}_subhist.old{len(glob1) + 1:02d}.dat"
        # Move the file
        os.rename(fname, fcopy)

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
        :Versions:
            * 2016-04-14 ``@ddalle``: v1.0 (``FinalizeFiles``)
            * 2023-07-06 ``@ddalle``: v1.1; instance method
        """
        # Read settings
        rc = self.read_case_json()
        # Get the project name
        fproj = self.get_project_rootname(j)
        # Get the last iteration number
        n = self.get_iter()
        # Don't use ``None`` for this
        if n is None:
            n = 0
        # Check for dual folder setup
        if os.path.isdir('Flow'):
            # Enter the flow folder
            os.chdir('Flow')
            qdual = True
            # History gets moved to parent
            fhist = os.path.join('..', 'run.%02i.%i' % (j, n))
        else:
            # Single folder
            qdual = False
            # History remains in present folder
            fhist = 'run.%02i.%i' % (j, n)
        # Assuming that worked, move the temp output file.
        if os.path.isfile('fun3d.out'):
            # Check if it's valid
            if self._getx_iter_stdoutfile('fun3d.out') is not None:
                # Move the file
                os.rename('fun3d.out', fhist)
        else:
            # Create an empty file
            fileutils.touch(fhist)
        # Rename the flow file, too.
        if rc.get_KeepRestarts(j):
            shutil.copy('%s.flow' % fproj, '%s.%i.flow' % (fproj, n))
        # Move back to parent folder if appropriate
        if qdual:
            os.chdir('..')

    # Prepare a case for "warm start"
    def prepare_warmstart(self):
        r"""Process WarmStart settings and copy files if appropriate

        :Call:
            >>> warmstart = runner.prepare_warmstart()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *warmstart*: ``True`` | ``False``
                Whether or not case is a valid warm-start
        :Versions:
            * 2023-03-14 ``@ddalle``: v1.0 (``PrepareWarmStart``)
            * 2023-07-06 ``@ddalle``: v1.1; instance method
        """
        # Read settings
        rc = self.read_case_json()
        # Check initial WarmStart setting
        if not rc.get_WarmStart(0):
            return False
        # Get folder
        fdir = rc.get_WarmStartFolder(0)
        # Check for an *fdir* input
        if fdir is not None:
            # Get conditions
            x = self.read_conditions()
            # Absolutize path to source folder
            srcdir = os.path.realpath(fdir % x)
            # Remember location
            workdir = os.getcwd()
            # Check if current folder
            return srcdir != workdir
        # Valid warm-start scenario
        return True

    # Function to set the most recent file as restart file.
    def set_restart_iter(self, n=None):
        r"""Set a given check file as the restart point

        :Call:
            >>> runner.set_restart_iter(n=None)
        :Inputs:
            *rc*: :class:`RunControlOpts`
                Run control options
            *n*: {``None``} :class:`int`
                Restart iteration number, defaults to latest available
        :Versions:
            * 2014-10-02 ``@ddalle``: v1.0 (``SetRestartIter``)
            * 2023-03-14 ``@ddalle``: v1.1; add WarmStart
            * 2023-07-06 ``@ddalle``: v1.2; instance method
        """
        # Check the input
        if n is None:
            n = self.get_restart_iter()
        # Read settings
        rc = self.read_case_json()
        # Read the namelist
        nml = self.read_namelist()
        # Flag to rewrite namelist
        nml_write_flag = False
        # Current restart setting
        restart_opt, nohist_opt = nml.GetRestart()
        # Check adapt method
        adapt_opt = rc.get_AdaptMethod()
        # Set restart flag
        if n > 0:
            # Get the phase
            j = self.get_phase()
            # Check if this is a phase restart
            nohist = True
            if os.path.isfile('run.%02i.%i' % (j, n)):
                # Nominal restart
                nohist = False
            elif j == 0:
                # Not sure how we could still be in phase 0
                nohist = False
            else:
                # Check for preceding phases
                f1 = glob.glob('run.%02i.*' % (j-1))
                n1 = rc.get_PhaseIters(j-1)
                # Read the previous namelist
                if n is not None and n1 is not None and (n > n1):
                    if (len(f1) > 0) and os.path.isfile("fun3d.out"):
                        # Current phase was already run, but run.{i}.{n}
                        # wasn't created
                        nml0 = self.read_namelist(j)
                    else:
                        nml0 = self.read_namelist(j - 1)
                else:
                    # Read the previous phase
                    nml0 = self.read_namelist(j - 1)
                # Get 'time_accuracy' parameter
                sec = 'nonlinear_solver_parameters'
                opt = 'time_accuracy'
                ta0 = nml0.get_opt(sec, opt)
                ta1 = nml.get_opt(sec, opt)
                # Check for a match
                nohist = (ta0 != ta1)
                # If mode switch, prevent Fun3D deleting history
                if nohist:
                    self.copy_hist(j - 1)
            # Ensure restart off for ref3 adapt
            if (adapt_opt == "refine/three"):
                # Get previous phase number
                jprev = max(0, j-1)
                # Check current flag
                if rc.get_AdaptPhase(jprev):
                    # Set the restart flag off
                    nml.SetRestart(False, nohist=nohist)
                    nml_write_flag = True
            elif (not restart_opt) or (nohist_opt != nohist):
                # Set the restart flag on
                nml.SetRestart(True, nohist=nohist)
                nml_write_flag = True
        else:
            # Check for warm-start flag
            warmstart = self.prepare_warmstart()
            # Check current flag
            if restart_opt != warmstart:
                # Set the restart flag on/off depending on warm-start config
                nml.SetRestart(warmstart)
                nml_write_flag = True
        # Write the namelist
        if nml_write_flag:
            nml.write()

    # Copy the histories
    def copy_hist(self, j: int):
        r"""Copy all FM and residual histories

        :Call:
            >>> runner.copy_hist(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number to use for storing histories
        :Versions:
            * 2016-10-28 ``@ddalle``: v1.0 (``CopyHist``)
            * 2023-07-06 ``@ddalle``: v1.1; instance method
        """
        # Read namelist
        nml = self.read_namelist(j)
        # Project name
        proj = self.get_project_rootname(j)
        # Get the list of FM files
        fmglob = glob.glob('%s_fm_*.dat' % proj)
        # Loop through FM files
        for f in fmglob:
            # Split words
            F = f.split('.')
            # Avoid re-copies
            if len(F) > 2:
                continue
            # Copy-to name
            fcopy = F[0] + ('.%02i.dat' % j)
            # Avoid overwrites
            if os.path.isfile(fcopy):
                continue
            # Copy the file
            os.rename(f, fcopy)
        # Copy the history file
        if os.path.isfile('%s_hist.dat' % proj):
            # Destination name
            fcopy = '%s_hist.%02i.dat' % (proj, j)
            # Avoid overwrites
            if not os.path.isfile(fcopy):
                # Copy the file
                os.rename('%s_hist.dat' % proj, fcopy)
        # Copy the history file
        if os.path.isfile('%s_subhist.dat' % proj):
            # Destination name
            fcopy = '%s_subhist.%02i.dat' % (proj, j)
            # Get time-accuracy option
            ta0 = nml.get_opt('nonlinear_solver_parameters', 'time_accuracy')
            # Avoid overwrites
            if not os.path.isfile(fcopy) and (ta0 != 'steady'):
                # Copy the file
                os.rename('%s_subhist.dat' % proj, fcopy)

    # Link best Tecplot files
    @casecntl.run_rootdir
    def link_plt(self):
        r"""Link the most recent Tecplot files to fixed file names

        :Call:
            >>> runner.link_plt()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Versions:
            * 2016-10-24 ``@ddalle``: v1.0 (``LinkPLT``)
            * 2023-07-06 ``@ddalle``: v1.1; instance method
        """
        # Determine phase number
        j = self.get_phase()
        # Need the namelist to figure out planes, etc.
        nml = self.read_namelist(j)
        # Get the project root name
        proj0, proj = self._get_project_roots()
        # Get the list of output surfaces
        fsrf = nml.get_opt("sampling_parameters", "label")
        # Pattern for globs
        pats = [
            "%s_%s_timestep*",
            "%s_%s",
        ]
        # Full list of Tecplot files
        allsurfs = ["tec_boundary", "volume"]
        allsurfs.extend(fsrf)
        # Initialize list of resulting link files
        fname = []
        # Initialize list of search patterns
        fglob = []
        # Loop through boundary, volume, and surfaces
        for fi in allsurfs:
            # Name of link to create
            fname.append('%s_%s' % (proj0, fi))
            # Lists of patterns to search for
            fglobi = [pat % (proj, fi) for pat in pats]
            fglob.append(fglobi)
        # Link the globs
        for i in range(len(fname)):
            # Loop through viz extensions
            for ext in (".tec", ".dat", ".plt", ".szplt"):
                # Append extensions to output and patterns
                fnamei = fname[i] + ext
                fglobi = [fj + ext for fj in fglob[i]]
                # Process the glob as well as possible
                LinkFromGlob(fnamei, fglobi)

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
        # Determine phase number
        j = self.get_phase()
        # Need the namelist to figure out planes, etc.
        nml = self.read_namelist(j)
        # Get the project root name
        proj = nml.get_opt('project', 'project_rootname')
        # Strip suffix
        if rc.get_Dual() or rc.get_Adaptive():
            # Strip adaptive section
            proj = proj[:-2]
        # Output
        return proj

    # Function to get the most recent working folder
    @casecntl.run_rootdir
    def get_working_folder(self) -> str:
        r"""Get working folder, ``.``,  or ``Flow/``

        :Call:
            >>> fdir = runner.get_working_folder()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *fdir*: ``"Flow"`` | ``"."``
                Location (relative to *runner.root_dir*) where ``nodet``
                will be run next
        :Versions:
            * 2024-07-29 ``@ddalle``: v1.0
        """
        # Check for Flow/ folder
        if os.path.isdir("Flow"):
            # Primal is run in Flow/ folder
            return "Flow"
        else:
            # Not using dual path
            return "."

    # Get project root name but "pyfun", not "pyfun02"
    def _get_project_roots(self):
        r"""This aims to return both ``pyfun[0-9][0-9]`` and ``pyfun``

        :Call:
            >>> proj0, proj = runner.get_project_baserootname()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *proj0*: :class:`str`
                Base project rootname, w/o adaptation counter
            *proj*: :class:`str`
                Pattern for project rootname w/ adapt counters
        :Versions:
            * 2024-03-22 ``@ddalle``: v1.0
        """
        # Read the options
        rc = self.read_case_json()
        # Determine phase number
        j = self.get_phase()
        # Need the namelist to figure out planes, etc.
        nml = self.read_namelist(j)
        # Get the project root name
        proj = nml.get_opt('project', 'project_rootname')
        # Strip suffix
        if rc.get_Dual() or rc.get_Adaptive():
            # Strip adaptive section
            proj0 = proj[:-2]
            # Search for 'pyfun00', 'pyfun01', ...
            proj = proj0 + "[0-9][0-9]"
        else:
            # Use the full project name if no adaptations
            proj0 = proj
        # Output
        return proj0, proj

   # --- Special readers ---
    # Read namelist
    @casecntl.run_rootdir
    def read_namelist(self, j=None):
        r"""Read case namelist file

        :Call:
            >>> nml = runner.read_namelist(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *nml*: :class:`cape.pyfun.namelist.Namelist`
                Namelist interface
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0
            * 2023-06-27 ``@ddalle``: v2.0; instance method
        """
        # Read ``case.json`` if necessary
        rc = self.read_case_json()
        # Process phase number
        if j is None and rc is not None:
            # Default to most recent phase number
            j = self.get_phase()
        # Get phase of namelist previously read
        nmlj = self.nml_j
        # Check if already read
        if isinstance(self.nml, Namelist) and nmlj == j and j is not None:
            # Return it!
            return self.nml
        # Check for `Flow` folder
        if os.path.isdir('Flow'):
            # Enter the folder
            qdual = True
            os.chdir('Flow')
        else:
            # No `Flow/` folder
            qdual = False
        # Check for folder with no working ``case.json``
        if rc is None:
            # Check for simplest namelist file
            if os.path.isfile('fun3d.nml'):
                # Read the currently linked namelist.
                nml = Namelist('fun3d.nml')
            else:
                # Look for namelist files
                fglob = glob.glob('fun3d.??.nml')
                # Sort it
                fglob.sort()
                # Read one of them.
                nml = Namelist(fglob[-1])
            # Return home if appropriate
            if qdual:
                os.chdir('..')
            return nml
        # Get the specified namelist
        nml = Namelist('fun3d.%02i.nml' % j)
        # Exit `Flow/` folder if necessary
        if qdual:
            os.chdir('..')
        # Output
        return nml

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
        :Versions:
            * 2024-11-05 ``@ddalle``: v1.0
        """
        # Project name
        fproj = self.get_project_rootname(j)
        # Use the project name with ".flow"
        return f"{fproj}.flow"

    # Get list of files needed for restart
    @casecntl.run_rootdir
    def get_restartfiles(self) -> list:
        r"""Get recent ``.flow`` and grid files to protect from clean

        :Call:
            >>> restartfiles = runner.get_restartfiles()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *restartfiles*: :class:`list`\ [:class:`str`]
                List of files not to delete during ``--clean``
        :Versions:
            * 2024-12-12 ``@ddalle``: v1.0
        """
        # Read namelist
        nml = self.read_namelist()
        # Get current project name
        proj = nml.get_opt("project", "project_rootname")
        # Protect the namelist and restart file
        restartfiles = [
            nml.fname,
            f"{proj}.flow",
            f"{proj}.mapbc",
        ]
        # Add in all grid files
        restartfiles.extend(glob.glob(f"{proj}.*ugrid"))
        # Output
        return restartfiles

    # Get list of files needed for reports
    def get_reportfiles(self) -> list:
        r"""Generate list of report files

        :Call:
            >>> filelist = runner.get_reportfiles()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *filelist*: :class:`list`\ [:class:`str`]
                List of files to protect
        :Verions:
            * 2024-09-25 ``@ddalle``: v1.0
        """
        # Initialize file list
        filelist = []
        # Get base rootname
        proj = self.get_project_baserootname()
        # Read namelist
        nml = self.read_namelist()
        # Read archivist
        a = self.get_archivist()
        # Tecplot file extensions
        exts = "(plt|tec|szplt)"
        # Get boundary output frequency
        freq = nml.get_opt("global", "boundary_animation_freq")
        # Base pattern for boundary output files
        pat = f"{proj}[0-9]*_tec_boundary"
        # Check for "timestep" label
        if freq == -1:
            # Timestep not in file name
            matchdict = a.search_regex(rf"{pat}\.{exts}")
        else:
            # Timestep in file name
            matchdict = a.search_regex(rf"{pat}_timestep[0-9]+\.{exts}")
        # Append results
        for matches in matchdict.values():
            filelist.extend(matches)
        # Output
        return filelist

    # Create TRIQ file if necessary
    def get_triq_file(self, stem: str = "tec_boundary"):
        r"""Get name of ``.triq`` file, creating it if necessary

        :Call:
            >>> ftriq, n, i0, i1 = runner.get_triq_file(stem)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *stem*: {``"tec_boundary"``} | :class:`str`
                Base name of surface/manifold to read
        :Outputs:
            *ftriq*: :class:`str`
                Name of ``triq`` file
            *n*: :class:`int`
                Number of iterations included
            *i0*: :class:`int`
                First iteration in the averaging
            *i1*: :class:`int`
                Last iteration in the averaging
        :Versions:
            * 2024-12-03 ``@ddalle``: v1.0
        """
        # First get name of raw file
        ftec, n, i0, i1 = self.get_plt_file(stem)
        # Get extnension and base
        basename, ext = ftec.rsplit('.', 1)
        # Name of triq file
        ftriq = f"{basename}.triq"
        # Convert .plt if necessary
        if os.path.isfile(ftriq):
            return ftriq, n, i0, i1
        # Check for ``.szplt`` format
        if ext == "szplt":
            # Convert it to .plt
            try:
                convert_szplt(ftec)
            except Exception:
                print(f"  Failed to convert '{ftec}' to PLT format")
                return None, n, i0, i1
            # Change file name
            ftec = f"{basename}.plt"
        # Mach number
        mach = self.get_mach()
        # Convert PLT file
        pltfile.Plt2Triq(ftec, ftriq, mach=mach)
        # Output
        return ftriq, n, i0, i1

    # Get averaging window for triq file
    def get_triq_filestats(self) -> casecntl.IterWindow:
        r"""Get start and end of averagine window for ``.triq`` file

        :Call:
            >>> window = runner.get_triq_filestats(ftriq)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *ftriq*: :class:`str`
                Name of latest ``.triq`` annotated triangulation file
        :Outputs:
            *window.ia*: :class:`int`
                Iteration at start of window
            *window.ib*: :class:`int`
                Iteration at end of window
        :Versions:
            * 2025-02-12 ``@ddalle``: v1.0
        """
        # Working here on surface
        stem = "tec_boundary"
        # Get root name of project
        basename = self.get_project_baserootname()
        # Glob for initial filter of files
        baseglob = f"{basename}*_{stem}*"
        # Form pattern for all possible output files
        # Part 1 matches "pyfun_tec_boundary" and "pyfun02_tec_boundary"
        # Part 2 matches "_timestep2500" or ""
        # Part 3 matches ".dat", ".plt", ".szplt", or ".tec"
        pat = (
            f"{basename}(?P<gn>[0-9][0-9]+)?_{stem}" +
            "(_timestep(?P<t>[1-9][0-9]*))?" +
            r"\.(?P<ext>dat|plt|szplt|tec)")
        # Find appropriate PLT file (or SZPLT ...)
        fplt, fmatch = fileutils.get_latest_regex(pat, baseglob)
        # Check for match
        if fplt is None:
            return casecntl.IterWindow(None, None)
        # Get the timestep number, if any
        t = fmatch.group("t")
        # Either way, we're going to need the run log phases and iters
        runlog = self.get_runlog()
        # Convert to list for iterative backward search
        runlist = list(runlog)
        # Get most recent
        if len(runlist):
            # Get last CAPE exit
            jlast, nlast = runlist.pop(-1)
        else:
            # No run logs yet
            jlast, nlast = 0, 0
        # Check if we found a timestep in the file name
        if t is None:
            # The iteration is from the last CAPE exit
            jplt, nplt = jlast, nlast
        else:
            # Got an iteration from timestep
            # We need to read iter history to check for FUN3D iteration
            # resets, e.g. at transition from RANS -> uRANS
            hist = CaseResid(basename)
            # In this case, default to the current phase
            jplt = self.get_phase()
            # Find the most recent time FUN3D reported *t*
            mask, = np.where(hist["solver_iter"] == int(t))
            # Use the last hit
            if mask.size == 0:
                # No matches? Cannot correct FUN3D's iter
                nplt = int(t)
            else:
                # Read CAPE iter from last time FUN3D reported *t*
                nplt = int(hist["i"][mask[-1]])
                # Check if we're *after* the last output
                if nplt <= nlast:
                    # This file came from a completed run; find which
                    mask1, = np.where(nplt <= runlog[:, 1])
                    # The last phase before *nplt* is the source
                    jplt = runlog[mask1[-1], 0]
                else:
                    # Add the most recent exit back to the runlist
                    runlist.append((jlast, nlast))
        # Until we find otherwise, assume there's no averaging
        nstrt = nplt
        # Track current phase
        jcur = jplt
        # Go backwards through runlog to see where averaging started
        while True:
            # Read the most appropriate namelist
            nmlj = self.read_namelist(jcur)
            # Check for time averaging
            tavg = nmlj.get_opt("time_avg_params", "itime_avg", vdef=0)
            # Process time-averaging
            if not tavg:
                # No time-averaging; do not update *nstrt*
                break
            # Need the preceding exit to see where averaging started
            if len(runlist):
                # Get last exit
                jcur, nlast = runlist.pop(-1)
                nstrt = nlast + 1
            else:
                # Started from zero
                nstrt = 1
                # No previous runs to check
                break
            # Check if we kept stats from *previous* run
            tprev = nmlj.get_opt(
                "time_avg_params", "user_prior_time_avg", vdef=1)
            # If we didn't keep prior stats; search is done
            if not tprev:
                break
        # Output
        return casecntl.IterWindow(nstrt, nlast)

    # Find boundary PLT file
    def get_plt_file(self, stem: str = "tec_boundary"):
        r"""Get most recent boundary ``plt`` file and its metadata

        :Call:
            >>> fplt, n, i0, i1 = runner.get_plt_file()
            >>> fplt, n, i0, i1 = runner.get_plt_file(stem)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *fplt*: :class:`str`
                Name of ``plt`` file
            *n*: :class:`int`
                Number of iterations included
            *i0*: :class:`int`
                First iteration in the averaging
            *i1*: :class:`int`
                Last iteration in the averaging
        :Versions:
            * 2016-12-20 ``@ddalle``: v1.0 (``GetPltFile``)
            * 2023-07-06 ``@ddalle``: v1.1; instance method
            * 2024-03-25 ``@ddalle``: v2.0
                - add *stem* imput
                - search for files w/o ``_timestep{n}`` in name
                - include previous runs in *n* if appropriate
                - more accurate accounting for FUN3D/s iter resets
        """
        # Get root name of project
        basename = self.get_project_baserootname()
        # Glob for initial filter of files
        baseglob = f"{basename}*_{stem}*"
        # Form pattern for all possible output files
        # Part 1 matches "pyfun_tec_boundary" and "pyfun02_tec_boundary"
        # Part 2 matches "_timestep2500" or ""
        # Part 3 matches ".dat", ".plt", ".szplt", or ".tec"
        pat = (
            f"{basename}(?P<gn>[0-9][0-9]+)?_{stem}" +
            "(_timestep(?P<t>[1-9][0-9]*))?" +
            r"\.(?P<ext>dat|plt|szplt|tec)")
        # Find appropriate PLT file (or SZPLT ...)
        fplt, fmatch = fileutils.get_latest_regex(pat, baseglob)
        # Check for at least one match
        if fplt is None:
            # No such files yet
            return fplt, None, None, None
        # Get the timestep number, if any
        t = fmatch.group("t")
        # Either way, we're going to need the run log phases and iters
        runlog = self.get_runlog()
        # Convert to list for iterative backward search
        runlist = list(runlog)
        # Get most recent
        if len(runlist):
            # Get last CAPE exit
            jlast, nlast = runlist.pop(-1)
        else:
            # No run logs yet
            jlast, nlast = 0, 0
        # Check if we found a timestep in the file name
        if t is None:
            # The iteration is from the last CAPE exit
            jplt, nplt = jlast, nlast
        else:
            # Got an iteration from timestep
            # We need to read iter history to check for FUN3D iteration
            # resets, e.g. at transition from RANS -> uRANS
            hist = CaseResid(basename)
            # In this case, default to the current phase
            jplt = self.get_phase()
            # Find the most recent time FUN3D reported *t*
            mask, = np.where(hist["solver_iter"] == int(t))
            # Use the last hit
            if mask.size == 0:
                # No matches? Cannot correct FUN3D's iter
                nplt = int(t)
            else:
                # Read CAPE iter from last time FUN3D reported *t*
                nplt = int(hist["i"][mask[-1]])
                # Check if we're *after* the last output
                if nplt <= nlast:
                    # This file came from a completed run; find which
                    mask1, = np.where(nplt <= runlog[:, 1])
                    # The last phase before *nplt* is the source
                    jplt = runlog[mask1[-1], 0]
                else:
                    # Add the most recent exit back to the runlist
                    runlist.append((jlast, nlast))
        # Until we find otherwise, assume there's no averaging
        nstrt = nplt
        # Track current phase
        jcur = jplt
        # Go backwards through runlog to see where averaging started
        while True:
            # Read the most appropriate namelist
            nmlj = self.read_namelist(jcur)
            # Check for time averaging
            tavg = nmlj.get_opt("time_avg_params", "itime_avg", vdef=0)
            # Process time-averaging
            if not tavg:
                # No time-averaging; do not update *nstrt*
                break
            # Need the preceding exit to see where averaging started
            if len(runlist):
                # Get last exit
                jcur, nlast = runlist.pop(-1)
                nstrt = nlast + 1
            else:
                # Started from zero
                nstrt = 1
                # No previous runs to check
                break
            # Check if we kept stats from *previous* run
            tprev = nmlj.get_opt(
                "time_avg_params", "user_prior_time_avg", vdef=1)
            # If we didn't keep prior stats; search is done
            if not tprev:
                break
        # Calculate how many iterations are averaged
        nstats = nplt - nstrt + 1
        # Output
        return fplt, nstats, nstrt, nplt

    # Search pattern for surface output files
    def get_flowviz_regex(self, stem: str) -> str:
        # Get root name of project
        basename = self.get_project_baserootname()
        # Constant stem
        stem = "tec_boundary"
        # Part 1 matches "pyfun_tec_boundary" and "pyfun02_tec_boundary"
        # Part 2 matches "_timestep2500" or ""
        # Part 3 matches ".dat", ".plt", ".szplt", or ".tec"
        pat = (
            f"{basename}(?P<gn>[0-9][0-9]+)?_{stem}" +
            "(_timestep(?P<t>[1-9][0-9]*))?" +
            r"\.(?P<ext>dat|plt|szplt|tec)")
        # Return it
        return pat

    # Search pattern for surface output files
    def get_surf_regex(self) -> str:
        # Constant stem
        stem = "tec_boundary"
        # Use general method
        return self.get_flowviz_regex(stem)

    def get_surf_pat(self) -> str:
        r"""Get glob pattern for candidate surface data files

        These can have false-positive matches because the actual search
        will be done by regular expression. Restricting this pattern can
        have the benefit of reducing how many files are searched by
        regex.

        :Call:
            >>> pat = runner.get_surf_pat()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *pat*: :class:`str`
                Glob file name pattern for candidate surface sol'n files
        :Versions:
            * 2025-01-24 ``@ddalle``: v1.0
        """
        # Get root name of project
        basename = self.get_project_baserootname()
        # Constant stem
        stem = "tec_boundary"
        # Glob for initial filter of files
        baseglob = f"{basename}*_{stem}*"
        # Return it
        return baseglob

   # --- Status ---
    # Function to chose the correct input to use from the sequence.
    def getx_phase(self, n: int):
        r"""Determine the phase number based on files in folder

        :Call:
            >>> i = casecntl.GetPhaseNumber(rc)
        :Inputs:
            *rc*: :class:`RunControlOpts`
                Options interface for run control
        :Outputs:
            *i*: :class:`int`
                Most appropriate phase number for a restart
        :Versions:
            * 2014-10-02 ``@ddalle``: v1.0 (``cape.pycart``)
            * 2015-10-19 ``@ddalle``: v1.0 (``GetPhaseNumber``)
            * 2023-07-06 ``@ddalle``: v1.1; instance method
        """
        # Read settings
        rc = self.read_case_json()
        # Global options
        qdual = rc.get_Dual()
        qadpt = rc.get_Adaptive()
        # Get phase sequence
        phases = self.get_phase_sequence()
        # Loop through possible input numbers.
        for i, j in enumerate(phases):
            # Check for output files.
            if len(glob.glob('run.%02i.*' % j)) == 0:
                # This run has not been completed yet.
                return j
            # Check the iteration numbers
            if rc.get_PhaseIters(i) is None:
                # Don't check null phases
                pass
            elif n is None:
                # No iters yet
                return j
            elif n < rc.get_PhaseIters(i):
                # This case has been run, not yet reached cutoff
                return j
            # Check for dual
            if qdual and rc.get_DualPhase(j):
                # Check for the dual output file
                if not os.path.isfile(
                        os.path.join('Adjoint', 'dual.%02i.out' % j)):
                    return i
            # Check for dual
            if qadpt and rc.get_AdaptPhase(i):
                # Check for weird hybrid setting
                if qdual:
                    # ``Flow/`` folder; other phases may be dual phases
                    fadpt = os.path.join('Flow', 'dual.%02i.out' % j)
                else:
                    # Purely adaptive; located in this folder
                    fadpt = 'adapt.%02i.out' % j
                # Check for the dual output file
                qadpt = os.path.isfile(fadpt)
                # Check for subseqnent phase outputs
                qnext = len(glob.glob("run.%02i.*" % (j+1))) > 0
                if not (qadpt or qnext):
                    return j
        # Case completed; just return the last phae
        return j

    # Check success
    def get_returncode(self):
        r"""Check for errors before continuing

        Currently the following checks are performed.

            * Check for NaN residual in the output file

        :Call:
            >>> ierr = runner.get_returncode())
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *ierr*: :class:`int`
                Return code
        :Versions:
            * 2016-04-18 ``@ddalle``: v1.0
            * 2023-06-02 ``@ddalle``: v1.1; return ``bool``; don't raise
            * 2023-07-06 ``@ddalle``: v1.2; instance method
            * 2024-06-17 ``@ddalle``: v1.3; was ``check_error()``
            * 2024-07-16 ``@ddalle``: v1.4; use *self.returncode*
        """
        # Get phase number
        j = self.get_phase(f=False)
        # Get last iteration run
        n = self.get_iter()
        # Don't use ``None`` for this
        if n is None:
            n = 0
        # Output file name
        fname = 'run.%02i.%i' % (j, n)
        # Check for the file
        if os.path.isfile(fname):
            # Get the last line from nodet output file
            line = fileutils.tail(fname)
            # Check if NaN is in there
            if 'NaN' in line:
                return casecntl.IERR_NANS
        # Otherwise no errors detected
        return getattr(self, "returncode", casecntl.IERR_OK)

    # Get current iteration
    def getx_iter(self):
        r"""Calculate most recent FUN3D iteration

        :Call:
            >>> n = runner.getx_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *n*: :class:`int`
                Iteration number
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0
            * 2016-04-28 ``@ddalle``: v1.1; ``Flow/`` folder
            * 2023-06-27 ``@ddalle``: v2.0; instance method
            * 2024-08-10 ``@ddalle``: v2.1; mostly getx_iter_running()
        """
        # Read the two sources
        _, ns = self.getx_iter_history()
        nr = self.getx_iter_running()
        # Process
        if nr in (0, None):
            # No running iterations; check history
            return ns
        else:
            # Some iterations saved and some running
            return nr

    # Get iteration if restart
    def getx_restart_iter(self):
        r"""Calculate number of iteration if case should restart

        :Call:
            >>> nr = runner.gets_restart_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *nr*: :class:`int`
                Restart iteration number
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0
            * 2016-04-19 ``@ddalle``: v1.1; check STDIO
            * 2020-01-15 ``@ddalle``: v1.2; sort globs better
            * 2023-07-05 ``@ddalle``: v1.3; moved to instance method
        """
        # Can just new, more robust getx_iter_running() here?
        n = self.getx_iter_running()
        return 0 if n is None else n
        # List of saved run files
        frun_glob = glob.glob('run.[0-9]*.[0-9]*')
        # More exact pattern check
        frun_pattern = []
        # Loop through glob finds
        for fi in frun_glob:
            # Above doesn't guarantee exact pattern
            try:
                # Split into parts
                _, s_phase, s_iter = fi.split(".")
                # Compute phase and iteration
                int(s_phase)
                int(s_iter)
            except Exception:
                continue
            # Append to filterted list
            frun_pattern.append(fi)
        # Sort by iteration number
        frun = sorted(frun_pattern, key=lambda f: int(f.split(".")[2]))
        # List the output files
        if os.path.isfile('fun3d.out'):
            # Only use the current file
            fflow = frun + ['fun3d.out']
        elif os.path.isfile(os.path.join('Flow', 'fun3d.out')):
            # Use the current file from the ``Flow/`` folder
            fflow = frun + [os.path.join('Flow', 'fun3d.out')]
        else:
            # Use the run output files
            fflow = frun
        # Initialize iteration number until informed otherwise.
        n = 0
        # Cumulative restart iteration number
        n0 = 0
        # Loop through the matches.
        for fname in fflow:
            # Check for restart of iteration counter
            lines = fileutils.grep('on_nohistorykept', fname)
            if len(lines) > 1:
                # Reset iteration counter
                n0 = n
                n = 0
            # Get the output report lines
            lines = fileutils.grep('current history iterations', fname)
            # Be safe
            try:
                # Split up line
                V = lines[-1].split()
                # Attempt to get existing iterations
                try:
                    # Format: "3000 + 2000 = 5000"
                    i0 = int(V[-5])
                except Exception:
                    # No restart...
                    # restart_read is 'off' or 'on_nohistorykept'
                    i0 = 0
                # Get the last write iteration number
                i = int(V[-1])
                # Update iteration number
                if i0 < n:
                    # Somewhere we missed an on_nohistorykept
                    n0 = n
                    n = i
                else:
                    # Normal situation
                    n = max(i, n)
            except Exception:
                pass
        # Output
        return n0 + n

    # Get iteration number from "history"
    @casecntl.run_rootdir
    def getx_iter_history(self):
        r"""Get the most recent iteration number for a history file

        :Call:
            >>> nh, n = runner.getx_history_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *nh*: :class:`int`
                Iterations from previous cases before Fun3D deleted history
            *n*: :class:`int` | ``None``
                Most recent iteration number
        :Versions:
            * 2015-10-20 ``@ddalle``: v1.0
            * 2016-04-28 ``@ddalle``: v1.1; for ``Flow/`` folder
            * 2016-10-29 ``@ddalle``: v1.2; handle Fun3D iteration reset
            * 2017-02-23 ``@ddalle``: v1.3; handle adapt project shift
            * 2023-06-27 ``@ddalle``: v2.0; instance method
        """
        # Read JSON settings
        rc = self.read_case_json()
        # Get adaptive settings
        qdual = rc.get_Dual()
        qadpt = rc.get_Adaptive()
        adapt_opt = rc.get_AdaptMethod()
        # Check for flow folder
        if qdual:
            os.chdir("Flow")
        # Read the project rootname
        try:
            rname = self.get_project_rootname()
        except Exception:
            # No iterations
            return None, None
        # Assemble file name.
        fname = "%s_hist.dat" % rname
        # Check for "pyfun00", "pyfun01", etc.
        if qdual or (qadpt and adapt_opt != "refine/three"):
            # Check for sequence of file names
            fnames = glob.glob(rname[:-2] + '??_hist.[0-9][0-9].dat')
            fnames.sort()
            # Single history file name(s)
            fhist = glob.glob("%s??_hist.dat" % rname[:-2])
            # Apppend the most recent one
            if len(fhist) > 0:
                # Get maximum file
                fnhist = max(fhist)
                # Check adaption numbers... don't use older adaption history
                if len(fnames) > 0:
                    # Get adaption number on both files
                    nr = len(rname) - 2
                    na0 = int(fnames[-1][nr:nr+2])
                    na1 = int(fnhist[nr:nr+2])
                    # Don't use pyfun01_hist.dat to append pyfun02_hist.03.dat
                    if na1 >= na0:
                        fnames.append(fnhist)
                else:
                    # No previous history; append
                    fnames.append(fnhist)
        else:
            nml = self.read_namelist()
            qrestart, _ = nml.GetRestart()
            # Need to allow for adapt soft reset?
            if ((adapt_opt == "refine/three") and not qrestart):
                fnames = glob.glob("%s[0-9][0-9]_hist.dat" % rname[:-2])
                fnames.sort()
            else:
                # Check for historical files
                fnames = glob.glob("%s_hist.[0-9][0-9].dat" % rname)
                fnames.sort()
                # Single history file name
                fnames.append("%s_hist.dat" % rname)
        # Loop through possible file(s)
        n = None
        nh = 0
        for fname in fnames:
            # Process the file
            ni = self.getx_iter_histfile(fname)
            # Add to history
            if ni is not None:
                # Check if any iterations have been found
                if n is None:
                    # First find
                    n = ni
                    # Check if this is a previous history
                    if len(fname.split('.')) == 3:
                        # Also save as history
                        nh = ni
                elif len(fname.split('.')) == 3 or adapt_opt == "refine/three":
                    # Add this history to prev [restarted iter count]
                    nh = n
                    n += ni
                else:
                    # New file for adaptive but not cumulative
                    n = nh + ni
        # Output
        return nh, n

    # Get the number of iterations from a single iterative history file
    def getx_iter_histfile(self, fname: str):
        r"""Get the most recent iteration number from a history file

        :Call:
            >>> n = runner.getx_iter_histfile(fname)
        :Inputs:
            *fname*: {``"pyfun_hist.dat"``} | :class:`str`
                Name of file to read
        :Outputs:
            *n*: :class:`int` | ``None``
                Most recent iteration number
        :Versions:
            * 2016-05-04 ``@ddalle``: v1.0; from :func:`GetHistoryIter`
            * 2023-06-27 ``@ddalle``: v2.0; rename *GetHistoryIterFile*
        """
        # Check for the file.
        if not os.path.isfile(fname):
            return None
        # Check the file.
        try:
            # Tail the file
            txt = fileutils.tail(fname)
            # Get the iteration number from first "word"
            return int(txt.split()[0])
        except Exception:
            return None

    # Get iteration from STDTOUT
    @casecntl.run_rootdir
    def getx_iter_running(self) -> Optional[int]:
        r"""Get the most recent iteration number for a running file

        :Call:
            >>> n = runner.getx_iter_running()
        :Outputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *n*: :class:`int` | ``None``
                Most recent iteration number
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0
            * 2016-04-28 ``@ddalle``: v1.1; handle ``Flow/`` folder
            * 2023-05-27 ``@ddalle``: v2.0; instance method
            * 2024-07-29 ``@ddalle``: v3.0
                - use :func:`fileutils.readline_reverse`
                - eliminate ``on_nohistorykept`` check
                - check for multiple files (was just ``fun3d.out``)

            * 2024-08-09 ``@ddalle``: v3.1; fix run.??.* sorting
            * 2024-08-10 ``@ddalle``: v3.2; check all run.??.* for rstrt
            * 2024-08-21 ``@ddalle``: v3.3; read prev file for 'off'
        """
        # Get all STDOUT files, in order
        runfiles = self.get_stdoutfiles()
        # Initialize discarded iteration count
        n_discard = 0
        # Loop through all the files
        for j, stdoutfile in enumerate(runfiles):
            # Get previous file (to use for 'off')
            prevfile = None if j == 0 else runfiles[j-1]
            # Check for discarded iter (restart is 'off' or 'on_nohist')
            n_discard += self._getx_i_stdout_discarded(stdoutfile, prevfile)
        # Loop through the files in reverse
        for stdoutfile in reversed(runfiles):
            # Read the file
            n = self._getx_iter_stdoutfile(stdoutfile)
            # Check for any find
            if n is not None:
                return n + n_discard
        # If no matches found, use ``None`` as the iter

    # Get list of STDOUT files
    def get_stdoutfiles(self) -> list:
        r"""Get list of STDOUT files in order they were run

        :Call:
            >>> runfiles = runner.get_stdoutfiles()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *runfiles*: :class:`list`\ [:class:`str`]
                List of run files, in ascending order
        :Versions:
            * 2024-08-09 ``@ddalle``: v1.0
        """
        # Get *previous* running files if any
        candidates = self.get_cape_stdoutfiles()
        # Get working folder
        fdir = self.get_working_folder()
        # Add "fun3d.out" to end of the list
        candidates += glob.glob(os.path.join(fdir, "fun3d.out"))
        # Initialize filetered output
        runfiles = []
        # Loop through candidates
        for runfile in candidates:
            # Check size or ends with .0
            if runfile.endswith(".0") or os.path.getsize(runfile) > 200:
                runfiles.append(runfile)
        # Output
        return runfiles

    # Read a single STDOUT file
    def _getx_iter_stdoutfile(self, fname: str) -> Optional[int]:
        r"""Determine current iteration from FUN3D STDOUT

        :Call:
            >>> n = runner._getx_iter_stdoutfile(fname)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *fname*: :class:`str`
                Name of file with STDOUT from ``nodet``
        :Outputs:
            *n*: ``None`` | :class:`int`
                Most recent iteration number
        :Versions:
            * 2024-07-29 ``@ddalle``: v1.0
            * 2024-07-30 ``@ddalle``: v2.0; revive *restart_read* check
            * 2024-08-10 ``@ddalle``: v2.1; use smaller functions
        """
        # Get restart setting
        restart_read = self._read_stdout_restart(fname)
        # If restart_read is "on", need to get restart iters
        if restart_read == "on":
            # Get number of iters in restart file
            nr = self._read_stdout_restart_iter(fname, restart_read)
        else:
            # Fresh history (according to FUN3D)
            nr = None
        # Initialize running iter
        n = None
        # Open file
        with open(fname, 'rb') as fp:
            # Move to EOF
            fp.seek(0, 2)
            # Loop through lines of file
            while True:
                # Read preceding line
                line = fileutils.readline_reverse(fp)
                # Check line against regex
                re_match = REGEX_F3DOUT.match(line)
                # Check for exit criteria
                if line == b'':
                    # Reached start of file w/o match
                    break
                elif re_match:
                    # Convert string to integer
                    n = int(re_match.group('iter'))
                    break
                elif b'current history iterations' in line:
                    # Directly specified
                    nr = None
                    n = int(line.split()[-1])
                    break
        # Output
        if n is not None:
            if nr is not None:
                # Return the sum
                return n + nr
            else:
                # Just the line-by-line count
                return n
        else:
            # Restart iter
            return nr

    # Get discareded restart read setting
    def _getx_i_stdout_discarded(
            self,
            fname: str,
            fprev: Optional[str] = None) -> int:
        r"""Get number of iterations discarded during restart

        If the ``restart_read`` setting is anything other than ``"on"``,
        the history iterations will be reported, but FUN3D will start
        over at ``0``.

        If ``restart_read`` is ``"on"``, this will return ``0``.

        :Call:
            >>> n = runner._getx_i_stdout_discarded(fname)
        :Outputs:
            *n*: :class:`int`
                Number of restart iterations not used
        """
        # Get restart setting
        restart_read = self._read_stdout_restart(fname)
        # Check flag
        if restart_read == "on":
            # No iterations discarded
            return 0
        elif restart_read == "off":
            # Check for previous file iters
            if fprev and os.path.getsize(fname) > 200:
                # Read iters from previous file
                nr = self._getx_iter_stdoutfile(fprev)
                return 0 if nr is None else nr
            # No previous file
            return 0
        else:
            # FUN3D reports iters in restart file but discards them
            nr = self._read_stdout_restart_iter(fname, restart_read)
            # Convert None -> 0
            return 0 if nr is None else nr

    # Get restart iter from stdout
    def _read_stdout_restart_iter(self, fname: str, rr: str) -> Optional[int]:
        r"""Get the reported number of iterations in the restart file

        :Call:
            >>> nr = runner._read_stdout_restart_iter(fname)
        :Outputs:
            *nr*: ``None`` | :class:`int`
                Number of iterations in restart file
        """
        # Search for text describing how many restart iters were
        try:
            lines = fileutils.grep("the restart files contains", fname, nmax=1)
        except UnicodeDecodeError:
            print(f"File {fname} in case {self.root_dir} is unreadable")
            nr = None
        # Try to convert it
        try:
            # Try to convert first match
            nr = int(lines[0].split('=')[-1])
        except Exception:
            # No iters found
            nr = None
        # Output
        return nr

    # Get restart setting
    def _read_stdout_restart(self, fname: str) -> Optional[str]:
        r"""Get the ``restart_read`` setting from FUN3D STDOUT

        :Call:
            >>> restart = runner._read_stdout_restart(fname)
        :Inputs:
            *fname*: :class:`str`
                Name of file
        :Outputs:
            *restart*: ``"off"`` | ``"on"`` | ``"on_nohistorykept"``
                Restart setting
        """
        # Find the "restart_read" setting
        lines = fileutils.grep(r"^\s*restart_read\s*=", fname, nmax=1)
        # Default (in case no match)
        restart_read = "off"
        # Check for match
        if len(lines) > 0:
            # Get setting
            restart_read = lines[0].split('=')[1].strip()
            restart_read = restart_read.strip('"').strip("'")
        # Output
        return restart_read

   # --- Conditions ---
    # Read Mach number from namelist
    def read_mach(self) -> float:
        r"""Read Mach number from namelist file

        :Call:
            >>> mach = runner.get_mach()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *key*: :class:`str`
                Name of run matrix key to query
            *f*: ``True`` | {``False``}
                Option to force re-read
        :Outputs:
            *mach*: :class:`float`
                Mach number
        :Versions:
            * 2024-12-03 ``@ddalle``: v1.0
        """
        # Read namelist
        nml = self.read_namelist()
        # Key section
        sec = "reference_physical_properties"
        # Get input type
        itype = nml.get_opt(sec, "dim_input_type")
        # Check equation type
        if itype == "dimensional-SI":
            # Read velocity, density, temperature
            v = nml.get_opt(sec, "velocity")  # m/s
            t = nml.get_opt(sec, "temperature")
            W = nml.get_opt(sec, "molecular_weight", vdef=28.964)
            g = nml.get_opt(sec, "gamma", vdef=1.4)
            R = 8314.46261815 / W
            # Temperature units
            t_units = nml.get_opt(sec, "temperature_units")
            # Convert temperature
            rt = 5/9 if t_units == "Rankine" else 1.0
            T = t * rt
            # Sound speed
            a = np.sqrt(g*R*T)
            # Mach nmumber
            return v / a
        else:
            # Read Mach number
            return nml.get_opt(sec, "mach_number")


# Find boundary TRIQ file
def GetTriqFile():
    r"""Get (create) most recent boundary ``triq`` file and its metadata

    :Call:
        >>> ftriq, n, i0, i1 = GetTriqFile()
    :Outputs:
        *ftriq*: :class:`str`
            Name of ``triq`` file
        *n*: :class:`int`
            Number of iterations included
        *i0*: :class:`int`
            First iteration in the averaging
        *i1*: :class:`int`
            Last iteration in the averaging
    :Versions:
        * 2024-12-03 ``@ddalle``: v1.0
    """
    # Instantiate runner
    runner = CaseRunner()
    # Call constituent method
    return runner.get_triq_file()


# Find boundary PLT file
def GetPltFile():
    r"""Get most recent boundary ``plt`` file and its metadata

    :Call:
        >>> fplt, n, i0, i1 = GetPltFile()
    :Outputs:
        *fplt*: :class:`str`
            Name of ``plt`` file
        *n*: :class:`int`
            Number of iterations included
        *i0*: :class:`int`
            First iteration in the averaging
        *i1*: :class:`int`
            Last iteration in the averaging
    :Versions:
        * 2016-12-20 ``@ddalle``: v1.0 (``GetPltFile``)
        * 2023-07-06 ``@ddalle``: v1.1; use ``CaseRunner``
    """
    # Instantiate runner
    runner = CaseRunner()
    # Call constituent method
    return runner.get_plt_file()


# Get best file based on glob
def GetFromGlob(fglb, fname=None):
    r"""Find the most recently edited file matching a glob

    :Call:
        >>> fname = casecntl.GetFromGlob(fglb, fname=None)
        >>> fname = casecntl.GetFromGlob(fglbs, fname=None)
    :Inputs:
        *fglb*: :class:`str`
            Glob for targeted file names
        *fglbs*: :class:`list`\ [:class:`str`]
            Multiple glob file name patterns
        *fname*: {``None``} | :class:`str`
            Optional alternate file name to consider
    :Outputs:
        *fbest*: :class:`str`
            Name of file matching glob that was most recently modified
    :Versions:
        * 2016-12-19 ``@ddalle``: v1.0
        * 2023-02-03 ``@ddalle``: v1.1; add *fname* input
        * 2023-03-26 ``@ddalle``: v1.2; multiple *fglbs*
        * 2024-06-21 ``@ddalle``: v1.3; disallow links to match globs
    """
    # Check for one or multiple globs
    if isinstance(fglb, (list, tuple)):
        # Combine list of globs
        fglob = []
        # Loop through multiples
        for fi in fglb:
            fglob.extend(glob.glob(fi))
    else:
        # List of files matching requested glob
        fglob = glob.glob(fglb)
    # Check for output file
    if fname is not None and os.path.isfile(fname):
        fglob.append(fname)
    # Reinitialize
    fglob1 = []
    # Loop through files
    for fi in fglob:
        # Check if file is a link or already present
        if os.path.islink(fi) or fi in fglob1:
            continue
        # Append
        fglob1.append(fi)
    # Check for empty glob
    if len(fglob1) == 0:
        return
    # Get modification times
    t = [os.path.getmtime(f) for f in fglob1]
    # Extract file with maximum index
    return fglob1[t.index(max(t))]


# Link best file based on name and glob
def LinkFromGlob(fname, fglb):
    r"""Link the most recent file to a generic Tecplot file name

    :Call:
        >>> casecntl.LinkFromGlob(fname, fglb)
        >>> casecntl.LinkFromGlob(fname, fglbs)
    :Inputs:
        *fname*: :class:`str`
            Name of unmarked file, like ``Components.i.plt``
        *fglb*: :class:`str`
            Glob for marked file names
        *fglbs*: :class:`list`\ [:class:`str`]
            Multiple glob file name patterns
    :Versions:
        * 2016-10-24 ``@ddalle``: v1.0
        * 2023-03-26 ``@ddalle``: v1.1; multiple *fglbs*
    """
    # Check for already-existing regular file
    if os.path.isfile(fname) and not os.path.islink(fname):
        return
    # Extract file with maximum index
    fsrc = GetFromGlob(fglb, fname=fname)
    # Exit if no matches
    if fsrc is None:
        return
    # Remove the link if necessary
    if os.path.islink(fname):
        # Check if link matches
        if os.readlink(fname) == fsrc:
            # Nothing to do
            return
        else:
            # Remove existing link to different file
            os.remove(fname)
    # Create the link if possible
    if os.path.isfile(fsrc):
        os.symlink(fsrc, fname)


# Link best Tecplot files
def LinkPLT():
    r"""Link the most recent Tecplot files to fixed file names

    :Call:
        >>> LinkPLT()
    :Versions:
        * 2016-10-24 ``@ddalle``: v1.0
        * 2023-07-06 ``@ddalle``: v1.1; use ``CaseRunner``
    """
    # Instantiate
    runner = CaseRunner()
    # Call link method
    runner.link_plt()

