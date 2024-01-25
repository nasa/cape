r"""
This module contains the important function :func:`case.run_fun3d`,
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

# Third-party modules
import numpy as np

# Local imports
from . import cmdrun
from . import cmdgen
from .. import fileutils
from ..cfdx import case
from .options.runctlopts import RunControlOpts
from .namelist import Namelist


# Regular expression to find a line with an iteration
_regex_dict = {
    "time": "(?P<time>[1-9][0-9]*)",
    "iter": "(?P<iter>[1-9][0-9]*)",
}
# Combine them; different format for steady and time-accurate modes
REGEX_F3DOUT = re.compile(r"\s*%(time)s?\s+%(iter)s\s{2,}[-0-9]" % _regex_dict)

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
class CaseRunner(case.CaseRunner):
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
    @case.run_rootdir
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
        """
        # Read settings
        rc = self.read_case_json()
        # Count number of times this phase has been run previously.
        nprev = len(glob.glob('run.%02i.*' % j))
        # Check for dual
        if rc.get_Dual():
            os.chdir('Flow')
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
        # Mesh generation and verification actions
        if j == 0 and n is None:
            # Run intersect and verify
            self.run_intersect(j, fproj)
            self.run_verify(j, fproj)
            # Create volume mesh if necessary
            self.run_aflr3(j, fproj, fmt=nml.GetGridFormat())
            # Check for mesh-only phase
            if nj is None or ni is None or ni <= 0 or nj < 0:
                # Name of next phase
                fproj_adapt = self.get_project_rootname(j+1)
                # AFLR3 output format
                fmt = nml.GetGridFormat()
                # Check for renamed file
                if fproj_adapt != fproj:
                    # Copy mesh
                    os.symlink(
                        '%s.%s' % (fproj, fmt),
                        '%s.%s' % (fproj_adapt, fmt))
                # Make sure *n* is not ``None``
                if n is None:
                    n = 0
                # Exit appropriately
                if rc.get_Dual():
                    os.chdir('..')
                # Create an output file to make phase number programs work
                fileutils.touch("run.%02i.%i" % (j, n))
                return
        # Prepare for restart if that's appropriate
        self.set_restart_iter()
        # Get *n* but ``0`` instead of ``None``
        n0 = 0 if (n is None) else n
        # Check if the primal solution has already been run
        if nprev == 0 or n0 < nj:
            # Get the `nodet` or `nodet_mpi` command
            cmdi = cmdgen.nodet(rc, j=j)
            # Call the command.
            cmdrun.callf(cmdi, f='fun3d.out')
            # Get new iteration number
            n1 = self.get_iter()
            # Check for lack of progress
            if n1 <= n0:
                # Mark failure
                self.mark_failure(f"No advance from iter {n0} in phase {j}")
                # Raise an exception for run()
                raise SystemError(
                    f"Cycle of phase {j} did not advance iteration count.")
            if len(glob.glob("nan_locations*.dat")):
                # Mark failure
                self.mark_failure("Found NaN location files")
                raise SystemError("Found NaN location files")
        else:
            # No new iteratoins
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
            cmdrun.callf(cmdi, f='dual.out')
            # Create the command to adapt
            cmdi = cmdgen.dual(rc, i=j, adapt=True)
            # Estimate error and adapt
            cmdrun.callf(cmdi, f='dual.out')
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
            # Run refine distance

            # Return home if appropriate
            if rc.get_Dual():
                os.chdir('..')

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
        # Formulate kw inputs for command line
        # There needs to be a call to the refine_translate_opts class
        kw_translate = {
            "function": "translate",
            "input_grid": 'pyfun%02i.lb8.ugrid' % j,
            "output_grid": 'pyfun%02i.meshb' % j
        }
        # Get project name
        fproj = self.get_project_rootname(j)
        # TODO: determine grid format
        # Set options to *rc* to save for command-line generation
        rc.set_RefineTranslateOpt("input_grid", f'{fproj}.lb8.ugrid')
        # Run the refine translate command
        cmdi = cmdgen.refine_translate(rc, i=j)
        # Call the command
        cmdrun.callf(cmdi, f="refine-translate.out")

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
        # Make sure "restart_read" is set to .true.
        nml.SetRestart(True)
        nml.write('fun3d.%02i.nml' % j)
        # Call the command.
        cmdrun.callf(cmdi, f='adapt.out')
        # Rename output file after completing that command
        os.rename('adapt.out', 'adapt.%02i.out' % j)

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
            # Set the restart flag on.
            nml.SetRestart(nohist=nohist)
        else:
            # Check for warm-start flag
            warmstart = self.prepare_warmstart()
            # Set the restart flag on/off depending on warm-start config
            nml.SetRestart(warmstart)
        # Write the namelist.
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
    @case.run_rootdir
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
            proj = proj0 + "??"
        else:
            # Use the full project name if no adaptations
            proj0 = proj
        # Get the list of output surfaces
        fsrf = nml.get_opt("sampling_parameters", "label")
        # Initialize file names
        fname = [
            '%s_tec_boundary' % proj0,
            '%s_volume' % proj0,
            '%s_volume' % proj0
        ]
        # Initialize globs
        fglob = [
            ['%s_tec_boundary_timestep*' % proj],
            ['%s_volume_timestep*' % proj],
            ['%s_volume' % proj]
        ]
        # Add special ones
        for fi in fsrf:
            fname.append('%s_%s' % (proj0, fi))
            fglob.append(
                ['%s_%s' % (proj, fi), '%s_%s_timestep*' % (proj, fi)])
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
    def get_project_rootname(self, j=None):
        r"""Read namelist and return project namelist

        :Call:
            >>> rname = runner.get_project_rootname(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase number
            *nml*: :class:`cape.pyfun.namelist.Namelist`
                Namelist interface; overrides *rc* and *i* if used
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

   # --- Special readers ---
    # Read namelist
    @case.run_rootdir
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
    # Find boundary PLT file
    def get_plt_file(self):
        r"""Get most recent boundary ``plt`` file and its metadata

        :Call:
            >>> fplt, n, i0, i1 = runner.get_plt_file()
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
        """
        # Read *rc* options to figure out iteration values
        rc = self.read_case_json()
        # Get current phase number
        j = self.get_phase()
        # Read the namelist to get prefix and iteration options
        nml = self.read_namelist(j)
        # =============
        # Best PLT File
        # =============
        # Prefix
        proj = self.get_project_rootname(j)
        # Create glob to search for
        fglb = '%s_tec_boundary_timestep[1-9]*.plt' % proj
        # Check in working directory?
        if rc.get_Dual():
            # Look in the 'Flow/' folder
            fglb = os.path.join('Flow', fglb)
        # Get file
        fplt = GetFromGlob(fglb)
        # Check for nothing...
        if fplt is None:
            # Check if we can fall back to a previous project
            if glob.fnmatch.fnmatch(proj, '*[0-9][0-9]'):
                # Allow any project
                fglb = f'{proj[:-2]}[0-9][0-9]_tec_boundary_timestep[1-9]*.plt'
                # Try again
                fplt = GetFromGlob(fglb)
                # Check for second-try miss
                if fplt is None:
                    return None, None, None, None
            else:
                # No file, global project name
                return None, None, None, None
        # Get the iteration number
        nplt = int(fplt.rstrip('.plt').split('timestep')[-1])
        # ============================
        # Actual Iterations after Runs
        # ============================
        # Glob of ``run.%02i.%i`` files
        fgrun = glob.glob('run.[0-9][0-9].[1-9]*')
        # Form dictionary of iterations
        nrun = []
        drun = {}
        # Loop through files
        for frun in fgrun:
            # Get iteration number
            ni = int(frun.split('.')[2])
            # Get phase number
            ji = int(frun.split('.')[1])
            # Save
            nrun.append(ni)
            drun[ni] = ji
        # Sort on iteration number
        nrun.sort()
        nrun = np.array(nrun)
        # Determine the phase that ended before this file was created
        krun = np.where(nplt > nrun)[0]
        # If no 'run.%02i.%i' before *nplt*, then use 0
        if len(krun) == 0:
            # Use current phase as reported
            nprev = 0
            nstrt = 1
            jstrt = j
        else:
            # Get the phase from the last run that finished before *nplt*
            kprev = krun[-1]
            nprev = nrun[kprev]
            jprev = drun[nprev]
            # Have we moved to the next phase?
            if nprev >= rc.get_PhaseIters(jprev):
                # We have *nplt* from the next phase
                mprev = rc.get_PhaseSequence().index(jprev)
                jstrt = rc.get_PhaseSequence(mprev+1)
            else:
                # Still running phase *jprev* to create *fplt*
                jstrt = jprev
            # First iteration included in PLT file
            nstrt = nprev + 1
        # Make sure we have the right namelist
        if j != jstrt:
            # Read the new namelist
            nml = self.read_namelist(jstrt)
        # ====================
        # Iteration Statistics
        # ====================
        # Check for averaging
        qavg = nml.get_opt('time_avg_params', 'itime_avg')
        # Number of iterations
        if qavg:
            # Time averaging included
            nStats = nplt - nprev
        else:
            # One iteration
            nStats = 1
            nstrt = nplt
        # ======
        # Output
        # ======
        return fplt, nStats, nstrt, nplt

   # --- Status ---
    # Function to chose the correct input to use from the sequence.
    def getx_phase(self, n: int):
        r"""Determine the phase number based on files in folder

        :Call:
            >>> i = case.GetPhaseNumber(rc)
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
        # Loop through possible input numbers.
        for i, j in enumerate(rc.get_PhaseSequence()):
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
    def check_error(self):
        r"""Check for errors before continuing

        Currently the following checks are performed.

            * Check for NaN residual in the output file

        :Call:
            >>> ierr = runner.check_error())
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
                return case.IERR_NANS
        # Otherwise no errors detected
        return case.IERR_OK

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
        """
        # Read the two sources
        nh, ns = self.getx_iter_history()
        nr = self.getx_iter_running()
        # Process
        if nr in (0, None):
            # No running iterations; check history
            return ns
        else:
            # Some iterations saved and some running
            return nh + nr

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
    @case.run_rootdir
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
        if qdual or qadpt:
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
                elif len(fname.split('.')) == 3:
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
    @case.run_rootdir
    def getx_iter_running(self):
        r"""Get the most recent iteration number for a running file

        :Call:
            >>> n = case.GetRunningIter()
        :Outputs:
            *n*: :class:`int` | ``None``
                Most recent iteration number
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0
            * 2016-04-28 ``@ddalle``: v1.1; handle ``Flow/`` folder
            * 2023-05-27 ``@ddalle``; v2.0; instance method
        """
        # Check for the file.
        if os.path.isfile('fun3d.out'):
            # Use the current folder
            fflow = 'fun3d.out'
        elif os.path.isfile(os.path.join('Flow', 'fun3d.out')):
            # Use the ``Flow/`` folder
            fflow = os.path.join('Flow', 'fun3d.out')
        else:
            # No current file
            return None
        # Check for flag to ignore restart history
        lines = fileutils.grep('on_nohistorykept', fflow)
        # Check whether or not to add restart iterations
        if len(lines) < 1:
            # Get the restart iteration line
            try:
                # Search for particular text
                lines = fileutils.grep('the restart files contains', fflow)
                # Process iteration count from the RHS of the last such line
                nr = int(lines[0].split('=')[-1])
            except Exception:
                # No restart iterations
                nr = None
        else:
            # Do not use restart iterations
            nr = None
        # Length of chunk at end of line to check
        nchunk = 10
        # Maximum number of chunks to scan
        mchunk = 8
        # Loop until chunk found with iteration number
        for ichunk in range(mchunk):
            # Get (cumulative) size of chunk and previous chunk
            ia = ichunk * nchunk
            ib = ia + nchunk
            # Get the last few lines of :file:`fun3d.out`
            lines = fileutils.tail(fflow, ib).strip().split('\n')
            lines.reverse()
            # Initialize output
            n = None
            # Try each line
            for line in lines[ia:]:
                try:
                    # Check for direct specification
                    if 'current history iterations' in line:
                        # Direct specification
                        n = int(line.split()[-1])
                        nr = None
                        break
                    # Use the iteration regular expression
                    match = REGEX_F3DOUT.match(line)
                    # Check for match
                    if match:
                        # Get the iteration number from the line
                        n = int(match.group('iter'))
                        # Search completed
                        break
                except Exception:
                    continue
            # Exit if valid line was found
            if n is not None:
                break
        # Output
        if n is None:
            return nr
        elif nr is None:
            return n
        else:
            return n + nr


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
        >>> fname = case.GetFromGlob(fglb, fname=None)
        >>> fname = case.GetFromGlob(fglbs, fname=None)
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
    # Check for empty glob
    if len(fglob) == 0:
        return
    # Get modification times
    t = [os.path.getmtime(f) for f in fglob]
    # Extract file with maximum index
    return fglob[t.index(max(t))]


# Link best file based on name and glob
def LinkFromGlob(fname, fglb):
    r"""Link the most recent file to a generic Tecplot file name

    :Call:
        >>> case.LinkFromGlob(fname, fglb)
        >>> case.LinkFromGlob(fname, fglbs)
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

