r"""
:mod:`cape.pylava.case`: LAVACURV case control module
=====================================================

This module contains LAVACURV-specific versions of some of the generic
methods from :mod:`cape.cfdx.case`.

All of the functions from :mod:`cape.case` are imported here.  Thus
they are available unless specifically overwritten by specific
:mod:`cape.pylava` versions.
"""

# Standard library modules
import os
from typing import Optional

# Third-party modules

# Local imports
from . import cmdgen
from .. import fileutils
from .dataiterfile import DataIterFile
from .yamlfile import RunYAMLFile
from .options.runctlopts import RunControlOpts
from ..cfdx import casecntl

# Constants
ITER_FILE = "data.iter"


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
        "yamlfile",
        "yamlfile_j",
    )

    # Names
    _modname = "pylava"
    _progname = "lavacurv"

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
        # Rename the STDOUT file
        if os.path.isfile("superlava.out"):
            # Move the file
            os.rename("superlava.out", fhist)
        else:
            # Create an empty file
            fileutils.touch(fhist)

    # Function to get total iteration number
    def getx_restart_iter(self):
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
        return self.getx_iter()

    # Get current iteration
    def getx_iter(self):
        r"""Get the most recent iteration number for LAVACURV case

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

   # --- Special readers ---
    # Read namelist
    @casecntl.run_rootdir
    def read_runyaml(self, j: Optional[int] = None) -> RunYAMLFile:
        r"""Read case namelist file

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
        # Read YAML file
        yamlfile = self.read_runyaml()
        # Maximum iterations
        maxiters = yamlfile.get_lava_subopt("nonlinearsolver", "iterations")
        if db.n >= maxiters:
            return True
        # Target convergence
        l2conv_target = yamlfile.get_lava_subopt("nonlinearsolver", "l2conv")
        # Apply it
        if l2conv_target:
            # Check reported convergence
            return db.l2conv <= l2conv_target
        else:
            # No convergence test
            return True
        # Perform parent check
        q = casecntl.CaseRunner.check_complete(self)
        # Quit if not complete
        if not q:
            return q

    @casecntl.run_rootdir
    def read_data_iter(
            self,
            fname: str = ITER_FILE,
            meta: bool = True) -> DataIterFile:
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
        :Versions:
            * 2024-08-02 ``@sneuhoff``; v1.0
            * 2024-10-11 ``@ddalle``: v2.0
        """
        # Check if file exists
        if os.path.isfile(fname):
            return DataIterFile(fname, meta=meta)
        else:
            # Empty instance
            return DataIterFile(None)
