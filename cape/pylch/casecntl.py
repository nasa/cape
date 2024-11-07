r"""
:mod:`cape.pylch.case`: Loci/CHEM case control module
=====================================================

This module contains Loci/CHEM-specific versions of some of the generic
methods from :mod:`cape.cfdx.case`.
"""

# Standard library modules
import os
from typing import Optional

# Third-party modules

# Local imports
from . import cmdgen
from .. import fileutils
from .varsfile import VarsFile
from .options.runctlopts import RunControlOpts
from ..cfdx import casecntl
from ..fileutils import tail

# Constants
ITER_FILE = "data.iter"


# Function to complete final setup and call the appropriate LAVA commands
def run_chem():
    r"""Setup and run the appropriate LAVACURV command

    :Call:
        >>> run_chem()
    :Versions:
        * 2024-10-17 ``@ddalle``: v1.0;
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
    _modname = "pylch"
    _progname = "chem"

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
            * 2024-10-17 ``@ddalle``: v1.0
        """
        # Run main executable
        self.run_chem(j)

    # Run chem one time
    @casecntl.run_rootdir
    def run_chem(self, j: int):
        r"""Run one phase of the ``chem`` executable

        :Call:
            >>> runner.run_superlava(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2024-10-17 ``@ddalle``: v1.0
        """
        # Read case settings
        rc = self.read_case_json()
        # Generate command
        cmdi = cmdgen.chem(rc, j)
        # Run the command
        self.callf(cmdi, f="chem.out", e="chem.err")

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
            * 2024-10-17 ``@ddalle``: v1.0
        """
        # Get the current iteration number
        n = self.get_iter()
        # Genrate name of STDOUT log, "run.{phase}.{n}"
        fhist = "run.%02i.%i" % (j, n)
        # Rename the STDOUT file
        if os.path.isfile("chem.out"):
            # Move the file
            os.rename("chem.out", fhist)
        else:
            # Create an empty file
            fileutils.touch(fhist)

    # Get current iteration
    @casecntl.run_rootdir
    def getx_iter(self) -> Optional[int]:
        r"""Get the most recent iteration number for Loci/CHEM case

        :Call:
            >>> n = runner.getx_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *n*: :class:`int` | ``None``
                Last iteration number
        :Versions:
            * 2024-10-17 ``@ddalle``: v1.0
        """
        # Residual file
        resid_file = os.path.join("output", "resid.dat")
        # Check for file
        if not os.path.isfile(resid_file):
            return
        # Read the last line
        line = tail(resid_file)
        # Parse first integer
        try:
            return int(line.split(1)[0])
        except Exception:
            return 0

   # --- Special readers ---
    # Read namelist
    @casecntl.run_rootdir
    def read_varsfile(self, j: Optional[int] = None) -> VarsFile:
        r"""Read case ``.vars`` file

        :Call:
            >>> yamlfile = runner.read_varsfile(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *varsfile*: :class:`VarsFIle`
                Loci/CHEM ``.vars`` file interface
        :Versions:
            * 2024-11-07 ``@ddalle``: v1.0
        """
        # Read ``case.json`` if necessary
        rc = self.read_case_json()
        # Process phase number
        if j is None and rc is not None:
            # Default to most recent phase number
            j = self.get_phase()
        # Get phase of namelist previously read
        yamlj = self.varsfile_j
        # Check if already read
        if isinstance(self.varsfile, VarsFile):
            if yamlj == j and j is not None:
                # Return it!
                return self.varsfile
        # Get name of file to read
        fbase = rc.get_lava_varsfile()
        fname = cmdgen.infix_phase(fbase, j)
        # Read it
        self.varsfile = VarsFile(fname)
        # Return it
        return self.varsfile
