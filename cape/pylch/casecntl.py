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
import h5py

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

    # Prepare files for a case
    def prepare_files(self, j: int):
        r"""Prepare files and links to run phase *j* of Loci/CHEM

        :Call:
            >>> runner.prepare_files(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2025-05-02 ``@ddalle``: v1.0
        """
        # Get project name
        proj = self.get_project_rootname()
        # Link to pylch.vars
        self.link_file(f"{proj}.{j:02d}.vars", f"{proj}.vars", f=True)

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
            return int(line.split(maxsplit=1)[0])
        except Exception:
            return 0

   # --- Custom settings ---
    def get_project_rootname(self, j: Optional[int] = None):
        r"""Get the project name for a Loci/CHEM case

        :Call:
            >>> proj = runner.get_project_rootname(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *proj*: :class:`str`
                Project name, usually ``"pylch"``
        :Versions:
            * 2025-05-02 ``@ddalle``: v1.0
        """
        # Read options
        rc = self.read_case_json()
        # # Get option
        return rc.get_opt("ProjectName", j=j, vdef="pylch")

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

    def write_mapbc2vog(self, fvog: str, j: int = 0):
        r"""Add MapBC to *fvog* VOG file and write"""
        # Ensure cntl
        cntl = getattr(self, "cntl", None)
        if cntl is None:
            # Read in cntl
            cntl = self.read_cntl()
        # Ensure mapbc
        mapbc = getattr(cntl, "MapBC", None)
        if mapbc is None:
            # Read in MapBC
            cntl.ReadMapBC(j)
        # Read mesh file
        with h5py.File(fvog, 'r+') as fp:
            mbcg = fp["surface_info"].create_group("mapbc")
            # Add mapbc to surface_info group
            mbcg.create_dataset("names", data=cntl.MapBC.Names)
            mbcg.create_dataset("surfid", data=cntl.MapBC.SurfID)
            mbcg.create_dataset("compid", data=cntl.MapBC.CompID)
            mbcg.create_dataset("bcs", data=cntl.MapBC.BCs)
