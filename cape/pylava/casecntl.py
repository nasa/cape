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
import numpy as np
import yaml

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
        # Generate command
        cmdi = cmdgen.superlava(rc, j)
        # Run the command
        self.callf(cmdi, f="superlava.out", e="superlava.err")

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
                Last iteration number
            *n*: :class:`int` | ``None``
        :Versions:
            * 2024-08-02 ``@sneuhoff``: v1.0
            * 2024-10-11 ``@ddalle``: v2.0; use DataIterFile(meta=True)
        """
        # Path to file
        fname = os.path.join(self.root_dir, ITER_FILE)
        # Read it, but only metadata
        db = DataIterFile(fname, meta=True)
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
    def check_complete(self) -> bool:
        r"""Check if a case is complete (DONE)
        Adds residual convergence as a stopping
        condition to cfdx.check_complete

        :Call:
            >>> q = runner.check_complete()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Versions:
            * 2024-09-16 ``@sneuhoff``: v1.0
        """
        # Read case JSON
        rc = self.read_case_json()
        # Determine current phase
        j = self.get_phase(rc)
        # Check if final phase
        if j < self.get_last_phase():
            return False
        # Get absolute iter
        n = self.get_iter()
        # Get restart iter (calculated in get_phase->get_restart_iter)
        nr = self.nr
        # Check for stop iteration
        qstop, nstop = self.get_stop_iter()
        # Check for convergence
        inp = self.read_case_inputfile()
        l2conv = float(inp['nonlinearsolver']['l2conv'])
        iterdata = self.read_data_iter()
        flowres = iterdata['flowres']
        if flowres[-1]/flowres[0] <= l2conv:
            return True
        # Check iteration number
        if nr is None:
            # No iterations complete
            return False
        elif qstop and (n >= nstop):
            # Stop requested by user
            return True
        elif nr < rc.get_LastIter():
            # Not enough iterations complete
            return False
        else:
            # All criteria met
            return True

    # Read input yaml
    def read_case_inputfile(self):
        r"""Read ``run.yaml``

        :Call:
            >>> inp = runner.read_case_inputfile()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *inp*: :class:`yaml`
                YAML dictionary of input file options
        :Versions:
            * 2024-09-16 ``@sneuhoff``: v1.0
        """
        # Read case JSON
        rc = self.read_case_json()
        # Absolute path
        fname = os.path.join(self.root_dir, rc.get_RunYaml())
        # Check current file
        if os.path.isfile(fname):
            with open(fname, 'r') as f:
                inp = yaml.safe_load(f)
        return inp

    def read_data_iter(self, fName="data.iter"):
        r"""From LAVA's plotHist.py: Read data from a history file.
        Should return a dict.

        :Versions:
            * 2024-08-02 ``@sneuhoff``; v1.0
        """
        fSize = os.path.getsize(fName)
        with open(fName, 'rb') as f:
            nVar = np.fromfile(f, dtype='i4', count=1)
            if len(nVar) == 0:
                raise ValueError('No data in provided history file')
            else:
                nVar = nVar[0]
            strSize = np.fromfile(f, dtype='i4', count=1)[0]
            strs = np.fromfile(f, dtype='c', count=nVar*strSize)
            keys = []

            for i in range(nVar):
                tmpStr = ""
                for j in range(strSize):
                    tmpStr = tmpStr + strs[i*strSize + j].decode('UTF-8')
                keys.append(tmpStr.strip().lower())

            # determine number of lines to read, in case write got interrupted
            remainingBytes = fSize - 8 - nVar * strSize
            lines = remainingBytes // (nVar * 8)

            buf = np.fromfile(f, dtype=float, count=-1, sep='')
            data = buf[0:lines*nVar].reshape((nVar, lines), order='f')

            # Convert to dict:
            dataDict = {}
            for i in range(len(keys)):
                dataDict[keys[i]] = data[i, :]

        return dataDict
