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

# Local imports
from . import cmdgen
from .. import fileutils
from .databook import CaseFM, CaseResid
from .dataiterfile import DataIterFile
from .runinpfile import CartInputFile
from .yamlfile import RunYAMLFile
from .options.runctlopts import RunControlOpts
from ..cfdx import casecntl

# Constants
ITER_FILE = "data.iter"
ITER_FILE_CART = os.path.join("monitor", "Cart.data.iter")


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
    def check_early_exit(self, j: Optional[int] = None) -> bool:
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
            opts["sover defaults"]["restart"].pop("file")
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
            # Get groups using archivist
            vgrp = a.search_regex(r"/(.+)\.[0-9]+\.([a-z0-9]+)")
            # Loop through keys:
            for fnstr in vgrp.keys():
                # Parse the filename to link to
                parse = re.findall(r"'(.*?)'", fnstr)
                # Append the output name and last file
                fname = f'{parse[0]}.{parse[1]}'
                fsrc = vgrp[fnstr][-1]
                # Link the files
                self.link_file(fname, fsrc)

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
        # Read it
        self.runinpfile = CartInputFile(fname)
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
        if not q:
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
