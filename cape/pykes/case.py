#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.pykes.case`: Kestrel individual-case module
=======================================================

This module provides interfaces to run a single case of Kestrel from
the case folder. It also provides tools such as :func:`GetCurrentIter`,
which determines how many cases have been run.

The functions of this module expect to be run from a Kestrel case folder
created by :mod:`cape.pykes`.

"""

# Standard library
import glob
import os
import re
import shutil

# Third-party
import yaml

# Local imports
from . import cmdgen
from .jobxml import JobXML
from ..cfdx import cmdrun
from ..cfdx import case
from .. import fileutils
from .options.runctlopts import RunControlOpts


# Help message for CLI
HELP_RUN_KESTREL = r"""
``run_kestrel.py``: Run Kestrel for one phase
================================================

This script determines the appropriate phase to run for an individual
case (e.g. if a restart is appropriate, etc.), sets that case up, and
runs it.

:Call:

    .. code-block:: console

        $ run_kestrel.py [OPTIONS]
        $ python -m cape.pykes run [OPTIONS]

:Options:

    -h, --help
        Display this help message and quit

:Versions:
    * 2021-10-21 ``@ddalle``: v1.0
"""


# Template file names
XML_FILE = "kestrel.xml"
XML_FILE_GLOB = "kestrel.[0-9]*.xml"
XML_FILE_TEMPLATE = "kestrel.%02i.xml"
LOG_FILE = os.path.join("log", "perIteration.log")
STDOUT_FILE = "kestrel.out"

# Maximum number of calls to run_phase()
NSTART_MAX = 20


# --- Execution ----
def run_kestrel() -> int:
    r"""Setup and run the appropriate Kestrel command

    This function runs one phase, but may restart the case by recursing
    if settings prescribe it.

    :Call:
        >>> run_kestrel()
    :Versions:
        * 2021-10-21 ``@ddalle``: v1.0
        * 2023-07-10 ``@ddalle``: v2.0; use ``CaseRunner``
    """
    # Get a case runner
    runner = CaseRunner()
    # Run it
    return runner.run()


# Case run class
class CaseRunner(case.CaseRunner):
   # --- Class attributes ---
    # Extra attributes
    __slots__ = (
        "xml",
        "xml_j",
    )

    # Help message
    _help_msg = HELP_RUN_KESTREL

    # Names
    _modname = "pykes"
    _progname = "pykes"
    _logprefix = "run"

    # Other options
    _nstart_max = NSTART_MAX

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
            * 2023-07-10 ``@ddalle``: v1.0
        """
        self.xml = None
        self.xml_j = None

    # Main runner
    @case.run_rootdir
    def run_phase(self, j: int):
        r"""Run one pass of one phase

        :Call:
            >>> runner.run_phase(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2021-11-02 ``@ddalle``: v1.0
            * 2023-07-10 ``@ddalle``: v1.1; instance method
        """
        # Count number of times this phase has been run previously.
        nprev = len(glob.glob("run.%02i.*" % j))
        # Read settings
        rc = self.read_case_json()
        # Read XML file
        xml = self.read_xml(j)
        # Get the last iteration number
        n = self.get_iter()
        # Set restart if appropriate
        if n is not None and n > 0:
            # This is a restart
            if not xml.get_restart():
                # It should be a restart (e.g. running Phase 0 2x)
                xml.set_restart()
        else:
            # This is *not* a restart
            if xml.get_restart():
                # This should not be a restart
                xml.set_restart(False)
        # Number of requested iters for the end of this phase
        nj = rc.get_PhaseIters(j)
        # Save initial iteration number, but replacing ``None``
        n0 = 0 if n is None else n
        # Check if we should run this phase
        if nprev == 0 or n0 < nj:
            # Reset number of iterations to current + nIter
            # This doesn't work until we can figure out restart iter
            # xml.set_kcfd_iters(n0 + mj)
            # Rewrite XML file
            xml.write()
            # Get the ``csi`` command
            cmdi = cmdgen.csi(rc, j)
            # Run the command
            cmdrun.callf(cmdi, f="kestrel.out")
            # Check new iteration number
            n1 = self.get_iter()
            # Check for lack of progress
            if n1 <= n0:
                # Error message
                msg = f"Phase {j} failed to advance from iteration {n0}"
                # Mark failure
                self.mark_failure(msg)
                # STDOUT
                print(f"    {msg}")

    # --- Case status ---
    # Get current iter
    @case.run_rootdir
    def getx_iter(self):
        r"""Get the most recent iteration number

        :Call:
            >>> n = runner.getx_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *n*: :class:`int` | ``None``
                Last iteration number
        :Versions:
            * 2021-11-05 ``@ddalle``: v1.0
            * 2023-07-10 ``@ddalle``: v1.1; instance method
        """
        # Check if log file exists
        if not os.path.isfile(LOG_FILE):
            return None
        # Otherwise open file to read last line
        line = fileutils.tail(LOG_FILE, n=1)
        # Try to get iteration number
        try:
            # First entry should be iteration number
            return int(line.split()[0])
        except ValueError:
            # Some other tailing line; probably no iterations yet
            return 0

   # --- File management ---
    # Prepare files before running cycle
    def prepare_files(self, j: int):
        r"""Prepare files appropriate to run phase *j*

        :Call:
            >>> runner.prepare_files(j)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2021-11-02 ``@ddalle``: v1.0
            * 2023-07-10 ``@ddalle``: v1.1; instance method
        """
        # XML file names
        fxml0 = XML_FILE
        fxmlj = XML_FILE_TEMPLATE % j
        # Check for "kestrel.xml" file
        if os.path.isfile(fxml0) or os.path.islink(fxml0):
            os.remove(fxml0)
        # Check for *j*
        if not os.path.isfile(fxmlj):
            raise OSError("Couldn't find file '%s'" % fxmlj)
        # Link "kestrel.02.xml" to "kestrel.xml", for example
        os.symlink(fxmlj, fxml0)

    # Clean up files after running one cycle
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
            * 2021-11-05 ``@ddalle``: v1.0
            * 2023-07-10 ``@ddalle``: v1.1; instance method
        """
        # Get the last iteration number
        n = self.get_iter()
        # Don't use ``None`` for this
        if n is None:
            n = 0
        # Name of history file
        fhist = "run.%02i.%i" % (j, n)
        # Assuming that worked, move the temp output file.
        if os.path.isfile(STDOUT_FILE):
            # Copy the file
            shutil.copy(STDOUT_FILE, fhist)
        else:
            # Create an empty file
            open(fhist, 'w').close()

   # --- Local readers ---
    # Read XML file
    def read_xml(self, j=None):
        r"""Read Kestrel ``.xml`` control file for one phase

        :Call:
            >>> xml = read_xml(rc=None, j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *xml*: :class:`JobXML`
                XML control file interface
        :Versions:
            * 2021-11-02 ``@ddalle``: v1.0
            * 2023-07-10 ``@ddalle``: v1.1; instance method
        """
        # Automatic phase option
        if j is None:
            j = self.get_phase()
        # Check for folder w/o "case.json"
        if j is None:
            if os.path.isfile(XML_FILE):
                # Use currently-linked file
                return JobXML(XML_FILE)
            else:
                # Look for template files
                xmlglob = glob.glob(XML_FILE_GLOB)
                # Sort it
                xmlglob.sort()
                # Use the last one
                return JobXML(xmlglob[-1])
        # Get specified version
        return JobXML(XML_FILE_TEMPLATE % j)


# --- File Management ---
# Get best file based on glob
def get_glob_latest(fglb):
    r"""Find the most recently edited file matching a glob

    :Call:
        >>> fname = get_glob_latest(fglb)
    :Inputs:
        *fglb*: :class:`str`
            Glob for targeted file names
    :Outputs:
        *fname*: :class:`str`
            Name of file matching glob that was most recently edited
    :Versions:
        * 2016-12-19 ``@ddalle``: v1.0
        * 2022-01-28 ``@ddalle``: v1.1; was GetFromGlob()
    """
    # List of files matching requested glob
    fglob = glob.glob(fglb)
    # Check for empty glob
    if len(fglob) == 0:
        return
    # Get modification times
    t = [os.path.getmtime(f) for f in fglob]
    # Extract file with maximum index
    fname = fglob[t.index(max(t))]
    # Output
    return fname


# Link best file based on name and glob
def link_glob_latest(fname, fglb):
    r"""Link the most recent file to a generic Tecplot file name

    :Call:
        >>> link_glob_latest(fname, fglb)
    :Inputs:
        *fname*: :class:`str`
            Name of unmarked file, like ``Components.i.plt``
        *fglb*: :class:`str`
            Glob for marked file names
    :Versions:
        * 2016-10-24 ``@ddalle``: v1.0
        * 2022-01-28 ``@ddalle``: v1.1; was LinKFromGlob()
    """
    # Check for already-existing regular file
    if os.path.isfile(fname) and not os.path.islink(fname):
        return
    # Remove the link if necessary
    if os.path.islink(fname):
        os.remove(fname)
    # Extract file with maximum index
    fsrc = get_glob_latest(fglb)
    # Exit if no matches
    if fsrc is None:
        return
    # Create the link if possible
    if os.path.isfile(fsrc):
        os.symlink(os.path.basename(fsrc), fname)


# Link best Tecplot files
def link_plt():
    r"""Link the most recent Tecplot files to fixed file names

    :Call:
        >>> link_plt(fdir=None)
    :Inputs:
        *fdir*: {``None``} | :class:`str`
            Specific folder in which to find latest file
    :Versions:
        * 2022-01-28 ``@ddalle``: v1.0
    """
    # Get case runner
    runner = CaseRunner()
    # Determine phase
    j = runner.get_phase()
    # Need the namelist to figure out planes, etc.
    xml = runner.read_xml(j)
    # Get the project root name
    proj = xml.get_job_name()
    # Name of file containing list of Tecplot exports
    fmg = os.path.join("outputs", "visualization", "%s.mg" % proj)
    # Exit if no such file
    if not os.path.isfile(fmg):
        return
    # Try to read the .mg file containing info about each PLT
    opts = yaml.load(open(fmg), Loader=yaml.CLoader)
    # Loop through expected data sources
    for data_source in opts.get("DataSources", []):
        # Check type
        if not isinstance(data_source, dict):
            continue
        # Check data type
        if data_source.get("DataType") != "Tecplot":
            continue
        # Get solution
        sol = data_source.get("Solution")
        if not isinstance(sol, dict):
            continue
        # Get path
        fpath = sol.get("Path")
        if fpath is None or "%ts" not in fpath:
            continue
        # Full path
        frel = os.path.join(
            "outputs", "visualization", fpath.replace("/", os.sep))
        # Substitute
        fglob = re.sub("%ts", "[0-9]*", frel)
        fname = re.sub("%ts", "", frel)
        # Link the latest
        link_glob_latest(fname, fglob)
    # Loop through any slices, surface extracts, or other VizMan outputs
    for fname, fglob in _get_vizman_candidates():
        link_glob_latest(fname, fglob)


def _get_vizman_candidates():
    # Initialize
    glob_list = []
    # Unstructured globs
    base = os.path.join(
        "outputs", "visualization", "Unstructured", "SurfaceExtract")
    if os.path.isdir(base):
        # Append both file candidates
        fname = os.path.join(base, "UnstructuredSurf.")
        fglob = os.path.join(base, "UnstructuredSurf_[0-9]*.")
        glob_list.append((fname + "tec", fglob + "tec"))
        glob_list.append((fname + "plt", fglob + "plt"))
    # Check for cut planes
    base = os.path.join(
        "outputs", "visualization", "Unstructured", "coordPlane")
    for fj in os.listdir(base):
        # Full relative path
        basej = os.path.join(base, fj)
        # Check for regular files
        if not os.path.isdir(basej):
            continue
        # Create candidates
        fname = os.path.join(basej, "Unstructured.")
        fglob = os.path.join(basej, "Unstructured_[0-9]*.")
        glob_list.append((fname + "tec", fglob + "tec"))
        glob_list.append((fname + "plt", fglob + "plt"))
    # Output
    return glob_list

