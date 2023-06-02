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
import sys

# Third-party
import yaml

# Local imports
from . import cmdgen
from .. import argread
from .. import text as textutils
from .jobxml import JobXML
from ..cfdx import bin
from ..cfdx import case as cc
from ..cfdx import queue
from ..tnakit import fileutils
from .options.runcontrol import RunControl


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

# Maximum number of calls to run_kestrel()
NSTART_MAX = 20


# --- Execution ----
def run_kestrel():
    r"""Setup and run the appropriate Kestrel command

    This function runs one phase, but may restart the case by recursing
    if settings prescribe it.

    :Call:
        >>> run_kestrel()
    :Versions:
        * 2021-10-21 ``@ddalle``: v0.1; started
    """
    # Process arguments
    a, kw = argread.readkeys(sys.argv)
    # Check for help argument.
    if kw.get('h') or kw.get('help'):
        # Display help and exit
        print(textutils.markdown(HELP_RUN_KESTREL))
        return cc.IERR_OK
    # Start RUNNING and initialize timer
    tic = cc.init_timer()
    # Read settings
    rc = read_case_json()
    # Initialize number of calls
    nstart = 0
    # Loop until complete or aborted by resubmission
    while nstart < NSTART_MAX:
        # Get phase number
        j = get_phase(rc)
        # Write the start time
        write_starttime(tic, rc, j)
        # Prepare files
        prepare_files(rc, j)
        # Prepare environment
        cc.PrepareEnvironment(rc, j)
        # Run appropriate commands
        try:
            run_phase(rc, j)
        except Exception:
            # Failure
            cc.mark_failure("run_phase")
            # Stop running marker
            cc.mark_stopped()
            # Return code
            return cc.IERR_RUN_PHASE
        # Clean up files
        finalize_files(rc, j)
        # Write timing data
        write_usertime(tic, rc, j)
        # Attempt counter
        nstart += 1
        # Explicit exit
        if check_complete(rc):
            break
        # Submit new PBS/Slurm job if appropriate
        q = resubmit_case(rc, j)
        # If new job started, this one should stop
        if q:
            break
    # Remove the RUNNING file
    cc.mark_stopped()
    # Return code
    return cc.IERR_OK


def run_phase(rc, j):
    r"""Run one pass of one phase

    :Call:
        >>> run_phase(rc, j)
    :Inputs:
        *rc*: :class:`RunControl`
            Options interface from ``case.json``
        *j*: :class:`int`
            Phase number
    :Versions:
        * 2021-11-02 ``@ddalle``: v1.0
    """
    # Count number of times this phase has been run previously.
    nprev = len(glob.glob("run.%02i.*" % j))
    # Read XML file
    xml = read_xml(rc, j)
    # Get the last iteration number
    n = get_current_iter()
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
        bin.callf(cmdi, f="kestrel.out")
        # Check new iteration number
        n1 = get_current_iter()
        # Check for lack of progress
        if n1 <= n0:
            raise SystemError(
                "Running phase %i did not advance iteration count" % j)


def resubmit_case(rc, j0):
    r"""Resubmit a case as a new job if appropriate

    :Call:
        >>> q = resubmit_case(rc, j0)
    :Inputs:
        *rc*: :class:`RunControl`
            Options interface from ``case.json``
        *j0*: :class:`int`
            Index of phase most recently run prior
            (may differ from :func:`get_phase` now)
    :Outputs:
        *q*: ``True`` | ``False``
            Whether or not a new job was submitted to queue
    :Versions:
        * 2022-01-20 ``@ddalle``: v1.0
    """
    # Get *current* phase
    j1 = get_phase(rc)
    # Job submission options
    qsub0 = rc.get_qsub(j0) or rc.get_slurm(j0)
    qsub1 = rc.get_qsub(j1) or rc.get_slurm(j1)
    # Trivial case if phase *j* is not submitted
    if not qsub1:
        return False
    # Check if *j1* is submitted and not *j0*
    if not qsub0:
        # Submit new phase
        _submit_job(rc, j1)
        return True
    # If rerunning same phase, check the *Continue* option
    if j0 == j1:
        if rc.get_Continue(j0):
            # Don't submit new job (continue current one)
            return False
        else:
            # Rerun same phase as new job
            _submit_job(rc, j1)
            return True
    # Now we know we're going to new phase; check the *Resubmit* opt
    if rc.get_Resubmit(j0):
        # Submit phase *j1* as new job
        _submit_job(rc, j1)
        return True
    else:
        # Continue to next phase in same job
        return False


def _submit_job(rc, j):
    # Get name of PBS script
    fpbs = get_pbs_script(j)
    # Check submission type
    if rc.get_qsub(j):
        # Submit PBS job
        return queue.pqsub(fpbs)
    elif rc.get_qsbatch(j):
        # Submit slurm job
        return queue.pqsbatch(fpbs)


def start_case(rc=None, j=None):
    r"""Start a case by either submitting it or calling locally

    :Call:
        >>> start_case()
    :Versions:
        * 2021-11-05 ``@ddalle``: v1.0
    """
    # Get the config
    rc = read_case_json()
    # Determine the run index.
    j = get_phase(rc)
    # Check qsub status.
    if rc.get_slurm(j):
        # Get the name of the PBS file
        fpbs = get_pbsscript(j)
        # Submit the Slurm case
        pbs = queue.psbatch(fpbs)
        return pbs
    elif rc.get_qsub(j):
        # Get the name of the PBS file.
        fpbs = get_pbsscript(j)
        # Submit the case.
        pbs = queue.pqsub(fpbs)
        return pbs
    else:
        # Simply run the case. Don't reset modules either.
        run_kestrel()


def check_complete(rc):
    r"""Check if case is complete as described

    :Call:
        >>> q = check_complete(rc)
    :Inputs:
        *rc*: :class:`RunControl`
            Options interface from ``case.json``
    :Outputs:
        *q*: ``True`` | ``False``
            Whether case has reached last phase w/ enough iters
    :Versions:
        * 2022-01-20 ``@ddalle``: v1.0
    """
    # Determine current phase
    j = get_phase(rc)
    # Check if last phase
    if j < rc.get_PhaseSequence(-1):
        return False
    # Get iteration number
    n = get_current_iter()
    # Check iteration number
    if n is None:
        return False
    elif n < rc.get_LastIter():
        return False
    else:
        # All criteria met
        return True


# --- File management ---
def prepare_files(rc, j=None):
    r"""Prepare files appropriate to run phase *j*

    :Call:
        >>> prepare_files(rc, j=None)
    :Inputs:
        *rc*: :class:`RunControl`
            Options interface from ``case.json``
        *j*: {``None``} | :class:`int`
            Phase number
    :Versions:
        * 2021-11-02 ``@ddalle``: v1.0
    """
    # Get phase number if needed
    if j is None:
        j = get_phase(rc)
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


def finalize_files(rc, j=None):
    r"""Clean up files after running one cycle of phase *j*

    :Call:
        >>> finalize_files(rc, j=None)
    :Inputs:
        *rc*: :class:`RunControl`
            Options interface from ``case.json``
        *j*: {``None``} | :class:`int`
            Phase number
    :Versions:
        * 2021-11-05 ``@ddalle``: v1.0
    """
    # Get phase number if necessary
    if j is None:
        # Get locally
        j = get_phase(rc)
    # Get the last iteration number
    n = get_current_iter()
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


# Function to determine which PBS script to call
def get_pbs_script(j=None):
    r"""Determine the file name of the PBS script to call

    This is a compatibility function for cases that do or do not have
    multiple PBS scripts in a single run directory

    :Call:
        >>> fpbs = case.get_pbs_script(j=None)
    :Inputs:
        *j*: {``None``} | :class:`int`
            Phase number
    :Outputs:
        *fpbs*: :class:`str`
            Name of PBS script to call
    :Versions:
        * 2014-12-01 ``@ddalle``: v1.0 (pycart)
        * 2022-01-20 ``@ddalle``: v1.0
    """
    # Form the full file name, e.g. run_cart3d.00.pbs
    if j is not None:
        # Create the name.
        fpbs = "run_kestrel.%02i.pbs" % j
        # Check for the file.
        if os.path.isfile(fpbs):
            # This is the preferred option if it exists.
            return fpbs
        else:
            # File not found; use basic file name
            return "run_kestrel.pbs"
    else:
        # Do not search for numbered PBS script if *i* is None
        return "run_kestrel.pbs"


# --- STATUS functions ---
def get_phase(rc):
    r"""Determine the phase number based on files in folder

    :Call:
        >>> j = get_phase(rc)
    :Inputs:
        *rc*: :class:`RunControl`
            Case *RunControl* options
    :Outputs:
        *j*: :class:`int`
            Most appropriate phase number for a (re)start
    :Versions:
        * 2021-10-21 ``@ddalle``: v1.0
    """
    # Get the iteration from which a restart would commence
    n = get_current_iter()
    # Start with phase 0 if ``None``
    if n is None:
        return rc.get_PhaseSequence(0)
    # Get last phase number
    j = rc.get_PhaseSequence(-1)
    # Special check for --skeleton cases
    if len(glob.glob("run.%02i.*" % j)) > 0:
        # Check iteration count
        if n >= rc.get_PhaseIters(j):
            return j
    # Loop through phases
    for j in rc.get_PhaseSequence():
        # Target iterations for this phase
        nt = rc.get_PhaseIters(j)
        # Check output files
        if len(glob.glob("run.%02i.*" % j)) == 0:
            # This phase has not been run
            return j
        # Check the iteration- numbers
        if nt is None:
            # Don't check null phases
            pass
        elif n < nt:
            # Case has been run but hasn't reached target
            return j
    # Case completed; just return the last phase
    return j


def get_current_iter():
    r"""Get the most recent iteration number

    :Call:
        >>> n = get_current_iter()
    :Outputs:
        *n*: :class:`int` | ``None``
            Last iteration number
    :Versions:
        * 2021-11-05 ``@ddalle``: v1.0
    """
    # Check if log file exists
    if not os.path.isfile(LOG_FILE):
        return None
    # Otherwise open file to read last line
    lines = fileutils.tail(LOG_FILE, n=1)
    # Attempt to unpack it
    if len(lines) == 0:
        # File exists but is empty
        return 0
    # Unpack singleton list of lines
    line, = lines
    # Trey to get iteration number
    try:
        # First entry should be iteration number
        return int(line.split()[0])
    except ValueError:
        # Some other tailing line; probably no iterations yet
        return 0


# --- Case settings ---
def read_case_json():
    r"""Read *RunControl* settings from ``case.json``

    :Call:
        >>> rc = read_case_json()
    :Outputs:
        *rc*: :class:`cape.pykes.options.runcontrol.RunControl`
            Case run control settings
    :Versions:
        * 2021-10-21 ``@ddalle``: v1.0
    """
    return cc.read_case_json(RunControl)


def read_xml(rc=None, j=None):
    r"""Read Kestrel ``.xml`` control file for one phase

    :Call:
        >>> xml = read_xml(rc=None, j=None)
    :Inputs:
        *rc*: {``None``} | :class:`RunControl`
            Options interface from ``case.json``
        *j*: {``None``} | :class:`int`
            Phase number
    :Outputs:
        *xml*: :class:`JobXML`
            XML control file interface
    :Versions:
        * 2021-11-02 ``@ddalle``: v1.0
    """
    # Read "case.json" if needed
    if rc is None:
        rc = read_case_json()
    # Automatic phase option
    if j is None and rc is not None:
        j = get_phase(rc)
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


# Function to determine which PBS script to call
def get_pbsscript(j=None):
    r"""Determine the file name of the PBS script to call

    This is a compatibility function for cases that do or do not have
    multiple PBS scripts in a single run directory

    :Call:
        >>> fpbs = get_pbsscript(j=None)
    :Inputs:
        *j*: {``None``} | :class:`int`
            Phase number
    :Outputs:
        *fpbs*: :class:`str`
            Name of PBS script to call
    :Versions:
        * 2021-11-05 ``@ddalle``: v1.0
    """
    # Form the full file name, e.g. run_cart3d.00.pbs
    if j is not None:
        # Create the name.
        fpbs = "run_kestrel.%02i.pbs" % j
        # Check for the file.
        if os.path.isfile(fpbs):
            # This is the preferred option if it exists.
            return fpbs
        else:
            # File not found; use basic file name
            return "run_kestrel.pbs"
    else:
        # Do not search for numbered PBS script if *i* is None
        return "run_kestrel.pbs"


# --- Timers ---
def write_starttime(tic, rc, j, fname="pykes_start.dat"):
    r"""Write the start time from *tic*

    :Call:
        >>> write_starttime(tic, rc, j, fname="pykes_start.dat")
    :Inputs:
        *tic*: :class:`datetime.datetime`
            Time to write into data file
        *rc*: :class:`RunControl`
            Options interface
        *j*: :class:`int`
            Phase number
        *fname*: {``"pykes_start.dat"``} | :class:`str`
            Name of file containing run start times
    :Versions:
        * 2021-10-21 ``@ddalle``: v1.0
    """
    # Call the function from :mod:`cape.cfdx.case`
    cc.WriteStartTimeProg(tic, rc, j, fname, "run_kestrel.py")


# Write time used
def write_usertime(tic, rc, i, fname="pykes_time.dat"):
    r"""Write time usage since time *tic* to file

    :Call:
        >>> toc = write_usertime(tic, rc, i, fname="pykes_time.dat")
    :Inputs:
        *tic*: :class:`datetime.datetime`
            Time from which timer will be measured
        *rc*: :class:`pyCart.options.runControl.RunControl`
            Options interface
        *i*: :class:`int`
            Phase number
        *fname*: :class:`str`
            Name of file containing CPU usage history
    :Outputs:
        *toc*: :class:`datetime.datetime`
            Time at which time delta was measured
    :Versions:
        * 2022-03-31 ``@ddalle``: v1.0
    """
    # Call the function from :mod:`cape.case`
    cc.WriteUserTimeProg(tic, rc, i, fname, 'run_kestrel.py')


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
    # Read the options
    rc = read_case_json()
    j = get_phase(rc)
    # Need the namelist to figure out planes, etc.
    xml = read_xml(rc, j)
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

