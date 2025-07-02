r"""
:mod:`cape.pyover.case`: OVERFLOW case control module
=====================================================

This module contains the important function :func:`run_overflow`, which
actually runs ``overrunmpi`` or whichever executable is specified by the
user, along with the utilities that support it.

It also contains OVERFLOW-specific versions of some of the generic
methods from :mod:`cape.cfdx.case`. For instance the function
:func:`GetCurrentIter` determines how many OVERFLOW iterations have been
run in the current folder, which is obviously a solver-specific task. It
also contains the function :func:`LinkQ` and :func:`LinkX` which creates
links to fixed file names from the most recent output created by
OVERFLOW, which is useful for creating simpler Tecplot layouts, for
example.

All of the functions from :mod:`cape.case` are imported here.  Thus they
are available unless specifically overwritten by specific
:mod:`cape.pyover` versions.
"""

# Standard library modules
import glob
import os
import shutil

# Third-party modules
import numpy as np

# Local imports
from . import cmdgen
from .. import fileutils
from ..cfdx import casecntl
from .options.runctlopts import RunControlOpts
from .overnmlfile import OverNamelist


global twall, dtwall, twall_avail

# File names
STOP_FILE = "STOP"


# Get string types based on major Python version
STR_TYPES = str


# Total wall time used
twall = 0.0
# Time used by last phase
dtwall = 0.0
# Default time avail
twall_avail = 1e99


# Help message for CLI
HELP_RUN_OVERFLOW = """
``run_overflow.py``: Run OVERFLOW for one phase
================================================

This script determines the appropriate phase to run for an individual
case (e.g. if a restart is appropriate, etc.), sets that case up, and
runs it.

:Call:

    .. code-block:: console

        $ run_overflow.py [OPTIONS]
        $ python -m cape.pyover run [OPTIONS]

:Options:

    -h, --help
        Display this help message and quit

:Versions:
    * 2014-10-02 ``@ddalle``: v1.0 (pycart)
    * 2016-02-02 ``@ddalle``: v1.0
    * 2021-10-01 ``@ddalle``: v2.0; part of :mod:`case`
"""

# Maximum number of calls to run_phase()
NSTART_MAX = 80


# Function to complete final setup and call the appropriate FUN3D commands
def run_overflow():
    r"""Setup and run the appropriate OVERFLOW command

    :Call:
        >>> run_overflow()
    :Versions:
        * 2016-02-02 ``@ddalle``: v1.0
        * 2021-10-08 ``@ddalle``: v1.1
        * 2023-07-08 ``@ddalle``: v2.0; use CaseRunner
    """
    # Get a case reader
    runner = CaseRunner()
    # Run it
    return runner.run()


# Class for running a case
class CaseRunner(casecntl.CaseRunner):
   # --- Class attributes ---
    # Slots
    __slots__ = (
        "nml",
        "nml_j",
    )

    # Help message
    _help_msg = HELP_RUN_OVERFLOW

    # Names
    _modname = "pyover"
    _progname = "overflow"
    _logprefix = "run"

    # Specific classes
    _rc_cls = RunControlOpts

   # --- Config ---
    # Initialize extra slots
    def init_post(self):
        r"""Custom initialization for ``pyover``

        :Call:
            >>> runner.init_post()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Versions:
            * 2023-07-08 ``@ddalle``: v1.0
        """
        self.nml = None
        self.nml_j = None

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
            * 2023-06-05 ``@ddalle``: v1.0; from ``run_overflow``
            * 2023-07-08 ``@ddalle``; v1.1; instance method
        """
        # Get the project name
        fproj = self.get_prefix()
        # Delete OVERFLOW namelist if present
        if os.path.isfile("over.namelist") or os.path.islink("over.namelist"):
            os.remove("over.namelist")
        # Create the correct namelist
        shutil.copy("%s.%02i.inp" % (fproj, j), "over.namelist")
        # Read case settings
        rc = self.read_case_json()
        # Get iteration pre-run
        n0 = self.get_iter()
        # Get the ``overrunmpi`` command
        cmdi = cmdgen.overrun(rc, j=j)
        # OVERFLOW creates its own "RUNNING" file
        self.mark_stopped()
        # Call the command
        self.callf(cmdi, f="overrun.out", e="overrun.err")
        # Recreate RUNNING file
        self.mark_running()
        # Check new iteration
        n = self.get_iter()
        # Check for no advance
        if n <= n0:
            # Mark failure
            self.mark_failure(f"No advance from iter {n0} in phase {j}")
            # Raise an exception for run()
            raise SystemError(f"No advance from iter {n0} in phase {j}")

   # --- File prep ---
    # Clean up immediately after running
    @casecntl.run_rootdir
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
        """
        # Get the project name
        fproj = self.get_prefix()
        # Get the most recent iteration number
        n = self.get_iter()
        # Assuming that worked, move the temp output file
        fout = "%s.%02i.out" % (fproj, j)
        flog = "%s.%02i.%i" % (fproj, j, n)
        flogj = flog + ".1"
        jlog = 1
        # Check if expected output file exists
        if os.path.isfile(fout):
            # Check if final file name already exists
            if os.path.isfile(flog):
                # Loop utnil we find a viable log file name
                while os.path.isfile(flogj):
                    # Increase counter
                    jlog += 1
                    flogj = "%s.%i" % (flog, jlog)
                # Move the existing log file
                os.rename(flog, flogj)
            # Move immediate output file to log location
            os.rename(fout, flog)

    # Write STOP iteration
    @casecntl.run_rootdir
    def write_stop_iter(self, n=0):
        r"""Create a ``STOP`` file and optionally set the stop iteration

        :Call:
            >>> runner.write_stop_iter(n)
        :Inputs:
            *n*: ``None`` | {``0``} | positive :class:`int`
                Iteration at which to stop; empty file if ``0`` or ``None``
        :Versions:
            * 2017-03-07 ``@ddalle``: v1.0 (``WriteStopIter``)
            * 2023-06-05 ``@ddalle``: v2.0; use context manager
            * 2023-07-08 ``@ddalle``: v2.1; rename, instance method
        """
        # Create the STOP file
        with open(STOP_FILE, "w") as fp:
            # Check if writing anything
            if n:
                fp.write("%i\n" % n)

   # --- Status ---
    # Function to chose the correct input to use from the sequence.
    def getx_phase(self, n: int):
        r"""Get the appropriate input number based on results available

        :Call:
            >>> j = runner.getx_phase(n)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *n*: :class:`int`
                Iteration number
        :Outputs:
            *j*: :class:`int`
                Most appropriate phase number for a restart
        :Versions:
            * 2014-10-02 ``@ddalle``: v1.0 (``cape.pycart.case``)
            * 2015-12-29 ``@ddalle``: v1.0 (``cape.pyfun.case``)
            * 2016-02-03 ``@ddalle``: v1.0 (``GetPhaseNumber``)
            * 2017-01-13 ``@ddalle``: v1.1;  no full ``run.%02.*`` seq
            * 2023-07-09 ``@ddalle``: v1.2; rename, instance method
        """
        # Initialize list of phases with adequate iters
        j_iter = []
        # Initialize list of phases with detected STDOUT files
        j_run = []
        # Read case settings
        rc = self.read_case_json()
        # Loop through possible input numbers.
        for i, j in enumerate(rc.get_PhaseSequence()):
            # Output file glob
            fglob = '%s.%02i.[0-9]*' % (rc.get_Prefix(j), j)
            # Check for output files.
            if len(glob.glob(fglob)) > 0:
                # This run has an output file
                j_run.append(i)
            # Check the iteration number.
            if n >= rc.get_PhaseIters(i):
                # The iterations are adequate for this phase
                j_iter.append(i)
        # Get phase numbers from the two types
        if len(j_iter) > 0:
            j_iter = max(j_iter) + 1
        else:
            j_iter = 0
        if len(j_run) > 0:
            j_run = max(j_run) + 1
        else:
            j_run = 0
        # Look for latest phase with both criteria
        i = min(j_iter, j_run)
        # Convert to phase number
        return rc.get_PhaseSequence(i)

    # Get current iteration
    def getx_iter(self):
        r"""Get the most recent iteration number for OVERFLOW case

        :Call:
            >>> n = runner.getx_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *n*: :class:`int` | ``None``
                Last iteration number
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0 (``GetCurrentIter``)
            * 2023-07-08 ``@ddalle``: v1.1; instance method
        """
        # Read the two sources
        nh = self.getx_history_iter()
        nr = self.getx_running_iter()
        no = self.getx_out_iter()
        # Process
        if nr is None and no is None:
            # No running iterations; check history
            return nh
        elif nr is None:
            # Intermediate step
            return no
        elif nh is None:
            # Only iterations are in running
            return nr
        else:
            # Some iterations saved and some running
            return max(nr, nh)

    # Get the number of finished iterations
    def getx_history_iter(self):
        r"""Get the most recent iteration number for a history file

        This function uses the last line from the file ``run.resid``

        :Call:
            >>> n = runner.getx_history_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *n*: :class:`int` | :class:`float` | ``None``
                Most recent iteration number
        :Versions:
            * 2016-02-01 ``@ddalle``: v1.0 (``GetHistoryIter``)
            * 2023-07-08 ``@ddalle``: v1.1; instance method
        """
        # Read the project rootname
        rname = self.get_prefix()
        # Assemble file name.
        fname = "%s.resid" % rname
        # Check for the file.
        if not os.path.isfile(fname):
            # Alternative file
            fname = "%s.tail.resid" % rname
        # Check for the file.
        if not os.path.isfile(fname):
            # No history to read.
            return 0.0
        # Parse from file
        return self._getx_iter_histfile(fname)

    # Get the last line (or two) from a running output file
    def getx_running_iter(self):
        r"""Get the most recent iteration number for a running file

        This function uses the last line from the file ``resid.tmp``

        :Call:
            >>> n = runner.getx_running_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *n*: :class:`int` | ``None``
                Most recent iteration number
        :Versions:
            * 2016-02-01 ``@ddalle``: v1.0 ( ``GetRunningIter``)
            * 2023-07-08 ``@ddalle``: v1.2; instance method
        """
        # Assemble file name.
        fname = "resid.tmp"
        # Check for the file.
        if not os.path.isfile(fname):
            # No history to read.
            return None
        # Read iteration from file
        return self._getx_iter_histfile(fname)

    # Get the last line (or two) from a running output file
    def getx_out_iter(self):
        r"""Get the most recent iteration number for a running file

        This function uses the last line from the file ``resid.out``

        :Call:
            >>> n = runner.getx_out_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *n*: :class:`int` | ``None``
                Most recent iteration number
        :Versions:
            * 2016-02-02 ``@ddalle``: v1.0 (``GetOutIter``)
            * 2023-07-08 ``@ddalle``: v1.1; instance method
        """
        # Assemble file name.
        fname = "resid.out"
        # Check for the file.
        if not os.path.isfile(fname):
            # No history to read.
            return None
        # Read iteration from file
        return self._getx_iter_histfile(fname)

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
            * 2015-10-19 ``@ddalle``: v1.0
        """
        # Get prefix
        rname = self.get_prefix()
        # Output glob
        fout = glob.glob('%s.[0-9][0-9]*.[0-9]*' % rname)
        # Initialize iteration number until informed otherwise.
        n = 0
        # Loop through the matches.
        for fname in fout:
            # Get the integer for this file.
            try:
                # Interpret the iteration number from file name
                i = int(fname.split('.')[-1])
            except Exception:
                # Failed to interpret this file name
                i = 0
            # Use the running maximum.
            n = max(i, n)
        # Output
        return n

    # Read a generic resid file to get latest iteration
    def _getx_iter_histfile(self, fname: str):
        # Read file
        try:
            # Tail the file
            line = fileutils.tail(fname, n=1)
            # Get the iteration number.
            return int(line.split(maxsplit=2)[1])
        except Exception:
            # Failure; return no-iteration result.
            raise ValueError(
                f"Unable to parse iteration number from '{fname}'\n" +
                f"Last line was:\n    {line[:20]}")

   # --- Local readers ---
    # Get the namelist
    @casecntl.run_rootdir
    def read_namelist(self, j=None):
        r"""Read case namelist file

        :Call:
            >>> nml = runner.read_namelist(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *i*: {``None``} | nonnegative :class:`int`
                Phase number (0-based)
        :Outputs:
            *nml*: :class:`OverNamelist`
                Namelist interface
        :Versions:
            * 2015-12-29 ``@ddalle``: v1.0 (``cape.pyfun.case``)
            * 2015-02-02 ``@ddalle``: v1.0 (``GetNamelist``)
            * 2016-12-12 ``@ddalle``: v1.1; *i* kwarg
            * 2023-07-09 ``@ddalle``: v1l.1; rename, instance method
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
        if isinstance(self.nml, OverNamelist) and nmlj == j and j is not None:
            # Return it!
            return self.nml
        # Check for detailed inputs
        if rc is None:
            # Check for simplest namelist file
            if os.path.isfile('over.namelist'):
                # Read the currently linked namelist.
                return OverNamelist('over.namelist')
            else:
                # Look for namelist files
                fglob = glob.glob('*.[0-9][0-9].inp')
                # Read one of them.
                return OverNamelist(fglob[0])
        else:
            # Get phase number
            if j is None:
                j = self.get_phase()
            # Read the namelist file.
            return OverNamelist('%s.%02i.inp' % (rc.get_Prefix(j), j))

   # --- Local options ---
    # Function to get prefix
    def get_prefix(self, j=None):
        r"""Read OVERFLOW file prefix

        :Call:
            >>> rname = runner.get_prefix(j=None)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *rname*: :class:`str`
                Project prefix
        :Versions:
            * 2016-02-01 ``@ddalle``: v1.0 (``GetPrefix``)
            * 2023-07-08 ``@ddalle``: v1.1; instance method
        """
        # Get options interface
        rc = self.read_case_json()
        # Read the prefix
        return rc.get_Prefix(j)

    # Get STOP iteration
    @casecntl.run_rootdir
    def get_stop_iter(self):
        r"""Get iteration at which to stop by reading ``STOP`` file

        If the file exists but is empty, returns ``0``; if file does not
        exist, returns ``None``; and otherwise reads the iteration

        :Call:
            >>> qstop, nstop = runner.get_stop_iter()
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        :Outputs:
            *nstop*: :class:`int` | ``None``
                Iteration at which to stop, if any
        :Versions:
            * 2017-03-07 ``@ddalle``: v1.0 (``GetStopIter``)
            * 2023-07-08 ``@ddalle``: v1.1; instance method
        """
        # Check for the file
        qstop = os.path.isfile(STOP_FILE)
        # If no file; exit
        if not qstop:
            return qstop, None
        # Otherwise, attempt to read it
        try:
            # Open the file
            with open(STOP_FILE, 'r') as fp:
                # Read the first line
                line = fp.readline()
            # Attempt to get an integer out of there
            n = int(line.split()[0])
            return n
        except Exception:
            # If empty file (or not readable), always stop
            return 0


# Check the number of iterations in an average
def checkqavg(fname):
    r"""Check the number of iterations in a ``q.avg`` file

    This function works by attempting to read a Fortran record at the
    very end of the file with exactly one (single-precision) integer.
    The function tries both little- and big-endian interpretations. If
    both methods fail, it returns ``1`` to indicate that the ``q`` file
    is a single-iteration solution.

    :Call:
        >>> nq = checkqavg(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of OVERFLOW ``q`` file
    :Outputs:
        *nq*: :class:`int`
            Number of iterations included in average
    :Versions:
        * 2016-12-29 ``@ddalle``: v1.0
    """
    # Open the file
    f = open(fname, 'rb')
    # Head to the end of the file, minus 12 bytes
    f.seek(-12, 2)
    # Try to read as a little-endian record at the end
    I = np.fromfile(f, count=3, dtype="<i4")
    # If that failed to read 3 ints, file has < 12 bits
    if len(I) < 3:
        f.close()
        return 1
    # Check if the little-endian read came up with something
    if (I[0] == 4) and (I[2] == 4):
        f.close()
        return I[1]
    # Try a big-endian read
    f.seek(-12, 2)
    I = np.fromfile(f, count=3, dtype=">i4")
    f.close()
    # Check for success
    if (I[0] == 4) and (I[2] == 4):
        # This record makes sense
        return I[1]
    else:
        # Could not interpret record; assume one-iteration q-file
        return 1


# Check the iteration number
def checkqt(fname):
    r"""Check the iteration number or time in a ``q`` file

    :Call:
        >>> t = checkqt(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of OVERFLOW ``q`` file
    :Outputs:
        *t*: ``None`` | :class:`float`
            Iteration number or time value
    :Versions:
        * 2016-12-29 ``@ddalle``: v1.0
    """
    # Open the file
    f = open(fname, 'rb')
    # Try to read the first record
    I = np.fromfile(f, count=1, dtype="<i4")
    # Check for valid read
    if len(I) == 0:
        f.close()
        return None
    # Check endianness
    if I[0] == 4:
        # Little endian
        ti = "<i4"
        tf = "<f"
    else:
        ti = ">i4"
        tf = ">f"
    # Read number of grids
    ng, i = np.fromfile(f, count=2, dtype=ti)
    # Check consistency
    if i != 4:
        f.close()
        return None
    # Read past the grid dimensions
    f.seek(4 + 12*ng, 1)
    # Read the number of states, num species, and end-of-record
    nq, nqc, i = np.fromfile(f, count=3, dtype=ti)
    # Read the header start-of-record marker to determine sp/dp
    i, = np.fromfile(f, count=1, dtype=ti)
    # Check for single precision
    if i == (13+max(2, nqc))*8 + 4:
        # Double-precision (usual)
        nf = 8
        tf = tf + "8"
    else:
        # Single-precision
        nf = 4
        tf = tf + "4"
    # Skip the first three entries of the header (REFMACH, ALPHA, REY)
    f.seek(3*nf, 1)
    # Read the time
    t, = np.fromfile(f, count=1, dtype=tf)
    # Close the file
    f.close()
    # Output
    return t


# Edit lines of a ``splitmq`` or ``splitmx`` input file
def EditSplitmqI(fin, fout, qin, qout):
    r"""Edit the I/O file names in a ``splitmq``/``splitmx`` input file

    :Call:
        >>> EditSplitmqI(fin, fout, qin, qout)
    :Inputs:
        *fin*: :class:`str`
            Name of template ``splitmq`` input file
        *fout*: :class:`str`
            Name of altered ``splitmq`` input file
        *qin*: :class:`str`
            Name of input solution or grid file
        *qout*: :class:`str`
            Name of output solution or grid file
    :Versions:
        * 2017-01-07 ``@ddalle``: v1.0
    """
    # Check for input file
    if not os.path.isfile(fin):
        raise ValueError("No template ``splitmq`` file '%s'" % fin)
    # Open the template and output files
    fi = open(fin, 'r')
    fo = open(fout, 'w')
    # Write the input and output solution/grid files
    fo.write('%s\n' % qin)
    fo.write('%s\n' % qout)
    # Ignore first two lines of input file
    fi.readline()
    fi.readline()
    # Copy the rest of the file
    fo.write(fi.read())
    # Close files
    fi.close()
    fo.close()


# Get best Q file
def GetQ():
    r"""Find most recent ``q.*`` file, with ``q.avg`` taking precedence

    :Call:
        >>> fq = GetQ()
    :Outputs:
        *fq*: ``None`` | :class:`str`
            Name of most recent averaged ``q`` file or newest ``q`` file
    :Versions:
        * 2016-12-29 ``@ddalle``: v1.0
    """
    # Get the list of q files
    qglob = glob.glob('q.save')+glob.glob('q.restart')+glob.glob('q.[0-9]*')
    qavgb = glob.glob('q.avg*')
    # Check for averaged files
    if len(qavgb) > 0:
        qglob = qavgb
    # Exit if no files
    if len(qglob) == 0:
        return None
    # Get modification times from the files
    tq = [os.path.getmtime(fq) for fq in qglob]
    # Get index of most recent file
    iq = np.argmax(tq)
    # Return that file
    return qglob[iq]


# Get best q file
def GetLatest(glb):
    r"""Get the most recent file matching a glob or list of globs

    :Call:
        >>> fq = GetLatest(glb)
        >>> fq = GetLatest(lglb)
    :Inputs:
        *glb*: :class:`str`
            File name glob
        *lblb*: :class:`list`\ [:class:`str`]
            List of file name globs
    :Outputs:
        *fq*: ``None`` | :class:`str`
            Name of most recent file matching glob(s)
    :Versions:
        * 2017-01-08 ``@ddalle``: v1.0
    """
    # Check type
    if type(glb).__name__ in ['list', 'ndarray']:
        # Initialize from list of globs
        fglb = []
        # Loop through globs
        for g in glb:
            # Add the matches to this glob (don't worry about duplicates)
            fglb += glob.glob(g)
    else:
        # Single glob
        fglb = glob.glob(glb)
    # Exit if none
    if len(fglb) == 0:
        return None
    # Get modification times from the files
    tg = [os.path.getmtime(fg) for fg in fglb]
    # Get index of most cecent file
    ig = np.argmax(tg)
    # return that file
    return fglb[ig]


# Generic link command that cleans out existing links before making a mess
def LinkLatest(fsrc, fname):
    r"""Create a symbolic link, but clean up existing links

    This prevents odd behavior when using :func:`os.symlink` when the
    link already exists.  It performs no action (rather than raising an
    error) when the source file does not exist or is ``None``.  Finally,
    if *fname* is already a full file, no action is taken.

    :Call:
        >>> LinkLatest(fsrc, fname)
    :Inputs:
        *fsrc*: ``None`` | :class:`str`
            Name of file to act as source for the link
        *fname*: :class:`str`
            Name of the link to create
    :Versions:
        * 2017-01-08 ``@ddalle``: v1.0
    """
    # Check for file
    if os.path.islink(fname):
        # Delete old links
        try:
            os.remove(fname)
        except Exception:
            pass
    elif os.path.isfile(fname):
        # Do nothing if full file exists with this name
        return
    # Check if the source file exists
    if (fsrc is None) or (not os.path.isfile(fsrc)):
        return
    # Create link
    try:
        os.symlink(fsrc, fname)
    except Exception:
        pass


# Link best Q file
def LinkQ():
    r"""Link the most recent ``q.*`` file to a fixed file name

    :Call:
        >>> LinkQ()
    :Versions:
        * 2016-09-06 ``@ddalle``: v1.0
        * 2016-12-29 ``@ddalle``: Moved file search to :func:`GetQ`
    """
    # Get the general best ``q`` file name
    fq = GetQ()
    # Get the best single-iter, ``q.avg``, and ``q.srf`` files
    fqv = GetLatest(["q.[0-9]*[0-9]", "q.save", "q.restart"])
    fqa = GetLatest(["q.[0-9]*.avg", "q.avg*"])
    fqs = GetLatest(["q.[0-9]*.srf", "q.srf*", "q.[0-9]*.surf", "q.surf*"])
    # Create links (safely)
    LinkLatest(fq,  'q.pyover.p3d')
    LinkLatest(fqv, 'q.pyover.vol')
    LinkLatest(fqa, 'q.pyover.avg')
    LinkLatest(fqs, 'q.pyover.srf')


# Get best Q file
def GetX():
    r"""Get the most recent ``x.*`` file

    :Call:
        >>> fx = GetX()
    :Outputs:
        *fx*: ``None`` | :class:`str`
            Name of most recent ``x.save`` or similar file
    :Versions:
        * 2016-12-29 ``@ddalle``: v1.0
    """
    # Get the list of q files
    xglob = (
        glob.glob('x.save') + glob.glob('x.restart') +
        glob.glob('x.[0-9]*') + glob.glob('grid.in'))
    # Exit if no files
    if len(xglob) == 0:
        return
    # Get modification times from the files
    tx = [os.path.getmtime(fx) for fx in xglob]
    # Get index of most recent file
    ix = np.argmax(tx)
    # Output
    return xglob[ix]


# Link best X file
def LinkX():
    r"""Link the most recent ``x.*`` file to a fixed file name

    :Call:
        >>> LinkX()
    :Versions:
        * 2016-09-06 ``@ddalle``: v1.0
    """
    # Get the best file
    fx = GetX()
    # Get the best surf grid if available
    fxs = GetLatest(["x.[0-9]*.srf", "x.srf*", "x.[0-9]*.surf", "x.surf*"])
    # Create links (safely)
    LinkLatest(fx,  'x.pyover.p3d')
    LinkLatest(fxs, 'x.pyover.srf')


# Function to determine newest triangulation file
def GetQFile(fqi="q.pyover.p3d"):
    r"""Get most recent OVERFLOW ``q`` file and its associated iterations

    Averaged solution files, such as ``q.avg`` take precedence.

    :Call:
        >>> fq, n, i0, i1 = GetQFile(fqi="q.pyover.p3d")
    :Inputs:
        *fqi*: {q.pyover.p3d} | q.pyover.avg | q.pyover.vol | :class:`str`
            Target Overflow solution file after linking most recent files
    :Outputs:
        *fq*: :class:`str`
            Name of ``q`` file
        *n*: :class:`int`
            Number of iterations included
        *i0*: :class:`int`
            First iteration in the averaging
        *i1*: :class:`int`
            Last iteration in the averaging
    :Versions:
        * 2016-12-30 ``@ddalle``: v1.0
        * 2017-03-28 ``@ddalle``: v1.1; from ```lineload` to ``case``
    """
    # Link grid and solution files
    LinkQ()
    LinkX()
    # Check for the input file
    if os.path.isfile(fqi):
        # Use the file (may be a link, in fact it usually is)
        fq = fqi
    else:
        # Best Q file available (usually "q.avg" or "q.save")
        fq = GetQ()
    # Check for q.avg iteration count
    n = checkqavg(fq)
    # Read the current "time" parameter
    i1 = checkqt(fq)
    # Get start parameter
    if (n is not None) and (i1 is not None):
        # Calculate start iteration
        i0 = i1 - n + 1
    else:
        # Cannot determine start iteration
        i0 = None
    # Output
    return fq, n, i0, i1

