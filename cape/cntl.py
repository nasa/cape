"""
:mod:`cape.cntl`: Base module for CFD operations and processing
=================================================================

This module provides tools and templates for tools to interact with
various CFD codes and their input files. The base class is
:class:`cape.cntl.Cntl`, and the derivative classes include
:class:`cape.pycart.cntl.Cntl`. This module creates folders for cases,
copies files, and can be used as an interface to perform most of the
tasks that Cape can accomplish except for running individual cases.

The control module is set up as a Python interface for the master
JSON file, which contains the settings to be used for a given CFD
project.

The derivative classes are used to read input files, set up cases,
submit and/or run cases, and be an interface for the various Cape
options as they are customized for the various CFD solvers. The
individualized modules are below.

    * :mod:`cape.pycart.cntl`
    * :mod:`cape.pyfun.cntl`
    * :mod:`cape.pyover.cntl`

:See also:
    * :mod:`cape.cfdx.case`
    * :mod:`cape.cfdx.options`
    * :mod:`cape.runmatrix`

"""

# Standard library modules
import functools
import glob
import importlib
import json
import os
import re
import shutil
import sys
import time

# Standard library partial imports
from datetime import datetime

# Third-party modules
import numpy as np

# Local modules
from .cfdx import options
from .cfdx import queue
from .cfdx import case
from . import convert
from . import console
from . import argread
from . import manage

# Functions and classes from other modules
from .runmatrix import RunMatrix
from .config    import ConfigXML, ConfigJSON

# Import triangulation
from .tri  import Tri, ReadTriFile
from .geom import RotatePoints


# Decorator for moving directories
def run_rootdir(func):
    r"""Decorator to run a function within a specified folder

    :Call:
        >>> func = run_rootdir(func)
    :Wrapper Signature:
        >>> v = cntl.func(*a, **kw)
    :Inputs:
        *func*: :class:`func`
            Name of function
        *cntl*: :class:`Cntl`
            Control instance from which to use *cntl.RootDir*
        *a*: :class:`tuple`
            Positional args to :func:`cntl.func`
        *kw*: :class:`dict`
            Keyword args to :func:`cntl.func`
    :Versions:
        * 2018-11-20 ``@ddalle``: Version 1.0
        * 2020-02-25 ``@ddalle``: Version 1.1: better exceptions
    """
    # Declare wrapper function to change directory
    @functools.wraps(func)
    def wrapper_func(self, *args, **kwargs):
        # Recall current directory
        fpwd = os.getcwd()
        # Go to specified directory
        os.chdir(self.RootDir)
        # Run the function with exception handling
        try:
            # Attempt to run the function
            v = func(self, *args, **kwargs)
        except Exception:
            # Go back to original folder
            os.chdir(fpwd)
            # Raise the error
            raise
        except KeyboardInterrupt:
            # Go back to original folder
            os.chdir(fpwd)
            # Raise the error
            raise
        # Go back to original folder
        os.chdir(fpwd)
        # Return function values
        return v
    # Apply the wrapper
    return wrapper_func


# Class to read input files
class Cntl(object):
    r"""Class to handle options, setup, and execution of CFD codes

    :Call:
        >>> cntl = cape.Cntl(fname="cape.json")
    :Inputs:
        *fname*: :class:`str`
            Name of JSON settings file from which to read options
    :Outputs:
        *cntl*: :class:`cape.cntl.Cntl`
            Instance of Cape control interface
        *cntl.opts*: :class:`cape.cfdx.options.Options`
            Options interface
        *cntl.x*: :class:`cape.runmatrix.RunMatrix`
            Run matrix interface
        *cntl.RootDir*: :class:`str`
            Working directory from which the class was generated
    :Versions:
        * 2015-09-20 ``@ddalle``: Started
        * 2016-04-01 ``@ddalle``: Version 1.0
    """
   # =============
   # Configuration
   # =============
   # <
    # Initialization method
    def __init__(self, fname="cape.json"):
        r"""Initialization method for :mod:`cape.cntl.Cntl`

        :Versions:
            * 2015-09-20 ``ddalle``: Version 1.0
        """
        # Check if file exists
        if not os.path.isfile(fname):
            # Raise error but suppress traceback
            os.sys.tracebacklimit = 0
            raise ValueError("No cape control file '%s' found" % fname)

        # Read settings
        self.opts = options.Options(fname=fname)

        #Save the current directory as the root
        self.RootDir = os.getcwd()

        # Import modules
        self.ImportModules()

        # Process the trajectory.
        self.x = RunMatrix(**self.opts['RunMatrix'])

        # Job list
        self.jobs = {}

        # Set umask
        os.umask(self.opts.get_umask())

        # Run any initialization functions
        self.InitFunction()

    # Output representation
    def __repr__(self):
        r"""Output representation method for Cntl class

        :Versions:
            * 2015-09-20 ``@ddalle``: Version 1.0
        """
        # Display basic information
        return "<cape.Cntl(nCase=%i)>" % self.x.nCase

    # Function to import user-specified modules
    def ImportModules(self):
        r"""Import user-defined modules if specified in the options

        All modules from the ``"Modules"`` global option of the JSON
        file (``cntl.opts["Modules"]``) will be imported and saved as
        attributes of *cntl*.  For example, if the user wants to use a
        module called :mod:`dac3`, it will be imported as *cntl.dac3*.
        A noncomprehensive list of disallowed module names is below.

            *DataBook*, *RootDir*, *jobs*, *opts*, *tri*, *x*

        The name of any method of this class is also disallowed.
        However, if the user wishes to import a module whose name is
        disallowed, he/she can use a dictionary to specify a different
        name to import the module as. For example, the user may import a
        module called :mod:`tri` as :mod:`mytri` using the following
        JSON syntax.

            .. code-block:: javascript

                "Modules": [{"tri": "mytri"}]

        :Call:
            >>> cntl.ImportModules()
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of Cape control interface
        :Versions:
            * 2014-10-08 ``@ddalle``: Version 1.0 (:mod:`pycart`)
            * 2015-09-20 ``@ddalle``: Version 1.0
        """
        # Get Modules.
        lmod = self.opts.get("Modules", [])
        # Ensure list.
        if not lmod:
            # Empty --> empty list
            lmod = []
        elif not isinstance(lmod, list):
            # Single string
            lmod = [lmod]
        # Loop through modules.
        for imod in lmod:
            # Check for dictionary
            if isinstance(imod, dict):
                # Get the file name and import name separately
                fmod = imod.keys()[0]
                nmod = imod[fmod]
                # Status update
                print("Importing module '%s' as '%s'" % (fmod, imod))
            else:
                # Import as the default name
                fmod = imod
                nmod = imod
                # Status update
                print("Importing module '%s'" % imod)
            # Load the module by its name
            self.__dict__[fmod] = importlib.import_module(nmod)

    # Function to apply initialization function
    def InitFunction(self):
        """Run one or more "initialization functions"

        This calls the function(s) in the global ``"InitFunction"`` option from
        the JSON file.  These functions must take *cntl* as an input, and they
        are usually from a module imported via the ``"Modules"`` option.  See
        the following example:

            .. code-block:: javascript

                "Modules": ["testmod"],
                "InitFunction": ["testmod.testfunc"]

        This leads pyCart to call ``testmod.testfunc(cntl)``.  For pyFun, this
        becomes ``testmod.testfunc(fun3d)``, etc.

        :Call:
            >>> cntl.InitFunction()
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Overall control interface
        :Versions:
            * 2017-04-04 ``@ddalle``: Version 1.0
        """
        # Get input functions
        lfunc = self.opts.get("InitFunction", [])
        # Ensure list
        lfunc = list(np.array(lfunc).flatten())
        # Loop through functions
        for func in lfunc:
            # Status update
            print("  InitFunction: %s()" % func)
            # Run the function
            exec("self.%s(self)" % func)


    # Call function to apply settings for case *i*
    def CaseFunction(self, i):
        """Apply a function at the beginning of :func:`PrepareCase(i)`

        This is meant to serve as a filter if a user wants to change the
        settings for some subset of the cases.  Using this function can change
        any setting, which can be dependent on the case number *i*.

        This calls the function(s) in the global ``"CaseFunction"`` option from
        the JSON file. These functions must take *cntl* as an input and the
        case number *i*. The function(s) are usually from a module imported via
        the ``"Modules"`` option. See the following example:

            .. code-block:: javascript

                "Modules": ["testmod"],
                "CaseFunction": ["testmod.testfunc"]

        This leads pyCart to call ``testmod.testfunc(cntl, i)`` at the
        beginning of :func:`PrepareCase` for each case *i* in the run matrix.
        The function is also called at the beginning of :func:`ApplyCase`

        :Call:
            >>> cntl.CaseFunction(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Overall control interface
            *i*: :class:`int`
                Case number
        :Versions:
            * 2017-04-05 ``@ddalle``: Version 1.0
        :See also:
            * :func:`cape.cntl.Cntl.InitFunction`
            * :func:`cape.cntl.Cntl.PrepareCase`
            * :func:`cape.pycart.cntl.Cntl.PrepareCase`
            * :func:`cape.pycart.cntl.Cntl.ApplyCase`
            * :func:`cape.pyfun.cntl.Cntl.PrepareCase`
            * :func:`cape.pyfun.cntl.Cntl.ApplyCase`
            * :func:`cape.pyover.cntl.Cntl.PrepareCase`
            * :func:`cape.pyover.cntl.Cntl.ApplyCase`
        """
        # Get input functions
        lfunc = self.opts.get("CaseFunction", [])
        # Ensure list
        lfunc = list(np.array(lfunc).flatten())
        # Loop through functions
        for func in lfunc:
            # Status update
            print("  Case Function: cntl.%s(%s)" % (func, i))
            # Run the function
            exec("self.%s(self, %s)" % (func, i))


    # Make a directory
    def mkdir(self, fdir):
        """Make a directory with the correct permissions

        :Call:
            >>> cntl.mkdir(fdir)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *fdir*: :class:`str`
                Directory to create
        :Versions:
            * 2015-09-27 ``@ddalle``: Version 1.0
        """
        # Get umask
        umask = self.opts.get_umask()
        # Apply mask
        dmask = 0o777 - umask
        # Make the directory.
        os.mkdir(fdir, dmask)
   # >

   # =============
   # Input Readers
   # =============
   # <
    # Function to prepare the triangulation for each grid folder
    @run_rootdir
    def ReadTri(self):
        """Read initial triangulation file(s)

        :Call:
            >>> cntl.ReadTri()
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
        :Versions:
            * 2014-08-30 ``@ddalle``: Version 1.0
        """
        # Only read triangulation if not already present.
        try:
            self.tri
            return
        except AttributeError:
            pass
        # Get the list of tri files.
        ftri = self.opts.get_TriFile()
        # Status update.
        print("  Reading tri file(s) from root directory.")
        # Name of config file
        fxml = self.opts.get_ConfigFile()
        # Check for a config file.
        if fxml is None:
            # Nothing to read
            cfg = None
        else:
            # Read config file
            cfg = ConfigXML(fxml)
        # Ensure list
        if type(ftri).__name__ not in ['list', 'ndarray']: ftri = [ftri]
        # Read first file
        tri = ReadTriFile(ftri[0])
        # Apply configuration
        if cfg is not None:
            tri.ApplyConfig(cfg)
        # Initialize number of nodes in each file
        tri.iTri = [tri.nTri]
        tri.iQuad = [tri.nQuad]
        # Loop through files
        for f in ftri[1:]:
            # Check for non-surface tri file
            if f.startswith('-'):
                # Not for writing in "VolTri"; don't intersect it
                qsurf = -1
                # Strip leading "-"
                f = f.lstrip("-")
            else:
                # This is a regular surface
                qsurf = 1
            # Read the next triangulation
            trii = ReadTriFile(f)
            # Apply configuration
            if cfg is not None:
                trii.ApplyConfig(cfg)
            # Append the triangulation
            tri.Add(trii)
            # Save the face counts
            tri.iTri.append(qsurf*tri.nTri)
            tri.iQuad.append(qsurf*tri.nQuad)
        # Save the triangulation and config.
        self.tri = tri
        self.tri.config = cfg
        # Check for AFLR3 bcs
        fbc = self.opts.get_aflr3_BCFile()
        # If present, map it.
        if fbc:
            # Map boundary conditions
            self.tri.ReadBCs_AFLR3(fbc)
        # Make a copy of the original to revert to after rotations, etc.
        self.tri0 = self.tri.Copy()

    # Read configuration (without tri file if necessary)
    @run_rootdir
    def ReadConfig(self, f=False):
        """Read ``Config.xml`` or ``Config.json`` file if not already present

        :Call:
            >>> cntl.ReadConfig()
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
        :Versions:
            * 2016-06-10 ``@ddalle``: Version 1.0
            * 2016-10-21 ``@ddalle``: Version 2.0, added ``Config.json``
            * 2020-09-01 ``@ddalle``: Version 2.1, add *f* kwarg
        """
        # Check for config
        if not f:
            try:
                self.config
                return
            except AttributeError:
                pass
            # Try to read from the triangulation
            try:
                self.config = self.tri.config
                return
            except AttributeError:
                pass
        # Name of config file
        fxml = self.opts.get_ConfigFile()
        # Split based on '.'
        fext = fxml.split('.')
        # Get the extension
        if len(fext) < 2:
            # Odd case, no extension given
            fext = 'json'
        else:
            # Get the extension
            fext = fext[-1].lower()
        # Read the configuration if it can be found
        if fxml is None or not os.path.isfile(fxml):
            # Nothing to read
            self.config = None
        elif fext == "xml":
            # Read XML config file
            self.config = ConfigXML(fxml)
        else:
            # Read JSON config file
            self.config = ConfigJSON(fxml)

   # >

   # ======================
   # Command-Line Interface
   # ======================
   # <
    # Baseline function
    def cli_preprocess(self, *a, **kw):
        """Preprocess command-line arguments and flags/keywords

        :Call:
            >>> a, kw = cntl.cli_preprocess(*a, **kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *kw*: :class:`dict` (``True`` | ``False`` | :class:`str`)
                Command-line keyword arguments and flags
        :Outputs:
            *a*: :class:`tuple`
                List of non-flag arguments with any additional preprocessing
            *kw*: :class:`dict`
                Flags with any additional preprocessing performed
        :Versions:
            * 2018-10-19 ``@ddalle``: Content from ``bin/`` executables
        """
        # Get constraints and convert text to list
        cons  = kw.get('cons',        '').split(',')
        cons += kw.get('constraints', '').split(',')
        # Set the constraints back into the keywords.
        kw['cons'] = [con.strip() for con in cons]

        # Process index list.
        if 'I' in kw:
            # Turn into a single list
            kw['I'] = self.x.ExpandIndices(kw['I'])

        # Get list of scripts in the "_old" section
        kwx = [ki['x'] for ki in kw.get('_old', {}) if 'x' in ki]
        # Append the last "-x" input
        if 'x' in kw:
            kwx.append(kw['x'])
        # Apply all scripts
        for fx in kwx:
            if sys.version_info.major == 2:
                # Built-in function
                execfile(fx)
            else:
                # Open file and execute it
                exec(open(fx).read())

        # Output
        return a, kw

    # Baseline function
    def cli_cape(self, *a, **kw):
        """Command-line interface

        This function is applied after the command-line arguments are parsed
        using :func:`cape.argread.readflagstar` and the control interface has
        already been read according to the ``-f`` flag.

        :Call:
            >>> cmd = cntl.cli_cape(*a, **kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *kw*: :class:`dict`
                Preprocessed command-line keyword arguments
        :Outputs:
            *cmd*: ``None`` | :class:`str`
                Name of command that was processed, if any
        :Versions:
            * 2018-10-19 ``@ddalle``: Content from ``bin/`` executables
        """
        # Check for recognized command
        if kw.get('c'):
            # Display status.
            self.DisplayStatus(**kw)
            return 'c'
        elif kw.get("PASS"):
            # Pass jobs
            self.MarkPASS(**kw)
            return "PASS"
        elif kw.get("ERROR"):
            # Mark jobs as permanent errors
            self.MarkERROR(**kw)
            return "ERROR"
        elif kw.get("unmark"):
            # Unmark jobs
            self.UnmarkCase(**kw)
            return "unmark"
        elif kw.get('batch'):
            # Process a batch job
            self.SubmitBatchPBS(os.sys.argv)
            return 'batch'
        elif kw.get('extend'):
            # Extend the number of iterations in a phase
            self.ExtendCases(**kw)
            return 'extend'
        elif kw.get('apply'):
            # Rewrite namelists and possibly add phases
            self.ApplyCases(**kw)
            return 'apply'
        elif kw.get('checkFM'):
            # Check aero databook
            self.CheckFM(**kw)
            return "checkFM"
        elif kw.get('checkLL'):
            # Check aero databook
            self.CheckLL(**kw)
            return "checkLL"
        elif kw.get('checkTriqFM'):
            # Check aero databook
            self.CheckTriqFM(**kw)
            return "checkTriqFM"
        elif kw.get('check'):
            # Check all
            print("---- Checking FM DataBook components ----")
            self.CheckFM(**kw)
            print("---- Checking LineLoad DataBook components ----")
            self.CheckLL(**kw)
            print("---- Checking TriqFM DataBook components ----")
            self.CheckTriqFM(**kw)
            # Output
            return "check"
        elif kw.get('aero') or kw.get('fm'):
            # Collect force and moment data.
            self.UpdateFM(**kw)
            return 'fm'
        elif kw.get('ll'):
            # Update line load data book
            self.UpdateLineLoad(**kw)
            return 'll'
        elif kw.get('triqfm'):
            # Update TriqFM data book
            self.UpdateTriqFM(**kw)
            return 'triqfm'
        elif kw.get('data', kw.get('db')):
            # Update all
            print("---- Updating FM DataBook components ----")
            self.UpdateFM(**kw)
            print("---- Updating LineLoad DataBook components ----")
            self.UpdateLL(**kw)
            print("---- Updating TriqFM DataBook components ----")
            self.UpdateTriqFM(**kw)
            # Output
            return "db"
        elif kw.get('archive'):
            # Archive cases
            self.ArchiveCases(**kw)
            return 'archive'
        elif kw.get('unarchive'):
            # Unarchive cases
            self.UnarchiveCases(**kw)
            return 'unarchive'
        elif kw.get('skeleton'):
            # Replace case with its skeleton
            self.SkeletonCases(**kw)
            return 'skeleton'
        elif kw.get('clean'):
            # Clean up cases
            self.CleanCases(**kw)
            return 'clean'
        elif kw.get('report'):
            # Get the report(s) to create.
            if kw['report'] == True:
                # First report
                rep = self.opts.get_ReportList()[0]
            else:
                # User-specified report
                rep = kw['report']
            # Get the report
            R = self.ReadReport(rep)
            # Update according to other options
            R.UpdateReport(**kw)
            return 'report'

    # Baseline function
    def cli(self, *a, **kw):
        """Command-line interface

        :Call:
            >>> cntl.cli(*a, **kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *kw*: :class:`dict` (``True`` | ``False`` | :class:`str`)
                Unprocessed keyword arguments
        :Outputs:
            *cmd*: ``None`` | :class:`str`
                Name of command that was processed, if any
        :Versions:
            * 2018-10-19 ``@ddalle``: Content from ``bin/`` executables
        """
        # Preprocess command-line inputs
        a, kw = self.cli_preprocess(*a, **kw)
        # Call the common interface
        cmd = self.cli_cape(*a, **kw)
        # Test for a command
        if cmd is not None:
            return
        # Submit jobs as fallback
        self.SubmitJobs(**kw)

    # Function to display current status
    def DisplayStatus(self, **kw):
        """Display current status for all cases

        This prints case names, current iteration numbers, and so on.  This is
        the function that is called when the user issues a system command like
        ``cape -c``.

        :Call:
            >>> cntl.DisplayStatus(j=False)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *j*: :class:`bool`
                Whether or not to display job ID numbers
            *cons*: :class:`list` (:class:`str`)
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2014-10-04 ``@ddalle``: Version 1.0
            * 2014-12-09 ``@ddalle``: Added constraints
        """
        # Force the "check" option to true.
        kw['c'] = True
        # Call the job submitter but don't allow submitting.
        self.SubmitJobs(**kw)

    # Master interface function
    def SubmitJobs(self, **kw):
        """Check jobs and prepare or submit jobs if necessary

        :Call:
            >>> cntl.SubmitJobs(**kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *c*: :class:`bool`
                If ``True``, only display status; do not submit new jobs
            *j*: :class:`bool`
                Whether or not to display job ID numbers
            *n*: :class:`int`
                Maximum number of jobs to submit
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2014-10-05 ``@ddalle``: Version 1.0
            * 2014-12-09 ``@ddalle``: Added constraints
        """
       # -----------------------
       # Command Determination
       # -----------------------
        # Get flag that tells pycart only to check jobs.
        qCheck = kw.get('c', False)
        # Get flag to show job IDs
        qJobID = kw.get('j', False)
        # Whether or not to delete cases
        qDel = kw.get('rm', False)
        # PBS flag
        qSlurm = self.opts.get_sbatch(0)
        # Check whether or not to kill PBS jobs
        qKill = kw.get('qdel', kw.get('kill', kw.get('scancel', False)))
        # Check whether to execute scripts
        ecmd = kw.get('exec', kw.get('e'))
        qExec = (ecmd is not None)
        # No submissions if we're just deleting.
        if qKill or qExec or qDel: qCheck = True
       # ---------
       # Options
       # ---------
        # Check if we should start cases
        if kw.get("nostart") or (not kw.get("start", True)):
            # Set cases up but do not start them
            q_strt = False
        else:
            # Set cases up and submit/start them
            q_strt = True
        # Check if we should submit INCOMP jobs
        if kw.get("norestart") or (not kw.get("restart", True)):
            # Do not submit jobs labeled "INCOMP"
            stat_submit = ["---"]
        elif q_strt:
            # Submit either new jobs or "INCOMP"
            stat_submit = ["---", "INCOMP"]
        else:
            # If not starting jobs, no reason for "INCOMP"
            stat_submit = ["---"]
        # Check for --no-qsub option
        if not kw.get('qsub', True):
            self.opts.set_qsub(False)
        if not kw.get('sbatch', True):
            self.opts.set_sbatch(False)
        # Check for skipping marked cases
        if kw.get("unmarked", False):
            # Unmarked cases
            q_umark = True
        elif kw.get("nomarked", False) or (not kw.get("marked", True)):
            # Only show unmarked cases
            q_umark = True
        else:
            # Show all cases
            q_umark = False
        # Maximum number of jobs
        nSubMax = int(kw.get('n', 10))
       # --------
       # Cases
       # --------
        # Get list of indices.
        I = self.x.GetIndices(**kw)
        # Get the case names.
        fruns = self.x.GetFullFolderNames(I)
       # -------
       # Queue
       # -------
        # Get the qstat info (safely; do not raise an exception).
        if qSlurm:
            # Slurm: squeue
            jobs = queue.squeue(u=kw.get('u'))
        else:
            # PBS: qstat
            jobs = queue.qstat(u=kw.get('u'))
        # Save the jobs.
        self.jobs = jobs
       # -------------
       # Formatting
       # -------------
        # Maximum length of one of the names
        if len(fruns) > 0:
            # Check the cases
            lrun = max([len(frun) for frun in fruns])
        else:
            # Just use a default value.
            lrun = 0
        # Make sure it's as long as the header
        lrun = max(lrun, 21)
        # Print the right number of '-' chars
        f = '-'; s = ' '
        # Create the string stencil.
        if qJobID:
            # Print status with job numbers.
            stncl = ('%%-%is ' * 7) % (4, lrun, 7, 11, 3, 8, 7)
            # Print header row.
            print(stncl % ("Case", "Config/Run Directory", "Status",
                "Iterations", "Que", "CPU Time", "Job ID"))
            # Print "---- --------" etc.
            print(f*4 + s + f*lrun + s + f*7 + s + f*11 + s + f*3 + s
                + f*8 + s + f*7)
        else:
            # Print status without job numbers.
            stncl = ('%%-%is ' * 6) % (4, lrun, 7, 11, 3, 8)
            # Print header row.
            print(stncl % ("Case", "Config/Run Directory", "Status",
                "Iterations", "Que", "CPU Time"))
            # Print "---- --------" etc.
            print(f*4 + s + f*lrun + s + f*7 + s + f*11 + s + f*3 + s + f*8)
       # -------
       # Loop
       # -------
        # Initialize number of submitted jobs
        nSub = 0
        # Initialize number of jobs in queue.
        nQue = 0
        # Number of deleted jobs
        nDel = 0
        # Initialize dictionary of statuses.3
        total = {'PASS':0, 'PASS*':0, '---':0, 'INCOMP':0,
            'RUN':0, 'DONE':0, 'QUEUE':0, 'ERROR':0, 'ZOMBIE':0}
        # Loop through the runs.
        for j in range(len(I)):
           # --- Case ID ---
            # Case index.
            i = I[j]
            # Extract case
            frun = fruns[j]
           # --- Mark check ---
            # Check for unmarked-only flag
            if q_umark and (self.x.PASS[i] or self.x.ERROR[i]):
                continue
           # --- Status ---
            # Check status.
            sts = self.CheckCaseStatus(i, jobs, u=kw.get('u'))
            # Get active job number.
            jobID = self.GetPBSJobID(i)
            # Append.
            total[sts] += 1
            # Get the current number of iterations
            n = self.CheckCase(i)
            # Get CPU hours
            t = self.GetCPUTime(i, running=(sts=='RUN'))
            # Convert to string
            if t is None:
                # Empty string
                CPUt = ""
            else:
                # Convert to %.1f
                CPUt = "%8.1f" % t
            # Switch on whether or not case is set up.
            if n is None:
                # Case is not prepared.
                itr = "/"
                que = "."
            else:
                # Case is prepared and might be running.
                # Get last iteration.
                nMax = self.GetLastIter(i)
                # Iteration string
                itr = "%i/%i" % (n, nMax)
                # Check the queue.
                if jobID in jobs:
                    # Get whatever the qstat command said.
                    que = jobs[jobID]["R"]
                else:
                    # Not found by qstat (or not a jobID at all)
                    que = "."
           # --- Display ---
            # Print info
            if qJobID and jobID in jobs:
                # Print job number.
                print(stncl % (i, frun, sts, itr, que, CPUt, jobID))
            elif qJobID:
                # Print blank job number.
                print(stncl % (i, frun, sts, itr, que, CPUt, ""))
            else:
                # No job number.
                print(stncl % (i, frun, sts, itr, que, CPUt))
           # --- Execution ---
            # Check for queue killing
            if qKill and (n is not None) and (jobID in jobs):
                # Delete it.
                self.StopCase(i)
                continue
            # Check for script
            if qExec:
                # Execute script
                self.ExecScript(i, ecmd)
                continue
            # Check for deletion
            if qDel and (not n) and (sts in ["INCOMP", "ERROR", "---"]):
                # Delete folder
                nDel += self.DeleteCase(i, **kw)
            elif qDel:
                # Delete but forcing prompt
                nDel += self.DeleteCase(i, prompt=True)
            # Check status.
            if qCheck: continue
            # If submitting is allowed, check the job status.
            if (sts in stat_submit) and self.FilterUser(i, **kw):
                # Prepare the job.
                self.PrepareCase(i)
                # Start (submit or run) case
                if q_strt:
                    self.StartCase(i)
                # Increase job number
                nSub += 1
            # Don't continue checking if maximum submissions reached.
            if nSub >= nSubMax: break
       # ---------
       # Summary
       # ---------
        # Extra line.
        print("")
        # State how many jobs submitted.
        if nSub:
            # Submitted/started?
            if q_strt:
                print("Submitted or ran %i job(s).\n" % nSub)
            else:
                # We can still count cases set up
                print("Set up %i job(s) but did not start.\n" % nSub)
        # State how many jobs deleted
        if nDel:
            print("Deleted %s jobs" % nDel)
        # Status summary
        fline = ""
        for key in total:
            # Check for any cases with the status.
            if total[key]:
                # At least one with this status.
                fline += ("%s=%i, " % (key,total[key]))
        # Print the line.
        if fline: print(fline)

    # Mark a case as PASS
    @run_rootdir
    def MarkPASS(self, **kw):
        """Mark one or more cases as **PASS** and rewrite matrix

        :Call:
            >>> cntl.MarkPASS(**kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints like ``'Mach<=0.5'``
            *flag*: {``"p"``} | ``"P"`` | ``"PASS"`` | ``"$p"``
                Marker to use to denote status
        :Versions:
            * 2019-06-14 ``@ddalle``: Version 1.0
        """
        # Get indices
        I = self.x.GetIndices(**kw)
        # Check sanity
        if len(I) > 100:
            raise ValueError(
                "Attempting to mark %i ERRORs; that's too many!" % len(I))
        # Process flag option
        flag = kw.get("flag", "p")
        # Loop through cases
        for i in I:
            # Mark case
            self.x.MarkPASS(i, flag=flag)
        # Write the trajectory
        self.x.WriteRunMatrixFile()

    # Mark a case as PASS
    @run_rootdir
    def MarkERROR(self, **kw):
        """Mark one or more cases as **ERROR** and rewrite matrix

        :Call:
            >>> cntl.MarkERROR(**kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints like ``'Mach<=0.5'``
            *flag*: {``"E"``} | ``"e"`` | ``"ERROR"`` | ``"$E"``
                Marker to use to denote status
        :Versions:
            * 2019-06-14 ``@ddalle``: Version 1.0
        """
        # Get indices
        I = self.x.GetIndices(**kw)
        # Check sanity
        if len(I) > 100:
            raise ValueError(
                "Attempting to mark %i ERRORs; that's too many!" % len(I))
        # Process flag option
        flag = kw.get("flag", "E")
        # Get deletion options
        qrm = kw.get("rm", False)
        qprompt = kw.get("prompt", True)
        # Loop through cases
        for i in I:
            # Mark case
            self.x.MarkERROR(i, flag=flag)
            # Delete folder (?)
            if qrm:
                self.DeleteCase(i, prompt=qprompt)
        # Write the trajectory
        self.x.WriteRunMatrixFile()

    # Remove PASS and ERROR markers
    @run_rootdir
    def UnmarkCase(self, **kw):
        """Mark one or more cases as **ERROR** and rewrite matrix

        :Call:
            >>> cntl.UnmarkCase(**kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2019-06-14 ``@ddalle``: Version 1.0
        """
        # Get indices
        I = self.x.GetIndices(**kw)
        # Loop through cases
        for i in I:
            # Mark case
            self.x.UnmarkCase(i)
        # Write the trajectory
        self.x.WriteRunMatrixFile()

    # Execute script
    @run_rootdir
    def ExecScript(self, i, cmd):
        """Execute a script in a given case folder

        This function is the interface to command-line calls using the ``-e``
        flag, such as ``pycart -e 'ls -lh'``.

        :Call:
            >>> ierr = cntl.ExecScript(i, cmd)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Case index (0-based)
        :Outputs:
            *ierr*: ``None`` | :class:`int`
                Exit status from the command
        :Versions:
            * 2016-08-26 ``@ddalle``: Version 1.0
        """
        # Get the case folder name
        frun = self.x.GetFullFolderNames(i)
        # Check for the folder
        if not os.path.isdir(frun):
            return
        # Enter the folder
        os.chdir(frun)
        # Status update
        print("  Executing system command:")
        # Check the input type
        typ = type(cmd).__name__
        # Execute based on type
        if typ == 'list':
            # Pass to safer subprocess command
            print("  %s" % cmd)
            ierr = sp.call(cmd)
        else:
            # Check if it's a file.
            if not cmd.startswith(os.sep):
                # First command could be a script name
                fcmd = cmd.split()[0]
                # Get file name relative to Cntl root directory
                fcmd = os.path.join(self.RootDir, fcmd)
                # Check for the file.
                if os.path.exists(fcmd):
                    # Copy the file here
                    shutil.copy(fcmd, '.')
                    # Name of the script
                    fexec = os.path.split(fcmd)[1]
                    # Strip folder names from command
                    cmd = "./%s %s" % (fexec, ' '.join(cmd.split()[1:]))
            # Status update
            print("    %s" % cmd)
            # Pass to dangerous system command
            ierr = os.system(cmd)
        # Output
        print("    exit(%s)" % ierr)
        return ierr

   # >

   # =============
   # Run Interface
   # =============
   # <
    # Apply user filter
    def FilterUser(self, i, **kw):
        # Get any 'user' trajectory keys
        ku = self.x.GetKeysByType('user')
        # Check if there's a user variable
        if len(ku) == 0:
            # No user filter
            return True
        elif len(ku) > 1:
            # More than one user constraint? sounds like a bad idea
            raise ValueError(
                "Found more than one USER run matrix value: %s" % ku)
        # Select the user key
        k = ku[0]
        # Get target user
        uid = kw.get('u', kw.get('user', os.environ['USER']))
        # Get the value of the user from the run matrix
        # Also, remove leading '@' character if present
        ui = self.x[k][i].lstrip('@').lower()
        # Check the actual constraint
        if ui == "":
            # No user constraint; pass
            return True
        elif ui == uid:
            # Correct user
            return True
        else:
            # Wrong user!
            return False


    # Function to start a case: submit or run
    @run_rootdir
    def StartCase(self, i):
        """Start a case by either submitting it

        This function checks whether or not a case is submittable.  If
        so, the case is submitted via :func:`cape.cfdx.queue.pqsub`,
        and otherwise the case is started using a system call.

        Before starting case, this function checks the folder using
        :func:`cape.cntl.CheckCase`; if this function returns ``None``,
        the case is not started.  Actual starting of the case is done
        using :func:`CaseStartCase`, which has a specific version for
        each CFD solver.

        :Call:
            >>> pbs = cntl.StartCase(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *pbs*: :class:`int` or ``None``
                PBS job ID if submitted successfully
        :Versions:
            * 2014-10-06 ``@ddalle``: Version 1.0
        """
        # Get case name
        frun = self.x.GetFullFolderNames(i)
        # Check status.
        if self.CheckCase(i) is None:
            # Case not ready
            print("    Attempted to start case '%s'." % frun)
            print("    However, case failed initial checks.")
            # Check again with verbose option
            self.CheckCase(i, v=True)
            return
        elif self.CheckRunning(i):
            # Case already running!
            return
        # Safely go to the folder.
        os.chdir(frun)
        # Print status.
        print("     Starting case '%s'" % frun)
        # Start the case by either submitting or calling it.
        pbs = self.CaseStartCase()
        # Display the PBS job ID if that's appropriate.
        if pbs:
            print("     Submitted job: %i" % pbs)
        # Output
        return pbs


    # Call the correct module to start the case
    def CaseStartCase(self):
        """Start a case by either submitting it or running it

        This function relies on :mod:`cape.cfdx.case`, and so it is
        customized for the correct solver only in that it calls the
        correct *case* module.

        :Call:
            >>> pbs = cntl.CaseStartCase()
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Cape control interface
        :Outputs:
            *pbs*: :class:`int` or ``None``
                PBS job ID if submitted successfully
        :Versions:
            * 2015-10-14 ``@ddalle``: Version 1.0
        """
        return case.StartCase()

    # Function to terminate a case: qdel and remove RUNNING file
    @run_rootdir
    def StopCase(self, i):
        """Stop a case if running

        This function deletes a case's PBS job and removes the :file:`RUNNING`
        file if it exists.

        :Call:
            >>> cntl.StopCase(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Cape control interface
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2014-12-27 ``@ddalle``: Version 1.0
        """
        # Check status.
        if self.CheckCase(i) is None:
            # Case not ready
            return
        # Get the case name and go there.
        frun = self.x.GetFullFolderNames(i)
        os.chdir(frun)
        # Stop the job if possible.
        case.StopCase()

   # >

   # ===========
   # Case Status
   # ===========
   # <
    # Get expected actual breaks of phase iters.
    def GetPhaseBreaks(self):
        """Get expected iteration numbers at phase breaks

        This fills in ``0`` entries in *RunControl>PhaseIters* and returns the
        filled-out list

        :Call:
            >>> PI = cntl.GetPhaseBreaks()
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Cape control interface
        :Outputs:
            *PI*: :class:`list` (:class:`int`)
        :Versions:
            * 2017-04-12 ``@ddalle``: Version 1.0
        """
        # Get list of phases to use
        PhaseSeq = self.opts.get_PhaseSequence()
        PhaseSeq = list(np.array(PhaseSeq).flatten())
        # Get number of sequences
        nSeq = len(PhaseSeq)
        # Get option values for *PhaseIters* and *nIter*
        PI = [self.opts.get_PhaseIters(j) for j in PhaseSeq]
        NI = [self.opts.get_nIter(j)      for j in PhaseSeq]
        # Ensure phase break for first phase
        PI[0] = max(PI[0], NI[0])
        # Loop through phases
        for i in range(1, nSeq):
            # Ensure at least *nIter* iterations beyond previous phase
            PI[i] = max(PI[i], PI[i-1]+NI[i])
        # Output
        return PI

    # Get last iter
    def GetLastIter(self, i):
        """Get minimum required iteration for a given run to be completed

        :Call:
            >>> nIter = cntl.GetLastIter(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Cape control interface
            *i*: :class:`int`
                Run index
        :Outputs:
            *nIter*: :class:`int`
                Number of iterations required for case *i*
        :Versions:
            * 2014-10-03 ``@ddalle``: Version 1.0
        """
        # Read the local case.json file.
        rc = self.ReadCaseJSON(i)
        # Check for null file
        if rc is None:
            return self.opts.get_PhaseIters(-1)
        # Option for desired iterations
        N = rc.get('PhaseIters', 0)
        # Output the last entry (if list)
        return options.getel(N, -1)

    # Function to determine if case is PASS, ---, INCOMP, etc.
    def CheckCaseStatus(self, i, jobs=None, auto=False, u=None):
        """Determine the current status of a case

        :Call:
            >>> sts = cntl.CheckCaseStatus(i, jobs=None, auto=False, u=None)
        :Inputs:
            *cart3d*: :class:`cape.pycart.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
            *jobs*: :class:`dict`
                Information on each job, ``jobs[jobID]`` for each submitted job
            *u*: :class:`str`
                User name (defaults to ``os.environ['USER']``)
        :Versions:
            * 2014-10-04 ``@ddalle``: Version 1.0
            * 2014-10-06 ``@ddalle``: Checking queue status
        """
        # Current iteration count
        n = self.CheckCase(i)
        # Try to get a job ID.
        jobID = self.GetPBSJobID(i)
        # Default jobs.
        if jobs is None:
            # Use current status.
            jobs = self.jobs
        # Check for auto-status
        if (jobs=={}) and auto:
            # Call qstat.
            if self.opts.get_sbatch(0):
                # Call slurm instead of PBS
                self.jobs = queue.squeue(u=u)
            else:
                # Use qstat to get job info
                self.jobs = queue.qstat(u=u)
            jobs = self.jobs
        # Check if the case is prepared.
        if self.CheckError(i):
            # Case contains :file:`FAIL`
            sts = "ERROR"
        elif n is None:
            # Nothing prepared.
            sts = "---"
        else:
            # Check if the case is running.
            if self.CheckRunning(i):
                # Case currently marked as running.
                sts = "RUN"
            elif self.CheckError(i):
                # Case has some sort of error.
                sts = "ERROR"
            else:
                # Get maximum iteration count.
                nMax = self.GetLastIter(i)
                # Get current phase
                j, jLast = self.CheckUsedPhase(i)
                # Check current count.
                if jobID in jobs:
                    # It's in the queue, but apparently not running.
                    if jobs[jobID]['R'] == "R":
                        # Job running according to the queue
                        sts = "RUN"
                    else:
                        # It's in the queue.
                        sts = "QUEUE"
                elif j < jLast:
                    # Not enough phases
                    sts = "INCOMP"
                elif n >= nMax:
                    # Not running and sufficient iterations completed.
                    sts = "DONE"
                else:
                    # Not running and iterations remaining.
                    sts = "INCOMP"
        # Check for zombies
        if (sts == "RUN") and self.CheckZombie(i):
            # Looks like it is running, but no files modified
            sts = "ZOMBIE"
        # Check if the case is marked as PASS
        if self.x.PASS[i]:
            # Check for cases marked but that can't be done.
            if sts == "DONE":
                # Passed!
                sts = "PASS"
            else:
                # Funky
                sts = "PASS*"
        # Output
        return sts

    # Check a case.
    @run_rootdir
    def CheckCase(self, i, v=False):
        """Check current status of run *i*

        Because the file structure is different for each solver, some
        of this method may need customization.  This customization,
        however, can be kept to the functions
        :func:`cape.cfdx.case.GetCurrentIter` and
        :func:`cape.cntl.Cntl.CheckNone`.

        :Call:
            >>> n = cntl.CheckCase(i, v=False)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
            *v*: ``True`` | {``False``}
                Verbose flag; prints messages if *n* is ``None``
        :Outputs:
            *n*: :class:`int` | ``None``
                Number of completed iterations or ``None`` if not set up
        :Versions:
            * 2014-09-27 ``@ddalle``: Version 1.0
            * 2015-09-27 ``@ddalle``: Generic version
            * 2015-10-14 ``@ddalle``: Removed dependence on :mod:`case`
            * 2017-02-22 ``@ddalle``: Added verbose flag
        """
        # Check input.
        if not isinstance(i, (int, np.int_)):
            raise TypeError(
                "Input to :func:`Cntl.CheckCase()` must be :class:`int`.")
        # Get the group name.
        frun = self.x.GetFullFolderNames(i)
        # Initialize iteration number.
        n = 0
        # Check if the folder exists.
        if (not os.path.isdir(frun)):
            # Verbosity option
            if v: print("    Folder '%s' does not exist" % frun)
            n = None
        # Check that test.
        if n is not None:
            # Go to the group folder.
            os.chdir(frun)
            # Check the history iteration
            try:
                n = self.CaseGetCurrentIter()
            except Exception:
                # At least one file missing that is required
                n = None
        # If zero, check if the required files are set up.
        if (n == 0) and self.CheckNone(v):
            n = None
        # Output.
        return n


    # Get the current iteration number from :mod:`case`
    def CaseGetCurrentIter(self):
        """Get the current iteration number from the appropriate module

        This function utilizes the :mod:`cape.cfdx.case` module, and so
        it must be copied to the definition for each solver's control
        class.

        :Call:
            >>> n = cntl.CaseGetCurrentIter()
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *n*: :class:`int` or ``None``
                Number of completed iterations or ``None`` if not set up
        :Versions:
            * 2015-10-14 ``@ddalle``: Version 1.0
        """
        return case.GetCurrentIter()

    # Check a case's phase output files
    @run_rootdir
    def CheckUsedPhase(self, i, v=False):
        """Check maximum phase number run at least once

        :Call:
            >>> j, n = cntl.CheckUsedPhase(i, v=False)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
            *v*: ``True`` | {``False``}
                Verbose flag; prints messages if *n* is ``None``
        :Outputs:
            *j*: :class:`int` | ``None``
                Phase number
            *n*: :class:`int` | ``None``
                Maximum phase number
        :Versions:
            * 2017-06-29 ``@ddalle``: Version 1.0
            * 2017-07-11 ``@ddalle``: Added second output
        """
         # Check input.
        if type(i).__name__ not in ["int", "int64", "int32"]:
            raise TypeError(
                "Input to :func:`Cntl.CheckCase()` must be :class:`int`.")
        # Get the group name.
        frun = self.x.GetFullFolderNames(i)
        # Initialize phase number.
        j = 0
        # Check if the folder exists.
        if (not os.path.isdir(frun)):
            # Verbosity option
            if v: print("    Folder '%s' does not exist" % frun)
            j = None
        # Check that test.
        if j is not None:
            # Go to the group folder.
            os.chdir(frun)
            # Read local settings
            try:
                # Read "case.json"
                rc = case.ReadCaseJSON()
                # Get phase list
                phases = list(self.opts.get_PhaseSequence())
            except Exception:
                # Get global phase list
                phases = list(self.opts.get_PhaseSequence())
            # Reverse the list
            phases.reverse()
            # Loop backwards
            for j in phases:
                # Check if any output files exist
                if len(glob.glob("run.%02i.[1-9]*" % j)) > 0:
                    # Found it.
                    break
        # Output.
        return j, phases[-1]

    # Check a case's phase number
    @run_rootdir
    def CheckPhase(self, i, v=False):
        """Check current phase number of run *i*

        :Call:
            >>> n = cntl.CheckPhase(i, v=False)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
            *v*: ``True`` | {``False``}
                Verbose flag; prints messages if *n* is ``None``
        :Outputs:
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2017-06-29 ``@ddalle``: Version 1.0
        """
        # Check input.
        if type(i).__name__ not in ["int", "int64", "int32"]:
            raise TypeError(
                "Input to :func:`Cntl.CheckCase()` must be :class:`int`.")
        # Get the group name.
        frun = self.x.GetFullFolderNames(i)
        # Initialize iteration number.
        n = 0
        # Check if the folder exists.
        if (not os.path.isdir(frun)):
            # Verbosity option
            if v: print("    Folder '%s' does not exist" % frun)
            n = None
        # Check that test.
        if n is not None:
            # Go to the group folder.
            os.chdir(frun)
            # Check the phase information
            try:
                n = self.CaseGetCurrentPhase()
            except Exception:
                # At least one file missing that is required
                n = 0
        # Output.
        return n

    # Get the current iteration number from :mod:`case`
    def CaseGetCurrentPhase(self):
        """Get the current phase number from the appropriate module

        This function utilizes the :mod:`cape.cfdx.case` module, and so
        it must be copied to the definition for each solver's control
        class.

        :Call:
            >>> j = cntl.CaseGetCurrentPhase()
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2017-06-29 ``@ddalle``: Version 1.0
        """
        # Be safe
        try:
            # Read the "case.json" folder
            rc = case.ReadCaseJSON()
            # Get the phase number
            return case.GetPhaseNumber(rc)
        except:
            return 0


    # Check if cases with zero iterations are not yet setup to run
    def CheckNone(self, v=False):
        """Check if the present working directory has the necessary files to run

        This function needs to be customized for each CFD solver so that it
        checks for the appropriate files.

        :Call:
            >>> q = cntl.CheckNone(v=False)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Cape control interface
            *v*: ``True`` | {``False``}
                Verbose flag; prints message if *q* is ``True``
        :Outputs:
            *q*: ```True`` | `False``
                Whether or not case is missing files
        :Versions:
            * 2015-09-27 ``@ddalle``: Version 1.0
            * 2017-02-22 ``@ddalle``: Added verbose flag
        """
        return False

    # Check if a case is running.
    @run_rootdir
    def CheckRunning(self, i):
        """Check if a case is currently running

        :Call:
            >>> q = cntl.CheckRunning(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Outputs:
            *q*: :class:`bool`
                If ``True``, case has :file:`RUNNING` file in it
        :Versions:
            * 2014-10-03 ``@ddalle``: Version 1.0
        """
        # Get run name
        frun = self.x.GetFullFolderNames(i)
        # Check for the RUNNING file.
        q = os.path.isfile(os.path.join(frun, 'RUNNING'))
        # Output
        return q

    # Check for a failure
    @run_rootdir
    def CheckError(self, i):
        """Check if a case has a failure

        :Call:
            >>> q = cntl.CheckError(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Outputs:
            *q*: :class:`bool`
                If ``True``, case has :file:`FAIL` file in it
        :Versions:
            * 2015-01-02 ``@ddalle``: Version 1.0
        """
        # Get run name
        frun = self.x.GetFullFolderNames(i)
        # Check for the RUNNING file.
        q = os.path.isfile(os.path.join(frun, 'FAIL'))
        # Check ERROR flag
        q = q or self.x.ERROR[i]
        # Output
        return q

    # Check for no unchanged files
    @run_rootdir
    def CheckZombie(self, i):
        """Check a case for ``ZOMBIE`` status

        A running case is declared a zombie if none of the listed files (by
        default ``*.out``) have been modified in the last 30 minutes.  However,
        a case cannot be a zombie unless it contains a ``RUNNING`` file and
        returns ``True`` from :func:`CheckRunning`.

        :Call:
            >>> q = cntl.CheckZombie(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Outputs:
            *q*: :class:`bool`
                ``True`` if no listed files have been modified recently
        :Versions:
            * 2017-04-04 ``@ddalle``: Version 1.0
        """
        # Check if case is running
        qrun = self.CheckRunning(i)
        # If not running, cannot be a zombie
        if not qrun:
            return False
        # Get run name
        frun = self.x.GetFullFolderNames(i)
        # Check if the folder exists
        if not os.path.isdir(frun):
            return False
        # Enter the folder
        os.chdir(frun)
        # List of files to check
        fzomb = self.opts.get("ZombieFiles", "*.out")
        # Ensure list
        fzomb = list(np.array(fzomb).flatten())
        # Create list of files matching globs
        fglob = []
        for fg in fzomb:
            fglob += glob.glob(fg)
        # Timeout time (in minutes)
        tmax = self.opts.get("ZombieTimeout", 30.0)
        t = tmax
        # Current time
        toc = time.time()
        # Loop through glob files
        for fname in fglob:
            # Get minutes since modification for *fname*
            ti = (toc - os.path.getmtime(fname))/60
            # Running minimum
            t = min(t, ti)
        # Output
        return (t >= tmax)
   # >

   # =================
   # Case Modification
   # =================
   # <
    # Function to extend one or more cases
    def ExtendCases(self, **kw):
        """Extend one or more case by a number of iterations

        By default, this applies to the final phase, but the phase number *j*
        can also be specified as input. The number of additional iterations is
        generally the nominal number of iterations that phase *j* would
        normally run.

        :Call:
            >>> cntl.ExtendCases(cons=[], j=None, extend=1, **kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of overall control interface
            *extend*: {``True``} | positive :class:`int`
                Extend phase *j* by *extend* nominal runs
            *j*: {``None``} | nonnegative :class:`int`
                Phase number
            *imax*: {``None``} | :class:`int`
                Do not increase iteration number beyond *imax*
            *cons*: :class:`list` (:class:`str`)
                List of constraints
            *I*: :class:`list` (:class:`int`)
                List of indices
        :Versions:
            * 2016-12-12 ``@ddalle``: Version 1.0
        """
        # Process inputs
        j = kw.get('j')
        n = kw.get('extend', 1)
        imax = kw.get('imax')
        # Convert inputs to integers
        if j:    j = int(j)
        if n:    n = int(n)
        if imax: imax = int(imax)
        # Restart inputs
        qsub = kw.get("restart", kw.get("qsub", False))
        nsub = kw.get("n", 150)
        jsub = 0
        # Loop through folders
        for i in self.x.GetIndices(**kw):
            # Status update
            print(self.x.GetFullFolderNames(i))
            # Extend case
            self.ExtendCase(i, n=n, j=j, imax=imax)
            # Start/submit the case?
            if qsub:
                # Check status
                sts = self.CheckCaseStatus(i)
                # Check if it's a submittable/restartable status
                if sts not in ['---', 'INCOMP']: continue
                # Try to start the case
                pbs = self.StartCase(i)
                # Check for a submission
                if pbs:
                    jsub += 1
                # Check submission limit
                if jsub >= nsub: return

    # Function to extend one or more cases
    def ApplyCases(self, **kw):
        """Extend one or more case by a number of iterations

        By default, this applies to the final phase, but the phase number *j*
        can also be specified as input. The number of additional iterations is
        generally the nominal number of iterations that phase *j* would
        normally run.

        :Call:
            >>> cntl.Applycases(cons=[], j=None, extend=1, **kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of overall control interface
            *extend*: {``True``} | positive :class:`int`
                Extend phase *j* by *extend* nominal runs
            *j*: {``None``} | nonnegative :class:`int`
                Phase number
            *cons*: :class:`list` (:class:`str`)
                List of constraints
            *I*: :class:`list` (:class:`int`)
                List of indices
        :Versions:
            * 2016-12-12 ``@ddalle``: Version 1.0
        """
        # Process inputs
        j = kw.get('j')
        n = kw.get('apply')
        # Handle raw ``-apply`` inputs vs. ``--apply $n``
        if n == True:
            # Use ``None`` to inherit phase count from *cntl*
            n = None
        else:
            # Convert input string to integer
            n = int(n)
        # Restart inputs
        qsub = kw.get("restart", kw.get("qsub", False))
        nsub = kw.get("n", 150)
        jsub = 0
        # Loop through folders
        for i in self.x.GetIndices(**kw):
            # Status update
            print(self.x.GetFullFolderNames(i))
            # Extend case
            self.ApplyCase(i, nPhase=n)
            # Start/submit the case?
            if qsub:
                # Check status
                sts = self.CheckCaseStatus(i)
                # Check if it's a submittable/restartable status
                if sts not in ['---', 'INCOMP']: continue
                # Try to start the case
                pbs = self.StartCase(i)
                # Check for a submission
                if pbs:
                    jsub += 1
                # Check submission limit
                if jsub >= nsub: return

    # Function to delete a case folder: qdel and rm
    @run_rootdir
    def DeleteCase(self, i, **kw):
        """Delete a case

        This function deletes a case's PBS job and removes the entire
        directory.  By default, the method prompts for user's confirmation
        before deleting; set *prompt* to ``False`` to delete without prompt.

        :Call:
            >>> n = cntl.DeleteCase(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Cape control interface
            *i*: :class:`int`
                Index of the case to check (0-based)
            *prompt*: {``True``} | ``False``
                Whether or not to prompt user before deleting case
        :Outputs:
            *n*: ``0`` | ``1``
                Number of folders deleted
        :Versions:
            * 2018-11-20 ``@ddalle``: Version 1.0
        """
        # Local function to perform deletion
        def del_folder(frun):
            # Delete the folder using :mod:`shutil`
            shutil.rmtree(frun)
            # Status update
            print("   Deleted folder '%s'" % frun)
        # Get the case name and go there.
        frun = self.x.GetFullFolderNames(i)
        # Check if folder exists
        if not os.path.isdir(frun):
            # Nothing to do
            n = 0
        # Check for prompt option
        elif kw.get('prompt', True):
            # Prompt text
            txt = "Delete case '%s'? y/n" % frun
            # Get option from user
            prompt = console.prompt_color(txt, "n")
            # Check option
            if (prompt is None) or (prompt.lower() != "y"):
                # Do not delete
                n = 0
            else:
                # Delete folder
                del_folder(frun)
                n = 1
        else:
            # Delete without prompt
            del_folder(frun)
            n = 1
        # Output
        return n
   # >

   # =========
   # Archiving
   # =========
   # <
    # Function to archive results and remove files
    @run_rootdir
    def ArchiveCases(self, **kw):
        """Archive completed cases and clean them up if specified

        :Call:
            >>> cntl.ArchiveCases()
            >>> cntl.ArchiveCases(cons=[], **kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of overall control interface
            *cons*: :class:`list` (:class:`str`)
                List of constraints
            *I*: :class:`list` (:class:`int`)
                List of indices
        :Versions:
            * 2016-12-09 ``@ddalle``: Version 1.0
        """
        # Get the format
        fmt = self.opts.get_ArchiveAction()
        # Check for directive not to archive
        if not fmt or not self.opts.get_ArchiveFolder(): return
        # Loop through folders
        for i in self.x.GetIndices(**kw):
            # Go to root folder
            os.chdir(self.RootDir)
            # Get folder name
            frun = self.x.GetFullFolderNames(i)
            # Status update
            print(frun)
            # Check if the case is ready to archive
            if not os.path.isdir(frun):
                print("  Folder does not exist.")
                continue
            # Get status
            sts = self.CheckCaseStatus(i)
            # Enter the case folder
            os.chdir(frun)
            # Perform cleanup
            self.CleanPWD()
            # Check status
            if sts != 'PASS':
                print("  Case is not marked PASS.")
                continue
            # Archive
            self.ArchivePWD(phantom=kw.get("phantom",False))

    # Individual case archive function
    def ArchivePWD(self, phantom=False):
        """Archive a single case in the current folder ($PWD)

        :Call:
            >>> cntl.ArchivePWD(phantom=False)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of overall control interface
            *phantom*: ``True`` | {``False``}
                Write actions to ``archive.log``; only delete if ``False``
        :Versions:
            * 2016-12-09 ``@ddalle``: Version 1.0
        """
        # Archive using the local module
        manage.ArchiveFolder(self.opts, phantom=False)

    # Function to archive results and remove files
    @run_rootdir
    def SkeletonCases(self, **kw):
        """Archive completed cases and delete all but a few files

        :Call:
            >>> cntl.SkeletonCases()
            >>> cntl.SkeletonCases(cons=[], **kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of overall control interface
            *cons*: :class:`list` (:class:`str`)
                List of constraints
            *I*: :class:`list` (:class:`int`)
                List of indices
        :Versions:
            * 2016-12-14 ``@ddalle``: Version 1.0
        """
        # Get the format
        fmt = self.opts.get_ArchiveAction()
        # Check for directive not to archive
        if not fmt or not self.opts.get_ArchiveFolder(): return
        # Loop through folders
        for i in self.x.GetIndices(**kw):
            # Go to root folder
            os.chdir(self.RootDir)
            # Get folder name
            frun = self.x.GetFullFolderNames(i)
            # Status update
            print(frun)
            # Check if the case is ready to archive
            if not os.path.isdir(frun):
                print("  Folder does not exist.")
                continue
            # Get status
            sts = self.CheckCaseStatus(i)
            # Enter the case folder
            os.chdir(frun)
            # Perform cleanup
            self.CleanPWD()
            # Check status
            if sts != 'PASS':
                print("  Case is not marked PASS.")
                continue
            # Archive
            self.SkeletonPWD(phantom=kw.get("phantom",False))

    # Individual case archive function
    def SkeletonPWD(self, phantom=False):
        """Delete most files in current folder, leaving only a skeleton

        :Call:
            >>> cntl.SkeletonPWD(phantom=False)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control interface
            *phantom*: ``True`` | {``False``}
                Write actions to ``archive.log``; only delete if ``False``
        :Versions:
            * 2017-12-14 ``@ddalle``: Version 1.0
        """
        # Archive using the local module
        manage.SkeletonFolder(self.opts, phantom=phantom)

    # Clean a set of cases
    @run_rootdir
    def CleanCases(self, **kw):
        """Clean a list of cases using *Progress* archive options only

        :Call:
            >>> cntl.CleanCases(**kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control interface
        :Versions:
            * 2017-03-13 ``@ddalle``: Version 1.0
        """
        # Loop through the folders
        for i in self.x.GetIndices(**kw):
            # Go to root folder
            os.chdir(self.RootDir)
            # Get folder name
            frun = self.x.GetFullFolderNames(i)
            # Check if the case is ready to archive
            if not os.path.isdir(frun):
                continue
            # Status update
            print(frun)
            # Enter the case folder
            os.chdir(frun)
            # Perform cleanup
            self.CleanPWD(phantom=kw.get("phantom",False))

    # Individual case archive function
    def CleanPWD(self, phantom=False):
        """Archive a single case in the current folder ($PWD)

        :Call:
            >>> cntl.CleanPWD(phantom=False)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control interface
            *phantom*: ``True`` | {``False``}
                Write actions to ``archive.log``; only delete if ``False``
        :Versions:
            * 2017-03-10 ``@ddalle``: Version 1.0
            * 2017-12-15 ``@ddalle``: Added *phantom* option
        """
        # Archive using the local module
        manage.CleanFolder(self.opts, phantom=phantom)

    # Unarchive cases
    @run_rootdir
    def UnarchiveCases(self, **kw):
        """Unarchive a list of cases

        :Call:
            >>> cntl.UnarchiveCases(**kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control interface
        :Versions:
            * 2017-03-13 ``@ddalle``: Version 1.0
        """
        # Loop through the folders
        for i in self.x.GetIndices(**kw):
            # Go to root folder
            os.chdir(self.RootDir)
            # Get folder name
            fgrp = self.x.GetGroupFolderNames(i)
            frun = self.x.GetFullFolderNames(i)
            fdir = self.x.GetFolderNames(i)
            # Status update
            print(frun)
            # Check if the group folder exists
            if not os.path.isdir(fgrp):
                # Greate folder
                self.mkdir(fgrp)
            # Check if the case is ready to archive
            if not os.path.isdir(frun):
                # Create folder temporarily
                self.mkdir(frun)
            # Enter the folder
            os.chdir(frun)
            # Run the unarchive command
            manage.UnarchiveFolder(self.opts)
            # Check if there were any files created
            fls = os.listdir('.')
            # If no files, delete the folder
            if len(fls) == 0:
                # Go up one level
                os.chdir('..')
                # Delete the folder
                os.rmdir(fdir)
   # >

   # =========
   # CPU Stats
   # =========
   # <
    # Get CPU hours (actually core hours)
    @run_rootdir
    def GetCPUTimeFromFile(self, i, fname='cape_time.dat'):
        """Read a Cape-style core-hour file

        :Call:
            >>> CPUt = cntl.GetCPUTimeFromFile(i, fname)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Cape control interface
            *i*: :class:`int`
                Case index
            *fname*: :class:`str`
                Name of file containing timing history
        :Outputs:
            *CPUt*: :class:`float` | ``None``
                Total core hours used in this job
        :Versions:
            * 2015-12-22 ``@ddalle``: Version 1.0
        """
        # Get the group name.
        frun = self.x.GetFullFolderNames(i)
        # Check if the folder exists.
        if (not os.path.isdir(frun)):
            return None
        # Go to the case folder.
        os.chdir(frun)
        # Check if the file exists.
        if not os.path.isfile(fname):
            return None
        # Read the time.
        try:
            # Read the first column of data
            CPUt = np.loadtxt(fname, comments='#', usecols=(0,), delimiter=',')
            # Return the total.
            return np.sum(CPUt)
        except Exception:
            # Could not read file
            return None

    # Get CPU hours currently running
    @run_rootdir
    def GetCPUTimeFromStartFile(self, i, fname='cape_start.dat'):
        """Read a Cape-style start time file and compare to current time

        :Call:
            >>> CPUt = cntl.GetCPUTimeFromStartFile(i, fname)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Cape control interface
            *i*: :class:`int`
                Case index
            *fname*: :class:`str`
                Name of file containing timing history
        :Outputs:
            *CPUt*: :class:`float` | ``None``
                Total core hours used in this job
        :Versions:
            * 2015-08-30 ``@ddalle``: Version 1.0
        """
        # Get the group name.
        frun = self.x.GetFullFolderNames(i)
        # Check if the folder exists.
        if (not os.path.isdir(frun)):
            return 0.0
        # Go to the case folder.
        os.chdir(frun)
        # Try to read the file
        nProc, tic = case.ReadStartTimeProg(fname)
        # Check for empty
        if tic is None:
            # Could not read or nothing to read
            return 0.0
        # Safety
        try:
            # Get current time
            toc = case.datetime.now()
            # Subtract time
            t = toc - tic
            # Calculate CPU hours
            CPUt = nProc * (t.days*24 + t.seconds/3600.0)
            # Output
            return CPUt
        except Exception:
            return 0.0

    # Get total CPU hours (core hours) with file names as inputs
    def GetCPUTimeBoth(self, i, fname, fstart, running=False):
        """Read Cape-style core-hour files from a case

        This function needs to be customized for each solver because it needs to
        know the name of the file in which timing data is saved.  It defaults to
        :file:`cape_time.dat`.  Modifying this command is a one-line fix with a
        call to :func:`cape.cntl.Cntl.GetCPUTimeFromFile` with the correct file
        name.

        :Call:
            >>> CPUt = cntl.GetCPUTimeBoth(i, fname, fstart, running=False)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Cape control interface
            *i*: :class:`int`
                Case index
        :Outputs:
            *CPUt*: :class:`float` | ``None``
                Total core hours used in this job
        :Versions:
            * 2015-12-22 ``@ddalle``: Version 1.0
            * 2016-08-30 ``@ddalle``: Checking for currently running cases
        """
        # Call the time from finished cases
        CPUf = self.GetCPUTimeFromFile(i, fname=fname)
        # Check for currently running case request
        if running:
            # Get time since last start
            CPUr = self.GetCPUTimeFromStartFile(i, fname=fstart)
            # Return the sum
            if CPUf is None:
                # No finished jobs
                return CPUr
            elif CPUr is None:
                # No running time
                return CPUf
            else:
                # Add them together
                return CPUf + CPUr
        else:
            # Just the time of finished jobs
            return CPUf

    # Get total CPU hours (actually core hours)
    def GetCPUTime(self, i, running=False):
        """Read a Cape-style core-hour file from a case

        This function needs to be customized for each solver because it needs to
        know the name of the file in which timing data is saved.  It defaults to
        :file:`cape_time.dat`.  Modifying this command is a one-line fix with a
        call to :func:`cape.cntl.Cntl.GetCPUTimeFromFile` with the correct file
        name.

        :Call:
            >>> CPUt = cntl.GetCPUTime(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Cape control interface
            *i*: :class:`int`
                Case index
        :Outputs:
            *CPUt*: :class:`float` | ``None``
                Total core hours used in this job
        :Versions:
            * 2015-12-22 ``@ddalle``: Version 1.0
            * 2016-08-30 ``@ddalle``: Checking for currently running cases
            * 2016-08-31 ``@ddalle``: Moved parts to :func:`GetCPUTimeBoth`
        """
        # File names
        fname = 'cape_time.dat'
        fstrt = 'cape_start.dat'
        # Call with base file names
        return self.GetCPUTimeBoth(i, fname, fstrt, running=running)

   # >

   # ========
   # PBS Jobs
   # ========
   # <
    # Get PBS name
    def GetPBSName(self, i, pre=None):
        """Get PBS name for a given case

        :Call:
            >>> lbl = cntl.GetPBSName(i, pre=None)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
            *pre*: {``None``} | :class:`str`
                Prefix for PBS job name
        :Outputs:
            *lbl*: :class:`str`
                Short name for the PBS job, visible via `qstat`
        :Versions:
            * 2014-09-30 ``@ddalle``: Version 1.0
            * 2016-12-20 ``@ddalle``: Moved to *x* and added prefix
        """
        # Call from trajectory
        return self.x.GetPBSName(i, pre=pre)

    # Get PBS job ID if possible
    @run_rootdir
    def GetPBSJobID(self, i):
        """Get PBS job number if one exists

        :Call:
            >>> pbs = cntl.GetPBSJobID(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Run index
        :Outputs:
            *pbs*: :class:`int` or ``None``
                Most recently reported job number for case *i*
        :Versions:
            * 2014-10-06 ``@ddalle``: Version 1.0
        """
        # Check the case.
        if self.CheckCase(i) is None: return None
        # Get the run name.
        frun = self.x.GetFullFolderNames(i)
        # Go there.
        os.chdir(frun)
        # Check for a "jobID.dat" file.
        if os.path.isfile('jobID.dat'):
            # Read the file.
            try:
                # Open the file and read the first line.
                line = open('jobID.dat').readline()
                # Get the job ID.
                pbs = int(line.split()[0])
            except Exception:
                # Unsuccessful reading for some reason.
                pbs = None
        else:
            # No file.
            pbs = None
        # Output
        return pbs

    # Write a PBS header
    def WritePBSHeader(self, f, i=None, j=0, typ=None, wd=None, pre=None):
        """Write common part of PBS or Slurm script

        :Call:
            >>> cntl.WritePBSHeader(f, i=None, j=0, typ=None, wd=None)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *f*: :class:`file`
                Open file handle
            *i*: {``None``} | :class:`int`
                Case index (ignore if ``None``); used for PBS job name
            *j*: :class:`int`
                Phase number
            *typ*: {``None``} | ``"batch"`` | ``"post"``
                Group of PBS options to use
            *wd*: {``None``} | :class:`str`
                Folder to enter when starting the job
            *pre*: {``None``} | :class:`str`
                PBS job name prefix, used for postprocessing
        :Versions:
            * 2015-09-30 ``@ddalle``: Separated from WritePBS
            * 2016-09-25 ``@ddalle``: Supporting "BatchPBS" and "PostPBS"
            * 2016-12-20 ``@ddalle``: Consolidated to *opts*, added prefix
        """
        # Get the shell name.
        if i is None:
            # Batch job
            lbl = '%s-batch' % self.__module__.split('.')[0].lower()
            # Ensure length
            if len(lbl) > 15: lbl = lbl[:15]
        else:
            # Case PBS job name
            lbl = self.GetPBSName(i, pre=pre)
        # Check the task manager
        if self.opts.get_sbatch(j):
            # Write the Slurm header
            self.opts.WriteSlurmHeader(f, lbl, j=j, typ=typ, wd=wd)
        else:
            # Call the function from *opts*
            self.opts.WritePBSHeader(f, lbl, j=j, typ=typ, wd=wd)

    # Write batch PBS job
    @run_rootdir
    def SubmitBatchPBS(self, argv):
        """Write a PBS script for a batch run

        :Call:
            >>> cntl.SubmitBatchPBS(argv)
        :Inputs:
            *argv*: :class:`list` (:class:`str`)
                List of command-line inputs
        :Versions:
            * 2016-09-25 ``@ddalle``: Version 1.0
        """
        # Write string... add quotes if needed
        def convertval(v):
            # Check type
            t = type(v).__name__
            # Check for certain characters if string
            if t not in ['str', 'unicode']:
                # Just convert to string
                return str(v)
            elif re.search("[,<>=]", v):
                # Add quotes
                return '"%s"' % v
            else:
                # Just convert to string
                return v
        # Convert keyword to string
        def convertkey(cmdi, k, v):
            if v == False:
                # Add --no- prefix
                cmdi.append('--no-%s' % k)
            elif v == True:
                # No extra value
                if len(k) == 1:
                    cmdi.append('-%s' % k)
                else:
                    cmdi.append('--%s' % k)
            else:
                # Append the key and value
                if len(k) == 1:
                    cmdi.append('-%s' % k)
                    cmdi.append('%s' % convertval(v))
                else:
                    cmdi.append('--%s' % k)
                    cmdi.append('%s' % convertval(v))
        # -------------------
        # Command preparation
        # -------------------
        # Process keys as keys
        a, kw = argread.readkeys(argv)
        # Check for alternate input method
        if kw.get('keys', False):
            # Do nothing
            pass
        elif kw.get('flags', False):
            # Reprorcess as flags
            a, kw = argread.readflags(argv)
        else:
            # Default: process like tar
            a, kw = argread.readflagstar(argv)
        # Initialize the command
        if 'prog' in kw:
            # Initialize command with specific program
            cmdi = [kw['prog']]
        else:
            # Get the name of the command (clear out full path)
            cmdj = os.path.split(argv[0])[-1]
            # Initialize command with same program as argv
            cmdi = [cmdj]
        # Loop through non-keyword arguments
        for ai in a: cmdi.append(a)
        # Turn off all QSUB operations unless --qsub given explicitly
        if 'qsub' not in kw: kw['qsub'] = False
        # Loop through _old arguments
        for d in kw.get("_old", []):
            # Check type
            if type(d).__name__ != "dict": continue
            # Number of keys
            K = d.keys()
            nk = len(K)
            # Check number of keys
            if nk != 1: continue
            # Convert to string
            convertkey(cmdi, K[0], d[K[0]])
        # Loop through keyword arguments
        for k in kw:
            # Check for skipped keys
            if k in ['batch', 'flags', 'keys', 'prog', '_old']:
                continue
            # Convert to string
            convertkey(cmdi, k, kw[k])
        # Turn off all QSUB operations unless --qsub given explicitly
        if 'qsub' not in kw: kw['qsub'] = False
        # ------------------
        # Folder preparation
        # ------------------
        # Create the folder if necessary
        if not os.path.isdir('batch-pbs'): os.mkdir('batch-pbs')
        # Enter the batch pbs folder
        os.chdir('batch-pbs')
        # ----------------
        # File preparation
        # ----------------
        # File name header
        prog = self.__module__.split('.')[0].lower()
        # Current time
        fnow = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # File name
        fpbs = '%s-%s.pbs' % (prog, fnow)
        # Write the file
        f = open(fpbs, 'w')
        # Write header
        self.WritePBSHeader(f, typ='batch', wd=self.RootDir)
        # Write the command
        f.write('\n# Run the command\n')
        f.write('%s\n\n' % (" ".join(cmdi)))
        # Close the file
        f.close()
        # ------------------
        # Submit and Cleanup
        # ------------------
        # Submit the job
        if self.opts.get_sbatch(0):
            # Submit Slurm job
            pbs = queue.sbatch(fpbs)
        else:
            # Submit PBS job
            pbs = queue.pqsub(fpbs)
   # >

   # ================
   # Case Preparation
   # ================
   # <
    # Prepare a case
    @run_rootdir
    def PrepareCase(self, i):
        """Prepare case for running if necessary

        This function creates the folder, copies mesh files, and saves settings
        and input files.  All of these tasks are completed only if they have not
        already been completed, and it needs to be customized for each CFD
        solver.

        :Call:
            >>> cntl.PrepareCase(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of case to analyze
        :Versions:
            * 2014-09-30 ``@ddalle``: Version 1.0
            * 2015-09-27 ``@ddalle``: Template version
        """
        # Get the existing status.
        n = self.CheckCase(i)
        # Quit if prepared.
        if n is not None: return None
        # Get the run name.
        frun = self.x.GetFullFolderNames(i)
        # Case function
        self.CaseFunction(i)
        # Make the directory if necessary.
        if not os.path.isdir(frun): self.mkdir(frun)
        # Go there.
        os.chdir(frun)
        # Write the conditions to a simple JSON file.
        self.x.WriteConditionsJSON(i)
        # Write a JSON files with contents of "RunControl" section
        self.WriteCaseJSON(i)

    # Function to apply transformations to config
    def PrepareConfig(self, i):
        """Apply rotations, translations, or other features to ``Config.xml``

        :Call:
            >>> cntl.PrepareConfig(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Case index
        :Versions:
            * 2016-08-23 ``@ddalle``: Version 1.0
        """
        # Get function for rotations, etc.
        keys = self.x.GetKeysByType(['translate', 'rotate', 'ConfigFunction'])
        # Exit if no keys
        if len(keys) == 0: return
        # Reset reference points
        self.opts.reset_Points()
        # Loop through keys.
        for key in keys:
            # Type
            kt = self.x.defns[key]['Type']
            # Filter on which type of configuration modification it is
            if kt == "ConfigFunction":
                # Special config.xml function
                self.PrepareConfigFunction(key, i)
            elif kt.lower() == "translate":
                # Component(s) translation
                self.PrepareConfigTranslation(key, i)
            elif kt.lower() == "rotate":
                # Component(s) translation
                self.PrepareConfigRotation(key, i)
        # Write the configuration file
        self.WriteConfig(i)

    # Write run control options to JSON file
    @run_rootdir
    def WriteCaseJSON(self, i, rc=None):
        """Write JSON file with run control and related settings for case *i*

        :Call:
            >>> cntl.WriteCaseJSON(i, rc=None)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Generic control class
            *i*: :class:`int`
                Run index
            *rc*: {``None``} | :class:`pyOver.options.runControl.RunControl`
                If specified, write specified "RunControl" options
        :Versions:
            * 2015-10-19 ``@ddalle``: Version 1.0
            * 2013-03-31 ``@ddalle``: Can now write other options
        """
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Check if it exists.
        if not os.path.isdir(frun):
            return
        # Go to the folder.
        os.chdir(frun)
        # Write folder.
        f = open('case.json', 'w')
        # Dump the Overflow and other run settings.
        if rc is None:
            # Write settings from the present options
            json.dump(self.opts['RunControl'], f, indent=1)
        else:
            # Write the settings given as input
            json.dump(rc, f, indent=1)
        # Close the file.
        f.close()

    # Read run control options from case JSON file
    @run_rootdir
    def ReadCaseJSON(self, i):
        """Read ``case.json`` file from case *i* if possible

        :Call:
            >>> rc = cntl.ReadCaseJSON(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class
            *i*: :class:`int`
                Run index
        :Outputs:
            *rc*: ``None`` | :class:`pyOver.options.runControl.RunControl`
                Run control interface read from ``case.json`` file
        :Versions:
            * 2016-12-12 ``@ddalle``: Version 1.0
            * 2017-04-12 ``@ddalle``: Added to :mod:`cape.Cntl`
        """
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Check if it exists.
        if not os.path.isdir(frun):
            return
        # Go to the folder.
        os.chdir(frun)
        # Check for file
        if not os.path.isfile('case.json'):
            # Nothing to read
            rc = None
        else:
            # Read the file
            rc = case.ReadCaseJSON()
        # Output
        return rc

   # >

   # =============
   # Geometry Prep
   # =============
   # <
    # Function to apply special triangulation modification keys
    def PrepareTri(self, i):
        """Rotate/translate/etc. triangulation for given case

        :Call:
            >>> cntl.PrepareTri(i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2014-12-01 ``@ddalle``: Version 1.0
            * 2016-04-05 ``@ddalle``: Moved from pyCart -> cape
        """
        # Get function for rotations, etc.
        keys = self.x.GetKeysByType(['translation', 'rotation', 'TriFunction'])
        # Reset reference points
        self.opts.reset_Points()
        # Loop through keys.
        for key in keys:
            # Type
            kt = self.x.defns[key]['Type']
            # Filter on which type of triangulation modification it is.
            if kt == "TriFunction":
                # Special triangulation function
                self.PrepareTriFunction(key, i)
            elif kt.lower() == "translation":
                # Component(s) translation
                self.PrepareTriTranslation(key, i)
            elif kt.lower() == "rotation":
                # Component(s) rotation
                self.PrepareTriRotation(key, i)

    # Apply a special triangulation function
    def PrepareTriFunction(self, key, i):
        """Apply special triangulation modification function for a case

        :Call:
            >>> cntl.PrepareTriFunction(key, i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *key*: :class:`str`
                Name of key
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2015-09-11 ``@ddalle``: Version 1.0
            * 2016-04-05 ``@ddalle``: Moved from pyCart -> cape
        """
        # Get the function for this *TriFunction*
        func = self.x.defns[key]['Function']
        # Apply it.
        exec("%s(self,%s,i=%i)" % (func, self.x[key][i], i))

    # Apply a special configuration function
    def PrepareConfigFunction(self, key, i):
        """Apply special configuration modification function for a case

        :Call:
            >>> cntl.PrepareConfigFunction(key, i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *key*: :class:`str`
                Name of key
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2016-08-23 ``@ddalle``: Copied from :func:`PrepareTriFunction`
        """
        # Get the function for this *TriFunction*
        func = self.x.defns[key]['Function']
        # Apply it.
        exec("%s(self,%s,i=%i)" % (func, self.x[key][i], i))

    # Apply a triangulation translation
    def PrepareTriTranslation(self, key, i):
        """Apply a translation to a component or components

        :Call:
            >>> cntl.PrepareTriTranslation(key, i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2015-09-11 ``@ddalle``: Version 1.0
            * 2016-04-05 ``@ddalle``: Moved from pyCart -> cape
        """
        # Get the options for this key.
        kopts = self.x.defns[key]
        # Get the components to translate.
        compID  = self.tri.GetCompID(kopts.get('CompID'))
        # Components to translate in opposite direction
        compIDR = self.tri.GetCompID(kopts.get('CompIDSymmetric', []))
        # Check for a direction
        if 'Vector' not in kopts:
            raise IOError(
                "Rotation key '%s' does not have a 'Vector'." % key)
        # Get the direction and its type
        vec = kopts['Vector']
        tvec = type(vec).__name__
        # Get points to translate along with it.
        pts  = kopts.get('Points', [])
        ptsR = kopts.get('PointsSymmetric', [])
        # Make sure these are lists.
        if type(pts).__name__  != 'list': pts  = list(pts)
        if type(ptsR).__name__ != 'list': ptsR = list(ptsR)
        # Check the type
        if tvec in ['list', 'ndarray']:
            # Specified directly.
            u = np.array(vec)
        else:
            # Named vector
            u = np.array(self.opts.get_Point(vec))
        # Form the translation vector
        v = u * self.x[key][i]
        # Translate the triangulation
        self.tri.Translate(v, compID=compID)
        self.tri.Translate(-v, compID=compIDR)
        # Loop through translation points.
        for pt in pts:
            # Get point
            x = self.opts.get_Point(pt)
            # Apply transformation.
            self.opts.set_Point(x+v, pt)
        # Loop through translation points.
        for pt in ptsR:
            # Get point
            x = self.opts.get_Point(pt)
            # Apply transformation.
            self.opts.set_Point(x-v, pt)

    # Apply a triangulation translation
    def PrepareConfigTranslation(self, key, i):
        """Apply a translation to a component or components

        :Call:
            >>> cntl.PrepareConfigTranslation(key, i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *key*: :class:`str`
                Name of variable from which to get value
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2016-08-23 ``@ddalle``: Version 1.0
        """
        # Get the options for this key.
        kopts = self.x.defns[key]
        # Get the components to translate.
        comps = kopts.get("CompID", [])
        comps = list(np.array(comps).flatten())
        # Components to translate in opposite direction
        compsR = kopts.get("CompIDSymmetric", [])
        compsR = list(np.array(compsR).flatten())
        # Get index of transformation (which order in Config.xml)
        I = kopts.get('TransformationIndex')
        # Process the transformation indices
        if type(I).__name__ != 'dict':
            # Initialize index dictionary.
            J = {}
            # Loop through all components
            for comp in (comps+compsR):
                J[comp] = I
            # Transfer values
            I = J
        # Check for a direction
        if 'Vector' not in kopts:
            raise KeyError(
                "Translation key '%s' does not have a 'Vector'." % key)
        # Get the direction and its type
        vec = kopts['Vector']
        tvec = type(vec).__name__
        # Get points to translate along with it.
        pts  = kopts.get('Points', [])
        ptsR = kopts.get('PointsSymmetric', [])
        # Make sure these are lists.
        if type(pts).__name__  != 'list': pts  = list(pts)
        if type(ptsR).__name__ != 'list': ptsR = list(ptsR)
        # Check the type
        if tvec in ['list', 'ndarray']:
            # Specified directly.
            u = np.array(vec)
        else:
            # Named vector
            u = np.array(self.opts.get_Point(vec))
        # Form the translation vector
        v = u * self.x[key][i]
        # Set the displacement for the positive translations
        for comp in comps:
            self.config.SetTranslation(comp, i=I.get(comp), Displacement=v)
        # Set the displacement for the negative translations
        for comp in compsR:
            self.config.SetTranslation(comp, i=I.get(comp), Displacement=-v)
        # Loop through translation points.
        for pt in pts:
            # Get point
            x = self.opts.get_Point(pt)
            # Apply transformation.
            self.opts.set_Point(x+v, pt)
        # Loop through translation points.
        for pt in ptsR:
            # Get point
            x = self.opts.get_Point(pt)
            # Apply transformation.
            self.opts.set_Point(x-v, pt)

    # Apply a triangulation rotation
    def PrepareTriRotation(self, key, i):
        """Apply a rotation to a component or components

        :Call:
            >>> cntl.PrepareTriRotation(key, i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2015-09-11 ``@ddalle``: Version 1.0
            * 2016-04-05 ``@ddalle``: Moved from pyCart -> cape
        """
        # ---------------
        # Read the inputs
        # ---------------
        # Get the options for this key.
        kopts = self.x.defns[key]
        # Rotation angle
        theta = self.x[key][i]
        # Get the components to translate.
        compID = self.tri.GetCompID(kopts.get('CompID'))
        # Components to translate in opposite direction
        compIDR = self.tri.GetCompID(kopts.get('CompIDSymmetric', []))
        # Get the components to translate based on a lever armg
        compsT  = kopts.get('CompIDTranslate', [])
        compsTR = kopts.get('CompIDTranslateSymmetric', [])
        # Symmetry applied to rotation vector.
        kv = kopts.get('VectorSymmetry', [1.0, 1.0, 1.0])
        kx = kopts.get('AxisSymmetry',   kv)
        kc = kopts.get('CenterSymmetry', kx)
        ka = kopts.get('AngleSymmetry', -1.0)
        # Convert symmetries: list -> numpy.ndarray
        if type(kv).__name__ == "list": kv = np.array(kv)
        if type(kx).__name__ == "list": kx = np.array(kx)
        if type(kc).__name__ == "list": kc = np.array(kc)
        # Get the reference points for translations based on this rotation
        xT = kopts.get('TranslateRefPoint', [0.0, 0.0, 0.0])
        # Get scale for translated points
        kt = kopts.get('TranslateScale', np.ones(3))
        # Ensure vector
        if type(kt).__name__ == 'list':
            # Ensure vector so that we can multiply it by another vector
            kt = np.array(kt)
        # Get vector
        vec = kopts.get('Vector')
        ax  = kopts.get('Axis')
        cen = kopts.get('Center')
        frm = kopts.get('Frame')
        # Get points to translate along with it.
        pts  = kopts.get('Points', [])
        ptsR = kopts.get('PointsSymmetric', [])
        # Make sure these are lists.
        if type(pts).__name__  != 'list': pts  = list(pts)
        if type(ptsR).__name__ != 'list': ptsR = list(ptsR)
        # ---------------------------
        # Process the rotation vector
        # ---------------------------
        # Check for an axis and center
        if vec is not None:
            # Check type
            if len(vec) != 2:
                raise KeyError(
                    "Rotation key '%s' vector must be exactly two points."
                    % key)
            # Get start and end points of rotation vector.
            v0 = np.array(self.opts.get_Point(vec[0]))
            v1 = np.array(self.opts.get_Point(vec[1]))
            # Convert to axis and center
            cen = v0
            ax  = v1 - v0
        else:
            # Get default axis if necessary
            if ax is None:
                ax = [0.0, 1.0, 0.0]
            # Get default center if necessary
            if cen is None:
                cen = [0.0, 0.0, 0.0]
            # Convert points
            cen = np.array(self.opts.get_Point(cen))
            ax  = np.array(self.opts.get_Point(ax))
        # Symmetry rotation vectors.
        axR  = kx*ax
        cenR = kc*cen
        # Form vectors
        v0  = cen;  v1  = ax + cen
        v0R = cenR; v1R = axR + cenR
        # Ensure a dictionary for reference points
        if type(xT).__name__ != 'dict':
            # Initialize dict (can't use an iterator to do this in old Python)
            yT = {}
            # Loop through components affected by this translation
            for comp in compsT+compsTR:
                yT[comp] = xT
            # Move the variable name
            xT = yT
        # Create full dictionary
        for comp in compsT+compsTR:
            # Get ref point for this component
            pt = xT.get(comp, xT.get('def', [0.0, 0.0, 0.0]))
            # Save it as a dimensionalized point
            xT[comp] = np.array(self.opts.get_Point(pt))
        # ---------------------
        # Apply transformations
        # ---------------------
        # Rotate the triangulation.
        self.tri.Rotate(v0,  v1,  theta,  compID=compID)
        self.tri.Rotate(v0R, v1R, ka*theta, compID=compIDR)
        # Points to be rotated
        X  = np.array([self.opts.get_Point(pt) for pt in pts])
        XR = np.array([self.opts.get_Point(pt) for pt in ptsR])
        # Reference points to be rotated
        XT  = np.array([xT[comp] for comp in compsT])
        XTR = np.array([xT[comp] for comp in compsTR])
        # Apply transformation
        Y   = RotatePoints(X,   v0,  v1,  theta)
        YT  = RotatePoints(XT,  v0,  v1,  theta)
        YR  = RotatePoints(XR,  v0R, v1R, ka*theta)
        YTR = RotatePoints(XTR, v0R, v1R, ka*theta)
        # Process translations caused by this rotation
        for j in range(len(compsT)):
            self.tri.Translate(kt*(YT[j]-XT[j]), compID=compsT[j])
        # Process translations caused by symmetric rotation
        for j in range(len(compsTR)):
            self.tri.Translate(kt*(YTR[j]-XTR[j]), compID=compsTR[j])
        # Apply transformation
        Y  = RotatePoints(X,  v0,  v1,  theta)
        YR = RotatePoints(XR, v0R, v1R, ka*theta)
        # Save the points.
        for j in range(len(pts)):
            # Set the new value.
            self.opts.set_Point(Y[j], pts[j])
        # Save the symmetric points.
        for j in range(len(ptsR)):
            # Set the new value.
            self.opts.set_Point(YR[j], ptsR[j])


    # Apply a configuration rotation
    def PrepareConfigRotation(self, key, i):
        """Apply a rotation to a component or components

        :Call:
            >>> cntl.PrepareConfigRotation(key, i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *key*: :class:`str`
                Name of the trajectory key
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2016-08-23 ``@ddalle``: Version 1.0
        """
        # ---------------
        # Read the inputs
        # ---------------
        # Get the options for this key.
        kopts = self.x.defns[key]
        # Rotation angle
        theta = self.x[key][i]
        # Get the components to rotate.
        comps = kopts.get('CompID', [])
        # Components to rotate in opposite direction
        compsR = kopts.get('CompIDSymmetric', [])
        # Get the components to translate based on a lever armg
        compsT  = kopts.get('CompIDTranslate', [])
        compsTR = kopts.get('CompIDTranslateSymmetric', [])
        # Ensure list
        if type(comps).__name__   != 'list': comps = [comps]
        if type(compsR).__name__  != 'list': compsR = [compsR]
        if type(compsT).__name__  != 'list': compsT = [compsT]
        if type(compsTR).__name__ != 'list': compsTR = [compsTR]
        # Get index of transformation (which order in Config.xml)
        I = kopts.get('TransformationIndex')
        # Symmetry applied to rotation vector.
        kv = kopts.get('VectorSymmetry', [1.0, 1.0, 1.0])
        kx = kopts.get('AxisSymmetry',   kv)
        kc = kopts.get('CenterSymmetry', kx)
        ka = kopts.get('AngleSymmetry', -1.0)
        # Convert symmetries: list -> numpy.ndarray
        if type(kv).__name__ == "list": kv = np.array(kv)
        if type(kx).__name__ == "list": kx = np.array(kx)
        if type(kc).__name__ == "list": kc = np.array(kc)
        # Get the reference points for translations based on this rotation
        xT  = kopts.get('TranslateRefPoint', [0.0, 0.0, 0.0])
        # Get scale for translated points
        kt = kopts.get('TranslateScale', np.ones(3))
        # Ensure vector
        if type(kt).__name__ == 'list':
            # Ensure vector so that we can multiply it by another vector
            kt = np.array(kt)
        # Get vector
        vec = kopts.get('Vector')
        ax  = kopts.get('Axis')
        cen = kopts.get('Center')
        frm = kopts.get('Frame')
        # Get points to translate along with it.
        pts  = kopts.get('Points', [])
        ptsR = kopts.get('PointsSymmetric', [])
        # Make sure these are lists.
        if type(pts).__name__  != 'list': pts  = list(pts)
        if type(ptsR).__name__ != 'list': ptsR = list(ptsR)
        # ---------------------------
        # Process the rotation vector
        # ---------------------------
        # Check for an axis and center
        if vec is not None:
            # Check type
            if len(vec) != 2:
                raise KeyError(
                    "Rotation key '%s' vector must be exactly two points."
                    % key)
            # Get start and end points of rotation vector.
            v0 = np.array(self.opts.get_Point(vec[0]))
            v1 = np.array(self.opts.get_Point(vec[1]))
            # Convert to axis and center
            cen = v0
            ax  = v1 - v0
        else:
            # Get default axis if necessary
            if ax is None:
                ax = [0.0, 1.0, 0.0]
            # Get default center if necessary
            if cen is None:
                cen = [0.0, 0.0, 0.0]
            # Convert points
            cen = np.array(self.opts.get_Point(cen))
            ax  = np.array(self.opts.get_Point(ax))
        # Symmetry rotation vectors.
        axR  = kx*ax
        cenR = kc*cen
        # Form vectors
        v0  = cen;  v1  = ax + cen
        v0R = cenR; v1R = axR + cenR
        # Ensure a dictionary for reference points
        if type(xT).__name__ != 'dict':
            # Initialize dict (can't use an iterator to do this in old Python)
            yT = {}
            # Loop through components affected by this translation
            for comp in compsT+compsTR:
                yT[comp] = xT
            # Move the variable name
            xT = yT
        # Create full dictionary
        for comp in compsT+compsTR:
            # Get ref point for this component
            pt = xT.get(comp, xT.get('def', [0.0, 0.0, 0.0]))
            # Save it as a dimensionalized point
            xT[comp] = np.array(self.opts.get_Point(pt))
        # Process the transformation indices
        if type(I).__name__ != 'dict':
            # Initialize index dictionary.
            J = {}
            # Loop through all components
            for comp in (comps+compsR+compsT+compsTR):
                J[comp] = I
            # Transfer values
            I = J
        # ---------------------
        # Apply transformations
        # ---------------------
        # Set the positive rotations.
        for comp in comps:
            self.config.SetRotation(comp, i=I.get(comp),
                Angle=theta, Center=cen, Axis=ax, Frame=frm)
        # Set the negative rotations.
        for comp in compsR:
            self.config.SetRotation(comp, i=I.get(comp),
                Angle=ka*theta, Center=cenR, Axis=axR, Frame=frm)
        # Points to be rotated
        X  = np.array([self.opts.get_Point(pt) for pt in pts])
        XR = np.array([self.opts.get_Point(pt) for pt in ptsR])
        # Reference points to be rotated
        XT  = np.array([xT[comp] for comp in compsT])
        XTR = np.array([xT[comp] for comp in compsTR])
        # Apply transformation
        Y   = RotatePoints(X,   v0,  v1,  theta)
        YT  = RotatePoints(XT,  v0,  v1,  theta)
        YR  = RotatePoints(XR,  v0R, v1R, ka*theta)
        YTR = RotatePoints(XTR, v0R, v1R, ka*theta)
        # Process translations caused by this rotation
        for j in range(len(compsT)):
            # Get component
            comp = compsT[j]
            # Apply translation
            self.config.SetTranslation(comp, i=I.get(comp),
                Displacement=kt*(YT[j]-XT[j]))
        # Process translations caused by symmetric rotation
        for j in range(len(compsTR)):
            # Get component
            comp = compsTR[j]
            # Apply translation
            self.config.SetTranslation(comp, i=I.get(comp),
                Displacement=kt*(YTR[j]-XTR[j]))
        # Save the points.
        for j in range(len(pts)):
            # Set the new value.
            self.opts.set_Point(Y[j], pts[j])
        # Save the symmetric points.
        for j in range(len(ptsR)):
            # Set the new value.
            self.opts.set_Point(YR[j], ptsR[j])

   # >

   # ==================
   # Thrust Preparation
   # ==================
   # <
    # Get exit area for SurfCT boundary condition
    def GetSurfCT_ExitArea(self, key, i, comp=None):
        """Get exit area for a *CT* trajectory key

        This can use either the area ratio (if available) or calculate from the
        exit Mach number.  The input area is determined from the component ID.
        If using the exit Mach number *M2*, the input Mach number *M1* is also
        needed.  The relationship between area ratio and exit Mach is given
        below.

            .. math::

                \\frac{A_2}{A_1} = \\frac{M_1}{M_2}\\left(
                    \\frac{1+\\frac{\\gamma-1}{2}M_2^2}{
                    1+\\frac{\\gamma-1}{2}M_1^2}
                \\right) ^ {\\frac{1}{2}\\frac{\\gamma+1}{\\gamma-1}}

        :Call:
            >>> A2 = cntl.GetSurfCT_ExitArea(key, i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *key*: :class:`str`
                Name of trajectory key to check
            *i*: :class:`int`
                Case number
            *comp*: {``None``} | :class:`str`
                Name of component for which to get BCs
        :Outputs:
            *A2*: :class:`list` (:class:`float`)
                Exit area for each component referenced by this key
        :Versions:
            * 2016-04-13 ``@ddalle``: Version 1.0
        """
        # Check for exit area
        A2 = self.x.GetSurfCT_ExitArea(i, key, comp=comp)
        # Check for a results
        if A2 is not None: return A2
        # Ensure triangulation if necessary
        self.ReadTri()
        # Get component(s)
        if comp is None:
            # Hopefully there is only one component
            comp = self.x.GetSurfCT_CompID(i, key)
            # Ensure one component
            if isinstance(comp, (list, np.ndarray)):
                comp = comp[0]
        # Input area(s)
        A1 = self.tri.GetCompArea(comp)
        # Check for area ratio
        AR = self.x.GetSurfCT_AreaRatio(i, key)
        # Check if we need to use Mach number
        if AR is None:
            # Get input and exit Mach numbers
            M1 = self.x.GetSurfCT_Mach(i, key)
            M2 = self.x.GetSurfCT_ExitMach(i, key)
            # Gas constants
            gam = self.GetSurfCT_Gamma(i, key)
            g1 = 0.5 * (gam+1) / (gam-1)
            g2 = 0.5 * (gam-1)
            # Ratio
            AR = M1/M2 * ((1+g2*M2*M2) / (1+g2*M1*M1))**g1
        # Return exit areas
        return A1*AR

    # Get exit Mach number for SurfCT boundary condition
    def GetSurfCT_ExitMach(self, key, i, comp=None):
        """Get exit Mach number for a *CT* trajectory key

        This can use either the ``"ExitMach"`` parameter (if available) or
        calculate from the area ratio.  If using the area ratio, the input Mach
        number is also needed.  The relationship between area ratio and exit
        Mach is given below.

            .. math::

                \\frac{A_2}{A_1} = \\frac{M_1}{M_2}\\left(
                    \\frac{1+\\frac{\\gamma-1}{2}M_2^2}{
                    1+\\frac{\\gamma-1}{2}M_1^2}
                \\right) ^ {\\frac{1}{2}\\frac{\\gamma+1}{\\gamma-1}}

        :Call:
            >>> M2 = cntl.GetSurfCT_ExitMach(key, i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *key*: :class:`str`
                Name of trajectory key to check
            *i*: :class:`int`
                Case number
            *comp*: {``None``} | :class:`str`
                Name of component for which to get BCs
        :Outputs:
            *M2*: :class:`float`
                Exit Mach number
        :Versions:
            * 2016-04-13 ``@ddalle``: Version 1.0
        """
        # Get exit Mach number
        M2 = self.x.GetSurfCT_ExitMach(i, key, comp=comp)
        # Check if we need to use area ratio
        if M2 is None:
            # Get input Mach number
            M1 = self.x.GetSurfCT_Mach(i, key, comp=comp)
            # Get area ratio
            AR = self.x.GetSurfCT_AreaRatio(i, key, comp=comp)
            # Ratio of specific heats
            gam = self.x.GetSurfCT_Gamma(i, key, comp=comp)
            # Calculate exit Mach number
            M2 = convert.ExitMachFromAreaRatio(AR, M1, gam)
        # Output
        return M2

    # Reference area
    def GetSurfCT_RefArea(self, key, i):
        """Get reference area for surface *CT* trajectory key

        This references the ``"RefArea"`` parameter of the definition for the
        run matrix variable *key*.  The user should set this parameter to
        ``1.0`` if thrust inputs are given as dimensional values.

        If this is ``None``, it returns the global reference area; if it is a
        string the reference area comes from the reference area for that
        component using ``cntl.opts.get_RefArea(comp)``.

        :Call:
            >>> Aref = cntl.GetSurfCT_RefArea(key, i)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *key*: :class:`str`
                Name of trajectory key to check
            *i*: :class:`int`
                Case number
        :Outputs:
            *Aref*: :class:`float`
                Reference area for normalizing thrust coefficients
        :Versions:
            * 2016-04-13 ``@ddalle``: Version 1.0
        """
        # Get *Aref* option
        Aref = self.x.GetSurfCT_RefArea(i, key)
        # Type
        t = type(Aref).__name__
        # Check type
        if Aref is None:
            # Use the default
            return self.opts.get_RefArea()
        elif t in ['str', 'unicode']:
            # Use the input as a component ID name
            return self.opts.get_RefArea(Aref)
        else:
            # Assume it's already given as the correct type
            return Aref

   # >

   # =================
   # DataBook Updaters
   # =================
   # <
    # Function to collect statistics
    @run_rootdir
    def UpdateFM(self, **kw):
        """Collect force and moment data

        :Call:
            >>> cntl.UpdateFM(cons=[], **kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *fm*, *aero*: {``None``} | :class:`str`
                Wildcard to subset list of FM components
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints like ``'Mach<=0.5'``
        :Outputs:
            *d*: :class:`dict` (:class:`numpy.ndarray` (:class:`float`))
                Dictionary of mean, min, max, std for each coefficient
        :Versions:
            * 2014-12-12 ``@ddalle``: Version 1.0
            * 2014-12-22 ``@ddalle``: Completely rewrote with DataBook class
            * 2017-04-25 ``@ddalle``: Added wild cards
            * 2018-10-19 ``@ddalle``: Renamed :func:`Aero` to :func:`UpdateFM`
        """
        # Get component option
        comp = kw.get("fm", kw.get("aero"))
        # Get full list of components
        comp = self.opts.get_DataBookByGlob(["FM","Force","Moment"], comp)
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Check if we are deleting or adding.
        if kw.get('delete', False):
            # Read the existing data book.
            self.ReadDataBook(comp=comp)
            # Delete cases.
            self.DataBook.DeleteCases(I, comp=comp)
        else:
            # Read an empty data book
            self.ReadDataBook(comp=[])
            # Read the results and update as necessary.
            self.DataBook.UpdateDataBook(I, comp=comp)

    # Update line loads
    @run_rootdir
    def UpdateLineLoad(self, **kw):
        """Update one or more line load data books

        :Call:
            >>> cntl.UpdateLineLoad(ll=None, **kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *ll*: :class:`str`
                Optional name of line load component to update
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints like ``'Mach<=0.5'``
            *pbs*: ``True`` | {``False``}
                Whether or not to calculate line loads with PBS scripts
        :Versions:
            * 2016-06-07 ``@ddalle``: Version 1.0
            * 2016-12-21 ``@ddalle``: Added *pbs* flag, may be temporary
            * 2017-04-25 ``@ddalle``: Removed *pbs*, added ``--delete``
        """
        # Get component option
        comp = kw.get("ll")
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Read the data book handle
        self.ReadDataBook(comp=[])
        self.ReadConfig()
        # Check if we are deleting or adding.
        if kw.get('delete', False):
            # Delete cases.
            self.DataBook.DeleteLineLoad(I, comp=comp)
        else:
            # Read the results and update as necessary.
            self.DataBook.UpdateLineLoad(I, comp=comp, conf=self.config)

    # Update TriqFM data book
    @run_rootdir
    def UpdateTriqFM(self, **kw):
        """Update one or more TriqFM data books

        :Call:
            >>> cntl.UpdateTriqFM(comp=None, **kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Control class
            *comp*: {``None``} | :class:`str`
                Name of TriqFM component
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2017-03-29 ``@ddalle``: Version 1.0
        """
        # Get component option
        comp = kw.get("triqfm")
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Read the data book handle
        self.ReadDataBook(comp=[])
        # Check if we are deleting or adding.
        if kw.get('delete', False):
            # Delete cases.
            self.DataBook.DeleteTriqFM(I, comp=comp)
        else:
            # Read the results and update as necessary.
            self.DataBook.UpdateTriqFM(I, comp=comp)

    # Update TriqPointGroup data book
    @run_rootdir
    def UpdateTriqPoint(self, **kw):
        """Update one or more TriqPoint extracted point sensor data books

        :Call:
            >>> cntl.UpdateTriqPoint(comp=None, **kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Control class
            *comp*: {``None``} | :class:`str`
                Name of TriqFM component
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2017-03-29 ``@ddalle``: Version 1.0
        """
        # Get component option
        comp = kw.get("pt")
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Read the data book handle
        self.ReadDataBook(comp=[])
        # Check if we are deleting or adding.
        if kw.get('delete', False):
            # Delete cases.
            self.DataBook.DeleteTriqPoint(I, comp=comp)
        else:
            # Read the results and update as necessary.
            self.DataBook.UpdateTriqPoint(I, comp=comp)
   # >

   # =================
   # DataBook Checkers
   # =================
   # <
    # Function to check FM component status
    def CheckFM(self, **kw):
        """Display missing force & moment components

        :Call:
            >>> cntl.CheckFM(**kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *fm*, *aero*: {``None``} | :class:`str`
                Wildcard to subset list of FM components
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2018-10-19 ``@ddalle``: Version 1.0
        """
        # Get component option
        comps = kw.get("fm", kw.get("aero",
            kw.get("checkFM", kw.get("check"))))
        # Get full list of components
        comps = self.opts.get_DataBookByGlob(["FM","Force","Moment"], comps)
        # Exit if no components
        if len(comps) == 0:
            return
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Check for a user key
        ku = self.x.GetKeysByType("user")
        # Check for a find
        if ku:
            # One key, please
            ku = ku[0]
        else:
            # No user key
            ku = None
        # Read the existing data book
        self.ReadDataBook(comp=comps)
        # Loop through the components
        for comp in comps:
            # Restrict the trajectory to cases in the databook
            self.DataBook[comp].UpdateRunMatrix()
        # Longest component name
        maxcomp = max(map(len, comps))
        # Format to include user and format to display iteration number
        fmtc = "    %%-%is: " % maxcomp
        fmti = "%%%ii" % int(np.ceil(np.log10(self.x.nCase)))
        # Loop through cases
        for i in I:
            # Skip marked errors
            if self.x.ERROR[i]:
                continue
            # Skip if we have a blocked user
            if ku:
                # Get the user
                ui = self.x[ku][i]
                # Simplify the value
                ui = ui.lstrip('@').lower()
                # Check if it's blocked
                if ui == "blocked": continue
            else:
                # Empty user
                ui = None
            # Get the last iteration for this case
            nLast = self.GetLastIter(i)
            # Initialize text
            txt = ""
            # Loop through components
            for comp in comps:
                # Get interface to component
                DBc = self.DataBook[comp]
                # See if it's missing
                j = DBc.x.FindMatch(self.x, i, **kw)
                # Check for missing case
                if j is None:
                    # Missing case
                    txt += (fmtc % comp)
                    txt += "missing\n"
                    continue
                # Otherwise, check iteration
                try:
                    # Get the recorded iteration number
                    nIter = DBc["nIter"][j]
                except KeyError:
                    # No iteration number found
                    nIter = nLast
                # Check for out-of date iteration
                if nIter < nLast:
                    # Out-of-date case
                    txt += (fmtc % comp)
                    txt += "out-of-date (%i --> %i)\n" % (nIter, nLast)
            # If we have any text, print a header
            if txt:
                # Folder name
                frun = self.x.GetFullFolderNames(i)
                # Print header
                if ku:
                    # Include user
                    print("Case %s: %s (%s)" % (fmti % i, frun, ui))
                else:
                    # No user
                    print("Case %s: %s" % (fmti % i, frun))
                # Display the text
                print(txt)
        # Loop back through the databook components
        for comp in comps:
            # Get component handle
            DBc = self.DataBook[comp]
            # Initialize text
            txt = ""
            # Loop through database entries
            for j in range(DBc.x.nCase):
                # Check for a find in master matrix
                i = self.x.FindMatch(DBc.x, j, **kw)
                # Check for a match
                if i is None:
                    # This case is not in the run matrix
                    txt += ("    Extra case: %s\n"
                        % DBc.x.GetFullFolderNames(j))
                    continue
                # Check for a user filter
                if ku:
                    # Get the user value
                    uj = DBc[ku][j]
                    # Strip it
                    uj = uj.lstrip('@').lower()
                    # Check if it's blocked
                    if uj == "blocked":
                        # Blocked case
                        txt += ("    Blocked case: %s\n"
                            % DBc.x.GetFullFolderNames(j))
            # If there is text, display the info
            if txt:
                # Header
                print("Checking component '%s'" % comp)
                print(txt[:-1])

    # Function to check LL component status
    def CheckLL(self, **kw):
        """Display missing line load components

        :Call:
            >>> cntl.CheckLL(**kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *fm*, *aero*: {``None``} | :class:`str`
                Wildcard to subset list of FM components
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2018-10-19 ``@ddalle``: Version 1.0
        """
        # Get component option
        comps = kw.get("ll", kw.get("checkLL", kw.get("check")))
        # Get full list of components
        comps = self.opts.get_DataBookByGlob("LineLoad", comps)
        # Exit if no components
        if len(comps) == 0:
            return
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Check for a user key
        ku = self.x.GetKeysByType("user")
        # Check for a find
        if ku:
            # One key, please
            ku = ku[0]
        else:
            # No user key
            ku = None
        # Read the existing data book
        self.ReadDataBook(comp=[])
        # Loop through the components
        for comp in comps:
            # Read the line load component
            self.DataBook.ReadLineLoad(comp)
            # Restrict the trajectory to cases in the databook
            self.DataBook.LineLoads[comp].UpdateRunMatrix()
        # Longest component name
        maxcomp = max(map(len, comps))
        # Format to include user and format to display iteration number
        fmtc = "    %%-%is: " % maxcomp
        fmti = "%%%ii" % int(np.ceil(np.log10(self.x.nCase)))
        # Loop through cases
        for i in I:
            # Skip if we have a blocked user
            if ku:
                # Get the user
                ui = self.x[ku][i]
                # Simplify the value
                ui = ui.lstrip('@').lower()
                # Check if it's blocked
                if ui == "blocked": continue
            else:
                # Empty user
                ui = None
            # Get the last iteration for this case
            nLast = self.GetLastIter(i)
            # Initialize text
            txt = ""
            # Loop through components
            for comp in comps:
                # Get interface to component
                DBc = self.DataBook.LineLoads[comp]
                # See if it's missing
                j = DBc.x.FindMatch(self.x, i, **kw)
                # Check for missing case
                if j is None:
                    # Missing case
                    txt += (fmtc % comp)
                    txt += "missing\n"
                    continue
                # Otherwise, check iteration
                try:
                    # Get the recorded iteration number
                    nIter = DBc["nIter"][j]
                except KeyError:
                    # No iteration number found
                    nIter = nLast
                # Check for out-of date iteration
                if nIter < nLast:
                    # Out-of-date case
                    txt += (fmtc % comp)
                    txt += "out-of-date (%i --> %i)\n" % (nIter, nLast)
            # If we have any text, print a header
            if txt:
                # Folder name
                frun = self.x.GetFullFolderNames(i)
                # Print header
                if ku:
                    # Include user
                    print("Case %s: %s (%s)" % (fmti % i, frun, ui))
                else:
                    # No user
                    print("Case %s: %s" % (fmti % i, frun))
                # Display the text
                print(txt)
        # Loop back through the databook components
        for comp in comps:
            # Get component handle
            DBc = self.DataBook.LineLoads[comp]
            # Initialize text
            txt = ""
            # Loop through database entries
            for j in range(DBc.x.nCase):
                # Check for a find in master matrix
                i = self.x.FindMatch(DBc.x, j, **kw)
                # Check for a match
                if i is None:
                    # This case is not in the run matrix
                    txt += ("    Extra case: %s\n"
                        % DBc.x.GetFullFolderNames(j))
                    continue
                # Check for a user filter
                if ku:
                    # Get the user value
                    uj = DBc[ku][j]
                    # Strip it
                    uj = uj.lstrip('@').lower()
                    # Check if it's blocked
                    if uj == "blocked":
                        # Blocked case
                        txt += ("    Blocked case: %s\n"
                            % DBc.x.GetFullFolderNames(j))
            # If there is text, display the info
            if txt:
                # Header
                print("Checking component '%s'" % comp)
                print(txt[:-1])

    # Function to check TriqFM component status
    def CheckTriqFM(self, **kw):
        """Display missing TriqFM components

        :Call:
            >>> cntl.CheckTriqFM(**kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *fm*, *aero*: {``None``} | :class:`str`
                Wildcard to subset list of FM components
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2018-10-19 ``@ddalle``: Version 1.0
        """
        # Get component option
        comps = kw.get("triqfm", kw.get("checkTriqFM", kw.get("check")))
        # Get full list of components
        comps = self.opts.get_DataBookByGlob("TriqFM", comps)
        # Exit if no components
        if len(comps) == 0:
            return
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Check for a user key
        ku = self.x.GetKeysByType("user")
        # Check for a find
        if ku:
            # One key, please
            ku = ku[0]
        else:
            # No user key
            ku = None
        # Read the existing data book
        self.ReadDataBook(comp=[])
        # Loop through the components
        for comp in comps:
            # Read the line load component
            self.DataBook.ReadTriqFM(comp)
            # Restrict the trajectory to cases in the databook
            self.DataBook.TriqFM[comp][None].UpdateRunMatrix()
        # Longest component name
        maxcomp = max(map(len, comps))
        # Format to include user and format to display iteration number
        fmtc = "    %%-%is: " % maxcomp
        fmti = "%%%ii" % int(np.ceil(np.log10(self.x.nCase)))
        # Loop through cases
        for i in I:
            # Skip if we have a blocked user
            if ku:
                # Get the user
                ui = self.x[ku][i]
                # Simplify the value
                ui = ui.lstrip('@').lower()
                # Check if it's blocked
                if ui == "blocked": continue
            else:
                # Empty user
                ui = None
            # Get the last iteration for this case
            nLast = self.GetLastIter(i)
            # Initialize text
            txt = ""
            # Loop through components
            for comp in comps:
                # Get interface to component
                DBc = self.DataBook.TriqFM[comp][None]
                # See if it's missing
                j = DBc.x.FindMatch(self.x, i, **kw)
                # Check for missing case
                if j is None:
                    # Missing case
                    txt += (fmtc % comp)
                    txt += "missing\n"
                    continue
                # Otherwise, check iteration
                try:
                    # Get the recorded iteration number
                    nIter = DBc["nIter"][j]
                except KeyError:
                    # No iteration number found
                    nIter = nLast
                # Check for out-of date iteration
                if nIter < nLast:
                    # Out-of-date case
                    txt += (fmtc % comp)
                    txt += "out-of-date (%i --> %i)\n" % (nIter, nLast)
            # If we have any text, print a header
            if txt:
                # Folder name
                frun = self.x.GetFullFolderNames(i)
                # Print header
                if ku:
                    # Include user
                    print("Case %s: %s (%s)" % (fmti % i, frun, ui))
                else:
                    # No user
                    print("Case %s: %s" % (fmti % i, frun))
                # Display the text
                print(txt)
        # Loop back through the databook components
        for comp in comps:
            # Get component handle
            DBc = self.DataBook.TriqFM[comp][None]
            # Initialize text
            txt = ""
            # Loop through database entries
            for j in range(DBc.x.nCase):
                # Check for a find in master matrix
                i = self.x.FindMatch(DBc.x, j, **kw)
                # Check for a match
                if i is None:
                    # This case is not in the run matrix
                    txt += ("    Extra case: %s\n"
                        % DBc.x.GetFullFolderNames(j))
                    continue
                # Check for a user filter
                if ku:
                    # Get the user value
                    uj = DBc[ku][j]
                    # Strip it
                    uj = uj.lstrip('@').lower()
                    # Check if it's blocked
                    if uj == "blocked":
                        # Blocked case
                        txt += ("    Blocked case: %s\n"
                            % DBc.x.GetFullFolderNames(j))
            # If there is text, display the info
            if txt:
                # Header
                print("Checking component '%s'" % comp)
                print(txt[:-1])

    # Function to check TriqFM component status
    def CheckTriqPoint(self, **kw):
        """Display missing TriqPoint components

        :Call:
            >>> cntl.CheckTriqPoint(**kw)
        :Inputs:
            *cntl*: :class:`cape.cntl.Cntl`
                Instance of control class containing relevant parameters
            *fm*, *aero*: {``None``} | :class:`str`
                Wildcard to subset list of FM components
            *I*: :class:`list` (:class:`int`)
                List of indices
            *cons*: :class:`list` (:class:`str`)
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2018-10-19 ``@ddalle``: Version 1.0
        """
        # Get component option
        comps = kw.get("pt", kw.get("checkPt", kw.get("check")))
        # Get full list of components
        comps = self.opts.get_DataBookByGlob("TriqPoint", comps)
        # Exit if no components
        if len(comps) == 0:
            return
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Check for a user key
        ku = self.x.GetKeysByType("user")
        # Check for a find
        if ku:
            # One key, please
            ku = ku[0]
        else:
            # No user key
            ku = None
        # Read the existing data book
        self.ReadDataBook(comp=[])
        # Component list for text
        complist = []
        # Loop through the components
        for comp in comps:
            # Read the line load component
            self.DataBook.ReadTriqPoint(comp)
            # Get point group
            DBG = self.DataBook.TriqPoint[comp]
            # Loop through points
            for pt in DBG.pts:
                # Restrict the trajectory to cases in the databook
                DBG[pt].UpdateRunMatrix()
                # Add to the list
                complist.append("%s/%s" % (comp, pt))

        # Longest component name (plus room for the '/' char)
        maxcomp = max(map(len, complist)) + 1
        # Format to include user and format to display iteration number
        fmtc = "    %%-%is: " % maxcomp
        fmti = "%%%ii" % int(np.ceil(np.log10(self.x.nCase)))
        # Loop through cases
        for i in I:
            # Skip if we have a blocked user
            if ku:
                # Get the user
                ui = self.x[ku][i]
                # Simplify the value
                ui = ui.lstrip('@').lower()
                # Check if it's blocked
                if ui == "blocked": continue
            else:
                # Empty user
                ui = None
            # Get the last iteration for this case
            nLast = self.GetLastIter(i)
            # Initialize text
            txt = ""
            # Loop through components
            for comp in comps:
                # Get point group
                DBG = self.DataBook.TriqPoint[comp]
                # Loop through points
                for pt in DBG.pts:
                    # Get interface to component
                    DBc = DBG[pt]
                    # See if it's missing
                    j = DBc.x.FindMatch(self.x, i, **kw)
                    # Check for missing case
                    if j is None:
                        # Missing case
                        txt += (fmtc % ("%s/%s" % (comp, pt)))
                        txt += "missing\n"
                        continue
                    # Otherwise, check iteration
                    try:
                        # Get the recorded iteration number
                        nIter = DBc["nIter"][j]
                    except KeyError:
                        # No iteration number found
                        nIter = nLast
                    # Check for out-of date iteration
                    if nIter < nLast:
                        # Out-of-date case
                        txt += (fmt % comp)
                        txt += "out-of-date (%i --> %i)\n" % (nIter, nLast)
            # If we have any text, print a header
            if txt:
                # Folder name
                frun = self.x.GetFullFolderNames(i)
                # Print header
                if ku:
                    # Include user
                    print("Case %s: %s (%s)" % (fmti % i, frun, ui))
                else:
                    # No user
                    print("Case %s: %s" % (fmti % i, frun))
                # Display the text
                print(txt)
        # Loop back through the databook components
        for comp in comps:
            # Get group
            DBG = self.DataBook.TriqPoint[comp]
            # Loop through points
            for pt in DBG.pts:
                # Get component handle
                DBc = DBG[pt]
                # Initialize text
                txt = ""
                # Loop through database entries
                for j in range(DBc.x.nCase):
                    # Check for a find in master matrix
                    i = self.x.FindMatch(DBc.x, j, **kw)
                    # Check for a match
                    if i is None:
                        # This case is not in the run matrix
                        txt += ("    Extra case: %s\n"
                            % DBc.x.GetFullFolderNames(j))
                        continue
                    # Check for a user filter
                    if ku:
                        # Get the user value
                        uj = DBc[ku][j]
                        # Strip it
                        uj = uj.lstrip('@').lower()
                        # Check if it's blocked
                        if uj == "blocked":
                            # Blocked case
                            txt += ("    Blocked case: %s\n"
                                % DBc.x.GetFullFolderNames(j))
                # If there is text, display the info
                if txt:
                    # Header
                    print("Checking point sensor '%s/%s'" % (comp, pt))
                    print(txt[:-1])
   # >
# class Cntl

