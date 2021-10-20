#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.pykes.cntl`: Kestrel control module
===============================================

This module provides tools to quickly setup basic or complex Kestrel run
matrices and serve as an executive for pre-processing, running,
post-processing, and managing the solutions. A collection of cases
combined into a run matrix can be loaded using the following commands.

    .. code-block:: pycon

        >>> import cape.pykes.cntl
        >>> cntl = cape.pykes.cntl.Cntl("pyKes.json")
        >>> cntl
        <cape.pyfun.Cntl(nCase=892)>
        >>> cntl.x.GetFullFolderNames(0)
        'poweroff/m1.5a0.0b0.0'


An instance of this :class:`cape.pyfun.cntl.Cntl` class has many
attributes, which include the run matrix (``cntl.x``), the options
interface (``cntl.opts``), and optionally the data book
(``cntl.DataBook``), the appropriate input files (such as
``cntl.``), and possibly others.

    ====================   =============================================
    Attribute              Class
    ====================   =============================================
    *cntl.x*              :class:`cape.runmatrix.RunMatrix`
    *cntl.opts*           :class:`cape.pykes.options.Options`
    *cntl.DataBook*       :class:`cape.pykes.dataBook.DataBook`
    *cntl.JobXML*         :class:`cape.pykes.namelist.Namelist`
    ====================   =============================================

:class:`cape.cntl.Cntl` class, so any methods available to the CAPE
class are also available here.

"""

# Standard library
import os
import shutil

# Third-party modules
import numpy as np

# Local imports
from . import options
#from . import manage
#from . import case
#from . import dataBook
from .jobxml   import JobXML
from .. import cntl as ccntl
from ..cfdx import report
from ..runmatrix import RunMatrix
from ..util import RangeString

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyKesFolder = os.path.split(_fname)[0]


# Class to read input files
class Cntl(ccntl.Cntl):
    r"""Class for handling global options and setup for Kestrel

    This class is intended to handle all settings used to describe a
    group of Kestrel cases.

    The settings are read from a JSON file.

    Defaults are read from the file
    ``options/pyKes.default.json``.

    :Call:
        >>> cntl = Cntl(fname="pyKes.json")
    :Inputs:
        *fname*: :class:`str`
            Name of pyKes input file
    :Outputs:
        *cntl*: :class:`cape.pykes.cntl.Cntl`
            Instance of the pyKes control class
    :Data members:
        *cntl.opts*: :class:`dict`
            Dictionary of options for this case (directly from *fname*)
        *cntl.x*: :class:`pyFun.runmatrix.RunMatrix`
            Values and definitions for variables in the run matrix
        *cntl.RootDir*: :class:`str`
            Absolute path to the root directory
    :Versions:
        * 2015-10-16 ``@ddalle``: Started
    """
  # ======
  # Config
  # ======
  # <
    # Initialization method
    def __init__(self, fname="pyKes.json"):
        r"""Initialization method

        :Versions:
            * 2015-10-16 ``@ddalle``: Version 1.0
        """
        # Force default
        if fname is None:
            fname = "pyKes.json"
        # Check if file exists
        if not os.path.isfile(fname):
            # Raise error but suppress traceback
            os.sys.tracebacklimit = 0
            raise ValueError("No pyKes control file '%s' found" % fname)

        # Read settings
        self.opts = options.Options(fname=fname)

        # Save the current directory as the root
        self.RootDir = os.getcwd()

        # Import modules
        self.ImportModules()

        # Process the trajectory.
        self.x = RunMatrix(**self.opts['RunMatrix'])

        # Job list
        self.jobs = {}

        # Read the namelist(s)
        self.ReadJobXML()

        # Set umask
        os.umask(self.opts.get_umask())

        # Run any initialization functions
        self.InitFunction()

    # Output representation
    def __repr__(self):
        r"""Output representation for the class."""
        # Display basic information from all three areas.
        return "<pyKes.Cntl(nCase=%i)>" % (
            self.x.nCase)
  # >

  # =======================
  # Command-Line Interface
  # =======================
  # <
    # Baseline function
    def cli(self, *a, **kw):
        r"""Command-line interface

        :Call:
            >>> cntl.cli(*a, **kw)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
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
        # Preemptive command
        if kw.get('check'):
            # Check all
            print("---- Checking FM DataBook components ----")
            self.CheckFM(**kw)
            print("---- Checking LineLoad DataBook components ----")
            self.CheckLL(**kw)
            print("---- Checking TriqFM DataBook components ----")
            self.CheckTriqFM(**kw)
            print("---- Checking TriqPoint DataBook components ----")
            self.CheckTriqPoint(**kw)
            # Quit
            return
        elif kw.get('data', kw.get('db')):
            # Update all
            print("---- Updating FM DataBook components ----")
            self.UpdateFM(**kw)
            print("---- Updating LineLoad DataBook components ----")
            self.UpdateLineLoad(**kw)
            print("---- Updating TriqFM DataBook components ----")
            self.UpdateTriqFM(**kw)
            print("---- Updating TriqPoint DataBook components ----")
            self.UpdateTriqPoint(**kw)
            # Output
            return
        # Call the common interface
        cmd = self.cli_cape(*a, **kw)
        # Test for a command
        if cmd is not None:
            return
        # Otherwise fall back to code-specific commands
        # Submit the jobs
        self.SubmitJobs(**kw)

  # >

  # ========
  # Readers
  # ========
  # <
    # Function to read the databook.
    def ReadDataBook(self, comp=None):
        r"""Read the current data book

        :Call:
            >>> cntl.ReadDataBook()
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class containing relevant parameters
        :Versions:
            * 2016-09-15 ``@ddalle``: Version 1.0
        """
        # Test for an existing data book.
        try:
            self.DataBook
            return
        except AttributeError:
            pass
        # Go to root directory.
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        # Ensure list of components
        if comp is not None:
            comp = list(np.array(comp).flatten())
        # Read the data book.
        self.DataBook = dataBook.DataBook(self.x, self.opts, comp=comp)
        # Save project name
        self.DataBook.proj = self.GetProjectRootName(None)
        # Return to original folder.
        os.chdir(fpwd)

    # Function to read a report
    def ReadReport(self, rep):
        r"""Read a report interface

        :Call:
            >>> R = cntl.ReadReport(rep)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class containing relevant parameters
            *rep*: :class:`str`
                Name of report
        :Outputs:
            *R*: :class:`pyFun.report.Report`
                Report interface
        :Versions:
            * 2018-10-19 ``@ddalle``: Version 1.0
        """
        # Read the report
        R = report.Report(self, rep)
        # Output
        return R
  # >

  # ========
  # XML job
  # ========
  # <
    # Read the namelist
    def ReadJobXML(self, j=0, q=True):
        r"""Read the :file:`fun3d.nml` file

        :Call:
            >>> cntl.ReadNamelist(j=0, q=True)
        :Inputs:
            *cntl*: :class:`cape.pykes.cntl.Cntl`
                Run matrix control interface
            *j*: {``0``} | :class:`int`
                Phase number
            *q*: {``True``} | ``False
                Option read to *JobXML*, else *JobXML0*
        :Versions:
            * 2021-10-18 ``@ddalle``: Version 1.0
        """
        # Namelist file
        fxml = self.opts.get_JobXML(j)
        # Check for empty value
        if fxml is None:
            return
        # Check for absolute path
        if not os.path.isabs(fxml):
            # Use path relative to JSON root
            fxml = os.path.join(self.RootDir, fxml)
        # Check again
        if not os.path.isabs(fxml):
            return
        # Read the file
        xml = JobXML(fxml)
        # Save it.
        if q:
            # Read to main slot for modification
            self.JobXML = xml
        else:
            # Template for reading original parameters
            self.JobXML0 = xml

    # Get namelist var
    def GetNamelistVar(self, sec, key, j=0):
        r"""Get a namelist variable's value

        The JSON file overrides the value from the namelist file

        :Call:
            >>> val = cntl.GetNamelistVar(sec, key, j=0)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of global pyFun settings object
            *sec*: :class:`str`
                Name of namelist section
            *key*: :class:`str`
                Variable to read
            *j*: :class:`int`
                Run sequence index
        :Outputs:
            *val*::class:`int`|:class:`float`|:class:`str`|:class:`list`
                Value
        :Versions:
            * 2015-10-19 ``@ddalle``: Version 1.0
        """
        # Get the namelist value.
        nval = self.Namelist.GetVar(sec, key)
        # Check for options value.
        if nval is None:
            # No namelist file value
            return self.opts.get_namelist_var(sec, key, j)
        elif 'Fun3D' not in self.opts:
            # No namelist in options
            return nval
        elif sec not in self.opts['Fun3D']:
            # No corresponding options section
            return nval
        elif key not in self.opts['Fun3D'][sec]:
            # Value not specified in the options namelist
            return nval
        else:
            # Default to the options
            return self.opts.get_namelist_var(sec, key, j)

    # Get the project rootname
    def GetProjectRootName(self, j=0):
        r"""Get the project root name

        The JSON file overrides the value from the namelist file if
        appropriate

        :Call:
            >>> name = cntl.GetProjectName(j=0)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of global pyFun settings object
            *j*: :class:`int`
                Phase number
        :Outputs:
            *name*: :class:`str`
                Project root name
        :Versions:
            * 2015-10-18 ``@ddalle``: Version 1.0
        """
        # Read the namelist.
        self.ReadNamelist(j, False)
        # Get the namelist value.
        nname = self.Namelist0.GetVar('project', 'project_rootname')
        # Get the options value.
        oname = self.opts.get_project_rootname(j)
        # Check for options value
        if nname is None:
            # Use the options value.
            name = oname
        elif 'Fun3D' not in self.opts:
            # No namelist options
            name = nname
        elif 'project' not in self.opts['Fun3D']:
            # No project options
            name = nname
        elif 'project_rootname' not in self.opts['Fun3D']['project']:
            # No rootname
            name = nname
        else:
            # Use the options value.
            name = oname
        # Check for adaptation number
        k = self.opts.get_AdaptationNumber(j)
        # Assemble project name
        if k is None:
            # No adaptation numbers
            return name
        else:
            # Append the adaptation number
            return '%s%02i' % (name, k)

    # Get the grid format
    def GetGridFormat(self, j=0):
        r"""Get the grid format

        The JSON file overrides the value from the namelist file

        :Call:
            >>> fmt = cntl.GetGridFormat(j=0)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of global pyFun settings object
            *j*: :class:`int`
                Run sequence index
        :Outputs:
            *fmt*: :class:`str`
                Project root name
        :Versions:
            * 2015-10-18 ``@ddalle``: Version 1.0
        """
        return self.GetNamelistVar('raw_grid', 'grid_format', j)

  # >

