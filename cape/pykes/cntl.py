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
from . import case
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

  # ================
  # Primary Setup
  # ================
  # <
    # Prepare a case
    @ccntl.run_rootdir
    def PrepareCase(self, i):
        r"""Prepare a case for running if necessary

        :Call:
            >>> cntl.PrepareCase(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE main control instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2021-10-26 ``@ddalle``: Version 1.0
        """
        # Check case
        n = self.CheckCase(i)
        # Quit if already prepared
        if f is not None:
            return
        # Run any case functions
        self.CaseFunction(i)
        # Prepare mesh
        self.PrepareMesh(i)
        # Get the run folder name
        frun = self.x.GetFullFolderNames(i)

    # Prepare the mesh for case *i*
    @ccntl.run_rootdir
    def PrepareMesh(self, i):
        r"""Prepare the mesh for case *i* if necessary
        
        :Call:
            >>> cntl.PrepareMesh(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE main control instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2021-10-26 ``@ddalle``: Version 1.0
        """
        # Get the case name
        frun = self.x.GetFullFolderNames(i)
        # Get the name of the group
        fgrp = self.x.GetGroupFolderNames(i)
        # Create folders
        if not os.path.isdir(fgrp):
            self.mkdir(fgrp)
        if not os.path.isdir(frun):
            self.mkdir(frun)
        # Status update
        print("  Case name: '%s' (index %i)" % (frun, i))
        # Initialize copied/linked list
        copiedfiles = set()
        # Linked files
        linkfiles = self.opts.get_MeshLinkFiles()
        # Loop through phases
        for j in self.opts.get_PhaseSequence():
            # Loop through candidates
            for fname in self.GetMeshFileNames(j):
                # Absolute source path
                fabs = self.abspath(fname)
                # Name in case folder
                fbase = os.path.basename(fabs)
                # Absolute destination folder
                fdest = os.path.join(self.RootDir, frun, fbase)
                # Check if already copied
                if fbase in copiedfiles:
                    continue
                # Check if file exists
                if not os.path.isfile(fabs):
                    continue
                # Check for file in folder
                if os.path.isfile(fdest):
                    # Remove it
                    os.remove(fdest)
                # Check copy vs link
                if fname in linkfiles:
                    # Link it
                    os.link(fabs, fdest)
                else:
                    # Copy it
                    shutil.copy(fabs, fdest)
                # Add to list already copied
                copiedfiles.add(fbase)
        # Enter the case folder
        os.chdir(frun)
        
  # >

  # ===============
  # Case Interface
  # ===============
  # <
    # Check if mesh is prepared
    def CheckMesh(self, i):
        r"""Check if the mesh for case *i* is prepared

        :Call:
            >>> q = cntl.CheckMesh(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                Main CAPE control instance
            *i*: :class:`int`
                Case index
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not mesh for case *i* is prepared
        :Versions:
            * 2021-10-26 ``@ddalle``: Version 1.0
        """
        # Name for this folder
        frun = self.x.GetFullFolderNames(i)
        # List of files checked
        checkedfiles = set()
        # Get *rc* to check phases
        rc = self.ReadCaseJSON(i)
        # If none yet, use local settings
        if rc is None:
            rc = self.opts
        # Loop through phases
        for j in rc.get_PhaseSequence():
            # Get file names
            for fname in self.GetMeshFileNames(j):
                # Get base name
                fbase = os.path.basename(fname)
                # Check if already checked
                if fbase in checkedfiles:
                    continue
                # Absolute path
                fabs = os.path.join(self.RootDir, frun, fbase)
                # Check if it exists
                if not os.path.isfile(fabs):
                    return False
        # All files found
        return True

    # Read run control options from case JSON file
    @ccntl.run_rootdir
    def ReadCaseJSON(self, i):
        r"""Read ``case.json`` file from case *i* if possible

        :Call:
            >>> rc = cntl.ReadCaseJSON(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of cape.pyover control class
            *i*: :class:`int`
                Run index
        :Outputs:
            *rc*: ``None`` | :class:`RunControl`
                Run control interface read from ``case.json`` file
        :Versions:
            * 2016-12-12 ``@ddalle``: Version 1.0
        """
        # Get the case name
        frun = self.x.GetFullFolderNames(i)
        # Check if it exists
        if not os.path.isdir(frun):
            return
        # Go to the folder
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

    # Find files referenced in XML
    def FindXMLPaths(self, j=0):
        r"""Find all *Path* and *File* elements

        :Call:
            >>> elems = cntl.FindXMLPaths(j=0)
        :Inputs:
            *cntl*: :class:`cape.pykes.cntl.Cntl`
                Run matrix control interface
            *j*: {``0``} | :class:`int`
                Phase number
        :Outputs:
            *elems*: :class:`list`\ [:class:`Element`]
                List of XML elements
        :Versions:
            * 2021-10-25 ``@ddalle``: Version 1.0
        """
        # Read XML file
        self.ReadJobXML(j=j, q=False)
        # Find all *Path* and *File* elements
        elems1 = self.JobXML0.findall_iter("Path")
        elems2 = self.JobXML0.findall_iter("File")
        # Return combination
        return elems1 + elems2
  # >

  # ================
  # File/Mesh Copy
  # ================
  # <
    # Find all mesh files
    def GetMeshFileNames(self, j=0):
        r"""Get list of copy/link files from both JSON and XML

        :Call:
            >>> meshfiles = cntl.GetMeshFiles(j=0)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE main control instance
            *j*: {``0``} | :class:`int`
                Phase number
        :Outputs:
            *meshfiles*: :class:`list`\ [:class:`str`]
                List of files to copy/link
        :Versions:
            * 2021-10-25 ``@ddalle``: Version 1.0
        """
        # Get file names from mesh
        meshfiles = self.opts.get_MeshFiles()
        # Get XML candidates
        elems = self.FindXMLPaths(j)
        # Loop through elements
        for elem in elems:
            # Get candidate file name
            fname = elem.text
            # Check if it exists
            if fname is None:
                # Empty element (?)
                continue
            if os.path.isabs(fname):
                # Already absolute
                fabs = fname
            else:
                # Absolutize from *RootDir*
                fabs = os.path.join(self.RootDir, fname)
            # Check if file exists
            if os.path.isfile(fabs):
                meshfiles.append(fname)
        # Output
        return meshfiles
  # >
