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


An instance of this :class:`cape.pykes.cntl.Cntl` class has many
attributes, which include the run matrix (``cntl.x``), the options
interface (``cntl.opts``), and optionally the data book
(``cntl.DataBook``), the appropriate input files (such as
``cntl.``), and possibly others.

    ====================   ============================================
    Attribute              Class
    ====================   ============================================
    *cntl.x*               :class:`cape.runmatrix.RunMatrix`
    *cntl.opts*            :class:`cape.pykes.options.Options`
    *cntl.DataBook*        :class:`cape.pykes.databook.DataBook`
    *cntl.JobXML*          :class:`cape.pykes.jobxml.JobXML`
    ====================   ============================================

:class:`cape.cfdx.cntl.Cntl` class, so any methods available to the CAPE
class are also available here.

"""

# Standard library
import os
import shutil

# Third-party modules


# Local imports
from . import casecntl
from . import databook
from . import options
from . import report
from .jobxml import JobXML
from ..cfdx import cntl as ccntl


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
  # =================
  # Class Attributes
  # =================
  # <
    # Names
    _solver = "kestrel"
    # Case module
    _databook_mod = databook
    _report_cls = report.Report
    # Options class
    _case_cls = casecntl.CaseRunner
    _opts_cls = options.Options
    # List of files to check for zombie status
    _fjson_default = "pyKes.json"
    _warnmode_default = ccntl.DEFAULT_WARNMODE
    _zombie_files = (
        "*.out",
        "log/*.log")
  # >

  # ==================
  # Init config
  # ==================
  # <
    def init_post(self):
        r"""Do ``__init__()`` actions specific to ``pyfun``

        :Call:
            >>> cntl.init_post()
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                CAPE run matrix control instance
        :Versions:
            * 2023-07-10 ``@ddalle``: v1.0
        """
        # Read the namelist(s)
        self.ReadJobXML()
  # >

  # ==================
  # Overall Settings
  # ==================
  # <
    # Job name
    def get_job_name(self, j=0):
        r"""Get "job name" for phase *j*

        :Call:
            >>> name = cntl.get_job_name(j=0)
        :Inputs:
            *cntl*: :class:`Cntl`
                Instance of main CAPE control class
            *j*: {``0``} | :class:`int`
                Phase number
        :Outputs:
            *name*: :class:`str`
                Job name for phase *j*
        :Versions:
            * 2021-1-05 ``@ddalle``: Version 1.0
        """
        # Get default
        name = self.opts.get_ProjectName(j)
        # Use *opts* as primary
        if name is not None:
            return name
        # Read XML template
        self.ReadJobXML(j, False)
        # Get XML setting
        name = self.JobXML0.get_job_name()
        # Check if found
        if name is None:
            return "pykes"
        else:
            return name
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
            *cntl*: :class:`cape.pykes.cntl.Cntl`
                CAPE main control instance
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
            # Quit
            return
        elif kw.get('data', kw.get('db')):
            # Update all
            print("---- Updating FM DataBook components ----")
            self.UpdateFM(**kw)
            print("---- Updating CaseProp DataBook components ----")
            self.UpdateCaseProp(**kw)
            print("---- Updating PyFunc DataBook components ----")
            self.UpdatePyFuncDataBook(**kw)
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

  # === Primary Setup ===
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
        if n is not None:
            return
        # Run any case functions
        self.CaseFunction(i)
        # Prepare mesh
        self.PrepareMesh(i)
        # Get the run folder name
        frun = self.x.GetFullFolderNames(i)
        # Write the "conditions.json" file
        os.chdir(frun)
        self.x.WriteConditionsJSON(i)
        os.chdir(self.RootDir)
        # Read the XML file
        self.ReadJobXML()
        # Check for any "CaseFunction" hooks
        casekeys = self.x.GetKeysByType("CaseFunction")
        # Get the list of functions
        casefuncs = [
            self.x.defns[key].get("Function") for key in casekeys
        ]
        # # Loop through the functions
        for (key, funcname) in zip(casekeys, casefuncs):
            # Get handle to module
            try:
                func = eval("self.%s" % funcname)
            except Exception:
                print(
                    "  CaseFunction key '%s' function '%s()' not found"
                    % (key, funcname))
                continue
            # Check if callable
            if not callable(func):
                print("  CaseFunction '%s' not callable! Skipping." % key)
                continue
            # Run it
            func(self.x[key][i], i=i)
        # Prepare the XML file(s)
        self.PrepareJobXML(i)
        # Write "case.json"
        self.WriteCaseJSON(i)
        # Write the PBS script(s)
        self.WritePBS(i)

    # Prepare the job'x XML file(s)
    @ccntl.run_rootdir
    def PrepareJobXML(self, i):
        r"""Write ``pykes.xml`` file(s) for case *i*

        :Call:
            >>> cntl.PrepareJobXML(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE main control instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2021-10-26 ``@ddalle``: Version 1.0
        """
        # Get job name
        job_name = self.get_job_name(i)
        # Get run matrix
        x = self.x
        # (Re)read XML
        self.ReadJobXML()
        # Get XML file instance
        xml = self.JobXML
        # Enforce main job name
        xml.set_job_name(job_name)
        # Get the case name
        frun = self.x.GetFullFolderNames(i)
        # Exit if not folder
        if not os.path.isdir(frun):
            return
        # Set any flight conditions
        # Get condition type
        known_cond = xml.get_freestream("KnownCond")
        # Mach number
        mach = x.GetMach(i)
        if mach is not None and (known_cond is None or "M" in known_cond):
            xml.set_mach(mach)
        # Angle of attack
        a = x.GetAlpha(i)
        if a is not None:
            xml.set_alpha(a)
        # Sideslip angle
        b = x.GetBeta(i)
        if b is not None:
            xml.set_beta(b)
        # Reynolds number
        rey = x.GetReynoldsNumber(i)
        if rey is not None and (known_cond is None or "Re" in known_cond):
            xml.set_rey(rey)
        # Pressure
        p = x.GetPressure(i)
        if p is not None and (known_cond is None or "P" in known_cond):
            xml.set_pressure(p)
        # Temperature
        t = x.GetTemperature(i)
        if t is not None and (known_cond is None or "T" in known_cond):
            xml.set_temperature(t)
        # Velocity
        v = x.GetVelocity(i)
        if v is not None and (known_cond is None or "Vel" in known_cond):
            xml.set_velocity(v)
        # Find all *Path* and *File* elements
        elems1 = xml.findall_iter("Path")
        elems2 = xml.findall_iter("File")
        # Remove paths from file names
        for elem in elems1 + elems2:
            # Get file name
            fname = elem.text
            # Check for any
            if fname is None:
                continue
            # Reset to base name
            elem.text = os.path.basename(fname)
        # Cumulative iteration tracker
        m_j1 = 0
        # Find run matrix keys for XML input tags
        keys = self.x.GetKeysByType("XMLInput")
        # Loop through any
        for key in keys:
            # Get value
            v = self.x.GetValue(key, i)
            # Get input name
            name = self.x.defns[key].get("Name")
            # Set it
            xml.set_input(name, v)
        # Loop through phases
        for j in self.opts.get_PhaseSequence():
            # Set the restart flag according to phase
            if j == 0:
                xml.set_restart(False)
            else:
                xml.set_restart(True)
            # Get the items from *XML* section for this phase
            for xmlitem in self.opts.select_xml_phase(j):
                # Set item
                xml.set_section_item(**xmlitem)
            # Cutoff iters for phase *j*
            m_j = self.opts.get_PhaseIters(j)
            # Number planned for one go at *j*
            n_j = self.opts.get_nIter(j)
            # Check for startup_iters
            if j == 0:
                # Startup iterations count toward total
                n_startup = xml.get_value(
                    "BodyHierarchy.Simulation.StartupIterations")
            else:
                # Startup iterations are only for non-restart
                n_startup = 0
            # Set number of iterations
            xml.set_kcfd_iters(m_j1 + n_j)
            # Update actual count
            m_j1 = max(m_j, m_j1 + n_startup + n_j)
            # Name of output file
            fxml = os.path.join(frun, "kestrel.%02i.xml" % j)
            # Write it
            xml.write(fxml)

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
            os.mkdir(fgrp)
        if not os.path.isdir(frun):
            os.mkdir(frun)
        # Status update
        print("  Case name: '%s' (index %i)" % (frun, i))
        # Initialize copied/linked list
        copiedfiles = set()
        # Generic copy files
        self.copy_files(i)
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
                    # Nothing to do
                    copiedfiles.add(fbase)
                    # Warning
                    print("    Not writing '%s'; file already exists" % fbase)
                    continue
                # Check copy vs link
                if fname in linkfiles:
                    # Link it
                    os.link(fabs, fdest)
                else:
                    # Copy it
                    shutil.copy(fabs, fdest)
                # Add to list already copied
                copiedfiles.add(fbase)

  # === Case Interface ===
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
        rc = self.read_case_json(i)
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

    # Function to apply namelist settings to a case
    def ApplyCase(self, i, nPhase=None, **kw):
        r"""Apply settings from *cntl.opts* to an individual case

        This rewrites each run namelist file and the :file:`case.json`
        file in the specified directories.

        :Call:
            >>> cntl.ApplyCase(i, nPhase=None)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                FUN3D control interface
            *i*: :class:`int`
                Case number
            *nPhase*: {``None``} | positive :class:`int`
                Last phase number (default determined by *PhaseSequence*)
        :Versions:
            * 2016-03-31 ``@ddalle``: Version 1.0
        """
        # Ignore cases marked PASS
        if self.x.PASS[i] or self.x.ERROR[i]:
            return
        # Case function
        self.CaseFunction(i)
        # Read ``case.json``.
        rc = self.read_case_json(i)
        # Get present options
        rco = self.opts["RunControl"]
        # Exit if none
        if rc is None:
            return
        # Get the number of phases in ``case.json``
        nSeqC = rc.get_nSeq()
        # Get number of phases from present options
        nSeqO = self.opts.get_nSeq()
        # Check for input
        if nPhase is None:
            # Default: inherit from json
            nPhase = nSeqO
        else:
            # Use maximum
            nPhase = max(nSeqC, int(nPhase))
        # Present number of iterations
        nIter = rc.get_PhaseIters(nSeqC)
        # Get nominal phase breaks
        PhaseIters = self.GetPhaseBreaks()
        # Loop through the additional phases
        for j in range(nSeqC, nPhase):
            # Append the new phase
            rc["PhaseSequence"].append(j)
            # Get iterations for this phase
            if j >= nSeqO:
                # Add *nIter* iterations to last phase iter
                nj = self.opts.get_nIter(j)
            else:
                # Process number of *additional* iterations expected
                nj = PhaseIters[j] - PhaseIters[j-1]
            # Set the iteration count
            nIter += nj
            rc.set_PhaseIters(nIter, j)
            # Status update
            print("  Adding phase %s (to %s iterations)" % (j, nIter))
        # Copy other sections
        for k in rco:
            # Don't copy phase and iterations
            if k in ["PhaseIters", "PhaseSequence"]:
                continue
            # Otherwise, overwrite
            rc[k] = rco[k]
        # Write it
        self.WriteCaseJSON(i, rc=rc)
        # Write the conditions to a simple JSON file
        self.WriteConditionsJSON(i)
        # (Re)Prepare mesh in case needed
        print("  Checking mesh preparations")
        self.PrepareMesh(i)
        # Rewriting phases
        print("  Writing xml files 0 to %s" % (nPhase-1))
        self.PrepareJobXML(i)
        # Write PBS scripts
        nPBS = self.opts.get_nPBS()
        print("  Writing PBS scripts 0 to %s" % (nPBS-1))
        self.WritePBS(i)
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
            *q*: {``True``} | ``False``
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
            else:
                print("    File warning: could not find mesh file named")
                print("      %s" % fabs)
        # Output
        return meshfiles
  # >
