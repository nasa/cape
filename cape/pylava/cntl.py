# -*- coding: utf-8 -*-
r"""
:mod:`cape.pylavacntl`: LAVA control module
==================================================

This module provides tools to quickly setup basic or complex LAVA
run matrices and serve as an executive for pre-processing, running,
post-processing, and managing the solutions. A collection of cases
combined into a run matrix can be loaded using the following commands.

    .. code-block:: pycon

        >>> import cape.pylava
        >>> cntl = cape.pylava.cntl.Cntl("pyLava.json")
        >>> cntl
        <cape.pylava.Cntl(nCase=907)>
        >>> cntl.x.GetFullFolderNames(0)
        'poweroff/m1.5a0.0b0.0'


An instance of this :class:`cape.pylava.cntl.Cntl` class has many
attributes, which include the run matrix (``cntl.x``), the options
interface (``cntl.opts``), and optionally the data book
(``cntl.DataBook``), the appropriate input files (such as
``cntl.Namelist``), and possibly others.

    ====================   =============================================
    Attribute              Class
    ====================   =============================================
    *cntl.x*               :class:`cape.runmatrix.RunMatrix`
    *cntl.opts*            :class:`cape.pylava.options.Options`
    *cntl.DataBook*        :class:`cape.pylava.databook.DataBook`
    *cntl.Namelist*        :class:`cape.pylava.namelist.Namelist`
    ====================   =============================================

Finally, the :class:`Cntl` class is subclassed from the
:class:`cape.cntl.Cntl` class, so any methods available to the CAPE
class are also available here.
"""

# Standard library
import os
import shutil

# Third-party

# Local imports
from . import options
from . import casecntl
from . import databook
from . import report
from .yamlfile import RunYAMLFile
from ..optdict import OptionsDict
from ..cfdx import cntl as capecntl
from ..cfdx.cmdgen import infix_phase


# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyLavaFolder = os.path.split(_fname)[0]


# Class to read input files
class Cntl(capecntl.Cntl):
    r"""Class for handling global options and setup for LAVA

    This class is intended to handle all settings used to describe a
    group of LAVA cases. For situations where it is not sufficiently
    customized, it can be used partially, e.g., to set up a Mach/alpha
    sweep for each single control variable setting.

    The settings are read from a yaml or json file.

    :Call:
        >>> cntl = cape.pylava.Cntl(fname="pyLava.yaml")
    :Inputs:
        *fname*: :class:`str`
            Name of cape.pylava input file
    :Outputs:
        *cntl*: :class:`cape.pylava.cntl.Cntl`
            Instance of the cape.pylava control class
    :Data members:
        *cntl.opts*: :class:`dict`
            Dictionary of options for this case (directly from *fname*)
        *cntl.x*: :class:`cape.pylava.runmatrix.RunMatrix`
            Values and definitions for variables in the run matrix
        *cntl.RootDir*: :class:`str`
            Absolute path to the root directory
    :Versions:
        * 2024-04-25 ``@sneuhoff``: v1.0
    """
  # === Class Attributes ===
    _solver = "lavacurv"
    _case_mod = casecntl
    _databook_mod = databook
    _case_cls = casecntl.CaseRunner
    _opts_cls = options.Options
    _report_mod = report
    _fjson_default = "pyLava.json"
    # _fjson_default = "pyLava.yaml"
    yaml_default = "run_default.yaml"
    _zombie_files = (
        "*.out",
        "*.log")

  # === __DUNDER__ ===
    # Initialization method
    def __init__(self, fname=None):
        r"""Initialization method for :mod:`cape.pylava.cntl.Cntl`
        Handles yaml files for cape input by conversion from
        yaml to json.

        :Versions:
            * 2024-10-01 ``sneuhoff``: v1.0
        """
        # Check if json or yaml
        fext = fname.split('.')[-1]
        if fext in ("yaml", "yml"):
            # Convert yaml to json
            fout = f"{fname}.json"
            inyaml = OptionsDict(fname)
            inyaml.write_jsonfile(fout)
            # Call parent init using json
            super().__init__(fout)
        else:
            # Assume it's json, call parent init
            super().__init__(fname)

  # === Init config ===
    def init_post(self):
        r"""Do ``__init__()`` actions specific to ``pylava``

        :Call:
            >>> cntl.init_post()
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
        :Versions:
            * 2024-10-09 ``@ddalle``: v1.0
        """
        # Read list of custom file control classes
        self.ReadRunYAML()

  # === Case Preparation ===
    # Prepare a case
    @capecntl.run_rootdir
    def PrepareCase(self, i: int):
        """Prepare case for running if necessary

        :Call:
            >>> cntl.PrepareCase(i)
        :Inputs:
            *cntl*: :class:`cape.pylava.cntl.Cntl`
                Instance of control class containing relevant parameters
            *i*: :class:`int`
                Index of case to analyze
        :Versions:
            * 2024-06-12 ``@sneuhoff``: v1.0
        """
        # Ensure case index is set
        self.opts.setx_i(i)
        # Get the existing status
        n = self.CheckCase(i)
        # Quit if prepared.
        if n is not None:
            return None
        # Get the run name.
        frun = self.x.GetFullFolderNames(i)
        # Case function
        self.CaseFunction(i)
        # Prepare the mesh (and create folders if necessary).
        self.PrepareMesh(i)
        # Go to case folder
        os.chdir(frun)
        # Write the conditions to a simple JSON file.
        self.x.WriteConditionsJSON(i)
        # Write a JSON file with contents of "RunControl" section
        self.WriteCaseJSON(i)
        # Write the PBS script.
        self.WritePBS(i)
        # Write YAML file
        self.PrepareRunYAML(i)

    # Prepare the mesh for case *i* (if necessary)
    @capecntl.run_rootdir
    def PrepareMesh(self, i: int):
        r"""Copy/link mesh files into case folder

        :Call:
            >>> cntl.PrepareMesh(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix controller instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        # Get the case name
        frun = self.x.GetFullFolderNames(i)
        # Create case folder if needed
        self.make_case_folder(i)
        # Enter the case folder
        os.chdir(frun)
        # ----------
        # Copy files
        # ----------
        # Get the configuration folder
        fcfg = self.opts.get_MeshConfigDir()
        fcfg_abs = os.path.join(self.RootDir, fcfg)
        # Get the names of the raw input files and target files
        fmsh = self.opts.get_MeshCopyFiles(i=i)
        # Loop through those files
        for j in range(len(fmsh)):
            # Original and final file names
            f0 = os.path.join(fcfg_abs, fmsh[j])
            f1 = os.path.split(fmsh[j])[1]
            # Skip if full file
            if os.path.isfile(f1):
                continue
            # Copy the file.
            if os.path.isfile(f0):
                shutil.copy(f0, f1)
        # Get the names of input files to link
        fmsh = self.opts.get_MeshLinkFiles(i=i)
        # Loop through those files
        for j in range(len(fmsh)):
            # Original and final file names
            f0 = os.path.join(fcfg_abs, fmsh[j])
            f1 = os.path.split(fmsh[j])[1]
            # Remove the file if necessary
            if os.path.islink(f1):
                os.remove(f1)
            # Skip if full file
            if os.path.isfile(f1):
                continue
            # Link the file.
            if os.path.isfile(f0) or os.path.isdir(f0):
                os.symlink(f0, f1)

  # === Input files ===
    # Read template YAML file
    def ReadRunYAML(self):
        r"""Read run YAML file, using template if setting is empty

        :Call:
            >>> cntl.ReadRunYAML()
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
        :Versions:
            * 2024-08-19 ``@sneuhoff``: v1.0 (``ReadInputFile()``)
            * 2024-10-09 ``@ddalle``: v2.0
        """
        # Get name of file to read
        fname = self.opts.get_RunYAMLFile()
        # Check for template
        if fname is None:
            # Read template
            fabs = os.path.join(PyLavaFolder, "templates", "run.yaml")
        else:
            # Absolutize
            fabs = os.path.join(self.RootDir, fname)
        # Read it
        self.YamlFile = RunYAMLFile(fabs)

    @capecntl.run_rootdir
    def PrepareRunYAML(self, i: int):
        r"""Prepare the run YAML file for each phase of one case

        :Call:
            >>> cntl.PrepareRunYAML(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        # Set case index for options
        self.opts.setx_i(i)
        # Set flight conditions
        self.PrepareRunYAMLFlightConditions(i)
        # Get user's selected file name
        yamlbase = self.opts.get_lava_yamlfile()
        # Get name of case folder
        frun = self.x.GetFullFolderNames(i)
        # Enter said folder
        os.chdir(frun)
        # Loop through phases
        for j in self.opts.get_PhaseSequence():
            # Select file name
            if isinstance(yamlbase, list):
                # Specified by phase
                yamlfile = self.opts.get_lava_yamlfile(j)
            else:
                # Add phase infix
                yamlfile = infix_phase(yamlbase, j)
            # Other preparation
            ...
            # Write file
            self.YamlFile.write_yamlfile(yamlfile)

    # Prepare the flight conditions
    def PrepareRunYAMLFlightConditions(self, i: int):
        r"""Prepare the flight conditions variables in a LAVA YAML file

        :Call:
            >>> cntl.PrepareRunYAMLFlightConditions(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        # Get properties
        u = self.x.GetVelocity(i, units="m/s")
        p = self.x.GetPressure(i, units="Pa")
        T = self.x.GetTemperature(i, units="K")
        M = self.x.GetMach(i)
        a = self.x.GetAlpha(i)
        b = self.x.GetBeta(i)
        # Get YAML interface
        opts = self.YamlFile
        # Get defaults
        fabs = os.path.join(PyLavaFolder, "templates", "run.yaml")
        DefaultYaml = RunYAMLFile(fabs)
        # Set velocity if any velocity setting was given
        if u is not None and M is not None:
            raise ValueError("Specify only one of umag and Mach")
        if u is not None:
            opts.set_umag(u)
        if M is not None:
            gamma = opts.get_refcond('gamma')
            if gamma is None:
                gamma = DefaultYaml.get_refcond('gamma')
            cp = opts.get_refcond('cp')
            if cp is None:
                cp = DefaultYaml.get_refcond('cp')
            R = ((gamma-1.0)/gamma)*cp
            T = opts.get_temperature()
            sos = (gamma*R*T)**0.5
            u = M*sos
            opts.set_umag(u)
        # Set angle of attack
        if a is not None:
            opts.set_alpha(a)
        # Set sideslip angle
        if b is not None:
            opts.set_beta(b)
        # Set pressure if specified
        if p is not None:
            opts.set_pressure(p)
        # Set temperature if specified
        if T is not None:
            opts.set_temperature(T)
        # Set nonlinear iterations
        np = int(self.opts.get_PhaseIters())
        opts.set_lava_subopt('nonlinearsolver', 'iterations', np)
