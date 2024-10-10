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
    *cntl.DataBook*        :class:`cape.pylava.dataBook.DataBook`
    *cntl.Namelist*        :class:`cape.pylava.namelist.Namelist`
    ====================   =============================================

Finally, the :class:`Cntl` class is subclassed from the
:class:`cape.cntl.Cntl` class, so any methods available to the CAPE
class are also available here.
"""

# Standard library
import os
import shutil
import math

# Third-party
import yaml

# Local imports
from . import options
from . import casecntl
from . import dataBook
from ..cfdx import cntl as capecntl
from ..cfdx.options.util import applyDefaults


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

    The settings are read from a JSON file.

    :Call:
        >>> cntl = cape.pylava.Cntl(fname="pyLava.json")
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
    _databook_mod = dataBook
    _case_cls = casecntl.CaseRunner
    _opts_cls = options.Options
    _fjson_default = "pyLava.json"
    yaml_default = "run_default.yaml"
    _zombie_files = (
        "*.out",
        "*.log",
    )

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
        self.ReadRunYaml()

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
        # Get the existing status.
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
        # Make the directory if necessary
        self.make_case_folder(i)
        # Go there.
        os.chdir(frun)
        # Write the conditions to a simple JSON file.
        self.x.WriteConditionsJSON(i)
        # Write a JSON file with contents of "RunControl" section
        self.WriteCaseJSON(i)
        # Write the PBS script.
        self.WritePBS(i)
        # Read the (template) input file in the root directory
        self.ReadInputFile()
        # Apply case specific conditions and write to case directory
        self.PrepareInputFile(i)

    # Prepare the mesh for case *i* (if necessary)
    @capecntl.run_rootdir
    def PrepareMesh(self, i: int):
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Create case folder if needed
        self.make_case_folder(i)
        # Enter the case folder.
        os.chdir(frun)
        # ----------
        # Copy files
        # ----------
        # Get the configuration folder
        fcfg = self.RootDir
        # Get the names of the raw input files and target files
        fmsh = self.opts.get_MeshCopyFiles(i=i)
        # Loop through those files
        for j in range(len(fmsh)):
            # Original and final file names
            f0 = os.path.join(fcfg, fmsh[j])
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
            f0 = os.path.join(fcfg, fmsh[j])
            f1 = os.path.split(fmsh[j])[1]
            # Remove the file if necessary
            if os.path.islink(f1):
                os.remove(f1)
            # Skip if full file
            if os.path.isfile(f1):
                continue
            # Link the file.
            if os.path.isfile(f0):
                os.symlink(f0, f1)
    # >

    def ReadInputFile(self):
        r"""Read the root-directory LAVA input file

        :Call:
            >>> cntl.ReadInputFile()
        :Inputs:
            *cntl*: :class:`cape.pylava.cntl.Cntl`
                Instance of global pylava settings object
        :Versions:
            * 2024-08-19 ``@sneuhoff``: v1.0
        """
        fpwd = os.getcwd()
        os.chdir(self.RootDir)
        fname = self.opts.get_RunYaml()
        if os.path.isfile(fname):
            with open(fname, 'r') as f:
                self.InputFile = yaml.safe_load(f)
        else:
            print(f"Input file {fname} not found.")

        # Ensure all dict keys are lower case
        self.InputFile = self.lower_dict_keys(self.InputFile)
        # Load in LAVA default options
        YamlDefaultsPath = PyLavaFolder+"/"+self.yaml_default
        with open(YamlDefaultsPath, 'r') as f:
            YamlDefaults = yaml.safe_load(f)
        # Ensure all dict keys are lower case
        YamlDefaults = self.lower_dict_keys(YamlDefaults)
        # Apply given options onto defaults
        applyDefaults(self.InputFile, YamlDefaults)
        os.chdir(fpwd)
    # >

    @capecntl.run_rootdir
    def PrepareInputFile(self, i: int):
        r"""Write LAVA input file for run case *i*

        :Call:
            >>> cntl.PrepareInputFile(i)
        :Inputs:
            *cntl*: :class:`cape.pylava.cntl.Cntl`
                Instance of global pylava settings object
            *i*: :class:`int`
                Run index
        :Versions:
            * 2024-08-19 ``@sneuhoff``: v1.0
        """
        # Get this case's run conditions from run matrix
        x = self.x
        # For bullet case, get Mach, alpha, beta
        Mach = x.GetMach(i)
        gamma = self.InputFile['referenceconditions']['gamma']
        temperature = self.InputFile['referenceconditions']['temperature']
        cp = self.InputFile['referenceconditions']['cp']
        gasconstant = ((gamma-1.0)/gamma)*cp
        soundspeed = math.sqrt(gamma*gasconstant*temperature)
        self.InputFile['nonlinearsolver']['iterations'] = int(self.opts.get_PhaseIters())

        self.InputFile['referenceconditions']['alpha'] = float(x.GetAlpha(i))
        self.InputFile['referenceconditions']['beta'] = float(x.GetBeta(i))
        self.InputFile['referenceconditions']['umag'] = float(Mach*soundspeed)

        # Get the case.
        frun = self.x.GetFullFolderNames(i)
        fout = os.path.join(frun, self.opts.get_RunYaml())
        with open(fout, "w") as f:
            yaml.dump(self.InputFile, f, default_flow_style=False)
    # >

    def lower_dict_keys(self, x):
        if isinstance(x, list):
            return [self.lower_dict_keys(v) for v in x]
        elif isinstance(x, dict):
            return dict((k.lower(), self.lower_dict_keys(v)) for k, v in x.items())
        else:
            return x
