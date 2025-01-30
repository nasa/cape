r"""
:mod:`cape.cfdx.cntlbase`: Abstract base module for run matrix control
=======================================================================

This module provides an abstract base class :class:`CntlBase`, which
contains much of the functionality for controlling a full CFD run matrix
in CAPE.

:See also:
    * :mod:`cape.cfdx.cntl`
"""


# Standard library modules
import copy
import functools
import getpass
import glob
import importlib
import os
import re
import shutil
import time
from abc import ABC, abstractmethod
from datetime import datetime
from io import IOBase
from typing import Optional

# Third-party modules
import numpy as np

# Local modules
from . import queue
from .. import convert
from .. import console
from .. import argread

# Functions and classes from other modules
from .runmatrix import RunMatrix
from .options import Options
from ..config import ConfigXML, ConfigJSON
from ..optdict import WARNMODE_WARN, WARNMODE_QUIET
from ..optdict.optitem import getel

# Import triangulation
from ..geom import RotatePoints
from ..trifile import ReadTriFile


# Constants
DEFAULT_WARNMODE = WARNMODE_WARN
MATRIX_CHUNK_SIZE = 1000
UGRID_EXTS = (
    "b4",
    "b8",
    "b8l",
    "lb4",
    "lb8",
    "lb8l",
    "lr4",
    "lr8",
    "r4",
    "r8",
)
DEG = np.pi / 180.0


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
        * 2018-11-20 ``@ddalle``: v1.0
        * 2020-02-25 ``@ddalle``: v1.1: better exceptions
        * 2023-06-16 ``@ddalle``: v1.2; use ``finally``
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
            # Raise the error
            raise
        except KeyboardInterrupt:
            # Raise the error
            raise
        finally:
            # Go back to original folder (always)
            os.chdir(fpwd)
        # Return function values
        return v
    # Apply the wrapper
    return wrapper_func


# Class to read input files
class CntlBase(ABC):
    r"""Class to handle options, setup, and execution of CFD codes

    :Call:
        >>> cntl = cape.CntlBase(fname="cape.json")
    :Inputs:
        *fname*: :class:`str`
            Name of JSON settings file from which to read options
    :Outputs:
        *cntl*: :class:`cape.cfdx.cntl.Cntl`
            Instance of Cape control interface
        *cntl.opts*: :class:`cape.cfdx.options.Options`
            Options interface
        *cntl.x*: :class:`cape.runmatrix.RunMatrix`
            Run matrix interface
        *cntl.RootDir*: :class:`str`
            Working directory from which the class was generated
    """
   # --- Class Attributes ---
    # Names
    _name = "cfdx"
    _solver = "cfdx"
    # Hooks to py{x} specific classes
    _opts_cls = Options
    # Other settings
    _fjson_default = "cape.json"
    _warnmode_default = WARNMODE_QUIET
    _warnmode_envvar = "CAPE_WARNMODE"
    _zombie_files = ["*.out"]

   # --- __DUNDER__ ---
    # Initialization method
    def __init__(self, fname=None):
        r"""Initialization method for :mod:`cape.cfdx.cntl.Cntl`

        :Versions:
            * 2015-09-20 ``ddalle``: v1.0
        """
        # Check fname
        if fname is None:
            fname = self._fjson_default
        # Check if file exists
        if not os.path.isfile(fname):
            # Raise error but suppress traceback
            os.sys.tracebacklimit = 0
            raise ValueError("No cape control file '%s' found" % fname)

        # Save the current directory as the root
        self.RootDir = os.getcwd()
        # Current case runner
        self.caserunner = None
        self.caseindex = None
        # Read options
        self.read_options(fname)
        # Import modules
        self.modules = {}
        self.ImportModules()
        # Process the trajectory.
        self.x = RunMatrix(**self.opts['RunMatrix'])
        # Save conditions w/i options
        self.opts.save_x(self.x)
        # Set initial index
        self.opts.setx_i(0)
        # Job list
        self.jobs = {}
        # Run cntl init functions, customize for py{x}
        self.init_post()
        # Run any initialization functions
        self.InitFunction()
        # Initialize slots
        self.DataBook = None

    # Output representation
    def __repr__(self):
        r"""Output representation method for Cntl class

        :Versions:
            * 2015-09-20 ``@ddalle``: v1.0
        """
        # Get class handle
        cls = self.__class__
        # Display basic information
        return "<%s.%s(nCase=%i)>" % (
            cls.__module__,
            cls.__name__,
            self.x.nCase)

   # --- Other Init ---
    def init_post(self):
        r"""Do ``py{x}`` specific initialization actions

        :Call:
            >>> cntl.init_post()
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                CAPE run matrix control instance
        :Versions:
            * 2023-05-31 ``@ddalle``: v1.0
        """
        pass

   # --- Hooks & Modules ---
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
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Instance of Cape control interface
        :Versions:
            * 2014-10-08 ``@ddalle``: v1.0 (:mod:`pycart`)
            * 2015-09-20 ``@ddalle``: v1.0
            * 2022-04-12 ``@ddalle``: v2.0; use *self.modules*
        """
        # Get module soption
        module_list = self.opts.get("Modules")
        # Exit if none
        if not module_list:
            return
        # Ensure list
        if not isinstance(module_list, list):
            raise TypeError(
                'Expected "Modules" option to be a list; got "%s"' %
                type(module_list).__name__)
        # Loop through modules
        for module_spec in module_list:
            # Check for list
            if isinstance(module_spec, list):
                # Check length
                if len(module_spec) != 2:
                    raise IndexError(
                        'Expected "list" type in "Modules" to have' +
                        ('len=2, got %i' % len(module_spec)))
                # Get the file name and import name separately
                import_name, as_name = module_spec
                # Status update
                print("Importing module '%s' as '%s'" % (import_name, as_name))
            else:
                # Import as the default name
                import_name = module_spec
                as_name = module_spec
                # Status update
                print("Importing module '%s'" % import_name)
            # Load the module by its name
            self.modules[as_name] = importlib.import_module(import_name)

    # Execute a function
    def exec_modfunction(self, funcname: str, a=None, kw=None, name=None):
        r"""Execute a function from *cntl.modules*

        :Call:
            >>> v = cntl.exec_modfunction(funcname, a, kw, name=None)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall control interface
            *func*: :class:`str`
                Name of function to execute, e.g. ``"mymod.myfunc"``
            *a*: {``None``} | :class:`tuple`
                Positional arguments to called function
            *kw*: {``None``} | :class:`dict`
                Keyworkd arguments to called function
            *name*: {``None``} | :class:`str`
                Hook name to use in status update
        :Outputs:
            *v*: **any**
                Output from execution of function
        :Versions:
            * 2022-04-12 ``@ddalle``: v1.0
        """
        # Status update if appropriate
        if name:
            print("  %s: %s()" % (name, funcname))
        # Default args and kwargs
        if a is None:
            a = tuple()
        elif not isinstance(a, tuple):
            a = a,
        if kw is None:
            kw = {}
        # Split name into module(s) and function name
        funcparts = funcname.split(".")
        # Has to be at least two parts
        if len(funcparts) < 2:
            raise ValueError(
                "Function spec '%s' must contain at least one '.'" % funcname)
        # Get module name
        modname = funcparts.pop(0)
        mod = self.modules.get(modname)
        # Cumulative spec
        spec = modname
        # Check for module
        if mod is None:
            raise KeyError('No module "%s" in Cntl instance' % modname)
        # Loop through remaining specs
        func = mod
        for j, part in enumerate(funcparts):
            # Get next spec
            mod = func
            func = mod.__dict__.get(part)
            spec = spec + "." + part
            # Check if found
            if func is None:
                raise AttributeError("No spec '%s' found" % spec)
        # Check if final spec is callable
        if not callable(func):
            raise TypeError("Spec '%s' is not callable" % spec)
        # Execute it and return value
        return func(*a, **kw)

    def _exec_funclist(self, funclist, a=None, kw=None, name=None):
        r"""Execute a list of functions in one category

        :Call:
            >>>  cntl._exec_funclist(funclist, a, kw, name=None)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall control interface
            *funclist*: :class:`list`\ [:class:`str`]
                List of function specs to execute
            *a*: {``None``} | :class:`tuple`
                Positional arguments to called function
            *kw*: {``None``} | :class:`dict`
                Keyworkd arguments to called function
            *name*: {``None``} | :class:`str`
                Hook name to use in status update
        :Versions:
            * 2022-04-12 ``@ddalle``: v1.0
        """
        # Exit if none
        if not funclist:
            return
        # Ensure list
        if not isinstance(funclist, list):
            if name:
                raise TypeError(
                    ('Option "%s" must be a list; got ' % name) +
                    ("'%s'" % type(funclist).__name__))
            else:
                raise TypeError(
                    "Expected list of functions; got " +
                    ("'%s'" % type(funclist).__name__))
        # Loop through functions
        for func in funclist:
            # Execute function
            self.exec_modfunction(func, a, kw, name)

    # Function to apply initialization function
    def InitFunction(self):
        r"""Run one or more functions a "initialization" hook

        This calls the function(s) in the global ``"InitFunction"``
        option from the JSON file.  These functions must take *cntl* as
        an input, and they are usually from a module imported via the
        ``"Modules"`` option.  See the following example:

            .. code-block:: javascript

                "Modules": ["testmod"],
                "InitFunction": ["testmod.testfunc"]

        This leads CAPE to call ``testmod.testfunc(cntl)``.

        :Call:
            >>> cntl.InitFunction()
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall control interface
        :Versions:
            * 2017-04-04 ``@ddalle``: v1.0
            * 2022-04-12 ``@ddalle``: v2.0; use _exec_funclist()
        """
        # Get input functions
        funclist = self.opts.get("InitFunction")
        # Execute each
        self._exec_funclist(funclist, self, name="InitFunction")

    # Call function to apply settings for case *i*
    def CaseFunction(self, i: int):
        r"""Run one or more functions at "prepare-case" hook

        This function is executed at the beginning of
        :func:`PrepareCase(i)`.

        This is meant to serve as a filter if a user wants to change the
        settings for some subset of the cases.  Using this function can
        change any setting, which can be dependent on the case *i*.

        This calls the function(s) in the global ``"CaseFunction"``
        option from the JSON file. These functions must take *cntl* as
        an input and the case number *i*. The function(s) are usually
        from a module imported via the ``"Modules"`` option. See the
        following example:

            .. code-block:: javascript

                "Modules": ["testmod"],
                "CaseFunction": ["testmod.testfunc"]

        This leads CAPE to call ``testmod.testfunc(cntl, i)`` at the
        beginning of :func:`PrepareCase` for each case *i* in the run
        matrix.  The function is also called at the beginning of
        :func:`ApplyCase` if ``pyfun --apply`` or similar is issued.

        :Call:
            >>> cntl.CaseFunction(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall control interface
            *i*: :class:`int`
                Case number
        :Versions:
            * 2017-04-05 ``@ddalle``: v1.0
        :See also:
            * :func:`cape.cfdx.cntl.Cntl.InitFunction`
            * :func:`cape.cfdx.cntl.Cntl.PrepareCase`
            * :func:`cape.pycart.cntl.Cntl.PrepareCase`
            * :func:`cape.pycart.cntl.Cntl.ApplyCase`
            * :func:`cape.pyfun.cntl.Cntl.PrepareCase`
            * :func:`cape.pyfun.cntl.Cntl.ApplyCase`
            * :func:`cape.pyover.cntl.Cntl.PrepareCase`
            * :func:`cape.pyover.cntl.Cntl.ApplyCase`
        """
        # Get input functions
        funclist = self.opts.get("CaseFunction")
        # Execute each
        self._exec_funclist(funclist, (self, i), name="CaseFunction")

   # --- Files ---
    # Absolutize
    def abspath(self, fname: str) -> str:
        r"""Absolutize a file name

        :Call:
            >>> fabs = cntl.abspath(fname)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE main control instance
            *fname*: :class:`str`
                A file name
        :Outputs:
            *fabs*: :class:`str`
                Absolute file path
        :Versions:
            * 2021-10-25 ``@ddalle``: v1.0
        """
        # Check if absolute
        if os.path.isabs(fname):
            # Already absolute
            return fname
        else:
            # Relative to *RootDir*
            return os.path.join(self.RootDir, fname)

   # --- Input Readers ---
    @abstractmethod
    def ReadDataBook(self, comp: Optional[str] = None):
        pass

    @abstractmethod
    def ReadReport(self, rep: str):
        pass

    # Function to prepare the triangulation for each grid folder
    @run_rootdir
    def ReadTri(self):
        r"""Read initial triangulation file(s)

        :Call:
            >>> cntl.ReadTri()
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
        :Versions:
            * 2014-08-30 ``@ddalle``: v1.0
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
        if not isinstance(ftri, (list, np.ndarray)):
            ftri = [ftri]
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
        r"""Read ``Config.xml`` or other surf configuration format

        :Call:
            >>> cntl.ReadConfig(f=False)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *f*: ``True`` | {``False``}
                Option to reread existing *cntl.config*
        :Versions:
            * 2016-06-10 ``@ddalle``: v1.0
            * 2016-10-21 ``@ddalle``: v2.0, added ``Config.json``
            * 2020-09-01 ``@ddalle``: v2.1, add *f* kwarg
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

   # --- Options ---
    # Read options (first time)
    def read_options(self, fjson: str):
        r"""Read options using appropriate class

        This allows users to set warning mode via an environment
        variable.

        :Call:
            >>> cntl.read_options(fjson)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                CAPE run matrix control instance
            *fjson*: :class:`str`
                Name of JSON file to read
        :Versions:
            * 2023-05-26 ``@ddalle``: v1.0; forked from __init__()
            * 2023-12-13 ``@ddalle``: v1.1; add *RootDir* and *JSONFile*
        """
        # Get class
        cls = self.__class__
        optscls = cls._opts_cls
        # Environment variable
        envvar = cls._warnmode_envvar
        warnmode_def = cls._warnmode_default
        # Read environment
        warnmode = os.environ.get(envvar, warnmode_def)
        # Convert to integer if string
        if isinstance(warnmode, str):
            warnmode = int(warnmode)
        # Read settings
        self.opts = optscls(fjson, _warnmode=warnmode)
        # Save root dir
        self.opts.set_RootDir(self.RootDir)
        # Follow any links
        freal = os.path.realpath(fjson)
        # Save path relative to RootDir
        frel = os.path.relpath(freal, self.RootDir)
        self.opts.set_JSONFile(frel)

    # Copy all options
    def SaveOptions(self):
        r"""Copy *cntl.opts* and store it as *cntl.opts0*

        :Call:
            >>> cntl.SaveOptions()
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE solver control interface
        :Versions:
            * 2021-07-31 ``@ddalle``: v1.0
        """
        # Copy the options
        self._opts0 = copy.deepcopy(self.opts)

    # Reset options to last "save"
    def RevertOptions(self):
        r"""Revert to *cntl.opts0* as working options

        :Call:
            >>> cntl.ResetOptions()
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE solver control interface
        :Versions:
            * 2021-07-31 ``@ddalle``: v1.0
        """
        # Get the saved options
        try:
            opts0 = self._opts0
        except AttributeError:
            opts0 = None
        # Check for null options
        if opts0 is None:
            raise AttributeError("No *cntl._opts0* options archived")
        # Revert options
        self.opts = copy.deepcopy(opts0)

   # --- Command-Line Interface ---
    # Preprocessor for idnices
    def cli_preprocess(self, *a, **kw):
        r"""Preprocess command-line arguments and flags/keywords

        :Call:
            >>> a, kw = cntl.cli_preprocess(*a, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *kw*: :class:`dict`\ [``True`` | ``False`` | :class:`str`]
                Command-line keyword arguments and flags
        :Outputs:
            *a*: :class:`tuple`
                List of sequential command-line parameters
            *kw*: :class:`dict`
                Flags with any additional preprocessing performed
        :Versions:
            * 2018-10-19 ``@ddalle``: v1.0
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

        # Get list of scripts in the "__replaced__" section
        kwx = [
            valj for optj, valj in kw.get('__replaced__', []) if optj == "x"
        ]
        # Append the last "-x" input
        if 'x' in kw:
            kwx.append(kw['x'])
        # Apply all scripts
        for fx in kwx:
            # Open file and execute it
            exec(open(fx).read())

        # Output
        return a, kw

    # CLI arg preprocesser
    def preprocess_kwargs(self, kw: dict):
        r"""Preprocess command-line arguments and flags/keywords

        This will effect the following CLI options:

        --cons CONS
            Comma-separated constraints split into a list

        -x FPY
            Each ``-x`` argument is executed (can be repeated)

        -I INDS
            Convert *INDS* like ``3-6,8`` to ``[3, 4, 5, 8]``

        :Call:
            >>> opts = cntl.cli_preprocess(*a, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *kw*: :class:`dict`\ [``True`` | ``False`` | :class:`str`]
                CLI keyword arguments and flags, modified in-place
        :Versions:
            * 2024-12-19 ``@ddalle``: v1.0
        """
        # Get constraints and convert text to list
        cons = kw.get('cons')
        if cons:
            kw["cons"] = [con.strip() for con in cons.split(',')]
        # Get explicit indices
        inds = kw.get("I")
        if inds:
            kw["I"] = self.x.ExpandIndices(inds)

        # Get list of scripts in the "__replaced__" section
        kwx = [
            valj for optj, valj in kw.get('__replaced__', []) if optj == "x"
        ]
        # Append the last "-x" input
        x = kw.pop("x", None)
        if x:
            kwx.append(x)
        # Apply all scripts
        for fx in kwx:
            # Open file and execute it
            exec(open(fx).read())

    # Baseline function
    def cli_cape(self, *a, **kw):
        r"""Common command-line interface

        This function is applied after the command-line arguments are
        parsed using :func:`cape.argread.readflagstar` and the control
        interface has already been read according to the ``-f`` flag.

        :Call:
            >>> cmd = cntl.cli_cape(*a, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *kw*: :class:`dict`
                Preprocessed command-line keyword arguments
        :Outputs:
            *cmd*: ``None`` | :class:`str`
                Name of command that was processed, if any
        :Versions:
            * 2018-10-19 ``@ddalle``: v1.0
            * 2023-06-21 ``@aburkhea``: v1.1; add --report options
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
        elif kw.get('dezombie'):
            # Kill ZOMBIE jobs and delete running file
            self.Dezombie(**kw)
            return "dezombie"
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
        # Check whether to execute scripts
        elif kw.get('exec', kw.get('e')):
            self.ExecScript(**kw)
            return 'e'
        elif kw.get('ll'):
            # Update line load data book
            self.UpdateLL(**kw)
            return 'll'
        elif kw.get('ts'):
            # Update time series data book
            self.UpdateTS(**kw)
            return 'ts'
        elif kw.get('triqfm'):
            # Update TriqFM data book
            self.UpdateTriqFM(**kw)
            return 'triqfm'
        elif kw.get("prop"):
            # Update CaseProp data book
            self.UpdateCaseProp(**kw)
            return 'prop'
        elif kw.get("dbpyfunc"):
            # Update PyFunc data book
            self.UpdateDBPyFunc(**kw)
            return "dbpyfunc"
        elif kw.get('data', kw.get('db')):
            # Update all
            print("---- Updating FM DataBook components ----")
            self.UpdateFM(**kw)
            print("---- Updating LineLoad DataBook components ----")
            self.UpdateLL(**kw)
            print("---- Updating TriqFM DataBook components ----")
            self.UpdateTriqFM(**kw)
            print("---- Updating CaseProp DataBook components ----")
            self.UpdateCaseProp(**kw)
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
            if kw['report'] is True:
                # First report
                rep = self.opts.get_ReportList()[0]
            else:
                # User-specified report
                rep = kw['report']
            # Get the report
            R = self.ReadReport(rep)
            # Check for force update
            R.force_update = kw.get("force", False)
            # Check if asking to delete cases
            if kw.get("rm", False):
                # Remove the cases dirs
                R.RemoveCases(**kw)
                return 'rm-report-case'
            else:
                # Update according to other options
                R.UpdateReport(**kw)
                return 'report'

    # Baseline function
    def cli(self, *a, **kw):
        r"""Command-line interface to ``cape``

        :Call:
            >>> cntl.cli(*a, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *kw*: :class:`dict`\ [``True`` | ``False`` | :class:`str`]
                Unprocessed keyword arguments
        :Outputs:
            *cmd*: ``None`` | :class:`str`
                Name of command that was processed, if any
        :Versions:
            * 2018-10-19 ``@ddalle``: v1.0
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
        r"""Display current status for all cases

        This prints case names, current iteration numbers, and so on.
        This is the function that is called when the user issues a
        system command like ``cape -c``.

        :Call:
            >>> cntl.DisplayStatus(j=False)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *j*: :class:`bool`
                Whether or not to display job ID numbers
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2014-10-04 ``@ddalle``: v1.0
            * 2014-12-09 ``@ddalle``: v2.0, ``--cons``
        """
        # Force the "check" option to true.
        kw['c'] = True
        # Call the job submitter but don't allow submitting.
        self.SubmitJobs(**kw)

    # Master interface function
    def SubmitJobs(self, **kw):
        r"""Check jobs and prepare or submit jobs if necessary

        :Call:
            >>> cntl.SubmitJobs(**kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *c*: :``True`` | {``False``}
                If ``True``, only display status; do not submit new jobs
            *j*: :class:`bool`
                Whether or not to display job ID numbers
            *n*: :class:`int`
                Maximum number of jobs to submit
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2014-10-05 ``@ddalle``: v1.0
            * 2014-12-09 ``@ddalle``: v2.0, ``--cons``
            * 2021-08-01 ``@ddalle``: v2.1, save/revert options
            * 2023-05-17 ``@ddalle``: v2.2, ``opts.setx_i()``
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
        # Check whether or not to kill PBS jobs
        qKill = kw.get('qdel', kw.get('kill', kw.get('scancel', False)))
        # Option to run "incremental" job
        q_incr = kw.get("incremental", False)
        # No submissions if we're just deleting.
        if qKill or qDel:
            qCheck = True
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
        # Check for showing errored casses
        if kw.get("errored", False) or kw.get("failed", False):
            q_error = True
        else:
            q_error = False
        # Maximum number of jobs to submit
        nSubMax = int(kw.get('n', 10))
        # Requested number to keep running
        nJob = self.opts["RunControl"].get_NJob()
        nJob = 0 if nJob is None else nJob
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
        # Check for --no-qstat
        qstat = kw.get("qstat", True)
        # Get the qstat info (safely; do not raise an exception)
        jobs = self.get_pbs_jobs(
            force=True,
            u=kw.get('u'),
            qstat=qstat)
        # Check for auto-submit options
        if (nJob > 0) and kw.get("auto", True):
            # Look for running cases
            nRunning = self.CountQueuedCases(jobs=jobs, u=kw.get('u'))
            # Reset nSubMax to the cape minus number running
            nSubMax = min(nSubMax, nJob - nRunning)
            # Status update
            print(f"Found {nRunning} running cases out of {nJob} max")
            # check to see if the max are already running
            if nRunning >= nJob and not qCheck:
                print(f"Aborting because >={nJob} cases already running.\n")
                return
            print("")
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
        f, s = '-', ' '
        # Create the string stencil.
        if qJobID:
            # Print status with job numbers.
            stncl = ('%%-%is ' * 7) % (4, lrun, 7, 11, 3, 8, 7)
            # Print header row.
            print(
                stncl % (
                    "Case", "Config/Run Directory", "Status",
                    "Iterations", "Que", "CPU Time", "Job ID"))
            # Print "---- --------" etc.
            print(
                f*4 + s + f*lrun + s + f*7 + s + f*11 + s + f*3 + s +
                f*8 + s + f*7)
        else:
            # Print status without job numbers.
            stncl = ('%%-%is ' * 6) % (4, lrun, 7, 11, 3, 8)
            # Print header row.
            print(
                stncl % (
                    "Case", "Config/Run Directory", "Status",
                    "Iterations", "Que", "CPU Time"))
            # Print "---- --------" etc.
            print(f*4 + s + f*lrun + s + f*7 + s + f*11 + s + f*3 + s + f*8)
       # -------
       # Loop
       # -------
        # Initialize number of submitted jobs
        nSub = 0
        # Number of deleted jobs
        nDel = 0
        # Initialize dictionary of statuses.3
        total = {
            'PASS': 0,
            'PASS*': 0,
            '---': 0,
            'INCOMP': 0,
            'RUN': 0,
            'DONE': 0,
            'QUEUE': 0,
            'ERROR': 0,
            'ZOMBIE': 0,
            'THIS_JOB': 0,
        }
        # Save current options
        if not qCheck:
            self.SaveOptions()
        # Loop through the runs.
        for j in range(len(I)):
           # --- Case ID ---
            # Case index
            i = I[j]
            # Set current state
            self.opts.setx_i(i)
            # Extract case
            frun = fruns[j]
           # --- Mark check ---
            # Check for unmarked-only flag
            if q_umark and (self.x.PASS[i] or self.x.ERROR[i]):
                continue
            if q_error and not self.x.ERROR[i]:
                continue
           # --- Status ---
            # Check status.
            sts = self.CheckCaseStatus(i, jobs, u=kw.get('u'), qstat=qstat)
            # Get active job number
            jobID = self.GetPBSJobID(i)
            # Append.
            total[sts] += 1
            # Get the current number of iterations
            n = self.CheckCase(i)
            # Get CPU hours
            t = self.GetCPUTime(i)
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
                # Isolate number
                try:
                    job_num = int(jobID.split(".")[0])
                except Exception:
                    job_num = jobID
                # Print job number
                print(stncl % (i, frun, sts, itr, que, CPUt, job_num))
            elif qJobID:
                # Print blank job number
                print(stncl % (i, frun, sts, itr, que, CPUt, ""))
            else:
                # No job number
                print(stncl % (i, frun, sts, itr, que, CPUt))
           # --- Execution ---
            # Check for queue killing
            if qKill and (n is not None) and (jobID in jobs):
                # Delete it.
                self.StopCase(i)
                continue
            # Check for deletion
            if qDel and (not n) and (sts in ["INCOMP", "ERROR", "---"]):
                # Delete folder
                nDel += self.DeleteCase(i, **kw)
            elif qDel:
                # Delete but forcing prompt
                nDel += self.DeleteCase(i, prompt=True)
            # Check status.
            if qCheck:
                continue
            # If submitting is allowed, check the job status.
            if (sts in stat_submit) and self.FilterUser(i, **kw):
                # Prepare the job
                self.PrepareCase(i)
                # Check for "incremental" option
                self._prepare_incremental(i, q_incr)
                # Start (submit or run) case
                if q_strt:
                    self.StartCase(i)
                # Increase job number
                nSub += 1
            # Revert to original optons
            self.RevertOptions()
            # Don't continue checking if maximum submissions reached.
            if nSub >= nSubMax:
                break
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
                fline += ("%s=%i, " % (key, total[key]))
        # Print the line.
        if fline:
            print(fline)

    # Mark a case as PASS
    @run_rootdir
    def MarkPASS(self, **kw):
        r"""Mark one or more cases as **PASS** and rewrite matrix

        :Call:
            >>> cntl.MarkPASS(**kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
            *flag*: {``"p"``} | ``"P"`` | ``"PASS"`` | ``"$p"``
                Marker to use to denote status
        :Versions:
            * 2019-06-14 ``@ddalle``: v1.0
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
        r"""Mark one or more cases as **ERROR** and rewrite matrix

        :Call:
            >>> cntl.MarkERROR(**kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
            *flag*: {``"E"``} | ``"e"`` | ``"ERROR"`` | ``"$E"``
                Marker to use to denote status
        :Versions:
            * 2019-06-14 ``@ddalle``: v1.0
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
        r"""Remove **PASS** or **ERROR** marking from one or more cases

        :Call:
            >>> cntl.UnmarkCase(**kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2019-06-14 ``@ddalle``: v1.0
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
    def ExecScript(self, **kw):
        r"""Execute a script in a given case folder

        This function is the interface to command-line calls using the
        ``-e`` flag, such as ``pycart -e 'ls -lh'``.

        :Call:
            >>> ierr = cntl.ExecScript(i, cmd)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Case index (0-based)
        :Outputs:
            *ierr*: ``None`` | :class:`int`
                Exit status from the command
        :Versions:
            * 2016-08-26 ``@ddalle``: v1.0
            * 2024-12-09 ``@jfdiaz3``:v1.1
        """
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Initialize return code
        ierr = 0
        # Get execute command
        cmd = kw.get('exec', kw.get('e'))
        for i in I:
            os.chdir(self.RootDir)
            # Get the case folder name
            frun = self.x.GetFullFolderNames(i)
            # Check for the folder
            if not os.path.isdir(frun):
                return
            print(f'{i} {frun}')
            # Enter the folder
            os.chdir(frun)
            # Set current case index
            self.opts.setx_i(i)
            # Status update
            print("  Executing system command:")
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
                    ncmd = "./%s %s" % (fexec, ' '.join(cmd.split()[1:]))
                else:
                    # Just run command as it is
                    ncmd = cmd
            # Status update
            print("    %s" % ncmd)
            # Pass to dangerous system command
            ierr = os.system(ncmd)
            # Output
            if ierr:
                print("    exit(%s)" % ierr)
        return ierr
