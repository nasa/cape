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
import shutil
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from io import IOBase
from typing import Any, Callable, Optional, Union

# Third-party modules
import numpy as np

# Local imports
from . import casecntlbase
from . import databookbase
from . import queue
from . import report
from .. import convert
from .. import console
from .casecntlbase import CaseRunnerBase
from .runmatrix import RunMatrix
from .logger import CntlLogger
from .options import Options
from .options.funcopts import UserFuncOpts
from ..config import ConfigXML, ConfigJSON
from ..errors import assert_isinstance
from ..optdict import WARNMODE_WARN, WARNMODE_QUIET
from ..optdict.optitem import getel
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


# Cache of one property for each case
class CaseCache(dict):
    r"""Cache of one property for cases in a run matrix

    :Call:
        >>> cache = CaseCache(prop)
    :Inputs:
        *prop*: :class:`str`
            Name of property being cached
    :Outputs:
        *cache*: :class:`CaseCache`
            Cache of property for each case, like a :class:`dict`
    """
    # Properties
    __slots__ = (
        "prop"
    )

    # Initialization
    def __init__(self, prop: str):
        #: :class:`str`
        #: Name of property being cached
        self.prop = prop

    # Get value
    def get_value(self, i: int) -> Any:
        r"""Get a value for a case, if any

        :Call:
            >>> val = cache.get_value(i)
        :Inputs:
            *cache*: :class:`CaseCache`
                Cache of property for cases in a run matrix
            *i*: :class:`int`
                Case index
        :Outputs:
            *val*: :class:`object` | ``None``
                Value, if present
        :Versions:
            * 2025-03-01 ``@ddalle``: v1.0
        """
        # Get value if any
        return self.get(i)

    # Save value
    def save_value(self, i: int, val: Any):
        r"""Save a value for a case

        :Call:
            >>> cache.save_value(i, val)
        :Inputs:
            *cache*: :class:`CaseCache`
                Cache of property for cases in a run matrix
            *i*: :class:`int`
                Case index
            *val*: :class:`object` | ``None``
                Value, if present
        :Versions:
            * 2025-03-01 ``@ddalle``: v1.0
        """
        # Save value
        self[i] = val

    # Clear value
    def clear_case(self, i: int):
        r"""Clear cache for one case, if present

        :Call:
            >>> cache.clear_case(i)
        :Inputs:
            *cache*: :class:`CaseCache`
                Cache of property for cases in a run matrix
            *i*: :class:`int`
                Case index
        :Versions:
            * 2025-03-01 ``@ddalle``: v1.0
        """
        self.pop(i, None)

    # Check case
    def check_case(self, i: int) -> bool:
        r"""Save a value for a case

        :Call:
            >>> q = cache.check_case(i)
        :Inputs:
            *cache*: :class:`CaseCache`
                Cache of property for cases in a run matrix
            *i*: :class:`int`
                Case index
        :Outputs:
            *q*: :class:`bool`
                Whether or not case *i* is present
        :Versions:
            * 2025-03-01 ``@ddalle``: v1.0
        """
        return i in self


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
    :Attributes:
        * :attr:`cache_iter`
    """
   # --- Class Attributes ---
    # Names
    _name = "cfdx"
    _solver = "cfdx"
    # Hooks to py{x} specific modules
    _case_mod = casecntlbase
    _databook_mod = databookbase
    _report_mod = report
    # Hooks to py{x} specific classes
    _case_cls = CaseRunnerBase
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
        # Initialize logger
        self.logger = None
        # Process the trajectory.
        self.x = RunMatrix(**self.opts['RunMatrix'])
        # Save conditions w/i options
        self.opts.save_x(self.x)
        # Set initial index
        self.opts.setx_i(0)
        # Job list
        self.jobs = {}
        self.jobqueues = []
        # Run cntl init functions, customize for py{x}
        self.init_post()
        # Run any initialization functions
        self.InitFunction()
        # Initialize slots
        self.DataBook = None
        #: Cache of current iteration for each case
        self.cache_iter = CaseCache("iter")

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
        assert_isinstance(module_list, list, '"Modules" option')
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

    # Execute a function by spec
    def exec_cntlfunction(self, funcspec: Union[str, dict]) -> Any:
        r"""Execute a *Cntl* function, accessing user-specified modules

        :Call:
            >>> v = cntl.exec_cntlfunction(funcname)
            >>> v = cntl.exec_cntlfunction(funcspec)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall control interface
            *funcname*: :class:`str`
                Name of function to execute, e.g. ``"mymod.myfunc"``
            *funcspec*: :class:`dict`
                Function opts parsed by :class:`UserFuncOpts`
        :Outputs:
            *v*: **any**
                Output from execution of function
        :Versions:
            * 2025-03-28 ``@ddalle``: v1.0
        """
        # Check type
        if isinstance(funcspec, dict):
            return self.exec_cntl_function_dict(funcspec)
        elif isinstance(funcspec, str):
            return self.exec_cntlfunction_str(funcspec)
        # Otherwise bad type

    # Execute a function by name only
    def exec_cntlfunction_str(self, funcname: str) -> Any:
        r"""Execute a function from *cntl.modules*

        :Call:
            >>> v = cntl.exec_modfunction(funcname, a, kw, name=None)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall control interface
            *funcname*: :class:`str`
                Name of function to execute, e.g. ``"mymod.myfunc"``
        :Outputs:
            *v*: **any**
                Output from execution of function
        :Versions:
            * 2025-03-28 ``@ddalle``: v1.0
        """
        return self.exec_modfunction(funcname)

    # Execute a function by dict
    def exec_cntl_function_dict(self, funcspec: dict):
        r"""Execute a *Cntl* function, accessing user-specified modules

        :Call:
            >>> v = cntl.exec_cntl_function_dict(funcspec)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall control interface
            *funcspec*: :class:`dict`
                Function opts parsed by :class:`UserFuncOpts`
        :Outputs:
            *v*: **any**
                Output from execution of function
        :Versions:
            * 2025-03-28 ``@ddalle``: v1.0
        """
        # Process options
        opts = UserFuncOpts(funcspec)
        # Get name
        functype = opts.get_opt("type")
        funcname = opts.get_opt("name")
        funcrole = opts.get_opt("role", vdef=funcname)
        # Check if present
        if funcname is None:
            raise ValueError(f"User-defined function has no name:\n{funcspec}")
        # Get argument names
        argnames = opts.get_opt("args", vdef=[])
        kwargdict = opts.get_opt("kwargs", vdef={})
        verbose = opts.get_opt("verbose", vdef=False)
        # Expand args
        a = [self._expand_funcarg(aj) for aj in argnames]
        kw = {k: self._expand_funcarg(v) for k, v in kwargdict.items()}
        # STDOUT tag
        name = funcrole if verbose else None
        # Execute
        return self._exec_pyfunc(functype, funcname, a, kw, name=name)

    def _expand_funcarg(self, argval: Union[Any, str]) -> Any:
        r"""Expand a function value

        :Call:
            >>> v = cntl._expand_funcarg(argval)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall control interface
        :Outputs:
            *v*: :class:`str` | :class:`float` | :class:`int`
                Expanded value, usually float or string
        :Versions:
            * 2025-03-28 ``@ddalle``: v1.0
        """
        # Check if string
        if not isinstance(argval, str):
            return argval
        # Check for $
        if not argval.startswith("$"):
            # Raw string
            return argval
        # Get current case index
        i = self.opts.i
        # Get argument name
        argname = argval.lstrip("$")
        # Check pre-defined values
        if argname == "cntl":
            return self
        elif argname == "i":
            return i
        elif argname == "runner":
            return self.ReadCaseRunner(i)
        elif argname in self.x.cols:
            return self.x[argname][i]
        else:
            return self.x.GetValue(argname, i)

    # Execute a function
    def exec_modfunction(
            self,
            funcname: str,
            a: Optional[Union[tuple, list]] = None,
            kw: Optional[dict] = None,
            name: Optional[str] = None) -> Any:
        r"""Execute a function from *cntl.modules*

        :Call:
            >>> v = cntl.exec_modfunction(funcname, a, kw, name=None)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall control interface
            *funcname*: :class:`str`
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
            * 2025-03-28 ``@ddalle``: v1.1; improve error messages
        """
        return self._exec_pyfunc("module", funcname, a, kw, name)

    # Execute a function
    def _exec_pyfunc(
            self,
            functype: str,
            funcname: str,
            a: Optional[Union[tuple, list]] = None,
            kw: Optional[dict] = None,
            name: Optional[str] = None) -> Any:
        r"""Execute a function from *cntl.modules*

        :Call:
            >>> v = cntl._exec_pyfunc(functype, funcname, a, kw, name)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall control interface
            *functype*: {``"module"``} | ``"cntl"`` | ``"runner"``
                Module source, general *cntl*, or *cntl.caserunner*
            *funcname*: :class:`str`
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
            * 2025-03-28 ``@ddalle``: v1.1; improve error messages
        """
        # Default args and kwargs
        a = tuple() if a is None else a
        a = a if isinstance(a, (tuple, list)) else (a,)
        kw = kw if isinstance(kw, dict) else {}
        # Check function type
        if functype == "cntl":
            # Get instance method from here
            func = getattr(self, funcname)
        elif functype == "runner":
            # Get case runner
            i = self.opts.i
            # Read Caserunner
            runner = self.ReadCaseRunner(i)
            # Get method from there
            func = getattr(runner, funcname)
        else:
            # Split name into module(s) and function name
            funcparts = funcname.rsplit(".", 1)
            # Has to be at least two parts
            if len(funcparts) < 2:
                raise ValueError(
                    f"User-defined function '{funcname}' has no module name; "
                    "must contain at least one '.'")
            # Get module name and function name
            modname, funcname = funcparts
            # Import module
            mod = self.import_module(modname)
            # Get function
            func = mod.__dict__.get(funcname)
        # Check if found
        if func is None:
            raise NameError(f"Name '{funcname}' is not defined")
        # Check if final spec is callable
        if not callable(func):
            raise TypeError(f"Name '{funcname}' is not callable")
        # Status update if appropriate
        if name:
            print("  %s: %s()" % (name, funcname))
        # Call function
        return func(*a, **kw)

    def import_module(self, modname: str):
        r"""Import a module by name, if possible

        :Call:
            >>> mod = cntl.import_module(modname)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall control interface
            *modname*: :class:`str`
                Name of module to import
        :Outputs:
            *mod*: **module**
                Python module
        :Versions:
            * 2025-03-28 ``@ddalle``: v1.0
        """
        # Get dict of module names
        modnamedict = self.opts.get_opt("ModuleNames", vdef={})
        # Get alias, if any
        fullmodname = modnamedict.get(modname, modname)
        # Check if module already imported
        if fullmodname not in sys.modules:
            # Status update
            print(f"Importing module '{modname}'")
        # Try to import module
        return importlib.import_module(fullmodname)

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
        assert_isinstance(funclist, list, "list of functions")
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
            * 2025-03-26 ``@ddalle``: v1.1; Windows compatibility fix
        """
        # Replace '/' -> '\' on Windows
        fname_sys = fname.replace('/', os.sep)
        # Check if absolute
        if os.path.isabs(fname_sys):
            # Already absolute
            return fname_sys
        else:
            # Relative to *RootDir*
            return os.path.join(self.RootDir, fname_sys)

    # Copy files
    @run_rootdir
    def copy_files(self, i: int):
        r"""Copy specified files to case *i* run folder

        :Call:
            >>> cntl.copy_files(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE main control instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2025-03-26 ``@ddalle``: v1.0
        """
        # Get list of files to copy
        files = self.opts.get_CopyFiles()
        # Check for any
        if files is None or len(files) == 0:
            return
        # Ensure case index is set
        self.opts.setx_i(i)
        # Create case folder
        self.make_case_folder(i)
        # Name of case folder
        frun = self.x.GetFullFolderNames(i)
        # Loop through files
        for fname in files:
            # Absolutize
            fabs = self.abspath(fname)
            # Get base file name
            fbase = os.path.basename(fabs)
            # Destination file
            fdest = os.path.join(self.RootDir, frun, fbase)
            # Check for overwrite
            if os.path.isfile(fdest):
                raise FileExistsError(f"  Cannot copy '{fname}'; file exists")
            # Copy file
            shutil.copy(fabs, fdest)

    # Link files
    @run_rootdir
    def link_files(self, i: int):
        r"""Link specified files to case *i* run folder

        :Call:
            >>> cntl.link_files(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE main control instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2025-03-26 ``@ddalle``: v1.0
        """
        # Get list of files to copy
        files = self.opts.get_LinkFiles()
        # Check for any
        if files is None or len(files) == 0:
            return
        # Ensure case index is set
        self.opts.setx_i(i)
        # Create case folder
        self.make_case_folder(i)
        # Name of case folder
        frun = self.x.GetFullFolderNames(i)
        # Loop through files
        for fname in files:
            # Absolutize
            fabs = self.abspath(fname)
            # Get base file name
            fbase = os.path.basename(fabs)
            # Destination file
            fdest = os.path.join(self.RootDir, frun, fbase)
            # Check for overwrite
            if os.path.isfile(fdest):
                raise FileExistsError(f"  Cannot copy '{fname}'; file exists")
            # Copy file
            os.symlink(fabs, fdest)

    @run_rootdir
    def PrepareMesh(self, i: int):
        r"""Prepare the mesh for case *i* if necessary

        :Call:
            >>> cntl.PrepareMesh(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class
            *i*: :class:`int`
                Case index
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0 (pyfun)
            * 2024-11-04 ``@ddalle``: v1.3 (pyfun)
            * 2024-11-07 ``@ddalle``: v1.0
        """
        # Ensure case index is set
        self.opts.setx_i(i)
        # Create case folder
        self.make_case_folder(i)
        # Copy/link generic files
        self.copy_files(i)
        self.link_files(i)
        # Prepare warmstart files, if any
        warmstart = self.PrepareMeshWarmStart(i)
        # Finish if case was warm-started
        if warmstart:
            return
        # Copy main files
        self.PrepareMeshFiles(i)
        # Prepare surface triangulation for AFLR3 if appropriate
        self.PrepareMeshTri(i)

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

    # Loop through cases
    def caseloop_verbose(self, casefunc: Callable, **kw):
        # Get list of indices
        inds = self.x.GetIndices(**kw)
        # Get indent
        indent = kw.get("indent", 0)
        tab = ' ' * indent
        # Default list of columns to display
        displaycols = [
            "i",
            "frun",
            "status",
            "progress",
            "queue",
            "time",
        ]
        # Default list of columns to count
        countercols = [
            "status",
        ]
        # Loop through cases
        for i in inds:
            # Get name of case
            frun = self.x.GetFullFolderNames(i)

    # Loop through cases
    def caseloop(self, casefunc: Callable, **kw):
        r"""Loop through cases and execute function for each case

        :Call:
            >>> cntl.caseloop(casefun, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                CAPE run matrix control instance
            *indent*: {``0``} | :class:`int`
                Number of spaces to indent each case name
        :Versions:
            * 2025-02-12 ``@ddalle``: v1.0
        """
        # Get list of indices
        inds = self.x.GetIndices(**kw)
        # Get indent
        indent = kw.get("indent", 0)
        tab = ' ' * indent
        # Loop through cases
        for i in inds:
            # Get case name
            frun = self.x.GetFullFolderNames(i)
            # Display it
            print(f"{tab}{frun}")
            # Run the function
            casefunc(i)

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
            nRunning = self.CountQueuedCases(u=kw.get('u'))
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
            'RUNNING': 0,
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
            # Check status
            sts = self.check_case_status(i)
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
                # Check the queue
                que = self.check_case_job(i)
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

   # --- Run Interface ---
    # Apply user filter
    def FilterUser(self, i: int, **kw) -> bool:
        r"""Determine if case *i* is assigned to current user

        :Call:
            >>> q = cntl.FilterUser(i, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Index of the case to check (0-based)
            *u*, *user*: :class:`str`
                User name (default: executing process's username) for
                comparing to run matrix
        :Outputs:
            *q*: :class:`bool`
                Whether user is owner of case *i*
        :Versions:
            2017-07-10 ``@ddalle``: v1.0
        """
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
        uid = kw.get('u', kw.get('user'))
        # Default
        if uid is None:
            uid = getpass.getuser()
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

    # Get case runner from a folder
    @abstractmethod
    def ReadFolderCaseRunner(self, fdir: str) -> CaseRunnerBase:
        r"""Read a ``CaseRunner`` from a folder by name

        :Call:
            >>> runner = cntl.ReadFolderCaseRunner(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        """
        pass

    # Read case runner
    @abstractmethod
    def ReadCaseRunner(self, i: int) -> CaseRunnerBase:
        r"""Read CaseRunner into slot

        :Call:
            >>> runner = cntl.ReadCaseRunner(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
        """
        pass

    # Function to start a case: submit or run
    @abstractmethod
    def StartCase(self, i: int):
        r"""Start a case by either submitting it or running it

        This function checks whether or not a case is submittable.  If
        so, the case is submitted via :func:`cape.cfdx.queue.pqsub`,
        and otherwise the case is started using a system call.

        Before starting case, this function checks the folder using
        :func:`cape.cfdx.cntl.CheckCase`; if this function returns ``None``,
        the case is not started.  Actual starting of the case is done
        using :func:`CaseStartCase`, which has a specific version for
        each CFD solver.

        :Call:
            >>> pbs = cntl.StartCase(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Outputs:
            *pbs*: :class:`int` | ``None``
                PBS job ID if submitted successfully
        """
        pass

    # Function to terminate a case: qdel and remove RUNNING file
    @abstractmethod
    def StopCase(self, i: int):
        r"""Stop a case if running

        This function deletes a case's PBS job and removes the
        ``RUNNING`` file if it exists.

        :Call:
            >>> cntl.StopCase(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Cape control interface
            *i*: :class:`int`
                Index of the case to check (0-based)
        """
        pass

    # Get expected actual breaks of phase iters
    def GetPhaseBreaks(self) -> list:
        r"""Get expected iteration numbers at phase breaks

        This fills in ``0`` entries in *RunControl* |>| *PhaseIters* and
        returns the filled-out list.

        :Call:
            >>> PI = cntl.GetPhaseBreaks()
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Cape control interface
        :Outputs:
            *PI*: :class:`list`\ [:class:`int`]
                Min iteration counts for each phase
        :Versions:
            * 2017-04-12 ``@ddalle``: v1.0
        """
        # Get list of phases to use
        PhaseSeq = self.opts.get_PhaseSequence()
        PhaseSeq = list(np.array(PhaseSeq).flatten())
        # Get option values for *PhaseIters* and *nIter*
        PI = [self.opts.get_PhaseIters(j) for j in PhaseSeq]
        NI = [self.opts.get_nIter(j) for j in PhaseSeq]
        # Initialize total
        ni = 0
        # Loop through phases
        for j, (phj, nj) in enumerate(zip(PI, NI)):
            # Check for specified cutoff
            if phj:
                # Use max of defined cutoff and *nj1*
                mj = max(phj, ni + nj)
            else:
                # Min value for next phase: last total + *nj*
                mj = ni + nj
            # Save new cutoff
            PI[j] = mj
        # Output
        return PI

    # Get last iter
    def GetLastIter(self, i: int) -> int:
        r"""Get minimum required iteration for a given case

        :Call:
            >>> nIter = cntl.GetLastIter(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Cape control interface
            *i*: :class:`int`
                Run index
        :Outputs:
            *nIter*: :class:`int`
                Number of iterations required for case *i*
        :Versions:
            * 2014-10-03 ``@ddalle``: v1.0
        """
        # Read the local case.json file.
        rc = self.read_case_json(i)
        # Check for null file
        if rc is None:
            return self.opts.get_PhaseIters(-1)
        # Option for desired iterations
        N = rc.get('PhaseIters', 0)
        # Output the last entry (if list)
        return getel(N, -1)

    # Get case index
    def GetCaseIndex(self, frun: str) -> Optional[int]:
        r"""Get index of a case in the current run matrix

        :Call:
            >>> i = cntl.GetCaseIndex(frun)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Cape control interface
            *frun*: :class:`str`
                Name of case, must match exactly
        :Outputs:
            *i*: :class:`int` | ``None``
                Index of case with name *frun* in run matrix, if present
        :Versions:
            * 2024-08-15 ``@ddalle``: v1.0
            * 2024-10-16 ``@ddalle``: v1.1; move to :class:`RunMatrix`
        """
        return self.x.GetCaseIndex(frun)

    # Count R/Q cases based on full status
    def CountRunningCases(self, I, jobs=None, u=None):
        r"""Count number of running cases via the batch system

        Also print a status of the running jobs.

        :Call:
            >>> n = cntl.CountRunningCases(I, jobs=None, u=None)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *jobs*: :class:`dict`
                Information on each job by ID number
            *u*: :class:`str`
                User name (defaults to process username)
        :Outputs:
            *n*: :class:`int`
                Number of running or queued jobs
        :Versions:
            * 2023-12-08 ``@dvicker``: v1.0
        """
        # Status update
        print("Checking for currently queued jobs")
        # Initialize counter
        total_running = 0
        # Loop through cases
        for i in I:
            # Check status of that case
            sts = self.CheckCaseStatus(i, jobs, u)
            # Add to counter if running
            running = 1 if (sts in ("RUN", "QUEUE")) else 0
            total_running += running
        # Return the counter
        return total_running

    # Count R/Q cases based on PBS/Slurm only
    def CountQueuedCases(
            self,
            I: Optional[list] = None,
            u: Optional[str] = None, **kw) -> int:
        r"""Count cases that have currently active PBS/Slurm jobs

        :Call:
            >>> n = cntl.CountQueuedCases(I=None, u=None, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *I*: {``None``} | :class:`list`\ [:class:`int`]
                List of indices
            *u*: :class:`str`
                User name (defaults to process username)
            *kw*: :class:`dict`
                Other kwargs used to subset the run matrix
        :Outputs:
            *n*: :class:`int`
                Number of running or queued jobs (not counting the job
                from which function is called, if applicable)
        :Versions:
            * 2024-01-12 ``@ddalle``: v1.0
            * 2024-01-17 ``@ddalle``: v1.1; check for *this_job*
            * 2025-05-01 ``@ddalle``: v1.2; remove *jobs* kwarg
        """
        # Status update
        print("Checking for currently queued jobs")
        # Check for ID of "this job" if called from a running job
        this_job = self.CheckBatch()
        # Initialize counter
        total_running = 0
        # Get full set of cases
        I = self.x.GetIndices(I=I, **kw)
        # Process jobs list
        jobs = self.get_pbs_jobs(u=u)
        # Loop through cases
        for i in I:
            # Get the JobID for that case
            jobid = self.GetPBSJobID(i)
            # Check if it's in the queue right now
            if (jobid in jobs) and (jobid != this_job):
                total_running += 1
        # Output
        return total_running

    # Function to determine if case is PASS, ---, INCOMP, etc.
    def CheckCaseStatus(
            self, i: int,
            jobs: Optional[dict] = None,
            auto: bool = False,
            u: Optional[str] = None,
            qstat: bool = True):
        r"""Determine the current status of a case

        :Call:
            >>> sts = cntl.CheckCaseStatus(i, jobs=None, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Index of the case to check (0-based)
            *jobs*: :class:`dict`
                Information on each job by ID number
            *u*: :class:`str`
                User name (defaults to process username)
            *qstat*: {``True``} | ``False``
                Option to call qstat/squeue to get job status
        :Versions:
            * 2014-10-04 ``@ddalle``: v1.0
            * 2014-10-06 ``@ddalle``: v1.1; check queue status
            * 2023-12-13 ``@dvicker``: v1.2; check for THIS_JOB
            * 2024-08-22 ``@ddalle``: v1.3; add *qstat*
        """
        # Current iteration count
        n = self.CheckCase(i)
        # Try to get a job ID.
        jobID = self.GetPBSJobID(i)
        # Get list of jobs
        jobs = self.get_pbs_jobs(u=u, qstat=qstat)
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

        # Get current job ID, if any
        current_jobid = self.CheckBatch()
        # Check current job ID against the one in this case folder
        if current_jobid == jobID:
            sts = "THIS_JOB"
        # Output
        return sts

    # Get information on all jobs from current user
    def get_pbs_jobs(
            self,
            force: bool = False,
            u: Optional[str] = None,
            server: Optional[str] = None,
            qstat: bool = True):
        r"""Get dictionary of current jobs active by one user

        :Call:
            >>> jobs = cntl.get_pbs_jobs(force=False, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *force*: ``True`` | {``False``}
                Query current queue even if *cntl.jobs* exists
            *u*: {``None``} | :class:`str`
                User name (defaults to process username)
            *server*: {``None``} | :class:`str`
                Name of non-default PBS/Slurm server
        :Outputs:
            *jobs*::class:`dict`
                Information on each job by ID number
        :Versions:
            * 2024-01-12 ``@ddalle``: v1.0
            * 2024-08-22 ``@ddalle``: v1.1; add *qstat* option
            * 2025-05-01 ``@ddalle``: v1.2; simplify flow
        """
        # Identifier for this user and queue
        qid = (server, u)
        # Check for existing jobs
        if (not force) and (qid in self.jobqueues):
            return self.jobs
        # Get current jobs
        jobs = self.jobs
        jobs = {} if jobs is None else jobs
        # Check for ``--no-qstat`` flag
        if not qstat:
            return jobs
        # Get list of jobs currently running for user *u*
        if self.opts.get_slurm(0):
            # Call slurm instead of PBS
            newjobs = queue.squeue(u=u)
        else:
            # Use qstat to get job info
            newjobs = queue.qstat(u=u, server=server)
        # Add to full list
        self.jobs.update(newjobs)
        # Note this status
        if qid not in self.jobqueues:
            self.jobqueues.append(qid)
        # Output
        return self.jobs

    def _get_qstat(self) -> queue.QStat:
        # Check current attribute
        if isinstance(self.jobs, queue.QStat):
            return self.jobs
        # Create one
        self.jobs = queue.QStat()
        # Set Slurm vs PBS
        rc = self.read_case_json()
        sched = "slurm" if rc.get_slurm(0) else "pbs"
        self.jobs.scheduler = sched
        # Output
        return self.jobs

    # Check a case
    @run_rootdir
    def CheckCase(
            self,
            i: int,
            force: bool = False,
            v: bool = False) -> Optional[int]:
        r"""Check current status of case *i*

        Because the file structure is different for each solver, some
        of this method may need customization.  This customization,
        however, can be kept to the functions
        :func:`cape.cfdx.casecntl.GetCurrentIter` and
        :func:`cape.cfdx.cntl.Cntl.CheckNone`.

        :Call:
            >>> n = cntl.CheckCase(i, v=False)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Index of the case to check (0-based)
            *v*: ``True`` | {``False``}
                Verbose flag; prints messages if *n* is ``None``
        :Outputs:
            *n*: :class:`int` | ``None``
                Number of completed iterations or ``None`` if not set up
        :Versions:
            * 2014-09-27 ``@ddalle``: v1.0
            * 2015-09-27 ``@ddalle``: v2.0; generic
            * 2015-10-14 ``@ddalle``: v2.1; no :mod:`case` req
            * 2017-02-22 ``@ddalle``: v2.2; add verbose flag
            * 2023-11-06 ``@ddalle``: v2.3; call ``setx_i(i)``
        """
        # Check input
        assert_isinstance(i, (int, np.integer), "case index")
        # Set options
        self.opts.setx_i(i)
        # Get the group name.
        frun = self.x.GetFullFolderNames(i)
        # Initialize iteration number.
        n = 0
        # Check if the folder exists.
        if v and (not os.path.isdir(frun)):
            # Verbosity option
            if v:
                print("    Folder '%s' does not exist" % frun)
        # Get case status
        n = self.GetCurrentIter(i, force)
        # If zero, check if the required files are set up
        if n == 0:
            # Get name of folder
            frun = self.x.GetFullFolderNames(i)
            # Check for case
            if os.path.isdir(frun):
                # Enter folder
                os.chdir(frun)
                # Check if prepared
                n = None if self.CheckNone(v) else 0
            else:
                # No folder
                n = None
        # Output
        return n

    # Get the current iteration number from :mod:`case`
    def GetCurrentIter(self, i: int, force: bool = False) -> int:
        r"""Get the current iteration number (using :mod:`case`)

        This function utilizes the :mod:`cape.cfdx.case` module, and so
        it must be copied to the definition for each solver's control
        class.

        :Call:
            >>> n = cntl.GetCurrentIter(i, force=False)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Index of the case to check (0-based)
            *force*: ``True`` | {``False``}
                Option to ignore cache
        :Outputs:
            *n*: :class:`int`
                Number of completed iterations
        :Versions:
            * 2015-10-14 ``@ddalle``: v1.0
            * 2023-07-07 ``@ddalle``: v2.0; use ``CaseRunner``
            * 2025-03-01 ``@ddalle``: v3.0; add caching
        """
        # Check if case is cached
        if (not force) and self.cache_iter.check_case(i):
            # Return cached value
            return self.cache_iter.get_value(i)
        # Instantiate case runner
        runner = self.ReadCaseRunner(i)
        # Check for non-existint case
        if runner is None:
            # No case to check
            n = None
        else:
            # Get iteration
            n = runner.get_iter()
        # Default to 0
        n = 0 if n is None else n
        # Save it
        self.cache_iter.save_value(i, n)
        # Return it
        return n

    # Check a case's phase output files
    @run_rootdir
    def CheckUsedPhase(self, i: int, v=False):
        r"""Check maximum phase number run at least once

        :Call:
            >>> j, n = cntl.CheckUsedPhase(i, v=False)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
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
            * 2017-06-29 ``@ddalle``: v1.0
            * 2017-07-11 ``@ddalle``: v1.1; verbosity option
            * 2025-03-02 ``@ddalle``: v2.0; use CaseRunner
        """
        # Check input
        assert_isinstance(i, (int, np.integer), "case index")
        # Read case runner
        runner = self.ReadCaseRunner(i)
        # Check if found
        if runner is None:
            # Verbosity option
            if v:
                # Get the group name
                frun = self.x.GetFullFolderNames(i)
                # Show it
                print("    Folder '%s' does not exist" % frun)
            # Phase
            j = 0
            jlast = self.opts.get_PhaseSequence()[-1]
            return j, jlast
        # Use runner's call
        return runner.get_phase_simple()

    # Check a case's phase number
    @run_rootdir
    def CheckPhase(self, i: int, v: bool = False) -> Optional[int]:
        r"""Check current phase number of run *i*

        :Call:
            >>> n = cntl.CheckPhase(i, v=False)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Index of the case to check (0-based)
            *v*: ``True`` | {``False``}
                Verbose flag; prints messages if *n* is ``None``
        :Outputs:
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2017-06-29 ``@ddalle``: v1.0
        """
        # Check input
        assert_isinstance(i, (int, np.integer), "case index")
        # Get the group name.
        frun = self.x.GetFullFolderNames(i)
        # Initialize iteration number.
        n = 0
        # Check if the folder exists.
        if (not os.path.isdir(frun)):
            # Verbosity option
            if v:
                print("    Folder '%s' does not exist" % frun)
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
        r"""Get the current phase number from the appropriate module

        This function utilizes the :mod:`cape.cfdx.case` module, and so
        it must be copied to the definition for each solver's control
        class.

        :Call:
            >>> j = cntl.CaseGetCurrentPhase()
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
        :Outputs:
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2017-06-29 ``@ddalle``: v1.0
            * 2023-07-06 ``@ddalle``: v1.1; use ``CaseRunner``
        """
        # Be safe
        try:
            # Instatiate case runner
            runner = self.ReadCaseRunner()
            # Get the phase number
            return runner.get_phase()
        except Exception:
            return 0

    # Check if cases with zero iterations are not yet setup to run
    def CheckNone(self, v=False):
        r"""Check if the current directory has the needed files to run

        This function needs to be customized for each CFD solver so that
        it checks for the appropriate files.

        :Call:
            >>> q = cntl.CheckNone(v=False)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Cape control interface
            *v*: ``True`` | {``False``}
                Verbose flag; prints message if *q* is ``True``
        :Outputs:
            *q*: ```True`` | `False``
                Whether or not case is missing files
        :Versions:
            * 2015-09-27 ``@ddalle``: v1.0
            * 2017-02-22 ``@ddalle``: v1.1, verbosity option
        """
        return False

    # Check if a case is running
    @abstractmethod
    def CheckRunning(self, i: int) -> bool:
        r"""Check if a case is currently running

        :Call:
            >>> q = cntl.CheckRunning(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Run index
        :Outputs:
            *q*: :class:`bool`
                If ``True``, case has :file:`RUNNING` file in it
        """
        pass

    # Check for a failure
    @abstractmethod
    def CheckError(self, i: int) -> bool:
        r"""Check if a case has a failure

        :Call:
            >>> q = cntl.CheckError(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Run index
        :Outputs:
            *q*: :class:`bool`
                If ``True``, case has ``FAIL`` file in it
        """
        pass

    # Check for no unchanged files
    @run_rootdir
    def CheckZombie(self, i):
        r"""Check a case for ``ZOMBIE`` status

        A running case is declared a zombie if none of the listed files
        (by default ``*.out``) have been modified in the last 30
        minutes.  However, a case cannot be a zombie unless it contains
        a ``RUNNING`` file and returns ``True`` from
        :func:`CheckRunning`.

        :Call:
            >>> q = cntl.CheckZombie(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Run index
        :Outputs:
            *q*: :class:`bool`
                ``True`` if no listed files have been modified recently
        :Versions:
            * 2017-04-04 ``@ddalle``: v1.0
            * 2021-01-25 ``@ddalle``: v1.1; use cls._zombie_files
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
        fzomb = self.opts.get("ZombieFiles", self.__class__._zombie_files)
        # Ensure list
        if not isinstance(fzomb, (list, tuple)):
            # Singleton glob, probably
            fzomb = [fzomb]
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

    # Check for if we are running inside a batch job
    def CheckBatch(self) -> int:
        r"""Check to see if we are running inside a batch job

        This looks for environment variables to see if this is running
        inside a batch job.  Currently supports slurm and PBS.

        :Call:
            >>> q = cntl.CheckBatch()
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
        :Outputs:
            *jobid*: :class:`int`
                ``0`` if no batch environment was detected
        :Versions:
            * 2023-12-13 ``@dvicker``: v1.0
            * 2023-12-18 ``@ddalle``: v1.1; debug
        """
        # Check which job manager
        if self.opts.get_slurm(0):
            # Slurm
            envid = os.environ.get('SLURM_JOB_ID', '0')
        else:
            # Slurm
            envid = os.environ.get('PBS_JOBID', '0')
        # Convert to integer
        return int(envid.split(".", 1)[0])

   # --- Case Modification ---
    # Function to clear out zombies
    def Dezombie(self, **kw):
        r"""Clean up any **ZOMBIE** cases

        :Call:
            >>> cntl.Dezombie(**kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Instance of overall control interface
            *extend*: {``True``} | positive :class:`int`
                Extend phase *j* by *extend* nominal runs
            *j*: {``None``} | :class:`int` >= 0
                Phase number
            *imax*: {``None``} | :class:`int`
                Do not increase iteration number beyond *imax*
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints
            *I*: :class:`list`\ [:class:`int`]
                List of indices
        :Versions:
            * 2021-10-14 ``@ddalle``: v1.0
        """
        # Zombie counter
        nzombie = 0
        # Cases
        I = self.x.GetIndices(**kw)
        # Largest size
        nlog = int(np.ceil(np.log10(max(1, np.max(I)))))
        # Print format
        fmt = "%%%ii %%s" % nlog
        # Loop through folders
        for i in I:
            # Get status
            sts = self.CheckCaseStatus(i)
            # Move to next case if not zombie
            if sts != "ZOMBIE":
                continue
            # Status update
            print(fmt % (i, self.x.GetFullFolderNames(i)))
            # qdel any cases
            self.StopCase(i)
            # Counter
            nzombie += 1
        # Final status
        print("Cleared up %i ZOMBIEs" % nzombie)

    # Function to extend one or more cases
    def ExtendCases(self, **kw):
        r"""Extend one or more case by a number of iterations

        By default, this applies to the final phase, but the phase
        number *j* can also be specified as input. The number of
        additional iterations is generally the nominal number of
        iterations that phase *j* would normally run.

        :Call:
            >>> cntl.ExtendCases(cons=[], extend=1, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Instance of overall control interface
            *extend*: {``True``} | positive :class:`int`
                Extend phase *j* by *extend* nominal runs
            *imax*: {``None``} | :class:`int`
                Do not increase iteration number beyond *imax*
            *j*, *phase*: {``None``} | :class:`int`
                Optional index of phase to extend
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints
            *I*: :class:`list`\ [:class:`int`]
                List of indices
        :Versions:
            * 2016-12-12 ``@ddalle``: v1.0
        """
        # Process inputs
        n = kw.get('extend', 1)
        j = kw.get("phase", kw.get("j", None))
        imax = kw.get('imax')
        # Convert inputs to integers
        if n:
            n = int(n)
        if imax:
            imax = int(imax)
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
                if sts not in ['---', 'INCOMP']:
                    continue
                # Try to start the case
                pbs = self.StartCase(i)
                # Check for a submission
                if pbs:
                    jsub += 1
                # Check submission limit
                if jsub >= nsub:
                    return

    # Extend a case
    @abstractmethod
    def ExtendCase(
            self,
            i: int,
            n: int = 1,
            j: Optional[int] = None,
            imax: Optional[int] = None):
        r"""Add iterations to case *i* by repeating the last phase

        :Call:
            >>> cntl.ExtendCase(i, n=1, j=None, imax=None)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
            *i*: :class:`int`
                Run index
            *n*: {``1``} | positive :class:`int`
                Add *n* times *steps* to the total iteration count
            *j*: {``None``} | :class:`int`
                Optional phase to extend
            *imax*: {``None``} | nonnegative :class:`int`
                Use *imax* as the maximum iteration count
        """
        pass

    # Function to extend one or more cases
    def ApplyCases(self, **kw):
        r"""Reapply settings to one or more cases

        :Call:
            >>> cntl.ApplyCases(cons=[], j=None, extend=1, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Instance of overall control interface
            *extend*: {``True``} | positive :class:`int`
                Extend phase *j* by *extend* nominal runs
            *j*: {``None``} | nonnegative :class:`int`
                Phase number
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints
            *I*: :class:`list`\ [:class:`int`]
                List of indices
        :Versions:
            * 2016-12-12 ``@ddalle``: v1.0
        """
        # Process inputs
        n = kw.get('apply', True)
        # Handle raw ``-apply`` inputs vs. ``--apply $n``
        if n is True:
            # Use ``None`` to inherit phase count from *cntl*
            n = None
        else:
            # Convert input string to integer
            n = int(n)
        # Restart inputs
        qsub = kw.get("restart", kw.get("qsub", False))
        nsub = kw.get("n", 150)
        jsub = 0
        # Save current copy of options
        self.SaveOptions()
        # Loop through folders
        for i in self.x.GetIndices(**kw):
            # Status update
            print(self.x.GetFullFolderNames(i))
            # Clear cache
            self.cache_iter.clear_case(i)
            # Extend case
            self.ApplyCase(i, nPhase=n)
            # Start/submit the case?
            if qsub:
                # Check status
                sts = self.CheckCaseStatus(i)
                # Check if it's a submittable/restartable status
                if sts not in ['---', 'INCOMP']:
                    continue
                # Try to start the case
                pbs = self.StartCase(i)
                # Check for a submission
                if pbs:
                    jsub += 1
                # Check submission limit
                if jsub >= nsub:
                    return
            # Revert options
            self.RevertOptions()

    # Function to delete a case folder: qdel and rm
    @run_rootdir
    def DeleteCase(self, i, **kw):
        r"""Delete a case

        This function deletes a case's PBS job and removes the entire
        directory.  By default, the method prompts for user's
        confirmation before deleting; set *prompt* to ``False`` to delete
        without prompt.

        :Call:
            >>> n = cntl.DeleteCase(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Cape control interface
            *i*: :class:`int`
                Index of the case to check (0-based)
            *prompt*: {``True``} | ``False``
                Whether or not to prompt user before deleting case
        :Outputs:
            *n*: ``0`` | ``1``
                Number of folders deleted
        :Versions:
            * 2018-11-20 ``@ddalle``: v1.0
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

   # --- Archiving ---
    # Function to archive results and remove files
    def ArchiveCases(self, **kw):
        r"""Archive completed cases and clean them up if specified

        :Call:
            >>> cntl.ArchiveCases()
            >>> cntl.ArchiveCases(cons=[], **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Instance of overall control interface
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints
            *I*: :class:`list`\ [:class:`int`]
                List of indices
        :Versions:
            * 2016-12-09 ``@ddalle``: v1.0
            * 2024-09-19 ``@ddalle``: v2.0
        """
        # Get test option
        test = kw.get("test", False)
        # Loop through the folders
        for i in self.x.GetIndices(**kw):
            # Get folder name
            frun = self.x.GetFullFolderNames(i)
            fabs = os.path.join(self.RootDir, frun)
            # Check if the case is ready to archive
            if not os.path.isdir(fabs):
                continue
            # Status update
            print(frun)
            # Run action
            self.CleanCase(i, test)
            # Check status
            if self.CheckCaseStatus(i) not in ('PASS', 'ERROR'):
                print("  Case is not marked PASS | FAIL")
                continue
            # Archive
            self.ArchiveCase(i, test)

    # Run ``--archive`` on one case
    @abstractmethod
    def ArchiveCase(self, i: int, test: bool = False):
        r"""Perform ``--archive`` archiving on one case

        There are no restrictions on the status of the case for this
        action.

        :Call:
            >>> cntl.CleanCase(i, test=False)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Instance of control interface
            *i*: :class:`int`
                Case index
            *test*: ``True`` | {``False``}
                Log file/folder actions but don't actually delete/copy
        :Versions:
            * 2024-09-18 ``@ddalle``: v1.0
        """
        pass

    # Function to archive results and remove files
    @run_rootdir
    def SkeletonCases(self, **kw):
        r"""Archive completed cases and delete all but a few files

        :Call:
            >>> cntl.SkeletonCases()
            >>> cntl.SkeletonCases(cons=[], **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Instance of overall control interface
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints
            *I*: :class:`list`\ [:class:`int`]
                List of indices
        :Versions:
            * 2016-12-14 ``@ddalle``: v1.0
            * 2024-09-19 ``@ddalle``: v2.0
        """
        # Get test option
        test = kw.get("test", False)
        # Loop through the folders
        for i in self.x.GetIndices(**kw):
            # Get folder name
            frun = self.x.GetFullFolderNames(i)
            fabs = os.path.join(self.RootDir, frun)
            # Check if the case is ready to archive
            if not os.path.isdir(fabs):
                continue
            # Status update
            print(frun)
            # Run action
            self.CleanCase(i, test)
            # Check status
            if self.CheckCaseStatus(i) not in ('PASS', 'ERROR'):
                print("  Case is not marked PASS | FAIL")
                continue
            # Archive and skeleton
            self.ArchiveCase(i, test)
            self.SkeletonCase(i, test)

    # Run ``--skeleton`` on one case
    @abstractmethod
    def SkeletonCase(self, i: int, test: bool = False):
        r"""Perform ``--skeleton`` archiving on one case

        There are no restrictions on the status of the case for this
        action.

        :Call:
            >>> cntl.SkeletonCase(i, test=False)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Instance of control interface
            *i*: :class:`int`
                Case index
            *test*: ``True`` | {``False``}
                Log file/folder actions but don't actually delete/copy
        :Versions:
            * 2024-09-18 ``@ddalle``: v1.0
        """
        pass

    # Clean a set of cases
    def CleanCases(self, **kw):
        r"""Clean a list of cases using *Progress* archive options only

        :Call:
            >>> cntl.CleanCases(**kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Instance of control interface
        :Versions:
            * 2017-03-13 ``@ddalle``: v1.0
            * 2024-09-18 ``@ddalle``: v2.0
        """
        # Test status
        test = kw.get("test", False)
        # Loop through the folders
        for i in self.x.GetIndices(**kw):
            # Get folder name
            frun = self.x.GetFullFolderNames(i)
            fabs = os.path.join(self.RootDir, frun)
            # Check if the case is ready to archive
            if not os.path.isdir(fabs):
                continue
            # Status update
            print(frun)
            # Run action
            self.CleanCase(i, test)

    # Run ``--clean`` on one case
    @abstractmethod
    def CleanCase(self, i: int, test: bool = False):
        r"""Perform ``--clean`` archiving on one case

        There are no restrictions on the status of the case for this
        action.

        :Call:
            >>> cntl.CleanCase(i, test=False)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Instance of control interface
            *i*: :class:`int`
                Case index
            *test*: ``True`` | {``False``}
                Log file/folder actions but don't actually delete/copy
        """
        pass

   # --- CPU Stats ---
    # Get total CPU hours (actually core hours)
    def GetCPUTime(self, i: int):
        r"""Read a CAPE-style core-hour file from a case

        This function needs to be customized for each solver because it
        needs to know the name of the file in which timing data is
        saved.  It defaults to ``cape_time.dat``.  Modifying this
        command is a one-line fix with a call to
        :func:`cape.cfdx.cntl.Cntl.GetCPUTimeFromFile` with the correct file
        name.

        :Call:
            >>> CPUt = cntl.GetCPUTime(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                CAPE control interface
            *i*: :class:`int`
                Case index
        :Outputs:
            *CPUt*: :class:`float` | ``None``
                Total core hours used in this job
        :Versions:
            * 2015-12-22 ``@ddalle``: v1.0
            * 2016-08-30 ``@ddalle``: v1.1; check for ``RUNNING``
            * 2016-08-31 ``@ddalle``: v1.2; use ``GetCPUTimeBoth``
            * 2023-07-09 ``@ddalle``: v2.0; use ``CaseRunner``
        """
        # Read case
        runner = self.ReadCaseRunner(i)
        # Check for null runner (probably haven't started case)
        if runner is None:
            return None
        # Return CPU time from that
        return runner.get_cpu_time()

   # --- Logging ---
    def log_main(
            self,
            msg: str,
            title: Optional[str] = None,
            parent: int = 0):
        r"""Write a message to primary log

        :Call:
            >>> runner.log_main(msg, title, parent=0)
        :Inputs:
            *runner*: :class:`CaseRunner`
                Controller to run one case of solver
            *msg*: :class:`str`
                Primary content of message
            *title*: {``None``} | :class:`str`
                Manual title (default is name of calling function)
            *parent*: {``0``} | :class:`int`
                Extra levels to use for calling function name
        :Versions:
            * 2025-04-30 ``@ddalle``: v1.0
        """
        # Name of calling function
        funcname = self.get_funcname(parent + 2)
        # Check for manual title
        title = funcname if title is None else title
        # Get logger
        logger = self.get_logger()
        # Log the message
        logger.log_main(title, msg)

    def get_logger(self) -> CntlLogger:
        r"""Get current logger and/or initialize one

        :Call:
            >>> logger = cntl.get_logger()
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
        :Outputs:
            *logger*: :class:`CntlLogger`
                Run matrix logger instance
        :Versions:
            * 2025-04-30 ``@ddalle``: v1.0
        """
        # Get current logger
        logger = getattr(self, "logger", None)
        # Check if present
        if isinstance(logger, CntlLogger):
            return logger
        # Get name of config
        jsonfile = self.opts._filenames[0]
        # Create one
        self.logger = CntlLogger(self.RootDir, jsonfile)

    def get_funcname(self, frame: int = 1) -> str:
        r"""Get name of calling function, mostly for log messages

        :Call:
            >>> funcname = runner.get_funcname(frame=1)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
            *frame*: {``1``} | :class:`int`
                Depth of function to seek title of
        :Outputs:
            *funcname*: :class:`str`
                Name of calling function
        :Versions:
            * 2025-04-30 ``@ddalle``
        """
        # Get frame of function calling this one
        func = sys._getframe(frame).f_code
        # Get name
        return func.co_name

   # --- PBS Jobs ---
    # Get PBS name
    def GetPBSName(self, i: int) -> str:
        r"""Get PBS name for a given case

        :Call:
            >>> lbl = cntl.GetPBSName(i, pre=None)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Run index
            *pre*: {``None``} | :class:`str`
                Prefix for PBS job name
        :Outputs:
            *lbl*: :class:`str`
                Short name for the PBS job, visile via ``qstat``
        :Versions:
            * 2014-09-30 ``@ddalle``: v1.0
            * 2016-12-20 ``@ddalle``: v1.1, moved to *x*
        """
        # Get max length of PBS/Slurm job name
        maxlen = self.opts.get_RunMatrixMaxJobNameLength()
        # Use JSON real file name as prefix
        prefix = self.opts.name
        prefix = f"{prefix}-" if prefix else ''
        # Call from trajectory
        return self.x.GetPBSName(i, prefix=prefix, maxlen=maxlen)

    # Get PBS job ID if possible
    @run_rootdir
    def GetPBSJobID(self, i: int) -> Optional[str]:
        r"""Get PBS job number if one exists

        :Call:
            >>> jobID = cntl.GetPBSJobID(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Run index
        :Outputs:
            *jobID*: ``None`` | :class:`str`
                Most recent PBS/Slurm job name, if able
        :Versions:
            * 2014-10-06 ``@ddalle``: v1.0
            * 2024-01-12 ``@ddalle``: v1.1; remove CheckCase() for speed
        """
        # Get case runner
        runner = self.ReadCaseRunner(i)
        # Check if case exists
        if runner is None:
            return
        # Read principal jobID if possible
        jobID = runner.get_job_id()
        # Repace '' -> None
        return None if jobID == '' else jobID

    # Write the PBS script
    @run_rootdir
    def WritePBS(self, i: int):
        r"""Write the PBS script(s) for a given case

        :Call:
            >>> cntl.WritePBS(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
            *i*: :class:`int`
                Run index
        :Versions:
            * 2014-10-19 ``@ddalle``: v1.0
            * 2023-10-20 ``@ddalle``: v1.1; arbitrary *frun* depth
            * 2024-08-01 ``@ddalle``: v2.0; solver-agnostic
        """
        # Get solver name
        name = self._solver
        # Get module name
        modname_parts = self.__class__.__module__.split('.')
        # Strip off last portion (cape.pyfun.cntl -> cape.pyfun)
        modname = ".".join(modname_parts[:-1])
        # Get the case name.
        frun = self.x.GetFullFolderNames(i)
        # Make folder if necessary
        self.make_case_folder(i)
        # Go to the folder.
        os.chdir(frun)
        # Determine number of unique PBS scripts.
        if self.opts.get_nPBS() > 1:
            # If more than one, use unique PBS script for each run.
            nPBS = self.opts.get_nSeq()
        else:
            # Otherwise use a single PBS script.
            nPBS = 1
        # Loop through the runs.
        for j in range(nPBS):
            # PBS script name.
            if nPBS > 1:
                # Put PBS number in file name.
                fpbs = f'run_{name}.{j:02d}.pbs'
            else:
                # Use single PBS script with plain name.
                fpbs = f'run_{name}.pbs'
            # Initialize the PBS script
            with open(fpbs, 'w') as fp:
                # Write the header
                self.WritePBSHeader(fp, i, j)
                # Initialize options to `run_fun3d.py`
                flgs = ''
                # Get specific python version
                pyexec = self.opts.get_PythonExec(j)
                # Use "python3" as default
                pyexec = "python3" if pyexec is None else pyexec
                # Call the main CAPE interface using python3 -m
                fp.write('\n# Call the main executable\n')
                fp.write(f"{pyexec} -m {modname} run {flgs}\n")

    # Write a PBS header
    def WritePBSHeader(
            self,
            fp: IOBase,
            i: Optional[int] = None,
            j: int = 0,
            typ: Optional[str] = None,
            wd: Optional[str] = None):
        r"""Write common part of PBS or Slurm script

        :Call:
            >>> cntl.WritePBSHeader(fp, i=None, j=0, typ=None, wd=None)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *fp*: :class:`IOBase`
                Open file handle
            *i*: {``None``} | :class:`int`
                Case index (ignore if ``None``); used for PBS job name
            *j*: :class:`int`
                Phase number
            *typ*: {``None``} | ``"batch"`` | ``"post"``
                Group of PBS options to use
            *wd*: {``None``} | :class:`str`
                Folder to enter when starting the job
        :Versions:
            * 2015-09-30 ``@ddalle``: v1.0, fork WritePBS()
            * 2016-09-25 ``@ddalle``: v1.1, "BatchPBS"
            * 2016-12-20 ``@ddalle``: v1.2
                - Consolidated to *opts*
                - Added *prefix*

            * 2024-08-15 ``@ddalle``: v1.3
                - Use *cntl.opts.name* as prefix
                - User-controlled job name length, longer default
        """
        # Get the shell name.
        if i is None:
            # Batch job
            lbl = '%s-batch' % self.__module__.split('.')[0].lower()
            # Max job name length
            maxlen = self.opts.get_RunMatrixMaxJobNameLength()
            # Ensure length
            lbl = lbl[:maxlen]
        else:
            # Case PBS job name
            lbl = self.GetPBSName(i)
        # Check the task manager
        if self.opts.get_slurm(j):
            # Write the Slurm header
            self.opts.WriteSlurmHeader(fp, lbl, j=j, typ=typ, wd=wd)
        else:
            # Call the function from *opts*
            self.opts.WritePBSHeader(fp, lbl, j=j, typ=typ, wd=wd)

    # Write batch PBS job
    @run_rootdir
    def run_batch(self, argv: list):
        r"""Write and submit PBS/Slurm script for a CLI

        :Call:
            >>> cntl.run_batch(argv)
        :Inputs:
            *argv*: :class:`list`\ [:class:`str`]
                List of command-line inputs
        :Versions:
            * 2024-12-20 ``@ddalle``: v1.0
        """
        # Create the folder if necessary
        if not os.path.isdir('batch-pbs'):
            os.mkdir('batch-pbs')
        # Enter the batch pbs folder
        os.chdir('batch-pbs')
        # File name header
        prog = self.__module__.split('.')[0].lower()
        # Current time
        fnow = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # File name
        fpbs = '%s-%s.pbs' % (prog, fnow)
        # Write the file
        with open(fpbs, 'w') as fp:
            # Write header
            self.WritePBSHeader(fp, typ='batch', wd=self.RootDir)
            # Write the command
            fp.write('\n# Run the command\n')
            fp.write('%s\n\n' % (" ".join(argv)))
        # Submit the job
        if self.opts.get_slurm(0):
            # Submit Slurm job
            pbs = queue.sbatch(fpbs)
        else:
            # Submit PBS job
            pbs = queue.pqsub(fpbs)
        # Output
        return pbs

   # --- Case Preparation ---
    # Prepare a case
    @run_rootdir
    def PrepareCase(self, i: int):
        r"""Prepare case for running if necessary

        This function creates the folder, copies mesh files, and saves
        settings and input files.  All of these tasks are completed only
        if they have not already been completed, and it needs to be
        customized for each CFD solver.

        :Call:
            >>> cntl.PrepareCase(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Index of case to analyze
        :Versions:
            * 2014-09-30 ``@ddalle``: v1.0
            * 2015-09-27 ``@ddalle``: v2.0, convert to template
        """
        # Ensure case index is set
        self.opts.setx_i(i)
        # Get the existing status
        n = self.CheckCase(i)
        # Quit if prepared
        if n is not None:
            return None
        # Clear the cache
        self.cache_iter.clear_case(i)
        # Get the run name
        frun = self.x.GetFullFolderNames(i)
        # Case function
        self.CaseFunction(i)
        # Make the directory if necessary
        self.make_case_folder(i)
        # Go there.
        os.chdir(frun)
        # Write the conditions to a simple JSON file.
        self.x.WriteConditionsJSON(i)
        # Write a JSON files with contents of "RunControl" section
        self.WriteCaseJSON(i)

    @run_rootdir
    def make_case_folder(self, i: int):
        r"""Create folder(s) if needed for case *i*

        :Call:
            >>> cntl.make_case_folder(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Index of case to analyze
        :Versions:
            * 2023-08-25 ``@ddalle``: v1.0 (CreateFolder)
            * 2023-10-20 ``@ddalle``: v2.0; support arbitrary depth
        """
        # Get the case name
        frun = self.x.GetFullFolderNames(i)
        # Loop through levels
        for fpart in frun.split(os.sep):
            # Check if folder exists
            if not os.path.isdir(fpart):
                # Create it
                os.mkdir(fpart)
            # Enter folder to prepare for next level
            os.chdir(fpart)

    # Function to apply transformations to config
    def PrepareConfig(self, i):
        r"""Apply rotations, translations, etc. to ``Config.xml``

        :Call:
            >>> cntl.PrepareConfig(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2016-08-23 ``@ddalle``: v1.0
        """
        # Ensure index is set
        self.opts.setx_i(i)
        # Get function for rotations, etc.
        keys = self.x.GetKeysByType(['translate', 'rotate', 'ConfigFunction'])
        # Exit if no keys
        if len(keys) == 0:
            return
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

    # Write conditions JSON file
    @run_rootdir
    def WriteConditionsJSON(self, i, rc=None):
        r"""Write JSON file with run matrix settings for case *i*

        :Call:
            >>> cntl.WriteConditionsJSON(i, rc=None)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Generic control class
            *i*: :class:`int`
                Run index
            *rc*: {``None``} | :class:`dict`
                If specified, write specified "RunControl" options
        :Versions:
            * 2021-09-08 ``@ddalle``: v1.0
        """
        # Get the case name
        frun = self.x.GetFullFolderNames(i)
        # Check if it exists.
        if not os.path.isdir(frun):
            return
        # Go to the folder.
        os.chdir(frun)
        # Write conditions
        self.x.WriteConditionsJSON(i)

    # Write run control options to JSON file
    @run_rootdir
    def WriteCaseJSON(self, i: int, rc: Optional[dict] = None):
        r"""Write JSON file with run control settings for case *i*

        :Call:
            >>> cntl.WriteCaseJSON(i, rc=None)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Generic control class
            *i*: :class:`int`
                Run index
            *rc*: {``None``} | :class:`dict`
                If specified, write specified "RunControl" options
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0
            * 2023-03-31 ``@ddalle``: v2.0; manual options input
            * 2023-08-29 ``@ddalle``: v2.1; call sample_dict()
            * 2024-08-24 ``@ddalle``: v2.2; use CaseRunner
            * 2025-01-23 ``@ddalle``: v2.3; eliminate *Arvhive* settings
        """
        # Get the case name
        frun = self.x.GetFullFolderNames(i)
        # Check if it exists
        if not os.path.isdir(frun):
            return
        # Ensure case index is set
        self.opts.setx_i(i)
        # Go to the folder
        os.chdir(frun)
        # Get "RunControl" section
        if rc is None:
            # Select
            rc = self.opts["RunControl"]
        # Sample to case *i*
        rc = self.opts.sample_dict(rc)
        # Remove *Archive* section
        rc.pop("Archive", None)
        # Read case runner
        runner = self.ReadCaseRunner(i)
        # Write settings
        runner.write_case_json(rc)

    # Read run control options from case JSON file
    @run_rootdir
    def read_case_json(self, i):
        r"""Read ``case.json`` file from case *i* if possible

        :Call:
            >>> rc = cntl.read_case_json(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Instance of control class
            *i*: :class:`int`
                Case index
        :Outputs:
            *rc*: ``None`` | :class:`dict`
                Run control interface read from ``case.json`` file
        :Versions:
            * 2016-12-12 ``@ddalle``: v1.0
            * 2017-04-12 ``@ddalle``: v1.1; add to :mod:`cape.cfdx`
            * 2023-06-29 ``@ddalle``: v2.0; use _case_mod
            * 2023-07-07 ``@ddalle``: v2.1; use CaseRunner
        """
        # Fall back if case doesn't exist
        try:
            # Get a case runner
            runner = self.ReadCaseRunner(i)
            # Get settings
            return runner.read_case_json()
        except Exception:
            # Fall back to None
            return None

   # --- Geometry "Points" ---
    # Evaluate "Points" positions w/o preparing tri or config
    def PreparePoints(self, i):
        r"""Calculate the value of each named ``"Point"`` for case *i*

        :Call:
            >>> x = cntl.PreparePoints(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
                Index of the case to check (0-based)
        :Versions:
            * 2022-03-07 ``@ddalle``: v1.0
        """
        # Reset points
        self.opts.reset_Points()
        # Loop through run matrix variables
        for key in self.x.cols:
            # Get type
            ktyp = self.x.defns[key].get("Type")
            # Check type
            if ktyp in ("rotation", "rotate"):
                self.PreparePointsRotation(key, i)
            elif ktyp in ("translation", "translate"):
                self.PreparePointsTranslation(key, i)

    # Apply a translation to "Points"
    def PreparePointsTranslation(self, key, i):
        r"""Apply a translation to named config points for one col

        :Call:
            >>> cntl.PreparePointsTranslation(key, i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *key*: :class:`str`
                Name of the trajectory key
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2022-03-07 ``@ddalle``: v1.0
        """
        # Get the options for this key.
        kopts = self.x.defns[key]
        # Check for a direction
        if 'Vector' not in kopts:
            raise KeyError(
                "Translation key '%s' does not have a 'Vector'." % key)
        # Get the direction and its type
        vec = kopts['Vector']
        # Get points to translate along with it.
        pts  = kopts.get('Points', [])
        ptsR = kopts.get('PointsSymmetric', [])
        # Make sure these are lists
        if not isinstance(pts, list):
            pts = [pts]
        if not isinstance(ptsR, list):
            ptsR = [ptsR]
        # Check the type
        if isinstance(vec, (list, np.ndarray)):
            # Specified directly.
            u = np.array(vec)
        else:
            # Named vector
            u = np.array(self.opts.get_Point(vec))
        # Form the translation vector
        v = u * self.x[key][i]
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

    # Apply a configuration rotation
    def PreparePointsRotation(self, key, i):
        r"""Apply a rotation to named config points for one col

        :Call:
            >>> cntl.PreparePointsRotation(key, i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *key*: :class:`str`
                Name of the trajectory key
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2022-03-07 ``@ddalle``: v1.0
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
        if not isinstance(comps, list):
            comps = [comps]
        if not isinstance(compsR, list):
            compsR = [compsR]
        if not isinstance(compsT, list):
            compsT = [compsT]
        if not isinstance(compsTR, list):
            compsTR = [compsTR]
        # Symmetry applied to rotation vector.
        kv = kopts.get('VectorSymmetry', [1.0, 1.0, 1.0])
        kx = kopts.get('AxisSymmetry',   kv)
        kc = kopts.get('CenterSymmetry', kx)
        ka = kopts.get('AngleSymmetry', -1.0)
        # Convert symmetries: list -> numpy.ndarray
        if isinstance(kv, list):
            kv = np.array(kv)
        if isinstance(kx, list):
            kx = np.array(kx)
        if isinstance(kc, list):
            kc = np.array(kc)
        # Get the reference points for translations based on this rotation
        xT  = kopts.get('TranslateRefPoint', [0.0, 0.0, 0.0])
        # Get scale for translated points
        kt = kopts.get('TranslateScale', np.ones(3))
        # Ensure vector
        if isinstance(kt, list):
            kt = np.array(kt)
        # Get vector
        vec = kopts.get('Vector')
        ax  = kopts.get('Axis')
        cen = kopts.get('Center')
        # Get points to translate along with it.
        pts  = kopts.get('Points', [])
        ptsR = kopts.get('PointsSymmetric', [])
        # Make sure these are lists
        if not isinstance(pts, list):
            pts = [pts]
        if not isinstance(ptsR, list):
            ptsR = [ptsR]
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
        v0 = cen
        v1 = ax + cen
        v0R = cenR
        v1R = axR + cenR
        # Ensure a dictionary for reference points
        if not isinstance(xT, dict):
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
        # Points to be rotated
        X  = np.array([self.opts.get_Point(pt) for pt in pts])
        XR = np.array([self.opts.get_Point(pt) for pt in ptsR])
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

   # --- Geometry Prep ---
    # Function to apply special triangulation modification keys
    def PrepareTri(self, i):
        r"""Rotate/translate/etc. triangulation for given case

        :Call:
            >>> cntl.PrepareTri(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2014-12-01 ``@ddalle``: v1.0
            * 2016-04-05 ``@ddalle``: v1.1, pycart -> cape
        """
        # Special key types
        tri_types = [
            "TriFunction",
            "TriRotate",
            "TriTranslate",
            "translation",
            "rotation",
        ]
        # Get function for rotations, etc.
        keys = self.x.GetKeysByType(tri_types)
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
            elif kt in ("TriTranslate", "translation"):
                # Component(s) translation
                self.PrepareTriTranslation(key, i)
            elif kt in ("TriRotate", "rotation"):
                # Component(s) rotation
                self.PrepareTriRotation(key, i)

    # Apply a special triangulation function
    def PrepareTriFunction(self, key, i):
        r"""Apply special surf modification function for a case

        :Call:
            >>> cntl.PrepareTriFunction(key, i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *key*: :class:`str`
                Name of key
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2015-09-11 ``@ddalle``: v1.0
            * 2016-04-05 ``@ddalle``: v1.1, pycart -> cape
            * 2022-04-13 ``@ddalle``: v2.0; exec_modfunction()
        """
        # Get the function for this *TriFunction*
        func = self.x.defns[key]['Function']
        # Form args and kwargs
        a = (self, self.x[key][i])
        kw = dict(i=i)
        # Apply it
        self.exec_modfunction(func, a, kw, name="TriFunction")

    # Apply a special configuration function
    def PrepareConfigFunction(self, key, i):
        r"""Apply special configuration modification function for a case

        :Call:
            >>> cntl.PrepareConfigFunction(key, i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *key*: :class:`str`
                Name of key
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2016-08-23 ``@ddalle``: v1.0
            * 2022-04-13 ``@ddalle``: v2.0; exec_modfunction()
        """
        # Get the function for this *ConfigFunction*
        func = self.x.defns[key]['Function']
        # Form args and kwargs
        a = (self, self.x[key][i])
        kw = dict(i=i)
        # Apply it
        self.exec_modfunction(func, a, kw, name="ConfigFunction")

    # Apply a triangulation translation
    def PrepareTriTranslation(self, key, i):
        r"""Apply a translation to a component or components

        :Call:
            >>> cntl.PrepareTriTranslation(key, i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2015-09-11 ``@ddalle``: v1.0
            * 2016-04-05 ``@ddalle``: v1.1, pycart -> cape
        """
        # Get the options for this key.
        kopts = self.x.defns[key]
        # Get the components to translate.
        compID  = self.tri.GetCompID(kopts.get('CompID'), warn=True)
        # Components to translate in opposite direction
        compIDR = self.tri.GetCompID(
            kopts.get('CompIDSymmetric', []), warn=True)
        # Check for a direction
        if 'Vector' not in kopts:
            raise IOError(
                "Rotation key '%s' does not have a 'Vector'." % key)
        # Get the direction and its type
        vec = kopts['Vector']
        # Get points to translate along with it.
        pts  = kopts.get('Points', [])
        ptsR = kopts.get('PointsSymmetric', [])
        # Make sure these are lists.
        if not isinstance(pts, list):
            pts  = list(pts)
        if not isinstance(ptsR, list):
            ptsR = list(ptsR)
        # Check the type
        if isinstance(vec, (list, np.ndarray)) and len(vec) == 2:
            # Vector b/w two points
            u0 = self.opts.get_Point(vec[0])
            u1 = self.opts.get_Point(vec[1])
            u = np.array(u1) - np.array(u0)
        else:
            # Named vector or already vector
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

    # Apply a config.xml translation
    def PrepareConfigTranslation(self, key, i):
        r"""Apply a translation to a component or components

        :Call:
            >>> cntl.PrepareConfigTranslation(key, i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *key*: :class:`str`
                Name of variable from which to get value
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2016-08-23 ``@ddalle``: v1.0
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
        if not isinstance(I, dict):
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
        # Make sure these are lists
        if not isinstance(pts, list):
            pts = list(pts)
        if not isinstance(ptsR, list):
            ptsR = list(ptsR)
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
        r"""Apply a rotation to a component or components

        :Call:
            >>> cntl.PrepareTriRotation(key, i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2015-09-11 ``@ddalle``: v1.0
            * 2016-04-05 ``@ddalle``: v1.1, pycart -> cape
        """
        # ---------------
        # Read the inputs
        # ---------------
        # Get the options for this key.
        kopts = self.x.defns[key]
        # Rotation angle
        theta = self.x[key][i]
        # Get the components to translate.
        compID = self.tri.GetCompID(kopts.get('CompID'), warn=True)
        # Components to translate in opposite direction
        compIDR = self.tri.GetCompID(
            kopts.get('CompIDSymmetric', []), warn=True)
        # Get the components to translate based on a lever armg
        compsT  = kopts.get('CompIDTranslate', [])
        compsTR = kopts.get('CompIDTranslateSymmetric', [])
        # Symmetry applied to rotation vector.
        kv = kopts.get('VectorSymmetry', [1.0, 1.0, 1.0])
        kx = kopts.get('AxisSymmetry',   kv)
        kc = kopts.get('CenterSymmetry', kx)
        ka = kopts.get('AngleSymmetry', -1.0)
        # Convert symmetries: list -> numpy.ndarray
        if isinstance(kv, list):
            kv = np.array(kv)
        if isinstance(kx, list):
            kx = np.array(kx)
        if isinstance(kc, list):
            kc = np.array(kc)
        # Get the reference points for translations based on this rotation
        xT = kopts.get('TranslateRefPoint', [0.0, 0.0, 0.0])
        # Get scale for translated points
        kt = kopts.get('TranslateScale', np.ones(3))
        # Ensure vector
        if isinstance(kt, list):
            # Ensure vector so that we can multiply it by another vector
            kt = np.array(kt)
        # Get vector
        vec = kopts.get('Vector')
        ax  = kopts.get('Axis')
        cen = kopts.get('Center')
        # Get points to translate along with it.
        pts  = kopts.get('Points', [])
        ptsR = kopts.get('PointsSymmetric', [])
        # Make sure these are lists.
        if not isinstance(pts, list):
            pts = list(pts)
        if not isinstance(ptsR, list):
            ptsR = list(ptsR)
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
        v0 = cen
        v1 = ax + cen
        v0R = cenR
        v1R = axR + cenR
        # Ensure a dictionary for reference points
        if not isinstance(xT, dict):
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
        r"""Apply a rotation to a component or components

        :Call:
            >>> cntl.PrepareConfigRotation(key, i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *key*: :class:`str`
                Name of the trajectory key
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2016-08-23 ``@ddalle``: v1.0
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
        # Options to modify GMP
        freeze_ax = kopts.get("FreezeGMPAxis", False)
        freeze_cen = kopts.get("FreezeGMPCenter", False)
        # Ensure list
        if not isinstance(comps, list):
            comps = [comps]
        if not isinstance(compsR, list):
            compsR = [compsR]
        if not isinstance(compsT, list):
            compsT = [compsT]
        if not isinstance(compsTR, list):
            compsTR = [compsTR]
        # Get index of transformation (which order in Config.xml)
        I = kopts.get('TransformationIndex')
        # Symmetry applied to rotation vector.
        kv = kopts.get('VectorSymmetry', [1.0, 1.0, 1.0])
        kx = kopts.get('AxisSymmetry',   kv)
        kc = kopts.get('CenterSymmetry', kx)
        ka = kopts.get('AngleSymmetry', -1.0)
        # Convert symmetries: list -> numpy.ndarray
        if not isinstance(kv, list):
            kv = np.array(kv)
        if not isinstance(kx, list):
            kx = np.array(kx)
        if not isinstance(kc, list):
            kc = np.array(kc)
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
        # Make sure these are lists
        if not isinstance(pts, list):
            pts  = list(pts)
        if not isinstance(ptsR, list):
            ptsR = list(ptsR)
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
        v0 = cen
        v1 = ax + cen
        v0R = cenR
        v1R = axR + cenR
        # Ensure a dictionary for reference points
        if not isinstance(xT, dict):
            # Initialize dict (can't use an iterator to do this in old Python)
            xT = {comp: xT for comp in compsT + compsTR}
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
        # Check for freeze options
        if freeze_ax:
            gmp_ax = None
            gmp_axR = None
        else:
            gmp_ax = ax
            gmp_axR = axR
        if freeze_cen:
            gmp_cen = None
            gmp_cenR = None
        else:
            gmp_cen = cen
            gmp_cenR = cenR
        # Set the positive rotations.
        for comp in comps:
            self.config.SetRotation(
                comp, i=I.get(comp),
                Angle=theta, Center=gmp_cen, Axis=gmp_ax, Frame=frm)
        # Set the negative rotations.
        for comp in compsR:
            self.config.SetRotation(
                comp, i=I.get(comp),
                Angle=ka*theta, Center=gmp_cenR, Axis=gmp_axR, Frame=frm)
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
            self.config.SetTranslation(
                comp, i=I.get(comp),
                Displacement=kt*(YT[j]-XT[j]))
        # Process translations caused by symmetric rotation
        for j in range(len(compsTR)):
            # Get component
            comp = compsTR[j]
            # Apply translation
            self.config.SetTranslation(
                comp, i=I.get(comp),
                Displacement=kt*(YTR[j]-XTR[j]))
        # Save the points.
        for j in range(len(pts)):
            # Set the new value.
            self.opts.set_Point(Y[j], pts[j])
        # Save the symmetric points.
        for j in range(len(ptsR)):
            # Set the new value.
            self.opts.set_Point(YR[j], ptsR[j])

   # --- Thrust Preparation ---
    # Get exit area for SurfCT boundary condition
    def GetSurfCT_ExitArea(self, key, i, comp=None):
        r"""Get exit area for a *CT* trajectory key

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
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
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
            * 2016-04-13 ``@ddalle``: v1.0
        """
        # Check for exit area
        A2 = self.x.GetSurfCT_ExitArea(i, key, comp=comp)
        # Check for a results
        if A2 is not None:
            return A2
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
        r"""Get exit Mach number for a *CT* trajectory key

        This can use either the ``"ExitMach"`` parameter (if available)
        or calculate from the area ratio.  If using the area ratio, the
        input Mach number is also needed.  The relationship between area
        ratio and exit Mach is given below.

            .. math::

                \frac{A_2}{A_1} = \frac{M_1}{M_2}\left(
                    \frac{1+\frac{\gamma-1}{2}M_2^2}{
                    1+\frac{\gamma-1}{2}M_1^2}
                \right) ^ {\frac{1}{2}\frac{\gamma+1}{\gamma-1}}

        :Call:
            >>> M2 = cntl.GetSurfCT_ExitMach(key, i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
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
            * 2016-04-13 ``@ddalle``: v1.0
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
        r"""Get reference area for surface *CT* trajectory key

        This references the ``"RefArea"`` parameter of the definition
        for the run matrix variable *key*.  The user should set this
        parameter to ``1.0`` if thrust inputs are given as dimensional
        values.

        If this is ``None``, it returns the global reference area; if it
        is a string the reference area comes from the reference area for
        that component using ``cntl.opts.get_RefArea(comp)``.

        :Call:
            >>> Aref = cntl.GetSurfCT_RefArea(key, i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *key*: :class:`str`
                Name of trajectory key to check
            *i*: :class:`int`
                Case number
        :Outputs:
            *Aref*: :class:`float`
                Reference area for normalizing thrust coefficients
        :Versions:
            * 2016-04-13 ``@ddalle``: v1.0
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

   # --- DataBook Components ---
    def get_transformation_matrix(
            self, topts: dict, i: int) -> Optional[np.ndarray]:
        r"""Calculate rotation matrix for a databook transformation

        :Call:
            >>> mat = cntl.get_transformation_matrix(topts, i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *topts*: :class:`dict`
                Definitions for transformation
            *i*: :class:`int`
                Case number
        :Outputs:
            *mat*: ``None`` | :class:`np.ndarray`\ [:class:`float`]
                Rotation matrix if *topts* is a rotation
        :Versions:
            * 2025-01-30 ``@ddalle``: v1.0
        """
        # Get the transformation type
        ttype = topts.get("Type", "")
        # Check type
        if ttype not in ("Euler321", "Euler123"):
            return
        # Use same as default in case it's obvious what they should be.
        kph = topts.get('phi', 0.0)
        kth = topts.get('theta', 0.0)
        kps = topts.get('psi', 0.0)
        # Extract roll
        if not isinstance(kph, str):
            # Fixed value
            phi = kph*DEG
        elif kph.startswith('-'):
            # Negative roll angle.
            phi = -self.x[kph[1:]][i]*DEG
        else:
            # Positive roll
            phi = self.x[kph][i]*DEG
        # Extract pitch
        if not isinstance(kth, str):
            # Fixed value
            theta = kth*DEG
        elif kth.startswith('-'):
            # Negative pitch
            theta = -self.x[kth[1:]][i]*DEG
        else:
            # Positive pitch
            theta = self.x[kth][i]*DEG
        # Extract yaw
        if not isinstance(kps, str):
            # Fixed value
            psi = kps*DEG
        elif kps.startswith('-'):
            # Negative yaw
            psi = -self.x[kps[1:]][i]*DEG
        else:
            # Positive pitch
            psi = self.x[kps][i]*DEG
        # Sines and cosines
        cph = np.cos(phi)
        cth = np.cos(theta)
        cps = np.cos(psi)
        sph = np.sin(phi)
        sth = np.sin(theta)
        sps = np.sin(psi)
        # Roll matrix
        R1 = np.array([[1, 0, 0], [0, cph, -sph], [0, sph, cph]])
        # Pitch matrix
        R2 = np.array([[cth, 0, -sth], [0, 1, 0], [sth, 0, cth]])
        # Yaw matrix
        R3 = np.array([[cps, -sps, 0], [sps, cps, 0], [0, 0, 1]])
        # Combined transformation matrix
        # Remember, these are applied backwards in order to undo the
        # original Euler transformation that got the component here.
        if ttype == "Euler321":
            return np.dot(R1, np.dot(R2, R3))
        elif ttype == "Euler123":
            return np.dot(R3, np.dot(R2, R1))

