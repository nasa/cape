r"""
:mod:`cape.cfdx.cntl`: Base module for CFD operations and processing
=====================================================================

This module provides tools and templates for tools to interact with
various CFD codes and their input files. The base class is
:class:`cape.cfdx.cntl.Cntl`, and the derivative classes include
:class:`cape.pycart.cntl.Cntl`. This module creates folders for cases,
copies files, and can be used as an interface to perform most of the
tasks that Cape can accomplish except for running individual cases.

The control module is set up as a Python interface for thec master
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
    * :mod:`cape.cfdx.casecntl`
    * :mod:`cape.cfdx.options`
    * :mod:`cape.cfdx.runmatrix`

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
from collections import Counter
from typing import Any, Callable, Optional, Union

# Third-party modules
import numpy as np

# Local imports
from . import casecntl
from . import databook
from . import queue
from . import report
from .. import textutils
from .casecntl import CaseRunner
from .cntlbase import CntlBase
from .dex import DataExchanger
from .logger import CntlLogger
from .options import Options
from .options.funcopts import UserFuncOpts
from .runmatrix import RunMatrix
from ..argread import ArgReader
from ..config import ConfigXML, ConfigJSON
from ..errors import assert_isinstance
from ..optdict import WARNMODE_WARN
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
COL_HEADERS = {
    "case": "Case Folder",
    "cpu-abbrev": "CPU Hours",
    "cpu-hours": "CPU Time",
    "frun": "Config/Run Directory",
    "gpu-abbrev": "GPU Hours",
    "gpu-hours": "GPU Hours",
    "group": "Group Folder",
    "i": "Case",
    "job": "Job ID",
    "job-id": "Job ID",
    "phase": "Phase",
    "progress": "Iterations",
    "queue": "Que",
    "status": "Status",
}
DEG = np.pi / 180.0
JOB_STATUSES = (
    'PASS',
    'PASS*',
    '---',
    'INCOMP',
    'RUN',
    'DONE',
    'QUEUE',
    'ERROR',
    'ERROR*',
    'FAIL',
    'ZOMBIE',
    'THIS_JOB',
)


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


# Convert ``a,b,c`` -> ``['a', 'b', 'c']``
def _split(v: Union[str, list]) -> list:
    # Check type
    if isinstance(v, str):
        return [vj.strip() for vj in v.split(',')]
    else:
        return v


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


# Arg parser for caseloop()
class CaseLoopArgs(ArgReader):
    __slots__ = ()
    _optmap = {
        "add_cols": "add-cols",
        "add_counters": "add-counters",
        "hide": "hide-cols",
        "hide_cols": "hide-cols",
        "hide_counters": "hide-counters",
        "j": "job",
    }
    _opttypes = {
        "sep": str,
    }
    _optconverters = {
        "add-cols": _split,
        "add-counters": _split,
        "cols": _split,
        "counters": _split,
        "hide-cols": _split,
        "hide-counters": _split,
    }
    _arglist = (
        "casefunc",
    )
    _rc = {
        "sep": " ",
    }


# Class to read input files
class Cntl(CntlBase):
    r"""Class to handle options, setup, and execution of CFD codes

    :Call:
        >>> cntl = cape.Cntl(fname="cape.json")
    :Inputs:
        *fname*: :class:`str`
            Name of JSON settings file from which to read options
    :Outputs:
        *cntl*: :class:`Cntl`
            Instance of CAPE control interface
    :Class attributes:
        * :attr:`_case_cls`
        * :attr:`_case_mod`
        * :attr:`_databook_mod`
        * :attr:`_fjson_default`
        * :attr:`_name`
        * :attr:`_opts_cls`
        * :attr:`_report_mod`
        * :attr:`_solver`
        * :attr:`_warnmode_default`
        * :attr:`_warnmode_envvar`
        * :attr:`_zombie_files`
    :Attributes:
        * :attr:`DataBook`
        * :attr:`RootDir`
        * :attr:`cache_iter`
        * :attr:`caseindex`
        * :attr:`caserunner`
        * :attr:`data`
        * :attr:`job`
        * :attr:`jobqueues`
        * :attr:`jobs`
        * :attr:`logger`
        * :attr:`modules`
        * :attr:`opts`
        * :attr:`x`
    :Versions:
        * 2015-09-20 ``@ddalle``: Started
        * 2016-04-01 ``@ddalle``: v1.0
    """
  # *** CLASS ATTRIBUTES ***
   # --- Names ---
    #: Name of this CAPE module
    #: :class:`str`
    _name = "cfdx"

    #: Name of CFD solver for this module
    #: :class:`str`
    _solver = "cfdx"

   # --- Specific modules ---
    #: Module for case control
    _case_mod = casecntl

    #: Solver-specific module for DataBook
    _databook_mod = databook

    #: Solver-specific module for automated reports
    _report_mod = report

   # --- Specific  classes ---
    #: Solver-specific class for running cases
    #: :class:`type`
    _case_cls = casecntl.CaseRunner

    #: Solver-specific class for CAPE options
    #: :class:`type`
    _opts_cls = Options

   # --- Other settings ---
    #: Name of default JSON file
    #: :class:`str`
    _fjson_default = "cape.json"

    #: Warning mode
    #: :class:`int`
    _warnmode_default = WARNMODE_WARN

    #: Environment variable to read warning mode from
    #: :class:`str`
    _warnmode_envvar = "CAPE_WARNMODE"

    #: List of files to check for zombie status
    #: :class:`list`\ [:class:`str`]
    _zombie_files = ["*.out"]

  # *** DUNDER ***
    # Initialization method
    def __init__(self, fname: Optional[str] = None):
        # Default file name
        fname = self._fjson_default if fname is None else fname
        # Check if file exists
        if not os.path.isfile(fname):
            # Raise error but suppress traceback
            os.sys.tracebacklimit = 0
            raise ValueError("No cape control file '%s' found" % fname)
        #: :class:`str`
        #: Root folder for this run matrix
        self.RootDir = os.getcwd()
        #: :class:`CaseRunner`
        #: Slot for the current case runner
        self.caserunner = None
        #: :class:`int`
        #: Case index of the current case runner
        self.caseindex = None
        #: :class:`cape.cfdx.options.Options`
        #: Options interface for this run matrix
        self.opts = None
        # Read options
        self.read_options(fname)
        #: :class:`dict`
        #: Dictionary of imported custom modules
        self.modules = {}
        #: :class:`CntlLogger`
        #: Run matrix logger instacnce
        self.logger = None
        #: :class:`cape.cfdx.runmatrix.RunMatrix`
        #: Run matrix instance
        self.x = RunMatrix(**self.opts['RunMatrix'])
        # Set run matrix w/i options
        self.opts.save_x(self.x)
        # Set initial index
        self.opts.setx_i(0)
        #: :class:`str`
        #: Job name to check
        self.job = None
        #: :class:`dict`\ [:class:`str`]
        #: Dictionary of PBS/slurm job IDs
        self.jobs = {}
        #: :class:`list`\ [:class:`str`]
        #: List of queues that have been checked
        self.jobqueues = []
        # Run cntl init functions, customize for py{x}
        self.init_post()
        # Run any initialization functions
        self.InitFunction()
        #: :class:`cape.cfdx.databook.DataBook`
        #: Interface to post-processed data
        self.DataBook = None
        #: :class:`dict`\ [:class:`DataExchanger`]
        #: Data extraction classes for each component of DataBook
        self.data = {}
        #: :class:`CaseCache`
        #: Cache of current iteration for each case
        self.cache_iter = CaseCache("iter")

    # Output representation
    def __repr__(self) -> str:
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

    __str__ = __repr__

  # *** OPTIONS ***
   # --- Other init ---
    def init_post(self):
        pass

   # --- I/O ---
    # Read options (first time)
    def read_options(self, fjson: str):
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

   # --- Options history ---
    # Copy all options
    def SaveOptions(self):
        # Copy the options
        self._opts0 = copy.deepcopy(self.opts)

    # Reset options to last "save"
    def RevertOptions(self):
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

   # --- Top-level options ---
    # Get the project rootname
    def GetProjectRootName(self, j: int = 0) -> str:
        # Get default, pyfun, pylava, etc.
        modname = self.__class__.__module__
        projname = modname.split('.')[1]
        # (base method, probably overwritten)
        return getattr(self, "_name", projname)

   # --- Phases ---
    # Get expected actual breaks of phase iters
    def GetPhaseBreaks(self) -> list:
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
                nj = 0 if nj is None else nj
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
        # Read the local case.json file.
        rc = self.read_case_json(i)
        # Check for null file
        if rc is None:
            return self.opts.get_PhaseIters(-1)
        # Option for desired iterations
        N = rc.get('PhaseIters', 0)
        # Output the last entry (if list)
        return getel(N, -1)

  # *** HOOKS ***
    # Function to import user-specified modules
    def ImportModules(self):
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

    # Call function to apply settings for case *i*
    def CaseFunction(self, i: int):
        # Get input functions
        funclist = self.opts.get("CaseFunction")
        # Execute each
        self._exec_funclist(funclist, (self, i), name="CaseFunction")

    # Function to apply initialization function
    def InitFunction(self):
        # Get input functions
        funclist = self.opts.get("InitFunction")
        # Execute each
        self._exec_funclist(funclist, self, name="InitFunction")

    # Execute a function by spec
    def exec_cntlfunction(self, funcspec: Union[str, dict]) -> Any:
        # Check type
        if isinstance(funcspec, dict):
            return self.exec_cntl_function_dict(funcspec)
        elif isinstance(funcspec, str):
            return self.exec_cntlfunction_str(funcspec)
        # Otherwise bad type

    # Execute a function by name only
    def exec_cntlfunction_str(self, funcname: str) -> Any:
        return self.exec_modfunction(funcname)

    # Execute a function by dict
    def exec_cntl_function_dict(self, funcspec: dict):
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

    # Execute a function
    def exec_modfunction(
            self,
            funcname: str,
            a: Optional[Union[tuple, list]] = None,
            kw: Optional[dict] = None,
            name: Optional[str] = None) -> Any:
        return self._exec_pyfunc("module", funcname, a, kw, name)

    def import_module(self, modname: str):
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

    def _expand_funcarg(self, argval: Union[Any, str]) -> Any:
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

    def _exec_funclist(
            self,
            funclist: list,
            a: Optional[tuple] = None,
            kw: Optional[dict] = None,
            name: Optional[str] = None):
        # Exit if none
        if not funclist:
            return
        # Ensure list
        assert_isinstance(funclist, list, "list of functions")
        # Loop through functions
        for func in funclist:
            # Execute function
            self.exec_modfunction(func, a, kw, name)

    # Execute a function
    def _exec_pyfunc(
            self,
            functype: str,
            funcname: str,
            a: Optional[Union[tuple, list]] = None,
            kw: Optional[dict] = None,
            name: Optional[str] = None) -> Any:
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

  # *** CASE PREPARATION ***
   # --- Mesh ---
    @run_rootdir
    def PrepareMesh(self, i: int):
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

    # Prepare the mesh for case *i* (if necessary)
    @run_rootdir
    def PrepareMeshUnstructured(self, i: int):
        # Ensure case index is set
        self.opts.setx_i(i)
        # Create case folder
        self.make_case_folder(i)
        # Copy/link basic files
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

    # Prepare the mesh for case *i* (if necessary)
    @run_rootdir
    def prepare_mesh_overset(self, i: int):
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

   # --- Mesh: location ---
    def GetCaseMeshFolder(self, i: int) -> str:
        # Check for a group setting
        if self.opts.get_GroupMesh():
            # Get the name of the group
            fgrp = self.x.GetGroupFolderNames(i)
            # Use that
            return fgrp
        # Case folder
        frun = self.x.GetFullFolderNames(i)
        # Get the CaseRunner
        runner = self.ReadCaseRunner(i)
        # Check for working folder
        workdir = runner.get_working_folder_()
        # Combine
        return os.path.join(frun, workdir)

   # --- Mesh: files ---
    @run_rootdir
    def PrepareMeshFiles(self, i: int) -> int:
        # Start counter
        n = 0
        # Get working folder
        workdir = self.GetCaseMeshFolder(i)
        # Create working folder if necessary
        if not os.path.isdir(workdir):
            os.mkdir(workdir)
        # Enter the working folder
        os.chdir(workdir)
        # Option to link instead of copying
        linkopt = self.opts.get_LinkMesh()
        # Loop through those files
        for fraw in self.GetInputMeshFileNames():
            # Get processed name of file
            fout = self.process_mesh_filename(fraw)
            # Absolutize input file
            fabs = self.abspath(fraw)
            # Copy fhe file.
            if os.path.isfile(fabs) and not os.path.isfile(fout):
                # Copy the file
                if linkopt:
                    os.symlink(fabs, fout)
                else:
                    shutil.copyfile(fabs, fout)
                # Counter
                n += 1
        # Output the count
        return n

    def PrepareMeshWarmStart(self, i: int) -> bool:
        # Ensure case index is set
        self.opts.setx_i(i)
        # Starting phase
        phase0 = self.opts.get_PhaseSequence(0)
        # Project name
        fproj = self.GetProjectRootName(phase0)
        # Get *WarmStart* settings
        warmstart = self.opts.get_WarmStart(phase0)
        warmstartdir = self.opts.get_WarmStartFolder(phase0)
        # If user defined a WarmStart source, expand it
        if warmstartdir is None or warmstart is False:
            # No *warmstart*
            return False
        else:
            # Read conditions
            x = {key: self.x[key][i] for key in self.x.cols}
            # Expand the folder name
            warmstartdir = warmstartdir % x
            # Absolutize path (already run in workdir)
            warmstartdir = os.path.realpath(warmstartdir)
            # Override *warmstart* if source and destination match
            warmstart = warmstartdir != os.getcwd()
        # Exit if WarmStart not turned on
        if not warmstart:
            return False
        # Get project name for source
        srcj = self.opts.get_WarmStartPhase(phase0)
        # Read case
        runner = self.ReadFolderCaseRunner(warmstartdir)
        # Project name
        src_project = runner.get_project_rootname(srcj)
        # Get restart file
        fsrc = runner.get_restart_file(srcj)
        fto = runner.get_restart_file(j=0)
        # Get nominal mesh file
        fmsh = self.opts.get_MeshFile(0)
        # Normalize it
        fmsh_src = self.process_mesh_filename(fmsh, src_project)
        fmsh_to = self.process_mesh_filename(fmsh, fproj)
        # Absolutize
        fmsh_src = os.path.join(warmstartdir, fmsh_src)
        # Check for source file
        if not os.path.isfile(fsrc):
            raise ValueError("No WarmStart source file '%s'" % fsrc)
        if not os.path.isfile(fmsh_src):
            raise ValueError("No WarmStart mesh '%s'" % fmsh_src)
        # Status message
        print("    WarmStart from folder")
        print("      %s" % warmstartdir)
        print("      Using restart file: %s" % os.path.basename(fsrc))
        print("      Using mesh file: %s" % os.path.basename(fmsh_src))
        # Copy files
        shutil.copy(fsrc, fto)
        shutil.copy(fmsh_src, fmsh_to)
        # Return status
        return True

   # --- Mesh: Surf ---
    def PrepareMeshTri(self, i: int):
        # Get mesh file and tri file settings
        meshfile = self.opts.get_MeshFile()
        trifile = self.opts.get_TriFile()
        # Option to run aflr3
        aflr3 = self.opts.get_aflr3()
        # Check for triangulation options
        if (trifile is None) or (meshfile is not None):
            return
        # Status update
        print("  Preparing surface triangulation...")
        # Starting phase
        phase0 = self.opts.get_PhaseSequence(0)
        # Project name
        fproj = self.GetProjectRootName(phase0)
        # Read the mesh
        self.ReadTri()
        # Revert to initial surface
        self.tri = self.tri0.Copy()
        # Apply rotations, translations, etc.
        self.PrepareTri(i)
        # AFLR3 boundary conditions file
        fbc = self.opts.get_aflr3_BCFile()
        # Enter case folder
        frun = self.x.GetFullFolderNames(i)
        os.chdir(self.RootDir)
        os.chdir(frun)
        # Check for those AFLR3 boundary conditions
        if fbc:
            # Absolute file name
            if not os.path.isabs(fbc):
                fbc = os.path.join(self.RootDir, fbc)
            # Copy the file
            shutil.copyfile(fbc, '%s.aflr3bc' % fproj)
        # Surface configuration file
        fxml = self.opts.get_ConfigFile()
        # Write it if necessary
        if fxml:
            # Absolute file name
            if not os.path.isabs(fxml):
                fxml = os.path.join(self.RootDir, fxml)
            # Copy the file
            if os.path.isfile(fxml):
                shutil.copyfile(fxml, f'{fproj}.xml')
        # Check intersection status.
        if self.opts.get_intersect():
            # Names of triangulation files
            fvtri = "%s.tri" % fproj
            fctri = "%s.c.tri" % fproj
            fftri = "%s.f.tri" % fproj
            # Write tri file as non-intersected; each volume is one CompID
            if not os.path.isfile(fvtri):
                self.tri.WriteVolTri(fvtri)
            # Write the existing triangulation with existing CompIDs.
            if not os.path.isfile(fctri):
                self.tri.WriteCompIDTri(fctri)
            # Write the farfield and source triangulation files
            if not os.path.isfile(fftri):
                self.tri.WriteFarfieldTri(fftri)
        elif self.opts.get_verify():
            # Names of surface mesh files
            fitri = "%s.i.tri" % fproj
            fsurf = "%s.surf" % fproj
            # Write the tri file
            if not os.path.isfile(fitri):
                self.tri.Write(fitri)
            # Write the AFLR3 surface file
            if not os.path.isfile(fsurf):
                self.tri.WriteSurf(fsurf)
        elif aflr3:
            # Names of surface mesh files
            fsurf = "%s.surf" % fproj
            # Write the AFLR3 surface file only
            if not os.path.isfile(fsurf):
                self.tri.WriteSurf(fsurf)
        else:
            # Write main tri file
            ext = getattr(self, "_tri_ext", "tri")
            ftri = f"{fproj}.{ext}"
            # Write it
            if not os.path.isfile(ftri):
                if ext == "fro":
                    self.tri.WriteFro(ftri)
                else:
                    self.tri.Write(ftri)

   # --- Mesh: File names ---
    # Get list of mesh file names that should be in a case folder
    def GetProcessedMeshFileNames(self) -> list:
        # Initialize output
        fname = []
        # Loop through input files.
        for f in self.GetInputMeshFileNames():
            # Get processed name
            fname.append(self.process_mesh_filename(f))
        # Output
        return fname

    # Get list of raw file names
    def GetInputMeshFileNames(self) -> list:
        # Get the file names from *opts*
        fname = self.opts.get_MeshFile()
        # Ensure list
        if fname is None:
            # Remove ``None``
            return []
        elif isinstance(fname, (list, np.ndarray, tuple)):
            # Return list-like as list
            return list(fname)
        else:
            # Convert to list
            return [fname]

    # Process a mesh file name to use the project root name
    def process_mesh_filename(
            self,
            fname: str,
            fproj: Optional[str] = None) -> str:
        # Get project name
        if fproj is None:
            fproj = self.GetProjectRootName()
        # Split names by '.'
        fsplt = fname.split('.')
        # Get final extension
        fext = fsplt[-1]
        # Get infix
        finfix = None if len(fsplt) < 2 else fsplt[-2]
        # Use project name plus the same extension.
        if finfix and finfix in UGRID_EXTS:
            # Copy second-to-last extension
            return f"{fproj}.{finfix}.{fext}"
        else:
            # Just the extension
            return f"{fproj}.{fext}"

   # --- Tri files ---
    # Function to prepare the triangulation for each grid folder
    @run_rootdir
    def ReadTri(self):
        # Only read triangulation if not already present
        tri = getattr(self, "tri", None)
        if tri is not None:
            return
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

   # --- Surface config ---
    # Read configuration (without tri file if necessary)
    @run_rootdir
    def ReadConfig(self, f: bool = False) -> Union[ConfigXML, ConfigJSON]:
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

  # *** CASE INTERFACE ***
   # --- CaseRunner ---
    # Get case runner from a folder
    @run_rootdir
    def ReadFolderCaseRunner(self, fdir: str) -> CaseRunner:
        # Check if folder exists
        if not os.path.isdir(fdir):
            raise ValueError(f"Cannot read CaseRunner: no folder '{fdir}'")
        # Read case runner
        return self._case_cls(fdir)

    # Instantiate a case runner
    @run_rootdir
    def ReadCaseRunner(self, i: int) -> CaseRunner:
        # Check the slot
        if (self.caseindex == i) and (self.caserunner is not None):
            # Check if folder still exists
            if os.path.isdir(self.caserunner.root_dir):
                # Already in the slot
                return self.caserunner
        # Get case name
        frun = self.x.GetFullFolderNames(i)
        fabs = os.path.join(self.RootDir, frun)
        # Check if case is present
        if not os.path.isdir(fabs):
            return
        # Instantiate
        self.caserunner = self._case_cls(fabs)
        self.caseindex = i
        # Save jobs
        self.caserunner.jobs = self.jobs
        self.caserunner.job = self.job
        # Save *cntl* so it doesn't have to read it
        self.caserunner.cntl = self
        # Output
        return self.caserunner

   # --- Start/stop ---
    # Function to start a case: submit or run
    @run_rootdir
    def StartCase(self, i: int):
        # Set case index
        self.opts.setx_i(i)
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
        # Print status
        print("     Starting case '%s'" % frun)
        # Get the case runner
        runner = self.ReadCaseRunner(i)
        # Remove case from cache
        self.cache_iter.clear_case(i)
        # Start the case by either submitting or calling it.
        ierr, pbs = runner.start()
        # Check for error
        if ierr:
            print(f"     Job failed with return code {ierr}")
        # Display the PBS job ID if that's appropriate.
        if pbs:
            print(f"     Submitted job: {pbs}")
        # Output
        return pbs

    # Function to terminate a case: qdel and remove RUNNING file
    @run_rootdir
    def StopCase(self, i: int):
        # Check status
        if self.CheckCase(i) is None:
            # Case not ready
            return
        # Read runner
        runner = self.ReadCaseRunner(i)
        # Stop the job if possible
        runner.stop_case()

   # --- Status counts ---
    # Count R/Q cases based on full status
    def CountRunningCases(
            self, I: list,
            jobs: Optional[dict] = None,
            u: Optional[str] = None) -> int:
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

   # --- Status ---
    # Function to determine if case is PASS, ---, INCOMP, etc.
    def CheckCaseStatus(
            self, i: int,
            jobs: Optional[dict] = None,
            auto: bool = False,
            u: Optional[str] = None,
            qstat: bool = True):
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

    # Check if cases with zero iterations are not yet setup to run
    def CheckNone(self, v: bool = False) -> bool:
        return False

   # --- Phase ---
    # Check a case's phase output files
    @run_rootdir
    def CheckUsedPhase(self, i: int, v: bool = False):
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
    def CaseGetCurrentPhase(self) -> int:
        # Be safe
        try:
            # Instatiate case runner
            runner = self.ReadCaseRunner()
            # Get the phase number
            return runner.get_phase()
        except Exception:
            return 0

   # --- Iteration ---
    # Check a case
    @run_rootdir
    def CheckCase(
            self,
            i: int,
            force: bool = False,
            v: bool = False) -> Optional[int]:
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

   # --- PBS/Slurm ---
    # Get information on all jobs from current user
    def get_pbs_jobs(
            self,
            force: bool = False,
            u: Optional[str] = None,
            server: Optional[str] = None,
            qstat: bool = True) -> dict:
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

  # *** CLI ***
   # --- Case loop ---
    # Loop through cases
    @CaseLoopArgs.check
    def caseloop_verbose(
            self, casefunc: Optional[Callable] = None, **kw) -> int:
        # Get list of indices
        inds = self.x.GetIndices(**kw)
        # Default list of columns to display
        defaultcols = [
            "i",
            "frun",
            "status",
            "progress",
            "queue",
            "cpu-hours",
        ]
        # Default list of columns to count
        defaultcountercols = [
            "status",
        ]
        # Process options
        add_cols = kw.get("add-cols")
        add_ctrs = kw.get("add-counters")
        hide_cols = kw.get("hide-cols")
        hide_ctrs = kw.get("hide-counters")
        sep = kw.get("sep", " ")
        # Get indent
        indent = kw.get("indent", 4)
        tab = ' ' * indent
        # Remove None
        add_cols = [] if not add_cols else add_cols
        add_ctrs = [] if not add_ctrs else add_ctrs
        hide_cols = [] if not hide_cols else hide_cols
        hide_ctrs = [] if not hide_ctrs else hide_ctrs
        # Process main list
        cols = kw.get("cols", defaultcols)
        ctrs = kw.get("counters", defaultcountercols)
        # Process additional cols
        for col in add_cols:
            if col not in cols:
                cols.append(col)
        for col in add_ctrs:
            if col not in ctrs:
                ctrs.append(col)
        # Process explicit hidden columns
        for col in hide_cols:
            if col in cols:
                cols.remove(col)
        for col in hide_ctrs:
            if col in ctrs:
                ctrs.remove(col)
        # Final column count
        ncol = len(cols)
        # Get headers
        headers = {
            col: self._header(col) for col in cols
        }
        # Ensure lengths are large enough for header
        maxlens = {
            col: max(self._maxlen(col, inds), len(headers[col]))
            for col in cols
        }
        # Header and horizontal line
        hdr_parts = [
            '%-*s' % (maxlens[col], header)
            for col, header in headers.items()
        ]
        hline_parts = ['-'*l for l in maxlens.values()]
        header = sep.join(hdr_parts)
        hline = sep.join(hline_parts)
        # Print header
        print(header)
        print(hline)
        # Initialize headers
        counters = {col: Counter() for col in ctrs}
        # Output counter
        n = 0
        # Loop through cases
        for i in inds:
            # Loop through columns
            for j, col in enumerate(cols):
                # Get length
                lj = maxlens[col]
                # Get value
                vj = self.getvalstr(col, i)
                # Print it
                sys.stdout.write("%-*s" % (lj, vj))
                sys.stdout.flush()
                # Print separator
                if j + 1 >= ncol:
                    # New line
                    sys.stdout.write('\n')
                else:
                    # Separator
                    sys.stdout.write(sep)
                sys.stdout.flush()
                # Count value if appropriate
                if col in counters:
                    counters[col].update((vj,))
            # Run case function
            if callable(casefunc):
                vi = casefunc(i)
                # Add to counter if appropriate
                ni = vi if isinstance(vi, (int, np.integer)) else 0
                n += ni
        # Blank line
        print("")
        # Process counters
        for col, counter in counters.items():
            # Skip if not in column list
            if col not in maxlens:
                continue
            # Length for this column
            lj = maxlens[col]
            # Check for special case
            if col == "status":
                # Loop through statuses in specified order
                for sts in JOB_STATUSES:
                    nj = counter.get(sts, 0)
                    if nj:
                        sys.stdout.write(f"{sts}={nj}, ")
                sys.stdout.write("\n")
                sys.stdout.flush()
            else:
                # Print column name
                print(f"{headers[col]}:")
                # Loop through values
                for vj, nj in counter.items():
                    print("%s- %*s: %i" % (tab, lj, vj, nj))
        # Output the accumulator
        return n

    # Loop through cases
    def caseloop(self, casefunc: Callable, **kw):
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

    # Get header for display column
    def _header(self, opt: str) -> str:
        return COL_HEADERS.get(opt, opt)

    # Get value for aribtrary value, skipping progress
    def _maxlen(self, opt: str, I: np.ndarray) -> int:
        # Check for special cases
        if opt == "progress":
            # Get anticipated max iteration
            jmax = self.opts.get_PhaseSequence(-1)
            imax = self.opts.get_PhaseIters(jmax)
            # Add some padding
            ipad = str(int(1.8*imax))
            # Create example string w/ max anticipated length
            return 2*len(ipad) + 1
        elif opt == "iter":
            # Get anticipated max iteration
            jmax = self.opts.get_PhaseSequence(-1)
            imax = self.opts.get_PhaseIters(jmax)
            # Add some padding
            ipad = str(int(1.8*imax))
            # Create example string w/ max anticipated length
            return len(ipad)
        elif opt == "cpu-hours":
            # Max it out at 8 ....
            return 8
        elif opt == "gpu-hours":
            # Max it out at 8 ...
            return 8
        elif opt in ("cpu-abbrev", "gpu-abbrev", "wall-abbrev"):
            return 5
            # Abbreviated counts
        elif opt == "job-id":
            # Just the integer portion of job ID
            return 8
        elif opt == "job":
            # Full name of job ID, int + (first part of server)
            return 16
        elif opt == "maxiter":
            # Get anticipated max iteration
            jmax = self.opts.get_PhaseSequence(-1)
            imax = self.opts.get_PhaseIters(jmax)
            # Add some padding
            ipad = str(int(1.8*imax))
            # Create example string w/ max anticipated length
            return len(ipad)
        elif opt == "phase":
            # Get anticipated max phase
            jmax = max(self.opts.get_PhaseSequence())
            # Create example string w/ max phase
            return 2*len(str(jmax)) + 1
        elif opt == "frun":
            # Get folder names
            fruns = self.x.GetFullFolderNames(I)
            # Return max length
            return max(map(len, fruns))
        elif opt == "group":
            # Get group folder names
            fruns = self.x.GetGroupFolderNames(I)
            # Return max length
            return max(map(len, fruns))
        elif opt == "case":
            # Get case folder name
            fruns = self.x.GetFolderNames(I)
            # Return max length
            return max(map(len, fruns))
        elif opt == "i":
            # Case index; avoid 0
            inds = np.fmax(2, I)
            return int(np.max(np.ceil(np.log10(inds))))
        elif opt == "status":
            # Case status
            return 7
        elif opt == "queue":
            # Queue indicator
            return 1
        else:
            # Get values
            vals = self.x.GetValue(opt, I)
            # Check for float
            if isinstance(vals[0], (float, np.floating)):
                return 8
            # Get max length when converted to str
            return max([len(str(v)) for v in vals])

   # --- Preprocess ---
    # CLI arg preprocesser
    def preprocess_kwargs(self, kw: dict):
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

   # --- Check ---
    # Function to display current status
    def DisplayStatus(self, **kw):
        # Call verbose caseloop
        self.caseloop_verbose(**kw)

   # --- Mark ---
    # Mark a case as PASS
    @run_rootdir
    def MarkPASS(self, **kw):
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
        # Get indices
        I = self.x.GetIndices(**kw)
        # Loop through cases
        for i in I:
            # Mark case
            self.x.UnmarkCase(i)
        # Write the trajectory
        self.x.WriteRunMatrixFile()

   # --- Execute script ---
    # Execute script
    @run_rootdir
    def ExecScript(self, **kw) -> int:
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

   # --- Cleanup ---
    # Function to clear out zombies
    def Dezombie(self, **kw):
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

   # --- Modify cases ---
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
        raise NotImplementedError(
            f"ExtendCase() not implemented for {self._name} module")

    # Function to extend one or more cases
    def ApplyCases(self, **kw):
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

    # Extend a case
    def ApplyCase(self, i: int, **kw):
        r"""Rewrite CAPE inputs for case *i*

        :Call:
            >>> cntl.ApplyCase(i, n=1, j=None, imax=None)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                CAPE main control instance
            *i*: :class:`int`
                Run index
            *qsub*: ``True`` | {``False``}
                Option to submit case after applying settings
            *nPhase*: :class:`int`
                Phase to apply settings to
        """
        raise NotImplementedError(
            f"ExtendCase() not implemented for {self._name} module")

    # Delete jobs
    def qdel_cases(self, **kw):
        r"""Kill/stop PBS job of cases

        This function deletes a case's PBS/Slurm jobbut not delete the
        foder for a case.

        :Call:
            >>> cntl.qdel_cases(**kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Cape control interface
            *I*: {``None``} | :class:`list`\ [:class:`int`]
                List of cases to delete
            *kw*: :class:`dict`
                Other subset parameters, e.g. *re*, *cons*
        :Versions:
            * 2025-06-22 ``@ddalle``: v1.0
        """
        # Delete one case (maybe)
        def qdel_case(i: int) -> int:
            # Check queue
            que = self.getval("queue", i)
            # Delete if status other than '.'
            if que and (que != '.'):
                return self.StopCase(i)
        # Case loop
        self.caseloop_verbose(qdel_case, **kw)

    # Delete cases
    def rm_cases(self, prompt: bool = True, **kw) -> int:
        r"""Delete one or more cases

        This function deletes a case's PBS job and removes the entire
        directory. By default, the method prompts for confirmation
        before deleting; set *prompt* to ``False`` to delete without
        prompt, but only cases with 0 iterations can be deleted this
        way.

        :Call:
            >>> n = cntl.rm_cases(prompt=True, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Cape control interface
            *prompt*: {``True``} | ``False``
                Whether or not to prompt user before deleting case
            *I*: {``None``} | :class:`list`\ [:class:`int`]
                List of cases to delete
            *kw*: :class:`dict`
                Other subset parameters, e.g. *re*, *cons*
        :Outputs:
            *n*: :class:`int`
                Number of folders deleted
        :Versions:
            * 2025-06-20 ``@ddalle``: v1.0
        """
        # Delete one case (maybe)
        def rm_case(i: int) -> int:
            # Check *prompt* overwrite
            if prompt:
                # No need to check
                prompti = True
            else:
                # Get case status
                sts = self.getval("status", i)
                n = self.getval("iter", i)
                # Always prompt if set up
                prompti = n or (sts not in ("INCOMP", "ERROR", "---"))
            # Delete
            return self.DeleteCase(i, prompt=prompti)
        # Case loop
        n = self.caseloop_verbose(rm_case, **kw)
        # Status message
        print(f"Deleted {n} cases")
        return n

    # Function to delete a case folder: qdel and rm
    @run_rootdir
    def DeleteCase(self, i: int, **kw):
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

  # *** RUN MATRIX ***
   # --- Values ---
    # Get value for specified property
    def getval(self, opt: str, i: int) -> Any:
        # Check for special cases
        if opt == "progress":
            # Get iteration
            n = self.CheckCase(i)
            nmax = self.GetLastIter(i)
            # Display '/' if case not set up
            if n is None:
                return '/'
            else:
                return f'{int(n)}/{nmax}'
        elif opt == "iter":
            # Get just iteration
            return self.CheckCase(i)
        elif opt == "i":
            # Case index
            return i
        elif opt == "cpu-hours":
            # Get CPU hours
            return self.GetCPUTime(i)
        elif opt == "cpu-abbrev":
            # Get CPU hours
            hr = self.GetCPUTime(i)
            # Shorten
            return textutils.pprint_n(hr)
        elif opt == "status":
            # Get case status
            return self.check_case_status(i)
        elif opt == "maxiter":
            # Case's indicated maximum req'd iteration
            return self.GetLastIter(i)
        elif opt == "phase":
            # Get case's current phase
            j, jmax = self.CheckUsedPhase(i)
            # Format string
            return f"{j}/{jmax}"
        elif opt == "frun":
            # Get full folder name
            return self.x.GetFullFolderNames(i)
        elif opt == "group":
            # Group folder name
            return self.x.GetGroupFolderNames(i)
        elif opt == "case":
            # Case folder name (no group)
            return self.x.GetFolderNames(i)
        elif opt == "job-id":
            # Get job name
            job = self.GetPBSJobID(i)
            # Get int
            return int(job.split('.', 0))
        elif opt == "job":
            # Get full PBS/Slurm job name
            return self.GetPBSJobID(i)
        elif opt == "status":
            # Get case status, INCOMP, DONE, etc.
            return self.check_case_status(i)
        elif opt == "queue":
            # Get PBS/Slurm queue indicator
            return self.check_case_job(i)
        else:
            return self.x.GetValue(opt, i)

    # Get value, ensuring string output
    def getvalstr(self, opt: str, i: int) -> str:
        # Get raw value
        v = self.getval(opt, i)
        # Don't display "None"
        v = '' if v is None else v
        # Check type
        if isinstance(v, str):
            # Return it
            return v
        elif isinstance(v, (float, np.floating)):
            # Convert
            return "%8g" % v
        else:
            # Use primary string method
            return str(v)

   # --- Filter ---
    # Apply user filter
    def FilterUser(self, i: int, **kw) -> bool:
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

   # --- Index ---
    # Get case index
    def GetCaseIndex(self, frun: str) -> Optional[int]:
        return self.x.GetCaseIndex(frun)

   # --- Run Interface ---

   # --- Case Preparation ---
    # Prepare ``CAPE-STOP-PHASE`` file
    def _prepare_incremental(self, i: int, j: Union[bool, int] = False):
        r"""Prepare a case to stop at end of specified phase

        :Call:
            >>> cntl._prepare_incremental(i, j=False)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Instance of control class
            *i*: :class:`int`
                Case index
            *j*: ``True`` | {``False``} | :class:`int`
                Option to stop at end of any phase (``True``) or
                specific phase number
        :Versions:
            * 2024-05-26 ``@ddalle``: v1.0
        """
        # Check option
        if j is False:
            # Normal case; no incremental option
            return
        # Run folder
        frun = self.x.GetFullFolderNames(i)
        # Absolutize
        fstop = os.path.join(self.RootDir, frun, casecntl.STOP_PHASE_FILE)
        # Create file
        with open(fstop, 'w') as fp:
            # Write phase number if *j* is an int
            if isinstance(j, (int, np.int32, np.int64)):
                fp.write(f"{j}\n")

   # --- Cases ---
    # Check if a case is running.
    @run_rootdir
    def CheckRunning(self, i):
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
        :Versions:
            * 2014-10-03 ``@ddalle``: v1.0
        """
        # Get run name
        frun = self.x.GetFullFolderNames(i)
        # Check for the RUNNING file.
        q = os.path.isfile(os.path.join(frun, casecntl.RUNNING_FILE))
        # Output
        return q

    # Check for a failure
    @run_rootdir
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
        :Versions:
            * 2015-01-02 ``@ddalle``: v1.0
        """
        # Get run name
        frun = self.x.GetFullFolderNames(i)
        # Check for the error file
        q = os.path.isfile(os.path.join(frun, casecntl.FAIL_FILE))
        # Check ERROR flag
        q = q or self.x.ERROR[i]
        # Output
        return q

    # Check for no unchanged files
    @run_rootdir
    def CheckZombie(self, i):
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
        # Check which job manager
        if self.opts.get_slurm(0):
            # Slurm
            envid = os.environ.get('SLURM_JOB_ID', '0')
        else:
            # Slurm
            envid = os.environ.get('PBS_JOBID', '0')
        # Convert to integer
        return int(envid.split(".", 1)[0])

    def check_case_job(self, i: int, active: bool = True) -> str:
        r"""Get queue status of the PBS/Slurm job from case *i*

        :Call:
            >>> s = cntl.check_case_job(i, active=True)
        :Inputs:
            *cntl*: :class:`Cntl`
                Controller for one CAPE run matrix
            *i*: :class:`int`
                Case index
            *active*: {``True``} | ``False``
                Whether or not to allow new calls to ``qstat``
        :Outputs:
            *s*: :class:`str`
                Job queue status

                * ``-``: not in queue
                * ``Q``: job queued (not running)
                * ``R``: job running
                * ``H``: job held
                * ``E``: job error status
        :Versions:
            * 2025-05-04 ``@ddalle``: v1.0
        """
        # Get case runner
        runner = self._read_runner(i, active)
        # Check for null case
        if runner is None:
            return '.'
        # Check
        return runner.check_queue()

    def check_case_status(self, i: int, active: bool = True) -> str:
        r"""Get queue status of the PBS/Slurm job from case *i*

        :Call:
            >>> sts = cntl.check_case_job(i, active=True)
        :Inputs:
            *cntl*: :class:`Cntl`
                Controller for one CAPE run matrix
            *i*: :class:`int`
                Case index
            *active*: {``True``} | ``False``
                Whether or not to allow new calls to ``qstat``
        :Outputs:
            *sts*: :class:`str`
                Case status

                * ``---``: folder does not exist
                * ``INCOMP``: case incomplete but [partially] set up
                * ``QUEUE``: case waiting in PBS/Slurm queue
                * ``RUNNING``: case currently running
                * ``DONE``: all phases and iterations complete
                * ``PASS``: *DONE* and marked by user as passed
                * ``ZOMBIE``: seems running; but no recent file updates
                * ``PASS*``: case marked by user but not *DONE*
                * ``ERROR``: case marked as error
                * ``FAIL``: case failed but not marked by user
        :Versions:
            * 2025-05-04 ``@ddalle``: v1.0
        """
        # Get case runner
        runner = self._read_runner(i, active)
        # Check for empty case
        if runner is None:
            # Check for mark
            if self.x.ERROR[i]:
                return 'ERROR'
            elif self.x.PASS[i]:
                return 'PASS*'
            return '---'
        # Check
        return runner.get_status()

    def _read_runner(self, i: int, active: bool = True) -> CaseRunner:
        r"""Read case runner and synch PBS jobs tracker

        :Call:
            >>> runner = cntl._read_runner(i, active=True)
        """
        # Get case runner
        runner = self.ReadCaseRunner(i)
        # Check for null case
        if runner is None:
            return
        # Check for active job trackers
        if isinstance(runner.jobs, queue.QStat):
            if isinstance(self.jobs, queue.QStat):
                # Both runner and cntl are active; combine
                self.jobs.update(runner.jobs)
                runner.jobs = self.jobs
            else:
                # Save the runner's version here
                self.jobs = runner.jobs
        elif isinstance(self.jobs, queue.QStat):
            # Save local instance in *runner*
            runner.jobs = self.jobs
        else:
            # Initialize
            self.jobs = runner._get_qstat()
        # Apply *active* setting
        runner.jobs.active = active
        # Output
        return runner

   # --- Case Modification ---
    # Extend a case
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
        :Versions:
            * 2016-12-12 ``@ddalle``: v1.0
            * 2024-08-27 ``@ddalle``: v2.0; move code to ``CaseRunner``
            * 2024-09-28 ``@ddalle``: v2.1; add *j*
        """
        # Ignore cases marked PASS
        if self.x.PASS[i] or self.x.ERROR[i]:
            return
        # Get the runner
        runner = self.ReadCaseRunner(i)
        # Extend the case
        runner.extend_case(m=n, j=j, nmax=imax)

  # *** REPORTING ***
    # Read report
    @run_rootdir
    def ReadReport(self, rep: str) -> report.Report:
        # Read the report
        rep = self.__class__._report_mod.Report(self, rep)
        # Output
        return rep

  # *** DATA EXTRACTION ***
   # --- Data Exchange ---
    def update_dex_case(self, comp: str, i: int) -> int:
        r"""Update one case of a *DataBook* component

        :Call:
            >>> n = cntl.update_dex_case(comp, i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix controller instance
            *comp*: :class:`str`
                Name of component to read
            *i*: :class:`int`
                Case to index
        :Outputs:
            *n*: ``0`` | ``1``
                Number of updates made
        :Versions:
            * 2025-07-25 ``@ddalle``: v1.0
        """
        ...

    def read_dex(self, comp: str, force: bool = False) -> DataExchanger:
        r"""Read a *DataBook* component using :class:`DataExchanger`

        :Call:
            >>> db = cntl.read_dex(comp, force=False)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix controller instance
            *comp*: :class:`str`
                Name of component to read
            *force*: ``True`` | {``False``}
                Option to re-read even if present in database
        :Outputs:
            *db*: :class:`DataExchanger`
                Data extracted from run matrix for comp *comp*
        :Versions:
            * 2025-07-25 ``@ddalle``: v1.0
        """
        # Get data dictionary
        dex = self.data
        dex = {} if self.data is None else dex
        # Check for component
        if (not force) and comp in dex:
            return dex[comp]
        # Read
        db = DataExchanger(self, comp)
        # Save it
        dex[comp] = db
        self.data = dex
        # Output
        return db

   # --- DataBook Init ---
    # Read the data book
    @run_rootdir
    def ReadDataBook(self, comp: Optional[str] = None) -> databook.DataBook:
        # Test if already read
        if self.DataBook is not None:
            return
        # Ensure list of components
        if comp is not None and not isinstance(comp, list):
            comp = [comp]
        # Get DataBook class
        databookmod = self.__class__._databook_mod
        # Instantiate class
        self.DataBook = databookmod.DataBook(self, comp=comp)
        # Call any custom functions
        self.ReadDataBookPost()

    # Call special post-read DataBook functions
    def ReadDataBookPost(self):
        r"""Do ``py{x}`` specific init actions after reading DataBook

        :Call:
            >>> cntl.ReadDataBookPost()
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                CAPE run matrix control instance
        :Versions:
            * 2023-05-31 ``@ddalle``: v1.0
        """
        pass

   # --- DataBook Updaters ---
    # Databook updater
    @run_rootdir
    def UpdateFM(self, **kw):
        r"""Collect force and moment data

        :Call:
            >>> cntl.UpdateFM(cons=[], **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *fm*, *aero*: {``None``} | :class:`str`
                Wildcard to subset list of FM components
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2014-12-12 ``@ddalle``: v1.0
            * 2014-12-22 ``@ddalle``: v2.0
                - Complete rewrite of DataBook class
                - Eliminate "Aero" class

            * 2017-04-25 ``@ddalle``: v2.1, add wildcards
            * 2018-10-19 ``@ddalle``: v3.0, rename from Aero()
        """
        # Get component option
        comp = kw.get("fm", kw.get("aero"))
        # If *comp* is ``True``, process all options
        if comp is True:
            comp = None
        # Get full list of components
        comp = self.opts.get_DataBookByGlob("FM", comp)
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

    # Function to collect statistics from generic-property component
    @run_rootdir
    def UpdateCaseProp(self, **kw):
        r"""Update generic-property databook for one or more comp

        :Call:
            >>> cntl.UpdateCaseProp(cons=[], **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *prop*: {``None``} | :class:`str`
                Wildcard to subset list of ``"Prop"`` components
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2022-04-08 ``@ddalle``: v1.0
        """
        # Get component option
        comp = kw.get("prop")
        # Get full list of components
        comp = self.opts.get_DataBookByGlob("CaseProp", comp)
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Make sure databook is present
        self.ReadDataBook(comp=[])
        # Check if we are deleting or adding.
        if kw.get('delete', False):
            # Delete cases.
            self.DataBook.DeleteCases(I, comp=comp)
        else:
            # Read the results and update as necessary.
            self.DataBook.UpdateDataBook(I, comp=comp)

    # Function to collect statistics from generic-property component
    @run_rootdir
    def UpdatePyFuncDataBook(self, **kw):
        r"""Update Python function databook for one or more comp

        :Call:
            >>> cntl.UpdatePyFuncDataBook(cons=[], **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *prop*: {``None``} | :class:`str`
                Wildcard to subset list of ``"PyFunc"`` components
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2022-04-10 ``@ddalle``: v1.0
        """
        # Get component option
        comp = kw.get("dbpyfunc")
        # Get full list of components
        comp = self.opts.get_DataBookByGlob("PyFunc", comp)
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Make sure databook is present
        self.ReadDataBook(comp=[])
        # Check if we are deleting or adding.
        if kw.get('delete', False):
            # Delete cases.
            self.DataBook.DeleteCases(I, comp=comp)
        else:
            # Read the results and update as necessary.
            self.DataBook.UpdateDataBook(I, comp=comp)

    # Update line loads
    @run_rootdir
    def UpdateLL(self, **kw):
        r"""Update one or more line load data books

        :Call:
            >>> cntl.UpdateLL(ll=None, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *ll*: {``None``} | :class:`str`
                Optional name of line load component to update
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
            *pbs*: ``True`` | {``False``}
                Whether or not to calculate line loads with PBS scripts
        :Versions:
            * 2016-06-07 ``@ddalle``: v1.0
            * 2016-12-21 ``@ddalle``: v1.1, Add *pbs* flag
            * 2017-04-25 ``@ddalle``: v1.2
                - Removed *pbs*
                - Added ``--delete``
        """
        # Get component option
        comp = kw.get("ll")
        # Check for True or False
        if comp is True:
            # Update all components
            comp = None
        elif comp is False:
            # Exit
            return
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

    @run_rootdir
    def UpdateSurfCp(self, **kw):
        r"""Collect surface pressure data

        :Call:
            >>> cntl.UpdateSurfCp(cons=[], **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *fm*, *aero*: {``None``} | :class:`str`
                Wildcard to subset list of FM components
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2014-12-12 ``@ddalle``: v1.0
            * 2014-12-22 ``@ddalle``: v2.0
                - Complete rewrite of DataBook class
                - Eliminate "Aero" class

            * 2017-04-25 ``@ddalle``: v2.1, add wildcards
            * 2018-10-19 ``@ddalle``: v3.0, rename from Aero()
        """
        # Get component option
        comp = kw.get("surfcp")
        # If *comp* is ``True``, process all options
        if comp is True:
            comp = None
        # Get full list of components
        comp = self.opts.get_DataBookByGlob("surfcp", comp)
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

    # Update time series
    @run_rootdir
    def UpdateTS(self, **kw):
        r"""Update one or more time series data books

        :Call:
            >>> cntl.UpdateTS(ts=None, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *ts*: {``None``} | :class:`str`
                Optional name of time series component to update
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
            *pbs*: ``True`` | {``False``}
                Whether or not to calculate line loads with PBS scripts
        :Versions:
            * 2016-06-07 ``@ddalle``: v1.0
            * 2016-12-21 ``@ddalle``: v1.1, Add *pbs* flag
            * 2017-04-25 ``@ddalle``: v1.2
                - Removed *pbs*
                - Added ``--delete``
        """
        # Get component option
        comp = kw.get("ts")
        # Check for True or False
        if comp is True:
            # Update all components
            comp = None
        elif comp is False:
            # Exit
            return
        # Get full list of components
        comp = self.opts.get_DataBookByGlob("TimeSeries", comp)
        # Apply constraints
        I = self.x.GetIndices(**kw)
        # Read the data book handle
        self.ReadDataBook(comp=[])
        self.ReadConfig()
        # Check if we are deleting or adding.
        if kw.get('delete', False):
            # Read the existing data book.
            self.ReadDataBook(comp=comp)
            # Delete cases.
            self.DataBook.DeleteCases(I, comp=comp)
        else:
            self.ReadDataBook(comp=[])
            # self.ReadDataBook(comp=[])
            # Read the results and update as necessary.
            self.DataBook.UpdateDataBook(I, comp=comp)

    # Update TriqFM data book
    @run_rootdir
    def UpdateTriqFM(self, **kw):
        r"""Update one or more TriqFM data books

        :Call:
            >>> cntl.UpdateTriqFM(comp=None, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Control class
            *comp*: {``None``} | :class:`str`
                Name of TriqFM component
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2017-03-29 ``@ddalle``: v1.0
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
        r"""Update one or more TriqPoint point sensor data books

        :Call:
            >>> cntl.UpdateTriqPoint(comp=None, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Control class
            *comp*: {``None``} | :class:`str`
                Name of TriqFM component
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2017-03-29 ``@ddalle``: v1.0
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

   # --- DataBook Checkers ---
    # Function to check FM component status
    def CheckFM(self, **kw):
        r"""Display missing force & moment components

        :Call:
            >>> cntl.CheckFM(**kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *fm*, *aero*: {``None``} | :class:`str`
                Wildcard to subset list of FM components
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2018-10-19 ``@ddalle``: v1.0
        """
        # Get component option
        comps = kw.get(
            "fm", kw.get(
                "aero", kw.get(
                    "checkFM", kw.get("check-fm", kw.get("check-db")))))
        # Get full list of components
        comps = self.opts.get_DataBookByGlob("FM", comps)
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
                if ui == "blocked":
                    continue
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
                    txt += (
                        "    Extra case: %s\n" % DBc.x.GetFullFolderNames(j))
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
                        txt += (
                            "    Blocked case: %s\n"
                            % DBc.x.GetFullFolderNames(j))
            # If there is text, display the info
            if txt:
                # Header
                print("Checking component '%s'" % comp)
                print(txt[:-1])

    # Function to check LL component status
    def CheckLL(self, **kw):
        r"""Display missing line load components

        :Call:
            >>> cntl.CheckLL(**kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *fm*, *aero*: {``None``} | :class:`str`
                Wildcard to subset list of FM components
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2018-10-19 ``@ddalle``: v1.0
        """
        # Get component option
        comps = kw.get(
            "ll", kw.get("checkLL", kw.get("check-ll", kw.get("check-db"))))
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
                if ui == "blocked":
                    continue
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
                    txt += (
                        "    Extra case: %s\n"
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
                        txt += (
                            "    Blocked case: %s\n"
                            % DBc.x.GetFullFolderNames(j))
            # If there is text, display the info
            if txt:
                # Header
                print("Checking component '%s'" % comp)
                print(txt[:-1])

    # Function to check TriqFM component status
    def CheckTriqFM(self, **kw):
        r"""Display missing TriqFM components

        :Call:
            >>> cntl.CheckTriqFM(**kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *fm*, *aero*: {``None``} | :class:`str`
                Wildcard to subset list of FM components
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2018-10-19 ``@ddalle``: v1.0
        """
        # Get component option
        comps = kw.get(
            "triqfm", kw.get(
                "checkTriqFM", kw.get("check-triqfm", kw.get("check-db"))))
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
                if ui == "blocked":
                    continue
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
                    txt += (
                        "    Extra case: %s\n"
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
                        txt += (
                            "    Blocked case: %s\n"
                            % DBc.x.GetFullFolderNames(j))
            # If there is text, display the info
            if txt:
                # Header
                print("Checking component '%s'" % comp)
                print(txt[:-1])

    # Function to check TriqFM component status
    def CheckTriqPoint(self, **kw):
        r"""Display missing TriqPoint components

        :Call:
            >>> cntl.CheckTriqPoint(**kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *fm*, *aero*: {``None``} | :class:`str`
                Wildcard to subset list of FM components
            *I*: :class:`list`\ [:class:`int`]
                List of indices
            *cons*: :class:`list`\ [:class:`str`]
                List of constraints like ``'Mach<=0.5'``
        :Versions:
            * 2018-10-19 ``@ddalle``: v1.0
        """
        # Get component option
        comps = kw.get(
            "pt", kw.get("checkPt", kw.get("check-pt", kw.get("check-db"))))
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
                if ui == "blocked":
                    continue
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
                        txt += (fmtc % ("%s/%s" % (comp, pt)))
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
                        txt += (
                            "    Extra case: %s\n"
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
                            txt += (
                                "    Blocked case: %s\n"
                                % DBc.x.GetFullFolderNames(j))
                # If there is text, display the info
                if txt:
                    # Header
                    print("Checking point sensor '%s/%s'" % (comp, pt))
                    print(txt[:-1])

  # *** FILE MANAGEMENT ***
   # --- Files ---
    # Absolutize
    def abspath(self, fname: str) -> str:
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
                print(f"  Replacing file '{fname}'")
                os.remove(fdest)
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

   # --- Archiving ---
    # Run ``--archive`` on one case
    def ArchiveCase(self, i: int, test: bool = False):
        r"""Perform ``--archive`` archiving on one case

        There are no restrictions on the status of the case for this
        action.

        :Call:
            >>> cntl.CleanCase(i, test=False)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix controller instance
            *i*: :class:`int`
                Case index
            *test*: ``True`` | {``False``}
                Log file/folder actions but don't actually delete/copy
        :Versions:
            * 2024-09-18 ``@ddalle``: v1.0
        """
        # Read case runner
        runner = self.ReadCaseRunner(i)
        # Run action
        runner.archive(test)

    # Run ``--skeleton`` on one case
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
        # Read case runner
        runner = self.ReadCaseRunner(i)
        # Run action
        runner.skeleton(test)

    # Run ``--clean`` on one case
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
        :Versions:
            * 2024-09-18 ``@ddalle``: v1.0
        """
        # Read case runner
        runner = self.ReadCaseRunner(i)
        # Run action
        runner.clean(test)

    # Unarchive cases
    def UnarchiveCases(self, **kw):
        r"""Unarchive a list of cases

        :Call:
            >>> cntl.UnarchiveCases(**kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Instance of control interface
        :Versions:
            * 2017-03-13 ``@ddalle``: v1.0
            * 2023-10-20 ``@ddalle``: v1.1; arbitrary-depth *frun*
            * 2024-09-20 ``@ddalle``: v2.0; use CaseArchivist
        """
        # Test status
        test = kw.get("test", False)
        # Loop through the folders
        for i in self.x.GetIndices(**kw):
            # Print case name
            print(self.x.GetFullFolderNames(i))
            # Create the case folder
            self.make_case_folder(i)
            # Read case runner
            runner = self.ReadCaseRunner(i)
            # Unarchive!
            runner.unarchive(test)

