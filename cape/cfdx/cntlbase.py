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
from abc import ABC, abstractmethod
from io import IOBase
from typing import Any, Callable, Optional, Union

# Third-party modules
import numpy as np

# Local imports
from . import casecntlbase
from . import databookbase
from . import queue
from .casecntlbase import CaseRunnerBase
from .options.runctlopts import RunControlOpts
from .logger import CntlLogger
from ..config import ConfigXML, ConfigJSON


# Class to read input files
class CntlBase(ABC):
    r"""Base class for :class:`cape.cfdx.cntl.Cntl`"""

  # *** CLASS ATTRIBUTES ***
    _name = "cfdx"
    _solver = "cfdx"
    _case_mod = casecntlbase
    _databook_mod = databookbase
    _report_mod = None
    _case_cls = None
    _opts_cls = None
    _fjson_default = None
    _warnmode_default = None
    _warnmode_envvar = None
    _zombie_files = None

  # *** DUNDER ***
    # Initialization method
    @abstractmethod
    def __init__(self, fname: Optional[str] = None):
        r"""Initialization method"""
        pass

    # Output representation
    @abstractmethod
    def __repr__(self) -> str:
        pass

    __str__ = __repr__

  # *** OPTIONS ***
   # --- Other init ---
    @abstractmethod
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

   # --- I/O ---
    # Read options (first time)
    @abstractmethod
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
        pass

   # --- Options history ---
    # Copy all options
    @abstractmethod
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
        pass

    # Reset options to last "save"
    @abstractmethod
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
        pass

   # --- Top-level options ---
    # Get the project rootname
    @abstractmethod
    def GetProjectRootName(self, j: int = 0) -> str:
        r"""Get the project root name

        The JSON file overrides the value from the namelist file if
        appropriate

        :Call:
            >>> name = cntl.GetProjectName(j=0)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
            *j*: {``0``} | :class:`int`
                Phase number
        :Outputs:
            *name*: :class:`str`
                Project root name
        :Versions:
            * 2015-10-18 ``@ddalle``: v1.0 (pyfun)
            * 2023-06-15 ``@ddalle``: v1.1; cleaner logic
            * 2024-10-22 ``@ddalle``: v2.0; moved to ``cfdx``
            * 2025-07-15 ``@ddalle``: v2.1; move into ``CntlBase``
        """
        pass

   # --- Phases ---
    # Get expected actual breaks of phase iters
    @abstractmethod
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
        pass

    # Get last iter
    @abstractmethod
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
        pass

  # *** HOOKS ***
    # Function to import user-specified modules
    @abstractmethod
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
        pass

    # Function to apply initialization function
    @abstractmethod
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
        pass

    # Call function to apply settings for case *i*
    @abstractmethod
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
        pass

    # Execute a function by spec
    @abstractmethod
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
        pass

    # Execute a function by name only
    @abstractmethod
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
        pass

    # Execute a function by dict
    @abstractmethod
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
        pass

    # Execute a function
    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def _exec_funclist(
            self,
            funclist: list,
            a: Optional[tuple] = None,
            kw: Optional[dict] = None,
            name: Optional[str] = None):
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
        pass

    # Execute a function
    @abstractmethod
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
        pass

  # *** CASE PREPARATION ***
   # --- Main ---
    # Prepare a case
    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    # Prepare ``CAPE-STOP-PHASE`` file
    @abstractmethod
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
        pass

   # --- Case settings ---
    # Write conditions JSON file
    @abstractmethod
    def WriteConditionsJSON(self, i: int):
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
        pass

    # Write run control options to JSON file
    @abstractmethod
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
        pass

   # --- PBS/Slurm ---
    # Write the PBS script
    @abstractmethod
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
        pass

    # Write a PBS header
    @abstractmethod
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
        pass

   # --- Hooks ---
    # Apply a special triangulation function
    @abstractmethod
    def PrepareTriFunction(self, key: str, i: int):
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
        pass

    # Apply a special configuration function
    @abstractmethod
    def PrepareConfigFunction(self, key: str, i: int):
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
        pass

   # --- Mesh ---
    @abstractmethod
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
        pass

    # Prepare the mesh for case *i* (if necessary)
    @abstractmethod
    def PrepareMeshUnstructured(self, i: int):
        r"""Prepare the mesh for case *i* if necessary

        :Call:
            >>> cntl.PrepareMeshUnstructured(i)
        :Inputs:
            *cntl*: :class:`cape.pyfun.cntl.Cntl`
                Instance of control class
            *i*: :class:`int`
                Case index
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0 (pyfun)
            * 2024-11-04 ``@ddalle``: v1.3 (pyfun)
            * 2024-11-07 ``@ddalle``: v1.0
            * 2025-07-15 ``@ddalle``: v1.1; name change
        """
        pass

    # Prepare the mesh for case *i* (if necessary)
    @abstractmethod
    def prepare_mesh_overset(self, i: int):
        r"""Copy/link mesh files from config folder into case folder

        :Call:
            >>> cntl.prepare_mesh_overset(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix controller instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2024-10-10 ``@ddalle``: v1.0
        """
        pass

   # --- Mesh: location ---
    @abstractmethod
    def GetCaseMeshFolder(self, i: int) -> str:
        r"""Get relative path to folder where mesh should be copied

        :Call:
            >>> fdir = cntl.GetCaseMeshFolder(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
            *i*: {``0``} | :class:`int`
                Case index
        :Outputs:
            *fdir*: :class:`str`
                Folder to copy file, relative to *cntl.RootDir*
        :Versions:
            * 2024-11-06 ``@ddalle``: v1.0
        """
        pass

   # --- Mesh: files ---
    @abstractmethod
    def PrepareMeshFiles(self, i: int) -> int:
        r"""Copy main unstructured mesh files to case folder

        :Call:
            >>> n = cntl.PrepareMeshFiles(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
            *i*: :class:`int`
                Case index
        :Outputs:
            *n*: :class:`int`
                Number of files copied
        :Versions:
            * 2024-11-05 ``@ddalle``: v1.0
            * 2025-04-02 ``@ddalle``: v1.1; add *LinkMesh* option
        """
        pass

    @abstractmethod
    def PrepareMeshWarmStart(self, i: int) -> bool:
        r"""Prepare *WarmStart* files for case, if appropriate

        :Call:
            >>> warmstart = cntl.PrepareMeshWarmStart(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                Name of main CAPE input (JSON) file
            *i*: :class:`int`
                Case index
        :Outputs:
            *warmstart*: :class:`bool`
                Whether or not case was warm-started
        :Versions:
            * 2024-11-04 ``@ddalle``: v1.0
        """
        pass

   # --- Mesh: Surf ---
    @abstractmethod
    def PrepareMeshTri(self, i: int):
        r"""Prepare surface triangulation for AFLR3, if appropriate

        :Call:
            >>> cntl.PrepareMeshTri(i)
        :Inputs:
            *cntl*: :class:`Cntl`
                CAPE run matrix control instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2024-11-01 ``@ddalle``: v1.0 (from pyfun's PrepareMesh())
        """
        pass

   # --- Mesh: File names ---
    # Get list of mesh file names that should be in a case folder
    @abstractmethod
    def GetProcessedMeshFileNames(self) -> list:
        r"""Return the list of mesh files that are written

        :Call:
            >>> fname = cntl.GetProcessedMeshFileNames()
        :Inputs:
            *cntl*: :class:`Cntl`
                Run matrix control instance for unstructured-mesh solver
        :Outputs:
            *fname*: :class:`list`\ [:class:`str`]
                List of file names written to case folders
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0
        """
        pass

    # Get list of raw file names
    @abstractmethod
    def GetInputMeshFileNames(self) -> list:
        r"""Return the list of mesh files from file

        :Call:
            >>> fnames = cntl.GetInputMeshFileNames()
        :Inputs:
            *cntl*: :class:`Cntl`
                Run matrix control instance for unstructured-mesh solver
        :Outputs:
            *fnames*: :class:`list`\ [:class:`str`]
                List of file names read from root directory
        :Versions:
            * 2015-10-19 ``@ddalle``: v1.0 (pyfun)
            * 2024-10-22 ``@ddalle``: v1.0
        """
        pass

    # Process a mesh file name to use the project root name
    @abstractmethod
    def process_mesh_filename(
            self,
            fname: str,
            fproj: Optional[str] = None) -> str:
        r"""Return a mesh file name using the project root name

        :Call:
            >>> fout = cntl.process_mesh_filename(fname, fproj=None)
        :Inputs:
            *cntl*: :class:`Cntl`
                Run matrix control instance for unstructured-mesh solver
            *fname*: :class:`str`
                Raw file name to be converted to case-folder file name
            *fproj*: {``None``} | :class;`str`
                Project root name
        :Outputs:
            *fout*: :class:`str`
                Name of file name using project name as prefix
        :Versions:
            * 2016-04-05 ``@ddalle``: v1.0 (pyfun)
            * 2023-03-15 ``@ddalle``: v1.1; add *fproj*
            * 2024-10-22 ``@ddalle``: v2.0; move to ``cfdx``
        """
        pass

   # --- Surface: read ---
    # Function to prepare the triangulation for each grid folder
    @abstractmethod
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
        pass

   # --- Surface: manipulation ---
    # Function to apply special triangulation modification keys
    @abstractmethod
    def PrepareTri(self, i: int):
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
        pass

    # Apply a triangulation translation
    @abstractmethod
    def PrepareTriTranslation(self, key: str, i: int):
        r"""Apply a translation to a component or components

        :Call:
            >>> cntl.PrepareTriTranslation(key, i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *key*: :class:`str`
                Run matrix key for translation def'n
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2015-09-11 ``@ddalle``: v1.0
            * 2016-04-05 ``@ddalle``: v1.1, pycart -> cape
        """
        pass

    # Apply a triangulation rotation
    @abstractmethod
    def PrepareTriRotation(self, key: str, i: int):
        r"""Apply a rotation to a component or components

        :Call:
            >>> cntl.PrepareTriRotation(key, i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *key*: :class:`str`
                Run matrix for rotation defn'
            *i*: :class:`int`
                Index of the case to check (0-based)
        :Versions:
            * 2015-09-11 ``@ddalle``: v1.0
            * 2016-04-05 ``@ddalle``: v1.1, pycart -> cape
        """
        pass

   # --- Surface: config ---
    # Function to apply transformations to config
    @abstractmethod
    def PrepareConfig(self, i: int):
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
        pass

    # Read configuration (without tri file if necessary)
    @abstractmethod
    def ReadConfig(self, f=False) -> Union[ConfigXML, ConfigJSON]:
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
        pass

    # Apply a config.xml translation
    @abstractmethod
    def PrepareConfigTranslation(self, key: str, i: int):
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
        pass

    # Apply a configuration rotation
    @abstractmethod
    def PrepareConfigRotation(self, key: str, i: int):
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
        pass

   # --- Geometry: points ---
    # Evaluate "Points" positions w/o preparing tri or config
    @abstractmethod
    def PreparePoints(self, i: int):
        r"""Calculate the value of each named ``"Point"`` for case *i*

        :Call:
            >>> x = cntl.PreparePoints(i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *i*: :class:`int`
                Case index
        :Versions:
            * 2022-03-07 ``@ddalle``: v1.0
        """
        pass

    # Apply a translation to "Points"
    @abstractmethod
    def PreparePointsTranslation(self, key: str, i: int):
        r"""Apply a translation to named config points for one col

        :Call:
            >>> cntl.PreparePointsTranslation(key, i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *key*: :class:`str`
                Name of the trajectory key
            *i*: :class:`int`
                Case index
        :Versions:
            * 2022-03-07 ``@ddalle``: v1.0
        """
        pass

    # Apply a configuration rotation
    @abstractmethod
    def PreparePointsRotation(self, key: str, i: int):
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
        pass

   # --- Thrust Preparation ---
    # Get exit area for SurfCT boundary condition
    @abstractmethod
    def GetSurfCT_ExitArea(
            self,
            key: str,
            i: int,
            comp: Optional[str] = None) -> float:
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
        pass

    # Get exit Mach number for SurfCT boundary condition
    @abstractmethod
    def GetSurfCT_ExitMach(
            self,
            key: str,
            i: int,
            comp: Optional[str] = None) -> float:
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
        pass

    # Reference area
    @abstractmethod
    def GetSurfCT_RefArea(self, key: str, i: int) -> float:
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
        pass

  # *** REPORTING ***
    @abstractmethod
    def ReadReport(self, rep: str):
        r"""Read a report interface

        :Call:
            >>> rep = cntl.ReadReport(rep)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                CAPE main control instance
            *rep*: :class:`str`
                Name of report
        :Outputs:
            *rep*: :class:`cape.cfdx.report.Report`
                Report interface
        :Versions:
            * 2018-10-19 ``@ddalle``: Version 1.0
        """
        pass

  # *** DATA EXTRACTION ***
   # --- DataBook ---
    @abstractmethod
    def ReadDataBook(self, comp: Optional[str] = None):
        r"""Read the current data book

        :Call:
            >>> cntl.ReadDataBook()
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                CAPE run matrix control instance
        :Versions:
            * 2016-09-15 ``@ddalle``: v1.0
            * 2023-05-31 ``@ddalle``: v2.0; universal ``cape.cntl``
        """
        pass

    # Call special post-read DataBook functions
    @abstractmethod
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

   # --- DataBook Components ---
    @abstractmethod
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
        pass

  # *** CASE INTERFACE ***
   # --- CaseRunner ---
    # Get case runner from a folder
    @abstractmethod
    def ReadFolderCaseRunner(self, fdir: str) -> CaseRunnerBase:
        r"""Read a ``CaseRunner`` from a folder by name

        :Call:
            >>> runner = cntl.ReadFolderCaseRunner(fdir)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *fdir*: :class:`str`
                Folder of case to read from
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

    @abstractmethod
    def _read_runner(self, i: int, active: bool = True) -> CaseRunnerBase:
        r"""Read case runner and synch PBS jobs tracker

        :Call:
            >>> runner = cntl._read_runner(i, active=True)
        """
        pass

   # --- Case settings ---
    # Read run control options from case JSON file
    @abstractmethod
    def read_case_json(self, i: int) -> RunControlOpts:
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
        pass

   # --- Start/stop ---
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

   # --- Counts by status ---
    # Count R/Q cases based on full status
    @abstractmethod
    def CountRunningCases(
            self, I: list,
            jobs: Optional[dict] = None,
            u: Optional[str] = None) -> int:
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
        pass

    # Count R/Q cases based on PBS/Slurm only
    @abstractmethod
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
        pass

   # --- Status ---
    # Get overall status using runner
    @abstractmethod
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
        pass

    # Function to determine if case is PASS, ---, INCOMP, etc.
    @abstractmethod
    def CheckCaseStatus(
            self, i: int,
            jobs: Optional[dict] = None,
            auto: bool = False,
            u: Optional[str] = None,
            qstat: bool = True) -> str:
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
        :Outputs:
            *sts*: :class:`str`
                Stats of case *i8
        :Versions:
            * 2014-10-04 ``@ddalle``: v1.0
            * 2014-10-06 ``@ddalle``: v1.1; check queue status
            * 2023-12-13 ``@dvicker``: v1.2; check for THIS_JOB
            * 2024-08-22 ``@ddalle``: v1.3; add *qstat*
        """
        pass

    # Check if cases with zero iterations are not yet setup to run
    @abstractmethod
    def CheckNone(self, v: bool = False) -> bool:
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
        pass

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
    @abstractmethod
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
        pass

    # Check for if we are running inside a batch job
    @abstractmethod
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
        pass

   # --- Phase ---
    # Check a case's phase output files
    @abstractmethod
    def CheckUsedPhase(self, i: int, v: bool = False):
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
        pass

    # Check a case's phase number
    @abstractmethod
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
        pass

    # Get the current iteration number from :mod:`case`
    @abstractmethod
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
        pass

   # --- Iteration ---
    # Check a case
    @abstractmethod
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
        pass

    # Get the current iteration number from :mod:`casecntl`
    @abstractmethod
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
        pass

   # --- PBS jobs ---
    # Get PBS/Slurm queue status indicator
    @abstractmethod
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
        pass

    # Get PBS name
    @abstractmethod
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
        pass

    # Get PBS job ID if possible
    @abstractmethod
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
        pass

    # Get information on all jobs from current user
    @abstractmethod
    def get_pbs_jobs(
            self,
            force: bool = False,
            u: Optional[str] = None,
            server: Optional[str] = None,
            qstat: bool = True) -> dict:
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
            *jobs*: :class:`dict`
                Information on each job by ID number
        :Versions:
            * 2024-01-12 ``@ddalle``: v1.0
            * 2024-08-22 ``@ddalle``: v1.1; add *qstat* option
            * 2025-05-01 ``@ddalle``: v1.2; simplify flow
        """
        pass

    @abstractmethod
    def _get_qstat(self) -> queue.QStat:
        pass

   # --- CPU Stats ---
    # Get total CPU hours (actually core hours)
    @abstractmethod
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
        pass

  # *** CLI ***
   # --- Case loop ---
    # Loop through cases
    @abstractmethod
    def caseloop_verbose(
            self, casefunc: Optional[Callable] = None, **kw) -> int:
        r"""Loop through cases and produce verbose table

        :Call:
            >>> n = cntl.caseloop_verbose(casefunc, **kw)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *casefunc*: {``None``} | :class:`Callable`
                Optional function to run in each case folder
            *I*: {``None``} | :class:`np.ndarray`\ [:class:`int`]
                Case indices
        :Outputs:
            *n*: :class:`int`
                Number of cases started/submitted
        :Versions:
            * 2025-06-19 ``@ddalle``: v1.0
        """
        pass

    # Loop through cases
    @abstractmethod
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
        pass

    # Get value for aribtrary value, skipping progress
    @abstractmethod
    def _maxlen(self, opt: str, I: np.ndarray) -> int:
        pass

    # Get header for display column
    @abstractmethod
    def _header(self, opt: str) -> str:
        pass

   # --- Preprocess ---
    # CLI arg preprocesser
    @abstractmethod
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
        pass

   # --- Check ---
    # Function to display current status
    @abstractmethod
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
            * 2014-12-09 ``@ddalle``: v2.0; ``--cons``
            * 2025-06-20 ``@ddalle``: v3.0; use `caseloop_verbose()`
        """
        pass

   # --- Mark ---
    # Mark a case as PASS
    @abstractmethod
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
        pass

    # Mark a case as PASS
    @abstractmethod
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
        pass

    # Remove PASS and ERROR markers
    @abstractmethod
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
        pass

   # --- Execute script ---
    # Execute script
    @abstractmethod
    def ExecScript(self, **kw) -> int:
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
        pass

   # --- Cleanup ---
    # Function to clear out zombies
    @abstractmethod
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
        pass

   # --- Modify cases ---
    # Function to extend one or more cases
    @abstractmethod
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
        pass

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
    @abstractmethod
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
        pass

    # Extend a case
    @abstractmethod
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
        pass

   # --- Delete/Stop ---
    # Delete jobs
    @abstractmethod
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
        pass

    # Delete cases
    @abstractmethod
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
        pass

    # Function to delete a case folder: qdel and rm
    @abstractmethod
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
        pass

   # --- Batch ---
    # Write batch PBS job
    @abstractmethod
    def run_batch(self, argv: list) -> str:
        r"""Write and submit PBS/Slurm script for a CLI

        :Call:
            >>> jobid = cntl.run_batch(argv)
        :Inputs:
            *argv*: :class:`list`\ [:class:`str`]
                List of command-line inputs
        :Outputs:
            *jobid*: :class:`str`
                PBS/Slurm job number/ID
        :Versions:
            * 2024-12-20 ``@ddalle``: v1.0
        """
        pass

  # *** RUN MATRIX ***
   # --- Values ---
    # Get value for specified property
    @abstractmethod
    def getval(self, opt: str, i: int) -> Any:
        r"""Get run matrix or case status value for one case

        :Call:
            >>> v = cntl.getval(opt, i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *opt*: :class:`str`
                Name of option or run matrix key
            *i*: :class:`int`
                Case index
        :Outputs:
            *v*: :class:`Any`
                Value of option *opt* for case *i*
        :Versions:
            * 2025-06-16 ``@ddalle``: v1.0
        """
        pass

    # Get value, ensuring string output
    @abstractmethod
    def getvalstr(self, opt: str, i: int) -> str:
        r"""Get value of run matrix variable as string

        :Call:
            >>> txt = cntl.getvalstr(opt, i)
        :Inputs:
            *cntl*: :class:`cape.cfdx.cntl.Cntl`
                Overall CAPE control instance
            *opt*: :class:`str`
                Name of option or run matrix key
            *i*: :class:`int`
                Case index
        :Outputs:
            *txt*: :class:`str`
                Text of value of option *opt* for case *i*
        :Versions:
            * 2025-06-16 ``@ddalle``: v1.0
        """
        pass

   # --- Filter ---
    # Apply user filter
    @abstractmethod
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
        pass

   # --- Index ---
    # Get case index
    @abstractmethod
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
        pass

  # *** FILE MANAGEMENT ***
   # --- Files ---
    # Absolutize
    @abstractmethod
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
        pass

    # Copy files
    @abstractmethod
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
        pass

    # Link files
    @abstractmethod
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
        pass

   # --- Archiving ---
    # Function to archive results and remove files
    @abstractmethod
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
        pass

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
    @abstractmethod
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
        pass

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
    @abstractmethod
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
        pass

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

    # Unarchive cases
    @abstractmethod
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
        pass

  # *** LOGGING ***
    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

  # *** GARBAGE ***
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
            'QUEUE': 0,
            'ERROR': 0,
            'ERROR*': 0,
            'FAIL': 0,
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
