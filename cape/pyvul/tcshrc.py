r"""
:mod:`cape.pyvul.tcshrc`: VULCAN-CFD tcsh startup files
=========================================================

The VULCAN-CFD ``vulcan`` executable is a ``tcsh`` script that calls
further ``tcsh`` scripts and shell aliases (``run_vulcan_int``,
``apart_vulcan``, ``pprep_vulcan``, ...) that are normally defined only
when the ``vulcan`` environment module is loaded into an interactive
shell.  Because ``LOADEDMODULES`` is exported to subprocesses, a child
``tcsh`` cannot recreate the aliases by loading the module again (the
load is a no-op), so the aliases must be provided through a shell
startup file.

CAPE accomplishes this portably:

1. Each case folder contains ``vulcan.tcshrc``, a ``tcsh`` startup
   file that defines the aliases found in the active ``vulcan``
   modulefile.  It is written by
   :func:`cape.pyvul.cntl.Cntl.PrepareCase` and rewritten by
   :class:`cape.pyvul.casecntl.CaseRunner` before each run, when the
   VULCAN environment is guaranteed to be loaded.
2. The user's ``~/.tcshrc`` sources the file named by the
   ``CAPE_TCSHRC`` environment variable, if present:

   .. code-block:: csh

       if ( $?CAPE_TCSHRC ) then
           source "$CAPE_TCSHRC"
       endif

   which :func:`update_user_tcshrc` appends to ``~/.tcshrc`` (or
   creates the file) during :func:`cape.pyvul.cntl.PrepareCase`.
3. :meth:`cape.pyvul.casecntl.CaseRunner.run_vulcan` exports
   ``CAPE_TCSHRC`` (along with ``CAPE_NUM_CPUS``, ``CAPE_VULCAN_CMND``,
   and ``VULCAN_ROOT``) in the environment of the ``vulcan`` call, so
   that every ``tcsh`` subprocess picks up the case's aliases.

"""

# Standard library
import glob
import os
import re
import shutil
from typing import Optional

# Name of the case-local tcsh startup file
TCSHRC_NAME = "vulcan.tcshrc"

# Environment variable that points ``~/.tcshrc`` at the case file
CAPE_TCSHRC_VAR = "CAPE_TCSHRC"

# Snippet added to the user's ``~/.tcshrc``
TCSHRC_SNIPPET = (
    'if ( $?%s ) then\n' % CAPE_TCSHRC_VAR +
    '    source "$%s"\n' % CAPE_TCSHRC_VAR +
    'endif\n')

# Detect the snippet (or an equivalent) in a ``~/.tcshrc`` file
REGEX_TCSHRC_SNIPPET = re.compile(r"if\s*\(\s*\$\?CAPE")

# Loaded-modules names, e.g. ``vulcan/2026-09-21``
REGEX_LOADED_VULCAN = re.compile(r"vulcan(?:/(?P<version>[^:]*))?$")

# Tcl ``set`` and ``set-alias`` commands in a modulefile
REGEX_TCL_SET = re.compile(
    r"^set\s+(?P<name>[A-Za-z_]\w*)\s+(?P<value>\S.*?)\s*$")
REGEX_TCL_SET_ALIAS = re.compile(
    r"^set-alias\s+(?P<name>\S+)\s+(?P<value>\{.*\}|\".*\"|\S.*?)\s*$")

# Tcl variable references, e.g. ``$root`` or ``${vulcanpath}``
REGEX_TCL_VAR = re.compile(r"\$\{?([A-Za-z_]\w*)\}?")


# Expand Tcl variables (from modulefile ``set`` commands) in a string
def _expand_tcl(value: str, tclvars: dict) -> str:
    r"""Substitute modulefile ``set`` variables into a string

    Unknown variables (e.g. ``$num_cpus``, which ``tcsh`` defines at
    alias use time) are left alone.

    :Call:
        >>> value = _expand_tcl(value, tclvars)
    :Inputs:
        *value*: :class:`str`
            String that may contain Tcl variable references
        *tclvars*: :class:`dict`
            Values of modulefile ``set`` variables
    :Outputs:
        *value*: :class:`str`
            String with known variables expanded
    """
    for _ in range(8):
        new = REGEX_TCL_VAR.sub(
            lambda m: tclvars.get(m.group(1), m.group(0)), value)
        if new == value:
            break
        value = new
    return value


# Find the active ``vulcan`` modulefile
def find_vulcan_modulefile(environ: Optional[dict] = None) -> Optional[str]:
    r"""Locate the modulefile for the current ``vulcan`` module

    The search uses the ``VULCAN_MODULEFILE`` environment variable if
    set, then the version from ``LOADEDMODULES`` resolved against
    ``MODULEPATH``, and finally the most recent ``vulcan/*`` file in
    ``MODULEPATH``.

    :Call:
        >>> fmodule = find_vulcan_modulefile(environ=None)
    :Inputs:
        *environ*: {``None``} | :class:`dict`
            Environment to search, defaults to :class:`os.environ`
    :Outputs:
        *fmodule*: :class:`str` | ``None``
            Name of modulefile found, or ``None`` if none exists
    """
    # Select environment
    env = os.environ if environ is None else environ
    # Direct override
    fmodule = env.get("VULCAN_MODULEFILE")
    if fmodule and os.path.isfile(fmodule):
        return fmodule
    # Module folders
    moddirs = [d for d in env.get("MODULEPATH", "").split(os.pathsep) if d]
    # Versions loaded in this environment
    versions = []
    for modname in env.get("LOADEDMODULES", "").split(":"):
        mtch = REGEX_LOADED_VULCAN.match(modname.strip())
        if mtch and mtch.group("version"):
            versions.append(mtch.group("version"))
    # Look for the loaded version(s)
    for version in versions:
        for d in moddirs:
            fmodule = os.path.join(d, "vulcan", version)
            if os.path.isfile(fmodule):
                return fmodule
    # Fall back to the most recent modulefile found
    candidates = []
    for d in moddirs:
        candidates += [
            f for f in glob.glob(os.path.join(d, "vulcan", "*"))
            if os.path.isfile(f)
        ]
    return max(candidates) if candidates else None


# Read aliases from a ``vulcan`` modulefile
def read_vulcan_aliases(
        fmodule: str,
        environ: Optional[dict] = None) -> list:
    r"""Convert modulefile ``set-alias`` commands to tcsh aliases

    Tcl variables set by top-level ``set`` commands in the modulefile
    (``$root``, ``$vulcanpath``, ``$tecplot_tools``, ...) are expanded.
    The shell flavor ``$vulcanshell`` is taken from the ``VULCANSHELL``
    environment variable, defaulting to ``"tcsh"``.  Variables that
    VULCAN defines at run time (``$num_cpus``, ``$vulcan_cmnd``,
    ``$ofn``) are preserved for ``tcsh`` to resolve when the alias is
    used.

    :Call:
        >>> aliases = read_vulcan_aliases(fmodule, environ=None)
    :Inputs:
        *fmodule*: :class:`str`
            Name of the ``vulcan`` modulefile to parse
        *environ*: {``None``} | :class:`dict`
            Environment for ``VULCANSHELL``, defaults to
            :class:`os.environ`
    :Outputs:
        *aliases*: :class:`list`\\ [\\ :class:`tuple`\\ (:class:`str`,
            :class:`str`)\\ ]
            Alias name/value pairs in modulefile order
    """
    # Select environment
    env = os.environ if environ is None else environ
    # Tcl variables from top-level ``set`` commands
    tclvars = {}
    # Aliases in definition order (later definitions win)
    aliases = {}
    # Tcl brace depth; ``set`` commands inside ``if`` blocks are
    # conditional and are not tracked
    depth = 0
    with open(fmodule) as f:
        for rawline in f:
            line = rawline.strip()
            if line and not line.startswith("#"):
                # ``set-alias`` is collected at any depth
                mtch = REGEX_TCL_SET_ALIAS.match(line)
                if mtch:
                    value = mtch.group("value")
                    # Strip Tcl quoting
                    if (value.startswith("{") and value.endswith("}")) or (
                            value.startswith('"') and value.endswith('"')):
                        value = value[1:-1]
                    aliases[mtch.group("name")] = value
                # Only unconditional ``set`` commands are tracked
                elif depth == 0:
                    mtch = REGEX_TCL_SET.match(line)
                    if mtch:
                        tclvars[mtch.group("name")] = mtch.group("value")
            # Update Tcl depth
            depth += rawline.count("{") - rawline.count("}")
    # Shell flavor used by aliases like ``apart_vulcan.$vulcanshell``
    tclvars.setdefault("vulcanshell", env.get("VULCANSHELL", "tcsh"))
    # Expand modulefile variables
    aliases = {k: _expand_tcl(v, tclvars) for k, v in aliases.items()}
    # Output
    return list(aliases.items())


# Get the VULCAN install root
def get_vulcan_root(environ: Optional[dict] = None) -> Optional[str]:
    r"""Get the VULCAN install root folder

    This is the folder that contains ``Executable``, ``Scripts``,
    ``Data_base``, and so on.

    :Call:
        >>> vroot = get_vulcan_root(environ=None)
    :Inputs:
        *environ*: {``None``} | :class:`dict`
            Environment to search, defaults to :class:`os.environ`
    :Outputs:
        *vroot*: :class:`str` | ``None``
            Path of the VULCAN root folder, or ``None`` if unknown
    """
    # Select environment
    env = os.environ if environ is None else environ
    # Check environment variables set by the module
    for key in ("VULCAN_ROOT", "vulcanpath", "version_path"):
        vroot = env.get(key)
        if vroot:
            if key == "version_path":
                vroot = os.path.join(vroot, "Vulcan")
            return vroot
    # Fall back to the ``vulcan`` script on the path
    exe = shutil.which("vulcan")
    if exe:
        fdir = os.path.dirname(os.path.realpath(exe))
        if os.path.basename(fdir) == "Scripts":
            return os.path.dirname(fdir)
    # Nothing found
    return None


# Generate the text of a case-local ``vulcan.tcshrc`` file
def vulcan_tcshrc_text(
        fmodule: Optional[str] = None,
        environ: Optional[dict] = None) -> str:
    r"""Create contents of a ``vulcan.tcshrc`` startup file

    :Call:
        >>> txt = vulcan_tcshrc_text(fmodule=None, environ=None)
    :Inputs:
        *fmodule*: {``None``} | :class:`str`
            Modulefile to read; defaults to
            :func:`find_vulcan_modulefile`
        *environ*: {``None``} | :class:`dict`
            Environment to search, defaults to :class:`os.environ`
    :Outputs:
        *txt*: :class:`str`
            Text of the ``tcsh`` startup file
    """
    # Select environment
    env = os.environ if environ is None else environ
    # Find the modulefile if necessary
    if fmodule is None:
        fmodule = find_vulcan_modulefile(environ=env)
    # Start the file
    txt = (
        "#!/bin/tcsh\n"
        "#\n"
        "# Written by CAPE (cape.pyvul); do not edit.\n"
        "# Defines the VULCAN-CFD tcsh aliases for the tcsh\n"
        "# subprocesses launched by the 'vulcan' command.\n"
    )
    # Check for a usable modulefile
    if not fmodule or not os.path.isfile(fmodule):
        return txt + (
            "#\n"
            "# WARNING: No 'vulcan' modulefile was found, so no\n"
            "# aliases are defined.  Load the vulcan module and\n"
            "# run the case again.\n"
        )
    # Read the aliases
    aliases = read_vulcan_aliases(fmodule, environ=env)
    if not aliases:
        return txt + (
            "#\n"
            f"# WARNING: No aliases found in '{fmodule}'.\n"
        )
    # Write the aliases
    txt += f"#\n# Modulefile: {fmodule}\n#\n"
    for name, value in aliases:
        # Protect any single quotes from the tcsh alias quoting
        value = value.replace("'", r"'\''")
        txt += f"alias {name} '{value}'\n"
    # Output
    return txt


# Write the case-local ``vulcan.tcshrc`` file
def write_vulcan_tcshrc(
        dirname: str = ".",
        environ: Optional[dict] = None) -> str:
    r"""Write ``vulcan.tcshrc`` into a case folder

    :Call:
        >>> fname = write_vulcan_tcshrc(dirname=".", environ=None)
    :Inputs:
        *dirname*: :class:`str`
            Folder in which to write ``vulcan.tcshrc``
        *environ*: {``None``} | :class:`dict`
            Environment to search, defaults to :class:`os.environ`
    :Outputs:
        *fname*: :class:`str`
            Full name of the file written
    """
    # File name
    fname = os.path.join(os.path.abspath(dirname), TCSHRC_NAME)
    # Generate and write
    txt = vulcan_tcshrc_text(environ=environ)
    with open(fname, "w") as f:
        f.write(txt)
    # Output
    return fname


# Build the environment for a VULCAN call
def get_vulcan_env(
        dirname: str = ".",
        nproc: Optional[int] = None,
        environ: Optional[dict] = None) -> dict:
    r"""Create the environment for running VULCAN-CFD

    :Call:
        >>> env = get_vulcan_env(dirname=".", nproc=None, environ=None)
    :Inputs:
        *dirname*: :class:`str`
            Case folder containing ``vulcan.tcshrc``
        *nproc*: {``None``} | :class:`int`
            Number of CPUs to request, written to ``CAPE_NUM_CPUS``
        *environ*: {``None``} | :class:`dict`
            Base environment, defaults to a copy of
            :class:`os.environ`
    :Outputs:
        *env*: :class:`dict`
            Environment for :func:`cape.cfdx.cmdrun.callf`
    """
    # Copy the base environment
    env = dict(os.environ if environ is None else environ)
    # Point ``~/.tcshrc`` at the case's aliases
    env[CAPE_TCSHRC_VAR] = os.path.join(os.path.abspath(dirname), TCSHRC_NAME)
    # Number of CPUs for this run
    if nproc is not None:
        env["CAPE_NUM_CPUS"] = str(int(nproc))
    # VULCAN locations
    vroot = get_vulcan_root(environ=environ)
    if vroot:
        env["VULCAN_ROOT"] = vroot
        env["CAPE_VULCAN_CMND"] = os.path.join(vroot, "Executable", "vulcan")
    # Output
    return env


# Install the ``CAPE_TCSHRC`` source line in ``~/.tcshrc``
def update_user_tcshrc(fname: Optional[str] = None) -> bool:
    r"""Ensure the user's ``~/.tcshrc`` sources ``$CAPE_TCSHRC``

    The following lines are appended to (or used to create) the
    startup file unless an equivalent ``if ( $?CAPE`` line is already
    present:

    .. code-block:: csh

        if ( $?CAPE_TCSHRC ) then
            source "$CAPE_TCSHRC"
        endif

    :Call:
        >>> didit = update_user_tcshrc(fname=None)
    :Inputs:
        *fname*: {``None``} | :class:`str`
            Name of startup file, defaults to ``~/.tcshrc``
    :Outputs:
        *didit*: :class:`bool`
            ``True`` if the file was created or modified
    """
    # Default file name
    if fname is None:
        fname = os.path.join(os.path.expanduser("~"), ".tcshrc")
    # Read any existing contents
    txt = ""
    if os.path.isfile(fname):
        with open(fname) as f:
            txt = f.read()
    # Check for an existing hook
    if REGEX_TCSHRC_SNIPPET.search(txt):
        return False
    # Append the snippet
    if txt and not txt.endswith("\n"):
        txt += "\n"
    with open(fname, "w") as f:
        f.write(txt + "\n" + TCSHRC_SNIPPET)
    # Report the modification
    return True
