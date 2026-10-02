r"""
:mod:`cape.capeconfig`: Interface to user-specific CAPE configuration
======================================================================

This file reads the settings file in either ``~/.capeconfig.json``,
``$XDG_CONFIG_HOME/cape/config.json``, or ``$CAPE_CONFIG_FILE``, as
controlled by :mod:`cape.sysutils`. It provides the class
:class:`CapeConfig`. The purpose of this configuration file is to set
user-specific preferences and SSH paths.

For example, on Linux, the user may specify a preferred PDF reader
application. Users may also inform the user of their local workstation
so that commands like ``cape open-pdf $PDFFILE`` will first send them
from an HPC login node to a local workstation and open it there.
"""

# Standard library
import getpass
import json
import os
import platform
import re
import shutil
import socket
from typing import Any, Optional

# Local imports
from .errors import CapeValueError
from .optdict import OptionsDict


# Environment variable name
CONFIG_ENVVAR = "CAPE_CONFIG_FILE"

# Cache to avoid a global variable
CONFIG_CACHE = {}

# Name of current user
USER = getpass.getuser()


# Default PDF applications for Linux
DEFAULT_PDF_VIEWERS_LINUX = [
    "okular",
    "evince",
    "google-chrome",
    "firefox",
]

# Default PNG applications for Linux
DEFAULT_PNG_VIEWERS_LINUX = [
    "display",
    "ristretto",
    "gwenview",
    "eog",
    "loupe",
    "google-chrome",
    "firefox",
]


# Class for processing jumphost
class JumpHostConfig(OptionsDict):
    # No attributes
    __slots__ = ()

    # Types
    _opttypes = {
        "_default_": str,
    }


# Initialize class
class CapeConfig(OptionsDict):
    # No attributes
    __slots__ = ()

    # Allowed options
    _optlist = (
        "AgentHistoryFile",
        "CacheDir",
        "HistoryFile",
        "JumpHost",
        "LocalHost",
        "LocalHostPatterns",
        "PDFReader",
        "PNGReader",
        "RemoteHost",
        "RemoteHostPatterns",
        "RemoteLoginCommands",
        "StateDir",
        "TUIHistoryFile",
    )

    # Aliases
    _optmap = {
        "pdf": "PDFReader",
        "PDFViewer": "PDFReader",
    }

    # Types
    _opttypes = {
        "AgentHistoryFile": str,
        "CacheDir": str,
        "HistoryFile": str,
        "LocalHost": str,
        "LocalHostPatterns": str,
        "PDFReader": str,
        "PNGReader": str,
        "RemoteHost": str,
        "RemoteHostPatterns": str,
        "RemoteLoginCommands": str,
        "StateDir": str,
        "TUIHistoryFile": str,
    }

    # Required lists
    _optlistdepth = {
        "LocalHostPatterns": 1,
        "RemoteHostPatterns": 1,
        "RemoteLoginCommands": 1,
    }

    # Defaults
    _rc = {
        "AgentHistoryFile": ".cape_agent_history",
        "HistoryFile": ".cape_history",
        "TUIHistoryFile": ".cape_tui_history",
    }

    # Environment variable
    _envvar = {
        "AgentHistoryFile": "CAPE_AGENT_HISTORY_FILE",
        "CacheDir": "CAPE_CACHE_DIR",
        "HistoryFile": "CAPE_HISTORY_FILE",
        "LocalHost": "CAPE_LOCAL_HOST",
        "PDFReader": "CAPE_PDF_READER",
        "PNGReader": "CAPE_PNG_READER",
        "RemoteHost": "CAPE_REMOTE_HOST",
        "StateDir": "CAPE_STATE_DIR",
        "TUIHistoryFile": "CAPE_TUI_HISTORY_FILE",
    }

    # Sections
    _sec_cls = {
        "JumpHost": JumpHostConfig,
    }

    # Descriptions
    _rst_descriptions = {
        "AgentHistoryFile": (
            "Location for history of CAPE-agentic commands; relative "
            "paths resolve under *StateDir*"),
        "CacheDir": (
            "Location for CAPE to cache files. Override with "
            "``$CAPE_CACHE_DIR``. Defaults to ``$XDG_CACHE_HOME/cape`` "
            "if that's set to an absolute path, else ``~/.cache/cape``"),
        "HistoryFile": (
            "Location for history of CAPE commands; relative paths "
            "resolve under *StateDir*"),
        "TUIHistoryFile": (
            "Location for history of CAPE TUI commands; relative paths "
            "resolve under *StateDir*"),
        "StateDir": (
            "Location for CAPE state files such as command histories. "
            "Override with ``$CAPE_STATE_DIR``. Defaults to "
            "``$XDG_STATE_HOME/cape`` if that's set to an absolute "
            "path, else ``~/.local/state/cape``"),
        "LocalHost": (
            "Name of 'local' machine; CAPE on remote systems will transfer "
            "files to this location for easier viewing. Override with "
            "``$CAPE_LOCAL_HOST``"),
        "LocalHostPatterns": (
            "List of regexes to tell CAPE that current host is 'local.'"),
        "PDFReader": (
            "Preferred PDF reader. Override with ``$CAPE_PDF_READER``."),
        "PNGReader": (
            "Preferred PNG image viewer. Override with ``$CAPE_PNG_READER``."),
        "RemoteHost": (
            "Remote host for ``cape receive-file`` to get files from. "
            "Override with ``$CAPE_REMOTE_HOST``."),
        "RemoteHostPatterns": (
            "List of regexes for host names to tell CAPE that current host "
            "is 'remote'; ``cape post-file`` and ``cape receive-file`` will "
            "not transfer files."),
        "RemoteLoginCommands": (
            "List of commands to configure CAPE on *RemoteHost*. A common "
            'example for HPC might be ``"module load cape"``.'),
    }

    # Get option (environment variable override)
    def get_opt(self, opt: str, vdef=None, **kw) -> Any:
        r"""Get value of an option with environment var override

        :Call:
            >>> v = opts.get_opt(opt, vdef=None)
        :Inputs:
            *opts*: :class:`CapeConfig`
                Cape configuration options instance
            *vdef*: {``None``} | **any**
                Default value
        :Outputs:
            *v*: **any**
                Option value, with following order of importance

                1. Environment variable
                2. Value set in ``~/.capeconfig``
                3. Default value from class
                4. *vdef*
        """
        # Normalize option name
        fullopt = self.apply_optmap(opt)
        # Check for environment variable
        envvar = self._envvar.get(fullopt)
        # Get environment variable if one is defined
        if (envvar is not None) and (envvar in os.environ):
            return os.environ[envvar]
        # Otherwise revert to regular OptionsDict behavior
        return OptionsDict.get_opt(self, opt, vdef=vdef, **kw)

    # Special getter; get *JumpHost* for *this* machine
    def get_JumpHost(self) -> Optional[str]:
        r"""Get name of SSH JumpHost needed to transfer files

        The result is based on the name of the current host

        :Call:
            >>> jumphost = opts.get_JumpHost()
        :Inputs:
            *opts*: :class:`CapeConfig`
                Cape configuration options instance
        :Outputs:
            *jumphost*: ``None`` | :class:`str`
                Name of jump host, if any
        """
        # Get jumphost map
        jumphostmap = self.get("JumpHost", {})
        # Get local host name
        host = socket.gethostname()
        # Check hosts
        for lh, jh in jumphostmap.items():
            # Check for match
            try:
                if re.fullmatch(lh, host):
                    return jh
            except Exception as e:
                raise CapeValueError(
                    f"Invalid *JumpHost* regex in .capeconfig:\n  '{lh}'\n"
                    f"Original message: {e.args[0]}")

    # Special getter; get *PDFReader* wtih appropriate default
    def get_PDFReader(self) -> Optional[str]:
        # Get user preference
        viewer = self.get_opt("PDFReader")
        # Use it if specified
        if viewer is not None:
            return viewer
        # Get system
        system = platform.system()
        if system == "Windows":
            return "start"
        elif system == "Darwin":
            return "open"
        # For Linux, find best available
        for viewer in DEFAULT_PDF_VIEWERS_LINUX:
            if shutil.which(viewer) is not None:
                return viewer

    # Special getter; get *PNGReader* with appropriate default
    def get_PNGReader(self) -> Optional[str]:
        # Get user preference
        viewer = self.get_opt("PNGReader")
        # Use it if specified
        if viewer is not None:
            return viewer
        # Get system
        system = platform.system()
        if system == "Windows":
            return "start"
        elif system == "Darwin":
            return "open"
        # For Linux, find best available
        for viewer in DEFAULT_PNG_VIEWERS_LINUX:
            if shutil.which(viewer) is not None:
                return viewer

    # Check if this is a local host
    def check_local(self) -> bool:
        r"""Check if the current host is "local"

        :Call:
            >>> q = opts.check_local()
        :Inputs:
            *opts*: :class:`CapeConfig`
                Cape configuration options instance
        :Outputs:
            *q*: ``True`` | ``False``
                Whether current host is "local"
        """
        # Default to *LocalHost* if *LocalHostPatterns* not specified
        pat = self.get_opt("LocalHost")
        pat1 = [] if (pat is None) else [pat]
        # Get list of "local" host patterns
        pats = self.get_opt("LocalHostPatterns", vdef=pat1)
        # Get local host name
        host = socket.gethostname()
        # Check patterhs
        for lpat in pats:
            # Check for match
            try:
                if re.fullmatch(lpat, host):
                    return True
            except Exception as e:
                raise CapeValueError(
                    "Invalid *LocalHostPatterns* regex in .capeconfig:"
                    f"\n  '{lpat}'\n"
                    f"Original message: {e.args[0]}")
        # No matches
        return False

    # Check if this is a remote host
    def check_remote(self) -> bool:
        r"""Check if the current host is "remote"

        :Call:
            >>> q = opts.check_remote()
        :Inputs:
            *opts*: :class:`CapeConfig`
                Cape configuration options instance
        :Outputs:
            *q*: ``True`` | ``False``
                Whether current host is "remote"
        """
        return not self.check_local()


# Add getters and setters(
_properties = (
    "CacheDir",
    "LocalHost",
    "LocalHostPatterns",
    "RemoteHost",
    "RemoteHostPatterns",
    "StateDir",
)
CapeConfig.add_properties(_properties)
CapeConfig.add_setters(("PDFReader", "PNGReader"))


# Command-line interface
def show_cape_config(opt: str) -> str:
    r"""Get a CAPE user configuration option as a string

    :Call:
        >>> txt = show_cape_config(opt)
    :Inputs:
        *opt*: :class:`str`
            Name of option
    :Outputs:
        *txt*: :class:`str`
            Value of option
    """
    # Get option value
    v = get_cape_opt(opt)
    # Convert to string
    if isinstance(v, list):
        return ','.join([str(vj) for vj in v])
    elif isinstance(v, dict):
        return json.dumps(v)
    else:
        return str(v)


def get_cape_opt(opt: str):
    r"""Get a CAPE user configuration option

    :Call:
        >>> v = get_cape_opt(opt)
    :Inputs:
        *opt*: :class:`str`
            Name of option
    :Outputs:
        *v*: :class:`str` | :class:`list` | :class:`dict`
            Value of option
    """
    # Read config
    opts = read_cape_config()
    # Standardize option name
    fullopt = opts.apply_optmap(opt)
    # Check for special cases
    if fullopt == "CacheDir":
        return get_cape_cachedir()
    elif fullopt == "StateDir":
        return get_cape_statedir()
    elif fullopt == "PDFReader":
        return get_pdf_viewer()
    # Get value
    return opts.get_opt(opt)


def set_cape_opt(opt: str, v: Any, blend: bool = False):
    r"""Set (or blend) a CAPE user configuration option

    :Call:
        >>> set_cape_opt(opt, v, blend=False)
    :Inputs:
        *opt*: :class:`str`
            Name of option
        *v*: :class:`str`
            Value to set
        *blend*: ``True`` | {``False``}
            Combine user's option with existing for list | dict
    """
    # Read config
    opts = read_cape_config()
    # Convert value ...
    if opt in CapeConfig._optlistdepth:
        # Convert to list
        v = v.split(',')
        # Check blending option
        if blend:
            # Get existing value if possible
            v0 = opts.get(opt, [])
            v0.extend(v)
        else:
            # Overwrite
            v0 = v
        # Save
        opts[opt] = v0
    elif opt == "JumpHost":
        # Convert to dictionary
        if isinstance(v, str):
            v = json.loads(v)
        # Check blending option
        if blend:
            # Get current option
            v0 = opts.get(opt, {})
            # Combine
            v0.update(v)
        else:
            # Overwrite
            v0 = v
        # Save
        opts[opt] = v0
    elif "." in opt:
        # Get parts
        sec, subopt = opt.split('.', 1)
        # Get current option
        vsec = opts.setdefault(sec, {})
        vsec[subopt] = v
    else:
        # Save option
        opts[opt] = v
    # Write updated uption
    opts.write_jsonfile(get_cape_configfile())


# Get default cache dir
def get_default_cachedir() -> str:
    r"""Get the default location for CAPE's cache folder

    This follows the XDG Base Directory Specification: if
    ``$XDG_CACHE_HOME`` is set to an absolute path, the default is
    ``$XDG_CACHE_HOME/cape``; otherwise it is ``~/.cache/cape``.

    :Call:
        >>> cachedir = get_default_cachedir()
    :Outputs:
        *cachedir*: :class:`str`
            Default location for CAPE to cache files
    """
    # Get XDG base cache folder
    xdg_cachedir = os.environ.get("XDG_CACHE_HOME", "")
    # Only absolute paths are valid according to the XDG spec
    if not os.path.isabs(xdg_cachedir):
        xdg_cachedir = os.path.join("~", ".cache")
    # Append CAPE-specific subfolder
    return os.path.join(xdg_cachedir, "cape")


# Get cache dir
def get_cape_cachedir() -> str:
    r"""Get (and create) cache folder

    The order of precedence is

    1.  The environment variable ``$CAPE_CACHE_DIR``
    2.  The *CacheDir* setting in ``~/.capeconfig.json``
    3.  The folder ``$XDG_CACHE_HOME/cape`` if ``$XDG_CACHE_HOME`` is
        set to an absolute path
    4.  The global default ``~/.cache/cape/``

    :Call:
        >>> cachedir = get_cape_cachedir()
    :Outputs:
        *cachedir*: :class:`str`
            Location of cache to use for temporary CAPE files
    """
    # Read config
    opts = read_cape_config()
    # Get CacheDir setting
    cachedir = opts.get_opt("CacheDir")
    # Use XDG-based default if none of those are set
    if cachedir is None:
        cachedir = get_default_cachedir()
    # Expand '~'
    cachedir = os.path.expanduser(cachedir)
    # Create it if necessary
    try:
        os.makedirs(cachedir, exist_ok=True)
    except PermissionError:
        pass
    # Output
    return cachedir


# Get default state dir
def get_default_statedir() -> str:
    r"""Get the default location for CAPE's state folder

    This follows the XDG Base Directory Specification: if
    ``$XDG_STATE_HOME`` is set to an absolute path, the default is
    ``$XDG_STATE_HOME/cape``; otherwise it is ``~/.local/state/cape``.

    :Call:
        >>> statedir = get_default_statedir()
    :Outputs:
        *statedir*: :class:`str`
            Default location for CAPE state files
    """
    # Get XDG base state folder
    xdg_statedir = os.environ.get("XDG_STATE_HOME", "")
    # Only absolute paths are valid according to the XDG spec
    if not os.path.isabs(xdg_statedir):
        xdg_statedir = os.path.join("~", ".local", "state")
    # Append CAPE-specific subfolder
    return os.path.join(xdg_statedir, "cape")


# Get state dir
def get_cape_statedir() -> str:
    r"""Get (and create) folder for CAPE state files

    The order of precedence is

    1.  The environment variable ``$CAPE_STATE_DIR``
    2.  The *StateDir* setting in ``~/.capeconfig.json``
    3.  The folder ``$XDG_STATE_HOME/cape`` if ``$XDG_STATE_HOME`` is
        set to an absolute path
    4.  The global default ``~/.local/state/cape/``

    :Call:
        >>> statedir = get_cape_statedir()
    :Outputs:
        *statedir*: :class:`str`
            Location of folder for CAPE state files such as histories
    """
    # Read config
    opts = read_cape_config()
    # Get StateDir setting
    statedir = opts.get_opt("StateDir")
    # Use XDG-based default if none of those are set
    if statedir is None:
        statedir = get_default_statedir()
    # Expand '~'
    statedir = os.path.expanduser(statedir)
    # Create it if necessary
    try:
        os.makedirs(statedir, exist_ok=True)
    except PermissionError:
        pass
    # Output
    return statedir


# Get a CAPE history file
def get_cape_histfile(opt: str = "HistoryFile") -> str:
    r"""Get full path to a CAPE history file

    Values that are relative paths (including the defaults) are
    resolved under the CAPE *StateDir*; see :func:`get_cape_statedir`.

    :Call:
        >>> histfile = get_cape_histfile(opt="HistoryFile")
    :Inputs:
        *opt*: {``"HistoryFile"``} | :class:`str`
            Name of history-file option, e.g. ``"AgentHistoryFile"``
    :Outputs:
        *histfile*: :class:`str`
            Full path to history file
    """
    # Read config
    opts = read_cape_config()
    # Get option value, expanding '~' if present
    histfile = os.path.expanduser(opts.get_opt(opt))
    # Join relative paths with *StateDir*
    if not os.path.isabs(histfile):
        histfile = os.path.join(get_cape_statedir(), histfile)
    # Output
    return histfile


# Get *JumpHost*
def get_cape_jumphost() -> Optional[str]:
    r"""Get *JumpHost* for SSH to reach intended target, based on $HOST

    :Call:
        >>> jumphost = get_cape_jumphost()
    :Outputs:
        *jumphost*: ``None`` | :class:`str`
            Name of SSH JumpHost needed, if needed
    """
    # Read config
    opts = read_cape_config()
    # Get *JumpHost*
    return opts.get_JumpHost()


# Get preferred PDF viewer
def get_pdf_viewer() -> str:
    r"""Get the preferred PDF viewer application based on system

    :Call:
        >>> viewer = get_pdf_viewer()
    :Outputs:
        *viewer*: :class:`str`
            Name of application to open PDF
    :Versions:
        * 2026-08-07 ``@ddalle``: v1.0
    """
    # Read config
    opts = read_cape_config()
    # Get *PDFReader*
    return opts.get_PDFReader()


# Get preferred PNG viewer
def get_png_viewer() -> str:
    r"""Get the preferred PNG viewer application based on system

    :Call:
        >>> viewer = get_png_viewer()
    :Outputs:
        *viewer*: :class:`str`
            Name of application to open PNG
    :Versions:
        * 2026-08-31 ``@ddalle``: v1.0
    """
    # Read config
    opts = read_cape_config()
    # Get *PNGReader*
    return opts.get_PNGReader()


# Check if local
def check_cape_local() -> bool:
    r"""Check if the current host is "local"

    :Call:
        >>> q = check_cape_local()
    :Outputs:
        *q*: ``True`` | ``False``
            Whether current host is "local"
    """
    # Read config
    opts = read_cape_config()
    # Check local
    return opts.check_local()


# Check fi remote
def check_cape_remote() -> bool:
    r"""Check if the current host is "remote"

    :Call:
        >>> q = check_cape_remote()
    :Outputs:
        *q*: ``True`` | ``False``
            Whether current host is "remote"
    """
    # Read config
    opts = read_cape_config()
    # Check remote
    return opts.check_remote()


# Read config file
def read_cape_config() -> CapeConfig:
    r"""Read config from location given by :func:`get_cape_configfile`

    If no config file exists, an empty one is created at that
    location (``~/.capeconfig.json`` or the XDG location
    ``$XDG_CONFIG_HOME/cape/config.json``).

    :Call:
        >>> opts = read_cape_config()
    :Outputs:
        *opts*: :class:`CapeConfig`
            CAPE user settings
    """
    # Check cache
    if USER in CONFIG_CACHE:
        return CONFIG_CACHE[USER]
    # Get config file
    configfile = get_cape_configfile()
    # Check if file exists
    if os.path.isfile(configfile):
        opts = CapeConfig(configfile)
    else:
        # Create initial instance
        opts = CapeConfig()
        # Write one
        try:
            # Create parent folder if necessary
            configdir = os.path.dirname(configfile)
            if configdir:
                os.makedirs(configdir, exist_ok=True)
            opts.write_jsonfile(configfile)
        except PermissionError:
            pass
    # Cache this instance
    CONFIG_CACHE[USER] = opts
    # Output
    return opts


# Get default config file
def get_default_configfile() -> str:
    r"""Get the default location for CAPE's config file

    This follows the XDG Base Directory Specification: if
    ``$XDG_CONFIG_HOME`` is set to an absolute path, the default is
    ``$XDG_CONFIG_HOME/cape/config.json``; otherwise it is
    ``~/.config/cape/config.json``.

    :Call:
        >>> configfile = get_default_configfile()
    :Outputs:
        *configfile*: :class:`str`
            Default location of CAPE config file
    """
    # Get XDG base config folder
    xdg_configdir = os.environ.get("XDG_CONFIG_HOME", "")
    # Only absolute paths are valid according to the XDG spec
    if not os.path.isabs(xdg_configdir):
        xdg_configdir = os.path.join("~", ".config")
    # Append CAPE-specific subfolder and file name
    return os.path.join(xdg_configdir, "cape", "config.json")


# Get current configuration file
def get_cape_configfile() -> str:
    r"""Get the location of CAPE's config file

    The order of precedence is

    1.  The environment variable ``$CAPE_CONFIG_FILE``
    2.  The legacy location ``~/.capeconfig.json``, if it exists
    3.  The file ``$XDG_CONFIG_HOME/cape/config.json`` if
        ``$XDG_CONFIG_HOME`` is set to an absolute path
    4.  The default ``~/.config/cape/config.json``

    :Call:
        >>> configfile = get_cape_configfile()
    :Outputs:
        *configfile*: :class:`str`
            Location of CAPE config file
    """
    # Check for environment variable
    configfile = os.environ.get(CONFIG_ENVVAR)
    # Environment variable wins if present
    if configfile:
        return os.path.expanduser(configfile)
    # Check for legacy config file
    legacy_configfile = os.path.expanduser(
        os.path.join("~", ".capeconfig.json"))
    if os.path.isfile(legacy_configfile):
        return legacy_configfile
    # Use XDG location; file gets created there if necessary
    return os.path.expanduser(get_default_configfile())

