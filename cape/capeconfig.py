r"""
:mod:`cape.capeconfig`: Interface to user-specific CAPE configuration
======================================================================

This file interacts with the settings file in either ``~/.capeconfig``
or ``$CAPE_CONFIGFILE``, as controlled by :mod:`cape.sysutils`. It
provides the class :class:`CapeConfig`. The purpose of this
configuration file is to set user-specific preferences and SSH paths.

For example, on Linux, the user may specify a preferred PDF reader
application. Users may also inform the user of their local workstation
so that commands like ``cape open-pdf $PDFFILE`` will first send them
from an HPC login node to a local workstation and open it there.
"""

# Standard library
import os
import re
import socket
from typing import Any, Optional

# Local imports
from .errors import CapeValueError
from .optdict import OptionsDict


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
        "CacheDir",
        "JumpHost",
        "LocalHost",
        "LocalHostPatterns",
        "PDFReader",
        "RemoteHost",
        "RemoteHostPatterns",
    )

    # Aliases
    _optmap = {
        "pdf": "PDFReader"
    }

    # Types
    _opttypes = {
        "CacheDir": str,
        "LocalHost": str,
        "LocalHostPatterns": str,
        "PDFReader": str,
        "RemoteHost": str,
        "RemoteHostPatterns": str,
    }

    # Required lists
    _optlistdepth = {
        "LocalHostPatterns": 1,
        "RemoteHostPatterns": 1,
    }

    # Environment variable
    _envvar = {
        "CacheDir": "CAPE_CACHE_DIR",
        "LocalHost": "CAPE_LOCAL_HOST",
        "PDFReader": "CAPE_PDF_READER",
        "RemoteHost": "CAPE_REMOTE_HOST",
    }

    # Sections
    _sec_cls = {
        "JumpHost": JumpHostConfig,
    }

    # Descriptions
    _rst_descriptions = {
        "CacheDir": "Location for CAPE to cache files",
        "LocalHost": (
            "Name of 'local' machine; CAPE on remote systems will transfer "
            "files to this location for easier viewing. Override with "
            "``$CAPE_LOCAL_HOST``"),
        "LocalHostPatterns": (
            "List of regexes to tell CAPE that current host is 'local.'"),
        "PDFReader": (
            "Preferred PDF reader. Override with ``$CAPE_PDF_READER``."),
        "RemoteHost": (
            "Remote host for ``cape receive-file`` to get files from. "
            "Override with ``$CAPE_REMOTE_HOST``."),
        "RemoteHostPatterns": (
            "List of regexes for host names to tell CAPE that current host "
            "is 'remote'; ``cape post-file`` and ``cape receive-file`` will "
            "not transfer files."),
    }

    # Get option (environment variable override)
    def get_opt(self, opt: str, vdef=None, **kw) -> Any:
        r"""Get value of an option with environment var override

        :Call:
            >>> v = opts.get_opt(opt, vdef=None)\
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


# Add getters and setters(
_properties = (
    "CacheDir",
    "LocalHost",
    "LocalHostPatterns",
    "PDFReader",
    "RemoteHost",
    "RemoteHostPatterns",
)
CapeConfig.add_properties(_properties)
