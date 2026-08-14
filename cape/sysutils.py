r"""
:mod:`cape.sysutils`: System utilities for using CAPE
======================================================

This module provides various "system" utilities such as providing a
universal Python method to open a PDF file for viewing.
"""

# Standard library
import os
import platform
import shutil
from subprocess import Popen, PIPE

# Local imports
from .errors import CapeFileNotFoundError


# Default PDF applications for Linux
DEFAULT_PDF_VIEWERS_LINUX = [
    "okular",
    "evince",
    "google-chrome",
    "firefox",
]

# Cache file name
CACHE_FILE = "tmp.tar.gz"


# Get current CACHE location
def get_cape_cachedir() -> str:
    r"""Get cache location for temp CAPE files

    Defaults to ``~/.cache/cape/``; override with ``$CAPE_CACHE_DIR``
    environment variable.

    :Call:
        >>> dirname = get_cape_cachedir()
    :Outputs:
        *dirname*: :class:`str`
            Name of folder; creates folder if necessary
    """
    # Default cache dir
    cachedir = os.path.expanduser("~/.cache/cape")
    # Check for environment variable
    cachedir = os.environ.get("CAPE_CACHE_DIR", cachedir)
    # Create it if necessary
    os.makedirs(cachedir, exist_ok=True)
    # Output
    return cachedir


# Get current configuration folder
def get_cape_configfile() -> str:
    ...


# Post file(s)
def post_file(pat1: str, *pats, v: bool = False) -> int:
    # Get [and create] cache dir
    cachedir = get_cape_cachedir()
    # Check verbose option
    flag = "czvf" if v else "czf"
    # Command to tar up requested file(s)
    cmdlist = ["tar", flag, os.path.join(cachedir, CACHE_FILE), pat1]
    cmdlist.extend(pats)
    # Run the command
    proc = Popen(cmdlist, stderr=PIPE)
    # Wait for command
    proc.communicate()
    # Check error status
    if proc.returncode:
        print(f"Failed to post file(s) '{pat1} {' '.join(pats)}'")
    # Exit code
    return proc.returncode


# Receive file(s)
def receive_file() -> list:
    ...


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


# Open a PDF
def open_pdf(fname: str, wait: bool = False) -> Popen:
    r"""Open a PDF file if found

    :Call:
        >>> open_pdf(fname, wait=False)
    :Inputs:
        *fname*: :class:`str`
            Name of file to open
        *wait*: ``True`` | {``False``}
            Option to wait until PDF is closed
    :Output:
        *proc*: :class:`subprocess.Popen`
            Subprocess handle
    :Versions:
        * 2026-08-07 ``@ddalle``: v1.0
    """
    # Check for file
    if not os.path.isfile(fname):
        raise CapeFileNotFoundError(f"No file '{fname}'")
    # Get viewer
    viewer = get_pdf_viewer()
    # Command to open it
    proc = Popen([viewer, fname], stdout=PIPE, stderr=PIPE)
    # Wait option
    if wait:
        proc.wait()
    # Return subprocess handle
    return proc
