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
