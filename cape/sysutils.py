r"""
:mod:`cape.sysutils`: System utilities for using CAPE
======================================================

This module provides various "system" utilities such as providing a
universal Python method to open a PDF file for viewing.
"""

# Standard library
import glob
import os
import tarfile
from subprocess import Popen, run, DEVNULL, PIPE
from typing import Optional, Tuple

# Local imports
from . import capeconfig
from .errors import CapeFileNotFoundError, CapeValueError

# Cache file name
CACHE_FILE = "tmp.tar.gz"


# Post file(s)
def post_file(pat1: str, *pats, v: bool = False) -> Tuple[int, list]:
    r"""Collect file(s) in tar ball; send to *RemoteHost* if necessary

    :Call:
        >>> ierr, filenames = post_file(pat1, *pats)
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *filenames*: :class:`list`\ [:class:`str`]
            List of posted file names
    """
    # Get [and create] cache dir
    cachedir = capeconfig.get_cape_cachedir()
    # Path to tar file
    tarpath = os.path.join(cachedir, CACHE_FILE)
    # Combine all patterns
    patterns = [pat1] + list(pats)
    # Create tar.gz file
    try:
        # Create tar file
        with tarfile.open(tarpath, "w:gz") as tar:
            # Full list of file names
            filenames = []
            # Loop through file name patterns
            for pattern in patterns:
                # Expand glob pattern
                matches = glob.glob(pattern)
                # Check if pattern matched anything
                if not matches:
                    print(f"Warning: pattern '{pattern}' matched no files")
                    continue
                # Add each matched file to archive
                for filepath in matches:
                    if v:
                        print(f"Adding: {filepath}")
                    # Add file to tarball
                    tar.add(filepath)
                    # Add to output
                    filenames.append(filepath)
        # Send the file
        _send_file(tarpath)
        # Return code
        return 0, filenames
    except Exception as e:
        print(f"Failed to post file(s) '{pat1} {' '.join(pats)}': {e}")
        return 1, []


# Receive file(s)
def receive_file() -> list:
    r"""Receive the current cached tarball

    :Call:
        >>> filenames = receive_file()
    :Outputs:
        *filenames*: :class:`list`\ [:class:`str`]
            List of unpacked files
    """
    # Get cache dir
    cachedir = capeconfig.get_cape_cachedir()
    # Path to tar file
    tarpath = os.path.join(cachedir, CACHE_FILE)
    # Receive file if necessary
    if capeconfig.check_cape_local():
        # Get remote host and config commands
        remote_host = capeconfig.get_cape_opt("RemoteHost")
        remote_cmds = capeconfig.get_cape_opt("RemoteLoginCommands")
        # Validate
        if remote_host is None:
            raise CapeValueError("No CAPE 'RemoteHost' setting found")
        # Format environment prep commands
        if remote_cmds:
            remote_env = '; '.join(remote_cmds) + '; '
        else:
            remote_env = ''
        # Remote command
        remote_cmd = (
            f'{remote_env}D=$(cape get-config CacheDir); '
            f'cat "$D/{CACHE_FILE}"')
        # Open local file to pipe into STDIN
        with open(tarpath, 'wb') as fp:
            # Run SSH and receive file there
            proc = run(['ssh', remote_host, remote_cmd], stdout=fp)
        # Check status
        if proc.returncode:
            raise CapeFileNotFoundError(
                f"Failed to receive {CACHE_FILE} from {remote_host}")
    # Check if tar file exists
    if not os.path.isfile(tarpath):
        return []
    # List to hold extracted file names
    extracted_files = []
    # Open and extract tar file
    with tarfile.open(tarpath, "r:gz") as tar:
        # Get list of members
        members = tar.getmembers()
        # Extract all files to cache dir
        tar.extractall(path=os.getcwd(), filter='data')
        # Collect file names
        extracted_files = [member.name for member in members]
    # Return list of extracted files
    return extracted_files


# Send file to remote host
def _send_file(localfile: Optional[str] = None):
    # Check if already remote
    if capeconfig.check_cape_remote():
        return
    # Default file name is the CACHE file
    if localfile is None:
        localfile = os.path.join(capeconfig.get_cape_cachedir(), CACHE_FILE)
    # Check for local file
    if not os.path.isfile(localfile):
        raise CapeFileNotFoundError(f"No file '{localfile}'")
    # Get remote host and config commands
    remote_host = capeconfig.get_cape_opt("RemoteHost")
    remote_cmds = capeconfig.get_cape_opt("RemoteLoginCommands")
    # Validate
    if remote_host is None:
        raise CapeValueError("No CAPE 'RemoteHost' setting found")
    # Format environment prep commands
    if remote_cmds:
        remote_env = '; '.join(remote_cmds) + '; '
    else:
        remote_env = ''
    # Base name of file (for destination to mimic)
    basename = os.path.basename(localfile)
    # Remote command
    remote_cmd = (
        f'{remote_env}D=$(cape get-config CacheDir); '
        f'cat > "$D/{basename}"')
    # Open local file to pipe into STDIN
    with open(localfile, 'rb') as fp:
        # Run SSH and receive file there
        run(['ssh', remote_host, remote_cmd], stdin=fp, check=True)


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
    return capeconfig.get_pdf_viewer()


# Open a PDF
def open_pdf(
        fname: str,
        wait: bool = False,
        local: Optional[bool] = None) -> Popen:
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
    # Check local option
    if local or capeconfig.check_cape_local():
        return _open_pdf_local(fname, wait)
    # Get name of "local" host to push file to
    local_host = capeconfig.get_cape_opt("LocalHost")
    # Fall-back if not specified
    if local_host is None:
        return _open_pdf_local(fname, wait)
    # Check if SSH jump-host is needed
    jump_host = capeconfig.get_cape_jumphost()
    # Commands to initialzie environment there
    remote_cmds = capeconfig.get_cape_opt("RemoteLoginCommands")
    # Format environment prep commands
    if remote_cmds:
        remote_env = '; '.join(remote_cmds) + '; '
    else:
        remote_env = ''
    # Format jumphost command
    if jump_host:
        # Include `-J` step
        base_cmd = ["ssh", "-J", jump_host, local_host]
    else:
        # Direct SSH login
        base_cmd = ["ssh", local_host]
    # Base name of file (for destination to mimic)
    basename = os.path.basename(fname)
    # Special command to detach remote app from SSH
    setsid = "" if wait else "setsid "
    # Remote command
    remote_cmd = (
        f'{remote_env}D=$(cape get-config CacheDir); '
        'V=$(cape get-config PDFReader); '
        f'cat > "$D/{basename}"; '
        f'{setsid}$V "$D/{basename}" &'
    )
    # Open the PDF file locally so we can pipe it through STDIN
    with open(fname, 'rb') as fp:
        proc = Popen(
            base_cmd + [remote_cmd],
            stdin=fp,
            stdout=DEVNULL,
            stderr=DEVNULL)
    # Wait option
    if wait:
        proc.communicate()
    # Return subprocess handle
    return proc


def _open_pdf_local(fname: str, wait: bool = False) -> Popen:
    # Get viewer
    viewer = get_pdf_viewer()
    # Command to open it
    proc = Popen([viewer, fname], stdout=PIPE, stderr=PIPE)
    # Wait option
    if wait:
        proc.wait()
    # Return subprocess handle
    return proc
