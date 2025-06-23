#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
``shellutils``: System calls with STDOUT capture, SFTP, and more
==================================================================

This module mainly provides two types of tools:

    *   convenience functions that wrap :class:`subprocess.Popen` that
        simplify common use cases and
    *   classes like :class:`Shell` and :class:`SSH` that create
        persistent processes, usually logged into a remote host.

The convenience functions include

    * :func:`call`
    * :func:`call_o`
    * :func:`call_oe`
    * :func:`call_q`
    * :func:`check_o`

Essentially each of these functions create an instance of
:class:`subprocess.Popen` and then call
:func:`subprocess.Popen.communicate`. They are all calls to
:func:`_call` but with different defaults and outputs for each case. The
``o`` stands for "output," as in STDOUT will be captured. The ``e`` is
for "error," referring to STDERR capture, and ``q`` is for "quiet,"
meaning that by default both STDOUT and STDERR will be suppressed.

Finally, :func:`check_o` is very similar to
:func:`subprocess.check_output`; it will capture STDOUT by default, but
it will also raise an exception if tne command's return code is nonzero.
All of these functions can also be used to execute a command on a remote
host using the *host* keyword argument.

The second category is collection of persistent-process classes:

*   :class:`SSH`
*   :class:`Shell`
*   :class:`SFTP`
*   :class:`SSHPortal`

The first works by logging into a remote host and creating a process so
that you can run a sequence of commands there. It starts with a

.. code-block:: bash

    ssh -q {host} {shell}

command, for example

.. code-block:: bash

    ssh -q pfe bash

It then sets STDIN, STDOUT, and STDERR of this process to be
non-blocking so that you can run multiple commands and query STDOUT and
the others more than one time.

The second class, :class:`Shell`, works in a similar way but on the
local host. It can be convenient because a :class:`Shell` instance has
its own current working directory. It also enables writing code for
which remote hosts and local hosts can share the vast majority of their
code.

:class:`SFTP` is a similar class that starts an SFTP session to a remote
host and behaves much like the system ``sftp`` itself. However, users
will generally prefer the fourth class to using this one directly.

The :class:`SSHPortal` is simply a combination of :class:`SSH` and
:class:`SFTP`. More specifically, it's a new class that have an instance
of both :class:`SSH` and :class:`SFTP` as attributes. Although this
requires two logins, it is usually preferable because it allows the
:class:`SSH` instance to do detailed checks and can create folders with
complex file permissions, while the :class:`SFTP` is used for the actual
transfer of data.

You can instantiate any of these classes (except :class:`Shell`, which
usually takes no arguments) by just giving it the name of the remote
host as a single positional parameter.

.. code-block:: python

    proc = SSH("pfe")

These classes provide several common functions, such as
:func:`SSH.listdir`, :func:`SSH.chdir`, :func:`SFTP.cd_local`, and
:func:`SSHPortal.get` and :func:`SSHPortal.put`. But they (except for
:class:`SFTP`) also have generic :func:`run` commands that allow for
running arbitrary code by writing the command to STDIN.
"""

# Standard library
import base64
import os
import posixpath
import re
import shutil
import socket
import sys
import time
from subprocess import Popen, PIPE


# Default encoding based on OS
if os.name == "nt":  # pragma no cover
    # Standard library imports
    import msvcrt
    from ctypes import byref, windll
    from ctypes.wintypes import DWORD, HANDLE, LPDWORD
    # Unicode is non-standard on Windows
    DEFAULT_ENCODING = "ascii"
    # Function to modify state of Windows pipes
    SetNamedPipeHandleState = windll.kernel32.SetNamedPipeHandleState
    SetNamedPipeHandleState.argtypes = [HANDLE, LPDWORD, LPDWORD, LPDWORD]
    # Flag for non-blocking read
    PIPE_NOWAIT = DWORD(1)
else:  # nt no cover
    DEFAULT_ENCODING = "utf-8"


# Wait times and effort counters
SLEEP_TIME = 0.2
SLEEP_STDOUT = 0.02
SLEEP_PROGRESS = 0.1
N_TIMEOUT = 100

# Regular expression for deciding if a path is local
REGEX_HOST2 = re.compile(r"((?P<host>[A-Za-z][A-Za-z0-9-.]+):)?(?P<path>.+)$")
REGEX_HOST1 = re.compile(
    r"ssh://(?P<host>[A-Za-z][A-Za-z0-9-.]+)(?P<path>/.+)$")


# Standard messages
_LPWD_PREFIX = "Local working directory: "
_PWD_PREFIX = "Remote working directory: "

# Disallowed characters
_FILENAME_MAXCHARS = 255
_FILENAME_CHAR_DENYLIST = frozenset(
    '/"\\|:*?<>,;=' +
    bytes(list(range(32)) + [127]).decode("ascii"))
_DIRNAME_CHAR_DENYLIST = frozenset(
    '"|:*?<>,;=' +
    bytes(list(range(32)) + [127]).decode("ascii"))
_GLOBNAME_CHAR_DENYLIST = frozenset(
    '"|:<>,;=' +
    bytes(list(range(32)) + [127]).decode("ascii"))

# Only show progress (by default) if STDOUT connected to a terminal
TTY = os.isatty(sys.stdout.fileno())


# Function to set a pipe to non-blocking read
def set_nonblocking(fp):
    # Check operating system
    if os.name == "nt":  # pragma no cover
        # Windows handle
        h = msvcrt.get_osfhandle(fp.fileno())
        # Set non-blocking mode
        SetNamedPipeHandleState(h, byref(PIPE_NOWAIT), None, None)
    else:
        # Posix
        os.set_blocking(fp.fileno(), False)


# Error class
class ShellutilsError(BaseException):
    pass


# Filename error
class ShellutilsFilenameError(ValueError, ShellutilsError):
    r"""Error class for invalid file names"""
    pass


class ShellutilsFileExistsError(FileExistsError, ShellutilsError):
    r"""Error class for already existing file"""
    pass


class ShellutilsFileNotFoundError(FileNotFoundError, ShellutilsError):
    r"""Error class for file/folder not found"""
    pass


class ShellutilsIsADirectoryError(IsADirectoryError, ShellutilsError):
    r"""Error class for attempting to access folder as if a file"""
    pass


# Special class for file transfer
class SSHPortal(object):
    r"""Combined SSH and SFTP interface

    This class works by logging into a remote host twice, once for a
    shell (SSH) and another for SFTP.

    :Call:
        >>> portal = SSHPortal(host, cwd=None, encoding="utf-8")
    :Inputs:
        *host*: :class:`str`
            Name of remote host to log into
        *cwd*: {``None``} | :class:`str`
            Optional initial absolute path of shell on *host*
        *encoding*: {``"utf-8"``} | :class:`str`
            Text encoding choice
    :Outputs:
        *portal*: :class:`SSHPortal`
            Combined SSH and SFTP interface
    :Attributes:
        * :attr:`cwd`
        * :attr:`sftp`
        * :attr:`ssh`
    """
   # --- Class attributes ---
    # Allowed instance attributes
    __slots__ = (
        "cwd",
        "sftp",
        "ssh",
    )

   # --- __dunder__ ---
    # Initialization
    def __init__(self, host: str, cwd=None, encoding=DEFAULT_ENCODING):
        r"""Initialization method"""
        # Open both processes
        #: :class:`SFTP` -- File transfer interface for this instance
        self.sftp = SFTP(host, encoding=encoding)
        #: :class:`SSH` -- Remote shell interface for this instance
        self.ssh = SSH(host, encoding=encoding)
        # Default working directory
        if cwd is None:
            cwd = os.getcwd()
        #: :class:`str` -- Initial local working directory
        self.cwd = cwd

   # --- Transfer ---
    def put(self, flocal: str, fremote=None, wait=True, **kw):
        r"""Transfer a file from local to remote host

        :Call:
            >>> portal.put(flocal, fremote=None, wait=True, **kw)
        :Inputs:
            *portal*: :class:`SSHPortal`
                Combined SSH and SFTP interface
            *flocal*: :class:`str`
                Name of local file to send
            *fremote*: {``None``} | :class:`str`
                Name of destination file on remote host; if ``None``,
                then ``os.path.basename(flocal)``
            *wait*: {``True``} | ``False``
                Option to block (wait) until transfer completes
            *progress*: {*wait*} | ``True`` | ``False``
                Option to display transfer progress percentage
            *fprog*: {``None``} | :class:`str`
                File name to show during transfer; default is *flocal*
                truncated to fit in current terminal width
        """
        # Process options
        progress = kw.get("progress", wait)
        fprog = kw.get("fprog")
        # Check for local file
        self.assert_isfile_local(flocal)
        # Get default remote file name
        if fremote is None:
            # Just use last portion (no folder names)
            fremote = os.path.basename(flocal)
        # Create a new file on remote host w/ correct permissions
        self.ssh.newfile(fremote)
        self.ssh.wait()
        # Now copy the file
        self.sftp.put(flocal, fremote)
        # If a progress indicator was requested, wait
        if wait:
            self._wait_put(flocal, fremote, progress=progress, fprog=fprog)

    def get(self, fremote: str, flocal=None, wait=True, **kw):
        r"""Transfer a file from remote to local host

        :Call:
            >>> portal.get(fremote, flocal=None, wait=True, **kw)
        :Inputs:
            *portal*: :class:`SSHPortal`
                Combined SSH and SFTP interface
            *fremote*: :class:`str`
                Name of remote file to send
            *flocal*: {``None``} | :class:`str`
                Name of destination file on local host; if ``None``,
                then ``os.path.basename(fremote)``
            *wait*: {``True``} | ``False``
                Option to block (wait) until transfer completes
            *progress*: {*wait*} | ``True`` | ``False``
                Option to display transfer progress percentage
            *fprog*: {``None``} | :class:`str`
                File name to show during transfer; default is *fremote*
                truncated to fit in current terminal width
        """
        # Process options
        progress = kw.get("progress", wait)
        fprog = kw.get("fprog")
        # Check for remote file
        self.ssh.assert_isfile(fremote)
        # Default local file name
        if flocal is None:
            # Use last portion (no folder names) (always Linux remote)
            flocal = posixpath.basename(fremote)
        # Delete local file if present
        if self.isfile_local(flocal):
            self.remove_local(flocal)
        # Copy the file
        self.sftp.get(fremote, flocal)
        # If a progress indicator was requested, wait
        if wait:
            self._wait_get(fremote, flocal, progress=progress, fprog=fprog)

    def _wait_get(self, fremote: str, flocal: str, progress=TTY, fprog=None):
        # Default name to use in progress indicator
        if fprog is None:
            # Use the base file name on the remote (source) side
            fprog = posixpath.basename(fremote)
        # Truncate
        fprog = self._trunc8_fname(fprog, 8)
        # Progress indicator
        while True:
            # Get local and remote sizes
            size_l = self._getsize_l(flocal)
            size_r = max(1, self._getsize_r(fremote))
            # Calculate fraciton
            prog_fraction = (100 * size_l) // size_r
            # Progress indicator
            if progress:
                sys.stdout.write("%s: %3s%%\r" % (fprog, prog_fraction))
                sys.stdout.flush()
            # Exit loop
            if size_l >= size_r:
                break
            # Wait before doing this again
            time.sleep(SLEEP_PROGRESS)
        # Clean up prompt
        if progress:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _wait_put(self, flocal: str, fremote: str, progress=TTY, fprog=None):
        # Default name to use in progress indicator
        if fprog is None:
            # Use the base file name on the remote (source) side
            fprog = os.path.basename(flocal)
        # Truncate
        fprog = self._trunc8_fname(fprog, 8)
        # Progress indicator
        while True:
            # Get local and remote sizes
            size_l = max(1, self._getsize_l(flocal))
            size_r = self._getsize_r(fremote)
            # Calculate fraciton
            prog_fraction = (100 * size_r) // size_l
            # Progress indicator
            if progress:
                sys.stdout.write("%s: %3s%%\r" % (fprog, prog_fraction))
                sys.stdout.flush()
            # Exit loop
            if size_r >= size_l:
                break
            # Wait before doing this again
            time.sleep(SLEEP_PROGRESS)  # pragma no cover
        # Clean up prompt
        if progress:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def _trunc8_fname(self, fname: str, n: int) -> str:
        # Length of current name
        l0 = len(fname)
        # Max width allowed (right now)
        maxwidth = shutil.get_terminal_size().columns - n
        # Check if truncation needed
        if l0 < maxwidth:
            # Use existing name
            return fname
        # Try to get leading folder
        if "/" in fname:
            # Get first folder, then everything else
            part1, part2 = fname.split("/", 1)
            # Try to truncate this
            fname = part1 + "/..." + part2[4 + len(part1) - maxwidth:]
        # Just truncate file name from end
        fname = fname[-maxwidth:]
        # Output
        return fname

   # --- Control ---
    def chdir_local(self, fdir: str):
        r"""Change the local working directory

        :Call:
            >>> portal.chdir_remote(fdir)
        :Inputs:
            *portal*: :class:`SSHPortal`
                Combined remote shell and file transfer portal
            *fdir*: :class:`str`
                Name of remote path, relative or absolute
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
            * 2023-08-12 ``@ddalle``: v1.1; one less line for testing
        """
        # Absolute path (temp)
        fabs = fdir
        # Check if relative
        if not os.path.isabs(fdir):
            # Relative to *cwd*
            fabs = os.path.join(self.cwd, fdir)
        # Assert folder exists
        if not os.path.isdir(fabs):
            raise ShellutilsFileNotFoundError(
                'No such file or directory "%s"' % fdir)
        # Change local location on SFTP
        self.sftp.cd_local(fdir.replace(os.sep, "/"))

    def chdir_remote(self, fdir: str):
        r"""Change the remote working directory

        :Call:
            >>> portal.chdir_remote(fdir)
        :Inputs:
            *portal*: :class:`SSHPortal`
                Combined remote shell and file transfer portal
            *fdir*: :class:`str`
                Name of remote path, relative or absolute
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Change shell on both locations
        self.ssh.chdir(fdir)
        self.sftp.cd_remote(fdir)

    def close(self):
        r"""Close both SSH and SFTP shells, waiting for running commands

        :Call:
            >>> portal.close()
        :Inputs:
            *portal*: :class:`SSHPortal`
                Combined remote shell and file transfer portal
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        self.ssh.close()
        self.sftp.close()

   # --- File stats ---
    def abspath_local(self, fname: str) -> str:
        r"""Return absolute path to local file/folder

        :Call:
            >>> fabs = portal.abspath_local(fname)
        :Inputs:
            *portal*: :class:`SSHPortal`
                Combined remote shell and file transfer portal
            *fname*: :class:`str`
                Local path to file/folder
        :Outputs:
            *fabs*: :class:`str`
                Local absolute path to *fname*
        """
        # Check input
        if os.path.isabs(fname):
            # Already absolute
            return fname
        else:
            # Prepend with working directory
            return os.path.join(self.cwd, fname)

    def assert_isfile_local(self, fname: str):
        r"""Assert that a file exists locally

        :Call:
            >>> portal.assert_isfile_local(fname)
        :Inputs:
            *portal*: :class:`SSHPortal`
                Combined remote shell and file transfer portal
            *fname*: :class:`str`
                Path to local file
        :Raises:
            :class:`ShellutilsFileNotFoundError` if *fname* does not
            exist relative to *portal.cwd*
        """
        # Check for file
        if not self.isfile_local(fname):
            raise ShellutilsFileNotFoundError(
                'No such file or folder "%s"' % fname)

    def getsize_local(self, fname: str) -> int:
        r"""Get size of local file

        :Call:
            >>> st_size = portal.getsize_local(fname)
        :Inputs:
            *portal*: :class:`SSHPortal`
                Combined remote shell and file transfer portal
            *fname*: :class:`str`
                Name of local file
        :Outputs:
            *st_size*: :class:`int`
                Size of file in bytes
        """
        # Check for local file
        self.assert_isfile_local(fname)
        # Absolutize path
        fabs = self.abspath_local(fname)
        # Get size
        return os.path.getsize(fabs)

    def getsize_remote(self, fname: str) -> int:
        r"""Get size of remote file

        :Call:
            >>> st_size = portal.getsize_local(fname)
        :Inputs:
            *portal*: :class:`SSHPortal`
                Combined remote shell and file transfer portal
            *fname*: :class:`str`
                Name of remote file
        :Outputs:
            *st_size*: :class:`int`
                Size of file in bytes
        """
        # Get size of remote path
        return self.ssh.getsize(fname)

    def isfile_local(self, fname: str) -> bool:
        r"""Check if local file exists

        :Call:
            >>> is_file = portal.isfile_local(fname)
        :Inputs:
            *portal*: :class:`SSHPortal`
                Combined remote shell and file transfer portal
            *fname*: :class:`str`
                Name of local file
        :Outputs:
            *is_file*: ``True`` | ``False``
                Whether *fname* exists
        """
        # Validate file name
        validate_absfilename(fname, sep=os.sep)
        # Absolutize
        fabs = self.abspath_local(fname)
        # Check for file
        return os.path.isfile(fabs)

    def remove_local(self, fname: str):
        r"""Delete a local file, if it exists

        :Call:
            >>> portal.remove_local(fname)
        :Inputs:
            *portal*: :class:`SSHPortal`
                Combined remote shell and file transfer portal
            *fname*: :class:`str`
                Name of local file
        """
        # Assert the file is present
        self.assert_isfile_local(fname)
        # Absolutize
        fabs = self.abspath_local(fname)
        # Remove it
        return os.remove(fabs)

    def _getsize_l(self, fname):
        r"""Get local file size but w/o errors"""
        try:
            return self.getsize_local(fname)
        except ShellutilsFileNotFoundError:
            return 0

    def _getsize_r(self, fname):
        r"""Get remote file size but w/o errors"""
        try:
            return self.getsize_remote(fname)
        except ShellutilsFileNotFoundError:
            return 0


# Base class for SSH, used for both SSH and SFTP
class SSHBase(object):
    r"""Base class for both :class:`SSH` and :class:`SFTP`

    This class does not have an :func:`__init__` method
    """
    # Close the portal
    def close(self):
        r"""Close SSH process

        :Call:
            >>> ssh.close()
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Run the exit command for good measure
        try:
            self.run("exit")
        except (ValueError, BrokenPipeError):
            # Can't run "exit" if STDIN pipe is already broken/closed
            pass
        # Run but wait for any transfers to finish
        self.proc.communicate()

    def read_stderr(self):
        r"""Read remote process's current STDERR buffer

        :Call:
            >>> stderr = ssh.read_stderr()
        :Inputs:
            *ssh*: :class:`SSHBase`
                Persistent SSH subprocess
        :Outputs:
            *stderr*: ``None`` | :class:`str`
                STDERR buffer, if any
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
            * 2023-08-11 ``@ddalle``: v2.0; os.read() to support Windows
        """
        # Attempt to read from STDERR
        try:
            stderr_bytes = os.read(self.proc.stderr.fileno(), 1024)
        except OSError:
            # No STDOUT to read
            return None
        # Decode
        return self._decode(stderr_bytes)

    def read_stdout(self):
        r"""Read remote process's current STDOUT buffer

        :Call:
            >>> stdout = ssh.read_stdout()
        :Inputs:
            *ssh*: :class:`SSHBase`
                Persistent SSH subprocess
        :Outputs:
            *stderr*: ``None`` | :class:`str`
                STDERR buffer, if any
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
            * 2023-08-11 ``@ddalle``: v2.0; os.read() to support Windows
        """
        # Attempt to read from STDOUT
        try:
            stdout_bytes = os.read(self.proc.stdout.fileno(), 1024)
        except OSError:
            # No STDOUT to read
            return None
        # Decode
        return self._decode(stdout_bytes)

    def run(self, cmdstr: str):
        r"""Run a command on remote host

        The command does not need to end with a newline (``\\n``).

        :Call:
            >>> ssh.run(cmdstr)
        :Inputs:
            *ssh*: :class:`SSHBase`
                Persistent SSH subprocess
            *cmdstr*: :class:`str`
                Command to run
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Send the command to STDIN
        self.proc.stdin.write(cmdstr.encode(self.encoding))
        # Terminate command
        self.proc.stdin.write(b"\n")
        # Make sure STDIN reads
        self.proc.stdin.flush()
        # Log the command
        self.log.append(cmdstr)

    def _decode(self, txt):
        if txt is None:
            # No current output
            return txt
        else:
            # Decode
            return txt.decode(self.encoding)


# Class to get a persistent SFTP process
class SFTP(SSHBase):
    r"""Interface to an SFTP process

    :Call:
        >>> sftp = SFTP(host, **kw)
    :Inputs:
        *host*: :class:`str`
            Name of remote host to log into
        *encoding*: {``"utf-8"``} | :class:`str`
            Text encoding choice
    :Outputs:
        *sftp*: :class:`SFTP`
            SFTP file transfer instance
    :Attributes:
        * :attr:`encoding`
        * :attr:`host`
        * :attr:`log`
        * :attr:`proc`
    """
   # --- Class attributes ---
    # Instance attribute list
    __slots__ = (
        "encoding",
        "host",
        "log",
        "proc",
    )

   # --- __dunder__ methods ---
    # Initialization method
    def __init__(self, host: str, **kw):
        r"""Initialization method"""
        #: :class:`str` -- Text encoding for this instance
        self.encoding = kw.get("encoding", DEFAULT_ENCODING)
        #: :class:`str` -- Name of remote host for this instance
        self.host = host
        #: :class:`subprocess.Popen` --
        #: Subprocess used to interface ``sftp`` executable
        self.proc = Popen(
            ["sftp", "-q", host], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        # Make sure to have non-blocking STDOUT and STDERR
        set_nonblocking(self.proc.stdout)
        set_nonblocking(self.proc.stderr)
        #: :class:`list`\ [:class:`str`] --
        #: Log messages
        self.log = []

   # --- Copy files ---
    # Put a file
    def put(self, flocal: str, fremote=None):
        r"""Transfer a file from local host to remote

        :Call:
            >>> sftp.put(flocal, fremote=None)
        :Inputs:
            *sftp*: :class:`SFTP`
                SFTP file transfer instance
            *flocal*: :class:`str`
                Name of file on local host
            *fremote*: {``None``} | :class:`str`
                Name of destination file on remote host, default is
                ``os.path.basename(flocal)``
        """
        # SFTP uses forward slashes
        flocal = flocal.replace(os.sep, '/')
        # Format command to copy file
        if fremote is None:
            # Copy file to same name in PWD
            sftp_cmd = "put %s" % flocal
        else:
            # Explicit name for definition
            sftp_cmd = "put %s %s" % (flocal, fremote)
        # Execute command
        self.run(sftp_cmd)
        # Save log
        self.log.append(sftp_cmd)

    # Get a file
    def get(self, fremote: str, flocal=None):
        r"""Transfer a file from remote host to local

        :Call:
            >>> sftp.get(fremote, flocal=None)
        :Inputs:
            *sftp*: :class:`SFTP`
                SFTP file transfer instance
            *fremote*: :class:`str`
                Name of file on remote host
            *flocal*: {``None``} | :class:`str`
                Name of destination file on local host, default is
                ``os.path.basename(fremote)``
        """
        # Format command to copy file
        if flocal is None:
            # Copy file to same name in PWD
            sftp_cmd = "get %s" % fremote
        else:
            # Explicit name for definition
            # SFTP uses forward slashes
            flocal_sftp = flocal.replace(os.sep, '/')
            # Copy command
            sftp_cmd = "get %s %s" % (fremote, flocal_sftp)
        # Execute command
        self.run(sftp_cmd)
        # Save log
        self.log.append(sftp_cmd)

   # --- Basic control ---
    # Wait until current put/get commands are done
    def wait(self):
        r"""Wait for any current commands to exit

        :Call:
            >>> sftp.wait()
        :Inputs:
            *sftp*: :class:`SFTP`
                SFTP file transfer instance
        """
        # Test if a command runs right away
        self.run("echo")
        # Loop until STDERR shows up
        while True:
            # Read STDERR
            stderr_str = self.read_stderr()
            # If nothing to report, a previous command is waiting
            if stderr_str is None:
                pass
            else:
                # Read STDERR
                line = stderr_str.strip().split("\n")[-1]
                # Check for precise expected result from running 'echo'
                if line == "Invalid command.":
                    # Clear out STDOUT, which will have "sftp> echo\n"
                    self.read_stdout()
                    break
            # Sleep
            time.sleep(SLEEP_TIME)

   # --- Shell ---
    def getcwd_local(self) -> str:
        r"""Get current working directory for local side

        :Call:
            >>> cwd = sftp.getcwd_local()
        :Inputs:
            *sftp*: :class:`SFTP`
                SFTP file transfer instance
        :Outputs:
            *cwd*: :class:`str`
                Absolute path to current working directory
        """
        # Make sure no file transfers are active
        self.wait()
        # Run command
        self.run("lpwd")
        # Read stdout
        txt = self.wait_stdout()
        # Get just the last line
        line = txt.strip().split("\n")[-1]
        # Check if it begins with prefix
        if line.startswith(_LPWD_PREFIX):
            return line[len(_LPWD_PREFIX):]

    def getcwd_remote(self) -> str:
        r"""Get current working directory for remote side

        :Call:
            >>> cwd = sftp.getcwd_remote()
        :Inputs:
            *sftp*: :class:`SFTP`
                SFTP file transfer instance
        :Outputs:
            *cwd*: :class:`str`
                Absolute path to current working directory
        """
        # Make sure no file transfers are active
        self.wait()
        # Run command
        self.run("pwd")
        # Read stdout
        txt = self.wait_stdout()
        # Get just the last line
        line = txt.strip().split("\n")[-1]
        # Check if it begins with prefix
        if line.startswith(_PWD_PREFIX):
            return line[len(_PWD_PREFIX):]

    def cd_local(self, path: str):
        r"""Change working directory on local side

        :Call:
            >>> sftp.cd_local(path)
        :Inputs:
            *sftp*: :class:`SFTP`
                SFTP file transfer instance
            *path*: :class:`str`
                Folder (absolute or relative) to change to
        """
        # Run command to change local directory
        self.run("lcd %s" % path)

    def cd_remote(self, path: str):
        r"""Change working directory on remote side

        :Call:
            >>> sftp.cd_remote(path)
        :Inputs:
            *sftp*: :class:`SFTP`
                SFTP file transfer instance
            *path*: :class:`str`
                Folder (absolute or relative) to change to
        """
        # Run command to change remote directory
        self.run("cd %s" % path)

   # --- Basic command tools ---
    # Read STDOUT, waiting a while
    def wait_stdout(self):
        r"""Read current STDOUT, or wait until some appears

        :Call:
            >>> txt = sftp.wait_stdout()
        :Inputs:
            *sftp*: :class:`SFTP`
                SFTP file transfer instance
        :Outputs:
            *txt*: :class:`str` | ``None``
                Text read from STDOUT, if any
        """
        # Read stdout
        for _ in range(N_TIMEOUT):
            # Read current STDOUT
            txt = self.read_stdout()
            # Exit if not None
            if txt is not None:
                return txt
            # Sleep before trying again
            time.sleep(SLEEP_STDOUT)


# Open an SSH termianl and leave it open
class SSH(SSHBase):
    r"""Class for persistent SSH subprocess

    This class opens :class:`subprocess.Popen` instance that logs into a
    remote host of your choosing and then waits for you to issue
    commands. It can also collect and return STDOUT and STDERR.

    :Call:
        >>> ssh = SSH(host, **kw)
    :Inputs:
        *host*: ``None`` | :class:`str`
            Name of server to login to using SSH (use local if ``None``)
        *executable*: {``"bash"``} | :class:`str`
            Executable to use on the remote host
        *encoding*: {``"utf-8"``} | :class:`str`
            Encoding for STDOUT, etc. bytes
    :Outputs:
        *ssh*: :class:`SSH`
            Persistent SSH subprocess
    :Attributes:
        * :attr:`encoding`
        * :attr:`executable`
        * :attr:`host`
        * :attr:`log`
        * :attr:`proc`
    """
   # --- Class attributes ---
    # Instance attribute list
    __slots__ = (
        "encoding",
        "executable",
        "host",
        "log",
        "proc",
        "_stdout",
    )

   # --- __dunder__ methods ---
    # Initialization method
    def __init__(self, host=None, **kw):
        #: :class:`str` -- Encoding for this method
        self.encoding = kw.get("encoding", DEFAULT_ENCODING)
        #: :class:`str` -- Executable for remote process
        self.executable = kw.get("executable", "bash")
        #: :class:`str` | ``None`` --
        #: Name of remote host for this process; if ``None`` then local
        #: process
        self.host = host
        # Form command depending on whether we have a *host* or not
        if host is None:
            # Local shell
            startcmd = [self.executable]
        else:
            # Remote
            startcmd = ["ssh", "-q", host, self.executable]
        #: :class:`subprocess.Popen` --
        #: Subprocess interface for ``ssh`` for this instance
        self.proc = Popen(startcmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        # Make sure to have non-blocking STDOUT and STDERR
        set_nonblocking(self.proc.stdout)
        set_nonblocking(self.proc.stderr)
        #: :class:`list`\ [:class:`str`] -- Log messages
        self.log = []
        # Initialize STDOUT container
        self._stdout = None

   # --- Interface commands ---
    def communicate(self, cmd_str: str):
        r"""Run a command and collect return code, STDOUT, and STDERR

        :Call:
            >>> ierr, stdout, stderr = ssh.communicate(cmd_str)
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
            *cmd_str*: :class:`str`
                Command to run on remote host
        :Outputs:
            *ierr*: :class:`int`
                Return code
            *stdout*: ``None`` | :class:`str`
                STDOUT from command, if any
            *stderr*: ``None`` | :class:`str`
                STDERR from command, if any
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # We need to use STDOUT, so wait for clean shell
        self.wait()
        # Clear STDERR
        self.read_stderr()
        # Run test command
        self.run(cmd_str)
        # Echo last returncode to STDOUT
        self.run("echo $?")
        # Wait for this command
        self.wait()
        # Collect STDOUT and STDERR
        stdoutraw = self.read_stdout()
        stderr = self.read_stderr()
        # Split STDOUT into lines
        stdoutlines = stdoutraw.split("\n")
        # Remove last line if STDOUT ends with "\n" (usual case)
        if stdoutraw.endswith("\n"):
            stdoutlines.pop(-1)  # pragma no cover
        # Reassemble STDOUT from original lines
        if len(stdoutlines) == 1:
            # No STDOUT
            stdout = None
        else:
            # Reassemble
            stdout = "\n".join(stdoutlines[:-1])  # pragma no cover
        # Convert last line, the return code, to integer
        try:
            # Get result of 'echo $?'
            returncode = int(stdoutlines[-1])
        except ValueError:  # pragma no cover
            # Probably 'echo $?' didn't run due to earlier failure?
            returncode = -1
        # Output
        return returncode, stdout, stderr

    def call(self, cmdstr: str):
        r"""Run a command and collect return code

        :Call:
            >>> ierr = ssh.call(cmd_str)
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
            *cmdstr*: :class:`str`
                Command to run on remote host
        :Outputs:
            *ierr*: :class:`int`
                Return code
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Do the full communicate() command
        returncode, _, _ = self.communicate(cmdstr)
        # Just return the returncode
        return returncode

   # --- Folder control ---
    def assert_isdir(self, fdir: str):
        r"""Assert that a folder exists on remote host

        :Call:
            >>> ssh.assert_isdir(fdir)
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
            *fdir*: :class:`str`
                Name of prospective folder to test
        :Raises:
            :class:`ShellutilsFileNotFoundError` if *fdir* is not a dir
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Check if folder exists first (also does *fdir* check)
        if not self.isdir(fdir):
            raise ShellutilsFileNotFoundError(
                f'No folder "{fdir}" {self._genr8_hostmsg()}')

    def chdir(self, wd: str):
        r"""Change the current working directory on the remote host

        :Call:
            >>> ssh.chdir(wd)
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Check if folder exists
        self.assert_isdir(wd)
        # Run the command
        self.run('cd "%s"' % wd)

    def getcwd(self) -> str:
        r"""Get current working directory

        :Call:
            >>> cwd = ssh.getcwd()
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
        :Outputs:
            *cwd*: :class:`str`
                Working directory on remote host
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Wait for any other commands to terminate
        self.wait()
        # Run "pwd" command
        self.run("pwd")
        # Get STDOUT
        return self.wait_stdout().rstrip("\n")

    def isdir(self, fdir: str) -> bool:
        r"""Test if remote folder *fdir* exists

        :Call:
            >>> q = ssh.isdir(fdir)
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
            *fdir*: :class:`str`
                Name of prospective folder to test
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not *fdir* is a folder on remote host
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Check for invalid folder names
        validate_dirname(fdir)
        # Run test command and use returncode
        returncode = self.call('test -d "%s"' % fdir)
        # If the folder exists, the returncode is ``0``
        return returncode == 0

    def listdir(self, fdir=".") -> list:
        r"""List contents of a folder

        :Call:
            >>> fnames = ssh.listdir(fdir=".")
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
            *fdir*: {``"."``} | :class:`str`
                Name of prospective folder to test
        :Outputs:
            *fnames*: :class:`list`\ [:class:`str`]
                List of files, folders, and links in *fdir*
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Check if folder exists first (also does *fdir* check)
        self.assert_isdir(fdir)
        # Run basic "ls" command
        self.run('ls "%s"' % fdir)
        # Get output
        stdout = self.wait_stdout()
        # Split lines
        return stdout.rstrip("\n").split("\n")

    def mkdir(self, fdir: str):
        r"""Create new folder

        :Call:
            >>> ssh.mkdir(fdir)
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
            *fdir*: {``"."``} | :class:`str`
                Name of prospective folder to test
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Split path
        froot, fbase = posixpath.split(fdir)
        # Check if *froot* exists
        if froot:
            self.assert_isdir(froot)
        # Make sure *fbase* does *not* exist
        if self.isdir(fbase) or self.isfile(fbase):
            raise ShellutilsFileExistsError(
                "Folder '%s' already exists" % fdir)
        # Create folder
        self.run('mkdir "%s"' % fdir)

    def _genr8_hostmsg(self):
        # Check for local host
        if self.host is None:
            # Use name of local machine
            return f'on local host "{socket.gethostname()}"'
        else:
            # Use name of host
            return f'on remote host "{self.host}"'

   # --- File data ---
    def assert_isfile(self, fname: str):
        r"""Assert that a file exists on remote host

        :Call:
            >>> ssh.assert_isfile(fdir)
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
            *fname*: :class:`str`
                Name of prospective file to test
        :Raises:
            :class:`ShellutilsFileNotFoundError` if *fname* is not file
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Check if folder exists first (also does *fdir* check)
        if not self.isfile(fname):
            raise ShellutilsFileNotFoundError(
                f'No file "{fname}" {self._genr8_hostmsg()}')

    def getmtime(self, fname: str) -> int:
        r"""Get modification time of remote file

        :Call:
            >>> mtime = ssh.getmtime(fname)
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
            *fname*: :class:`str`
                Name of file on remote host
        :Outputs:
            *fsize*: :class:`int`
                Size of file in bytes
        :Versions:
            * 2024-04-23 ``@ddalle``: v1.0
        """
        # Validate file name
        self.assert_isfile(fname)
        # Run ``stat``
        self.run('stat --printf "%%Y\n" "%s"' % fname)
        # Get STDOUT
        return int(self.wait_stdout().strip())

    def getsize(self, fname: str) -> int:
        r"""Get size of remote file

        :Call:
            >>> fsize = ssh.getsize(fname)
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
            *fname*: :class:`str`
                Name of file on remote host
        :Outputs:
            *fsize*: :class:`int`
                Size of file in bytes
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Validate file name
        self.assert_isfile(fname)
        # Run ``stat``
        self.run('stat --printf "%%s\n" "%s"' % fname)
        # Get STDOUT
        return int(self.wait_stdout().strip())

    def isfile(self, fname: str) -> bool:
        r"""Test if remote file *fname* exists and is a file

        :Call:
            >>> q = ssh.isfile(fname)
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
            *fname*: :class:`str`
                Name of prospective folder to test
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not *fname* is a file on remote host
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Check for invalid folder names
        validate_absfilename(fname)
        # Run test command and use returncode
        returncode = self.call('test -f "%s"' % fname)
        # If the folder exists, the returncode is ``0``
        return returncode == 0

    def remove(self, fname: str):
        r"""Remove a file or link

        :Call:
            >>> ssh.remove(fname)
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
            *fname*: :class:`str`
                Name of prospective folder to test
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Check if file exists
        if self.isdir(fname):
            # Cannot remove a folder
            raise ShellutilsIsADirectoryError(
                "Is a directory: '%s'" % fname)
        elif self.isfile(fname):
            # Remove it
            self.run('rm "%s"' % fname)
        else:
            # No file to delete
            raise ShellutilsFileNotFoundError(
                "No file to delete: '%s'" % fname)

    def newfile(self, fname: str):
        r"""Create an empty file w/ correct ACLs based on parent folder

        The purpose of this command is to create an empty file with the
        correct name but obeying the default ACLs of the parent folder.
        SFTP does not upload files with the correct access control
        lists, so this file creates a new file using ``touch``. When the
        empty file already exists, SFTP does not change the ACLs, so the
        result is a file with the expected permissions. If the file
        already exists, this function deletes it.

        :Call:
            >>> ssh.newfile(fname)
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
            *fname*: :class:`str`
                Name of prospective folder to test
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Check if file exists
        if self.isdir(fname):
            # Cannot remove a folder
            raise ShellutilsIsADirectoryError(
                "Is a directory: '%s'" % fname)
        elif self.isfile(fname):
            # Remove it
            self.run('rm "%s"' % fname)
        # Now create it with correct permissions from ACLs
        self.run('touch "%s"' % fname)

    def touch(self, fname: str):
        r"""Create empty file or update modification time

        :Call:
            >>> ssh.touch(fname)
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
            *fname*: :class:`str`
                Name of prospective folder to test
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Split path into dir and basename
        fdir, fbase = posixpath.split(fname)
        # Check for folder
        if fdir:
            self.assert_isdir(fdir)
        # Validate file name
        validate_filename(fbase)
        # Run touch command
        self.run('touch "%s"' % fname)

   # --- Basic control ---
    # Wait until current put/get commands are done
    def wait(self, timeout=None, dt=SLEEP_STDOUT):
        r"""Wait for any running commands to terminate

        The method for this is to echo a random string and wait for it
        to show up in STDOUT. The random string is then removed from the
        STDOUT buffer so that other methods can access the intended
        STDOUT text.

        :Call:
            >>> ierr = ssh.wait()
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
            *timeout*: {``None``} | :class:`float` | :class:`int`
                Maximum seconds to wait, unlimited if ``None``
            *dt*: {``0.02``} | :class:`float`
                Seconds to wait between polls
        :Outputs:
            *ierr*: ``0`` | ``1`` | ``2``
                Status indicator:

                * ``0``: success
                * ``1``: wait() call timed out
                * ``2``: keyboard interrupt
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Allow for keyboard interrupt
        try:
            return self._wait(timeout, dt)
        except KeyboardInterrupt:  # pragma no cover
            # Manually interrupted
            return 2

    def _wait(self, timeout=None, dt=SLEEP_STDOUT):
        # Generate random string
        # Use "base64" encoding to convert 32 random bytes to string
        rand_str = base64.b64encode(os.urandom(32)).decode("ascii")
        # Echo it
        self.echo(rand_str)
        # Reset STDOUT holder
        self._stdout = None
        # Initial time
        tic = time.time()
        # Read stdout
        while (timeout is None) or (time.time() - tic < timeout):
            # Read current STDOUT
            txt = self._read_stdout()
            # Exit if not None
            if txt is not None:
                # Get last line
                lines = txt.rstrip("\n").split("\n")
                # Check value
                if lines[-1] == rand_str:
                    # Success! Got random string back; shell is ready
                    # Save any other STDOUT we got
                    self._save_stdout("\n".join(lines[:-1]))
                    # Success!
                    return 0
                else:
                    # Save any other STDOUT we got
                    self._save_stdout(txt)
            # Sleep before trying again
            time.sleep(dt)
        # Otherwise failed to get output back
        return 1  # pragma no cover

    def echo(self, msg: str):
        r"""Echo text to STDOUT on remote host

        :Call:
            >>> ssh.echo(msg)
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
            *msg*: :class:`str`
                Text to echo
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Check executable
        if self.executable == "bash":
            self._echo_bash(msg)

    def _echo_bash(self, msg: str):
        self.run("echo '%s'" % msg)

   # --- Basic command tools ---
    # Read STDOUT, waiting a while
    def wait_stdout(self, timeout=None, dt=SLEEP_STDOUT):
        r"""Capture STDOUT, but wait for any running process to finish

        :Call:
            >>> stdout = ssh.wait_stdout(timeout=None, dt=0.02)
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
            *timeout*: {``None``} | :class:`float` | :class:`int`
                Maximum seconds to wait, unlimited if ``None``
            *dt*: {``0.02``} | :class:`float`
                Seconds to wait between polls
        :Outputs:
            *stdout*: :class:`str`
                STDOUT but waiting until it's nonempty
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
        """
        # Initial time
        tic = time.time()
        # Initial text
        msg = None
        # Read stdout
        while (timeout is None) or (time.time() - tic < timeout):
            # Read current STDOUT
            txt = self.read_stdout()
            # Check if anything was read (this time)
            if txt is None and msg is not None:
                # Read something previously but not this time
                return msg
            if txt is not None:
                # Check if we had a previous read
                if msg is None:
                    # First read
                    msg = txt
                else:
                    # Append to previous read
                    msg += txt  # pragma no cover
            # Sleep before trying again
            time.sleep(dt)

    def read_stdout(self):
        r"""Read remote process's current STDOUT buffer

        :Call:
            >>> stdout = ssh.read_stdout()
        :Inputs:
            *ssh*: :class:`SSH`
                Persistent SSH subprocess
        :Outputs:
            *stdout*: ``None`` | :class:`str`
                STDOUT buffer, if any
        :Versions:
            * 2022-12-19 ``@ddalle``: v1.0
            * 2023-08-11 ``@ddalle``: v2.0; os.read() to support Windows
        """
        # Check for existing STDOUT buffer
        if self._stdout:
            # Transfer buffer
            msg = self._stdout
            # Reset
            self._stdout = None
            # Output
            return msg
        # Otherwise reread
        return self._read_stdout()

    def _read_stdout(self):
        # Attempt to read from STDOUT
        try:
            stdout_bytes = os.read(self.proc.stdout.fileno(), 1024)
        except OSError:
            # No STDOUT to read
            return None
        # Decode
        return self._decode(stdout_bytes)

    def _save_stdout(self, txt):
        # Don't save empty text
        if not txt:
            return
        # Check if any current buffer
        if self._stdout is None:
            # Start new buffer
            self._stdout = txt
        else:
            # Append to buffer
            self._stdout = self._stdout + txt  # pragma no cover


# Open a subprocess shell and leave it open
class Shell(SSH):
    r"""Class for persistent local subprocess

    This class opens :class:`subprocess.Popen` instance that starts a
    shell process (usually BASH) on your local host and then waits for
    you to issue commands. It can also collect and return STDOUT and
    STDERR.

    :Call:
        >>> shell = Shell(**kw)
    :Inputs:
        *host*: :class:`str`
            Name of server to login to using SSH
        *executable*: {``"bash"``} | :class:`str`
            Executable to use on the remote host
        *encoding*: {``"utf-8"``} | :class:`str`
            Encoding for STDOUT, etc. bytes
    :Outputs:
        *shell*: :class:`Shell`
            Persistent local subprocess
    :Attributes:
        * :attr:`encoding`
        * :attr:`executable`
        * :attr:`host`
        * :attr:`log`
        * :attr:`proc`
    """
   # --- __dunder__ methods ---
    # Initialization method
    def __init__(self, **kw):
        SSH.__init__(self, None, **kw)


# Call a command and capture output
def check_o(cmd, **kw):
    r"""Run a system command and capture STDOUT

    :Call:
        >>> out, err = check_o(cmd, **kw)
    :Inputs:
        *cmd*: :class:`list`\ [:class:`str`]
            System command to run, broken into parts as for
            :func:`subprocess.call`
        *stdout*: ``None`` | {*PIPE*} | :class:`file`
            Destination for standard output
        *stderr*: {``None``} | *PIPE* | :class:`file`
            Destination for standard error messages
        *encoding*: {``"utf-8"``} | :class:`str`
            Name of encoding to use for converting strings to bytes
        *host*: {``None``} | :class:`str`
            Name of remote host (if not ``None``) on which to run
        *cwd*: {``os.getcwd()``} | :class:`str`
            Folder in which to run command
        *executable*: {``"sh"``} | :class:`str`
            Name of shell to use if on remote host
    :Outputs:
        *out*: :class:`str`
            Captured STDOUT decoded as a :class:`str`
        *err*: ``None`` | :class:`str`
            Captured STDERR if *stdout* is *subprocess.PIPE*
    :Versions:
        * 2020-12-24 ``@ddalle``: v1.0
    """
    # Default STDOUT
    kw.setdefault("stdout", PIPE)
    # Call basis command
    stdout, stderr, ierr = _call(cmd, **kw)
    # Check for errors
    if ierr:
        raise SystemError(
            "Return code '%i' while calling `%s`" %
            (ierr, " ".join(cmd)))
    # Output
    return stdout, stderr


# Call a command without default captures
def call(cmd, **kw):
    r"""Run a system command and ignore STDOUT and STDERR

    Setting *stdout* and/or *stderr* to *PIPE* will suppress them.
    Setting them to ``None`` (the default) will cause them to display
    normally and not be captured.

    :Call:
        >>> ierr = call(cmd, **kw)
    :Inputs:
        *cmd*: :class:`list`\ [:class:`str`]
            System command to run, broken into parts as for
            :func:`subprocess.call`
        *stdout*: {``None``} | *PIPE* | :class:`file`
            Destination for standard output
        *stderr*: {``None``} | *PIPE* | :class:`file`
            Destination for standard error messages
        *encoding*: {``"utf-8"``} | :class:`str`
            Name of encoding to use for converting strings to bytes
        *host*: {``None``} | :class:`str`
            Name of remote host (if not ``None``) on which to run
        *cwd*: {``os.getcwd()``} | :class:`str`
            Folder in which to run command
        *executable*: {``"sh"``} | :class:`str`
            Name of shell to use if on remote host
    :Outputs:
        *ierr*: :class:`int`
            Return code from executing command
    :Versions:
        * 2020-12-24 ``@ddalle``: v1.0
    """
    # Call basis command
    _, _, ierr = _call(cmd, **kw)
    # Only return return code
    return ierr


# Call a command and capture output
def call_oe(cmd, **kw):
    r"""Run a system command and capture STDOUT

    :Call:
        >>> out, err, ierr = call_oe(cmd, **kw)
    :Inputs:
        *cmd*: :class:`list`\ [:class:`str`]
            System command to run, broken into parts as for
            :func:`subprocess.call`
        *stdout*: ``None`` | {*PIPE*} | :class:`file`
            Destination for standard output
        *stderr*: ``None`` | {*PIPE*} | :class:`file`
            Destination for standard error messages
        *encoding*: {``"utf-8"``} | :class:`str`
            Name of encoding to use for converting strings to bytes
        *host*: {``None``} | :class:`str`
            Name of remote host (if not ``None``) on which to run
        *cwd*: {``os.getcwd()``} | :class:`str`
            Folder in which to run command
        *executable*: {``"sh"``} | :class:`str`
            Name of shell to use if on remote host
    :Outputs:
        *out*: :class:`str`
            Captured STDOUT decoded as a :class:`str`
        *err*: ``None`` | :class:`str`
            Captured STDERR if *stdout* is *subprocess.PIPE*
        *ierr*: :class:`int`
            Return code from executing command
    :Versions:
        * 2021-07-19 ``@ddalle``: v1.0
    """
    # Default STDOUT
    kw.setdefault("stdout", PIPE)
    kw.setdefault("stderr", PIPE)
    # Call basis command
    return _call(cmd, **kw)


# Call a command and capture output
def call_o(cmd, **kw):
    r"""Run a system command and capture STDOUT

    :Call:
        >>> out, err, ierr = call_o(cmd, **kw)
    :Inputs:
        *cmd*: :class:`list`\ [:class:`str`]
            System command to run, broken into parts as for
            :func:`subprocess.call`
        *stdout*: ``None`` | {*PIPE*} | :class:`file`
            Destination for standard output
        *stderr*: {``None``} | *PIPE* | :class:`file`
            Destination for standard error messages
        *encoding*: {``"utf-8"``} | :class:`str`
            Name of encoding to use for converting strings to bytes
        *host*: {``None``} | :class:`str`
            Name of remote host (if not ``None``) on which to run
        *cwd*: {``os.getcwd()``} | :class:`str`
            Folder in which to run command
        *executable*: {``"sh"``} | :class:`str`
            Name of shell to use if on remote host
    :Outputs:
        *out*: :class:`str`
            Captured STDOUT decoded as a :class:`str`
        *err*: ``None`` | :class:`str`
            Captured STDERR if *stdout* is *subprocess.PIPE*
        *ierr*: :class:`int`
            Return code from executing command
    :Versions:
        * 2020-12-24 ``@ddalle``: v1.0
    """
    # Default STDOUT
    kw.setdefault("stdout", PIPE)
    # Call basis command
    return _call(cmd, **kw)


# Call a command and suppress STDOUT and STDERR
def call_q(cmd, **kw):
    r"""Run a system command, suppressing STDOUT and STDERR

    :Call:
        >>> ierr = call_q(cmd, **kw)
    :Inputs:
        *cmd*: :class:`list`\ [:class:`str`]
            System command to run, broken into parts as for
            :func:`subprocess.call`
        *stdout*: ``None`` | {*PIPE*} | :class:`file`
            Destination for standard output
        *stderr*: ``None`` | {*PIPE*} | :class:`file`
            Destination for standard error messages
        *encoding*: {``"utf-8"``} | :class:`str`
            Name of encoding to use for converting strings to bytes
        *host*: {``None``} | :class:`str`
            Name of remote host (if not ``None``) on which to run
        *cwd*: {``os.getcwd()``} | :class:`str`
            Folder in which to run command
        *executable*: {``"sh"``} | :class:`str`
            Name of shell to use if on remote host
    :Outputs:
        *ierr*: :class:`int`
            Return code from executing command
    :Versions:
        * 2020-12-24 ``@ddalle``: v1.0
    """
    # Hide STDOUT and STDERR
    kw.setdefault("stdout", PIPE)
    kw.setdefault("stderr", PIPE)
    # Call basis command
    _, _, ierr = _call(cmd, **kw)
    # Only return return code
    return ierr


# Call a command
def _call(cmd, **kw):
    r"""Generic system interface

    This command can either capture STDOUT [and/or STDERR], print it
    to the terminal, or pipe it to a file.  It more or less
    encapsulates the capabilities of

        * :func:`subprocess.call` and
        * :func:`subprocess.check_output`

    although it works by using :class:`subprocess.Popen` directly.

    :Call:
        >>> out, err, ierr = _call(cmd, **kw)
    :Inputs:
        *cmd*: :class:`list`\ [:class:`str`]
            System command to run, broken into parts as for
            :func:`subprocess.call`
        *stdout*: {``None``} | *PIPE* | :class:`file`
            Destination for standard output
        *stderr*: {``None``} | *PIPE* | :class:`file`
            Destination for standard error messages
        *encoding*: {``"utf-8"``} | :class:`str`
            Name of encoding to use for converting strings to bytes
        *host*: {``None``} | :class:`str`
            Name of remote host (if not ``None``) on which to run
        *cwd*: {``None``} | :class:`str`
            Folder in which to run command
        *executable*: {``"sh"``} | :class:`str`
            Name of shell to use if on remote host
    :Outputs:
        *out*: ``None`` | :class:`str`
            Captured STDOUT if *stdout* is *subprocess.PIPE*
        *err*: ``None`` | :class:`str`
            Captured STDERR if *stdout* is *subprocess.PIPE*
        *ierr*: :class:`int`
            Return code from executing command
    :Versions:
        * 2020-12-24 ``@ddalle``: v1.0
        * 2021-01-27 ``@ddalle``: v1.1; fix default cwd w/ host
    """
    # Process keyword args
    cwd = kw.get("cwd")
    stdout = kw.get("stdout")
    stderr = kw.get("stderr")
    encoding = kw.get("encoding", DEFAULT_ENCODING)
    host = kw.get("host")
    executable = kw.get("executable", "sh")
    # Check if remote
    if host:
        # Create a process
        proc = Popen(
            ["ssh", "-q", host, executable],
            stdin=PIPE, stdout=stdout, stderr=stderr)
        # Go to the folder
        if cwd is not None:
            proc.stdin.write(("cd %s\n" % cwd).encode(encoding))
        # Write the command
        proc.stdin.write(" ".join(cmd).encode(encoding))
        # Send commands
        proc.stdin.flush()
    else:
        # Create process
        proc = Popen(
            cmd, stdin=PIPE, cwd=cwd, stdout=stdout, stderr=stderr)
    # Get the results
    stdout, stderr = proc.communicate()
    # Decode output
    if isinstance(stdout, bytes):
        stdout = stdout.decode(encoding)
    if isinstance(stderr, bytes):
        stderr = stderr.decode(encoding)
    # Output
    return stdout, stderr, proc.returncode


# Validate an absolute file name
def validate_absfilename(fname: str, sep='/'):
    r"""Check if a file name is valid, allowing for folder names

    This version only disallows punctuation characters that cause
    problems on at least one common file system. Single quotes and
    foreign language characters are allowed.

    It also restricts the file name length to 255 characters, even
    though this is not strictly speaking a hard limit on Windows
    systems.

    :Call:
        >>> validate_absfilename(fname, sep='/')
    :Inputs:
        *fname*: :class:`str`
            Name of file
        *sep*: {``'/'``} | ``"\"``
            Path separator
    :Versions:
        * 2022-12-19 ``@ddalle``: v1.0
        * 2023-10-25 ``@ddalle``: v1.1; add *sep*
    """
    # Get base file name
    fbase = fname.split(sep)[-1]
    # Check base name
    validate_filename(fbase)
    # Search through characters of *fname*
    _check_str_denylist(fname, _DIRNAME_CHAR_DENYLIST, "file name")


# Validate a folder name
def validate_dirname(fdir: str, sep='/'):
    r"""Check if a folder name is valid

    This version only disallows punctuation characters that cause
    problems on at least one common file system. Single quotes and
    foreign language characters are allowed.

    It also restricts the file name length to 255 characters, even
    though this is not strictly speaking a hard limit on Windows
    systems.

    :Call:
        >>> validate_dirname(fdir, sep='/')
    :Inputs:
        *fdir*: :class:`str`
            Name of file
    :Versions:
        * 2022-12-19 ``@ddalle``: v1.0
        * 2023-10-25 ``@ddalle``: v1.1; add *sep*
    """
    # Get base file name
    fbase = fdir.split(sep)[-1]
    # Check base name
    _check_str_denylist(fbase, _FILENAME_CHAR_DENYLIST, "folder name")
    # Check length
    _check_fname_len(fbase)
    # On windows, files cannot end with "."
    if fbase.endswith(".") and fbase not in (".", ".."):
        raise ShellutilsFilenameError(
            "File name '%s' cannot end with '.' on Windows" % fbase)
    # Search through characters of *fname*
    _check_str_denylist(fdir, _DIRNAME_CHAR_DENYLIST, "folder name")


# Validate a file name
def validate_filename(fname: str):
    r"""Check if a file name is valid

    This version only disallows punctuation characters that cause
    problems on at least one common file system. Single quotes and
    foreign language characters are allowed.

    It also restricts the file name length to 255 characters, even
    though this is not strictly speaking a hard limit on Windows
    systems.

    :Call:
        >>> validate_filename(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file
    :Versions:
        * 2022-12-19 ``@ddalle``: v1.0
    """
    # Check length
    _check_fname_len(fname)
    # Search through characters of *fname*
    _check_str_denylist(fname, _FILENAME_CHAR_DENYLIST, "file name")
    # On windows, files cannot end with "."
    if fname.endswith("."):
        raise ShellutilsFilenameError(
            "File name '%s' cannot end with '.' on Windows" % fname)


# Validate a glob pattern
def validate_globname(pattern: str):
    r"""Check if a glob pattern is valid

    This version only disallows punctuation characters that cause
    problems on at least one common file system. Single quotes and
    foreign language characters are allowed.

    It also restricts the file name length to 255 characters, even
    though this is not strictly speaking a hard limit on Windows
    systems.

    :Call:
        >>> validate_globname(pattern)
    :Inputs:
        *fname*: :class:`str`
            Name of file
    :Versions:
        * 2022-12-19 ``@ddalle``: v1.0
    """
    # Search through characters of *fname*
    _check_str_denylist(pattern, _GLOBNAME_CHAR_DENYLIST, "wildcard pattern")


# Identify local vs remote
def identify_host(where=None):
    r"""Identify possible remote host/local path breakdown

    :Call:
        >>> host, path = identify_host(where)
    :Inputs:
        *where*: {``None``} | :class:`str`
            Local path like ``"/home/user"`` or remote path like
            ``"pfe://home"``
    :Outputs:
        *host*: ``None`` | :class:`str`
            Remote host nae preceding ``:`` if any
        *path*: :class:`str`
            Path on *host*; matches *where* if no host
    :Versions:
        * 2022-03-06 ``@ddalle``: v1.0
    """
    # Default to current location
    if where is None:
        return None, os.getcwd()
    # Check against ssh://{host}/{path} pattern
    match = REGEX_HOST1.match(where)
    # Check for amatch
    if match:
        return match.group("host"), match.group("path")
    # Check against pattern (guaranteed to match)
    match = REGEX_HOST2.match(where)
    # Return groups
    return match.group("host"), match.group("path")


# Check string against a denylist
def _check_str_denylist(fname: str, denylist: frozenset, title: str):
    # Loop through characters of string
    for j, c in enumerate(fname):
        # Check if character is denied
        if j == 1 and c == ":":
            # Allow paths like "C:\Users"
            pass
        elif c in denylist:
            # Create error message
            msg = 'Character "%%s" is not allowed in %s "%%s"' % title
            # Raise exception
            raise ShellutilsFilenameError(msg % (c, fname))


# Check file name length
def _check_fname_len(fname: str):
    # Check length
    if len(fname) > _FILENAME_MAXCHARS:
        raise ShellutilsFilenameError(
            "File name with %i characters is not allowed (max %i)"
            % (len(fname), _FILENAME_MAXCHARS))
