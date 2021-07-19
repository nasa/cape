#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
:mod:`shellutils`: Convenient system calls; remote or local
=============================================================

:Versions:
    * 2021-07-19 ``@ddalle``: Version 1.0
"""

# Standard library
import os

# Direct standard library imports
from subprocess import Popen, PIPE


# Call a command and capture output
def check_o(cmd, **kw):
    r"""Run a system command and capture STDOUT

    :Call:
        >>> out, err = call_o(cmd, **kw)
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
        *host*: {*repo.host*} | ``None`` | :class:`str`
            Name of remote host (if not ``None``) on which to run
        *executable*: {``"sh"``} | :class:`str`
            Name of shell to use if on remote host
    :Outputs:
        *out*: :class:`str`
            Captured STDOUT decoded as a :class:`str`
        *err*: ``None`` | :class:`str`
            Captured STDERR if *stdout* is *subprocess.PIPE*
    :Versions:
        * 2020-12-24 ``@ddalle``: Version 1.0
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
        *host*: {*repo.host*} | ``None`` | :class:`str`
            Name of remote host (if not ``None``) on which to run
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
        * 2020-12-24 ``@ddalle``: Version 1.0
    """
    # Default STDOUT
    kw.setdefault("stdout", PIPE)
    # Call basis command
    return _call(cmd, **kw)


# Call a command without default captures
def call(cmd, **kw):
    r"""Run a system command and ignore STDOUT and STDERR

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
        *host*: {*repo.host*} | ``None`` | :class:`str`
            Name of remote host (if not ``None``) on which to run
        *executable*: {``"sh"``} | :class:`str`
            Name of shell to use if on remote host
    :Outputs:
        *ierr*: :class:`int`
            Return code from executing command
    :Versions:
        * 2020-12-24 ``@ddalle``: Version 1.0
    """
    # Call basis command
    _, _, ierr = _call(cmd, **kw)
    # Only return return code
    return ierr


# Call a command and suppress STDOUT and STDERR
def call_q(self, cmd, **kw):
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
        *host*: {*repo.host*} | ``None`` | :class:`str`
            Name of remote host (if not ``None``) on which to run
        *executable*: {``"sh"``} | :class:`str`
            Name of shell to use if on remote host
    :Outputs:
        *ierr*: :class:`int`
            Return code from executing command
    :Versions:
        * 2020-12-24 ``@ddalle``: Version 1.0
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
        *host*: {*repo.host*} | ``None`` | :class:`str`
            Name of remote host (if not ``None``) on which to run
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
        * 2020-12-24 ``@ddalle``: Version 1.0
    """
    # Process keyword args
    cwd = kw.get("cwd", os.getcwd())
    stdout = kw.get("stdout")
    stderr = kw.get("stderr")
    encoding = kw.get("encoding", "utf-8")
    host = kw.get("host")
    executable = kw.get("executable", "sh")
    # Check if remote
    if host:
        # Create a process
        proc = Popen(
            ["ssh", "-q", host, executable],
            stdin=PIPE, stdout=stdout, stderr=stderr)
        # Go to the folder
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

