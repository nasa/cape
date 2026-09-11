r"""
:mod:`cape.tui`: Rich interactive terminal user interface to CAPE
====================================================================

This module provides a Rich-based interactive interface for the main
CAPE tools (that is, running CFD), launched using

.. code-block:: console

    $ cape tui

It provides the same core features as :mod:`cape.ui`, namely
CAPE-aware tab-completion and a dedicated history file, with a more
interactive display:

* **Rich display**
    A startup banner, colored headers and footers around each
    command showing its exit status and wall time, and a session
    summary on exit.

* **TUI meta-commands**
    Commands starting with ``:`` are handled by the TUI itself:

    * ``:help [<cmd>]``: table of CAPE commands, or details of one
    * ``:history [<n>]``: table of the *n* most recent commands
    * ``:!N``: rerun command No. ``N`` from the history table
    * ``:status``: show summary of the current session
    * ``:cd <dir>``: change folder (plain ``cd <dir>`` also works)
    * ``:pwd``: show current working folder
    * ``:clear``: clear the screen
    * ``:exit`` / ``:quit``: exit the TUI

* **Smart execution**
    Commands starting with ``cape``, ``pycart``, ``pyfun``, etc. are
    run in-process using :func:`cape.cfdx.cli.main`, while all other
    commands are run as system subprocesses.

"""

from __future__ import annotations

# Standard library
import os
import readline
import re
import shlex
import socket
import subprocess
import time
from typing import Optional, Tuple

# CAPE imports
from .. import capeconfig
from ..ui.promptutils import CAPE_EXECS

# Local imports
from .tuiutils import (
    CONSOLE,
    EXIT_CMDS,
    TuiCompleter,
    get_dirname,
    make_prompt,
    render_banner,
    render_command_rule,
    render_error,
    render_help,
    render_history,
    render_meta_help,
    render_session_stats,
    render_status_rule,
)


# Constants
CAPE_HISTORY_LENGTH = 1000

# Match commands to run with CAPE's in-process CLI
REGEX_CAPE_CLI = re.compile(rf"\$?\s*({'|'.join(CAPE_EXECS)})( |$)")

# Help-me topics for :help
META_HELP_TOPICS = ("tui", "meta")


# Main function
def main(cls: Optional[type] = None) -> Tuple[int, dict]:
    r"""Main Rich-based interactive TUI function

    :Call:
        >>> ierr, result = main(cls)
    :Inputs:
        *cls*: :class:`type`
            :class:`cape.argread.ArgReader` subtype, for completions
    :Outputs:
        *ierr*: :class:`int`
            Return code
        *result*: :class:`dict`
            Information about results of commands run

            * ``commands``: :class:`int` number of commands run
            * ``failures``: :class:`int` number of failed commands
            * ``tui_commands``: :class:`int` number of TUI meta-cmds
            * ``duration``: :class:`float` session wall time (sec)
    """
    # Get history file
    histfile = get_tui_histfile()
    # Read CAPE TUI history from previous sessions
    try:
        readline.read_history_file(histfile)
        readline.set_history_length(CAPE_HISTORY_LENGTH)
    except FileNotFoundError:
        pass
    # Enable tab completion (optional)
    readline.parse_and_bind("tab: complete")
    # Default completions class
    if cls is None:
        from ..cfdx.cli import CfdxFrontDesk
        cls = CfdxFrontDesk
    # Create autocompleter
    completer = TuiCompleter(cls)
    readline.set_completer(completer)
    # Get hostname
    hostname = socket.gethostname().split('.')[0]
    # Render banner
    render_banner(histfile)
    # Session statistics
    t0 = time.perf_counter()
    stats = {
        "commands": 0,
        "failures": 0,
        "tui_commands": 0,
        "duration": 0.0,
    }
    # Return code tracker
    ierr = 0
    # Loop until user requests exit
    while True:
        # Generate a prompt
        user_prompt = make_prompt(hostname, get_dirname(), ierr == 0)
        # Get user input
        try:
            user_message = input(user_prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        # Recycle if empty message given
        if not user_message:
            continue
        # Check for exit commands
        if user_message in EXIT_CMDS:
            break
        # Check for TUI meta-commands
        if user_message.startswith(":"):
            ierr = handle_meta_cmd(user_message, cls, stats, histfile)
            # Check for exit request
            if ierr is None:
                ierr = 0
                break
        # Check for folder-change commands
        elif user_message.startswith("cd "):
            ierr = run_cd(user_message)
            stats["commands"] += 1
            stats["failures"] += int(ierr != 0)
        # Run the command
        else:
            ierr = run_command(user_message)
            stats["commands"] += 1
            stats["failures"] += int(ierr != 0)
    # Save readline history on exit
    try:
        readline.write_history_file(histfile)
    except Exception:
        pass
    # Finalize statistics
    stats["duration"] = time.perf_counter() - t0
    # Render exit summary
    render_session_stats(stats, histfile, title="CAPE TUI summary")
    # Return code
    return ierr, stats


# Handle a TUI meta-command
def handle_meta_cmd(
        user_message: str,
        cls: type,
        stats: dict,
        histfile: str) -> Optional[int]:
    r"""Handle a TUI meta-command starting with ``:``

    :Call:
        >>> ierr = handle_meta_cmd(user_message, cls, stats, histfile)
    :Inputs:
        *user_message*: :class:`str`
            Entire command entered by user, e.g. ``":help run"``
        *cls*: :class:`type`
            :class:`cape.argread.ArgReader` subtype, for completions
        *stats*: :class:`dict`
            Session statistics (updated in place)
        *histfile*: {``None``} | :class:`str`
            Name of command history file
    :Outputs:
        *ierr*: :class:`int` | ``None``
            Return code; ``None`` if TUI should exit
    """
    # Split meta-command
    parts = shlex.split(user_message)
    # Name of meta-command
    metacmd = parts[0]
    # Count it
    stats["tui_commands"] += 1
    # Exit commands
    if metacmd in (":exit", ":quit"):
        return None
    elif metacmd == ":clear":
        CONSOLE.clear()
        return 0
    # Simple no-argument meta-commands
    try:
        arg = parts[1]
    except IndexError:
        arg = None
    # Other basic meta-commands
    if metacmd == ":pwd":
        CONSOLE.print(os.getcwd(), style="cyan")
        ierr = 0
    elif metacmd == ":cd":
        # Change to folder *arg*, or home folder
        ierr = run_cd(f"cd {arg}" if arg else "cd ~")
    elif metacmd == ":help":
        # Help about TUI or a CAPE command
        if arg in META_HELP_TOPICS:
            ierr = render_meta_help()
        else:
            ierr = render_help(cls, arg)
    elif metacmd == ":history":
        # Optional count of history entries
        try:
            n = int(arg) if arg else 25
        except ValueError:
            render_error(f"Bad history count: '{arg}'")
            return 1
        ierr = render_history(n)
    elif metacmd == ":status":
        render_session_stats(stats, histfile)
        ierr = 0
    elif metacmd.startswith(":!"):
        # Rerun command from history
        ierr = run_history_cmd(metacmd[2:])
    else:
        # Unknown meta-command
        metacmds = ":cd :clear :exit :help :history :pwd :status"
        render_error(f"Unrecognized TUI command: '{metacmd}'")
        CONSOLE.print(f"Try one of: {metacmds}")
        ierr = 16
    # Output
    return ierr


# Run command No. N from the history
def run_history_cmd(txt: str) -> int:
    r"""Rerun command No. *N* from the readline history

    :Call:
        >>> ierr = run_history_cmd(txt)
    :Inputs:
        *txt*: :class:`str`
            History entry number, e.g. ``"12"`` from ``:!12``
    :Outputs:
        *ierr*: :class:`int`
            Return code of rerun command
    """
    # Parse the index
    try:
        j = int(txt)
    except ValueError:
        render_error(f"Bad history entry number: '{txt}'")
        return 16
    # Check range
    nhist = readline.get_current_history_length()
    if j < 1 or j > nhist:
        render_error(f"History entry out of range 1:{nhist}: {j}")
        return 16
    # Get the command
    cmd = readline.get_history_item(j)
    # Check for meta-command
    if cmd.startswith(":"):
        render_error(f"Cannot rerun TUI command: {cmd}")
        return 16
    # Get the command; status
    CONSOLE.print(f"Rerunning history entry {j}:", style="cyan")
    CONSOLE.print(cmd, style="bold")
    # Run it (counts toward stats in caller? no; self-contained)
    return run_command(cmd)


# Handle folder-change commands
def run_cd(user_message: str) -> int:
    r"""Handle a folder-change command such as ``cd powerless/``

    :Call:
        >>> ierr = run_cd(user_message)
    :Inputs:
        *user_message*: :class:`str`
            Entire command entered by user, e.g. ``"cd powerless/"``
    :Outputs:
        *ierr*: :class:`int`
            Return code, ``0`` on success
    """
    # Split off the folder name
    parts = user_message.split(' ', 1)
    # Get folder
    target = os.path.expanduser(parts[1]) if len(parts) > 1 else "~"
    # Change folder
    try:
        os.chdir(target)
        ierr = 0
    except FileNotFoundError:
        render_error(f"Folder not found: '{target}'")
        ierr = 2
    except PermissionError:
        render_error(f"Permission denied: '{target}'")
        ierr = 13
    # Output
    return ierr


# Run a system or in-process CAPE command
def run_command(user_message: str) -> int:
    r"""Run one command from the CAPE TUI

    Commands starting with a CAPE executable (``cape``, ``pycart``,
    ``pyfun``, etc.) are run in-process using
    :func:`cape.cfdx.cli.main`; all others are run as system
    subprocesses.

    :Call:
        >>> ierr = run_command(user_message)
    :Inputs:
        *user_message*: :class:`str`
            Entire command entered by user, e.g. ``"cape c -I 4:8"``
    :Outputs:
        *ierr*: :class:`int`
            Return code of the command
    """
    # Delayed import of CAPE CLI (to avoid circular imports)
    from ..cfdx import cli
    # Try to split the command
    try:
        cmdlist = shlex.split(user_message)
    except ValueError as err:
        render_error(str(err))
        return 16
    # Show the command heading
    render_command_rule(user_message)
    # Start timer
    t0 = time.perf_counter()
    # Run the command
    try:
        # Check for CAPE in-process command
        if REGEX_CAPE_CLI.match(user_message):
            # Run in-process
            ierr = cli.main(argv=cmdlist)
        else:
            # Run as a system command
            proc = subprocess.run(cmdlist)
            ierr = proc.returncode
    except FileNotFoundError:
        # Command not installed
        render_error(f"Command not found: '{cmdlist[0]}'")
        ierr = 127
    except PermissionError:
        render_error(f"Permission denied: '{cmdlist[0]}'")
        ierr = 13
    except KeyboardInterrupt:
        CONSOLE.print("KeyboardInterrupt", style="bold red")
        ierr = 130
    except Exception:
        # Unexpetect error: show full traceback
        CONSOLE.print_exception(show_locals=False)
        ierr = 128
    # Duration of command
    dt = time.perf_counter() - t0
    # Render the status
    render_status_rule(user_message, ierr, dt)
    # Output
    return ierr


# Get the name of the history file
def get_tui_histfile() -> str:
    r"""Get the name of the CAPE TUI history file

    The file is controlled by the *TUIHistoryFile* option from the
    user's ``~/.capeconfig.json`` (or the ``$CAPE_TUI_HISTORY_FILE``
    environment variable). Relative paths are joined with the CAPE
    *CacheDir*.

    :Call:
        >>> histfile = get_tui_histfile()
    :Outputs:
        *histfile*: :class:`str`
            Name of TUI command history file
    """
    # Get history file
    histfile = capeconfig.get_cape_opt("TUIHistoryFile")
    # If relative path, join with CacheDir
    if not os.path.isabs(histfile):
        cachedir = capeconfig.get_cape_opt("CacheDir")
        histfile = os.path.join(cachedir, histfile)
    # Output
    return os.path.expanduser(histfile)
