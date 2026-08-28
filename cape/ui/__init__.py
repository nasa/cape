r"""
:mod:`cape.ui` Readline-based user interface to CAPE
=======================================================

This class provides an interactive interface for the main CAPE tools
(that is, running CFD) without major third-party dependencies. This
simplified CAPE UI is launched using

.. code-block:: console

    $ cape ui

There are two main applications to using this user interface:

1.  **Targeted tab-completion**

    The user interface provides CAPE-aware tab completions based on
    details of the CLI defined in :mod:`cape.cfdx.cli`, including

    * inferring the sub-command such as ``cape extend``, ``cape run``,
      etc. (see :attr:`cape.cfdx.cli.CfdxFrontDesk._cmdlist`);
    * knowledge of which options are available for each sub-command
      (for example ``cape edit-json -`` and ``cape c -`` will have
      different tab-completion suggestions);
    * knowledge of which options expect file names.

2.  **Dedicated history**

    The CAPE UI maintains a separate history file so that users can
    recall previous commands using the "UP" arrow key.

While using the CAPE UI pseudo-shell, users will still be able to run
non-CAPE commands, though in that case tab-completion may not be as
smart as the native shell.
"""

# Standard library
import os
import readline
import shlex
import socket
import subprocess
from typing import Optional, Tuple

# Local imports
from .. import capeconfig
from .promptutils import CfdxCompleter, sprintf_color_rl


# Constants
CAPE_HISTORY_LENGTH = 1000
EXIT_CMDS = (
    "exit",
    "quit",
    "exit()",
    "quit()",
)


# Main function
def main(cls: Optional[type] = None) -> Tuple[int, dict]:
    r"""Main interactive UI function

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
    """
    # Get history file
    histfile = capeconfig.get_cape_opt("HistoryFile")
    # If relative path, join with CacheDir
    if not os.path.isabs(histfile):
        cachedir = capeconfig.get_cape_opt("CacheDir")
        histfile = os.path.join(cachedir, histfile)
    # Read CAPE history from previous sessions
    try:
        readline.read_history_file(histfile)
        readline.set_history_length(CAPE_HISTORY_LENGTH)
    except FileNotFoundError:
        pass
    # Enable tab completion (optional)
    readline.parse_and_bind("tab: complete")
    # Get hostname
    hostname = socket.gethostname().split('.')[0]
    # Count number of commands run
    n_failure = 0
    n_commands = 0
    # Create autocompleter
    completer = CfdxCompleter(cls)
    readline.set_completer(completer)
    # Loop until user requests exit
    while True:
        # Get last two parts
        _dir, basename = os.path.split(os.getcwd())
        parname = os.path.basename(_dir)
        # Generate a prompt
        dirname = os.path.join(parname, basename)
        user_prompt = sprintf_color_rl(
            f"CAPE {hostname}:{dirname}$ ", ["bold", "green"])
        # Get user input
        try:
            user_message = input(user_prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            ierr = 0
            break
        # Recycle if empty message given
        if not user_message:
            continue
        # Check for special commands
        if user_message.strip().startswith("cd "):
            parts = user_message.split(' ', 1)
            try:
                os.chdir(parts[1])
            except FileNotFoundError:
                print(f"CAPE> Folder not found: '{parts[1]}")
                n_failure += 1
            except PermissionError:
                print(f"CAPE> Permission denied: '{parts[1]}'")
                n_failure += 1
            # Count as a command
            n_commands += 1
            # Skip cd
            continue
        elif user_message in EXIT_CMDS:
            break
        # Run the command
        try:
            proc = subprocess.run(shlex.split(user_message))
            ierr = proc.returncode
        except PermissionError:
            ierr = 13
        except Exception:
            ierr = 1
        except KeyboardInterrupt:
            ierr = 0
            break
        # Count commands
        n_commands += 1
        if ierr:
            n_failure += 1
    # Save readline history on exit
    try:
        readline.write_history_file(histfile)
    except Exception:
        pass
    # Return code
    return ierr, {"commands": n_commands, "failures": n_failure}


