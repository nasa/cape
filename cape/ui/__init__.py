r"""
:mod:`cape.ui` Readline-based user interface to CAPE
=======================================================

"""

# Standard library
import os
import readline
import shlex
import socket
import subprocess
from typing import Tuple

# Local imports
from . import capeconfig
from .promptutils import sprintf_color_rl


# Constatnts
CAPE_HISTORY_LENGTH = 1000


# Main function
def main() -> Tuple[int, dict]:
    # Get history file
    histfile = capeconfig.get_cape_opt("HistoryFile")
    # If relative path, join with CacheDir
    if not os.path.isabs(histfile):
        cachedir = capeconfig.get_cape_opt("CacheDir")
        histfile = os.path.join(cachedir, histfile)
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
    n_commands = 0
    # Loop until user requests exit
    while True:
        # Get last two parts
        _dir, basename = os.path.split(os.getcwd())
        parname = os.path.basename(_dir)
        # Generate a prompt
        dirname = os.path.join(parname, basename)
        user_prompt = sprintf_color_rl(
            f"{hostname}:{dirname}$ ", ["bold", "green"])
        try:
            user_message = input(user_prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        # Recycle if empty prompt given
        if not user_message:
            continue
        # Check for special commands
        if user_message.strip().startswith("cd "):
            parts = user_message.split(' ', 1)
            try:
                os.chdir(parts[1])
            except FileNotFoundError:
                print(f"CAPE> Folder not found: '{parts[1]}")
            except PermissionError:
                print(f"CAPE> Permission denied: '{parts[1]}'")
        # Run the command
        ierr = subprocess.run(shlex.split(user_message))
        # Count commands
        n_commands += 1
    # Return code
    return ierr.returncode, {"n_commands": n_commands}


