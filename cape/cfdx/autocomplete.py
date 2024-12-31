
# Standard library
import fnmatch
import glob
import os
import shlex
import sys

# Local imports
from . import cli


# Option and command lists
CMDLIST = cli.CfdxFrontDesk._cmdlist
OPTLIST = cli.CfdxFrontDesk._optlist
OPTLISTNEG = cli.CfdxFrontDesk._help_opt_negative


def genr8_completions(line: str) -> list:
    # Split into words
    words = shlex.split(line)
    # Get current index
    i = len(words)
    # Check for terminal space (moved onto next word)
    i = i + 1 if line.endswith(' ') else i
    # Get last word
    cur = '' if line.endswith(' ') else words[-1]
    # Check for first word
    if i <= 2:
        # Check option vs cmdname
        if cur.startswith("-"):
            # Found an option name; filter available commands
            return _genr8_opt_completions(cur)
        else:
            # Filter command list
            matches = fnmatch.filter(CMDLIST, f"{cur}*")
        # Return those
        return matches
    # Filter options if processing an option
    if cur.startswith("-"):
        return _genr8_opt_completions(cur)
    # Get previous word
    prev = words[-2]
    prevopt = prev.lstrip('-') if prev.startswith('-') else ''
    # Check for special cases
    if prevopt in ("f", "json"):
        return genr8_file_completions(cur, ".json")
    # Fall back to glob
    return genr8_file_completions(cur)


def genr8_file_completions(prefix: str, ext: str = '') -> list:
    # Get basic matches
    matchlist = glob.glob(f"{prefix}*")
    # Filter files plus folders
    extmatch = fnmatch.filter(matchlist, f"*{ext}")
    # Add folders
    dirnames = [
        fname + os.sep
        for fname in matchlist if os.path.isdir(fname)]
    # Return all completions
    completions = extmatch + dirnames
    # Check for single directory
    if len(completions) == 1 and os.path.isdir(completions[0]):
        # List contents of that folder
        return genr8_file_completions(completions[0], ext)
    # Output
    return completions


def _genr8_opt_completions(cur: str) -> list:
    # Check prefix
    if cur.startswith("--no-"):
        # Filter option list based on portion after negation
        prefix = "--no-"
        optname = cur[5:]
        # Find matches
        matches = fnmatch.filter(OPTLISTNEG, f"{optname}*")
        # Reapply the prefix
        return [prefix + comp for comp in matches]
    if cur.startswith('-'):
        # Filter option list
        optname = cur.lstrip('-')
        # Find matches
        matches = fnmatch.filter(OPTLIST, f"{optname}*")
        # Use correct number of hyphens
        return ['-' * max(1, min(2, len(opt))) + opt for opt in matches]


if __name__ == "__main__":
    # Get last argument
    line = sys.argv[1]
    # Generate completions
    completions = genr8_completions(line)
    # Print completions
    print('\n'.join(completions))
