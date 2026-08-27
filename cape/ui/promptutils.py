#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
``promptutils``: Useful tools for interactive CLI prompts
============================================================

"""

from __future__ import annotations

# Standard library
import fnmatch
import glob
import os
import re
import readline
import shlex
from typing import Any, Callable, Optional

# Local imports
from ..argread import ArgReader

# Fix prompt colors for Windows
try:
    # Import colorama
    from colorama import init
    # Initialize colorama to support ANSI escape codes on Windows
    init(autoreset=True)
except ModuleNotFoundError:
    init = None


# Console colors and attributes
CONSOLE = {
    'black':     '\x1b[30m',
    'blink':     '\x1b[05m',
    'blue':      '\x1b[34;01m',
    'bold':      '\x1b[01m',
    'brown':     '\x1b[33m',
    'darkblue':  '\x1b[34m',
    'darkgray':  '\x1b[30;01m',
    'darkgreen': '\x1b[32m',
    'darkred':   '\x1b[31m',
    'faint':     '\x1b[02m',
    'fuchsia':   '\x1b[35;01m',
    'green':     '\x1b[32;01m',
    'italic':    "\x1b[3m",
    'lightgray': '\x1b[37m',
    'plain':     "\x1b[0m",
    'purple':    '\x1b[35m',
    'red':       '\x1b[31;01m',
    'reset':     '\x1b[39;49;00m',
    'standout':  '\x1b[03m',
    'strikethrough': '\x1b[29m',
    'teal':      '\x1b[36;01m',
    'turquoise': '\x1b[36m',
    'underline': '\x1b[04m',
    'white':     '\x1b[37;01m',
    'yellow':    '\x1b[33;01m',
}


# Regular expression to recognize "@{n}" entries
REGEX_AT = re.compile("@([0-9]+)")

# Generic completer settings
readline.set_completer_delims(' \t\n')
readline.parse_and_bind("tab: complete")

# CAPE main executable names
CAPE_EXECS = [
    "cape",
    "pycart",
    "pyfun",
    "pykes",
    "pylava",
    "pylch",
    "pyover",
]


class CfdxCompleter:
    __slots__ = (
        "matches",
        "cls",
    )

    def __init__(self, cls: type["ArgReader"]):
        r"""Initialize a CAPE autocompleter"""
        #: :class:`list`\ [:class:`str`]
        #: Current list of matches
        self.matches = None
        #: :class:`type`
        #: Subclass of :class:`cape.argread.ArgReader` for completion
        self.cls = cls

    def __call__(self, text: str, state: int) -> Optional[str]:
        # Get list of suggestions starting with *text*
        if state == 0:
            self.matches = self.genr8_suggestions(text)
        # Use suggestion if given
        if state < len(self.matches):
            return self.matches[state]
        # Default to None
        return None

    def genr8_suggestions(self, text: str) -> list:
        # Get position
        line = readline.get_line_buffer()
        # Split line back into argv
        argv = shlex.split(line)
        # Get index of current word
        if text in argv:
            # Proper index
            j = argv.index(text)
        else:
            # Failed, probably because of quotes
            j = max(0, len(argv) - 1)
        # Special case for first word
        if j == 0:
            # Get three types of completions
            xcape = complete_xcape(text)
            xpath = complete_pathcmds(text)
            xfile = complete_xfilenames(text)
            print(f"\n> cape: {xcape}")
            print(f"> path: {xpath}")
            print(f"> file: {xfile}")
            # Output
            return xcape + xpath + xfile
        # Default to file names
        return complete_filenames(text)

    def genr8_extra_suggestions(self, text: str) -> list:
        return []


# Get list of files matching current glob
def complete_filenames(text: str) -> list[str]:
    r"""Return a tab-completion suggestion based on file names

    :Call:
        >>> suggestions = complete_filenames(text)
    :Inputs:
        *text*: :class:`str`
            User input so far
    :Outputs:
        *suggestions*: ::class:`list`\ [class:`str`]
            Extant files whose names start with *text*
    """
    return sorted(glob.glob(text + '*'))


# Get list of matching CAPE executables
def complete_xcape(text: str) -> list[str]:
    r"""Return tab-completion suggestions of CAPE executables

    :Call:
        >>> suggestions = complete_xcape(text)
    :Inputs:
        *text*: :class:`str`
            User input so far
    :Outputs:
        *suggestions*: ::class:`list`\ [class:`str`]
            Sortec CAPE file names that start with *text*
    """
    return fnmatch.filter(CAPE_EXECS, f"{text}*")


# Get list of executable files matching current glob
def complete_xfilenames(text: str) -> list[str]:
    r"""Return a filtered list of executable file names

    :Call:
        >>> suggestions = complete_xfilenames(text, state)
    :Inputs:
        *text*: :class:`str`
            User input so far; must start with ``"./"``
    :Outputs:
        *suggestions*: ::class:`list`\ [class:`str`]
            Extant files whose names start with *text*
    """
    # Only check ./ completions
    if not text.startswith("./"):
        return []
    # Initialize output
    matches = set()
    # Loop through matching files
    for fname in complete_filenames(text):
        # Check if executable
        if os.path.exists(fname) and os.access(fname, os.X_OK):
            # Criteria met!
            matches.add(fname)
    # Output
    return sorted(matches)


# Get list of $PATH executables matching current pattern
def complete_pathcmds(text: str) -> list[str]:
    r"""Return a tab-completion suggestion from ``$PATH`` executables

    :Call:
        >>> suggestions = complete_pathcmds(text)
    :Inputs:
        *text*: :class:`str`
            User input so far
    :Outputs:
        *suggestions*: ::class:`list`\ [class:`str`]
            Reachable executalbes whose names start with *text*
    """
    # Initialize output
    matches = set()
    # Loop through PATH folders
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        # Ensure folder actually exists
        if not folder:
            continue
        # Look for executables in that folder
        try:
            # Loop through files in folder
            for name in os.listdir(folder):
                # Apply filder
                if not name.startswith(text):
                    continue
                # Get absolute path
                path = os.path.join(folder, name)
                # Check if executable
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    # Criteria met!
                    matches.add(name)
        except OSError:
            pass
    # Convert to list
    return sorted(matches)


# Function to get user input using a colored prompt
def prompt_color(
        txt: str,
        vdef: Optional[Any] = None,
        vopt: Optional[list] = None,
        color: str = "green",
        prompt: str = '>',
        completer: Optional[Callable] = None,
        glob: bool = False) -> Any:
    r"""Get user input using a colorized prompt

    :Call:
        >>> v = prompt_color(txt, vdef=None)
    :Inputs:
        *txt*: :class:`str`
            Text of the question on the same line as prompt
        *vdef*: {``None``} | :class:`object`
            Default value (if any)
        *vopt*: {``None``} | :class:`list`
            List of possible or suggested values (optional)
        *color*: {``"green"``} | :class:`str`
            Color name
        *completer*: {``None``} | **callable**
            Function to return list of suggestions given current text
        *prompt*: {``">"``} | :class:`str`
            Character(s) to use as prompt
    :Outputs:
        *v*: :class:`str` | *vdef* | ``vopt[j]``
            User input or default value
    """
    # Default vdef --> vopt
    vopt = vdef if (vopt is None) else vopt
    # Three versions of option list; two will be empty
    msg1 = _dumps_vopt_list(txt, vdef, vopt, prompt)
    msg2 = _dumps_vdef(txt, vdef, vopt, prompt)
    msg3 = _dumps_plain(txt, vdef, vopt, prompt)
    # Combine all three
    msg = msg1 + msg2 + msg3
    # Create a completer
    comp = PromptCompleter(glob, vopt)
    # Check for custom function
    comp.func = completer
    # Turn custom completin class on
    readline.set_completer(comp)
    # Substantiate default
    vdef = vopt if vdef is None else vdef
    vdef = vdef if not isinstance(vdef, list) else vdef[0]
    # Read input from command line (ignore lead/trail spaces)
    vraw = input_color(msg, color)
    # Check if it's an "@"
    if msg1 and REGEX_AT.fullmatch(vraw):
        # Get the number provided by user
        n = int(REGEX_AT.match(vraw).group(1))
        # Return that value (0-based)
        v = vopt[n - 1]
    elif vdef and (not vraw):
        # Use the default value instead
        v = vdef
    else:
        # Return the user's value, even if empty
        v = vraw
    # Inform user what value was used
    print(f"--> using '{v}'")
    # Output
    return v


# Make a raw request
def input_color(prompt: str, color: str = "black") -> str:
    r"""Modify built-in :func:`input` to also set color

    :Call:
        >>> raw = input_color(prompt, color)
    :Inputs:
        *prompt*: :class:`str`
            Text to display prior to requesting input
        *color*: :class:`str`
            Common name of color to use for prompt
    :Outputs:
        *raw*: :class:`str`
            User's input
    """
    # Get color
    col = CONSOLE.get(color, CONSOLE["black"])
    reset = CONSOLE["reset"]
    # Form a prompt with formatting
    prompt_txt = f"{col}{prompt}{reset}"
    # Request a response
    return input(prompt_txt).strip()


# Print with color
def print_color(msg: str, color: Optional[str | list] = None):
    r"""Print message using a specified ANSI color

    :Call:
        >>> print_color(msg, color=None)
    :Inputs:
        *prompt*: :class:`str`
            Text to display prior to requesting input
        *color*: {``None``} | :class:`str` | :class:`list`
            Name(s) of color (or style) to use for prompt
    """
    # Print combined string
    print(sprintf_color(msg, color))


# Generate string with color
def sprintf_color(msg: str, color: Optional[str | list] = None) -> str:
    r"""Create string with optional color instructions

    :Call:
        >>> prompt = sprintf_color(msg, color=None)
    :Inputs:
        *prompt*: :class:`str`
            Text to display prior to requesting input
        *color*: {``None``} | :class:`str` | :class:`list`
            Name(s) of color (or style) to use for prompt
    :Outputs:
        *prompt*: :class:`str`
            String with color instructions at beginning and end
    """
    # Check for color
    if color is None:
        return msg
    # Ensure list
    colors = color if isinstance(color, (list, tuple)) else [color]
    # Initialize color chars
    colchars = []
    # Loop through colors
    for colj in colors:
        # Check if present
        if colj not in CONSOLE:
            raise ValueError(f"Could not find color '{colj}'")
        # Append
        colchars.append(CONSOLE[colj])
    # Combine color info
    col = ''.join(colchars)
    # Get reset marker
    reset = CONSOLE["reset"]
    # Form a prompt with formatting
    return f"{col}{msg}{reset}"


# Generate string with color for readline prompts
def sprintf_color_rl(msg: str, color: Optional[str | list] = None) -> str:
    r"""Create string with color for readline input prompts

    This wraps ANSI escape codes with readline's special markers
    (\x01 and \x02) so readline can correctly calculate prompt width
    and handle line wrapping/cursor positioning.

    :Call:
        >>> prompt = sprintf_color_rl(msg, color=None)
    :Inputs:
        *msg*: :class:`str`
            Text to display prior to requesting input
        *color*: {``None``} | :class:`str` | :class:`list`
            Name(s) of color (or style) to use for prompt
    :Outputs:
        *prompt*: :class:`str`
            String with readline-wrapped color instructions
    """
    # Check for color
    if color is None:
        return msg
    # Ensure list
    colors = color if isinstance(color, (list, tuple)) else [color]
    # Initialize color chars
    colchars = []
    # Loop through colors
    for colj in colors:
        # Check if present
        if colj not in CONSOLE:
            raise ValueError(f"Could not find color '{colj}'")
        # Append
        colchars.append(CONSOLE[colj])
    # Combine color info and wrap with readline markers
    col = '\x01' + ''.join(colchars) + '\x02'
    # Get reset marker and wrap it too
    reset = '\x01' + CONSOLE["reset"] + '\x02'
    # Form a prompt with formatting
    return f"{col}{msg}{reset}"


# Display list of options
def _dumps_vopt_list(
        txt: str,
        vdef: Optional[Any] = None,
        vopt: Optional[list] = None,
        prompt: str = '>') -> str:
    # Check if options give are a list
    if not isinstance(vopt, (list, tuple)):
        return ''
    # Initial portion of prompt using pre-specified prompt
    msg = f"{txt}:\n"
    # Default default value is first entry in *vopt_list*
    if vdef is None:
        vdef = vopt[0]
    # Check if given a list of default values
    if isinstance(vdef, (list, tuple)):
        # Use the first
        vdef = vdef[0]
    # Check if default is in option; if soe get the index
    jdef = None if vdef not in vopt else vopt.index(vdef)
    # Loop through options
    for j, opt in enumerate(vopt):
        # Format message
        if j == jdef:
            # Highlight first option as the true default
            msgj = f"    @{j+1}: [{opt}]\n"
        else:
            # Use option number
            msgj = f"    @{j+1}: {opt}\n"
        # Append to overall prompt
        msg += msgj
    # Append user input prompt
    return msg + prompt + ' '


# Display list of options
def _dumps_vdef(
        txt: str,
        vdef: Optional[Any] = None,
        vopt: Optional[list] = None,
        prompt: str = '>') -> str:
    # Check if options give are a list
    if isinstance(vopt, (list, tuple)):
        return ''
    # Check for any reasonable default
    if (vdef is None) and (vopt is None):
        return ''
    # Use *vdef* or *vopt*
    vdef = vopt if vdef is None else vdef
    # Form the prompt with single default value
    return f"{txt} [{vdef}]:\n{prompt} "


# Display with no list or default
def _dumps_plain(
        txt: str,
        vdef: Optional[Any] = None,
        vopt: Optional[list] = None,
        prompt: str = '>') -> str:
    # Check for any default/option list; use prior function
    if (vdef is not None) or (vopt is not None):
        return ''
    # Form the prompt without default value
    return f"{txt}:\n{prompt} "
