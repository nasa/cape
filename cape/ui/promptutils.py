#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
:mod:`cape.promptutils`: Useful tools for interactive CLI prompts
===================================================================

This includes a CAPE-specific autocompletion class
:class:`CfdxCompleter`, which is used in :mod:`cape.ui`.

"""

from __future__ import annotations

# Standard library
import fnmatch
import glob
import os
import re
import readline
import shlex
from typing import Any, Optional

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
    # Foreground colors
    'black':     '\x1b[30m',
    'darkgray':  '\x1b[30;01m',
    'red':       '\x1b[31;01m',
    'darkred':   '\x1b[31m',
    'green':     '\x1b[32;01m',
    'darkgreen': '\x1b[32m',
    'yellow':    '\x1b[33;01m',
    'brown':     '\x1b[33m',
    'blue':      '\x1b[34;01m',
    'darkblue':  '\x1b[34m',
    'purple':    '\x1b[35m',
    'fuchsia':   '\x1b[35;01m',
    'turquoise': '\x1b[36m',
    'teal':      '\x1b[36;01m',
    'white':     '\x1b[37;01m',
    'lightgray': '\x1b[37m',
    # Bright foreground colors
    'bright-black':   '\x1b[90m',
    'bright-red':     '\x1b[91m',
    'bright-green':   '\x1b[92m',
    'bright-yellow':  '\x1b[93m',
    'bright-blue':    '\x1b[94m',
    'bright-magenta': '\x1b[95m',
    'bright-cyan':    '\x1b[96m',
    'bright-white':   '\x1b[97m',
    # Background colors
    'bg-black':     '\x1b[40m',
    'bg-red':       '\x1b[41m',
    'bg-green':     '\x1b[42m',
    'bg-yellow':    '\x1b[43m',
    'bg-blue':      '\x1b[44m',
    'bg-magenta':   '\x1b[45m',
    'bg-cyan':      '\x1b[46m',
    'bg-white':     '\x1b[47m',
    # Bright background colors
    'bg-bright-black':   '\x1b[100m',
    'bg-bright-red':     '\x1b[101m',
    'bg-bright-green':   '\x1b[102m',
    'bg-bright-yellow':  '\x1b[103m',
    'bg-bright-blue':    '\x1b[104m',
    'bg-bright-magenta': '\x1b[105m',
    'bg-bright-cyan':    '\x1b[106m',
    'bg-bright-white':   '\x1b[107m',
    # Text attributes
    'bold':       '\x1b[01m',
    'faint':      '\x1b[02m',
    'italic':     '\x1b[03m',
    'underline':  '\x1b[04m',
    'blink':      '\x1b[05m',
    'standout':   '\x1b[03m',
    'reverse':    '\x1b[07m',
    'conceal':    '\x1b[08m',
    'strikethrough': '\x1b[09m',
    # Special turn-off sequences
    'un-bold':          "\x1b[22m",
    'un-italic':        "\x1b[23m",
    'un-underline':     "\x1b[24m",
    'un-blink':         "\x1b[25m",
    'un-reverse':       "\x1b[27m",
    'un-strikethrough': "\x1b[29m",
    # Reset
    'plain':      '\x1b[0m',
    'reset':      '\x1b[39;49;00m',
}


# Regular expression to recognize "@{n}" entries
REGEX_AT = re.compile("@([0-9]+)")
REGEX_QUOTE = re.compile('"' + "'")

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
        "cmdname",
        "parser",
        "subparser",
        "solver",
        "role",
    )

    def __init__(self, cls: type["ArgReader"]):
        r"""Initialize a CAPE autocompleter"""
        #: :class:`list`\ [:class:`str`]
        #: Current list of matches
        self.matches = None
        #: :class:`type`
        #: Subclass of :class:`cape.argread.ArgReader` for completion
        self.cls = cls
        #: :class:`cape.argread.ArgReader`
        #: Instance of *cls* instantiated with args so far
        self.parser = cls()
        #: :class:`str` | ``None``
        #: Name of sub-command
        self.cmdname = None
        #: :class:`str` | ``None``
        #: Name of current solver based on first word of command
        self.solver = None
        #: :class:`str` | ``None``
        #: Role of current word
        self.role = None

    def __call__(self, text: str, state: int) -> Optional[str]:
        # Get list of suggestions starting with *text*
        if state == 0:
            self.matches = self.get_suggestions(text)
        # Use suggestion if given
        if state < len(self.matches):
            return self.matches[state]
        # Default to None
        return None

    def get_suggestions(self, text: str) -> list[str]:
        r"""Get completions; add ``' '`` or ``'/'`` if appropriate

        :Call:
            >>> suggestions = comp.genr8_get_suggestionssuggestions(text)
        :Inputs:
            *comp*: :class:`CfdxCompleter`
                CAPE front desk autocompleter
            *text*: :class:`str`
                Current text of current word
        :Outputs
            *suggestions*: :class:`list`\ [:class:`str`]
                List of suggested completions for current word
        """
        # Get matches
        matches = self.genr8_suggestions(text)
        # Move on to next word if unique suggestion and not a folder
        if len(matches) == 1:
            # Get that unique match
            mtch = matches[0]
            # Check if it's a folder
            if (self.role == "filename") and os.path.isdir(mtch):
                # Add a slash
                matches[0] = mtch + os.sep
            else:
                # Add a space to move onto next option
                matches[0] = mtch + " "
        # Output
        return matches

    def genr8_suggestions(self, text: str) -> list[str]:
        r"""Generate list of suggestions based on current prompt

        :Call:
            >>> suggestions = comp.genr8_suggestions(text)
        :Inputs:
            *comp*: :class:`CfdxCompleter`
                CAPE front desk autocompleter
            *text*: :class:`str`
                Current text of current word
        :Outputs
            *suggestions*: :class:`list`\ [:class:`str`]
                List of suggested completions for current word
        """
        # Reset role
        self.role = None
        # Get position
        line = readline.get_line_buffer()
        # Split line back into argv
        argv = shlex.split(line.lstrip('$').lstrip())
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
            self.role = "pathcmd"
            # Output
            return xcape + xpath + xfile
        # Set solver name based on word 0
        if argv[0] in CAPE_EXECS:
            # Running one of the CAPE commands
            self.solver = argv[0]
        else:
            # Not a CAPE command
            self.role = "filename"
            return complete_filenames(text)
        # Check for option vs value
        if text.startswith("-"):
            # Filter option list
            self.decide_cmdname(argv, j)
            # Filter existing options
            opts = self.genr8_optlist(text)
            # Filter to multi-letter if given two dashes
            if text.startswith("--"):
                opts = [o for o in opts if len(o) > 1]
            # Initialize output
            suggestions = []
            # Add one or two dashes
            for opt in opts:
                # Check length
                prefix = '-' if (len(opt) == 1) else '--'
                # Append to list
                suggestions.append(f"{prefix}{opt}")
            # Good
            return suggestions
        elif ("=" in text) and not REGEX_QUOTE.match(text):
            # Using alternate option=value syntax
            opt, text = text.split('=', 1)
            # Get suggested values
            vals = self.genr8_optvals(opt, text)
            # Append LHS to completesions
            suggestions = [f'{opt}={v}' for v in vals]
            return suggestions
        # For arg 1, suggest using canonical format
        if j == 1 and len(argv) <= 2:
            return self.genr8_cmdnames(text)
        # Check if we're in an option or not
        if j > 1:
            # Get previous argument to see if it's an option
            prev = argv[j - 1]
            opt = prev.lstrip('-')
            if prev.startswith("-") and opt not in self.cls._optlist_noval:
                # Get completions for that value
                return self.genr8_optvals(opt, text)
        # For new words, let's recommend options
        if len(text) == 0:
            # Filter option list
            self.decide_cmdname(argv, j)
            opts = self.genr8_optlist('')
            # Append '-' to each
            suggestions = []
            # Add one or two dashes
            for opt in opts:
                # Check length
                prefix = '-' if (len(opt) == 1) else '--'
                # Append to list
                suggestions.append(f"{prefix}{opt}")
            # Output
            self.role = "optname"
            return suggestions
        # Default to file names
        self.role = "filename"
        return complete_filenames(text)

    def decide_cmdname(self, argv: list, j: int):
        # Attempt to decide sub-command
        try:
            # Parse arguments so far
            cmdname, _ = self.parser.decide_cmdname(argv[:j])
            # Save it if successful
            if cmdname == "run" and len(argv) > 1 and argv[1] != "run":
                # Avoid defaulting to "run" too early
                self.cmdname = None
            elif cmdname == "ui" and len(argv) > 1 and argv[1] != "ui":
                # Avoid defaulting to "ui" too early
                self.cmdname = None
            else:
                # Save the result
                self.cmdname = cmdname
        except Exception:
            # Current command doesn't map to sub-parser or is invalid
            self.cmdname = None

    def genr8_cmdnames(self, text: str) -> list[str]:
        r"""Suggest list of CAPE command name completions

        :Call:
            >>> cmds = comp.genr8_cmdnames(text)
        :Inputs:
            *comp*: :class:`CfdxCompleter`
                CAPE front desk autocompleter
            *text*: :class:`str`
                Current user input to match against command names
        :Outputs:
            *cmds*: :class:`list`\ [:class:`str`]
                List of command names matching *text* pattern
        """
        # Fitler existing options
        self.role = "cmdname"
        return fnmatch.filter(self.cls._cmdlist, f"{text}*")

    def genr8_optlist(self, text: str) -> list[str]:
        r"""Suggest list of option name completions

        :Call:
            >>> opts = comp.genr8_optlist(text)
        :Inputs:
            *comp*: :class:`CfdxCompleter`
                CAPE front desk autocompleter
            *text*: :class:`str`
                Current user input to match against option names
        :Outputs:
            *opts*: :class:`list`\ [:class:`str`]
                List of options matching *text* pattern
        """
        # Get name of option so far (w/o '--')
        optpat = text.lstrip('-')
        # Get full list of available options
        if self.cmdname is None:
            # Use full set
            optlist = self.cls._optlist
        else:
            # Use subset
            subcls = self.cls._cmdparsers.get(self.cmdname)
            # Check before assuming that worked
            if subcls is None:
                optlist = self.cls._optlist
            else:
                optlist = subcls._optlist
        # Filter existing options
        opts = fnmatch.filter(optlist, f"{optpat}*")
        # Output
        self.role = "optname"
        return opts

    def genr8_optvals(self, opt: str, text: str) -> list[str]:
        r"""Suggest list of possible option values

        :Call:
            >>> suggestions = comp.genr8_optvals(opt, text)
        :Inputs:
            *comp*: :class:`CfdxCompleter`
                CAPE front desk autocompleter
            *opt*: :class:`str`
                Name of option whose values are being completed
            *text*: :class:`str`
                Current user input to match against allowed values
        :Outputs:
            *suggestions*: :class:`list`\\[:class:`str`]
                List of allowed values for *opt* that start with *text*
        """
        # Create pattern
        pat = f"{text}*"
        # Check what kind of option it is
        if opt in self.cls._optvals:
            # Filter the allowed option values
            self.role = "optval"
            return fnmatch.filter(self.cls.get_optvals(opt), pat)
        elif opt in self.cls._optlist_file:
            # Search for files
            self.role = "filename"
            return complete_filenames(text)
        # Otherwise no special completions
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
