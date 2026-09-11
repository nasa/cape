r"""
:mod:`cape.promptutils`: Simple tools for interactive CLI prompts
===================================================================

This module provides tools for auto-completion, colored formatting, and
more when prompting users for values interactively.
"""

# Standard library
import fnmatch
import glob
import re
import readline
from typing import Any, Callable, Optional

# Local imports
from .argread.clitext import CONSOLE

# Third-party OPTIONAL
try:
    from colorama import init

    # Initialize colorama to support ANSI escape codes on Windows
    init(autoreset=True)
except Exception:
    pass


# Regular expression to recognize "@{n}" entries
REGEX_AT = re.compile("@([0-9]+)")

# Generic completer settings
readline.set_completer_delims(' \t\n')
readline.parse_and_bind("tab: complete")


class PromptCompleter:
    __slots__ = (
        "glob",
        "vopt",
        "func",
    )

    def __init__(self, glob: bool = False, vopt: Optional[list] = None):
        self.glob = glob
        self.vopt = vopt
        self.func = None

    def __call__(self, text: str, state: int) -> Optional[str]:
        # Get list of suggestions starting with *text*
        suggestions = self.genr8_suggestions(text)
        # Append ``None`` (for no-match case) and index it
        suggestions.append(None)
        return suggestions[state]

    def genr8_suggestions(self, text: str) -> list:
        # Initialize list of values
        suggestions = []
        # Get list of values
        if isinstance(self.vopt, (tuple, list)):
            suggestions.extend(fnmatch.filter(self.vopt, f"{text}*"))
        # Complete on file names if appropriate
        if self.glob:
            suggestions.extend(glob.glob(f"{text}*"))
        # Add any extra suggestions (custom)
        extra = self.genr8_extra_suggestions(text)
        suggestions.extend(extra)
        # Check for extra function
        if callable(self.func):
            custom = self.func(text)
            if isinstance(custom, (list, tuple)):
                suggestions.extend(custom)
        # Output
        return suggestions

    def genr8_extra_suggestions(self, text: str) -> list:
        return []


def complete_glob(text: str, state: int) -> Optional[str]:
    r"""Return a tab-completion suggestion based on file names

    :Call:
        >>> suggestion = complete_glob(text, state)
    :Inputs:
        *text*: :class:`str`
            User input so far
        *state*: :class:`int`
            Return the suggestion of index *state*, (usually ``0``)
    :Outputs:
        *suggestion*: :class:`str` | ``None``
            A file whose name starts with *text* if possible
    """
    return (glob.glob(text + '*') + [None])[state]


# Function to get user input using a colored prompt
def prompt_color(
        txt: str,
        vdef: Optional[Any] = None,
        vopt: Optional[list] = None,
        color: str = "green",
        prompt: str = '>',
        completer: Optional[Callable] = None,
        glob: bool = False,
        show: bool = True,
        oneline: bool = False,
        pcolor: Optional[str] = None,
        atcolor: Optional[str] = None,
        ocolor: Optional[str] = None) -> Any:
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
            Color name for entire prompt
        *completer*: {``None``} | **callable**
            Function to return list of suggestions given current text
        *prompt*: {``">"``} | :class:`str`
            Character(s) to use as prompt
        *glob*: ``True`` | {``False``}
            Use glob to suggest file-name-based completions
        *show*: {``True``} | ``False``
            Print the final selection for confirmation
        *oneline*: ``True`` | {``False``}
            Option to display values as compact one-line list ``[y]/n``
    :Outputs:
        *v*: :class:`str` | *vdef* | ``vopt[j]``
            User input or default value
    """
    # Default vdef --> vopt
    vopt = vdef if (vopt is None) else vopt
    # Convert extra color options to color
    pcol = '' if pcolor is None else CONSOLE.get(pcolor, '')
    acol = '' if pcolor is None else CONSOLE.get(atcolor, '')
    ocol = '' if pcolor is None else CONSOLE.get(ocolor, '')
    # Args passed to option printers
    args = (txt, vdef, vopt, prompt, pcol, acol, ocol)
    # Three versions of option list; two will be empty
    if oneline:
        msg1 = _dumps_vopt_oneline(*args)
    else:
        msg1 = _dumps_vopt_list(*args)
    msg2 = _dumps_vdef(*args)
    msg3 = _dumps_plain(*args)
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
    if show:
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
    col = CONSOLE.get(color, '')
    reset = CONSOLE["reset"] if color else ''
    # Form a prompt with formatting
    prompt_txt = f"{col}{prompt}{reset}"
    # Request a response
    return input(prompt_txt).strip()


# Display simple list of options
def _dumps_vopt_oneline(
        txt: str,
        vdef: Optional[Any] = None,
        vopt: Optional[list] = None,
        prompt: str = '>',
        *args) -> str:
    # Check if options give are a list
    if not isinstance(vopt, (list, tuple)):
        return ''
    # Initial portion of prompt using pre-specified prompt
    msg = f"{txt} "
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
        # Add separator
        if j > 0:
            msg += '/'
        # Format message
        if j == jdef:
            # Highlight first option as the true default
            msgj = f"[{opt}]"
        else:
            # Use option number
            msgj = opt
        # Append to overall prompt
        msg += msgj
    # Append user input prompt
    return msg + prompt + ' '


# Display list of options
def _dumps_vopt_list(
        txt: str,
        vdef: Optional[Any] = None,
        vopt: Optional[list] = None,
        prompt: str = '>',
        pcol: str = '',
        acol: str = '',
        ocol: str = '') -> str:
    # Check if options give are a list
    if not isinstance(vopt, (list, tuple)):
        return ''
    # Resets
    rp = CONSOLE["reset"] if pcol else ''
    ra = CONSOLE["reset"] if acol else ''
    ro = CONSOLE["reset"] if ocol else ''
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
            msgj = f"    {acol}@{j+1}{ra}: [{ocol}{opt}{ro}]\n"
        else:
            # Use option number
            msgj = f"    {acol}@{j+1}{ra}: {ocol}{opt}{ro}\n"
        # Append to overall prompt
        msg += msgj
    # Append user input prompt
    return msg + pcol + prompt + rp + ' '


# Display list of options
def _dumps_vdef(
        txt: str,
        vdef: Optional[Any] = None,
        vopt: Optional[list] = None,
        prompt: str = '>', *args) -> str:
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
        prompt: str = '>', *args) -> str:
    # Check for any default/option list; use prior function
    if (vdef is not None) or (vopt is not None):
        return ''
    # Form the prompt without default value
    return f"{txt}:\n{prompt} "
