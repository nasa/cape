#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.tnakit.promptutils`: Interactive Console Prompt
==============================================================


"""

# System utilities
import os


# Console colors and attributes
CONSOLE_COLORS = {
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
    'lightgray': '\x1b[37m',
    'purple':    '\x1b[35m',
    'red':       '\x1b[31;01m',
    'reset':     '\x1b[39;49;00m',
    'standout':  '\x1b[03m',
    'teal':      '\x1b[36;01m',
    'turquoise': '\x1b[36m',
    'underline': '\x1b[04m',
    'white':     '\x1b[37;01m',
    'yellow':    '\x1b[33;01m',
}


# Function to get user input using a colored prompt
def prompt(txt, vdef=None, color="green"):
    r"""Get user input using a colorized prompt
    
    :Call:
        >>> v = prompt(txt, vdef=None)
    :Inputs:
        *txt*: :class:`str`
            Text of the question on the same line as prompt
        *vdef*: {``None``} | :class:`str`
            Default value (if any)
        *color*: {``"green"``} | :class:`str`
            Color name
    :Outputs:
        *v*: ``None`` | :class:`str`
            Raw (unevaluated) input from :func:`input` function
    :Versions:
        * 2018-08-23 ``@ddalle``: Version 1.0 (in :mod:`aerohub`)
        * 2021-08-24 ``@ddalle``: Version 1.1
            - was :func:`prompt_color`
            - returns ``None`` on empty response
    """
    # Check for default value
    if vdef:
        # Form the prompt with default value
        ptxt = "> %s [%s]: " % (txt, vdef)
    else:
        # Form the prompt without default value
        ptxt = "> %s: " % txt
    # Get code for console color
    if color not in CONSOLE_COLORS:
        raise ValueError("Unrecognized console color '%s'" % color)
    # Get console color code
    console_color = CONSOLE_COLORS[color]
    # Prepend and append escape sequences
    ptxt = console_color + ptxt + CONSOLE_COLORS["reset"]
    # Check version
    if os.sys.version_info.major > 2:
        # Use the :func:`input` function
        v = input(ptxt)
    else:
        # Use the older :func:`raw_input` function
        v = raw_input(ptxt)
    # Check default value again
    if v == "":
        # Use the default value instead
        return vdef
    else:
        # Return the user's value
        return v

