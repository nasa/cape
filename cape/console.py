#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.console`: Console and Prompt Tools
==============================================

This module provides tools to interact with user prompts.  In contains a
function :func:`prompt_color` that handles the Python version dependence
(:func:`input` vs :func:`raw_input`) and also provides a colorful
prompt.
"""

# Standard library
import os


# Console colors and attributes
con = {
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
def prompt_color(txt, vdef=None, color="green"):
    r"""Get user input using a colorized prompt
    
    :Call:
        >>> v = prompt_color(txt, vdef=None)
    :Inputs:
        *txt*: :class:`str`
            Text of the question on the same line as prompt
        *vdef*: {``None``} | :class:`str`
            Default value (if any)
        *color*: {``"green"``} | :class:`str`
            Color name
    :Outputs:
        *v*: :class:`str`
            Raw (unevaluated) input from :func:`input` function
    :Versions:
        * 2018-08-23 ``@ddalle``: Version 1.0
    """
    # Check for default value
    if vdef:
        # Form the prompt with default value
        ptxt = "> %s [%s]: " % (txt, vdef)
    else:
        # Form the prompt without default value
        ptxt = "> %s: " % txt
    # Prepend and append escape sequences
    ptxt = con.get(color, con["black"]) + ptxt + con["reset"]
    # Check version
    if os.sys.version.startswith("3"):
        # Use the :func:`input` function
        v = input(ptxt)
    else:
        # Use the older :func:`raw_input` function
        v = raw_input(ptxt)
    # Check default value again
    if vdef and (not v):
        # Use the default value instead
        return vdef
    else:
        # Return the user's value, even if empty
        return v

