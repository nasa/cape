#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.console`: Console and Prompt Tools
==============================================

This module provides tools to interact with user prompts.  In contains a
function :func:`prompt_color` that handles the Python version dependence
(:func:`input` vs :func:`raw_input`) and also provides a colorful
prompt.

It also contains :func:`prompt_menu`, which displays one option per line
in the style of the ``promptutils`` module from ``aerohub``.
"""

# Standard library
import re


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

# Regular expression to recognize "@{n}" entries
REGEX_AT = re.compile(r"@([0-9]+)")


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
    # Use the :func:`input` function
    v = input(ptxt)
    # Check default value again
    if vdef and (not v):
        # Use the default value instead
        return vdef
    else:
        # Return the user's value, even if empty
        return v


# Function to get user input from a menu of options, one per line
def prompt_menu(txt, vopt, vdef=None, color="green", prompt=">"):
    r"""Get user input from a list of options, one per line

    This prompt is modeled after ``aerohub.promptutils.prompt_color``:
    each option is displayed on its own line with a ``@j`` index, and
    the default value (if any) is wrapped in square brackets. The user
    may press Enter to accept the default, type ``@j`` to select option
    number *j*, or type a value directly.

    For example::

        Action:
            @1: [approve]
            @2: extend
            @3: skip
        >

    :Call:
        >>> v = prompt_menu(txt, vopt, vdef=None, color="green")
        >>> v = prompt_menu(txt, vopt, vdef=None, color="green", prompt=">")
    :Inputs:
        *txt*: :class:`str`
            Title of the question, shown above the option list
        *vopt*: :class:`list`\ [:class:`str` | :class:`tuple`]
            List of options to show, one per line; entries may also be
            ``(display, value)`` tuples, where *display* (which may
            contain newlines for sub-items) is shown while *value* is
            used for defaults and ``@j`` selections
        *vdef*: {``None``} | :class:`str`
            Default value (if any); an option whose value is equal to
            *vdef* is highlighted with brackets
        *color*: {``"green"``} | :class:`str`
            Color name
        *prompt*: {``">"``} | :class:`str`
            Character(s) to use as the final in-line prompt
    :Outputs:
        *v*: :class:`str`
            User input, *vdef* on empty input, or ``vopt[j-1]`` if user
            entered ``@j`` for valid *j*
    :Versions:
        * 2026-09-06 ``@ddalle``: v1.0
    """
    # Start menu with title line
    lines = [f"{txt}:"]
    # Loop through options
    for j, opt in enumerate(vopt):
        # Check for (display, value) format
        if isinstance(opt, tuple):
            disp, val = opt
        else:
            disp, val = opt, opt
        # Highlight the default value with square brackets
        if val == vdef:
            disp = f"[{disp}]"
        # Add one line per option
        lines.append(f"    @{j+1}: {disp}")
    # Combine title and options and append in-line prompt
    ptxt = "\n".join(lines) + f"\n{prompt} "
    # Prepend and append escape sequences
    ptxt = con.get(color, con["black"]) + ptxt + con["reset"]
    # Use the :func:`input` function
    v = input(ptxt).strip()
    # Check for "@j" selection
    match = REGEX_AT.fullmatch(v)
    if match:
        # Convert to zero-based index
        j = int(match.group(1)) - 1
        # Check bounds
        if 0 <= j < len(vopt):
            # Return the value of that option
            opt = vopt[j]
            return opt[1] if isinstance(opt, tuple) else opt
        # Otherwise return raw invalid entry
        return v
    # Check default value
    if vdef and (not v):
        # Use the default value instead
        return vdef
    # Return the user's value, even if empty
    return v
