#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.tnakit.textutils.wrap`: Text wrapping and indenting toolkit
=======================================================================

This module contains common functions for wrapping long strings into a
limited-length lines
"""


# Wrap text
def wrap_text(txt, cwidth=79, indent=4, cwidth1=None, indent1=None):
    r"""Convert a long string into multiple lines of text
    
    :Call:
        >>> lines = wrap_text(txt, cwidth=79, indent=4, **kw)
    :Inputs:
        *txt*: :class:`str` | :class:`unicode`
            A string, which may be very long
        *cwidth*: {``79``} | :class:`int`
            Maximum line length
        *indent*: {``4``} | :class:`int`
            Number of leading white spaces in each line
        *cwidth1*: {*cwidth*} | :class:`int`
            Maximum length for the first line
        *indent1*: {*indent*} | :class:`int`
            Number of leading white spaces in first line
    :Outputs:
        *lines*: :class:`list` (:class:`str`)
            List of lines, each less than *cwidth* chars unless there
            is a word that is longer than *cwidth* chars
    :Versions:
        * 2018-03-07 ``@ddalle``: Version 1.0
    """
    # Check max width
    if not isinstance(cwidth, int):
        raise TypeError("Max width must be int")
    elif cwidth < 1:
        raise ValueError("Max width must be positive")
    # Check indent
    if not isinstance(indent, int):
        raise TypeError("Indent must be int")
    elif indent < 0:
        raise ValueError("Indent must be nonnegative")
    # Process width for first line
    if cwidth1 is None:
        cwidth1 = cwidth
    elif not isinstance(cwidth1, int):
        raise TypeError("Max width for line 1 must be int")
    elif cwidth1 < 1:
        raise ValueError("Max width for line 1 must be positive")
    # Process indent for first line
    if indent1 is None:
        indent1 = indent
    elif not isinstance(indent1, int):
        raise TypeError("Indent for line 1 must be int")
    elif indent1 < 0:
        raise ValueError("Indent for line 1 must be nonnegative")
    # Updated widths after subtracting indent
    w0 = cwidth - indent
    w1 = cwidth1 - indent1
    # Wrap text without indent
    lines = _wrap_text(txt, w0, w1)
    # Check for null indent
    if (indent == 0) and (indent1 == 0):
        # No alteration required
        return lines
    # Turn indents into spaces
    tab1 = " " * indent1
    tab0 = " " * indent
    # Wrap other lines
    for (j, line) in enumerate(lines):
        # Add indent
        if j == 0:
            lines[j] = tab1 + line
        else:
            lines[j] = tab0 + line
    # Output
    return lines


# Wrap text
def _wrap_text(txt, cwidth=79, cwidth1=None):
    r"""Convert a long string into multiple lines of text
    
    :Call:
        >>> lines = wrap_text(txt, cwidth=79)
    :Inputs:
        *txt*: :class:`str` | :class:`unicode`
            A string, which may be very long
        *cwidth*: {``79``} | :class:`int`
            Maximum line length
        *cwidth1*: {*cwidth*} | :class:`int`
            Maximum length for the first line
    :Outputs:
        *lines*: :class:`list` (:class:`str`)
            List of lines, each less than *cwidth* chars unless there
            is a word that is longer than *cwidth* chars
    :Versions:
        * 2018-03-07 ``@ddalle``: Version 1.0
    """
    # Divide into words
    W = txt.split()
    # Initialize output
    lines = []
    # Current line length
    line = ""
    li = 0
    # Maximum length for current line
    if cwidth1 is None:
        # Use general option
        cwidthi = cwidth
    else:
        # Use specific option for first line
        cwidthi = cwidth1
    # Loop through words
    for w in W:
        # Length of this word
        lw = len(w)
        # Get target length for current line
        # Check special case
        if lw > cwidthi:
            # Terminate previous line if appropriate
            if li > 0:
                # Save the line
                lines.append(line)
                # Shift to general line width
                cwidthi = cwidth
            # Length of word divided by number of lines
            nline = lw // cwidthi
            # Initialize line cutt indices
            ia = 0
            ib = cwidthi
            # Create full lines with parts of this word
            for i in range(nline):
                # Get the next chunk of this word
                lines.append(w[ia:ib])
                # Move to general line width
                cwidthi = cwidth
                # Update indices
                ia = ib
                ib += cwidthi
            # Reset with any remaining part of this word
            line = w[ia:]
            li = len(line)
        elif li > 0 and (lw + li + 1 > cwidthi):
            # Terminate previous line
            lines.append(line)
            # Shift to general line width
            cwidthi = cwidth
            # Reset using this word
            line = w
            li = lw
        elif li == 0 and (lw + li > cwidthi):
            # Exact one-word line
            lines.append(line)
            # Shift to general line width
            cwidthi = cwidth
            # Reset using this word
            line = w
            li = lw
        elif li == 0:
            # Start new line
            line = w
            li = lw
        else:
            # Append to existing line
            line = line + " " + w
            # Update length
            li += (lw + 1)
    # Append last line if not empty.
    if li > 0:
        lines.append(line)
    # Return empty text if necessary
    if len(lines) == 0:
        lines = [""]
    # Output
    return lines

