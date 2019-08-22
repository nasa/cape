#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`tnakit.text.wrap`: Text wrapping and indenting toolkit
==============================================================

This module contains common functions for wrapping long strings into a
limited-length lines
"""


# Wrap text
def wrap_text(txt, cwidth=79, cwidth1=None):
    """Convert a long string into multiple lines of text
    
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
            List of lines, each less than *cwidth* chars unless there is a word
            that is longer than *cwidth* chars
    :Versions:
        * 2018-03-07 ``@ddalle``: First version
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

