#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.testutils.fileutils`: File system utilities
=======================================================

This small module contains utilities for the CAPE testing system to
perform tasks on file trees.  This includes expanding a list of file
wildcard patterns into lists of extant files/folders.

"""

# Standard library modules
import os
import glob
import shutil
import re


# Regular expression for finding numbers
expr_float = "([+-]?[0-9]+(\.[0-9]+)?([edED][+-][0-9]+)?)"
expr_interval = "(?P<c1>[[(])(?P<v1>%s),\s*(?P<v2>%s)(?P<c2>[)\]])" % (
    expr_float, expr_float)

# Compile float recognizer
regex_float = re.compile(expr_float)
regex_interval = re.compile(expr_interval)


# Process list of files
def expand_file_list(fglobs, typ="f", error=True):
    """Expand a list of file globs
    
    :Call:
        >>> fnames = expand_file_list(fglobs, typ="f", error=True)
        >>> fnames = expand_file_list(fglob, typ="f", error=True)
    :Inputs:
        *fglob*: :class:`str` | ``None``
            Single file pattern
        *fglobs*: :class:`list`\ [:class:`str`]
            List of zero or more file patterns
        *typ*: ``"d"`` | {``"f"``} | ``"l"``
            File type; directory, file, or link
        *error*: {``True``} | ``False``
            Whether or not to raise an exception if a glob has not
            matches
    :Outputs:
        *fnames*: :class:`list`\ [:class:`str`]
            List of matches to *fglob*, without duplicates
    :Versions:
        * 2019-07-03 ``@ddalle``: First version
    """
    # Ensure lists
    if not fglobs:
        fglobs = []
    elif not isinstance(fglobs, (tuple, list)):
        fglobs = [fglobs]
    # Initialize outputs
    fnames = []
    # Loop through entries
    for pattern in fglobs:
        # Get the matches
        fglob = glob.glob(pattern)
        # Check for empty glob
        if len(fglob) == 0:
            # Check for error
            if error:
                raise ValueError("No matches for pattern '%s'" % pattern)
            # Don't process further
            continue
        # Sort these matches
        fglob.sort()
        # Loop through matches to add them to overall list
        for fname in fglob:
            # Filter by type
            if typ in ["file", "f"]:
                # File: no links and no dirs
                if os.path.islink(fname):
                    continue
                elif os.path.isdir(fname):
                    continue
            elif typ in ["dir", "d"]:
                # Directory: no links and only dirs
                if os.path.islink(fname):
                    continue
                elif not os.path.isdir(fname):
                    continue
            elif typ in ["link", "l"]:
                # Link: link only
                if not os.path.islink(fname):
                    continue
            else:
                # Unrecognized type
                raise ValueError(
                    ("File type must be 'f', 'd', or 'l'; ") +
                    ("received '%s'" % typ))
            # Check if already in list
            if fname not in fnames:
                # Append to list
                fnames.append(fname)
    # Output
    return fnames


# Compare file to target
def compare_files(fn1, fn2, **kw):
    """Compare two lines of text using various rules
    
    :Call:
        >>> q, i = compare_files(fn1, fn2)
    :Inputs:
        *fn1*: :class:`str`
            Nominal output for comparison
        *fn2*: :class:`str`
            Target output or pattern
        *MAX_LINES*: {``100000``} | :class:`int` > 0
            Maximum number of lines to check in a file
        *REGULAR_EXPRESSION*: ``True`` | {``False``}
            Treat *line2* as a regular expression (do not use with
            other options)
        *NORMALIZE_WHITESPACE*: ``True`` | {``False``}
            Option to ignore whitespace during comparisons
        *ELLIPSIS*: {``True``} | ``False``
            Allow string ``"..."`` to represent any string at most
            once in *line2*
        *VALUE_INTERVAL*: {``True``} | ``False``
            Allow syntax ``"<valint>[$VMIN,$VMAX]"`` to represent any
            value between *VMIN* and *VMAX* when used in *line2*
    :Outputs:
        *q*: ``True`` | ``False``
            Whether or not two files match
        *i*: ``None`` | :class:`int` > 0
            Line at which first difference occurs, if any
    :Versions:
        * 2019-07-05 ``@ddalle``: First version
    """
    # Maximum line count
    imax = kw.pop("MAX_LINES", 100000)
    # Check for duplicate file
    if fn1 == fn2:
        raise ValueError("Output file and target are the same file")
    # Check if first file (nominal output) exists
    if not os.path.isfile(fn1):
        # Missing required file
        return False
    # Check if second file (target output) exists
    if not os.path.isfile(fn2):
        # Test successful if no target
        return True
    # Open both files
    f1 = open(fn1, 'r')
    f2 = open(fn2, 'r')
    # Initialize status as PASS until contrary evidence
    q = True
    # Loop through lines
    for i in range(imax):
        # Read next line
        line1 = f1.readline()
        line2 = f2.readline()
        # Exit if EOF
        if line1 == "":
            # Make sure *line2* is also empty
            if line2 != "":
                # Failure status
                q = False
            # Exit loop
            break
        # Compare lines
        try:
            # Attempt comparison
            qi = compare_lines(line1, line2, **kw)
        except ValueError as e:
            # Report line number
            f1.close()
            f2.close()
            raise ValueError(
                ("In line %i of file '%s':\n" % (i + 1, fn2)) + e.message)
        if not qi:
            # Failure
            q = False
            break
    # Close files
    f1.close()
    f2.close()
    # No line number if successful comparison
    if q is True:
        # No failure line
        i = None
    else:
        # Switch to 1-based indexing
        i += 1
    # Output
    return q, i


# Compare two lines
def compare_lines(line1, line2, **kw):
    """Compare two lines of text using various rules
    
    :Call:
        >>> q = compare_lines(line1, line2)
    :Inputs:
        *line1*: :class:`str`
            Nominal output for comparison
        *line2*: :class:`str`
            Target output or pattern
        *REGULAR_EXPRESSION*: ``True`` | {``False``}
            Treat *line2* as a regular expression (do not use with
            other options)
        *NORMALIZE_WHITESPACE*: ``True`` | {``False``}
            Option to ignore whitespace during comparisons
        *ELLIPSIS*: {``True``} | ``False``
            Allow string ``"..."`` to represent any string at most
            once in *line2*
        *VALUE_INTERVAL*: {``True``} | ``False``
            Allow syntax ``"<valint>[$VMIN,$VMAX]"`` to represent any
            value between *VMIN* and *VMAX* when used in *line2*
    :Outputs:
        *q*: ``True`` | ``False``
            Whether or not two lines match according to options
    :Versions:
        * 2019-07-05 ``@ddalle``: First version
    """
    # Get options
    normws = kw.get("NORMALIZE_WHITESPACE", False)
    qregex = kw.get("REGULAR_EXPRESSION", False)
    ellips = kw.get("ELLIPSIS", True)
    valint = kw.get("VALUE_INTERVAL", True)
    # Check for disallowed combinations
    if qregex and normws:
        raise ValueError(
            "Can't combine REGULAR_EXPRESSION and NORMALIZE_WHITESPACE")
    elif qregex and ellips:
        raise ValueError(
            "Can't combine REGULAR_EXPRESSION and ELLIPSIS")
    elif qregex and valint:
        raise ValueError(
            "Can't combine REGULAR_EXPRESSION and VALUE_INTERVAL")
    # Check for regular expression before anything else
    if qregex:
        # Line 2 is the regular expression
        match = re.match(line2, line1)
        # Check for the match (convert to boolean)
        if match:
            return True
        else:
            return False
    # Check for whitespace option
    if normws:
        # Remove whitespace in both
        line1 = line1.replace(" ", "").replace("\t", "").replace("\n", "")
        line2 = line2.replace(" ", "").replace("\t", "").replace("\n", "")
    # Check for ellipsis
    if ellips and ("..." in line2):
        # Find location of ellipsis
        ia = line2.index("...")
        ib = ia + 3
        ic = len(line2)
        # Check length of line 1
        if len(line1) < ic - 3:
            # Not enough characters to match
            return False
        # Just copy over whatever's in those positions from *line1*
        line2 = line2[:ia] + line1[ia:ib-ic] + line2[ib:]
        # Check for multiple ellipses
        if "..." in line2:
            raise ValueError("Can't use '...' twice with ELLIPSIS")
    # Check for value intervals
    if valint:
        # Loop until intervals removed
        while "<valint>" in line2:
            # Find the <valint> flag
            ia = line2.index("<valint>")
            # Get allowed interval from remaining characters
            match = regex_interval.match(line2[ia+8:])
            # Check for invalid expression
            if not match:
                raise ValueError("Invalid interval: %s" % line2[ia:])
            # Get value from *line1*
            m1 = regex_float.match(line1[ia:])
            # Check for valid float
            if not m1:
                return False
            # Get values
            vmin = float(re.sub("[Dd]", "e", match.group("v1")))
            vmax = float(re.sub("[Dd]", "e", match.group("v2")))
            # Get value from *line1*
            v = float(re.sub("[Dd]", "e", m1.group()))
            # Check test
            if (v < vmin) or (v > vmax):
                # Outside interval (don't need to check open/closed
                return False
            elif (match.group("c1") == "(") and (v == vmin):
                # Edge of open interval
                return False
            elif (match.group("c2") == ")") and (v == vmax):
                # Edge of open interval
                return False
            # Lengths of regular expressions
            l1 = m1.end()
            l2 = match.end()
            # Make replacement
            line2 = line2[:ia] + line1[ia:ia+l1] + line2[ia+l2+8:]
    # At this point, just compare the lines
    return line1 == line2
            
        
    
