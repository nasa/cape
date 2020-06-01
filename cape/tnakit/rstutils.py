#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.tnakit.rstutils`: Tools for writing ReST files
=========================================================

This module contains tools to write text formatted for reStructuredText
(reST). This includes various markup options and tools for creating the
directives to include images.  Of particular interest are the functions
:func:`rst_image_table_lines` and :func:`rst_image_table`, which
simplifies the important task of creating a table that includes several
images.
"""

# Regular expressions
import re

# Local modules
from . import typeutils
from .textutils import wrap


# Function to create input or output reST dict for docstring
def rst_param_list(keys, types, descrs, alts={}, **kw):
    r"""Write a docstring for a list of parameters

    :Call:
        >>> txt = rst_param_list(keys, types, descrs, alts={}, **kw)
    :Inputs:
        *keys*: :class:`list`\ [:class:`str`]
            List of parameters to describe
        *types*: :class:`dict`\ [:class:`str`]
            Docstring of types for each *key* in *keys*
        *descrs*: :class:`dict`\ [:class:`str`]
            Docstring of descriptions for each *key* in *keys*
        *alts*: {``{}``} | :class:`dict`\ [:class:`str`]
            Dictionary of optional names for certain keys
        *indent*: {``4``} | :class:`int` > 0
            Indent spaces for each line
        *tab*: {``4``} | :class:`int` > 0
            Additional spaces for further indentation
        *width*: {``72``} | :class:`int`
            Maximum width for line of text (including spaces)
        *strict*: {``True``} | ``False``
            If ``False``, skip keys with missing type or description
    :Outputs:
        *txt*: :class:`str`
            Text in CAPE docstring format describing *keys*
    :Versions:
        * 2019-12-19 ``@ddalle``: First version
    """
    # Get other options
    indent = kw.get("indent", 4)
    tab = kw.get("tab", 4)
    width = kw.get("width", 72)
    strict = kw.get("strict", True)
    # Check types
    if not isinstance(keys, list):
        raise TypeError("Parameter list must be a list")
    elif not isinstance(types, dict):
        raise TypeError("Parameter types must be a dict")
    elif not isinstance(descrs, dict):
        raise TypeError("Parameter descriptions must be a dict")
    elif not isinstance(alts, dict):
        raise TypeError("Alternate parameter names must be a dict")
    elif not isinstance(indent, int):
        raise TypeError("Indent must be an int")
    elif not isinstance(tab, int):
        raise TypeError("Additional tab width must be an int")
    # Convert indent and tab to strs
    tab1 = " " * indent
    # Number of keys
    nkey = len(keys)
    # Initialize text; ensuring unicode in Python 2 and 3
    txt = u""
    # Loop through keys
    for (i, k) in enumerate(keys):
        # Check type
        if not typeutils.isstr(k):
            raise TypeError("Parameter %i is not a string" % i)
        # Get type and text
        rst_type = types.get(k)
        rst_desc = descrs.get(k)
        # Check types
        if rst_type is None:
            # Missing type string
            if strict:
                raise KeyError("No type for parameter '%s'" % k)
            else:
                continue
        elif rst_desc is None:
            # Missing description
            if strict:
                raise KeyError("No description for parameter '%s'" % k)
            else:
                continue
        elif not typeutils.isstr(rst_type):
            # Type spec is a string, *strict* doesn't apply here
            raise TypeError(
                "Type description for parameter '%s' is not a string" % k)
        elif not typeutils.isstr(rst_desc):
            # Description must be string, *strict* enforcement
            raise TypeError(
                "Description for parameter '%s' is not a string" % k)
        # Find alternate names
        key_alt = [k1 for (k1, k2) in alts.items() if (k2 == k)]
        # Wrap the text
        lines_desc = wrap.wrap_text(rst_desc, width, indent+tab)
        # Form into single string
        txt_desc = "\n".join(lines_desc) + "\n"
        # First indent: skip if first line
        if i > 0:
            txt += tab1
        # Append text for primary variable name
        txt += "*%s*" % k
        # Add alternate names
        for ka in key_alt:
            # Delimiter and alternate name
            txt += ", *%s*" % ka
        # Delimiter and type description
        txt += ": "
        txt += rst_type
        txt += "\n"
        # Description (strip final \n)
        if i + 1 == nkey:
            txt += txt_desc[:-1]
        else:
            txt += txt_desc
    # Output
    return txt
    


# Write a title
def rst_title(title, char="-", margin=2, before=False):
    """Create an ReST title from string
    
    This adds characters before (optionally) and after the title, creating
    valid syntax for ReStructuredText titles.  Examples of valid titles with
    the input string ``"Sample Title"`` are shown below:
    
    .. code-block:: rst
    
        Sample Title
        --------------
        
        ==============
        Sample Title
        ==============
        
        ************
        Sample Title
        ************
    
    :Call:
        >>> txt = rst_title(title, char="-", margin=2, before=False)
    :Inputs:
        *title*: :class:`str` | :class:`unicode`
            Input string, must be one line
        *char*: {``"-"``} | ``"="`` | ``"*"`` | ``"^"``
            Valid reST title adornment character (there are more than 4)
        *margin*: {``2``} | :class:`int` >= 0
            Extra characters to place after end of character
        *before*: ``True`` | {``False``}
            Whether or not to use and adornment line before title
    :Outputs:
        *txt*: :class:`str` | :class:`unicode`
            String with title, adornment line(s), and 2 or 3 newline chars
    :Versions:
        * 2019-02-22 ``@ddalle``: First version
    """
    # Check input type
    if not typeutils.isstr(title):
        t = title.__class__.__name__
        raise TypeError("Title must be a string (%s)" % t)
    # Check margin
    if margin < 0:
        raise ValueError("Cannot use negative margin")
    # Check input character
    if len(char) > 1:
        raise ValueError(
            "Adornment char '%s' is more than one character" % char)
    elif re.match("\w", char):
        raise ValueError(
            "Adornment char '%s' is invalid" % char)
    # Title length
    L = len(title)
    # Create adornment line
    line = char*(L+margin) + "\n"
    # Required part
    txt = title + "\n" + line
    # Check for prefix adornment line
    if before:
        txt = line + txt
    # Output
    return txt


# Write an rst option
def rst_directive_option(k, v):
    """Convert an option name and value to an reST directive option
    
    :Call:
        >>> txt = rst_directive_option(k, v)
    :Inputs:
        *k*: :class:`str`
            Name of the option
        *v*: :class:`any`
            Value for the option
    :Outputs:
        *txt*: :class:`str` | :class:`unicode`
            reST version of the option declaration
    :Cases:
        =============   ==========================
        *v*             *txt*
        =============   ==========================
        ``None``        ``""``
        ``True``        ``":%s:" % k``
        *Else*          ``":%s: %s" % (k, v)``
        =============   ==========================
    :Versions:
        * 2019-02-24 ``@ddalle``: First version
    """
    # Check value
    if v is None:
        # No option
        return ""
    elif v is True:
        # Turn option one with no arguments
        return ":%s:" % k
    else:
        # Use name and value
        return ":%s: %s" % (k, v)


# Generic conversion
def py2rst(v, **kw):
    r"""Convert a generic Python object to reST text

    :Call:
        >>> txt = py2rst(v, **kw)
    :Inputs:
        *v*: :class:`object`
            Generic Python value
        *repr_types*: {``None``} | :class:`type` | :class:`tuple`
            Type or tuple of types for which to use :func:`repr`
        *str_types*: {``None``} | :class:`type` | :class:`tuple`
            Type or tuple of types for which to use :func:`str`
        *strict*: {``True``} | ``False``
            Whether or not to raise exception for other types
    :Outputs:
        *txt*: :class:`str`
            Marked up reST text
    :Versions:
        * 2019-04-25 ``@ddalle``: First version
    """
    # Check for special types to use :func:`repr` for
    repr_types = kw.get("repr_types")
    str_types  = kw.get("str_types")
    # Get universal options dictionaries
    formats = kw.pop("fmts", kw.pop("formats", {}))
    markups = kw.pop("markups", {})
    # Filter input type
    if (repr_types is not None) and isinstance(v, repr_types):
        # Use class's __repr__ method
        # Get options
        markup = markups.get("__repr__", kw.get("markup", False))
        # Conversion
        return py2rst_any_repr(v, markup=markup)
    elif (str_types is not None) and isinstance(v, str_types):
        # Use class's __str__ method
        # Get options
        markup = markups.get("__str__", kw.get("markup", False))
        # Conversion
        return py2rst_any_str(v, markup=markup)
    elif v is None:
        # Convert ``None``
        # Get options
        markup = markups.get("None", kw.get("markup", True))
        # Conversion
        return py2rst_none(markup=markup)
    elif isinstance(v, bool):
        # Convert boolean
        markup = markups.get("bool", kw.get("markup", True))
        # Conversion
        return py2rst_bool(v, markup=markup)
    elif isinstance(v, int):
        # Convert integer
        # Get options
        markup = markups.get("int", kw.get("markup", False))
        fmt    = formats.get("int", kw.get("fmt", "%s"))
        # Conversion
        return py2rst_int(v, markup=markup, fmt=fmt)
    elif isinstance(v, float):
        # Convert float
        # Get options
        markup = markups.get("float", kw.get("markup", False))
        fmt    = formats.get("float", kw.get("fmt", "%s"))
        # Conversion
        return py2rst_float(v, markup=markup, fmt=fmt)
    elif typeutils.isstr(v):
        # Convert string
        # Copy options
        kw_str = dict(kw)
        # Transfer any class-specific options
        if "str" in markups:
            kw_str["markup"] = markups["str"]
        # Convert string
        return py2rst_str(v, **kw_str)
    elif isinstance(v, (list, tuple, set)):
        # Convert list
        return py2rst_list(v, **kw)
    elif isinstance(v, dict):
        # Dictionary
        return py2rst_dict(v, **kw)
    elif isinstance(v, typeutils.moduletype):
        # Module
        return py2rst_mod(v, **kw)
    elif kw.get("strict", True):
        # Unrecognized type
        raise TypeError("Unrecognized type '%s'" % type(v))
    else:
        # Use class's __repr__ method
        # Get options
        markup = markups.get("__repr__", kw.get("markup", False))
        # Conversion
        return py2rst_any_repr(v, markup=markup)


# Convert Python `mod` to reST
def py2rst_mod(mod, markup=True):
    r"""Convert a Python module to reST text

    :Call:
        >>> txt = py2rst_mod(mod, markup=True)
    :Inputs:
        *mod*: :class:`module`
            A Python module
        *markup*: {``True``} | ``False``
            Option to use reST ``:mod:`` role
    :Outputs:
        *txt*: :class:`str`
            Text representation for reST
    :Versions:
        * 2019-12-27 ``@ddalle``: First version
    """
    # Check input types
    if not isinstance(mod, typeutils.moduletype):
        raise TypeError(
            "Cannot convert type '%s' to module" % mod.__class__.__name__)
    # Convert text
    if markup:
        # Use role
        return ":mod:`%s`" % mod.__name__
    else:
        # No decoration
        return mod.__name__


# Convert Python `int` to reST
def py2rst_int(i, markup=False, fmt="%s"):
    r"""Convert a Python :class:`int` to reST text

    :Call:
        >>> txt = py2rst_int(i, markup=False, fmt="%s")
    :Inputs:
        *i*: :class:`int`
            Integer to represent
        *markup*: ``True`` | {``False``}
            Whether or not to add backticks around converted text
        *fmt*: {``"%s"``} | :class:`str`
            Format string to convert *i* (printf style)
    :Outputs:
        *txt*: :class:`str`
            Text representation for reST
    :Versions:
        * 2019-04-24 ``@ddalle``: First version
    """
    # Check input types
    if i is True:
        raise TypeError("Disallowed conversion of 'True' to int")
    elif i is False:
        raise TypeError("Disallowed conversion of 'False' to int")
    if not isinstance(i, int):
        raise TypeError(
            "Cannot convert type '%s' to int" % i.__class__.__name__)
    elif not typeutils.isstr(fmt):
        raise TypeError("Format flag must be string")
    # Convert text
    try:
        txt = fmt % i
    except TypeError:
        # Conversion string not allowed
        raise TypeError(
            "Could not convert float using format flag '%s'" % fmt)
    # Check for markup
    if markup:
        # Use double-backticks for markup
        txt = "``%s``" % txt
    # Output
    return txt


# Convert Python `float` to reST
def py2rst_float(x, markup=False, fmt="%s"):
    r"""Convert a Python :class:`float` to reST text

    :Call:
        >>> txt = py2rst_float(x, markup=False, fmt="%s")
    :Inputs:
        *x*: :class:`float`
            Floating-point number to represent
        *markup*: ``True`` | {``False``}
            Whether or not to add backticks around converted text
        *fmt*: {``"%s"``} | :class:`str`
            Format string to convert *i* (printf style)
    :Outputs:
        *txt*: :class:`str`
            Text representation for reST
    :Versions:
        * 2019-04-24 ``@ddalle``: First version
    """
    # Check input types
    if not isinstance(x, float):
        raise TypeError(
            "Cannot convert type '%s' to float" % x.__class__.__name__)
    elif not typeutils.isstr(fmt):
        raise TypeError("Format flag must be string")
    # Convert text
    try:
        txt = fmt % x
    except TypeError:
        # Conversion string not allowed
        raise TypeError(
            "Could not convert float using format flag '%s'" % fmt)
    # Check for markup
    if markup:
        # Use double-backticks for markup
        txt = "``%s``" % txt
    # Output
    return txt


# Convert boolean
def py2rst_bool(q, markup=True):
    r"""Convert a Python :class:`bool` to reST text

    :Call:
        >>> txt = py2rst_bool(q, markup=True)
    :Inputs:
        *q*: ``True`` | ``False`` | :class:`bool`
            Boolean or subclass quantity to represent
        *markup*: {``True``} | ``False``
            Whether or not to add backticks around converted text
    :Outputs:
        *txt*: :class:`str`
            Text representation for reST
    :Versions:
        * 2019-04-24 ``@ddalle``: First version
    """
    # Check input types
    if not isinstance(q, bool):
        raise TypeError(
            "Cannot convert type '%s' to bool" % q.__class__.__name__)
    # Convert text
    txt = repr(q)
    # Check for markup
    if markup:
        # Use double-backticks for markup
        txt = "``%s``" % txt
    # Output
    return txt


# Convert null/None
def py2rst_none(markup=True):
    r"""Convert Python ``None`` to reST text

    :Call:
        >>> txt = py2rst_none(markup=True)
    :Inputs:
        *markup*: {``True``} | ``False``
            Whether or not to add backticks around converted text
    :Outputs:
        *txt*: :class:`str`
            Text representation for reST
    :Versions:
        * 2019-04-24 ``@ddalle``: First version
    """
    # Check for markup
    if markup:
        # Use double-backticks for markup
        return "``None``"
    else:
        # No double-backticks
        return "None"


# Convert a string
def py2rst_str(s, **kw):
    r"""Convert a Python :class:`str` to reST text

    :Call:
        >>> txt = py2rst_str(s, **kw)
    :Inputs:
        *s*: :class:`str`
            String to convert
        *indent*: {``0``} | :class:`int` ≥ 0
            Number of spaces of indentation
        *hang*: {*indent*} | :class:`int` ≥ 0
            Number of spaces of indent for continuation lines
        *maxcols*: {``79``} | :class:`int` > 0
            Maximum number of columns in a line
        *markuplength*: {``50``} | :class:`int` ≥ 0
            Cutoff length for default marked strings
    :Outputs:
        *txt*: :class:`str`
            Marked up reST text
    :Versions:
        * 2019-04-25 ``@ddalle``: First version
    """
    # Check input types
    if not typeutils.isstr(s):
        raise TypeError(
            "Cannot convert type '%s' to str" % s.__class__.__name__)
    # Length of string
    L = len(s)
    # Maximum number of marked-up columns
    mlength = kw.pop("markuplength", 50)
    # Get default option for markup
    if L > mlength:
        markup = False
    else:
        markup = True
    # Get remaining options
    maxcols = kw.get("maxcols", 79)
    markup = kw.get("markup", markup)
    indent = kw.get("indent", 0)
    hang = kw.get("hang", indent)
    # Turn of indentation while marked up
    if markup:
        indent = 0
        hang = 0
    # Reduced line widths
    cwidth1 = maxcols - indent
    cwidth  = maxcols - hang
    # Check for backticks
    if markup:
        cwidth1 -= 4
        cwidth  -= 4
    # Wrap the string
    lines = wrap.wrap_text(s, cwidth1=cwidth1, cwidth=cwidth, indent=0)
    # Check for markup
    if markup:
        # Initialize lines
        lines_fmt = []
        # Loop through lines
        for line in lines:
            # Check for double quotes
            if '"' in line:
                # Use single quotes
                delim = "'"
            else:
                # Use double quotes
                delim = '"'
            # Add double-backticks to each line
            lines_fmt.append("``%s%s%s``" % (delim, line, delim))
        # Handoff lines reference
        lines = lines_fmt
    # Initialize string
    txt = (" "*indent) + lines[0]
    # Add remaining lines
    for line in lines[1:]:
        txt += "\n" + (" "*hang) + line
    # Output
    return txt


# Convert a list
def py2rst_list(V, **kw):
    r"""Convert a list to reST text

    :Call:
        >>> txt = py2rst_list(V, **kw)
    :Inputs:
        *V*: :class:`list` | :class:`tuple`
            List of Python quantitites to convert
        *markup*: ``True`` | {``False``}
            Default value for option to add backticks
        *indent*: {``0``} | :class:`int` ≥ 0
            Number of spaces of indentation
        *hang*: {*indent*} | :class:`int` ≥ 0
            Number of spaces of indent for continuation lines
        *maxcols*: {``79``} | :class:`int` > 0
            Maximum number of columns in a line
        *markuplength*: {``50``} | :class:`int` ≥ 0
            Cutoff length for default marked strings
        *fmt*: ``True`` | ``False``
            Format string for :class:`int` and :class:`float` entries
        *markups*: :class:`dict`\ [:class:`bool`]
            Specific *markup* for types ``"int"``, ``"float"``,
            ``"None"``, ``"str"``, ``"__repr__"``, and ``"__str__"``
        *fmts*: :class:`dict`\ [:class:`str`]
            Specific *fmt* for types ``"int"`` and/or ``"float"``
    :Versions:
        * 2019-04-25 ``@ddalle``: First version
    """
    # Check input types
    if not isinstance(V, (list, tuple, set)):
        raise TypeError(
            "Cannot convert type '%s' to list" % V.__class__.__name__)
    # Length to consider a single string
    minstrlen = kw.get("minstrlen", 32)
    # Check for all strings
    if minstrlen:
        # Check type of each element
        qstr = all([typeutils.isstr(v) for v in V])
    else:
        # Don't check if minstrlen is turned off
        qstr = False
    # Check string length
    if qstr:
        # Get total length
        L = sum([len(v) for v in V])
        # If insufficiently long, don't trigger line continuation
        if L < minstrlen:
            qstr = False
    # Check for string conversion
    if qstr:
        return py2rst_str("\n".join(V), **kw)
    # Otherwise begin actual list
    # Get remaining options
    maxcols = kw.get("maxcols", 79)
    indent = kw.pop("indent", 0)
    hang = kw.pop("hang", indent)
    # Refine maximum columns
    kw["maxcols"] = maxcols - 4 - max(indent, hang)
    # Create list of strings (indentation removed)
    txts = [py2rst(v, **kw) for v in V]
    # Get line sets
    lines_table = [txt.split("\n") for txt in txts]
    # Find maximum length and number of rows for each entry of table
    ncol_table = [max(map(len, lines)) for lines in lines_table]
    # Overall maximum length of a row
    ncol = max(ncol_table)
    # Convert indent counts to string
    indenttab = " " * indent
    hangingtab = " " * hang
    # Create row separator, Format for data row
    rowsep = "+" + ("-" * (ncol + 2)) + "+\n"
    rowfmt = hangingtab + ("| %%-%is |\n" % ncol)
    rowstr = hangingtab + rowsep
    # Joined text for table
    txt = indenttab + rowsep
    # Loop through entries of the table
    for lines in lines_table:
        # Loop through rows
        for line in lines:
            # Add this line
            txt += (rowfmt % line)
        # Add the row separator
        txt += rowstr
    # Output (strip last \n)
    return txt[:-1]


# Convert dictionary to reST
def py2rst_dict(v, **kw):
    """Convert a :class:`dict` to reST text

    :Call:
        >>> txt = py2rst_dict(v, **kw)
    :Inputs:
        *v*: :class:`dict`
            Python dictionary to convert to reST text
        *alignfields*: ``True`` | {``False``}
            Option to align right-hand side if text fits on one row
        *sort*: ``True`` | {``False``}
            Whether or not to sort dictionary keys
        *sortkey*: {``None``} | :class:`function`
            Function to use for sorting if *sort* is ``True``
        *sortreverse*: ``True`` | {``False``}
            Option to reverse the sort if *sort* is ``True``
        *markup*: ``True`` | {``False``}
            Default value for option to add backticks
        *indent*: {``0``} | :class:`int` ≥ 0
            Number of spaces of indentation
        *hang*: {*indent*} | :class:`int` ≥ 0
            Number of spaces of indent for continuation lines
        *tab*: {``4``} | :class:`int` ≥ 0
            Number of spaces to add to deeper indentation levels
        *maxcols*: {``79``} | :class:`int` > 0
            Maximum number of columns in a line
        *markuplength*: {``50``} | :class:`int` ≥ 0
            Cutoff length for default marked strings
        *fmt*: ``True`` | ``False``
            Format string for :class:`int` and :class:`float` entries
        *markups*: :class:`dict`\ [:class:`bool`]
            Specific *markup* for types ``"int"``, ``"float"``,
            ``"None"``, ``"str"``, ``"__repr__"``, and ``"__str__"``
        *fmts*: :class:`dict`\ [:class:`str`]
            Specific *fmt* for types ``"int"`` and/or ``"float"``
    :Versions:
        * 2019-04-25 ``@ddalle``: First version
    """
    # Check input types
    if not isinstance(v, dict):
        raise TypeError(
            "Cannot convert type '%s' to list" % v.__class__.__name__)
    # Check for sorting
    if kw.get("sort"):
        # Sort the keys
        keys = sorted(
            v.keys(),
            key=kw.get("sortkey"),
            reverse=kw.get("sortreverse", False))
    else:
        # Use the keys in the order that comes out
        keys = v.keys()
    # Get alignmnment option
    align = kw.pop("alignfields", kw.pop("align", False))
    # Get remaining options
    maxcols = kw.pop("maxcols", 79)
    indent = kw.pop("indent", 0)
    tab = kw.pop("tab", 4)
    kw.pop("hang", None)
    # Initialize string
    txt = ""
    # Get length of longest key if appropriate
    if align:
        # Maximum key name length plus 3 for ":: "
        nlhs = max(map(len, keys)) + 3
    # Loop through keys
    for key in keys:
        # Get the value
        val = v[key]
        # Start of output: label for the key in reST
        keytxt = (" " * indent) + (":%s:" % key)
        # Length of this option (plus a space)
        nk = len(keytxt)
        # Check for alignment option
        if not align:
            # Unaligned: indent and key name and one space
            nkey = nk + 1
            nsep = 1
        else:
            # Aligned: indent and key name plus sufficient spaces
            nkey = indent + nlhs
            nsep = nkey - nk
        # Create options
        kw_key = dict(indent=0, maxcols=maxcols-nkey, **kw)
        # First attempt to see if it fits on one line
        if isinstance(val, (dict, list)):
            # This cannot fit on one line
            valtxt = "\n"
        else:
            # Convert the value to reST on one line
            valtxt = py2rst(val, **kw_key)
        # Check if it fit on one line
        if "\n" not in valtxt:
            # Complete the line
            txt += keytxt + (" "*nsep) + valtxt + "\n"
            continue

        # Redo conversion with an indent
        kw_key = dict(indent=indent + tab, maxcols=maxcols, **kw)
        # Second conversion
        valtxt = py2rst(val, **kw_key)
        # Add newline character
        valtxt = valtxt.rstrip("\n") + "\n"
        # Add this entry to the total
        txt += (keytxt + "\n" + valtxt)
    # Output (w/o final \n)
    return txt[:-1]

    
# Convert any quantity using :func:`str`
def py2rst_any_str(v, markup=True):
    """Convert a Python to reST text using :func:`str`

    :Call:
        >>> txt = py2rst_any_str(v, markup=True)
    :Inputs:
        *v*: any
            Quantity to convert using :func:`repr`
        *markup*: {``True``} | ``False``
            Whether or not to add backticks around converted text
    :Outputs:
        *txt*: :class:`str`
            Text representation for reST
    :Versions:
        * 2019-04-25 ``@ddalle``: First version
    """
    # Convert text
    txt = str(v)
    # Check for markup
    if markup:
        # Use double-backticks for markup
        txt = "``%s``" % txt
    # Output
    return txt


# Convert any quantity using :func:`repr`
def py2rst_any_repr(v, markup=True):
    """Convert a Python to reST text using :func:`repr`

    :Call:
        >>> txt = py2rst_any_repr(v, markup=True)
    :Inputs:
        *v*: any
            Quantity to convert using :func:`repr`
        *markup*: {``True``} | ``False``
            Whether or not to add backticks around converted text
    :Outputs:
        *txt*: :class:`str`
            Text representation for reST
    :Versions:
        * 2019-04-25 ``@ddalle``: First version
    """
    # Convert text
    txt = repr(v)
    # Check for markup
    if markup:
        # Use double-backticks for markup
        txt = "``%s``" % txt
    # Output
    return txt


# List of common options for all directives
directive_common_options = [
    "class",
    "name"
]
# List of options for "image" directive
directive_img_options = directive_common_options + [
    "alt",
    "height",
    "width",
    "scale",
    "align",
    "target"
]
# List of options for "figure" directive
directive_fig_options = directive_img_options + [
    "figwidth"
]
# List of options for "table" directive
directive_table_options = directive_common_options + [
    "align",
    "widths",
    "width"
]


# Write a directive for an image
def rst_image_lines(ffig, directive="image", **kw):
    """Write the reST command to include a figure
    
    :Call:
        >>> lines = rst_image_lines(ffig, directive="image", **kw)
    :Inputs:
        *ffig*: :class:`str`
            Name of image to include (often like ``"myimage.*"``)
        *directive*: {``"image"``} | ``"figure"``
            Which image-inclusion command to use
        *tab*: {``"    "``} | :class:`int`
            Indent character for a tab (default is 4 spaces)
        *sub*, *tag*: {``None``} | :class:`str`
            Abbreviated name  to use for a substitution instruction
        *width*: {``None``} | :class:`str`
            Image width option value
        *name*: {``None``} | :class:`str`
            Reference "name" to be used by later ``:ref:`` calls
        *target*: {``None``} | :class:`str`
            URI for clickable image
        *alt*: {``None``} | :class:`str`
            Alternate text if image cannot be displayed
        *figwidth*: {``None``} | :class:`str`
            Width for figure, which contains an image (``"figure"`` only)
        *maxcol*, *maxcols*, *cwidth*: {``79``} | :class:`int` > 0
            Column number at which to wrap
    :Outputs:
        *lines*: :class:`list` (:class:`str`)
            List of lines of text
    :Versions:
        * 2019-02-22 ``@ddalle``: First version
    """
    # Check value
    if directive not in ["image", "figure"]:
        raise ValueError(
            "Directive '%s' is not an image directive"  % directive)
    # Tab
    tab = kw.pop("tab", "    ")
    # Check for substitution
    tag = kw.pop("tag", kw.pop("sub", None))
    # Initialize text
    if tag:
        # Does not work with "figure"
        if directive == "figure":
            raise ValueError(
                "Substitutions are not allowed for 'figure' directives")
        # Initialize a substitution directive
        lines = [".. |%s| %s:: %s" % (tag, directive, ffig)]
    else:
        # Direct inclusion
        lines = [".. %s:: %s" % (directive, ffig)]
    # Process appropriate option list
    if directive == "image":
        # "image" directive
        opt_list = directive_img_options
    elif directive == "figure":
        # "figure" directive
        opt_list = directive_fig_options
    else:
        # Bad directive
        raise ValueError(
            ("Image directive '%s' must be either " % directive) +
            ("'image' or 'figure'"))
    # Loop through other keywords
    for k in kw:
        # Check option
        if k not in opt_list:
            raise ValueError(
                "Invalid option '%s' for '%s' directive" % (k, directive))
        # Get value
        v = kw[k]
        # Create line for this text
        line = rst_directive_option(k, v)
        # Check if empty
        if line:
            lines.append(tab + line)
    # Output
    return lines
# def rst_image_lines


# Write a directive for an image
def rst_image(ffig, caption=None, **kw):
    """Write the reST command to include an image
    
    :Call:
        >>> txt = rst_image(ffig, caption=None, **kw)
    :Inputs:
        *ffig*: :class:`str`
            Name of image to include (often like ``"myimage.*"``)
        *caption*: {``None``} | :class:`str`
            Caption to use to describe image
        *indent*: {``""``} | :class:`str`
            Indentation to place before each line
        *tab*: {``"    "``} | :class:`int`
            Indent character for a tab (default is 4 spaces)
    :See Also:
        * :func:`rst_image_lines`
    :Outputs:
        *txt*: :class:`str`
            Text of directive
    :Versions:
        * 2019-02-24 ``@ddalle``: First version
    """
    # Get an indent
    indent = kw.pop("indent", kw.get("tab", ("    ")))
    # Get lines
    lines = rst_image_lines(ffig, **kw)
    # Create text
    txt = ""
    # Loop through lines
    for line in lines:
        # Include indent and new line
        txt += (indent + line.rstrip() + "\n")
    # Output
    return txt
# def rst_image


# Write a directive for an image
def rst_figure(ffig, caption, **kw):
    """Write the reST command to include an image
    
    :Call:
        >>> txt = rst_figure(ffig, caption, **kw)
    :Inputs:
        *ffig*: :class:`str`
            Name of image to include (often like ``"myimage.*"``)
        *caption*: :class:`str` | :class:`unicode`
            Caption to use to describe figure
        *tab*: {``"    "``} | :class:`int`
            Indent character for a tab (default is 4 spaces)
        *indent*: {*tab*} | :class:`str`
            Indentation to place before each line
        *width*: {``None``} | :class:`str`
            Image width option value
        *maxcol*, *maxcols*, *cwidth*: {``79``} | :class:`int` > 0
            Column number at which to wrap caption
    :See Also:
        * :func:`rst_image_lines`
    :Outputs:
        *txt*: :class:`str`
            Text of directive
    :Versions:
        * 2019-02-24 ``@ddalle``: First version
    """
    # Get an indent and tab
    tab    = kw.get("tab", "    ")
    indent = kw.pop("indent", tab)
    # Get lines
    lines = rst_image_lines(ffig, directive="figure", **kw)
    # Maximum width
    cwidth = kw.pop("cwidth", 79)
    cwidth = kw.pop("maxcol", cwidth)
    cwidth = kw.pop("maxcols", cwidth)
    # Calibrate max width to remove indent and tab
    cwidth = cwidth - len(tab) - len(indent)
    # Create initial text
    txt = ""
    # Loop through lines
    for line in lines:
        # Include indent and new line
        txt += (indent + line.rstrip() + "\n")
    # Blank line for caption
    txt += (indent + "\n")
    # Wrap caption
    caption_lines = wrap.wrap_text(caption, cwidth)
    # Append lthe lines
    for line in caption_lines:
        txt += (indent + tab + line.rstrip() + "\n")
    # Output
    return txt
# def rst_figure


# Create a table of images
def rst_image_table_lines(imgtable, **kw):
    """Process lines used to create a reST table of images
    
    :Call:
        >>> lines, subs = rst_image_table_lines(imgtable, **kw)
    :Inputs:
        *imgtable*: :class:`list` (:class:`list`)
            List of rows; each row is a list of cell contents
        *cell11*: :class:`tuple` | :class:`str`
            See :func:`unpack_image_cell`
        *colwidth*, *cellwidth*: {``72/ncol``} | :class:`int` > 0
            Minimum column width
    :Outputs:
        *lines*: :class:`list` (:class:`str`)
            List of lines describing the table itself, without indentation
        *subs*: :class:`str`
            Substitution definition text to insert after the table
    :Versions:
        * 2019-02-26 ``@ddalle``: First version
    """
   # --- Input Checks ---
    # Check type
    if imgtable.__class__.__name__ != "list":
        raise TypeError(
            "Image table input must be a list (got '%s')"
            % imgtable.__class__.__name__)
    # Check row types
    for i, row in enumerate(imgtable):
        # Get type
        t = row.__class__.__name__
        # Check it
        if t != "list":
            raise TypeError("Row %i is not a list (got '%s')" % (i, t))
    # Number of columns
    ncol_rows = [len(row) for row in imgtable]
    # Get min and max counts
    ncol_min = min(ncol_rows)
    ncol_max = min(ncol_rows)
    # Check consistency
    if ncol_min != ncol_max:
        raise ValueError(
            "Some rows have different lengths (%i to %i)" %
            (ncol_min, ncol_max))
    # Number of columns
    ncol = ncol_min
   # --- Global Options ---
   # --- Initialization ---
    # Initialize table of texts
    lines_table = []
    captions_table = []
    # Initialize substitution text
    subs_txt = ""
    # Running cell width (max)
    cellwidth = 0
   # --- Cell Contents ---
    # Loop through rows
    for (i, row) in enumerate(imgtable):
        # Initialize row of table
        lines_row = []
        captions_row = []
        # Loop through columns
        for (j, cell) in enumerate(row):
            # Unpack cell to figure name, caption, and keywords
            ffig, cap, kwj = unpack_image_cell(cell)
            # Apply default images
            kwc = dict(kw, **kwj)
            # Get column width
            cwidth = kwc.pop("cellwidth", kw.pop("colwidth", 72//ncol))
            # Check for a substitution
            tag = kwc.get("tag", kwc.get("sub"))
            # Create lines
            lines_cell = rst_image_lines(ffig, **kwc)
            # If there's a substitution, use it
            if tag:
                # Put the lines in the substitution area
                subs_txt += "\n"
                subs_txt += "\n".join(lines_cell)
                subs_txt += "\n"
                # Reset the lines to put in the table
                lines_cell = ["|%s|" % tag]
            # Save caption
            captions_row.append(cap)
            # Get maximum width of any line in this cell
            widthj = max([len(line)+1 for line in lines_cell])
            # Cumulative maximum width
            cellwidth = max(cellwidth, widthj)
            cellwidth = min(cellwidth, cwidth)
            # Save the lines
            lines_row.append(lines_cell)
        # Append the row cells
        lines_table.append(lines_row)
        # Save captions
        captions_table.append(captions_row)
   # --- Text ---
    # Create horizontal before/after delimiter for each cell
    hcell = ("-"*cellwidth) + "+"
    # Print instruction for each cell
    fcell = ("%%-%is|" % cellwidth)
    # Create horizontal delimiter
    hline = "+" + (hcell*ncol)
    # Initialize output: top bar
    lines = [hline]
    # Loop through rows: get line counts
    for lines_row, captions_row in zip(lines_table, captions_table):
        # Initialize line count
        nline_row = 0
        # Save wrapped lines
        row = []
        # Loop through columns: get line counts
        for lines_cell, cap in zip(lines_row, captions_row):
            # Check for caption
            if cap:
                # Wrap caption
                lines_cap = wrap.wrap_text(cap, cwidth=cellwidth-1)
                # Append both line contents
                cell = lines_cell + [""] + lines_cap
            else:
                # No caption
                cell = lines_cell
            # Number of lines in directive
            nline_cell = len(cell)
            # Max line count
            nline_row = max(nline_row, nline_cell)
            # Save contents
            row.append(cell)
        # Loop through line count
        for i in range(nline_row):
            # Start the line
            line = "|"
            # Loop through columns
            for col in row:
                # Number of lines in this column
                nrow_col = len(col)
                # Check if this is past the end of the cell
                if i >= nrow_col:
                    # Append empty cell
                    line += (fcell % "")
                else:
                    # Append contents of cell
                    line += (fcell % col[i])
            # Save the line
            lines.append(line)
        # Vertical delimiter
        lines.append(hline)
   # --- Output ---
    # Output
    return lines, subs_txt
# def rst_image_table_lines


# Create a table of images
def rst_image_table(imgtable, caption=None, directive="table", **kw):
    """Process lines used to create a reST table of images
    
    :Call:
        >>> txt = rst_image_table(imgtable, **kw)
    :Inputs:
        *imgtable*: :class:`list` (:class:`list`)
            List of rows; each row is a list of cell contents
        *cell11*: :class:`tuple` | :class:`str`
            See :func:`unpack_image_cell`
        *colwidth*, *cellwidth*: {``72/ncol``} | :class:`int` > 0
            Minimum column width
        *cell_options*: {``{}``} | :class:`dict`
            Default cell options, passed to :func:`rst_image_table_lines`
    :Outputs:
        *lines*: :class:`list` (:class:`str`)
            List of lines describing the table itself, without indentation
        *subs*: :class:`str`
            Substitution definition text to insert after the table
    :Versions:
        * 2019-02-26 ``@ddalle``: First version
    """
   # --- Global Options ---
    # Pop off indent option
    tab    = kw.get("tab", "    ")
    indent = kw.pop("indent", tab)
    # Get total textwidth
    twidth = kw.get("textwidth", 79)
   # --- Directive ---
    # Check value
    if directive is None:
        # No inclusion text
        txt = ""
    elif directive == "table":
        # Initialize directive
        txt = indent + ".. table::"
        # Check for caption
        if caption:
            # Indent for hanging lines (11 characters in ".. table:: ")
            indent2 = (" "*11) + indent
            # Number of available text columns for caption
            wtxt = twidth - len(indent2)
            # Wrap it
            lines_cap = wrap.wrap_text(caption, wtxt)
            # Add first line
            txt += (" " + lines_cap[0] + "\n")
            # Add hanging lines if any
            for line in lines_cap[1:]:
                txt += (indent2 + line + "\n")
        else:
            # End caption line
            txt += "\n"
    else:
        # Invalid directive
        raise ValueError(
            "Table directive '%s' should be 'table' or None" % directive)
   # --- Directive Options ---
    # No options if no directive
    if directive:
        # Indent for subsequent lines
        indent3 = indent + tab
        # Loop through possible options
        for k in directive_table_options:
            # Get value
            v = kw.pop(k, None)
            # Check if present
            if v is None:
                continue
            # Convert to text
            txtk = rst_directive_option(k, v)
            # Add it to the text
            txt += (indent3 + txtk + "\n")
        # Blank line
        txt += "\n"
    else:
        # Indent for table is just indent
        indent3 = indent
   # --- Table Lines ---
    # Get cell options
    kw_cell = kw.pop("cell_options", kw.pop("image_options", {}))
    # Form options for :func:`rst_image_table_lines`
    kwc = dict(kw, **kw_cell)
    # Call main function
    lines, subs_txt = rst_image_table_lines(imgtable, **kwc)
    # Loop through the lines
    for line in lines:
        # Indent and add to text
        txt += (indent3 + line + "\n")
    # Add in the substitution text
    txt += subs_txt
   # --- Output ---
    # Output
    return txt


# Unpack image table cell
def unpack_image_cell(cell):
    """Unpack contents of an image table cell
    
    :Call:
        >>> ffig, cap, kwj = unpack_image_cell(cell)
        >>> ffig, cap, kwj = unpack_image_cell(ffig)
        >>> ffig, cap, kwj = unpack_image_cell((ffig,))
        >>> ffig, cap, kwj = unpack_image_cell((ffig, cap))
        >>> ffig, cap, kwj = unpack_image_cell((ffig, kwj))
        >>> ffig, cap, kwj = unpack_image_cell((ffig, cap, kwj))
    :Inputs:
        *cell*: :class:`tuple` | :class:`list` | :class:`str`
            Tuple of figure name, caption, and options or just figure name
        *ffig*: :class:`str`
            Name of figure to include
        *cap*: :class:`str`
            Caption to use with cell contents
        *kwj*: :class:`dict`
            Dictionary of specific options for this cell
    :Outputs:
        *ffig*: :class:`str`
            Name of figure to include
        *cap*: {``None``} | :class:`str`
            Caption to use with cell contents
        *kwj*: {``{}``} | :class:`dict`
            Dictionary of specific options for this cell
    :Versions:
        * 2019-02-25 ``@ddalle``: First version
    """
    # Unpack cell
    try:
        # Unpack figure, caption, and keywords
        ffig, cap, kwj = cell
    except ValueError:
        # Failed to unpack three
        try:
            # Unpack figure and caption
            ffig, cap = cell
            # Check "caption" type
            if cap.__class__.__name__ == "dict":
                # Keywords given instead of caption
                kwj = cap
                cap = None
            else:
                # No keywords
                kwj = {}
        except Exception:
            # Single output
            try:
                # Single tuple
                ffig = cell,
            except Exception:
                # Single string
                ffig = cell
            # Now caption or keywords
            cap = None
            kwj = None
    # Output
    return ffig, cap, kwj
# def unpack_image_cell

