#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.cli`: Other ``cape`` command-line interface functions
=================================================================

This module defines functions that do the work of auxiliary ``cape``
command-line interface commands such as

    .. code-block:: console

        $ cape-expandjson

"""

# Standard library modules
import os
import shutil
import sys

# Local imprts
from . import argread
from . import text as textutils
from .cfdx.options.util import expandJSONFile


# Template help messages
_help_help = r"""-h, --help
        Display this help message and exit"""

_help = {
    "help": _help_help,
}


HELP_EXPANDJSON = r"""
``cape-expandjson``: Expand a JSON file and remove comments
=============================================================

This function performs two tasks:
    
    * Removes comments starting with either ``"//"`` or ``"#"``
    * Replaces ``JSONFile(fname)`` with the contents of *fname*

The comments are removed recursively, so comments within children files
are allowed.  If the output file is the same as the input file, the
suffix ``.old`` is added to the original file, and the expanded contents
are written to a new file with the same name as the input file.

The default output file *OFILE* puts the infix "expanded" before the
``.json`` of the input file. So for example

    ``myfile.json`` -> ``myfile.expanded.json``

:Usage:
    .. code-block:: console

        $ cape-expandjson IFILE OFILE [OPTIONS]
        $ cape-expandjson -i IFILE [-o OFILE] [OPTIONS]

:Inputs:
    * *IFILE*: name of input file
    * *OFILE*: output file for *IFILE*

:OPTIONS:
    -h, --help
        Display this help message and exit
        
    -i IFILE
        Use *IFILE* as input file

    -o OFILE
        Use *OFILE* as output file

:Versions:
    * 2015-10-27 ``@ddalle``: Version 1.0: :func:`ExpandJSON`
    * 2021-10-12 ``@ddalle``: Version 2.0; move to :mod:`cli`
"""


# Python functions
def expand_json(*a, **kw):
    r"""Expand one or more JSON files
    
    This function performs two tasks:
    
        * Removes comments starting with either ``'//'`` or ``'#'``
        * Replaces ``JSONFile(fname)`` with the contents of *fname*
    
    The comments are removed recursively, so comments within children
    files are allowed.  If the output file is the same as the input
    file, the suffix ``.old`` is added to the original file, and the
    expanded contents are written to a new file with the same name as
    the input file.
    
    :Call:
        >>> expand_json(ifile, ofile, **kw)
        >>> expand_json(i=ifile, o=ofile, **kw)
    :Inputs:
        *ifile*: :class:`str`
            Name of input file
        *ofile*: :class:`str`
            Name of output file
            (default "file.json" -> "file.expanded.json")
    :Versions:
        * 2015-10-27 ``@ddalle``: Version 1.0
        * 2021-10-12 ``@ddalle``: Version 2.0; single file
    """
    # Get input file name
    ifile = _get_i(*a, **kw)
    # Get output file name
    ofile = _get_o_infix(ifile, "json", "expanded", *a, **kw)
    # Check for matches
    if ifile == ofile:
        # Copy original file
        shutil.copy(ifile, ifile + ".old")
    # Read the file
    lines = open(ifile).readlines()
    # Strip the comments and expand subfiles
    lines = expandJSONFile(lines)
    # Write to the output file
    with open(ofile, "w") as fp:
        fp.write(lines)


# CLI functions
def main_expandjson():
    r"""CLI for :func:`expand_json`

    :Call:
        >>> main_expandjson()
    :Versions:
        * 2021-10-12 ``@ddalle``: Version 1.0
    """
    _main(expand_json, HELP_EXPANDJSON)


def _main(func, doc):
    r"""Command-line interface template

    :Call:
        >>> _main(func, doc)
    :Inputs:
        *func*: **callable**
            API function to call after processing args
        *doc*: :class:`str`
            Docstring to print with ``-h``
    :Versions:
        * 2021-10-01 ``@ddalle``: Version 1.0
    """
    # Process the command-line interface inputs.
    a, kw = argread.readkeys(sys.argv)
    # Check for a help option.
    if kw.get('h', False) or kw.get("help", False):
        print(textutils.markdown(doc))
        return
    # Run the main function.
    func(*a, **kw)


# Process sys.argv
def _get_i(*a, **kw):
    r"""Process input file name

    :Call:
        >>> fname_in = _get_i(*a, **kw)
        >>> fname_in = _get_i(fname)
        >>> fname_in = _get_i(i=None)
    :Inputs:
        *a[0]*: :class:`str`
            Input file name specified as first positional arg
        *i*: :class:`str`
            Input file name as kwarg; supersedes *a*
    :Outputs:
        *fname_in*: :class:`str`
            Input file name
    :Versions:
        * 2021-10-01 ``@ddalle``: Version 1.0
    """
    # Get the input file name
    if len(a) == 0:
        # Defaults
        fname_in = None
    else:
        # Use the first general input
        fname_in = a[0]
    # Prioritize a "-i" input
    fname_in = kw.get('i', fname_in)
    # Must have a file name.
    if fname_in is None:
        # Required input.
        raise ValueError("Required at least 1 arg or 'i' kwarg")
    # Output
    return fname_in


def _get_o_infix(fname_in, ext1, infix, *a, **kw):
    r"""Process output file name

    :Call:
        >>> fname_out = _get_o(fname_in, ext1, infix, *a, **kw)
    :Inputs:
        *fname_in*: :class:`str`
            Input file name
        *ext1*: :class:`str`
            Expected file extension for *fname_in*
        *infix*: :class:`str`
            Default file extension for *fname_out*
        *a[1]*: :class:`str`
            Output file name specified as first positional arg
        *o*: :class:`str`
            Output file name as kwarg; supersedes *a*
    :Outputs:
        *fname_out*: :class:`str`
            Output file name
    :Versions:
        * 2021-10-01 ``@ddalle``: Version 1.0
    """
    # Strip *ext1* as starter for default
    if ext1 and fname_in.endswith(ext1):
        # Specified extension
        ext = ext1
    else:
        # Use last extension
        ext = fname_in.split(".")[-1]
    # Strip expected file extension
    fname_out = fname_in[:-len(ext)]
    # Add supplied *infix* and repeat extension
    fname_out = fname_out + infix + "." + ext
    # Get the output file name
    if len(a) >= 2:
        fname_out = a[1]
    # Prioritize a "-o" input.
    fname_out = kw.get('o', fname_out)
    # Output
    return fname_out

