#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.testutils.argread`: Command-Line Argument Processor
===============================================================

Parse command-line inputs for test utilities.  This is similar to the
:mod:`cape.argread` module but simpler in order to make sure the test
suite can still run if there is a problem import :mod:`cape`.

"""

# Process options using any dash as keyword
def readkeys(argv):
    """
    Read list of command-line strings and convert to 
    
    :Call:
        >>> args, kwargs = readkeys(argv)
    :Inputs:
        *argv*: :class:`list`\ [:class:`str`]
            Input strings; first entry is ignored (from ``sys.argv``)
    :Outputs:
        *args*: :class:`list`
            List of general inputs with no keyword names
        *kwargs*: :class:`dict`
            Dictionary of inputs specified with option flags
        *kwargs['_old']*: :class:`list` (:class:`tuple`)
            List of argument-value pairs that were overwritten
    :Versions:
        * 2014-06-10 ``@ddalle``: First version
        * 2017-04-10 ``@ddalle``: Added ``--no-v`` --> ``v=False``
        * 2019-07-03 ``@ddalle``: Copied from :mod:`cape.argread`
    """
    # Check the input.
    if type(argv) is not list:
        raise TypeError('Input must be a list of strings.')
    # Initialize outputs.
    args = []
    kwargs = {}
    old = []
    # Get the number of arguments.
    argc = len(argv)
    # Exit if no arguments
    if len(argv) < 1:
        return args, kwargs
    # Pop first argument
    argv.pop(0)
    # Loop through arguments
    while len(argv) > 0:
        # Get next argument
        a = argv.pop(0)
        # Check if it starts with a `-`
        if a.startswith('-'):
            # Key name starts after '-'s
            k = a.lstrip('-')
            # Check for ``--no-$opt`` flags
            if k.startswith("no-"):
                # Strip the 'no-' part
                k = k[3:]
                q = False
            else:
                # Normal situation
                q = True
            # Check if already present
            if k in kwargs:
                old.append((k, kwargs[k]))
            # Check for a following argument
            if len(argv) > 0:
                # Get next argument
                a1 = argv[0]
                # Check if *that* is an option
                if a1.startswith("-"):
                    # Following flag; no value for this one
                    v = q
                else:
                    # No hyphen; must be value for this flag
                    v = argv.pop(0)
            else:
                # No following argument
                v = q
            # Save to dictionary
            kwargs[k] = v
        else:
            # Append to args
            args.append(a)
    # Save overwritten items
    kwargs["_old"] = old
    # Output
    return args, kwargs

