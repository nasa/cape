#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
---------------------------------------------------------------------
:mod:`cape.tnakit.plotutils.genopts`: Generic Plotting Options Tools
---------------------------------------------------------------------

This is a module that contains utilities for processing options to
plotting functions and is not specific to Matplotlib or another fixed
plot library.

"""


# Method to transfer a key from one dictionary to another
def transfer_key(opts1, opts2, k, k2=None):
    """Remove a key *k* from *opts1* and set it in *opts2* if present

    :Call:
        >>> transfer_key(opts1, opts2, k, k2=None)
    :Inputs:
        *opts1*: :class:`dict`
            Original dictionary from which key may be removed
        *opts2*: :class:`dict`
            If *k* is in *opts1*, the value is transferred to *opts2*
        *k*: :class:`str`
            Name of key to check and transfer
    :Versions:
        * 2019-03-08 ``@ddalle``: First version
    """
    # Default value of k2
    if k2 is None:
        k2 = k
    # Check if *k* is present in *opts1*
    if k in opts1:
        # Transfer value and remove it
        opts2[k2] = opts1.pop(k)


# Display any remaining options
def display_remaining(funcname, indent, **kw):
    """Display warning for any remaining (unused) options

    :Call:
        >>> display_remaining(funcname, indent, **kw)
    :Inputs:
        *funcname*: ``None`` | :class:`str`
            Name of function to display in warnings
        *indent*: ``None`` | :class:`str`
            Indentation (usually zero or more spaces) before warning
    :Versions:
        * 2019-03-08 ``@ddalle``: First version
    """
    # Default indent
    if indent is None:
        indent = ""
    # Check for function name
    if funcname:
        # Loop through keys
        for k in kw:
            # Print warning
            print(indent +
                  ("Warning: In function '%s', " % funcname) +
                  ("unused option '%s'" % k))
    else:
        # Loop through keys
        for k in kw:
            # Print warning
            print(indent + ("Warning: unused option '%s'" % k))


# Remove ``None`` keys
def denone(opts):
    """Remove any keys whose value is ``None``

    :Call:
        >>> opts = denone(opts)
    :Inputs:
        *opts*: :class:`dict`
            Any dictionary
    :Outputs:
        *opts*: :class:`dict`
            Input with any keys whose value is ``None`` removed; returned for
            convenience but input is also affected
    :Versions:
        * 2019-03-01 ``@ddalle``: First version
    """
    # Loop through keys
    for (k, v) in dict(opts).items():
        # Check if ``None``
        if v is None:
            opts.pop(k)
    # Output
    return opts

