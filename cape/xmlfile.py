# -*- coding: utf-8 -*-
r"""
:mod:`cape.xmlfile`: Extended interface to XML files
======================================================

This module provides the class :class:`XMLFile`, which extends slightly
the built-in class :class:`xml.etree.ElmentTree`. Compared to the
standard library class, :class:`XMLFile` has a more top-level interface.

Specifically, it is possible to find and/or edit properties of
subelements that are arbitrarily deep within the file using methods for
the top-level class. This is convenient (for example) for CFD solvers
using XML files as their input because it eases the process of changing
minor settings (for example the angle of attack) without searching
through multiple levels of elements and subelements.

"""

# Standard library
from ml.etree import ElementTree


# Primary class
class XMLFile(object):
    r"""Interface to XML files

    :Call:
        >>> xml = XMLFile(fxml)
        >>> xml = XMLFile(et)
        >>> xml = XMLFile(opts)
        >>> xml = XMLFile(txt)
    :Versions:
        * 2021-10-06 ``@ddalle``: Version 0.0: Started
    """
    # Initialization method
    def __init__(self, arg0, **kw):
        r"""Initialization method

        :Versions:
            * 2021-10-06 ``@ddalle``: Version 1.0
        """
        pass

