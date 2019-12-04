#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.attdb.ftypes.basefile`: Common ATTDB file type attributes
=====================================================================

This module provides the class :class:`DBResponseNull` as a subclass of
:class:`dict` that contains methods common to each of the other (mostly)
databases.  The :class:`DBResponseNull` class has most of the database
properties and methods but does not define "response surface"
capabilities that come from other classes that inherit from
:class:`DBResponseNull`.

Finally, having this common template class provides a single point of
entry for testing if an object is based on a product of the
:mod:`cape.attdb.rdb` module.  The following Python sample tests if
any Python object *db* is an instance of any class from this data-file
collection.

    .. code-block:: python

        isinstance(db, cape.attdb.rdb.DBResponseNull)
"""

# Standard library modules
import os
import warnings

# Third-party modules
import numpy as np

# CAPE modules
import cape.attdb.ftypes as ftypes



# Declare base class
class DBResponseNull(ftypes.DataFile):
    r"""Basic database template without responses
    
    :Call:
        >>> db = DBResponseNull(fname=None, **kw)
    :Inputs:
        *fname*: {``None``} | :class:`str`
            File name; extension is used to guess data format
        *csv*: {``None``} | :class:`str`
            Explicit file name for :class:`CSVFile` read
        *textdata*: {``None``} | :class:`str`
            Explicit file name for :class:`TextDataFile`
        *simplecsv*: {``None``} | :class:`str`
            Explicit file name for :class:`CSVSimple`
    :Outputs:
        *db*: :class:`cape.attdb.rdb.rdbnull.DBResponseNull`
            Generic database
    :Versions:
        * 2019-12-04 ``@ddalle``: First version
    """
    pass
