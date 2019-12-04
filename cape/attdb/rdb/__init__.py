#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`cape.attdb.rdb`: Response database modules
====================================================

This module contains containers for basic (mostly) numeric databases
that do not have inherent response surfaces attached to them.

The :class:`DBResponseNull` class created in this module is a template
for (mostly) numeric databases throughout the :mod:`cape.attdb`
package.  The data storage aspects are handled by a data file interface
class, :mod:`cape.attdb.ftypes`, which provides tools for reading and
writing several different file formats.

"""

# Local direct imports
from .rdbnull import DBResponseNull
