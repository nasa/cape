#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`cape.tnakit` module is a toolkit of common tools that are not
necessarily specific to CAPE or database management but are not part of
the standard library either.

For example, it contains utilities, mostly in
:mod:`cape.tnakit.typeutils`, that generalize code between Python 2 and
Python 3.  It also contains useful tools built only on the standard
library (and are compatible with both Python 2.7 and Python 3.6+) like
creating formatted reStructuredText representations of Python objects
(in :mod:`cape.tnakit.rstutils`).

Other extensions to standard Python conventions are provided, like a
handler for keyword arguments with many possible values
(:mod:`cape.tnakit.kwutils`).

Another useful package provides extra options for plotting
(:mod:`cape.tnakit.plot_mpl`).

"""

__version__ = "1.0"

