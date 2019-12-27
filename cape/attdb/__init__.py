#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`cape.attdb` module is named for its original intention of
handling "databases" for the Aero Task Team of the Space Launch System
program. The databases created for technical disciplines tend to differ
significantly and fundamentally from the typical databases in the
software world, which are little more than a series of spreadsheets in
a different data format.

Technical databases also contain "lookup" rules and have meaning at
conditions that are not directly represented in the database.  For
example, an integrated force & moment database might contain 6 output
force & moment coefficients that are a function of Mach number, angle
of attack, and sideslip angle.  The file might only have data at Mach
1.2 and 1.4, but it still has to provide an answer when the user wants
to know the forces & moments at Mach 1.32.  The "response database"
contains some data along with explicitly specified rules.

The :mod:`cape.attdb` module is intended to provide a library of tools
useful to this situation, with an obvious focus on the data products
that are needed to support aerodynamic analysis of launch vehicles.

"""

__version__ = '1.0'

