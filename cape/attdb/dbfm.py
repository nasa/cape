#!/us/bin/env python
# -*- coding: utf-8 -*-
r"""
-----------------------------------------------------------------
:mod:`cape.attdb.dbfm`: Aero Task Team Force & Moment Databases
-----------------------------------------------------------------

This module provides customizations of :mod:`cape.attdb.rdb` that are
especially useful for launch vehicle force & moment databases.  The
force & moment coefficient names follow common missile naming
conventions:

    =======  ===============  ============================
    Name     Symbol           Description
    =======  ===============  ============================
    *CA*     :math:`C_A`      Axial force
    *CY*     :math:`C_Y`      Side force
    *CN*     :math:`C_N`      Normal force
    *CLL*    :math:`C_\ell`   Rolling moment
    *CLM*    :math:`C_m`      Pitching moment
    *CLN*    :math:`C_n`      Yawing moment
    =======  ===============  ============================


"""

# Standard library modules
import os
import sys

# Third-party modules
import numpy as np

# Local modules
from . import rdb


