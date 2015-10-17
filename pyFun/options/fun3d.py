"""
Interface to FUN3D namelist options
===================================

This module provides a class to mirror the Fortran namelist capability.  For
now, nonunique section names are not allowed.
"""

# Ipmort options-specific utilities
from util import rc0, odict

# Class for namelist settings
class Fun3d(odict):
    """Dictionary-based interface for FUN3D namelists"""
    
    # 
    pass
