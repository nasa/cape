"""
pyCart
======

A Python interface for Cart3D, a Cartesian-grid, cut-cell CFD tool.
"""

# Set version number.  Don't forget to update it.
__version__ = 0.1

# Configuration file processor
import json
# File system and operating system management
import os
# More powerful text processing
import re


# Maximum number of case
MAXCASES = 10000


# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyCartFolder = os.path.split(_fname)[0]
TemplateFodler = os.path.join(PyCartFolder, "templates")


# Import classes and methods from the submodules
from tri  import Tri, ReadTri, WriteTri
from cntl import Cntl, Trajectory, ReadTrajectoryFile, CreateFolders
from post import LoadsCC


# Get the conversion tools directly.
from convert import *
