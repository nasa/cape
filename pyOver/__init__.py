"""
*****************
The pyOver module
*****************

The :mod:`pyCart` module contains the top-level interface for OVERFLOW setup. It
loads the most important methods from the various submodules so that they are
easier to access. Most tasks using the pyCart API can be accessed by loading
this module.

    .. code-block:: python
    
        import pyOver
        
For example the following will read in a global settings instance assuming that
the present working directory contains the correct files.  (If not, various
defaults will be used, but it is unlikely that the resulting setup will be what
you intended.)

    .. code-block:: python
        
        import pyOver
        oflow = pyOver.Overflow()
        
A simpler example is to simply read a `.tri` file, rotate it about the *x*-axis
by 20 degrees, and write it to a new file.

    .. code-block:: python
    
        # Import the module.
        import pyOver
        # Read the .tri file.
        tri = pyOver.Tri('bJet.i.tri')
        # Rotate it.
        tri.Rotate([0.,0.,0.], [1.,0.,0.], 20)
        # Write it to a new file.
        tri.Write('bJet_rotated.i.tri')
        
Most of the pyCart submodules essentially contain a single class definition, and
that class is accessible directly from the :mod:`pyCart` module.

The following classes are imported in this module, so that code like
``pyOver.Tri`` will work (although ``pyOver.tri.Tri``) will also work.

    * :class:`pyOver.tri.Tri`
    * :class:`pyOver.overflow.Overflow`
    * :class:`pyOver.trajectory.Trajectory`
"""

# System
import os

# Save version number
version = "0.5"

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyOverFolder = os.path.split(_fname)[0]

# Import Control module
#from overflow  import Overflow, Trajectory

