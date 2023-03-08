r"""
The :mod:`cape.pycart` module contains the top-level interface for
Cart3D setup and execution.

    .. code-block:: python

        import cape.pycart
        
For example the following will read in a global settings instance
assuming that the present working directory contains the correct files.
(If not, various defaults will be used, but it is unlikely that the
resulting setup will be what you intended.)

    .. code-block:: python

        import cape.pycart
        cntl = cape.pycart.Cntl()
        
A simpler example is to simply read a ``.tri`` file, rotate it about the
*x*-axis by 20 degrees, and write it to a new file.

    .. code-block:: python

        # Import the module.
        import cape.pycart
        # Read the .tri file.
        tri = pyCart.Tri('bJet.i.tri')
        # Rotate it.
        tri.Rotate([0.,0.,0.], [1.,0.,0.], 20)
        # Write it to a new file.
        tri.Write('bJet_rotated.i.tri')
        
Most of the pyCart submodules essentially contain a one or more class
definitions, and some of these classes are accessible directly from
:mod:`cape.pycart`.

The module also contains the :mod:`cape.pycart.bin` module, which
contains functions that run the main Cart3D binaries: ``autoInputs``,
``cubes``, ``mgPrep``, and ``flowCart``.
    
Modules included within this one are outlined below.

    * Core modules:
        - :mod:`cape.pycart.cntl`
        - :mod:`cape.pycart.case`
        - :mod:`cape.pycart.manage`
        - :mod:`cape.pycart.dataBook`
        - :mod:`cape.pycart.options`
        
    * Cart3D and Cape files
        - :mod:`cape.pycart.inputCntl`
        - :mod:`cape.pycart.aeroCsh`
        - :mod:`cape.pycart.preSpecCntl`
        - :mod:`cape.pycart.runmatrix`
        - :mod:`cape.pycart.tri`
        
    * Supporting modules
        - :mod:`cape.pycart.cmd`
        - :mod:`cape.pycart.bin`
        - :mod:`cape.pycart.report`
        - :mod:`cape.pycart.util`

"""

# System
import os


# Save version number
version = "1.0"
__version__ = version

# Get the root directory of the module.
_fname = os.path.abspath(__file__)

# Saved folder names
PyCartFolder = os.path.split(_fname)[0]

# Import classes and methods from the submodules
from .cntl import Cntl, RunMatrix
from .inputCntl   import InputCntl
from .aeroCsh     import AeroCsh
from .preSpecCntl import PreSpecCntl
from .dataBook    import CaseFM, CaseResid


