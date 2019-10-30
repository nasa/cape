"""
:mod:`cape.pyover.options.Config`: pyOver configurations options
=================================================================

This module provides options for defining some aspects of the surface
configuration for an OVERFLOW run.  Because OVERFLOW contains its own
instructions for most surface family definitions, etc. in a specified
``mixsur.fmp`` file, this section does very little.

The reference area (``"RefArea"``) and reference length (``"RefLength"``)
parameters are also defined in this section. OVERFLOW does not have two
separate reference lengths, so there is no ``"RefSpan"`` parameter. These
parameters are not used as inputs to OVERFLOW, which are already part of the
inputs created by ``mixsur`` or ``usurp``.  However, they can be used to
dimensionalize results during post-processing.

Many parameters are inherited from the :class:`cape.config.Config` class, so
readers are referred to that module for defining points by name along with
several other methods.

One aspect of ``"Config"`` that matches the behavior of the version for other
solvers is that it can be used to define reference points.  A sample is shown
below.

    .. code-block:: javascript
    
        "Config": {
            "RefLength": 2.0,
            "RefArea": 3.14159,
            "Points": {
                "P01": [15.0, 1.0, 0.0],
                "P02": [15.0, 0.0, 1.0]
            }
        }

:See Also:
    * :mod:`cape.cfdx.options.Config`
    * :mod:`cape.config`
    * :mod:`cape.pyover.overNamelist`
"""

# Import options-specific utilities
from .util import rc0

# Import base class
import cape.cfdx.options.Config

# Class for PBS settings
class Config(cape.cfdx.options.Config):
    """
    Configuration options for OVERFLOW
    
    :Call:
        >>> opts = Config(**kw)
    :Versions:
        * 2015-12-29 ``@ddalle``: First version
    """
    
# class Config

