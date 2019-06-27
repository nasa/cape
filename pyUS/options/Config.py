"""
:mod:`pyUS.options.Config`: pyUS configurations options
==========================================================

This module provides options for defining some aspects of the surface
configuration for a US3D run.  It can point to a surface configuration file
such as :file:`Config.xml` or :file:`Config.json` that reads an instance of
:class:`cape.config.Config` or :class:`cape.config.ConfigJSON`, respectively.
This surface configuration file is useful for grouping individual components
into families using a format very similar to how they are defined for Cart3D.

The ``"Config"`` section also defines which components are requested by FUN3D
for iterative force & moment history reporting.  For the moment histories, this
section also specifies the moment reference points (moment center in FUN3D
nomenclature) for each component.

This is the section in which the user specifies which components to track
forces and/or moments on, and in addition it defines a moment reference point
for each component.

The reference area (``"RefArea"``) and reference length (``"RefLength"``)
parameters are also defined in this section.

Many parameters are inherited from the :class:`cape.config.Config` class, so
readers are referred to that module for defining points by name along with
several other methods.

Like other solvers, the ``"Config"`` section is also used to define the
coordinates of named points.  These can be specified as point sensors to be
reported on directly by US3D and/or define named points for other sections of
the JSON file.

:See Also:
    * :mod:`cape.options.Config`
    * :mod:`cape.config`
"""


# Import options-specific utilities
from .util import rc0

# Import base class
import cape.options.Config

# Class for PBS settings
class Config(cape.options.Config):
    """
    Configuration options for Fun3D
    
    :Call:
        >>> opts = Config(**kw)
    :Versions:
        * 2015-10-20 ``@ddalle``: First version
    """
   # ------------------
   # Component Mapping
   # ------------------
   # [
    # Get inputs for a particular component
    def get_ConfigInput(self, comp):
        """Return the input for a particular component
        
        :Call:
            >>> inp = opts.get_ConfigInput(comp)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
        :Outputs:
            *inp*: :class:`str` | :class:`list` (:class:`int`)
                List of BCs in this component
        :Versions:
            * 2015-10-20 ``@ddalle``: First version
        """
        # Get the inputs.
        conf_inp = self.get("Inputs", {})
        # Get the definitions
        return conf_inp.get(comp)
        
    # Set inputs for a particular component
    def set_ConfigInput(self, comp, inp):
        """Set the input for a particular component
        
        :Call:
            >>> opts.set_ConfigInput(comp, nip)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *inp*: :class:`str` | :class:`list` (:class:`int`)
                List of BCs in this component
        :Versions:
            * 2015-10-20 ``@ddalle``: First version
        """
        # Ensure the field exists.
        self.setdefault("Inputs", {})
        # Set the value.
        self["Inputs"][comp] = inp
   # ]
   
   # ------------
   # Other Files
   # ------------
   # [
   
   
   # ]
        
   # ----------
   # Points
   # ----------
   # [
   
   
   # ]
   
# class Config

