"""Interface for configuration control: :mod:`pyCart.options.Config`"""


# Import options-specific utilities
from util import rc0

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
        conf_inp = self.get("Inputs")
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
    
# class Config

