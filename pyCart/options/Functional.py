"""Interface for Cart3D adaptation settings"""


# Import options-specific utilities
from util import odict


# Class for output functional settings
class Functional(odict):
    """Dictionary-based interface for Cart3D output functionals"""
    
    # Function to return all the optForce dicts found
    def get_optForces(self):
        """Return a list of output forces to be used in functional
        
        :Call:
            >>> optForces = opts.get_optForces()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *optForces*: :class:`list` (:class:`dict`)
                List of output force dictionaries
        :Versions:
            * 2014-11-19 ``@ddalle``: First version
        """
        # Initialize output
        optForces = {}
        # Loop through keys
        for k in self.keys():
            # Get the key value.
            v = self[k]
            # Check if it's a dict.
            if type(v).__name__ != "dict": continue
            # Check if it's a force
            if v.get('Type', 'optForce') == 'optForce':
                # Append the key.
                optForces[k] = v
        # Output
        return optForces
        
