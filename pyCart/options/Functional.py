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
        
    # Function to return all the optSensor dicts found
    def get_optSensors(self):
        """Return a list of output sensors to be used in functional
        
        :Call:
            >>> optSensors = opts.get_optSensors()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *optSensors*: :class:`list` (:class:`dict`)
                List of output sensor dictionaries
        :Versions:
            * 2015-05-06 ``@ddalle``: First version
        """
        # Initialize output
        optSensors = {}
        # Loop through keys.
        for k in self.keys():
            # Get the key value.
            v = self[k]
            # Check if it's a dict.
            if type(v).__name__ != "dict": continue
            # Check if it's a sensor.
            if v.get('Type', 'optForce') == 'optSensor':
                # Append the key.
                optSensors[k] = v
        # Output
        return optSensors
        
    # Function to return all the optMoment_Point dicts found
    def get_optMoments(self):
        """Return a list of moment coefficients to be used in functional
        
        :Call:
            >>> optMoments = opts.get_optMoments()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *optMoments*: :class:`list` (:class:`dict`)
                List of output moment coefficient dictionaries
        :Versions:
            * 2015-05-14 ``@ddalle``: First version
        """
        # Initialize output
        optMoments = {}
        # Loop through keys.
        for k in self.keys():
            # Get the key value.
            v = self[k]
            # Check if it's a dict.
            if type(v).__name__ != "dict": continue
            # Check if it's a sensor.
            if v.get('Type', 'optForce') in ['optMoment', 'optMoment_Point']:
                # Append the key.
                optMoments[k] = v
        # Output
        return optMoments
    
