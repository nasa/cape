"""Interface for Cart3D adaptation settings"""


# Import options-specific utilities
from .util import odict, getel


# Class for output functional settings
class Functional(odict):
    """Dictionary-based interface for output/objective functions"""
    
    # Function to get functionals
    def get_FuncCoeffs(self, fn):
        """Get the list of terms in a function
        
        :Call:
            >>> coeffs = opts.get_FuncCoeffs(fn)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fn*: :class:`str`
                Function name
        :Outputs:
            *coeffs*: :class:`list` (:class:`str`)
                List of coefficients
        :Versions:
            * 2016-04-24 ``@ddalle``: First version
        """
        # Get the functional options
        fopts = self.get('Functions', {})
        fopts = fopts.get(fn, {})
        # Default keys
        coeffs = self.keys() + []
        # Remove function defs
        if 'Functions' in coeffs:
            coeffs.remove('Functions')
        # Get the coeffs
        return fopts.get('coeffs', coeffs)
        
    # Function to get functional type
    def get_FuncType(self, fn):
        """Get the functional type
        
        :Call:
            >>> typ = opts.get_FuncType(fn)
        
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fn*: :class:`str`
                Function name
        :Outputs:
            *typ*: {``adapt``} | ``objective`` | ``constraint``
                Function type
        :Versions:
            * 2016-04-25 ``@ddalle``: First version
        """
        # Get the functional options
        fopts = self.get('Functions', {})
        fopts = fopts.get(fn, {})
        # Get the coeffs
        return fopts.get('type', 'adapt')
        
    # Get functional type
    def get_OptFuncs(self):
        """Get list of objective functions
        
        :Call:
            >>> fns = opts.get_OptFuncs()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *fns*: :class:`list` (:class:`str`)
                List of objective functions 
        :Versions:
            * 2016-04-25 ``@ddalle``: First version
        """
        # Initialize
        fns = []
        # Get the functions
        fopts = self.get('Functions', {})
        # Loop through keys
        for fn in fopts.keys():
            # Get the type
            typ = fopts.get('type', 'adapt')
            # Check
            if typ in ['opt', 'objective']:
                fns.append(fn)
        # Output
        return fns
    
    # Get adaptive function
    def get_AdaptFuncs(self):
        """Get list of adaptation functions
        
        :Call:
            >>> fns = opts.get_AdaptFuncs()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *fns*: :class:`list` (:class:`str`)
                List of adaptive output functions
        :Versions:
            * 2016-04-26 ``@ddalle``: First version
        """
        # Initialize
        fns = []
        # Get the functions
        fopts = self.get('Functions', {})
        # loop through keys
        for fn in fopts.keys():
            # Get the type
            typ = fopts.get('type', 'adapt')
            # Check
            if typ in ['adapt', 'output']:
                fns.append(fn)
        # Output
        return fns
    
    # Get constraint function
    def get_ConstraintFuncs(self):
        """Get list of adaptation functions
        
        :Call:
            >>> fns = opts.get_ConstraintFuncs()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *fns*: :class:`list` (:class:`str`)
                List of adaptive output functions
        :Versions:
            * 2016-04-27 ``@ddalle``: First version
        """
        # Initialize
        fns = []
        # Get the functions
        fopts = self.get('Functions', {})
        # loop through keys
        for fn in fopts.keys():
            # Get the type
            typ = fopts.get('type', 'adapt')
            # Check
            if typ in ['con', 'constraint']:
                fns.append(fn)
        # Output
        return fns
            
    
    # Function to get weight for a certain functional term
    def get_FuncCoeffWeight(self, coeff, j=None):
        """Get the weight of a named functional term
        
        :Call:
            >>> w = opts.get_FuncCoeffWeight(coeff, j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *coeff*: :class:`str`
                Coefficient name
            *j*: :class:`int`
                Phase number
        :Outputs:
            *w*: :class:`float`
                Function weight
        :Versions:
            * 2016-04-24 ``@ddalle``: First version
        """
        # Get the term
        fopts = self.get(coeff, {})
        # Get the weight
        return getel(fopts.get('weight', 1.0), j)
        
    # Function to get target value for a certain functional term
    def get_FuncCoeffTarget(self, coeff, j=None):
        """Get the target value for a named functional term
        
        :Call:
            >>> t = opts.get_FuncCoeffTarget(coeff, j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *coeff*: :class:`str`
                Coefficient name
            *j*: :class:`int`
                Phase number
        :Outputs:
            *t*: :class:`float`
                Target value, term is ``w*(v-t)**p``
        :Versions:
            * 2016-04-25 ``@ddalle``: First version
        """
        # Get the term
        fopts = self.get(coeff, {})
        # Get the target
        return getel(fopts.get('target', 0.0), j)
        
    # Function to get exponent for a certain functional term
    def get_FuncCoeffPower(self, coeff, j=None):
        """Get exponent for a certain functional term
        
        :Call:
            >>> p = opts.get_FuncCoeffPower(coeff, j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *coeff*: :class:`str`
                Coefficient name
            *j*: :class:`int`
                Phase number
        :Outputs:
            *p*: :class:`float`
                Exponent, term is ``w*(v-t)**p``
        :Versions:
            * 2016-04-25 ``@ddalle``: First version
        """
        # Get the term
        fopts = self.get(coeff, {})
        # Get the exponent
        return getel(fopts.get('power', 1.0), j)
        
    # Function to get the component for a certain functional term
    def get_FuncCoeffCompID(self, coeff, j=None):
        """Get the component to apply functional term to
        
        :Call:
            >>> compID = opts.get_FuncCoeffCompID(coeff, j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *coeff*: :class:`str`
                Coefficient name
            *j*: :class:`int`
                Phase number
        :Outputs:
            *compID*: :class:`str` | :class:`int` | :class:`list`
                Component name or number
        :Versions:
            * 2016-04-25 ``@ddalle``: First version
        """
        # Get the term
        fopts = self.get(coeff, {})
        # Get the exponent
        return getel(fopts.get('compID', []), j)
        
    # Function to return all the optForce dicts found
    def get_optForces(self):
        """Return a list of output forces to be used in functional
        
        :Call:
            >>> optForces = opts.get_optForces()
        :Inputs:
            *opts*: :class:`cape.options.Options`
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
            *opts*: :class:`cape.options.Options`
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
            *opts*: :class:`cape.options.Options`
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
        
# class Functional

