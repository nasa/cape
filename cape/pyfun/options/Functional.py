"""Interface for Cart3D adaptation settings"""


# Import options-specific utilities
from .util import getel
# Import base class
import cape.cfdx.options.Functional


# Class for output functional settings
class Functional(cape.cfdx.options.Functional.Functional):
    """Dictionary-based interface for output/objective functions"""
    
    # Get adaptive function coefficients
    def get_AdaptCoeffs(self):
        """Get the adaptive output function coefficients
        
        :Call:
            >>> coeffs = opts.get_AdaptCoeffs()
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
        :Ouutputs:
            *coeffs*: :class:`list`\ [:class:`str`]
                List of coefficients in the adaptive function
        :Versions:
            * 2016-04-26 ``@ddalle``: First version
        """
        # Get adaptive functions
        fns = self.get_AdaptFuncs()
        # Get the first
        if len(fns) >= 1:
            # First function
            fn = fns[0]
        else:
            # No function; use all coeffs
            fn = None
        # Default list of coefficients
        coeffs = self.keys() + []
        # Remove 'Functions' if necessary
        if 'Functions' in coeffs:
            coeffs.remove('Functions')
        # Check for function
        if fn is None:
            # Return default list
            return coeffs
        else:
            # Get the coefficients from 
            return self[fn].get('coeffs', coeffs)
    
    # Function to get weight for a certain functional term
    def get_FuncCoeffType(self, coeff, j=None):
        """Get the weight of a named functional term
        
        :Call:
            >>> typ = opts.get_FuncCoeffType(coeff, j=None)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *coeff*: :class:`str`
                Coefficient name
            *j*: :class:`int`
                Phase number
        :Outputs:
            *typ*: {cd} | cl | ca | cy | cn | cll | clm | cln | cx | cy | cz
                Value type, class of coefficient
        :Versions:
            * 2016-04-24 ``@ddalle``: First version
        """
        # Get the term
        fopts = self.get(coeff, {})
        # Get the weight
        return getel(fopts.get('type', 'cd'), j)
    
# class Functional

