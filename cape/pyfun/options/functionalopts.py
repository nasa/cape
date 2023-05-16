"""Interface for Cart3D adaptation settings"""


# Local imports
from ...optdict import OptionsDict, FLOAT_TYPES


# Class for individual function opts
class FunctionOpts(OptionsDict):
    # Attributes
    __slots__ = ()

    # Options
    _optlist = {
        "coeffs",
        "parent",
        "type",
    }

    # Aliases
    _optmap = {
        "cols": "coeffs",
    }

    # Types
    _opttypes = {
        "coeffs": str,
        "parent": str,
        "type": str,
    }

    # List depth
    _optlistdepth = {
        "coeffs": 1,
    }

    # Values
    _optvals = {
        "type": ("adapt", "constraint", "objective"),
    }

    # Defaults
    _rc = {
        "type": "adapt",
    }


# Collection of functions
class FunctionCollectionOpts(OptionsDict):
    # Attributes
    __slots__ = ()

    # Types
    _opttypes = {
        "_default_": dict,
    }

    # Sections
    _sec_cls_opt = "Parent"
    _sec_cls_optmap = {
        "_default_": FunctionOpts,
    }

    # Get option for a function
    def get_FunctionalFuncOpt(self, fn: str, opt: str, j=None, **kw):
        r"""Get option for a specific functional function

        :Call:
            >>> v = opts.get_FunctionalFuncOpt(fn, opt, j=None, **kw)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *fn*: :class:`str`
                Name of function
            *opt*: :class:`str`
                Name of functional option
        :Outputs:
            *v*: :class:`object`
                Value of ``opts[fn][opt]`` or as appropriate
        :Versions:
            * 2023-05-16 ``@ddalle``: v1.0
        """
        return self.get_subopt(fn, opt, j=j, **kw)

    # Get list of optimization functions
    def get_FunctionalAdaptFuncs(self):
        r"""Get list of adaptation functions

        :Call:
            >>> funcs = opts.get_FunctionalAdaptFuncs()
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
        :Outputs:
            *funcs*: :class:`list`\ [:class:`str`]
                List of function names
        :Versions:
            * 2016-04-25 ``@ddalle``: v1.0
            * 2023-05-16 ``@ddalle``: v2.0
        """
        # List of functions
        funcs = []
        # Loop through keys
        for fn in self:
            # Get type
            typ = self.get_FunctionalFuncOpt(fn, "type")
            # Check type
            if typ == "adapt":
                funcs.append(fn)
        # Output
        return funcs

    # Get list of optimization functions
    def get_FunctionalConstraintFuncs(self):
        r"""Get list of constraint functions

        :Call:
            >>> funcs = opts.get_FunctionalConstraintFuncs()
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
        :Outputs:
            *funcs*: :class:`list`\ [:class:`str`]
                List of function names
        :Versions:
            * 2016-04-25 ``@ddalle``: v1.0
            * 2023-05-16 ``@ddalle``: v2.0
        """
        # List of functions
        funcs = []
        # Loop through keys
        for fn in self:
            # Get type
            typ = self.get_FunctionalFuncOpt(fn, "type")
            # Check type
            if typ == "constraint":
                funcs.append(fn)
        # Output
        return funcs

    # Get list of optimization functions
    def get_FunctionalOptFuncs(self):
        r"""Get list of objective functions

        :Call:
            >>> funcs = opts.get_FunctionalOptFuncs()
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
        :Outputs:
            *funcs*: :class:`list`\ [:class:`str`]
                List of function names
        :Versions:
            * 2016-04-25 ``@ddalle``: v1.0
            * 2023-05-16 ``@ddalle``: v2.0
        """
        # List of functions
        funcs = []
        # Loop through keys
        for fn in self:
            # Get type
            typ = self.get_FunctionalFuncOpt(fn, "type")
            # Check type
            if typ == "objective":
                funcs.append(fn)
        # Output
        return funcs




# Class for output functional settings
class FunctionalOpts(OptionsDict):
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

