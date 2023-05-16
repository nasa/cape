"""Interface for Cart3D adaptation settings"""


# Local imports
from ...optdict import OptionsDict, FLOAT_TYPES, INT_TYPES


# Class for individual function opts
class FunctionalFuncOpts(OptionsDict):
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
        "coeffs": ["entire"],
        "type": "adapt",
    }


# Collection of functions
class FunctionalFuncCollectionOpts(OptionsDict):
    # Attributes
    __slots__ = ()

    # Types
    _opttypes = {
        "_default_": dict,
    }

    # Sections
    _sec_cls_opt = "parent"
    _sec_cls_optmap = {
        "_default_": FunctionalFuncOpts,
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


# Coefficient
class FunctionalCoeffOpts(OptionsDict):
    # Attributes
    __slots__ = ()

    # Options
    _optlist = {
        "compID",
        "parent",
        "power",
        "target",
        "weight",
    }

    # Types
    _opttypes = {
        "compID": INT_TYPES + (str,),
        "target": FLOAT_TYPES,
        "parent": str,
        "power": FLOAT_TYPES + INT_TYPES,
        "weight": FLOAT_TYPES,
    }

    # Defaults
    _rc = {
        "power": 1.0,
        "target": 0.0,
        "weight": 1.0,
    }


# Class for collection of coefficient defns
class FunctionalCoeffCollectionOpts(OptionsDict):
    # Attributes
    __slots__ = ()

    # Types
    _opttypes = {
        "_default_": dict,
    }

    # Sections
    _sec_cls_opt = "Parent"
    _sec_cls_optmap = {
        "_default_": FunctionalFuncOpts,
    }

    # Get option for a function
    def get_FunctionalCoeffOpt(self, coeff: str, opt: str, j=None, **kw):
        r"""Get option for a specific functional coefficient

        :Call:
            >>> v = opts.get_FunctionalCoeffOpt(coeff, opt, j=None, **kw)
        :Inputs:
            *opts*: :class:`OptionsDict`
                Options interface
            *coeff*: :class:`str`
                Name of coefficient
            *opt*: :class:`str`
                Name of functional option
        :Outputs:
            *v*: :class:`object`
                Value of ``opts[fn][opt]`` or as appropriate
        :Versions:
            * 2023-05-16 ``@ddalle``: v1.0
        """
        return self.get_subopt(coeff, opt, j=j, **kw)


# Class for output functional settings
class FunctionalOpts(OptionsDict):
    # Attributes
    __slots__ = ()

    # Available options
    _optlist = {
        "Coeffs",
        "Functions",
    }

    # Types
    _opttypes = {
        "Coeffs": dict,
        "Functions": dict,
    }

    # Sections
    _sec_cls = {
        "Coeffs": FunctionalCoeffCollectionOpts,
        "Functions": FunctionalFuncCollectionOpts,
    }

    # Get adaptive function coefficients
    def get_FunctionalAdaptCoeffs(self):
        r"""Get list of coefficients in any adaptation function

        :Call:
            >>> coeffs = opts.get_AdaptCoeffs()
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
        :Ouutputs:
            *coeffs*: :class:`list`\ [:class:`str`]
                List of coefficients in the adaptive function
        :Versions:
            * 2016-04-26 ``@ddalle``: v1.0
        """
        # Initialize coefficient list
        coeffs = set()
        # Loop through adaptive functions
        for fn in self.get_FunctionalAdaptFuncs():
            # Get list of coefficients
            fncoeffs = self.get_FunctionalFuncOpt(fn, "coeffs")
            # Combine
            if fncoeffs is not None:
                coeffs.update(fncoeffs)
        # Output
        return coeffs


# Promote subsections
FunctionalOpts.promote_sections()

