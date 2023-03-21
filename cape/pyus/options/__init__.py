"""

This module provides tools to read, access, modify, and write settings for
:mod:`cape.pyfun`.  The class is based off of the built-in :class:`dict` class, so
its default behavior, such as ``opts['InputInp']`` or
``opts.get('InputInp')`` are also present.  In addition, many convenience
methods, such as ``opts.get_CFDSOLVER_ires()``, are also provided.

In addition, this module controls default values of each pyUS
parameter in a two-step process.  The precedence used to determine what the
value of a given parameter should be is below.

    1. Values directly specified in the input file, :file:`pyUS.json`

    2. Values specified in the default control file,
       ``$PYFUN/settings/pyUS.default.json``

    3. Hard-coded defaults from this module

"""

# Local imports
from . import util
from ...cfdx import options
from .DataBook    import DataBook
from .Report      import Report
from .runControl  import RunControl
from .inputInp    import InputInpOpts
from .Mesh        import Mesh
from .Config      import Config


# Class definition
class Options(options.Options):
    r"""Options interface for :mod:`cape.pyfun`

    :Call:
        >>> opts = Options(fname=None, **kw)
    :Inputs:
        *fname*: :class:`str`
            File to be read as a JSON file with comments
        *kw*: :class:`dict`
            Dictionary to be transformed into :class:`pyCart.options.Options`
    :Versions:
        * 2014-07-28 ``@ddalle``: v1.0
        * 2023-03-20 ``@ddalle``: v2.0; use :mod:`cape.optdict`
    """
   # ================
   # Class attributes
   # ================
   # <
    # Additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "InputInp",
    }

    # Aliases
    _optmap = {
        "input.inp": "InputInp",
    }

    # Types
    _opttypes = {
        "InputInp": str,
    }

    # Descripitons
    _rst_descriptions = {
        "InputInp": "name of ``input.inp`` template file",
    }

    # New or replaced sections
    _sec_cls = {
        "US3D": InputInpOpts,
    }
   # >

   # =============
   # Configuration
   # =============
   # <
    # Initialization hook
    def init_post(self):
        r"""Initialization hook for :class:`Options`

        :Call:
            >>> opts.init_post()
        :Inputs:
            *opts*: :class:`Options`
                Options interface
        :Versions:
            * 2022-10-23 ``@ddalle``: Version 1.0
        """
        # Read the defaults
        defs = util.getPyUSDefaults()
        # Apply the defaults.
        self = util.applyDefaults(self, defs)
        # Add extra folders to path.
        self.AddPythonPath()
   # >


# Add properties
Options.add_properties(
    (
        "InputInp",
    ))
# Add methods from subsections
Options.promote_sections()
