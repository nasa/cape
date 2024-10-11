r"""
:mod:`cape.cfdx.environopts`: Options for environment variables
------------------------------------------------------------------

This module provides the class :class:`EnvironOpts`, which controls
options for environment variables that users may wish to set at the
beginning of each individual case in a run matrix.
"""

# Local imports
from ...optdict import (
    INT_TYPES,
    WARNMODE_ERROR,
    OptionsDict)


# Environment class
class EnvironOpts(OptionsDict):
    r"""Class for environment variables

    :Call:
        >>> opts = EnvironOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of environment variables
    :Outputs:
        *opts*: :class:`cape.options.runControl.Environ`
            System environment variable options interface
    :Versions:
        * 2015-11-10 ``@ddalle``: v1.0 (Environ)
        * 2022-10-28 ``@ddalle``: v2.0; OptionsDict
    """
    # Class attributes
    _opttypes = {
        "_default_": INT_TYPES + (str,),
    }

    # Get an environment variable by name
    def get_Environ(self, opt, j=0, i=None):
        r"""Get an environment variable setting by name

        :Call:
            >>> val = opts.get_Environ(opt, j=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *opt*: :class:`str`
                Name of the environment variable
            *i*: {``None``} | :class:`int`
                Case index
            *j*: {``0``} | ``None`` | :class:`int`
                Phase number
        :Outputs:
            *val*: :class:`str`
                Value to set the environment variable to
        :Versions:
            * 2015-11-10 ``@ddalle``: v1.0
            * 2022-10-29 ``@ddalle``: v2.0; OptionsDict methods
        """
        # Get value
        val = self.get_opt(opt, j, i=i, mode=WARNMODE_ERROR)
        # Return a string
        return str(val)

    # Set an environment variable by name
    def set_Environ(self, opt, val, j=None):
        r"""Set an environment variable setting by name

        :Call:
            >>> val = opts.get_Environ(opts, j=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *opt*: :class:`str`
                Name of the environment variable
            *val*: :class:`str`
                Value to set the environment variable to
            *j*: {``None``} | :class:`int`
                Phase index
        :Versions:
            * 2015-11-10 ``@ddalle``: v1.0
            * 2022-10-29 ``@ddalle``: v2.0; OptionsDict methods
        """
        self.set_opt(opt, val, j, mode=WARNMODE_ERROR)
