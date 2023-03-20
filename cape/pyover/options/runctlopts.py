r"""
This is the ``"RunControl"`` options module specific to
:mod:`cape.pyover`. It is based primarily on

    :mod:`cape.cfdx.options.runctlopts`

and provides the class

    :class:`cape.pyover.options.runctlopts.RunControlOpts`

Among the OVERFLOW-specific options interfaced here are the options for
the OVERFLOW executables, defined in the ``"overrun"`` section
"""

# Local imports
from .archiveopts import ArchiveOpts
from ...cfdx.options import runctlopts
from ...optdict import INT_TYPES, OptionsDict


# Class for OVERFLOW command-line interface
class OverrunOpts(OptionsDict):
    r"""Class for ``overrun`` settings

    :Call:
        >>> opts = OverrunOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Raw options
    :Outputs:
        *opts*: :class:`OverrunOpts`
            CLI options for ``overrun`` or ``overrunmpi``
    :Versions:
        * 2016-02-01 ``@ddalle``: Version 1.0 (``flowCart``)
        * 2022-11-04 ``@ddalle``: Version 2.0; use ``optdict``
    """
    # Additional attributes
    __slots__ = ()

    # Accepted options
    _optlist = {
        "args",
        "aux",
        "cmd",
        "nthreads",
    }

    # Option types
    _opttypes = {
        "args": str,
        "aux": str,
        "cmd": str,
        "nthreads": INT_TYPES,
    }

    # Defaults
    _rc = {
        "args": "",
        "aux": '"-v pcachem -- dplace -s1\"',
        "cmd": "overrunmpi",
    }

    # Descriptions
    _rst_descriptions = {
        "args": "string of extra args to ``overrun``",
        "aux": "auxiliary CLI args to ``overrun``",
        "cmd": "name of OVERFLOW executable to use",
        "nthreads": "number of OpenMP threads",
    }

    # Get dictionary of options
    def get_overrun_kw(self, j=None, **kw):
        r"""Get other inputs to ``overrun``

        :Call:
            >>> kw = opts.get_overrun_kw(i=None)
        :Inputs:
            *opts*: :class:`cape.pyover.options.Options`
                Options interface
            *i*: :class:`int`
                Phase number
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of additional arguments to ``overrun``
        :Versions:
            * 2016-02-01 ``@ddalle``: Version 1.0
            * 2022-11-03 ``@ddalle``: Version 2.0; :class:`OptionsDict`
        """
        # Intiialize output
        kw = {}
        # Loop through output
        for k in self:
            # Check for named keys
            if k in ("args", "aux", "cmd"):
                continue
            # Otherwise, append the key, but select the phase
            kw[k] = self.get_opt(k, j=j, **kw)
        # Output
        return kw


# Add properties
OverrunOpts.add_properties(OverrunOpts._optlist, prefix="overrun_")


# Class for Report settings
class RunControlOpts(runctlopts.RunControlOpts):
    # Additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "Prefix",
        "overrun",
    }

    # Types
    _opttypes = {
        "Prefix": str,
    }

    # Defaults
    _rc = {
        "MPI": True,
        "Prefix": "run",
    }

    # Descriptions
    _rst_descriptions = {
        "Prefix": "project root name, or file prefix",
    }

    # Additional or renewed sections
    _sec_cls = {
        "Archive": ArchiveOpts,
        "overrun": OverrunOpts,
    }


# Add properties
RunControlOpts.add_properties(RunControlOpts._optlist)
# Promote sections
RunControlOpts.promote_sections()
