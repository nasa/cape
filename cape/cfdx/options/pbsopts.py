r"""
:mod:`cape.cfdx.options.pbsopts`: PBS script options
=====================================================

This portion of the options is universal, and so it is only encoded in
the :mod:`cape` module. The :mod:`cape.pycart` module, for example, does
not have a modified version. It contains options for specifying which
architecture to use, how many nodes to request, etc.

"""


# Import options-specific utilities
from ...optdict import OptionsDict, ARRAY_TYPES, INT_TYPES


# Class for PBS settings
class PBSOpts(OptionsDict):
    r"""Options class for PBS jobs

    :Call:
        >>> opts = PBS(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of PBS options
    :Outputs:
        *opts*: :class:`cape.options.pbs.PBS`
            PBS options interface
    :Versions:
        * 2014-12-01 ``@ddalle``: v1.0
        * 2022-10-13 ``@ddalle``: v2.0; :class:`OptionsDict`
    """
   # --- Class attributes ---
    # No extra attributes
    __slots__ = tuple()

    # Identifiers
    _name = "options for PBS job control"

    # Allowed parameters
    _optlist = {
        "A",
        "S",
        "W",
        "aoe",
        "e",
        "j",
        "k",
        "mem",
        "model",
        "mpiprocs",
        "ncpus",
        "ngpus",
        "o",
        "ompthreads",
        "p",
        "place",
        "q",
        "r",
        "select",
        "site_needed",
        "walltime"
    }

    # Aliases
    _optmap = {
        "site-needed": "site_needed",
    }

    # Types
    _opttypes = {
        "S": str,
        "W": str,
        "j": str,
        "k": str,
        "r": str,
        "aoe": str,
        "model": str,
        "mem": INT_TYPES,
        "mpiprocs": INT_TYPES,
        "ncpus": INT_TYPES,
        "ngpus": INT_TYPES,
        "ompthreads": INT_TYPES,
        "place": str,
        "q": str,
        "select": INT_TYPES,
        "site_needed": str,
        "walltime": str,
    }

    # Possible values
    _optvals = {
        "r": ("n", "y"),
        "place": ("shared", "excl"),
    }
    
    # List-like control
    _optlistdepth = {
        "site_needed": 1,
    }

    # Defaults
    _rc = {
        "S": "/bin/bash",
        "W": "",
        "j": "oe",
        "place": "excl",
        "q": "normal",
        "r": "n",
        "select": 1,
        "walltime": "8:00:00",
    }

    # Descriptions
    _rst_descriptions = {
        "A": "account name(s) or number(s)",
        "M": "email recipient list",
        "S": "shell to execute PBS job",
        "W": "PBS *W* setting, usually for setting group",
        "aoe": "architecture operating environment",
        "e": "explicit STDERR file name",
        "j": "PBS 'join' setting",
        "k": "PBS 'keep' setting; live updates to STD(OUT|ERR)",
        "m": "email notification setting",
        "mem": "Requested cpu memory (on gpu node)",
        "model": "model type/architecture",
        "mpiprocs": "number of MPI processes per node",
        "ncpus": "number of cores (roughly CPUs) per node",
        "ngpus": "number of gpus per node",
        "o": "explicit STDOUT file name",
        "ompthreads": "number of OMP threads",
        "p": "PBS priority",
        "place": "Share or exclusive access to requested gpu nodes",
        "q": "PBS queue name",
        "r": "rerun-able setting",
        "select": "number of nodes",
        "site_needed": "list of manually requested hard drives to mount",
        "walltime": "maximum job wall time",
    }

    # Get the number of unique PBS jobs.
    def get_nPBS(self) -> int:
        r"""Return the maximum number of unique PBS inputs

        For example, if a case is set up to be run in two parts, and the
        first phase needs only one node (*select=1*) while the second
        phase needs 10 nodes (*select=10*), then the input file should
        have ``"select": [1, 10]``, and the output of this function will
        be ``2``.

        :Call:
            >>> n = opts.get_nPBS()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *n*: :class:`int`
                Number of unique PBS scripts
        :Versions:
            * 2014-12-01 ``@ddalle``: First version
        """
        # Initialize the number of jobs.
        n = 1
        # Loop the keys.
        for k, v in self.items():
            # Skip if it's got a list depth
            if self.getx_cls_key("_optlistdepth", k):
                continue
            # Try to use getter so map is applied
            func = getattr(self, f"get_PBS_{k}", None)
            if func:
                v = func()
            # Test if it's a list
            if isinstance(v, ARRAY_TYPES):
                # Update the length.
                n = max(n, len(v))
        # Output
        return n


# Identical subclasses
class BatchPBSOpts(PBSOpts):
    __slots__ = ()
    _name = "options for PBS job control for batch commands"


class PostPBSOpts(PBSOpts):
    __slots__ = ()


# Add all properties
PBSOpts.add_properties(PBSOpts._optlist, prefix="PBS_")
BatchPBSOpts.add_properties(PBSOpts._optlist, prefix="BatchPBS_")
PostPBSOpts.add_properties(PBSOpts._optlist, prefix="PostPBS_")
