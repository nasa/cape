r"""
:mod:`cape.cfdx.options.slurmopts`: SLURM script options
==========================================================

This portion of the options is universal, and so it is only encoded in
the :mod:`cape` module. The :mod:`cape.pycart` module, for example, does
not have a modified version. It contains options for specifying which
architecture to use, how many nodes to request, etc.
"""


# Import options-specific utilities
from ...optdict import OptionsDict, ARRAY_TYPES, INT_TYPES


# Class for Slurm settings
class SlurmOpts(OptionsDict):
    r"""Dictionary-based options for Slurm jobs submitted via ``sbatch``

    :Call:
        >>> opts = SlurmOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of Slurm options
    :Outputs:
        *opts*: :class:`SlurmOpts`
            Slurm options interface
    :Versions:
        * 2022-10-31 ``@ddalle``: v1.0
    """
   # --- Class attributes ---
    # No extra attributes
    __slots__ = ()

    # Identifiers
    _name = "options for Slurm job control"

    # Options list
    _optlist = {
        "A",
        "C",
        "N",
        "b",
        "gid",
        "n",
        "other",
        "p",
        "shell",
        "time",
    }

    # Aliases
    _optmap = {
        "account": "A",
        "begin": "b",
        "constraint": "C",
        "q": "p",
        "t": "time",
    }

    # Types
    _opttypes = {
        "A": str,
        "C": str,
        "N": INT_TYPES,
        "gid": str,
        "n": INT_TYPES,
        "other": dict,
        "p": str,
        "shell": str,
        "time": str,
    }

    # Defaults
    _rc = {
        "N": 1,
        "time": "8:00:00",
        "n": 40,
        "p": "normal",
        "shell": "/bin/bash",
    }

    # Descriptions
    _rst_descriptions = {
        "A": "Slurm job account name",
        "C": "constraint(s) on Slurm job",
        "N": "number of Slurm nodes",
        "b": "constraints on when to acllocate job",
        "gid": "Slurm job group ID",
        "n": "number of CPUs per node",
        "other": "dict of additional Slurm options using ``=`` char",
        "p": "Slurm queue name",
        "shell": "Slurm job shell name",
        "time": "max Slurm job wall time",
    }

    # Get the number of unique Slurm jobs.
    def get_nSlurm(self) -> int:
        r"""Return the maximum number of unique Slurm inputs

        For example, if a case is set up to be run in two parts, and the
        first phase needs only one node (*N=1*) while the second
        phase needs 10 nodes (*N=10*), then the input file should
        have  ``"N": [1, 10]``, and the output of this function
        will be ``2``.

        :Call:
            >>> n = opts.get_nSlurm()
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
        :Outputs:
            *n*: :class:`int`
                Number of unique Slurm scripts
        :Versions:
            * 2014-12-01 ``@ddalle``: Version 1.0
        """
        # Initialize the number of jobs.
        n = 1
        # Loop the keys.
        for v in self.values():
            # Test if it's a list
            if isinstance(v, ARRAY_TYPES):
                # Update the length.
                n = max(n, len(v))
        # Output
        return n


# Identical subclasses
class BatchSlurmOpts(SlurmOpts):
    __slots__ = ()
    _name = "options for Slurm job control for batch commands"


class PostSlurmOpts(SlurmOpts):
    __slots__ = ()


# Add all properties
SlurmOpts.add_properties(SlurmOpts._optlist, prefix="Slurm_")
BatchSlurmOpts.add_properties(SlurmOpts._optlist, prefix="BatchSlurm_")
PostSlurmOpts.add_properties(SlurmOpts._optlist, prefix="PostSlurm_")

