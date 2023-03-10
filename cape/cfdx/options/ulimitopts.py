"""
:mod:`cape.cfdx.options.ulimitopts`: System resource options
================================================================

This module provides a class to access options relating to system
resources. Specifically, this pertains to options usually set by
``ulimit`` from the command line.

The class provided in this module, :class:`ULimitOpts`, is
loaded in the ``"RunControl"`` section of the JSON file or main options
interface.
"""

# Ipmort options-specific utilities
from ...optdict import INT_TYPES, OptionsDict, WARNMODE_ERROR


# Resource limits class
class ULimitOpts(OptionsDict):
    r"""Class for resource limit options

    :Call:
        >>> opts = ULimitOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of system resource options
    :Outputs:
        *opts*: :class:`ULimitOpts`
            System resource options interface
    :Versions:
        * 2015-11-10 ``@ddalle``: v1.0 (``ulimit``)
        * 2022-10-31 ``@ddalle``: v2.0; use :class:`OptionsDict`
    """
    # List of recognized options
    _optlist = {
        "c",
        "d",
        "e",
        "f",
        "i",
        "l",
        "m",
        "n",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "x",
    }

    # Aliases
    _optmap = {
        "core_file_size": "c",
        "data_segment": "d",
        "file_locks": "x",
        "file_size": "f",
        "locked_memory": "l",
        "max_processes": "u",
        "message_queue_size": "q",
        "open_files": "n",
        "pending_signals": "i",
        "pipe_size": "p",
        "processes": "u",
        "real_time_priority": "r",
        "scheduling_priority": "e",
        "set_size": "m",
        "stack_size": "s",
        "time_limit": "t",
        "user_processes": "u",
        "virtual_memory": "v",
    }

    # Types (all int or "unlimited")
    _opttypes = {
        "_default_": INT_TYPES + (str,),
    }

    # Defaults
    _rc = {
        "c": 0,
        "d": "unlimited",
        "e": 0,
        "f": "unlimited",
        "i": 127556,
        "l": 64,
        "m": "unlimited",
        "n": 1024,
        "p": 8,
        "q": 819200,
        "r": 0,
        "s": 4194304,
        "t": "unlimited",
        "u": 127812,
        "v": "unlimited",
        "x": "unlimited",
    }

    # Descriptions
    _rst_descriptions = {
        "T": "max number of threads, ``ulimit -T``",
        "b": "max socket buffer size, ``ulimit -b``",
        "c": "core file size limit, ``ulimit -c``",
        "d": "process data segment limit, ``ulimit -d``",
        "e": "max scheduling priority, ``ulimit -e``",
        "f": "max size of files written by shell, ``ulimit -f``",
        "i": "max number of pending signals, ``ulimit -i``",
        "l": "max size that may be locked into memory, ``ulimit -l``",
        "m": "max resident set size, ``ulimit -m``",
        "n": "max number of open files, ``ulimit -n``",
        "p": "pipe size in 512-byte blocks, ``ulimit -p``",
        "q": "max bytes in POSIX message queues, ``ulimit -q``",
        "r": "max real-time scheduling priority, ``ulimit -r``",
        "s": "stack size limit, ``ulimit -s``",
        "t": "max amount of cpu time in s, ``ulimit -t``",
        "u": "max number of procs avail to one user, ``ulimit -u``",
        "v": "max virtual memory avail to shell, ``ulimit -v``",
        "x": "max number of file locks, ``ulimit -x``",
    }

    # Get a ulimit setting
    def get_ulimit(self, u, j=0, i=None):
        r"""Get a resource limit (``ulimit``) setting by name

        :Call:
            >>> l = opts.get_ulimit(u, i=0)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *u*: :class:`str`
                Name of the ``ulimit`` flag
            *j*: ``None`` | {``0``} | :class:`int`
                Phase index
            *i*: {``None``} | :class:`int`
                Case number
        :Outputs:
            *l*: :class:`int`
                Value of the resource limit
        :Versions:
            * 2015-11-10 ``@ddalle``: v1.0
            * 2022-10-30 ``@ddalle``: v2.0; use :mod:`optdict`
        """
        # Get setting with strict warings
        return self.get_opt(u, j=j, i=i, mode=WARNMODE_ERROR)

    # Set a ulimit setting
    def set_ulimit(self, u, l=None, j=None):
        r"""Set a resource limit (``ulimit``) by name and value

        :Call:
            >>> opts.set_ulimit(u, l=None, i=None)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *u*: :class:`str`
                Name of the ``ulimit`` flag
            *l*: :class:`int`
                Value of the limit
            *j*: {``None``} | :class:`int`
                Phase index
        :Versions:
            * 2015-11-10 ``@ddalle``: v1.0
            * 2022-10-30 ``@ddalle``: v2.0; use :mod:`optdict`
            * 2023-03-10 ``@ddalle``: v2.1; remove *i* input
        """
        # Get default if appropriate
        if l is None:
            l = self.get_opt_default(u)
        # Set option
        self.set_opt(u, l, j=j)


# Add properties
ULimitOpts.add_properties(ULimitOpts._optlist)
ULimitOpts.add_properties(ULimitOpts._optmap)
