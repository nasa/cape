r"""
:mod:`cape.cfdx.caseutils`: Common functions for case-control modules
======================================================================

This module provides tools that are used in more than one case-control
modules. For example, :func:`run_rootdir` is used in more than one
place, and putting it here makes the import graph simpler to understand.

Examples of case-control modules include

*   :mod:`cape.cfdx.casecntl`
*   :mod:`cape.cfdx.archivist`
*   :mod:`cape.cfdx.logger`
"""

# Standard library
import functools
import os


# Decorator for moving directories
def run_rootdir(func):
    r"""Decorator to run a function within a specified folder

    :Call:
        >>> func = run_rootdir(func)
    :Wrapper Signature:
        >>> v = runner.func(*a, **kw)
    :Inputs:
        *func*: :class:`func`
            Name of function
        *runner*: :class:`CaseRunner`
            Controller to run one case of solver
        *a*: :class:`tuple`
            Positional args to :func:`cntl.func`
        *kw*: :class:`dict`
            Keyword args to :func:`cntl.func`
    :Versions:
        * 2020-02-25 ``@ddalle``: v1.1 (:mod:`cape.cntl`)
        * 2023-06-16 ``@ddalle``: v1.0
    """
    # Declare wrapper function to change directory
    @functools.wraps(func)
    def wrapper_func(self, *args, **kwargs):
        # Recall current directory
        fpwd = os.getcwd()
        # Go to specified directory
        os.chdir(self.root_dir)
        # Run the function with exception handling
        try:
            # Attempt to run the function
            v = func(self, *args, **kwargs)
        except Exception:
            # Raise the error
            raise
        except KeyboardInterrupt:
            # Raise the error
            raise
        finally:
            # Go back to original folder
            os.chdir(fpwd)
        # Return function values
        return v
    # Apply the wrapper
    return wrapper_func
