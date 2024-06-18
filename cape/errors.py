r"""
:mod:`cape.errors`: Customized exceptions for CAPE
=======================================================

This module provides CAPE-specific exceptions and methods to perform
standardized checks throughout :mod:`cape`. One purpose of creating
customized exceptions is to differentiate CAPE bugs and exceptions
raised by CAPE intentionally due to problems with user input or errors
detected in the execution of the integrated CFD solvers.
"""


# Basic error family
class CapeError(Exception):
    r"Base exception class for exceptions intentionally raised by CAPE"
    pass


# Runtime error
class CapeRuntimeError(RuntimeError, CapeError):
    r"""CAPE exception that does not fit other categories"""
    pass
