r"""
:mod:`cape.cfdx.casecntlbase`: Abstract base classes for case interface
=======================================================================

This module provides an abstract base class for the
:class:`cape.cfdx.casecntl.CaseRunner` class that controls the CAPE
interface to individual CFD cases. The base class is
:mod:`CaseRunnerBase`.
"""

# Standard library
from abc import ABC, abstractmethod


# Constants:
# Name of file that marks a case as currently running
RUNNING_FILE = "RUNNING"
# Name of file marking a case as in a failure status
FAIL_FILE = "FAIL"
# Name of file to stop at end of phase
STOP_PHASE_FILE = "CAPE-STOP-PHASE"
# Case settings
RC_FILE = "case.json"
# Run matrix conditions
CONDITIONS_FILE = "conditions.json"


# Definition
class CaseRunnerBase(ABC):
    pass
