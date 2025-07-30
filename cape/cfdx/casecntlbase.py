r"""
:mod:`cape.cfdx.casecntlbase`: Abstract base classes for case interface
=======================================================================

This module provides an abstract base class for the
:class:`cape.cfdx.casecntl.CaseRunner` class that controls the CAPE
interface to individual CFD cases. The base class is
:mod:`CaseRunnerBase`.
"""

# Standard library

# Local imports
from .archivist import CaseArchivist
from .options import RunControlOpts


# Meta class to merge _dex_cls
class MetaCaseRunner(type):
    r"""Metaclass for :class:`CaseRunner`

    This metaclass ensures that new subclasses of :class:`CaseRunner`,
    for example in :mod:`cape.pyfun` or :mod:`cape.pylava`, merge their
    :attr:`_dex_cls` class attribute values.
    """
    def __new__(metacls, name: str, bases: tuple, namespace: dict):
        r"""Initialize a new subclass, but combine ``_dex-cls`` attr

        :Call:
            >>> cls = metacls.__new__(name, bases, namespace)
        :Inputs:
            *metacls*: :class:`type`
                The :class:`MetaArgReader` metaclass
            *name*: :class:`str`
                Name of new class being created
            *bases*: :class:`tuple`\ [:class:`type`]
                Bases for new class
            *namespace*: :class:`dict`
                Attributes, methods, etc. for new class
        :Outputs:
            *cls*: :class:`type`
                New class using *metacls* instead of :class:`type`
        """
        # Initialize the new class
        cls = type.__new__(metacls, name, bases, namespace)
        # Return the new class
        return cls

    @classmethod
    def combine_dex_cls(metacls, clsj: type, cls: type):
        r"""Combine the ``_dex_cls`` from a class and one of its bases

        :Call:
            >>> metacls.combine_dex_cls(clsj, cls)
        :Inputs:
            *metacls*: :class:`type`
                The :class:`MetaArgReader` metaclass
            *clsj*: :class:`type`
                Parent class (basis) to combine into *cls*
            *cls*: :class:`type`
                New class in which to save combined attributes
        """
        metacls.combine_dict(clsj, cls, "_dex_cls")

    @classmethod
    def combine_dict(metacls, clsj: type, cls: type, attr: str):
        r"""Combine one dict-like class attribute of *clsj* and *cls*

        :Call:
            >>> metacls.combine_dict(clsj, cls, attr)
        :Inputs:
            *metacls*: :class:`type`
                The :class:`MetaArgReader` metaclass
            *clsj*: :class:`type`
                Parent class (basis) to combine into *cls*
            *cls*: :class:`type`
                New class in which to save combined attributes
            *attr*: :class:`str`
                Name of attribute to combine
        """
        # Get initial properties
        vj = getattr(clsj, attr, None)
        vx = cls.__dict__.get(attr)
        # Check for both
        qj = isinstance(vj, dict)
        qx = isinstance(vx, dict)
        if not (qj and qx):
            return
        # Copy dict from basis
        combined_dict = dict(vj)
        # Combine results
        combined_dict.update(vx)
        # Save combined list
        setattr(cls, attr, combined_dict)


# Definition
class CaseRunnerBase(metaclass=MetaCaseRunner):
    r"""Abstract base class for :class:`cape.cfdx.casecntl.CaseRunner`

    The main purpose for this class is to provide useful type
    annotations for :mod:`cape.cfdx.cntl` without circular imports.

    :Call:
        >>> runner = CaseRunnerBase()
    :Outputs:
        *runner*: :class:`CaseRunner`
            Controller to run one case of solver
    :Class attributes:
        * :attr:`_modname`
        * :attr:`_progname`
        * :attr:`_logprefix`
        * :attr:`_rc_cls`
        * :attr:`_archivist_cls`
        * :attr:`_dex_cls`
    """
    # Maximum number of starts
    _nstart_max = 100

    # Names
    #: :class:`str`, Name of module
    _modname = "cfdx"
    #: :class:`str`, Name of main program controlled
    _progname = "cfdx"
    #: :class:`str`, Prefix for log files
    _logprefix = "run"

    # Specific classes
    #: Class for interpreting *RunControl* options from ``case.json``
    _rc_cls = RunControlOpts
    #: Class for case archiving instances
    _archivist_cls = CaseArchivist
    #: Classes for extracting types of data from case
    _dex_cls = {}

