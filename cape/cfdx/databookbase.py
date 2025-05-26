r"""
:class:`cape.cfdx.databookbase`: Abstract base classes for DataBook
=====================================================================

This module provides abstract base classes for the "DataBook" capability
in CAPE, which collects post-processed data from CFD cases.

The main class is :class:`DataBookBase`, which is the base class for
:class:`cape.cfdx.databook.DataBook`.

:See also:
    * :mod:`cape.cfdx.databook`
"""


# Standard library
from abc import ABCMeta, abstractmethod
from typing import Optional

# Local imports


# Aerodynamic history class
class DataBookBase(dict, metaclass=ABCMeta):
    r"""Interface to the data book for a given CFD run matrix

    :Call:
        >>> DB = DataBook(cntl, **kw)
    :Inputs:
        *cntl*: :class:`Cntl`
            CAPE control class instance
        *RootDir*: :class:`str`
            Root directory, defaults to ``os.getcwd()``
        *targ*: {``None``} | :class:`str`
            Option to read duplicate data book as a target named *targ*
    :Outputs:
        *DB*: :class:`cape.cfdx.databook.DataBook`
            Instance of the Cape data book class
        *DB.x*: :class:`cape.runmatrix.RunMatrix`
            Run matrix of rows saved in the data book
        *DB[comp]*: :class:`cape.cfdx.databook.DBComp`
            Component data book for component *comp*
        *DB.Components*: :class:`list`\ [:class:`str`]
            List of force/moment components
        *DB.Targets*: :class:`dict`
            Dictionary of :class:`DataBookTarget` target data books
    """
    # Initialization method
    @abstractmethod
    def __init__(
            self,
            cntl,
            RootDir: Optional[str] = None,
            targ: Optional[str] = None, **kw):
        pass

    # Command-line representation
    def __repr__(self):
        r"""Representation method

        :Versions:
            * 2014-12-22 ``@ddalle``: v1.0
        """
        # Get class
        cls = self.__class__
        clsname = cls.__name__
        # Get module
        modname = cls.__module__
        # Get base module
        modbase = modname.split('.')[1]
        # Initialize string
        return f"<{modbase}.{clsname}, ncomp={len(self.Components)}>"

    # String conversion
    __str__ = __repr__
