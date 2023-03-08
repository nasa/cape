r"""
This module contains various modules that are meant to provide
interfaces to individual file formats.  For example, Fortran-style
namelists can be read, altered, and written using
:mod:`cape.filecntl.namelist`.

The template module upon which most or all individual class are built
is :mod:`cape.filecntl.filecntl`, which provides the template class
:class:`cape.filecntl.filecntl.FileCntl`.  That class is also imported
in this top-level :mod:`cape.filecntl` package.

"""

# Import primary class
from .filecntl import FileCntl

