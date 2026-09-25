r"""
The :mod:`cape.pyvul` module contains the interfaces for VULCAN-UNS, the
unstructured-mesh version of VULCAN-CFD. Most tasks using the
CAPE/VULCAN API can be accessed by loading the :mod:`cape.pyvul.cntl`
class and instantiating :class:`cape.pyvul.cntl.Cntl`.

    .. code-block:: python

        import cape.pyvul.cntl

For example the following will read in a global settings instance
assuming that the present working directory contains the correct files.
(If not, various defaults will be used, but it is unlikely that the
resulting setup will be what you intended.)

    .. code-block:: python

        import cape.pyvul.cntl
        cntl = cape.pyvul.cntl.Cntl()

Most of the pyFun submodules essentially contain a single class
definition, which is derived from a similarly named :mod:`cape` module.
For example, :class:`cape.pyfun.databook.DBComp` is subclassed to
:class:`cape.cfdx.databook.DBComp`, but several functions are edited
because their functionality needs customization for FUN3D.  For
example, reading iterative force & moment histories require a
customized method for each solver.

"""


