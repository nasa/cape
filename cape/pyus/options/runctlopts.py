r"""

Options interface for aspects of running a case of US3D. The settings
are read from the ``"RunControl"`` of a JSON file, and the contents of
this section are written to ``case.json`` within each run folder.

The methods of :class:`cape.cfdx.options.runctlopts.RunControlOpts` are
also present. These control options such as whether to submit as a PBS
job, whether or not to use MPI, etc.

:See Also:
    * :mod:`cape.cfdx.options.runctlopts`
    * :mod:`cape.cfdx.options.ulimitopts`
    * :mod:`cape.pyus.options.archiveopts`
"""

# Local imports
from ...cfdx.options import runctlopts
from ...cfdx.options.util import ExecOpts
from .archiveopts import ArchiveOpts


# Class for inputs to the US3D executable
class US3DRunOpts(ExecOpts):
    # Additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "gas",
        "grid",
        "input",
    }

    # Types
    _opttypes = {
        "gas": str,
        "grid": str,
        "input": str,
    }

    # Defaults
    _rc = {
        "grid": "grid.h5",
        "input": "input.inp",
    }

    # Descriptions
    _rst_descriptions = {
        "gas": "name of gas model to use for US3D",
        "grid": "name of US3D grid file",
        "input": "name of US3D input control file",
    }


# Add properties
US3DRunOpts.add_properties(US3DRunOpts._optlist)


# Class for inputs to the ``us3d-prepar`` executable
class US3DPreparOpts(ExecOpts):
    # Additional attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "conn",
        "grid",
    }

    # Types
    _opttypes = {
        "conn": str,
        "grid": str,
    }

    # Defaults
    _rc = {
        "conn": "conn.h5",
        "grid": "pyus.cas",
    }

    # Descriptions
    _rst_descriptions = {
        "conn": "name of connectivity file made by ``us3d-prepar``",
        "grid": "name of input grid to ``us3d-prepar``",
    }


# Add properties
US3DPreparOpts.add_properties(US3DPreparOpts._optlist)


# Class for Report settings
class RunControlOpts(runctlopts.RunControlOpts):
    # No additional atributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "us3d",
        "us3d-prepar",
    }

    # Section map
    _sec_cls = {
        "Archive": ArchiveOpts,
        "us3d": US3DRunOpts,
        "us3d-prepar": US3DPreparOpts,
    }


# Promote subections
RunControlOpts.promote_sections()

