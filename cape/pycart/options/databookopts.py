r"""
:mod:`cape.pycart.options.databookopts`: Pycart databook definition options
===========================================================================

This module provides database options specific to pyCart/Cart3D.  The
vast majority of database options are common to all solvers and are thus
inherited from :class:`cape.cfdx.options.DataBook.DataBook`.

For force and/or moment components (``"Type": "FM"`` or
``"Type": "Force"``), each component requested in the databook must also
be listed appropriately as a force and/or moment in the ``input.cntl``
file. These can be written manually to the template ``input.cntl`` file
or controlled via the :class:`cape.pycart.options.Config.Config` class.

The pyCart version of this module alters the default list of columns for
inclusion in the data book. For point sensors this includes a column
called *RefLev* that specifies the number of refinements of the mesh at
the location of that point sensor (which my vary from case to case
depending on mesh adaptation options). Point sensors also save the
values of state variables at that point, which for Cart3D are the
following columns.

    ==============  ==============================================
    Column          Description
    ==============  ==============================================
    *X*             *x*-coordinate of the point
    *Y*             *y*-coordinate of the point
    *Z*             *z*-coordinate of the point
    *Cp*            Pressure coefficient
    *dp*            :math:`(p-p_{\\infty})/(\\gamma p_{\\infty})`
    *rho*           Density over freestream density
    *u*             *x*-velocity over freestream sound speed
    *v*             *y*-velocity over freestream sound speed
    *w*             *z*-velocity over freestream sound speed
    *P*             Pressure over gamma times freestream pressure
    ==============  ==============================================

The full description of the JSON options can be found in a
:ref:`CAPE section <cape-json-DataBook>` and a
:ref:`pyCart section <pycart-json-DataBook>`.

:See Also:
    * :mod:`cape.cfdx.options.DataBook`
    * :mod:`cape.pycart.options.Config.Config`
"""

# Template module
from ...cfdx.options import databookopts


# Class for "IterPoint" components
class DBIterPointOpts(databookopts.DBIterPointOpts):
    # No attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "Points",
    }

    # Option types
    _opttypes = {
        "Points": str,
    }

    # List depth
    _optlistdepth = {
        "Points": 1,
    }

    # Defaults
    _rc = {
        "Cols": ["x", "y", "z", "cp", "dp", "rho", "u", "v", "w", "p"],
        "IntCols": ["RefLev"],
    }

    # Descriptions
    _rst_descriptions = {
        "Points": "list of individual point sensors",
    }


# Class for "IterPoint" components
class DBIterLineOpts(DBIterPointOpts):
    # No attributes
    __slots__ = ()

    # Additional options
    _optlist = {
        "Points",
    }

    # Option types
    _opttypes = {
        "Points": str,
    }

    # List depth
    _optlistdepth = {
        "Points": 1,
    }


# Class for DataBook options
class DataBookOpts(databookopts.DataBookOpts):
    r"""Dictionary-based interface for DataBook specifications
    
    :Versions: 
        * 2023-03-16 ``@ddalle``: v1.0
    """
    # No attributes
    __slots__ = ()

    # Section map
    _sec_cls_optmap = {
        "FM": databookopts.DBFMOpts,
        "PointSensor": DBIterPointOpts,
        "LineLoad": databookopts.DBLineLoadOpts,
        "LineSensor": DBIterLineOpts,
        "PyFunc": databookopts.DBPyFuncOpts,
        "TriqFM": databookopts.DBTriqFMOpts,
        "TriqPoint": databookopts.DBTriqPointOpts,
    }

    # Allowed values
    _optvals = {
        "Type": {
            "FM",
            "PointSensor",
            "LineSensor",
            "LineLoad",
            "PyFunc",
            "TriqFM",
            "TriqPoint",
        },
    }

