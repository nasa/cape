r"""
:mod:`cape.pycart.options.configopts`: Cart3Dcomponent config options
======================================================================

This module provides options for defining some aspects of the surface
configuration for a Cart3D run. In addition to specifying a template
:file:`Config.xml` file for naming individual components or groups of
components, the ``"Config"`` section also contains user-defined points
and a set of parameters that are written to the Cart3D input file
``input.cntl``.

This is the section in which the user specifies which components to
track forces and/or moments on, and in addition it defines a moment
reference point for each component.

The reference area (``"RefArea"``) and reference length
(``"RefLength"``) parameters are also defined in this section. Cart3D
does not have two separate reference lengths, so there is no
``"RefSpan"`` parameter.

Most parameters are inherited from
:class:`cape.cfdx.options.confiopts.Config`.

The ``"Xslices"``, ``"Yslices"``, and ``"Zslices"`` parameters are used
by Cart3D to define the coordinates at which Tecplot slices are
extracted. These are written to the output file ``cutPlanes.plt`` or
``cutPlanes.dat``. Furthermore, these coordinates can be tied to
user-defined points that may vary for each case in a run matrix.

The following example has a user-defined coordinate for a point that is
on a component called ``"Fin2"``.  Assuming there is a trajectory key
that rotates ``"Fin2"``, the *y*-slice and *z*-slice coordinates will
automatically be updated according to the fin position.

    .. code-block:: javascript

        "Config": {
            // Define a point at the tip of a fin
            "Points": {
                "Fin2": [7.25, -0.53, 0.00]
            },
            // Write a *y* and *z* slice through the tip of the fin
            "Yslices": ["Fin2"],
            "Zslices": ["Fin2"]
        }

Cart3D also contains the capability for point sensors, which record the
state variables at a point, and line sensors, which provide the state at
several points along a line.  The line sensors are particularly useful
for extracting a sonic boom signature.  Both point sensors and line
sensors can also be used as part of the definition of an objective
function for mesh refinement. In addition, the sensors can be used to
extract not only the conditions at the final iteration but also the
history of relevant conditions at each iteration.

:See Also:
    * :mod:`cape.cfdx.options.configopts`
    * :mod:`cape.config`
    * :mod:`cape.pycart.inputCntl`
"""


# Local imports
from ...cfdx.options import configopts
from ...optdict import ARRAY_TYPES, FLOAT_TYPES


# Class for PBS settings
class ConfigOpts(configopts.ConfigOpts):
    # No new attributes
    __slots__ = ()

    # Accepted options
    _optlist = {
        "LineSensors",
        "Xslices",
        "Yslices",
        "Zslices",
    }

    # Aliases
    _optmap = {
        "XSlices": "Xslices",
        "YSlices": "Yslices",
        "ZSlices": "Zslices",
    }

    # Types
    _opttypes = {
        "LineSensors": dict,
        "Xslices": FLOAT_TYPES + (str,),
        "Yslices": FLOAT_TYPES + (str,),
        "Zslices": FLOAT_TYPES + (str,),
    }

    # Descriptions
    _rst_descriptions = {
        "LineSensors": "dictionary of line sensor definitions",
        "PointSensors": "dictionary of point sensor definitions",
        "Xslices": r"*x*\ -slice(s) to export",
        "Yslices": r"*y*\ -slice(s) to export",
        "Zslices": r"*z*\ -slice(s) to export",
    }

    # Get value from list-like entry e.g. "Xslices"
    def _get_pt_comp(self, opt, idir, j=None, **kw):
        # Get value
        v = self.get_opt(opt, j=j, **kw)
        # Check for list
        if not isinstance(v, ARRAY_TYPES):
            return self.get_Point(v)
        # Loop throuch entries
        return [self.get_Point(vj)[idir] for vj in v]

    # Get cut plane extraction coordinate(s)
    def get_Xslices(self, j=None, **kw):
        r"""Return the *x*\ -slice(s) to export

        :Call:
            >>> x = opts.get_Xslices(j=None)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *j*: {``None``} | :class:`int`
                Index of cut plane coordinate to extract
        :Outputs:
            *x*: :class:`float` | :class:`np.ndarray`
                Cut plane coordinate(s)
        :Versions:
            * 2014-10-08 ``@ddalle``: Version 1.0
            * 2022-11-01 ``@ddalle``: Version 2.0; :mod:`optdict`
        """
        return self._get_pt_comp("Xslices", 0, j=j, **kw)

    # Get cut plane extraction coordinate(s)
    def get_Yslices(self, j=None, **kw):
        r"""Return the *y*\ -slice(s) to export

        :Call:
            >>> y = opts.get_Yslices(j=None)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *j*: {``None``} | :class:`int`
                Index of cut plane coordinate to extract
        :Outputs:
            *y*: :class:`float` | :class:`np.ndarray`
                Cut plane coordinate(s)
        :Versions:
            * 2014-10-08 ``@ddalle``: Version 1.0
            * 2022-11-01 ``@ddalle``: Version 2.0; :mod:`optdict`
        """
        return self._get_pt_comp("Yslices", 1, j=j, **kw)

    # Get cut plane extraction coordinate(s)
    def get_Zslices(self, j=None, **kw):
        r"""Return the *z*\ -slice(s) to export

        :Call:
            >>> z = opts.get_Zslices(j=None)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *j*: {``None``} | :class:`int`
                Index of cut plane coordinate to extract
        :Outputs:
            *z*: :class:`float` | :class:`np.ndarray`
                Cut plane coordinate(s)
        :Versions:
            * 2014-10-08 ``@ddalle``: Version 1.0
            * 2022-11-01 ``@ddalle``: Version 2.0; :mod:`optdict`
        """
        return self._get_pt_comp("Zslices", 2, j=j, **kw)


# Properties to add setters and adders
_PROPERTIES = (
    "LineSensors",
    "PointSensors")
_SET_PROPERTIES = (
    "Xslices",
    "Yslices",
    "Zslices")
_ADD_PROPERTIES = _PROPERTIES + _SET_PROPERTIES

ConfigOpts.add_properties(_PROPERTIES)
ConfigOpts.add_setters(_SET_PROPERTIES)
ConfigOpts.add_extenders(_ADD_PROPERTIES)
