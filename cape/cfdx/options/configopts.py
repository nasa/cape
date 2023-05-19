r"""
:mod:`cape.cfdx.options.configopts`: Options for CFD component config
=====================================================================

This module provides options for CFD component configurations,
including:

    * Pointing to a surface component hierarchy file
    * Telling the solver which components to report on each iteration
    * Defining moment reference points
    * Defining other points used by other options

The :func:`ConfigOpts.get_ConfigFile` typically points to an external
file that associates names with each numbered surface and provides a
tree of parent/child relationships between components and groups of
components.

Another aspect is to define ``"Points"`` by name. This allows the moment
reference point for a configuration and not have to repeat the
coordinates over and over again. Furthermore, named points can be
transformed by other functions automatically. For example, a moment
reference point can be translated and rotated along with a component, or
a set of four points defining a right-handed coordinate system can be
kept attached to a certain component.
"""

# Standard library
import copy

# Local imports
from ...optdict import ARRAY_TYPES, FLOAT_TYPES, OptionsDict


# Class for PBS settings
class ConfigOpts(OptionsDict):
    r"""Options class for CFD configuration

    It is primarily used for naming surface components, grouping them,
    defining moment reference points, defining other points, and
    requesting components of interest.

    :Call:
        >>> opts = Config(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of configuration
    :Outputs:
        *opts*: :class:`ConfigOpts`
            CFD component configuration option interface
    :Versions:
        * 2014-09-29 ``@ddalle``: Version 1.0 (``Config``)
        * 2022-11-01 ``@ddalle``: Version 2.0; :class:`OptDict`
    """
    # Additional attributes
    __slots__ = ("_Points",)

    # Accepted options
    _optlist = {
        "Components",
        "ConfigFile",
        "Points",
        "RefArea",
        "RefLength",
        "RefPoint",
        "RefSpan",
    }

    # Aliases
    _optmap = {
        "File": "ConfigFile",
    }

    # Types
    _opttypes = {
        "Components": str,
        "ConfigFile": str,
        "Points": dict,
        "RefArea": (dict,) + FLOAT_TYPES,
        "RefLength": (dict,) + FLOAT_TYPES,
        "RefPoint": (dict, str) + FLOAT_TYPES,
        "RefSpan": (dict,) + FLOAT_TYPES,
    }

    # List-like entries
    _optlistdepth = {
        "Components": 1,
    }

    # Defaults
    _rc = {
        "Components": [],
        "ConfigFile": "Config.xml",
        "Points": {},
        "RefArea": 1.0,
        "RefLength": 1.0,
        "RefPoint": [0.0, 0.0, 0.0],
    }

    # Descriptions
    _rst_descriptions = {
        "Components": "list of components to request from solver",
        "ConfigFile": "configuration file name",
        "Points": "dictionary of reference point locations",
        "RefArea": "reference area [for a component]",
    }

    # Initialization method
    def init_post(self):
        r"""Initialization hook for :class:`ConfigOpts`

        :Call:
            >>> opts.init_post()
        :Inputs:
            *opts*: :class:`ConfigOpts`
                CFD component configuration option interface
        :Versions:
            * 2022-11-01 ``@ddalle``: Version 1.0
        """
        # Store a copy of point locations
        self._Points = copy.deepcopy(self.get_opt('Points'))

    # Reset the points
    def reset_Points(self):
        r"""Reset all points to original locations

        :Call:
            >>> opts.reset_Points()
        :Inptus:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
        :Versions:
            * 2016-04-18 ``@ddalle``: Version 1.0
            * 2022-11-01 ``@ddalle``: Version 1.1; :func:`deepcopy`
        """
        self['Points'] = copy.deepcopy(self._Points)

    # Tool for reference area
    def get_refcol(self, col: str, comp=None):
        r"""Get value of a dictionary option like ``"RefArea"``

        :Call:
            >>> vref = opts.get_refcol(col, comp=None)
        :Inputs:
            *opts*: :class:`ConfigOpts`
                CFD component configuration option interface
            *col*: :class:`str`
                Name of ``"Config"`` option
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *vref*: :class:`float`
                Reference quantity [for *comp*]
        :Versions:
            * 2022-11-01 ``@ddalle``: Version 1.0
            * 2023-05-19 ``@ddalle``: v1.1; mod for ``OptionsDict``
        """
        # Get scalar or dictionary
        vmap = self.get_opt(col)
        # Check type
        if isinstance(vmap, dict):
            # Check if *comp* is present
            if (comp in vmap):
                # Return the specific component.
                vref = vmap[comp]
            else:
                # Get overall default
                vdef = self.get_opt_default(opt)
                # Check for JSON-specified default
                vref = vmap.get("_default_", vdef)
        else:
            # It's just a number.
            vref = vmap
        # Output
        return vref

    def set_refcol(self, col, v, comp=None):
        r"""Set value of a dictionary option like ``"RefArea"``

        :Call:
            >>> opts.set_refcol(col, v, comp=None)
        :Inputs:
            *opts*: :class:`ConfigOpts`
                CFD component configuration option interface
            *col*: :class:`str`
                Name of ``"Config"`` option
            *v*: :class:`object`
                Reference quantity [for *comp*]
            *comp*: {``None``} | :class:`str`
                Name of component
        :Versions:
            * 2022-11-01 ``@ddalle``: Version 1.0
        """
        # Get map, ensuring dict
        vmap = self.setdefault(col, {})
        # Replace *float* with *dict*
        if not isinstance(vmap, dict):
            # Replace existing value with singleton dict
            self[col] = {"_default_": vmap}
            # Update variable
            vmap = self[col]
        # Assign the specified value
        if comp is None:
            # Set it to _default_
            vmap["_default_"] = v
        else:
            # Specific component
            vmap[comp] = v

    # Get reference area for a given component.
    def get_RefArea(self, comp=None):
        r"""Return the reference area [for a component]

        The *comp* input has an affect if the ``"RefArea"`` option is a
        :class:`dict`. Otherwise all values of *comp* wil return the
        same *Aref*

        :Call:
            >>> Aref = opts.get_RefArea()
            >>> Aref = opts.get_RefArea(comp=None)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *Aref*: :class:`float`
                Reference area [for *comp*]
        :Versions:
            * 2014-09-29 ``@ddalle``: Version 1.0
            * 2022-11-01 ``@ddalle``: Version 2.0; :func:`get_refcol`
        """
        return self.get_refcol("RefArea", comp=comp)

    # Set the reference area for a given component.
    def set_RefArea(self, Aref, comp=None):
        r"""Set the reference area [of a component]

        :Call:
            >>> opts.set_RefArea(Aref)
            >>> opts.set_RefArea(Aref, comp=None)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *Aref*: :class:`float`
                Reference area [for *comp*]
            *comp*: {``None``} | :class:`str`
                Name of component
        :Versions:
            * 2014-09-29 ``@ddalle``: Version 1.0
            * 2022-11-01 ``@ddalle``: Version 2.0; :func:`set_refcol`
        """
        return self.set_refcol("RefArea", Aref, comp=comp)

    # Get reference length for a given component.
    def get_RefLength(self, comp=None):
        r"""Return the reference length [for a component]

        The *comp* argument has an effect if the``"RefLength"`` option
        is a :class:`dict`

        :Call:
            >>> Lref = opts.get_RefLength()
            >>> Lref = opts.get_RefLength(comp=None)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *Lref*: :class:`float`
                Reference length [of *comp*]
        :Versions:
            * 2014-09-29 ``@ddalle``: Version 1.0
            * 2022-11-01 ``@ddalle``: Version 2.0; :func:`get_refcol`
        """
        return self.get_refcol("RefLength", comp=comp)

    # Set the reference length for a given component.
    def set_RefLength(self, Lref, comp=None):
        r"""Set the reference length [of a component]

        :Call:
            >>> opts.set_RefLength(Lref)
            >>> opts.set_RefLength(Lref, comp=None)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *Lref*: :class:`float`
                Reference length [of *comp*]
            *comp*: {``None``} | :class:`str`
                Name of component
        :Versions:
            * 2014-09-29 ``@ddalle``: Version 1.0
            * 2022-11-01 ``@ddalle``: Version 2.0; :func:`set_refcol`
        """
        return self.set_refcol("RefLength", Lref, comp=comp)

    # Get reference length for a given component.
    def get_RefSpan(self, comp=None):
        r"""Return the reference span [for a component]

        This falls back to ``"RefLength"`` if appropriate

        :Call:
            >>> bref = opts.get_RefSpan()
            >>> bref = opts.get_RefSpan(comp=None)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *bref*: :class:`float`
                Reference span [for *comp*]
        :Versions:
            * 2017-02-19 ``@ddalle``: Version 1.0; from get_RefLength
            * 2022-11-01 ``@ddalle``: Version 2.0; :func:`get_refcol`
        """
        # Get reference span
        bref = self.get_refcol("RefSpan", comp=comp)
        # Check if defined
        if bref is None:
            # Use reference length
            return self.get_refcol("RefLength", comp=comp)
        else:
            # Return specific reference span
            return bref

    # Set the reference length for a given component.
    def set_RefSpan(self, bref, comp=None):
        r"""Set the reference span [for a component]

        :Call:
            >>> opts.set_RefSpan(bref)
            >>> opts.set_RefSpan(bref, comp=None)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *bref*: :class:`float`
                Reference span [for *comp*]
            *comp*: {``None``} | :class:`str`
                Name of component
        :Versions:
            * 2017-02-19 ``@ddalle``: Copied from :func:`set_RefLength`
            * 2022-11-01 ``@ddalle``: Version 2.0; :func:`set_refcol`
        """
        return self.set_refcol("RefSpan", bref, comp=comp)

    # Get points
    def get_Point(self, name=None):
        r"""Return the coordinates of a point by name

        If the input is a point, it is simply returned

        :Call:
            >>> x = opts.get_Point(name=None)
            >>> x = opts.get_Point(x)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *name*: :class:`str`
                Point name
        :Outputs:
            *x*: [:class:`float`, :class:`float`, :class:`float`]
                Coordinates of that point
        :Versions:
            * 2015-09-11 ``@ddalle``: Version 1.0
            * 2022-11-01 ``@ddalle``: Version 1.1; ARRAY_TYPES
        """
        # If it's already a vector, use it
        if isinstance(name, ARRAY_TYPES):
            return name
        # Get the specified points
        P = self.get('Points', {})
        # Check input consistency.
        if name not in P:
            raise KeyError(
                "Point named '%s' is not defined in the 'Config' section."
                % name)
        # Get the coordinates.
        return P[name]

    # Set the value of a point.
    def set_Point(self, x, name: str):
        r"""Set or alter the coordinates of a point by name

        :Call:
            >>> opts.set_Point(x, name)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *x*: [:class:`float`, :class:`float`, :class:`float`]
                Coordinates of that point
            *name*: :class:`str`
                Point name
        :Versions:
            * 2015-09-11 ``@ddalle``: Version 1.0
            * 2022-11-01 ``@ddalle``: Versoin 1.1; ARRAY_TYPES
        """
        # Make sure that "Points" are included.
        self.setdefault('Points', {})
        # Check the input
        if not isinstance(x, ARRAY_TYPES):
            # Not a vector
            raise TypeError(
                "Cannot set point '%s' to a non-array value." % name)
        elif len(x) < 2 or len(x) > 3:
            # Not a 2- or 3-vector
            raise IndexError(
                "Point '%s' with size %i is not a 2D or 3D point"
                % (name, len(x)))
        # Set it
        self["Points"][name] = list(x)

    # Expand point names
    def expand_Point(self, x):
        r"""Expand points that are specified by name instead of value

        :Call:
            >>> x = opts.expand_Point(x)
            >>> x = opts.expand_Point(s)
            >>> X = opts.expand_Point(d)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *x*: :class:`list`\ [:class:`float`]
                Point
            *s*: :class:`str`
                Point name
            *d*: :class:`dict`
                Dictionary of points and point names
        :Outputs:
            *x*: [:class:`float`, :class:`float`, :class:`float`]
                Point
            *X*: :class:`dict`
                Dictionary of points
        :Versions:
            * 2015-09-12 ``@ddalle``: Version 1.0
        """
        # Check input type.
        if isinstance(x, str):
            # Single point name
            return self.get_Point(x)
        elif x is None:
            # Null input
            return []
        elif isinstance(x, ARRAY_TYPES):
            # Check length
            n = len(x)
            # Check length
            if n in (2, 3):
                # Check first entry
                if isinstance(x[0], FLOAT_TYPES):
                    # Already a point
                    return x
            # Otherwise, this is a list of points
            return [self.get_Point(xk) for xk in x]
        elif not isinstance(x, dict):
            # Unrecognized
            raise TypeError(
                "Cannot expand points of type '%s'"
                % type(x).__name__)
        # Initialize output dictionary
        X = {pt: self.get_Point(v) for pt, v in x.items()}
        # Output
        return X

    # Get moment reference point for a given component.
    def get_RefPoint(self, comp=None):
        r"""Return the moment reference point [for a component]

        :Call:
            >>> x = opts.get_RefPoint()
            >>> x = opts.get_RefPoint(comp=None)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: :class:`str` or :class:`int`
                Name of component or component index
        :Outputs:
            *x*: [:class:`float`, :class:`float`, :class:`float`]
                Moment reference point [for *comp*]
        :Versions:
            * 2014-09-29 ``@ddalle``: Version 1.0
            * 2022-11-01 ``@ddalle``: Version 2.0; :func:`get_refcol`
        """
        # Get the defined value
        x = self.get_refcol("RefPoint", comp=comp)
        # Output
        return self.expand_Point(x)

    # Set the reference length for a given component.
    def set_RefPoint(self, x, comp=None):
        r"""Set the moment reference point [for a component]

        :Call:
            >>> opts.set_RefPoint(x)
            >>> opts.set_RefPoint(x, comp=None)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *x*: [:class:`float`, :class:`float`, :class:`float`]
                Global moment reference point or that for a component
            *comp*: :class:`str` or :class:`int`
                Name of component or component index
        :Versions:
            * 2014-09-29 ``@ddalle``: Version 1.0
            * 2022-11-01 ``@ddalle``: Version 2.0; :func:`set_refcol`
        """
        return self.set_refcol("RefPoint", x, comp=comp)


# Create methods
ConfigOpts.add_properties(["ConfigFile"])
ConfigOpts.add_properties(["Components"], prefix="Config")
