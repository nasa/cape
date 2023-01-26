r"""
:mod:`cape.cfdx.options.databookopts`: Databook definition options
===================================================================

This module contains the basic interface that define the ``"DataBook"``
of a CAPE configuration, which controls which information from the CFD
runs are extracted and collected elsewhere.

The :class:`DataBookOpts` class defined here defines the list of
databook components in the option *Components*, and each component has
its own entry in the ``"DataBook"`` section. A component that has no
further definitions is usually interpreted as a force & moment
component (*Type* is ``"FM"``), but other databook component types can
also be used.

"""

# Standard library
import fnmatch

# Local imports
from ...optdict import (
    OptionsDict,
    OptdictKeyError,
    BOOL_TYPES,
    FLOAT_TYPES,
    INT_TYPES,
    USE_PARENT)
from .util import rc0


# Template class for databook component
class DBCompOpts(OptionsDict):
    # No attbitues
    __slots__ = ()

    # Recognized options
    _optlist = {
        "CompID",
        "DNStats",
        "NLastStats",
        "NMaxStats",
        "NMin",
        "NStats",
        "Type",
    }

    # Aliases
    _optmap = {
        "Component": "CompID",
        "NAvg": "nStats",
        "NFirst": "NMin",
        "NLast": "nLastStats",
        "NMax": "nLastStats",
        "dnStats": "DNStats",
        "nAvg": "NStats",
        "nFirst": "NMin",
        "nLast": "NLastStats",
        "nLastStats": "NLastStats",
        "nMax": "NLastStats",
        "nMaxStats": "NMaxStats",
        "nMin": "NMin",
        "nStats": "NStats",
    }

    # Types
    _opttypes = {
        "DNStats": INT_TYPES,
        "NLastStats": INT_TYPES,
        "NMaxStats": INT_TYPES,
        "NMin": INT_TYPES,
        "NStats": INT_TYPES,
        "Type": str,
    }

    # Defaults
    _rc = {
        "Type": "FM",
    }

    # Descriptions
    _rst_descriptions = {
        "CompID": "surface componet(s) to use for this databook component",
        "DNStats": "increment for candidate window sizes",
        "NLastStats": "specific iteration at which to extract stats",
        "NMaxStats": "max number of iters to include in averaging window",
        "NMin": "first iter to consider for use in databook [for a comp]",
        "NStats": "iterations to use in averaging window [for a comp]",
        "Type": "databook component type",
    }


# Class for "TriqFM" components
class DBTriqFMOpts(DBCompOpts):
    # No attbitues
    __slots__ = ()

    # Recognized options
    _optlist = {
        "AbsProjTol",
        "AbsTol",
        "CompProjTol",
        "CompTol",
        "ConfigFile",
        "MapTri",
        "OutputFormat",
        "Patches",
        "RelProjTol",
        "RelTol",
    }

    # Aliases
    _optmap = {
        "Config": "ConfigFile",
        "MapTriFile": "MapTri",
        "antol": "AbsProjTol",
        "atol": "AbsTol",
        "cntol": "CompProjTol",
        "ctol": "CompTol",
        "rntol": "RelProjTol",
        "rtol": "RelTol",
    }

    # Types
    _opttypes = {
        "AbsProjTol": FLOAT_TYPES,
        "AbsTol": FLOAT_TYPES,
        "CompProjTol": FLOAT_TYPES,
        "CompTol": FLOAT_TYPES,
        "ConfigFile": str,
        "MapTri": str,
        "OutputFormat": str,
        "OutputSurface": BOOL_TYPES,
        "Patches": str,
        "RelProjTol": FLOAT_TYPES,
        "RelTol": FLOAT_TYPES,
    }

    # Specified values
    _optvals = {
        "OutputFormat": {"dat", "plt", "dat"},
    }

    # List options
    _optlistdepth = {
        "Patches": 1,
    }

    # Defaults
    _rc = {
        "OutputFormat": "plt",
        "OutputSurface": True,
    }

    # Descriptions
    _rst_descriptions = {
        "AbsProjTol": "absolute projection tolerance",
        "AbsTol": "absolute tangent tolerance for surface mapping",
        "CompProjTol": "projection tolerance relative to size of component",
        "CompTol": "tangent tolerance relative to component",
        "ConfigFile": "configuration file for surface groups",
        "OutputFormat": "output format for component surface files",
        "OutputSurface": "whether or not to write TriqFM surface",
        "MapTri": "name of a tri file to use for remapping CFD surface comps",
        "Patches": "list of patches for a databook component",
        "RelProjTol": "projection tolerance relative to size of geometry",
        "RelTol": "relative tangent tolerance for surface mapping",
    }


# Class for "TriqPoint" components
class DBTriqPointOpts(DBCompOpts):
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

    # Descriptions
    _rst_descriptions = {
        "Points": "list of individual point sensors",
    }

# Class for "PyFunc" components
class DBPyFuncOpts(DBCompOpts):
    # No attbitues
    __slots__ = ()

    # Recognized options
    _optlist = {
        "Function",
    }

    # Aliases
    _optmap = {}

    # Types
    _opttypes = {
        "Function": str,
    }

    # Defaults
    _rc = {}

    # Descriptions
    _rst_descriptions = {
        "Function": "Python function name",
    }


# Calss for line load options
class DBLineLoadOpts(DBCompOpts):
    # No attbitues
    __slots__ = ()

    # Recognized options
    _optlist = {
        "NCut",
        "SectionType",
    }

    # Aliases
    _optmap = {
        "nCut": "NCut",
    }

    # Types
    _opttypes = {
        "NCut": INT_TYPES,
    }

    # Allowed values
    _optvals = {
        "SectionType": {"dlds", "clds", "slds"},
    }

    # Defaults
    _rc = {
        "NCut": 200,
        "SectionType": "dlds",
    }

    # Descriptions
    _rst_descriptions = {
        "NCut": "Number of cuts to make using ``triload`` (-> +1 slice)",
        "SectionType": "line load section type",
    }


# Class for overall databook
class DataBookOpts(OptionsDict):
    r"""Dictionary-based interface for DataBook specifications

    :Call:
        >>> opts = DataBookOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of options
    :Outputs:
        *opts*: :class:`cape.cfdx.options.databookopts.DataBookOpts`
            Data book options interface
    :Versions:
        * 2014-12-20 ``@ddalle``: Version 1.0
    """
  # ================
  # Class Attributes
  # ================
  # <
    # No attbitues
    __slots__ = ()

    # Recognized options
    _optlist = {
        "Components",
        "Folder",
        "DNStats",
        "NLastStats",
        "NMaxStats",
        "NMin",
        "NStats",
    }

    # Aliases
    _optmap = {
        "Dir": "Folder",
        "NAvg": "nStats",
        "NFirst": "NMin",
        "NLast": "nLastStats",
        "NMax": "nLastStats",
        "dnStats": "DNStats",
        "nAvg": "NStats",
        "nFirst": "NMin",
        "nLast": "NLastStats",
        "nLastStats": "NLastStats",
        "nMax": "NLastStats",
        "nMaxStats": "NMaxStats",
        "nMin": "NMin",
        "nStats": "NStats",
    }

    # Types
    _opttypes = {
        "Components": str,
        "Folder": str,
        "DNStats": INT_TYPES,
        "NLastStats": INT_TYPES,
        "NMaxStats": INT_TYPES,
        "NMin": INT_TYPES,
        "NStats": INT_TYPES,
    }

    # Defaults
    _rc = {
        "Folder": "data",
        "NMin": 0,
        "NStats": 0,
    }

    # Descriptions
    _rst_descriptions = {
        "AbsProjTol": "absolute projection tolerance",
        "AbsTol": "absolute tangent tolerance for surface mapping",
        "CompProjTol": "projection tolerance relative to size of component",
        "CompTol": "tangent tolerance relative to component",
        "Components": "list of databook components",
        "Folder": "folder for root of databook",
        "DNStats": "increment for candidate window sizes",
        "MapTri": "name of a tri file to use for remapping CFD surface comps",
        "NCut": "number of ``'LineLoad'`` cuts for ``triload``",
        "NLastStats": "specific iteration at which to extract stats",
        "NMaxStats": "max number of iters to include in averaging window",
        "NMin": "first iter to consider for use in databook [for a comp]",
        "NStats": "iterations to use in averaging window [for a comp]",
        "Patches": "list of patches for a databook component",
        "Points": "list of individual point sensors",
        "RelProjTol": "projection tolerance relative to size of geometry",
        "RelTol": "tangent tolerance relative to overall geometry scale",
        "SectionType": "line load section type",
    }

    # Key defining additional *_xoptlist*
    _xoptkey = "Components"

    # Section map
    _sec_cls_opt = "Type"
    _sec_cls_optmap = {
        "LineLoad": DBLineLoadOpts,
        "PyFunc": DBPyFuncOpts,
        "TriqFM": DBTriqFMOpts,
        "TriqPont": DBTriqPointOpts,
    }

    # Parent for each section
    _sec_parent = {
        "Type": None,
        "_default_": USE_PARENT,
    }
  # >

  # =================
  # Global Components
  # =================
  # <
    # Get the targets for a specific component
    def get_CompTargets(self, comp):
        r"""Get the list of targets for a specific data book component

        :Call:
            >>> targs = opts.get_CompTargets(comp)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *targs*: :class:`list`\ [:class:`str`]
                List of targets for that component
        :Versions:
            * 2014-12-21 ``@ddalle``: Version 1.0
        """
        # Get the component options.
        copts = self.get(comp, {})
        # Get the targets.
        targs = copts.get('Targets', {})
        # Make sure it's a dict.
        if type(targs).__name__ not in ['dict']:
            raise TypeError("Targets for component '%s' are not a dict" % comp)
        # Output
        return targs

    # Get list of point in a point sensor group
    def get_DBGroupPoints(self, name):
        r"""Get the list of points in a group

        For example, get the list of point sensors in a point sensor
        group

        :Call:
            >>> pts = opts.get_DBGroupPoints(name)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *name*: :class:`str`
                Name of data book group
        :Outputs:
            *pts*: :class:`list`\ [:class:`str`]
                List of points (by name) in the group
        :Versions:
            * 2015-12-04 ``@ddalle``: Version 1.0
            * 2016-02-17 ``@ddalle``: Version 1.1; generic version
        """
        # Check.
        if name not in self:
            raise KeyError("Data book group '%s' not found" % name)
        # Check for points.
        pts = self[name].get("Points", [name])
        # Check if it's a list.
        if type(pts).__name__ in ['list', 'ndarray']:
            # Return list as-is
            return pts
        else:
            # Singleton list
            return [pts]
  # >

  # =======================
  # Class methods
  # =======================
  # <
    @classmethod
    def add_compgetters(cls, optlist, prefix=None, name=None, doc=True):
        r"""Add list of component-specific getters with common settings

        :Call:
            >>> cls.add_compgetters(optlist, prefix=None, name=None)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *optlist*: :class:`list`\ [:class:`str`]
                Name of options to process
            *prefix*: {``None``} | :class:`str`
                Optional prefix, e.g. ``opt="a", prefix="my"`` will add
                functions :func:`get_my_a` and :func:`set_my_a`
            *name*: {*opt*} | :class:`str`
                Alternate name to use in name of get and set functions
            *doc*: {``True``} | ``False``
                Whether or not to add docstring to functions
        :Versions:
            * 2022-11-08 ``@ddalle``: Version 1.0
        """
        for opt in optlist:
            cls.add_compgetter(opt, prefix=prefix, name=name, doc=doc)

    @classmethod
    def add_compgetter(cls, opt: str, prefix=None, name=None, doc=True):
        r"""Add getter method for option *opt*

        For example ``cls.add_property("a")`` will add a function
        :func:`get_a`, which has a signatures like
        :func:`OptionsDict.get_opt` except that it doesn't have the
        *opt* input.

        :Call:
            >>> cls.add_compgetter(opt, prefix=None, name=None)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *opt*: :class:`str`
                Name of option
            *prefix*: {``None``} | :class:`str`
                Optional prefix in method name
            *name*: {*opt*} | :class:`str`
                Alternate name to use in name of get and set functions
            *doc*: {``True``} | ``False``
                Whether or not to add docstring to getter function
        :Versions:
            * 2022-11-08 ``@ddalle``: Version 1.0
        """
        # Check if acting on original OptionsDict
        cls._assert_subclass()
        # Default name
        name, fullname = cls._get_funcname(opt, name, prefix)
        funcname = "get_" + fullname

        # Define function
        def func(self, comp=None, j=None, i=None, **kw):
            try:
                return self._get_opt_comp(opt, comp=comp, j=j, i=i, **kw)
            except Exception:
                raise

        # Generate docstring if requrested
        if doc:
            func.__doc__ = cls._genr8_compg_docstring(opt, name, prefix)
        # Modify metadata of *func*
        func.__name__ = funcname
        func.__qualname__ = "%s.%s" % (cls.__name__, funcname)
        # Save function
        setattr(cls, funcname, func)

    @classmethod
    def _genr8_compg_docstring(cls, opt: str, name, prefix, indent=8, tab=4):
        r"""Create automatic docstring for component getter function

        :Call:
            >>> txt = cls._genr8_compg_docstring(opt, name, prefx, **kw)
        :Inputs:
            *cls*: :class:`type`
                A subclass of :class:`OptionsDict`
            *opt*: :class:`str`
                Name of option
            *name*: {*opt*} | :class:`str`
                Alternate name to use in name of functions
            *prefx*: ``None`` | :class:`str`
                Optional prefix, e.g. ``opt="a", prefix="my"`` will add
                functions :func:`get_my_a` and :func:`set_my_a`
            *indent*: {``8``} | :class:`int` >= 0
                Number of spaces in lowest-level indent
            *tab*: {``4``} | :class:`int` > 0
                Number of additional spaces in each indent
        :Outputs:
            *txt*: :class:`str`
                Contents for ``get_{opt}`` function docstring
        :Versions:
            * 2022-10-03 ``@ddalle``: Version 1.0
        """
        # Expand tabs
        tab1 = " " * indent
        tab2 = " " * (indent + tab)
        tab3 = " " * (indent + 2*tab)
        # Normalize option name
        name, funcname = cls._get_funcname(opt, name, prefix)
        # Apply aliases if anny
        fullopt = cls.get_cls_key("_optmap", opt, vdef=opt)
        # Create title
        title = 'Get %s\n\n' % cls._genr8_rst_desc(fullopt)
        # Generate signature
        signature = (
            "%s>>> %s = opts.get_%s(comp=None, i=None, **kw)\n"
            % (tab2, name, funcname))
        # Generate class description
        rst_cls = cls._genr8_rst_cls(indent=indent, tab=tab)
        # Generate *opt* description
        rst_opt = cls._genr8_rst_opt(opt, indent=indent, tab=tab)
        # Form full docstring
        return (
            title +
            tab1 + ":Call:\n" +
            signature +
            tab1 + ":Inputs:\n" +
            rst_cls +
            tab2 + "*comp*: {``None``} | :class:`str`\n" +
            tab3 + "Name of databook component\n" +
            tab2 + "*i*: {``None``} | :class:`int`\n" +
            tab3 + "Case index\n" +
            tab1 + ":Outputs:\n" +
            rst_opt
        )
  # >

  # =================
  # Common Properties
  # =================
  # <
    # Generic subsection
    def _get_opt_comp(self, opt: str, comp=None, **kw):
        r"""Get an option, from a specific subsection if possible

        :Call:
            >>> v = opts._get_opt_comp(opt, comp=None, **kw)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *opt*: :class:`str`
                Name of option to access
            *comp*: {``None``} | :class:`str`
                Name of specific databook component
        :Outputs:
            *v*: :class:`object`
                Value of *opt* from either *opts* or *opts[comp]*
        :Versions:
            * 2022-11-08 ``@ddalle``: Version 1.0
            * 2022-12-14 ``@ddalle``: Version 2.0; get_subopt()
        """
        # No phases for databook
        kw["j"] = None
        # Check for *comp*
        if comp is None:
            # Get option from global
            return self.get_opt(opt, **kw)
        elif comp not in self:
            # Check valiid comp
            if comp not in self.get_DataBookComponents():
                raise ValueError("No DataBook component named '%s'" % comp)
            # Attempt to return global option
            return self.get_opt(opt, **kw)
        else:
            # Use cascading options
            return self.get_subopt(comp, opt, **kw)

    # CompID: special default
    def get_DataBookCompID(self, comp: str, **kw):
        r"""Get *CompID* opton for a component

        :Call:
            >>> compid = opts.get_DataBookCompID(comp, **kw)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of databook component
        :Outputs:
            *compid*: :class:`int` | :class:`str` | :class:`list`
                Value of *opt* from either *opts* or *opts[comp]*
        :Versions:
            * 2023-01-22 ``@ddalle``: Version 1.0
        """
        # Check validity of component
        if comp not in self.get_DataBookComponents():
            raise ValueError("No DataBook component named '%s'" % comp)
        # Get suboption
        compid = self.get_subopt(comp, "CompID", **kw)
        # Check for null result
        if compid is None:
            # Default is name of component
            return comp
        else:
            # Return nontrivial result
            return compid
  # >

  # =======
  # Targets
  # =======
  # <
    # Get the targets
    def get_DataBookTargets(self):
        """Get the list of targets to be used for the data book
        
        :Call:
            >>> targets = opts.get_DataBookTargets()
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
        :Outputs:
            *targets*: :class:`dict`\ [:class:`dict`]
                Dictionary of targets
        :Versions:
            * 2014-12-20 ``@ddalle``: Version 1.0
        """
        # Output
        return self.get('Targets', {})
        
    # Get a target by name
    def get_DataBookTargetByName(self, targ):
        """Get a data book target option set by the name of the target
        
        :Call:
            >>> topts = opts.get_DataBookTargetByName(targ)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *targ*: :class:`str`
                Name of the data book target
        :Outputs:
            * 2015-12-15 ``@ddalle``: Version 1.0
        """
        # Get the set of targets
        DBTs = self.get_DataBookTargets()
        # Check if it's present
        if targ not in DBTs:
            raise KeyError("There is no DBTarget called '%s'" % targ)
        # Output
        return DBTs[targ]
    
    # Get type for a given target
    def get_DataBookTargetType(self, targ):
        """Get the target data book type
        
        This can be either a generic target specified in a single file or a Cape
        data book that has the same description as the present data book
        
        :Call:
            >>> typ = opts.get_DataBookTargetType(targ)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *targ*: :class:`str`
                Name of the data book target
        :Outputs:
            *typ*: {``"generic"``} | ``"cape"``
                Target type, generic CSV file or duplicate data book
        :Versions:
            * 2016-06-27 ``@ddalle``: Version 1.0
        """
        # Get the set of targets
        DBTs = self.get_DataBookTargets()
        # Check if it's present
        if targ not in DBTs:
            raise KeyError("There is no DBTarget called '%s'" % targ)
        # Get the type
        return DBTs[targ].get('Type', 'generic')
        
    # Get data book target directory
    def get_DataBookTargetDir(self, targ):
        """Get the folder for a data book duplicate target
        
        :Call:
            >>> fdir = opts.get_DataBookTargetDir(targ)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *targ*: :class:`str`
                Name of the data book target
        :Outputs:
            *typ*: {``"generic"``} | ``"cape"``
                Target type, generic CSV file or duplicate data book
        :Versions:
            * 2016-06-27 ``@ddalle``: Version 1.0
        """
        # Get the set of targets
        DBTs = self.get_DataBookTargets()
        # Check if it's present
        if targ not in DBTs:
            raise KeyError("There is no DBTarget called '%s'" % targ)
        # Get the type
        return DBTs[targ].get('Folder', 'data')
  # >

  # ================
  # Component Config
  # ================
  # <
    # Get data book components by type
    def get_DataBookByType(self, typ):
        """Get the list of data book components with a given type
        
        :Call:
            >>> comps = opts.get_DataBookByType(typ)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *typ*: ``"Force"`` | ``"FM"`` | ``"LineLoad"`` | :class:`str`
                Data book type
        :Outputs:
            *comps*: :class:`list`\ [:class:`str`]
                List of data book components with ``"Type"`` matching *typ*
        :Versions:
            * 2016-06-07 ``@ddalle``: Version 1.0
        """
        # Initialize components
        comps = []
        # Get list of types
        for comp in self.get_DataBookComponents():
            # Check the type
            if typ == self.get_DataBookType(comp):
                # Append the component to the list
                comps.append(comp)
        # Output
        return comps
        
    # Get list of components matching a type and list of wild cards
    def get_DataBookByGlob(self, typ, comp=None):
        """Get list of components by type and list of wild cards
        
        :Call:
            >>> comps = opts.get_DataBookByGlob(typ, comp)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *typ*: FM | Force | Moment | LineLoad | TriqFM
                Data book type
            *comp*: {``None``} | :class:`str`
                List of component wild cards, separated by commas
        :Outputs:
            *comps*: :class:`str`
                All components meeting one or more wild cards
        :Versions:
            * 2017-04-25 ``@ddalle``: Version 1.0
        """
        # Check for list of types
        if type(typ).__name__ not in ['ndarray', 'list']:
            # Ensure list
            typ = [typ]
        # Get list of all components with matching type
        comps_all = []
        for t in typ:
            comps_all += self.get_DataBookByType(t)
        # Check for default option
        if comp in [True, None]:
            return comps_all
        # Initialize output
        comps = []
        # Ensure input is a list
        if type(comp).__name__ in ['list', 'ndarray']:
            comps_in = comp
        else:
            comps_in = [comp]
        # Initialize wild cards
        comps_wc = []
        # Split by comma
        for c in comps_in:
            comps_wc += c.split(",")
        # Loop through components to check if it matches
        for c in comps_all:
            # Loop through components
            for pat in comps_wc:
                # Check if it matches
                if fnmatch.fnmatch(c, pat):
                    # Add the component to the list
                    comps.append(c)
                    break
        # Output
        return comps
            
    # Get the data type of a specific component
    def get_DataBookType(self, comp):
        """Get the type of data book entry for one component
        
        :Call:
            >>> ctype = opts.get_DataBookType(comp)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *ctype*: {Force} | Moment | FM | PointSensor | LineLoad
                Data book entry type
        :Versions:
            * 2015-12-14 ``@ddalle``: Version 1.0
        """
        # Get the component options.
        copts = self.get(comp, {})
        # Return the type
        return copts.get("Type", "FM")
        
    # Get list of components in a component
    def get_DataBookCompID(self, comp):
        """Get list of components in a data book component
        
        :Call:
            >>> compID = opts.get_DataBookCompID(comp)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of data book component/field
        :Outputs:
            *compID*: :class:`str` | :class:`int` | :class:`list`
                Component or list of components to which this DB applies
        :Versions:
            * 2016-06-07 ``@ddalle``: Version 1.0
        """
        # Get the options for that component
        copts = self.get(comp, {})
        # Get the componetns
        return copts.get('CompID', comp)
        
    # Get the coefficients for a specific component
    def get_DataBookCoeffs(self, comp):
        """Get the list of data book coefficients for a specific component
        
        :Call:
            >>> coeffs = opts.get_DataBookCoeffs(comp)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *coeffs*: :class:`list`\ [:class:`str`]
                List of coefficients for that component
        :Versions:
            * 2014-12-21 ``@ddalle``: Version 1.0
        """
        # Get the component options.
        copts = self.get(comp, {})
        # Check for manually-specified coefficients
        coeffs = copts.get("Cols", self.get("Coefficients", []))
        # Check the type.
        if not isinstance(coeffs, list):
            raise TypeError(
                "Coefficients for component '%s' must be a list." % comp)
        # Exit if that exists.
        if len(coeffs) > 0:
            return coeffs
        # Check the type.
        ctype = self.get_DataBookType(comp)
        # Default coefficients
        if ctype in ["Force", "force"]:
            # Force only, body-frame
            coeffs = ["CA", "CY", "CN"]
        elif ctype in ["Moment", "moment"]:
            # Moment only, body-frame
            coeffs = ["CLL", "CLM", "CLN"]
        elif ctype in ["DataFM", "FM", "full", "Full"]:
            # Force and moment
            coeffs = ["CA", "CY", "CN", "CLL", "CLM", "CLN"]
        elif ctype in ["TriqFM"]:
            # Extracted force and moment
            coeffs = [
                "CA",  "CY",  "CN", 
                "CAv", "CYv", "CNv",
                "Cp_min", "Cp_max",
                "Ax", "Ay", "Az"
            ]
        elif ctype in ["PointSensor", "TriqPoint"]:
            # Default to list of points for a point sensor
            coeffs = ["x", "y", "z", "cp"]
        # Output
        return coeffs
        
    # Get coefficients for a specific component/coeff
    def get_DataBookCoeffStats(self, comp, coeff):
        """Get the list of statistical properties for a specific coefficient
        
        :Call:
            >>> sts = opts.get_DataBookCoeffStats(comp, coeff)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of data book component
            *coeff*: :class:`str`
                Name of data book coefficient, e.g. "CA", "CY", etc.
        :Outputs:
            *sts*: :class:`list` (mu | std | min | max | err)
                List of statistical properties for this coefficient
        :Versions:
            * 2016-03-15 ``@ddalle``: Version 1.0
        """
        # Get the component options
        copts = self.get(comp, {})
        # Get the coefficient
        sts = copts.get(coeff)
        # Get type
        typ = self.get_DataBookType(comp)
        # Process default if necessary
        if sts is not None:
            # Non-default; check the type
            if type(sts).__name__ not in ['list', 'ndarray']:
                raise TypeError(
                    "List of statistical properties must be a list")
            # Output
            return sts
        # Data book type
        typ = self.get_DataBookType(comp)
        # Check data book type
        if typ in ["TriqFM", "TriqPoint", "PointSensor"]:
            # No iterative history
            return ['mu']
        # Others; iterative history available
        if coeff in ['x', 'y', 'z', 'X', 'Y', 'Z']:
            # Coordinates
            return ['mu']
        elif coeff in ['CA', 'CY', 'CN', 'CLL', 'CLM', 'CLN']:
            # Body-frame force/moment
            return ['mu', 'min', 'max', 'std', 'err']
        elif coeff in ['CL', 'CN', 'CS']:
            # Stability-frame force/moment
            return ['mu', 'min', 'max', 'std', 'err']
        elif typ in ["PyFunc"]:
            return ["mu"]
        else:
            # Default for most states
            return ['mu', 'std', 'min', 'max']
        
    # Get additional float columns
    def get_DataBookFloatCols(self, comp):
        """Get additional numeric columns for component (other than coeffs)
        
        :Call:
            >>> fcols = opts.get_DataBookFloatCols(comp)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of data book component
        :Outputs:
            *fcols*: :class:`list`\ [:class:`str`]
                List of additional float columns
        :Versions:
            * 2016-03-15 ``@ddalle``: Version 1.0
        """
        # Get the component options
        copts = self.get(comp, {})
        # Get data book default
        fcols_db = self.get("FloatCols")
        # Get float columns option
        fcols = copts.get("FloatCols")
        # Check for default
        if fcols is not None:
            # Manual option
            return fcols
        elif fcols_db is not None:
            # Data book option
            return fcols_db
        else:
            # Global default
            return []
            
    # Get integer columns
    def get_DataBookIntCols(self, comp):
        """Get integer columns for component
        
        :Call:
            >>> fcols = opts.get_DataBookFloatCols(comp)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of data book component
        :Outputs:
            *fcols*: :class:`list`\ [:class:`str`]
                List of additional float columns
        :Versions:
            * 2016-03-15 ``@ddalle``: Version 1.0
        """
        # Get the component options
        copts = self.get(comp, {})
        # Get type
        ctyp = self.get_DataBookType(comp)
        # Get data book default
        icols_db = self.get("IntCols")
        # Get float columns option
        icols = copts.get("IntCols")
        # Check for default
        if icols is not None:
            # Manual option
            return icols
        elif icols_db is not None:
            # Data book option
            return icols_db
        elif ctyp in ["TriqPoint", "PointSensor", "PyFunc"]:
            # Limited default
            return ['nIter']
        else:
            # Global default
            return ['nIter', 'nStats']
        
    # Get full list of columns for a specific component
    def get_DataBookCols(self, comp):
        """Get the full list of data book columns for a specific component
        
        This includes the list of coefficients, e.g. ``['CA', 'CY', 'CN']``;
        statistics such as ``'CA_min'`` if *nStats* is greater than 0; and
        targets such as ``'CA_t'`` if there is a target for *CA*.
        
        :Call:
            >>> cols = opts.get_DataBookCols(comp)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *cols*: :class:`list`\ [:class:`str`]
                List of coefficients and other columns for that coefficient
        :Versions:
            * 2014-12-21 ``@ddalle``: Version 1.0
        """
        # Data columns (from CFD)
        dcols = self.get_DataBookDataCols(comp)
        # Output
        return dcols
        
    # Get full list of data columns for a specific component
    def get_DataBookDataCols(self, comp):
        """Get the list of data book columns for a specific component
        
        This includes the list of coefficients, e.g. ``['CA', 'CY', 'CN']``;
        statistics such as ``'CA_min'`` if *nStats* is greater than 0.
        
        :Call:
            >>> cols = opts.get_DataBookDataCols(comp)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *cols*: :class:`list`\ [:class:`str`]
                List of coefficients and other columns for that coefficient
        :Versions:
            * 2014-12-21 ``@ddalle``: Version 1.0
            * 2022-04-08 ``@ddalle``: Version 2.0; cooeff-spec suffixes
        """
        # Get the list of coefficients.
        coeffs = self.get_DataBookCoeffs(comp)
        # Initialize output
        cols = [] + coeffs
        # Get the number of iterations used for statistics
        nStats = self.get_nStats()
        # Process statistical columns.
        if nStats > 0:
            # Loop through columns.
            for coeff in coeffs:
                # Get stat cols for this coeff
                scols = self.get_DataBookCoeffStats(comp, coeff)
                # Don't double-count the mean
                if "mu" in scols:
                    scols.remove("mu")
                # Append all statistical columns.
                cols += [coeff + "_" + suf for suf in scols]
        # Output.
        return cols
        
    # Get list of target data columns for a specific component
    def get_DataBookTargetCols(self, comp):
        """Get the list of data book target columns for a specific component
        
        :Call:
            >>> cols = opts.get_DataBookDataCols(comp)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *cols*: :class:`list`\ [:class:`str`]
                List of coefficient target values
        :Versions:
            * 2014-12-21 ``@ddalle``: Version 1.0
        """
        # Initialize output
        cols = []
        # Process targets.
        targs = self.get_CompTargets(comp)
        # Loop through the targets.
        for c in targs:
            # Append target column
            cols.append(c+'_t')
        # Output
        return cols
  # >
  
  # ======================
  # Iterative Force/Moment
  # ======================
  # <
        
    # Get the transformations for a specific component
    def get_DataBookTransformations(self, comp):
        """
        Get the transformations required to transform a component's data book
        into the body frame of that component.
        
        :Call:
            >>> tlist = opts.get_DataBookTransformations(comp)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *tlist*: :class:`list`\ [:class:`dict`]
                List of targets for that component
        :Versions:
            * 2014-12-22 ``@ddalle``: Version 1.0
        """
        # Get the options for the component.
        copts = self.get(comp, {})
        # Get the value specified, defaulting to an empty list.
        tlist = copts.get('Transformations', [])
        # Make sure it's a list.
        if type(tlist).__name__ not in ['list', 'ndarray']:
            # Probably a single transformation; put it in a list
            tlist = [tlist]
        # Output
        return tlist
  # >
      
  # ===========
  # Line Loads
  # ===========
  # <
    # Get momentum setting
    def get_DataBookMomentum(self, comp):
        """Get 'Momentum' flag for a data book component
        
        :Call:
            >>> qm = opts.get_DataBookMomentum(comp)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *qm*: ``True`` | {``False``}
                Whether or not to include momentum
        :Versions:
            * 2016-06-07 ``@ddalle``: Version 1.0
        """
        # Global data book setting
        db_qm = self.get("Momentum", False)
        # Get component options
        copts = self.get(comp, {})
        # Get the local setting
        return copts.get("Momentum", db_qm)
        
    # Get guage pressure setting
    def get_DataBookGauge(self, comp):
        """Get 'Gauge' flag for a data book component
        
        :Call:
            >>> qg = opts.get_DataBookGauge(comp)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *qg*: {``True``} | ``False``
                Option to use gauge forces (freestream pressure as reference)
        :Versions:
            * 2017-03-29 ``@ddalle``: Version 1.0
        """
        # Global data book setting
        db_qg = self.get("Gauge", True)
        # Get component options
        copts = self.get(comp, {})
        # Get the local setting
        return copts.get("Gauge", db_qg)
        
    # Get trim setting
    def get_DataBookTrim(self, comp):
        """Get 'Trim' flag for a data book component
        
        :Call:
            >>> iTrim = opts.get_DataBookTrim(comp)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *iTrim*: ``0`` | {``1``}
                Trim setting; no output if ``None``
        :Versions:
            * 2016-06-07 ``@ddalle``: Version 1.0
        """
        # Global data book setting
        db_trim = self.get("Trim", 1)
        # Get component options
        copts = self.get(comp, {})
        # Get the local setting
        return copts.get("Trim", db_trim)
  # >
  

# Options available to subclasses
_SETTER_PROPS = (
    "DNStats",
    "NMin",
    "NStats",
    "NStatsMax",
)
DataBookOpts.add_compgetters(_SETTER_PROPS, prefix="DataBook")
DataBookOpts.add_setters(_SETTER_PROPS, prefix="DataBook")

# Options only available to sections
_GETTER_PROPS = (
    "AbsProjTol",
    "AbsTol",
    "CompProjTol",
    "CompTol",
    "ConfigFile",
    "Function",
    "MapTri",
    "NCut",
    "OutputFormat",
    "Patches",
    "Points",
    "RelProjTol",
    "RelTol",
    "SectionType",
)
DataBookOpts.add_compgetters(_GETTER_PROPS, prefix="DataBook")

# Normal top-level properties
_PROPS = (
    "Components",
    "Folder",
)
DataBookOpts.add_properties(_PROPS, prefix="DataBook")


# Class for target data
class DBTargetOpts(OptionsDict):
    """Dictionary-based interface for data book targets
    
    :Call:
        >>> opts = DBTarget(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of PBS options
    :Outputs:
        *opts*: :class:`cape.options.DataBook.DBTarget`
            Data book target options interface
    :Versions:
        * 2014-12-01 ``@ddalle``: Version 1.0
    """
    
    # Get the maximum number of refinements
    def get_TargetName(self):
        """Get the name/identifier for a given data book target
        
        :Call:
            >>> Name = opts.get_TargetName()
        :Inputs:
            *opts*: :class:`cape.options.DataBook.DBTarget`
                Options interface
        :Outputs:
            *Name*: :class:`str`
                Identifier for the target
        :Versions:
            * 2014-08-03 ``@ddalle``: Version 1.0
        """
        return self.get('Name', 'Target')
        
    # Get the label
    def get_TargetLabel(self):
        """Get the name/identifier for a given data book target
        
        :Call:
            >>> lbl = opts.get_TargetLabel()
        :Inputs:
            *opts*: :class:`cape.options.DataBook.DBTarget`
                Options interface
        :Outputs:
            *lbl*: :class:`str`
                Label for the data book target to be used in plots and reports 
        :Versions:
            * 2015-06-04 ``@ddalle``: Version 1.0
        """
        # Default to target identifier
        return self.get('Label', self.get_TargetName())
        
    # Get the components that this target describes
    def get_TargetComponents(self):
        """Get the list of components described by this component
        
        Returning ``None`` is a flag to use all components from the data book.
        
        :Call:
            >>> comps = opts.get_TargetComponents()
        :Inputs:
            *opts*: :class:`cape.options.DataBook.DBTarget`
                Options interface
        :Outputs:
            *comps*: :class:`list`\ [:class:`str`]
                List of components (``None`` if not specified)
        :Versions:
            * 2015-06-03 ``@ddalle``: Version 1.0
        """
        # Get the list
        comps = self.get('Components')
        # Check type.
        if type(comps).__name__ in ['str', 'unicode']:
            # String: make it a list.
            return [comps]
        else:
            # List, ``None``, or nonsense
            return comps
        
    # Get the file name
    def get_TargetFile(self):
        """Get the file name for the target
        
        :Call:
            >>> fname = opts.get_TargetFile()
        :Inputs:
            *opts*: :class:`cape.options.DataBook.DBTarget`
                Options interface
        :Outputs:
            *fname*: :class:`str`
                Name of the file
        :Versions:
            * 2014-12-20 ``@ddalle``: Version 1.0
        """
        return self.get('File', 'Target.dat')
        
    # Get the directory name
    def get_TargetDir(self):
        """Get the directory for the duplicate target data book
        
        :Call:
            >>> fdir = opts.get_TargetDir()
        :Inputs:
            *opts*: :class:`cape.options.DataBook.DBTarget`
                Options interface
        :Outputs:
            *fdir*: :class:`str`
                Name of the directory (relative to root directory)
        :Versions:
            * 2016-06-27 ``@ddalle``: Version 1.0
        """
        return self.get('Folder', 'data')
        
    # Get the target type
    def get_TargetType(self):
        """Get the target type for a target data book
        
        :Call:
            >>> typ = opts.get_TargetType()
        :Inputs:
            *opts*: :class:`cape.otpions.DataBook.DBTarget`
                Options interface
        :Outputs:
            *typ*: {``"generic"``} | ``"cape"``
                Target type, generic CSV file or duplicate data book
        :Versions:
            * 2016-06-27 ``@ddalle``: Version 1.0
        """
        return self.get('Type', 'generic')
        
    # Get tolerance
    def get_Tol(self, xk):
        """Get the tolerance for a particular trajectory key
        
        :Call:
            >>> tol = opts.get_Tol(xk)
        :Inputs:
            *opts*: :class:`cape.options.DataBook.DBTarget`
                Options interface
            *xk*: :class:`str`
                Name of trajectory key
        :Outputs:
            *tol*: :class:`float`
                Tolerance to consider as matching value for a trajectory key
        :Versions:
            * 2015-12-16 ``@ddalle``: Version 1.0
        """
        # Get tolerance option set
        tolopts = self.get("Tolerances", {})
        # Get the option specific to this key
        return tolopts.get(xk, None)
        
    # Get the delimiter
    def get_Delimiter(self):
        """Get the delimiter for a target file
        
        :Call:
            >>> delim = opts.get_Delimiter()
        :Inputs:
            *opts*: :class:`cape.options.DataBook.DBTarget`
                Options interface
        :Outputs:
            *delim*: :class:`str`
                Delimiter text
        :Versions:
            * 2014-12-21 ``@ddalle``: Version 1.0
        """
        return self.get('Delimiter', rc0('Delimiter'))
        
    # Get the comment character.
    def get_CommentChar(self):
        """Get the character to used to mark comments
        
        :Call:
            >>> comchar = opts.get_CommentChar()
        :Inputs:
            *opts*: :class:`cape.options.DataBook.DBTarget`
                Options interface
        :Outputs:
            *comchar*: :class:`str`
                Comment character (may be multiple characters)
        :Versions:
            * 2014-12-21 ``@ddalle``: Version 1.0
        """
        return self.get('Comment', '#')
    
    # Get trajectory conversion
    def get_RunMatrix(self):
        """Get the trajectory translations
        
        :Call:
            >>> traj = opts.get_RunMatrix()
        :Inputs:
            *opts*: :class:`cape.options.DataBook.DBTarget`
                Options interface
        :Outputs:
            *comchar*: :class:`str`
                Comment character (may be multiple characters)
        :Versions:
            * 2014-12-21 ``@ddalle``: Version 1.0
        """
        return self.get('RunMatrix', {})    
# class DBTarget

