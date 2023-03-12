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
    ARRAY_TYPES,
    BOOL_TYPES,
    FLOAT_TYPES,
    INT_TYPES,
    USE_PARENT)


# Template class for databook component
class DBCompOpts(OptionsDict):
    # No attbitues
    __slots__ = ()

    # Recognized options
    _optlist = {
        "Cols",
        "CompID",
        "DNStats",
        "NLastStats",
        "NMaxStats",
        "NMin",
        "NStats",
        "Targets",
        "Type",
    }

    # Aliases
    _optmap = {
        "Coefficients": "Cols",
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
        "tagets": "Targets",
    }

    # Types
    _opttypes = {
        "Cols": list,
        "DNStats": INT_TYPES,
        "NLastStats": INT_TYPES,
        "NMaxStats": INT_TYPES,
        "NMin": INT_TYPES,
        "NStats": INT_TYPES,
        "Type": str,
        "Targets": dict,
    }

    # Defaults
    _rc = {
        "Targets": {},
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


# Class for "IterPoint" components
class DBFMOpts(DBCompOpts):
    # No attributes
    __slots__ = ()


# Class for "IterPoint" components
class DBIterPointOpts(DBCompOpts):
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


# Class for target data
class DBTargetOpts(OptionsDict):
    r"""Dictionary-based interface for data book targets

    :Call:
        >>> opts = DBTarget(fjson, **kw)
        >>> opts = DBTarget(mydict, **kw)
    :Inputs:
        *fjson*: {``None``} | :class:`str`
            Name of JSON file with settings
        *mydict*: :class:`dict`
            Existing options structure
        *kw*: :class:`dict`
            Additional options from keyword arguments
    :Outputs:
        *opts*: :class:`DBTargetOptions`
            Data book target options interface
    :Versions:
        * 2014-12-01 ``@ddalle``: v1.0
        * 2023-03-11 ``@ddalle``: v2.0; use :mod:`optdict`
    """
  # ================
  # Class Attributes
  # ================
  # <
    # No attbitues
    __slots__ = ()

    # Known options
    _optlist = {
        "CommentChar",
        "Components",
        "Delimiter",
        "File",
        "Folder",
        "Label",
        "Name",
        "Tolerances",
        "Translations",
        "Type",
    }

    # Aliases
    _optmap = {
        "Comment": "CommentChar",
        "Dir": "Folder",
        "RunMatrix": "Translations",
        "Tolerance": "Tolerances",
        "delim": "Delimiter",
        "tol": "Tolerances",
        "tols": "Tolerances",
        "trans": "Translations",
    }

    # Types
    _opttypes = {
        "CommentChar": str,
        "Components": str,
        "Delimiter": str,
        "File": str,
        "Folder": str,
        "Label": str,
        "Name": str,
        "Tolerances": dict,
        "Translations": dict,
        "Type": str,
    }

    # Allowed values
    _optvals = {
        "Type": (
            "generic",
            "databook",
        ),
    }

    # List keys
    _optlistdepth = {
        "Components": 1,
    }

    # Defaults
    _rc = {
        "CommentChar": "#",
        "Delimiter": ",",
        "Folder": "data",
        "Type": "generic",
        "tol": 1e-6,
    }

    # Descriptions
    _rst_descriptions = {
        "CommentChart": "Character(s) denoting a comment line in target file",
        "Components": "List of databook components with data from this target",
        "Delimiter": "Delimiter in databook target data file",
        "File": "Name of file from which to read data",
        "Folder": "Name of folder from which to read data",
        "Label": "Label to use when plotting this target",
        "Name": "Internal *name* to use for target",
        "Tolerances": "Dictionary of tolerances for run matrix keys",
        "Type": "DataBook Target type",
    }
  # >

    # Get tolerance
    def get_Tol(self, col: str):
        r"""Get the tolerance for a particular trajectory key

        :Call:
            >>> tol = opts.get_Tol(xk)
        :Inputs:
            *opts*: :class:`DBTargetOpts`
                Options interface
            *col*: :class:`str`
                Name of trajectory key
        :Outputs:
            *tol*: {``None``} | :class:`float`
                Max distance for a match for column *col*
        :Versions:
            * 2015-12-16 ``@ddalle``: v1.0
            * 2023-03-11 ``@ddalle``: v2.0
        """
        # Get tolerance option set
        tolopts = self.get("Tolerances", {})
        # Get the option specific to this key
        return tolopts.get(col, self.__class__._rc["tol"])


# Properties
_GETTER_PROPS = (
    "CommentChar",
    "Components",
    "Delimiter",
    "File",
    "Folder",
    "Label",
    "Name",
    "Tolerances",
    "Translations",
    "Type",
)
DBTargetOpts.add_getters(_GETTER_PROPS)


# Collection of databook targets
class DBTargetCollectionOpts(OptionsDict):
  # ================
  # Class Attributes
  # ================
  # <
    # No attbitues
    __slots__ = ()

    # Section classes
    _sec_cls_opt = "Type"
    _sec_cls_optmap = {
        "_default_": DBTargetOpts,
    }
  # >

  # ==================
  # Config
  # ==================
  # <
    # Check component exists
    def assert_DataBookTarget(self, targ: str):
        r"""Ensure *comp* is in the list of ``"DataBook"`` components

        :Call:
            >>> opts.assert_DataBookTarget(comp)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *targ*: :class:`str`
                Name of databook target
        :Versions:
            * 2023-03-12 ``@ddalle``: v1.0
        """
        # Check validity of component
        if targ not in self:
            raise ValueError("No DataBook Target named '%s'" % targ)
   # >

  # =======================
  # Class methods
  # =======================
  # <
    @classmethod
    def add_targgetters(cls, optlist, prefix=None, name=None, doc=True):
        r"""Add list of getters for DBTarget properties

        :Call:
            >>> cls.add_targgetters(optlist, prefix=None, name=None)
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
            * 2023-03-12 ``@ddalle``: v1.0
        """
        for opt in optlist:
            cls.add_targgetter(opt, prefix=prefix, name=name, doc=doc)

    @classmethod
    def add_targgetter(cls, opt: str, prefix=None, name=None, doc=True):
        r"""Add getter method for databook target option *opt*

        For example ``cls.add_property("a")`` will add a function
        :func:`get_a`, which has a signatures like
        :func:`OptionsDict.get_opt` except that it doesn't have the
        *opt* input.

        :Call:
            >>> cls.add_targgetter(opt, prefix=None, name=None)
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
            * 2022-11-08 ``@ddalle``: v1.0
        """
        # Check if acting on original OptionsDict
        cls._assert_subclass()
        # Default name
        name, fullname = cls._get_funcname(opt, name, prefix)
        funcname = "get_" + fullname

        # Define function
        def func(self, targ: str, j=None, i=None, **kw):
            try:
                return self._get_opt_targ(targ, opt, j=j, i=i, **kw)
            except Exception:
                raise

        # Generate docstring if requrested
        if doc:
            func.__doc__ = cls._genr8_targg_docstring(opt, name, prefix)
        # Modify metadata of *func*
        func.__name__ = funcname
        func.__qualname__ = "%s.%s" % (cls.__name__, funcname)
        # Save function
        setattr(cls, funcname, func)

    @classmethod
    def _genr8_targg_docstring(cls, opt: str, name, prefix, indent=8, tab=4):
        r"""Create automatic docstring for DBTarget getter function

        :Call:
            >>> txt = cls._genr8_targg_docstring(opt, name, prefx, **kw)
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
            * 2023-03-12 ``@ddalle``: v1.0
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
            "%s>>> %s = opts.get_%s(targ, i=None, **kw)\n"
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
            tab2 + "*targ*: :class:`str`\n" +
            tab3 + "Name of databook target\n" +
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
    # Generic target option getter
    def _get_opt_targ(self, targ: str, opt: str, **kw):
        r"""Get an option from a specific databook target

        :Call:
            >>> v = opts._get_opt_targ(targ, opt, **kw)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *targ*: :class:`str`
                Name of DataBook target
            *opt*: :class:`str`
                Name of option to access
        :Outputs:
            *v*: :class:`object`
                Value of *opt* from specific target
        :Versions:
            * 2023-03-12 ``@ddalle``: v1.0
        """
        # No phases for databook
        kw["j"] = None
        # Assert target exists
        self.assert_DataBookTarget(targ)
        # Use cascading options
        return self.get_subopt(targ, opt, **kw)
  # >

  # ======================
  # Special Properties
  # ======================
  # <
    # Get a target by name
    def get_DataBookTargetByName(self, name: str):
        r"""Get a data book target by *Name*, using user-defined name

        :Call:
            >>> topts = opts.get_DataBookTargetByName(name)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *name*: :class:`str`
                Name of the data book target
        :Outputs:
            *topts*: :class:`DBTargetOpts`
                Databook target options
        :Versions:
            * 2015-12-15 ``@ddalle``: v1.0
            * 2023-03-12 ``@ddalle``: v2.0; use :mod:`optdict`
        """
        # Loop through candidates
        for targ in self:
            # Get name
            targ_name = self.get_DataBookTargetName(targ)
            # Check for match
            if targ_name == name:
                return self[targ]
        # If reaching this point, no target found
        raise KeyError("There is no DBTarget named '%s'" % name)

    # Get "name", falling back to *targ*
    def get_DataBookTargetName(self, targ: str, **kw):
        r"""Get *Name* from databook target, falling back to *targ*

        :Call:
            >>> name = opts.get_DataBookTargetName(targ, **kw)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *targ*: :class:`str`
                Name of databook target
        :Outputs:
            *name*: *targ* | :class:`str`
                User-defined *Name* of target or *targ*
        :Versions:
            * 2023-03-12 ``@ddalle``: v1.0
        """
        # Get *Name*, if possible
        name = self._get_opt_targ(targ, "Name", **kw)
        # Use default
        if name is None:
            # Use key from *DataBook* > *Targets*
            return targ
        else:
            # User-defined
            return name

    # Get "Label", falling back to ":"Name"
    def get_DataBookTargetLabel(self, targ: str, **kw):
        r"""Get *Label* from databook target, falling back to *Name*

        :Call:
            >>> lbl = opts.get_DataBookTargetLabel(targ, **kw)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *targ*: :class:`str`
                Name of databook target
        :Outputs:
            *lbl*: *targ* | :class:`str`
                User-defined *Label* of target or *Name* or *targ*
        :Versions:
            * 2023-03-12 ``@ddalle``: v1.0
        """
        # Get *Name*, if possible
        name = self._get_opt_targ(targ, "Label", **kw)
        # Use default
        if name is None:
            # Use *Name* as fallback
            return self.get_DataBookTargetName(targ, **kw)
        else:
            # User-defined
            return name
  # >


# Add getters
_GETTER_PROPS = (
    "CommentChar",
    "Components",
    "Delimiter",
    "File",
    "Folder",
    "Tolerances",
    "Translations",
    "Type",
)
DBTargetCollectionOpts.add_targgetters(_GETTER_PROPS, prefix="DataBookTarget")


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
        * 2014-12-20 ``@ddalle``: v1.0
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
        "Targets",
        "Type",
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
        "Type": str,
    }

    # Allowed values
    _optvals = {
        "Type": {
            "FM",
            "IterPoint",
            "LineLoad",
            "PyFunc",
            "TriqFM",
            "TriqPoint",
        },
    }

    # Defaults
    _rc = {
        "Folder": "data",
        "NMin": 0,
        "NStats": 0,
        "Type": "FM",
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
        "Type": "Default component type",
    }

    # Key defining additional *_xoptlist*
    _xoptkey = "Components"

    # Section map
    _sec_cls = {
        "Targets": DBTargetCollectionOpts,
    }
    _sec_cls_opt = "Type"
    _sec_cls_optmap = {
        "FM": DBFMOpts,
        "IterPoint": DBIterPointOpts,
        "LineLoad": DBLineLoadOpts,
        "PyFunc": DBPyFuncOpts,
        "TriqFM": DBTriqFMOpts,
        "TriqPoint": DBTriqPointOpts,
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
    def get_CompTargets(self, comp: str):
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
            * 2014-12-21 ``@ddalle``: v1.0
        """
        # Assert *comp* is valid
        self.assert_DataBookComponent(comp)
        # Check if present
        if comp not in self:
            return {}
        # If present, use subopt
        return self.get_subopt(comp, "Targets", vdef={})
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
            * 2022-11-08 ``@ddalle``: v1.0
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
            * 2022-11-08 ``@ddalle``: v1.0
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
            * 2022-10-03 ``@ddalle``: v1.0
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
            * 2022-11-08 ``@ddalle``: v1.0
            * 2022-12-14 ``@ddalle``: v2.0; get_subopt()
            * 2023-03-10 ``@ddalle``: v2.1; cleaner *comp* check
        """
        # No phases for databook
        kw["j"] = None
        # Check for *comp*
        if comp is None:
            # Get option from global
            return self.get_opt(opt, **kw)
        # Assert component exists
        self.assert_DataBookComponent(comp)
        # Check if it's an implicit component
        if comp not in self:
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
            * 2023-01-22 ``@ddalle``: v1.0
        """
        # Check component exists
        self.assert_DataBookComponent(comp)
        # Get suboption
        compid = self.get_subopt(comp, "CompID", **kw)
        # Check for null result
        if compid is None:
            # Default is name of component
            return comp
        else:
            # Return nontrivial result
            return compid

    # Check component exists
    def assert_DataBookComponent(self, comp: str):
        r"""Ensure *comp* is in the list of ``"DataBook"`` components

        :Call:
            >>> opts.assert_DataBookComponent(comp)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of databook component
        :Versions:
            * 2023-03-10 ``@ddalle``: v1.0
        """
        # Check validity of component
        if comp not in self.get_DataBookComponents():
            raise ValueError("No DataBook component named '%s'" % comp)
  # >

  # ================
  # Component Config
  # ================
  # <
    # Get data book components by type
    def get_DataBookByType(self, typ: str) -> list:
        r"""Get the list of data book components with a given type

        :Call:
            >>> comps = opts.get_DataBookByType(typ)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *typ*: ``"FM"`` | ``"LineLoad"`` | :class:`str`
                Data book type
        :Outputs:
            *comps*: :class:`list`\ [:class:`str`]
                List of components with ``"Type"`` matching *typ*
        :Versions:
            * 2016-06-07 ``@ddalle``: v1.0
            * 2023-03-09 ``@ddalle``: v1.1; validate *typ*
        """
        # Validate input
        self.validate_DataBookType(typ)
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
    def get_DataBookByGlob(self, typ, pat=None):
        r"""Get list of components by type and list of wild cards

        :Call:
            >>> comps = opts.get_DataBookByGlob(typ, pat=None)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *typ*: ``"FM"`` | :class:`str`
                Target value for ``"Type"`` of matching components
            *pat*: {``None``} | :class:`str` | :class:`list`
                List of component name patterns
        :Outputs:
            *comps*: :class:`str`
                All components meeting one or more wild cards
        :Versions:
            * 2017-04-25 ``@ddalle``: v1.0
            * 2023-02-06 ``@ddalle``: v1.1; improved naming
            * 2023-03-09 ``@ddalle``: v1.2; validate *typ*
        """
        # Get list of all components with matching type
        comps_all = self.get_DataBookByType(typ)
        # Check for default option
        if pat is None:
            return comps_all
        # Initialize output
        comps = []
        # Ensure input is a list
        if isinstance(pat, ARRAY_TYPES):
            # Already a list
            pats = pat
        else:
            # Read as string: comma-separated list
            pats = pat.split(",")
        # Loop through components to check if it matches
        for comp in comps_all:
            # Loop through components
            for pat in pats:
                # Check if it matches
                if fnmatch.fnmatch(comp, pat):
                    # Add the component to the list
                    comps.append(comp)
                    break
        # Output
        return comps

    # Validate type
    def validate_DataBookType(self, typ: str):
        r"""Ensure that *typ* is a recognized DataBook *Type*

        :Call:
            >>> opts.validate_DataBookType(typ)
        :Inputs:
            *opts*: :class:`cape.cfdx.options.Options`
                Options interface
            *typ*: ``"FM"`` | :class:`str`
                Target value for ``"Type"`` of matching components
        :Raises:
            :class:`ValueError`
        :Versions:
            * 2023-03-09 ``@ddalle``: v1.0
        """
        # Check value
        if typ not in self.__class__._sec_cls_optmap:
            raise TypeError(f"Unrecognized DabaBook type '{typ}'")

    # Get the coefficients for a specific component
    def get_DataBookCoeffs(self, comp):
        r"""Get the list of data book coefficients for a specific component

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
            * 2014-12-21 ``@ddalle``: v1.0
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
            * 2016-03-15 ``@ddalle``: v1.0
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
        r"""Get additional numeric columns for component (other than coeffs)

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
            * 2016-03-15 ``@ddalle``: v1.0
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
        r"""Get integer columns for component

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
            * 2016-03-15 ``@ddalle``: v1.0
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
        r"""Get the full list of data book columns for a specific component

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
            * 2014-12-21 ``@ddalle``: v1.0
        """
        # Data columns (from CFD)
        dcols = self.get_DataBookDataCols(comp)
        # Output
        return dcols

    # Get full list of data columns for a specific component
    def get_DataBookDataCols(self, comp):
        r"""Get the list of data book columns for a specific component

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
            * 2014-12-21 ``@ddalle``: v1.0
            * 2022-04-08 ``@ddalle``: v2.0; cooeff-spec suffixes
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
        r"""Get the list of data book target columns for a specific component

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
            * 2014-12-21 ``@ddalle``: v1.0
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
        r"""
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
            * 2014-12-22 ``@ddalle``: v1.0
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
            * 2016-06-07 ``@ddalle``: v1.0
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
            * 2017-03-29 ``@ddalle``: v1.0
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
            * 2016-06-07 ``@ddalle``: v1.0
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
    "Type",
)
DataBookOpts.add_compgetters(_GETTER_PROPS, prefix="DataBook")

# Normal top-level properties
_PROPS = (
    "Components",
    "Folder",
)
DataBookOpts.add_properties(_PROPS, prefix="DataBook")

# Normal top-level get-only
_GETTER_PROPS = (
    "Targets",
)
DataBookOpts.add_getters(_GETTER_PROPS, prefix="DataBook")

# Upgrade subsections
DataBookOpts.promote_sections()
