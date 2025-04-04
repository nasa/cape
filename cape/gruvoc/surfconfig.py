r"""
:mod:`gruvoc.surfconfig`: Surface name, ID, and property management
====================================================================

This module provides an interface to surface grid configuration control.
Perhaps the primary goal is simply to associate a name with each ID
number in a surface grid.
"""


# Standard library
import os.path as op
import re
from typing import Any, Optional, Union

# Third-party
import numpy as np
from numpy import int32, ndarray

# Local imports
from .errors import GruvocValueError, assert_isinstance, assert_isfile
from .fileutils import tail
from ._vendor.optdict import OptionsDict


# List of file types
FACE_CONFIG_TYPES = (
    "mapbc",
    "uh3d",
)
FULL_CONFIG_TYPES = (
    "xml",
    "json",
    "mixur",
)
CONFIG_TYPES = FACE_CONFIG_TYPES + FULL_CONFIG_TYPES

# Integer regular expression
REGEX_POSINT = re.compile("[0-9]+")
REGEX_INT = re.compile("[+-]?[0-9]+")

# Integer types
INT_TYPES = (int, np.integer)


# Options for JSON config files
class JSONOpts(OptionsDict):
    # No attributes
    __slots__ = ()

    # Allowed options
    _optlist = (
        "Tree",
        "Properties",
    )

    # Aliases
    _optmap = {
        "Props": "Properties",
        "properties": "Properties",
        "props": "Properties",
        "tree": "Tree",
    }

    # Types
    _opttypes = {
        "Tree": dict,
        "Properties": dict,
    }


# Surface configuration class
class SurfConfig(object):
    r"""Surface properties and configurations

    :Call:
        >>> cfg = SurfConfig(fname=None, **kw)
    :Outputs:
        *cfg*: :class:`SurfConfig`
            Surface configuration
    :Attributes:
        * :attr:`faces`
        * :attr:`fdir`
        * :attr:`fname`
        * :attr:`parents`
        * :attr:`props`
    """
  # === Basic ===
   # --- Class attributes ---
    __slots__ = (
        "facenames",
        "faces",
        "fdir",
        "fname",
        "parents",
        "props",
        "tree",
    )

   # --- __dunder__ methods ---
    def __init__(self, fname: Optional[str] = None, **kw):
        # Initialize slots
        #: :class:`list`\ [:class:`str`]
        #: Ordered list of faces
        self.faces = []
        #: :class:`dict`\ [:class:`list`\ [:class:`str`]]
        #: Lists of direct children of each family component
        self.tree = {}
        #: :class:`str` | ``None``
        #: Base name of file from which configuration was read
        self.fname = None
        #: :class:`str` | ``None``
        #: Absolute path to folder containing source file
        self.fdir = None
        # Optional slots
        self.facenames = None
        #: :class:`dict`\ [:class:`list`\ [:class:`str`]]]
        #: List of parent components of each component
        self.parents = {}
        #: :class:`dict`\ [:class:`dict`]
        #: Properties for each *face* or *family*
        self.props = {}
        # Read generic file type
        if fname is not None:
            self.read(fname)
            return
        # Check for kwargs
        fjson = kw.get("json")
        fmapbc = kw.get("mapbc")
        fmixsur = kw.get("mixsur")
        fuh3d = kw.get("uh3d")
        fxml = kw.get("xml")
        # Read individual
        if fjson:
            self.read_json(fjson)
        elif fmapbc:
            self.read_mapbc(fmapbc)
        elif fxml:
            self.read_xml(fxml)
        elif fmixsur:
            self.read_mixsur(fmixsur)
        elif fuh3d:
            self.read_uh3d(fuh3d)

   # --- Support ---
    def _save_fname(self, fname: str):
        # Absolutize
        fabs = fname if op.isabs(fname) else op.abspath(fname)
        # Save parts
        self.fdir, self.fname = op.split(fabs)

  # === Read ===
   # --- Main reader ---
    def read(self, fname: str):
        # Identify the configuration type
        ext = identify_config_filetype(fname)
        # Read if applicable
        if ext == "mapbc":
            self.read_mapbc(fname)
        elif ext == "uh3d":
            self.read_uh3d(fname)
        elif ext == "json":
            self.read_json(fname)

   # --- Format-specific ---
    def read_json(self, fname: str):
        r"""Read surface configuration from JSON file

        :Call:
            >>> cfg.read_json(fname)
        :Inputs:
            *cfg*: :class:`SurfConfig`
                Surface component configuration instance
            *fname*: :class:`str`
                Name of file to read
        """
        # Read the file
        opts = JSONOpts(fname)
        # Save the tree
        self.tree = opts["Tree"]
        self.props = opts["Properties"]
        # Set parent for each element as listed in *Tree*
        for parent, children in self.tree.items():
            # Loop through listed children
            for child in children:
                self.add_parent(parent, child)

    def read_mapbc(self, fname: str):
        # Ensure file exists
        assert_isfile(fname)
        # Initialize parts
        names = self.make_facenames()
        # Save file name
        self._save_fname(fname)
        # Read the lines of the file
        lines = open(fname).readlines()
        # Loop through remaining lines
        for j, line in enumerate(lines):
            # Ignore first line (should be number of faces)
            if j == 0:
                continue
            # Split into parts
            parts = line.split()
            # Validate
            if len(parts) != 3:
                raise GruvocValueError(
                    f"In line {j+1} of mapbc file '{self.fname}'; " +
                    f"expected 3 parts but got {len(parts)}")
            # Split the parts to individual variables
            raw_id, raw_bc, face = parts
            # Validate numeric
            assert_posint(raw_id, f"Line {j+1} ID of '{fname}', '{raw_id}',")
            assert_posint(raw_bc, f"Line {j+1} BC of '{fname}', '{raw_bc}',")
            # Get/initialize properties for this face
            props = self._add_props(face)
            # Save two properties
            props["CompID"] = int(raw_id)
            props["bc"] = int(raw_bc)
            # Save name
            names[int(raw_id)] = face
            # Add to list of faces
            self.faces.append(face)

    def read_uh3d(self, fname: str):
        # Ensure file exists
        assert_isfile(fname)
        # Save file name
        self._save_fname(fname)
        # Open file
        with open(fname, 'r') as fp:
            # Discard first line
            fp.readline()
            # Read counts
            line = fp.readline()
            # Number of components
            raw_ncomp = line.strip().split(',')[-1].strip()
            # Validate count
            assert_posint(raw_ncomp, f"Number of comps on line 2 of '{fname}'")
        # Convert components to int
        ncomp = int(raw_ncomp)
        # Read last *ncomp*
        lines = tail(fname, ncomp + 1).strip().split("\n")
        # Loop through number of comps
        for j, line in enumerate(lines[:ncomp]):
            # Split into parts
            parts = line.strip().split(', ', maxsplit=1)
            # Split into raw ID number and name
            raw_id, face = parts
            # Validate numeric
            assert_posint(raw_id, f"Line {j+1} ID of '{fname}', '{raw_id}',")
            # Save face
            self.add_face(face, int32(raw_id))

  # === Data ===
   # --- From CompID ---
    def get_name(self, surf_id: int) -> str:
        r"""Find face name from a single component ID

        :Call:
            >>> face = cfg.get_name(surf_id)
        :Inputs:
            *cfg*: :class:`SurfConfig`
                Surface component configuration instance
            *surf_id*: :class:`int`
                Surface index
        """
        # Check input
        assert_isinstance(surf_id, INT_TYPES, "face ID number")
        # Get dictionary of face names
        names = self.make_facenames()
        # Get surface if present
        return names.get(surf_id, f"boundary {surf_id}")

    def make_facenames(self) -> dict:
        # Check if present
        if self.facenames is not None:
            return self.facenames
        # Create if necessary
        return self.create_facenames()

    def create_facenames(self) -> dict:
        # Initialize dictionary
        names = {}
        # Loop through faces
        for j, face in enumerate(self.faces):
            # Get CompID
            surf_id = self.get_prop(face, "CompID")
            # Validate
            assert_isinstance(
                surf_id, INT_TYPES, f"CompID for face {j+1}, '{face}'")
            # Save
            names[surf_id] = face
        # Save
        self.facenames = names
        # Output
        return names

   # --- From name ---
    def get_surf_ids(self, comp: str) -> ndarray:
        # Get ID for top-level zone (if any)
        surf_id = self.get_prop(comp, "CompID")
        # Initialize
        surf_ids = [] if surf_id is None else [surf_id]
        # Find children of *comp*, if applicable
        child_comps = self.get_children(comp)
        # Exit if no children
        if child_comps is None:
            return np.array(surf_ids, dtype="int32")
        # Loop through children
        for child_comp in child_comps:
            # Recurse into *child_comp*
            child_ids = self.get_surf_ids(child_comp)
            # Combine
            surf_ids.extend(child_ids)
        # Return sorted unique array
        return np.unique(surf_ids)

   # --- Components ---
    def get_comp_ids(self, comp: str) -> list:
        r"""Get list of component IDs within a component, expanding tree

        :Call:
            >>> comp_ids = cfg.get_comp_ids(comp)
        :Inputs:
            *cfg*: :class:`SurfConfig`
                Surface component configuration instance
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *comp_ids*: :class:`list`\ [:class:`int`]
                List of component IDs in *comp* and its children
        """
        # Get initial component
        compid = self.get_prop_comp(comp, "CompID")
        # Initialize list
        comp_ids = [] if compid is None else [compid]
        # Get tree
        for child in self.tree.get(comp, []):
            # Get component IDs of children (recursive)
            child_ids = self.get_comp_ids(child)
            # Combine
            comp_ids.extend(child_ids)
        # Output
        return sorted(comp_ids)

    def get_complist(self) -> list:
        r"""Get full list components from *tree* and *props*

        :Call:
            >>> complist = cfg.get_complist()
        :Inputs:
            *cfg*: :class:`SurfConfig`
                Surface component configuration instance
        :Outputs:
            *complist*: :class:`list`\ [:class:`str`]
                List of components mentioned in configuration
        """
        # Initialize set of components
        compset = set()
        # Loop through tree
        for parent, children in self.tree.items():
            # Add both
            compset.add(parent)
            compset.update(children)
        # Add in any properties not in tree
        compset.update(self.props.keys())
        # Output
        return sorted(list(compset))

   # --- Tree ---
    def get_children(self, comp: str) -> list:
        # Check inputs
        assert_isinstance(comp, str)
        self.assert_comp(comp)
        # Check if it's in the tree
        if comp not in self.tree:
            return []
        # Initialize
        children = list(self.tree[comp])
        # Recurse
        for child in list(children):
            # Get children thereof
            grandchildren = self.get_children(child)
            # Append
            for grandchild in grandchildren:
                if grandchild not in children:
                    children.append(grandchild)
        # Output
        return children

    def add_parent(self, parent: str, child: str):
        r"""Add a parent-child relationship for two named components

        :Call:
            >>> cfg.add_parent(parent, child)
        :Inputs:
            *cfg*: :class:`SurfConfig`
                Surface component configuration instance
            *parent*: :class:`str`
                Name of parent component
            *child*: :class:`str`
                Name of child component
        :Attributes:
            :attr:`parents`: *parent* added to ``cfg.parents[child]``
        """
        # Initialize
        parents = self.parents.setdefault(child, [])
        # Check if *parent* is already a parent
        if parent in parents:
            return
        # Check for previous parents
        if len(parents):
            # Get property
            primary_parent = self.get_prop(child, "Parent")
            # Check if this is the priority parent
            if primary_parent == parent:
                # Place this parent in position 0
                parents.insert(0, parent)
                return
        # Append parent in order listed in *Tree*
        parents.append(parent)

    def _get_sorting_id(self, comp: str) -> Union[int, float]:
        # Check for direct listing
        id0 = self.get_prop_comp(comp, "CompID")
        # Use it if found
        if id0 is not None:
            return id0
        # Get children
        children = self.get_children(comp)
        # Check for any
        if len(children) == 0:
            return float(np.nan)
        # Get IDs for each
        children_ids = [self._get_sorting_id(child) for child in children]
        # Use minimum
        return np.nanmin(children_ids) - 0.01

   # --- Removal ---
    def renumber_comps(self):
        r"""Renumber used components 1, 2, ...

        :Call:
            >>> cfg.renumber_comps()
        :Inputs:
            *cfg*: :class:`SurfConfig`
                Surface component configuration instance
        """
        # Clear parents
        self.clear_parent_comp_ids()
        # Initialize list of tuples: (comp, compid)
        complist = []
        # Loop through properties
        for comp, prop in self.props.items():
            # Check for *CompID*
            compid = prop.get("CompID")
            # Save to list if found
            if compid is not None:
                complist.append((comp, compid))
        # Sort the list by ascending CompID
        complist.sort(key=lambda x: x[1])
        # Renumber
        for jb, (comp, ja) in enumerate(complist):
            # Reset component list
            self.props[comp]["OldCompID"] = ja
            self.props[comp]["CompID"] = jb + 1

    def apply_comp_dict(self, configdict: dict):
        r"""Renumber components to match dict and remove unused comps

        :Call:
            >>> cfg.apply_comp_dict(configdict)
        :Inputs:
            *cfg*: :class:`SurfConfig`
                Surface component configuration instance
            *configdict*: :class:`dict`\ [:class:`int`]
                Dictionary of components to keep and their final IDs
        """
        # Validate input
        assert_isinstance(configdict, dict, "configuration ID dictionary")
        for comp, compid in configdict.items():
            assert_isinstance(
                compid, INT_TYPES, f"target component ID for '{comp}'")
        # Loop through properties
        for comp, props in self.props.items():
            # Check if this is a target ID
            if comp in configdict:
                # Set ID
                props["CompID"] = configdict[comp]
            else:
                # Remove the CompID if any is present
                props.pop("CompID", None)
        # Get list of IDs
        config_ids = list(configdict.values())
        # Remove unused components
        self.restrict_comp_ids(config_ids)

    def clear_parent_comp_ids(self):
        r"""Remove *CompID* setting from any comp that is a parent

        This means that only the base-level components are numbered.

        :Call:
            >>> cfg.clear_parent_comp_ids()
        :Inputs:
            *cfg*: :class:`SurfConfig`
                Surface component configuration instance
        """
        # Loop through parents
        for parent in self.tree:
            # Get properties
            if parent not in self.props:
                continue
            # Delete CompID
            self.props[parent].pop("CompID", None)

    def restrict_comp_ids(self, comp_ids: Union[list, ndarray]):
        r"""Restrict configuration to comps with specified IDs

        :Call:
            >>> cfg.restrict_comp_ids(comp_ids)
        :Inputs:
            *cfg*: :class:`SurfConfig`
                Surface component configuration instance
            *comp_ids*: :class:`list`\ [:class:`int`]
                List/array of CompIDs to keep
        """
        # Make sure to reduce the size of target ID list
        unique_ids = np.unique(comp_ids)
        # Loop through all components
        for comp in self.get_complist():
            # Get components thereof
            ids = self.get_comp_ids(comp)
            # Check for intersection
            if len(np.intersect1d(ids, unique_ids)) == 0:
                self.remove_comp(comp)
            # If component has children, check the comp-specific ID
            compid = self.get_prop_comp(comp, "CompID")
            # Check if present
            if (compid is not None) and not np.any(unique_ids == compid):
                # Remove it
                self.props[comp].pop("CompID")

    def remove_comp(self, comp: str):
        r"""Remove a component from configuration, including tree

        :Call:
            >>> cfg.remove_comp(comp)
        :Inputs:
            *cfg*: :class:`SurfConfig`
                Surface component configuration instance
            *comp*: :class:`str`
                Name of component to remove
        :Attributes:
            *comp* removed from both :attr:`props` and :attr:`tree`
        """
        # Check if *comp* is in *props*
        if comp in self.props:
            self.props.pop(comp)
        # Check for parents of *comp*
        parents = self.parents.pop(comp, [])
        # Remove from tree of all parents
        for parent in parents:
            self.tree[parent].remove(comp)
        # Check if it's in the tree
        if comp not in self.tree:
            return
        # Get children and remove from tree
        children = self.tree.pop(comp)
        # Exit if no parents
        if len(parents) == 0:
            return
        # Get primary
        grandparent = parents[0]
        # Get tree thereof
        tree = self.tree[grandparent]
        # Remove parent from tee of grandparent
        if comp in tree:
            tree.remove(comp)
        # Add children to the tree of the first parent
        self.tree[grandparent].extend(children)
        # Update the parent of each child; grandparent becomes parent
        for child in children:
            # Index of *comp*
            i = self.parents[child].index(comp)
            # Replace
            self.parents[child][i] = grandparent

   # --- Properties ---
    def get_prop(self, comp: str, opt: str) -> Any:
        r"""Get any property for one component, including from parents

        :Call:
            >>> val = cfg.get_prop(comp, opt)
        :Inputs:
            *cfg*: :Class:`SurfConfig`
                Surface configuration
            *comp*: :class:`str`
                Name of component (face/family)
            *opt*: :class:`str`
                Name of poperty to set
        :Outputs:
            *val*: :class:`object`
                Value to set for that property
        """
        # Get properties dict
        props = self.props.get(comp, {})
        # Get property
        val = props.get(opt)
        # Use it
        if val is not None:
            return val
        # Seek parents
        for parent in self.parents.get(comp, []):
            # Recurse
            val = self.get_prop(parent, opt)
            # Use it if able
            if val is not None:
                return val

    def get_prop_comp(self, comp: str, opt: str) -> Any:
        r"""Get any property for one component, no recursion

        :Call:
            >>> val = cfg.get_prop(comp, opt)
        :Inputs:
            *cfg*: :Class:`SurfConfig`
                Surface configuration
            *comp*: :class:`str`
                Name of component (face/family)
            *opt*: :class:`str`
                Name of poperty to set
        :Outputs:
            *val*: :class:`object`
                Value to set for that property
        """
        # Get properties dict
        props = self.props.get(comp, {})
        # Get property
        return props.get(opt)

    def set_prop(self, comp: str, opt: str, val: Any):
        r"""Set any property for one component

        :Call:
            >>> cfg.set_prop(comp, opt, val)
        :Inputs:
            *cfg*: :Class:`SurfConfig`
                Surface configuration
            *comp*: :class:`str`
                Name of component (face/family)
            *opt*: :class:`str`
                Name of poperty to set
            *val*: :class:`object`
                Value to set for that property
        """
        # Get properties
        props = self._add_props(comp)
        # Set property
        props[opt] = val

    def add_face(self, comp: str, comp_id: int):
        r"""Add a new face with specified ID number

        :Call:
            >>> cfg.add_face(comp, comp_id)
        :Inputs:
            *cfg*: :Class:`SurfConfig`
                Surface configuration
            *comp*: :class:`str`
                Name of component (face/family)
            *comp_id*: :class:`int`
                ID number for *comp*
        """
        # Check input types
        assert_isinstance(comp, str)
        assert_isinstance(comp_id, INT_TYPES)
        # Check if *comp* is already a face
        if comp not in self.faces:
            # Save it
            self.faces.append(comp)
        # Get or create properties
        props = self._add_props(comp)
        # Set component ID number
        props["CompID"] = comp_id

    def _add_props(self, comp: str) -> dict:
        r"""Get all properties for one face; add it if not present

        :Call:
            >>> props = cfg._add_props(comp)
        :Inputs:
            *cfg*: :Class:`SurfConfig`
                Surface configuration
            *comp*: :class:`str`
                Name of component (face/family)
        :Outputs:
            *props*: :class:`dict`
                Properties for *comp*
        """
        return self.props.setdefault(comp, {})

   # --- Errors ---
    def assert_comp(self, comp: str):
        r"""Test of *comp* is in the configuration

        :Call:
            >>> cfg.assert_comp(comp)
        :Inputs:
            *cfg*: :Class:`SurfConfig`
                Surface configuration
            *comp*: :class:`str`
                Name of component (face/family)
        :Raises:
            :class:`GruvocValueError` if *comp* is not present
        """
        # Check if *comp* is valid
        if not ((comp in self.tree) or (comp in self.props)):
            raise GruvocValueError(f"No comp named '{comp}' in {self.fname}")


# Identify file extension
def identify_config_filetype(fname: str) -> str:
    r"""Identify the probable file format based on a file name

    :Call:
        >>> ext = identify_config_filetype(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read/write
    :Ouputs:
        *ext*: :class:`str`
            Probable file type
    """
    # Check types
    assert_isinstance(fname, str, "name of config file")
    # Get file extension
    ext = fname.split(".")[-1]
    # Check against known types
    if ext in CONFIG_TYPES:
        # Valid
        return ext
    # Special cases
    if ext == "i" and fname.startswith("mixsur."):
        return "mixsur"
    # Otherwise not detected
    raise GruvocValueError(
        f"Could not identify config type from file name '{fname}'")


def assert_posint(txt: str, description: str):
    if REGEX_POSINT.fullmatch(txt) is None:
        raise GruvocValueError(f"{description} is not a positive integer")
