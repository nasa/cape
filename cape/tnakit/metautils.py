# -*- coding: utf-8 -*-
r"""
This module provides various utilities for creating databases of
metadata.  For example, the :class:`ModulePropDB` class can be used to
collect properties for each module in a package.
"""

# Standard library
import os
import re
import json

# Specific imports
from collections import OrderedDict

# Local modules
from . import typeutils


# Local function to perform merger of two dicts
# This is a function so that it can be recursed
def merge_dict(opts1, opts2):
    """Merge two dictionaries, using value from *opts1* in conflict

    :Call:
        >>> merge_dict(opts1, opts2)
    :Inputs:
        *opts1*: :class:`dict`
            First dictionary
        *opts2*: :class:`dict`
            Second dictionary
    :Versions:
        * 2019-04-07 ``@ddalle``: First version
    """
    # Loop through options of input dict
    for k, v in opts2.items():
        # Check if *k* is in *self*
        if k not in opts1:
            # New value
            opts1[k] = v
            continue
        # Otherwise check type of input value
        if not isinstance(v, dict):
            # Save non-dict value
            opts1[k] = v
            continue
        # Get current value
        u = opts1[k]
        # Then check type of existing value
        if not isinstance(u, dict):
            # Replace non-dict with dict
            opts1[k] = v
            continue
        # Otherwise, they are both dicts; recurse
        merge_dict(u, v)


# Local function to perform merger of two dicts
# This is a function so that it can be recursed
def merge_dict_default(opts1, opts2):
    """Merge two dictionaries, using value from *opts2* in conflict

    :Call:
        >>> merge_dict_default(opts1, opts2)
    :Inputs:
        *opts1*: :class:`dict`
            First dictionary
        *opts2*: :class:`dict`
            Second dictionary
    :Versions:
        * 2019-04-07 ``@ddalle``: First version
    """
    # Loop through options of input dict
    for k, v in opts2.items():
        # Check if *k* is in *self*
        if k not in opts1:
            # New value
            opts1[k] = v
            continue
        # Get current value
        u = opts1[k]
        # Otherwise check type of input value
        if not isinstance(v, dict):
            # Don't replace existing dict with non-dict
            continue
        # Then check type of existing value
        if not isinstance(u, dict):
            # Don't replace existing non-dict with dict
            continue
        # Otherwise, they are both dicts; recurse
        merge_dict_default(u, v)


# Class for single module's metadata
class ModuleMetadata(dict):
    r"""Read metadata from a datakit module

    :Versions:
        * 2021-08-12 ``@ddalle``: Version 0.1; started
    """
    def __init__(self, mod=None, **kw):
        r"""Initialization method

        :Versions:
            * 2021-08-12 ``@ddalle``: Version 1.0
        """
        # Name of JSON file to search for
        fjson = kw.pop("_json", "meta.json")
        # Check class of first argument
        if mod is None:
            # No JSON file to read
            fdir = None
            # No base dictionary
            basedict = None
        elif isinstance(mod, type(os)):
            # Got a module; find the directory
            fdir = os.path.dirname(mod.__file__)
            # No starting-point dict
            basedict = None
        elif typeutils.isstr(mod):
            # Got a string; check if it's a dir
            if os.path.isdir(mod):
                # Already a folder name
                fdir = os.path.abspath(mod)
            elif os.path.isfile(mod):
                # Ensure absolute path
                fabs = os.path.realpath(mod)
                # Split into folder/file
                fdir = os.path.dirname(fabs)
            else:
                # Invalid string
                raise ValueError("Could not find file/folder '%s'" % mod)
            # No starting-point dict
            basedict = None
        elif isinstance(mod, (dict, list, tuple)):
            # No file to read
            fdir = None
            # Got a dictionary
            basedict = mod
        elif typeutils.isfile(mod):
            # Got an open file
            fabs = os.path.abs(mod.name)
            # Split into folder/file
            fdir, fjson = os.path.split(fabs)
        else:
            # Unrecognized type
            raise TypeError(
                "Unable to process input of type '%s'" %
                mod.__class__.__name__)
        # Initialize dictionary if given a valid dict(basedict)
        if basedict is not None:
            dict.__init__(self, basedict)
        # Save JSON file locations
        self.fdir = fdir
        self.fjson = fjson
        # Read the JSON file
        if fdir is None:
            # No file to read
            self.fabs = None
        else:
            # Get absolute path
            self.fabs = os.path.join(fdir, fjson)
            # Check for already-opened file
            if typeutils.isfile(mod):
                # Read from open file
                self.read_json(mod)
            else:
                # Read from file name
                self.read_json(self.fabs)
        # Add any keyword arguments
        self.update(**kw)
        
    # Read a JSON file
    def read_json(self, fjson):
        r"""Read a JSON file into the keys of a metadata object

        :Call:
            >>> meta.read_json(fjson)
            >>> meta.read_json(f)
        :Inputs:
            *meta*: :class:`ModuleMetadata`
                Metadata object for one module
            *fjson*: :class:`str`
                Name of JSON file to read
            *f*: **file**
                Already opened file handle
        :Versions:
            * 2021-08-12 ``@ddalle``: Version 1.0
        """
        # Check for file
        if typeutils.isfile(fjson):
            # Read open file handle
            self._read_json(fjson)
        else:
            # Check for JSON file
            if not os.path.isfile(fjson):
                return
            # Open the file and read it
            with open(fjson, "r") as f:
                self._read_json(f)

    # Read a file handle
    def _read_json(self, f):
        r"""Read a JSON file form its handle

        :Call:
            >>> meta._read_json(f)
        :Inputs:
            *meta*: :class:`ModuleMetadata`
                Metadata object for one module
            *f*: **file**
                Already opened file handle
        :Versions:
            * 2021-08-12 ``@ddalle``: Version 1.0
        """
        # Read the file
        opts = json.load(f)
        # Save the options therein
        self.update(**opts)
        

# Metadata class for module properties
class ModulePropDB(dict):
    """Module properties database
    
    :Call:
        >>> props = ModulePropDB(fname)
        >>> props = ModulePropDB(opts, **kw)
        >>> props = ModulePropDB(optlist, **kw)
    :Inputs:
        *fname*: :class:`str`
            Name of JSON file to read
        *opts*: :class:`dict`
            Dictionary of options to convert to database
        *optlist*: :class:`list` | :class:`tuple`
            Ordered key/value pair list
        *kw*: :class:`dict`
            Keywords to convert to or merge into database
    :Outputs:
        *props*: :class:`ModulePropDB`
            Module property database instance
        *props.settings*: :class:`dict`
            Settings popped from ``".settings"`` key from input
    :Versions:
        * 2019-04-15 ``@ddalle``: First version
    """
    
    # Attribute for settings (separate from metadata)
    settings = {}
    # Name of file
    fjson = None
    
    # Initialization method
    def __init__(self, *a, **kw):
        """Initialization method
        
        :Versions:
            * 2019-04-13 ``@ddalle``: First version
        """
        # Number of args
        na = len(a)
        # Check for nonzero args
        if na > 1:
            raise ValueError("Expecting 0 or 1 args; received %i" % na)
        elif na > 0:
            # Get first arg and its type
            a0 = a[0]
            # Check for a string
            if typeutils.isstr(a0):
                # Otherwise, read the file
                self.read_json(a0)
                return
            elif isinstance(a0, dict):
                # Received a dict
                A = (a0,)
            elif isinstance(a0, (list, tuple)):
                # Create a dict (cannot use :func:`dict.__init__`)
                A = (dict(a0),)
            else:
                raise TypeError(
                    ("ModulePropDB '__init__' requires a ") +
                    ("'str', 'dict', or 'list' ") +
                    ("but received '%s'" % a0.__class__.__name__))
        else:
            # No inputs
            A = tuple()
        # If reaching this point, get settings
        dict.__init__(self, *A, **kw)
        # Remove settings if appropriate
        settings = self.pop(".settings", {})
        # Save those as settings
        self.settings = settings
        

    # Read from JSON file
    @classmethod
    def from_json(cls, fname):
        """Create metadata instance from JSON file

        :Call:
            >>> meta = MetaData.from_json(fname)
        :Inputs:
            *props*: :class:`ModulePropDB`
                Module property database instance
            *fname*: :class:`str`
                Name of JSON file
        :Versions:
            * 2019-04-07 ``@ddalle``: First versoin
        """
        # Initialize empty instance
        meta = cls()
        # Read JSON file
        meta.read_json(fname)
        # Return the object
        return meta
        
    # List modules
    def list_modules(self):
        """Get a list of modules in appropriate order
        
        :Call:
            >>> mods = props.list_modules()
        :Inputs:
            *props*: :class:`ModulePropDB`
                Module property database interface instance
        :Outputs:
            *mods*: :class:`list` (:class:`str`)
                List of modules in database, sorted in appropriate order
        :Versions:
            * 2019-04-14 ``@ddalle``: First version
        """
        # Get keys of database
        keys = self.keys()
        # Remove all keys starting with "."
        for k in keys:
            if k.startswith("."):
                keys.pop(k)
        # Return sorted version of what's left
        return sorted(keys)
        
    # Get ordered database for a particular module
    def get_ordered_db(self, mod):
        """Get an :class:`OrderedDict` database for one module
        
        :Call:
            >>> moddb = props.get_ordered_db(mod)
        :Inputs:
            *props*: :class:`ModulePropDB`
                Module property database interface instance
            *mod*: :class:`str`
                Name of module to sort
        :Outputs:
            *moddb*: :class:`OrderedDict`
                Properties for module *mod*
        :Versions:
            * 2019-04-14 ``@ddalle``: First version
        """
        # Initialize
        moddb = OrderedDict()
        # Get existing :class:`dict` for this module
        opts = self[mod]
        # Get key order from *.settings*
        keys = self.settings["Keys"] + sorted(opts.keys())
        # Sort keys of *opts* according to order specified in *.settings*
        for k in sorted(opts.keys(), key=keys.index):
            # Save the property to :class:`OrderedDict`
            moddb[k] = opts[k]
        # Output
        return moddb


    # Get ordered settings
    def get_ordered_settings(self):
        """Get ordered version of metadata *settings* attribute
        
        :Call:
            >>> settings = props.get_ordered_settings()
        :Inputs:
            *props*: :class:`ModulePropDB`
                Module property database instance
        :Outputs:
            *settings*: :class:`OrderedDict`
                Properly ordered settings
        :Versions:
            * 2019-09-27 ``@ddalle``: First version
        """
        # Initialize settings
        settings = OrderedDict()
        # List of keys
        key_list = self.settings["Keys"]
        # Save settings in order
        settings["Keys"] = list(key_list)
        # Loop through other keys
        for k in sorted(self.settings.keys()):
            # Skip "Keys" entry
            if k == "Keys":
                continue
            # Get value from .settings
            v = self.settings[k]
            # Check type
            if isinstance(v, dict):
                # Order the dictionary by *key_list* order
                settings[k] = OrderedDict((
                    (kj, v[kj]) for kj in key_list if kj in v))
            else:
                # Save key as is
                settings[k] = v
        # Output
        return settings

        
    # Return a property from one module, using defaults
    def get_property(self, mod, k, vdef=None):
        """Return a property from one module, using defaults
        
        :Call:
            >>> v = props.get_property(mod, k, vdef=None)
        :Inputs:
            *props*: :class:`ModulePropDB`
                Module property database interface instance
            *mod*: :class:`str`
                Name of module to sort
            *k*: :class:`str`
                Name of property/key to return
            *vdef*: {``None``} | :class:`str` | :class:`bool`
                Default value if not present in DB or defaults
        :Outputs:
            *v*: :class:`str` | :class:`bool` | ``None`` | :class:`dict`
                Properties for module *mod*
        :Versions:
            * 2019-04-17 ``@ddalle``: First version
        """
        # Get database for module *mod*
        moddb = self.get(mod, {})
        # Check if present
        if k in moddb:
            return moddb[k]
        # Get defaults
        defs = self.settings.get("Defaults", {})
        # Get property from defaults
        return defs.get(k, vdef)
        
    # Read a file
    def read_json(self, fname):
        """Read a JSON metadata file

        :Call:
            >>> props.read_json(fname)
        :Inputs:
            *props*: :class:`ModulePropDB`
                Module property database instance
            *fname*: :class:`str`
                Name of JSON file
        :Versions:
            * 2019-04-06 ``@ddalle``: First versoin
        """
        # Save file name
        self.fjson = fname
        # Test if file exists
        if not os.path.isfile(fname):
            return
        # Open the file
        with open(fname, 'r') as f:
            # Read the file
            opts = json.load(f)
        # Check for ``"_settings"`` key
        # If present, blend it into existing settings
        merge_dict(self.settings, opts.pop(".settings", {}))
        # Merge settings
        self.merge(opts)

    # Write
    def write_json(self, fname, **kw):
        """Write metadata to JSON file

        :Call:
            >>> props.write_json(fname)
        :Inputs:
            *props*: :class:`ModulePropDB`
                Module property database instance
            *fname*: :class:`str`
                Name of JSON file
            *indent*: {``4``} | :class:`int` > 0
                Spaces in indentation level
        :Versions:
            * 2019-04-06 ``@ddalle``: First versoin
        """
        # Indent
        indent = kw.pop("indent", 4)
        # Create object for output
        opts = OrderedDict()
        # Save the settings
        opts[".settings"] = self.get_ordered_settings()
        # Add the individual module DBs in the appropriate order
        for mod in self.list_modules():
            # Get the properties for this module, sorted appropriately
            db = self.get_ordered_db(mod)
            # Store properly sorted database
            opts[mod] = db
        # Open the file for writing
        with open(fname, 'w') as f:
            # Write metadata
            json.dump(opts, f, indent=indent, separators=(",", ": "))
        

    # Merge options
    def merge(self, opts):
        """Merge a dictionary, where *opts* overrides *props*

        :Call:
            >>> props.merge(opts)
        :Inputs:
            *props*: :class:`ModulePropDB`
                Module property database instance
            *opts*: :class:`dict`
                Dictionary of metadata to merge
        :Versions:
            * 2019-04-07 ``@ddalle``: First version
        """
        # Apply the recursive function
        merge_dict(self, opts)

    # Merge options
    def mergedefault(self, opts):
        """Merge a dictionary, where *props* overrides *props*

        :Call:
            >>> props.mergedefault(opts)
        :Inputs:
            *props*: :class:`ModulePropDB`
                Module property database instance
            *opts*: :class:`dict`
                Dictionary of metadata to merge
        :Versions:
            * 2019-04-07 ``@ddalle``: First version
        """
        # Apply the recursive function
        merge_dict_default(self, opts)
    
    # Compare properties
    def compare_module(self, mod, modopts):
        """Compare specified properties to those of a particular module
        
        :Call:
            >>> match = props.compare_module(mod, modopts)
        :Inputs:
            *props*: :class:`ModulePropDB`
                Module property database instance
            *mod*: :class:`str`
                Name of module to test
            *modopts*: :class:`dict`
                Dictionary of properties to compare to module *mod*
        :Outputs:
            *match*: ``True`` | ``False``
                Whether or not *mod* matches all values in *modopts*
        :Versions:
            * 2019-04-15 ``@ddalle``: First version
        """
        # Check input
        if not isinstance(modopts, dict):
            raise TypeError("Target for comparisons must be a 'dict'")
        # Get database for that module
        moddb = self.get(mod, {})
        # Get list of available keys
        keylist = self.settings.get("Keys", [])
        # Loop through input keys
        for k, v in modopts.items():
            # Replace "_" with "-"
            k = k.replace("_", "-")
            # Check if it's a key
            if k not in keylist:
                raise ValueError(
                    ("Key '%s' is not a recognized key\n" % k) +
                    ("Available keys: %s" % keylist))
            # Check comparison
            if moddb.get(k) != v:
                # Mismatch
                return False
        # Otherwise, all criteria met
        return True
        
    # Compare properties
    def compare_module_all(self, mod, *a, **kw):
        """Search for specified properties in a particular module
        
        :Call:
            >>> q, keys = props.compare_module_all(mod, *a, **kw)
            >>> q, keys = props.compare_module_all(mod, v1, v2, ...)
            >>> q, keys = props.compare_module_all(mod, k1=v1, k2=v2)
        :Inputs:
            *props*: :class:`ModulePropDB`
                Module property database instance
            *k1*: :class:`str`
                Name of first key to search (``_`` replaced with ``-``)
            *k2*: :class:`str`
                Name of second key to search
            *v1*: :class:`str` | :class:`any`
                Test value 1 (for key *k1* if specified, else any key)
            *v2*: :class:`str` | :class:`any`
                Test value 2 (for key *k2* if specified)
            *kw*: :class:`dict`
                Keyword arguments of options to match
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not all properties were matched
            *keys*: :class:`list`[:class:`str`]
                Dictionary of modules that match, with keys that match
        :Versions:
            * 2019-04-16 ``@ddalle``: First version
        """
        # Initialize match flag
        q = True
        # Initialize match list
        keys = []
        # Get list of avialable keys
        keylist = self.settings.get("Keys", [])
        # Get database for requested module
        moddb = self.get(mod, {})
        # Loop through unkeyed search items
        for v in a:
            # Flag for at least one match of this value
            qv = False
            # Check if it's a string
            v_isstr = typeutils.isstr(v)
            # If so, test based on regular expression
            if v_isstr:
                t = re.compile(v).search
            # Loop through entries of module database
            for mk, mv in moddb.items():
                # If both are strings, use regular expression search
                if v_isstr and typeutils.isstr(mv):
                    # Regular expression test
                    qmk = t(mv) is not None
                elif (v is None) or (v is True) or (v is False):
                    # Test using "is"
                    qmk = (v is mv)
                else:
                    # Test equivalence
                    qmk = (v == mv)
                # If there's a match, update list
                if qmk and (mk not in keys):
                    keys.append(mk)
                # Update search flag
                qv = qv or qmk
            # Update overall match test
            q = q and qv
        # Loop through keyword search items
        for k, v in kw.items():
            # Check if present in module
            if k not in moddb:
                # No match for at least one key
                q = False
                continue
            # Get test value
            mv = moddb[k]
            # Check if both values are strings
            v_isstr = typeutils.isstr(v)
            m_isstr = typeutils.isstr(mv)
            # If so, test based on regular expression
            if v_isstr and m_isstr:
                # Compile regular expression
                t = re.compile(v).search
                # Perform test
                qk = t(mv) is not None
            elif (v is None) or (v is True) or (v is False):
                # Test using "is"
                qk = (v is mv)
            else:
                # Test equivalence
                qk = (v == mv)
            # If there's a match, update list
            if qk and (k not in keys):
                keys.append(k)
            # Update search flag
            q = q and qk
        # Output
        return q, keys
                
                

    # Search database
    def search_db(self, *a, **kw):
        """Search module database for modules that match criteria
        
        :Call:
            >>> mods = props.search_db(**kw)
            >>> mods = props.search_db(opts, **kw)
        :Inputs:
            *props*: :class:`ModulePropDB`
                Module property database instance
            *opts*: :class:`dict`
                Dictionary of options to match
            *kw*: :class:`dict`
                Keyword arguments of options to match
        :Outputs:
            *mods*: :class:`list`[:class:`str`]
                List of module names
        :Versions:
            * 2019-04-16 ``@ddalle``: First version
        """
        # Create options to compare to
        opts = dict(*a, **kw)
        # Initialize list of matches
        mods = []
        # Loop through databases
        for mod in self:
            # Compare the database
            if self.compare_module(mod, opts):
                # Save this database
                mods.append(mod)
        # Output
        return mods

    # Search database by value
    def search(self, *a, **kw):
        """Search module database for values regardless of key
        
        Checks are made using regular expressions (in particular,
        :func:`re.search`) if both the database value and the test
        value are strings.  If no keyword is specified, the value will
        be searched in each key. 
        
        :Call:
            >>> moddb = props.search(*a, **kw)
            >>> moddb = props.search(v1, v2, ..., **kw)
            >>> moddb = props.search(k1=v1, k2=v2)
        :Inputs:
            *props*: :class:`ModulePropDB`
                Module property database instance
            *k1*: :class:`str`
                Name of first key to search (``_`` replaced with ``-``)
            *k2*: :class:`str`
                Name of second key to search
            *v1*: :class:`str` | :class:`any`
                Test value 1 (for key *k1* if specified, else any key)
            *v2*: :class:`str` | :class:`any`
                Test value 2 (for key *k2* if specified)
            *kw*: :class:`dict`
                Keyword arguments of options to match
        :Outputs:
            *moddb*: :class:`dict`[:class:`list`[:class:`str`]]
                Dictionary of modules that match, with keys that match
        :Versions:
            * 2019-04-16 ``@ddalle``: First version
        """
        # Get list of available keys
        keylist = self.settings.get("Keys", [])
        # Initialize output dictionary
        moddb = {}
        # Loop through modules
        for mod in self:
            # Compare the database for this module
            q, keys = self.compare_module_all(mod, *a, **kw)
            # If it matched, append to output
            if q:
                moddb[mod] = keys
        # Output
        return moddb
        
# class Metadata
