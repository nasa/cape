"""
:mod:`cape.trajectory`: Run matrix interface
==============================================

This module provides a class :class:`cape.trajectory.Trajectory` for
interacting with a list of cases. Usually this is the list of cases defined as
the run matrix for a set of CFD solutions, and it is defined in the
``"Trajectory"`` :ref:`section of the JSON file <cape-json-trajectory>`.

However, the contents of the :class:`cape.trajectory.Trajectory` may have a
list of cases that differs from the run matrix, for example containing instead
the cases contained in a data book.

The key defining parameter of a run matrix is the list of independent
variables, which are referred to as "trajectory keys" within Cape. For example,
a common list of trajectory keys for an inviscid setup is ``["mach", "alpha",
"beta"]``. If the run matrix is loaded as *x*, then the value of the Mach
number for case number *i* would be ``x.mach[i]``. If the name of the key to
query is held within a variable *k*, use the following syntax to get the value.

    .. code-block:: python
    
        # Name of trajectory key
        k = "alpha"
        # Case number
        i = 10
        # Value of that key for case *i*
        getattr(x,k)[i]

Each case has a particular folder name.  To get the name of the folder for case
*i*, use the syntax

    .. code-block:: python
        
        // Group folder name
        x.GetGroupFolderNames(i)
        // Case folder name
        x.GetFolderNames(i)
        // Combined group and case folder name
        x.GetFullFolderNames(i)

The trajectory class also contains several methods for filtering cases.  For
example, the user may get the list of indices of cases with a Mach number
greater than 1.0, or the user may restrict to cases containing the text "a1.0". 
These use the methods :func:`cape.trajectory.Trajectory.Filter` and
:func:`cape.trajectory.Trajectory.FilterString`, respectively.  Filtering
examples are shown below.

    .. code-block:: python
    
        // Constraints
        I = x.Filter(cons=['mach>=0.5', 'mach<1.0'])
        // Contains exact text
        I = x.FilterString('m0.9')
        // Contains text with wildcards (like file globs)
        I = x.FilterWildcard('m?.?a-*')
        // Contains a regular expression
        I = x.FilterRegex('m0\.[5-8]+a')
        
These methods are all combined in the
:func:`cape.trajectory.Trajectory.GetIndices` method.  The
:func:`cape.trajectory.Trajectory.GetSweeps` method provides a capability to
split a run matrix into groups in which the cases in each group satisfy
user-specified constraints, for example having the same angle of attack.

Also provided are methods such as :func:`cape.trajectory.Trajectory.GetAlpha`,
which allows the user to easily access the angle of attack for case *i* even if
the run matrix is defined using total angle of attack and roll angle.
Similarly, :func:`cape.trajectory.Trajectory.GetReynoldsNumber` returns the
Reynolds number per grid unit even if the run matrix uses static or dynamic
pressure, which relieves the user from having to do the conversions before
creating the run matrix conditions.
"""

# Basic numerics
import numpy as np
# More powerful text processing
import re, fnmatch
# File system and operating system management
import os
# Conversions and constants
from . import convert
# Units tool
from .units import mks


# Trajectory class
class Trajectory(object):
    """
    Read a list of configuration variables
    
    :Call:
        >>> x = cape.Trajectory(**traj)
        >>> x = cape.Trajectory(File=fname, Keys=keys)
    :Inputs:
        *traj*: :class:`dict`
            Dictionary of options from ``opts["Trajectory"]``
    :Keyword arguments:
        *File*: :class:`str`
            Name of file to read, defaults to ``'Trajectory.dat'``
        *Keys*: :class:`list` of :class:`str` items
            List of variable names, defaults to ``['Mach','alpha','beta']``
        *Prefix*: :class:`str`
            Prefix to be used for each case folder name
        *GroupPrefix*: :class:`str`
            Prefix to be used for each grid folder name
        *GroupMesh*: :class:`bool`
            Whether or not cases in same group can share volume grids
        *Definitions*: :class:`dict`
            Dictionary of definitions for each key
    :Outputs:
        *x*: :class:`cape.trajectory.Trajectory`
            Instance of the trajectory class
    :Data members:
        *x.nCase*: :class:`int`
            Number of cases in the trajectory
        *x.prefix*: :class:`str`
            Prefix to be used in folder names for each case in trajectory
        *x.GroupPrefix*: :class:`str`
            Prefix to be used for each grid folder name
        *x.keys*: :class:`list`, *dtype=str*
            List of variable names used
        *x.text*: :class:`dict`, *dtype=list*
            Lists of variable values taken from trajectory file
        *x.Mach*: :class:`numpy.ndarray`, *dtype=float*
            Vector of Mach numbers in trajectory
        ``getattr(x, key)``: :class:`numpy.ndarray`, *dtype=float*
            Vector of values of each variable specified in *keys*
    :Versions:
        2014-05-28 ``@ddalle``: First version
        2014-06-05 ``@ddalle``: Generalized for user-defined keys
    """
  # =============
  # Configuration
  # =============
  # <
    # Initialization method
    def __init__(self, **kwargs):
        """Initialization method"""
        # Check for an empty trajectory
        if kwargs.get('Empty', False): return
        # Process the inputs.
        fname = kwargs.get('File', None)
        keys = kwargs.get('Keys', ['Mach', 'alpha', 'beta'])
        prefix = kwargs.get('Prefix', "F")
        groupPrefix = kwargs.get('GroupPrefix', "Grid")
        # Process the definitions.
        defns = kwargs.get('Definitions', {})
        # Save properties.
        self.keys = keys
        self.prefix = prefix
        self.GroupPrefix = groupPrefix
        # List of PASS and ERROR markers
        self.PASS = []
        self.ERROR = []
        # Save freestream state
        self.gas = kwargs.get("Freestream", {})
        # Process the key definitions.
        self.ProcessKeyDefinitions(defns)
        # Read the file.
        if fname and os.path.isfile(fname):
            self.ReadTrajectoryFile(fname)
        # Loop through the keys to see if any were specified in the inputs.
        for key in keys:
            # Check inputs for that key.
            if key not in kwargs: continue
            # Check the specification type.
            if type(kwargs[key]).__name__ not in ['list']:
                # Use the same value for all cases
                self.text[key] = [str(kwargs[key])] * len(self.text[keys[0]])
            else:
                # Set it with the new value.
                self.text[key] = [str(v) for v in kwargs[key]]
        # Check if PASS markers are specified.
        if 'PASS' in kwargs:
            self.PASS = kwargs['PASS']
        # Check if ERROR markers are specified.
        if 'ERROR' in kwargs:
            self.ERROR = kwargs['ERROR']
        # Convert PASS and ERROR list to numpy.
        self.PASS  = np.array(self.PASS)
        self.ERROR = np.array(self.ERROR)
        # Save the number of cases.
        nCase = len(self.text[keys[0]])
        self.nCase = nCase
        # Number of entries
        nPass = len(self.PASS)
        nErr  = len(self.ERROR)
        # Make sure PASS and ERROR fields have correct length
        if nPass < nCase:
            self.PASS = np.hstack((self.PASS, False*np.ones(nCase-nPass)))
        if nErr < nCase:
            self.ERROR = np.hstack((self.ERROR, False*np.ones(nCase-nErr)))
        # Create the numeric versions.
        for key in keys:
            # Check the key type.
            if 'Value' not in self.defns[key]:
                raise KeyError(
                    "Definition for trajectory key '%s' is incomplete." % key)
            if self.defns[key]['Value'] == 'float':
                # Normal numeric value
                setattr(self, key,
                    np.array([float(v) for v in self.text[key]]))
            elif self.defns[key]['Value'] == 'int':
                # Normal numeric value
                setattr(self, key,
                    np.array([int(v) for v in self.text[key]]))
            elif self.defns[key]['Value'] == 'hex':
                # Hex numeric value
                setattr(self, key,
                    np.array([eval('0x'+v) for v in self.text[key]]))
            elif self.defns[key]['Value'] in ['oct', 'octal']:
                # Octal value
                setattr(self, key,
                    np.array([eval('0o'+v) for v in self.text[key]]))
            elif self.defns[key]['Value'] in ['bin', 'binary']:
                # Binary value
                setattr(self, key,
                    np.array([eval('0b'+v) for v in self.text[key]]))
            else:
                # Assume string
                setattr(self, key, np.array(self.text[key]))
        # Process the groups (conditions in a group can use same grid).
        self.ProcessGroups()
        
    # Function to display things
    def __repr__(self):
        """
        Return the string representation of a trajectory.
        
        This looks like ``<cape.Trajectory(nCase=N, keys=['Mach','alpha'])>``
        """
        # Return a string.
        return '<cape.Trajectory(nCase=%i, keys=%s)>' % (self.nCase,
            self.keys)
        
    # Function to display things
    def __str__(self):
        """
        Return the string representation of a trajectory.
        
        This looks like ``<cape.Trajectory(nCase=N, keys=['Mach','alpha'])>``
        """
        # Return a string.
        return '<cape.Trajectory(nCase=%i, keys=%s)>' % (self.nCase,
            self.keys)
        
    # Copy the trajectory
    def Copy(self):
        """Return a copy of the trajectory
        
        :Call:
            >>> y = x.Copy()
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the trajectory class
        :Outputs:
            *y*: :class:`cape.trajectory.Trajectory`
                Separate trajectory with same data
        :Versions:
            * 2015-05-22 ``@ddalle``
        """
        # Initialize an empty trajectory.
        y = Trajectory(Empty=True)
        # Copy the fields.
        y.defns  = {}
        y.abbrv  = dict(self.abbrv)
        y.keys   = list(self.keys)
        y.text   = self.text.copy()
        y.prefix = self.prefix
        y.PASS   = self.PASS.copy()
        y.ERROR  = self.ERROR.copy()
        y.nCase  = self.nCase
        # Copy definitions
        for k in self.defns:
            y.defns[k] = dict(self.defns[k])
        # Group-related info
        y.GroupPrefix  = self.GroupPrefix
        y.GroupKeys    = self.GroupKeys
        y.NonGroupKeys = self.NonGroupKeys
        # Loop through keys to copy values.
        for k in self.keys:
            # Copy the array
            setattr(y,k, getattr(self,k).copy())
        # Process groups to make it a full trajectory.
        self.ProcessGroups()
        # Output
        return y
    
  # >
    
  # ========
  # File I/O
  # ========
  # <
    # Function to read a file
    def ReadTrajectoryFile(self, fname):
        """Read trajectory variable values from file
        
        :Call:
            >>> x.ReadTrajectoryFile(fname)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the trajectory class
            *fname*: :class:`str`
                Name of trajectory file
        :Versions:
            * 2014-10-13 ``@ddalle``: Cut code from __init__ method
        """
        # Open the file.
        f = open(fname)
        # Extract the keys.
        keys = self.keys
        # Number of variables
        nVar = len(keys)
        # Loop through the lines.
        for line in f.readlines():
            # Strip the line.
            line = line.strip()
            # Check for empty line or comment
            if line.startswith('#') or len(line)==0:
                continue
            # Group string literals by '"' and "'"
            grp1 = re.findall('"[^"]*"', line)
            grp2 = re.findall("'[^']*'", line)
            # Make replacements
            for i in range(len(grp1)):
                line = line.replace(grp1[i], "grp1-%s" % i)
            for i in range(len(grp2)):
                line = line.replace(grp2[i], "grp2-%s" % i)
            # Separate by any white spaces and/or at most one comma
            v = re.split("\s*,{0,1}\s*", line)
            # Substitute back in original literals
            for i in range(len(grp1)):
                # Replacement text
                txt = "grp1-%s" % i
                # Original, quotes stripped
                raw = grp1[i].strip('"')
                # Make replacements
                v = [vi.replace(txt, raw) for vi in v]
            # Substitute back in original literals
            for i in range(len(grp2)):
                # Replacement text
                txt = "grp2-%s" % i
                # Original, quotes stripped
                raw = grp2[i].strip("'")
                # Make replacements
                v = [vi.replace(txt, raw) for vi in v]
            # Check v[0]
            if v[-1].lower() in ['$p', 'pass']:
                # Case is marked as passed.
                self.PASS.append(True)
                self.ERROR.append(False)
                # Shift the entries.
                v.pop()
            elif v[0].lower() in ['p', '$p', 'pass']:
                # Case is marked as passed.
                self.PASS.append(True)
                self.ERROR.append(False)
                # Shift the entries.
                v.pop(0)
            elif v[-1].lower() in ['$e', 'error']:
                # Case is marked as error.
                self.PASS.append(False)
                self.ERROR.append(True)
                # Shift the entries.
                v.pop()
            elif v[0].lower() in ['e', '$e', 'error']:
                # Case is marked as error.
                self.PASS.append(False)
                self.ERROR.append(True)
                # Shift the entries.
                v.pop(0)
            else:
                # Case is unmarked.
                self.PASS.append(False)
                self.ERROR.append(False)
            # Save the strings.
            for k in range(nVar):
                # Check for text.
                if k < len(v):
                    # Save the text.
                    self.text[keys[k]].append(v[k])
                elif self.defns[keys[k]]['Value'] == 'str':
                    # No text (especially useful for optional labels)
                    # Default value.
                    v0 = self.defns[keys[k]].get('Default', '')
                    self.text[keys[k]].append(v0)
                else:
                    # No text (especially useful for optional labels)
                    v0 = self.defns[keys[k]].get('Default', '0')
                    self.text[keys[k]].append(str(v0))
        # Close the file.
        f.close()
        
    # Function to write a JSON file with the trajectory variables.
    def WriteConditionsJSON(self, i, fname="conditions.json"):
        """Write a simple JSON file with exact trajectory variables
        
        :Call:
            >>> x.WriteConditionsJSON(i, fname="conditions.json")
        :Inputs:
            *i*: :class:`int`
                Index of the run case to print
            *fname*: :class:`str`
                Name of file to create
        :Versions:
            * 2014-11-18 ``@ddalle``: First version
        """
        # Open the file.
        f = open(fname, 'w')
        # Create the header.
        f.write('{\n')
        # Number of keys
        n = len(self.keys)
        # Loop through the keys.
        for j in range(n):
            # Name of the key.
            k = self.keys[j]
            # If it's a string, add quotes.
            if self.defns[k]["Value"] in ['str', 'char']:
                # Use quotes.
                q = '"'
            else:
                # No quotes.
                q = ''
            # Test if a comma is needed.
            if j >= n-1:
                # No comma for last line.
                c = ''
            else:
                # Yes, a comma is needed.
                c = ','
            # Get the value.
            v = getattr(self,k)[i]
            # Initial portion of line.
            line = ' "%s": %s%s%s%s\n' % (k, q, v, q, c)
            # Write the line.
            f.write(line)
        # Close out the JSON object.
        f.write('}\n')
        # Close the file.
        f.close()
  # >
    
  # ===============
  # Key Definitions
  # ===============
  # <
    # Function to process the role that each key name plays.
    def ProcessKeyDefinitions(self, defns):
        """
        Process definitions for the function of each trajectory variable
        
        Many variables have default definitions, such as ``'Mach'``,
        ``'alpha'``, etc.  For user-defined trajectory keywords, defaults will
        be used for aspects of the definition that are missing from the inputs.
        
        :Call:
            >>> x.ProcessKeyDefinitions(defns)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *defns*: :class:`dict`
                Dictionary of keyword definitions or partial definitions
        :Effects:
            *x.text*: :class:`dict`
                Text for each variable and each break point is initialized
            *x.defns*: :class:`dict`
                Definition dictionary is created after processing defaults
            *x.abbrv*: :class:`dict`
                Dictionary of abbreviations for each trajectory key
        :Versions:
            * 2014-06-05 ``@ddalle``: First version
            * 2014-06-17 ``@ddalle``: Overhauled to read from ``defns`` dict
        """
        # Overall default key
        odefkey = defns.get('Default', {})
        # Process the mandatory fields.
        odefkey.setdefault('Group', True)
        odefkey.setdefault('Type', "Group")
        odefkey.setdefault('Format', '%s')
        # Initialize the dictionaries.
        self.text = {}
        self.defns = {}
        self.abbrv = {}
        # Initialize the fields.
        for key in self.keys:
            # Initialize the text for this key.
            self.text[key] = []
            # Process defaults.
            if key.lower() in ['m', 'mach']:
                # Mach number; non group
                defkey = {
                    "Group": False,
                    "Type": "Mach",
                    "Value": "float",
                    "Format": "%s",
                    "Abbreviation": "m"
                }
            elif key in ['Alpha', 'alpha', 'aoa']:
                # Angle of attack; non group
                defkey = {
                    "Group": False,
                    "Type": "alpha",
                    "Value": "float",
                    "Format": "%s",
                    "Abbreviation": "a"
                }
            elif key in ['Beta', 'beta', 'aos']:
                # Sideslip angle; non group
                defkey = {
                    "Group": False,
                    "Type": "beta",
                    "Value": "float",
                    "Format": "%s",
                    "Abbreviation": "b"
                }
            elif key.lower() in ['alpha_t', 'alpha_total', 'aoav']:
                # Total angle of attack; non group
                defkey = {
                    "Group": False,
                    "Type": "alpha_t",
                    "Value": "float",
                    "Format": "%s",
                    "Abbreviation": "a"
                }
            elif key.lower() in ['phi', 'phiv']:
                # Total roll angle; non group
                defkey = {
                    "Group": False,
                    "Type": "phi",
                    "Value": "float",
                    "Format": "%s",
                    "Abbreviation": "r"
                }
            elif key.lower() in ['re', 'rey', 'reynolds', 'reynolds_number']:
                # Reynolds number per unit
                defkey = {
                    "Group": False,
                    "Type": "Re",
                    "Value": "float",
                    "Format": "%.2e",
                    "Label": False,
                    "Abbreviation": "Re"
                }
            elif key == "T" or key.lower() in ['tinf', 'temp', 'temperature']:
                # Static temperature
                defkey = {
                    "Group": False,
                    "Type": "T",
                    "Value": "float",
                    "Format": "%s",
                    "Label": False,
                    "Abbreviation": "T"
                }
            elif key in ['p', 'P', 'pinf', 'pressure']:
                # Static freestream pressure
                defkey = {
                    "Group": False,
                    "Type": "p",
                    "Value": "float",
                    "Format": "%s",
                    "Label": False,
                    "Abbreviation": "p"
                }
            elif key in ['q', 'qinf', 'qbar']:
                # Dynamic pressure
                defkey = {
                    "Group": False,
                    "Type": "q",
                    "Value": "float",
                    "Format": "%s",
                    "Label": False,
                    "Abbreviation": "q"
                }
            elif key in ['U', 'V']:
                # Freestream speed
                defkey = {
                    "Group": False,
                    "Type": "V",
                    "Value": "float",
                    "Format": "%s",
                    "Label": False,
                    "Abbreviation": "V"
                }
            elif key.lower() in ['gamma']:
                # Freestream ratio of specific heats
                defkey = {
                    "Group": False,
                    "Type": "gamma",
                    "Value": "float",
                    "Format": "%s",
                    "Abbreviation": "g"
                }
            elif key.lower() in ['p0', 'p_total', 'total_pressure'
                    ] or key.startswith('p0'):
                # Surface stagnation pressure ratio
                defkey = {
                    "Group": False,
                    "Type": "SurfBC",
                    "Value": "float",
                    "Format": "%s",
                    "Label": False,
                    "Abbreviation": "p0",
                    "RefPressure": 1.0,
                    "RefTemperature": 1.0,
                    "TotalPressure": None,
                    "TotalTemperature": "T0",
                    "CompID": []
                }
            elif key.startswith('CT'):
                # Thrust coefficient
                defkey = {
                    "Group": False,
                    "Type": "SurfCT",
                    "Value": "float",
                    "Format": "%s",
                    "Label": True,
                    "Abbreviation": "CT",
                    "RefDynamicPressure": None,
                    "RefArea": None,
                    "AreaRatio": 4.0,
                    "MachNumber": 1.0,
                    "TotalTemperature": "T0",
                    "CompID": []
                }
            elif key.lower() in ['t0', 't_total', 'total_temperature']:
                # Surface stagnation temperature ratio
                defkey = {
                    "Group": False,
                    "Type": "value",
                    "Value": "float",
                    "Format": "%s",
                    "Label": False,
                    "Abbreviation": "T0"
                }
            elif key.lower() in ['label', 'suffix']:
                # Extra label for case (non-group)
                defkey = {
                    "Group": False,
                    "Type": "Label",
                    "Value": "str",
                    "Format": "%s",
                    "Abbreviation": ""
                }
            elif key.lower() in ['value']:
                # Just holding a value
                defkey = {
                    "Group": False,
                    "Type": "value",
                    "Value": "float",
                    "Format": "%s",
                    "Label": False
                }
            elif key.lower() in ['tag', 'tags']:
                # Just holding a value
                defkey = {
                    "Group": False,
                    "Type": "value",
                    "Value": "str",
                    "Format": "%s",
                    "Label": False,
                    "Abbreviation": "tag"
                }
            elif key.lower() in ["user", "uid", "userfilter"]:
                # Filter on which user can submit
                defkey = {
                    "Group": False,
                    "Type": "user",
                    "Value": "str",
                    "Format": "%s",
                    "Label": False,
                    "Abbreviation": "user"
                }
            elif key.lower() in ['config', 'GroupPrefix']:
                # Group name or prefix, e.g. 'poweroff', 'poweron', etc.
                defkey = {
                    "Group": True,
                    "Type": "Config",
                    "Value": "str",
                    "Format": "%s",
                    "Abbreviation": ""
                }
            elif key in ['GroupLabel', 'GroupSuffix']:
                # Extra label for groups
                defkey = {
                    "Group": True,
                    "Type": "GroupLabel",
                    "Value": "str",
                    "Format": "%s",
                    "Abbreviation": ""
                }
            else:
                # Start with default key
                defkey = odefkey
                # Set the abbreviation to the full name.
                defkey["Abbreviation"] = key
            # Check if the input has that key defined.
            optkey = defns.get(key, {})
            # Loop through properties.
            for k in defkey.keys():
                optkey.setdefault(k, defkey[k])
            # Save the definitions.
            self.defns[key] = optkey
            # Save the abbreviations.
            self.abbrv[key] = optkey.get("Abbreviation", key)
        
    # Process the groups that need separate grids.
    def ProcessGroups(self):
        """
        Split trajectory variables into groups.  A "group" is a set of
        trajectory conditions that can use the same mesh.
        
        :Call:
            >>> x.ProcessGroups()
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
        :Effects:
            Creates fields that save the properties of the groups.  These fields
            are called *x.GroupKeys*, *x.GroupX*, *x.GroupID*.
        :Versions:
            * 2014-06-05 ``@ddalle``: First version
        """
        # Initialize matrix of group-generating key values.
        x = []
        # Initialize list of group variables.
        gk = []
        ngk = []
        # Loop through the keys to check for groups.
        for key in self.keys:
            # Check the definition for the grouping status.
            if self.defns[key]['Group']:
                # Append to matrix of group-only variables for ALL conditions.
                x.append(getattr(self,key))
                # Append to the list of group variables.
                gk.append(key)
            else:
                # Append to the list of groupable variables.
                ngk.append(key)
        # Save.
        self.GroupKeys = gk
        self.NonGroupKeys = ngk
        # Turn *x* into a proper matrix.
        x = np.transpose(np.array(x))
        # Unfortunately NumPy arrays do not support testing very well.
        y = [list(xi) for xi in x]
        # Initialize list of unique conditions.
        Y = []
        # Initialize the groupID numbers.
        gID = []
        # Test for case of now groups.
        if y == []:
            # List of group==0 nodes.
            gID = np.zeros(self.nCase)
        # Loop through the full set of conditions.
        for yi in y:
            # Test if it's in the existing set of unique conditions.
            if not (yi in Y):
                # If not, append it.
                Y.append(yi)
            # Save the index.
            gID.append(Y.index(yi))
        # Convert the result and save it.
        self.GroupX = np.array(Y)
        # Save the group index for *all* conditions.
        self.GroupID = np.array(gID)
        return None
        
    # Get all keys by type
    def GetKeysByType(self, KeyType):
        """Get all keys by type
        
        :Call:
            >>> keys = x.GetKeysByType(KeyType)
            >>> keys = x.GetKeysByType(KeyTypes)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of pyCart trajectory class
            *KeyType*: :class:`str`
                Key type to search for
            *KeyTypes*: :class:`list` (:class:`str`)
                List of key types to search for
        :Outputs:
            *keys*: :class:`numpy.ndarray` (:class:`str`)
                List of keys such that ``x[key]['Type']`` matches *KeyType*
        :Versions:
            * 2014-10-07 ``@ddalle``: First version
        """
        # List of key types
        KT = np.array([self.defns[k]['Type'] for k in self.keys])
        # Class of input
        kt = type(KeyType).__name__
        # Depends on the type of what we are searching for
        if kt.startswith('str') or kt=='unicode':
            # Return matches
            return np.array(self.keys)[KT == KeyType]
        elif kt not in ['list', 'ndarray']:
            # Not usable
            raise TypeError("Cannot search for keys of type '%s'" % KeyType)
        # Initialize list of matches to all ``False``
        U = np.zeros(len(self.keys), dtype="bool")
        # Loop through types given as input.
        for k in KeyType:
            # Search for this kind of key.
            U = np.logical_or(U, KT == k)
        # Output.
        return np.array(self.keys)[np.where(U)[0]]
        
    # Get first key by type
    def GetFirstKeyByType(self, KeyType):
        """Get all keys by type
        
        :Call:
            >>> key = x.GetFirstKeyByType(KeyType)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of pyCart trajectory class
            *KeyType*: :class:`str`
                Key type to search for
        :Outputs:
            *key*: :class:`str` | ``None``
                First key such that ``x.defns[key]['Type']`` matches *KeyType*
        :Versions:
            * 2018-04-13 ``@ddalle``: First version
        """
        # Loop through keys
        for k in self.keys:
            # Get the type
            typ = self.defns[k].get("Type", "value")
            # Check for a match
            if typ == KeyType:
                return k
        # No match
        return None
        
    # Get keys by type of its value
    def GetKeysByValue(self, val):
        """Get all keys with specified type of value
        
        :Call:
            >>> keys = x.GetKeysByValue(val)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of pyCart trajectory class
            *val*: :class:`str`
                Key value class to search for
        :Outputs:
            *keys*: :class:`numpy.ndarray` (:class:`str`)
                List of keys such that ``x[key]['Value']`` matches *val*
        :Versions:
            * 2014-10-07 ``@ddalle``: First version
        """
        # List of key types
        KV = np.array([self.defns[k]['Value'] for k in self.keys])
        # Return matches
        return np.array(self.keys)[KV == val]
        
    # Function to get the group index from the case index
    def GetGroupIndex(self, i):
        """Get group index from case index
        
        :Call:
            k = x.GetGroupIndex(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *i*: :class:`int`
                Index of case
        :Outputs:
            *j*: :class:`int`
                Index of group that contains case *i*
        :Versions:
            * 2014-09-27 ``@ddalle``: First versoin
        """
        # Check inputs.
        if not type(i).__name__.startswith('int'):
            raise TypeError("Input to :func:`Trajectory.GetGroupIndex` must"
                + " be :class:`int`.")
        # Get name of group for case *i*.
        grp = self.GetGroupFolderNames(i)
        # Get the list of all unique groups.
        grps = self.GetUniqueGroupFolderNames()
        # Find the index.
        j = np.where(grps == grp)[0][0]
        # Output
        return j
        
    # Get name of key based on type
    def GetKeyName(self, typ, key=None):
        """Get name of key by specified type; defaulting to first key with type
        
        A ValueError exception is raised if input key has incorrect type or if
        no keys have that type.
        
        :Call:
            >>> k = x.GetKeyName(typ, key=None)
        :Inputs:
            *typ*: :class:`str`
                Name of key type, for instance 'alpha_t'
            *key*: {``None``} | :class:`str`
                Name of trajectory key
        :Outputs:
            *k*: :class:`str`
                Key meeting those requirements
        :Versions:
            * 2016-08-29 ``@ddalle``: First version
        """
        # Process key
        if key is None:
            # Key types
            KeyTypes = [self.defns[k]['Type'] for k in self.keys]
            # Check for key.
            if typ not in KeyTypes:
                raise ValueError("No trajectory keys of type '%s'" % typ)
            # Get first key
            return self.GetKeysByType(typ)[0]
        else:
            # Check the key
            if key not in self.keys:
                # Undefined key
                raise KeyError("No trajectory key '%s'" % key)
            elif self.defns[key]['Type'] != typ:
                # Wrong type
                raise ValueError(
                    ("Requested key '%s' with type '%s', but " % (key,typ)) +
                    ("actual type is '%s'" % (self.defns[key]['Type'])))
            # Output the key
            return key
    
  # >
    
  # ============
  # Folder Names
  # ============
  # <
    # Function to assemble a folder name based on a list of keys and an index
    def _AssembleName(self, keys, prefix, i):
        """
        Assemble names using common code.
        
        :Call:
            >>> dname = x._AssembleName(keys, prefix, i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *keys*: :type:`list` (:class:`str`)
                List of keys to use for this folder name
            *prefix*: :class:`str`
                Header for name of each case folder
            *i*: :class:`int`
                Index of case to process
        :Outputs:
            *dname*: :class:`str` or :class:`list`
                Name containing value for each key in *keys*
        :Versions:
            * 2014-06-05 ``@ddalle``: First version
            * 2014-10-03 ``@ddalle``: Added suffixes
        """
        # Process the key types.
        types = [self.defns[k].get("Type","") for k in keys]
        # Check for a prefix.
        if "Config" in types:
            # Figure out which key it is
            j = types.index("Config")
            # Get the specified prefix.
            fpre = getattr(self,keys[j])[i]
            # Initialize the name.
            if fpre:
                # Use the specified prefix/config
                dname = str(fpre)
            else:
                # Use the input/default prefix/config
                dname = str(prefix)
            # Add underscore if more keys remaining.
            if len(types) > 1: dname += "_"
        elif prefix:
            # The prefix is likely to be the whole name.
            dname = str(prefix)
            # Add underscore if there are keys.
            if (keys is not None) and (len(keys)>0): dname += "_"
        else:
            # Initialize an empty string.
            dname = ""
        # Append based on the keys.
        for k in keys:
            # Skip text
            if self.defns[k]["Value"] == "str": continue
            # Check for unlabeled values
            if (not self.defns[k].get("Label", True)): continue
            # Skip unentered values
            if (i>=len(self.text[k])) or (not self.text[k][i]): continue
            # Check for "SkipZero" flag
            if self.defns[k].get("SkipIfZero", False): continue
            # Make the string of what's going to be printed.
            # This is something like ``'%.2f' % x.alpha[i]``.
            lbl = self.defns[k]["Format"] % getattr(self,k)[i]
            # Append the text in the trajectory file.
            dname += self.abbrv[k] + lbl 
        # Check for suffix keys.
        for k in keys:
            # Only look for labels.
            if self.defns[k].get("Type") != "Label": continue
            # Check the value.
            if (i < len(self.text[k])) and self.text[k][i].strip():
                # Add underscore if necessary.
                if dname: dname += "_"
                # Add the label itself
                dname += (self.abbrv[k] + self.text[k][i])
        # Return the result.
        return dname
        
        
    # Get PBS name
    def GetPBSName(self, i, pre=None):
        """Get PBS name for a given case
        
        :Call:
            >>> lbl = x.GetPBSName(i, pre=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *i*: :class:`int`
                Run index
            *pre*: {``None``} | :class:`str`
                Prefix to be added to a PBS job name
        :Outputs:
            *lbl*: :class:`str`
                Short name for the PBS job, visible via `qstat`
        :Versions:
            * 2014-09-30 ``@ddalle``: First version
            * 2016-12-20 ``@ddalle``: Moved to *x* and added prefix
        """
        # Initialize label.
        if pre:
            # Initialize job name with prefix
            lbl = '%s_' % pre
        else:
            # No prefix
            lbl = ''
        # Loop through keys.
        for k in self.keys[0:]:
            # Skip it if not part of the label.
            if not self.defns[k].get('Label', True):
                continue
            # Default print flag
            if self.defns[k]['Value'] == 'float':
                # Float: get two decimals if nonzero
                sfmt = '%.2f'
            else:
                # Simply use string
                sfmt = '%s'
            # Non-default strings
            slbl = self.defns[k].get('PBSLabel', self.abbrv[k])
            sfmt = self.defns[k].get('PBSFormat', sfmt)
            # Apply values
            slbl = slbl + (sfmt % getattr(self,k)[i])
            # Strip underscores
            slbl = slbl.replace('_', '')
            # Strop trailing zeros and decimals if float
            if self.defns[k]['Value'] == 'float':
                slbl = slbl.rstrip('0').rstrip('.')
            # Append to the label.
            lbl += slbl
        # Check length.
        if len(lbl) > 15:
            # 16-char limit (or is it 15?)
            lbl = lbl[:15]
        # Output
        return lbl
    
    # Function to return the full folder names.
    def GetFullFolderNames(self, i=None, prefix=None):
        """
        List full folder names for each of the cases in a trajectory.
        
        The folder names will be of the form
    
            ``Grid/F_m2.0a0.0b-0.5/``
            
        if there are no trajectory keys that require separate grids or
        
            ``Grid_d1.0/F_m2.0a0.0b-0.5/``
            
        if there is a key called ``"delta"`` with abbreviation ``'d'`` that
        requires a separate mesh each time the value of that key changes.  All
        keys in the trajectory file are included in the folder name at one of
        the two levels.  The number of digits used will match the number of
        digits in the trajectory file.
        
        :Call:
            >>> dname = x.GetFullFolderNames()
            >>> dname = x.GetFullFolderNames(i=None, prefix="F")
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *i*: :class:`int` or :class:`list`
                Index of cases to process or list of cases.  If this is
                ``None``, all cases will be processed.
            *prefix*: :class:`str`
                Header for name of each case folder
        :Outputs:
            *dname*: :class:`str` or :class:`list`
                Folder name or list of folder names
        :Versions:
            * 2014-06-05 ``@ddalle``: First version
        """
        # Get the two components.
        glist = self.GetGroupFolderNames(i)
        flist = self.GetFolderNames(i, prefix)
        # Check for list or not.
        if type(glist) is list:
            # Return the list of combined strings.
            return [os.path.join(glist[i],flist[i]) for i in range(len(glist))]
        else:
            # Just join the one.
            return os.path.join(glist, flist)
    
    # Function to list directory names
    def GetFolderNames(self, i=None, prefix=None):
        """
        List folder names for each of the cases in a trajectory.
        
        The folder names will be of the form
    
            ``F_m2.0a0.0b-0.5/``
            
        if the prefix is ``'F'``, or
        
            ``m2.0a0.0b-0.5/``
            
        if the prefix is empty.
            
        Trajectory keys that require separate meshes for each value of the key
        will not be part of the folder name.  The number of digits used will
        match the number of digits in the trajectory file.
        
        :Call:
            >>> dname = x.GetFolderNames()
            >>> dname = x.GetFolderNames(i=None, prefix="F")
        :Inputs:
            *T*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *i*: :class:`int` or :class:`list`
                Index of cases to process or list of cases.  If this is
                ``None``, all cases will be processed.
            *prefix*: :class:`str`
                Header for name of each folder
        :Outputs:
            *dname*: :class:`str` or :class:`list`
                Folder name or list of folder names
        :Versions:
            * 2014-05-28 ``@ddalle``: First version
            * 2014-06-05 ``@ddalle``: Refined to variables that use common grid
        """
        # Process the prefix.
        if prefix is None: prefix = self.prefix
        # Process the index list.
        if i is None: i = range(self.nCase)
        # Get the variable names.
        keys = self.NonGroupKeys
        # Check for a list.
        if np.isscalar(i):
            # Get the name.
            dlist = self._AssembleName(keys, prefix, i)
        else:
            # Initialize the list.
            dlist = []
            # Loop through the conditions.
            for j in i:
                # Get the folder name.
                dname = self._AssembleName(keys, prefix, j)
                # Append to the list.
                dlist.append(dname)
        # Return the list.
        return dlist
        
    # Function to get grid folder names
    def GetGroupFolderNames(self, i=None):
        """
        Get names of folders that require separate meshes
        
        :Call:
            >>> x.GetGroupFolderNames()
            >>> x.GetGroupFolderNames(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *i*: :class:`int` or :class:`list`
                Index of cases to process or list of cases.  If this is
                ``None``, all cases will be processed.
        :Outputs:
            *dname*: :class:`str` or :class:`list`
                Folder name or list of folder names
        :Versions:
            * 2014-06-05 ``@ddalle``: First version
        """
        # Check for prefix variables.
        #if 
        # Set the prefix.
        prefix = self.GroupPrefix
        # Process the index list.
        if i is None: i = range(self.nCase)
        # Get the names of variables requiring separate grids.
        keys = self.GroupKeys
        # Check for a list.
        if np.isscalar(i):
            # Get the name.
            dlist = self._AssembleName(keys, prefix, i)
        else:
            # Initialize the list.
            dlist = []
            # Loop through the conditions.
            for j in i:
                # Get the folder name.
                dname = self._AssembleName(keys, prefix, j)
                # Append to the list.
                dlist.append(dname)
        # Return the list.
        return dlist
        
    # Function to get grid folder names
    def GetUniqueGroupFolderNames(self, i=None):
        """
        Get unique names of folders that require separate meshes
        
        :Call:
            >>> x.GetUniqueGroupFolderNames()
            >>> x.GetUniqueGroupFolderNames(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *i*: :class:`int` or :class:`list`
                Index of group(s) to process
        :Outputs:
            *dname*: :class:`str` or :class:`list`
                Folder name or list of folder names
        :Versions:
            * 2014-09-03 ``@ddalle``: First version
        """
        # Get all group folder names
        dlist = self.GetGroupFolderNames()
        # Transform to unique list.
        dlist = np.unique(dlist)
        # Check for an index filter.
        if i:
            # Return either a single value or sublist
            return dlist[i]
        else:
            # Return the whole list
            return dlist
    
  # >
   
  # =======
  # Filters
  # =======
  # <
    # Function to filter cases
    def Filter(self, cons, I=None):
        """Filter cases according to a set of constraints
        
        The constraints are specified as a list of strings that contain
        inequalities of variables that are in *x.keys*.
        
        For example, if *m* is the name of a key (presumably meaning Mach
        number), and *a* is a variable presumably representing angle of attack,
        the following example finds the indices of all cases with Mach number
        greater than 1.5 and angle of attack equal to ``2.0``.
        
            >>> i = x.Filter(['m>1.5', 'a==2.0'])
            
        A warning will be produces if one of the constraints does not correspond
        to a trajectory variable or cannot be evaluated for any other reason.
        
        :Call:
            >>> i = x.Filter(cons)
            >>> i = x.Fitler(cons, I)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *cons*: :class:`list` (:class:`str`)
                List of constraints
            *I*: :class:`list` (:class:`int`)
                List of initial indices to consider
        :Outputs:
            *i*: :class:`numpy.ndarray` (:class:`int`)
                List of indices that match constraints
        :Versions:
            * 2014-12-09 ``@ddalle``: First version
        """
        # Initialize the conditions.
        if I is None:
            # Consider all indices
            i = np.arange(self.nCase) > -1
        else:
            # Start with all indices failed.
            i = np.arange(self.nCase) < -1
            # Set the specified indices to True
            i[I] = True
        # Check for None
        if cons is None: cons = []
        # Loop through constraints
        for con in cons:
            # Check for empty constraints.
            if len(con.strip()) == 0: continue
            # Perform substitutions on constraint
            con = con.strip().replace('=', '==')
            con = con.replace('====', '==')
            con = con.replace('>==', '>=')
            con = con.replace('<==', '<=')
            con = con.replace('!==', '!=')
            # Constraint may fail with bad input.
            try:
                # Apply the constraint.
                i = np.logical_and(i, eval('self.' + con))
            except Exception:
                # Print a warning and move on.
                print("Constraint '%s' failed to evaluate." % con)
        # Output
        return np.where(i)[0]
        
    # Function to filter by checking if string is in the name
    def FilterString(self, txt, I=None):
        """Filter cases by whether or not they contain a substring
        
        :Call:
            >>> i = x.FilterString(txt)
            >>> i = x.FilterString(txt, I)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *txt*: :class:`str`
                Substring to use as filter
            *I*: :class:`list` (:class:`int`)
                List of initial indices to consider
        :Outputs:
            *i*: :class:`numpy.ndarray` (:class:`int`)
                List of indices that match constraints
        :Versions:
            * 2015-11-02 ``@ddalle``: First version
        """
        # Initialize the conditions.
        if I is None:
            # Consider all indices
            i = np.arange(self.nCase) > -1
        else:
            # Start with all indices failed.
            i = np.arange(self.nCase) < -1
            # Set the specified indices to True
            i[I] = True
        # Loop through conditions
        for j in np.where(i)[0]:
            # Get the case name.
            frun = self.GetFullFolderNames(j)
            # Check if the string is in there.
            if txt not in frun:
                i[j] = False
        # Output
        return np.where(i)[0]
        
    # Function to filter by checking if string is in the name
    def FilterWildcard(self, txt, I=None):
        """Filter cases by whether or not they contain a substring
        
        This function uses file wild cards, so for example
        ``x.FilterWildcard('*m?.0*')`` matches any case whose name contains
        ``m1.0`` or ``m2.0``, etc.  To make sure the ``?`` is a number, use
        ``*m[0-9].0``.  To obtain a filter that matches both ``m10.0`` and
        ``m1.0``, see :func:`FilterRegex`.
        
        :Call:
            >>> i = x.FilterWildcard(txt)
            >>> i = x.FilterWildcard(txt, I)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *txt*: :class:`str`
                Wildcard to use as filter
            *I*: :class:`list` (:class:`int`)
                List of initial indices to consider
        :Outputs:
            *i*: :class:`numpy.ndarray` (:class:`int`)
                List of indices that match constraints
        :Versions:
            * 2015-11-02 ``@ddalle``: First version
        """
        # Initialize the conditions.
        if I is None:
            # Consider all indices
            i = np.arange(self.nCase) > -1
        else:
            # Start with all indices failed.
            i = np.arange(self.nCase) < -1
            # Set the specified indices to True
            i[I] = True
        # Loop through conditions
        for j in np.where(i)[0]:
            # Get the case name.
            frun = self.GetFullFolderNames(j)
            # Check if the string is in there.
            if not fnmatch.fnmatch(frun, txt):
                i[j] = False
        # Output
        return np.where(i)[0]
        
    # Function to filter by checking if string is in the name
    def FilterRegex(self, txt, I=None):
        """Filter cases by whether or not they match a regular expression
        
        :Call:
            >>> i = x.FilterRegex(txt)
            >>> i = x.FilterRegex(txt, I)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *txt*: :class:`str`
                Wildcard to use as filter
            *I*: :class:`list` (:class:`int`)
                List of initial indices to consider
        :Outputs:
            *i*: :class:`numpy.ndarray` (:class:`int`)
                List of indices that match constraints
        :Versions:
            * 2015-11-02 ``@ddalle``: First version
        """
        # Initialize the conditions.
        if I is None:
            # Consider all indices
            i = np.arange(self.nCase) > -1
        else:
            # Start with all indices failed.
            i = np.arange(self.nCase) < -1
            # Set the specified indices to True
            i[I] = True
        # Loop through conditions
        for j in np.where(i)[0]:
            # Get the case name.
            frun = self.GetFullFolderNames(j)
            # Check if the name matches the regular expression.
            if not re.search(txt, frun):
                i[j] = False
        # Output
        return np.where(i)[0]
        
        
    # Function to expand indices
    def ExpandIndices(self, itxt):
        """Expand string of subscripts into a list of indices
        
        :Call:
            >>> I = x.ExpandIndices(itxt)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *itxt*: :class:`str` or :class:`unicode`
                Text of subscripts, separated by ';'
        :Outputs:
            *I*: :class:`list` (:class:`int`)
                Array of indices matching any of the input indices
        :Examples:
            >>> x.ExpandIndices(':5')
            array([0, 1, 2, 3, 4])
            >>> x.ExpandIndices(':4;7,8')
            array([0, 1, 2, 3, 7, 8])
        :Versions:
            * 2015-03-10 ``@ddalle``: First version
        """
        # Check the input.
        if type(itxt).__name__ in ['list', 'ndarray']:
            # Already split
            ITXT = itxt
        elif type(itxt).__name__ in ['str', 'unicode']:
            # Split.
            ITXT = itxt.split(';')
        else:
            # Invalid format
            return []
        # Get the full list of indices.
        I0 = range(self.nCase)
        # Initialize output
        I = []
        # Split the input by semicolons.
        for i in ITXT:
            # Ignore []
            i = i.lstrip('[').rstrip(']')
            try:
                # Check for a ':'
                if ':' in i:
                    # Add a range.
                    I += eval('I0[%s]' % i)
                elif ',' in i:
                    # List
                    I += list(eval(i))
                else:
                    # Individual case
                    I += [eval(i)]
            except Exception:
                # Status update.
                print("Index specification '%s' failed to evaluate." % i)
        # Return the matches.
        return I
        
    # Get indices
    def GetIndices(self, **kw):
        """Get indices from either list or constraints or both
        
        :Call:
            >>> I = x.GetIndices(I=None, cons=[], **kw)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *I*: :class:`list` | :class:`str`
                Array of indices or text of indices
            *cons*: :class:`list` (:class:`str`) | :class:`str`
                List of constraints or text list using commas
            *re*: :class:`str`
                Regular expression to test against folder names
            *filter*: :class:`str`
                Exact test to test against folder names
            *glob*: :class:`str`
                Wild card to test against folder names
        :Outputs:
            *I*: :class:`numpy.ndarray` (:class:`int`)
                Array of indices
        :Versions:
            * 2015-03-10 ``@ddalle``: First version
            * 2016-02-17 ``@ddalle``: Upgraded to handle text
        """
        # Get special kwargs
        I = kw.get("I")
        cons = kw.get("cons", [])
        # Check index for string
        if type(I).__name__ in ['str', 'unicode']:
            # Process indices
            I = self.ExpandIndices(I)
        # Check constraints for string
        if type(cons).__name__ in ['str', 'unicode']:
            # Separate into list of constraints
            cons = cons.split(',')
        # Initialize indices using "I" and "cons"
        I = self.Filter(cons, I)
        # Check for regular expression filter
        if kw.get('re') not in [None, '']:
            # Filter by regular expression
            I = self.FilterRegex(kw.get('re'), I)
        # Check for wildcard filter
        if kw.get('glob') not in [None, '']:
            # Filter by wildcard glob
            I = self.FilterWildcard(kw.get('glob'), I)
        # Check for simple substring
        if kw.get('filter') not in [None, '']:
            # Filter by substring
            I = self.FilterString(kw.get('filter'), I)
        # Output
        return I
    
  # >
   
  # ==================
  # Trajectory Subsets
  # ==================
  # <
    # Function to get sweep based on constraints
    def GetSweep(self, M, **kw):
        """
        Return a list of indices meeting sweep constraints
        
        The sweep uses the index of the first entry of ``True`` in *M*, i.e.
        ``i0=np.where(M)[0][0]``.  Then the sweep contains all other points that
        meet all criteria with respect to trajectory point *i0*.
        
        For example, using ``EqCons=['mach']`` will cause the method to return
        points with *x.mach* matching *x.mach[i0]*.
        
        :Call:
            >>> I = x.GetSweep(M, **kw)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *M*: :class:`numpy.ndarray` (:class:`bool`)
                Mask of which trajectory points should be considered
            *i0*: {``np.where(M)[0][0]``} | :class:`int`
                Index of case to use as seed of sweep
            *SortVar*: :class:`str`
                Variable by which to sort each sweep
            *EqCons*: :class:`list` (:class:`str`)
                List of trajectory keys which must match (exactly) the first
                point in the sweep
            *TolCons*: :class:`dict` (:class:`float`)
                Dictionary whose keys are trajectory keys which must match the
                first point in the sweep to a specified tolerance and whose
                values are the specified tolerances
            *IndexTol*: :class:`int`
                If specified, only trajectory points in the range
                ``[i0,i0+IndexTol]`` are considered for the sweep
        :Outputs:
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of trajectory point indices in the sweep
        :Versions:
            * 2015-05-24 ``@ddalle``: First version
            * 2017-06-27 ``@ddalle``: Added special variables
        """
        # Check for an *i0* point.
        if not np.any(M): return np.array([])
        # Copy the mask.
        m = M.copy()
        # Sort key.
        xk = kw.get('SortVar')
        # Get constraints
        EqCons  = kw.get('EqCons',  [])
        TolCons = kw.get('TolCons', {})
        # Ensure no NoneType
        if EqCons  is None: EqCons = []
        if TolCons is None: TolCons = {}
        # Get the first index.
        i0 = kw.get("i0", np.where(M)[0][0])
        # Test validity
        if i0 >= M.size:
            raise IndexError("Seed index %s exceeds dimenions of mask" % i0)
        if not M[i0]:
            raise ValueError("Seed index %s is masked to False" % i0)
        # Check for an IndexTol.
        itol = kw.get('IndexTol', self.nCase)
        # Max index to consider.
        if type(itol).__name__.startswith('int'):
            # Possible maximum index
            imax = min(self.nCase, i0+itol)
        else:
            # Do not reject points based on index.
            imax = self.nCase
        # Filter if necessary.
        if imax < self.nCase:
            # Remove from the mask
            m[imax:] = False
        # Loop through equality constraints.
        for c in EqCons:
            # Get the key (for instance if matching ``k%10``)
            k = re.split('[^a-zA-Z_]', c)[0]
            # Check for the key.
            if k in self.keys:
                # Get the target value.
                x0 = getattr(self,k)[i0]
                # Get the value
                V = eval('self.%s' % c)
            elif k == "alpha":
                # Get the target value.
                x0 = self.GetAlpha(i0)
                # Extract matrix values
                V = self.GetAlpha()
            elif k == "beta": 
                # Get the target value
                x0 = self.GetBeta(i0)
                # Extract matrix values
                V = self.GetBeta()
            elif k in ["alpha_t", "aoav"]:
                # Get the target value.
                x0 = self.GetAlphaTotal(i0)
                # Extract matrix values
                V = self.GetAlphaTotal()
            elif k in ["phi", "phiv"]:
                # Get the target value.
                x0 = self.GetPhi(i0)
                # Extract matrix values
                V = self.GetPhi()
            elif k in ["alpha_m", "aoam"]:
                # Get the target value.
                x0 = self.GetAlphaManeuver(i0)
                # Extract matrix values
                V = self.GetAlphaManeuver()
            elif k in ["phi_m", "phim"]:
                # Get the target value.
                x0 = self.GetPhiManeuver(i0)
                # Extract matrix values
                V = self.GetPhiManeuver()
            else:
                raise KeyError(
                    "Could not find trajectory key for constraint '%s'." % c)
            # Evaluate constraint
            qk = np.abs(V - x0) <= 1e-10
            # Check for special modifications
            if k in ["phi", "phi_m", "phiv", "phim"]:
                # Get total angle of attack
                aoav = self.GetAlphaTotal()
                # Test and combine with any "aoav=0" cases
                qk = np.logical_or(qk, np.abs(aoav)<=1e-10)
            # Combine constraint
            m = np.logical_and(m, qk)
        # Loop through tolerance-based constraints.
        for c in TolCons:
            # Get the key (for instance if matching 'i%10', key is 'i')
            k = re.split('[^a-zA-Z_]', c)[0]
            # Get tolerance.
            tol = TolCons[c]
            # Check for the key.
            if k in self.keys:
                # Get the target value.
                x0 = getattr(self,k)[i0]
                # Get the values
                V = eval('self.%s' % c)
            elif k == "alpha":
                # Get the target value.
                x0 = self.GetAlpha(i0)
                # Get trajectory values
                V = self.GetAlpha()
            elif k == "beta":
                # Get the target value
                x0 = self.GetBeta(i0)
                # Get trajectory values
                V = self.GetBeta()
            elif k in ["alpha_m", "aoam"]:
                # Get the target value.
                x0 = self.GetAlphaManeuver(i0)
                # Extract matrix values
                V = self.GetAlphaManeuver()
            elif k in ["phi_m", "phim"]:
                # Get the target value.
                x0 = self.GetPhiManeuver(i0)
                # Extract matrix values
                V = self.GetPhiManeuver()
            else:
                raise KeyError(
                    "Could not find trajectory key for constraint '%s'." % c)
            # Evaluate constraint
            qk = np.abs(x0-V) <= tol
            # Check for special modifications
            if k in ["phi", "phi_m", "phiv", "phim"]:
                # Get total angle of attack
                aoav = self.GetAlphaTotal()
                # Test and combine with any "aoav=0" cases
                qk = np.logical_or(qk, np.abs(aoav)<=1e-10)
            # Combine constraints
            m = np.logical_and(m, qk)
        # Initialize output.
        I = np.arange(self.nCase)
        # Apply the final mask.
        J = I[m]
        # Check for a sort variable.
        if (xk is not None):
            # Sort based on that key.
            if xk in self.keys:
                # Sort based on trajectory key
                vx = getattr(self,xk)[J]
            elif xk.lower() in ["alpha"]:
                # Sort based on angle of attack
                vx = self.GetAlpha(J)
            elif xk.lower() in ["beta"]:
                # Sort based on angle of sideslip
                vx = self.GetBeta(J)
            elif xk.lower() in ["alpha_t", "aoav"]:
                # Sort based on total angle of attack
                vx = self.GetAlphaTotal(J)
            elif xk.lower() in ["alpha_m", "aoam"]:
                # Sort based on total angle of attack
                vx = self.GetAlphaManeuver(J)
            elif xk.lower() in ["phi_m", "phim"]:
                # Sort based on velocity roll
                vx = self.GetPhiManeuver(J)
            elif xk.lower() in ["phi", "phiv"]:
                # Sort based on velocity roll
                vx = self.GetPhi(J)
            else:
                # Unable to sort
                raise ValueError("Unable to sort based on variable '%s'" % xk)
            # Order
            j = np.argsort(vx)
            # Sort the indices.
            J = J[j]
        # Output
        return J
            
    # Function to get sweep based on constraints
    def GetCoSweep(self, x0, i0, **kw):
        """
        Return a list of indices meeting sweep constraints
        
        The sweep uses point *i0* of co-trajectory *x0* as the reference for the
        constraints.
        
        For example, using ``EqCons=['mach']`` will cause the method to return
        points with *x.mach* matching *x0.mach[i0]*.
        
        :Call:
            >>> I = x.GetSweep(x0, i0, **kw)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *x0*: :class:`cape.trajectory.Trajectory`
                Another instance of the pyCart trajectory class
            *i0*: :class:`int`
                Index of point in *x0* to use as reference for constraints
            *SortVar*: :class:`str`
                Variable by which to sort each sweep
            *EqCons*: :class:`list` (:class:`str`)
                List of trajectory keys which must match (exactly) the first
                point in the sweep
            *TolCons*: :class:`dict` (:class:`float`)
                Dictionary whose keys are trajectory keys which must match the
                first point in the sweep to a specified tolerance and whose
                values are the specified tolerances
            *IndexTol*: :class:`int`
                If specified, only trajectory points in the range
                ``[i0,i0+IndexTol]`` are considered for the sweep
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of *x* indices to consider in the sweep
        :Outputs:
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of *x* indices in the sweep
        :Versions:
            * 2015-06-03 ``@ddalle``: First version
        """
        # Handle to all indices
        I0 = np.arange(self.nCase)
        # Check for list of indices
        I = kw.get('I')
        # Initial mask
        if I is not None and len(I) > 0:
            # Initialize mask
            m = np.arange(self.nCase) < 0
            # Consider cases in initial list
            m[I] = True
        else:
            # Initialize mask to ``True``.
            m = np.arange(self.nCase) > -1
        # Sort key.
        xk = kw.get('SortVar')
        # Get constraints
        EqCons  = kw.get('EqCons',  [])
        TolCons = kw.get('TolCons', {})
        # Ensure no NoneType
        if EqCons  is None: EqCons = []
        if TolCons is None: TolCons = {}
        # Check for an IndexTol.
        itol = kw.get('IndexTol', self.nCase)
        # Max index to consider.
        if type(itol).__name__.startswith('int'):
            # Possible maximum index
            imax = min(self.nCase, i0+itol)
        else:
            # Do not reject points based on index.
            imax = self.nCase
        # Filter if necessary.
        if imax < self.nCase:
            # Remove from the mask
            m[imax:] = False
        # Loop through equality constraints.
        for c in EqCons:
            # Get the key (for instance if matching ``k%10``)
            k = re.split('[^a-zA-Z_]', c)[0]
            # Check for the key.
            if k in self.keys:
                # Get the target value.
                v0 = getattr(x0,k)[i0]
                # Get the value
                V = eval('self.%s' % c)
            elif k == "alpha":
                # Get the target value.
                v0 = x0.GetAlpha(i0)
                # Get trajectory values
                V = self.GetAlpha()
            elif k == "beta": 
                # Get the target value
                v0 = x0.GetBeta(i0)
                # Extract matrix values
                V = self.GetBeta()
            elif k in ["alpha_t", "aoav"]:
                # Get the target value
                v0 = x0.GetAlphaTotal(i0)
                # Extract matrix values
                V = self.GetAlphaTotal()
            elif k in ["alpha_m", "aoam"]:
                # Get the target value.
                v0 = x0.GetAlphaManeuver(i0)
                # Extract matrix values
                V = self.GetAlphaManeuver()
            elif k in ["phi_m", "phim"]:
                # Get the target value.
                v0 = x0.GetPhiManeuver(i0)
                # Extract matrix values
                V = self.GetPhiManeuver()
            elif k in ["phi", "phiv"]:
                # Get the target value.
                v0 = x0.GetPhi(i0)
                # Extract matrix values
                V = self.GetPhi()
            else:
                raise KeyError(
                    "Could not find trajectory key for constraint '%s'." % c)
            # Evaluate constraint
            m = np.logical_and(m, np.abs(V - v0) < 1e-10)
        # Loop through tolerance-based constraints.
        for c in TolCons:
            # Get the key (for instance if matching 'i%10', key is 'i')
            k = re.split('[^a-zA-Z_]', c)[0]
            # Get tolerance.
            tol = TolCons[c]
            # Check for the key.
            if k in self.keys:
                # Get the target value.
                v0 = getattr(x0,k)[i0]
                # Evaluate the trajectory values
                V = eval('self.%s' % c)
            elif k == "alpha":
                # Get the target value.
                v0 = x0.GetAlpha(i0)
                # Get trajectory values
                V = self.GetAlpha()
            elif k == "beta": 
                # Get the target value
                v0 = x0.GetBeta(i0)
                # Extract matrix values
                V = self.GetBeta()
            elif k in ["alpha_t", "aoav"]:
                # Get the target value
                v0 = x0.GetAlphaTotal(i0)
                # Extract matrix values
                V = self.GetAlphaTotal()
            elif k in ["alpha_m", "aoam"]:
                # Get the target value.
                v0 = x0.GetAlphaManeuver(i0)
                # Extract matrix values
                V = self.GetAlphaManeuver()
            elif k in ["phi_m", "phim"]:
                # Get the target value.
                v0 = x0.GetPhiManeuver(i0)
                # Extract matrix values
                V = self.GetPhiManeuver()
            elif k in ["phi", "phiv"]:
                # Get the target value.
                v0 = x0.GetPhi(i0)
                # Extract matrix values
                V = self.GetPhi()
            else:
                raise KeyError(
                    "Could not find trajectory key for constraint '%s'." % c)
            # Evaluate constraint
            m = np.logical_and(m, np.abs(v0-V) <= tol)
        # Initialize output.
        I = np.arange(self.nCase)
        # Apply the final mask.
        J = I[m]
        # Check for a sort variable.
        if xk is not None:
            # Sort based on that key.
            j = np.argsort(getattr(self,xk)[J])
            # Sort the indices.
            J = J[j]
        # Output
        return J
        
    # Function to get set of sweeps based on criteria
    def GetSweeps(self, **kw):
        """
        Return a list of index sets in which each list contains cases that
        satisfy specified criteria.
        
        For example, using ``EqCons=['mach']`` will cause the method to return
        lists of points with the same Mach number.
        
        :Call:
            >>> J = x.GetSweeps(**kw)
        :Inputs:
            *cons*: :class:`list` (:class:`str`)
                List of global constraints; only points satisfying these
                constraints will be in one of the output sweeps
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of indices to restrict to
            *SortVar*: :class:`str`
                Variable by which to sort each sweep
            *EqCons*: :class:`list` (:class:`str`)
                List of trajectory keys which must match (exactly) the first
                point in the sweep
            *TolCons*: :class:`dict` (:class:`float`)
                Dictionary whose keys are trajectory keys which must match the
                first point in the sweep to a specified tolerance and whose
                values are the specified tolerances
            *IndexTol*: :class:`int`
                If specified, only trajectory points in the range
                ``[i0,i0+IndexTol]`` are considered for the sweep
        :Outputs:
            *J*: :class:`list` (:class:`numpy.ndarray` (:class:`int`))
                List of trajectory point sweeps
        :Versions:
            * 2015-05-25 ``@ddalle``: First version
        """
        # Expand global index constraints.
        I0 = self.GetIndices(I=kw.get('I'), cons=kw.get('cons'))
        # Initialize mask (list of ``False`` with *nCase* entries)
        M = np.arange(self.nCase) < 0
        # Set the mask to ``True`` for any cases passing global constraints.
        M[I0] = True
        # Set initial mask
        # (This is to allow cases that meet initial constraints to appear in
        #  multiple sweeps while still disallowing cases that don't meet cons)
        M0 = M.copy()
        # Initialize output.
        J = []
        # Initialize target output
        JT = []
        # Safety check: no more than *nCase* sets.
        i = 0
        # Loop through cases.
        while np.any(M) and i<self.nCase:
            # Increase number of sweeps.
            i += 1
            # Seed for this sweep
            i0 = np.where(M)[0][0]
            # Get the current sweep
            #  (Use initial mask *M0* for validity but seed based on *M*)
            I = self.GetSweep(M0, i0=i0, **kw)
            # Save the sweep
            J.append(I)
            # Update the mask
            M[I] = False
        # Output
        return J
    
  # >
   
  # =================
  # Flight Conditions
  # =================
  # <
   # ---------
   # Angles
   # ---------
   # [
    # Get angle of attack
    def GetAlpha(self, i=None):
        """Get the angle of attack
        
        :Call:
            >>> alpha = x.GetAlpha(i=None)
        :Inputs:
            *x*: :class;`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
        :Outputs:
            *alpha*: :class:`float` | :class:`np.ndarray`
                Angle of attack in degrees
        :Versions:
            * 2016-03-24 ``@ddalle``: First version
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Process the key types
        KeyTypes = [self.defns[k]['Type'] for k in self.keys]
        # Check for angle of attack
        if 'alpha' in KeyTypes:
            # Find the key
            k = self.GetKeysByType('alpha')[0]
            # Return the value
            return getattr(self,k)[i]
        # Check for total angle of attack
        if 'alpha_t' in KeyTypes:
            # Get the key
            k = self.GetKeysByType('alpha_t')[0]
            # Get the value
            av = getattr(self,k)[i]
            # Check for roll angle
            if 'phi' in KeyTypes:
                # Get the key
                k = self.GetKeysByType('phi')[0]
                # Get the value
                rv = getattr(self,k)[i]
            else:
                # Default value
                rv = 0.0
            # Convert to aoa and aos
            a, b = convert.AlphaTPhi2AlphaBeta(av, rv)
            # Output
            return a
        # No info
        return None
        
    # Get maneuver angle of attack
    def GetAlphaManeuver(self, i=None):
        """Get the signed total angle of attack
        
        :Call:
            >>> am = x.GetAlphaManeuver(i)
        :Inputs:
            *x*: :class;`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
        :Outputs:
            *am*: :class:`float`
                Signed maneuver angle of attack [deg]
        :Versions:
            * 2017-06-27 ``@ddalle``: First version
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Process the key types
        KeyTypes = [self.defns[k]['Type'] for k in self.keys]
        # Check for total angle of attack
        if 'alpha_t' in KeyTypes:
            # Find the key
            k = self.GetKeysByType('alpha_t')[0]
            # Get that value
            aoav = getattr(self,k)[i]
            # Check for 'phi'
            if 'phi' in KeyTypes:
                # Find that key
                kph = self.GetKeysByType('phi')[0]
                # Get that value
                phiv = getattr(self,kph)[i]
            else:
                # Use 0 for the roll angle
                phiv = 0.0
            # Convert to aoam, phim
            aoam, phim = convert.AlphaTPhi2AlphaMPhi(aoav, phiv)
            # Output
            return aoam
        # Check for angle of attack
        if 'alpha' in nKeyTypes:
            # Get the key
            k = self.GetKeysByType('alpha')[0]
            # Get the value
            a = getattr(self,k)[i]
            # Check for sideslip
            if 'beta' in KeyTypes:
                # Get the key
                k = self.GetKeysByType('beta')[0]
                # Get the value
                b = getattr(self,k)[i]
            else:
                # Default value
                b = 0.0
            # Convert to alpha total, phi
            av, rv = convert.AlphaBeta2AlphaMPhi(a, b)
            # Output
            return av
        # no info
        return None
        
        
    # Get total angle of attack
    def GetAlphaTotal(self, i=None):
        """Get the total angle of attack
        
        :Call:
            >>> av = x.GetAlphaTotal(i)
        :Inputs:
            *x*: :class;`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
        :Outputs:
            *av*: :class:`float`
                Total angle of attack in degrees
        :Versions:
            * 2016-03-24 ``@ddalle``: First version
            * 2017-06-25 ``@ddalle``: Added default *i* = ``None``
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Process the key types
        KeyTypes = [self.defns[k]['Type'] for k in self.keys]
        # Check for total angle of attack
        if 'alpha_t' in KeyTypes:
            # Find the key
            k = self.GetKeysByType('alpha_t')[0]
            # Get that value
            aoav = getattr(self,k)[i]
            # Check for 'phi'
            if 'phi' in KeyTypes:
                # Find that key
                kph = self.GetKeysByType('phi')[0]
                # Get that value
                phiv = getattr(self,kph)[i]
            else:
                # Use 0 for the roll angle
                phiv = 0.0
            # Convert to aoam, phim
            aoav, phiv = convert.AlphaMPhi2AlphaTPhi(aoav, phiv)
            # Output
            return aoav
        # Check for angle of attack
        if 'alpha' in nKeyTypes:
            # Get the key
            k = self.GetKeysByType('alpha')[0]
            # Get the value
            a = getattr(self,k)[i]
            # Check for sideslip
            if 'beta' in KeyTypes:
                # Get the key
                k = self.GetKeysByType('beta')[0]
                # Get the value
                b = getattr(self,k)[i]
            else:
                # Default value
                b = 0.0
            # Convert to alpha total, phi
            av, rv = convert.AlphaBeta2AlphaTPhi(a, b)
            # Output
            return av
        # no info
        return None
        
    # Get angle of sideslip
    def GetBeta(self, i=None):
        """Get the sideslip angle
        
        :Call:
            >>> beta = x.GetBeta(i)
        :Inputs:
            *x*: :class;`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
        :Outputs:
            *beta*: :class:`float`
                Angle of sideslip in degrees
        :Versions:
            * 2016-03-24 ``@ddalle``: First version
            * 2017-06-25 ``@ddalle``: Added default *i* = ``None``
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Process the key types
        KeyTypes = [self.defns[k]['Type'] for k in self.keys]
        # Check for angle of attack
        if 'beta' in KeyTypes:
            # Find the key
            k = self.GetKeysByType('beta')[0]
            # Return the value
            return getattr(self,k)[i]
        # Check for total angle of attack
        if 'alpha_t' in KeyTypes:
            # Get the key
            k = self.GetKeysByType('alpha_t')[0]
            # Get the value
            av = getattr(self,k)[i]
            # Check for roll angle
            if 'phi' in KeyTypes:
                # Get the key
                k = self.GetKeysByType('phi')[0]
                # Get the value
                rv = getattr(self,k)[i]
            else:
                # Default value
                rv = 0.0
            # Convert to aoa and aos
            a, b = convert.AlphaTPhi2AlphaBeta(av, rv)
            # Output
            return b
        # No info
        return None
        
    # Get velocity roll angle
    def GetPhi(self, i=None):
        """Get the velocity roll angle
        
        :Call:
            >>> phiv = x.GetPhi(i)
        :Inputs:
            *x*: :class;`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
        :Outputs:
            *phiv*: :class:`float`
                Velocity roll angle in degrees
        :Versions:
            * 2016-03-24 ``@ddalle``: First version
            * 2017-06-25 ``@ddalle``: Added default *i* = ``None``
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Process the key types
        KeyTypes = [self.defns[k]['Type'] for k in self.keys]
        # Check for total angle of attack
        if 'phi' in KeyTypes:
            # Find the key
            k = self.GetKeysByType('phi')[0]
            # Return the value
            return getattr(self,k)[i]
        # Check for angle of attack
        if 'alpha' in nKeyTypes:
            # Get the key
            k = self.GetKeysByType('alpha')[0]
            # Get the value
            a = getattr(self,k)[i]
            # Check for sideslip
            if 'beta' in KeyTypes:
                # Get the key
                k = self.GetKeysByType('beta')[0]
                # Get the value
                b = getattr(self,k)[i]
            else:
                # Default value
                b = 0.0
            # Convert to alpha total, phi
            av, rv = convert.AlphaBeta2AlphaTPhi(a, b)
            # Output
            return rv
        # no info
        return None
        
    # Get maneuver angle of attack
    def GetPhiManeuver(self, i=None):
        """Get the signed maneuver roll angle
        
        :Call:
            >>> phim = x.GetPhiManeuver(i)
        :Inputs:
            *x*: :class;`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
        :Outputs:
            *phim*: :class:`float`
                Signed maneuver roll angle [deg]
        :Versions:
            * 2017-06-27 ``@ddalle``: First version
            * 2017-07-20 ``@ddalle``: Added default *i* = ``None``
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Process the key types
        KeyTypes = [self.defns[k]['Type'] for k in self.keys]
        # Check for total angle of attack
        if 'phi' in KeyTypes:
            # Find the key
            k = self.GetKeysByType('phi')[0]
            # Get that value
            phiv = getattr(self,k)[i]
            # Check for 'phi'
            if 'alpha_t' in KeyTypes:
                # Find that key
                k = self.GetKeysByType('alpha_t')[0]
                # Get that value
                aoav = getattr(self,k)[i]
            else:
                # Use 0 for the roll angle
                aoav = 1.0
            # Convert to aoam, phim
            aoam, phim = convert.AlphaTPhi2AlphaMPhi(aoav, phiv)
            # Output
            return phim
        # Check for angle of attack
        if 'alpha' in nKeyTypes:
            # Get the key
            k = self.GetKeysByType('alpha')[0]
            # Get the value
            a = getattr(self,k)[i]
            # Check for sideslip
            if 'beta' in KeyTypes:
                # Get the key
                k = self.GetKeysByType('beta')[0]
                # Get the value
                b = getattr(self,k)[i]
            else:
                # Default value
                b = 0.0
            # Convert to alpha total, phi
            av, rv = convert.AlphaBeta2AlphaMPhi(a, b)
            # Output
            return av
        # no info
        return None
   # ]
   
   # -------------------------
   # Dimensionless Parameters
   # -------------------------
   # [
    # Get Reynolds number
    def GetReynoldsNumber(self, i=None, units=None):
        """Get Reynolds number (per foot)
        
        :Call:
            >>> Re = x.GetReynoldsNumber(i=None, units=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int` | :class:`list`
                Case number(s)
            *units*: {``None``} | :class:`str` | :class:`unicode`
                Requested units for output
        :Outputs:
            *Re*: :class:`float`
                Reynolds number [1/inch | 1/ft]
        :Versions:
            * 2016-03-23 ``@ddalle``: First version
            * 2017-07-19 ``@ddalle``: Added default conditions
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Check for Reynolds number key
        k = self.GetFirstKeyByType("Re")
        # Default unit system
        us = self.gas.get("UnitSystem", "fps")
        # Check default units based on input
        if us == "mks":
            # MKS: 1/m in general
            udef = "1/m"
        else:
            # FPS: 1/inch ... watch out for 1/ft
            udef = "1/inch"
        # Check for Reynolds number key
        if k is not None:
            # Get value directly
            return self.GetKeyValue(k, i, udef=udef, units=units)
        # Get parameters that could be used
        kM = self.GetFirstKeyByType("Mach")
        kT = self.GetFirstKeyByType("T")
        kp = self.GetFirstKeyByType("p")
        kq = self.GetFirstKeyByType("q")
        # Likely to need *gamma*
        gam = self.GetGamma(i)
        # Likely to need gas constant
        R = self.GetNormalizedGasConstant(i, units="m^2/s^2/K")
        # Consider cases
        if kM and kT and kq:
            # Get values
            M = self.GetMach(i)
            T = self.GetTemperature(i, units="K")
            q = self.GetDynamicPressure(i, units="Pa")
            # Calculate static pressure
            p = q / (0.5*gam*M*M)
            # Get viscosity
            mu = self.GetViscosity(i, units="kg/m/s")
            # Get dimensional speed
            U = M*np.sqrt(gam*R*T)
            # Get density
            rho = p / (R*T)
            # Convert Reynolds number
            Re = rho*U/mu
        elif kM and kT and kp:
            # Get values
            M = self.GetMach(i)
            T = self.GetTemperature(i)
            p = self.GetPressure(i)
            # Get viscosity
            mu = self.GetViscosity(i)
            # Get dimensional speed
            U = M*np.sqrt(gam*R*T)
            # Get density
            rho = p / (R*T)
            # Convert Reynolds number
            Re = rho*U/mu
        else:
            # Unprocessed
            return None
        # Reduce by requested units
        if units is None:
            # Use default inputs
            return Re / mks(udef)
        else:
            # Requested units
            return Re / mks(units)
        
        
        
        # Process the key types.
        KeyTypes = [self.defns[k]['Type'] for k in self.keys]
        # Check for Reynolds number
        if 'Re' in KeyTypes:
            # Find the key.
            k = self.GetKeysByType('Re')[0]
            # Set the value.
            return getattr(self,k)[i]
        # If reached here, we need 'T' and "Mach' at least
        if 'Mach' not in KeyTypes:
            return None
        elif 'T' not in KeyTypes:
            return None
        # Get temperature and Mach keys
        kM = self.GetKeysByType('Mach')[0]
        kT = self.GetKeysByType('T')[0]
        # Get temperature and Mach values
        M = getattr(self,kM)[i]
        T = getattr(self,kT)[i]
        # Get units
        u = self.defns[kT].get('Units', 'fps')
        # Check for pressure specifier
        if 'p' in KeyTypes:
            # Find the key.
            k = self.GetKeysByType('p')[0]
            # Get the pressure
            p = getattr(self,k)[i]
            # Calculate Reynolds number
            if u.lower() in ['fps', 'r']:
                # Imperial units
                return convert.ReynoldsPerFoot(p, T, M)/12
            else:
                # MKS units
                return convert.ReynoldsPerMeter(p, T, M)
        elif 'q' in KeyTypes:
            # Find the key.
            k = self.GetKeysByType('q')[0]
            # Get the dynamic pressure
            q = getattr(self,k)[i]
            # Calculate static pressure
            p = q / (0.7*M*M)
            # Calculate Reynolds number
            if u.lower() in ['fps', 'r']:
                # Imperial units
                return convert.ReynoldsPerFoot(p, T, M)/12
            else:
                # MKS units
                return convert.ReynoldsPerMeter(p, T, M)
        # If we reach here, missing info.
        return None
        
    # Get Mach number
    def GetMach(self, i=None):
        """Get Mach number
        
        :Call:
            >>> M = x.GetMach(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case number
        :Outputs:
            *M*: :class:`float`
                Mach number
        :Versions:
            * 2016-03-24 ``@ddalle``: First version
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Search for key
        k = self.GetFirstKeyByType("Mach")
        # Check if found
        if k is not None:
            # Return the value
            return getattr(self,k)[i]
        # If we reach this point... no conversion figured out
        return None
   # ]
   
   # -------------
   # Utilities
   # -------------
   # [
    # Get unitized value
    def GetKeyValue(self, k, i=None, units=None, udef="1"):
        """Get the value of one key with appropriate dimensionalization
        
        :Call:
            >>> v = x.GetKeyValue(k, i=None, units=None, udef="1")
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *k*: :class:`str`
                Name of run matrix variable (trajectory key)
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
            *units*: {``None``} | ``"mks"`` | :class:`str`
                Output units
        :Outputs:
            *v*: :class:`float`
                Value of key *k* for case *i* in units of *units*
        :Versions:
            * 2018-04-13 ``@ddalle``: First version
        """
        # Check if key exists
        if k not in self.keys:
            raise KeyError("No trajectory key called '%s'" % k)
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Get value of the variable for case(s) *i*
        v = getattr(self,k)[i]
        # Get the units
        u = self.defns[k].get("Units", udef)
        # Check units
        if units is None:
            # No conversion
            return v
        elif units.lower() == "mks":
            # Convert input units to MKS
            return v * mks(u)
        else:
            # Convert to requested units
            return v * mks(u)/mks(units)
   # ]
   
   # ----------------
   # State Variables
   # ----------------
   # [
   
    # Get freestream density
    def GetDensity(self, i=None, units=None):
        """Get freestream density
        :Call:
            >>> rho = x.GetDensity(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
            *units*: {``None``} | ``"mks"`` | ``"F"`` | ``"R"``
                Output units
        :Outputs:
            *r*: :class:`float`
                freestream density [ ]
        :Versions:
            * 2018-04-13 ``@jmeeroff``: First version
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Check for dynamic pressure key
        kr = self.GetFirstKeyByType('rho')
        # Check for dynamic pressure
        if kr is not None:
            # Get value directly
            return self.GetKeyValue(kr, i)
        # If we reach this point, we need two other parameters
        kM = self.GetFirstKeyByType("Mach")
        kT = self.GetFirstKeyByType("T")
        kV = self.GetFirstKeyByType("V")
        kp = self.GetFirstKeyByType("p")
        kq = self.GetFirstKeyByType("q")
        kR = self.GetFirstKeyByType("Re")
        # Get the ratio of specific heats in case we need to use it
        gam = self.GetGamma(i)
        # Search for a combination of parameters we can interpret
        if kV and kq:
            # Dynamic pressure and velocity
            q   = self.GetKeyValue(kq, i)
            V   = self.GetKeyValue(kV, i)
            # Calculate density
            rho = 2*q*(1/V)*(1/V)  
        elif kT and kp:
            # Pressure and Temperature (ideal gas law)
            p   = self.GetKeyValue(kp, i)
            T   = self.GetKeyValue(kT, i)
            # Get gas constant R
            R = self.GetNormalizedGasConstant(i)
            # Calculate density
            rho = p*(1/T)*(1/R)
        elif kq and kM and kT:
            # dynamic pressure and mach number (with Temperature)
            q   = self.GetKeyValue(kq, i)
            M   = self.GetKeyValue(kM, i)
            T   = self.GetKeyValue(kT, i)
            # Get gas constant
            R = self.GetNormalizedGasConstant(i)
            # Sound speed
            a = np.sqrt(gam*R*T)
            # Velocity
            V = a*M
            # Dynamic pressure
            rho = 2*q*(1/V)*(1/V)          
       
        # Output with units
        if units is None:
            # No conversion
            return rho
        else:
            # Apply expected units
            return rho / mks("units")
            
        # If we reach this point... not trying other conversions
        return None
        
    # Get velocity
    def GetVelocity(self, i=None, units=None):
        """Get velocity
        :Call:
            >>> U = x.GetVelocity(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
            *units*: {``None``} | ``"mks"`` | ``"F"`` | ``"R"``
                Output units
        :Outputs:
            *r*: :class:`float`
                velocity [ mps | fps ]
        :Versions:
            * 2018-04-13 ``@jmeeroff``: First version
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Check for dynamic pressure key
        kV = self.GetFirstKeyByType('V')
        # Check for dynamic pressure
        if kV is not None:
            # Get value directly
            return self.GetKeyValue(kV, i)
        # If we reach this point, we need two other parameters
        kM = self.GetFirstKeyByType("Mach")
        kT = self.GetFirstKeyByType("T")
        kr = self.GetFirstKeyByType("rho")
        kp = self.GetFirstKeyByType("p")
        kq = self.GetFirstKeyByType("q")
        kR = self.GetFirstKeyByType("Re")
        # Get the ratio of specific heats in case we need to use it
        gam = self.GetGamma(i)
        # Search for a combination of parameters we can interpret
        if kM and kT:
            # Mach number and Temperature
            M   = self.GetKeyValue(kM, i)
            T   = self.GetKeyValue(kT, i)
            # Get gas constant
            R = self.GetNormalizedGasConstant(i)
            # Sound speed
            a = np.sqrt(gam*R*T)
            # Calculate velocity
            U = M * a  
        elif kr and kq:
            # Density and dynamic pressure
            r   = self.GetKeyValue(kr, i)
            q   = self.GetKeyValue(kq, i)
            # Calculate velocity
            U = np.sqrt(2*q/r)
        elif kM and kp and kr:
            # Mach number, pressure, and density
            M   = self.GetKeyValue(kM, i)
            p   = self.GetKeyValue(kp, i)
            r   = self.GetKeyValue(kr, i)
            # gas constant
            R = self.GetNormalizedGasConstant(i)
            # speed of sound
            a = np.sqrt(gam*p/r)
            # Calculate velocity
            U = M * a  

        # Output with units
        if units is None:
            # No conversion
            return U
        else:
            # Apply expected units
            return U / mks("units")
            
        # If we reach this point... not trying other conversions
        return None        
        
        
        
    # Get freestream temperature
    def GetTemperature(self, i=None, units=None):
        """Get static freestream temperature
        
        :Call:
            >>> T = x.GetTemperature(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
            *units*: {``None``} | ``"mks"`` | ``"F"`` | ``"R"``
                Output units
        :Outputs:
            *T*: :class:`float`
                Static temperature [R | K]
        :Versions:
            * 2016-03-24 ``@ddalle``: First version
            * 2017-06-25 ``@ddalle``: Added default *i* = ``None``
            * 2018-04-13 ``@ddalle``: Units
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Search for temperature key
        k = self.GetFirstKeyByType('T')
        # Check unit system
        us = self.gas.get("UnitSystem", "fps").lower()
        # Check default units based on input
        if us == "mks":
            # MKS: Kelvins
            udef = "K"
        else:
            # FPS: degrees Rankine
            udef = "R"
        # Check for temperature key hit
        if k is not None:
            # Get appropriately unitized value
            return self.GetKeyValue(k, i, units=units, udef=udef)
        # If we reach this point... not trying other conversions
        return None
        
    # Get freestream stagnation temperature
    def GetTotalTemperature(self, i=None):
        """Get freestream stagnation temperature
        
        :Call:
            >>> T0 = x.GetTotalTemperature(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
        :Outputs:
            *T0*: :class:`float`
                Freestream stagnation temperature [R | K]
        :Versions:
            * 2016-08-30 ``@ddalle``: First version
            * 2017-07-20 ``@ddalle``: Added default cases
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Get temperature, Mach number, and ratio of specific heats
        T = self.GetTemperature(i)
        M = self.GetMach(i)
        g = self.GetGamma(i)
        # Calculate stagnation temperature
        return T * (1 + 0.5*(g-1)*M*M)
        
    # Get freestream pressure
    def GetPressure(self, i=None):
        """Get static freestream pressure (in psf or Pa)
        
        :Call:
            >>> p = x.GetPressure(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
        :Outputs:
            *p*: :class:`float`
                Static pressure [psf | Pa]
        :Versions:
            * 2016-03-24 ``@ddalle``: First version
            * 2017-07-20 ``@ddalle``: Added default cases
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Process the key types
        KeyTypes = [self.defns[k]['Type'] for k in self.keys]
        # Check for static pressure
        if 'p' in KeyTypes:
            # Find the key.
            k = self.GetKeysByType('p')[0]
            # Output
            return getattr(self,k)[i]
        # If we reach this point, we need at least Mach and temperature
        # Get temperature and Mach keys
        kM = self.GetKeysByType('Mach')[0]
        kT = self.GetKeysByType('T')[0]
        # Get temperature and Mach values
        M = getattr(self,kM)[i]
        T = getattr(self,kT)[i]
        # Get units
        u = self.defns[kT].get('Units', 'fps')
        # Check for pressure specifier
        if 'Re' in KeyTypes:
            # Find the key.
            k = self.GetKeysByType('Re')[0]
            # Get the Reynolds number
            Re = getattr(self,k)[i]
            # Calculate Reynolds number
            if u.lower() in ['fps', 'r']:
                # Imperial units
                return convert.PressureFPSFromRe(Re*12, T, M)
            else:
                # MKS units
                return convert.PressureMKSFromRe(Re, T, M)
        elif 'q' in KeyTypes:
            # Find the key.
            k = self.GetKeysByType('q')[0]
            # Get the dynamic pressure
            q = getattr(self,k)[i]
            # Calculate static pressure
            return q / (0.7*M*M)
        # If we reach here, missing info.
        return None
    
    # Get freestream pressure
    def GetDynamicPressure(self, i=None, units=None):
        """Get dynamic freestream pressure (in psf or Pa)
        
        :Call:
            >>> q = x.GetDynamicPressure(i=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int` | :class:`list`
                Case number(s)
            *units*: {``None``} | ``"mks"`` | :class:`str`
                Output units
        :Outputs:
            *q*: :class:`float`
                Dynamic pressure [psf | Pa | *units*]
        :Versions:
            * 2016-03-24 ``@ddalle``: First version
            * 2017-07-20 ``@ddalle``: Added default cases
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Check for dynamic pressure key
        kq = self.GetFirstKeyByType('q')
        # Default unit system
        us = self.gas.get("UnitSystem", "fps")
        # Check default units based on input
        if us == "mks":
            # MKS: N/m^2 ... kg/(m*s)
            udef = "Pa"
        else:
            # FPS: lbf/ft^2
            udef = "psf"
        # Check for dynamic pressure
        if kq is not None:
            # Get value directly
            return self.GetKeyValue(kq, i, units=units, udef=udef)
        # If we reach this point, we need two other parameters
        kM = self.GetFirstKeyByType("Mach")
        kT = self.GetFirstKeyByType("T")
        kV = self.GetFirstKeyByType("V")
        kp = self.GetFirstKeyByType("p")
        kr = self.GetFirstKeyByType("rho")
        kR = self.GetFirstKeyByType("Re")
        # Get the ratio of specific heats in case we need to use it
        gam = self.GetGamma(i)
        # The gas constant is often needed
        R = self.GetNormalizedGasConstant(i)
        # Search for a combination of parameters we can interpret
        if kV and kr:
            # Density and velocity; easy
            rho = self.GetKeyValue(kr, i)
            V   = self.GetKeyValue(kV, i)
            # Calculate dynamic pressure
            q = 0.5*rho*V*V
        elif kM and kp:
            # Pressure and Mach
            p   = self.GetKeyValue(kp, i)
            M   = self.GetKeyValue(kM, i)
            # Calculate dynamic pressure
            q = 0.5*gam*p*M*M
        elif kM and kr and kT:
            # Density and Mach (and temperature to get speed)
            rho = self.GetKeyValue(kr, i)
            M   = self.GetKeyValue(kM, i)
            T   = self.GetKeyValue(kT, i)
            # Sound speed
            a = np.sqrt(gam*R*T)
            # Velocity
            V = a*M
            # Dynamic pressure
            q = 0.5*rho*V*V
        elif kV and kp and kT:
            # Pressure, Mach, and temperature
            p = self.GetKeyValue(kp, i)
            M = self.GetKeyValue(kM, i)
            V = self.GetKeyValue(kV, i)
            # Sound speed
            a = np.sqrt(gam*R*T)
            # Mach number
            M = V/a
            # Dynamic pressure
            q = 0.5*gam*p*M*M
        elif kR and kM and kT:
            # Get Reynolds number, Mach number, and temperature
            Re = self.GetKeyValue(kR, i)
            M  = self.GetKeyValue(kM, i)
            T  = self.GetKeyValue(kT, i)
            # Sound speed
            a = np.sqrt(gam*R*T)
            # Velocity
            U = M*a
            # Viscosity
            mu = self.GetViscosity(i)
            # Velocity
            V = M*a
            # Pressure
            p = Re*mu*R*T/V
            # Dynamic pressure
            q = 0.5*gam*p*M*M
            
        # Output with units
        if units is None:
            # No conversion
            return q
        else:
            # Apply expected units
            return q / mks("units")
        
    # Get viscosity
    def GetViscosity(self, i=None, units=None):
        """Get the dynamic viscosity for case(s) *i*
        
        :Call:
            >>> mu = x.GetViscosity(i=None, units=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int` | :class:`list`
                Case number(s)
            *units*: {``None``} | ``"mks"`` | :class:`str`
                Output units
        :Outputs:
            *q*: :class:`float`
                Dynamic pressure [psf | Pa | *units*]
        :Versions:
            * 2018-04-13 ``@ddalle``: First version
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Check for dynamic pressure key
        k = self.GetFirstKeyByType('mu')
        # Default unit system
        us = self.gas.get("UnitSystem", "fps")
        # Check default units based on input
        if us == "mks":
            # MKS: Kelvins
            udef = "kg/m/s"
        else:
            # FPS: degrees Rankine
            udef = "lbm/ft/s"
        # Check for dynamic pressure
        if k is not None:
            # Get value directly
            return self.GetKeyValue(k, i, units=units, udef=udef)
        # Get temperature
        T = self.GetTemperature(i, units="K")
        # Reference parameters
        mu0 = self.GetSutherland_mu0(i, units="kg/m/s")
        T0  = self.GetSutherland_T0(i, units="K")
        C   = self.GetSutherland_C(i, units="K")
        # Sutherland's law
        mu = convert.SutherlandMKS(T, mu0=mu0, T0=T0, C=C)
        # Check for units
        if units is None:
            # No conversion
            return mu / mks(udef)
        else:
            # Convert to requested units
            return mu / mks(units)
        
        
    # Get freestream stagnation pressure
    def GetTotalPressure(self, i=None):
        """Get freestream stagnation pressure (in psf or Pa)
        
        :Call:
            >>> p0 = x.GetTotalPressure(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
        :Outputs:
            *p0*: :class:`float`
                Stagnation pressure [psf | Pa]
        :Versions:
            * 2016-08-30 ``@ddalle``: First version
            * 2017-07-20 ``@ddalle``: Added default cases
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Get pressure, Mach number, and gamma
        p = self.GetPressure(i)
        M = self.GetMach(i)
        g = self.GetGamma(i)
        # Other gas constants
        g2 = 0.5*(g-1)
        g3 = g/(g-1)
        # Calculate stagnation pressure
        return p * (1+g2*M*M)**g3
   # ]
   
   # -------------------------
   # Thermodynamic Properties
   # -------------------------
   # [
    # Get parameter from freestream state
    def GetGasProperty(self, k, vdef=None):
        """Get property from the ``"Freestream"`` section
        
        :Call:
            >>> v = x.GetGasProperty(k, vdef=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *k*: :class:`str`
                Name of parameter
            *vdef*: {``None``} | :class:`any`
                Default value for the parameter
        :Outputs:
            *v*: :class:`float` | :class:`str` | :class:`any`
                Value of the 
        :Versions:
            * 2016-03-24 ``@ddalle``: First version
        """
        # Get the parameter
        return self.gas.get(k, vdef)
    
    # Get ratio of specific heats
    def GetGamma(self, i=None):
        """Get freestream ratio of specific heats
        
        :Call:
            >>> gam = x.GetGamma(i)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
        :Outputs:
            *gam*: :class:`float`
                Ratio of specific heats
        :Versions:
            * 2016-03-29 ``@ddalle``: First version
            * 2017-07-20 ``@ddalle``: Added default cases
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Attempt to find a special key
        kg = self.GetFirstKeyByType('gamma')
        # Check for a matching key
        if kg is None:
            # Get value from the gas definition
            return self.gas.get("Gamma", 1.4)
        else:
            # Use the trajectory value
            return getattr(self,k)[i]
            
    # Get molecular weight
    def GetMolecularWeight(self, i=None, units=None):
        """Get averaged freestream gas molecular weight
        
        :Call:
            >>> W = x.GetMolecularWeight(i=None, units=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
            *units*: {``None``} | :class:`str`
                Requested units of output
        :Outputs:
            *W*: :class:`float`
                Molecular weight [kg/kmol | *units* ]
        :Versions:
            * 2018-04-13 ``@ddalle``: First version
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Attempt to find a special key
        kW = self.GetFirstKeyByType('MW')
        # Check for a matching key
        if kW is None:
            # Check for molecular weight and gas constant from "Freestream"
            MW = self.gas.get("MolecularWeight")
            R  = self.gas.get("GasConstant")
            Ru = self.gas.get("UniversalGasConstant", 8314.4598)
            # Check special cases
            if (MW is None) and (R is None):
                # Use default molecular weight
                W = Ru/287.00
            elif MW is None:
                # Infer from gas constant
                W = Ru/R
            else:
                # Molecular weight is present
                W = MW
        else:
            # Get from run matrix
            W = getattr(self,kW)[i]
        # Output with units
        if units is None:
            # No conversion
            return W
        else:
            # Reduce by requested units
            return W / mks(units)
    
    # Get normalized gas constant
    def GetNormalizedGasConstant(self, i=None, units=None):
        """Get averaged freestream gas molecular weight
        
        :Call:
            >>> R = x.GetNormalizedGasConstant(i=None, units=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
            *units*: {``None``} | :class:`str`
                Requested units of output
        :Outputs:
            *R*: :class:`float`
                Normalized gas constant [J/kg*K | *units* ]
        :Versions:
            * 2018-04-13 ``@ddalle``: First version
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Attempt to find a special key
        kW = self.GetFirstKeyByType('MW')
        # Check for a matching key
        if kW is None:
            # Check for molecular weight and gas constant from "Freestream"
            MW = self.gas.get("MolecularWeight")
            R  = self.gas.get("GasConstant")
            Ru = self.gas.get("UniversalGasConstant", 8314.4598)
            # Check special cases
            if (MW is None) and (R is None):
                # Use default gas constant
                R = 287.00
            elif R is None:
                # Infer from molecular weight
                R = Ru/MW
            else:
                # Gas constant is present
                R = R
        else:
            # Get from run matrix
            R = getattr(self,kR)[i]
        # Output with units
        if units is None:
            # No conversion
            return R
        else:
            # Reduce by requested units
            return R / mks(units)
    
    # Sutherland's law reference viscosity
    def GetSutherland_mu0(self, i=None, units=None):
        """Get reference viscosity for Sutherland's Law
        
        :Call:
            >>> mu0 = x.GetSutherland_mu0(i=None, units=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
            *units*: {``None``} | :class:`str`
                Requested units of output
        :Outputs:
            *mu0*: :class:`float`
                Reference viscosity [ kg/m*s | *units* ]
        :Versions:
            * 2018-04-13 ``@ddalle``: First version
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Default unit system
        us = self.gas.get("UnitSystem", "fps")
        # Check default units based on input
        if us == "mks":
            # MKS kg/m/s
            udef = "kg/m/s"
        else:
            # FPS: ... maybe slug/ft/s instead of expected lbm/ft/s
            udef = "slug/ft/s"
        # Check for override on units
        udef = self.gas.get("Sutherland_mu0_Units", udef)
        # Conversion factor
        cdef = mks(udef)
        # Default value
        mudef = 1.716e-5 / cdef
        # Get value from freestream state
        mu0 = self.gas.get("Sutherland_mu0", mudef)
        # Check for units
        if units is None:
            # No conversion
            return mu0
        else:
            # Reduce by requested units
            return mu0 * cdef / mks(units)
    
    # Sutherland's law reference temperature
    def GetSutherland_T0(self, i=None, units=None):
        """Get reference temperature for Sutherland's Law
        
        :Call:
            >>> T0 = x.GetSutherland_T0(i=None, units=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
            *units*: {``None``} | :class:`str`
                Requested units of output
        :Outputs:
            *T0*: :class:`float`
                Reference temperature [ K | *units* ]
        :Versions:
            * 2018-04-13 ``@ddalle``: First version
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Default unit system
        us = self.gas.get("UnitSystem", "fps")
        # Check default units based on input
        if us == "mks":
            udef = "K"
        else:
            udef = "R"
        # Check for units
        udef = self.gas.get("Sutherland_T0_Units", udef)
        # Conversion factor
        cdef = mks(udef)
        # Default temperature
        Tdef = 273.15 / cdef
        # Get value from freestream state
        T0 = self.gas.get("Sutherland_T0", Tdef)
        # Check for units
        if units is None:
            # No conversion
            return T0
        else:
            # Reduce by requested units
            return T0 * cdef / mks(units)
    
    # Sutherland's law reference temperature
    def GetSutherland_C(self, i=None, units=None):
        """Get reference temperature for Sutherland's Law
        
        :Call:
            >>> C = x.GetSutherland_C(i=None, units=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: {``None``} | :class:`int`
                Case number (return all if ``None``)
            *units*: {``None``} | :class:`str`
                Requested units of output
        :Outputs:
            *C*: :class:`float`
                Reference temperature [ K | *units* ]
        :Versions:
            * 2018-04-13 ``@ddalle``: First version
        """
        # Default list
        if i is None:
            i = np.arange(self.nCase)
        # Default unit system
        us = self.gas.get("UnitSystem", "fps")
        # Check default units based on input
        if us == "mks":
            udef = "K"
        else:
            udef = "R"
        # Check for units
        udef = self.gas.get("Sutherland_C_Units", udef)
        # Conversion factor
        cdef = mks(udef)
        # Default temperature
        Cdef = 110.33333 / cdef
        # Get value from freestream state
        C = self.gas.get("Sutherland_C", Cdef)
        # Check for units
        if units is None:
            # No conversion
            return C
        else:
            # Reduce by requested units
            return C * cdef / mks(units)
    
   # ]
  # >
   
  # ===========
  # SurfBC Keys
  # ===========
  # <
    # Get generic input for SurfBC key
    def GetSurfBC_ParamType(self, key, k, comp=None):
        """Get generic parameter value and type for a surface BC key
        
        :Call:
            >>> v, t = x.GetSurfBC_ParamType(key, k, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *k*: :class:`str`
                Name of input parameter to find
            *key*: :class:`str`
                Name of trajectory key to use
            *comp*: ``None`` | :class:`str`
                If *v* is a dict, use *v[comp]* if *comp* is nontrivial
            *typ*: ``"SurfBC"`` | :class:`str`
                Trajectory key type to process
        :Outputs:
            *v*: ``None`` | :class:`any`
                Value of the parameter
            *t*: :class:`str`
                Name of the type of *v* (``type(v).__name__``)
        :Versions:
            * 2016-08-29 ``@ddalle``: First version
        """
        # Get the raw parameter
        v = self.defns[key].get(k)
        # Type
        t = type(v).__name__
        # Check for dictionary
        if (t == 'dict') and (comp is not None):
            # Reprocess for this component.
            v = v.get(comp)
            t = type(v).__name__
        # Output
        return v, t
        
    # Process input for generic type
    def GetSurfBC_Val(self, i, key, v, t, vdef=None, **kw):
        """Default processing for processing a key by value
        
        :Call:
            >>> V = x.GetSurfBC_Val(i, key, v, t, vdef=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *key*: :class:`str`
                Name of trajectory key to use
            *v*: ``None`` | :class:`any`
                Value of the parameter
            *t*: :class:`str`
                Name of the type of *v* (``type(v).__name__``)
            *vdef*: ``None``  | :class:`any`
                Default value for *v* if *v* is ``None``
        :Outputs:
            *V*: :class:`any`
                Processed key, for example ``getattr(x,key)[i]``
        :Versions:
            * 2016-08-29 ``@ddalle``: First version
        """
        # Process the type and value
        if v is None:
            # Check for default value
            if vdef is None:
                # Use the value directly from the trajectory key
                return getattr(self, key)[i]
            else:
                # Use the default value
                return vdef
        elif t in ['str', 'unicode']:
            # Check for specially-named keys
            if v in kw:
                # Use another function
                return getattr(self, kw[v])(i)
            else:
                # Use the value of a different key.
                return getattr(self, v)[i]
        else:
            # Use the value directly.
            return v
            
    # Process input for generic SurfBC key/param
    def GetSurfBC_Param(self, i, key, k, comp=None, vdef=None, **kw):
        """Process a single parameter of a SurfBC key
        
        :Call:
            >>> v = x.GetSurfBC_Param(i, key, k, comp=None, vdef=None, **kw)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: :class:`str`
                Name of trajectory key to use
            *k*: :class:`str`
                Name of input parameter to find
            *comp*: ``None`` | :class:`str`
                If *v* is a dict, use *v[comp]* if *comp* is nontrivial
            *vdef*: ``None`` | :class:`any`
                Default value for *v* if *v* is ``None``
            *typ*: {``"SurfBC"``} | :class:`str`
                Trajectory key type to process
        :Keyword arguments:
            *j*: :class:`str`
                Name of function to use if parameter is a string
        :Outputs:
            *v*: ``None`` | :class:`any`
                Value of the parameter
        :Versions:
            * 2016-08-31 ``@ddalle``: First version
        """
        # Process keywords
        typ = kw.get('typ', 'SurfBC')
        # Process key
        key = self.GetKeyName(typ, key)
        # Get the parameter and value
        v, t = self.GetSurfBC_ParamType(key, k, comp=comp)
        # Get value
        return self.GetSurfBC_Val(i, key, v, t, vdef=vdef, **kw)
    
    # Get thrust for SurfCT input
    def GetSurfCT_Thrust(self, i, key=None, comp=None):
        """Get thrust input for surface *CT* key
        
        :Call:
            >>> CT = x.GetSurfCT_Thrust(i, key=None, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
            *comp*: ``None`` | :class:`str`
                Name of component to access if *CT* is a :class:`dict`
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *CT*: :class:`float`
                Thrust parameter, either thrust or coefficient
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
            * 2016-08-29 ``@ddalle``: Added component capability
        """
        # Process as SurfCT key
        return self.GetSurfBC_Param(i, key, 'Thrust', comp=comp, typ='SurfCT')
            
    # Get reference dynamic pressure
    def GetSurfCT_RefDynamicPressure(self, i, key=None, comp=None):
        """Get reference dynamic pressure for surface *CT* key
        
        :Call:
            >>> qinf = x.GetSurfCT_RefDynamicPressure(i, key=None, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *qinf*: :class:`float`
                Reference dynamic pressure to use, this divides the *CT* value
        :Versions:
            * 2016-04-12 ``@ddalle``: First version
            * 2016-08-29 ``@ddalle``: Added component capability
        """
        # Special translations
        kw = {
            "freestream": "GetDynamicPressure",
            "inf":        "GetDynamicPressure",
        }
        # Name of parameter
        k = 'RefDynamicPressure'
        # Process the key
        return self.GetSurfBC_Param(i, key, k, comp=comp, typ='SurfCT', **kw)
            
    # Get total temperature
    def GetSurfCT_RefPressure(self, i, key=None, comp=None):
        """Get reference pressure input for surface *CT* total pressure
        
        :Call:
            >>> Tref = x.GetSurfCT_RefPressure(i, key=None, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *pref*: :class:`float`
                Reference pressure for normalizing *T0*
        :Versions:
            * 2016-04-13 ``@ddalle``: First version
        """
        # Call the SurfBC equivalent
        return self.GetSurfBC_RefPressure(i, key, comp=comp, typ="SurfCT")
            
    # Get pressure calibration factor
    def GetSurfCT_PressureCalibration(self, i, key=None, comp=None):
        """Get pressure calibration factor for *CT* key
        
        :Call:
            >>> fp = x.GetSurfCT_PressureCalibration(i, key=None, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *fp*: {``1.0``} | :class:`float`
                Pressure calibration factor
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Call the SurfBC equivalent
        return self.GetSurfBC_PressureCalibration(i, key,
            comp=comp, typ="SurfCT")
            
    # Get pressure additive calibration
    def GetSurfCT_PressureOffset(self, i, key=None, comp=None):
        """Get offset used for calibration of static or stagnation pressure
        
        The value used by :mod:`cape` is given by
        
        .. math::
            
            \\tilde{p} = \\frac{b + ap}{p_\\mathit{ref}}
            
        where :math:`\\tilde{p}` is the value used in the namelist, *b* is the
        value from this function, *a* is the result of
        :func:`GetSurfBC_PressureCalibration`, *p* is the input value from the
        JSON file, and :math:`p_\\mathit{ref}` is the value from
        :func:`GetSurfBC_RefPressure`.  In code, this is
        
        .. code-block:: python
        
            p_tilde = (bp + fp*p) / pref
        
        :Call:
            >>> bp = x.GetSurfCT_PressureOffset(i, key=None, comp=None)
        :Inputs:
            *x*: :Class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *bp*: {``0.0``} | :class:`float`
                Stagnation or static pressure offset
        :Versions:
            * 2016-08-29 ``@ddalle``: First version
        """
        # Call the SurfBC equivalent
        return self.GetSurfBC_PressureOffset(i, key, comp=comp, typ="SurfCT")
        
    # Get stagnation pressure input for SurfBC input
    def GetSurfCT_TotalPressure(self, i, key=None, comp=None):
        """Get stagnation pressure input for surface *CT* key
        
        :Call:
            >>> p0 = x.GetSurfCT_TotalPressure(i, key=None, comp=None, **kw)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
            *typ*: {``"SurfBC"``} | ``"SurfCT"`` | 
                Type to use for checking properties of *key*
        :Outputs:
            *p0*: :class:`float`
                Stagnation pressure parameter, usually *p0/pinf*
        :Versions:
            * 2016-03-28 ``@ddalle``: First version
        """
        # Call the SurfBC equivalent
        return self.GetSurfBC_TotalPressure(i, key, comp=comp, typ="SurfCT")
            
    # Get total temperature
    def GetSurfCT_TotalTemperature(self, i, key=None, comp=None):
        """Get total temperature input for surface *CT* key
        
        :Call:
            >>> T0 = x.GetSurfCT_TotalTemperature(i, key=None, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *T0*: :class:`float`
                Total temperature of thrust conditions
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Call the SurfBC equivalent
        return self.GetSurfBC_TotalTemperature(i, key,
            comp=comp, typ="SurfCT")
        
    # Get pressure calibration factor
    def GetSurfCT_TemperatureCalibration(self, i, key=None, comp=None):
        """Get temperature calibration factor for *CT* key
        
        :Call:
            >>> fT = x.GetSurfCT_Temperature(i, key=None, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *fT*: {``1.0``} | :class:`float`
                Temperature calibration factor
        :Versions:
            * 2016-08-30 ``@ddalle``: First version
        """
        # Call the SurfBC equivalent
        return self.GetSurfBC_TemperatureCalibration(i, key,
            comp=comp, typ="SurfCT")
    
    # Get pressure additive calibration
    def GetSurfCT_TemperatureOffset(self, i, key=None, comp=None):
        """Get offset used for calibration of static or stagnation temperature
        
        The value used by :mod:`cape` is given by
        
        .. math::
            
            \\tilde{T} = \\frac{b + aT}{T_\\mathit{ref}}
            
        where :math:`\\tilde{T}` is the value used in the namelist, *b* is the
        value from this function, *a* is the result of
        :func:`GetSurfBC_TemperatureCalibration`, *T* is the input value from
        the JSON file, and :math:`T_\\mathit{ref}` is the value from
        :func:`GetSurfBC_RefTemperature`.  In code, this is
        
        .. code-block:: python
        
            T_tilde = (bt + ft*T) / Tref
        
        :Call:
            >>> bt = x.GetSurfCT_TemperatureOffset(i, key=None, comp=None)
        :Inputs:
            *x*: :Class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *bt*: {``0.0``} | :class:`float`
                Stagnation or static temperature offset
        :Versions:
            * 2016-08-29 ``@ddalle``: First version
        """
        # Call the SurfBC equivalent
        return self.GetSurfBC_TemperatureOffset(i, key,
            comp=comp, typ="SurfCT")
            
    # Get total temperature
    def GetSurfCT_RefTemperature(self, i, key=None, comp=None):
        """Get reference temperature input for surface *CT* total temperature
        
        :Call:
            >>> Tref = x.GetSurfCT_RefTemperature(i, key=None, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *Tref*: :class:`float`
                Reference temperature for normalizing *T0*
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Call the SurfBC equivalent
        return self.GetSurfBC_RefTemperature(i, key, comp=comp, typ="SurfCT")
    
    # Get Mach number
    def GetSurfCT_Mach(self, i, key=None, comp=None):
        """Get Mach number input for surface *CT* key
        
        :Call:
            >>> M = x.GetSurfCT_TotalTemperature(i, key=None, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *T0*: :class:`float`
                Total temperature of thrust conditions
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Call the SurfBC equivalent
        return self.GetSurfBC_Mach(i, key, comp=comp, typ="SurfCT")
    
    # Get exit Mach number input for SurfCT input
    def GetSurfCT_ExitMach(self, i, key=None, comp=None):
        """Get Mach number input for surface *CT* key
        
        :Call:
            >>> M2 = x.GetSurfCT_ExitMach(i, key=None, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *M2*: :class:`float`
                Nozzle exit Mach number
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Get the parameter and value
        v, t = self.GetSurfBC_ParamType(key, 'ExitMach', comp=comp)
        # Process the option
        if v is None:
            # Flag to use the vehicle value from *cntl.opts*
            return None
        elif t in ['str', 'unicode']:
            # Use this value as a key
            return getattr(self,v)[i]
        else:
            # Use the fixed value
            return v
    
    # Get area ratio
    def GetSurfCT_AreaRatio(self, i, key=None, comp=None):
        """Get area ratio for surface *CT* key
        
        :Call:
            >>> AR = x.GetSurfCT_AreaRatio(i, key=None, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *AR*: :class:`float`
                Area ratio
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Get the parameter and value
        v, t = self.GetSurfBC_ParamType(key, 'AreaRatio', comp=comp)
        # Process the option
        if v is None:
            # Flag to use the vehicle value from *cntl.opts*
            return None
        elif t in ['str', 'unicode']:
            # Use this value as a key
            return getattr(self,v)[i]
        else:
            # Use the fixed value
            return v
    
    # Get area ratio
    def GetSurfCT_ExitArea(self, i, key=None, comp=None):
        """Get exit area for surface *CT* key
        
        :Call:
            >>> A2 = x.GetSurfCT_ExitArea(i, key=None, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *A2*: :class:`float`
                Exit area
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Get the parameter and value
        v, t = self.GetSurfBC_ParamType(key, 'ExitArea', comp=comp)
        # Process the option
        if v is None:
            # Flag to use the vehicle value from *cntl.opts*
            return None
        elif t in ['str', 'unicode']:
            # Use this value as a key
            return getattr(self,v)[i]
        else:
            # Use the fixed value
            return v
            
    # Get reference area
    def GetSurfCT_RefArea(self, i, key=None, comp=None):
        """Get reference area for surface *CT* key, this divides *CT* value
        
        If this is ``None``, it defaults to the vehicle reference area
        
        :Call:
            >>> Aref = x.GetSurfCT_RefArea(i, key=None, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *ARef*: {``None``} | :class:`float`
                Reference area; if ``None``, use the vehicle area 
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Get the parameter and value
        v, t = self.GetSurfBC_ParamType(key, 'RefArea', comp=comp)
        # Process the option
        if v is None:
            # Flag to use the vehicle value from *cntl.opts*
            return None
        elif t in ['str', 'unicode']:
            # Use this value as a key
            return getattr(self,v)[i]
        else:
            # Use the fixed value
            return v
    
    # Get component ID(s) for input SurfCT key
    def GetSurfCT_CompID(self, i, key=None, comp=None):
        """Get component ID input for surface *CT* key
        
        :Call:
            >>> compID = x.GetSurfCT_CompID(i, key=None, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *compID*: :class:`list` (:class:`int` | :class:`str`)
                Surface boundary condition Mach number
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Call the SurfBC equivalent
        return self.GetSurfBC_CompID(i, key, comp=comp, typ="SurfCT")
    
    # Get mass species
    def GetSurfCT_Species(self, i, key=None, comp=None):
        """Get species input for surface *CT* key
        
        :Call:
            >>> Y = x.GetSurfCT_Species(i, key=None, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *Y*: :class:`list` (:class:`float`)
                Vector of mass fractions
        :Versions:
            * 2016-08-30 ``@ddalle``: First version
        """
        # Call the SurfBC equivalent
        return self.GetSurfBC_Species(i, key, comp=comp, typ="SurfCT")
    
    # Get grid name(s)/number(s) for input SurfCT key
    def GetSurfCT_Grids(self, i, key=None, comp=None):
        """Get list of grids for surface *CT* key
        
        :Call:
            >>> compID = x.GetSurfCT_Grids(i, key=None, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *compID*: :class:`list` (:class:`int` | :class:`str`)
                Surface boundary condition Mach number
        :Versions:
            * 2016-08-29 ``@ddalle``: First version
        """
        # Call the SurfBC equivalent
        return self.GetSurfBC_Grids(i, key, comp=comp, typ="SurfCT")
        
    # Get ratio of specific heats for SurfCT key
    def GetSurfCT_Gamma(self, i, key=None, comp=None):
        """Get ratio of specific heats input for surface *CT* key
        
        :Call:
            >>> gam = x.GetSurfCT_Gamma(i, key=None, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfCT`` key
            *comp*: {``None``} | :class:`str`
                Name of component
        :Outputs:
            *gam*: {``1.4``} | :class:`float`
                Ratio of specific heats
        :Versions:
            * 2016-04-11 ``@ddalle``: First version
        """
        # Call the SurfBC equivalent
        return self.GetSurfBC_Gamma(i, key, comp=comp, typ="SurfCT")
        
    # Get stagnation pressure input for SurfBC input
    def GetSurfBC_TotalPressure(self, i, key=None, comp=None, **kw):
        """Get stagnation pressure input for surface BC key
        
        :Call:
            >>> p0 = x.GetSurfBC_TotalPressure(i, key=None, comp=None, **kw)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
            *typ*: {``"SurfBC"``} | ``"SurfCT"`` | 
                Type to use for checking properties of *key*
        :Outputs:
            *p0*: :class:`float`
                Stagnation pressure parameter, usually *p0/pinf*
        :Versions:
            * 2016-03-28 ``@ddalle``: First version
        """
        # Type
        typ = kw.get('typ', 'SurfBC')
        # Process key
        key = self.GetKeyName(typ, key)
        # Get the parameter and value
        v, t = self.GetSurfBC_ParamType(key, 'TotalPressure', comp=comp)
        # Default process
        return self.GetSurfBC_Val(i, key, v, t)
            
    # Get reference pressure
    def GetSurfBC_RefPressure(self, i, key=None, comp=None, **kw):
        """Get reference pressure for surface BC key
        
        :Call:
            >>> pinf = x.GetSurfBC_RefPressure(i, key=None, comp=None, **kw)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
            *typ*: {``"SurfBC"``} | :class:`str`
                Trajectory key type to process
        :Outputs:
            *pinf*: :class:`float`
                Reference pressure to use, this divides the *p0* value
        :Versions:
            * 2016-03-28 ``@ddalle``: First version
        """
        # Type
        typ = kw.get('typ', 'SurfBC')
        # Process key
        key = self.GetKeyName(typ, key)
        # Get the parameter and value
        v, t = self.GetSurfBC_ParamType(key, 'RefPressure', comp=comp)
        # Process the option
        if v is None:
            # Use dimensional temperature
            return 1.0
        elif t in ['str', 'unicode']:
            # Check for special keys
            if v.lower() in ['freestream', 'inf']:
                # Use frestream value
                return self.GetPressure(i)
            elif v.lower() in ['total', 'stagnation']:
                # Use freestream stagnation value
                return self.GetTotalPressure(i)
            else:
                # Use this as a key
                return getattr(self,v)[i]
        else:
            # Use the fixed value
            return v
            
    # Get pressure scaling
    def GetSurfBC_PressureCalibration(self, i, key=None, comp=None, **kw):
        """Get total pressure scaling factor used for calibration
        
        :Call:
            >>> fp = x.GetSurfBC_PressureCalibration(i,key=None,comp=None,**kw)
        :Inputs:
            *x*: :Class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
            *typ*: {``"SurfBC"``} | :class:`str`
                Trajectory key type to process
        :Outputs:
            *fp*: {``1.0``} | :class:`float`
                Pressure calibration factor
        :Versions:
            * 2016-04-12 ``@ddalle``: First version
        """
        # Type
        typ = kw.get('typ', 'SurfBC')
        # Process key
        key = self.GetKeyName(typ, key)
        # Get the parameter and value
        v, t = self.GetSurfBC_ParamType(key, 'PressureCalibration', comp=comp)
        # Default process
        return self.GetSurfBC_Val(i, key, v, t, vdef=1.0)
            
    # Get pressure additive calibration
    def GetSurfBC_PressureOffset(self, i, key=None, comp=None, **kw):
        """Get offset used for calibration of static or stagnation pressure
        
        The value used by :mod:`cape` is given by
        
        .. math::
            
            \\tilde{p} = \\frac{b + ap}{p_\\mathit{ref}}
            
        where :math:`\\tilde{p}` is the value used in the namelist, *b* is the
        value from this function, *a* is the result of
        :func:`GetSurfBC_PressureCalibration`, *p* is the input value from the
        JSON file, and :math:`p_\\mathit{ref}` is the value from
        :func:`GetSurfBC_RefPressure`.  In code, this is
        
        .. code-block:: python
        
            p_tilde = (bp + fp*p) / pref
        
        :Call:
            >>> bp = x.GetSurfBC_PressureOffset(i, key=None, comp=None, **kw)
        :Inputs:
            *x*: :Class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
            *typ*: {``"SurfBC"``} | :class:`str`
                Trajectory key type to process
        :Outputs:
            *bp*: {``0.0``} | :class:`float`
                Stagnation or static pressure offset
        :Versions:
            * 2016-08-29 ``@ddalle``: First version
        """
        # Type
        typ = kw.get('typ', 'SurfBC')
        # Process key
        key = self.GetKeyName(typ, key)
        # Get the parameter and value
        v, t = self.GetSurfBC_ParamType(key, 'PressureOffset', comp=comp)
        # Process the option
        if v is None:
            # Use no offset
            return 0.0
        elif t in ['str', 'unicode']:
            # Check for special keys
            if v.lower() in ['freestream', 'inf']:
                # Use frestream value
                return self.GetPressure(i)
            elif v.lower() in ['total', 'stagnation']:
                # Use freestream stagnation value
                return self.GetTotalPressure(i)
            else:
                # Use this as a key
                return getattr(self,v)[i]
        else:
            # Use the fixed value
            return v
        
    # Get stagnation temperature input for SurfBC input
    def GetSurfBC_TotalTemperature(self, i, key=None, comp=None, **kw):
        """Get stagnation pressure input for surface BC key
        
        :Call:
            >>> T0 = x.GetSurfBC_TotalTemperature(i, key=None, comp=None, **kw)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
            *typ*: {``"SurfBC"``} | :class:`str`
                Trajectory key type to process
        :Outputs:
            *T0*: :class:`float`
                Stagnation temperature parameter, usually *T0/Tinf*
        :Versions:
            * 2016-03-28 ``@ddalle``: First version
        """
        # Type
        typ = kw.get('typ', 'SurfBC')
        # Process key
        key = self.GetKeyName(typ, key)
        # Special key names
        kw_funcs = {
            "freestream": "GetTemperature",
            "inf":        "GetTemperature",
            "total":      "GetTotalTemperature",
            "stagnation": "GetTotalTemperature",
        }
        # Get the value
        return self.GetSurfBC_Param(i, key, 'TotalTemperature', 
            comp=comp, typ=typ, vdef=0.0, **kw_funcs)
    
    # Get reference temperature
    def GetSurfBC_RefTemperature(self, i, key=None, comp=None, **kw):
        """Get reference temperature for surface BC key
        
        :Call:
            >>> Tinf = x.GetSurfBC_RefTemperature(i, key=None, comp=None, **kw)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
            *typ*: {``"SurfBC"``} | :class:`str`
                Trajectory key type to process
        :Outputs:
            *Tinf*: :class:`float`
                Reference temperature to use, this divides the *T0* value
        :Versions:
            * 2016-03-28 ``@ddalle``: First version
        """
        # Type
        typ = kw.get('typ', 'SurfBC')
        # Process key
        key = self.GetKeyName(typ, key)
        # Special key names
        kw_funcs = {
            "freestream": "GetTemperature",
            "inf":        "GetTemperature",
            "total":      "GetTotalTemperature",
            "stagnation": "GetTotalTemperature",
        }
        # Get the value
        return self.GetSurfBC_Param(i, key, 'RefTemperature', 
            comp=comp, typ=typ, vdef=0.0, **kw_funcs)
            
    # Get pressure scaling
    def GetSurfBC_TemperatureCalibration(self, i, key=None, comp=None, **kw):
        """Get total/static temperature scaling factor used for calibration
        
        :Call:
           >>> fT=x.GetSurfBC_TemperatureCalibration(i,key=None,comp=None,**kw)
        :Inputs:
            *x*: :Class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
            *typ*: {``"SurfBC"``} | :class:`str`
                Trajectory key type to process
        :Outputs:
            *fp*: {``1.0``} | :class:`float`
                Pressure calibration factor
        :Versions:
            * 2016-08-30 ``@ddalle``: First version
        """
        # Type
        typ = kw.get('typ', 'SurfBC')
        # Process key
        key = self.GetKeyName(typ, key)
        # Get the parameter and value
        v, t = self.GetSurfBC_ParamType(key,'TemperatureCalibration',comp=comp)
        # Default process
        return self.GetSurfBC_Val(i, key, v, t, vdef=1.0)
        
    # Get pressure additive calibration
    def GetSurfBC_TemperatureOffset(self, i, key=None, comp=None, **kw):
        """Get offset used for calibration of static or stagnation temperature
        
        The value used by :mod:`cape` is given by
        
        .. math::
            
            \\tilde{T} = \\frac{b + aT}{T_\\mathit{ref}}
            
        where :math:`\\tilde{T}` is the value used in the namelist, *b* is the
        value from this function, *a* is the result of
        :func:`GetSurfBC_TemperatureCalibration`, *T* is the input value from
        the JSON file, and :math:`T_\\mathit{ref}` is the value from
        :func:`GetSurfBC_RefTemperature`.  In code, this is
        
        .. code-block:: python
        
            T_tilde = (bt + ft*T) / Tref
        
        :Call:
            >>> bt = x.GetSurfBC_TemperatureOffset(i,key=None, comp=None, **kw)
        :Inputs:
            *x*: :Class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
            *typ*: {``"SurfBC"``} | :class:`str`
                Trajectory key type to process
        :Outputs:
            *bt*: {``0.0``} | :class:`float`
                Stagnation or static temperature offset
        :Versions:
            * 2016-08-29 ``@ddalle``: First version
        """
        # Type
        typ = kw.get('typ', 'SurfBC')
        # Process key
        key = self.GetKeyName(typ, key)
        # Get the parameter and value
        v, t = self.GetSurfBC_ParamType(key, 'TemperatureOffset', comp=comp)
        # Special key names
        kw_funcs = {
            "freestream": "GetTemperature",
            "inf":        "GetTemperature",
            "total":      "GetTotalTemperature",
            "stagnation": "GetTotalTemperature",
        }
        # Get the value
        return self.GetSurfBC_Param(i, key, 'TemperatureOffset', 
            typ=typ, vdef=0.0, **kw_funcs)
            
    # Get Mach number input for SurfBC input
    def GetSurfBC_Mach(self, i, key=None, comp=None, **kw):
        """Get Mach number input for surface BC key
        
        :Call:
            >>> M = x.GetSurfBC_Mach(i, key=None, comp=None, **kw)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
            *typ*: {``"SurfBC"``} | :class:`str`
                Trajectory key type to process
        :Outputs:
            *M*: :class:`float`
                Surface boundary condition Mach number
        :Versions:
            * 2016-03-28 ``@ddalle``: First version
        """
        # Type
        typ = kw.get('typ', 'SurfBC')
        # Process key
        key = self.GetKeyName(typ, key)
        # Get the parameter and value
        v, t = self.GetSurfBC_ParamType(key, 'Mach', comp=comp)
        # Default process
        return self.GetSurfBC_Val(i, key, v, t, vdef=1.0)
            
    # Get ratio of specific heats input for SurfBC key
    def GetSurfBC_Gamma(self, i, key=None, comp=None, **kw):
        """Get ratio of specific heats for surface BC key
        
        :Call:
            >>> gam = x.GetSurfBC_Gamma(i, key=None, comp=None, **kw)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
            *typ*: {``"SurfBC"``} | :class:`str`
                Trajectory key type to process
        :Outputs:
            *gam*: :class:`float`
                Surface boundary condition ratio of specific heats
        :Versions:
            * 2016-03-29 ``@ddalle``: First version
            * 2016-08-29 ``@ddalle``: Added *comp*
        """
        # Type
        typ = kw.get('typ', 'SurfBC')
        # Process key
        key = self.GetKeyName(typ, key)
        # Get the parameter and value
        v, t = self.GetSurfBC_ParamType(key, 'Gamma', comp=comp)
        # Default process
        return self.GetSurfBC_Val(i, key, v, t, vdef=1.4)
        
    # Get species
    def GetSurfBC_Species(self, i, key=None, comp=None, **kw):
        """Get species information for a surface BC key
        
        The species can be specified using several different manners.
        
        The first and most common approach is to simply set a fixed list, for
        example setting ``"Species"`` to ``[0.0, 1.0, 0.0]`` to specify use of
        the second species, or ``[0.2, 0.8, 0.1]`` to specify a mix of three
        difference species.
        
        A second method is to specify an integer.  For example, if ``"Species"``
        is set to ``2`` and ``"nSpecies"`` is set to ``4``, the output will be
        ``[0.0, 1.0, 0.0, 0.0]``.
        
        The third method is a generalization of the first.  An alternative to
        simply setting a fixed list of numeric species mass fractions, the
        entries in the list can depend on the values of other trajectory keys.
        For example, setting ``"Species"`` to ``['YH2', 'YO2']`` will translate
        the mass fractions according to the values of trajectory keys ``"YH2"``
        and ``"YO2"``.
        
        :Call:
            >>> Y = x.GetSurfBC_Species(i, key=None, comp=None, **kw)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
            *typ*: {``"SurfBC"``} | :class:`str`
                Trajectory key type to process
        :Outputs:
            *Y*: :class:`list` (:class:`float`)
                List of species mass fractions for boundary condition
        :Versions:
            * 2016-08-29 ``@ddalle``: First version
        """
        # Type
        typ = kw.get('typ', 'SurfBC')
        # Process key
        key = self.GetKeyName(typ, key)
        # Get the parameter and value
        v, t = self.GetSurfBC_ParamType(key, 'Species', comp=comp)
        # Default process
        Y = self.GetSurfBC_Val(i, key, v, t, vdef=[1.0])
        # Type
        tY = type(Y).__name__
        # Check for integer
        if tY == 'int':
            # Get number of species
            nY = self.GetSurfBC_nSpecies(i, key, comp=comp, typ=typ)
            # Make list with one nonzero component
            Y = [float(i+1==Y) for i in range(nY)]
        elif tY != 'list':
            # Must be list or int
            raise TypeError("Species specification must be int or list")
        # Loop through components
        for i in range(len(Y)):
            # Get the value and type
            y = Y[i]
            ty = type(y).__name__
            # Convert if string
            if ty in ['str', 'unicode']:
                # Get the value of a different key
                Y[i] = getattr(self,y)[i]
        # Output
        return Y
            
    # Get number of species
    def GetSurfBC_nSpecies(self, i, key=None, comp=None, **kw):
        """Get number of species for a surface BC key
        
        :Call:
            >>> nY = x.GetSurfBC_nSpecies(i, key=None, comp=None, **kw)
        :Inptus:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
            *typ*: {``"SurfBC"``} | :class:`str`
                Trajectory key type to process
        :Outputs:
            *nY*: {``1``} | :class:`int`
                Number of species
        :Versions:
            * 2016-08-29 ``@ddalle``: First version
        """
        # Type
        typ = kw.get('typ', 'SurfBC')
        # Process key
        key = self.GetKeyName(typ, key)
        # Get the parameter and value
        v, t = self.GetSurfBC_ParamType(key, 'nSpecies', comp=comp)
        # Default process
        return self.GetSurfBC_Val(i, key, v, t)
        
            
    # Get component ID(s) input for SurfBC key
    def GetSurfBC_CompID(self, i, key=None, comp=None, **kw):
        """Get component ID input for surface BC key
        
        :Call:
            >>> compID = x.GetSurfBC_CompID(i, key=None, comp=None, **kw)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
            *typ*: {``"SurfBC"``} | :class:`str`
                Trajectory key type to process
        :Outputs:
            *compID*: :class:`list` | :class:`str` | :class:`dict`
                Surface boundary condition component ID(s)
        :Versions:
            * 2016-03-28 ``@ddalle``: First version
        """
        # Type
        typ = kw.get('typ', 'SurfBC')
        # Process key
        key = self.GetKeyName(typ, key)
        # Get the parameter and value
        v, t = self.GetSurfBC_ParamType(key, 'CompID', comp=comp)
        # Do not process this one
        return v
            
    # Get column index input for SurfBC key
    def GetSurfBC_BCIndex(self, i, key=None, comp=None, **kw):
        """Get namelist/column/etc. index for a surface BC key
        
        :Call:
            >>> inds = x.GetSurfBC_BCIndex(i, key=None, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
            *typ*: {``"SurfBC"``} | :class:`str`
                Trajectory key type to process
        :Outputs:
            *inds*: :class:`list` | :class:`str` | :class:`dict`
                Column index for each grid or component
        :Versions:
            * 2016-08-29 ``@ddalle``: First version
        """
        # Type
        typ = kw.get('typ', 'SurfBC')
        # Process key
        key = self.GetKeyName(typ, key)
        # Get the parameter and value
        v, t = self.GetSurfBC_ParamType(key, 'BCIndex', comp=comp)
        # Default process
        return self.GetSurfBC_Val(i, key, v, t)
            
    # Get component ID(s) input for SurfBC key
    def GetSurfBC_Grids(self, i, key=None, comp=None, **kw):
        """Get list of grids for surface BC key
        
        :Call:
            >>> grids = x.GetSurfBC_Grids(i, key=None, comp=None)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Run matrix interface
            *i*: :class:`int`
                Case index
            *key*: ``None`` | :class:`str`
                Name of key to use; defaults to first ``SurfBC`` key
            *comp*: {``None``} | :class:`str`
                Name of component
            *typ*: {``"SurfBC"``} | :class:`str`
                Trajectory key type to process
        :Outputs:
            *grids*: :class:`list` (:class:`int` | :class:`str`)
                Surface boundary condition grids
        :Versions:
            * 2016-08-29 ``@ddalle``: First version
        """
        # Type
        typ = kw.get('typ', 'SurfBC')
        # Process key
        key = self.GetKeyName(typ, key)
        # Get the parameter and value
        v, t = self.GetSurfBC_ParamType(key, 'Grids', comp=comp)
        # Process
        return self.GetSurfBC_Val(i, key, v, t)
    
  # >
# class Trajectory
