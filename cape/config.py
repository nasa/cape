"""
Surface configuration module: :mod:`cape.config`
================================================

This is a module to interact with :file:`Config.xml` files.  In general, it can
be used to create groups of surfaces using an XML file format.  This comes from
the Cart3D/OVERFLOW convention, but it can be used with other modules as well.

It is typical for a surface definition, whether a triangulation, system of
overset structured grids, or mixed quads and triangles, to have each surface
polygon to have a numbered component ID.  This allows a user to group
triangles and quads or other polygons together in some relevant way.  For
example, the user may tag each polygon on the left wing with the component ID of
``12``, and the entire surface is broken out in a similar fashion.

The :mod:`cape.config` module allows the user to do two main things: give
meaningful names to these component IDs and group component IDs together.  For
example, it is usually much more convenient to refer to ``"left_wing"`` than
remember to put ``"12"`` in all the data books, reports, etc.  In addition, a
user usually wants to know the integrated force on the entire airplane (or
whatever other type of configuration is under investigation), so it is useful to
make another component called ``"vehicle"`` that contains ``"left_wing"``,
``"right_wing"``, and ``"fuselage"``.  The user specifies this in the XML file
using the following syntax.

    .. code-block:: xml
    
        <?xml version="1.0" encoding="ISO-8859-1"?>
        <Configuration Name="airplane" Source="Components.i.tri">
        
        <Component Name="vehicle" Type="container">
        </Component>
        
        <Component Name="fuselage" Type="tri">
        <Data> Face Label=1 </Data> </Component>
        
        <Component Name="right_wing" Type="tri">
        <Data> Face Label=11 </Data> </Component>
        
        <Component Name="left_wing" Type="tri">
        <Data> Face Label=12 </Data> </Component>
        
        </Configuration>
        
The *Source* attribute of the first tag is not that important; it's placed there
based on a Cart3D template.  The choice of encoding is probably also
unimportant, but having something there may prevent problems.

The major limitation of this capability at present is that a component may not
have multiple parents.  A parent may have parent, allowing the user to subdivide
groups into smaller groups, but the user may not, for example, split the vehicle
into left half and right half and also create components for forward half and
aft half.
"""

# File checker.
import os
# Import xml parser
import xml.etree.ElementTree as ET
# Process unique lists.
import numpy as np

# Utility functions and classes from CAPE
from .util    import RangeString
from .options import util

# Configuration class
class Config:
    """Configuration class for interfacing :file:`Config.xml` files
    
    :Call:
        >>> cfg = cape.Config(fname='Config.xml')
    :Inputs:
        *fname*: :class:`str`
            Name of configuration file to read
    :Outputs:
        *cfg*: :class:`cape.config.Config`
            Instance of configuration class
    :Versions:
        * 2014-10-12 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self, fname="Config.xml"):
        """Initialization method"""
        # Check for the file.
        if not os.path.isfile(fname):
            # Save an empty component dictionary.
            self.faces = {}
            return
        # Save file name
        self.fname = fname
        # Read the XML file.
        e = ET.parse(fname)
        # Save it
        self.XML = e
        # Get the list of components.
        self.Comps = e.findall('Component')
        # Get the names.
        self.Names = [c.get('Name') for c in self.Comps]
        # Initialize face data
        self.faces = {}
        # Initialize the transformation data
        self.transform = {}
        # Check for unnamed component.
        if None in self.Names:
            raise ValueError("At least one component in "
                + "'%s' is lacking a name." % self.fname)
        # Loop through points to get the labeled faces.
        for c in self.Comps:
            # Check the type.
            if c.get('Type') == 'tri':
                # Triangulation, face label
                self.ProcessTri(c)
            elif c.get('Type') == 'struc':
                # Structured grid list
                self.ProcessStruc(c)
    
    # Function to display things
    def __repr__(self):
        """
        Return the string representation of a :file:`Config.xml` file.
        
        This looks like ``<cape.Config(nComp=N, faces=['Core', ...])>``
        """
        # Return a string.
        return '<cape.Config(nComp=%i, faces=%s)>' % (
            len(self.faces), self.faces.keys())
        
    # Process a tri component
    def ProcessTri(self, c):
        """Process a GMP component of type ``'tri'``
        
        :Call:
            >>> cfg.ProcessTri(c)
        :Inputs:
            *cfg*: :class:`cape.config.Config`
                Instance of configuration class
            *c*: :class:`xml.Element`
                XML interface to element with tag ``'Component'``
        :Versions:
            * 2016-08-23 ``@ddalle``: First version
        """
        # Get the children
        D = c.getchildren()
        # Get the component name
        comp = c.get('Name')
        # Loop through children
        for d in D:
            # Check the element's type
            if d.tag == 'Data':
                # Process the face label
                compID = self.ProcessTriData(comp, d)
            elif d.tag == 'Transform':
                # Process the transformation
                self.ProcessTransform(comp, d)
            else:
                # Unrecognized tag
                raise ValueError("Unrecognized tag '%s' for component '%s'" %
                    (x.tag, comp))
            # Process any parents.
            self.AppendParent(c, compID)
        
    # Process a structured grid component
    def ProcessStruc(self, c):
        """Process a GMP component of type ``'struc'``
        
        :Call:
            >>> cfg.ProcessStruc(c)
        :Inputs:
            *cfg*: :class:`cape.config.Config`
                Instance of configuration class
            *c*: :class:`xml.Element`
                XML interface to element with tag ``'Component'``
        :Versions:
            * 2016-08-23 ``@ddalle``: First version
        """
        # Get the children
        D = c.getchildren()
        # Get the component name
        comp = c.get('Name')
        # Loop through children
        for d in D:
            # Check the element's type
            if d.tag == 'Data':
                # Process the face label
                compID = self.ProcessStrucData(comp, d)
            elif d.tag == 'Transform':
                # Process the transformation
                self.ProcessTransform(comp, d)
            else:
                # Unrecognized tag
                raise ValueError("Unrecognized tag '%s' for component '%s'" %
                    (x.tag, comp))
            # Process any parents.
            self.AppendParent(c, compID)
            
    # Process face label data
    def ProcessTriData(self, comp, d):
        """Process a GMP data element with text for "Face Label"
        
        :Call:
            >>> compID = cfg.ProcessTriData(comp, d)
        :Inputs:
            *cfg*: :class:`cape.config.Config`
                Instance of configuration class
            *d*: :class:`xml.Element`
                XML interface to element with tag ``'Data'``
        :Outputs:
            *compID*: :class:`int`
                Component ID number from "Face Label"
        :Attributes:
            *cfg.faces[comp]*: :class:`int`
                Gets set to *compID*
        :Versions:
            * 2016-08-23 ``@ddalle``: First version
        """
        # Get text
        txt = d.text
        # Check for null text
        if txt is None:
            raise ValueError(
                "Component '%s' has no text in its 'Data' element"
                % comp)
        # Split text on '='
        V = txt.split('=')
        # Check for validity
        if V[0].strip() != "Face Label" or len(V) != 2:
            raise ValueError(
                "Component '%s' has no 'Face Label' in its data element"
                % comp)
        # Set the component
        try:
            # Try to use an integer
            compID = int(V[1])
        except Exception:
            # Just set the text; may be useful
            compID = V[1]
        # Set the component.
        self.faces[comp] = compID
        # Output
        return compID
        
    # Process grid list data
    def ProcessStrucData(self, comp, d):
        """Process a GMP data element with text for "Grid List"
        
        :Call:
            >>> compID = cfg.ProcessStrucData(comp, d)
        :Inputs:
            *cfg*: :class:`cape.config.Config`
                Instance of configuration class
            *d*: :class:`xml.Element`
                XML interface to element with tag ``'Data'``
        :Outputs:
            *compID*: :class:`list` (:class:`int`)
                Grid numbers "Grid List"
        :Attributes:
            *cfg.faces[comp]*: :class:`list` (:class:`int`)
                Gets set to *compID*
        :Versions:
            * 2016-08-23 ``@ddalle``: First version
        """
        # Get text
        txt = d.text
        # Check for null text
        if txt is None:
            raise ValueError(
                "Component '%s' has no text in its 'Data' element"
                % comp)
        # Split text on '='
        V = txt.split('=')
        # Check for validity
        if V[0].strip() != "Grid List" or len(V) != 2:
            raise ValueError(
                "Component '%s' has no 'Grid List' in its data element"
                % comp)
        # Set the component
        try:
            # Initialize list
            compID = []
            # Loop through entries
            for rng in V[1].split(','):
                # Split the range, something like '187-195' or '187'
                v = rng.split('-')
                # Check for single value or range
                if len(v) == 1:
                    # Single value
                    compID.append(int(v[0]))
                else:
                    # Start, end range
                    compID += range(int(v[0]), int(v[1])+1)
            # Convert to array
            compID = list(np.unique(compID))
        except Exception:
            # Just set the text; may be useful
            compID = V[1]
        # Set the component.
        self.faces[comp] = compID
        # Output
        return compID
        
            
    # Process transformation
    def ProcessTransform(self, comp, t):
        """Process a GMP transformation
        
        :Call:
            >>> cfg.ProcessTransform(t)
        :Inputs:
            *cfg*: :class:`cape.config.Config`
                Instance of configuration class
            *t*: :class:`xml.Element`
                XML interface to element with tag ``'Transform'``
        :Versions:
            * 2016-08-23 ``@ddalle``: First version
        """
        # Check the tag
        if t.tag != "Transform":
            raise ValueError("Element '%s' does not have 'Transform' tag" % t)
        # Process transformation
        X = t.getchildren()
        # Initialize transformations
        self.transform[comp] = []
        # Loop through children
        for x in X:
            # Check rotation/translation
            if x.tag == "Rotate":
                # Get center
                try:
                    # Process the current value as a vector
                    cent = [float(v) for v in x.get("Center").split(',')]
                except Exception:
                    # Unset center
                    cent = x.get("Center")
                # Process axis
                try:
                    # Process current value as a vector
                    ax = [float(v) for v in x.get("Axis").split(',')]
                except Exception:
                    # Unset axis
                    ax = x.get("Axis")
                # Process angle
                try:
                    # Process current value as a float
                    ang = float(x.get("Angle"))
                except Exception:
                    # Unset angle
                    ang = x.get("Angle")
                # Process frame
                frame = x.get("Frame")
                # Add rotation to the list
                self.transform[comp].append({
                    "Type": "Rotate",
                    "Center": cent,
                    "Axis": ax,
                    "Angle": ang,
                    "Frame": frame
                })
            elif x.tag == "Translate":
                # Get displacement
                try:
                    # Process the current value as a vector
                    dx = [float(v) for v in x.get("Displacement").split(",")]
                except Exception:
                    # Unset displacement
                    dx = x.get("Displacement")
                # Add translation to the list
                self.transform[comp].append({
                    "Type": "Translate",
                    "Displacement": dx
                })
            else:
                # Unrecognized type
                raise ValueError("Unrecognized transform tag '%s' for '%s'" %
                    (x.tag, comp))
        

    # Function to recursively append components to parents and their parents
    def AppendParent(self, c, compID):
        """Append a component ID to a parent container and its parents
        
        :Call:
            >>> cfg.AppendParent(c, compID)
        :Inputs:
            *Comps*: :class:`list` (:class:`xml.etree.Element`)
                List of XML tags with type 'Component'
            *comp*: :class:`dict`
                Dictionary of component ID numbers in each labeled component
            *k*: :class:`int`
                Index of XML tag to process
            *compID*: :class:`int`
                Component ID number to add to parents' lists
        :Outputs:
            *comp*: :class:`dict`
                Dictionary with *compID* appended in appropriate places
        :Versions:
            * 2014-10-13 ``@ddalle``: First version
        """
        # Check for a parent.
        parent = c.get("Parent")
        # Check for no parent
        if parent is None: return
        # Check for recursion.
        if c.get('Name') == parent:
            # Recursive parent situation
            raise ValueError('Component "%s" is its own parent.' % parent)
        elif parent not in self.Names:
            # Parent could not be found
            raise KeyError('Component "%s" has invalid parent "%s"' %
                (c.get("Name"), parent))
        # Initialize the parent if necessary
        self.faces.setdefault(parent, [])
        # Add this face label or component to the container list
        if type(compID).__name__ in ['list', 'ndarray']:
            # List of components
            self.faces[parent] += list(compID)
        else:
            # Single component
            self.faces[parent].append(compID)
        # Eliminate doubles.
        self.faces[parent] = list(np.unique(self.faces[parent]))
        # Get the parent's index
        k0 = self.Names.index(parent)
        # Get the parent's element
        p = self.Comps[k0]
        # Append *compID* to parent's parent, if any
        self.AppendParent(p, compID)
        
    # Set transformation
    def SetRotation(self, comp, i=None, **kw):
        """Modify or add a rotation for component *comp*
        
        :Call:
            >>> cfg.SetRotation(comp, i=None, **kw)
        :Inputs:
            *cfg*: :class:`cape.config.Config`
                Instance of configuration class
            *i*: {``None``} | :class:`int`
                Index of the rotation
            *Center*: {``[0.0, 0.0, 0.0]``} | :class:`list` | :class:`str`
                Point about which to rotate
            *Axis*: {``[0.0, 1.0, 0.0]``} | :class:`list` | :class:`str`
                Axis about which to rotate
            *Angle*: {``0.0``} | :class:`float` | :class:`str`
                Angle for rotation
            *Frame*: {``"Body"``} | ``None``
                Rotation type, body frame or Overflow frame
        :Versions:
            * 2016-08-23 ``@ddalle``: First version
        """
        # Ensure there is a transformation for this component.
        self.transform.setdefault(comp, [])
        # Get components
        T = self.transform[comp]
        # Number of transformations currently defined
        n = len(T)
        # Default index
        if i is None: i = n
        # Check if transformation *i* exists
        if i < n:
            # Transformation exists; check type
            typ = T[i].get("Type", "Rotate")
            # Check type
            if typ != "Rotate":
                raise ValueError("Transform %s for '%s' is not a rotation"
                    % (i, comp))
        elif i > n:
            # Cannot add this rotation
            raise ValueError(
                ("Cannot set transformation %s for '%s'\n" % (i, comp)) +
                ("because only %s transformations are currently defined" % n))
        # Process input values
        R = {
            "Type": "Rotate",
            "Center":  kw.get("Center", [0.0, 0.0, 0.0]),
            "Axis":    kw.get("Axis",   [0.0, 1.0, 0.0]),
            "Angle":   kw.get("Angle",  0.0),
            "Frame":   kw.get("Frame")
        }
        # Apply changes as appropriate
        if i == n:
            # Add the whole rotation
            T.append(R)
        else:
            # Ensure type
            T[i]["Type"] = "Rotate"
            # Only apply either blank settings or directly-specified values
            for k in ["Center", "Axis", "Angle", "Frame"]:
                # Check if parameter in *T[i]*
                T[i].setdefault(k, R[k])
                # Check if we should overwrite current settings
                if k in kw: T[i][k] = R[k]
        
    # Set transformation
    def SetTranslation(self, comp, i=None, **kw):
        """Modify or add a translation for component *comp*
        
        :Call:
            >>> cfg.SetTranslation(comp, i=0, **kw)
        :Inputs:
            *cfg*: :class:`cape.config.Config`
                Instance of configuration class
            *i*: {``0``} | :class:`int`
                Index of the rotation
            *Displacement*: {``[0.0, 0.0, 0.0]``} | :class:`list` | :class:`str`
                Vector to move component
        :Versions:
            * 2016-08-23 ``@ddalle``: First version
        """
        # Ensure there is a transformation for this component.
        self.transform.setdefault(comp, [])
        # Get components
        T = self.transform[comp]
        # Number of transformations currently defined
        n = len(T)
        # Default index
        if i is None: i = n
        # Check if transformation *i* exists
        if i < n:
            # Transformation exists; check type
            typ = T[i].get("Type", "Translate")
            # Check type
            if typ != "Translate":
                raise ValueError("Transform %s for '%s' is not a translation"
                    % (i, comp))
        elif i > n:
            # Cannot add this rotation
            raise ValueError(
                ("Cannot set transformation %s for '%s'\n" % (i, comp)) +
                ("because only %s transformations are currently defined" % n))
        # Process input values
        R = {
            "Type": "Translate",
            "Displacement": kw.get("Displacement", [0.0, 0.0, 0.0])
        }
        # Apply changes as appropriate
        if i == n:
            # Add the whole rotation
            T.append(R)
        else:
            # Ensure type
            T[i]["Type"] = "Translate"
            # Only apply either blank settings or directly-specified values
            for k in ["Displacement"]:
                # Check if parameter in *T[i]*
                T[i].setdefault(k, R[k])
                # Check if we should overwrite current settings
                if k in kw: T[i][k] = R[k]
        
    # Write the file
    def Write(self, fname=None):
        """Write the configuration to file
        
        :Call:
            >>> cfg.Write(fname=None)
        :Inputs:
            *cfg*: :class:`cape.config.Config`
                Instance of configuration class
            *fname*: {``None``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-08-23 ``@ddalle``: First version
        """
        # Default file name
        if fname is None:
            fname = self.fname
        # Open the file for writing
        f = open(fname, 'w')
        # Write the header.
        f.write("<?xml version='1.0' encoding='utf-8'?>\n")
        # Get the Configuration properties
        conf = self.XML.getroot().attrib
        # Write the configuration tag
        f.write("<Configuration")
        # Loop through the keys
        for k in conf:
            # Write the key and the value
            f.write(' %s="%s"' % (k, conf[k]))
        # Close the opening configuration tag
        f.write(">\n\n")
        # Loop through components
        for comp in self.Names:
            # Write the component element
            self.WriteComponent(f, comp)
        # Close the root element
        f.write("</Configuration>\n")
        # Close the file.
        f.close()
        
    # Function to write a component
    def WriteComponent(self, f, comp):
        """Write a "Component" element to file
        
        Data (either "Face Label" or "Grid List") is written, and any
        transformations are also written.
        
        :Call:
            >>> cfg.WriteComponent(f, comp)
        :Inputs:
            *cfg*: :class:`cape.config.Config`
                Instance of configuration class
            *f*: :class:`file`
                File handle open for writing
            *comp*: :class:`str`
                Name of component to write
        :Versions:
            * 2016-08-23 ``@ddalle``: First version
        """
        # Get the component index
        i = self.Names.index(comp)
        # Get the component interface
        c = self.Comps[i]
        # Begin the tag.
        f.write("  <Component")
        # Loop through the attributes
        for k in c.attrib:
            f.write(' %s="%s"' % (k, c.attrib[k]))
        # Close the component opening tag
        f.write(">\n")
        # Check type
        if c.get("Type", "tri") == "tri":
            # Write the data element with "Face Label"
            self.WriteComponentData(f, comp, label="Face Label")
        else:
            # Write the data element with "Grid List"
            self.WriteComponentData(f, comp, label="Grid List")
        # Write transformation
        if comp in self.transform:
            # Write the transformation
            self.WriteComponentTransform(f, comp)
        # Close the component element.
        f.write("  </Component>\n\n")
        
    # Method to write data
    def WriteComponentData(self, f, comp, label=None):
        """Write a "Data" element to file
        
        :Call:
            >>> cfg.WriteComponentData(f, comp, label=None)
        :Inputs:
            *cfg*: :class:`cape.config.Config`
                Instance of configuration class
            *comp*: :class:`str`
                Name of component to write
            *label*: {``None``} | ``"Face Label"`` | ``"Grid List"``
                Label used to specify data
        :Versions:
            * 2016-08-23 ``@ddalle``: First version
        """
        # Write the data tag
        f.write("    <Data>")
        # Get the list of components in the component.
        compID = self.GetCompID(comp)
        # Type (int, list, or str)
        typ = type(compID).__name__
        # Determine default label
        if label is None:
            if typ.startswith('int'):
                # Assuming we have a single face
                label = "Face Label"
            else:
                # Guessing it's a grid list
                label = "Grid List"
        # Write the label
        f.write(" %s=" % label)
        # Check for easy stuff
        if typ in ['int', 'str']:
            # Write a string
            f.write("%s </Data>\n" % compID)
            return
        # Number of components
        n = len(compID)
        # Exit if appropriate
        if n == 0:
            f.write("</Data>\n")
            return
        # Write the list
        f.write(RangeString(compID))
        # Close the element.
        f.write(" </Data>\n")
            
    # Method to write transformation(s)
    def WriteComponentTransform(self, f, comp):
        """Write a "Transform" element to file
        
        :Call:
            >>> cfg.WriteComponentData(f, comp, label=None)
        :Inputs:
            *cfg*: :class:`cape.config.Config`
                Instance of configuration class
            *comp*: :class:`str`
                Name of component to write
            *label*: {``None``} | ``"Face Label"`` | ``"Grid List"``
                Label used to specify data
        :Versions:
            * 2016-08-23 ``@ddalle``: First version
        """
        # Check if component has one or more transformations
        if comp not in self.transform: return
        # Write the transform tag
        f.write("    <Transform>\n")
        # Loop through transformations
        for R in self.transform[comp]:
            # Get the type
            typ = R.get("Type", "Translate")
            # Rotation or translation
            if typ == "Rotate":
                # Open the rotation tag
                f.write("%6s<Rotate" % "")
                # Get the properties
                cent  = R.get("Center", [0.0, 0.0, 0.0])
                ax    = R.get("Axis",   [0.0, 1.0, 0.0])
                ang   = R.get("Angle",  0.0)
                frame = R.get("Frame")
                # Convert center to string 
                if type(cent).__name__ in ['list', 'ndarray']:
                    # Ensure doubles
                    cent = ", ".join(['%.12e'%v for v in cent])
                # Convert axis to string
                if type(ax).__name__ in ['list', 'ndarray']:
                    # Convert to float and then string
                    ax = ", ".join([str(float(v)) for v in ax])
                # Write values
                f.write(' Center="%s"' % cent)
                f.write(' Axis="%s"'   % ax)
                f.write(' Angle="%s"'  % ang)
                # Only write frame if it exists
                if frame is not None:
                    f.write(' Frame="%s"' % frame)
            elif typ == "Translate":
                # Open the rotation tag
                f.write("%6s<Translate" % "")
                # Get the properties
                dx = R.get("Displacement", [0.0, 0.0, 0.0])
                # Convert center to string 
                if type(dx).__name__ in ['list', 'ndarray']:
                    # Ensure doubles
                    dx = ", ".join(['%.12e'%v for v in dx])
                # Write values
                f.write(' Displacement="%s"' % dx)
            # Close the element
            f.write(" />\n")
        # Close the element
        f.write("    </Transform>\n")
        
        
    # Method to get CompIDs from generic input
    def GetCompID(self, face):
        """Return a list of component IDs from generic input
        
        :Call:
            >>> compID = cfg.GetCompID(face)
        :Inputs:
            *cfg*: :class:`cape.config.Config`
                Instance of configuration class
            *face*: :class:`str` | :class:`int` | :class:`list`
                Component number, name, or list of component numbers and names
        :Outputs:
            *compID*: :class:`list` (:class:`int`)
                List of component IDs
        :Versions:
            * 2014-10-12 ``@ddalle``: First version
        """
        # Initialize the list.
        compID = []
        # Process the type.
        if type(face).__name__ in ['list', 'numpy.ndarray']:
            # Loop through the inputs.
            for f in face:
                # Call this function so it passes to the non-array portion.
                compID += self.GetCompID(f)
        elif face in self.faces:
            # Process the face
            cID = self.faces[face]
            # Check if it's a list.
            if type(cID).__name__ == 'list':
                # Add the list.
                compID += cID
            else:
                # Single component.
                compID.append(cID)
        else:
            # Just append it (as an integer).
            try:
                compID.append(int(face))
            except Exception:
                pass
        # Output
        return compID
    
    # Method to copy a configuration
    def Copy(self):
        """Copy a configuration interface
        
        :Call:
            >>> cfg2 = cfg.Copy()
        :Inputs:
            *cfg*: :class:`cape.config.Config`
                Instance of configuration class
        :Outputs:
            *cfg2*: :class:`cape.config.Config`
                Copy of input
        :Versions:
            * 2014-11-24 ``@ddalle``: First version
        """
        # Initialize object.
        cfg = Config()
        # Copy the dictionaries.
        cfg.faces = self.faces.copy()
        cfg.transform = self.transform.copy()
        # Output
        return cfg
# class Config

# Alternate configuration
class ConfigJSON(object):
    """Configuration 
    
    """
    # Initialization method
    def __init__(self, fname="Config.json"):
        # Read the settings from an expanded and decommented JSON file
        opts = util.loadJSONFile(fname)
        # Convert to special options class
        opts = util.odict(**opts)
        # Major sections
        self.props = opts.get("Properties", {})
        self.tree  = opts.get("Tree", {})
        # Save
        self.opts = opts
        # Initialize component list
        self.comps = []
        self.faces = {}
        ## Loop through properties to get CompIDs
        #for c in props:
        #    # Get the property
        #    prop = props[c]
        #    # Get the compID
        #    compID = prop.get("CompID", None)
        #    # Assign it
        #    if compID is not None and compID not in self.comps:
        #        self.faces[c] = compID
        # Loop through the tree
        for c in self.opts["Tree"]:
            # Check if already processed
            if c in self.faces:
                continue
            # Process the first component
            self.AppendChild(c)
            
    
    # Process children
    def AppendChild(self, c):
        # Initialize component
        compID = []
        # Get the children
        C = self.tree.get(c, [])
        # Loop through children
        for child in C:
            # Check if it has been processed
            if child in self.faces:
                # Get the components to add from that child
                compID += self.faces[child]
                continue
            # Otherwise, check if this is also a parent
            if child in self.tree:
                # Nest
                compID += self.ProcessChild(child)
                continue
            # Get the component ID from the "Properties" section
            prop = self.props.get(child, {})
            # Check for component
            if "CompID" not in prop:
                raise ValueError(("Component '%s' is not a parent " % child) +
                    'and has no "CompID"')
            # Get the component
            cID = prop["CompID"]
            # Set the component for *child*
            self.faces[child] = cID
            # Append to the current component's list
            compID.append(cID)
        # Save the results
        self.faces[c] = compID
        # Output
        return compID
                

# class ConfigJSON

