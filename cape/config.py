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
        
    # Eliminate all CompID numbers not actually used
    def RestrictCompID(self, compIDs):
        """Restrict component IDs in *cfg.faces* to those in a specified list
        
        :Call:
            >>> cfg.RestrictCompID(compIDs)
        :Inputs:
            *cfg*: :class:`cape.config.Config`
                XML-based configuration interface
            *compIDs*: :class:`list` (:class:`int`)
                List of relevant component IDs
        :Versions:
            * 2016-11-05 ``@ddalle``: First version
        """
        # Check inputs
        t = type(compIDs).__name__
        if t not in ['list', 'ndarray']:
            raise TypeError(
                ("List of relevant component ID numbers must have type ") + 
                ("'int' or 'ndarray'; received '%s'" % t))
        # Check length
        if len(compIDs) < 1:
            raise ValueError("Invalid request to restrict to an empty list")
        # Check first element
        t = type(compIDs[0]).__name__
        if not t.startswith('int'):
            raise TypeError(
                ("List of relevant component ID numbers must be made ") +
                ("up of integers; received type '%s'" % t))
        # Loop through all keys
        for face in self.faces.keys():
            # Get the current parameters
            c = self.faces[face]
            t = type(c).__name__
            # Check the type
            if t.startswith('int'):
                # Check for the compID at all
                if c not in compIDs:
                    # Delete the face
                    del self.faces[face]
            else:
                # Intersect the current value with the target list
                F = np.intersect1d(c, compIDs)
                # Check for intersections
                if len(F) == 0:
                    # Delete the face
                    del self.faces[face]
                else:
                    # Use the restricted subset
                    self.faces[face] = F
                
        
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
        
    # Copy the handle
    def WriteXML(self, fname=None):
        """Write the configuration to file
        
        :Call:
            >>> cfg.WriteXML(fname=None)
        :Inputs:
            *cfg*: :class:`cape.config.Config`
                Instance of configuration class
            *fname*: {``None``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-08-23 ``@ddalle``: First version
        """
        self.Write(fname)
        
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
            
    
    # Get a defining component ID from the *Properties* section
    def GetPropCompID(self, comp):
        """Get a *CompID* from the "Properties" section without recursion
        
        :Call:
            >>> compID = cfg.GetPropCompID(comp)
        :Inputs:
            *cfg*: :class:`cape.config.Config`
                XML-based configuration interface
            *c*: :class:`str`
                Name of component in "Tree" section
        :Outputs:
            *compID*: :class:`int`
                Full list of component IDs in *c* and its children
        :Versions:
            * 2016-10-21 ``@ddalle``: First version
        """
        # Get the properties for the component
        compID = self.GetCompID(comp)
        # Type
        t = type(prop).__name__
        # Check if it's an integer
        if t.startswith('int'):
            # Valid single-component ID
            return compID
        else:
            # Missing or multiple components
            return None
    
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
    """JSON-based surface configuration interface
    
    :Call:
        >>> cfg = ConfigJSON(fname="Config.json")
    :Inputs:
        *fname*: {``"Config.json"``} | :class:`str`
            Name of JSON file from which to read configuration tree and props
    :Outputs:
        *cfg*: :class:`cape.config.ConfigJSON`
            JSON-based configuration interface
    :Attributes:
        *cfg.faces*: :class:`dict` (:class:`list` | :class:`int`)
            Dict of the component ID or IDs in each named face
        *cfg.comps*: :class:`list` (:class:`str`)
            List of components with no children
        *cfg.parents*: :class:`dict` (:class:`list` (:class:`str`))
            List of parent(s) by name for each component
    :Versions:
        * 2016-10-21 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self, fname="Config.json"):
        """Initialization method
        
        :Versions:
            * 2016-10-21 ``@ddalle``: First version
        """
        # Read the settings from an expanded and decommented JSON file
        opts = util.loadJSONFile(fname)
        # Convert to special options class
        opts = util.odict(**opts)
        # Get major sections
        self.props = opts.get("Properties")
        self.tree  = opts.get("Tree")
        self.order = opts.get("Order")
        # Save the major sections
        if self.props is None: self.props = opts
        if self.tree  is None: self.tree = opts
        # Save
        self.opts = opts
        # Initialize component list
        self.comps = []
        self.faces = {}
        self.parents = {}
        # Loop through the tree
        for c in self.opts["Tree"]:
            # Check if already processed
            if c in self.faces:
                continue
            # Process the first component
            self.AppendChild(c)
    
    # Function to display things
    def __repr__(self):
        """
        Return the string representation of a :file:`Config.xml` file.
        
        This looks like ``<cape.Config(nComp=N, faces=['Core', ...])>``
        """
        # Return a string.
        return '<cape.ConfigJSON(nComp=%i, faces=%s)>' % (
            len(self.faces), self.faces.keys())
    
    # Method to get CompIDs from generic input
    def GetCompID(self, face):
        """Return a list of component IDs from generic input
        
        :Call:
            >>> compID = cfg.GetCompID(face)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                Instance of configuration class
            *face*: :class:`str` | :class:`int` | :class:`list`
                Component number, name, or list of component numbers and names
        :Outputs:
            *compID*: :class:`list` (:class:`int`)
                List of component IDs
        :Versions:
            * 2014-10-12 ``@ddalle``: First version
            * 2016-10-21 ``@ddalle``: Copied from ``Config.xml``
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
        cfg.comps = self.comps
        cfg.props = self.props.copy()
        cfg.tree  = self.tree.copy()
        cfg.parents = self.parents.copy()
        # Output
        return cfg
    
    # Process children
    def AppendChild(self, c, parent=None):
        """Process one component of the tree and recurse into any children
        
        :Call:
            >>> compID = cfg.AppendChild(c, parent=None)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration interface
            *c*: :class:`str`
                Name of component in "Tree" section
            *parent*: {``None``} | :class:`str`
                Name of parent component when called recursively
        :Outputs:
            *compID*: :class:`list` (:class:`int`)
                Full list of component IDs in *c* and its children
        :Versions:
            * 2016-10-21 ``@ddalle``: First version
        """
        # Check for compID
        cID = self.GetPropCompID(c)
        # Initialize component
        if cID is None:
            # Initialize with no compID (pure container)
            compID = []
        else:
            # Initialize with one compID (container + fall back single comp)
            compID = [cID]
        # Initialize parents list if necessary
        self.parents.setdefault(c, [])
        # Check for parent and append it if not already present
        if parent is not None and parent not in self.parents[c]:
            self.parents[c].append(parent)
        # Get the children of this component
        C = self.tree.get(c, [])
        # Loop through children
        for child in C:
            # Check if it has been processed
            if child in self.faces:
                # Get the components to add from that child
                f = self.faces[child]
                # Check the type
                if type(f).__name__.startswith('int'):
                    compID.append(f)
                else:
                    compID += f
                # Update parent list
                if c not in self.parents[child]:
                    self.parents[child].append(c)
                continue
            # Otherwise, check if this is also a parent
            if child in self.tree:
                # Nest
                compID += self.AppendChild(child, parent=c)
                continue
            # Get the component ID from the "Properties" section
            cID = self.GetPropCompID(child)
            # Check for component
            if cID is None:
                # Missing property
                print(("Skipping component '%s'; not a parent " % child) +
                    'and has no "CompID"')
            # Set the component for *child*
            self.faces[child] = cID
            self.parents[child] = [c]
            self.comps.append(child)
            # Append to the current component's list
            compID.append(cID)
        # Save the results
        self.faces[c] = compID
        # Output
        return compID
        
    # Eliminate all CompID numbers not actually used
    def RestrictCompID(self, compIDs):
        """Restrict component IDs in *cfg.faces* to those in a specified list
        
        :Call:
            >>> cfg.RestrictCompID(compIDs)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration interface
            *compIDs*: :class:`list` (:class:`int`)
                List of relevant component IDs
        :Versions:
            * 2016-11-05 ``@ddalle``: First version
        """
        # Check inputs
        t = type(compIDs).__name__
        if t not in ['list', 'ndarray']:
            raise TypeError(
                ("List of relevant component ID numbers must have type ") + 
                ("'int' or 'ndarray'; received '%s'" % t))
        # Check length
        if len(compIDs) < 1:
            raise ValueError("Invalid request to restrict to an empty list")
        # Check first element
        t = type(compIDs[0]).__name__
        if not t.startswith('int'):
            raise TypeError(
                ("List of relevant component ID numbers must be made ") +
                ("up of integers; received type '%s'" % t))
        # Loop through all keys
        for face in self.faces.keys():
            # Get the current parameters
            c = self.faces[face]
            t = type(c).__name__
            # Check the type
            if t.startswith('int'):
                # Check for the compID at all
                if c not in compIDs:
                    # Delete the face
                    del self.faces[face]
            else:
                # Intersect the current value with the target list
                F = np.intersect1d(c, compIDs)
                # Check for intersections
                if len(F) == 0:
                    # Delete the face
                    del self.faces[face]
                else:
                    # Use the restricted subset
                    self.faces[face] = F
    
    # Get list of components that are not parents
    def GetTriFaces(self):
        """Get the names of faces that are of type "tri" (not containers)
        
        :Call:
            >>> comps = cfg.GetTriFaces()
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration instance
        :Outputs:
            *comps*: :class:`list` (:class:`str`)
                List of lowest-level component names
        :Versions:
            * 2016-11-07 ``@ddalle``: First version
        """
        # Initialize
        comps = []
        # Loop through all faces
        for face in self.faces:
            # Get the compID information for this face
            compID = self.faces[face]
            # Type
            t = type(compID).__name__
            # Check
            if compID is None:
                # Empty instruction
                continue
            if t.startswith('int'):
                # Valid face (non-list integer instruction)
                comps.append(face)
            else:
                # Get the compID from the "Properties" section
                c = self.GetPropCompID(face)
                # Check for length one
                if len(compID) != 1:
                    # Multiple faces
                    continue
                elif compID[0] == c:
                    # One face, matching "Properties" section
                    comps.append(face)
                else:
                    # One face, not matching "Properties" section
                    continue
        # Output
        return comps
                
            
    # Write configuration
    def WriteXML(self, fname="Config.xml", Name=None, Source=None):
        """Write a GMP-type ``Config.xml`` file
        
        :Call:
            >>> cfg.WriteXML(fname="Config.xml", Name=None, Source=None)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration instance
            *fname*: {``"Config.xml"``} | :class:`str`
                Name of file to write
            *Name*: {``None``} | :class:`str`
                Name of the configuration, defaults to *fname*
            *Source*: {``"Components.i.tri"``} | :class:`str`
                Name of the "source" triangulation file, has no effect
        :Versions:
            * 2016-11-06 ``@ddalle``: First version
        """
        # Open the file.
        f = open(fname, 'w')
        # Write opening handle
        f.write('<?xml version="1.0" encoding="utf-8"?>\n\n')
        # Get the name and source
        if Name   is None: Name = fname
        if Source is None: Source = "Components.i.tri"
        # Write the "configuration" element
        f.write('<Configuration Name="%s" Source="%s">\n\n' % (Name,Source))
        # Get sorted faces
        faces = self.SortCompIDs()
        # Loop through the elements
        for face in faces:
            # Get the compID
            compID = self.faces[face]
            # Don't mess around with ``None``
            if compID is None: continue
            # Type
            t = type(compID).__name__
            # Check if it's a basic face or a container
            if t.startswith('int'):
                # Integers are already faces
                q = True
            else:
                # Get the compID from properties
                c = self.GetPropCompID(face)
                # Check for length one
                if len(compID) > 1:
                    # Multiple faces
                    q = False
                elif compID[0] == c:
                    # One face, matching "Properties" section
                    q = True
                    compID = compID[0]
                else:
                    # One component, but from a single child
                    q = False
            # Get parent
            parent = self.parents[face]
            # Type
            t = type(parent).__name__
            # Check for list
            if t in ['list', 'ndarray']:
                # Check for multiple components
                if len(parent) == 0:
                    # No parents
                    print("  No parent for component '%s'" % face)
                    parent = None
                elif len(parent) > 1:
                    # Let's warn for now, verbose
                    print(
                        ("  WARNING: Component '%s' has multiple " % face) +
                        ("parents (%s); using first entry" % parent))
                    parent = parent[0]
                else:
                    # Take first entry
                    parent = parent[0]
            # Common portion of face label
            f.write('  <Component Name="%s" ' % face)
            # Check for parent
            if parent is not None:
                f.write('Parent="%s" ' % parent)
                # Append the parent to the list
                if parent not in faces:
                    faces.append(parent)
            # Write the right type of component
            if q:
                # Write single-face
                f.write('Type="tri">\n')
                f.write('    <Data>Face Label=%i</Data>\n' % compID)
                f.write('  </Component>\n\n')
            else:
                # Write container
                f.write('Type="container">\n')
                f.write('  </Component>\n\n')
        # Close the "Configuration" element
        f.write("</Configuration>\n")
        # Close the file
        f.close()
        
    # Write .mapbc file
    def WriteFun3DMapBC(self, fname):
        """Write a Fun3D ".mapbc" file
        
        :Call:
            >>> cfg.WriteFun3DMapBC(fname)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration instance
            *fname*: :class:`str`
                Name of mapbc file to write
        :Versions:
            * 2016-11-07 ``@ddalle``: First version
        """
        # Get the list of tri faces
        faces = self.GetTriFaces()
        # Initialize final list and dictionary of BCs
        comps0 = []
        bcs = {}
        compIDs = {}
        # List used for sorting tri faces
        inds = []
        # Get the manually-set "Order" paremter
        compOrder = self.opts.get("Order", [])
        # Loop through *faces*
        for face in faces:
            # Get the BCs from the "Properties" section
            bc      = self.GetProperty(face, 'bc')
            fun3dbc = self.GetProperty(face, 'fun3d_bc')
            aflr3bc = self.GetProperty(face, 'aflr3_bc')
            # Turn into a single bc
            if fun3dbc is None:
                # No explicit Fun3D boundary condition; check overall 'bc'
                if bc is None:
                    # No boundary condition: default (viscous wall=4000)
                    fun3dbc = 4000
                elif aflr3bc is None:
                    # The 'bc' parameter prefers to affect AFLR3
                    fun3dbc = 4000
                    aflr3bc = bc
                else:
                    # Otherwise, use the *bc* parameter to set Fun3D BC
                    fun3dbc = bc
            elif aflr3bc is None:
                # No explicit Fun3D bc
                if bc is None:
                    # No AFLR3 setting; use default
                    aflr3bc = -1
                else:
                    # Copy from *bc*
                    aflr3bc = bc
            # Check for valid wall boundary condition
            if (aflr3bc == 3) or (fun3dbc == False):
                # This is a source; do not add it to the Fun3D BCs
                continue
            # Otherwise, add the component
            comps0.append(face)
            bcs[face] = fun3dbc
            # Set the component ID
            compID = self.GetPropCompID(face)
            compIDs[face] = compID
            # Get the sorting parameter
            if face in compOrder:
                # Use the index in the "Order" section as a sort key
                inds.append(compOrder.index(face))
            else:
                # Use the CompID to sort
                inds.append(compID)
        # Sort the components
        I = np.argsort(inds)
        # Start final component list
        comps = []
        # Use this sorting order to reorder *comps*
        for i in I:
            comps.append(comps0[i])
        
        # Open the .mapbc file
        f = open(fname, 'w')
        # Write the number of components
        f.write("%9s %i\n" % (" ", len(comps)))
        # Loop through components
        for comp in comps:
            # Write compID, BC, and name
            f.write("%7i   %4i   %s\n" % (compIDs[comp], bcs[comp], comp))
        # Close the file
        f.close()
        
    # Renumber a component
    def RenumberCompID(self, face, compID):
        """Renumber the component ID number
        
        This affects *cfg.faces* for *face* and each of its parents, and it
        also resets the component ID number in *cfg.props*.
        
        :Call:
            >>> cfg.RenumberCompID(face, compID)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration
            *face*: :class:`str`
                Name of component to rename
        :Versions:
            * 2016-11-09 ``@ddalle``: First version
        """
        # Get the current component number
        compi = self.faces[face]
        t = type(compi).__name__
        # Reset it
        if t in ['list', 'ndarray']:
            # Extract the original component ID from singleton list
            compi = compi[0]
            # Reset it (list)
            self.faces[face] = [compID]
        else:
            # Reset it (number)
            self.faces[face] = compID
        # Get the component ID from "Properties"
        compp = self.props[face]
        t = type(compp).__name__
        # Check for single number
        if t == 'dict':
            # Set the CompID property
            self.props[face]["CompID"] = compID
        else:
            # Single number
            self.props[face] = compID
        # Loop through the parents
        self.RenumberCompIDParent(face, compi, compID)
        
        
    # Renumber the parents of one component.
    def RenumberCompIDParent(self, face, compi, compo):
        """Recursively renumber the component ID numbers for parents of *face* 
        
        :Call:
            >>> cfg.RenumberCompIDParent(face, compi, compo)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration
            *face*: :class:`str`
                Name of component for which parents should be renumbered
            *compi*: :class:`int`
                Incoming component ID number
            *compo*: :class:`int`
                Outgoing component ID number
        :Versions:
            * 2016-11-09 ``@ddalle``: First version
        """
        # Get parent
        parents = self.parents[face]
        # Check None
        if parents is None: return
        # Loop through parents
        for parent in parents:
            # Get the parent's face data
            comp = self.faces[parent]
            t = type(comp).__name__
            # Replace the parent value
            if t == 'list':
                # Check for membership
                if compi not in comp: continue
                # Make the replacement
                comp[comp.index(compi)] = compo
            else:
                # Check for membership
                I = np.where(comp==compi)[0]
                if len(I) == 0: continue
                # Make the replacement
                comp[I[0]] = compo
            # Recurse
            self.RenumberCompIDParent(parent, compi, compo)
            
    # Reset component IDs
    def ResetCompIDs(self):
        """Renumber component IDs 1 to *n*
        
        :Call:
            >>> comps = cfg.ResetCompIDs()
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration instance
        :Versions:
            * 2016-11-09 ``@ddalle``: First version
        """
        # Get the list of tri faces
        comps = self.SortCompIDs()
        # Loop through faces in order
        for i in range(len(comps)):
            # Get the face
            face = comps[i]
            # Renumber
            self.RenumberCompID(face, i)
        
    # Renumber Component IDs 1 to *n*
    def SortCompIDs(self):
        """Get ordered list of components
        
        :Call:
            >>> comps = cfg.SortCompIDs()
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration instance
        :Outputs:
            *comps*: :class:`list` (:class:`str`)
                List of components
        :Versions:
            * 2016-11-09 ``@ddalle``: First version
        """
        # Get the list of tri faces
        faces = self.GetTriFaces()
        # Initialize final list and dictionary of BCs
        comps0 = []
        bcs = {}
        compIDs = {}
        # List used for sorting tri faces
        inds = []
        # Get the manually-set "Order" paremter
        compOrder = self.opts.get("Order", [])
        # Loop through *faces*
        for face in faces:
            # Get the BCs from the "Properties" section
            bc      = self.GetProperty(face, 'bc')
            fun3dbc = self.GetProperty(face, 'fun3d_bc')
            aflr3bc = self.GetProperty(face, 'aflr3_bc')
            # Turn into a single bc
            if fun3dbc is None:
                # No explicit Fun3D boundary condition; check overall 'bc'
                if bc is None:
                    # No boundary condition: default (viscous wall=4000)
                    fun3dbc = 4000
                elif aflr3bc is None:
                    # The 'bc' parameter prefers to affect AFLR3
                    fun3dbc = 4000
                    aflr3bc = bc
                else:
                    # Otherwise, use the *bc* parameter to set Fun3D BC
                    fun3dbc = bc
            elif aflr3bc is None:
                # No explicit Fun3D bc
                if bc is None:
                    # No AFLR3 setting; use default
                    aflr3bc = -1
                else:
                    # Copy from *bc*
                    aflr3bc = bc
            # Otherwise, add the component
            comps0.append(face)
            bcs[face] = fun3dbc
            # Set the component ID
            compID = self.GetPropCompID(face)
            compIDs[face] = compID
            # Get the sorting parameter
            if face in compOrder:
                # Use the index in the "Order" section as a sort key
                inds.append(compOrder.index(face))
            else:
                # Use the CompID to sort
                inds.append(compID)
        # Sort the components
        I = np.argsort(inds)
        # Start final component list
        comps = []
        # Use this sorting order to reorder *comps*
        for i in I:
            comps.append(comps0[i])
        # Output
        return comps
    
    # Get a defining component ID from the *Properties* section
    def GetPropCompID(self, comp):
        """Get a *CompID* from the "Properties" section without recursion
        
        :Call:
            >>> compID = cfg.GetPropCompID(comp)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration interface
            *c*: :class:`str`
                Name of component in "Tree" section
        :Outputs:
            *compID*: :class:`int`
                Full list of component IDs in *c* and its children
        :Versions:
            * 2016-10-21 ``@ddalle``: First version
        """
        # Get the properties for the component
        prop = self.props.get(comp, {})
        # Type
        t = type(prop).__name__
        # Check for component
        if t.startswith('int'):
            # Directly specified CompID (no other properties)
            return prop
        elif t.startswith('float'):
            # Convert to integer
            return int(prop)
        elif t not in ['dict', 'odict']:
            # Not a valid type
            raise TypeError(("Properties for component '%s' " % comp) +
                "must be either a 'dict' or 'int'")
        elif "CompID" not in prop:
            # Missing property
            return None
        else:
            # Get the component ID number from the property dict
            return prop["CompID"]
    
    # Get a property
    def GetProperty(self, comp, k):
        """Get a cascading property from a component or its parents
        
        :Call:
            >>> v = cfg.GetProperty(comp, k)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration interface
            *comp*: :class:`str`
                Name of component to query
            *k*: :class:`str`
                Name of property to query
        :Outputs:
            *v*: ``None`` | :class:`any`
                Value of *k* from *comp* with fallback to parents
        :Versions:
            * 2016-10-21 ``@ddalle``: First version
        """
        # Get component properties
        opts = self.props.get(comp, {})
        # Check type
        if type(opts).__name__ not in ['dict', 'odcit']:
            opts = {}
        # Check for the property
        if k in opts:
            return opts[k]
        # Loop through parents
        for parent in self.parents[comp]:
            # Get the value from that parent (note: this may recurse)
            v = self.GetProperty(parent, k)
            # Check for success (otherwise try next parent if there is one)
            if v is not None:
                return v
        # If this point is reached, could not find property in any parent
        return None
    
# class ConfigJSON

