#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.config`: Surface configuration module
=================================================

This is a module to interact with Cart3D's ``Config.xml`` or similar
files whose primary purpose is to describe and label surface geometry.
In general, it can be used to create groups of surfaces using an XML,
JSON, or MIXSUR file format.  This originates from a Cart3D/OVERFLOW
convention, but it can be used with other modules as well.  Presently
there are three classes, which each interface with a specific type of
file:

    ======================== ================= =============================
    Class                    Common File       Description
    ======================== ================= =============================
    :class:`ConfigXML`       ``Config.xml``    GMP component XML file
    :class:`ConfigJSON`      ``Config.json``   Cape JSON surf config file
    :class:`ConfigMIXSUR`    ``mixsur.i``      CGT input families stream
    ======================== ================= =============================

It is typical for a surface definition, whether a triangulation, system
of overset structured grids, or mixed quads and triangles, to have each
surface polygon to have a numbered component ID.  This allows a user to
group triangles and quads or other polygons together in some relevant
way.  For example, the user may tag each polygon on the left wing with
the component ID of ``12``, and the entire surface is broken out in a
similar fashion.

The :mod:`cape.config` module allows the user to do two main things:
give meaningful names to these component IDs and group component IDs
together. For example, it is usually much more convenient to refer to
``"left_wing"`` than remember to put ``"12"`` in all the data books,
reports, etc. In addition, a user usually wants to know the integrated
force on the entire airplane (or whatever other type of configuration is
under investigation), so it is useful to make another component called
``"vehicle"`` that contains ``"left_wing"``, ``"right_wing"``, and
``"fuselage"``. The user specifies this in the XML file using the
following syntax.

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

The *Source* attribute of the first tag is not that important; it's
placed there based on a Cart3D template. The choice of encoding is not
crucial but does affect the validity of the XML file, which some
applications may check.

The major limitation of the XML format is that a component may not have
multiple parents. A parent may have parent, allowing the user to
subdivide groups into smaller groups, but the user may not, for example,
split the vehicle into left half and right half and also create
components for forward half and aft half.

An alternative version of the same is to use the JSON configuration
format developed for Cape.  It allows mixed parents like the
forward/aft, left/right example described above, and it also allows the
users to specify boundary conditions within the config file, which can
consolidate information about your surface that would otherwise require
multiple files.  The version of the above configuration in JSON form is
below.

    .. code-block:: javascript

        {
            "Tree": {
                "vehicle": [
                    "fuselage",
                    "right_wing",
                    "left_wing"
                ]
            },
            "Properties": {
                "fuselage": {
                    "CompID": 1
                },
                "right_wing": 11,
                "left_wing": 12
            }
        }

The ``"Properties"`` section allows generic options, for example those
in the table below.  If the *Properties* for a face is not a
:class:`dict`, it must be an :class:`int`, which is assumed to be the
*CompID* parameter for that face.

    ======================   ==========================================
    Component Property       Description
    ======================   ==========================================
    *CompID*                 Surface component integer
    *Parent*                 Name of principal parent if ambiguous
    *fun3d_bc*               Boundary condition for FUN3D
    *aflr3_bc*               Boundary condition for AFLR3
    *blds*                   Initial BL spacing, AFLR3
    *bldel*                  Prism layer height, AFLR3
    ======================   ==========================================

Finally, the :class:`ConfigMIXSUR` file interprets the streams that are
often given as inputs to the Chimera Grid Tools executables ``mixsur``
or ``usurp``.  These functions are used to divide an overset structured
grid system into surface components and can also be used to create
unique surface triangulations.  These input streams are often saved as a
file, by convention ``mixsur.i``, and read into the CGT executable using
a call such as ``mixsur < mixsur.i``.

"""

# Standard library
import os
import re

# Standard library: direct imports
import xml.etree.ElementTree as ET

# Third-party modules
import numpy as np

# CAPE modules
from .util import RangeString, SplitLineGeneral
from .cfdx.options import util


# Configuration class
class ConfigXML(object):
    r"""Interface to Cart3D ``Config.xml`` files

    :Call:
        >>> cfg = cape.ConfigXML(fname='Config.xml')
    :Inputs:
        *fname*: :class:`str`
            Name of configuration file to read
    :Outputs:
        *cfg*: :class:`cape.config.ConfigXML`
            XML surface config instance
    :Versions:
        * 2014-10-12 ``@ddalle``: Version 1.0
    """
    # Initialization method
    def __init__(self, fname="Config.xml"):
        r"""Initialization method

        :Versions:
            * 2014-10-12 ``@ddalle``: Version 1.0
        """
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
        # Initialize Tris
        self.comps = []
        # Initialize the transformation data
        self.transform = {}
        # Check for unnamed component.
        if None in self.Names:
            raise ValueError(
                "At least one component in "
                + ("'%s' is lacking a name." % self.fname))
        # Loop through points to get the labeled faces.
        for c in self.Comps:
            # Check the type.
            if c.get('Type') == 'tri':
                # Triangulation, face label
                self.ProcessTri(c)
                # Get Component list
                self.CompIDs = [self.faces.get(comp,0) for comp in self.comps]
            elif c.get('Type') == 'struc':
                # Structured grid list
                self.ProcessStruc(c)

    # Function to display things
    def __repr__(self):
        r"""Representation method

        Form: ``<cape.ConfigXML(nComp=N, faces=['Core', ...])>``

        :Versions:
            * 2014-10-12 ``@ddalle``: Version 1.0
        """
        # Return a string.
        return '<cape.ConfigXML(nComp=%i, faces=%s)>' % (
            len(self.faces), self.faces.keys())

    # Process a tri component
    def ProcessTri(self, c):
        r"""Process a GMP component of type ``"tri"``

        :Call:
            >>> cfg.ProcessTri(c)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
            *c*: :class:`xml.Element`
                XML interface to element with tag ``"Component"``
        :Versions:
            * 2016-08-23 ``@ddalle``: Version 1.0
            * 2021-09-30 ``@ddalle``: Version 1.1
                - :func:`Element.getchildren` removed in Python 3.9
                - *c* is iterable
        """
        # Get the component name
        comp = c.get('Name')
        # Loop through children
        for d in c:
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
        r"""Process a GMP component of type ``'struc'``

        :Call:
            >>> cfg.ProcessStruc(c)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
            *c*: :class:`xml.Element`
                XML interface to element with tag ``"Component"``
        :Versions:
            * 2016-08-23 ``@ddalle``: Version 1.0
            * 2021-09-30 ``@ddalle``: Version 1.1
                - :func:`Element.getchildren` removed in Python 3.9
                - *c* is iterable
        """
        # Get the component name
        comp = c.get('Name')
        # Loop through children
        for d in c:
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
        r"""Process a GMP data element with text for "Face Label"

        :Call:
            >>> compID = cfg.ProcessTriData(comp, d)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
            *d*: :class:`xml.Element`
                XML interface to element with tag ``'Data'``
        :Outputs:
            *compID*: :class:`int`
                Component ID number from "Face Label"
        :Attributes:
            *cfg.faces[comp]*: :class:`int`
                Gets set to *compID*
        :Versions:
            * 2016-08-23 ``@ddalle``: Version 1.0
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
        # Save as a component
        self.comps.append(comp)
        # Output
        return compID

    # Process grid list data
    def ProcessStrucData(self, comp, d):
        r"""Process a GMP data element with text for "Grid List"

        :Call:
            >>> compID = cfg.ProcessStrucData(comp, d)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
            *d*: :class:`xml.Element`
                XML interface to element with tag ``'Data'``
        :Outputs:
            *compID*: :class:`list`\ [:class:`int`]
                Grid numbers "Grid List"
        :Attributes:
            *cfg.faces[comp]*: :class:`list`\ [:class:`int`]
                Gets set to *compID*
        :Versions:
            * 2016-08-23 ``@ddalle``: Version 1.0
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
        r"""Process a GMP transformation

        :Call:
            >>> cfg.ProcessTransform(t)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
            *t*: :class:`xml.Element`
                XML interface to element with tag ``'Transform'``
        :Versions:
            * 2016-08-23 ``@ddalle``: Version 1.0
            * 2021-09-30 ``@ddalle``: Version 1.1; iterate *t* directly
        """
        # Check the tag
        if t.tag != "Transform":
            raise ValueError("Element '%s' does not have 'Transform' tag" % t)
        # Initialize transformations
        self.transform[comp] = []
        # Loop through children
        for x in t:
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
        r"""Append a component ID to a parent container and its parents

        :Call:
            >>> cfg.AppendParent(c, compID)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
            *c*: :class:`xml.Element`
                XML interface to element with tag ``'Component'``
            *compID*: :class:`int`
                Component ID number to add to parents' lists
        :Outputs:
            *comp*: :class:`dict`
                Dictionary with *compID* appended in appropriate places
        :Versions:
            * 2014-10-13 ``@ddalle``: Version 1.0
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
        r"""Restrict component IDs in *cfg.faces* to manual list

        :Call:
            >>> cfg.RestrictCompID(compIDs)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML-based configuration interface
            *compIDs*: :class:`list`\ [:class:`int`]
                List of relevant component IDs
        :Versions:
            * 2016-11-05 ``@ddalle``: Version 1.0
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
        for face in list(self.faces.keys()):
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
        r"""Modify or add a rotation for component *comp*

        :Call:
            >>> cfg.SetRotation(comp, i=None, **kw)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
            *comp*: :class:`str`
                Name of component
            *i*: {``None``} | :class:`int`
                Index of the rotation
            *Center*: {``[0.0,0.0,0.0]``} | :class:`list` | :class:`str`
                Point about which to rotate
            *Axis*: {``[0.0, 1.0, 0.0]``} | :class:`list` | :class:`str`
                Axis about which to rotate
            *Angle*: {``0.0``} | :class:`float` | :class:`str`
                Angle for rotation
            *Frame*: {``"Body"``} | ``None``
                Rotation type, body frame or Overflow frame
        :Versions:
            * 2016-08-23 ``@ddalle``: Version 1.0
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
            "Center":  kw.get("Center"),
            "Axis":    kw.get("Axis"),
            "Angle":   kw.get("Angle"),
            "Frame":   kw.get("Frame"),
        }
        # Apply changes as appropriate
        if i == n:
            # Add the whole rotation
            T.append(R)
        else:
            # Ensure type
            T[i]["Type"] = "Rotate"
            # Only apply either blank settings or directly-specified values
            for k, v in R.items():
                # Check if we should overwrite current settings
                if v is not None:
                    T[i][k] = v

    # Set transformation
    def SetTranslation(self, comp, i=None, **kw):
        r"""Modify or add a translation for component *comp*

        :Call:
            >>> cfg.SetTranslation(comp, i=0, **kw)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
            *comp*: :class:`str`
                Name of component
            *i*: {``0``} | :class:`int`
                Index of the rotation
            *Displacement*: :class:`list` | :class:`str`
                Vector to move component
        :Versions:
            * 2016-08-23 ``@ddalle``: Version 1.0
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
        r"""Write the configuration to file

        :Call:
            >>> cfg.Write(fname=None)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
            *fname*: {``None``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-08-23 ``@ddalle``: Version 1.0
        """
        # Default file name
        if fname is None:
            fname = self.fname
        # Open the file for writing
        f = open(fname, 'w')
        # Write the header.
        f.write("<?xml version='1.0' encoding='utf-8'?>\n")
        # Get the Configuration properties
        try:
            conf = self.XML.getroot().attrib
        except AttributeError:
            conf = {"Name": fname, "Source": "cape.tri"}
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
    def WriteXML(self, fname=None, name=None):
        r"""Write the configuration to file

        :Call:
            >>> cfg.WriteXML(fname=None)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
            *fname*: {``None``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-08-23 ``@ddalle``: Version 1.0
        """
        self.Write(fname)

    # Function to write a component
    def WriteComponent(self, f, comp):
        r"""Write a "Component" element to file

        Data (either "Face Label" or "Grid List") is written, and any
        transformations are also written.

        :Call:
            >>> cfg.WriteComponent(f, comp)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
            *f*: :class:`file`
                File handle open for writing
            *comp*: :class:`str`
                Name of component to write
        :Versions:
            * 2016-08-23 ``@ddalle``: Version 1.0
            * 2017-08-25 ``@ddalle``: Version 1.1, skip negative IDs
        """
        # Get the component index
        i = self.Names.index(comp)
        # Get the component interface
        c = self.Comps[i]
        # Process the type
        if c.get("Type", "tri") == "tri":
            # Exit if not in *faces*
            if comp not in self.faces:
                return
            # Get the list of components in the component.
            compID = self.GetCompID(comp)
            # Check if negative
            if any(compID) < 0: return
            print("       %s: %s" % (comp, compID))
            # Single component
            lbl = "Face Label"
        else:
            # Grid list
            lbl = "Grid List"
        # Begin the tag.
        f.write("  <Component")
        # Loop through the attributes
        for k in c.attrib:
            f.write(' %s="%s"' % (k, c.attrib[k]))
        # Close the component opening tag
        f.write(">\n")
        # Write the data element with "Face Label"
        self.WriteComponentData(f, comp, label=lbl)
        # Write transformation
        if comp in self.transform:
            # Write the transformation
            self.WriteComponentTransform(f, comp)
        # Close the component element.
        f.write("  </Component>\n\n")

    # Method to write data
    def WriteComponentData(self, f, comp, label=None):
        r"""Write a "Data" element to file

        :Call:
            >>> cfg.WriteComponentData(f, comp, label=None)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
            *comp*: :class:`str`
                Name of component to write
            *label*: {``None``} | ``"Face Label"`` | ``"Grid List"``
                Label used to specify data
        :Versions:
            * 2016-08-23 ``@ddalle``: Version 1.0
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
        r"""Write a "Transform" element to file

        :Call:
            >>> cfg.WriteComponentData(f, comp, label=None)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
            *comp*: :class:`str`
                Name of component to write
            *label*: {``None``} | ``"Face Label"`` | ``"Grid List"``
                Label used to specify data
        :Versions:
            * 2016-08-23 ``@ddalle``: Version 1.0
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
        r"""Return a list of component IDs from generic input

        :Call:
            >>> compID = cfg.GetCompID(face)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
            *face*: :class:`str` | :class:`int` | :class:`list`
                Component number, name, or list thereof
        :Outputs:
            *compID*: :class:`list`\ [:class:`int`]
                List of component IDs
        :Versions:
            * 2014-10-12 ``@ddalle``: Version 1.0
        """
        # Initialize the list.
        compID = []
        # Process the type.
        if type(face).__name__ in ['list', 'ndarray']:
            # Loop through the inputs.
            for f in face:
                # Call this function so it passes to the non-array portion.
                compID += self.GetCompID(f)
                # Sort components
                compID.sort()
        elif face in self.faces:
            # Process the face
            cID = self.faces[face]
            # Check if it's a list.
            if type(cID).__name__ in ['list', 'ndarray']:
                # Add the list.
                compID += list(cID)
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

    # Get name of a compID
    def GetCompName(self, compID):
        r"""Get the name of a component by its number

        :Call:
            >>> face = cfg.GetCompName(compID)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
            *compID*: :class:`int`
                Component ID number
        :Outputs:
            *face*: ``None`` | :class:`str`
                Name of so-numbered component, if any
        :Versions:
            * 2017-03-30 ``@ddalle``: Version 1.0
        """
        # Make the list of current compIDs
        self.CompIDs = [self.faces[c] for c in self.comps if c in self.faces]
        # Check if CompID is present
        if compID not in self.CompIDs:
            # No match
            return
        # Get an array of matches
        CompIDs = np.array(self.CompIDs)
        # Find matches
        I = np.where(compID == CompIDs)[0]
        # Loop through them to make sure it's *still* in *self.faces*
        for i in I:
            # Candidate
            face = self.comps[i]
            # Check if present
            if face in self.faces:
                # Output
                return face

    # Get a defining component ID from the *Properties* section
    def GetPropCompID(self, comp):
        r"""Get a *CompID* from the "Properties" section w/o recursion

        :Call:
            >>> compID = cfg.GetPropCompID(comp)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML-based configuration interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *compID*: :class:`int`
                Full list of component IDs in *c* and its children
        :Versions:
            * 2016-10-21 ``@ddalle``: Version 1.0
        """
        # Get the properties for the component
        compID = self.GetCompID(comp)
        # Type
        t = type(compID).__name__
        # Check if it's an integer
        if t.startswith('int'):
            # Valid single-component ID
            return compID
        elif (t in ['list', 'ndarray']) and (len(compID)==1):
            # Valid singleton list
            return compID[0]
        else:
            # Missing or multiple components
            return None

    # Method to copy a configuration
    def Copy(self):
        r"""Copy a configuration interface

        :Call:
            >>> cfg2 = cfg.Copy()
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
        :Outputs:
            *cfg2*: :class:`cape.config.ConfigXML`
                Copy of input
        :Versions:
            * 2014-11-24 ``@ddalle``: Version 1.0
        """
        # Initialize object.
        cfg = ConfigXML()
        # Copy the dictionaries.
        cfg.faces = self.faces.copy()
        cfg.transform = self.transform.copy()
        cfg.Names = list(self.Names)
        cfg.Comps = list(self.Comps)
        # Output
        return cfg
# class Config


# Config based on MIXSUR
class ConfigMIXSUR(object):
    r"""Class to build a surf configuration from a ``mixsur`` file

    :Call:
        >>> cfg = ConfigMIXSUR(fname="mixsur.i", usurp=True)
    :Inptus:
        *fname*: {``"mixsur.i"``} | :class:`str`
            Name of ``mixsur`` input file
        *usurp*: {``True``} | ``False``
            Whether or not to number components as with ``usurp`` output
    :Outputs:
        *cfg*: :class:`cape.config.ConfigMIXSUR`
            ``mixsur``-based configuration interface
    :Attributes:
        *cfg.faces*: :class:`dict` (:class:`list` | :class:`int`)
            Dict of component ID or IDs in each named face
        *cfg.comps*: :class:`list` (:class:`str`)
            List of components with no children
        *cfg.parents*: :Class:`dict` (:class:`list` (:class:`str`))
            List of parent(s) by name for each component
        *cfg.IDs*: :class:`list`\ [:class:`int`]
            List of unique component ID numbers
    :Versions:
        * 2016-12-29 ``@ddalle``: Version 1.0
    """
    # Initialization method
    def __init__(self, fname="mixsur.i", usurp=True):
        r"""Initialization method

        :Versions:
            * 2016-12-29 ``@ddalle``: Version 1.0
        """
        # Initialize the data products
        self.faces = {}
        self.comps = []
        self.parents = {}
        self.refs = []
        # Grid list, for inputs
        self.grids = []
        # Open the file
        f = open(fname)
        # Read the first line
        V = self.readline(f)
        # Save the Mach number, angle of attack, and other conditions
        try:
            self.FSMACH = float(V[0])
            self.ALPHA  = float(V[1])
            self.BETA   = float(V[2])
            self.REY    = float(V[3])
            self.GAMINF = float(V[4])
            self.TINF   = float(V[5])
        except Exception:
            pass
        # Read the next line
        V = self.readline(f)
        # Get number of reference conditions
        try:
            self.NREF = int(V[0])
        except Exception:
            self.NREF = 1
        # Initialize reference quantities
        self.Lref = np.zeros(self.NREF)
        self.Aref = np.zeros(self.NREF)
        self.XMRP = np.zeros(self.NREF)
        self.YMRP = np.zeros(self.NREF)
        self.ZMRP = np.zeros(self.NREF)
        # Loop through reference lines
        for i in range(self.NREF):
            # Read reference quantities
            V = self.readline(f)
            # Save reference quantities
            try:
                self.Lref[i] = float(V[0])
                self.Aref[i] = float(V[1])
                self.XMRP[i] = float(V[2])
                self.YMRP[i] = float(V[3])
                self.ZMRP[i] = float(V[4])
            except Exception:
                pass
        # Number of components
        V = self.readline(f)
        nsurf = int(V[0])
        self.NSURF = nsurf
        # Initialize components
        self.IDs = range(self.NSURF)
        # Loop through the actual inputs, which we don't really need for cfg
        for k in range(self.NSURF):
            # Read the number of grids
            V = self.readline(f)
            # Number of subsets
            nsub = int(V[0])
            # Reference condition number to use
            try:
                nref = int(V[1])
            except Exception:
                nref = 1
            # Initialize component grid information
            comp = {}
            # Save number of subsets
            comp["NSUBS"] = nsub
            # Save reference condition index
            comp["NREF"] = nref
            # Initialize grids
            subs = np.zeros((nsub,8))
            # Loop through subsets
            for j in range(nsub):
                # Read the subset input
                V = self.readline(f)
                # Save the information
                for i in range(8): subs[j,i] = int(V[i])
            # Save the subsets
            comp["SUBS"] = subs
            # Read the number of PRIS
            V = self.readline(f)
            # Number of PRIs
            npri = int(V[0])
            # Initialize PRIs
            pris = np.zeros((npri,2))
            # Loop through PRIs
            for j in range(npri):
                # Read the subset input
                V = self.readline(f)
                # Save the information
                for i in range(2): pris[j,i] = int(V[i])
            # Save PRIs
            comp["PRIS"] = pris
            # Append component
            self.grids.append(comp)
        # Read the number of components
        V = self.readline(f)
        ncomp = int(V[0])
        # Save number of components
        self.NCOMP = ncomp
        # Initialize map of CompID numbers
        # This gives the relationship between the surface number at the end of
        # mixsur input file to the actual CompID number in the grid.i.tri file
        self.Surf2CompID = {}
        # Component ID offset
        if usurp:
            # Number 1, 2, ... , NSURF
            noff = 0
        else:
            # Number NCOMP-NSURF+1, NCOMP-NSURF+2, ... , NCOMP
            noff = ncomp - nsurf
        # Loop through number of components (including groups)
        for k in range(ncomp):
            # Read the name of the component
            V = self.readline(f)
            face = V[0]
            # Read the number of component IDs
            V = self.readline(f)
            ni = int(V[0])
            # Read the reference condition index
            try:
                nref = int(V[1])
            except Exception:
                nref = 1
            # Read the components
            V = self.readline(f)
            # Determine if this is a single component or group
            if k >= ncomp-nsurf:
                # Convert component to integer
                compID = int(V[0]) + noff
                # Check *usurp*
                # Add to list of components
                self.comps.append(face)
                # Save the reference condition
                self.refs.append(nref)
                # Add to the map
                self.Surf2CompID[face] = compID
            else:
                # Convert list of component numbers to integers
                compID = [int(v)+noff for v in V[:ni]]
            # Save components
            self.faces[face] = compID
            # Initialize parents
            self.parents[face] = []
        # Close the file
        f.close()
        # Process parents
        for face in self.faces:
            # Get parents
            self.FindParents(face)

    # Check parent
    def FindParents(self, face):
        r"""Find the parents of a single face

        :Call:
            >>> cfg.FindParents(face)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigMIXSUR`
                Configuration interface for ``mixsur`` triangulations
            *face*: :class:`str`
                Name of face to check
        :Versions:
            * 2016-12-29 ``@ddalle``: Version 1.0
        """
        # Component
        comp = self.faces[face]
        # Initialize parents
        parents = []
        # Check type
        if face in self.comps:
            # Single face
            for par in self.faces:
                # Do not process single comps
                if par in self.comps: continue
                # Check if this comp is in there
                if comp in self.faces[par]:
                    parents.append(par)
        else:
            # *face* is a group
            for par in self.faces:
                # Do not process single comps
                if par in self.comps: continue
                # Do not process self
                if par == face: continue
                # Get components in *par*
                comppar = self.faces[par]
                # Initialize parent test
                qpar = True
                # Loop through components in *face*
                for c in comp:
                    # Check if *c* is not in *par*
                    if c not in comppar:
                        qpar = False
                        break
                # Append to list if *qpar* is ``True``
                if qpar:
                    parents.append(par)
        # Save the parent list
        self.parents[face] = parents


    # Method to copy a configuration
    def Copy(self):
        r"""Copy a configuration interface

        :Call:
            >>> cfg2 = cfg.Copy()
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
        :Outputs:
            *cfg2*: :class:`cape.config.ConfigXML`
                Copy of input
        :Versions:
            * 2014-11-24 ``@ddalle``: Version 1.0
        """
        # Initialize object.
        cfg = ConfigMIXSUR()
        # Copy the dictionaries.
        cfg.faces = self.faces.copy()
        cfg.parents = self.parents.copy()
        cfg.comps = list(self.comps)
        cfg.IDs = list(self.IDs)
        # Output
        return cfg

    # Read a line of text
    def readline(self, f=None, n=100):
        r"""Read a non-blank line from a CGT-like input file

        :Call:
            >>> V = cfg.readline(f=None, n=100)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigMIXSUR`
                Configuration interface for ``mixsur``
            *f*: {``None``} | :class:`file`
                File handle; defaults to *cfg.f*
            *n*: {``100``} | :class:`int` > 0
                Maximum number of lines to check
        :Outputs:
            *V*: :class:`list` (:class:`str`)
                List of substrings split by commas or spaces
        :Versions:
            * 2016-12-29 ``@ddalle``: Version 1.0
        """
        # Default file handle
        if f is None:
            try:
                # See if one is stored in *self*
                f = self.f
            except AttributeError:
                # Nothing to read!
                return
        # Initialize null output
        V = []
        # Line counter
        nline = 0
        # Loop until nonempty line is read
        while (len(V) == 0) and (nline < n):
            # Read the next line
            line = f.readline()
            # Check for end-of-file (EOF)
            if line == "": return
            # Split it
            V = SplitLineGeneral(line)
        # Output
        return V

    # Method to get CompIDs from generic input
    def GetCompID(self, face):
        r"""Return a list of component IDs from generic input

        :Call:
            >>> compID = cfg.GetCompID(face)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigMIXSUR`
                XML surface config instance
            *face*: :class:`str` | :class:`int` | :class:`list`
                Component number, name, or list thereof
        :Outputs:
            *compID*: :class:`list`\ [:class:`int`]
                List of component IDs
        :Versions:
            * 2014-10-12 ``@ddalle``: Version 1.0 (:class:`ConfigXML`)
            * 2016-12-29 ``@ddalle``: Version 1.0
        """
        # Initialize the list.
        compID = []
        # Process the type.
        if type(face).__name__ in ['list', 'ndarray']:
            # Loop through the inputs.
            for f in face:
                # Call this function so it passes to the non-array portion.
                compID += self.GetCompID(f)
                # Sort components
                compID.sort()
        elif face in self.faces:
            # Process the face
            cID = self.faces[face]
            # Check if it's a list.
            if type(cID).__name__ in ['list', 'ndarray']:
                # Add the list.
                compID += list(cID)
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

    # Get name of a compID
    def GetCompName(self, compID):
        r"""Get the name of a component by its number

        :Call:
            >>> face = cfg.GetCompName(compID)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
            *compID*: :class:`int`
                Component ID number
        :Outputs:
            *face*: ``None`` | :class:`str`
                Name of so-numbered component, if any
        :Versions:
            * 2017-03-30 ``@ddalle``: Version 1.0
        """
        # Make sure there is a list of CompIDs
        try:
            self.CompIDs
        except AttributeError:
            # Make the list
            self.CompIDs = [self.faces[comp] for comp in self.comps]
        # Check if CompID is present
        if compID in self.CompIDs:
            # Get the component name
            return self.comps[self.CompIDs.index(compID)]
        else:
            # CompID not found
            return None
# class ConfigMIXSUR


# Alternate configuration
class ConfigJSON(object):
    r"""JSON-based surface configuration interface

    :Call:
        >>> cfg = ConfigJSON(fname="Config.json")
    :Inputs:
        *fname*: {``"Config.json"``} | :class:`str`
            Name of JSON file from which to read tree and properties
    :Outputs:
        *cfg*: :class:`cape.config.ConfigJSON`
            JSON-based configuration interface
    :Attributes:
        *cfg.faces*: :class:`dict`\ [:class:`list` | :class:`int`]
            Dict of the component ID or IDs in each named face
        *cfg.comps*: :class:`list`\ [:class:`str`]
            List of components with no children
        *cfg.parents*: :class:`dict`\ [:class:`list`\ [:class:`str`]]
            List of parent(s) by name for each component
        *cfg.IDs*: :class:`list`\ [:class:`int`]
            List of unique component ID numbers
        *cfg.name*: ``None`` | :class:`str`
            Optional string to identify what is being represented
    :Versions:
        * 2016-10-21 ``@ddalle``: Version 1.0
    """
    # Initialization method
    def __init__(self, fname="Config.json"):
        r"""Initialization method

        :Versions:
            * 2016-10-21 ``@ddalle``: Version 1.0
        """
        # Check for a file
        if fname is not None:
            # Read the settings from an expanded and decommented JSON file
            opts = util.loadJSONFile(fname)
            # Convert to special options class
            opts = util.odict(**opts)
        else:
            # Empty options
            opts = util.odict()
        # Get major sections
        self.article = None
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
        self.IDs = []
        self.faces = {}
        self.parents = {}
        self._skipped_faces = set()
        # Loop through the tree
        for c in self.tree:
            # Check if already processed
            if c in self.faces:
                continue
            # Process the first component
            self.AppendChild(c)

    # Function to display things
    def __repr__(self):
        r"""Representation method

        Template: ``<cape.ConfigJSON(nComp=N, faces=['Core', ...])>``
        """
        # Return a string.
        return '<cape.ConfigJSON(nComp=%i, faces=%s)>' % (
            len(self.faces), self.faces.keys())

    # Method to get CompIDs from generic input
    def GetCompID(self, face):
        r"""Return a list of component IDs from generic input

        :Call:
            >>> compID = cfg.GetCompID(face)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                XML surface config instance
            *face*: :class:`str` | :class:`int` | :class:`list`
                Component number, name, or list thereof
        :Outputs:
            *compID*: :class:`list`\ [:class:`int`]
                List of component IDs
        :Versions:
            * 2014-10-12 ``@ddalle``: Version 1.0 (:class:`ConfigXML`)
            * 2016-10-21 ``@ddalle``: Version 1.0
        """
        # Initialize the list.
        compID = []
        # Process the type.
        if type(face).__name__ in ['list', 'ndarray']:
            # Loop through the inputs.
            for f in face:
                # Call this function so it passes to the non-array portion.
                compID += self.GetCompID(f)
                # Sort component numbers
                compID.sort()
        elif face in self.faces:
            # Process the face
            cID = self.faces[face]
            # Check if it's a list.
            if type(cID).__name__ in ['list', 'ndarray']:
                # Add the list.
                compID += list(cID)
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

    # Get name of a compID
    def GetCompName(self, compID):
        r"""Get the name of a component by its number

        :Call:
            >>> face = cfg.GetCompName(compID)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
            *compID*: :class:`int`
                Component ID number
        :Outputs:
            *face*: ``None`` | :class:`str`
                Name of so-numbered component, if any
        :Versions:
            * 2017-03-30 ``@ddalle``: Version 1.0
        """
        # Make sure there is a list of CompIDs
        try:
            self.CompIDs
        except AttributeError:
            # Make the list
            self.CompIDs = [self.faces[comp] for comp in self.comps]
        # Check if CompID is present
        if compID in self.CompIDs:
            # Get the component name
            return self.comps[self.CompIDs.index(compID)]
        else:
            # CompID not found
            return None

    # Method to copy a configuration
    def Copy(self):
        r"""Copy a configuration interface

        :Call:
            >>> cfg2 = cfg.Copy()
        :Inputs:
            *cfg*: :class:`cape.config.ConfigXML`
                XML surface config instance
        :Outputs:
            *cfg2*: :class:`cape.config.ConfigXML`
                Copy of input
        :Versions:
            * 2014-11-24 ``@ddalle``: Version 1.0
        """
        # Initialize object.
        cfg = ConfigJSON(fname=None)
        # Copy the dictionaries.
        cfg.faces = self.faces.copy()
        cfg.props = self.props.copy()
        cfg.tree  = self.tree.copy()
        cfg.parents = self.parents.copy()
        cfg.comps = list(self.comps)
        cfg.IDs = list(self.IDs)
        # Output
        return cfg

    # Process children
    def AppendChild(self, c, parent=None):
        r"""Process one component of the tree and recurse

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
            *compID*: :class:`list`\ [:class:`int`]
                Full list of component IDs in *c* and its children
        :Versions:
            * 2016-10-21 ``@ddalle``: Version 1.0
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
                if f is None:
                    # Do not append
                    print(
                        "Skipping '%s' child comp '%s'; no CompID" % (c, child))
                    continue
                elif isinstance(f, list):
                    # Add two lists together
                    compID += f
                elif isinstance(f, int):
                    # Append singleton face
                    compID.append(f)
                else:
                    # Invalid type
                    print(
                        "Skipping '%s' child '%s'; invalid type '%s'"
                        % (c, child, type(f)))
                    continue
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
                # Check if skipped already
                if child in self._skipped_faces:
                    continue
                # Missing property
                print(
                    ("Skipping component '%s'; not a parent " % child) +
                    'and has no "CompID"')
                # Save skipped face and get out of here
                self._skipped_faces.add(child)
                continue
            elif cID in self.IDs:
                # Duplicate entry
                raise ValueError(
                    ("Face '%s' uses component ID number " % c) +
                    ("%s, which was already in use" % cID))
            # Set the component for *child*
            self.faces[child] = cID
            self.IDs.append(cID)
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
        r"""Restrict component IDs in *cfg.faces* to manual list

        :Call:
            >>> cfg.RestrictCompID(compIDs)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration interface
            *compIDs*: :class:`list`\ [:class:`int`]
                List of relevant component IDs
        :Versions:
            * 2016-11-05 ``@ddalle``: Version 1.0
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
        for face in list(self.faces.keys()):
            # Get the current parameters
            c = self.faces[face]
            t = type(c).__name__
            # Check the type
            if t.startswith('int'):
                # Check for the compID at all
                if c not in compIDs:
                    # Delete the face
                    del self.faces[face]
                    # Delete the component name
                    if face in self.comps:
                        i = self.comps.index(face)
                        del self.comps[i]
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
        r"""Get names of faces that are of type "tri" (not containers)

        :Call:
            >>> comps = cfg.GetTriFaces()
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration instance
        :Outputs:
            *comps*: :class:`list`\ [:class:`str`]
                List of lowest-level component names
        :Versions:
            * 2016-11-07 ``@ddalle``: Version 1.0
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
    def WriteXML(self, fname="Config.xml", name=None, source=None):
        r"""Write a GMP-type ``Config.xml`` file

        :Call:
            >>> cfg.WriteXML(fname="Config.xml", name=None, source=None)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration instance
            *fname*: {``"Config.xml"``} | :class:`str`
                Name of file to write
            *name*: {``None``} | :class:`str`
                Name of the configuration, defaults to *cfg.name*
            *source*: {``"Components.i.tri"``} | :class:`str`
                Name of the "source" tri file, has no effect
        :Versions:
            * 2016-11-06 ``@ddalle``: Version 1.0
        """
        # Open the file.
        f = open(fname, 'w')
        # Write opening handle
        f.write('<?xml version="1.0" encoding="utf-8"?>\n\n')
        # Get the name and source
        if name is None:
            name = self.name
        if source is None:
            source = "Components.i.tri"
        # Write the "configuration" element
        cname = os.path.basepath(fname)
        f.write('<Configuration Name="%s" Source="%s">\n\n' % (cname, source))
        # Get sorted faces
        faces = self.SortCompIDs()
        # Loop through the elements
        for face in faces:
            # Check if not present for this config *name*
            if not self.GetProperty(face, "Present", name=name, vdef=True):
                continue
            # Get the compID
            compID = self.faces.get(face)
            # Don't mess around with ``None``
            if compID is None:
                continue
            # Check if it's a basic face or a container
            if isinstance(compID, int):
                # Integers are already faces
                q = True
                # Check for valid face
                if compID < 0: continue
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
                    # Check validity
                    if compID < 0: continue
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
                    # Get primary parent
                    ppar = self.GetProperty(face, 'Parent')
                    # Check for primary parent.
                    if ppar is None:
                        # Let's warn for now, verbose
                        print(
                            ("  WARNING: Comp '%s' has multiple " % face) +
                            ("parents (%s); using first entry" % parent))
                        parent = parent[0]
                    else:
                        # Use the "Parent" from the "Properties" section
                        parent = ppar
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

    # Write .aflr3bc file
    def WriteAFLR3BC(self, fname):
        r"""Write a file that list AFLR3 boundary conditions for components

        :Call:
            >>> cfg.WriteAFLR3BC(fname)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration instance
            *fname*: :class:`str`
                Name of AFLR3 boundary condition file to write
        :Versions:
            * 2017-05-05 ``@ddalle``: Version 1.0
        """
        # Get maximum length of a component for nice formatting
        L = max([len(face) for face in self.faces])
        # Create format string
        fmt = "%%-%is" % L
        # Open the file
        f = open(fname, 'w')
        # Loop through faces
        for face in self.props:
            # Get properties
            prop = self.props[face]
            # Go to next face if not a dictionary of properties
            if prop.__class__.__name__ != "dict": continue
            # Check if "aflr3_bc" is present
            bc = prop.get("aflr3_bc", prop.get("bc"))
            # Skip if not present
            if bc is None: continue
            # If we reach this point, write the component name and the BC
            f.write(fmt % face)
            f.write("%3i" % bc)
            # Get initial spacing and BL growth distance
            blds = prop.get("blds")
            bldel = prop.get("bldel")
            # Skip if no *blds* value
            if blds is None:
                # Give a warning if -1
                if bc == -1:
                    print("  Warning: component '%s' has BC of -1 but no blds"
                        % face)
            else:
                # Write the initial spacing
                f.write("  %-6s" % blds)
            # Check for the distance
            if bldel is not None:
                f.write("  %-6s" % bldel)
            # Go to next line
            f.write("\n")
        # Close the file
        f.close()

    # Write .mapbc file
    def WriteFun3DMapBC(self, fname):
        r"""Write a Fun3D ".mapbc" file

        :Call:
            >>> cfg.WriteFun3DMapBC(fname)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration instance
            *fname*: :class:`str`
                Name of mapbc file to write
        :Versions:
            * 2016-11-07 ``@ddalle``: Version 1.0
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
            # Set the component ID
            compID = self.GetPropCompID(face)
            # Check for negative component
            if compID < 0: continue
            # Add the component
            comps0.append(face)
            bcs[face] = fun3dbc
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
        r"""Renumber the component ID number

        This affects *cfg.faces* for *face* and each of its parents, and
        it also resets the component ID number in *cfg.props*.

        :Call:
            >>> cfg.RenumberCompID(face, compID)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration
            *face*: :class:`str`
                Name of component to rename
        :Versions:
            * 2016-11-09 ``@ddalle``: Version 1.0
        """
        # Get the current component number
        compi = self.faces[face]
        # Reset it
        if isinstance(compi, (list, np.ndarray)):
            # Extract the original component ID from singleton list
            compi = compi[0]
            # Reset it as a face (from list)
            self.faces[face] = compID
            # Add to list of components
            if face not in self.comps:
                self.comps.append(face)
        else:
            # Reset it (number)
            self.faces[face] = compID
        # Get the component ID from "Properties"
        compp = self.props[face]
        # Check for single number
        if isinstance(compp, dict):
            # Set the CompID property
            self.props[face]["CompID"] = compID
        else:
            # Single number
            self.props[face] = compID
        # Loop through the parents
        self.RenumberCompIDParent(face, compi, compID)

    # Renumber the parents of one component.
    def RenumberCompIDParent(self, face, compi, compo):
        r"""Recursively renumber the parents of *face*

        :Call:
            >>> cfg.RenumberCompIDParent(face, compi, compo)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration
            *face*: :class:`str`
                Name of component whose parents should be renumbered
            *compi*: :class:`int`
                Incoming component ID number
            *compo*: :class:`int`
                Outgoing component ID number
        :Versions:
            * 2016-11-09 ``@ddalle``: Version 1.0
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
        r"""Renumber component IDs 1 to *n*

        :Call:
            >>> comps = cfg.ResetCompIDs()
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration instance
        :Versions:
            * 2016-11-09 ``@ddalle``: Version 1.0
        """
        # Get the list of tri faces
        comps = self.SortCompIDs()
        # Loop through faces in order
        for i in range(len(comps)):
            # Get the face
            face = comps[i]
            # Renumber
            self.RenumberCompID(face, i+1)

    # Renumber Component IDs 1 to *n*
    def SortCompIDs(self):
        r"""Get ordered list of components

        :Call:
            >>> comps = cfg.SortCompIDs()
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration instance
        :Outputs:
            *comps*: :class:`list`\ [:class:`str`]
                List of components
        :Versions:
            * 2016-11-09 ``@ddalle``: Version 1.0
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
        r"""Get a *CompID* from the "Properties" section

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
            * 2016-10-21 ``@ddalle``: Version 1.0
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
    def GetProperty(self, comp, k, name=None, vdef=None):
        r"""Get a cascading property from a component or its parents

        :Call:
            >>> v = cfg.GetProperty(comp, k, name=None, vdef=None)
        :Inputs:
            *cfg*: :class:`cape.config.ConfigJSON`
                JSON-based configuration interface
            *comp*: :class:`str`
                Name of component to query
            *k*: :class:`str`
                Name of property to query
            *name*: {``None``} | :class:`str`
                Name to filter if *k* has multiple values; defaults to
                *cfg.name* if applicable
            *vdef*: {``None``} | **any**
        :Outputs:
            *v*: *vdef* | **any**
                Value of *k* from *comp* with fallback to parents
        :Versions:
            * 2016-10-21 ``@ddalle``: Version 1.0
            * 2022-03-15 ``@ddalle``: Version 2.0; add *name*
            * 2022-04-14 ``@ddalle``: Version 2.1; add *vdef*
        """
        # Check for the property
        v, q = self._get_property(comp, k, name)
        # Check for the property
        if q:
            return v
        # Loop through parents until one is reached
        for parent in self.parents[comp]:
            # Get the value from that parent (note: this may recurse)
            v = self.GetProperty(parent, k, name)
            # Check for success (otherwise try next parent if there is one)
            if v is not None:
                return v
        # If this point is reached, could not find property in any parent
        return vdef

    def _get_property(self, comp, k, name=None):
        # Get component properties
        opts = self.props.get(comp, {})
        # Check type
        if opts is None:
            # Default
            opts = {}
        elif not isinstance(opts, dict):
            # Process a single option for *CompID*
            opts = {"CompID": opts}
        # Check for the property
        if k not in opts:
            # Not present
            return None, False
        # Get value
        val = opts[k]
        # Check if a dictionary
        if not isinstance(val, dict):
            # Return singleton value
            return val, True
        # Default *name*
        if name is None:
            name = self.name
        # If no *name*, return the whole dict
        if name is None:
            return val, True
        # Filter it
        for namej, vj in val.items():
            # Check it
            if re.match(namej, name):
                return vj, True

