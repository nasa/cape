"""
Module to interface with configuration files: :mod:`pyCart.Config`
==================================================================

This is a module to interact with :file:`Config.xml` files.
"""

# File checker.
import os
# Import xml parser
import xml.etree.ElementTree as ET
# Process unique lists.
from numpy import unique


# Function to recursively append components to parents and their parents
def AppendParent(Comps, comp, k, compID):
    """Append a component ID to a parent container and its parents
    
    :Call:
        >>> comp = AppendParent(Comps, comp, k, compID)
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
        * 2014.10.13 ``@ddalle``: First version
    """
    # Get the component.
    c = Comps[k]
    # Check for a parent.
    parent = c.get("Parent")
    # Get the names.
    Names = [c.get('Name') for c in Comps]
    # Check if that's a recognized component.
    if parent in comp:
        # Add this face label to the container list.
        comp[parent].append(compID)
        # Eliminate doubles.
        comp[parent] = list(unique(comp[parent]))
        # Get the parent tag.
        k0 = Names.index(parent)
        # Append *compID* to parent's parent, if any
        comp = AppendParent(Comps, comp, k0, compID)
    # Output
    return comp

# Configuration class
class Config:
    """Configuration class for interfacing :file:`Config.xml` files
    
    :Call:
        >>> cfg = pyCart.Config(fname='Config.xml')
    :Inputs:
        *fname*: :class:`str`
            Name of configuration file to read
    :Outputs:
        *cfg*: :class:`pyCart.config.Config`
            Instance of configuration class
    :Versions:
        * 2014.10.12 ``@ddalle``: First version
    """
    
    # Initialization method
    def __init__(self, fname="Config.xml"):
        """Initialization method"""
        # Check for the file.
        if not os.path.isfile(fname):
            # Save an empty component dictionary.
            self.faces = {}
            return
        # Read the XML file.
        e = ET.parse(fname)
        # Get the list of components.
        Comps = e.findall('Component')
        # Get the names.
        Names = [c.get('Name') for c in Comps]
        # Check for unnamed component.
        if None in Names:
            raise IOError("At least one component in "
                + "'%s' is lacking a name." % fname)
        # Initialize containers and individual tris
        comp = {}
        # Loop through components to get containers.
        for c in Comps:
            # Check the type.
            if c.attrib.get('Type') != 'container':
                continue
            # Get the name.
            compName = c.attrib.get('Name')
            # That's all; initialize an empty list of components.
            comp[compName] = []
        # Loop through points to get the labeled faces.
        for k in range(len(Comps)):
            # Extract key.
            c = Comps[k]
            # Check the type.
            if c.attrib.get('Type') != 'tri':
                continue
            # Get the name.
            compName = c.attrib.get('Name')
            # Try to read the CompID
            try:
                # Get the text of the 'Data' element.
                txt = c.getchildren()[0].text
                # Process the integer.
                compID = int(txt.split('=')[-1])
            except Exception:
                raise IOError("Could not process Face Label "
                    + "for component '%s'." % compName)
            # Save the component.
            comp[compName] = compID
            # Process any parents.
            comp = AppendParent(Comps, comp, k, compID)
        # Save the individually labeled faces.
        self.faces = comp
    
    # Function to display things
    def __repr__(self):
        """
        Return the string representation of a :file:`Config.xml` file.
        
        This looks like ``<pyCart.Config(nComp=N, faces=['Core', ...])>``
        """
        # Return a string.
        return '<pyCart.Config(nComp=%i, faces=%s)>' % (
            len(self.faces), self.faces.keys())
        
    # Method to get CompIDs from generic input
    def GetCompID(self, face):
        """Return a list of component IDs from generic input
        
        :Call:
            >>> compID = cfg.GetCompID(face)
        :Inputs:
            *face*: :class:`str` or :class:`int` or :class:`list`
                Component number, name, or list of component numbers and names
        :Outputs:
            *compID*: :class:`list`(:class:`int`)
                List of component IDs
        :Versions:
            * 2014.10.12 ``@ddalle``: First version
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
    
