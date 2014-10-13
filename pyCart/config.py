"""
Module to interface with configuration files: :mod:`pyCart.Config`
==================================================================

This is a module to interact with :file:`Config.xml` files.
"""

# File checker.
import os
# Import xml parser
import xml.etree.ElementTree as ET

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
            self.comp = {}
            return
        # Read the XML file.
        e = ET.parse(fname)
        # Get the list of components.
        Comps = e.findall('Component')
        # Initialize containers and individual tris
        comp = {}
        # Loop through components to get containers.
        for c in Comps:
            # Check the type.
            if c.attrib.get('Type') != 'container':
                continue
            # Get the name.
            compName = c.attrib.get('Name')
            # Check for error.
            if compName is None:
                raise IOError("At least one component in "
                    + "'%s' is lacking a name." % fname)
            # That's all; initialize an empty list of components.
            comp[compName] = []
        # Loop through points to get the labeled faces.
        for c in Comps:
            # Check the type.
            if c.attrib.get('Type') != 'tri':
                continue
            # Get the name.
            compName = c.attrib.get('Name')
            # Check for error.
            if compName is None:
                raise IOError("At least one component in "
                    + "'%s' is lacking a name." % fname)
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
            # Get the parent name, if any.
            parent = c.get("Parent")
            # Check for a parent and one that's in the current list.
            if parent in comp:
                # Add this face label to the container list.
                comp[parent].append(compID)
        # Save the individually labeled faces.
        self.faces = comp
        
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
        elif type(face).__name__ == 'str':
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
            # Just append it.
            compID.append(face)
        # Output
        return compID
    
