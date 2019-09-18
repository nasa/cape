"""
:mod:`cape.pyfun.faux`: FAUXGeom interface module
==================================================

This is a module for interacting with FUN3D input files that define geometry
for adaptation.  Files defined in the FAUXGeom file ``faux_input`` can have
their surface meshes refined while other surfaces must be frozen.

:See also:
    * :func:`cape.pyfun.cntl.Cntl.ReadFAUXGeom`
    * :func:`cape.pyfun.cntl.Cntl.PrepareFAUXGeom`
    * :func:`cape.pyfun.cntl.Cntl.ReadFreezeSurfs`
    * :func:`cape.pyfun.cntl.Cntl.PrepareFreezeSurfs`
"""

# System modules
import os.path

# Base this class off of the main file control class.
class FAUXGeom(object):
    """File control class for :file:`faux_input`
    
    :Call:
        >>> faux = pyFun.FAUXGeom()
    :Inputs:
        *fname*: :class:`str`
            Name of ``faux_input`` file or template
    :Outputs:
        *faux*: :class:`pyFun.faux.FAUXGeom`
            Interface for ``faux_input`` file
        *faux.nSurf*: :class:`int`
            Number of MapBC surfaces with geometry descriptions
        *faux.Surfs*: :class:`list` (:class:`int`)
            List of surface indices
        *faux.Geom*: :class:`dict` (:class:`float` | :class:`list`)
            Dictionary of geometry definitions
    :Versions:
        * 2017-02-23 ``@ddalle``: First version
    """
    
    # Initialization method (not based off of FileCntl)
    def __init__(self, fname="faux_input"):
        """Initialization method
        
        :Versions:
            * 2017-02-23 ``@ddalle``: First version
        """
        # Save the file name.
        self.fname = fname
        # Initialize
        self.Surfs = []
        self.Geom = {}
        self.nSurf = 0
        # Read the file
        if (fname is not None) and os.path.isfile(fname):
            self.Read(fname)
            
    # String representation
    def __repr__(self):
        """String representation
        
        :Versions:
            * 2017-04-08 ``@ddalle``: First version
        """
        return "<FAUXGeom fname='%s', nSurf=%s>" % (self.fname, self.nSurf)
        
        
    # Read a ``faux_input`` file or template
    def Read(self, fname):
        """Read a ``faux_input`` input file or template
        
        :Call:
            >>> faux.Read(fname)
        :Inputs:
            *fname*: :class:`str`
                Name of ``faux_input`` file or template
        :Versions:
            * 2017-02-23 ``@ddalle``: First version
        """
        # Open the file
        lines = open(fname).readlines()
        # Initialize
        self.Surfs = []
        self.Geom = {}
        # Loop through lines
        i = 1
        nline = len(lines)
        while i < nline:
            # Get the values
            V = self.ConvertToVal(lines[i])
            # Check contents
            if len(V) < 3:
                raise ValueError(
                    ("Failure reading FAUXGeom line:\n%s\n" % lines[i]) +
                    ("Must be surface ID (int), geom type (str), and coord"))
            # Get the type
            typ = V[1]
            # Check it
            if typ in ["xplane", "yplane", "zplane"]:
                # Valid x-plane, y-plane, or z-plane
                self.Geom[V[0]] = {V[1]: V[2]}
            elif typ in ["general_plane"]:
                # Read the next line, which is the normal
                i += 1
                n = self.ConvertToVal(lines[i])
                # Valid general plane
                self.Geom[V[0]] = {V[1]: V[2], "normal": n}
            # Check if surface already counted
            if V[0] not in self.Surfs:
                # Increase surface count
                self.nSurf += 1
                # Add to the list
                self.Surfs.append(V[0])
            # Move to next line
            i += 1
        # Sort surface list
        self.Surfs.sort()
        
    # Convert a string to a value
    def ConvertToVal(self, val):
        """Convert a text description to a Python value
        
        :Call:
            >>> v = faux.ConvertToVal(val)
        :Inputs:
            *faux*: :class:`pyFun.faux.FAUXGeom`
                Interface for ``faux_input`` file
            *val*: :class:`str` | :class:`unicode`
                Text of the value from file
        :Outputs:
            *v*: :class:`str` | :class:`int` | :class:`float` | :class:`list`
                Evaluated value of the text
        :Versions:
            * 2017-02-23 ``@ddalle``: First version
        """
        # Check inputs
        if type(val).__name__ not in ['str', 'unicode']:
            # Not text; return as is
            return val
        # Initialize output
        V = []
        # Loop through parts
        for vi in val.split():
            # Attempt to interpret as an integer
            try:
                V.append(int(vi))
                continue
            except Exception:
                pass
            # Attempt to interpret as a float
            try:
                V.append(float(vi))
            except Exception:
                # Just keep as a string
                V.append(vi)
        # Output (list if needed)
        if len(V) == 1:
            # Single output
            return V[0]
        else:
            # Return list
            return V
        
    # Set value for a plane
    def SetGeom(self, comp, geom):
        """Set geometry definition for a component
        
        :Call:
            >>> faux.SetGeom(comp, geom)
        :Inputs:
            *faux*: :class:`pyFun.faux.FAUXGeom`
                Interface for ``faux_input`` file
            *comp*: :class:`int`
                Component index
            *geom*: :class:`dict`
                Geometry description
        :Versions:
            * 2017-02-23 ``@ddalle``: First version
        """
        # Check if component already defined.
        if comp not in self.Surfs:
            # Append the component
            self.Surfs.append(comp)
            # Sort again
            self.Surfs.sort()
            # Increase count
            self.nSurf += 1
        # Save the instruction
        self.Geom[comp] = geom
        
    # Write a ``faux_input`` file
    def Write(self, fname="faux_input"):
        """Write FAUXGeom input file
        
        :Call:
            >>> faux.Write(fname="faux_input")
        :Inputs:
            *faux*: :class:`pyFun.faux.FAUXGeom`
                Interface for ``faux_input`` file
            *fname*: {``"faux_input"``} | :class:`str`
                Name of file to write
        :Versions:
            * 2017-02-23 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'w')
        # Write the number of surfaces.
        f.write('%i\n' % self.nSurf)
        # Loop through instructions
        for comp in self.Surfs:
            # Get the instruction
            geom = self.Geom[comp]
            # Check the type
            if 'general_plane' in geom:
                # Write a generic plane
                f.write(" %i general_plane %s" % (comp,geom["general_plane"]))
                # Write the normal
                f.write("  ")
                f.write(" ".join([str(v) for v in geom["normal"]]))
                f.write("\n")
            elif 'xplane' in geom:
                # Write the xplane
                f.write(" %i xplane %s\n" % (comp, geom["xplane"]))
            elif 'yplane' in geom:
                # Write the xplane
                f.write(" %i yplane %s\n" % (comp, geom["yplane"]))
            elif 'zplane' in geom:
                # Write the xplane
                f.write(" %i zplane %s\n" % (comp, geom["zplane"]))
        # Close the file
        f.close()
        
