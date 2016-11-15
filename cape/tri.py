"""
Surface triangulation module: :mod:`cape.tri`
=============================================

This module provides the utilities for interacting with Cart3D or Plot3D type
triangulations, including annotated triangulations (including ``.triq`` files).
Triangulations can also be read from the UH3D, UNV, and AFLR3 surf formats.

The module consists of individual classes that are built off of a base
triangulation class :class:`cape.tri.TriBase`.  Methods that are written for
the TriBase class apply to all other classes as well.

Some triangulation methods are written in Python/C using the :mod:`cape._cape`
module.  For some repeated tasks (especially writing triangulations to file),
creating a compiled version can lead to significant time savings.  These are
relatively simple to compile, but fall-back methods are provided using purely
Python code in each case.  The convention used for this situation is to provide
a method like :func:`cape.tri.TriBase.WriteFast` for the compiled version and
:func:`cape.tri.TriBase.WriteSlow` for the Python version.
"""

# Required modules
# Numerics
import numpy as np
# File system and operating system management
import os, shutil
import subprocess as sp
# Specific commands to copy files and call commands.
from shutil import copy
# Ordered dictionaries
from collections import OrderedDict

# Utilities
from .util import GetTecplotCommand, TecFolder, ParaviewFolder
from .config import Config, ConfigJSON

# Input/output module
from . import io
from . import geom

# Attempt to load the compiled helper module.
try:
    from . import _cape as pc
except ImportError:
    pass

# Function to get a non comment line
def _readline(f, comment='#'):
    """Read line that is nonempty and not a comment
    
    :Call:
        >>> line = _readline(f, comment='#')
    :Inputs:
        *f*: :class:`file`
            File instance
        *comment*: :class:`str`
            Character(s) that begins a comment
    :Outputs:
        *line*: :class:`str`
            Nontrivial line or `''` if at end of file
    :Versions:
        * 2015-11-19 ``@ddalle``: First version
    """
    # Read a line.
    line = f.readline()
    # Check for empty line (EOF)
    if line == '': return line
    # Process stripped line
    lstrp = line.strip()
    # Check if otherwise empty or a comment
    while (lstrp=='') or lstrp.startswith(comment):
        # Read the next line.
        line = f.readline()
        # Check for empty line (EOF)
        if line == '': return line
        # Process stripped line
        lstrp = line.strip()
    # Return the line.
    return line
# end _readline

# Function to read a single triangulation file
def ReadTriFile(fname, fmt=None):
    """Read a single triangulation file
    
    :Call:
        >>> tri = ReadTriFile(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of Cart3D tri, IDEAS unv, UH3D, or AFLR3 surf file
        *fmt*: {``None``} | ``"tri"`` | ``"uh3d"`` | ``"unv"`` | ``"surf"``
            Format to use; by default determine from the file extension
    :Outputs:
        *tri*: :class:`cape.tri.Tri`
            Triangulation
    :Versions:
        * 2016-04-06 ``@ddalle``: First version
    """
    # Split based on '.'
    fext = fname.split('.')
    # Get the extension
    if len(fext) < 2:
        # Odd case, no extension given
        fext = 'tri'
    else:
        # Get the extension
        fext = fext[-1]
    # Determine whether or not to use *fmt* override of format
    if fmt is None: fmt = fext.lower()
    # Read using the appropriate format
    if fmt == 'surf':
        # AFLR3 surface file
        return Tri(surf=fname)
    elif fmt == 'uh3d':
        # UH3D surface file
        return Tri(uh3d=fname)
    elif fmt == 'unv':
        # Weird IDEAS triangulation thing
        return Tri(unv=fname)
    elif fmt == 'triq':
        # Read triq file
        return Triq(fname)
    else:
        # Assume Cart3D triangulation file
        return Tri(fname)
# end ReadTriFile


# Triangulation class
class TriBase(object):
    """Cape base triangulation class
    
    This class provides an interface for a basic triangulation without
    surface data.  It can be created either by reading an ASCII file or
    specifying the data directly.
    
    When no component numbers are specified, the object created will label
    all triangles ``1``.
    
    :Call:
        >>> tri = cape.tri.TriBase(fname=fname, c=None)
        >>> tri = cape.tri.TriBase(uh3d=uh3d, c=None)
        >>> tri = cape.tri.TriBase(Nodes=Nodes, Tris=Tris, CompID=CompID)
    :Inputs:
        *fname*: :class:`str`
            Name of triangulation file to read (format based on extension)
        *tri*: :class:`str`
            Name of triangulation file (Cart3D TRI format)
        *uh3d*: :class:`str`
            Name of triangulation file (UH3D format)
        *c*: :class:`str`
            Name of configuration file (e.g. ``Config.xml``)
        *nNode*: :class:`int`
            Number of nodes in triangulation
        *Nodes*: :class:`np.ndarray` (:class:`float`), (*nNode*, 3)
            Matrix of *x,y,z*-coordinates of each node
        *nTri*: :class:`int`
            Number of triangles in triangulation
        *Tris*: :class:`np.ndarray` (:class:`int`), (*nTri*, 3)
            Indices of triangle vertex nodes
        *CompID*: :class:`np.ndarray` (:class:`int`), (*nTri*)
            Component number for each triangle
    :Data members:
        *tri.nNode*: :class:`int`
            Number of nodes in triangulation
        *tri.Nodes*: :class:`np.ndarray` (:class:`float`), (*nNode*, 3)
            Matrix of *x,y,z*-coordinates of each node
        *tri.nTri*: :class:`int`
            Number of triangles in triangulation
        *tri.Tris*: :class:`np.ndarray` (:class:`int`), (*nTri*, 3)
            Indices of triangle vertex nodes
        *tri.CompID*: :class:`np.ndarray` (:class:`int`), (*nTri*)
            Component number for each triangle
    :Versions:
        * 2014-05-23 ``@ddalle``: First version
        * 2014-06-02 ``@ddalle``: Added UH3D reading capability
        * 2015-11-19 ``@ddalle``: Added AFLR3 surface capability
    """
    # ======
    # Config
    # ======
  # <
    # Initialization method
    def __init__(self, fname=None, c=None,
        tri=None, uh3d=None, surf=None, unv=None,
        xml=None, json=None,
        nNode=None, Nodes=None, nTri=None, Tris=None,
        nQuad=None, Quads=None, CompID=None):
        """Initialization method"""
        # Versions:
        #  2014-05-23 @ddalle: First version
        #  2014-06-02 @ddalle: Added UH3D reading capability
        #  2015-11-19 @ddalle: Added XML reading and AFLR3 surfs
        
        # Check if file is specified.
        if tri is not None:
            # Read from file.
            self.Read(fname)
        elif uh3d is not None:
            # Read from the other format.
            self.ReadUH3D(uh3d)
        elif unv is not None:
            # Read bogus format
            self.ReadUnv(unv)
        elif surf is not None:
            # Read AFLR3 surface format
            self.ReadSurf(surf)
        elif fname is not None:
            # Guess type from file extensions
            self.ReadBest(fname)
            
        else:
            # Process inputs.
            # Check counts.
            if nNode is None:
                # Get dimensions if possible.
                if Nodes is not None:
                    # Use the shape.
                    nNode = Nodes.shape[0]
                else:
                    # No nodes
                    nNode = 0
            # Check counts.
            if nTri is None:
                # Get dimensions if possible.
                if Tris is not None:
                    # Use the shape.
                    nTri = Tris.shape[0]
                else:
                    # No nodes
                    nTri = 0
            # Check counts.
            if nQuad is None:
                # Get dimensions if possible.
                if Quads is not None:
                    # Use the shape.
                    nQuad = Quads.shape[0]
                else:
                    # No nodes
                    nQuad = 0
            # Save the components.
            self.nNode = nNode
            self.Nodes = Nodes
            self.nTri = nTri
            self.Tris = Tris
            self.nQuad = nQuad
            self.Quads = Quad
            self.CompID = CompID
        
        # Check for configuration
        if xml is not None:
            # Read a Config.xml type of file
            self.ReadConfigXML(c)
        elif json is not None:
            # Read a Config.json type of file
            self.ReadConfigJSON(c)
        elif c is not None:
            # Read the configuration
            self.ReadConfig(c)
        # Check if we should apply it
        try:
            # Check for two opinions about how tris should be numbered
            self.Conf
            self.config
            # Use the explicity one
            self.ApplyConfig(self.config)
        except AttributeError:
            pass
        
    # Method that shows the representation of a triangulation
    def __repr__(self):
        """Return the string representation of a triangulation.
        
        This looks like ``<cape.tri.Tri(nNode=M, nTri=N)>``
        
        :Versions:
            * 2014-05-27 ``@ddalle``: First version
        """
        return '<cape.tri.Tri(nNode=%i, nTri=%i)>' % (self.nNode, self.nTri)
        
    # String representation is the same
    __str__ = __repr__
    
    # Function to read using the best guess at format
    def ReadBest(self, fname):
        """Read a file using the extension to guess format
        
        :Call:
            >>> tri.ReadBest(fname)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
                Triangulation or unstructured surface mesh interface
            *fname*: :class:`str`
                Name of file, use the extension to guess format
        :Versions:
            * 2016-10-21 ``@ddalle``: First version
        """ 
        # Split based on '.'
        fext = fname.split('.')
        # Get the extension
        if len(fext) < 2:
            # Odd case, no extension given
            fext = 'tri'
        else:
            # Get the extension
            fext = fext[-1].lower()
        # Guess the format
        if fext == 'surf':
            # AFLR3 surface file
            self.ReadSurf(fname)
        elif fext == 'uh3d':
            # UH3D surface file
            self.ReadUH3D(fname)
        elif fext == 'unv':
            # Weird IDEAS triangulation thing
            self.ReadUnv(fname)
        elif fext == 'triq':
            # Read triq file
            self.ReadTriQ(fname)
        else:
            # Assume Cart3D triangulation file
            self.Read(fname)
        
        
    # Function to copy a triangulation and unlink it.
    def Copy(self):
        """Copy a triangulation and unlink it
        
        :Call:
            >>> tri2 = tri.Copy()
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
        :Outputs:
            *tri2*: :class:`cape.tri.Tri`
                Triangulation with same values as *tri* but not linked
        :Versions:
            * 2014-06-12 ``@ddalle``: First version
        """
        # Make a new triangulation with no information.
        typ = type(self).__name__
        # Initialize the correct type.
        if typ == 'Triq':
            # Initialize with state
            tri = Triq()
        elif type == 'TriBase':
            # Initialize base object
            tri = TriBase()
        else:
            # Default to surface geometry definition
            tri = Tri()
        # Copy over the scalars.
        tri.nNode = self.nNode
        tri.nTri  = self.nTri
        # Make new copies of the arrays.
        tri.Nodes  = self.Nodes.copy()
        tri.Tris   = self.Tris.copy()
        tri.CompID = self.CompID.copy()
        # Copy BL parameters
        try:
            tri.blds = self.blds.copy()
            tri.bldel = self.bldel.copy()
        except Exception:
            pass
        # Other Tri info
        try:
            tri.BCs = self.BCs.copy()
        except Exception:
            pass
        # Copy Quad info
        try:
            tri.nQuad = self.nQuad
            tri.Quads = self.Quads.copy()
            tri.CompIDQuad = self.CompIDQuad.copy()
            tri.BCsQuad = self.BCsQuad.copy()
        except Exception:
            pass
        # Try to copy the configuration list.
        try: tri.Conf = self.Conf.copy()
        except Exception: pass
        # Try to copy the configuration.
        try: tri.config = self.config.Copy()
        except Exception: pass
        # Try to copy the original barriers.
        try: tri.iTri = self.iTri
        except Exception: pass
        # Try to copy the state
        try:
            tri.q = self.q.copy()
            tri.nq = tri.shape[1]
        except Exception:
            pass
        # Try to copy the state length
        try:
            tri.n = self.n
        except Exception:
            tri.n = 1
        # Output the new triangulation.
        return tri
    
  # >
    
    # ===========
    # TRI Readers
    # ===========
  # <
  
    # ++++++++++++++++
    # Full TRI Readers
    # ++++++++++++++++
   # {
    # Function to read a .tri file
    def Read(self, fname, n=1):
        """Read a triangulation file (from ``.tri`` or ``.triq`` file)
        
        File type is automatically detected and may be any one of the following
        
            * ASCII
            * Double-precision little-endian Fortran unformatted
            * Single-precision little-endian Fortran unformatted
            * Double-precision big-endian Fortran unformatted
            * Single-precision big-endian Fortran unformatted
        
        :Call:
            >>> tri.Read(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: :class:`str`
                Name of triangulation file to read
        :Versions:
            * 2014-06-02 ``@ddalle``: First version
        """
        # Get the file type
        self.GetTriFileType(fname)
        # Check if ASCII
        if self.filetype == 'ascii':
            # Read the ASCII file
            self.ReadASCII(fname, n=n)
        else:
            # Read the binary file
            self.ReadTriBin(fname)
            # Save number of iterations included in average
            self.n = n
        
    # Function to read a .tri file
    def ReadASCII(self, fname, n=1):
        """Read a triangulation file from an ASCII file
        
        :Call:
            >>> tri.ReadASCII(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: :class:`str`
                Name of triangulation file to read
            *n*: {``1``} | :class:`int` > 0
                Number of iterations included in average (for ``triq`` files)
        :Versions:
            * 2014-06-02 ``@ddalle``: First version
            * 2016-08-18 ``@ddalle``: Moved from :func:`Read` to enable binary
        """
        # Open the file
        fid = open(fname, 'r')
        # Read the first line.
        line = fid.readline().strip()
        V = line.split()
        # Process the line into two integers.
        nNode, nTri = (int(v) for v in V[0:2])
        # Check for number of states
        if len(V) == 3:
            nq = int(V[2])
        else:
            nq = 0
        
        # Read the nodes.
        self.ReadNodes(fid, nNode)
        # Read the Tris.
        self.ReadTris(fid, nTri)
        # Read or assign component IDs.
        self.ReadCompID(fid)
        # Read the sate.
        self.ReadQ(fid, nNode, nq)
        
        # Close the file.
        fid.close()
        
        # No quads
        self.nQuad = 0
        self.Quads = np.zeros((0,4))
        
        # Save extension
        self.ext = 'ascii'
        
        # Weight: number of files included in file
        self.n = n
            
    # Read TRI file as a binary file
    def ReadTriBin(self, fname, ni=4, nf=4):
        """Read binary unformatted triangulation file
        
        :Call:
            >>> tri.ReadTriBin(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-08-18 ``@ddalle``: First version
        """
        # Open the file for binary reading
        fid = open(fname, 'rb')
        # Get the byte order and precision
        try:
            bo = self.byteorder
            ni = self.bytecount
        except AttributeError:
            bo = os.sys.byteorder
            ni = 4
        # Read flags
        if bo == 'big':
            # Big-endian integer and float
            fi = '>i%i' % ni
            ff = '>f%i' % ni
        else:
            # Little-endian integer and float
            fi = '<i%i' % ni
            ff = '<f%i' % ni
        # Read the first record marker
        R = np.fromfile(fid, count=1, dtype=fi)
        # Number of integers in first line
        r = R[0] / ni
        # Read header line; also read 
        H = np.fromfile(fid, count=r+2, dtype=fi)
        # Set number of nodes and tris
        self.nNode = H[0]
        self.nTri = H[1]
        # Set *nq* if possible
        if r > 2:
            # Some state variables are included
            self.nq = H[2]
        else:
            # No state variables
            self.nq = 0
        # Number of nodes calculated from nodal coord block record marker
        nNode2 = H[-1] / ni / 3
        # Check for doubles
        if nNode2 / self.nNode == 2:
            # Boost to double-precision floats
            nf = 2*ni
            ff[-1] = str(nf)
        elif nNode2 == self.nNode:
            # Number of bytes per float equals that of ints
            nf = ni
        else:
            # Inconsistent
            raise ValueError(
                "Expecting %i nodes but %i indicated by record marker" %
                (self.nNode, nNode2))
        # Read the nodal coordinates
        P = np.fromfile(fid, count=3*self.nNode, dtype=ff)
        # Reshape to nNode x 3 matrix
        self.Nodes = P.reshape((self.nNode, 3))
        # Read end-of-record for nodes and start-of-record for tris
        R = np.fromfile(fid, count=2, dtype=fi)
        # Number of tris reported by record marker
        nTri2 = R[1] / ni / 3
        # Check consistency
        if nTri2 != self.nTri:
            # Inconsistent
            raise ValueError(
                "Expecting %i tris but %i indicated by record marker" %
                (self.nTri, nTri2))
        # Read the node indices for each tri
        T = np.fromfile(fid, count=3*self.nTri, dtype=fi)
        # Reshape to nTri x 3
        self.Tris = T.reshape((self.nTri, 3))
        # End-of-record
        R = np.fromfile(fid, count=1, dtype=fi)
        # Start-of-record for compIDs
        R = np.fromfile(fid, count=1, dtype=fi)
        # Check for end of file
        if R.size == 0:
            # Quit
            fid.close()
            return
        # Read the Component IDs
        self.CompID = np.fromfile(fid, count=self.nTri, dtype=fi)
        # End-of-record
        R = np.fromfile(fid, count=1, dtype=fi)
        # Start-of-record for states
        R = np.fromfile(fid, count=1, dtype=fi)
        # Check end-of-file and then consistency
        if R.size == 0:
            # Quit
            fid.close()
            return
        elif R[0] != self.nNode*nf*self.nq:
            # Inconsistent
            raise ValueError(
                "Expecting %i states but %s indicated by record marker" %
                (self.nNode, float(R[0])/nf/self.nq))
        # Read the states
        self.q = np.fromfile(fid, count=self.nNode*self.nq, dtype=ff)
        # Close the file
        fid.close()
        # Count (used for averaging triq files)
        self.n = 1
   # }
   
    # +++++++++++++
    # TRI File Type
    # +++++++++++++
   # {
    # Get byte order
    def GetTriFileType(self, fname):
        """Get the byte order and precision for a TRI file
        
        The function works by setting attributes of the triangulation
        
        :Call:
            >>> tri.GetTriFileType(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: :class:`str`
                Name of file to write
        :Attributes:
            *tri.filetype*: {``'ascii'``} | ``'binary'``
                File type
            *tri.byteorder*: ``'big'`` | ``'little'``
                Endianness
            *tri.precision*: ``4`` | ``8``
                Number of bytes in one entry
        :Versions:
            * 2016-08-18 ``@ddalle``: First version
        """
        # Open the file; attempt a binary read
        fid = open(fname, 'rb')
        # Read the first entry as a double-precision little-endian int
        r = np.fromfile(fid, count=1, dtype='<i4')
        # Check for success
        if r >= 8 and r <= 12:
            # Little-endian success
            self.filetype  = 'binary'
            self.byteorder = 'little'
            # Read the number of nodes (comma unpacks the 1-element array)
            nNode, = np.fromfile(fid, count=1, dtype='<i4')
            # Finish the record
            np.fromfile(fid, count=r/4, dtype='<i4')
            # Read the record marker for the nodes
            nb, = np.fromfile(fid, count=1, dtype='<i4')
            # Check how many bytes are there
            if nb/12 == nNode:
                # Single-precision
                self.bytecount = 4
                self.ext = 'lb4'
            else:
                # Double-precision
                self.bytecount = 8
                self.ext = 'lb8'
            # Finished
            fid.close()
            return
        # Go to beginning of file again
        fid.seek(0)
        # Read the first bit as a double-precision big-endian int
        r = np.fromfile(fid, count=1, dtype='>i4')
        # Check for success
        if r >= 8 and r <= 12:
            # Success
            self.filetype  = 'binary'
            self.byteorder = 'big'
            # Read the number of nodes (comma unpacks the 1-element array)
            nNode, = np.fromfile(fid, count=1, dtype='>i4')
            # Finish the record
            np.fromfile(fid, count=r/4, dtype='>i4')
            # Read the record marker for the nodes
            nb, = np.fromfile(fid, count=1, dtype='>i4')
            # Check how many bytes are there
            if nb/12 == nNode:
                # Single-precision
                self.bytecount = 4
                self.ext = 'b4'
            else:
                # Double-precision
                self.bytecount = 8
                self.ext = 'b8'
            # Finished
            fid.close()
            return
        # Close the file
        fid.close()
        # Attempt to read as an ASCII file
        fid = open(fname, 'r')
        # Read the first line
        line = fid.readline()
        # Close the file
        fid.close()
        # Check if it makes sense
        try:
            # See if each entry can be converted to an integer
            [int(v) for v in line.split()]
            # Set file type
            self.filetype = 'ascii'
        except Exception:
            # No valid interpretation
            raise ValueError("File did not match any of the following types:\n"
                + "  Double-precision little-endian Fortran unformatted\n"
                + "  Single-precision little-endian Fortran unformatted\n"
                + "  Double-precision big-endian Fortran unformatted\n"
                + "  Single-precision big-endian Fortran unformatted\n"
                + "  ASCII")
   # }
   
    # ++++++++++++
    # Nodes (Slow)
    # ++++++++++++
   # {
    # Function to read node coordinates from .tri file
    def ReadNodes(self, f, nNode):
        """Read node coordinates from a .tri file.
        
        :Call:
            >>> tri.ReadNodes(f, nNode)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase` or derivative
                Triangulation instance
            *f*: :class:`file`
                Open file handle
            *nNode*: :class:`int`
                Number of nodes to read
        :Effects:
            *tri.Nodes*: :class:`np.ndarray` (:class:`float`) (*nNode*, 3)
                Matrix of nodal coordinates
            *tri.blds*: :class:`np.ndarray` (:class:`float`) (*nNode*,)
                Vector of initial boundary layer spacings
            *tri.bldel*: :class:`np.ndarray` (:class:`float`) (*nNode*,)
                Vector of boundary layer thicknesses
            *f*: :class:`file`
                File remains open
        :Versions:
            * 2014-06-16 ``@ddalle``: First version
        """
        # Save the node count.
        self.nNode = nNode
        # Read the nodes.
        Nodes = np.fromfile(f, dtype=float, count=nNode*3, sep=" ")
        # Reshape into a matrix.
        self.Nodes = Nodes.reshape((nNode,3))
        
    # Function to read node coordinates from .triq+ file
    def ReadNodesSurf(self, f, nNode):
        """Read node coordinates from an AFLR3 ``.surf`` file
        
        :Call:
            >>> tri.ReadNodesSurf(f, nNode)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase` or derivative
                Triangulation instance
            *f*: :class:`file`
                Open file handle
            *nNode*: :class:`int`
                Number of tris to read
        :Effects:
            *tri.Nodes*: :class:`np.ndarray` (:class:`float`) (*nNode*, 3)
                Matrix of nodal coordinates
            *tri.blds*: :class:`np.ndarray` (:class:`float`) (*nNode*,)
                Vector of initial boundary layer spacings
            *tri.bldel*: :class:`np.ndarray` (:class:`float`) (*nNode*,)
                Vector of boundary layer thicknesses
            *f*: :class:`file`
                File remains open
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        # Save the node count.
        self.nNode = nNode
        # Read the nodes.
        Nodes = np.fromfile(f, dtype=float, count=nNode*5, sep=" ")
        # Reshape into a matrix.
        Nodes = Nodes.reshape((nNode,5))
        # Save nodes
        self.Nodes = Nodes[:,:3]
        # Save boundary layer spacings
        self.blds = Nodes[:,3]
        # Save boundary layer thicknesses
        self.bldel = Nodes[:,4]
   # }
    
    # +++++++++++
    # Tris (Slow)
    # +++++++++++
   # {
    # Function to read triangle indices from .triq+ files
    def ReadTris(self, f, nTri):
        """Read triangle node indices from a .tri file.
        
        :Call:
            >>> tri.ReadTris(f, nTri)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase` or derivative
                Triangulation instance
            *f*: :class:`file`
                Open file handle
            *nTri*: :class:`int`
                Number of tris to read
        :Effects:
            Reads and creates *tri.Tris*; file remains open.
        :Versions:
            * 2014-06-16 ``@ddalle``: First version
        """
        # Save the tri count.
        self.nTri = nTri
        # Read the Tris
        Tris = np.fromfile(f, dtype=int, count=nTri*3, sep=" ")
        # Reshape into a matrix.
        self.Tris = Tris.reshape((nTri,3))
    
    # Function to read triangles from .surf file
    def ReadTrisSurf(self, f, nTri):
        """Read triangle node indices, comp IDs, and BCs from AFLR3 file
        
        :Call:
            >>> tri.ReadTrisSurf(f, nTri)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase` or derivative
                Triangulation instance
            *f*: :class:`file`
                Open file handle
            *nTri*: :class:`int`
                Number of tris to read
        :Effects:
            *tri.Tris*: :class:`np.ndarray` (:class:`int`) (*nTri*, 3)
                Matrix of nodal coordinates
            *tri.CompID*: :class:`np.ndarray` (:class:`int`) (*nTri*,)
                Vector of component IDs for each triangle
            *tri.BCs*: :class:`np.ndarray` (:class:`int`) (*nTri*,)
                Vector of boundary condition flags
            *f*: :class:`file`
                File remains open
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        # Save the tri count
        self.nTri = nTri
        # Exit if no tris
        if nTri == 0:
            self.Tris = np.zeros((0,3))
            self.CompID = np.zeros(0, dtype=int)
            self.BCs = np.zeros(0, dtype=int)
            return
        # Read the tris
        Tris = np.fromfile(f, dtype=int, count=nTri*6, sep=" ")
        # Reshape into a matrix
        Tris = Tris.reshape((nTri,6))
        # Save the triangles
        self.Tris = Tris[:,:3]
        # Save the component IDs.
        self.CompID = Tris[:,3]
        # Save the boundary conditions.
        self.BCs = Tris[:,5]
        
   # }
    
    # ++++++++++++
    # Quads (Slow)
    # ++++++++++++
   # {
    # Function to read quads from .surf file
    def ReadQuadsSurf(self, f, nQuad):
        """Read quad node indices, compIDs, and BCs from AFLR3 file
        
        :Call:
            >>> tri.ReadQuadsSurf(f, nQuad)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase` or derivative
                Triangulation instance
            *f*: :class:`file`
                Open file handle
            *nTri*: :class:`int`
                Number of tris to read
        :Effects:
            *tri.Quads*: :class:`np.ndarray` (:class:`int`) (*nQuad*, 4)
                Matrix of nodal coordinates
            *tri.CompIDQuad*: :class:`np.ndarray` (:class:`int`) (*nQuad*,)
                Vector of component IDs for each quad
            *tri.BCsQuad*: :class:`np.ndarray` (:class:`int`) (*nQuad*,)
                Vector of boundary condition flags
            *f*: :class:`file`
                File remains open
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        # Save the tri count
        self.nQuad = nQuad
        # Exit if no tris
        if nQuad == 0:
            self.Quads = np.zeros((0,3))
            self.CompIDQuad = np.zeros(0, dtype=int)
            self.BCsQuad = np.zeros(0, dtype=int)
            return
        # Read the tris
        Quads = np.fromfile(f, dtype=int, count=nQuad*7, sep=" ")
        # Reshape into a matrix
        Quads = Tris.reshape((nQuad,6))
        # Save the triangles
        self.Quads = Quads[:,:4]
        # Save the component IDs.
        self.CompIDQuad = Quads[:,4]
        # Save the boundary conditions.
        self.BCsQuad = Quads[:,6]
   # }
   
    # ++++++++
    # Comp IDs
    # ++++++++
   # {
    # Function to read the component identifiers
    def ReadCompID(self, f):
        """Read component IDs from a .tri file.
        
        :Call:
            >>> tri.ReadCompID(f)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase` or derivative
                Triangulation instance
            *f*: :class:`str`
                Open file handle
        :Effects:
            Reads and creates *tri.CompID* if not at end of file.  Otherwise all
            components are labeled ``1``.
        :Versions:
            * 2014-06-16 ``@ddalle``: First version
        """
        # Check for end of file.
        if f.tell() == os.fstat(f.fileno()).st_size:
            # Use default component ids.
            self.CompID = np.ones(self.nTri)
        else:
            # Read from file.
            self.CompID = np.fromfile(f, dtype=int, count=self.nTri, sep=" ")
   # }
    
    # +++++++++
    # State (Q)
    # +++++++++
   # {
    # Function to read node coordinates from .triq+ file
    def ReadQ(self, f, nNode, nq):
        """Read node states from a ``.triq`` file.
        
        :Call:
            >>> triq.ReadQ(f, nNode, nq)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase` or derivative
                Triangulation instance
            *f*: :class:`file`
                Open file handle
            *nNode*: :class:`int`
                Number of nodes to read
            *nq*: :class:`int`
                Number of state variables at each node
        :Effects:
            Reads and creates *tri.Nodes*; file remains open.
        :Versions:
            * 2015-09-14 ``@ddalle``: First version
        """
        # Save the state count.
        self.nq = nq
        # Check for null case
        if nq is None: return
        # Read the nodes.
        q = np.fromfile(f, dtype=float, count=nNode*nq, sep=" ")
        # Reshape into a matrix.
        self.q = q.reshape((nNode,nq))
   # }
    
  # >
    
    # ===========
    # TRI Writers
    # ===========
  # <
    
    # ++++++++++++++++
    # Full TRI Writers
    # ++++++++++++++++
   # {
    # Fall-through function to write the triangulation to file.
    def Write(self, fname='Components.i.tri', **kw):
        """Write triangulation to file using fastest method available
        
        :Call:
            >>> tri.WriteSlow(fname='Components.i.tri', v=True, **kw)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
            *v*: :class:`bool`
                Whether or not
            *ascii*: ``True`` | {``False``}
                Whether or not to use ASCII (text file)
            *b4*: ``True`` | {``False``}
                Whether or not to use single-precision big-endian
            *lb4*: ``True`` | {``False``}
                Whether or not to use single-precision little-endian
            *b8*: ``True`` | {``False``}
                Whether or not to use double-precision big-endian
            *lb8*: ``True`` | {``False``}
                Whether or not to use double-precision little-endian
            *byteorder*: ``"big"`` | ``"little"`` | {``None``}
                Byte order
            *endian*: ``"big"`` | ``"little"`` | {``None``}
                Byte order
            *bytecount*: ``4`` | ``8`` | {``None``}
                Byte count, ``4`` for single precision is default
            *byteswap*: ``True`` | ``False`` | {``None``}
                Use byte order opposite to *os.sys.byteorder*
            *bin*: ``True`` | ``False`` | {``None``}
                Force binary output; can be used to get default binary format
            *dp*: ``True`` | {``False``}
                Use double-precision (default is single-precision)
            *sp*: ``True`` | {``False``}
                Use single-precision (no effect since default is single)
        :Examples:
            >>> tri = cape.ReadTri('bJet.i.tri')
            >>> tri.Write('bjet2.tri')
        :Versions:
            * 2014-05-23 ``@ddalle``: First version
            * 2015-01-03 ``@ddalle``: Added C capability
            * 2015-02-25 ``@ddalle``: Added status update
            * 2016-10-02 ``@ddalle``: Now checking for binary/ASCII
        """
        # Status update.
        if kw.get('v', False):
            print("    Writing triangulation: '%s'" % fname)
        # Get the extension
        ext = self.GetOutputFileType(**kw)
        # Check text vs. binary
        if ext == 'ascii':
            # Try the ASCII writers
            self.WriteASCII(fname)
        elif ext == 'lb4':
            # Little-endian single
            self.WriteTri_lb4(fname)
        elif ext == 'b4':
            # Big-endian single
            self.WriteTri_b4(fname)
        elif ext == 'lb8':
            # Little-endian double
            self.WriteTri_lb8(fname)
        elif ext == 'b8':
            # Big-endian double
            self.WriteTri_b8(fname)
            
    # Process file type by options
    def GetOutputFileType(self, **kw):
        """Determine output file type from keyword inputs
        
        Many of the possible inputs are in conflict.  For example, it is
        possible to use ``endian="big"`` and ``byteorder="little"``.  In the
        input table below, any value that evaluates as ``True`` supersedes all
        inputs listed below it.  For example, if ``ascii==True``, no other
        inputs have any effect.
        
        If none of the inputs is specified, the function will try to access
        *tri.ext*.  If that does not exist, the default output is ``"ascii"``.
        
        The default byte order is determined from *os.sys.byteorder*, but this
        is overridden if the environment variable $F_UFMTENDIAN is ``"big"`` or
        $GFORTRAN_CONVERT_UNIT is ``"big_endian"``.
        
        The five possible outputs are described below.
        
            =================  =======================================
            *ext*              *Description*
            =================  =======================================
            ``"ascii"``        Text file
            ``"b4"``           Big-endian, single-precision
            ``"lb4"``          Little-endian, single-precision
            ``"b8"``           Big-endian, double-precision
            ``"lb8"``          Little-endian, double-precision
            =================  =======================================
        
        :Call:
            >>> ext = tri.GetOutputFileType(**kw)
        :Inputs:
            *ascii*: ``True`` | {``False``}
                Whether or not to use ASCII (text file)
            *b4*: ``True`` | {``False``}
                Whether or not to use single-precision big-endian
            *lb4*: ``True`` | {``False``}
                Whether or not to use single-precision little-endian
            *b8*: ``True`` | {``False``}
                Whether or not to use double-precision big-endian
            *lb8*: ``True`` | {``False``}
                Whether or not to use double-precision little-endian
            *byteorder*: ``"big"`` | ``"little"`` | {``None``}
                Byte order
            *endian*: ``"big"`` | ``"little"`` | {``None``}
                Byte order
            *bytecount*: ``4`` | ``8`` | {``None``}
                Byte count, ``4`` for single precision is default
            *byteswap*: ``True`` | ``False`` | {``None``}
                Use byte order opposite to *os.sys.byteorder*
            *bin*: ``True`` | ``False`` | {``None``}
                Force binary output; can be used to get default binary format
            *dp*: ``True`` | {``False``}
                Use double-precision (default is single-precision)
            *sp*: ``True`` | {``False``}
                Use single-precision (no effect since default is single)
        :Outputs:
            *ext*: {``ascii``} | ``b4`` | ``b8`` | ``lb4`` | ``lb8``
                File type
        :Outputs:
            * 2016-10-02 ``@ddalle``: First version
        """
        # Check for easy cases
        if kw.get('ascii'):
            # ASCII file
            return 'ascii'
        elif kw.get('b4'):
            # Big-endian single-precision
            return 'b4'
        elif kw.get('lb4'):
            # Little-endian single precision
            return 'lb4'
        elif kw.get('b8'):
            # Big-endian double-precision
            return 'b8'
        elif kw.get('lb8'):
            # Little-endian double precision
            return 'lb8'
        # Get non-specific keyword arguments
        bo = kw.get('byteorder', kw.get('endian'))
        bs = kw.get('byteswap')
        bc = kw.get('bytecount')
        # Check for at least one binary flag
        if (not kw.get('bin')) and bo==None and bs==None and bc==None:
            # Get the existing extension
            try:
                return self.ext
            except AttributeError:
                # Default to ASCII
                return 'ascii'
        # Default byte order
        if bo is None:
            # Check for byteswap flag
            if bs:
                # Reverse default byte order (ignore env variables)
                if os.sys.byteorder == "big":
                    bs = "little"
                else:
                    bs = "big"
            else:
                # Use the default
                bs = io.sbo
        # Default byte count
        if bc is None:
            # Check for single-precision and double-precision flags
            if kw.get('dp'):
                # Double precision flag
                bc = 8
            else:
                # Single precision flag or default
                bc = 4
        # Process output
        if bo == "big":
            if bc == 8:
                return 'b8'
            elif bc == 4:
                return 'b4'
            else:
                raise ValueError("Could not interpret bytecount '%s'" % bc)
        elif bo == "little":
            if bc == 8:
                return 'lb8'
            elif bc == 4:
                return 'lb4'
            else:
                raise ValueError("Could not interpret bytecount '%s'" % bc)
        else:
            raise ValueError("Could not interpret byte order '%s'" % bo)
            
   # }
    
    # +++++++++++++++++
    # ASCII TRI Writers
    # +++++++++++++++++
   # {
    # Write ASCII with fall-through to Python method
    def WriteASCII(self, fname='Components.i.tri'):
        """Write triangulation to file using fastest method available
        
        :Call:
            >>> tri.WriteSlow(fname='Components.i.tri', v=True)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
            *v*: :class:`bool`
                Whether or not
        :Examples:
            >>> tri = cape.ReadTri('bJet.i.tri')
            >>> tri.Write('bjet2.tri')
        :Versions:
            * 2014-05-23 ``@ddalle``: First version
            * 2015-01-03 ``@ddalle``: Added C capability
            * 2015-02-25 ``@ddalle``: Added status update
            * 2016-10-02 ``@ddalle``: Moved from :func:`tri.Write`
        """
        # Try the fast way.
        try:
            # Fast method using compiled C.
            self.WriteFast(fname)
        except Exception:
            # Slow method using Python code.
            self.WriteSlow_ASCII(fname)
    
    # Function to write a triangulation to file as fast as possible.
    def WriteFast(self, fname='Components.i.tri'):
        """Try using a compiled function to write to file
        
        :Call:
            >>> tri.WriteFast(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
        :Versions:
            * 2015-01-03 ``@ddalle``: First version
        """
        # Write the nodes.
        pc.WriteTri(self.Nodes, self.Tris)
        # Write the component IDs.
        pc.WriteCompID(self.CompID)
        # Check the file name.
        if fname != "Components.pyCart.tri":
            # Move the file.
            os.rename("Components.pyCart.tri", fname)
    
    # Function to write a triangulation to file the old-fashioned way.
    def WriteSlow_ASCII(self, fname='Components.i.tri', nq=None):
        """Write a triangulation to file
        
        :Call:
            >>> tri.WriteASCIISlow(fname='Components.i.tri')
            >>> tri.WriteASCIISlow(fname='Components.i.tri', nq=None)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
            *nq*: {``None``} | :class:`int`
                Number of states, override the value of *tri.nq*
        :Examples:
            >>> tri = cape.ReadTri('bJet.i.tri')
            >>> tri.Write('bjet2.tri')
        :Versions:
            * 2014-05-23 ``@ddalle``: First version
        """
        # Open the file for creation.
        fid = open(fname, 'w')
        # Get number of state vars
        if nq is None:
            try:
                nq = self.nq
            except AttributeError:
                # No attribute for number of states
                nq = 0
        # Check for extra header info
        if nq == 0 or nq is None:
            # Write the number of nodes and triangles.
            fid.write('%i  %i\n' % (self.nNode, self.nTri))
        else:
            # Write the number of states as well
            fid.write('%i %i %i\n' % (self.nNode, self.nTri, nq))
        # Write the nodal coordinates, tris, and component ids.
        np.savetxt(fid, self.Nodes, fmt="%+15.8e", delimiter=' ')
        np.savetxt(fid, self.Tris,  fmt="%i",      delimiter=' ')
        np.savetxt(fid, self.CompID, fmt="%i",      delimiter=' ')
        # Close the file.
        fid.close()
        
   # }
    
    # ++++++++++++++
    # Binary Writers
    # ++++++++++++++
   # {
    # Write TRI file as lb4 file
    def WriteTri_lb4(self, fname):
        """Write a triangulation as a little-endian single-precision file
        
        :Call:
            >>> tri.WriteTri_lb4(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-10-10 ``@ddalle``: First version
        """
        try:
            # Compiled (C) version
            self.WriteFast_lb4(fname)
        except Exception:
            # Python fall-back function
            self.WriteSlow_lb4(fname)
            
    # Write TRI file as little-endian single-precision
    def WriteFast_lb4(self, fname='Components.i.tri'):
        """Use compiled C code to write single-precision little-endian tri
        
        :Call:
            >>> tri.WriteFast_lb4(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-10-10 ``@ddalle``: First version
        """
        # 
        pc.WriteTri_lb4(self.Nodes, self.Tris, self.CompID)
        # Check the file name and rename if necessary
        if fname != "Components.pyCart.tri":
            os.rename("Components.pyCart.tri", fname)
    
    # Write TRI file as little-endian single-precision
    def WriteSlow_lb4(self, fname='Components.i.tri'):
        """Use Python code to write single-precision little-endian tri
        
        :Call:
            >>> tri.WriteSlow_lb4(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-09-05 ``@ddalle``: First version
        """
        # Check for state vars
        try:
            qq = self.nq > 0
        except AttributeError:
            # No attribute for number of states
            qq = False
        # Open the file
        fid = open(fname, 'wb')
        # Write the header
        if qq:
            # Write with *q*
            io.write_record_lb4_i(fid, [self.nNode, self.nItri, self.nq])
        else:
            # No q values
            io.write_record_lb4_i(fid, [self.nNode, self.nTri])
        # Write the nodes, tris, and compIDs
        io.write_record_lb4_f(fid, self.Nodes)
        io.write_record_lb4_i(fid, self.Tris)
        io.write_record_lb4_i(fid, self.CompID)
        # Write states if appropriate
        if qq: io.write_record_lb4_f(fid, self.q)
        # Close the file
        fid.close()
            
    # Write TRI file as b4 file
    def WriteTri_b4(self, fname):
        """Write a triangulation as a big-endian single-precision file
        
        :Call:
            >>> tri.WriteTri_b4(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-10-10 ``@ddalle``: First version
        """
        try:
            # Compiled (C) version
            self.WriteFast_b4(fname)
        except Exception:
            # Python fall-back function
            self.WriteSlow_b4(fname)
            
    # Write TRI file as big-endian single-precision
    def WriteFast_b4(self, fname='Components.i.tri'):
        """Use compiled C code to write single-precision big-endian tri
        
        :Call:
            >>> tri.WriteFast_lb4(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-10-10 ``@ddalle``: First version
        """
        # 
        pc.WriteTri_b4(self.Nodes, self.Tris, self.CompID)
        # Check the file name and rename if necessary
        if fname != "Components.pyCart.tri":
            os.rename("Components.pyCart.tri", fname)
        
    # Write TRI file as big-endian single-precision
    def WriteSlow_b4(self, fname='Components.i.tri'):
        """Use compiled code to write single-precision big-endian tri
        
        :Call:
            >>> tri.WriteSlow_b4(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-09-05 ``@ddalle``: First version
        """
        # Check for state vars
        try:
            qq = self.nq > 0
        except AttributeError:
            # No attribute for number of states
            qq = False
        # Open the file
        fid = open(fname, 'wb')
        # Write the header
        if qq:
            # Write with *q*
            io.write_record_b4_i(fid, [self.nNode, self.nItri, self.nq])
        else:
            # No q values
            io.write_record_b4_i(fid, [self.nNode, self.nTri])
        # Write the nodes, tris, and compIDs
        io.write_record_b4_f(fid, self.Nodes)
        io.write_record_b4_i(fid, self.Tris)
        io.write_record_b4_i(fid, self.CompID)
        # Write states if appropriate
        if qq: io.write_record_b4_f(fid, self.q)
        # Close the file
        fid.close()
    
    # Write TRI file as lb8 file
    def WriteTri_lb8(self, fname):
        """Write a triangulation as a little-endian double-precision file
        
        :Call:
            >>> tri.WriteTri_lb4(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-10-10 ``@ddalle``: First version
        """
        try:
            # Compiled (C) version
            self.WriteFast_lb8(fname)
        except Exception:
            # Python fall-back function
            self.WriteSlow_lb8(fname)
            
    # Write TRI file as little-endian double-precision
    def WriteFast_lb8(self, fname='Components.i.tri'):
        """Use compiled C code to write double-precision little-endian tri
        
        :Call:
            >>> tri.WriteFast_lb4(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-10-10 ``@ddalle``: First version
        """
        # 
        pc.WriteTri_lb8(self.Nodes, self.Tris, self.CompID)
        # Check the file name and rename if necessary
        if fname != "Components.pyCart.tri":
            os.rename("Components.pyCart.tri", fname)
        
    # Write TRI file as little-endian double-precision
    def WriteSlow_lb8(self, fname='Components.i.tri'):
        """Use Python code to write double-precision little-endian tri
        
        :Call:
            >>> tri.WriteSlow_lb8(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-09-05 ``@ddalle``: First version
        """
        # Check for state vars
        try:
            qq = self.nq > 0
        except AttributeError:
            # No attribute for number of states
            qq = False
        # Open the file
        fid = open(fname, 'wb')
        # Write the header
        if qq:
            # Write with *q*
            io.write_record_lb4_i(fid, [self.nNode, self.nItri, self.nq])
        else:
            # No q values
            io.write_record_lb4_i(fid, [self.nNode, self.nTri])
        # Write the nodes, tris, and compIDs
        io.write_record_lb8_f(fid, self.Nodes)
        io.write_record_lb4_i(fid, self.Tris)
        io.write_record_lb4_i(fid, self.CompID)
        # Write states if appropriate
        if qq: io.write_record_lb8_f(fid, self.q)
        # Close the file
        fid.close()
            
    # Write TRI file as b8 file
    def WriteTri_b8(self, fname):
        """Write a triangulation as a big-endian double-precision file
        
        :Call:
            >>> tri.WriteTri_b8(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-10-10 ``@ddalle``: First version
        """
        try:
            # Compiled (C) version
            self.WriteFast_b8(fname)
        except Exception:
            # Python fall-back function
            self.WriteSlow_b8(fname)
            
    # Write TRI file as big-endian single-precision
    def WriteFast_b8(self, fname='Components.i.tri'):
        """Use compiled C code to write double-precision big-endian tri
        
        :Call:
            >>> tri.WriteFast_b8(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-10-10 ``@ddalle``: First version
        """
        # 
        pc.WriteTri_b8(self.Nodes, self.Tris, self.CompID)
        # Check the file name and rename if necessary
        if fname != "Components.pyCart.tri":
            os.rename("Components.pyCart.tri", fname)
        
    # Write TRI file as big-endian double-precision
    def WriteSlow_b8(self, fname='Components.i.tri'):
        """Use Python code to write double-precision big-endian tri
        
        :Call:
            >>> tri.WriteSlow_b8(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-09-05 ``@ddalle``: First version
        """
        # Check for state vars
        try:
            qq = self.nq > 0
        except AttributeError:
            # No attribute for number of states
            qq = False
        # Open the file
        fid = open(fname, 'wb')
        # Write the header
        if qq:
            # Write with *q*
            io.write_record_b4_i(fid, [self.nNode, self.nItri, self.nq])
        else:
            # No q values
            io.write_record_b4_i(fid, [self.nNode, self.nTri])
        # Write the nodes, tris, and compIDs
        io.write_record_b8_f(fid, self.Nodes)
        io.write_record_b4_i(fid, self.Tris)
        io.write_record_b4_i(fid, self.CompID)
        # Write states if appropriate
        if qq: io.write_record_b8_f(fid, self.q)
        # Close the file
        fid.close()
    
   # }
  # >
    
    # =============
    # Other Readers
    # =============
  # <
    # Read from a .uh3d file.
    def ReadUH3D(self, fname):
        """Read a triangulation file (from ``*.uh3d``)
        
        :Call:
            >>> tri.ReadUH3D(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: :class:`str`
                Name of triangulation file to read
        :Versions:
            * 2014-06-02 ``@ddalle``: First version
            * 2014-10-27 ``@ddalle``: Added draft of reading component names
        """
        # Open the file
        fid = open(fname, 'r')
        # Read the first line and discard.
        line = fid.readline()
        # Read the second line and split by commas.
        data = fid.readline().split(',')
        # Process the number of nodes and tris
        nNode = int(data[0])
        nTri = int(data[2])
        # Save the statistics.
        self.nNode = nNode
        self.nTri = nTri
        self.nQuad = 0
        
        # Initialize the nodes.
        Nodes = np.zeros((nNode, 3))
        # Loop through the nodes.
        for i in range(nNode):
            # Read the next line.
            Nodes[i] = np.fromfile(fid, dtype=float, count=4, sep=",")[1:4]
        # Save
        self.Nodes = Nodes
        
        # Initialize the Tris and component numbers
        Tris = np.zeros((nTri, 3))
        CompID = np.ones(nTri)
        # Loop through the lines.
        for i in range(nTri):
            # Read the line.
            d = np.fromfile(fid, dtype=int, count=5, sep=",")
            # Save the indices.
            Tris[i] = d[1:4]
            # Save the component number.
            CompID[i] = d[4]
        # Save.
        self.Tris   = np.array(Tris,   dtype=int)
        self.CompID = np.array(CompID, dtype=int)
        
        # Set location.
        ftell = -1
        # Initialize components.
        Conf = {}
        # Check for named components
        while fid.tell() != ftell:
            # Save the position.
            ftell = fid.tell()
            # Read next line.
            v = fid.readline().split(',')
            # Check if it could be a line like "'1', 'Entire'"
            if len(v) != 2: break
            # Try to convert it.
            try:
                # Get an index.
                cid = int(v[0])
                # Get the component name.
                cname = v[1].strip().strip('\'')
                # Save it.
                if cname in Conf:
                    # Append this compID to the list for this component
                    if type(Conf[cname]).__name__ == 'list':
                        # Append to the list
                        Conf[cname].append(cid)
                    else:
                        # Create a list with the two entries
                        Conf[cname] = [Conf[cname], cid]
                else:
                    # Start a hash for this *cname*
                    Conf[cname] = cid
            except Exception:
                break
        # Save the named components.
        self.Conf = Conf
        # Close the file.
        fid.close()
        
    # Read surface file
    def ReadSurf(self, fname):
        """Read an AFLR3 surface file
        
        :Call:
            >>> tri.ReadUH3D(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: :class:`str`
                Name of triangulation file to read
        :Versions:
            * 2014-06-02 ``@ddalle``: First version
            * 2014-10-27 ``@ddalle``: Added draft of reading component names
        """
        # Open the file
        fid = open(fname, 'r')
        # Read the first line.
        line = fid.readline().strip()
        # Process the first line.
        nTri, nQuad, nNode = (int(v) for v in line.split())
        
        # Read the nodes.
        self.ReadNodesSurf(fid, nNode)
        # Read the Tris.
        self.ReadTrisSurf(fid, nTri)
        # Read the Quads.
        self.ReadQuadsSurf(fid, nQuad)
        
        # Close the file.
        fid.close()
    
    # Function to read IDEAS UNV files
    def ReadUnv(self, fname):
        """Read an IDEAS format UNV triangulation
        
        :Call:
            >>> tri.ReadUnv(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: :class:`str`
                File name
        :Versions:
            * 2015-12-13 ``@ddalle``: First version
        """
        # Check for the file
        if not os.path.isfile(fname):
            raise SystemError("File '%s' does not exist" % fname)
        # Status update
        print("  Reading number of points, edges, and tris")
        # grep command to get number of points
        # Line type: "    iNode     1      1     11\n"
        cmdi = ['egrep "^\s+[0-9]+\s+1\s+1\s+11\s*" %s | tail -1' % fname]
        # Get last line with a point
        line = sp.Popen(cmdi, stdout=sp.PIPE, shell=True).communicate()[0]
        # Get number of nodes
        nNode = int(line.split()[0])
        # Command to get number of edges
        # Line type: "   iEdge  11  2  1  7  2"
        cmdi = ['egrep "^\s+[0-9]+\s+11\s+2\s+1\s+7\s+2\s*$" %s | tail -1'
        % fname]
        # Get the last line with an edge declaration
        line = sp.Popen(cmdi, stdout=sp.PIPE, shell=True).communicate()[0]
        # Get number of tris
        nEdge = int(line.split()[0])
        # Command to get number of tris
        # Line type: "   iTri  41  2  1  7  3"
        cmdi = ['egrep "^\s+[0-9]+\s+41\s+2\s+1\s+7\s+3\s*$" %s | tail -1'
        % fname]
        # Get the last line with a tri declaration
        line = sp.Popen(cmdi, stdout=sp.PIPE, shell=True).communicate()[0]
        # Get number of tris
        nTri = int(line.split()[0]) - nEdge
        # Initialize.
        self.nNode = nNode
        self.nTri  = nTri
        self.Nodes = np.zeros((nNode, 3), dtype=float)
        self.Tris  = np.zeros((nTri,  3), dtype=int)
        # Initialize a component
        self.CompID = np.ones((nTri), dtype=int)
        # Status update
        print("  Reading %i nodes" % nNode)
        # Read the file
        f = open(fname, 'r')
        # First 19 liens are discarded
        for i in range(19): f.readline()
        # Read the points.
        for j in np.arange(nNode):
            # Discard declaration line
            f.readline()
            # Get nodal coordinates
            line = f.readline()
            self.Nodes[j,:] = [float(v) for v in line.split()]
        # Discard three lines
        for j in range(3): f.readline()
        # Status update
        print("  Discarding %i edges" % nEdge)
        # Loop through the edges
        for j in np.arange(nEdge):
            # Discard the declaration lines
            f.readline()
            f.readline()
            # Discard edges
            f.readline()
        # Status update
        print("  Reading %i triangle faces" % nTri)
        # Loop through the faces
        for j in np.arange(nTri):
            # Discard the declaration line
            f.readline()
            # Get node indices
            self.Tris[j] = [int(v) for v in f.readline().split()]
        # Discard three lines
        for j in range(3): f.readline()
        # Initialize components
        iComp = 0
        Conf = {}
        # Save the named components.
        self.Conf = Conf
        # Check for components
        line = "-1"
        while line != '':
            # Read the line
            line = f.readline()
            # Check for end
            if line.strip() in ["-1", ""]: break
            # Move to next component ID.
            iComp += 1
            # Read number of points in component
            kTri = int(line.split()[-1])
            # Get the component name
            comp = f.readline().strip()
            # Status update
            print("    Mapping component '%s' -> %i" % (comp, iComp))
            self.Conf[comp] = iComp
            # Read the indices of tris in that group
            KTri = np.fromfile(f, dtype=int, count=4*kTri, sep=" ")
            # Assign the compID for the corresponding tris
            self.CompID[KTri[1::4]-nEdge-1] = iComp
        # Close the file.
        f.close()
  # >
    
    # =============
    # Other Writers
    # =============
  # <
    
    # ++++
    # TRIQ
    # ++++
   # <
    # Fall-through function to write the triangulation to file.
    def WriteTriq(self, fname='Components.i.triq', v=True):
        """Write q-triangulation to file using fastest method available
        
        :Call:
            >>> triq.WriteTriq(fname='Components.i.triq', v=True)
        :Inputs:
            *triq*: :class:`cape.tri.Triq`
                Triangulation instance to be written
            *fname*: :class:`str`
                Name of triangulation file to create
            *v*: :class:`bool`
                Whether or not
        :Examples:
            >>> triq = cape.ReadTriq('bJet.i.triq')
            >>> triq.Write('bjet2.triq')
        :Versions:
            * 2014-05-23 ``@ddalle``: First version
            * 2015-01-03 ``@ddalle``: Added C capability
            * 2015-02-25 ``@ddalle``: Added status update
            * 2015-09-14 ``@ddalle``: Copied from :func:`TriBase.WriteTri`
        """
        # Status update.
        if v:
            print("     Writing triangulation: '%s'" % fname)
        # Try the fast way.
        try:
            # Fast method using compiled C.
            self.WriteTriqFast(fname)
        except Exception:
            # Slow method using Python code.
            self.WriteTriqSlow(fname)
        
    # Function to write a triq file the old-fashioned way.
    def WriteTriqSlow(self, fname='Components.i.triq'):
        """Write a triangulation file with state to file
        
        :Call:
            >>> triq.WriteTriqSlow(fname='Components.i.triq')
        :Inputs:
            *triq*: :class:`cape.tri.Triq`
                Triangulation instance to be written
            *fname*: :class:`str`
                Name of triangulation file to create
        :Examples:
            >>> triq = cape.ReadTriQ('bJet.i.triq')
            >>> triq.Write('bjet2.triq')
        :Versions:
            * 2015-09-14 ``@ddalle``: First version
        """
        print("Label 110: Failed")
        # Write the Common portion of the triangulation
        self.WriteSlow(fname=fname)
        # Open the file to append.
        fid = open(fname, 'a')
        # Loop through states.
        for qi in self.q:
            # Write the pressure coefficient.
            fid.write('%.6f\n' % qi[0])
            # Line of text for the remaining state variables.
            line = ' ' + ' '.join(['%.6f' % qij for qij in qi[1:]]) + '\n'
            # Write it
            fid.write(line)
        # Close the flie.
        fid.close()
        
    # Function to write a triq file via C function
    def WriteTriqFast(self, fname='Components.i.triq'):
        """Write a triangulation file with state to file via Python/C
        
        :Call:
            >>> triq.WriteTriqFast(fname='Components.i.triq')
        :Inputs:
            *triq*: :class:`cape.tri.Triq`
                Triangulation instance to be written
            *fname*: :class:`str`
                Name of triangulation file to create
        :Examples:
            >>> triq = cape.ReadTriQ('bJet.i.triq')
            >>> triq.Write('bjet2.triq')
        :Versions:
            * 2015-09-14 ``@ddalle``: First version
        """
        # Write the nodes.
        pc.WriteTriQ(self.Nodes, self.Tris, self.CompID, self.q)
        # Check the file name.
        if fname != "Components.pyCart.tri":
            # Move the file.
            os.rename("Components.pyCart.tri", fname)
   # >
    
    # ++++++++++++
    # UH3D Writers
    # ++++++++++++
   # {
    # Function to write a UH3D file
    def WriteUH3D(self, fname='Components.i.uh3d'):
        """Write a triangulation to a UH3D file
        
        :Call:
            >>> tri.WriteUH3D(fname='Components.i.uh3d')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
        :Examples:
            >>> tri = cape.ReadTri('bJet.i.tri')
            >>> tri.WriteUH3D('bjet2.uh3d')
        :Versions:
            * 2015-04-17 ``@ddalle``: First version
        """
        # Initialize labels
        lbls = {}
        # If read from a UH3D file, there is a tri.Conf attribute
        try:
            # Loop through the named components
            for gID in self.Conf:
                # Get the value
                cID = self.Conf[gID]
                # Add it to the list
                lbls[cID] = gID
        except Exception:
            pass
        # Try to invert the configuration.
        try:
            # Loop through named components.
            for gID in self.config.faces:
                # Get the value.
                cID = self.config.GetCompID(gID)
                # Check the length.
                if len(cID) != 1: continue
                # Add it to the list.
                lbls[cID[0]] = gID
        except Exception:
            pass
        # Write the file.
        self.WriteUH3DSlow(fname, lbls)
    
    # Function to write a UH3D file the old-fashioned way.
    def WriteUH3DSlow(self, fname='Components.i.uh3d', lbls={}):
        """Write a triangulation to a UH3D file
        
        :Call:
            >>> tri.WriteUH3DSlow(fname='Components.i.uh3d', lbls={})
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
            *lbls*: :class:`dict`
                Optioan dict of names for component IDs, e.g. ``{1: "body"}`` 
        :Versions:
            * 2015-04-17 ``@ddalle``: First version
        """
        # Number of component IDs
        nID = len(np.unique(self.CompID))
        # Open the file for creation.
        fid = open(fname, 'w')
        # Write the author line.
        fid.write(' file created by cape\n')
        # Write the information line.
        fid.write('%i, %i, %i, %i, %i, %i\n' %
            (self.nNode, self.nNode, self.nTri, self.nTri, nID, nID))
        # Loop through the nodes.
        for i in np.arange(self.nNode):
            # Write the line (with 1-based node index).
            fid.write('%i, %.12f, %.12f, %.12f\n' %
                (i+1, self.Nodes[i,0], self.Nodes[i,1], self.Nodes[i,2]))
        # Loop through the triangles.
        for k in np.arange(self.nTri):
            # Write the line (with 1-based triangle index and CompID).
            fid.write('%i, %i, %i, %i, %i\n' % (k+1, self.Tris[k,0], 
                self.Tris[k,1], self.Tris[k,2], self.CompID[k]))
        # Loop through the component names.
        for k in range(1,nID+1):
            # Get the name that will be written.
            lbl = lbls.get(k, str(k))
            # Write the label.
            fid.write("%i, '%s'\n" % (k, lbl))
        # Write termination line.
        fid.write('99,99,99,99,99\n')
        # Close the file.
        fid.close()
        
   # }
    
    # +++++++++++
    # STL Writers
    # +++++++++++
   # {
    # Write STL using python language
    def WriteSTL(self, fname='Components.i.stl', v=False):
        """Write a triangulation to an STL file
        
        :Call:
            >>> tri.WriteSTL(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
        :Versions:
            * 2015-11-22 ``@ddalle``: First version
        """
        # Status update.
        if v:
            print("    Writing triangulation: '%s'" % fname)
        # Try the fast way.
        try:
            # Fast method using compiled C.
            self.WriteSTLFast(fname)
        except Exception:
            # Slow method using Python code.
            self.WriteSTLSlow(fname)
        
    # Write STL using python language
    def WriteSTLSlow(self, fname='Components.i.stl'):
        """Write a triangulation to an STL file
        
        :Call:
            >>> tri.WriteSTLSlow(fname='Components.i.stl')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
        :Versions:
            * 2015-11-22 ``@ddalle``: First version
        """
        # Ensure that normals have been calculated
        self.GetNormals()
        # Open the file for creation.
        f = open(fname, 'w')
        # Header
        f.write('solid\n')
        # Loop through triangles
        for i in np.arange(self.nTri):
            # Triangle
            ti = self.Tris[i]
            # Normal
            ni = self.Normals[i]
            # Vertices
            x0 = self.Nodes[ti[0]]
            x1 = self.Nodes[ti[1]]
            x2 = self.Nodes[ti[2]]
            # Write header and normal vector
            f.write('   facet normal   %12.5e %12.5e %12.5e\n' % tuple(ni))
            # Write vertices
            f.write('      outer loop\n')
            f.write('         vertex   %12.5e %12.5e %12.5e\n' % tuple(x0))
            f.write('         vertex   %12.5e %12.5e %12.5e\n' % tuple(x1))
            f.write('         vertex   %12.5e %12.5e %12.5e\n' % tuple(x2))
            # Close the loop
            f.write('      endloop\n')
            f.write('   endfacet\n')
        # End header
        f.write('endsolid\n')
        # Close the file.
        f.close()
    
    # Function to write a triangulation to file as fast as possible.
    def WriteSTLFast(self, fname='Components.i.stl'):
        """Try using a compiled function to write to file
        
        :Call:
            >>> tri.WriteFast(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
        :Versions:
            * 2016-04-08 ``@ddalle``: First version
        """
        # Write the nodes.
        pc.WriteTriSTL(self.Nodes, self.Tris)
        # Check the file name.
        if fname != "Components.pyCart.stl":
            # Move the file.
            os.rename("Components.pyCart.stl", fname)
   # }
        
    # ++++++++++
    # AFLR3 Surf
    # ++++++++++
   # {
    # Function to write a UH3D file
    def WriteSurf(self, fname='Components.i.surf'):
        """Write a triangulation to a AFLR3 surface file
        
        :Call:
            >>> tri.WriteSurf(fname='Components.i.surf')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
        :Versions:
            * 2015-11-19 ``@ddalle``: First version
        """
        # Status update
        print("    Writing ALFR3 surface: '%s'" % fname)
        # Make sure we have BL parameters
        try:
            self.blds
        except AttributeError:
            self.blds = np.zeros(self.nNode)
        try:
            self.bldel
        except AttributeError:
            self.bldel = np.zeros(self.nNode)
        # Make sure we have quads
        try:
            self.nQuad
        except AttributeError:
            self.nQuad = 0
        # Write the file.
        try:
            # Try compiled versoin
            self.WriteSurfFast(fname)
        except Exception:
            # Fall back to slow version
            self.WriteSurfSlow(fname)
    
    # Function to write a SURF file the old-fashioned way.
    def WriteSurfSlow(self, fname="Components.surf"):
        """Write an AFLR3 ``surf`` surface mesh file
        
        :Call:
            >>> tri.WriteSurfSlow(fname='Components.surf')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
        :Versions:
            * 2015-11-19 ``@ddalle``: First version
            * 2016-04-05 ``@ddalle``: Added quads, *blds*, and *bldel*
        """
        # Open the file for creation.
        fid = open(fname, 'w')
        # Write the number of tris, quads, points
        fid.write('%i %i %i\n' % (self.nTri, self.nQuad, self.nNode))
        # Loop through the nodes.
        for i in np.arange(self.nNode):
            # Write the line (with 1-based node index).
            fid.write('%15.8e %15.8e %15.8e %s %s\n' % (
                self.Nodes[i,0], self.Nodes[i,1], self.Nodes[i,2],
                self.blds[i], self.bldel[i]))
        # Loop through the triangles.
        for k in np.arange(self.nTri):
            # Write the line (with 1-based triangle index and CompID).
            fid.write('%i %i %i %i 0 %i\n' % (self.Tris[k,0], 
                self.Tris[k,1], self.Tris[k,2], self.CompID[k], self.BCs[k]))
        # Loop through the quads.
        for k in np.arange(self.nQuad):
            # Write the line (with 1-based quad indx and CompID)
            fid.write('%i %i %i %i %i 0 %i\n' % (self.Quads[k,0],
                self.Quads[k,1], self.Quads[k,2], self.Quads[k,3],
                self.CompIDQuad[k], self.BCsQuad[k]))
        # Close the file.
        fid.close()
    
    # Function to write a triangulation to file as fast as possible.
    def WriteSurfFast(self, fname='Components.i.surf'):
        """Try using a compiled function to write to AFLR3 ``surf`` file
        
        :Call:
            >>> tri.WriteSurfFast(fname='Components.i.surf')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance to be translated
            *fname*: :class:`str`
                Name of triangulation file to create
        :Versions:
            * 2015-01-03 ``@ddalle``: First version
        """
        # Write the nodes.
        pc.WriteSurf(
            self.Nodes, self.blds,       self.bldel,
            self.Tris,  self.CompID,     self.BCs,
            self.Quads, self.CompIDQuad, self.BCsQuad)
        # Check the file name.
        if fname != "Components.pyCart.surf":
            # Move the file.
            os.rename("Components.pyCart.surf", fname)
   
   # }
  # >
    
    # =====================
    # Multiple File Reading
    # =====================
  # <
    # Add a second triangulation without destroying component numbers.
    def Add(self, tri):
        """Add a second triangulation file.
        
        If the new triangulation begins with a component ID less than the
        maximum component ID of the existing triangulation, the components of
        the second triangulation are offset.  For example, if both
        triangulations have components 1, 2, and 3; the IDs of the second
        triangulation, *tri2*, will be changed to 4, 5, and 6.
        
        No checks are performed, and intersections are not analyzed.
        
        :Call:
            >>> tri.Add(tri2)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance to be altered
            *tri2*: :class:`cape.tri.Tri`
                Triangulation instance to be added to the first
        :Effects:
            All nodes and triangles from *tri2* are added to *tri*.  As a
            result, the number of nodes, number of tris, and number of
            components in *tri* will all increase.
        :Versions:
            * 2014-06-12 ``@ddalle``: First version
            * 2014-10-03 ``@ddalle``: Auto detect CompID overlap
        """
        # Concatenate the node matrix.
        self.Nodes = np.vstack((self.Nodes, tri.Nodes))
        # Concatenate the triangle node index matrix.
        self.Tris = np.vstack((self.Tris, tri.Tris + self.nNode))
        # Get the current component ID lists from both tries.
        CompID0 = np.unique(self.CompID)
        CompID1 = np.unique(tri.CompID)
        # Concatenate the component vector.
        if np.any(np.intersect1d(CompID0, CompID1)):
            # Number of components in the original triangulation
            nC = np.max(self.CompID)
            # Adjust CompIDs to avoid overlap.
            self.CompID = np.hstack((self.CompID, tri.CompID + nC))
        else:
            # Add the components raw (don't offset CompID.
            self.CompID = np.hstack((self.CompID, tri.CompID))
        # Update the statistics.
        self.nNode += tri.nNode
        self.nTri  += tri.nTri
        # Check for config
        try:
            # Check for a configuration in both triangulations
            self.config
            tri.config
            # Loop through the faces in the added configuration
            for face in tri.config.faces:
                # Check if the face is also in the current configuration
                if face in self.config.faces:
                    # Union the two faces
                    self.config.faces[face] = np.union1d(
                        self.config.faces[face], tri.config.faces[face])
                else:
                    # Add the face
                    self.config.faces[face] = tri.config.faces[face]
        except AttributeError:
            # No configurations to merge
            pass
        
    # Add a second triangulation without altering component numbers.
    def AddRawCompID(self, tri):
        """
        Add a second triangulation to the current one without changing 
        component numbers of either triangulation.  No checks are performed,
        and intersections are not analyzed.
        
        :Call:
            >>> tri.AddRawCompID(tri2)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance to be altered
            *tri2*: :class:`cape.tri.Tri`
                Triangulation instance to be added to the first
        :Effects:
            All nodes and triangles from *tri2* are added to *tri*.  As a
            result, the number of nodes, number of tris, and number of
            components in *tri* will all increase.
        :Versions:
            * 2014-06-12 ``@ddalle``: First version
        """
        # Concatenate the node matrix.
        self.Nodes = np.vstack((self.Nodes, tri.Nodes))
        # Concatenate the triangle node index matrix.
        self.Tris = np.vstack((self.Tris, tri.Tris + self.nNode))
        # Concatenate the component vector.
        self.CompID = np.hstack((self.CompID, tri.CompID))
        # Update the statistics.
        self.nNode += tri.nNode
        self.nTri  += tri.nTri
        # Done
        return None
        
  # >
    
    # ===============
    # Intersect Tools
    # ===============
  # <
    # Function to write .tri file with one CompID per break
    def WriteVolTri(self, fname='Components.tri'):
        """Write a .tri file with one CompID per break in *tri.iTri*
        
        This is a necessary step of running `intersect` because each polyhedron
        (i.e. water-tight volume) must have a single uniform component ID before
        running `intersect`.
        
        :Call:
            >>> tri.WriteCompIDTri(fname='Components.c.tri')
        :Inputs:
            *tri*: :class:`cape.tri.TriBase` or derivative
                Triangulation instance
            *fname*: :class:`str`
                Name of .tri file for use as input to `intersect`
        :Versions:
            * 2015-02-24 ``@ddalle``: First version
        """
        # Copy the triangulation.
        tri = self.Copy()
        # Current maximum CompID
        comp0 = np.max(self.CompID)
        # Set first volume.
        tri.CompID[:self.iTri[0]] = comp0 + 1
        # Loop through volumes as marked in *tri.iTri*
        for k in range(len(self.iTri)-1):
            # Set the CompID for each tri in that volume.
            tri.CompID[self.iTri[k]:self.iTri[k+1]] = comp0 + k + 2
        # Write the triangulation to file.
        tri.Write(fname)
        
    # Function to map each face's CompID to the closest match from another tri
    def MapSubCompID(self, tric, compID, kc=None):
        """
        Map CompID of each face to the CompID of the nearest face in another
        triangulation.  This is a common step after running `intersect`.
        
        :Call:
            >>> tri.MapSubCompID(tric, compID, iA=0, iB=-1)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase` or derivative
                Triangulation instance
            *tric*: :class:`cape.tri.TriBase` or derivative
                Triangulation with more desirable CompIDs to be copied
            *compID*: :class:`int`
                Component ID to map from *tric*
            *k1*: :class:`numpy.ndarray` (:class:`int`)
                Indices of faces in *tric* to considerider
        :Versions:
            * 2015-02-24 ``@ddalle``: First version
        """
        # Default last index.
        if kc is None: kc = np.arange(tric.nTri)
        # Indices of tris to map.
        K1 = np.where(self.CompID == compID)[0]
        # Check for a single component to map (volume really is one CompID).
        if len(np.unique(tric.CompID[kc])) == 1:
            # Map that component to each face in *k*.
            self.CompID[K1] = tric.CompID[kc[0]]
            # That's it.
            return
        # Make copy of the target indices.
        K0 = np.array(kc).copy()
        # Extract target triangle vertices
        x0 = tric.Nodes[tric.Tris[K0]-1,0]
        y0 = tric.Nodes[tric.Tris[K0]-1,1]
        z0 = tric.Nodes[tric.Tris[K0]-1,2]
        # Current vertices
        x1 = self.Nodes[self.Tris[K1]-1,0]
        y1 = self.Nodes[self.Tris[K1]-1,1]
        z1 = self.Nodes[self.Tris[K1]-1,2]
        # Length scale
        tol = 1e-6 * np.sqrt(np.sum(
            (np.max(self.Nodes,0)-np.min(self.Nodes,0))**2))
        # Start with the first tri.
        k0 = 0
        k1 = 0
        # Loop until one of the two sets of faces is exhausted.
        while (k0<len(K0)-1) and (k1<len(K1)-1):
            # Current point from intersected geometry.
            xk = x1[k1]; yk = y1[k1]; zk = z1[k1]
            # Distance to current intersected triangle.
            d0 = np.sqrt(
                (x0[k0:,0]-xk[0])**2 + (x0[k0:,1]-xk[1])**2 +
                (x0[k0:,2]-xk[2])**2 + (y0[k0:,0]-yk[0])**2 +
                (y0[k0:,1]-yk[1])**2 + (y0[k0:,2]-yk[2])**2 +
                (z0[k0:,0]-zk[0])**2 + (z0[k0:,1]-zk[1])**2 +
                (z0[k0:,2]-zk[2])**2)
            # Find the index of this tri in the target set.
            i0 = np.where(d0 <= tol)[0]
            # Check for match.
            if len(i0) == 0:
                # No match.
                k1 += 1
            else:
                # Take the first point.
                k0 += i0[0]
            # Try to match all the remaining points.
            n = min(len(K0)-k0, len(K1)-k1)
            # Calculate total of distances between vertices.
            dk = np.sqrt(np.sum((x1[k1:k1+n]-x0[k0:k0+n])**2 +
                (y1[k1:k1+n]-y0[k0:k0+n])**2 +
                (z1[k1:k1+n]-z0[k0:k0+n])**2, 1))
            # Check for a match.
            if not np.any(dk<=tol): continue
            # Find the first tri that does _not_ match.
            j = np.where(dk<=tol)[0][-1] + 1
            # Copy these *j* CompIDs.
            self.CompID[K1[k1:k1+j]] = tric.CompID[K0[k0:k0+j]]
            # Move to next tri in intersected surface.
            k1 += j; k0 += j
        
        # Find the triangles that are _still_ the old CompID
        K = np.where(self.CompID == compID)[0]
        
        # Calculate the centroids of the target components.
        x0 = np.mean(tric.Nodes[tric.Tris[K0]-1, 0], 1)
        y0 = np.mean(tric.Nodes[tric.Tris[K0]-1, 1], 1)
        z0 = np.mean(tric.Nodes[tric.Tris[K0]-1, 2], 1)
        # Calculate centroids of current tris.
        x1 = np.mean(self.Nodes[self.Tris-1,0], 1)
        y1 = np.mean(self.Nodes[self.Tris-1,1], 1)
        z1 = np.mean(self.Nodes[self.Tris-1,2], 1)
        # Loop through components.
        for i in K:
            # Find the closest centroid from *tric*.
            j = np.argmin((x0-x1[i])**2 + (y0-y1[i])**2 + (z0-z1[i])**2)
            # Map it.
            self.CompID[i] = tric.CompID[K0[j]]
            
    # Function to fully map component IDs
    def MapCompID(self, tric, tri0):
        """
        Map CompIDs from pre-intersected triangulation to an intersected
        triangulation.  In standard cape terminology, this is a transformation
        from :file:`Components.o.tri` to :file:`Components.i.tri`
        
        :Call:
            >>> tri.MapCompID(tric, tri0)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation interface
            *tric*: :class:`cape.tri.Tri`
                Full CompID breakdown prior to intersection
            *tri0*: :class:`cape.tri.Tri`
                Input triangulation to `intersect`
        :Versions:
            * 2015-02-25 ``@ddalle``: First version
        """
        # Get the components from the pre-intersected triangulation.
        comps = np.unique(tri0.CompID)
        # Loop through comps.
        for compID in comps:
            # Get the faces with that comp ID (before intersection)
            kc = np.where(tri0.CompID == compID)[0]
            # Map the compIDs for that component.
            self.MapSubCompID(tric, compID, kc)
  # >
    
    # ================
    # CompID Interface
    # ================
  # <
    # Function to read configuration file based on file extension
    def ReadConfig(self, c):
        """Read a configuration file using extension to guess type
        
        :Call:
            >>> tri.ReadConfig(c)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *c*: :class:`str`
                Configuration file name
        :Versions:
            * 2016-10-21 ``@ddalle``: First version
        """
        # Split based on '.'
        fext = c.split('.')
        # Get the extension
        if len(fext) < 2:
            # Odd case, no extension given
            fext = 'json'
        else:
            # Get the extension
            fext = fext[-1].lower()
        # Read configuration
        if fext == "xml":
            # Read an XML file
            self.ReadConfigXML(c)
        else:
            # Read a JSON file
            self.ReadConfigJSON(c)
        
    # Function to read Config.xml
    def ReadConfigXML(self, c):
        """Read a ``Config.xml`` file labeling and grouping of component IDs
        
        :Call:
            >>> tri.ReadConfigXML(c)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *c*: :class:`str`
                Configuration file name
        :Versions:
            * 2015-11-19 ``@ddalle``: First version
        """
        # Read the configuration and save it.
        self.config = Config(c)
        # Restrict to a subset
        self.RestrictConfigCompID()
        
    # Function to read Config.json
    def ReadConfigJSON(self, c):
        """Read a ``Config.json`` file labeling and grouping of component IDs
        
        :Call:
            >>> tri.ReadConfigJSON(c)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *c*: :class:`str`
                Configuration file name
        :Versions:
            * 2016-10-21 ``@ddalle``: First version
        """
        # Read the configuration and save it
        self.config = ConfigJSON(c)
        
    # Function to map component ID numbers to those in a Config.
    def ApplyConfig(self, cfg):
        """Change component IDs to match a configuration file
        
        Any component that is named in *tri.Conf* and *cfg.faces* has its
        component ID changed to match its intended value in *cfg*, which is an
        interface to :file:`Config.xml` files.  Note that *tri.Conf* is only
        created if the triangulation is read from a UH3D file.
        
        For example, if *tri* has a component ``'Body'`` that initially has
        component ID of 4, but the user wants that component ID to instead be
        104, then ``tri.Conf['Body']`` will be ``4``, and ``cfg.faces['Body']``
        will be ``104``.  The result of applying this method is that all faces
        in *tri.compID* that are labeled with a ``4`` will get changed to
        ``104``.
        
        This process uses a working copy of *tri* to avoid problems with the
        order of changing the component numbers.
        
        :Call:
            >>> tri.ApplyConfig(cfg)
            >>> tri.ApplyConfig(fcfg)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *cfg*: :class:`cape.config.Config`
                Configuration instance
            *fcfg*: :class:`str`
                Name of XML config file
        :Versions:
            * 2014-11-10 ``@ddalle``: First version
        """
        # Check for Conf in the triangulation.
        try:
            self.Conf
        except AttributeError:
            return
        # Check for string input
        if type(cfg).__name__ in ['str', 'unicode']:
            # Read the config
            cfg = Config(cfg)
        # Make a copy of the component IDs.
        compID = self.CompID.copy()
        # Check for components.
        for k in self.Conf:
            # Check if the component is in the cfg.
            cID = cfg.GetPropCompID(k)
            # Check for valid result (above only returns int or None)
            if cID:
                # Get the number or list of numbers from *Conf*
                kID = self.Conf[k]
                # Process type
                if type(kID).__name__ != 'list': kID = [kID]
                # Initialize indices of tris with matching compIDs
                I = np.ones_like(cID)
                # Loop through additional entries
                for kj in kID:
                    # Use *or* operation to search for other matches
                    I = np.logical_or(I, compID==kj)
                # Assign the new value.
                self.CompID[compID==self.Conf[k]] = cID
                # Save it in the Conf, too.
                self.Conf[k] = cID
                # Save the compID as an int in the *config* just for clarity
                #self.config.faces[k] = cID
        # Restrict
        self.RestrictConfigCompID()
        
    # Write a new Config.xml file
    def WriteConfigXML(self, fname="Config.xml"):
        """Write a ``Config.xml`` file specific to this triangulation
        
        :Call:
            >>> tri.WriteConfigXML(fname="Config.xml")
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
                Triangulation interface
            *fname*: {``"Config.xml"``} | :class:`str`
                Name of GMP configuration file to write
        :Versions:
            * 2016-11-06 ``@ddalle``: First version
        """
        # Check for a configuration
        try:
            self.config
        except AttributeError:
            raise AttributeError(
                ("Cannot write GMP file '%s' " % fname) +
                ("because tri instance does not have a config interface"))
        # Write the XML
        self.config.WriteXML(fname)
    
    # Restrict component IDs to those actually used in this triangulation
    def RestrictConfigCompID(self):
        """Restrict the component IDs in the *config* to those in *tri.CompID*
        
        :Call:
            >>> tri.RestrictConfigCompID()
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
        :Versions:
            * 2016-11-05 ``@ddalle``: First version
        """
        # Get list of component IDs
        compIDs = np.unique(self.CompID)
        # Call the method from the *config* handle
        try:
            self.config.RestrictCompID(compIDs)
        except Exception:
            # Print a warning
            print("WARNING: Attempt to restrict *config* component IDs failed")
            
    # Renumber component IDs 1 to *n*
    def RenumberCompIDs(self):
        """Renumber component ID numbers 1 to *n*
        
        :Call:
            >>> tri.RenumberCompIDs()
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
                Triangulation interface
        :Versions:
            * 2016-11-09 ``@ddalle``: First version
        """
        # Check for a configuration
        try:
            self.config
        except AttributeError:
            raise AttributeError(
                ("Cannot reorder component IDs without a ConfigJSON object"))
        # Get list of component IDs in ascending order
        faces = self.config.SortCompIDs()
        # List of component IDs in current list
        compIDs = np.unique(self.CompID)
        # Initialize new list
        CompID = np.zeros_like(self.CompID)
        # Initial component number
        ncomp = 0
        # Loop through sorted faces
        for face in faces:
            # Get the component number
            compi = self.config.faces[face]
            # Make sure it's an integer
            if type(compi).__name__ in ['list', 'ndarray']:
                # Extract element from singleton
                compi = compi[0]
            # Check if that compID is present
            if compi not in compIDs: continue
            # Otherwise, reset it.
            I = np.where(self.CompID==compi)[0]
            ncomp += 1
            CompID[I] = ncomp
            # Renumber the components in the *config*
            self.config.RenumberCompID(face, ncomp)
        # Check for zero
        if np.any(CompID == 0):
            # General warning
            print("  WARNING: At least one tri has unset component ID")
            # Get the locations of missed triangles
            I = np.where(CompID == 0)[0]
            # Print the list of original component IDs from here
            print("           List of original component IDs that were missed")
            print("              %s" % np.unique(self.CompID[I]))
        # Reset component IDs
        self.CompID = CompID
       
   
    # Function to get compIDs by name
    def GetCompID(self, face=None):
        """Get components by name
        
        :Call:
            >>> compID = tri.GetCompID()
            >>> compID = tri.GetCompID(face)
            >>> compID = tri.GetCompID(comp)
            >>> compID = tri.GetCompID(comps)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation interface
            *face*: :class:`str`
                Component name
            *comp*: :class:`int`
                Component ID
            *comps*: :class:`list` (:class:`int` | :class:`str`)
                List of component names or IDs
        :Outputs:
            *compID*: :class:`list` (:class:`int`)
                List of component IDs
        :Versions:
            * 2014-10-12 ``@ddalle``: First version
            * 2016-03-29 ``@ddalle``: Edited docstring
        """
        # Process input into a list of component IDs.
        try:
            # Best option is to use the Config.xml file
            return self.config.GetCompID(face)
        except Exception:
            # Check for scalar
            if face is None:
                # No contents; this might break otherwise
                return list(np.unique(self.CompID))
            elif type(face).__name__  in ['list', 'ndarray']:
                # Return the list
                return face
            else:
                # Make a singleton list
                return [face]
       
    # Function to get node indices from component ID(s)
    def GetNodesFromCompID(self, compID=None):
        """Find node indices from face component ID(s)
        
        :Call:
            >>> i = tri.GetNodesFromCompID(comp)
            >>> i = tri.GetNodesFromCompID(comps)
            >>> i = tri.GetNodesFromCompID(compID)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *comp*: :class:`str`
                Name of component
            *comps*: :class:`list` (:class:`int` | :class:`str`)
                List of component IDs or names
            *compID*: :class:`int`
                Component number
        :Outputs:
            *i*: :class:`numpy.array` (:class:`int`)
                Node indices, 0-based
        :Versions:
            * 2014-09-27 ``@ddalle``: First version
        """
        # Process inputs.
        if compID is None:
            # Return all the tris.
            return np.arange(self.nNode)
        elif compID == 'entire':
            # Return all the tris.
            return np.arange(self.nNode)
        # Get matches from tris and quads
        kTri  = self.GetTrisFromCompID(compID)
        kQuad = self.GetQuadsFromCompID(compID)
        # Initialize with all false
        I = np.arange(self.nNode) < 0
        # Check for triangular matches
        if len(kTri) > 0:
            # Mark matches
            I[self.Tris[kTri]-1] = True
        # Check for quadrangle matches
        if len(kQuad) > 0:
            # Mark matches
            I[self.Quads[kQuad]] = True
        # Output
        return np.where(I)[0]
        
    # Function to get tri indices from component ID(s)
    def GetTrisFromCompID(self, compID=None):
        """Find indices of triangles with specified component ID(s)
        
        :Call:
            >>> k = tri.GetTrisFromCompID(comp)
            >>> k = tri.GetTrisFromCompID(comps)
            >>> k = tri.GetTrisFromCompID(compID)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *comp*: :class:`str`
                Name of component
            *comps*: :class:`list` (:class:`int` | :class:`str`)
                List of component IDs or names
            *compID*: :class:`int`
                Component number
        :Outputs:
            *k*: :class:`numpy.ndarray` (:class:`int`, shape=(N,))
                List of triangle indices in requested component(s)
        :Versions:
            * 2015-01-23 ``@ddalle``: First version
        """
        # Process inputs.
        if compID is None:
            # Return all the tris.
            return np.arange(self.nTri)
        elif compID == 'entire':
            # Return all the tris.
            return np.arange(self.nTri)
        # Get list of components
        comps = self.GetCompID(compID)
        # Check for single match
        if len(comps) == 1:
            # Get a single component.
            K = self.CompID == comps[0]
        else:
            # Initialize with all False (same size as number of tris)
            K = self.CompID < 0
            # List of components.
            for comp in comps:
                # Add matches for component *ii*.
                K = np.logical_or(K, self.CompID==comp)
        # Turn boolean vector into vector of indices]
        return np.where(K)[0]
    
    # Function to get tri indices from component ID(s)
    def GetQuadsFromCompID(self, compID=None):
        """Find indices of triangles with specified component ID(s)
        
        :Call:
            >>> k = tri.GetQuadsFromCompID(comp)
            >>> k = tri.GetQuadsFromCompID(comps)
            >>> k = tri.GetQuadsFromCompID(compID)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *comp*: :class:`str`
                Name of component
            *comps*: :class:`list` (:class:`int` | :class:`str`)
                List of component IDs or names
            *compID*: :class:`int`
                Component number
        :Outputs:
            *k*: :class:`numpy.ndarray` (:class:`int`, shape=(N,))
                List of quad indices in requested component(s)
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        # Be careful because not everyone has quads
        try:
            # Process inputs.
            if compID is None:
                # Return all the tris.
                return np.arange(self.nQuad)
            elif compID == 'entire':
                # Return all the tris.
                return np.arange(self.nQuad)
            elif self.nQuad == 0:
                # No quads to check for
                return np.array([], dtype=int)
            # Get list of components
            comps = self.GetCompID(compID)
            # Check for single match
            if len(comps) == 1:
                # Get a single component.
                K = self.CompIDQuad == comps[0]
            else:
                # Initialize with all False (same size as number of tris)
                K = self.CompIDQuad < 0
                # List of components.
                for comp in comps:
                    # Add matches for component *ii*.
                    K = np.logical_or(K, self.CompIDQuad==comp)
            # Turn boolean vector into vector of indices]
            I =  np.where(K)[0]
        except AttributeError:
            # No quads
            return np.zeros(0, dtype=int)
    
    # Get subtriangulation from CompID list
    def GetSubTri(self, i=None):
        """
        Get the portion of the triangulation that contains specified component
        ID(s).
        
        :Call:
            >>> tri0 = tri.GetSubTri(i=None)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *i*: :class:`int` or :class:`list` (:class:`int`)
                Component ID or list of component IDs
        :Outputs:
            *tri0*: :class:`cape.tri.Tri`
                Copied triangulation containing only faces with CompID in *i*
        :Versions:
            * 2015-01-23 ``@ddalle``: First version
        """
        # Get the triangle indices.
        k = self.GetTrisFromCompID(i)
        # Make a copy of the triangulation.
        tri0 = self.Copy()
        # Restrict *tri0* to the matching faces.
        tri0.Tris = tri0.Tris[k]
        tri0.CompID = tri0.CompID[k]
        # Save the reduced number of tris.
        tri0.nTri = k.size
        # Output
        return tri0
                
  # >
  
    # ===============
    # FUN3D Interface
    # ===============
  # <
  # >
    
    # =========================
    # AFLR3 Boundary Conditions
    # =========================
  # <
    # Map boundary condition tags from config
    def MapBCs_ConfigAFLR3(self):
        """Map boundary conditions from ``"Config.json"`` file format
        
        :Call:
            >>> tri.MapBCs_ConfigAFLR3()
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
        :Versions:
            * 2016-10-21 ``@ddalle``: First version
        """
        # Check for configuration
        self.config
        self.config.comps
        # Initialize the BCs to -1 (grow boundary layer)
        self.BCs = -1 * np.ones_like(self.CompID)
        # Initialize quad BCs
        try:
            self.BCsQuad = -1 * np.ones(self.nQuad, dtype=int)
        except AttributeError:
            self.BCsQuad = np.ones(0, dtype=int)
        # Initialize the boundary layer spacings
        self.blds = np.zeros(self.nNode)
        self.bldel = np.zeros(self.nNode)
        # Loop through BCs
        for comp in self.config.comps:
            # Get the tris, quads, and nodes matching the component ID
            IT = self.GetTrisFromCompID(comp)
            IQ = self.GetQuadsFromCompID(comp)
            IN = self.GetNodesFromCompID(comp)
            # Get the boundary condition for this comp
            BC = self.config.GetProperty(comp, 'aflr3_bc')
            # Fallback
            if BC is None:
                BC = self.config.GetProperty(comp, 'BC')
            # Check for a BC find
            if BC is None:
                print("  Component '%s' had no 'BC' or 'aflr3_bc' property"
                    % comp)
            # Apply to the appropriate tris and quads
            if len(IT) > 0:
                try:
                    self.BCs[IT] = BC
                except Exception:
                    print("BC=%s" % BC)
            if len(IQ) > 0:
                self.BCsQuad[IQ] = BC
                
            # Get the boundary layer growth parameters
            blds = self.config.GetProperty(comp, 'blds')
            bldel = self.config.GetProperty(comp, 'bldel')
            # Check for nodes in this component
            if len(IN) > 0:
                # Check for the property
                if blds  is not None: self.blds[IN]  = blds
                if bldel is not None: self.bldel[IN] = bldel
            
        
    # Map boundary condition tags
    def MapBCs_AFLR3(self, compID=None, BCs={}, blds={}, bldel={}):
        """Initialize and map boundary condition indices for AFLR3
        
        :Call:
            >>> tri.MapBCs_AFLR3(compID=[], BCs={}, blds={}, bldel={})
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *compID*: :class:`list` (:class:`str` | :class:`int`) | ``None``
                List of components to preserve order; defaults to ``BCs.keys()``
            *BCs*: :class:`dict` (:class:`str` | :class:`int`)
                Dictionary of BC flags for CompIDs or component names
            *blds*: :class:`dict` (:class:`str` | :class:`int`)
                Dictionary of BL spacings for CompIDs or component names
            *bldel*: :class:`dict` (:class:`str` | :class:`int`)
                Dictionary of BL thicknesses for CompIDs or component names
        :Versions:
            * 2015-11-19 ``@ddalle``: First version
            * 2016-04-05 ``@ddalle``: Added BL spacing and thickness
        """
        # Initialize the BCs to -1 (grow boundary layer)
        self.BCs = -1 * np.ones_like(self.CompID)
        # Initialize quad BCs
        try:
            self.BCsQuad = -1 * np.ones(self.nQuad, dtype=int)
        except AttributeError:
            self.BCsQuad = np.ones(0, dtype=int)
        # Initialize the boundary layer spacings
        self.blds = np.zeros(self.nNode)
        self.bldel = np.zeros(self.nNode)
        # Default keys
        if compID is None:
            compID = BCs.keys()
        # Loop through BCs
        for comp in compID:
            # Get the tris matching the component ID
            I = self.GetTrisFromCompID(comp)
            # Modify those BCs
            # Check node count
            if len(I) > 0:
                self.BCs[I] = BCs[comp]
            # Get the quads from the matching component ID
            I = self.GetQuadsFromCompID(comp)
            # Modify those BCs.
            if len(I) > 0:
                self.BCsQuad[I] = BCs[comp]
        # Loop through boundary layer spacings
        for comp in blds:
            # Get the nodes
            I = self.GetNodesFromCompID(comp)
            # Check node count
            if len(I) == 0:
                print("Warning: No nodes mapped for component '%s'" % comp)
                continue
            # Modify those BL spacings
            self.blds[I] = blds[comp]
            # Check for BL thicknesses
            if comp in bldel:
                self.bldel[I] = bldel[comp]
        # Loop through boundary layer thicknesses
        for comp in bldel:
            # Make sure not already processed
            if comp in blds: continue
            # Get the nodes
            I = self.GetNodesFromCompID(comp)
            # Check node count
            if len(I) == 0:
                print("Warning: No nodes mapped for component '%s'" % comp)
                continue
            # Modify those BL thicknesses
            self.bldel[I] = bldel[comp]
            
    # Read boundary condition map
    def ReadBCs_AFLR3(self, fname):
        """Initialize and map boundary condition indices for AFLR3 from file
        
        :Call:
            >>> tri.ReadBCs_AFLR3(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: :class:`str`
                Name of boundary condition map file
        :Versions:
            * 2015-11-19 ``@ddalle``: First version
            * 2016-04-05 ``@ddalle``: Added BL spacing and thickness
        """
        # Read the boundary condition file
        f = open(fname, 'r')
        # Initialize boundary condition map
        compID = []
        BCs = OrderedDict()
        blds = OrderedDict()
        bldel = OrderedDict()
        # Loop through lines
        line = "start"
        while line != '':
            # Read the line.
            line = _readline(f)
            # Exit at end of file
            if line == '': break
            # Split line
            V = line.split()
            # Get the component name
            comp = V[0]
            # Get the boundary condition flag
            bc = int(V[1])
            # Append to the list (ordered)
            compID.append(comp)
            # Save the boundary condition
            BCs[comp] = bc
            # Check length
            if len(V) < 3: continue
            # Get the boundary layer spacing
            bldsi = float(V[2])
            # Save BL spacing
            blds[comp] = bldsi
            # Check length
            if len(V) < 4: continue
            # Get the boundary layer thickness
            bldeli = float(V[3])
            # Save the BL thickness
            bldel[comp] = bldeli
        # Close the file.
        f.close()
        # Apply the boundary conditions
        self.MapBCs_AFLR3(compID, BCs, blds=blds, bldel=bldel)
    
  # >
        
    # =============
    # Geometry Info
    # =============
  # <
    
    # ++++
    # Tris
    # ++++
   # {
    # Get normals and areas
    def GetNormals(self):
        """Get the normals and areas of each triangle
        
        :Call:
            >>> tri.GetNormals()
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
        :Effects:
            *tri.Areas*: :class:`ndarray`, shape=(tri.nTri,)
                Area of each triangle is created
            *tri.Normals*: :class:`ndarray`, shape=(tri.nTri,3)
                Unit normal for each triangle is saved
        :Versions:
            * 2014-06-12 ``@ddalle``: First version
            * 2016-01-23 ``@ddalle``: Added a check before calculating
        """
        # Check for normals.
        if hasattr(self, 'Normals'): return
        # Extract the vertices of each tri.
        x = self.Nodes[self.Tris-1, 0]
        y = self.Nodes[self.Tris-1, 1]
        z = self.Nodes[self.Tris-1, 2]
        # Get the deltas from node 0 to node 1 or node 2
        x01 = np.vstack((x[:,1]-x[:,0], y[:,1]-y[:,0], z[:,1]-z[:,0]))
        x02 = np.vstack((x[:,2]-x[:,0], y[:,2]-y[:,0], z[:,2]-z[:,0]))
        # Calculate the dimensioned normals
        n = np.cross(np.transpose(x01), np.transpose(x02))
        # Calculate the area of each triangle.
        A = np.fmax(1e-10, np.sqrt(np.sum(n**2, 1)))
        # Normalize each component.
        n[:,0] /= A
        n[:,1] /= A
        n[:,2] /= A
        # Save the areas.
        self.Areas = A/2
        # Save the unit normals.
        self.Normals = n
        
        
    # Get edge lengths
    def GetLengths(self):
        """Get the lengths of edges
        
        :Call:
            >>> tri.GetLengths()
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
        :Effects:
            *tri.Lengths*: :class:`numpy.ndarray`, shape=(tri.nTri,3)
                Length of edge of each triangle
        :Versions:
            * 2015-02-21 ``@ddalle``: First version
        """
        # Extract the vertices of each tri.
        x = self.Nodes[self.Tris-1, 0]
        y = self.Nodes[self.Tris-1, 1]
        z = self.Nodes[self.Tris-1, 2]
        # Get the deltas from node 0->1, 1->2, 2->1
        x01 = np.vstack((x[:,1]-x[:,0], y[:,1]-y[:,0], z[:,1]-z[:,0]))
        x12 = np.vstack((x[:,2]-x[:,1], y[:,2]-y[:,1], z[:,2]-z[:,1]))
        x20 = np.vstack((x[:,0]-x[:,2], y[:,0]-y[:,2], z[:,0]-z[:,2]))
        # Calculate lengths.
        self.Lengths = np.vstack((
            np.sqrt(np.sum(x01**2, 0)),
            np.sqrt(np.sum(x12**2, 0)),
            np.sqrt(np.sum(x20**2, 0)))).transpose()
   # }
    
    # +++++
    # Nodes
    # +++++
   # {
    # Get averaged normals at nodes
    def GetNodeNormals(self):
        """Get the area-averaged normals at each node
        
        :Call:
            >>> tri.GetNodeNormals()
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
        :Effects:
            *tri.NodeNormals*: :class:`np.ndarray`, shape=(tri.nNode,3)
                Unit normal at each node averaged from neighboring triangles
        :Versions:
            * 2016-01-23 ``@ddalle``: First version
        """
        # Ensure normals are present
        self.GetNormals()
        # Initialize node normals
        NN = np.zeros((self.nNode, 3))
        # Get areas
        TA = np.transpose([self.Areas, self.Areas, self.Areas])
        # Add in the weighted tri areas for each column of nodes in the tris
        NN[self.Tris[:,0]-1,:] += (self.Normals*TA)
        NN[self.Tris[:,1]-1,:] += (self.Normals*TA)
        NN[self.Tris[:,2]-1,:] += (self.Normals*TA)
        # Calculate the length of each of these vectors
        L = np.fmax(1e-10, np.sqrt(np.sum(NN**2, 1)))
        # Normalize.
        NN[:,0] /= L
        NN[:,1] /= L
        NN[:,2] /= L
        # Save it.
        self.NodeNormals = NN
   # }
   
    # +++++
    # Edges
    # +++++
   # {
    # Get edges
    def GetEdges(self):
        """Get the list of edges
        
        :Call:
            >>> tri.GetEdges()
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
                Triangulation instance
        :Effects:
            *tri.Edges*: :class:`np.ndarray`, shape=(3*nTri, 2)
                Array of node indices defining each edge
        :Versions:
            * 2016-09-29 ``@ddalle``: First version
        """
        # Check for edges
        try:
            self.Edges
            return
        except Exception:
            pass
        # Edges from 0->1, 1->2, 2->0 edges of each tri
        E = np.vstack((
            self.Tris[:,[0,1]],
            self.Tris[:,[1,2]],
            self.Tris[:,[2,0]]))
        # Sort by end node then start node
        I = np.lexsort((E[:,1], E[:,0]))
        # Save sorted edges
        self.Edges = E[I,:]
   # }
    
    # ++++++++++
    # Components
    # ++++++++++
   # {
    # Get normals and areas
    def GetCompArea(self, compID, n=None):
        """
        Get the total area of a component, or get the total area of a component
        projected to a plane with a given normal vector.
        
        :Call:
            >>> A = tri.GetCompArea(compID)
            >>> A = tri.GetCompArea(compID, n)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *compID*: :class:`int`
                Index of the component of which to find the area
            *n*: :class:`numpy.ndarray`
                Unit normal vector to use for projection
        :Outputs:
            *A*: :class:`float`
                Area of the component
        :Versions:
            * 2014-06-13 ``@ddalle``: First version
        """
        # Check for areas.
        try:
            self.Areas
        except AttributeError:
            self.GetNormals()
        # Find the indices of tris in the component.
        k = self.GetTrisFromCompID(compID)
        # Check for direction projection.
        if n is None:
            # No projection
            return np.sum(self.Areas[k])
        else:
            # Extract the normals and copy to new matrix.
            N = self.Normals[k].copy()
            # Dot those normals with the requested vector.
            N[:,0] *= n[0]
            N[:,1] *= n[1]
            N[:,2] *= n[2]
            # Sum to get the dot product.
            d = np.sum(N, 1)
            # Multiply this dot product by the area of each tri
            return np.sum(self.Areas[k] * d)
    
    # Get normals and areas
    def GetCompNormal(self, compID):
        """Get the area-averaged unit normal of a component
        
        :Call:
            >>> n = tri.GetCompNormal(compID)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *compID*: :class:`int`
                Index of the component of which to find the normal
        :Outputs:
            *n*: :class:`numpy.ndarray` shape=(3,)
                Area-averaged unit normal
        :Versions:
            * 2014-06-13 ``@ddalle``: First version
        """
        # Check for areas.
        try:
            self.Areas
        except AttributeError:
            self.GetNormals()
        # Find the indices of tris in the component.
        i = self.CompID == compID
        # Extract those normals and areas.
        N = self.Normals[i].copy()
        A = self.Areas[i].copy()
        # Weight the normals.
        N[:,0] *= A
        N[:,1] *= A
        N[:,2] *= A
        # Compute the mean.
        n = np.mean(N, 0)
        # Unitize.
        return n / np.sqrt(np.sum(n**2))
    
    # Get centroid of component
    def GetCompCentroid(self, compID):
        """Get the centroid of a component
        
        :Call:
            >>> [x, y] = tri.GetCompCentroid(compID)
            >>> [x, y, z] = tri.GetCompCentroid(compID)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *compID*: :class:`int`
                Index of the component of which to find the normal
        :Outputs:
            *x*: :class:`float`
                Coordinate of the centroid
            *y*: :class:`float`
                Coordinate of the centroid
            *z*: :class:`float`
                Coordinate of the centroid
        :Versions:
            * 2016-03-29 ``@ddalle``: First version
        """
        # Check for areas.
        try:
            self.Areas
        except AttributeError:
            self.GetNormals()
        # Get tris
        k = self.GetTrisFromCompID(compID)
        # Get corresponding nodes
        i = self.Tris[k,:] - 1
        # Get areas of those components
        A = self.Areas[k]
        # Total area
        AT = np.sum(A)
        # Dimensions
        nd = self.Nodes.shape[1]
        # Get coordinates
        if nd == 2:
            # 2D coordinates
            x = np.mean(self.Nodes[i,0], axis=1)
            y = np.mean(self.Nodes[i,1], axis=1)
            # Weighting
            xc = np.sum(x*A) / AT
            yc = np.sum(y*A) / AT
            # Output
            return np.array([xc, yc])
        else:
            # 3D coordinates
            x = np.mean(self.Nodes[i,0], axis=1)
            y = np.mean(self.Nodes[i,1], axis=1)
            z = np.mean(self.Nodes[i,2], axis=1)
            # Weighted averages
            xc = np.sum(x*A) / AT
            yc = np.sum(y*A) / AT
            zc = np.sum(z*A) / AT
            # Output
            return np.array([xc, yc, zc])
    
    # Function to add a bounding box based on a component and buffer
    def GetCompBBox(self, compID=[], **kwargs):
        """
        Find a bounding box based on the coordinates of a specified component
        or list of components, with an optional buffer or buffers in each
        direction
        
        :Call:
            >>> xlim = tri.GetCompBBox(compID, **kwargs)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *compID*: :class:`int` or :class:`str` or :class:`list`
                Component or list of components to use for bounding box
            *pad*: :class:`float`
                Buffer to add in each dimension to min and max coordinates
            *xpad*: :class:`float`
                Buffer to minimum and maximum *x*-coordinates
            *ypad*: :class:`float`
                Buffer to minimum and maximum *y*-coordinates
            *zpad*: :class:`float`
                Buffer to minimum and maximum *z*-coordinates
            *xp*: :class:`float`
                Buffer for the maximum *x*-coordinate
            *xm*: :class:`float`
                Buffer for the minimum *x*-coordinate
            *yp*: :class:`float`
                Buffer for the maximum *y*-coordinate
            *ym*: :class:`float`
                Buffer for the minimum *y*-coordinate
            *zp*: :class:`float`
                Buffer for the maximum *z*-coordinate
            *zm*: :class:`float`
                Buffer for the minimum *z*-coordinate
        :Outputs:
            *xlim*: :class:`numpy.ndarray` (:class:`float`), shape=(6,)
                List of *xmin*, *xmax*, *ymin*, *ymax*, *zmin*, *zmax*
        :Versions:
            * 2014-06-16 ``@ddalle``: First version
            * 2014-08-03 ``@ddalle``: Changed "buff" --> "pad"
        """
        # Process it into a list of component IDs.
        compID = self.GetCompID(compID)
        # Quit if none specified.
        if not compID: return None
        # Get the overall buffer.
        pad = kwargs.get('pad', 0.0)
        # Get the other buffers.
        xpad = kwargs.get('xpad', pad)
        ypad = kwargs.get('ypad', pad)
        zpad = kwargs.get('zpad', pad)
        # Get the directional buffers.
        xp = kwargs.get('xp', xpad)
        xm = kwargs.get('xm', xpad)
        yp = kwargs.get('yp', ypad)
        ym = kwargs.get('ym', ypad)
        zp = kwargs.get('zp', zpad)
        zm = kwargs.get('zm', zpad)
        # List of components; initialize with first.
        i = self.CompID == compID[0]
        # Loop through remaining components.
        for k in compID[1:]:
            i = np.logical_or(i, self.CompID == k)
        # Get the coordinates of each vertex of included tris.
        x = self.Nodes[self.Tris[i,:]-1, 0]
        y = self.Nodes[self.Tris[i,:]-1, 1]
        z = self.Nodes[self.Tris[i,:]-1, 2]
        # Get the extrema
        xmin = np.min(x) - xm
        xmax = np.max(x) + xp
        ymin = np.min(y) - ym
        ymax = np.max(y) + yp
        zmin = np.min(z) - zm
        zmax = np.max(z) + zp
        # Return the list.
        return np.array([xmin, xmax, ymin, ymax, zmin, zmax])
   
   # }
    
  # >
    
    # ==================
    # Edge/Curve Tracing
    # ==================
  # <
    # Get the closest node to a point
    def GetClosestNode(self, x):
        """Get the index of a node closest to a 3D point
        
        :Call:
            >>> i, L = tri.GetClosestNode(x)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
                Triangulation instance
            *x*: :class:`np.ndarray` shape=(3,)
                Coordinates of a point
        :Outputs:
            *i*: :class:`int`
                Index of closest node (1-based) to point *x*
            *L*: :class:`float`
                Value of the distance
        :Versions:
            * 2016-09-29 ``@ddalle``: First version
        """
        # Get deltas in each axis
        dx = self.Nodes[:,0] - x[0]
        dy = self.Nodes[:,1] - x[1]
        dz = self.Nodes[:,2] - x[2]
        # Get distances
        L = np.sqrt(dx*dx + dy*dy + dz*dz)
        # Find minimum
        i = np.argmin(L)
        # Output
        return i + 1, L[i]
        
    # Trace a curve
    def TraceCurve(self, Y, **kw):
        """Extract nodes along a piecewise linear curve
        
        :Call:
            >>> X = tri.TraceCurve(Y, **kw)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
                Triangulation instance
            *Y*: :class:`np.ndarray` shape=(n,3)
                List of points defining piecewise linear curve
            *dtol*: {``0.05``} | :class:`float`
                Maximum distance from curve as fraction of reference length
            *atol*: {``60.0``} | :class:`float`
                Maximum dot product between triangle edge and curve segment
        :Outputs:
            *X*: :class:`np.ndarray` shape=(m,3)
                Sequential list of nodes that trace a curve
        :Versions:
            * 2016-09-29 ``@ddalle``: First version
        """
        # Spatial tolerance
        dtol = kw.get('dtol', 0.05)
        # Find the node closest to the start of the curve
        icur, d0 = self.GetClosestNode(Y[0])
        # Characteristic length of the curve
        dy = np.max(np.max(Y, axis=0) - np.min(Y, axis=0))
        # Check for acceptable tolreance
        if d0/dy > dtol:
            return np.array([], dtype='int')
        # Initialize nodes
        I = np.zeros(5000, dtype='int')
        # Save first node
        ni = 1
        I[ni-1] = icur
        # Loop through the curve until no matching node on curve is found
        jcur = 0
        while icur is not None:
            # Set previous tolerance
            # Find the next node that lies on or near the curve
            icur, jcur = self.TraceCurve_NextNode(icur, Y, jcur, **kw)
            # Check for match
            if icur is None: break
            # Save the node and increase the count
            I[ni] = icur
            ni += 1
        # Check for trivial curves
        if ni == 1:
            return np.array([], dtype='int')
        # Return all indices
        return self.Nodes[I[:ni]-1,:]
        
        
    # Get next point on a curve
    def TraceCurve_NextNode(self, icur, Y, jcur, **kw):
        """Find the next node of the triangulation by following a curve
        
        :Call:
            >>> inew, jnew = tri.TraceCurve_NextNode(icur, Y, jcur, **kw)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
                Triangulation instance
            *icur*: :class:`int`
                Index (1-based) of current node
            *Y*: :class:`np.ndarray` shape=(n,3)
                List of points defining piecewise linear curve
            *jcur*: :class:`int`
                Number of curve segments to discount from search
            *dtol*: {``0.05``} | :class:`float`
                Maximum distance from curve as fraction of reference length
            *atol*: {``60.0``} | :class:`float`
                Maximum dot product between triangle edge and curve segment
        :Outputs:
            *inew*: :class:`int`
                Index (1-based) of next node along curve
            *jnew*: :class:`int`
                Number of curve segments to discount from next search
        :Versions:
            * 2016-09-29 ``@ddalle``: First version
        """
        # Direction tolerance
        atol = np.cos(kw.get('atol', 60.0) * np.pi/180)
        # Distance tolerance
        dtol = kw.get('dtol', 0.05)
        # Get the indices of neighboring nodes (leave zero-based)
        I = self.Edges[self.Edges[:,0]==icur, 1]
        # Get coordinates of neighboring nodes
        X = self.Nodes[I-1,:]
        # Current node
        x = self.Nodes[icur-1]
        # Initialize best find
        ds = 1e16
        d = 1e16
        # Number of points in curve
        nY = Y.shape[0] - 1
        # Check for last node
        if jcur >= nY:
            return None, None
        # Get vector of the first available segment of the curve
        dy0 = Y[jcur+1,:] - Y[jcur,:]
        Ly0 = np.sqrt(dy0[0]**2 + dy0[1]**2 + dy0[2]**2)
        # Loop through nodes
        for i in range(len(X)):
            # Get new point
            xi = X[i]
            # Vector from *x* to current point
            dxi = xi - x
            Lxi = np.sqrt(dxi[0]**2 + dxi[1]**2 + dxi[2]**2)
            # Check if we are going in the right direction
            if np.sum(dxi*dy0) / (Lxi*Ly0) < atol:
                # Do not check
                continue
            # Get distance from *xi* to the remaining curve points
            di, dsi, ji = self.TraceCurve_GetDistance(Y[jcur:,:], xi)
            # Length of edge (from *tri*)
            Li = np.sqrt(np.sum((xi-x)**2))
            # Normalized distance from curve to point
            di /= Li
            # Compare distance to tolerance
            if di > dtol:
                # Not close enough
                continue
            elif ji == 0:
                # Distance from *x* to *xi*
                dsi += np.sqrt(np.sum((xi-x)**2))
            else:
                # Distance from *x* to *Y[jcur]*
                dsi += np.sqrt(np.sum((Y[jcur+1]-x)**2))
                # Distance from *Y[jcur+ji]* to *xi*
                dsi += np.sqrt(np.sum((xi-Y[jcur+ji])**2))
            # Check distance
            if (dsi < ds) and (di < 1.5*d or di < 1e-5):
                # Update
                jnew = jcur + ji
                inew = I[i]
                # Save distances
                ds = dsi
                d = di
        # Check for a match
        if d < dtol:
            # Found new point
            return inew, jnew
        else:
            # No match
            return None, None
        
    # Get distance from curve and arc length
    def TraceCurve_GetDistance(self, Y, x, **kw):
        """Find distance between a generic curve and a point
        
        :Call:
            >>> d, ds, j = tri.TraceCurve_GetDistance(Y, x, **kw)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
                Triangulation instance
            *Y*: :class:`np.ndarray` shape=(n,3)
                List of points defining piecewise linear curve
            *x*: :class:`np.ndarray` shape=(3,)
                Test point
            *dtol*: {``0.05``} | :class:`float`
                Maximum distance from curve as fraction of reference length
            *atol*: {``60.0``} | :class:`float`
                Maximum dot product between triangle edge and curve segment
        :Outputs:
            *d*: :class:`float`
                Minimum distance from curve to *x*
            *ds*: :class:`float`
                Total arc length of curve segments before the closest point
            *j*: :class:`int`
                Index of segment in which closest point is located
        :Versions:
            * 2016-09-29 ``@ddalle``: First version
        """
        # Get distance from point to curve and length of each curve segment
        D = geom.DistancePointToCurve(x, Y)
        # Find minimum
        j = np.argmin(D)
        d = D[j]
        # Cumulative arc length of first *j* segments
        if j == 0:
            # No segments cut
            return d, 0.0, j
        elif j == 1:
            # One segment cut; no total segments cut
            ds = 0.0
        else:
            # Intervals
            dX = Y[2:j+1,:] - Y[1:j,:]
            # Lengths
            ds = np.sum(np.sqrt(np.sum(dX**2, axis=1)))
        # Add the distance from *Y[j]* to *x*
        ds += np.sqrt(np.sum((Y[j]-x)**2))
        # Output
        return d, ds, j
  # >
    
    # ========
    # Plotting
    # ========
  # <
    # Create a 3-view of a component (or list of) using TecPlot
    def Tecplot3View(self, fname, i=None):
        """Create a 3-view PNG of a component(s) using TecPlot
        
        :Call:
            >>> tri.Tecplot3View(fname, i=None)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: :class:`str`
                Created file is ``'%s.png' % fname``
            *i*: :class:`str` or :class:`int` or :class:`list` (:class:`int`)
                Component name, ID or list of component IDs
        :Versions:
            * 2015-01-23 ``@ddalle``: First version
        """
        # Get the subtriangulation.
        tri0 = self.GetSubTri(i)
        # Name of .tri file
        ftri = '%s.tri' % fname
        # Write triangulation file.
        tri0.Write(ftri)
        # Hide output.
        f = open('/dev/null', 'w')
        # Convert it to an STL.
        print("     Converting to STL: '%s' -> 'comp.stl'" % ftri)
        sp.call(['tri2stl', '-i', ftri, '-o', 'comp.stl'], stdout=f)
        # Cleanup.
        for fi in ['iso-comp.mcr', 'iso-comp.lay']:
            # Check for the file.
            if os.path.isfile(fi):
                # Delete it.
                os.remove(fi)
        # Copy the template layout file and macro.
        copy(os.path.join(TecFolder, 'iso-comp.lay'), '.')
        copy(os.path.join(TecFolder, 'iso-comp.mcr'), '.')
        # Get the command for tecplot
        t360 = GetTecplotCommand()
        # Create the image.
        print("     Creating image '%s.png' using `%s`" % (fname, t360))
        sp.call([t360, '-b', '-p', 'iso-comp.mcr'], stdout=f)
        # Close the output file.
        f.close()
        # Rename the PNG
        os.rename('iso-comp.png', '%s.png' % fname)
        # Cleanup.
        for f in ['iso-comp.mcr', 'iso-comp.lay', 'comp.stl']:
            # Check for the file.
            if os.path.isfile(f):
                # Delete it.
                os.remove(f)
    
    # Function to plot all components!
    def TecplotExplode(self):
        """
        Create a 3-view of each available named component in *tri.config* (read
        from :file:`Config.xml`) if available.  If not, create a 3-view plot for
        each *CompID*, e.g. :file:`1.png`, :file:`2.png`, etc.
        
        :Call:
            >>> tri.Tecplot3View(fname, i=None)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
        :Versions:
            * 2015-01-23 ``@ddalle``: First version
        """
        # Plot "entire.png"
        print("Plotting entire surface ...")
        print("    entire.png")
        # Create the 3-view using the name "entire" (much like Cart3D)
        self.TecPlot3View('entire', None)
        # Check for a config.
        try:
            # Appropriate status update.
            print("Plotting each named component in config ...")
            # Loop through named faces.
            for comp in self.config.faces:
                # Status update
                print("    %s.png" % comp)
                # Get the CompIDs for that face.
                k = self.config.GetCompID(comp)
                # Create the 3-view using that name.
                self.Tecplot3View(comp, k)
        except Exception:
            # Loop through CompID.
            print("FAILED.")
            print("Plotting each numbered CompID ...")
            # Loop through the available CompIDs
            for i in np.unique(self.CompID):
                # Status update.
                print("    %s.png" % i)
                # Create the 3-view plot for just that CompID==i
                self.Tecplot3View(i, i)
        
    
    # Create a surface view of a component using Paraview
    def ParaviewPlot(self, fname, i=None, r='x', u='y'):
        """Create a plot of the surface of one component using Paraview
        
        :Call:
            >>> tri.ParaviewPlot(fname, i=None, r='x', u='y')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: :class:`str`
                Created file is ``'%s.png' % fname``
            *i*: :class:`str` or :class:`int` or :class:`list` (:class:`int`)
                Component name, ID or list of component IDs
            *r*: :class:`str` | :class:`list` (:class:`int`)
                Axis pointing to the right in plot
            *u*: :class:`str` | :class:`list` (:class:`int`)
                Axis pointing upward in plot
        :Versions:
            * 2015-11-22 ``@ddalle``: First version
        """
        # Get the subtriangulation
        tri0 = self.GetSubTri(i)
        # Name of .tri and .stl files
        ftri = '%s.tri' % fname
        # Write the triangulation file
        tri0.Write(ftri)
        # Hide output
        f = open('/dev/null', 'w')
        # Convert to STL
        print("      Converting to STL: '%s' -> comp.stl'" % ftri)
        sp.call(['tri2stl', '-i', ftri, '-o', 'comp.stl'], stdout=f)
        # Cleanup if any old files
        for fi in ['cape_stl.py']:
            if os.path.isfile(fi): os.remove(fi)
        # Copy the template Paraview script
        copy(os.path.join(ParaviewFolder, 'cape_stl.py'), '.')
        # Create the image.
        print("      Creating image '%s.png' using `pvpython`" % fname)
        sp.call(['pvpython', 'cape_stl.py', str(r), str(u)], stdout=f)
        # Close null output file.
        fclose()
        # Rename the PNG.
        os.rename('cape_stl.png', '%s.png' % fname)
        # Cleanup.
        for f in ['cape_stl.py', 'comp.stl']:
            # Check for the file.
            if os.path.isfile(f):
                # Delete it.
                os.remove(f)
        
  # >
    
    # =====================
    # Geometry Manipulation
    # =====================
  # <
    # Function to translate the triangulation
    def Translate(self, *a, **kw):
        """Translate the nodes of a triangulation object
            
        The offset coordinates may be specified as individual inputs or a
        single vector of three coordinates.
        
        :Call:
            >>> tri.Translate(dR, compID)
            >>> tri.Translate(dx, dy, dz, compID=None)
            >>> tri.Translate(dy=dy, compID=None)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance to be translated
            *dR*: :class:`numpy.ndarray` | :class:`list`
                List of three coordinates to use for translation
            *dx*: :class:`float`
                *x*-coordinate offset
            *dy*: :class:`float`
                *y*-coordinate offset
            *dz*: :class:`float`
                *z*-coordinate offset
            *compID*: :class:`int` | :class:`str` | :class:`list`
                Component ID(s) to which to apply translation
        :Versions:
            * 2014-05-23 ``@ddalle``: First version
            * 2014-10-08 ``@ddalle``: Exported functionality to function
            * 2016-04-08 ``@ddalle``: Redid inputs
        """
        # Get component ID
        compID = kw.get('compID')
        # Check regular arguments
        if len(a) == 1:
            # Get first input
            dR = a[0]
            # Vector
            dx, dy, dz = tuple(dR)
            # No component ID
            compID = None
        elif len(a) == 2:
            # Vector
            R = a[0]
            # Components
            compID = a[1]
        elif len(a) == 3:
            # Get the values
            dR = a
            # No component ID
            compID = None
        elif len(a) == 4:
            # Vector
            dR = a[:3]
            # Components
            compID = a[3]
        elif len(a) == 0:
            # Defaults
            dR = [0.0, 0.0, 0.0]
            compID = None
        else:
            # Bad input count
            raise ValueError("Must use exactly 0 to 4 non-keyword inputs")
        # Check length and type of displacements
        if type(dR).__name__ not in ['list', 'ndarray']:
            # Not a vector
            raise TypeError("Single input must be a vector")
        elif len(dR) != 3:
            # Not a 3-vector
            raise ValueError("Single input vector must have three values")
        # Get the keyword-values
        dx = kw.get('dx', dR[0])
        dy = kw.get('dy', dR[1])
        dz = kw.get('dz', dR[2])
        # Process components
        compID = kw.get('compID', compID)
        # Process the node indices to be rotated.
        i = self.GetNodesFromCompID(compID)
        # Extract the points.
        X = self.Nodes[i,:]
        # Apply the translation.
        Y = geom.TranslatePoints(X, [dx, dy, dz])
        # Save the translated points.
        self.Nodes[i,:] = Y
        
    # Function to rotate a triangulation about an arbitrary vector
    def Rotate(self, v1, v2, theta, compID=None):
        """Rotate the nodes of a triangulation object.
        
        :Call:
            >>> tri.Rotate(v1, v2, theta)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance to be rotated
            *v1*: :class:`numpy.ndarray`, *shape* = (3,)
                Start point of rotation vector
            *v2*: :class:`numpy.ndarray`, *shape* = (3,)
                End point of rotation vector
            *theta*: :class:`float`
                Rotation angle in degrees
            *compID*: :class:`int` | :class:`str` | :class:`list`
                Component ID(s) to which to apply translation
        :Versions:
            * 2014-05-27 ``@ddalle``: First version
            * 2014-10-07 ``@ddalle``: Exported functionality to function
        """
        # Get the node indices.
        i = self.GetNodesFromCompID(compID)
        # Extract the points.
        X = self.Nodes[i,:]
        # Apply the rotation.
        Y = geom.RotatePoints(X, v1, v2, theta)
        # Save the rotated points.
        self.Nodes[i,:] = Y
  # >
    
# class TriBase


# Regular triangulation class
class Tri(TriBase):
    """Cape surface mesh interface
    
    This class provides an interface for a basic triangulation without
    surface data.  It can be created either by reading an ASCII file or
    specifying the data directly.
    
    When no component numbers are specified, the object created will label
    all triangles ``1``.
    
    :Call:
        >>> tri = cape.Tri(fname=fname, c=None)
        >>> tri = cape.Tri(surf=surf, c=None)
        >>> tri = cape.Tri(uh3d=uh3d, c=None)
        >>> tri = cape.Tri(unv=unv, c=None)
        >>> tri = cape.Tri(nNode=nNode, Nodes=Nodes, **kw)
    :Inputs:
        *fname*: :class:`str`
            Name of triangulation file to read (Cart3D format)
        *surf*: :class:`str`
            Name of AFLR3 surface file
        *uh3d*: :class:`str`
            Name of triangulation file (UH3D format)
        *unv*: :class:`str`
            Name of IDEAS surface triangulation file
        *c*: :class:`str`
            Name of configuration file (usually ``Config.xml``)
    :Keyword arguments:
        Data members can be defined directly using keyword arguments
    :Data members:
        *tri.nNode*: :class:`int`
            Number of nodes in triangulation
        *tri.Nodes*: :class:`np.ndarray` (:class:`float`), (*nNode*, 3)
            Matrix of *x,y,z*-coordinates of each node
        *tri.nTri*: :class:`int`
            Number of triangles in triangulation
        *tri.Tris*: :class:`np.ndarray` (:class:`int`), (*nTri*, 3)
            Indices of triangle vertex nodes
        *tri.CompID*: :class:`np.ndarray` (:class:`int`), (*nTri*,)
            Component number for each triangle
        *tri.BCs*: :class:`np.ndarray` (:class:`int`), (*nTri*,)
            Boundary condition flag for each triangle
        *tri.nQuad*: :class:`int`
            Number of quads in surface
        *tri.Quads*: :class:`np.ndarray` (:class:`int`), (*nQuad*, 4)
            Indices of quad vertex nodes
        *tri.CompIDQuad*: :class:`np.ndarray` (:class:`int`), (*nQuad*,)
            Component number for each quad
        *tri.BCsQuad*: :class:`np.ndarray` (:class:`int`), (*nQuad*,)
            Boundary condition flag for each quad
        *tri.blds*: :class:`np.ndaray` (:class:`float`), (*nNode*,)
            Boundary layer initial spacing for each node
        *tri.bldel*: :class:`np.ndarray` (:class:`float`), (*nNode*,)
            Boundary layer thicknesses for each node
    :Versions:
        * 2014-05-23 ``@ddalle``: First version
        * 2016-04-05 ``@ddalle``: Many input formats
    """
    # Initialization method
    def __init__(self, fname=None, c=None, **kw):
        """Initialization method
        
        :Versions:
            * 2014-05-23 ``@ddalle``: First version
            * 2014-06-02 ``@ddalle``: Added UH3D reading capability
            * 2016-04-05 ``@ddalle``: Added AFLR3 and cleaned up inputs
        """
        
        # Check if file is specified.
        if 'tri' in kw:
            # Read from file.
            self.Read(kw['tri'])
        elif 'uh3d' in kw:
            # Read from the UH3D format
            self.ReadUH3D(kw['uh3d'])
        elif 'surf' in kw:
            # Read from AFLR3 surface
            self.ReadSurf(kw['surf'])
        elif 'unv' in kw:
            # I don't know what's up with this format
            self.ReadUnv(kw['unv'])
        elif fname is not None:
            # Guess type from file extensions
            self.ReadBest(fname)
        else:
            # Process raw inputs.
            # Nodes, tris, and quads
            Nodes = kw.get('Nodes', np.zeros((0,3)))
            Tris  = kw.get('Tris',  np.zeros((0,3), dtype=int))
            Quads = kw.get('Quads', np.zeros((0,4), dtype=int))
            # Ensure arrays
            Nodes = np.array(Nodes)
            Tris  = np.array(Tris)
            Quads = np.array(Quads)
            # Number of nodes
            nNode = kw.get('nNode', Nodes.shape[0])
            nTri  = kw.get('nTri',  Tris.shape[0])
            nQuad = kw.get('nQuad', Quads.shape[0])
            # Component IDs
            self.CompID = kw.get('CompID')
            self.CompIDQuad = kw.get('CompIDQuad')
            # Boundary condition flags
            self.BCs = kw.get('BCs')
            self.BCsQuad = kw.get('BCsQuad')
            # BL growth parameters
            self.blds  = kw.get('blds',  np.zeros(nNode))
            self.bldel = kw.get('bldel', np.zeros(nNode))
            # Save the nodes
            self.nNode = nNode
            self.Nodes = Nodes
            # Save the triangle definitions
            self.nTri = nTri
            self.Tris = Tris
            # Save the quad definitions
            self.nQuad = nQuad
            self.Quads = Quads
        
        # Check for configuration
        if 'xml' in kw:
            # Read a Config.xml type of file
            self.ReadConfigXML(kw['xml'])
        elif 'json' in kw:
            # Read a Config.json type of file
            self.ReadConfigJSON(kw['json'])
        elif c is not None:
            # Read the configuration
            self.ReadConfig(c)
        # Check if we should apply it
        try:
            # Check for two opinions about how tris should be numbered
            self.Conf
            self.config
            # Use the explicity one
            self.ApplyConfig(self.config)
        except AttributeError:
            pass
        
    # Method that shows the representation of a triangulation
    def __repr__(self):
        """Return the string representation of a triangulation.
        
        This looks like ``<cape.tri.Tri(nNode=M, nTri=N)>``
        
        :Versions:
            * 2014-05-27 ``@ddalle``: First version
        """
        return '<cape.tri.Tri(nNode=%i, nTri=%i)>' % (self.nNode, self.nTri)
    
# class Tri


# Regular triangulation class
class Triq(TriBase):
    """Class for surface geometry with solution values at each point
    
    This class is based on the concept of Cart3D ``triq`` files, which are also
    utilized by some Overflow utilities, including ``overint``.
    
    :Call:
        >>> triq = cape.Triq(fname=fname, c=None)
        >>> triq = cape.Triq(Nodes=Nodes, Tris=Tris, CompID=CompID, q=q)
    :Inputs:
        *fname*: :class:`str`
            Name of triangulation file to read (Cart3D format)
        *c*: :class:`str`
            Name of configuration file (usually ``Config.xml``)
        *nNode*: :class:`int`
            Number of nodes in triangulation
        *Nodes*: :class:`np.ndarray` (:class:`float`), (*nNode*, 3)
            Matrix of *x,y,z*-coordinates of each node
        *nTri*: :class:`int`
            Number of triangles in triangulation
        *Tris*: :class:`np.ndarray` (:class:`int`), (*nTri*, 3)
            Indices of triangle vertex nodes
        *CompID*: :class:`np.ndarray`, (*nTri*)
            Component number for each triangle
        *nq*: :class:`int`
            Number of state variables at each node
        *q*: :class:`np.ndarray` (:class:`float`), (*nNode*, *nq*)
            State vector at each node
    :Data members:
        *triq.nNode*: :class:`int`
            Number of nodes in triangulation
        *triq.Nodes*: :class:`np.ndarray` (:class:`float`), (*nNode*, 3)
            Matrix of *x,y,z*-coordinates of each node
        *triq.nTri*: :class:`int`
            Number of triangles in triangulation
        *triq.Tris*: :class:`np.ndarray` (:class:`int`), (*nTri*, 3)
            Indices of triangle vertex nodes
        *triq.CompID*: :class:`np.ndarray` (:class:`int`), (*nTri*)
            Component number for each triangle
        *triq.nq*: :class:`int`
            Number of state variables at each node
        *triq.q*: :class:`np.ndarray` (:class:`float`), (*nNode*, *nq*)
            State vector at each node
        *triq.n*: :class:`int`
            Number of files averaged in this triangulation (used for weight)
    """
    # Initialization method
    def __init__(self, fname=None, n=1, nNode=None, Nodes=None, c=None,
        nTri=None, Tris=None, CompID=None, nq=None, q=None):
        """Initialization method
        
        :Versions:
            * 2014-05-23 ``@ddalle``: First version
            * 2014-06-02 ``@ddalle``: Added UH3D reading capability
        """
        # Check if file is specified.
        if fname is not None:
            # Read from file.
            self.Read(fname, n=n)
            
        else:
            # Process inputs.
            # Check counts.
            if nNode is None:
                # Get dimensions if possible.
                if Nodes is not None:
                    # Use the shape.
                    nNode = Nodes.shape[0]
                else:
                    # No nodes
                    nNode = 0
            # Check counts.
            if nTri is None:
                # Get dimensions if possible.
                if Tris is not None:
                    # Use the shape.
                    nTri = Tris.shape[0]
                else:
                    # No nodes
                    nTri = 0
            # Check state
            if nq is None:
                # Get dimensions if possible.
                if q is not None:
                    # Use the shape.
                    nq = q.shape[1]
                else:
                    # No states
                    nq = 0
            # Save the components.
            self.nNode = nNode
            self.Nodes = Nodes
            self.nTri = nTri
            self.Tris = Tris
            self.CompID = CompID
            self.nq = nq
            self.n = n
            self.q = q
            
        # Check for configuration
        if c is not None:
            self.config = Config(c)
            
        # End
        return None
        
    # Method that shows the representation of a triangulation
    def __repr__(self):
        """Return the string representation of a triangulation.
        
        This looks like ``<cape.tri.Triq(nNode=M, nTri=N)>``
        
        :Versions:
            * 2014-05-27 ``@ddalle``: First version
        """
        return '<cape.tri.Triq(nNode=%i, nTri=%i, nq=%i)>' % (
            self.nNode, self.nTri, self.nq)
        
    # Function to read a .triq file
    def Read(self, fname, n=1):
        """Read a q-triangulation file (from ``*.triq``)
        
        :Call:
            >>> triq.Read(fname)
            >>> triq.Read(fname, n=1)
        :Inputs:
            *triq*: :class:`cape.tri.Triq`
                Triangulation instance
            *fname*: :class:`str`
                Name of triangulation file to read
            *n*: :class:`int`
                Number of iterations used for weighted averaging
        :Versions:
            * 2015-09-14 ``@ddalle``: First version
        """
        # Open the file
        fid = open(fname, 'r')
        # Read the first line.
        line = fid.readline().strip()
        # Process the first line.
        try:
            # Three entries
            nNode, nTri, nq = (int(v) for v in line.split()[0:3])
        except Exception:
            # Not a TriQ file.
            nNode, nTri = (int(v) for v in line.split()[0:2])
            # No state
            nq = 0
        
        # Read the nodes.
        self.ReadNodes(fid, nNode)
        # Read the Tris.
        self.ReadTris(fid, nTri)
        # Read or assign component IDs.
        self.ReadCompID(fid)
        # Read the sate.
        self.ReadQ(fid, nNode, nq)
        
        # Close the file.
        fid.close()
        
        # Weight: number of files included in file
        self.n = n
        
    # Function to write a .triq file
    def Write(self, fname):
        """Write a q-triangulation ``.triq`` file
        
        :Call:
            >>> triq.Write(fname)
        :Inputs:
            *triq*: :class:`cape.tri.Triq`
                Triangulation instance
            *fname*: :class:`str`
                Name of triangulation file to write
        :Versions:
            * 2015-09-14 ``@ddalle``: First version
        """
        self.WriteTriq(fname)
        
    # Function to calculate weighted average.
    def WeightedAverage(self, triq):
        """Calculate weighted average with a second triangulation
        
        :Call:
            >>> triq.WeightedAverage(triq2)
        :Inputs:
            *triq*: :class:`cape.tri.Triq`
                Triangulation instance
            *triq2*: class:`cape.tri.Triq`
                Second triangulation instance
        :Versions:
            * 2015-09-14 ``@ddalle``: First version
        """
        # Check consistency.
        if self.nNode != triq.nNode:
            raise ValueError("Triangulations must have same number of nodes.")
        elif self.nTri != triq.nTri:
            raise ValueError("Triangulations must have same number of tris.")
        elif self.n > 0 and self.nq != triq.nq:
            raise ValueError("Triangulations must have same number of states.")
        # Degenerate case.
        if self.n == 0:
            # Use the second input.
            self.q = triq.q
            self.n = triq.n
            self.nq = triq.nq
        # Weighted average
        self.q = (self.n*self.q + triq.n*triq.q) / (self.n+triq.n)
        # Update count.
        self.n += triq.n
        
    
# class Triq


# Function to read .tri files
def ReadTri(fname):
    """Read a basic triangulation file
    
    :Call:
        >>> tri = cape.ReadTri(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of `.tri` file to read
    :Outputs:
        *tri*: :class:`cape.tri.Tri`
            Triangulation instance
    :Examples:
        >>> tri = cape.ReadTri('bJet.i.tri')
        >>> tri.nNode
        92852
    :Versions:
        * 2014-05-27 ``@ddalle``: First version
    """
    # Create the tri object and return it.
    return Tri(fname)
    
    
# Global function to write a triangulation (just calls tri method)
def WriteTri(fname, tri):
    """Write a triangulation instance to file
    
    :Call:
        >>> cape.WriteTri(fname, tri)
    :Inputs:
        *fname*: :class:`str`
            Name of `.tri` file to read
        *tri*: :class:`cape.tri.Tri`
            Triangulation instance
    :Examples:
        >>> tri = cape.ReadTri('bJet.i.tri')
        >>> cape.WriteTri('bjet2.tri', tri)
    :Versions:
        * 2014-05-23 ``ddalle``: First version
    """
    # Call the triangulation's write method.
    tri.Write(fname)
    return None
# def WriteTri
    
