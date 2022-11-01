# -*- coding: utf-8 -*-
r"""
:mod:`cape.tri`: Surface triangulation module
=============================================

This module provides the utilities for interacting with Cart3D or
Plot3D type triangulations, including annotated triangulations
(including ``.triq`` files). Triangulations can also be read from
several other formats, including UH3D, UNV, and AFLR3 surf formats.

The module consists of individual classes that are built off of a base
triangulation class :class:`TriBase`. Methods that are written for the
:class:`TriBase` class apply to all other classes as well.

Some triangulation methods are written in Python/C using the
:mod:`cape._cape` module. For some repeated tasks (especially writing
triangulations to file), creating a compiled version can lead to
significant time savings. These are relatively simple to compile, but
fall-back methods are provided using purely Python code in each case.
The convention used for this situation is to provide
a method like :func:`TriBase.WriteFast` for the compiled version and
:func:`TriBase.WriteSlow` for the Python version.

"""

# Standard library
import os
import subprocess as sp
import sys
from datetime import datetime
from shutil import copy
from collections import OrderedDict

# Third-party modules
import numpy as np

# Local inputs
from . import io
from . import geom
from . import util
from .cfdx import options
from .cfdx import volcomp
from .tnakit import plot_mpl as pmpl
from .config import ConfigXML, ConfigJSON, ConfigMIXSUR
from .cgns import CGNS


# Constants
INT_TYPES = (int, np.int64, np.int32)

# Default tolerances for mapping triangulations
atoldef = options.rc.get("atoldef", 1e-2)
rtoldef = options.rc.get("rtoldef", 1e-4)
ctoldef = options.rc.get("ctoldef", 1e-4)
ztoldef = options.rc.get("ztoldef", 5e-2)
antoldef = options.rc.get("antoldef", 2e-2)
rntoldef = options.rc.get("rntoldef", 1e-4)
cntoldef = options.rc.get("cntoldef", 1e-4)
rztoldef = options.rc.get("rztoldef", 1e-5)


# Attempt to load the compiled helper module.
try:
    if sys.version_info.major == 2:
        # Python 2 extension
        import _cape2 as _cape
    else:
        # Python 3 extension
        import _cape3 as _cape
except ImportError:
    # No module
    _cape = None


# Function to get a non comment line
def _readline(f, comment='#'):
    r"""Read line that is nonempty and not a comment

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
        * 2015-11-19 ``@ddalle``: Version 1.0
    """
    # Read a line.
    line = f.readline()
    # Check for empty line (EOF)
    if line == '': return line
    # Process stripped line
    lstrp = line.strip()
    # Check if otherwise empty or a comment
    while (lstrp == '') or lstrp.startswith(comment):
        # Read the next line.
        line = f.readline()
        # Check for empty line (EOF)
        if line == '':
            return line
        # Process stripped line
        lstrp = line.strip()
    # Return the line.
    return line


# Function to read a single triangulation file
def ReadTriFile(fname, fmt=None):
    r"""Read a single triangulation file

    :Call:
        >>> tri = ReadTriFile(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of Cart3D tri, IDEAS unv, UH3D, or AFLR3 surf file
        *fmt*: {``None``} | ``"tri"`` | ``"uh3d"`` | :class:`str`
            Format to use; by default determine from the file extension
    :Outputs:
        *tri*: :class:`cape.tri.Tri`
            Triangulation
    :Versions:
        * 2016-04-06 ``@ddalle``: Version 1.0
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


# Triangulation class
class TriBase(object):
    r"""Cape base triangulation class

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
        * 2014-05-23 ``@ddalle``: Version 1.0
        * 2014-06-02 ``@ddalle``: Added UH3D reading capability
        * 2015-11-19 ``@ddalle``: Added AFLR3 surface capability
    """
  # ======
  # Config
  # ======
  # <
    # Initialization method
    def __init__(
            self, fname=None, c=None,
            tri=None, uh3d=None, surf=None, unv=None, cgns=None,
            xml=None, json=None, mixsur=None, uh3dc=None,
            nNode=None, Nodes=None, nTri=None, Tris=None,
            nQuad=None, Quads=None, CompID=None):
        r"""Initialization method

        :Versions:
        * 2014-05-23 ``@ddalle``: Version 1.0
        * 2014-06-02 ``@ddalle``: V1.1: Add UH3D reading capability
        * 2015-11-19 ``@ddalle``: V2.0: Add XML reading and AFLR3 surfs
        """
        # Save file name
        self.fname = fname
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
        elif cgns is not None:
            # Read CGNS surface format
            self.ReadCGNS(cgns)
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
            self.Quads = Quads
            self.CompID = CompID

        # Check for UH3D component list
        if uh3dc is not None:
            # Read a component list like from UH3D
            self.ReadUH3DCompIDList(uh3dc)

        # Check for configuration
        if xml is not None:
            # Read a Config.xml type of file
            self.ReadConfigXML(c)
        elif json is not None:
            # Read a Config.json type of file
            self.ReadConfigJSON(c)
        elif mixsur is not None:
            # Read a mixsur.inp type of file
            self.ReadConfigMIXSUR(mixsur)
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
            * 2014-05-27 ``@ddalle``: Version 1.0
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
            * 2016-10-21 ``@ddalle``: Version 1.0
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
        elif fext == 'cgns':
            # Read CGNS file
            self.ReadCGNS(fname)
        elif fext == 'triq':
            # Read triq file
            self.ReadTriQ(fname)
        else:
            # Assume Cart3D triangulation file
            self.Read(fname)

    # Function to copy a triangulation and unlink it.
    def Copy(self):
        r"""Copy a triangulation and unlink it

        :Call:
            >>> tri2 = tri.Copy()
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
        :Outputs:
            *tri2*: :class:`cape.tri.Tri`
                Triangulation with same values as *tri* but not linked
        :Versions:
            * 2014-06-12 ``@ddalle``: Version 1.0
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
            tri.nq = tri.q.shape[1]
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
            * 2014-06-02 ``@ddalle``: Version 1.0
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
        # Ensure quads are present
        try:
            self.nQuad
        except AttributeError:
            self.nQuad = 0

    # Function to read a .triq file
    def ReadTriQ(self, fname, n=1):
        """Read an annotated triangulation file (``.triq``)

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
            *n*: {``1``} | positive :class:`int`
                Number of snapshots averaged into ``triq`` file
        :Versions:
            * 2017-01-11 ``@ddalle``: Points to :func:`ReadTri`
        """
        # Use previous function
        self.Read(fname, n=n)

    # Function to read a .tri file
    def ReadASCII(self, fname, n=1):
        r"""Read a triangulation file from an ASCII file

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
            * 2014-06-02 ``@ddalle``: Version 1.0
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
        self.Quads = np.zeros((0, 4))

        # Save extension
        self.ext = 'ascii'

        # Weight: number of files included in file
        self.n = n

    # Read TRI file as a binary file
    def ReadTriBin(self, fname, ni=4, nf=4):
        r"""Read binary unformatted triangulation file

        :Call:
            >>> tri.ReadTriBin(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-08-18 ``@ddalle``: Version 1.0
        """
        # Open the file for binary reading
        fid = open(fname, 'rb')
        # Get the byte order and precision
        try:
            bo = self.byteorder
            ni = self.bytecount
        except AttributeError:
            bo = sys.byteorder
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
        r = R[0] // ni
        # Read header line; also read
        H = np.array(np.fromfile(fid, count=r+2, dtype=fi), dtype='int')
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
        P = np.array(P, dtype='float')
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
        T = np.array(T, dtype='int')
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
        C = np.fromfile(fid, count=self.nTri, dtype=fi)
        # Convert single-to-double if necessary
        self.CompID = np.array(C, dtype='int')
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
        q = np.fromfile(fid, count=self.nNode*self.nq, dtype=ff)
        # Reshape and save
        self.q = np.array(q.reshape((self.nNode, self.nq)), dtype='float')
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
        r"""Get the byte order and precision for a TRI file

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
            * 2016-08-18 ``@ddalle``: Version 1.0
        """
        # Open the file; attempt a binary read
        fid = open(fname, 'rb')
        # Read the first entry as a double-precision little-endian int
        r, = np.fromfile(fid, count=1, dtype='<i4')
        # Check for success
        if r >= 8 and r <= 12:
            # Little-endian success
            self.filetype  = 'binary'
            self.byteorder = 'little'
            # Read the number of nodes (comma unpacks the 1-element array)
            nNode, = np.fromfile(fid, count=1, dtype='<i4')
            # Finish the record
            np.fromfile(fid, count=r//4, dtype='<i4')
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
            np.fromfile(fid, count=r//4, dtype='>i4')
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
            raise ValueError(
                "File did not match any of the following types:\n"
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
            *tri*: :class:`cape.tri.TriBase`
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
            * 2014-06-16 ``@ddalle``: Version 1.0
        """
        # Save the node count.
        self.nNode = nNode
        # Read the nodes.
        Nodes = np.fromfile(f, dtype=float, count=nNode*3, sep=" ")
        # Reshape into a matrix.
        self.Nodes = Nodes.reshape((nNode, 3))

    # Function to read node coordinates from .triq+ file
    def ReadNodesSurf(self, f, nNode):
        """Read node coordinates from an AFLR3 ``.surf`` file

        :Call:
            >>> tri.ReadNodesSurf(f, nNode)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
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
            * 2016-04-05 ``@ddalle``: Version 1.0
        """
        # Save the node count.
        self.nNode = nNode
        # Read the nodes.
        Nodes = np.fromfile(f, dtype=float, count=nNode*5, sep=" ")
        # Reshape into a matrix.
        Nodes = Nodes.reshape((nNode, 5))
        # Save nodes
        self.Nodes = Nodes[:, :3]
        # Save boundary layer spacings
        self.blds = Nodes[:, 3]
        # Save boundary layer thicknesses
        self.bldel = Nodes[:, 4]
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
            *tri*: :class:`cape.tri.TriBase`
                Triangulation instance
            *f*: :class:`file`
                Open file handle
            *nTri*: :class:`int`
                Number of tris to read
        :Effects:
            Reads and creates *tri.Tris*; file remains open.
        :Versions:
            * 2014-06-16 ``@ddalle``: Version 1.0
        """
        # Save the tri count.
        self.nTri = nTri
        # Read the Tris
        Tris = np.fromfile(f, dtype=int, count=nTri*3, sep=" ")
        # Reshape into a matrix.
        self.Tris = Tris.reshape((nTri, 3))

    # Function to read triangles from .surf file
    def ReadTrisSurf(self, f, nTri):
        """Read triangle node indices, comp IDs, and BCs from AFLR3 file

        :Call:
            >>> tri.ReadTrisSurf(f, nTri)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
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
            * 2016-04-05 ``@ddalle``: Version 1.0
        """
        # Save the tri count
        self.nTri = nTri
        # Exit if no tris
        if nTri == 0:
            self.Tris = np.zeros((0, 3))
            self.CompID = np.zeros(0, dtype=int)
            self.BCs = np.zeros(0, dtype=int)
            return
        # Read the tris
        Tris = np.fromfile(f, dtype=int, count=nTri*6, sep=" ")
        # Reshape into a matrix
        Tris = Tris.reshape((nTri, 6))
        # Save the triangles
        self.Tris = Tris[:, :3]
        # Save the component IDs.
        self.CompID = Tris[:, 3]
        # Save the boundary conditions.
        self.BCs = Tris[:, 5]

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
            *tri*: :class:`cape.tri.TriBase`
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
            * 2016-04-05 ``@ddalle``: Version 1.0
        """
        # Save the tri count
        self.nQuad = nQuad
        # Exit if no tris
        if nQuad == 0:
            self.Quads = np.zeros((0, 3))
            self.CompIDQuad = np.zeros(0, dtype=int)
            self.BCsQuad = np.zeros(0, dtype=int)
            return
        # Read the tris
        Quads = np.fromfile(f, dtype=int, count=nQuad*7, sep=" ")
        # Reshape into a matrix
        Quads = Quads.reshape((nQuad, 7))
        # Save the triangles
        self.Quads = Quads[:, :4]
        # Save the component IDs.
        self.CompIDQuad = Quads[:, 4]
        # Save the boundary conditions.
        self.BCsQuad = Quads[:, 6]
   # }

   # ++++++++
   # Comp IDs
   # ++++++++
   # {
    # Function to read the component identifiers
    def ReadCompID(self, f):
        r"""Read component IDs from a .tri file.

        :Call:
            >>> tri.ReadCompID(f)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
                Triangulation instance
            *f*: :class:`str`
                Open file handle
        :Effects:
            Reads and creates *tri.CompID* if not at end of file.
            Otherwise all components are labeled ``1``.
        :Versions:
            * 2014-06-16 ``@ddalle``: Version 1.0
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
        r"""Read node states from a ``.triq`` file.

        :Call:
            >>> triq.ReadQ(f, nNode, nq)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
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
            * 2015-09-14 ``@ddalle``: Version 1.0
        """
        # Save the state count.
        self.nq = nq
        # Check for null case
        if nq is None: return
        # Read the nodes.
        q = np.fromfile(f, dtype=float, count=nNode*nq, sep=" ")
        # Reshape into a matrix.
        self.q = q.reshape((nNode, nq))
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
        r"""Write triangulation to file using fastest method available

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
            *fmt*: {``None``} | ``"ascii"`` | ``"b4"`` | ``"lb4"``
                Format specified by text
            *b4*: ``True`` | {``False``}
                Whether or not to use single-precision big-endian
            *lb4*: ``True`` | {``False``}
                Whether or not to use single-precision little-endian
            *b8*: ``True`` | {``False``}
                Whether or not to use double-precision big-endian
            *lb8*: ``True`` | {``False``}
                Whether or not to use double-precision little-endian
            *r4*: ``True`` | {``False``}
                Whether or not to use single-precision big-endian Fortran
            *lr4*: ``True`` | {``False``}
                Whether or not to use single-precision little-endian Fortran
            *r8*: ``True`` | {``False``}
                Whether or not to use double-precision big-endian Fortran
            *lr8*: ``True`` | {``False``}
                Whether or not to use double-precision little-endian Fortran
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
            * 2014-05-23 ``@ddalle``: Version 1.0
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
        elif ext == 'lr4':
            # Little-endian single
            self.WriteTri_lr4(fname)
        elif ext == 'r4':
            # Big-endian single
            self.WriteTri_r4(fname)
        elif ext == 'lr8':
            # Little-endian double
            self.WriteTri_lr8(fname)
        elif ext == 'r8':
            # Big-endian double
            self.WriteTri_r8(fname)
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
            ``"r4"``           Big-endian, single-precision record
            ``"lr4"``          Little-endian, single-precision record
            ``"r8"``           Big-endian, double-precision record
            ``"lr8"``          Little-endian, double-precision record
            ``"b4"``           Big-endian, single-precision
            ``"lb4"``          Little-endian, single-precision
            ``"b8"``           Big-endian, double-precision
            ``"lb8"``          Little-endian, double-precision
            =================  =======================================

        :Call:
            >>> ext = tri.GetOutputFileType(**kw)
        :Inputs:
            *fmt*: {``None``} | ``"ascii"`` | ``"b4"`` | ``"lb4"``
                Format specified by text
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
            *r4*: ``True`` | {``False``}
                Whether or not to use single-precision big-endian Fortran
            *lr4*: ``True`` | {``False``}
                Whether or not to use single-precision little-endian Fortran
            *r8*: ``True`` | {``False``}
                Whether or not to use double-precision big-endian Fortran
            *lr8*: ``True`` | {``False``}
                Whether or not to use double-precision little-endian Fortran
            *record*, *fortran*: {``True``} | ``False``
                Whether or not to use Fortran-style record markers
            *byteorder*, *endian*: ``"big"`` | ``"little"`` | {``None``}
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
            * 2016-10-02 ``@ddalle``: Version 1.0
        """
        # Check for format flag
        if 'fmt' in kw:
            # Return lower format
            return kw["fmt"].lower()
        # Check for easy cases
        if kw.get('ascii'):
            # ASCII file
            return 'ascii'
        elif kw.get('r4'):
            # Big-endian single-precision
            return 'r4'
        elif kw.get('lr4'):
            # Little-endian single precision
            return 'lr4'
        elif kw.get('r8'):
            # Big-endian double-precision
            return 'r8'
        elif kw.get('lr8'):
            # Little-endian double precision
            return 'lr8'
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
        fo = kw.get('fortran', kw.get('record'))
        # Check for at least one binary flag
        if (not kw.get('bin') and
                bo is None and bs is None and bc is None):
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
                if fo:
                    return 'r8'
                else:
                    return 'b8'
            elif bc == 4:
                if fo:
                    return 'r4'
                else:
                    return 'b4'
            else:
                raise ValueError("Could not interpret bytecount '%s'" % bc)
        elif bo == "little":
            if bc == 8:
                if fo:
                    return 'lr8'
                else:
                    return 'lb8'
            elif bc == 4:
                if fo:
                    return 'lr4'
                else:
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
            * 2014-05-23 ``@ddalle``: Version 1.0
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
            * 2015-01-03 ``@ddalle``: Version 1.0
        """
        # Write the nodes.
        _cape.WriteTri(self.Nodes, self.Tris)
        # Write the component IDs.
        _cape.WriteCompID(self.CompID)
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
            * 2014-05-23 ``@ddalle``: Version 1.0
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
            * 2016-10-10 ``@ddalle``: Version 1.0
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
            * 2016-10-10 ``@ddalle``: Version 1.0
        """
        #
        _cape.WriteTri_lb4(self.Nodes, self.Tris, self.CompID)
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
            * 2016-09-05 ``@ddalle``: Version 1.0
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
            io.write_record_lr4_i(fid, [self.nNode, self.nTri, self.nq])
        else:
            # No q values
            io.write_record_lr4_i(fid, [self.nNode, self.nTri])
        # Write the nodes, tris, and compIDs
        io.write_record_lr4_f(fid, self.Nodes)
        io.write_record_lr4_i(fid, self.Tris)
        io.write_record_lr4_i(fid, self.CompID)
        # Write states if appropriate
        if qq: io.write_record_lr4_f(fid, self.q)
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
            * 2016-10-10 ``@ddalle``: Version 1.0
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
            * 2016-10-10 ``@ddalle``: Version 1.0
        """
        #
        _cape.WriteTri_b4(self.Nodes, self.Tris, self.CompID)
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
            * 2016-09-05 ``@ddalle``: Version 1.0
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
            io.write_record_r4_i(fid, [self.nNode, self.nTri, self.nq])
        else:
            # No q values
            io.write_record_r4_i(fid, [self.nNode, self.nTri])
        # Write the nodes, tris, and compIDs
        io.write_record_r4_f(fid, self.Nodes)
        io.write_record_r4_i(fid, self.Tris)
        io.write_record_r4_i(fid, self.CompID)
        # Write states if appropriate
        if qq: io.write_record_r4_f(fid, self.q)
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
            * 2016-10-10 ``@ddalle``: Version 1.0
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
            * 2016-10-10 ``@ddalle``: Version 1.0
        """
        #
        _cape.WriteTri_lb8(self.Nodes, self.Tris, self.CompID)
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
            * 2016-09-05 ``@ddalle``: Version 1.0
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
            io.write_record_lr4_i(fid, [self.nNode, self.nTri, self.nq])
        else:
            # No q values
            io.write_record_lr4_i(fid, [self.nNode, self.nTri])
        # Write the nodes, tris, and compIDs
        io.write_record_lr8_f(fid, self.Nodes)
        io.write_record_lr4_i(fid, self.Tris)
        io.write_record_lr4_i(fid, self.CompID)
        # Write states if appropriate
        if qq: io.write_record_lr8_f(fid, self.q)
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
            * 2016-10-10 ``@ddalle``: Version 1.0
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
            * 2016-10-10 ``@ddalle``: Version 1.0
        """
        #
        _cape.WriteTri_b8(self.Nodes, self.Tris, self.CompID)
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
            * 2016-09-05 ``@ddalle``: Version 1.0
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
            io.write_record_r4_i(fid, [self.nNode, self.nTri, self.nq])
        else:
            # No q values
            io.write_record_r4_i(fid, [self.nNode, self.nTri])
        # Write the nodes, tris, and compIDs
        io.write_record_r8_f(fid, self.Nodes)
        io.write_record_r4_i(fid, self.Tris)
        io.write_record_r4_i(fid, self.CompID)
        # Write states if appropriate
        if qq: io.write_record_r8_f(fid, self.q)
        # Close the file
        fid.close()

    # Write TRI file as lr4 file
    def WriteTri_lr4(self, fname):
        """Write a triangulation as a little-endian single-precision file

        :Call:
            >>> tri.WriteTri_lr4(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-10-10 ``@ddalle``: Version 1.0
        """
        try:
            # Compiled (C) version
            self.WriteFast_lr4(fname)
        except Exception:
            # Python fall-back function
            self.WriteSlow_lr4(fname)

    # Write TRI file as little-endian single-precision
    def WriteFast_lr4(self, fname='Components.i.tri'):
        """Use compiled C code to write single-precision little-endian tri

        :Call:
            >>> tri.WriteFast_lr4(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-10-10 ``@ddalle``: Version 1.0
        """
        #
        _cape.WriteTri_lr4(self.Nodes, self.Tris, self.CompID)
        # Check the file name and rename if necessary
        if fname != "Components.pyCart.tri":
            os.rename("Components.pyCart.tri", fname)

    # Write TRI file as little-endian single-precision
    def WriteSlow_lr4(self, fname='Components.i.tri'):
        """Use Python code to write single-precision little-endian tri

        :Call:
            >>> tri.WriteSlow_lr4(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-09-05 ``@ddalle``: Version 1.0
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
            io.write_record_lr4_i(fid, [self.nNode, self.nTri, self.nq])
        else:
            # No q values
            io.write_record_lr4_i(fid, [self.nNode, self.nTri])
        # Write the nodes, tris, and compIDs
        io.write_record_lr4_f(fid, self.Nodes)
        io.write_record_lr4_i(fid, self.Tris)
        io.write_record_lr4_i(fid, self.CompID)
        # Write states if appropriate
        if qq: io.write_record_lr4_f(fid, self.q)
        # Close the file
        fid.close()

    # Write TRI file as b4 file
    def WriteTri_r4(self, fname):
        """Write a triangulation as a big-endian single-precision file

        :Call:
            >>> tri.WriteTri_r4(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-10-10 ``@ddalle``: Version 1.0
        """
        try:
            # Compiled (C) version
            self.WriteFast_r4(fname)
        except Exception:
            # Python fall-back function
            self.WriteSlow_r4(fname)

    # Write TRI file as big-endian single-precision
    def WriteFast_r4(self, fname='Components.i.tri'):
        """Use compiled C code to write single-precision big-endian tri

        :Call:
            >>> tri.WriteFast_lb4(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-10-10 ``@ddalle``: Version 1.0
        """
        #
        _cape.WriteTri_r4(self.Nodes, self.Tris, self.CompID)
        # Check the file name and rename if necessary
        if fname != "Components.pyCart.tri":
            os.rename("Components.pyCart.tri", fname)

    # Write TRI file as big-endian single-precision
    def WriteSlow_r4(self, fname='Components.i.tri'):
        """Use compiled code to write single-precision big-endian tri

        :Call:
            >>> tri.WriteSlow_r4(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-09-05 ``@ddalle``: Version 1.0
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
            io.write_record_r4_i(fid, [self.nNode, self.nTri, self.nq])
        else:
            # No q values
            io.write_record_r4_i(fid, [self.nNode, self.nTri])
        # Write the nodes, tris, and compIDs
        io.write_record_r4_f(fid, self.Nodes)
        io.write_record_r4_i(fid, self.Tris)
        io.write_record_r4_i(fid, self.CompID)
        # Write states if appropriate
        if qq: io.write_record_r4_f(fid, self.q)
        # Close the file
        fid.close()

    # Write TRI file as lb8 file
    def WriteTri_lr8(self, fname):
        """Write a triangulation as a little-endian double-precision file

        :Call:
            >>> tri.WriteTri_lr8(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-10-10 ``@ddalle``: Version 1.0
        """
        try:
            # Compiled (C) version
            self.WriteFast_lr8(fname)
        except Exception:
            # Python fall-back function
            self.WriteSlow_lr8(fname)

    # Write TRI file as little-endian double-precision
    def WriteFast_lr8(self, fname='Components.i.tri'):
        """Use compiled C code to write double-precision little-endian tri

        :Call:
            >>> tri.WriteFast_lr8(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-10-10 ``@ddalle``: Version 1.0
        """
        #
        _cape.WriteTri_lr8(self.Nodes, self.Tris, self.CompID)
        # Check the file name and rename if necessary
        if fname != "Components.pyCart.tri":
            os.rename("Components.pyCart.tri", fname)

    # Write TRI file as little-endian double-precision
    def WriteSlow_lr8(self, fname='Components.i.tri'):
        """Use Python code to write double-precision little-endian tri

        :Call:
            >>> tri.WriteSlow_lr8(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-09-05 ``@ddalle``: Version 1.0
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
            io.write_record_lr4_i(fid, [self.nNode, self.nTri, self.nq])
        else:
            # No q values
            io.write_record_lr4_i(fid, [self.nNode, self.nTri])
        # Write the nodes, tris, and compIDs
        io.write_record_lr8_f(fid, self.Nodes)
        io.write_record_lr4_i(fid, self.Tris)
        io.write_record_lr4_i(fid, self.CompID)
        # Write states if appropriate
        if qq: io.write_record_lr8_f(fid, self.q)
        # Close the file
        fid.close()

    # Write TRI file as b8 file
    def WriteTri_r8(self, fname):
        """Write a triangulation as a big-endian double-precision file

        :Call:
            >>> tri.WriteTri_r8(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-10-10 ``@ddalle``: Version 1.0
        """
        try:
            # Compiled (C) version
            self.WriteFast_r8(fname)
        except Exception:
            # Python fall-back function
            self.WriteSlow_r8(fname)

    # Write TRI file as big-endian single-precision
    def WriteFast_r8(self, fname='Components.i.tri'):
        """Use compiled C code to write double-precision big-endian tri

        :Call:
            >>> tri.WriteFast_r8(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-10-10 ``@ddalle``: Version 1.0
        """
        #
        _cape.WriteTri_r8(self.Nodes, self.Tris, self.CompID)
        # Check the file name and rename if necessary
        if fname != "Components.pyCart.tri":
            os.rename("Components.pyCart.tri", fname)

    # Write TRI file as big-endian double-precision
    def WriteSlow_r8(self, fname='Components.i.tri'):
        """Use Python code to write double-precision big-endian tri

        :Call:
            >>> tri.WriteSlow_r8(fname='Components.i.tri')
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangultion instance to be translated
            *fname*: {``'Components.i.tri'``} | :class:`str`
                Name of file to write
        :Versions:
            * 2016-09-05 ``@ddalle``: Version 1.0
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
            io.write_record_r4_i(fid, [self.nNode, self.nTri, self.nq])
        else:
            # No q values
            io.write_record_r4_i(fid, [self.nNode, self.nTri])
        # Write the nodes, tris, and compIDs
        io.write_record_r8_f(fid, self.Nodes)
        io.write_record_r4_i(fid, self.Tris)
        io.write_record_r4_i(fid, self.CompID)
        # Write states if appropriate
        if qq: io.write_record_r8_f(fid, self.q)
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
        r"""Read a triangulation file (from ``*.uh3d``)

        :Call:
            >>> tri.ReadUH3D(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: :class:`str`
                Name of triangulation file to read
        :Versions:
            * 2014-06-02 ``@ddalle``: Version 1.0
            * 2014-10-27 ``@ddalle``: Added draft of reading component names
        """
        # Open the file
        fid = open(fname, 'r')
        # Read the first line and discard
        fid.readline()
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

        # Read component names
        self.ReadUH3DCompIDList(fid)
        # Close the file.
        fid.close()

    # Read component names from UH3D-like list
    def ReadUH3DCompIDList(self, fname):
        """Read a UH3D-like component list to *tri.Conf*

        :Call:
            >>> tri.ReadUH3DCompIDList(fname)
            >>> tri.ReadUH3DCompIDList(fid)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: :class:`str`
                Name of component list file to read
            *fid*: :class:`file`
                Already opened file handle from which to read component names
        :Versions:
            * 2018-02-23 ``@ddalle``: Split from :func:`ReadUH3D`
        """
        # Open file if necessary
        if hasattr(fname, "read"):
            # File handle
            fid = fname
            qfile = True
        else:
            # Open the file
            fid = open(fname, 'r')
            qfile = False
        # Initialize components.
        Conf = {}
        # Initialize check for end of file
        ftell = -1
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

        # Close the file if appropriate
        if not qfile:
            fid.close()

    # Read surface file
    def ReadSurf(self, fname):
        r"""Read an AFLR3 surface file

        :Call:
            >>> tri.ReadUH3D(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *fname*: :class:`str`
                Name of triangulation file to read
        :Versions:
            * 2014-06-02 ``@ddalle``: Version 1.0
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
            * 2015-12-13 ``@ddalle``: Version 1.0
        """
        # Check for the file
        if not os.path.isfile(fname):
            raise SystemError("File '%s' does not exist" % fname)
        # Status update
        print("  Reading number of points, edges, and tris")
        # grep command to get number of points
        # Line type: "    iNode     1      1     11\n"
        cmdi = [r'egrep "^\s+[0-9]+\s+1\s+1\s+11\s*" %s | tail -1' % fname]
        # Get last line with a point
        line = sp.Popen(cmdi, stdout=sp.PIPE, shell=True).communicate()[0]
        # Get number of nodes
        nNode = int(line.split()[0])
        # Command to get number of edges
        # Line type: "   iEdge  11  2  1  7  2"
        cmdi = [
            r'egrep "^\s+[0-9]+\s+11\s+2\s+1\s+7\s+2\s*$" %s | tail -1'
            % fname
        ]
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

    # Read CGNS file
    def ReadCGNS(self, fname):
        """Read a surface triangulated (with optional quads) CGNS file

        :Call:
            >>> tri.ReadCGNS(fname)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation
            *fname*: :class:`str`
                Name of file to read
        :Versions:
            * 2018-03-02 ``@ddalle``: Version 1.0
        """
       # --- Input ---
        # Read the CGNS file
        cgns = CGNS(fname)
       # --- CGNSBase_t ---
        # Get the *CGNSBase_t* node
        K_zt = cgns.GetNodeIndex(label="CGNSBase_t")
        # Check for multiple zones
        if len(K_zt) > 1:
            raise NotImplementedError(
                "No support for multiple CGNSBase_t nodes")
        elif len(K_zt) == 0:
            raise ValueError("No *CGNSBase_t* nodes")
        # Get the data
        data = cgns.Data[K_zt[0]]
        # Check the grid dimension
        #if data[0] != 3:
        #    raise ValueError("Expecting CellDimension=3, found '%s'" % data[0])
        # Check the physical dimension
        if data[1] != 3:
            raise ValueError("Expecting PhysicalDimension=3, found '%s'"
                % data[1])
       # --- ZoneType_t ---
        # Get the *ZoneType_t* node
        K_zt = cgns.GetNodeIndex(label="ZoneType_t")
        # Check for multiple zones
        if len(K_zt) > 1:
            raise NotImplementedError(
                "No support for multiple grid paradigms")
        elif len(K_zt) == 0:
            raise ValueError("No *ZoneType_t* nodes")
        # Keep the *ZoneType_t* index
        k_zt = K_zt[0]
        # Check the type
        if cgns.Data[k_zt] != "Unstructured":
            raise ValueError("CGNS file must be 'Unstructured' type")
       # --- Zone_t ---
        # Get any *Zone_t* nodes
        K_z = cgns.GetNodeIndex(label="Zone_t")
        # Check for multiple zones
        if len(K_z) > 1:
            raise NotImplementedError(
                "No support for multiple Zone_t nodes")
        elif len(K_z) == 0:
            raise ValueError("No *Zone_t* nodes")
        # Save this index
        k_z = K_z[0]
        # Get the data
        data = cgns.Data[k_z]
        # Number of nodes
        nNode = data[0]
        nElem = data[1]
       # --- Coordinates ---
        # Get *GridCoordinates_t* node
        kx = cgns.GetNodeIndex("CoordinateX", "DataArray_t")
        ky = cgns.GetNodeIndex("CoordinateY", "DataArray_t")
        kz = cgns.GetNodeIndex("CoordinateZ", "DataArray_t")
        # Get data type
        dtx = cgns.DataTypes[kx].replace("R", "f")
        # Initialize nodes
        Nodes = np.zeros((nNode, 3), dtype=dtx)
        # Read the coordinates
        Nodes[:,0] = cgns.Data[kx]
        Nodes[:,1] = cgns.Data[ky]
        Nodes[:,2] = cgns.Data[kz]
       # --- Elements ---
        # Initialize configuration
        Conf = {}
        # Running maximum component number
        ncomp = 0
        # Get indices of each CompID, *Eelements_t* data type
        KE = cgns.GetNodeIndex(label="Elements_t")
        # Process first component
        k = KE[0]
        # Element definitions
        ka, kb, ElemsK = cgns.GetCompIDInfo(cgns.NodeAddresses[k])
        # Initialize Elements array
        Elems = np.zeros((nElem, 4), dtype=ElemsK.dtype)
        # Initialize CompID array
        CompID = np.zeros(nElem, dtype="int")
        # Loop through remaining components
        for k in KE:
            # Name
            name = cgns.NodeNames[k]
            # Remove "_TRI" and "_QUA" suffixes
            if name.endswith("_TRI"):
                name = name[:-4]
            elif name.endswith("_QUA"):
                name = name[:-4]
            # Address
            addr = cgns.NodeAddresses[k]
            # Get component data
            ka, kb, ElemsK = cgns.GetCompIDInfo(addr)
            # Tris or Qauds?
            nVert = ElemsK.shape[1]
            # Save data
            Elems[ka-1:kb,:nVert] = ElemsK
            # Check if it's a new name
            if name in Conf:
                # Find existing index or list
                compID = Conf[name]
            elif name.endswith('1') and name[:-1] in Conf:
                # Use the previous conf
                compID = Conf[name[:-1]]
            else:
                # New component
                ncomp += 1
                compID = ncomp
                # Save the new name
                Conf[name] = compID
            # Save CompID
            CompID[ka-1:kb] = compID
       # --- Separate Tris and Quads ---
        # Find indices of triangles
        ITri  = np.where(Elems[:,3] == 0)[0]
        IQuad = np.where(Elems[:,3] != 0)[0]
        # Extract elements
        Tris  = Elems[ITri,:3]
        Quads = Elems[IQuad,:]
        # Counts
        nTri  = len(ITri)
        nQuad = len(IQuad)
        # Extract component ID arrays
        CompIDTri  = CompID[ITri]
        CompIDQuad = CompID[IQuad]
       # --- Output ---
        # Save nodes
        self.nNode = nNode
        self.Nodes = Nodes
        # Save tris
        self.nTri = nTri
        self.Tris = Tris
        # Save quads
        self.nQuad = nQuad
        self.Quads = Quads
        # Save component numbers
        self.CompID = CompIDTri
        self.CompIDQuad = CompIDQuad
        # Save the simple configuration
        self.Conf = Conf


  # >

  # =============
  # Other Writers
  # =============
  # <

   # ++++
   # TRIQ
   # ++++
   # {
    # Fall-through function to write the triangulation to file.
    def WriteTriq(self, fname='Components.i.triq',**kw):
        """Write q-triangulation to file using fastest method available

        :Call:
            >>> triq.WriteTriq(fname='Components.i.triq', **kw)
        :Inputs:
            *triq*: :class:`cape.tri.Triq`
                Triangulation instance to be written
            *fname*: :class:`str`
                Name of triangulation file to create
            *v*: :class:`bool`
                Whether or not
            *ascii*: ``True`` | {``False``}
                Whether or not to use ASCII (text file)
            *fmt*: {``None``} | ``"ascii"`` | ``"b4"`` | ``"lb4"``
                Format specified by text
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
            >>> triq = cape.ReadTriq('bJet.i.triq')
            >>> triq.Write('bjet2.triq', b4=True)
        :Versions:
            * 2014-05-23 ``@ddalle``: Version 1.0
            * 2015-01-03 ``@ddalle``: Added C capability
            * 2015-02-25 ``@ddalle``: Added status update
            * 2015-09-14 ``@ddalle``: Copied from :func:`TriBase.WriteTri`
        """
        # Status update.
        if kw.get('v', False):
            print("    Writing triangulation: '%s'" % fname)
        # Get the extension
        ext = self.GetOutputFileType(**kw)
        # Check text vs. binary
        if ext == 'ascii':
            # Try the ASCII writers
            self.WriteTriqASCII(fname)
        elif ext == 'lr4':
            # Little-endian single Fortran
            self.WriteSlow_lr4(fname)
        elif ext == 'r4':
            # Big-endian single Fortran
            self.WriteSlow_r4(fname)
        elif ext == 'lr8':
            # Little-endian double Fortran
            self.WriteSlow_lr8(fname)
        elif ext == 'r8':
            # Big-endian double Fortran
            self.WriteSlow_r8(fname)
        elif ext == 'lb4':
            # Little-endian single
            self.WriteSlow_lb4(fname)
        elif ext == 'b4':
            # Big-endian single
            self.WriteSlow_b4(fname)
        elif ext == 'lb8':
            # Little-endian double
            self.WriteSlow_lb8(fname)
        elif ext == 'b8':
            # Big-endian double
            self.WriteSlow_b8(fname)

    # Fall-through function to write the triangulation to file.
    def WriteTriqASCII(self, fname='Components.i.triq', v=True, **kw):
        """Write q-triangulation to file using fastest method available

        :Call:
            >>> triq.WriteTriqASCII(fname='Components.i.triq', v=True)
        :Inputs:
            *triq*: :class:`cape.tri.Triq`
                Triangulation instance to be written
            *fname*: :class:`str`
                Name of triangulation file to create
            *v*: :class:`bool`
                Verbosity flag
        :Versions:
            * 2014-05-23 ``@ddalle``: Version 1.0
            * 2015-01-03 ``@ddalle``: Added C capability
            * 2015-02-25 ``@ddalle``: Added status update
            * 2015-09-14 ``@ddalle``: Copied from :func:`TriBase.WriteTri`
            * 2017-03-29 ``@ddalle``: Moved from :func:`WriteTriq`
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
            * 2015-09-14 ``@ddalle``: Version 1.0
        """
        # Write the Common portion of the triangulation
        self.WriteSlow_ASCII(fname=fname)
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
            * 2015-09-14 ``@ddalle``: Version 1.0
        """
        # Write the nodes.
        _cape.WriteTriQ(self.Nodes, self.Tris, self.CompID, self.q)
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
            * 2015-04-17 ``@ddalle``: Version 1.0
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
            * 2015-04-17 ``@ddalle``: Version 1.0
        """
        # List of actual component IDs
        cID = list(np.unique(self.CompID))
        # Number of component IDs
        nID = len(cID)
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
            # Get the mapped component number
            kID = cID.index(self.CompID[k]) + 1
            # Write the line (with 1-based triangle index and CompID).
            fid.write('%i, %i, %i, %i, %i\n' % (k+1, self.Tris[k,0],
                self.Tris[k,1], self.Tris[k,2], kID))
        # Loop through the component names.
        for k in range(nID):
            # Get the actual component number
            c = cID[k]
            # Get the name that will be written.
            lbl = lbls.get(c, str(k+1))
            # Write the label.
            fid.write("%i, '%s'\n" % (k+1, lbl))
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
            * 2015-11-22 ``@ddalle``: Version 1.0
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
            * 2015-11-22 ``@ddalle``: Version 1.0
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
            * 2016-04-08 ``@ddalle``: Version 1.0
        """
        # Write the nodes.
        _cape.WriteTriSTL(self.Nodes, self.Tris)
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
            * 2015-11-19 ``@ddalle``: Version 1.0
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
            * 2015-11-19 ``@ddalle``: Version 1.0
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
            fid.write('%i %i %i %i 7 %i\n' % (self.Tris[k,0],
                self.Tris[k,1], self.Tris[k,2], self.CompID[k], self.BCs[k]))
        # Loop through the quads.
        for k in np.arange(self.nQuad):
            # Write the line (with 1-based quad indx and CompID)
            fid.write('%i %i %i %i %i 7 %i\n' % (self.Quads[k,0],
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
            * 2015-01-03 ``@ddalle``: Version 1.0
        """
        # Write the nodes.
        _cape.WriteSurf(
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
            * 2014-06-12 ``@ddalle``: Version 1.0
            * 2014-10-03 ``@ddalle``: Auto detect CompID overlap
        """
        # Concatenate the node matrix.
        self.Nodes = np.vstack((self.Nodes, tri.Nodes))
        # Concatenate the triangle node index matrix.
        self.Tris = np.vstack((self.Tris, tri.Tris + self.nNode))
        # Concatenate *q* values if appropriate
        try:
            self.q = np.vstack((self.q, tri.q))
        except AttributeError:
            pass
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
                    # Get the two faces
                    face0 = self.config.faces[face]
                    face1 = tri.config.faces[face]
                    # Types
                    q0 = face0.__class__.__name__.startswith("int")
                    q1 = face1.__class__.__name__.startswith("int")
                    # Check the situation
                    if q0 and q1 and (face0 == face1):
                        # Do nothing
                        pass
                    else:
                        # Convert to array
                        face0 = np.array(face0).flatten()
                        face1 = np.array(face1).flatten()
                        # Union the two faces
                        u = np.union1d(face0, face1)
                        # Save as a list
                        self.config.faces[face] = list(u)
                else:
                    # Add the face
                    self.config.faces[face] = tri.config.faces[face]
        except AttributeError:
            # No configurations to merge
            pass
        # Check for Confg
        try:
            # Check for a configuration in both triangulations
            self.Conf
            tri.Conf
            # Loop through the faces in the added configuration
            for face in tri.Conf:
                # Check if present
                if face not in self.Conf:
                    # Just add it (easy case)
                    self.Conf[face] = tri.Conf[face]
                    continue
                # Otherwise, get both values
                face0 = self.Conf[face]
                face1 = self.Conf[face]
                # Check types
                q0 = face0.__class__.__name__.startswith("int")
                q1 = face1.__class__.__name__.startswith("int")
                # Combine them
                if q0 and q1:
                    # Check if different
                    if face0 != face1:
                        # Use a list of both values
                        self.Conf[face] = [face0, face1]
                elif q1:
                    # New triangulation is an integer
                    if face1 not in face0:
                        # Append to the list
                        self.Conf[face].append(face1)
                elif q0:
                    # Original configuration is an integer
                    if face0 not in face1:
                        # Append to the list
                        face1.append(face0)
                    # Save the list
                    self.Conf[face] = face1
                else:
                    # Both are lists; loop through new set
                    for k in face1:
                        # Check if present
                        if k not in face0:
                            self.Conf[face].append(k)
        except AttributeError:
            # No *Conf* to merge
            pass

    # Add a second triangulation without altering component numbers.
    def AddRawCompID(self, tri, warn=False, newnodes=True):
        """
        Add a second triangulation to the current one without changing
        component numbers of either triangulation.  No checks are performed,
        and intersections are not analyzed.

        :Call:
            >>> tri.AddRawCompID(tri2, warn=False, newnodes=True)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance to be altered
            *tri2*: :class:`cape.tri.Tri`
                Triangulation instance to be added to the first
            *warn*: ``True`` | {``False``}
                Whether or not to warn about components in both
            *newnodes*: {``True``} | ``False``
                If ``False``, do not add nodes of second triangulation
        :Effects:
            All nodes and triangles from *tri2* are added to *tri*.  As a
            result, the number of nodes, number of tris, and number of
            components in *tri* will all increase.
        :Versions:
            * 2014-06-12 ``@ddalle``: Version 1.0
            * 2019-06-17 ``@ddalle``: Added *newnodes* keyword
        """
        # Check for separate node lists
        if newnodes:
            # Concatenate the node matrix
            self.Nodes = np.vstack((self.Nodes, tri.Nodes))
            # Concatenate the triangle node index matrix
            self.Tris = np.vstack((self.Tris, tri.Tris + self.nNode))
            # Update number of nodes
            self.nNode += tri.nNode
        else:
            # Concatenate the triangle node index matrix, unaltered
            self.Tris = np.vstack((self.Tris, tri.Tris))
        # Check for overlaps
        if warn:
            # Unique component lists
            CompID0 = np.unique(self.CompID)
            CompID1 = np.unique(tri.CompID)
            # Intersection
            CompID_common = np.intersect1d(CompID0, CompID1)
            # Loop through those
            for c in CompID_common:
                # Start of warning message
                msg = "  WARNING [AddRawCompID]: component %i " % c
                # Try to get face names
                face0 = self.GetCompName(c)
                face1 = tri.GetCompName(c)
                # Add names to message
                msg += "(%s, %s) " % (face0, face1)
                # Complete message
                msg += "in both grids"
                # Show it
                print(msg)
        # Concatenate the component vector
        self.CompID = np.hstack((self.CompID, tri.CompID))
        # Loop through confs
        try:
            # See if added triangulation has *tri.Conf*
            tri.Conf
            # See if *self* has *Conf*
            try:
                self.Conf
            except AttributeError:
                self.Conf = {}
            # Loop through components
            for comp in tri.Conf:
                # Get value and type
                vt = tri.Conf[comp]
                tt = type(vt).__name__
                # Test if already present
                if comp in self.Conf:
                    # Get value
                    vs = self.Conf[comp]
                    # Get type
                    ts = type(v).__name__
                    # Append
                    if ts in ['list', 'ndarray']:
                        # Add to list
                        if tt in ['list', 'ndarray']:
                            # Combine lists
                            self.Conf[comp] = list(vs) + list(vt)
                        else:
                            # Append to existing list
                            self.Conf[comp] = list(vs) + [vt]
                    else:
                        # Create list
                        if tt in ['list', 'ndarray']:
                            # Create list and add to it
                            self.Conf[comp] = [vs] + list(vt)
                        else:
                            # Create doubleton list
                            self.Conf[comp] = [vs, vt]
                else:
                    # Add it
                    self.Conf[comp] = vt
        except Exception:
            pass
        # Update the number of tris
        self.nTri += tri.nTri

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
            >>> tri.WriteVolTri(fname='Components.c.tri')
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
                Triangulation instance
            *fname*: :class:`str`
                Name of .tri file for use as input to `intersect`
        :Versions:
            * 2015-02-24 ``@ddalle``: Version 1.0
        """
        # Copy the triangulation.
        tri = self.Copy()
        # Current maximum CompID
        comp0 = np.max(self.CompID)
        # Set first volume.
        tri.CompID[:self.iTri[0]] = comp0 + 1
        # Loop through volumes as marked in *tri.iTri*
        for k in range(len(self.iTri)-1):
            # Component number
            compn = comp0 + k + 2
            # Get indices
            k1 = self.iTri[k]
            k2 = self.iTri[k+1]
            # Check for negative *iTri* value
            # If negative; don't write in ``intersect`` input triangulation
            if k2 < 0:
                # Write a negative component number
                compn = -compn
            # Use positive triangle indices
            k1 = abs(k1)
            k2 = abs(k2)
            # Set the CompID for each tri in that volume.
            tri.CompID[k1:k2] = compn
        # Ignore negative triangles
        kKeep = (tri.CompID > 0)
        tri.Tris   = tri.Tris[kKeep, :]
        tri.CompID = tri.CompID[kKeep]
        tri.nTri   = tri.Tris.shape[0]
        # Write the triangulation to file.
        tri.Write(fname)

    # Function to write c.tri file with original CompIDs but w/o farfield
    def WriteCompIDTri(self, fname='Components.tri'):
        """Write a .tri file with the original components

        This provides a component map for the output of ``intersect``.
        Supplemental surfaces, such as farfield triangles or grid refinement
        sources, are not written.  Specifically, triangles with negative
        component IDs are not written.

        :Call:
            >>> tri.WriteCompIDTri(fname='Components.c.tri')
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
                Triangulation instance
            *fname*: :class:`str`
                Name of .tri file to use for mapping intersected tris
        :Versions:
            * 2017-05-25 ``@ddalle``: Version 1.0
        """
        # Copy the triangulation.
        tri = self.Copy()
        # Initialize indices of triangles to keep
        kKeep = np.arange(self.nTri) < 0
        # Initialize start of first zone
        k2 = 0
        # Loop through volumes as marked in *tri.iTri*
        for k in range(len(self.iTri)):
            # Get indices
            k1 = abs(k2)
            k2 = self.iTri[k]
            # Keep positive *iTri* values
            if k2 > 0:
                # Keep this zone
                kKeep[k1:k2] = True
            else:
                # Do not keep this zone
                kKeep[k1:-k2] = False
        # Ignore negative triangles
        tri.Tris   = tri.Tris[kKeep,:]
        tri.CompID = tri.CompID[kKeep]
        tri.nTri   = tri.Tris.shape[0]
        # Write the triangulation to file.
        tri.Write(fname)

    # Function to write f.tri file with supplemental surfaces
    def WriteFarfieldTri(self, fname='Components.f.tri'):
        """Write a .tri file supplemental surfaces not intersected

        This stores the triangles that are excluded from the input to
        ``intersect``.  This would include farfield surfaces, grid refinement
        boxes, or bodies that are known not to intersect any others.

        :Call:
            >>> tri.WriteFarfieldTri(fname='Components.f.tri')
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
                Triangulation instance
            *fname*: :class:`str`
                Name of .tri file to use for mapping intersected tris
        :Versions:
            * 2017-05-25 ``@ddalle``: Version 1.0
        """
        # Copy the triangulation.
        tri = self.Copy()
        # Initialize indices of triangles to keep
        kKeep = np.arange(self.nTri) < 0
        # Initialize start of first zone
        k2 = 0
        # Loop through volumes as marked in *tri.iTri*
        for k in range(len(self.iTri)):
            # Get indices
            k1 = abs(k2)
            k2 = self.iTri[k]
            # Keep positive *iTri* values
            if k2 > 0:
                # Keep this zone
                kKeep[k1:k2] = False
            else:
                # Do not keep this zone
                kKeep[k1:-k2] = True
        # Ignore negative triangles
        tri.Tris   = tri.Tris[kKeep,:]
        tri.CompID = tri.CompID[kKeep]
        tri.nTri   = tri.Tris.shape[0]
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
            *tri*: :class:`cape.tri.TriBase`
                Triangulation instance
            *tric*: :class:`cape.tri.TriBase`
                Triangulation with more desirable CompIDs to be copied
            *compID*: :class:`int`
                Component ID to map from *tric*
            *k1*: :class:`numpy.ndarray` (:class:`int`)
                Indices of faces in *tric* to considerider
        :Versions:
            * 2015-02-24 ``@ddalle``: Version 1.0
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
        K0 = kc.copy()
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
            * 2015-02-25 ``@ddalle``: Version 1.0
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
        r"""Read a configuration file using extension to guess type

        :Call:
            >>> tri.ReadConfig(c)
            >>> tri.ReadConfig(cfg)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *c*: :class:`str`
                Configuration file name
            *cfg*: :class:`ConfigJSON` | :class:`ConfigXML`
                Pre-existing configuration
        :Versions:
            * 2016-10-21 ``@ddalle``: Version 1.0
            * 2022-03-17 ``@ddalle``: Version 2.0; allow *cfg* input
        """
        # Check for exisgint config
        if isinstance(c, ConfigJSON):
            # Pre-read JSON config
            self.config = c
            return
        elif isinstance(c, ConfigXML):
            # Pre-read XML config
            self.config = c
            return
        elif isinstance(c, ConfigMIXSUR):
            # Pre-read MIXSUR config
            self.config = c
            return
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
        elif fext == "json":
            # Read a JSON file
            self.ReadConfigJSON(c)
        elif (fext == "i") or (c.startswith("fomoco") or c.startswith("mixsur")):
            # Try a ``mixsur.i`` file
            self.ReadConfigMIXSUR(c)
        else:
            # Cascade: try XML first
            try:
                self.ReadConfigXML(c)
                return
            except Exception:
                pass
            # Cascade: try JSON second
            try:
                self.ReadConfigJSON(c)
                return
            except Exception:
                pass
            # Cascade: try MIXSUR third
            try:
                self.ReadConfigMIXSUR(c)
                return
            except Exception:
                pass

    # Function to read Config.xml
    def ReadConfigXML(self, c, restrict=False):
        r"""Read an XML file labeling and grouping of component IDs

        :Call:
            >>> tri.ReadConfigXML(c, restrict=False)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *c*: :class:`str`
                Configuration file name
            *restrict*: ``True`` | {``False``}
                Option to eliminate faces that are not present in triangulation
        :Versions:
            * 2015-11-19 ``@ddalle``: Version 1.0
        """
        # Read the configuration and save it.
        self.config = ConfigXML(c)
        # Restrict to a subset
        if restrict:
            self.RestrictConfigCompID()

    # Function to read Config.json
    def ReadConfigJSON(self, c):
        """Read a JSON file labeling and grouping of component IDs

        :Call:
            >>> tri.ReadConfigJSON(c)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *c*: :class:`str`
                Configuration file name
        :Versions:
            * 2016-10-21 ``@ddalle``: Version 1.0
        """
        # Read the configuration and save it
        self.config = ConfigJSON(c)

    # Function to read Config.json
    def ReadConfigMIXSUR(self, c):
        """Read a ``mixsur.i`` file labeling and grouping of component IDs

        :Call:
            >>> tri.ReadConfigMixsur(c)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *c*: :class:`str`
                Configuration file name
        :Versions:
            * 2017-04-05 ``@ddalle``: Version 1.0
        """
        # Read the configuration and save it
        self.config = ConfigMIXSUR(c)

    # Function to map component ID numbers to those in a Config.
    def ApplyConfig(self, cfg):
        r"""Change component IDs to match a configuration file

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
            * 2014-11-10 ``@ddalle``: Version 1.0
        """
        # Check for Conf in the triangulation.
        try:
            self.Conf
        except AttributeError:
            return
        # Check for string input
        if type(cfg).__name__ in ['str', 'unicode']:
            # Read the config
            cfg = ConfigXML(cfg)
        # Make a copy of the component IDs
        compID = self.CompID.copy()
        # Try to make a copy of the quad component IDs
        try:
            compIDQuad = self.CompIDQuad.copy()
        except AttributeError:
            compIDQuad = np.zeros(0, dtype="int")
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
                I = np.zeros_like(compID)
                J = np.zeros_like(compIDQuad)
                # Loop through additional entries
                for kj in kID:
                    # Use *or* operation to search for other matches
                    I = np.logical_or(I, compID==kj)
                    J = np.logical_or(J, compIDQuad==kj)
                # Convert to indices
                I1 = np.where(I)[0]
                J1 = np.where(J)[0]
                # Assign the new values
                if len(I1) > 0:
                    self.CompID[I] = cID
                if len(J1) > 0:
                    self.CompIDQuad[J] = cID
                # Save it in the Conf, too.
                self.Conf[k] = cID
                # Save the compID as an int in the *config* just for clarity
                #self.config.faces[k] = cID
        # Restrict
        #self.RestrictConfigCompID()

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
            * 2016-11-06 ``@ddalle``: Version 1.0
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
            * 2016-11-05 ``@ddalle``: Version 1.0
        """
        # Get list of component IDs
        compIDs = np.unique(self.CompID)
        compIDsQ = self.__dict__.get("CompIDQuad", None)
        # Attempt to add Quads
        if compIDsQ is not None:
            compIDs = np.union1d(compIDs, np.unique(self.CompIDQuad))
        # Call the method from the *config* handle
        self.config.RestrictCompID(compIDs)

    # Renumber component IDs 1 to *n*
    def RenumberCompIDs(self):
        """Renumber component ID numbers 1 to *n*

        :Call:
            >>> tri.RenumberCompIDs()
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
                Triangulation interface
        :Versions:
            * 2016-11-09 ``@ddalle``: Version 1.0
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
        # Create a copy used for error checking
        tri = self.Copy()
        # Clean up extra entries in faces
        for face in self.config.faces:
            # Skip if already processed
            if face in faces:
                continue
            # Get the component number
            compf = self.config.faces[face]
            # Make sure it's a list
            if not isinstance(compf, (list, np.ndarray)):
                # This should have been a face
                faces.append(face)
                continue
            # Loop through components
            for compi in list(compf):
                # Delete it if not in the renumbered list
                if compi not in compIDs:
                    compf.remove(compi)
        # Get full list of faces
        face_candidates = list(faces)
        # Loop through all named faces
        for face in sorted(self.config.faces.keys()):
            # Skip if already present
            if face in face_candidates:
                continue
            # Get component number(s)
            compi = self.config.faces[face]
            # Check type
            if isinstance(compi, int):
                # Single face; good candidate
                face_candidates.append(face)
            elif len(compi) != 1:
                # List; not a candidate
                continue
            # Get name from *compi* (to check for parent with one child)
            facei = self.GetCompName(compi[0])
            # Check if name for *compi* matches *face*
            if face == facei:
                # Singleton face; candidate
                face_candidates.append(face)
        # Loop through sorted faces
        for face in face_candidates:
            # Get the component number
            compi = self.config.faces[face]
            # Check for singleton candidate (list w/ >1 entry not candidate)
            if not isinstance(compi, int):
                # Extract element from singleton
                compi = compi[0]
            # Check if that compID is present
            if compi not in compIDs:
                # This component is not present in the current triangulation
                # Mark for deletion
                self.config.RenumberCompID(face, -compi)
                continue
            # Otherwise, assign new component ID number
            # Find indices of triangles in this component
            I = np.where(self.CompID==compi)[0]
            # Increase component count
            ncomp += 1
            # Assign updated component number
            CompID[I] = ncomp
            # Renumber the components in the *config*
            self.config.RenumberCompID(face, ncomp)
        # Check for zero
        if np.any(CompID == 0):
            # General warning
            print(
                "  WARNING [RenumberCompIDs]: "
                + "At least one tri has unset component ID")
            # Get the locations of missed triangles
            I = np.where(CompID == 0)[0]
            # Get list of missed components
            C = np.unique(self.CompID[I])
            # Print the list of original component IDs from here
            print("  Table of original component IDs that were missed:")
            print("      CompID  Name")
            # Loop through such missing components
            for cID in C:
                # Attempt to get the name
                face = tri.GetCompName(cID)
                # Error message
                print("%4s  %6s  %s" % ("", cID, face))
        # Reset component IDs
        self.CompID = CompID

    # Function to get compIDs by name
    def GetCompID(self, face=None):
        """Get components by name or number

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
            * 2014-10-12 ``@ddalle``: Version 1.0
            * 2016-03-29 ``@ddalle``: Edited docstring
            * 2017-02-10 ``@ddalle``: Added fallback to *tri.Conf*
        """
        # Process input into a list of component IDs.
        try:
            # Best option is to use the Config.xml file
            return self.config.GetCompID(face)
        except Exception:
            # Fall back to *tri.Conf* or just process raw numbers
            return self.GetConfCompID(face)

    # Get name of a compID
    def GetCompName(self, compID):
        """Get the name of a component by its number

        :Call:
            >>> face = tri.GetCompName(compID)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation interface
            *compID*: :class:`int`
                Component ID number
        :Outputs:
            *face*: {``""``} | :class:`str`
                Name of so-numbered component, if any
        :Versions:
            * 2017-03-30 ``@ddalle``: Version 1.0
        """
        # Try to use the UH3D dictionary
        try:
            self.Conf
            # Get list of available components from Conf
            Comps = []
            CompIDs = []
            # Loop through candidate comps
            for comp in self.Conf:
                # Get value
                v = self.Conf[comp]
                # Append if appropriate
                if isinstance(v, INT_TYPES):
                    # Check for match
                    if v == compID:
                        return comp
        except AttributeError:
            # There's no *tri.Conf*
            pass
        # Try both configuration interfaces
        try:
            # Use the *Config* class first
            face = self.config.GetCompName(compID)
        except AttributeError:
            # No *Config* attribute
            face = None
        # Exit if found
        if face is None:
            # No match
            return ""
        else:
            return face

    # Get compIDs by name or number from *tri.Conf*
    def GetConfCompID(self, face=None):
        """Get components by name or number from *tri.Conf* dictionary

        :Call:
            >>> compID = tri.GetConfCompID()
            >>> compID = tri.GetConfCompID(face)
            >>> compID = tri.GetConfCompID(comp)
            >>> compID = tri.GetConfCompID(comps)
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
            * 2017-02-10 ``@ddalle``: Version 1.0
        """
        # Check for scalar
        if face is None:
            # No contents; this might break otherwise
            return list(np.unique(self.CompID))
        elif type(face).__name__  in ['list', 'ndarray']:
            # Return the list
            faces = face
        else:
            # Make a singleton list
            faces = [face]
        # Process the *tri.Conf* with default
        try:
            Conf = self.Conf
        except AttributeError:
            Conf = {}
        # Check if present
        compID = []
        # Loop through faces
        for face in faces:
            # Check type
            if type(face).__name__.startswith('int'):
                # Append integer face
                compID.append(face)
            else:
                # Get comp from *tri.Conf*
                comp = Conf.get(face)
                # Check type
                if comp is None:
                    # This face is not present
                    continue
                elif type(comp).__name__ == "list":
                    # List of components
                    compID += comp
                else:
                    # Single component
                    compID.append(comp)
        # Sort the list
        compID.sort()
        # Use this list
        return compID

    # Get *tri.Conf* dictionary
    def GetConfFromConfig(self):
        """Create *tri.Conf* dictionary using *tri.config* if appropriate

        :Call:
            >>> tri.GetConfFromConfig()
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation interface
        :Attributes:
            *tri.Conf*: :class:`dict`
                Dictionary of face names coped from *tri.config.faces*
        :Versions:
            * 2017-02-10 ``@ddalle``: Version 1.0
            * 2022-10-27 ``@ddalle``: Version 2.0; only single-component
        """
        # Check for existing *Conf*
        if hasattr(self, "Conf"):
            return
        # Initialize dictionary
        self.Conf = {}
        # Get key attributes
        faces = self.config.faces
        # Loop through "components" (non-groups)
        for comp in self.config.comps:
            # Get compID if any
            compID = faces.get(comp)
            # Save single face ID
            if compID is not None:
                self.Conf[comp] = faces[comp]

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
            * 2014-09-27 ``@ddalle``: Version 1.0
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
            I[self.Quads[kQuad]-1] = True
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
            * 2015-01-23 ``@ddalle``: Version 1.0
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

    # Get tri indices from node indices
    def GetTrisFromNodes(self, I, skip=1):
        r"""Find indices of triangles from node indices

        :Call:
            >>> K = tri.GetTrisFromNodes(I, skip=1)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *I*: :class:`np.ndarray`\ [:class:`int`]
                Array of node indices
            *skip*: {``1``} | :class:`int` > 0
                Only analyze every *skip*\ th node
        :Outputs:
            *K*: :class:`np.ndarray`\ [:class:`int`]
                Array of triangle indices
        :Versions:
            * 2019-05-14 ``@ddalle``: Version 1.0
        """
        # Initialize True|{False} array of each triangle
        Q = np.arange(self.nTri) < 0
        # Handle to triangle indices
        T = self.Tris
        # Loop through nodes
        for i in I[::skip]:
            # Status update
            sys.stdout.write("   %5i/%5i\r" % (i, np.max(I)))
            sys.stdout.flush()
            # Process three columns of *Tris*
            Q[T[:,0] == i] = True
            Q[T[:,1] == i] = True
            Q[T[:,2] == i] = True
        # Clean up prompt
        sys.stdout.write("%14s\r" % "")
        sys.stdout.flush()
        # Get indices
        return np.where(Q)[0]

    # Get components from compIDs
    def GetFacesFromTris(self, K, nmin=10):
        r"""Find components from triangles

        :Call:
            >>> faces = tri.GetFacesFromTris(K, nmin=10)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *K*: :class:`np.ndarray`\ [:class:`int`]
                Array of triangle indices
            *nmin*: {``10``} | :class:`int` > 0
                Only return faces with at least *nmin* triangles
        :Outputs:
            *faces*: :class:`list`\ [:class:`str`]
                List of face names (if available) or numbers
        :Versions:
            * 2019-05-14 ``@ddalle``: Version 1.0
        """
        # Component numbers of *K* triangles
        CompID = self.CompID[K]
        # Get list of component IDs
        CompIDs = np.unique(CompID)
        # Initialize faces
        faces = []
        # Loop through found component IDs
        for comp in CompIDs:
            # Count number of triangles
            ntri = np.where(CompID == comp)[0].size
            # Skip if less than *nmin*
            if ntri < nmin:
                continue
            # Otherwise get face name/number
            face = self.GetCompName(comp)
            # Save it
            if face:
                # Save face name
                faces.append(face)
            else:
                # Save number
                faces.append(comp)
        # Output
        return faces

    # Get components from compIDs
    def GetFacesFromQuads(self, K, nmin=10):
        r"""Find components from triangles

        :Call:
            >>> faces = tri.GetFacesFromQuads(K, nmin=10)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *K*: :class:`np.ndarray`\ [:class:`int`]
                Array of quadrilateral indices
            *nmin*: {``10``} | :class:`int` > 0
                Only return faces with at least *nmin* triangles
        :Outputs:
            *faces*: :class:`list`\ [:class:`str`]
                List of face names (if available) or numbers
        :Versions:
            * 2019-05-14 ``@ddalle``: Version 1.0
        """
        # Component numbers of *K* triangles
        CompID = self.CompIDQuad[K]
        # Get list of component IDs
        CompIDs = np.unique(CompID)
        # Initialize faces
        faces = []
        # Loop through found component IDs
        for comp in CompIDs:
            # Count number of triangles
            ntri = np.where(CompID == comp)[0].size
            # Skip if less than *nmin*
            if ntri < nmin:
                continue
            # Otherwise get face name/number
            face = self.GetCompName(comp)
            # Save it
            if face:
                # Save face name
                faces.append(face)
            else:
                # Save number
                faces.append(comp)
        # Output
        return faces

    # Function to get tri indices from component ID(s)
    def GetQuadsFromCompID(self, compID=None):
        r"""Find indices of triangles with specified component ID(s)

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
            * 2016-04-05 ``@ddalle``: Version 1.0
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
            # Turn boolean vector into vector of indices
            I =  np.where(K)[0]
            return I
        except AttributeError:
            # No quads
            return np.zeros(0, dtype=int)

    # Get subtriangulation from CompID list
    def GetSubTri(self, i=None):
        r"""
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
            * 2015-01-23 ``@ddalle``: Version 1.0
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
        # Trim unused nodes to save space
        tri0.RemoveUnusedNodes()
        # Output
        return tri0

    # Eliminate unused nodes
    def RemoveUnusedNodes(self, v=False):
        r"""Remove any nodes that are not used in any triangles

        :Call:
            >>> tri.RemoveUnusedNodes(v=False)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *v*: ``True`` | {``False``}
                Verbosity flag
        :Versions:
            * 2017-02-10 ``@ddalle``: Version 1.0
            * 2019-06-18 ``@ddalle``: NO LOOPS
        """
        # Get nodes that are used
        N = np.unique(self.Tris)
        # Output number of nodes
        nNode = N.size
        # Status update
        if v:
            print("Removing %i unused nodes" % (self.nNode - nNode))
        # Check for no unused nodes
        if nNode == self.nNode:
            return
        # Initialize array of new indices
        # This is a map from node i -> I[i-1]
        I = np.zeros(self.nNode, dtype="int")
        # Create the map using vector operation
        I[N-1] = np.arange(1, nNode+1)
        # Extract triangles
        T = self.Tris
        # Renumber triangles using *I* as a map
        self.Tris = I[T-1]
        # Downselect nodes
        self.nNode = nNode
        self.Nodes = self.Nodes[N-1,:]
        # Downselect *q* if available
        try:
            self.q = self.q[N-1,:]
        except AttributeError:
            pass


    # Eliminate small triangles
    def RemoveSmallTris(self, smalltri=1e-5, v=False, recurse=True):
        r"""Remove any triangles that are below a certain size

        :Call:
          >>> tri.RemoveSmallTris(smalltri=1e-5, v=False, recurse=True)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *smalltri*: {``1e-5``} | :class:`float` > 0
                Minimum allowable triangle area
            *v*: ``True`` | {``False``}
                Verbosity flag
            *recurse*: {``True``} | ``False``
                Whether or not to remove newly created small tris
        :Versions:
            * 2017-06-19 ``@ddalle``: Version 1.0
        """
        # Calculate areas
        self.GetNormals()
        # Filter areas
        K = np.where(self.Areas <= smalltri)[0]
        # Number of triangles to remove
        nsmall = K.size
        # Check for nothing to do
        if nsmall == 0:
            return
        # Status update
        if v:
            print("Removing %i small triangles (A<=%.2e)"
                % (nsmall, smalltri))
        # Get the node indices of the small tris
        I = self.Tris[K] - 1
        # Get the coordinates of all nodes involved in small triangles
        X = self.Nodes[I, 0]
        Y = self.Nodes[I, 1]
        Z = self.Nodes[I, 2]
        # Edge distance components
        dx = X[:,[1,2,0]] - X
        dy = Y[:,[1,2,0]] - Y
        dz = Z[:,[1,2,0]] - Z
        # Distances
        D = np.sqrt(dx*dx + dy*dy + dz*dz)
        # Estimate edge tolerance
        dmin = np.sqrt(smalltri)
        # Find the shortest edge of each small triangle
        J = np.argmin(D, axis=1)
        # Combine start/end node indices
        I0 = np.array([I[i][[j, (j + 1) % 3]] for (i, j) in enumerate(J)])
        # Additional node replacements
        I2 = []
        # Loop through the tri to check for neighbor-neighbor issues
        for (i, k) in enumerate(K):
            # Check maximum length
            if np.max(D[i]) <= dmin:
                # Get the nodes
                i1, i2, i3 = I[i]
                # Just shift the triangle to a point
                I2.append([i1, i2])
                I2.append([i1, i3])
                I2.append([i2, i3])
        # Append additional deletions
        if len(I2) > 0:
            I0 = np.vstack((I0, I2))
        # Sort the node pairs so we mark the same node for deletion
        # regardless of order
        I0.sort(axis=1)
        # Each node in the left column is replaced by node in right
        # To avoid loops, we sort in dictionary order; higher node
        # indices are favored
        O1 = np.lexsort((I0[:,1], I0[:,0]))
        # Apply sorting
        I0 = I0[O1]
        # Initialize node index map; node i --> node I1[i] (0-based)
        I1 = np.arange(self.nNode)
        # Loop through node replacements
        for (ia, ib) in I0:
            # Make assignment
            I1[ia] = ib
        # Outgoing nodes
        IA = np.unique(I0[:,0])
        # Make new triangle index array with replacements
        T = I1[self.Tris - 1] + 1
        # Once the nodes have been updated, look for tris with repeats
        # This can happen if a (sufficiently) large triangle has an
        # edge removed by being adjacent to a small tri
        # First we find the delta-node-index for each edge
        DI = T[:,[1,2,0]] - T
        # Find the smallest node-delta in absolute terms for each tri
        DImin = np.min(np.abs(DI), axis=1)
        # Any time there's a zero in a row marks a trivialized tri
        K1 = np.where(DImin == 0)[0]
        # Let's be safe and make sure the original tris are removed
        K2 = np.union1d(K, K1)
        # Total removal count
        ndel = K2.size
        # Final removal count
        if v:
            print("Removing %i additional tris trivialized by edge removal"
                % (ndel - nsmall))
        # Remove the small or trivialized tris from orig and remapped
        T1 = np.delete(self.Tris, K2, axis=0)
        T2 = np.delete(T, K2, axis=0)
        # Save triangles
        self.Tris = T2
        # Update number of triangles
        self.nTri = self.Tris.shape[0]
        # Delete area calculations, some of which will need updating
        delattr(self, "Areas")
        delattr(self, "Normals")
        #delattr(self, "EdgeTable")
        # Final removal count
        if v:
            print("Removing %i triangles in total" % ndel)
        # Component IDs should be there, but let's be safe
        try:
            self.CompID = np.delete(self.CompID, K2, axis=0)
        except AttributeError:
            pass
        # Remove those removed nodes
        self.RemoveUnusedNodes(v=v)
        # Recurse if needed
        if recurse and (ndel > 0):
            # Status update
            if v:
                print("Recursing to check for *new* small triangles")
            # Recursive call
            self.RemoveSmallTris(smalltri, v=v)


    # Map triangles to components based on another file
    def MapTriCompID(self, tri, **kw):
        r"""Map component IDs of a separate triangulation

        :Call:
            >>> tri.MapTriCompID(tric, **kw)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *tric*: :class:`cape.tri.Tri`
                Triangulation with alternative component labels
            *compID*: {``None``} | :class:`int` | :class:`str` | :class:`list`
                Only consider tris in this component(s)
        :Versions:
            * 2017-02-09 ``@ddalle``: Version 1.0
        """
        # Check triangulation type
        tt = type(tri).__name__
        if not tt.startswith("Tri"):
            raise TypeError(
                "Triangulation for mapping must be 'Tri', or 'Triq'")
        # Check for null operation
        if tri.nTri == 0: return
        # Only consider triangles in this component
        compID = kw.get('compID')
        # Process primary tolerances
        atol  = kw.get("atol",  kw.get("AbsTol",  atoldef))
        rtol  = kw.get("rtol",  kw.get("RelTol",  rtoldef))
        ctol  = kw.get("ctol",  kw.get("CompTol", ctoldef))
        antol = kw.get("ntol",  kw.get("ProjTol", antoldef))
        antol = kw.get("antol", kw.get("AbsProjTol",  antol))
        rntol = kw.get("rntol", kw.get("RelProjTol",  rntoldef))
        cntol = kw.get("cntol", kw.get("CompProjTol", cntoldef))
        # Get scale of the entire triangulation
        L = tri.GetCompScale()
        # Initialize scales of components
        LC = {}
        # Put together absolute and relative tols
        tol  = atol   + rtol*L
        ntol = antol  + rntol*L
        # Filter the triangles that have a chance of intersecting
        if compID is None:
            # Bet bounding box from *tri*
            bbox = tri.GetCompBBox(pad=tol)
            # Get triangles with at least one node in that *BBox*
            K = self.FilterTrisBBox(bbox)
        else:
            # Get candidate triangles directly
            K = self.GetTrisFromCompID(compID)
        # Verbose flag
        v = kw.get("v", False)
        # Ensure the centers are present
        self.GetCenters()
        # Get list of unique component IDs
        comps = np.unique(self.CompID)
        # Mapping *tri.CompID* to *self.CompID*
        compmap = {}
        facemap = {}
        # Loop through columns
        for i, k in enumerate(K):
            # Get triangle number
            k = K[i]
            # Status update if verbose
            if v and ((i+1) % (1000*v) == 0):
                sys.stdout.write("  Mapping triangle %i/%i\r" % (i+1, len(K)))
                sys.stdout.flush()
            # Perform search
            T = tri.GetNearestTri(self.Centers[k,:], n=1)
            # Get components
            c1 = T.get("c1")
            # Make sure component scale is present
            if c1 not in LC:
                # Get the component scale
                LC[c1] = tri.GetCompScale(c1)
                # Check if the component is already used by *tri*
                if c1 in comps:
                    # Need to shift the component number
                    c = c1 + max(comps)
                else:
                    # Already have the component
                    c = c1
                # Save the component map
                compmap[c1] = c
            # Get overall tolerances
            toli  = tol + ctol*LC[c1]
            ntoli = ntol + cntol*LC[c1]
            # Filter results
            if (T["t1"] > toli) or (T["z1"] > ntoli):
                continue
            # Save new component ID
            self.CompID[k] = compmap[c1]
        # Clean up prompt
        if v:
            sys.stdout.write("%72s\r" % "")
            sys.stdout.flush()
        # Update *self.config* if applicable
        try:
            # Loop through faces in the target map
            for face in tri.config.faces:
                # Get component ID(s); guarantee list
                comps = np.array(tri.config.faces[face]).flatten()
                # Get mapped component numbers
                cmapd = []
                # Loop through comps
                for comp in comps:
                    # Skip if not used
                    if comp not in compmap:
                        # Use the existing component number
                        cmapd.append(comp)
                    else:
                        # Save the component from the new guy
                        cmapd.append(compmap[comp])
                # Check length
                if len(cmapd) == 0:
                    # No matches
                    continue
                elif len(cmapd) == 1:
                    # Save single match
                    self.config.faces[face] = cmapd[0]
                else:
                    # Save list
                    self.config.faces[face] = cmapd
        except AttributeError:
            pass
        # Set a *Conf* dictionary if necessary
        self.GetConfFromConfig()
        # Get Config from mapping try:
        try:
            # Extract from the map
            Conf = tri.Conf
        except AttributeError:
            # Create a default *Conf* dictionary
            Conf = {}
        # Initialize conf
        try:
            self.Conf
        except AttributeError:
            # Create default *Conf* dictionary
            self.Conf = {}
        # Loop through faces in the target map
        for face in Conf:
            # Get component ID(s); guarantee list
            comps = np.array(Conf[face]).flatten()
            # Get mapped component numbers
            cmapd = []
            # Loop through comps
            for comp in comps:
                # Skip if not used
                if comp not in compmap: continue
                # Save the component
                cmapd.append(compmap[comp])
            # Check length
            if len(cmapd) == 0:
                # No matches
                continue
            elif len(cmapd) == 1:
                # Save single match
                self.Conf[face] = cmapd[0]
                # Try to set the config value
                try:
                    self.config.faces[face] = cmapd[0]
                except AttributeError:
                    pass
            else:
                # Save list
                self.Conf[face] = cmapd
                # Try to set the config value
                try:
                    self.config.faces[face] = cmapd
                except AttributeError:
                    pass
        # Output compmap
        return compmap

    # Extract and write subtris after mapping
    def ExtractMappedComps(self, tric, comps=None, **kw):
        r"""Map component names from a template *tri* and write component files

        :Call:
            >>> tris = tri.ExtractMappedComps(tric, comps=[], **kw)
            >>> triu = tri.ExtractMappedComps(tric, comps=[], join=True, **kw)
        :Inputs:
            *tri*: :class:`cape.tri.Tri` | :class:`cape.tri.Triq`
                Triangulation or annotated triangulation instance
            *tric*: :class:`cape.tri.Tri`
                Triangulation with alternative component labels
            *comps*: :class:`list` (:class:`str`)
                List of *tric* faces to write
            *join*: ``True`` | {``False``}
                Return a single triangulation with all *comps*
        :Outputs:
            *tris*: :class:`dict` (:class:`cape.tri.Tri`)
                Dictionary of triangulations for each *comp* in *comps*
            *triu*: :class:`cape.tri.Tri`
                Single joined triangulation if *join* is ``True``
        :Versions:
            * 2016-02-10 ``@ddalle``: Version 1.0
        """
        # Initialize output
        tris = {}
        # Verbose
        v = kw.get("v", False)
        # Default component list: ALL
        if comps is None:
            # Get from *tric.config* or *tric.Conf*
            try:
                # Read from JSON-based config interface
                comps = tric.config.comps
            except AttributeError:
                try:
                    # Read from UH3D face dictionary
                    comps = tric.Conf.keys()
                except AttributeError:
                    # No components
                    comps = []
        else:
            # Ensure input components makes a list
            comps = list(np.array(comps).flatten())
        # Extract requested components
        trik = tric.GetSubTri(comps)
        # Perform mapping
        tri = self.Copy()
        tri.MapTriCompID(trik, **kw)
        # Check for joined
        if kw.get("join", False):
            # Extract components
            triu = tri.GetSubTri(comps)
            # Output
            return triu
        # Loop through components
        for comp in comps:
            # Check type
            if type(comp).__name__.startswith("int"):
                raise TypeError(
                    "Component '%s' is an integer; must be a string" % comp)
            # Status update
            if v:
                print("Mapping and extracting component '%s'" % comp)
            # Extract the component
            trii = tri.GetSubTri(comp)
            # Save it
            tris[comp] = trii
        # Output
        return tris

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
            * 2016-10-21 ``@ddalle``: Version 1.0
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
        # Initialize array of nodes touched
        ntouch = np.ones(self.nNode, dtype=bool)
        ttouch = np.ones(self.nTri,  dtype=bool)
        qtouch = np.ones(self.nQuad, dtype=bool)
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
                raise ValueError(
                    "  Component '%s' had no 'BC' or 'aflr3_bc' property"
                    % comp)
            # Apply to the appropriate tris and quads
            if len(IT) > 0:
                # Set BCs
                self.BCs[IT] = BC
                # Note touched BCs
                ttouch[IT] = False
            if len(IQ) > 0:
                # Set BCs
                self.BCsQuad[IQ] = BC
                # Note touched BCs
                qtouch[IQ] = False

            # Get the boundary layer growth parameters
            blds = self.config.GetProperty(comp, 'blds')
            bldel = self.config.GetProperty(comp, 'bldel')
            # Check for nodes in this component
            if len(IN) > 0:
                # Check for viscous BL with no spacing
                if (BC < 0) and ((blds is None) or (blds <= 0.0)):
                    raise ValueError(
                        ("  Component '%s' has viscous BL " % comp) +
                        ("(aflr_bc=-1) but no blds=%s" % blds))
                # Nodes touched
                ntouch[IN] = False
                # Check for the property
                if blds  is not None: self.blds[IN]  = blds
                if bldel is not None: self.bldel[IN] = bldel
            else:
                # Status message for ignored component
                print("  Component '%s' has no nodes" % comp)
        # Check for untouched triangles
        if np.any(ttouch):
            # Get said untouched triangles
            K = np.where(ttouch)[0]
            # Count
            nK = K.size
            # Warning
            print(
                ("  WARNING [MapBCs_ConfigAFLR3]: ") +
                ("%i tris with no BC information" % nK))
            # Get face names
            faces = self.GetFacesFromTris(K, nmin=10)
            # List of faces
            print("  Faces with at least 10 triangles:")
            for face in faces:
                print("    %s" % face)
        # Check for untouched triangles
        if np.any(qtouch):
            # Get said untouched triangles
            K = np.where(qtouch)[0]
            # Count
            nK = K.size
            # Warning
            print(
                ("  WARNING [MapBCs_ConfigAFLR3]: ") +
                ("%i quads with no BC information" % nK))
            # Get face names
            faces = self.GetFacesFromQuads(K, nmin=10)
            # List of faces
            print("  Faces with at least 10 quads:")
            for face in faces:
                print("    %s" % face)


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
            * 2015-11-19 ``@ddalle``: Version 1.0
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
                print(
                    ("Warning [MapBCs_AFLR3]: ") +
                    ("No nodes mapped for component '%s'" % comp))
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
            * 2015-11-19 ``@ddalle``: Version 1.0
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
    # Get coordinates of nodes for each triangle
    def GetTriNodes(self):
        """Get the nodal coordinates of each triangle

        :Call:
            >>> tri.GetTriNodes()
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
        :Attributes:
            *tri.TriX*: :class:`np.ndarray` (:class:`float` shape=(nTri,3))
                *x*-coordinates of each node of each triangle
            *tri.TriY*: :class:`np.ndarray` (:class:`float` shape=(nTri,3))
                *y*-coordinates of each node of each triangle
            *tri.TriZ*: :class:`np.ndarray` (:class:`float` shape=(nTri,3))
                *z*-coordinates of each node of each triangle
        :Versions:
            * 2017-12-22 ``@ddalle``: Version 1.0
        """
        # Check for centers
        try:
            self.TriX
            self.TriY
            self.TriZ
            return
        except AttributeError:
            pass
        # Calculate the center of each tri, one coordinate at a time
        self.TriX = self.Nodes[self.Tris-1, 0]
        self.TriY = self.Nodes[self.Tris-1, 1]
        self.TriZ = self.Nodes[self.Tris-1, 2]

    # Get centers of nodes
    def GetCenters(self):
        """Get the centroids of each triangle

        :Call:
            >>> tri.GetCenters()
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
        :Attributes:
            *tri.Centers*: :class:`np.ndarray` (:class:`float` shape=(nTri,3))
                Center of each triangle
        :Versions:
            * 2017-02-09 ``@ddalle``: Version 1.0
        """
        # Check for centers
        try:
            self.Centers
            return
        except AttributeError:
            pass
        # Calculate the center of each tri, one coordinate at a time
        x = np.mean(self.Nodes[self.Tris-1, 0], axis=1)
        y = np.mean(self.Nodes[self.Tris-1, 1], axis=1)
        z = np.mean(self.Nodes[self.Tris-1, 2], axis=1)
        # Save the centers
        self.Centers = util.stackcol((x,y,z))

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
            * 2014-06-12 ``@ddalle``: Version 1.0
            * 2016-01-23 ``@ddalle``: Added a check before calculating
        """
        # Check for normals.
        try:
            self.Normals
            return
        except AttributeError:
            pass
        # Extract the vertices of each tri.
        x = self.Nodes[self.Tris-1, 0]
        y = self.Nodes[self.Tris-1, 1]
        z = self.Nodes[self.Tris-1, 2]
        # Get the deltas from node 0 to node 1 or node 2
        x01 = util.stackcol((x[:,1]-x[:,0], y[:,1]-y[:,0], z[:,1]-z[:,0]))
        x02 = util.stackcol((x[:,2]-x[:,0], y[:,2]-y[:,0], z[:,2]-z[:,0]))
        # Calculate the dimensioned normals
        n = np.cross(x01, x02)
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

    # Get normals and areas
    def GetAreaVectors(self):
        """Get the normals and areas of each triangle

        :Call:
            >>> tri.GetAreaVectors()
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
        :Effects:
            *tri.AreaVectors*: :class:`ndarray`, shape=(tri.nTri,)
                Area of each triangle is created
            *tri.Normals*: :class:`ndarray`, shape=(tri.nTri,3)
                Unit normal for each triangle is saved
        :Versions:
            * 2014-06-12 ``@ddalle``: Version 1.0
            * 2016-01-23 ``@ddalle``: Added a check before calculating
        """
        # Check for normals.
        try:
            self.AreaVectors
            return
        except AttributeError:
            pass
        # Extract the vertices of each tri.
        x = self.Nodes[self.Tris-1, 0]
        y = self.Nodes[self.Tris-1, 1]
        z = self.Nodes[self.Tris-1, 2]
        # Get the deltas from node 0 to node 1 or node 2
        x01 = util.stackcol((x[:,1]-x[:,0], y[:,1]-y[:,0], z[:,1]-z[:,0]))
        x02 = util.stackcol((x[:,2]-x[:,0], y[:,2]-y[:,0], z[:,2]-z[:,0]))
        # Calculate the dimensioned normals
        n = np.cross(x01, x02)
        # Save the unit normals.
        self.AreaVectors = n

    # Get right-handed coordinate system
    def GetBasisVectors(self):
        """Get a right-handed coordinate basis for all triangles

        :Call:
            >>> tri.GetBasisVectors()
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
        :Effects:
            *tri.e1*: :class:`np.ndarray` (:class:`float`, shape=(nTri,3))
                Unit vector pointing from node 1 to node 2 of each tri
            *tri.e2*: :class:`np.ndarray` (:class:`float`, shape=(nTri,3))
                Unit vector completing right-handed coordinate system
            *tri.e3*: :class:`np.ndarray` (:class:`float`, shape=(nTri,3))
                Unit normal of each triangle
        :Versions:
            * 2017-02-09 ``@ddalle``: Version 1.0
        """
        # Check for all the requested attributes
        try:
            self.e1
            self.e2
            self.e3
            return
        except AttributeError:
            pass
        # Extract the vertices of each tri.
        X = self.Nodes[self.Tris-1, 0]
        Y = self.Nodes[self.Tris-1, 1]
        Z = self.Nodes[self.Tris-1, 2]
        # Get the deltas from node 0 to node 1 or node 2
        X01 = util.stackcol((X[:,1]-X[:,0], Y[:,1]-Y[:,0], Z[:,1]-Z[:,0]))
        X02 = util.stackcol((X[:,2]-X[:,0], Y[:,2]-Y[:,0], Z[:,2]-Z[:,0]))
        # Calculate the dimensioned normals
        n = np.cross(X01, X02)
        # Calculate the area of each triangle.
        A = np.sqrt(np.sum(n**2, 1))
        # Calculate the length of each 0->1 segment
        L = np.sqrt(np.sum(X01**2, 1))
        # Normalize each component.
        e3 = n.copy()
        e3[:,0] /= A
        e3[:,1] /= A
        e3[:,2] /= A
        # Normalize 0->1 segment as tangent
        e1 = X01.copy()
        e1[:,0] /= L
        e1[:,1] /= L
        e1[:,2] /= L
        # Get final axis to complete right-handed system
        e2 = np.cross(e3, e1)
        # Save basis
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3


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
            * 2015-02-21 ``@ddalle``: Version 1.0
        """
        try:
            self.Lengths
            return
        except AttributeError:
            pass
        # Extract the vertices of each tri.
        x = self.Nodes[self.Tris-1, 0]
        y = self.Nodes[self.Tris-1, 1]
        z = self.Nodes[self.Tris-1, 2]
        # Get the deltas from node 0->1, 1->2, 2->1
        x01 = np.vstack((x[:,1]-x[:,0], y[:,1]-y[:,0], z[:,1]-z[:,0]))
        x12 = np.vstack((x[:,2]-x[:,1], y[:,2]-y[:,1], z[:,2]-z[:,1]))
        x20 = np.vstack((x[:,0]-x[:,2], y[:,0]-y[:,2], z[:,0]-z[:,2]))
        # Calculate lengths.
        self.Lengths = util.stackcol((
            np.sqrt(np.sum(x01**2, 0)),
            np.sqrt(np.sum(x12**2, 0)),
            np.sqrt(np.sum(x20**2, 0))))

    # Get nearest triangle to a point
    def GetNearestTri(self, x, n=4, **kw):
        """Get the triangle that is nearest to a point, and the distance

        :Call:
            >>> T = tri.GetNearestTri(x, n=4, **kw)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *x*: :class:`np.ndarray` (:class:`float`, shape=(3,))
                Array of *x*, *y*, and *z* coordinates of test point
            *n*: {``4``} | :class:`int`
                Number of *tri* components to search
            *ztol*: {_ztol_} | positive :class:`float`
                Maximum extra projection distance
            *rztol*: {_antol_} | positive :class:`float`
                Maximum relative projection distance
        :Outputs:
            *T*: :class:`dict`
                Dictionary of match parameters
            *T["k1"]*: :class:`int`
                Index of triangle nearest to test point
            *T["c1"]*: :class:`int`
                Component ID of triangle *k1*
            *T["d1"]*: :class:`float`
                Distance from triangle *k1* to test point
            *T["z1"]*: :class:`float`
                Projection distance of point to triangle *k1*
            *T["t1"]*: :class:`float`
                Tangential distance of point to triangle *k1*
            *T["k2"]*: ``None`` | :class:`int`
                Index of nearest triangle outside component *c1*
            *T["c2"]*: :class:`int`
                Component ID of triangle *k2*
            *T["d2"]*: :class:`float`
                Distance from triangle *k2* to test point
            *T["z2"]*: :class:`float`
                Projection distance of point to triangle *k2*
            *T["k3"]*: ``None`` | :class:`int`
                Index of nearest trioutside components *c1* and *c2*
            *T["k4"]*: ``None`` | :class:`int`
                Index of nearest trioutside components *c1*, *c2*, *c3*
        :Versions:
            * 2017-02-06 ``@ddalle``: Version 1.0
            * 2017-02-07 ``@ddalle``: Version 1.1; search for 2nd comp
            * 2017-02-08 ``@ddalle``: Version 1.2; 3rd and 4th comp
        """
        # Get coordinates
        self.GetBasisVectors()
        # Extract coordinate basis function
        e1 = self.e1
        e2 = self.e2
        e3 = self.e3
        # Extract the vertices of each tri.
        self.GetTriNodes()
        X = self.TriX
        Y = self.TriY
        Z = self.TriZ
        # Extract test point coordinates
        y = x[1]
        z = x[2]
        x = x[0]
        # Get the projection distance
        zi = (x-X[:,0])*e3[:,0] + (y-Y[:,0])*e3[:,1] + (z-Z[:,0])*e3[:,2]
        zi = np.abs(zi)
        # Get minimum projection distance
        kmin = np.nanargmin(zi)
        zmin = zi[kmin]
        # Process max tol
        ztol = kw.get("ztol", ztoldef)
        rztol = kw.get("rztol", rztoldef)
        # Scale of vehicle
        bbox = self.GetCompBBox()
        # Use largest dimension of bbox
        Lref = np.max(bbox[1::2] - bbox[::2])
        # Relative tolerance
        ztol = ztol + rztol*Lref
        # Get indices of points within *zmin* and *ztol*
        I = zi <= zmin + ztol
        K = np.where(I)[0]
        # Preselect subsets
        XI = X[I, :]
        YI = Y[I, :]
        ZI = Z[I, :]
        # Filter best 25 candidates
        if K.size > 25:
            # Centers
            XC = np.mean(XI, axis=1)
            YC = np.mean(YI, axis=1)
            ZC = np.mean(ZI, axis=1)
            # L1 distance to each center
            L1 = np.abs(XC - x) + np.abs(YC - y) + np.abs(ZC - z)
            # Sort closest 25
            J = np.argsort(L1)[:25]
            K = K[J]
            # Redo subsets
            XI = XI[J, :]
            YI = YI[J, :]
            ZI = ZI[J, :]
        # These operations are tested to run as fast as possible
        XI0, XI1, XI2 = XI.T
        YI0, YI1, YI2 = YI.T
        ZI0, ZI1, ZI2 = ZI.T
        # Downselect the basis vectors
        e10, e11, e12 = e1[K, :].T
        e20, e21, e22 = e2[K, :].T
        # Convert the test point into coordinates aligned with first edge
        xi = (x-XI0)*e10 + (y-YI0)*e11 + (z-ZI0)*e12
        yi = (x-XI0)*e20 + (y-YI0)*e21 + (z-ZI0)*e22
        zi = zi[K]
        # Initialize transformed triangles
        XI = np.zeros_like(XI)
        YI = np.zeros_like(XI)
        # Convert the second and third vertices
        # The commented line should be all zeros
        XI[:, 1] = ((XI1-XI0)*e10 + (YI1-YI0)*e11 + (ZI1-ZI0)*e12)
        XI[:, 2] = ((XI2-XI0)*e10 + (YI2-YI0)*e11 + (ZI2-ZI0)*e12)
        # YI[:, 1] = ((XI1-XI0)*e20 + (YI1-YI0)*e21 + (ZI1-ZI0)*e22)
        YI[:, 2] = ((XI2-XI0)*e20 + (YI2-YI0)*e21 + (ZI2-ZI0)*e22)
        # Get distance to each triangle within the plane of each triangle
        DI = geom.dist2_tris_to_pt(XI, YI, xi, yi)
        # Get total distance from point to each triangle
        D = zi*zi + DI
        # Get index of minimum distance
        i1 = np.nanargmin(D)
        k1 = K[i1]
        # Find the component ID
        c1 = self.CompID[k1]
        # Initialize output
        T = {
            "k1": k1,
            "c1": c1,
            "d1": np.sqrt(D[i1]),
            "t1": DI[i1],
            "z1": abs(zi[i1]),
        }
        # Initialize submask
        I1 = K != c1
        C1 = self.CompID[I]
        # Loop through until we find up to four components
        for nj in range(n-1):
            # Tag
            sj = str(nj + 2)
            # Downselect available triangle indices
            J = np.where(I1)[0]
            # Check for no remaining triangles
            if J.size == 0:
                return T
            # Find nearest match from remaining triangles
            i = np.nanargmin(D[J])
            j = J[i]
            k = K[j]
            c = self.CompID[k]
            # Save parameters
            T["k"+sj] = k
            T["c"+sj] = c
            T["d"+sj] = np.sqrt(D[j])
            T["z"+sj] = zi[j]
            T["t"+sj] = DI[j]
            # Update mask
            I1[C1 == c] = False
        # Output (if 4 components)
        return T
    # Edit default tolerances
    GetNearestTri.__doc__=GetNearestTri.__doc__.replace("_ztol_",str(ztoldef))
    GetNearestTri.__doc__=GetNearestTri.__doc__.replace("_rztol_",str(rztoldef))

    # Get tris by bbox
    def FilterTrisBBox(self, bbox):
        """Get the list of Tris in a specified rectangular prism

        :Call:
            >>> K = tri.FilterTrisBBox(bbox)
            >>> K = tri.FilterTrisBBox([xmin, xmax, ymin, ymax, zmin, zmax])
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *bbox*: :class:`list` | :class:`np.ndarray`
                List of minimum and maximum coordinates
        :Outputs:
            *K*: :class:`np.ndarray` (:class:`int`)
                List of 1-based tri numbers that intersect BBox
        :Versions:
            * 2017-02-17 ``@ddalle``: Version 1.0
        """
        # Compute vertices
        x = self.Nodes[self.Tris-1,0]
        y = self.Nodes[self.Tris-1,1]
        z = self.Nodes[self.Tris-1,2]
        # Unpack inputs
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        # Initialize array
        K = (self.CompID > -1)
        # Go through each coordinate
        K = np.logical_and(K, np.min(x,axis=1) <= xmax)
        K = np.logical_and(K, np.max(x,axis=1) >= xmin)
        K = np.logical_and(K, np.min(y,axis=1) <= ymax)
        K = np.logical_and(K, np.max(y,axis=1) >= ymin)
        K = np.logical_and(K, np.min(z,axis=1) <= zmax)
        K = np.logical_and(K, np.max(z,axis=1) >= zmin)
        # Output
        return np.where(K)[0]
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
            * 2016-01-23 ``@ddalle``: Version 1.0
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
            * 2016-09-29 ``@ddalle``: Version 1.0
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

    # Get edges
    def GetEdgeTable(self):
        """Get the list of edges and triangle of origin

        :Call:
            >>> tri.GetEdgeTable()
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
                Triangulation instance
        :Effects:
            *tri.EdgeTable*: :class:`np.ndarray`, shape=(3*nTri, 3)
                Array of node indices defining each edge
        :Versions:
            * 2019-06-20 ``@ddalle``: Version 1.0
        """
        # Check for edges
        try:
            self.EdgeTable
            return
        except Exception:
            pass
        # Groups: edges from 0->1, 1->2, 2->0
        i0 = 0
        i1 = self.nTri
        i2 = 2*i1
        i3 = 3*i1
        # Initialize table of edge node 0, node 1, tri index
        E = np.zeros((3*self.nTri, 3), dtype="int")
        # Triangulation handles
        T = self.Tris
        # Edge 0: 0->1
        E[i0:i1, 0] = T[:,0]
        E[i0:i1, 1] = T[:,1]
        E[i0:i1, 2] = np.arange(1, self.nTri+1)
        # Edge 1: 1->2
        E[i1:i2, 0] = T[:,1]
        E[i1:i2, 1] = T[:,2]
        E[i1:i2, 2] = E[i0:i1, 2]
        # Edge 2: 2->0
        E[i2:i3, 0] = T[:,2]
        E[i2:i3, 1] = T[:,0]
        E[i2:i3, 2] = E[i0:i1, 2]
        # Sort by end node then start node
        I = np.lexsort((E[:,1], E[:,0]))
        # Save sorted edges
        self.EdgeTable = E[I,:]

    # Find neighbor
    def FindTriFromEdge(self, i0, i1):
        """Find the triangle index from a specified edge

        :Call:
            >>> k = tri.FindTriFromEdge(i0, i1)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
                Triangulation instance
            *i0*: :class:`int` > 0
                Edge start node index [1-based]
            *i1*: :class:`int` > 0
                Edge end node index [1-based]
        :Outputs:
            *k*: :class:`int` >= 0
                Triangle index containing edge *i0* -> *i1* [1-based],
                if no match, returns ``0``
        :Versions:
            * 2019-06-20 ``@ddalle``: Version 1.0
        """
        # Get edge table
        self.GetEdgeTable()
        # Get handle
        ET = self.EdgeTable
        # Find edges starting with node *i0*
        j0 = np.where(ET[:,0] == i0)[0]
        # Of these, find the one with *i1* as the end
        j1 = np.where(ET[j0,1] == i1)[0]
        # Check validity
        if j1.size != 1:
            return 0
        # Get edge index
        j = j0[j1[0]]
        # Get triangle index from table
        return ET[j, 2]


    # Find neighbors of a triangle
    def FindNeighbors(self, k):
        """Find the triangles neighboring one triangle

        :Call:
            >>> k0, k1, k2 = tri.FindNeighbors(k)
        :Inputs:
            *tri*: :class:`cape.tri.TriBase`
                Triangulation instance
            *k*: :class:`int` > 0
                Triangle index [1-based]
        :Outputs:
            *K*: :class:`np.ndarray`\ [:class:`int`]
                Neighboring triangles of *k*
            *k0*: :class:`int` >= 0
                Triangle index sharing edge 1 of triangle *k*
            *k1*: :class:`int` >= 0
                Triangle index sharing edge 1 of triangle *k*
            *k2*: :class:`int` >= 0
                Triangle index sharing edge 1 of triangle *k*
        :Versions:
            * 2019-06-20 ``@ddalle``: Version 1.0
        """
        # Get nodes of this triangle
        i0, i1, i2 = self.Tris[k]
        # Find triangles containing reversed edges
        k0 = self.FindTriFromEdge(i1, i0)
        k1 = self.FindTriFromEdge(i2, i1)
        k2 = self.FindTriFromEdge(i0, i2)
        # Output
        return np.array([k0, k1, k2])
   # }

   # ++++++++++
   # Components
   # ++++++++++
   # {
    # Get normals and areas
    def GetCompArea(self, compID=None, n=None):
        """
        Get the total area of a component, or get the total area of a component
        projected to a plane with a given normal vector.

        :Call:
            >>> A = tri.GetCompArea(compID)
            >>> A = tri.GetCompArea(compID, n)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *compID*: {``None``} | :class:`int`
                Index of the component of which to find the area
            *n*: :class:`numpy.ndarray`
                Unit normal vector to use for projection
        :Outputs:
            *A*: :class:`float`
                Area of the component
        :Versions:
            * 2014-06-13 ``@ddalle``: Version 1.0
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
    def GetCompAreaVector(self, compID, n=None):
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
            * 2014-06-13 ``@ddalle``: Version 1.0
        """
        # Check for areas.
        self.GetAreaVectors()
        # Find the indices of tris in the component.
        k = self.GetTrisFromCompID(compID)
        # Add up component areas
        return np.sum(self.AreaVectors[k], axis=0)

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
            * 2014-06-13 ``@ddalle``: Version 1.0
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
            * 2016-03-29 ``@ddalle``: Version 1.0
        """
        # Check for areas.
        try:
            self.Areas
        except AttributeError:
            self.GetNormals()
        # Get tris
        k = self.GetTrisFromCompID(compID)
        # Check for no triangles
        if k.size == 0:
            raise ValueError("Found no tris for comp '%s'" % compID)
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
    def GetCompBBox(self, compID=None, **kwargs):
        """
        Find a bounding box based on the coordinates of a specified component
        or list of components, with an optional buffer or buffers in each
        direction

        :Call:
            >>> xlim = tri.GetCompBBox(compID, **kwargs)
        :Inputs:
            *tri*: :class:`cape.tri.Tri`
                Triangulation instance
            *compID*: {``None``} | :class:`int` | :class:`str` | :class:`list`
                Component or list of components to use for bounding box; if
                ``None`` return bounding box for entire triangulation
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
            * 2014-06-16 ``@ddalle``: Version 1.0
            * 2014-08-03 ``@ddalle``: Changed "buff" --> "pad"
            * 2017-02-08 ``@ddalle``: CompID ``None`` gets BBox for full tri
        """
        # List of components; initialize with first.
        i = self.GetTrisFromCompID(compID)
        # Check for null component
        if i is None or len(i) == 0:
            return
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

    # Get length of diagonal of BBox
    def GetCompScale(self, compID=None, **kw):
        """Get diagonal length of bounding box of a component(s)

        :Call:
            >>> L = tri.GetCompScale(compID, **kw)
        :Inputs:
            *compID*: {``None``} | :class:`int` | :class:`str` | :class:`list`
                Component or list of components to use for bounding box; if
                ``None`` return bounding box for entire triangulation
            *pad*: :class:`float`
                Buffer to add in each dimension to min and max coordinates
            *kw*: :class:`dict`
                Keyword arguments passed to :func:`GetCompBBox`
        :Outputs:
            *L*: nonnegative :class:`float`
                Length of the diagonal of the bounding box
        :Versions:
            * 2017-02-08 ``@ddalle``: Version 1.0
        """
        # Get the bounding box
        BBox = self.GetCompBBox(compID, **kw)
        # Check for null result
        if BBox is None: return 0.0
        # Get the components
        dx = BBox[1] - BBox[0]
        dy = BBox[3] - BBox[2]
        dz = BBox[5] - BBox[4]
        # Get the length
        return np.sqrt(dx*dx + dy*dy + dz*dz)

    def GetCompProjectedArea(self, nhat, compID=None, ds=None, **kw):
        r"""Get projected area of a component(s)

        :Call:
            >>> L = tri.GetCompProjectedArea(compID, **kw)
        :Inputs:
            *nhat*: :class:`np.ndarray`\ [:class:`float`]
                Projection vector [*nx*, *ny*, *nz*]
            *compID*: {``None``} | :class:`int` | :class:`str`
                Component or list of components to use for area
                projection; if``None`` return projected area for entire
                triangulation
            *ds*: :class:`float`
                Resolution of projection plane 
            *img*: {``None``} | :class:`str`
                Optional file name for projection figure
        :Outputs:
            *A*: :class:`float`
                Projected area 
        :Versions:
            * 2020-11-05 ``@dschauer``: Version 1.0
            * 2020-11-13 ``@ddalle``: Version 2.0
                - Vectorized masking of projected plane
                - Driven by triangles instead of grid

            * 2020-11-20 ``@ddalle``: Version 2.1
                - Remove debug hooks
                - Add *img* for output fig generation
                - Clean up unit vectors and docstring
        """
        # Check for trivial *nhat*
        if not isinstance(nhat, np.ndarray):
            raise ValueError(
                "Projection vector must be ndarray, got '%s'" % type(nhat))
        elif nhat.size !=3:
            raise IndexError(
                "Projection vector has length %i; expected 3" % nhat.size)
        elif np.max(np.abs(nhat)) < 1e-8:
            raise ValueError(
                "Projection vector has zero or near-zero magnitude")
        # Get the bounding box of the component
        bbox = self.GetCompBBox(compID)
        # Exit if *compID* not found
        if bbox is None:
            # Not possible
            return
        # Get default *ds* if necessary
        if ds is None:
            # Get the diagonal length
            L = np.sqrt(np.sum((bbox[1::2]-bbox[::2])**2))
            # Use .001 times that
            ds = 1e-3 * L

        # Define y and z unit vectors
        yhat = np.array([0.0, 1.0, 0.0])
        zhat = np.array([0.0, 0.0, 1.0])
        zero = np.array([0.0, 0.0, 0.0])
        # Get vectors of the projection plane
        e1v = np.cross(yhat, nhat)
        # Check for trivial local *x* (*nhat* parallel to *yhat*)
        if np.max(np.abs(e1v)) < min(ds, 0.001):
            # Use *zhat* instead of *yhat*
            e1v = np.cross(zhat, nhat)
        # Complete right-handed system
        e2v = np.cross(nhat, e1v)
        # Get unit vectors
        e1 = e1v / np.sqrt((e1v**2).sum())
        e2 = e2v / np.sqrt((e2v**2).sum())

        # Project nodes to the plane
        e1p = np.dot(self.Nodes, e1)
        e2p = np.dot(self.Nodes, e2)

        # Find the bounds for the projection plane, add some pad
        pad = 2.0 * ds 
        # Only count nodes in *compID*
        I = self.GetNodesFromCompID(compID)
        # Min/max of projected points
        e1p_min = np.min(e1p[I]) - pad
        e1p_max = np.max(e1p[I]) + pad
        e2p_min = np.min(e2p[I]) - pad
        e2p_max = np.max(e2p[I]) + pad

        # Discretize the projection plane, use nodes not midpoints?
        e1p_np = int(np.ceil((e1p_max - e1p_min)/ds))
        e2p_np = int(np.ceil((e2p_max - e2p_min)/ds))
        e1p_dis = np.linspace(e1p_min, e1p_max, e1p_np+1)
        e2p_dis = np.linspace(e2p_min, e2p_max, e2p_np+1)
        # Switch to midpoints
        e1p_dis = 0.5*(e1p_dis[:-1] + e1p_dis[1:])
        e2p_dis = 0.5*(e2p_dis[:-1] + e2p_dis[1:])
        # Create a 2D mesh
        e1grid, e2grid = np.meshgrid(e1p_dis, e2p_dis, indexing='ij')
        # Flatten back into longer 1D arrays
        e1grid = e1grid.flatten()
        e2grid = e2grid.flatten()

        # Area of a single square on the projection plane
        de1p = e1p_dis[1] - e1p_dis[0]
        de2p = e2p_dis[1] - e2p_dis[0]
        a = de1p * de2p

        # Create a mask for the discretized projection plane
        mask = np.zeros(e1p_np*e2p_np, dtype="bool")

        # Get the triangles in *compID*
        K = self.GetTrisFromCompID(compID)
        # Unpack the triangles using zero-based indexing
        T = self.Tris[K] - 1
        # Get the edges of the triangles
        xt1 = e1p[T[:,1]] - e1p[T[:,0]]
        xt2 = e1p[T[:,2]] - e1p[T[:,1]]
        xt3 = e1p[T[:,0]] - e1p[T[:,2]]
        yt1 = e2p[T[:,1]] - e2p[T[:,0]]
        yt2 = e2p[T[:,2]] - e2p[T[:,1]]
        yt3 = e2p[T[:,0]] - e2p[T[:,2]]
        # Assemble edge vectors?
        # Get the normals
        zt = xt1*yt2 - xt2*yt1
        # Figure out which triangles need to be flipped
        tmask = zt < 0
        # Flip them
        T[tmask,:] = T[tmask,::-1]

        # Loop through triangles
        for t in T:
            # Calculate the vector from node 0 to the whole discr mesh
            xj = e1grid - e1p[t[0]]
            yj = e2grid - e2p[t[0]]
            # Get the first tangent vector
            xtj = e1p[t[1]] - e1p[t[0]]
            ytj = e2p[t[1]] - e2p[t[0]]
            # Calculate dot product and flag positive ones
            # Here (xtj, ytj) is the vector along edge0
            # And then (-ytj, xtj) is the vector that points left of it
            # *maskj* marks points in the discretized plane left of edge
            maskj = -ytj*xj + xtj*yj >= 0.0
            # Calculate the vector from node 1 to mesh
            xj = e1grid - e1p[t[1]]
            yj = e2grid - e2p[t[1]]
            # Tangent of edge 1
            xtj = e1p[t[2]] - e1p[t[1]]
            ytj = e2p[t[2]] - e2p[t[1]]
            # Calculate points to the left of edge 1 (and edge 0)
            maskj = np.logical_and(maskj, -ytj*xj + xtj*yj >= 0.0)
            # Now node 2 to mesh
            xj = e1grid - e1p[t[2]]
            yj = e2grid - e2p[t[2]]
            # Tangent of edge 2
            xtj = e1p[t[0]] - e1p[t[2]]
            ytj = e2p[t[0]] - e2p[t[2]]
            # Calculate points to the left of edge 2 (and edges 0 and 1)
            maskj = np.logical_and(maskj, -ytj*xj + xtj*yj >= 0.0)
            # Update the global mask
            mask = np.logical_or(mask, maskj)

        # Output file name
        img = kw.get("img")
        # Plot the results if requested
        if img is not None:
            # Just-in-time PyPlot import
            pmpl.mpl._import_pyplot()
            # Handle to usual PyPlot module
            plt = pmpl.mpl.plt
            # Reverse mask
            mask_ = np.logical_not(mask)
            # Get new figure
            fig = plt.figure()
            # Draw points inside and outside of projection
            plt.plot(e1grid[mask], e2grid[mask], 'ro', markersize=2)
            plt.plot(e1grid[mask_], e2grid[mask_], 'bo', markersize=2)
            # Draw projected triangulation
            plt.triplot(e1p, e2p, T, lw=0.2, color='k')
            # Save figure
            fig.savefig(img)

        # Calculate the area
        A = a * np.sum(mask)
        # Output
        return A


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
            * 2016-09-29 ``@ddalle``: Version 1.0
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
            * 2016-09-29 ``@ddalle``: Version 1.0
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
            * 2016-09-29 ``@ddalle``: Version 1.0
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
            * 2016-09-29 ``@ddalle``: Version 1.0
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
            * 2015-01-23 ``@ddalle``: Version 1.0
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
        copy(os.path.join(util.TECPLOT_TEMPLATES, 'iso-comp.lay'), '.')
        copy(os.path.join(util.TECPLOT_TEMPLATES, 'iso-comp.mcr'), '.')
        # Get the command for tecplot
        t360 = util.GetTecplotCommand()
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
            * 2015-01-23 ``@ddalle``: Version 1.0
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
            * 2015-11-22 ``@ddalle``: Version 1.0
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
        copy(os.path.join(util.PARAVIEW_TEMPLATES, 'cape_stl.py'), '.')
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
            * 2014-05-23 ``@ddalle``: Version 1.0
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
            * 2014-05-27 ``@ddalle``: Version 1.0
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
        >>> tri = cape.Tri(cgns=cgns, c=None)
        >>> tri = cape.Tri(unv=unv, c=None)
        >>> tri = cape.Tri(nNode=nNode, Nodes=Nodes, **kw)
    :Inputs:
        *fname*: :class:`str`
            Name of triangulation file to read (Cart3D format)
        *surf*: :class:`str`
            Name of AFLR3 surface file
        *uh3d*: :class:`str`
            Name of triangulation file (UH3D format)
        *cgns*: :class:`str`
            Name of mixed quad/tri file (CGNS format)
        *unv*: :class:`str`
            Name of IDEAS surface triangulation file
        *c*: :class:`str`
            Name of configuration file (usually ``Config.xml`` or ``pyfun.json``)
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
        * 2014-05-23 ``@ddalle``: Version 1.0
        * 2016-04-05 ``@ddalle``: Many input formats
    """
    # Initialization method
    def __init__(self, fname=None, c=None, **kw):
        """Initialization method

        :Versions:
            * 2014-05-23 ``@ddalle``: Version 1.0
            * 2014-06-02 ``@ddalle``: Added UH3D reading capability
            * 2016-04-05 ``@ddalle``: Added AFLR3 and cleaned up inputs
        """
        # Save file name
        self.fname = fname
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
        elif 'cgns' in kw:
            # Read CGNS surface format
            self.ReadCGNS(kw['cgns'])
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

        # Read UH3D-like component list
        if 'uh3dc' in kw:
            # Read a UH3D component list
            self.ReadUH3DCompIDList(kw['uh3dc'])

        # Check for configuration
        if 'xml' in kw:
            # Read a Config.xml type of file
            self.ReadConfigXML(kw['xml'])
        elif 'json' in kw:
            # Read a Config.json type of file
            self.ReadConfigJSON(kw['json'])
        elif 'mixsur' in kw:
            # Read a mixsur.inp type of file
            self.ReadConfigMIXSUR(kw['mixsur'])
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
            * 2014-05-27 ``@ddalle``: Version 1.0
        """
        return '<cape.tri.Tri(nNode=%i, nTri=%i)>' % (self.nNode, self.nTri)



# Regular triangulation class
class Triq(TriBase):
    r"""Class for surface geometry with solution values at each point

    This class is based on the concept of Cart3D ``triq`` files, which
    are also utilized by some Overflow utilities, including ``overint``.

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
  # ======
  # Config
  # ======
  # <
    # Initialization method
    def __init__(self, fname=None, n=1, nNode=None, Nodes=None, c=None,
        nTri=None, Tris=None, CompID=None, nq=None, q=None):
        r"""Initialization method

        :Versions:
            * 2014-05-23 ``@ddalle``: Version 1.0
            * 2014-06-02 ``@ddalle``: Added UH3D reading capability
        """
        # Save file name
        self.fname = fname
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
            self.ReadConfig(c)

    # Method that shows the representation of a triangulation
    def __repr__(self):
        """Return the string representation of a triangulation.

        This looks like ``<cape.tri.Triq(nNode=M, nTri=N)>``

        :Versions:
            * 2014-05-27 ``@ddalle``: Version 1.0
        """
        return '<cape.tri.Triq(nNode=%i, nTri=%i, nq=%i)>' % (
            self.nNode, self.nTri, self.nq)
  # >

  # ================
  # Modified Writers
  # ================
  # <
    # Function to write a .triq file
    def Write(self, fname, **kw):
        """Write a q-triangulation ``.triq`` file

        :Call:
            >>> triq.Write(fname, **kw)
        :Inputs:
            *triq*: :class:`cape.tri.Triq`
                Triangulation instance
            *fname*: :class:`str`
                Name of triangulation file to write
            *b4*: ``True`` | {``False``}
                Write single-precision big-endian
        :Versions:
            * 2015-09-14 ``@ddalle``: Version 1.0
        """
        self.WriteTriq(fname, **kw)
  # >

  # =========
  # Averaging
  # =========
  # <
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
            * 2015-09-14 ``@ddalle``: Version 1.0
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
  # >

  # ==============
  # Interpolation
  # ==============
  # <
    # Interpolate state
    def InterpSurfPoint(self, x, **kw):
        r"""Interpolate *triq.q* to the nearest point on the surface

        :Call:
            >>> x0, q = triq.InterpSurfPoint(x, **kw)
        :Inputs:
            *triq*: :class:`cape.tri.Triq`
                Annotated triangulation interface
            *x*: :class:`np.ndarray` (:class:`float`, shape=(3,))
                Array of *x*, *y*, and *z* coordinates of test point
            *k*: {``None``} | :class:`int`
                Pre-specified index of nearest triangle (0-based)
            *k1*: {``None``} | :class:`int`
                Pre-specified index of nearest triangle (1-based)
            *z*: {``None``} | :class:`float`
                Pre-specified projection distance of *x* to tri *k1*
            *kw*: :class:`dict`
                Keyword arguments passed to :func:`Tri.GetNearestTri`
        :Outputs:
            *x0*: :class:`np.ndarray` shape=(3,)
                Point projected onto the surface
            *q*: :class:`np.ndarray` shape=(*triq.nq*,)
                Interpolated state from *triq.q*
        :Versions:
            * 2017-10-10 ``@ddalle``: Version 1.0
            * 2018-10-12 ``@serogers``: Version 2.0; subtriangles
            * 2022-03-10 ``@ddalle``: Version 2.1; skip GetNearestTri()
        """
        # Check options
        k = kw.get("k")
        z = kw.get("z")
        k1 = kw.get("k1")
        # Try to use 1-based *tri*
        if k1 is not None and k is None:
            k = k1 - 1
        # Check if we already have nearest tri
        if k is None or z is None:
            # Get the nearest triangle to point *x*
            T = self.GetNearestTri(x, **kw)
            # Nearest triangle
            k = T["k1"]
            # Projection distance
            z = T["z1"]
        else:
            # Make sure basis vecotrs are present
            self.GetBasisVectors()
        # Extract the node numbers
        i0, i1, i2 = self.Tris[k] - 1
        # Get nodal coordinates
        x0 = self.Nodes[i0]
        x1 = self.Nodes[i1]
        x2 = self.Nodes[i2]
        # Use sub-triangles to compute weights
        # If the projected point xp is outside of the triangle,
        # then the sum of a0,a1,a2 will be greater than the total
        # area of the triangle, but this method scales the weights
        # to account for this
        #
        # Projected point
        xp = x - z * self.e3[k]
        # Dot products
        dp0 = np.cross(xp-x1, xp-x2)
        dp1 = np.cross(xp-x2, xp-x0)
        dp2 = np.cross(xp-x0, xp-x1)
        # Areas of the sub triangles
        a0 = np.sqrt(np.dot(dp0, dp0))
        a1 = np.sqrt(np.dot(dp1, dp1))
        a2 = np.sqrt(np.dot(dp2, dp2))
        # Area of the entire triangle (actually three subtriangles)
        sa = a0 + a1 + a2
        # Compute the weights for each node
        w0 = a0/sa
        w1 = a1/sa
        w2 = a2/sa
        # Get states
        q0 = self.q[i0]
        q1 = self.q[i1]
        q2 = self.q[i2]
        # Interpolation
        q = w0*q0 + w1*q1 + w2*q2
        return xp, q
  # >

  # ============
  # Force/Moment
  # ============
  # <
    # Calculate forces and moments
    def GetSkinFriction(self, comp=None, **kw):
        """Calculate vectors of pressure, momentum, and viscous forces on tris

        :Call:
            >>> cf_x, cf_y, cf_z = triq.GetSkinFriction(comp=None, **kw)
        :Inputs:
            *triq*: :class:`cape.tri.Triq`
                Annotated surface triangulation
            *comp*: {``None``} | :class:`str` | :class:`int` | :class:`list`
                Subset component ID or name or list thereof
            *incm*, *momentum*: ``True`` | {``False``}
                Include momentum (flow-through) forces in total
            *gauge*: {``True``} | ``False``
                Calculate gauge forces (``True``) or absolute (``False``)
            *save*: ``True`` | {``False``}
                Store vectors of forces for each triangle as attributes
            *xMRP*: {``0.0``} | :class:`float`
                *x*-coordinate of moment reference point
            *yMRP*: {``0.0``} | :class:`float`
                *y*-coordinate of moment reference point
            *zMRP*: {``0.0``} | :class:`float`
                *z*-coordinate of moment reference point
            *MRP*: {[*xMRP*, *yMRP*, *zMRP*]} | :class:`list` (len=3)
                Moment reference point
            *m*, *mach*: {``1.0``} | :class:`float`
                Freestream Mach number
            *RefArea*, *Aref*: {``1.0``} | :class:`float`
                Reference area
            *RefLength*, *Lref*: {``1.0``} | :class:`float`
                Reference length (longitudinal)
            *RefSpan*, *bref*: {*Lref*} | :class:`float`
                Reference span (for rolling and yawing moments)
            *Re*, *Rey*: {``1.0``} | :class:`float`
                Reynolds number per grid unit (units same as *triq.Nodes*)
            *gam*, *gamma*: {``1.4``} | :class:`float` > 1
                Freestream ratio of specific heats
        :Utilized Attributes:
            *triq.nNode*: :class:`int`
                Number of nodes
            *triq.q*: :class:`np.ndarray` (:class:`float` shape=(*nNode*,*nq*))
                Vector of 5, 9, or 13 states on each node
        :Outputs:
            *cf_x*: :class:`np.ndarray`
                *x*-component of skin friction coefficient
            *cf_y*: :class:`np.ndarray`
                *y*-component of skin friction coefficient
            *cf_z*: :class:`np.ndarray`
                *z*-component of skin friction coefficient
        :Versions:
            * 2017-04-03 ``@ddalle``: Version 1.0
        """
       # --------------
       # Viscous Forces
       # --------------
        if self.nq == 9:
            # Viscous stresses given directly
            cf_x = Q[I,6]
            cf_y = Q[I,7]
            cf_z = Q[I,8]
            # Output
            return cf_x, cf_y, cf_z
        elif self.nq < 13:
            # TRIQ file only contains inadequate info for viscous forces
            cf_x = np.zeros(nNode)
            cf_y = np.zeros(nNode)
            cf_z = np.zeros(nNode)
            # Output
            return cf_x, cf_y, cf_z
       # ------
       # Inputs
       # ------
        # Get Reynolds number per grid unit
        REY = kw.get("Re", kw.get("Rey", 1.0))
        # Freestream mach number
        mach = kw.get("RefMach", kw.get("mach", kw.get("m", 1.0)))
        # Freestream pressure and gamma
        gam  = kw.get("gamma", 1.4)
        pref = kw.get("p", 1.0/gam)
        # Dynamic pressure
        qref = 0.5*gam*pref*mach**2
        # Reference length/area
        Aref = kw.get("RefArea",   kw.get("Aref", 1.0))
        Lref = kw.get("RefLength", kw.get("Lref", 1.0))
        # Volume limiter
        SMALLVOL = kw.get("SMALLVOL", 1e-20)
        SMALLTRI = kw.get("SMALLTRI", 1e-12)
       # --------
       # Geometry
       # --------
        # Component for subsetting
        K = self.GetTrisFromCompID(comp)
        # Select nodes
        I = self.GetNodesFromCompID(comp)
        # Number of nodes and tris
        nNode = I.shape[0]
        nTri = K.shape[0]
        # Store node indices for each tri
        T = self.Tris[K,:] - 1
        v0 = T[:,0]
        v1 = T[:,1]
        v2 = T[:,2]
        # Handle to state variables
        Q = self.q
        # Extract the vertices of each tri.
        x = self.Nodes[T, 0]
        y = self.Nodes[T, 1]
        z = self.Nodes[T, 2]
        # Get the deltas from node 0->1 and 0->2
        x01 = util.stackcol((x[:,1]-x[:,0], y[:,1]-y[:,0], z[:,1]-z[:,0]))
        x02 = util.stackcol((x[:,2]-x[:,0], y[:,2]-y[:,0], z[:,2]-z[:,0]))
        # Calculate the dimensioned normals
        N = 0.5*np.cross(x01, x02)
        # Scalar areas of each triangle
        A = np.sqrt(np.sum(N**2, axis=1))
       # -----
       # Areas
       # -----
        # Calculate components
        Avec = np.sum(N, axis=0)
        # Overset grid information
        # Inverted Reynolds number [in]
        REI = mach / REY
        # Extract coordinates
        X1 = self.Nodes[v0,0]
        Y1 = self.Nodes[v0,1]
        Z1 = self.Nodes[v0,2]
        X2 = self.Nodes[v1,0]
        Y2 = self.Nodes[v1,1]
        Z2 = self.Nodes[v1,2]
        X3 = self.Nodes[v2,0]
        Y3 = self.Nodes[v2,1]
        Z3 = self.Nodes[v2,2]
        # Calculate coordinates of L=2 points
        xlp1 = X1 + Q[v0,10]
        ylp1 = Y1 + Q[v0,11]
        zlp1 = Z1 + Q[v0,12]
        xlp2 = X2 + Q[v1,10]
        ylp2 = Y2 + Q[v1,11]
        zlp2 = Z2 + Q[v1,12]
        xlp3 = X3 + Q[v2,10]
        ylp3 = Y3 + Q[v2,11]
        zlp3 = Z3 + Q[v2,12]
        # Calculate volume of prisms
        VOL = volcomp.VolTriPrism(X1,Y1,Z1, X2,Y2,Z2, X3,Y3,Z3,
            xlp1,ylp1,zlp1, xlp2,ylp2,zlp2, xlp3,ylp3,zlp3)
        # Filter small prisms
        IV = VOL > SMALLVOL
        # Filter small areas
        IV = np.logical_and(IV, A > SMALLTRI)
        # Downselect areas
        VAX = N[IV,0]
        VAY = N[IV,1]
        VAZ = N[IV,2]
        # Average dynamic viscosity
        mu = np.mean(Q[T[IV,:],6], axis=1)
        # Velocity derivatives
        UL = np.mean(Q[T[IV,:],7], axis=1)
        VL = np.mean(Q[T[IV,:],8], axis=1)
        WL = np.mean(Q[T[IV,:],9], axis=1)
        # Sheer stress multiplier
        FTMUJ = mu*REI/VOL[IV]
        # Stress flux
        ZUVW = (1.0/3.0) * (VAX*UL + VAY*VL + VAZ*WL)
        # Stress tensor
        TXX = 2.0*FTMUJ * (UL*VAX - ZUVW)
        TYY = 2.0*FTMUJ * (VL*VAY - ZUVW)
        TZZ = 2.0*FTMUJ * (WL*VAZ - ZUVW)
        TXY = FTMUJ * (VL*VAX + UL*VAY)
        TYZ = FTMUJ * (WL*VAY + VL*VAZ)
        TXZ = FTMUJ * (UL*VAZ + WL*VAX)
        # Initialize viscous forces
        Fv = np.zeros((nTri, 3))
        # Save results from non-zero volumes
        Fv[IV,0] = (TXX*VAX + TXY*VAY + TXZ*VAZ)/A[IV]
        Fv[IV,1] = (TXY*VAX + TYY*VAY + TYZ*VAZ)/A[IV]
        Fv[IV,2] = (TXZ*VAX + TYZ*VAY + TZZ*VAZ)/A[IV]
        # Initialize friction coefficients
        cf_x = np.zeros(self.nNode)
        cf_y = np.zeros(self.nNode)
        cf_z = np.zeros(self.nNode)
        # Initialize areas
        Af = np.zeros(self.nNode)
        # Add friction values weighted by areas
        cf_x[T[:,0]] += (Fv[:,0] * A)
        cf_x[T[:,1]] += (Fv[:,0] * A)
        cf_x[T[:,2]] += (Fv[:,0] * A)
        cf_y[T[:,0]] += (Fv[:,1] * A)
        cf_y[T[:,1]] += (Fv[:,1] * A)
        cf_y[T[:,2]] += (Fv[:,1] * A)
        cf_z[T[:,0]] += (Fv[:,2] * A)
        cf_z[T[:,1]] += (Fv[:,2] * A)
        cf_z[T[:,2]] += (Fv[:,2] * A)
        # Accumulate areas
        Af[T[:,0]] += A
        Af[T[:,1]] += A
        Af[T[:,2]] += A
        # Filter small areas
        Af = Af[I]
        IA = (Af > SMALLTRI)
        # Downselect
        cf_x = cf_x[I]
        cf_y = cf_y[I]
        cf_z = cf_z[I]
        # Divide by area
        cf_x[IA] /= Af[IA]
        cf_y[IA] /= Af[IA]
        cf_z[IA] /= Af[IA]
        # Output
        return cf_x, cf_y, cf_z


    # Calculate forces and moments
    def GetTriForces(self, comp=None, **kw):
        """Calculate vectors of pressure, momentum, and viscous forces on tris

        :Call:
            >>> C = triq.GetTriForces(comp=None, **kw)
        :Inputs:
            *triq*: :class:`cape.tri.Triq`
                Annotated surface triangulation
            *comp*: {``None``} | :class:`str` | :class:`int` | :class:`list`
                Subset component ID or name or list thereof
            *incm*, *momentum*: ``True`` | {``False``}
                Include momentum (flow-through) forces in total
            *gauge*: {``True``} | ``False``
                Calculate gauge forces (``True``) or absolute (``False``)
            *save*: ``True`` | {``False``}
                Store vectors of forces for each triangle as attributes
            *xMRP*: {``0.0``} | :class:`float`
                *x*-coordinate of moment reference point
            *yMRP*: {``0.0``} | :class:`float`
                *y*-coordinate of moment reference point
            *zMRP*: {``0.0``} | :class:`float`
                *z*-coordinate of moment reference point
            *MRP*: {[*xMRP*, *yMRP*, *zMRP*]} | :class:`list` (len=3)
                Moment reference point
            *m*, *mach*: {``1.0``} | :class:`float`
                Freestream Mach number
            *RefArea*, *Aref*: {``1.0``} | :class:`float`
                Reference area
            *RefLength*, *Lref*: {``1.0``} | :class:`float`
                Reference length (longitudinal)
            *RefSpan*, *bref*: {*Lref*} | :class:`float`
                Reference span (for rolling and yawing moments)
            *Re*, *Rey*: {``1.0``} | :class:`float`
                Reynolds number per grid unit (units same as *triq.Nodes*)
            *gam*, *gamma*: {``1.4``} | :class:`float` > 1
                Freestream ratio of specific heats
        :Utilized Attributes:
            *triq.nNode*: :class:`int`
                Number of nodes
            *triq.q*: :class:`np.ndarray` (:class:`float` shape=(*nNode*,*nq*))
                Vector of 5, 9, or 13 states on each node
        :Output Attributes:
            *triq.Fp*: :class:`np.ndarray` shape=(*nTri*,3)
                Vector of pressure forces on each triangle
            *triq.Fm*: :class:`np.ndarray` shape=(*nTri*,3)
                Vector of momentum (flow-through) forces on each triangle
            *triq.Fv*: :class:`np.ndarray` shape=(*nTri*,3)
                Vector of viscous forces on each triangle
        :Outputs:
            *C*: :class:`dict` (:class:`float`)
                Dictionary of requested force/moment coefficients
            *C["CA"]*: :class:`float`
                Overall axial force coefficient
        :Versions:
            * 2017-02-11 ``@ddalle``: Started
            * 2017-02-15 ``@ddalle``: Version 1.0
        """
       # ------
       # Inputs
       # ------
        # Which things to calculate
        incm = kw.get("incm", kw.get("momentum", False))
        gauge = kw.get("gauge", True)
        # Get Reynolds number per grid unit
        REY = kw.get("Re", kw.get("Rey", 1.0))
        # Freestream mach number
        mach = kw.get("RefMach", kw.get("mach", kw.get("m", 1.0)))
        # Freestream pressure and gamma
        gam  = kw.get("gamma", 1.4)
        pref = kw.get("p", 1.0/gam)
        # Dynamic pressure
        qref = 0.5*gam*pref*mach**2
        # Reference length/area
        Aref = kw.get("RefArea",   kw.get("Aref", 1.0))
        Lref = kw.get("RefLength", kw.get("Lref", 1.0))
        bref = kw.get("RefSpan",   kw.get("bref", Lref))
        # Moment reference point
        MRP = kw.get("MRP", np.array([0.0, 0.0, 0.0]))
        xMRP = kw.get("xMRP", MRP[0])
        yMRP = kw.get("yMRP", MRP[1])
        zMRP = kw.get("zMRP", MRP[2])
        # Volume limiter
        SMALLVOL = kw.get("SMALLVOL", 1e-20)
        SMALLTRI = kw.get("SMALLTRI", 1e-5)
       # --------
       # Geometry
       # --------
        # Component for subsetting
        K = self.GetTrisFromCompID(comp)
        # Number of tris
        nTri = K.shape[0]
        # Store node indices for each tri
        T = self.Tris[K,:] - 1
        v0 = T[:,0]
        v1 = T[:,1]
        v2 = T[:,2]
        # Extract the vertices of each tri.
        x = self.Nodes[T, 0]
        y = self.Nodes[T, 1]
        z = self.Nodes[T, 2]
        # Get the deltas from node 0->1 and 0->2
        x01 = util.stackcol((x[:,1]-x[:,0], y[:,1]-y[:,0], z[:,1]-z[:,0]))
        x02 = util.stackcol((x[:,2]-x[:,0], y[:,2]-y[:,0], z[:,2]-z[:,0]))
        # Calculate the dimensioned normals
        N = 0.5*np.cross(x01, x02)
        # Scalar areas of each triangle
        A = np.sqrt(np.sum(N**2, axis=1))
       # -----
       # Areas
       # -----
        # Calculate components
        Avec = np.sum(N, axis=0)
       # ---------------
       # Pressure Forces
       # ---------------
        # State handle
        Q = self.q
        # Calculate average *Cp* (first state variable)
        Cp = np.sum(Q[T,0], axis=1)/3
        # Forces are inward normals
        Fp = -util.stackcol((Cp*N[:,0], Cp*N[:,1], Cp*N[:,2]))
        # Vacuum
        Fvac = -2/(gam*mach*mach)*N
       # ---------------
       # Momentum Forces
       # ---------------
        # Check which type of state variables we have (if any)
        if self.nq < 5:
            # TRIQ file only contains pressure info
            Fm = np.zeros((nTri, 3))
        elif self.nq == 6:
            # Cart3D style: $\hat{u}=u/a_\infty$
            # Average density
            rho = np.mean(Q[T,1], axis=1)
            # Velocities
            U = np.mean(Q[T,2], axis=1)
            V = np.mean(Q[T,3], axis=1)
            W = np.mean(Q[T,4], axis=1)
            # Mass flux [kg/s]
            phi = -rho*(U*N[:,0] + V*N[:,1] + W*N[:,2])
            # Force components
            Fm = util.stackcol((phi*U,phi*V,phi*W))
        else:
            # Conventional: $\hat{u}=\frac{\rho u}{\rho_\infty a_\infty}$
            # Average density
            rho = np.mean(Q[T,1], axis=1)
            # Average mass flux components
            rhoU = np.mean(Q[T,2], axis=1)
            rhoV = np.mean(Q[T,3], axis=1)
            rhoW = np.mean(Q[T,4], axis=1)
            # Average mass flux components
            U = (Q[v0,2]/Q[v0,1] + Q[v1,2]/Q[v1,1] + Q[v2,2]/Q[v2,1])/3
            V = (Q[v0,3]/Q[v0,1] + Q[v1,3]/Q[v1,1] + Q[v2,3]/Q[v2,1])/3
            W = (Q[v0,4]/Q[v0,1] + Q[v1,4]/Q[v1,1] + Q[v2,4]/Q[v2,1])/3
            # Average mass flux, done wrongly for consistency with `triload`
            phi = -(U*N[:,0] + V*N[:,1] + W*N[:,2])
            # Force components
            Fm = util.stackcol((phi*rhoU,phi*rhoV,phi*rhoW))
       # --------------
       # Viscous Forces
       # --------------
        if self.nq == 9:
            # Viscous stresses given directly
            FXV = np.mean(Q[T,6], axis=1) * A
            FYV = np.mean(Q[T,7], axis=1) * A
            FZV = np.mean(Q[T,8], axis=1) * A
            # Force components
            Fv = util.stackcol((FXV, FYV, FZV))
        elif self.nq >= 13:
            # Overset grid information
            # Inverted Reynolds number [in]
            REI = mach / REY
            # Extract coordinates
            X1 = self.Nodes[v0,0]
            Y1 = self.Nodes[v0,1]
            Z1 = self.Nodes[v0,2]
            X2 = self.Nodes[v1,0]
            Y2 = self.Nodes[v1,1]
            Z2 = self.Nodes[v1,2]
            X3 = self.Nodes[v2,0]
            Y3 = self.Nodes[v2,1]
            Z3 = self.Nodes[v2,2]
            # Calculate coordinates of L=2 points
            xlp1 = X1 + Q[v0,10]
            ylp1 = Y1 + Q[v0,11]
            zlp1 = Z1 + Q[v0,12]
            xlp2 = X2 + Q[v1,10]
            ylp2 = Y2 + Q[v1,11]
            zlp2 = Z2 + Q[v1,12]
            xlp3 = X3 + Q[v2,10]
            ylp3 = Y3 + Q[v2,11]
            zlp3 = Z3 + Q[v2,12]
            # Calculate volume of prisms
            VOL = volcomp.VolTriPrism(X1,Y1,Z1, X2,Y2,Z2, X3,Y3,Z3,
                xlp1,ylp1,zlp1, xlp2,ylp2,zlp2, xlp3,ylp3,zlp3)
            # Filter small prisms
            IV = VOL > SMALLVOL
            # Downselect areas
            VAX = N[IV,0]
            VAY = N[IV,1]
            VAZ = N[IV,2]
            # Average dynamic viscosity
            mu = np.mean(Q[T[IV,:],6], axis=1)
            # Velocity derivatives
            UL = np.mean(Q[T[IV,:],7], axis=1)
            VL = np.mean(Q[T[IV,:],8], axis=1)
            WL = np.mean(Q[T[IV,:],9], axis=1)
            # Sheer stress multiplier
            FTMUJ = mu*REI/VOL[IV]
            # Stress flux
            ZUVW = (1.0/3.0) * (VAX*UL + VAY*VL + VAZ*WL)
            # Stress tensor
            TXX = 2.0*FTMUJ * (UL*VAX - ZUVW)
            TYY = 2.0*FTMUJ * (VL*VAY - ZUVW)
            TZZ = 2.0*FTMUJ * (WL*VAZ - ZUVW)
            TXY = FTMUJ * (VL*VAX + UL*VAY)
            TYZ = FTMUJ * (WL*VAY + VL*VAZ)
            TXZ = FTMUJ * (UL*VAZ + WL*VAX)
            # Initialize viscous forces
            Fv = np.zeros((nTri, 3))
            # Save results from non-zero volumes
            Fv[IV,0] = (TXX*VAX + TXY*VAY + TXZ*VAZ)
            Fv[IV,1] = (TXY*VAX + TYY*VAY + TYZ*VAZ)
            Fv[IV,2] = (TXZ*VAX + TYZ*VAY + TZZ*VAZ)
        else:
            # TRIQ file only contains inadequate info for viscous forces
            Fv = np.zeros((nTri, 3))
       # ------------
       # Finalization
       # ------------
        # Normalize
        Fp /= (Aref)
        Fm /= (qref*Aref)
        Fv /= (qref*Aref)
        Fvac /= (Aref)
        # Centers of nodes
        xc = np.mean(x, axis=1)
        yc = np.mean(y, axis=1)
        zc = np.mean(z, axis=1)
        # Calculate pressure moments
        Mpx = ((yc-yMRP)*Fp[:,2] - (zc-zMRP)*Fp[:,1])/bref
        Mpy = ((zc-zMRP)*Fp[:,0] - (xc-xMRP)*Fp[:,2])/Lref
        Mpz = ((zc-xMRP)*Fp[:,1] - (yc-yMRP)*Fp[:,0])/bref
        # Calculate vacuum pressure moments
        Mcx = ((yc-yMRP)*Fvac[:,2] - (zc-zMRP)*Fvac[:,1])/bref
        Mcy = ((zc-zMRP)*Fvac[:,0] - (xc-xMRP)*Fvac[:,2])/Lref
        Mcz = ((zc-xMRP)*Fvac[:,1] - (yc-yMRP)*Fvac[:,0])/bref
        # Calculate momentum moments
        Mmx = ((yc-yMRP)*Fm[:,2] - (zc-zMRP)*Fm[:,1])/bref
        Mmy = ((zc-zMRP)*Fm[:,0] - (xc-xMRP)*Fm[:,2])/Lref
        Mmz = ((zc-xMRP)*Fm[:,1] - (yc-yMRP)*Fm[:,0])/bref
        # Calculate viscous moments
        Mvx = ((yc-yMRP)*Fv[:,2] - (zc-zMRP)*Fv[:,1])/bref
        Mvy = ((zc-zMRP)*Fv[:,0] - (xc-xMRP)*Fv[:,2])/Lref
        Mvz = ((zc-xMRP)*Fv[:,1] - (yc-yMRP)*Fv[:,0])/bref
        # Assemble
        Mp = util.stackcol((Mpx,Mpy,Mpz))
        Mvac = util.stackcol((Mcx,Mcy,Mcz))
        Mm = util.stackcol((Mmx,Mmy,Mmz))
        Mv = util.stackcol((Mvx,Mvy,Mvz))
        # Add up forces
        if gauge:
            # Use *pinf* as reference pressure
            if incm:
                # Include all forces
                F = Fp + Fm + Fv
                M = Mp + Mm + Mv
            else:
                # Include viscous
                F = Fp + Fv
                M = Mp + Mv
        else:
            # Use p=0 as reference pressure
            if incm:
                # Include all forces
                F = Fp + Fvac + Fm + Fv
                M = Mp + Mvac + Mm + Mv
            else:
                # Disinclude momentum
                F = Fp + Fvac + Fv
                M = Mp + Mvac + Mv
        # Save information
        if kw.get("save", False):
            self.F = F
            self.Fp = Fp
            self.Fm = Fm
            self.Fv = Fv
            self.M = M
            self.Mc = Mc
            self.Mp = Mp
            self.Mm = Mm
            self.Mv = Mv
        # Dictionary of results
        C = {}
        # Save areas
        C["Ax"] = Avec[0]
        C["Ay"] = Avec[1]
        C["Az"] = Avec[2]
        # Total forces
        C["CA"] =  np.sum(F[:,0])
        C["CY"] =  np.sum(F[:,1])
        C["CN"] =  np.sum(F[:,2])
        C["CLL"] = np.sum(M[:,0])
        C["CLM"] = np.sum(M[:,1])
        C["CLN"] = np.sum(M[:,2])
        # Pressure contributions
        C["CAp"] =  np.sum(Fp[:,0])
        C["CYp"] =  np.sum(Fp[:,1])
        C["CNp"] =  np.sum(Fp[:,2])
        C["CLLp"] = np.sum(Mp[:,0])
        C["CLMp"] = np.sum(Mp[:,1])
        C["CLNp"] = np.sum(Mp[:,2])
        # Vacuum forces
        C["CAvac"] = np.sum(Fvac[:,0])
        C["CYvac"] = np.sum(Fvac[:,1])
        C["CNvac"] = np.sum(Fvac[:,2])
        C["CLLvac"] = np.sum(Mvac[:,0])
        C["CLMvac"] = np.sum(Mvac[:,1])
        C["CLNvac"] = np.sum(Mvac[:,2])
        # Flow-through contributions
        C["CAm"] =  np.sum(Fm[:,0])
        C["CYm"] =  np.sum(Fm[:,1])
        C["CNm"] =  np.sum(Fm[:,2])
        C["CLLm"] = np.sum(Mm[:,0])
        C["CLMm"] = np.sum(Mm[:,1])
        C["CLNm"] = np.sum(Mm[:,2])
        # Viscous contributions
        C["CAv"] =  np.sum(Fv[:,0])
        C["CYv"] =  np.sum(Fv[:,1])
        C["CNv"] =  np.sum(Fv[:,2])
        C["CLLv"] = np.sum(Mv[:,0])
        C["CLMv"] = np.sum(Mv[:,1])
        C["CLNv"] = np.sum(Mv[:,2])
        # Output
        return C


  # >



# Function to read .tri files
def ReadTri(fname):
    r"""Read a basic triangulation file

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
        * 2014-05-27 ``@ddalle``: Version 1.0
    """
    # Create the tri object and return it.
    return Tri(fname)


# Global function to write a triangulation (just calls tri method)
def WriteTri(fname, tri):
    r"""Write a triangulation instance to file

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
        * 2014-05-23 ``ddalle``: Version 1.0
    """
    # Call the triangulation's write method.
    tri.Write(fname)
    return None

