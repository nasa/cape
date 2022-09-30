#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.plt`: Python interface to Tecplot PLT files
========================================================

This module provides the class :class:`cape.plt.Plt`, which intends to read and
write Tecplot binary or ASCII PLT files for surface grid solutions.  It does
not use the TecIO library to avoid causing unnecessary dependencies for the
software.

This class cannot read any generic ``.plt`` file; it focuses on surface grids
with a mix of triangles and quads.  In particular it is closely paired with
the :mod:`trifile` triangulation module.  The initial driving cause for
creating this module was to read FUN3D boundary solution files and convert them
to annotated Cart3D ``triq`` format for input to ``triload`` and other
post-processing based on the :mod:`trifile` module.

See also:

    * :mod:`trifile`
    * :mod:`cape.pyfun.plt`
    * :mod:`pc_Tri2Plt`
    * :mod:`pc_Plt2Tri`

"""

# Standard library
import re

# Third-party
import numpy as np

# Local imports
from . import io as capeio
from . import tri as trifile
from . import util as capeutil


# Text patterns
REGEX_VARS = re.compile("variables", re.IGNORECASE)
REGEX_ZONE = re.compile("zone", re.IGNORECASE)

# Zone types
ORDERED = 0
FELINESEG = 1
FETRIANGLE = 2
FEQUADRILATERAL = 3
FETETRAHEDRON = 4
FEBRICK = 5
FEPOLYGON = 6
FEPOLYHEDRON = 7

# Other options
N_AUX_MAX = 100


# Convert a PLT to TRIQ
def Plt2Triq(fplt, ftriq=None, **kw):
    """Convert a Tecplot PLT file to a Cart3D annotated triangulation (TRIQ)
    
    :Call:
        >>> Plt2Triq(fplt, ftriq=None, **kw)
    :Inputs:
        *fplt*: :class:`str`
            Name of Tecplot PLT file
        *ftriq*: {``None``} | :class:`str`
            Name of output file (default: replace extension with ``.triq``)
        *mach*: {``1.0``} | positive :class:`float`
            Freestream Mach number for skin friction coeff conversion
        *triload*: {``True``} | ``False``
            Whether or not to write a triq tailored for ``triloadCmd``
        *avg*: {``True``} | ``False``
            Use time-averaged states if available
        *rms*: ``True`` | {``False``}
            Use root-mean-square variation instead of nominal value
    :Versions:
        * 2016-12-20 ``@ddalle``: First version
    """
    # Output file name
    if ftriq is None:
        # Default: strip .plt and add .triq
        ftriq = fplt.rstrip('plt').rstrip('dat') + 'triq'
    # TRIQ settings
    ll  = kw.get('triload', True)
    avg = kw.get('avg', True)
    rms = kw.get('rms', False)
    # Read the PLT file
    plt = Plt(fplt)
    # Create the TRIQ interface
    triq = plt.CreateTriq(**kw)
    # Get output file extension
    ext = triq.GetOutputFileType(**kw)
    # Write triangulation
    triq.Write(ftriq, **kw)

# Get an object from a list
def getind(V, k, j=None):
    """Get an index of a variable in a list if possible
    
    :Call:
        >>> i = getind(V, k, j=None)
    :Inputs:
        *V*: :class:`list` (:class:`str`)
            List of headers
        *k*: :class:`str`
            Header name
        *j*: :class:`int` | {``None``}
            Default index if *k* not in *V*
    :Outputs:
        *i*: :class:`int` | ``None``
            Index of *k* in *V* if possible
    :Versions:
        * 2016-12-19 ``@ddalle``: First version
    """
    # Check if *k* is in *V*
    if k in V:
        # Return the index
        return V.index(k)
    else:
        # Return ``None``
        return j

# Tecplot class
class Plt(object):
    """Interface for Tecplot PLT files
    
    :Call:
        >>> plt = cape.plt.Plt(fname=None, dat=None, triq=None, **kw)
    :Inputs:
        *fname*: {``None``} | :class:`str`
            Name of binary PLT file to read
        *dat*: {``None``} | :class:`str`
            Name of ASCII file to read
        *triq*: {``None``} | :class:`trifile.Triq`
            Annotated triangulation interface
    :Outputs:
        *plt*: :class:`cape.plt.Plt`
            Tecplot PLT interface
        *plt.nVar*: :class:`int`
            Number of variables
        *plt.Vars*: :class:`list` (:class:`str`)
            List of of variable names
        *plt.nZone*: :class:`int`
            Number of zones
        *plt.Zone*: :class:`int`
            Name of each zone
        *plt.nPt*: :class:`np.ndarray` (:class:`int`, *nZone*)
            Number of points in each zone
        *plt.nElem*: :class:`np.ndarray` (:class:`int`, *nZone*)
            Number of elements in each zone
        *plt.Tris*: :class:`list` (:class:`np.ndarray` (*N*,4))
            List of triangle node indices for each zone
    :Versions:
        * 2016-11-22 ``@ddalle``: First version
        * 2017-08-24 ``@ddalle``: Added ASCII input capability
    """
    # Initialization method
    def __init__(self, fname=None, dat=None, triq=None, **kw):
        """Initialization method
        
        :Versions:
            * 2016-11-21 ``@ddalle``: Started
            * 2016-11-22 ``@ddalle``: First version
        """
        # Check for an input file
        if fname is not None:
            # Read the file
            self.Read(fname)
        elif dat is not None:
            # Read an ASCII file
            self.ReadDat(dat)
        elif triq is not None:
            # Convert from Triq
            self.ConvertTriq(triq, **kw)
        else:
            # Create empty zones
            self.nZone = 0
            self.q = []
            self.qmin = []
            self.qmax = []
            self.Tris = []
    
    # Tec Boundary reader
    def Read(self, fname):
        """Read a Fun3D boundary Tecplot binary file
        
        :Call:
            >>> plt.Read(fname)
        :Inputs:
            *plt*: :class:`pyFun.plt.Plt`
                Tecplot PLT interface
            *fname*: :class:`str`
                Name of file to read
        :Versions:
            * 2016-11-22 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the opening string
        s = np.fromfile(f, count=1, dtype='|S8')
        # Get string
        if len(s) == 0:
            # No string
            header = None
        else:
            # Get and convert
            header = s[0].decode("ascii")
        # Check it
        if header != '#!TDV112':
            f.close()
            raise ValueError("File '%s' must start with '#!TDV112'" % fname)
        # Throw away the next two integers
        self.line2 = np.fromfile(f, count=2, dtype='i4')
        # Read the title
        self.title = capeio.read_lb4_s(f)
        # Get number of variables (, unpacks the list)
        self.nVar, = np.fromfile(f, count=1, dtype='i4')
        # Loop through variables
        self.Vars = []
        for i in range(self.nVar):
            # Read the name of variable *i*
            self.Vars.append(capeio.read_lb4_s(f))
        # Initialize zones
        self.nZone = 0
        self.Zones = []
        self.ParentZone = []
        self.StrandID = []
        self.QVarLoc = []
        self.VarLocs = []
        self.t = []
        self.ZoneType = []
        self.ZoneAux = []
        self.nPt = []
        self.nElem = []
        # This number should be 299.0
        marker, = np.fromfile(f, dtype='f4', count=1)
        # Read until no more zones
        while True:
            # Test the marker
            if marker == 357.0:
                # End of header
                break
            elif marker == 299.0:
                # Increase zone count
                self.nZone += 1
                # Read zone name
                zone = capeio.read_lb4_s(f).strip('"')
                # Save it
                self.Zones.append(zone)
                # Parent zone
                i, = np.fromfile(f, count=1, dtype='i4')
                self.ParentZone.append(i)
                # Strand ID
                i, = np.fromfile(f, count=1, dtype='i4')
                self.StrandID.append(i)
                # Solution time
                v, = np.fromfile(f, count=1, dtype='f8')
                self.t.append(v)
                # Read a -1 and then the zone type
                i, zt = np.fromfile(f, count=2, dtype='i4')
                self.ZoneType.append(zt)
                # Check zone type
                if zt == ORDERED:
                    raise ValueError("Ordered zone type not implemented")
                # Read option related fo variable location
                # 0: data at notes
                # 1: specify for each var
                vl, = np.fromfile(f, count=1, dtype='i4')
                # Check for var location
                self.QVarLoc.append(vl)
                if vl == 0:
                    # Nothing to specify
                    self.VarLocs.append([])
                else:
                    # Read variable locations... {0: "node", 1: "cell"}
                    self.VarLocs.append(
                        np.fromfile(f, dtype='i4', count=self.nVar))
                # Two options about face neighbors
                neighbor_opt, n_neighbor = np.fromfile(f, dtype='i4', count=2)
                if n_neighbor > 0:
                    raise ValueError("Local face neighbors not implemented")
                # Number of points
                nPt, = np.fromfile(f, count=1, dtype="i4")
                # Check polygon/polyhedron
                if zt in (FEPOLYGON, FEPOLYHEDRON):
                    raise ValueError(
                        "Arbitrary polygon/polyhedron zones not implemented")
                # Number of elements
                nElem, = np.fromfile(f, count=1, dtype="i4")
                # Cell dims
                celldims = np.fromfile(f, count=3, dtype="i4")
                if np.any(celldims):
                    raise ValueError(
                        "In zone %i, expected cell dims to be zero" % self.nZone)
                # Save point and element count
                self.nPt.append(nPt)
                self.nElem.append(nElem)
                # Intitialize aux data
                auxdict = {}
                self.ZoneAux.append(auxdict)
                # Check optio nfor aux name/value paris
                for naux in range(N_AUX_MAX):
                    # Read aux flag
                    aux, = np.fromfile(f, count=1, dtype="i4")
                    # Check flag
                    if aux == 0:
                        break
                    # Read name
                    auxname = capeio.read_lb4_s(f)
                    # Read data type (must be 0)
                    auxtype = np.fromfile(f, count=1, dtype="i4")
                    if auxtype != 0:
                        raise ValueError(
                            "Aux data type %i in zone %i not supported"
                            % (auxtype, self.nZone))
                    # Read string property
                    auxval = capeio.read_lb4_s(f)
                    # Save it
                    auxdict[auxname] = auxval
                # Read some zeros at the end.
            elif marker == 799.0:
                # Auxiliary data
                name = capeio.read_lb4_s(f).strip('"')
                # Read format
                fmt, = np.fromfile(f, count=1, dtype='i4')
                # Check value of *fmt*
                if fmt != 0:
                    raise ValueError(
                        ("Dataset Auxiliary data value format is %i; " % fmt) +
                        ("expected 0"))
                # Read value
                val = capeio.read_lb4_s(f).strip('"')
            else:
                # Unknown marker
                raise ValueError(
                    "Expecting end-of-header marker 357.0\n" +
                    ("  Found: %s" % marker))
            # This number should be 299.0
            marker, = np.fromfile(f, dtype='f4', count=1)
        # Convert arrays
        self.nPt = np.array(self.nPt)
        self.nElem = np.array(self.nElem)
        # This number should be 299.0
        marker, = np.fromfile(f, dtype='f4', count=1)
        # Initialize format list
        self.fmt = np.zeros((self.nZone, self.nVar), dtype='i4')
        # Initialize values and min/max
        self.qmin = np.zeros((self.nZone, self.nVar))
        self.qmax = np.zeros((self.nZone, self.nVar))
        self.q = []
        # Initialize node numbers
        self.Tris = []
        # Read until no more zones
        n = -1
        while marker == 299.0:
            # Next zone
            n += 1
            npt = self.nPt[n]
            nelem = self.nElem[n]
            # Read zone type
            self.fmt[n] = np.fromfile(f, dtype='i4', count=self.nVar)
            # Check for passive variables
            ipass, = np.fromfile(f, dtype='i4', count=1)
            if ipass != 0:
                np.fromfile(f, dtype='i4', count=self.nVar)
            # Check for variable sharing
            ishare, = np.fromfile(f, dtype='i4', count=1)
            if ishare != 0:
                np.fromfile(f, dtype='i4', count=self.nVar)
            # Zone number to share with
            zshare, = np.fromfile(f, dtype='i4', count=1)
            # Read the min and max variables
            qi = np.fromfile(f, dtype='f8', count=(self.nVar*2))
            self.qmin[n] = qi[0::2]
            self.qmax[n] = qi[1::2]
            # Read the actual data
            if self.fmt[n][0] == 2:
                # Read doubles
                qi = np.fromfile(f, dtype="f8", count=self.nVar*npt)
            else:
                # Read floats
                qi = np.fromfile(f, dtype='f4', count=self.nVar*npt)
            # Reshape
            qi = np.transpose(np.reshape(qi, (self.nVar, npt)))
            self.q.append(qi)
            # Get zone type
            zt = self.ZoneType[n]
            # Number of nodes per face
            if zt == FETRIANGLE:
                # Triangles
                melem = 3
            elif zt == FEQUADRILATERAL:
                # Quads (often used for tris, too, w/ repeated node)
                melem = 4
            elif zt == FETETRAHEDRON:
                # Tetrahedra, 4 nodes
                melem = 4
            elif zt == FEBRICK:
                # Hex; also used for pyramids and prisms
                melem = 8
            else:
                raise ValueError(
                    "Zone type %i (zone %i) is unsupported" % (zt, n + 1))
            # Read the tris
            ii = np.fromfile(f, dtype='i4', count=(melem*nelem))
            # Reshape and save
            self.Tris.append(np.reshape(ii, (nelem, melem)))
            # Read the next marker
            i = np.fromfile(f, dtype='f4', count=1)
            # Check length
            if len(i) == 1:
                marker = i[0]
            else:
                break
        # Close the file
        f.close()
    
    # Write Tec Boundary
    def Write(self, fname, Vars=None, **kw):
        """Write a Fun3D boundary Tecplot binary file
        
        :Call:
            >>> plt.Write(fname, Vars=None, **kw)
        :Inputs:
            *plt*: :class:`pyFun.plt.Plt`
                Tecplot PLT interface
            *fname*: :class:`str`
                Name of file to read
            *Vars*: {``None``} | :class:`list` (:class:`str`)
                List of variables (by default, use all variables)
            *CompID*: {``range(len(plt.nZone))``} | :class:`list`
                Optional list of zone numbers to use
        :Versions:
            * 2017-03-29 ``@ddalle``: First version
            * 2017-05-16 ``@ddalle``: Added variable list
            * 2017-12-18 ``@ddalle``: Added *CompID* input
        """
        # Default variable list
        if Vars is None: Vars = self.Vars
        # Number of variables
        nVar = len(Vars)
        # Check for CompID list
        IZone = kw.get("CompID", range(self.nZone))
        # Number of output zones
        nZone = len(IZone)
        # Indices of variabels
        IVar = np.array([self.Vars.index(v) for v in Vars])
        # Open the file
        f = open(fname, 'wb')
        # Write the opening string
        s = np.array('#!TDV112', dtype='|S8')
        # Write it
        s.tofile(f)
        # Write specifier
        capeio.tofile_ne4_i(f, [1, 0])
        # Write title
        capeio.tofile_ne4_s(f, self.title)
        # Write number of variables
        capeio.tofile_ne4_i(f, nVar)
        # Loop through variable names
        for var in Vars:
            capeio.tofile_ne4_s(f, var)
        # Write zones
        for i in IZone:
            # Write goofy zone marker
            capeio.tofile_ne4_f(f, 299.0)
            # Write zone name
            capeio.tofile_ne4_s(f, '"%s"' % self.Zones[i])
            # Write parent zone (usually -1)
            try:
                capeio.tofile_ne4_i(f, self.ParentZone[i])
            except Exception:
                capeio.tofile_ne4_i(f, -1)
            # Write the StrandID
            try:
                capeio.tofile_ne4_i(f, self.StrandID[i])
            except Exception:
                capeio.tofile_ne4_i(f, 1000+i)
            # Write the time
            try:
                capeio.tofile_ne8_f(f, self.t[i])
            except Exception:
                capeio.tofile_ne8_f(f, 0.0)
            # Write -1
            capeio.tofile_ne4_i(f, -1)
            # Write the zone type (3 for triangles)
            try:
                capeio.tofile_ne4_i(f, self.ZoneType[i])
            except Exception:
                capeio.tofile_ne4_i(f, 3)
            # Write a bunch of zeros
            try:
                # Check for variable locations (node- or cell-centered)
                if self.QVarLoc[i]:
                    # Variable locations are marked
                    capeio.tofile_ne4_i(1)
                    # Try to write the markers
                    try:
                        capeio.tofile_ne4_i(self.VarLocs[i][IVar])
                    except Exception:
                        capeio.tofile_ne4_i(np.zeros(nVar))
                else:
                    # Write a zero and move on
                    capeio.tofile_ne4_i(f, 0)
            except Exception:
                # Default to marking all variables node-centered
                capeio.tofile_ne4_i(f, 1)
                capeio.tofile_ne4_i(f, np.zeros(nVar))
            # Two unused or weird variables
            capeio.tofile_ne4_i(f, np.zeros(2))
            # Write number of pts, elements
            capeio.tofile_ne4_i(f, [self.nPt[i], self.nElem[i]])
            # Write some more zeros
            capeio.tofile_ne4_i(f, np.zeros(4))
        # Write end-of-header marker
        capeio.tofile_ne4_f(f, 357.0)
        # Loop through the zones again
        for n in IZone:
            # Write marker
            capeio.tofile_ne4_f(f, 299.0)
            # Extract sizes
            npt = self.nPt[n]
            nelem = self.nElem[n]
            # Write variable types (usually 1 for float type, I think)
            try:
                capeio.tofile_ne4_i(f, self.fmt[n][IVar])
            except Exception:
                capeio.tofile_ne4_i(f, np.ones(nVar))
            # Just set things as passive variables like FUN3D
            capeio.tofile_ne4_i(f, 1)
            capeio.tofile_ne4_i(f, np.zeros(nVar))
            # Just set things to share with -1, because that makes sense
            # somehow.  I guess it is a commercial format, so go figure.
            capeio.tofile_ne4_i(f, 1)
            capeio.tofile_ne4_i(f, -1*np.ones(nVar))
            # Save the *zshare* value
            capeio.tofile_ne4_i(f, -1)
            # Form matrix of qmin[0], qmax[0], qmin[1], ...
            qex = np.vstack((self.qmin[n][IVar],
                self.qmax[n][IVar])).transpose()
            # Save *qmin* and *qmax*
            capeio.tofile_ne8_f(f, qex)
            # Save the actual data
            capeio.tofile_ne4_f(f, np.transpose(self.q[n][:,IVar]))
            # Write the tris (this may need to be generalized!)
            capeio.tofile_ne4_i(f, self.Tris[n])
        # Close the file
        f.close()
    
    # Tec Boundary reader
    def ReadDat(self, fname):
        """Read an ASCII Tecplot data file
        
        :Call:
            >>> plt.ReadData(fname)
        :Inputs:
            *plt*: :class:`pyFun.plt.Plt`
                Tecplot PLT interface
            *fname*: :class:`str`
                Name of file to read
        :Versions:
            * 2017-08-24 ``@ddalle``: Version 1.0
            * 2022-08-30 ``@ddalle``: Version 1.1; default title
        """
        # Open the file
        f = open(fname, 'r')
        # Throw away the next two integers
        self.line2 = np.zeros(2, dtype="i4")
        # Read the title line
        line = f.readline().strip()
        # Check for a title
        if REGEX_VARS.match(line):
            # No title
            self.title = "Untitled"
        else:
            # Save the title
            self.title = line.split("=")[1].strip('"')
            # Read the variables line
            line = f.readline().strip()
        # Read variable names
        varpart = line.split("=")[1]
        varlist = re.split(r"[\s,]+", varpart.strip())
        # Save the variable list
        self.Vars = [varname.strip('"') for varname in varlist]
        # Get number of variables (, unpacks the list)
        nVar = len(self.Vars)
        self.nVar = nVar
        # Initialize zones
        self.nZone = 0
        self.Zones = []
        self.ParentZone = []
        self.StrandID = []
        self.QVarLoc = []
        self.VarLocs = []
        self.t = []
        self.ZoneType = []
        self.nPt = []
        self.nElem = []
        # Initialize data variables
        self.q = []
        self.qmin = np.zeros((0,self.nVar))
        self.qmax = np.zeros((0,self.nVar))
        self.fmt = np.zeros((0,self.nVar))
        # Initialize node numbers
        self.Tris = []
        # Read the title line
        line = f.readline().strip()
        # Read until no more zones
        while REGEX_ZONE.match(line):
            # Increase zone count
            self.nZone += 1
            # Split line by commas
            L = line[5:].split(",")
            L = [s.strip() for s in L]
            # Convert to dictionary
            D = {}
            # Loop through values
            for s in L:
                # Get key and value
                k, v = s.split("=")
                # Save key
                D[k.lower()] = v
            # Save the title
            v = D.get("t", "zone %i" % self.nZone)
            self.Zones.append(v)
            # Parent zone
            v = int(D.get("parent", -1))
            self.ParentZone.append(v)
            # Strand ID
            v = int(D.get("strandid", 1000))
            self.StrandID.append(v)
            # Solution time
            v = float(D.get("solutiontime", 0))
            self.t.append(v)
            # Get zone type
            zt = D.get("f", "feblock")
            # Save zone type
            if zt.lower() == "feblock":
                self.ZoneType.append(3)
            else:
                # Some other zone type?
                self.ZoneType.append(0)
            # Element type
            et = D.get("et", "").lower()
            # Check some other aspect about the zone
            vl = int(D.get("varloc", 1))
            # Check for var location
            self.QVarLoc.append(vl)
            if vl == 0:
                # Nothing to specify
                self.VarLocs.append([])
            else:
                # Read variable locations... {0: "node", 1: "cell"}
                self.VarLocs.append(np.zeros(self.nVar, dtype="int"))
            # Number of points, elements
            nPt   = int(D.get("i", D.get("n", 0)))
            nElem = int(D.get("j", D.get("e", 0)))
            self.nPt.append(nPt)
            self.nElem.append(nElem)
            # Read the actual data
            qi = np.fromfile(f, count=(nVar*nPt), sep=" ")
            # Reshape
            qi = np.reshape(qi, (nPt, nVar))
            self.q.append(qi)
            # Save mins and maxes
            qmini = np.min(qi, axis=0)
            qmaxi = np.max(qi, axis=0)
            # Append min and max values
            self.qmin = np.vstack((self.qmin, [qmini]))
            self.qmax = np.vstack((self.qmax, [qmaxi]))
            # Check shape
            if et == "triangle":
                nPtElem = 3
            else:
                nPtElem = 4
            # Read the tris
            ii = np.fromfile(f, count=(nPtElem*nElem), sep=" ", dtype="int")
            # Reshape and save
            self.Tris.append(np.reshape(ii-1, (nElem, nPtElem)))
            # Read next line (empty or title of next zone)
            line = f.readline().strip()
            # Check for variable list
            if REGEX_VARS.match(line):
                # Read another line
                line = f.readline().strip()
        # Convert arrays
        self.nPt = np.array(self.nPt)
        self.nElem = np.array(self.nElem)
        # Transpose qmin, qmax
        #self.qmin = self.qmin.transpose()
        #self.qmax = self.qmax.transpose()
        # Set format list
        self.fmt = np.ones((self.nZone, self.nVar), dtype='i4')
        # Close the file
        f.close()
        
    # Write ASCII file 
    def WriteDat(self, fname, Vars=None, **kw):
        """Write Tecplot PLT file to ASCII format (``.dat``)
        
        :Call:
            >>> plt.WriteDat(fname, Vars=None, **kw)
        :Inputs:
            *plt*: :class:`cape.plt.Plt`
                Tecplot PLT interface
            *fname*: :class:`str`
                Name of DAT file to write
            *Vars*: {``None``} | :class:`list` (:class:`str`)
                List of variables (by default, use all variables)
            *CompID*: {``range(len(plt.nZone))``} | :class:`list`
                Optional list of zone numbers to use
        :Versions:
            * 2017-03-30 ``@ddalle``: First version
            * 2017-05-16 ``@ddalle``: Added variable list
            * 2017-12-18 ``@ddalle``: Added *CompID* input
        """
        # Default variable list
        if Vars is None: Vars = self.Vars
        # Number of variables
        nVar = len(Vars)
        # Indices of variabels
        IVar = np.array([self.Vars.index(v) for v in Vars])
        # Check for CompID list
        IZone = kw.get("CompID", range(self.nZone))
        # Number of output zones
        nZone = len(IZone)
        # Create the file
        f = open(fname, 'w')
        # Write the title
        f.write('title="%s"\n' % self.title)
        # Write the variable names header
        f.write('variables = %s\n' % " ".join(Vars))
        # Loop through zones
        for n in IZone:
            # Write the zone name
            f.write('zone t="%s"' % self.Zones[n].strip('"').strip("'"))
            # Write the time
            f.write(', solutiontime=%14.7E' % self.t[n])
            # Write the strandid
            f.write(', strandid=%s' % self.StrandID[n])
            # Write the number of nodes and elements
            f.write(', i=%s, j=%s' % (self.nPt[n], self.nElem[n]))
            # Write some header that appears fixed.
            f.write(", f=feblock\n")
            # Extract the state
            q = self.q[n][:,IVar]
            # Number of rows of 7
            nrow = int(np.ceil(q.shape[0]/7.0))
            # Loop through the variables
            for j in range(nVar):
                # Writing each row
                for i in range(nrow):
                    # Extract data in septuplets and write to file
                    q[7*i:7*(i+1),j].tofile(f, sep=" ", format="%14.7E")
                    # Add a newline character
                    f.write("\n")
            # Write the TRI nodes
            np.savetxt(f, self.Tris[n]+1, fmt="%10i")
        # Close the file
        f.close()
        
    # Create from a TRIQ
    def ConvertTriq(self, triq, **kw):
        """Create a PLT object by reading data from a Tri/Triq object
        
        :Call:
            >>> plt.ConvertTriq(triq)
        :Inputs:
            *plt*: :class:`cape.plt.Plt`
                Tecplot PLT interface
            *triq*: :class:`trifile.Triq`
                Surface triangulation with or without solution (*triq.q*)
            *CompIDs*: :class:`list` (:class:`int`)
                List of CompIDs to consider
        :Versions:
            * 2017-03-30 ``@ddalle``: First version
        """
        # Get CompIDs option
        CompIDs = kw.get("CompIDs")
        # Default: all components
        if CompIDs is None:
            # Get unique compIDS
            CompIDs = np.unique(triq.CompID)
        # Get number of zones
        self.nZone = len(CompIDs)
        # Try to get number of states
        try:
            # This should be an attribute
            nq = triq.nq
        except Exception:
            # No extra states
            nq = 0
        # Process variables
        qvars = kw.get("vars", kw.get("Vars"))
        # Process default
        if qvars is None:
            # Process default list based on number of states
            if nq == 1:
                # Pressure coefficient
                qvars = ["cp"]
            elif nq == 6:
                # Cart3D states
                qvars = ["cp", "rho", "u", "v", "w", "p"]
            elif nq == 9:
                # FUN3D states (common for unstructured)
                qvars = [
                    "cp",
                    "rho", "u", "v", "w", "p",
                    "cf_x", "cf_y", "cf_z"
                ]
            elif nq == 13:
                # OVERFLOW states from OVERINT
                # Check if we can get skin friction
                try:
                    # Try to calculate skin friction
                    cf_x, cf_y, cf_z = triq.GetSkinFriction(comp=CompIDs, **kw)
                    # Get nodes affected
                    I = triq.GetNodesFromCompID(CompIDs)
                    # Get the key states
                    qvars = [
                        "cp",
                        "rho", "rhou", "rhov", "rhow", "e",
                        "cf_x", "cf_y", "cf_z"
                    ]
                    # Downsize state variables
                    triq.q = triq.q[:,:9]
                    # Update number of states
                    triq.nq = 9
                    nq = triq.nq
                    # Save the skin friction.
                    triq.q[I,6] = cf_x
                    triq.q[I,7] = cf_y
                    triq.q[I,8] = cf_z
                except Exception as e:
                    # Print a warning
                    print("    WARNING: failed to calculate skin friction")
                    # Fall back to the native states
                    qvars = [
                        "cp",
                        "rho", "rhou", "rhov", "rhow", "e",
                        "mu", "u_eta", "v_eta", "w_eta",
                        "dx", "dy", "dz"
                    ]
            else:
                # Unknown
                qvars = ["var%s" % i for i in range(nq)]
        # Set number of variables
        self.nVar = 3 + len(qvars)
        # Check number of variables
        if len(qvars) != nq:
            raise ValueError(
                ("Found %s variables in TRIQ input but " % nq) +
                ("input list\n  %s\nhas %s variables" % (qvars, len(qvars))))
        # Full list of variables
        self.Vars = ["x", "y", "z"] + qvars
        # Initialize zone names
        self.Zones = []
        # Set the title
        self.title = "tecplot geometry and solution file"
        # Save some nonsense
        self.line2 = np.array([1, 0], dtype='i4')
        # Initialize a bunch of other properties
        self.QVarLoc = []
        self.VarLocs = []
        self.StrandID = []
        self.ParentZone = []
        # Get the boundary names and other header info
        for n in range(self.nZone):
            # Get the component name and number
            compID = CompIDs[n]
            name = triq.GetCompName(compID)
            # Status update
            if kw.get('v', False):
                print("Initializing zone '%s' (%s/%s)" % (name, n+1, self.nZone))
            # Append the title
            if name:
                # Include the boundary name
                self.Zones.append("boundary %s %s" % (compID, name))
            else:
                # Just use the compID in the title
                self.Zones.append("boundary %s" % compID)
            # Append some parameters
            self.QVarLoc.append(0)
            self.VarLocs.append([])
            self.StrandID.append(1000 + n)
            self.ParentZone.append(-1)
        # Initialize zone sizes
        self.nPt = []
        self.nElem = []
        # Initialize the geometry
        self.Tris = []
        # Initialize the states
        self.q = []
        # Initialize of each min/max for each var/zone combo
        self.qmin = np.zeros((self.nZone, self.nVar))
        self.qmax = np.zeros((self.nZone, self.nVar))
        # Format nonsense
        self.fmt = np.ones((self.nZone, self.nVar))
        # Loop through the zones to process the data
        for n in range(self.nZone):
            # Status update
            if kw.get('v', False):
                print("Creating zone '%s' (%s/%s)" % 
                    (self.Zones[n], n+1, self.nZone))
            # Get the CompID in question
            comp = CompIDs[n]
            # Get the nodes and tris in that comp
            I = triq.GetNodesFromCompID(comp)
            K = triq.GetTrisFromCompID(comp)
            # Get the nodes and overall-index tris
            Nodes = triq.Nodes[I,:]
            Tris = triq.Tris[K,:]
            # Get the counts
            self.nPt.append(I.size)
            self.nElem.append(K.size)
            # Downselect the node indices so they are numbered 1 to *n*
            # (Also shift to zero-based)
            T = capeutil.TrimUnused(Tris) - 1
            # Form the state matrix for this zone
            if nq > 0:
                # Include *q* variables
                q = np.hstack((Nodes, triq.q[I,:self.nVar]))
            else:
                # Only use nodes as nodal variables
                q = np.array(Nodes)
            # Save the min/max
            self.qmin[n,:] = np.min(q, axis=0)
            self.qmax[n,:] = np.max(q, axis=0)
            # Save the states as single-precision
            self.q.append(np.array(q, dtype="f4"))
            # Save the triangles (nodal indices)
            # We have to append the third node to the end
            # Apparently Tecplot might think of tris as degenerate quads, which
            # could be convenient at a later time.
            self.Tris.append(np.hstack((T, T[:,[2]])))
        # Convert to array
        self.nPt = np.array(self.nPt)
        self.nElem = np.array(self.nElem)
        # Process time step
        t = float(kw.get("t", 1.0))
        self.t = list(t*np.ones(self.nZone))

    # Create a triq file
    def CreateTriq(self, **kw):
        """Create a Cart3D annotated triangulation (``triq``) interface
        
        The primary purpose is creating a properly-formatted triangulation for
        calculating line loads with the Chimera Grid Tools function
        ``triloadCmd``, which requires the five fundamental states plus the
        pressure coefficient for inviscid sectional loads.  For complete
        sectional loads including viscous contributions, the Tecplot interface
        must also have the skin friction coefficients.
        
        The *triq* interface will have either 6 or 9 states, depending on
        whether or not the viscous coefficients are present.
        
        Alternatively, if the optional input *triload* is ``False``, the *triq*
        output will have whatever states are present in *plt*.
        
        :Call:
            >>> triq = plt.CreateTriq(mach=1.0, triload=True, **kw)
        :Inputs:
            *plt*: :class:`pyFun.plt.Plt`
                Tecplot PLT interface
            *mach*: {``1.0``} | positive :class:`float`
                Freestream Mach number for skin friction coeff conversion
            *CompID*: {``range(len(plt.nZone))``} | :class:`list`
                Optional list of zone numbers to use
            *triload*: {``True``} | ``False``
                Whether or not to write a triq tailored for ``triloadCmd``
            *avg*: {``True``} | ``False``
                Use time-averaged states if available
            *rms*: ``True`` | {``False``}
                Use root-mean-square variation instead of nominal value
        :Outputs:
            *triq*: :class:`trifile.Triq`
                Annotated Cart3D triangulation interface
        :Versions:
            * 2016-12-19 ``@ddalle``: First version
        """
        # Inputs
        triload = kw.get('triload', True)
        # Boundary number map?
        mapbc = kw.get('mapbc', True)
        # Averaging?
        avg = kw.get('avg', True)
        # Write RMS values?
        rms = kw.get('rms', False)
        # Freestream Mach number; FUN3D writes cf/1.4*pinf instead of cf/qinf
        mach = float(kw.get('mach', kw.get('m', kw.get('minf', 1.0))))
        # Total number of points (if no emissions)
        nNode = np.sum(self.nPt)
        # Rough number of tris
        nElem = np.sum(self.nElem)
        # Initialize
        Nodes = np.zeros((nNode, 3))
        Tris  = np.zeros((2*nElem, 3), dtype=int)
        # Initialize component IDs
        CompID = np.zeros(2*nElem, dtype=int)
        # Counters
        iNode = 0
        iTri  = 0
        # Error message for coordinates
        msgx = ("  Warning: triq file conversion requires '%s'; " +
            "not found in this PLT file")
        # Check required states
        for v in ['x', 'y', 'z']:
            # Check for the state
            if v not in self.Vars:
                raise ValueError(msgx % v)
        # Process the states
        if triload:
            # Select states appropriate for ``triload``
            qtype = 2
            # Error message for required states
            msgr = ("  Warning: triq file for line loads requires '%s'; " +
                "not found in this PLT file")
            msgv = ("  Warning: Viscous line loads require '%s'; " +
                "not found in this PLT file")
            # Check required states
            for v in ['rho', 'u', 'v', 'w', 'p', 'cp']:
                # Check for the state
                if v not in self.Vars:
                    # Print warning
                    print(msgr % v)
                    # Fall back to type 0
                    qtype = 0
            # Check viscous states
            for v in ['cf_x', 'cf_y', 'cf_z']:
                # Check for the state
                if v not in self.Vars:
                    # Print warning
                    print(msgv % v)
                    # Fall back to type 0 or 1
                    qtype = min(qtype, 1)
        else:
            # Use the states that are present
            qtype = 0
        # Find the states in the variable list
        jx = self.Vars.index('x')
        jy = self.Vars.index('y')
        jz = self.Vars.index('z')
        # Check for nominal states
        jcp  = getind(self.Vars, 'cp')
        jrho = getind(self.Vars, 'rho')
        ju   = getind(self.Vars, 'u')
        jv   = getind(self.Vars, 'v')
        jw   = getind(self.Vars, 'w')
        jp   = getind(self.Vars, 'p')
        jcfx = getind(self.Vars, 'cf_x')
        jcfy = getind(self.Vars, 'cf_y')
        jcfz = getind(self.Vars, 'cf_z')
        # Check for time average
        if avg:
            jcp  = getind(self.Vars, 'cp_tavg',   jcp)
            jrho = getind(self.Vars, 'rho_tavg',  jrho)
            ju   = getind(self.Vars, 'u_tavg',    ju)
            jv   = getind(self.Vars, 'v_tavg',    jv)
            jw   = getind(self.Vars, 'w_tavg',    jw)
            jp   = getind(self.Vars, 'p_tavg',    jp)
            jcfx = getind(self.Vars, 'cf_x_tavg', jcfx)
            jcfy = getind(self.Vars, 'cf_y_tavg', jcfy)
            jcfz = getind(self.Vars, 'cf_z_tavg', jcfz)
        # Check for RMS variation
        if rms:
            jcpr  = getind(self.Vars, 'cp_trms',   jcp)
            jrhor = getind(self.Vars, 'rho_trms',  jrho)
            jur   = getind(self.Vars, 'u_trms',    ju)
            jvr   = getind(self.Vars, 'v_trms',    jv)
            jwr   = getind(self.Vars, 'w_trms',    jw)
            jpr   = getind(self.Vars, 'p_trms',    jp)
            jcfxr = getind(self.Vars, 'cf_x_trms', jcfx)
            jcfyr = getind(self.Vars, 'cf_y_trms', jcfy)
            jcfzr = getind(self.Vars, 'cf_z_trms', jcfz)
        # Initialize the states
        if qtype == 0:
            # Use all the states from the PLT file
            nq = self.nVar - 3
            # List of states to save (exclude 'x', 'y', 'z')
            J = [i for i in range(self.nVar)
                if self.Vars[i] not in ['x','y','z']
            ]
        elif qtype == 1:
            # States adequate for pressure and momentum
            nq = 6
        elif qtype == 2:
            # Full set of states including viscous
            nq = 9
        # Initialize state
        q = np.zeros((nNode, nq))
        # Reset node count
        npt = 0
        # Check for CompID list
        IZone = kw.get("CompID", range(self.nZone))
        # Loop through the components
        for k in IZone:
            # Extract tris
            T = self.Tris[k]
            # Number of points and elements
            kNode = self.nPt[k]
            kTri  = self.nElem[k]
            # Increment node count
            npt += kNode
            # Check for quads
            iQuad = np.where(T[:,-1] != T[:,-2])[0]
            kQuad = len(iQuad)
            # if np.any(self.Tris[k][:,-1] != self.Tris[k][:,-2]):
            #     raise ValueError(
            #         ("Detected a quad face in zone %s " % k) +
            #         ("(%s); not yet supported " % self.Zones[k]) +
            #         "for converting PLT files for line loads")
            # Save the nodes
            Nodes[iNode:iNode+kNode,0] = self.q[k][:,jx]
            Nodes[iNode:iNode+kNode,1] = self.q[k][:,jy]
            Nodes[iNode:iNode+kNode,2] = self.q[k][:,jz]
            # Save the states
            if qtype == 0:
                # Save all states appropriately
                for j in range(len(J)):
                    q[iNode:iNode+kNode,j] = self.q[k][:,J[j]]
            elif qtype == 1:
                # Get the appropriate states, primary only
                if rms:
                    # Variation
                    cp   = self.q[k][:,jcpr]
                    rhoa = self.q[k][:,jrho]
                    rho  = self.q[k][:,jrhor]
                    u    = self.q[k][:,jur] * rhoa
                    v    = self.q[k][:,jvr] * rhoa
                    w    = self.q[k][:,jwr] * rhoa
                    p    = self.q[k][:,jpr]
                else:
                    # Nominal states
                    cp  = self.q[k][:,jcp]
                    rho = self.q[k][:,jrho]
                    u   = self.q[k][:,ju] * rho
                    v   = self.q[k][:,jv] * rho
                    w   = self.q[k][:,jw] * rho
                    p   = self.q[k][:,jp]
                # Save the states
                q[iNode:iNode+kNode,0] = cp
                q[iNode:iNode+kNode,1] = rho
                q[iNode:iNode+kNode,2] = u
                q[iNode:iNode+kNode,3] = v
                q[iNode:iNode+kNode,4] = w
                q[iNode:iNode+kNode,5] = p
            elif qtype == 2:
                # Scaling 
                cf = mach*mach/2
                # Get the appropriate states, including viscous
                if rms:
                    # Variation
                    cp   = self.q[k][:,jcpr]
                    rhoa = self.q[k][:,jrho]
                    rho  = self.q[k][:,jrhor]
                    u    = self.q[k][:,jur] * rhoa
                    v    = self.q[k][:,jvr] * rhoa
                    w    = self.q[k][:,jwr] * rhoa
                    p    = self.q[k][:,jpr]
                    cfx  = self.q[k][:,jcfxr] * cf
                    cfy  = self.q[k][:,jcfyr] * cf
                    cfz  = self.q[k][:,jcfzr] * cf
                else:
                    # Nominal states
                    cp  = self.q[k][:,jcp]
                    rho = self.q[k][:,jrho]
                    u   = self.q[k][:,ju] * rho
                    v   = self.q[k][:,jv] * rho
                    w   = self.q[k][:,jw] * rho
                    p   = self.q[k][:,jp]
                    cfx = self.q[k][:,jcfx] * cf
                    cfy = self.q[k][:,jcfy] * cf
                    cfz = self.q[k][:,jcfz] * cf
                # Save the states
                q[iNode:iNode+kNode,0] = cp
                q[iNode:iNode+kNode,1] = rho
                q[iNode:iNode+kNode,2] = u
                q[iNode:iNode+kNode,3] = v
                q[iNode:iNode+kNode,4] = w
                q[iNode:iNode+kNode,5] = p
                q[iNode:iNode+kNode,6] = cfx
                q[iNode:iNode+kNode,7] = cfy
                q[iNode:iNode+kNode,8] = cfz
            # Save the node numbers
            Tris[iTri:iTri+kTri,:] = (T[:,:3] + iNode + 1)
            # Save the quads
            if kQuad > 0:
                # Select the elements first; cannot combine operations
                TQ = T[iQuad,:]
                # Select nodes 1,3,4 to get second triangle
                Tris[iTri+kTri:iTri+kTri+kQuad,:] = TQ[:,[0,2,3]]+iNode+1
            # Increase the running node count
            iNode += kNode
            # Try to read the component ID
            try:
                # Name of the zone should be 'boundary 9 CORE_Body' or similar
                comp = int(self.Zones[k].split()[1])
            except Exception:
                # Otherwise just number 1 to *n*
                comp = np.max(CompID) + 1
            # Check for converting the compID (e.g. FUN3D 'mapbc' file)
            if mapbc is not None:
                try:
                    comp = mapbc.CompID[comp-1]
                except Exception:
                    pass
            # Number of elements
            kElem = kTri + kQuad
            # Save the component IDs
            CompID[iTri:iTri+kElem] = comp
            # Increase the running tri count
            iTri += kElem
        # Downselect Tris and CompID
        Tris = Tris[:iTri,:]
        CompID = CompID[:iTri]
        # Downselect nodes
        Nodes = Nodes[:npt,:]
        q = q[:npt,:]
        # Create the triangulation
        triq = trifile.Triq(Nodes=Nodes, Tris=Tris, q=q, CompID=CompID)
        # Output
        return triq

    # Create a triq file
    def CreateTri(self, **kw):
        r"""Create a Cart3D triangulation (``.tri``) file
        
        :Call:
            >>> tri = plt.CreateTri(**kw)
        :Inputs:
            *plt*: :class:`pyFun.plt.Plt`
                Tecplot PLT interface
            *CompID*: {``range(len(plt.nZone))``} | :class:`list`
                Optional list of zone numbers to use
        :Outputs:
            *tri*: :class:`trifile.Tri`
                Cart3D triangulation interface
        :Versions:
            * 2016-12-19 ``@ddalle``: Version 1.0
            * 2021-01-06 ``@ddalle``: Version 1.1; fork CreateTriq()
        """
        # Boundary number map?
        mapbc = kw.get('mapbc', True)
        # Total number of points (if no emissions)
        nNode = np.sum(self.nPt)
        # Rough number of tris
        nElem = np.sum(self.nElem)
        # Initialize
        Nodes = np.zeros((nNode, 3))
        Tris  = np.zeros((2*nElem, 3), dtype=int)
        # Initialize component IDs
        CompID = np.zeros(2*nElem, dtype=int)
        # Counters
        iNode = 0
        iTri  = 0
        # Error message for coordinates
        msgx = ("  Warning: tri file conversion requires '%s'; " +
            "not found in this PLT file")
        # Check required states
        for v in ['x', 'y', 'z']:
            # Check for the state
            if v not in self.Vars:
                raise ValueError(msgx % v)
        # Find the states in the variable list
        jx = self.Vars.index('x')
        jy = self.Vars.index('y')
        jz = self.Vars.index('z')
        # Reset node count
        npt = 0
        # Check for CompID list
        IZone = kw.get("CompID", range(self.nZone))
        # Loop through the components
        for k in IZone:
            # Extract tris
            T = self.Tris[k]
            # Number of points and elements
            kNode = self.nPt[k]
            kTri  = self.nElem[k]
            # Increment node count
            npt += kNode
            # Check for quads
            if T.shape[1] == 4:
                # Check for duplicated index 
                iQuad = np.where(T[:,-1] != T[:,-2])[0]
                kQuad = len(iQuad)
            else:
                # Pure triangles
                iQuad = np.zeros(0, dtype="int")
                kQuad = 0
            # Save the nodes
            Nodes[iNode:iNode+kNode,0] = self.q[k][:,jx]
            Nodes[iNode:iNode+kNode,1] = self.q[k][:,jy]
            Nodes[iNode:iNode+kNode,2] = self.q[k][:,jz]
            # Save the node numbers
            Tris[iTri:iTri+kTri,:] = (T[:,:3] + iNode + 1)
            # Save the quads
            if kQuad > 0:
                # Select the elements first; cannot combine operations
                TQ = T[iQuad,:]
                # Select nodes 1,3,4 to get second triangle
                Tris[iTri+kTri:iTri+kTri+kQuad,:] = TQ[:,[0,2,3]]+iNode+1
            # Increase the running node count
            iNode += kNode
            # Try to read the component ID
            try:
                # Name of the zone should be 'boundary 9 CORE_Body' or similar
                comp = int(self.Zones[k].split()[1])
            except Exception:
                # Otherwise just number 1 to *n*
                comp = np.max(CompID) + 1
            # Check for converting the compID (e.g. FUN3D 'mapbc' file)
            if mapbc is not None:
                try:
                    comp = mapbc.CompID[comp-1]
                except Exception:
                    pass
            # Number of elements
            kElem = kTri + kQuad
            # Save the component IDs
            CompID[iTri:iTri+kElem] = comp
            # Increase the running tri count
            iTri += kElem
        # Downselect Tris and CompID
        Tris = Tris[:iTri,:]
        CompID = CompID[:iTri]
        # Downselect nodes
        Nodes = Nodes[:npt,:]
        # Create the triangulation
        tri = trifile.Tri(Nodes=Nodes, Tris=Tris, CompID=CompID)
        # Output
        return tri
# class Plt

