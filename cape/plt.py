"""
Tecplot PLT File Interface for Fun3D
====================================


"""

# Basic numerics
import numpy as np
# Useful tool for more complex binary I/O methods
import cape.io
import cape.tri
import cape.util

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
        >>> plt = cape.plt.Plt(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
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
    """
    
    def __init__(self, fname=None, triq=None, **kw):
        """Initialization method
        
        :Versions:
            * 2016-11-21 ``@ddalle``: Started
            * 2016-11-22 ``@ddalle``: First version
        """
        # Check for an input file
        if fname is not None:
            # Read the file
            self.Read(fname)
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
        # Check it
        if len(s)==0 or s[0]!='#!TDV112':
            f.close()
            raise ValueError("File '%s' must start with '#!TDV112'" % fname)
        # Throw away the next two integers
        self.line2 = np.fromfile(f, count=2, dtype='i4')
        # Read the title
        self.title = cape.io.read_lb4_s(f)
        # Get number of variables (, unpacks the list)
        self.nVar, = np.fromfile(f, count=1, dtype='i4')
        # Loop through variables
        self.Vars = []
        for i in range(self.nVar):
            # Read the name of variable *i*
            self.Vars.append(cape.io.read_lb4_s(f))
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
        # This number should be 299.0
        marker, = np.fromfile(f, dtype='f4', count=1)
        # Read until no more zones
        while marker == 299.0:
            # Increase zone count
            self.nZone += 1
            # Read zone name
            zone = cape.io.read_lb4_s(f).strip('"')
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
            # Check some other aspect about the zone
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
            # Two miscellaneous options about user-defined face something
            np.fromfile(f, dtype='i4', count=2)
            # Number of points, elements
            nPt, nElem = np.fromfile(f, count=2, dtype='i4')
            self.nPt.append(nPt)
            self.nElem.append(nElem)
            # Read some zeros at the end.
            np.fromfile(f, count=4, dtype='i4')
            # This number should be 299.0
            marker, = np.fromfile(f, dtype='f4', count=1)
        # Check for end-of-header marker
        if marker != 357.0:
            raise ValueError(
                "Expecting end-of-header marker 357.0\n" +
                ("  Found: %s" % marker))
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
            ishare = np.fromfile(f, dtype='i4', count=1)
            if ishare != 0:
                np.fromfile(f, dtype='i4', count=self.nVar)
            # Zone number to share with
            zshare, = np.fromfile(f, dtype='i4', count=1)
            # Read the min and max variables
            qi = np.fromfile(f, dtype='f8', count=(self.nVar*2))
            self.qmin[n] = qi[0::2]
            self.qmax[n] = qi[1::2]
            # Read the actual data
            qi = np.fromfile(f, dtype='f4', count=(self.nVar*npt))
            # Reshape
            qi = np.transpose(np.reshape(qi, (self.nVar, npt)))
            self.q.append(qi)
            # Read the tris
            ii = np.fromfile(f, dtype='i4', count=(4*nelem))
            # Reshape and save
            self.Tris.append(np.reshape(ii, (nelem,4)))
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
    def Write(self, fname):
        """Write a Fun3D boundary Tecplot binary file
        
        :Call:
            >>> plt.Write(fname)
        :Inputs:
            *plt*: :class:`pyFun.plt.Plt`
                Tecplot PLT interface
            *fname*: :class:`str`
                Name of file to read
        :Versions:
            * 2017-03-29 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'wb')
        # Write the opening string
        s = np.array('#!TDV112', dtype='|S8')
        # Write it
        s.tofile(f)
        # Write specifier
        cape.io.tofile_ne4_i(f, [1, 0])
        # Write title
        cape.io.tofile_ne4_s(f, self.title)
        # Write number of variables
        cape.io.tofile_ne4_i(f, self.nVar)
        # Loop through variable names
        for var in self.Vars:
            cape.io.tofile_ne4_s(f, var)
        # Write zones
        for i in range(self.nZone):
            # Write goofy zone marker
            cape.io.tofile_ne4_f(f, 299.0)
            # Write zone name
            cape.io.tofile_ne4_s(f, '"%s"' % self.Zones[i])
            # Write parent zone (usually -1)
            try:
                cape.io.tofile_ne4_i(f, self.ParentZone[i])
            except Exception:
                cape.io.tofile_ne4_i(f, -1)
            # Write the StrandID
            try:
                cape.io.tofile_ne4_i(f, self.StrandID[i])
            except Exception:
                cape.io.tofile_ne4_i(f, 1000+i)
            # Write the time
            try:
                cape.io.tofile_ne8_f(f, self.t[i])
            except Exception:
                cape.io.tofile_ne8_f(f, 0.0)
            # Write -1
            cape.io.tofile_ne4_i(f, -1)
            # Write the zone type (3 for triangles)
            try:
                cape.io.tofile_ne4_i(f, self.ZoneType[i])
            except Exception:
                cape.io.tofile_ne4_i(f, 3)
            # Write a bunch of zeros
            try:
                # Check for variable locations (node- or cell-centered)
                if self.QVarLoc[i]:
                    # Variable locations are marked
                    cape.io.tofile_ne4_i(1)
                    # Try to write the markers
                    try:
                        cape.io.tofile_ne4_i(self.VarLocs[i])
                    except Exception:
                        cape.io.tofile_ne4_i(np.zeros(self.nVar))
                else:
                    # Write a zero and move on
                    cape.io.tofile_ne4_i(f, 0)
            except Exception:
                # Default to marking all variables node-centered
                cape.io.tofile_ne4_i(f, 1)
                cape.io.tofile_ne4_i(f, np.zeros(self.nVar))
            # Two unused or weird variables
            cape.io.tofile_ne4_i(f, np.zeros(2))
            # Write number of pts, elements
            cape.io.tofile_ne4_i(f, [self.nPt[i], self.nElem[i]])
            # Write some more zeros
            cape.io.tofile_ne4_i(f, np.zeros(4))
        # Write end-of-header marker
        cape.io.tofile_ne4_f(f, 357.0)
        # Loop through the zones again
        for n in range(self.nZone):
            # Write marker
            cape.io.tofile_ne4_f(f, 299.0)
            # Extract sizes
            npt = self.nPt[n]
            nelem = self.nElem[n]
            # Write variable types (usually 1 for float type, I think)
            try:
                cape.io.tofile_ne4_i(f, self.fmt[n])
            except Exception:
                cape.io.tofile_ne4_i(f, np.ones(self.nVar))
            # Just set things as passive variables like FUN3D
            cape.io.tofile_ne4_i(f, 1)
            cape.io.tofile_ne4_i(f, np.zeros(self.nVar))
            # Just set things to share with -1, because that makes sense
            # somehow.  I guess it is a commercial format, so go figure.
            cape.io.tofile_ne4_i(f, 1)
            cape.io.tofile_ne4_i(f, -1*np.ones(self.nVar))
            # Save the *zshare* value
            cape.io.tofile_ne4_i(f, -1)
            # Form matrix of qmin[0], qmax[0], qmin[1], ...
            qex = np.vstack((self.qmin[n], self.qmax[n])).transpose()
            # Save *qmin* and *qmax*
            cape.io.tofile_ne8_f(f, qex)
            # Save the actual data
            cape.io.tofile_ne4_f(f, np.transpose(self.q[n]))
            # Write the tris (this may need to be generalized!)
            cape.io.tofile_ne4_i(f, self.Tris[n])
        # Close the file
        f.close()
        
    # Write ASCII file 
    def WriteDat(self, fname):
        """Write Tecplot PLT file to ASCII format (``.dat``)
        
        :Call:
            >>> plt.WriteDat(fname)
        :Inputs:
            *plt*: :class:`cape.plt.Plt`
                Tecplot PLT interface
            *fname*: :class:`str`
                Name of DAT file to write
        :Versions:
            * 2017-03-30 ``@ddalle``: First version
        """
        # Create the file
        f = open(fname, 'w')
        # Write the title
        f.write('title="%s"\n' % self.title)
        # Write the variable names header
        f.write('variables = %s\n' % " ".join(self.Vars))
        # Loop through zones
        for n in range(self.nZone):
            # Write the zone name
            f.write('zone t="%s"' % self.Zones[n])
            # Write the time
            f.write(', solutiontime=%14.7E' % self.t[n])
            # Write the strandid
            f.write(', strandid=%s' % self.StrandID[n])
            # Write the number of nodes and elements
            f.write(', i=%s, j=%s' % (self.nPt[n], self.nElem[n]))
            # Write some header that appears fixed.
            f.write(", f=feblock\n")
            # Extract the state
            q = self.q[n]
            # Number of rows of 7
            nrow = int(np.ceil(q.shape[0]/7.0))
            # Loop through the variables
            for j in range(self.nVar):
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
        
    # SZPLOT Boundary reader
    def ReadSzplt(self, fname):
        raise NotImplementedError
        # Open the file
        f = open(fname, 'rb')
        # Read the opening string
        s = np.fromfile(f, count=1, dtype='|S8')
        # Check it
        if len(s)==0 or s[0]!='#!SZPLT ':
            f.close()
            raise ValueError("File '%s' must start with '#!SZPLT '" % fname)
        # Read a revision number string
        self.rev = np.fromfile(f, count=1, dtype='|S34')
        
    # Create from a TRIQ
    def ConvertTriq(self, triq, **kw):
        """Create a PLT object by reading data from a Tri/Triq object
        
        :Call:
            >>> plt.ConvertTriq(triq)
        :Inputs:
            *plt*: :class:`cape.plt.Plt`
                Tecplot PLT interface
            *triq*: :class:`cape.tri.Triq`
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
                qvars = ["var%s" for i in range(nq)]
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
            T = cape.util.TrimUnused(Tris) - 1
            # Form the state matrix for this zone
            if nq > 0:
                # Include *q* variables
                q = np.hstack((Nodes, triq.q[I,:self.nVar]))
            else:
                # Only use nodes as nodal variables
                q = np.hstack((Nodes))
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
            >>> triq = plt.CreateTriq(mach=1.0, triload=True, avg=True, rms=False)
        :Inputs:
            *plt*: :class:`pyFun.plt.Plt`
                Tecplot PLT interface
            *mach*: {``1.0``} | positive :class:`float`
                Freestream Mach number for skin friction coeff conversion
            *triload*: {``True``} | ``False``
                Whether or not to write a triq tailored for ``triloadCmd``
            *avg*: {``True``} | ``False``
                Use time-averaged states if available
            *rms*: ``True`` | {``False``}
                Use root-mean-square variation instead of nominal value
        :Outputs:
            *triq*: :class:`cape.tri.Triq`
                Annotated Cart3D triangulation interface
        :Versions:
            * 2016-12-19 ``@ddalle``: First version
        """
        # Inputs
        triload = kw.get('triload', True)
        # Averaging?
        avg = kw.get('avg', True)
        # Write RMS values?
        rms = kw.get('rms', False)
        # Freestream Mach number; FUN3D writes cf/1.4*pinf instead of cf/qinf
        mach = kw.get('mach', kw.get('m', kw.get('minf', 1.0)))
        # Total number of points
        nNode = np.sum(self.nPt)
        # Rough number of tris
        nElem = np.sum(self.nElem)
        # Initialize
        Nodes = np.zeros((nNode, 3))
        Tris  = np.zeros((nElem, 3), dtype=int)
        # Initialize component IDs
        CompID = np.zeros(nElem, dtype=int)
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
        # Loop through the components
        for k in range(self.nZone):
            # Number of points and elements
            kNode = self.nPt[k]
            kTri  = self.nElem[k]
            # Check for quads
            if np.any(self.Tris[k][:,-1] != self.Tris[k][:,-2]):
                raise ValueError("Detected a quad face; not yet supported " +
                    "for converting PLT files for line loads")
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
            Tris[iTri:iTri+kTri,:] = (self.Tris[k][:,:3] + iNode + 1)
            # Increase the running node count
            iNode += kNode
            # Try to read the component ID
            try:
                # The name of the zone should be 'boundary 9 CORE_Body' or sim
                comp = int(self.Zones[k].split()[1])
            except Exception:
                # Otherwise just number 1 to *n*
                comp = np.max(CompID) + 1
            # Save the component IDs
            CompID[iTri:iTri+kTri] = comp
            # Increase the running tri count
            iTri += kTri
        # Create the triangulation
        triq = cape.tri.Triq(Nodes=Nodes, Tris=Tris, q=q, CompID=CompID)
        # Output
        return triq
        
# class Plt

