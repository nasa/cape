"""
Tecplot PLT File Interface for Fun3D
====================================


"""

# Basic numerics
import numpy as np
# Useful tool for more complex binary I/O methods
import cape.io
import cape.tri

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
    triq = plt.CreateTriq(triload=ll, avg=avg, rms=rms)
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
        >>> plt = pyFun.plt.Plt(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *plt*: :class:`pyFun.plt.Plt`
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
    
    def __init__(self, fname):
        """Initialization method
        
        :Versions:
            * 2016-11-21 ``@ddalle``: Started
            * 2016-11-22 ``@ddalle``: First version
        """
        # Read the file
        self.Read(fname)
    
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
        np.fromfile(f, count=2, dtype='i4')
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
            # Read a lot of zeros
            np.fromfile(f, dtype='i4', count=(self.nVar+3))
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
            raise ValueError("Expecting end-of-header marker 357.0")
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
        
    # Create a triq file
    def CreateTriq(self, triload=True, avg=True, rms=False):
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
            >>> triq = plt.CreateTriq(triload=True, avg=True, rms=False)
        :Inputs:
            *plt*: :class:`pyFun.plt.Plt`
                Tecplot PLT interface
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
                    cfx  = self.q[k][:,jcfxr]
                    cfy  = self.q[k][:,jcfyr]
                    cfz  = self.q[k][:,jcfzr]
                else:
                    # Nominal states
                    cp  = self.q[k][:,jcp]
                    rho = self.q[k][:,jrho]
                    u   = self.q[k][:,ju] * rho
                    v   = self.q[k][:,jv] * rho
                    w   = self.q[k][:,jw] * rho
                    p   = self.q[k][:,jp]
                    cfx = self.q[k][:,jcfx]
                    cfy = self.q[k][:,jcfy]
                    cfz = self.q[k][:,jcfz]
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
        
    
