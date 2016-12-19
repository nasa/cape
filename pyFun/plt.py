"""
Tecplot PLT File Interface for Fun3D
====================================


"""

# Basic numerics
import numpy as np
# Useful tool for more complex binary I/O methods
import cape.io
import cape.tri

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
    def CreateTriq(self, triload=True):
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
            >>> triq = plt.CreateTriq(triload=True)
        :Inputs:
            *plt*: :class:`pyFun.plt.Plt`
                Tecplot PLT interface
            *triload*: {``True``} | ``False``
                Whether or not to write a triq tailored for ``triloadCmd``
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
            # Indices of vars to use
            jcp  = self.Vars.index('cp')
            jrho = self.Vars.index('rho')
            ju   = self.Vars.index('u')
            jv   = self.Vars.index('v')
            jw   = self.Vars.index('w')
            jp   = self.Vars.index('p')
        elif qtype == 2:
            # Full set of states including viscous
            nq = 9
            # Indices of vars to use
            jcp  = self.Vars.index('cp')
            jrho = self.Vars.index('rho')
            ju   = self.Vars.index('u')
            jv   = self.Vars.index('v')
            jw   = self.Vars.index('w')
            jp   = self.Vars.index('p')
            jcfx = self.Vars.index('cf_x')
            jcfy = self.Vars.index('cf_y')
            jcfz = self.Vars.index('cf_z')
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
                # Save the primary states appropriately
                q[iNode:iNode+kNode,0] = self.q[k][:,jcp]
                q[iNode:iNode+kNode,1] = self.q[k][:,jrho]
                q[iNode:iNode+kNode,2] = self.q[k][:,ju]/self.q[k][:,jrho]
                q[iNode:iNode+kNode,3] = self.q[k][:,jv]/self.q[k][:,jrho]
                q[iNode:iNode+kNode,4] = self.q[k][:,jw]/self.q[k][:,jrho]
                q[iNode:iNode+kNode,5] = self.q[k][:,jp]
            elif qtype == 2:
                # Save the primary states appropriately
                q[iNode:iNode+kNode,0] = self.q[k][:,jcp]
                q[iNode:iNode+kNode,1] = self.q[k][:,jrho]
                q[iNode:iNode+kNode,2] = self.q[k][:,ju]/self.q[k][:,jrho]
                q[iNode:iNode+kNode,3] = self.q[k][:,jv]/self.q[k][:,jrho]
                q[iNode:iNode+kNode,4] = self.q[k][:,jw]/self.q[k][:,jrho]
                q[iNode:iNode+kNode,5] = self.q[k][:,jp]
                q[iNode:iNode+kNode,6] = self.q[k][:,jcfx]
                q[iNode:iNode+kNode,7] = self.q[k][:,jcfy]
                q[iNode:iNode+kNode,8] = self.q[k][:,jcfz]
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
        
    
