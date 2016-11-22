"""
Tecplot PLT File Interface for Fun3D
====================================


"""

# Basic numerics
import numpy as np
# Useful tool for more complex binary I/O methods
import cape.io

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
    
