"""
Tecplot PLT File Interface for Fun3D
====================================


"""

# Basic numerics
import numpy as np
# Useful tool for more complex binary I/O methods
import cape.io

# Tecplot class
class plt(object):
    
    
    def __init__(self, fname):
        """Initialization method
        
        :Versions:
            * 2016-11-21 ``@ddalle``: Started
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
            # Read a -1
            np.fromfile(f, count=1, dtype='i4')
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
            np.fromfile(f, dtype='i4', count=(nVar+3))
            # Number of points, elements
            nPt, nElem = np.fromfile(f, count=2, dtype='i4')
            self.nPt.append(nPt)
            self.nElem.append(nElem)
            # Read three zeros at the end.
            np.fromfile(f, count=4, dtype='i4')
        #
        
        # Close the file
        #f.close()
        return f
    
