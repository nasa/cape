"""
Module to interface with "over.namelist" files: :mod:`pyOver.namelist`
======================================================================

This is a module built off of the :mod:`cape.fileCntl` module customized for
manipulating OVERFLOW namelists.
"""

# Numerics
import numpy as np

# Import the base file control class.
import cape.namelist2

# Function to compare boundary indices
def gti(a, b):
    """Altered greater-than test for Fortran array indices
    
    Negative indices are always considered to be greater than positive ones,
    and negative indices closer to zero are the largest.  The general pattern
    is ``1 < 2 < 20 < -20 < -1``, and ``-1`` is the maximum possible value.
    
    :Call:
        >>> q = gti(a, b)
    :Inputs:
        *a*: :class:`int` | :class:`float`
            First test value
        *b*: :class:`int` | :class:`float`
            Second test value
    :Outputs:
        *q*: ``True`` | ``False``
            Whether or not *a* > *b*
    :Versions:
        * 2016-12-30 ``@ddalle``: First version
    """
    # Check signs
    if a*b > 0:
        # Both negative of both positive; use normal test
        return a > b
    elif a < 0:
        # *a* is negative and *b* is positive
        return True
    elif b < 0:
        # *b* is negative and *a* is positive
        return False
    else:
        # Can't process ``0``
        raise ValueError("Index of ``0`` is not valid for Fortran")

# Function to compare boundary indices
def gteqi(a, b):
    """Altered greater-than-or-equal-to test for Fortran array indices
    
    Negative indices are always considered to be greater than positive ones,
    and negative indices closer to zero are the largest.  The general pattern
    is ``1 < 2 < 20 < -20 < -1``, and ``-1`` is the maximum possible value.
    
    :Call:
        >>> q = gteqi(a, b)
    :Inputs:
        *a*: :class:`int` | :class:`float`
            First test value
        *b*: :class:`int` | :class:`float`
            Second test value
    :Outputs:
        *q*: ``True`` | ``False``
            Whether or not *a* > *b*
    :Versions:
        * 2016-12-30 ``@ddalle``: First version
    """
    # Check if the two values are equal
    if a == b:
        # Equal
        return True
    else:
        # Go to altered gt test
        return gti(a, b)

# Function to compare boundary indices
def lti(a, b):
    """Altered less-than test for Fortran array indices
    
    Negative indices are always considered to be greater than positive ones,
    and negative indices closer to zero are the largest.  The general pattern
    is ``1 < 2 < 20 < -20 < -1``, and ``-1`` is the maximum possible value.
    
    :Call:
        >>> q = lti(a, b)
    :Inputs:
        *a*: :class:`int` | :class:`float`
            First test value
        *b*: :class:`int` | :class:`float`
            Second test value
    :Outputs:
        *q*: ``True`` | ``False``
            Whether or not *a* > *b*
    :Versions:
        * 2016-12-30 ``@ddalle``: First version
    """
    # Check signs
    if a*b > 0:
        # Both negative of both positive; use normal test
        return a < b
    elif a < 0:
        # *a* is negative and *b* is positive
        return False
    elif b < 0:
        # *b* is negative and *a* is positive
        return True
    else:
        # Can't process ``0``
        raise ValueError("Index of ``0`` is not valid for Fortran")

# Function to compare boundary indices
def lteqi(a, b):
    """Altered less-than-or-equal-to test for Fortran array indices
    
    Negative indices are always considered to be greater than positive ones,
    and negative indices closer to zero are the largest.  The general pattern
    is ``1 < 2 < 20 < -20 < -1``, and ``-1`` is the maximum possible value.
    
    :Call:
        >>> q = lteqi(a, b)
    :Inputs:
        *a*: :class:`int` | :class:`float`
            First test value
        *b*: :class:`int` | :class:`float`
            Second test value
    :Outputs:
        *q*: ``True`` | ``False``
            Whether or not *a* > *b*
    :Versions:
        * 2016-12-30 ``@ddalle``: First version
    """
    # Check if the two values are equal
    if a == b:
        # Equal
        return True
    else:
        # Go to altered gt test
        return lti(a, b)

# Altered maximum function
def maxi(a, b):
    """Altered maximum function for array indices
    
    Negative indices are always considered to be greater than positive ones,
    and negative indices closer to zero are the largest.  The general pattern
    is ``1 < 2 < 20 < -20 < -1``, and ``-1`` is the maximum possible value.
    
    :Call:
        >>> c = maxi(a, b)
    :Inputs:
        *a*: :class:`int` | :class:`float`
            First test value
        *b*: :class:`int` | :class:`float`
            Second test value
    :Outputs:
        *c*: :class:`int` | :class:`float`
            Either *a* or *b* depending on which is greater
    :Versions:
        * 2016-12-30 ``@ddalle``: First version
    """
    # Test a,b
    if gti(a, b):
        return a
    else:
        return b

# Altered minimum function
def mini(a, b):
    """Altered minimum function for array indices
    
    Negative indices are always considered to be greater than positive ones,
    and negative indices closer to zero are the largest.  The general pattern
    is ``1 < 2 < 20 < -20 < -1``, and ``-1`` is the maximum possible value.
    
    :Call:
        >>> c = maxi(a, b)
    :Inputs:
        *a*: :class:`int` | :class:`float`
            First test value
        *b*: :class:`int` | :class:`float`
            Second test value
    :Outputs:
        *c*: :class:`int` | :class:`float`
            Either *a* or *b* depending on which is greater
    :Versions:
        * 2016-12-30 ``@ddalle``: First version
    """
    # Use altered test
    if lti(a, b):
        return a
    else:
        return b
# def mini


# Base this class off of the main file control class.
class OverNamelist(cape.namelist2.Namelist2):
    """
    File control class for :file:`over.namelist`
    ============================================
            
    This class is derived from the :class:`pyCart.fileCntl.FileCntl` class, so
    all methods applicable to that class can also be used for instances of this
    class.
    
    :Call:
        >>> nml = pyOver.Namelist2()
        >>> nml = pyOver.Namelist2(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of namelist file to read, defaults to ``'over.namelist'``
    :Outputs:
        *nml*: :class:`pyOver.overNamelist.OverNamelist`
            Interface to OVERFLOW input namelist
    :Version:
        * 2016-01-31 ``@ddalle``: First version
    """
    
    # Initialization method (not based off of FileCntl)
    def __init__(self, fname="over.namelist"):
        """Initialization method"""
        # Read the file.
        self.Read(fname)
        # Save the file name.
        self.fname = fname
        # Split into sections.
        self.UpdateNamelist()
        # Get grid names
        self.GetGridNames()
        
    # Function to get list of grid names
    def GetGridNames(self):
        """Get the list of grid names in an OVERFLOW namelist
        
        :Call:
            >>> nml.GetGridNames()
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
        :Versions:
            * 2016-01-31 ``@ddalle``: First version
        """
        # Get list indices of the 'GRDNAM' lists
        I = self.GetGroupByName('GRDNAM', None)
        # Save the names as an array (for easier searching)
        self.GridNames = [self.GetKeyFromGroupIndex(i, 'NAME') for i in I]
        # Save the indices of those lists
        self.iGrid = I
        
    # Get grid number
    def GetGridNumberByName(self, grdnam):
        """Get the number of a grid from its name
        
        :Call:
            >>> i = nml.GetGridNumberByName(grdnam)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *grdnam*: :class:`str`
                Name of the grid
        :Outputs:
            *i*: :class:`int`
                Grid number 
        :Versions:
            * 2016-01-31 ``@ddalle``: First version
        """
        # Check for integer
        if type(grdnam).__name__.startswith('int'):
            return grdnam
        # Check if the grid is present
        if grdnam not in self.GridNames:
            raise KeyError("No grid named '%s' was found" % grdnam)
        # Return the index
        return self.GridNames.index(grdnam)
    
    # Get grid number (alias)
    GetGridNumber = GetGridNumberByName
    
    # Write SPLITMQ.I file
    def WriteSplitmqI(self, fname="splitmq.i", wall=True):
        """Write a ``splitmq.i`` file to extract surface and second layer
        
        :Call:
            >>> nml.WriteSplitmqI(fname="splitmq.i", wall=True)
        :Inputs:
            *nml*: :clas:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *fname*: {``"splitmq.i"``} | :class:`str`
                Name of ``splitmq`` input file to write
            *wall*: {``True``} | ``False``
                Only include walls if ``True``; else include thrust BCs
        :Versions:
            * 2016-12-30 ``@ddalle``: First version
        """
        # Open the output file
        f = open(fname, 'w')
        # Write header
        f.write('q.p3d\n')
        f.write('q.save\n')
        # Valid boundary condition types
        wtyp = range(1,10)
        # Check for other surface BCs such as thrust BCs
        if wall == False:
            wtyp += [42, 153]
        # Loop through grids
        gn = 0
        for grid in self.GridNames:
            # Increase grid number
            gn += 1
            # Get boundary conditions and directions
            ibtyp = self.GetKeyFromGrid(grid, 'BCINP', 'IBTYP')
            ibdir = self.GetKeyFromGrid(grid, 'BCINP', 'IBDIR')
            # Check for off-body grid
            if ibtyp is None:
                # Off-body grid
                continue
            elif type(ibtyp).__name__ == "int":
                # Integer; create list
                ibtyp = [ibtyp]
                ibdir = [ibdir]
            # Check for other non-boundary grid
            J = np.intersect1d(ibtyp, wtyp)
            # If no walls, skip this grid
            if len(J) == 0: continue
            # Get range of indices
            jbcs = self.GetKeyFromGrid(grid, 'BCINP', 'JBCS')
            jbce = self.GetKeyFromGrid(grid, 'BCINP', 'JBCE')
            kbcs = self.GetKeyFromGrid(grid, 'BCINP', 'KBCS')
            kbce = self.GetKeyFromGrid(grid, 'BCINP', 'KBCE')
            lbcs = self.GetKeyFromGrid(grid, 'BCINP', 'LBCS')
            lbce = self.GetKeyFromGrid(grid, 'BCINP', 'LBCE')
            # Enlist
            if type(jbcs).__name__ == "int":
                jbcs = [jbcs]; jbce = [jbce]
                kbcs = [kbcs]; kbce = [kbce]
                lbcs = [lbcs]; lbce = [lbce]
            # Loop through the three directions
            for k in [1, 2, 3]:
                # Initialize range
                ja = -1; jb = 1
                ka = -1; kb = 1
                la = -1; lb = 1
                # Loop through boundary conditions, looking only for walls
                for i in range(len(ibtyp)):
                    # Check for valid BC type
                    if ibtyp[i] not in wtyp: continue
                    # Check direction
                    if ibdir[i] != k: continue
                    # Compare boundaries
                    ja = mini(ja, jbcs[i]); jb = maxi(jb, jbce[i])
                    ka = mini(ka, kbcs[i]); kb = maxi(kb, kbce[i])
                    la = mini(la, lbcs[i]); lb = maxi(lb, lbce[i])
                # Check for valid range
                if (k==3) and lteqi(ja,jb) and lteqi(ka,kb):
                    # Write L=1,2
                    la = 1; lb = 2
                elif (k==1) and lteqi(ka,kb) and lteqi(la,lb):
                    # Write J=1,2
                    ja = 1; jb = 2
                elif (k==2) and lteqi(la,lb) and lteqi(ja,jb):
                    # Write k=1,2
                    ka = 1; kb = 2
                else:
                    continue
                # Write the range
                f.write("%5i," % gn)
                f.write("%9i,%6i,%6i," % (ja, jb, 1))
                f.write("%9i,%6i,%6i," % (ka, kb, 1))
                f.write("%9i,%6i,%6i," % (la, lb, 1))
                f.write("\n")
        # Close the file
        f.close()
    
    # Apply dictionary of options to a grid
    def ApplyDictToGrid(self, grdnam, opts):
        """Apply a dictionary of settings to a grid
        
        :Call:
            >>> nml.ApplyDictToGrid(grdnam, opts)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *grdnam*: :class:`str` | :class:`int`
                Grid name or index
            *opts*: :class:`dict`
                Dictionary of options to apply
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Get the indices of the requested grid's groups
        jbeg, jend = self.GetGroupIndexByGridName(grdnam)
        # Get the list of group names
        grps = [self.Groups[j].lower() for j in range(jbeg,jend)]
        # Loop through the major keys
        for grp in opts:
            # Use lower-case group name for Fortran consistency
            grpl = grp.lower()
            # Find the group index
            if grpl in grps:
                # Get the group index (global)
                jgrp = jbeg + grps.index(grpl)
            else:
                # Insert the group
                self.InsertGroup(jend, grp)
                jgrp = jend
                # Update info
                jend += 1
            # Loop through the keys
            for k in opts[grp]:
                # Set the value.
                self.SetKeyInGroupIndex(jgrp, k, opts[grp][k])
                
    # Apply a dictionary of options to all grids
    def ApplyDictToALL(self, opts):
        """Apply a dictionary of settings to all grids
        
        :Call:
            >>> nml.ApplyDictToALL(opts)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *opts*: :class:`dict`
                Dictionary of options to apply
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Loop through groups
        for igrd in range(len(self.GridNames)):
            # Apply settings to individual grid
            self.ApplyDictToGrid(igrd, opts)
        
    # Get a quantity from a grid (with fallthrough)
    def GetKeyFromGrid(self, grdnam, grp, key, i=None):
        """Get the value of a key for a grid with a specific name
        
        This function uses fall-through, so if a setting is not explicitly
        defined for grid *grdnam*, it will check the preceding grid, and the
        grid before that, etc.
        
        :Call:
            >>> val = nml.GetKeyFromGrid(grdnam, grp, key, i=None)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *grdnam*: :class:`str`
                Name of the grid
            *grp*: :class:`str`
                Name of the namelist group of key to query
            *key*: :class:`str`
                Name of the key to query
            *i*: {``None``} | ``':'`` | :class:`int`
                Index to use in the namelist, e.g. "BCPAR(*i*)"
        :Outputs:
            *val*: :class:`str` | :class:`int` | :class:`float` | :class:`bool`
                Value from the namelist
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
            * 2016-08-29 ``@ddalle``: Added namelist indices
        """
        # Use lower case for Fortran consistency
        grpl = grp.lower()
        # Get the indices of the requested grid
        jbeg, jend = self.GetGroupIndexByGridName(grdnam)
        # Get the list of group names
        grps = [self.Groups[j].lower() for j in range(jbeg,jend)]
        # Check if the group is in the requested grid definition
        if grpl in grps:
            # Group explicitly in *grdnam* defn
            jgrp = jbeg + grps.index(grpl)
        else:
            # Get the groups for grid 0 (i.e. first grid)
            jbeg, jend = self.GetGroupIndexByGridName(0)
            grps = [self.Groups[j].lower() for j in range(jbeg,jend)]
            # If not in grid 0, this is not a valid request
            if grpl not in grps:
                raise KeyError("No group named '%s' in grid definitions" % grp)
            # Otherwise, loop backwards until we find it (fallthrough)
            igrid = self.GetGridNumberByName(grdnam)
            # Loop until found
            while igrid > 0:
                # Move backwards a grid
                igrid -= 1
                # Get the groups for that grid
                jbeg, jend = self.GetGroupIndexByGridName(igrid)
                grps = [self.Groups[j].lower() for j in range(jbeg,jend)]
                # Test for a match
                if grpl in grps:
                    # Use this group index; discontinue search
                    jgrp = jbeg + grps.index(grpl)
                    break
        # Get the key from this group.
        return self.GetKeyFromGroupIndex(jgrp, key, i=i)
        
    # Set a quantity for a specific grid
    def SetKeyForGrid(self, grdnam, grp, key, val, i=None):
        """Set the value of a key for a grid with a specific name
        
        :Call:
            >>> nml.SetKeyForGrid(grdnam, grp, key, val, i=None)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *grdnam*: :class:`str` | :class:`int`
                Name or number of the grid
            *grp*: :class:`str`
                Name of the namelist group
            *key*: :class:`str`
                Name of the key to set
            *val*: :class:`str` | :class:`float` | :class:`bool` | ...
                Value to set the key to
            *i*: {``None``} | ``':'`` | :class:`int`
                Index to use in the namelist, e.g. "BCPAR(*i*)"
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
            * 2016-08-29 ``@ddalle``: Added namelist indices
        """
        # Use lower-case group name for Fortran consistency
        grpl = grp.lower()
        # Get the indices of the requested grid's groups
        jbeg, jend = self.GetGroupIndexByGridName(grdnam)
        # Get the list of group names
        grps = [self.Groups[j].lower() for j in range(jbeg,jend)]
        # Check if the group is in the requested grid
        if grpl in grps:
            # Get the overall index of the requested group
            jgrp = jbeg + grps.index(grpl)
        else:
            # Insert a new group at the end of this grid definition
            self.InsertGroup(jend, grp)
            # Use the new group
            jgrp = jend
        # Set the key in that group
        self.SetKeyInGroupIndex(jgrp, key, val, i)
        
    # Get list of lists in a grid
    def GetGroupNamesByGridName(self, grdnam):
        """Get the list names in a grid definition
        
        :Call:
            >>> grps = nml.GetGroupNamesByGridName(grdnam)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *grdnam*: :class:`str`
                Name of the grid
        :Outputs:
            *grps*: :class:`list` (:class:`str`)
                List of group names in the grid *grdnam* definition
        :Versions:
            * 2016-01-31 ``@ddalle``: First version
        """
        # Get the start and end indices
        jbeg, jend = self.GetGroupIndexByGridName(grdnam)
        # Return the corresponding list
        return [self.Groups[j] for j in range(jbeg,jend)]
    
    # Get start and end of list indices in a grid
    def GetGroupIndexByGridName(self, grdnam):
        """Get the indices of the first and last list in a grid by name
        
        :Call:
            >>> jbeg, jend = nml.GetGroupIndexByGridName(grdnam)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *grdnam*: :class:`str`
                Name of the grid
        :Outputs:
            *jbeg*: :class:`int`
                Index of first list in the grid definition
            *jend*: :class:`int`
                Index of last list in the grid definition
        :Versions:
            * 2016-01-31 ``@ddalle``: First version
        """
        # Get the grid number
        grdnum = self.GetGridNumberByName(grdnam)
        # Get the list index of this grid's start
        jbeg = self.iGrid[grdnum]
        # Number of grids
        nGrid = len(self.GridNames)
        # Get the list index of the last list in this grid
        if grdnum >= nGrid-1:
            # Use the last list
            jend = len(self.Groups)
        else:
            # Use the list before the start of the next grid
            jend = self.iGrid[grdnum+1]
        # Output
        return jbeg, jend
        
    # Get FLOINP value
    def GetFLOINP(self, key):
        """Return value of key from the $FLOINP group
        
        :Call:
            >>> val = nml.GetFLOINP(key)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *key*: :class:`str`
                Name of field to query
        :Outputs:
            *val*: :class:`float` | :class:`list`
                Value of field *key* in group ``"FLOINP"``
        :Versions:
            * 2016-02-01 ``@ddalle``
        """
        return self.GetKeyInGroupName('FLOINP', key)
        
    # Set FLOINP value
    def SetFLOINP(self, key, val):
        """Set the value of key in the $FLOINP group
        
        :Call:
            >>> nml.SetFLOINP(key, val)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *key*: :class:`str`
                Name of field to query
            *val*: :class:`float` | :class:`list`
                Value of field *key* in group ``"FLOINP"``
        :Versions:
            * 2016-02-01 ``@ddalle``
        """
        self.SetKeyInGroupName('FLOINP', key, val)
        
    # Get GLOBAL value
    def GetGLOBAL(self, key):
        """Return value of key from the $GLOBAL group
        
        :Call:
            >>> val = nml.GetGLOBAL(key)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *key*: :class:`str`
                Name of field to query
        :Outputs:
            *val*: :class:`int` | :class:`bool` | :class:`list`
                Value of field *key* in group ``"GLOBAL"``
        :Versions:
            * 2016-02-01 ``@ddalle``
        """
        return self.GetKeyInGroupName('GLOBAL', key)
        
    # Set GLOBAL value
    def SetGLOBAL(self, key, val):
        """Set value of key from the $GLOBAL group
        
        :Call:
            >>> nml.GetGLOBAL(key, val)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *key*: :class:`str`
                Name of field to query
            *val*: :class:`int` | :class:`bool` | :class:`list`
                Value of field *key* in group ``"GLOBAL"``
        :Versions:
            * 2016-02-01 ``@ddalle``
        """
        self.SetKeyInGroupName('GLOBAL', key, val)
        
    # Function set the Mach number.
    def SetMach(self, mach):
        """Set the freestream Mach number
        
        :Call:
            >>> nml.SetMach(mach)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *mach*: :class:`float`
                Mach number
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Replace the line or add it if necessary.
        self.SetKeyInGroupName('FLOINP', 'FSMACH', mach)
        
    # Function to get the current Mach number.
    def GetMach(self):
        """Find the current Mach number
        
        :Call:
            >>> mach = nml.GetMach()
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
        :Outputs:
            *M*: :class:`float` (or :class:`str`)
                Mach number specified in :file:`input.cntl`
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Get the value.
        return self.GetKeyFromGroupName('FLOINP', 'FSMACH')
        
    # Function to set the angle of attack
    def SetAlpha(self, alpha):
        """Set the angle of attack
        
        :Call:
            >>> nml.SetAlpha(alpha)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *alpha*: :class:`float`
                Angle of attack
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Replace the line or add it if necessary.
        self.SetKeyInGroupName('FLOINP', 'ALPHA', alpha)
        
    # Get the angle of attack
    def GetAlpha(self):
        """Return the angle of attack
        
        :Call:
            >>> alpha = nml.GetAlpha()
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
        :Outputs:
            *alpha*: :class:`float`
                Angle of attack
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Get the value
        return self.GetKeyFromGroupName('FLOINP', 'ALPHA')
        
    # Function to set the sideslip angle
    def SetBeta(self, beta):
        """Set the sideslip angle
        
        :Call:
            >>> nml.SetBeta(beta)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *beta*: :class:`float`
                Sideslip angle
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        # Replace the line or add it if necessary.
        self.SetKeyInGroupName('FLOINP', 'BETA', beta)
        
    # Get the slideslip angle
    def GetBeta(self):
        """Get the sideslip angle
        
        :Call:
            >>> beta = nml.GetBeta()
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
        :Outputs:
            *beta*: :class:`float`
                Sideslip angle
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        return self.GetKeyFromGroupName('FLOINP', 'BETA')
        
    # Set the temperature
    def SetTemperature(self, T):
        """Set the freestream temperature
        
        :Call:
            >>> nml.SetTemperature(T)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *T*: :class:`float`
                Freestream temperature
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        self.SetKeyInGroupName('FLOINP', 'TINF', T)
        
    # Get the temperature
    def GetTemperature(self):
        """Get the freestream temperature
        
        :Call:
            >>> T = nml.GetTemperature()
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
        :Outputs:
            *T*: :class:`float`
                Freestream temperature
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        return self.GetKeyInGroupName('FLOINP', 'TINF')        
        
    # Set the Reynolds number
    def SetReynoldsNumber(self, Re):
        """Set the Reynolds number per unit length
        
        :Call:
            >>> nml.SetReynoldsNumber(Re)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *Re*: :class:`float`
                Reynolds number per unit length
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        self.SetKeyInGroupName('FLOINP', 'REY', Re)
        
    # Get the Reynolds number
    def GetReynoldsNumber(self):
        """Get the Reynolds number per unit length
        
        :Call:
            >>> Re = nml.GetReynoldsNumber()
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
        :Outputs:
            *Re*: :class:`float`
                Reynolds number per unit length
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        return self.GetKeyInGroupName('FLOINP', 'REY')
        
    # Set the number of iterations
    def SetnIter(self, nIter):
        """Set the number of iterations
        
        :Call:
            >>> nml.SetnIter(nIter)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *nIter*: :class:`int`
                Number of iterations to run
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        self.SetKeyInGroupName('GLOBAL', 'NSTEPS', nIter)
        
    # Get the number of iterations
    def GetnIter(self):
        """Get the number of iterations
        
        :Call:
            >>> nIter = nml.GetnIter()
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
        :Outputs:
            *nIter*: :class:`int`
                Number of iterations to run
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        return self.GetKeyInGroupName('GLOBAL', 'NSTEPS')
        
    # Set the restart setting
    def SetRestart(self, q=True):
        """Set or unset restart flag
        
        :Call:
            >>> nml.SetRestart(q=True)
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
            *q*: :class:`bool`
                Whether or not to run as a restart
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        self.SetKeyInGroupName('GLOBAL', 'RESTRT', q)
        
    # Get the restart setting
    def GetRestart(self):
        """Get the current restart flag
        
        :Call:
            >>> q = nml.GetRestart()
        :Inputs:
            *nml*: :class:`pyOver.overNamelist.OverNamelist`
                Interface to OVERFLOW input namelist
        :Outputs:
            *q*: :class:`bool`
                Whether or not to run as a restart
        :Versions:
            * 2016-02-01 ``@ddalle``: First version
        """
        return self.GetKeyInGroupName('GLOBAL', 'RESTRT')
    
    
# class Namelist

        
