#!/usr/bin/env python
"""
Python interface to Plot3D files
================================

:Versions:
    * 2016-02-26 ``@ddalle``: First version
"""

# Numerics
import numpy as np
# System interface
import os
# Basic Plot3D class
import cape.plot3d


# OVERFLOW Plot3D template
class P3D(cape.plot3d.X):
    
        
    # Determine byte order
    def get_byteorder(self, endian=None):
        """Determine the proper byte order automatically if necessary
        
        :Call:
            >>> q.get_byteorder(endian=None)
        :Inputs:
            *q*: :class:`pyOver.plot3d.Q`
                General OVERFLOW q-file interface
            *endian*: {``None``} | "big" | "little"
                Manually-specified byte order
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Check for manual override
        if endian is not None:
            self.endian = endian
        # If no override, open the file
        self.open()
        # Read the first entry, which is a record marker for the first line
        # Try little-endian first
        i0 = np.fromfile(self.f, count=1, dtype='<i4')
        # Check the value
        if i0 == 4 or i0%12 == 8:
            # This was the correct byte order
            self.endian = "little"
        else:
            # Big-endian
            self.endian = "big"
        
    # Determine read flags
    def get_dtypes(self):
        """Save NumPy data types to prevent later look-ups
        
        :Call:
            >>> q.get_dtypes()
        :Inputs:
            *q*: :class:`pyOver.plot3d.Q`
                General OVERFLOW q-file interface
        :Data members:
            *q.itype*: '<i4' | '>i4'
                Data type for integer entries
            *q.ftype*: '<f8' | '>f8'
                Data type for floating-point entries
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Check endianness
        if self.endian == "little":
            # Little-endian 64-bit flags
            self.itype = "<i4"
            self.ftype = "<f8"
        else:
            # Big-endian 64-bit flags
            self.itype = ">i4"
            self.ftype = ">f8"
    
    # Get grid indices
    def expand_grid_indices(self, IG, **kw):
        """Expand grid indices using a variety of input methods
        
        Portions of a grid can be extracted either by using a list of indices
        using the *J*, *K*, *L* keyword arguments, individual indices using the
        *J*, *K*, *L* keyword arguments, or start-end subsets using *JS*, *JE*,
        etc.  The *J* keyword takes precedence over *JS*, *JE* if bot hare
        specified.
        
        :Call:
            >>> J, K, L = q.expand_grid_indices(IG, **kw)
        :Inputs:
            *q*: :class:`pyOver.plot3d.Q`
                General OVERFLOW q-file interface
            *IG*: :class:`int`
                Grid number (one-based index)
        :Keyword arguments:
            *J*: :class:`int` | :class:`list` (:class:`int`) 
                Single grid index, *j* direction
            *JS*: :class:`int`
                Start index, *j* direction
            *JE*: :class:`int`
                End index, *j* direction
            *K*: :class:`int` | :class:`list` (:class:`int`)
                Single grid index, *k* direction
            *KS*: :class:`int`
                Start index, *k* direction
            *KE*: :class:`int`
                End index, *k* direction
            *L*: :class:`int` | :class:`list` (:class:`int`)
                Single grid index, *l* direction
            *LS*: :class:`int`
                Start index, *l* direction
            *LE*: :class:`int`
                End index, *l* direction
        :Outputs:
            *J*: :class:`np.ndarray` (:class:`int`) 
                Array of grid indices in *j* direction
            *K*: :class:`np.ndarray` (:class:`int`) 
                Array of grid indices in *k* direction
            *L*: :class:`np.ndarray` (:class:`int`) 
                Array of grid indices in *l* direction
        :Versions:
            * 2016-03-07 ``@ddalle``: First version
        """
        # Grid sizes
        JD = self.JD[IG-1]
        KD = self.KD[IG-1]
        LD = self.LD[IG-1]
        # Process index start indices
        JS = kw.get("JS", 1)
        KS = kw.get("KS", 1)
        LS = kw.get("LS", 1)
        # Process index end indices
        JE = kw.get("JE", JD)
        KE = kw.get("KE", KD)            
        LE = kw.get("LE", LD)
        # Check for negative indices
        if JS < 0: JS += JD
        if KS < 0: KS += KD
        if LS < 0: LS += LD
        if JE < 0: JE += JD
        if KE < 0: KE += KD
        if LE < 0: LE += LD
        # Default indices
        JDEF = JS if JS==JE else np.arange(JS,JE+1)
        KDEF = KS if KS==KE else np.arange(KS,KE+1)
        LDEF = LS if LS==LE else np.arange(LS,LE+1)
        # Process direct indices
        J = kw.get("J", JDEF) - 1
        K = kw.get("K", KDEF) - 1
        L = kw.get("L", LDEF) - 1
        # Output
        return J, K, L
    
    
# class P3D
    

# OVERFLOW q class
class Q(P3D):
    """
    General OVERFLOW ``q`` file interface
    
    :Call:
        >>> q = pyOver.plot3d.Q(fname, endian=None)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *endian*: {``None``} | "big" | "little"
            Manually-specified byte order
    :Outputs:
        *q*: :class:`pyOver.plot3d.Q`
            General OVERFLOW q-file interface
    :Versions:
        * 2016-02-26 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self, fname, endian=None):
        """Initialization method
        
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Save the file name
        self.fname = fname
        # File handle
        self.f = None
        # Save required states
        # Reference viscosity
        self.mu0  = 2.2696e-8
        # Sutherland's law reference temperature [R]
        self.TREF = 198.6
        # Gas constant [unitus ridiculous]
        self.R = 1716.0 
        # Determine endianness
        self.get_byteorder(endian)
        # Get flags
        self.get_dtypes()
        # Read the file
        self.Read()
    
    # Read the file
    def Read(self):
        """Read an OVERFLOW generic Q file
        
        :Call:
            >>> q.Read()
        :Inputs:
            *q*: :class:`pyOver.plot3d.Q`
                General OVERFLOW q-file interface
        :Data members:
            *q.nGrid*: :class:`int`
                Number of grids
            *q.mGrid*: :class:`bool`
                Whether or not file is a multiple-grid file
            *q.Q*: :class:`list` (:class:`numpy.ndarray`)
                List of solution arrays
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Open file if necessary
        if self.closed == True:
            self.open()
        # Get number of grids
        nGrid = self.GetNGrid()
        # Read grid dimensions
        self.GetGridDims()
        # Initialize header quantities
        self.InitHeaders()
        # Loop through grids
        for i in range(nGrid):
            # Read headers
            self.ReadQHeader(i+1)
            self.ReadQData(i+1)
        # Reread if q.restart...
        
        # Close the file
        self.close()
        # Freestream viscosity
        self.MUINF = self.mu0 * self.TINF**1.5 / (self.TINF+self.TREF)
        # Freestream speed of sound
        self.AINF = np.sqrt(self.GAMINF * self.R * self.TINF)
        # Freestream speed
        self.UINF = self.AINF * self.FSMACH
        # Freestream density [slug/ft^3]
        self.RHOINF = self.REY * self.MUINF / (self.UINF/12.0)
        # Freestream pressure
        self.PINF = self.RHOINF * self.R * self.TINF
        # Dynamic pressure
        self.QINF = 0.5*self.RHOINF * self.UINF**2
            
    # Get the number of grids
    def GetNGrid(self):
        """Read the number of grids and determine multiple grid status
        
        :Call:
            >>> nGrid = q.GetNGrid()
        :Inputs:
            *q*: :class:`pyOver.plot3d.Q`
                General OVERFLOW q-file interface
        :Outputs:
            *nGrid*: :class:`int`
                Number of grids
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Open the file
        self.open()
        # Read the first bit
        i0 = np.fromfile(self.f, count=1, dtype=self.itype)
        # If this is 4, it's a multiple grid
        if i0 == 4:
            # This is a multiple grid file (even if it has one grid)
            self.mGrid = True
            # Read the next file
            self.nGrid = self.read_int()
            # Read the next bit
            self.read_int()
        else:
            # This is a single-grid file
            self.mGrid = False
            # One grid
            self.nGrid = 1
            # Go back to the beginning
            self.f.seek(0)
        # Output
        return self.nGrid
        
    # Read the dimensions for each grid
    def GetGridDims(self):
        """Read the dimensions for each grid
        
        :Call:
            >>> q.GetGridDims()
        :Inputs:
            *q*: :class:`pyOver.plot3d.Q`
                General OVERFLOW q-file interface
        :Data members:
            *q.JD*: :class:`numpy.ndarray` (:class:`int` size=*q.nGrid*)
                *J* dimensions of each grid
            *q.KD*: :class:`numpy.ndarray` (:class:`int` size=*q.nGrid*)
                *K* dimensions of each grid
            *q.LD*: :class:`numpy.ndarray` (:class:`int` size=*q.nGrid*)
                *L* dimensions of each grid
            *q.NQ*: :class:`int`
                Number of conserved variables plus one for *gamma*
            *q.NQC*: :class:`int`
                Number of species concentrations, ``0`` if not using
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Read the start-of-record indicator
        i0 = self.read_int()
        # Number of integers to read
        ni = 3*self.nGrid + 2
        # Expected length
        i1 = 4*ni
        # Check the value
        if i0 != i1:
            # This should be 20, indicating 5 integers
            raise ValueError(
                ("Grid dimensions record starts with %i\n" % i0) +
                ("Expecting %i, which indicates 3*NGRID+2 entries." % i1))
        # Read the values
        D = self.read_int(ni)
        # Read end-of-record
        self.read_int()
        # Save dimensions of each grid
        self.JD = D[0:-2:3]
        self.KD = D[1:-2:3]
        self.LD = D[1:-2:3]
        # Save *NQ*, the number of conserved values
        self.NQ = D[-2]
        # Save *NQC*, the number of species
        self.NQC = D[-1]
        
    # Initialize grids
    def InitHeaders(self):
        """Initialize reference quantities for each grid
        
        :Call:
            >>> q.InitHeaders()
        :Inputs:
            *q*: :class:`pyOver.plot3d.Q`
                Generic OVERFLOW module
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Copy number of grids to save space
        nGrid = self.nGrid
        # Expected number of *RGAS* values
        nRGAS = max(2, self.NQC)
        # Initialize the values
        self._REFMACH = np.zeros(nGrid, dtype=self.ftype)
        self._ALPHA   = np.zeros(nGrid, dtype=self.ftype)
        self._REY     = np.zeros(nGrid, dtype=self.ftype)
        self._TIME    = np.zeros(nGrid, dtype=self.ftype)
        self._GAMINF  = np.zeros(nGrid, dtype=self.ftype)
        self._BETA    = np.zeros(nGrid, dtype=self.ftype)
        self._TINF    = np.zeros(nGrid, dtype=self.ftype)
        self._IGAMMA  = np.zeros(nGrid, dtype=self.itype)
        self._HTINF   = np.zeros(nGrid, dtype=self.ftype)
        self._HT1     = np.zeros(nGrid, dtype=self.ftype)
        self._HT2     = np.zeros(nGrid, dtype=self.ftype)
        self._RGAS    = np.zeros((nGrid, nRGAS), dtype=self.ftype)
        self._FSMACH  = np.zeros(nGrid, dtype=self.ftype)
        self._TVREF   = np.zeros(nGrid, dtype=self.ftype)
        self._DTVREF  = np.zeros(nGrid, dtype=self.ftype)
        
    # Read the header info
    def ReadQHeader(self, IG=None):
        """Read header info assuming the file marker is in the correct place
        
        :Call:
            >>> q.ReadQHeader(IG=None)
        :Inputs:
            *q*: :class:`pyOver.plot3d.Q`
                Generic OVERFLOW module
            *IG*: :class:`int`
                Grid number to read, defaults to ``len(q.Q)+1``
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Expected number of *RGAS* values
        nRGAS = max(2, self.NQC)
        # Expected number of entries before and after *IGAMMA*
        n1 = 7
        n2 = 6 + nRGAS
        # Expected number of bytes in the record
        i1 = (n1+n2)*8 + 4
        # Read start-of-record
        i0 = self.read_int()
        # Check it
        if i0 != i1:
            self.close()
            raise ValueError(
                "Header record contains %i bytes; expecting %i" % (i0, i1))
        # Read the first block of header data
        F1 = self.read_float(n1)
        # Read the IGAM setting
        self.IGAMMA = self.read_int()
        # Read the second block of header data
        F2 = self.read_float(n2)
        # Unpack the headers
        self.REFMACH = F1[0]
        self.ALPHA   = F1[1]
        self.REY     = F1[2]
        self.TIME    = F1[3]
        self.GAMINF  = F1[4]
        self.BETA    = F1[5]
        self.TINF    = F1[6]
        self.HTINF   = F2[0]
        self.HT1     = F2[1]
        self.HT2     = F2[2]
        self.RGAS    = F2[3:3+nRGAS]
        self.FSMACH  = F2[3+nRGAS]
        self.TVREF   = F2[4+nRGAS]
        self.DTVREF  = F2[5+nRGAS]
        # Save appropriately
        if IG is not None:
            # zero-based grid number
            ig = IG - 1
            # Save the reference quantities in the grid lists
            self._REFMACH[ig] = self.REFMACH
            self._ALPHA[ig]   = self.ALPHA
            self._REY[ig]     = self.REY
            self._TIME[ig]    = self.TIME
            self._GAMINF[ig]  = self.GAMINF
            self._BETA[ig]    = self.BETA
            self._TINF[ig]    = self.TINF
            self._IGAMMA[ig]  = self.IGAMMA
            self._HTINF[ig]   = self.HTINF
            self._HT1[ig]     = self.HT1
            self._HT2[ig]     = self.HT2
            self._RGAS[ig]    = self.RGAS
            self._FSMACH[ig]  = self.FSMACH
            self._TVREF[ig]   = self.TVREF
            self._DTVREF[ig]  = self.DTVREF
        # Read end-of-record
        self.read_int()
        
    # Read data and unpack it
    def ReadQData(self, IG=None):
        """Read the data
        
        :Call:
            >>> q.ReadQData(IG=None)
        :Inputs:
            *q*: :class:`pyOver.plot3d.Q`
                General OVERFLOW q-file interface
            *IG*: :class:`int`
                Grid number to read, defaults to ``len(q.Q)+1``
        :Data members:
            ``q.Q[IG-1]``: :class:`numpy.ndarray` (:class:`float`)
                Solution array
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Make sure there is a solution vector
        try:
            self.Q
        except AttributeError:
            self.Q = []
        # Process grid number
        if IG is None:
            # Use current length of Q
            IG = len(self.Q) + 1
            # Append to the solution list
            self.Q.append(None)
        elif IG > len(self.Q):
            # Append to the solution list
            self.Q += [None for i in range(len(self.Q),IG)]
        # Total number of states
        NQ = self.NQ + self.NQC
        # Get dimensions
        JD = self.JD[IG-1]
        KD = self.KD[IG-1]
        LD = self.LD[IG-1]
        # Total number of points
        NPT = JD*KD*LD
        # Expected length
        i1 = NPT * NQ * 8
        # Read record length
        i0 = self.read_int()
        # Check consistency
        if i0 != i1:
            self.close()
            raise ValueError(
                "Record for grid %i has incorrect length" % (IG))
        # Read the values
        qi = self.read_float(NPT*NQ)
        # Reshape and save
        self.Q[IG-1] = np.reshape(qi, (NQ, JD, KD, LD))
        # Read end-of-record
        self.read_int()
        
    # Extract CP
    def get_Cp(self, IG, **kw):
        """Get pressure coefficients from a grid
        
        :Call:
            >>> Cp = q.get_Cp(IG, **kw)
        :Inputs:
            *q*: :class:`pyOver.plot3d.Q`
                General OVERFLOW q-file interface
            *IG*: :class:`int`
                Grid number (one-based index)
        :Keyword arguments:
            *J*: :class:`int` | :class:`list` (:class:`int`) 
                Single grid index, *j* direction
            *JS*: :class:`int`
                Start index, *j* direction
            *JE*: :class:`int`
                End index, *j* direction
            *K*: :class:`int` | :class:`list` (:class:`int`)
                Single grid index, *k* direction
            *KS*: :class:`int`
                Start index, *k* direction
            *KE*: :class:`int`
                End index, *k* direction
            *L*: :class:`int` | :class:`list` (:class:`int`)
                Single grid index, *l* direction
            *LS*: :class:`int`
                Start index, *l* direction
            *LE*: :class:`int`
                End index, *l* direction
        :Outputs:
            *Cp*: :class:`float` | :class:`numpy.ndarray` (:class:`float`)
                Pressure coefficient or array of pressure coefficients
        :See also:
            :func:`expand_grid_indices`
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Expand grid indices
        J, K, L = self.expand_grid_indices(IG, **kw)
        # Extract freestream states
        M_inf = self.FSMACH
        g_inf = self.GAMINF
        # Extract the *q* grid
        Q = self.Q[IG-1]
        # Get normalized density and energy
        rhostar = Q[0,J,K,L]
        # Get the velocity components
        ustar = Q[1,J,K,L] / rhostar
        vstar = Q[2,J,K,L] / rhostar
        wstar = Q[3,J,K,L] / rhostar
        # Velocity
        U2star = ustar*ustar + vstar*vstar + wstar*wstar
        # Get internal energy
        estar = Q[4,J,K,L]/rhostar - 0.5*U2star
        # Ratios of specific heats
        gam = Q[5,J,K,L]
        # Non-dimensional pressures
        pstar = (gam-1)*rhostar*estar
        # Pressure coefficient
        Cp = (gam*pstar-1)/(0.5*g_inf*M_inf**2)
        # Output
        return Cp
        
    # Extract Mach number
    def get_M(self, IG, **kw):
        """Get Mach numbers from a grid
        
        :Call:
            >>> M = q.get_M(IG, **kw)
        :Inputs:
            *q*: :class:`pyOver.plot3d.Q`
                General OVERFLOW q-file interface
            *IG*: :class:`int`
                Grid number (one-based index)
        :Keyword arguments:
            *J*: :class:`int` | :class:`list` (:class:`int`) 
                Single grid index, *j* direction
            *JS*: :class:`int`
                Start index, *j* direction
            *JE*: :class:`int`
                End index, *j* direction
            *K*: :class:`int` | :class:`list` (:class:`int`)
                Single grid index, *k* direction
            *KS*: :class:`int`
                Start index, *k* direction
            *KE*: :class:`int`
                End index, *k* direction
            *L*: :class:`int` | :class:`list` (:class:`int`)
                Single grid index, *l* direction
            *LS*: :class:`int`
                Start index, *l* direction
            *LE*: :class:`int`
                End index, *l* direction
        :Outputs:
            *M*: :class:`float` | :class:`numpy.ndarray` (:class:`float`)
                Mach number or array of Mach numbers
        :See also:
            :func:`expand_grid_indices`
        :Versions:
            * 2016-03-07 ``@ddalle``: First version
        """
        # Expand grid indices
        J, K, L = self.expand_grid_indices(IG, **kw)
        # Extract freestream states
        M_inf = self.FSMACH
        g_inf = self.GAMINF
        # Extract the *q* grid
        Q = self.Q[IG-1]
        # Get normalized density and energy
        rhostar = Q[0,J,K,L]
        # Get the velocity components
        ustar = Q[1,J,K,L] / rhostar
        vstar = Q[2,J,K,L] / rhostar
        wstar = Q[3,J,K,L] / rhostar
        # Velocity
        U2star = ustar*ustar + vstar*vstar + wstar*wstar
        # Get internal energy
        estar = Q[4,J,K,L]/rhostar - 0.5*U2star
        # Ratios of specific heats
        gam = Q[5,J,K,L]
        # Non-dimensional pressures
        pstar = (gam-1)*rhostar*estar
        # Sound speed
        astar = np.sqrt(gam*pstar/rhostar)
        # Mach number
        M = np.sqrt(U2star) / astar
        # Output
        return M
        
    # Extract dimensional pressure
    def get_p(self, IG, **kw):
        """Get dimensional point pressures from a grid in lb/ft^2
        
        :Call:
            >>> p = q.get_p(IG, **kw)
        :Inputs:
            *q*: :class:`pyOver.plot3d.Q`
                General OVERFLOW q-file interface
            *IG*: :class:`int`
                Grid number (one-based index)
        :Keyword arguments:
            *J*: :class:`int` | :class:`list` (:class:`int`) 
                Single grid index, *j* direction
            *JS*: :class:`int`
                Start index, *j* direction
            *JE*: :class:`int`
                End index, *j* direction
            *K*: :class:`int` | :class:`list` (:class:`int`)
                Single grid index, *k* direction
            *KS*: :class:`int`
                Start index, *k* direction
            *KE*: :class:`int`
                End index, *k* direction
            *L*: :class:`int` | :class:`list` (:class:`int`)
                Single grid index, *l* direction
            *LS*: :class:`int`
                Start index, *l* direction
            *LE*: :class:`int`
                End index, *l* direction
        :Outputs:
            *p*: :class:`float` | :class:`numpy.ndarray` (:class:`float`)
                Pressure or array of pressures [psf]
        :See also:
            :func:`expand_grid_indices`
        :Versions:
            * 2016-03-07 ``@ddalle``: First version
        """
        # Expand grid indices
        J, K, L = self.expand_grid_indices(IG, **kw)
        # Extract freestream states
        M_inf = self.FSMACH
        g_inf = self.GAMINF
        p_inf = self.PINF
        # Extract the *q* grid
        Q = self.Q[IG-1]
        # Get normalized density and energy
        rhostar = Q[0,J,K,L]
        # Get the velocity components
        ustar = Q[1,J,K,L] / rhostar
        vstar = Q[2,J,K,L] / rhostar
        wstar = Q[3,J,K,L] / rhostar
        # Velocity
        U2star = ustar*ustar + vstar*vstar + wstar*wstar
        # Get internal energy
        estar = Q[4,J,K,L]/rhostar - 0.5*U2star
        # Ratios of specific heats
        gam = Q[5,J,K,L]
        # Non-dimensional pressures
        pstar = (gam-1)*rhostar*estar
        # Dimensional pressures
        return gam*p_inf*pstar
        
    # Extract dimensional temperature
    def get_T(self, IG, **kw):
        """Get dimensional point temperatures from a grid in degrees Rankine
        
        :Call:
            >>> p = q.get_p(IG, **kw)
        :Inputs:
            *q*: :class:`pyOver.plot3d.Q`
                General OVERFLOW q-file interface
            *IG*: :class:`int`
                Grid number (one-based index)
        :Keyword arguments:
            *J*: :class:`int` | :class:`list` (:class:`int`) 
                Single grid index, *j* direction
            *JS*: :class:`int`
                Start index, *j* direction
            *JE*: :class:`int`
                End index, *j* direction
            *K*: :class:`int` | :class:`list` (:class:`int`)
                Single grid index, *k* direction
            *KS*: :class:`int`
                Start index, *k* direction
            *KE*: :class:`int`
                End index, *k* direction
            *L*: :class:`int` | :class:`list` (:class:`int`)
                Single grid index, *l* direction
            *LS*: :class:`int`
                Start index, *l* direction
            *LE*: :class:`int`
                End index, *l* direction
        :Outputs:
            *T*: :class:`float` | :class:`numpy.ndarray` (:class:`float`)
                Temperature or array of temperatures [R]
        :See also:
            :func:`expand_grid_indices`
        :Versions:
            * 2016-03-07 ``@ddalle``: First version
        """
        # Expand grid indices
        J, K, L = self.expand_grid_indices(IG, **kw)
        # Extract freestream states
        M_inf = self.FSMACH
        g_inf = self.GAMINF
        T_inf = self.TINF
        # Extract the *q* grid
        Q = self.Q[IG-1]
        # Get normalized density and energy
        rhostar = Q[0,J,K,L]
        # Number of species
        NQC = max(1, self.NQC)
        # Check for nontrivial gas constant
        if NQC == 0:
            # Use the value
            Rstar = 1.0
        else:
            # Dimension of *Q*
            NQ = self.NQ
            # Relative values of gas constant
            RGAS = self.RGAS
            # Initialize gas constant
            Rstar = 0.0
            # Loop through species
            for i in range(NQC):
                Rstar += self.RGAS[i] * Q[NQ-NQC-1+i,J,K,L]/rhostar
        # Get the velocity components
        ustar = Q[1,J,K,L] / rhostar
        vstar = Q[2,J,K,L] / rhostar
        wstar = Q[3,J,K,L] / rhostar
        # Velocity
        U2star = ustar*ustar + vstar*vstar + wstar*wstar
        # Get internal energy
        estar = Q[4,J,K,L]/rhostar - 0.5*U2star
        # Temperature
        return estar
        # Ratios of specific heats
        gam = Q[5,J,K,L]
        # Non-dimensional pressures
        Tstar = (gam-1)*estar / Rstar
        # Dimensional pressures
        return g_inf*T_inf*Tstar            
        
# class Q

# OVERFLOW x class
class X(P3D):
    """
    General OVERFLOW ``x`` file interface
    
    :Call:
        >>> x = pyOver.plot3d.X(fname, endian=None)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *endian*: {``None``} | "big" | "little"
            Manually-specified byte order
    :Outputs:
        *x*: :class:`pyOver.plot3d.X`
            General OVERFLOW x-file interface
    :Versions:
        * 2016-08-31 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self, fname, endian=None):
        """Initialization method
        
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Save the file name
        self.fname = fname
        # File handle
        self.f = None
        # Determine endianness
        self.get_byteorder(endian)
        # Get flags
        self.get_dtypes()
        
        
        
        
        
# CLASS x
