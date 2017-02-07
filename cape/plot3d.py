#!/usr/bin/env python
"""
Python interface to Plot3D files
================================

:Versions:
    * 2016-02-26 ``@ddalle``: First version
"""

# System interface
import os
# Numerics
import numpy as np

# Input/output module
from . import io

# General Plot3D class...
class Plot3D(object):
    """General Plot3D class
    
    :Call:
        >>> q = Plot3D(fname, endian=None)
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
        *endian*: ``None`` | ``"big"`` | ``"little"``
            Manually-specified byte-order
    :Outputs:
        *q*: :class:`cape.plot3d.Plot3D`
            Plot3D interface
    :Versions:
        * 2016-02-26 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self, fname=None, X=None):
        """Initialization method
        
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Save file name
        self.fname = os.path.abspath(fname)
        # Endianness
        self.endian = endian
        # File handle
        self.f = None
        # Status
        self.closed = True
        # Save reasonable default data types
        self.itype = 'i4'
        self.ftype = 'f8'
    
    # Open the file
    def open(self, mode='rb'):
        """Open the file with the correct mode
        
        :Call:
            >>> q.open(mode='rb')
        :Inputs:
            *q*: :class:`cape.plot3d.Plot3D`
                Plot3D file interface
            *mode*: {'rb'} | 'wb' | 'rb+' | :class:`str`
                File mode to use
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Close the file if necessary
        self.close()
        # Open the file and save the file
        self.f = open(self.fname, mode)
        # Set status
        self.closed = False
        
    # Close the file
    def close(self):
        """Close the file if it is open
        
        :Call:
            >>> q.close()
        :Inputs:
            *q*: :class:`cape.plot3d.Plot3D`
                Plot3D file interface
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Set status
        self.closed = True
        # Check if file is open
        if type(self.f).__name__ != "file":
            # Not a file!
            return
        elif self.f.closed == True:
            # File already closed
            return
        else:
            # Close the file
            self.f.close()
            
    # Read a number of integers
    def read_int(self, count=1):
        """Read *count* integers using the correct data type
        
        :Call:
            >>> I = q.read_int(count=1)
        :Inputs:
            *q*: :class:`cape.plot3d.Plot3D`
                Plot3D file interface
            *count*: :class:`int`
                Number of entries to read
        :Outputs:
            *I*: :class:`numpy.ndarray` (dtype=:class:`int` size=*count*)
                Array of *count* integers
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Check file status
        if self.closed:
            return np.array([])
        # Read data
        return np.fromfile(self.f, count=count, dtype=self.itype)
            
    # Read a number of integers
    def read_float(self, count=1):
        """Read *count* floats using the correct data type
        
        :Call:
            >>> F = q.read_float(count=1)
        :Inputs:
            *q*: :class:`cape.plot3d.Plot3D`
                Plot3D file interface
            *count*: :class:`int`
                Number of entries to read
        :Outputs:
            *F*: :class:`numpy.ndarray` (dtype=:class:`infloat` size=*count*)
                Array of *count* floats
        :Versions:
            * 2016-02-26 ``@ddalle``: First version
        """
        # Check file status
        if self.closed:
            return np.array([])
        # Read data
        return np.fromfile(self.f, count=count, dtype=self.ftype)
# class Plot3D

# Plot3D Multiple-Grid file
class X(object):
    
    def __init__(self, fname=None, X=None):
        """Initialization method
        
        :Versions:
            * 2016-10-11 ``@ddalle``: First version
        """
        # Check for a file to read
        if fname is not None:
            self.Read(fname)
            
            
    def Read(self, fname, **kw):
        """Read a Plot3D grid file of any format
        
        :Call:
            >>> x.Read(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
        :Attributes:
            *x.x*: :class:`list` (:class:`np.ndarray`)
                List of x-coordinate arrays
            *x.y*: :class:`list` (:class:`np.ndarray`)
                List of y-coordinate arrays
            *x.z*: :class:`list` (:class:`np.ndarray`)
                List of z-coordinate arrays
        :Versions:
            * 2016-10-15 ``@ddalle``: First version
        """
        # Check for keywords
        if kw.get('lb8'):
            # Read as little-endian double
            self.Read_lb8(fname)
            return
        elif kw.get('b8'):
            # Read as big-endian double
            self.Read_b8(fname)
            return
        elif kw.get('lb4'):
            # Read as little-endian single
            self.Read_lb4(fname)
            return
        elif kw.get('b4'):
            # Read as big-endian single
            self.Read_b4(fname)
            return
        elif kw.get('ascii'):
            # Read as an ASCII file
            self.Read_ASCII(fname)
        # Get the basic file type
        ext = self.GetFileType(fname)
        # Check extension
        if ext == 'lb8':
            # Read as little-endian double
            self.Read_lb8(fname)
        elif ext == 'b8':
            # Read as big-endian double
            self.Read_b8(fname)
        elif ext == 'lb4':
            # Read as little-endian single
            self.Read_lb4(fname)
        elif ext == 'b4':
            # Read as big-endian single
            self.Read_b4(fname)
        elif ext == 'ascii':
            # Read as an ASCII file
            self.Read_ASCII(fname)
        
    # Determine file type blindly
    def GetFileType(self, fname):
        """Get full file type of a Plot3D grid file
        
        :Call:
            >>> ext = x.GetBasicFileType(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Outputs:
            *ext*: {``ascii``} | ``lb8`` | ``lb4`` | ``b8`` | ``b4``
                File type in the form of a file extension code
        :Attributes:
            *x.byteorder*: ``"little"`` | ``"big"`` | {``None``}
                Endianness of binary file
            *x.filetype*: ``"binary"`` | {``"ascii"``}
                Basic file type
            *x.p3dtype*: ``"multiple"`` | {``"single"``}
                Plot3D zone type
        :Versions:
            * 2016-10-14 ``@ddalle``: First version
        """
        # Get basic file type
        self.GetBasicFileType(fname)
        # Check file type
        if self.filetype == 'ascii':
            # This requires no action
            self.ext = 'ascii'
            return
        # Open the file
        f = open(fname, 'rb')
        # Check byte order
        if self.byteorder == 'little':
            # First part of extension
            ext = 'lb'
            # Check for number-of-grids-marker
            if self.p3dtype == 'multiple':
                # Read number of grids record
                ng = io.read_record_lb4_i(f)
            # Read first record
            dims = io.read_record_lb4_i(f)
            # Read the record marker
            r, = np.fromfile(f, count=1, dtype='<i4')
        else:
            # First part of extension
            ext = 'b'
            # Check for number-of-grids-marker
            if self.p3dtype == 'multiple':
                # Read number of grids record
                ng = io.read_record_b4_i(f)
            # Read first record
            dims = io.read_record_b4_i(f)
            # Ead record marker for first grid
            r, = np.fromfile(f, count=1, dtype='>i4')
        # Close the file
        f.close()
        # Number of points in the first grid
        # (Assume at least one grid)
        npt = dims[0] * dims[1] * dims[2]
        # Use the number of bytes in the record marker to check single/double
        if r/24 == npt:
            # Double-precision
            self.ext = ext + '8'
        elif r/12 == npt:
            # Single-precision
            self.ext = ext + '4'
        else:
            # Unusual number of bits per float
            nb = r / npt / 3 * 8
            raise ValueError("Found %i bits per float, must be 32 or 64" % nb)
        # Output
        return self.ext
    
    # Determine basic aspects of file type (do not determine single/double)
    def GetBasicFileType(self, fname):
        """Determine if a file is ASCII, little-endian, or big-endian
        
        Also determine if the file is single-zone or multiple-zone.  The
        function does not check for single-precision or double-precision
        
        :Call:
            >>> x.GetBasicFileType(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Attributes:
            *x.byteorder*: ``"little"`` | ``"big"`` | {``None``}
                Endianness of binary file
            *x.filetype*: ``"binary"`` | {``"ascii"``}
                Basic file type
            *x.p3dtype*: ``"multiple"`` | {``"single"``}
                Plot3D zone type
        :Versions:
            * 2016-10-14 ``@ddalle``: First version
        """
        # Open file
        f = open(fname, 'rb');
        # Read first record marker as little-endian
        r, = np.fromfile(f, count=1, dtype='<i4')
        # Check for success (or coherence)
        if r == 4:
            # Success; multiple zone
            self.byteorder = 'little'
            self.filetype = 'binary'
            self.p3dtype = 'multiple'
            f.close()
            return 
        # Try to read of single zone
        f.seek(r, 1)
        # Try to read end-of-record
        R = np.fromfile(f, count=1, dtype='<i4')
        # Check it
        if len(R) == 1 and R[0] == r:
            # Little-endian single-zone
            self.byteorder = 'little'
            self.filetye = 'binary'
            self.p3dtype = 'single'
            f.close()
            return
        # Return to the original location
        f.seek(0)
        # Read first record marker as big-endian
        r, = np.fromfile(f, count=1, dtype='>i4')
        # Check for success
        if r == 4:
            # Success; multiple zone
            self.byteorder = 'big'
            self.filetype = 'binary'
            self.p3dtype = 'multiple'
            f.close()
            return 
        # Try to read of single zone
        f.seek(r, 1)
        # Try to read end-of-record
        R = np.fromfile(f, count=1, dtype='>i4')
        # Check it
        if len(R) == 1 and R[0] == r:
            # Little-endian single-zone
            self.byteorder = 'big'
            self.filetye = 'binary'
            self.p3dtype = 'single'
            f.close()
            return
        # Apparently it's ASCII
        f.close()
        f = open(fname, 'r')
        # Read first line
        line = f.readline()
        # Check if it's a single entry
        if len(line.split()) == 1:
            # Multiple-zone
            self.filetype = 'ascii'
            self.p3dtype = 'multiple'
        else:
            # Single-zone
            self.filetype = 'ascii'
            self.p3dtype = 'single'
        f.close()
    
    # Read big-endian double-precision
    def Read_b8(self, fname):
        """Read a Plot3D grid as a big-endian double-precision file
        
        :Call:
            >>> x.Read_b8(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first record
        R = io.read_record_b4_i(f)
        # Check for single zone or multiple zone
        if len(R) == 1:
            # Number of zones
            self.NG = R[0]
            # Read dimensions
            R = io.read_record_b4_i(f)
        else:
            # Single zone
            self.NG = 1
        # Resize dimensions
        self.dims = np.resize(R, (self.NG, 3))
        self.NJ = self.dims[:,0]
        self.NK = self.dims[:,1]
        self.NL = self.dims[:,2]
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.append([0], np.cumsum(npt))
        # Initialize coordinates
        self.X = np.zeros((3,mpt[-1]))
        # Loop through the grids
        for i in range(self.NG):
            # Read record
            R = io.read_record_b8_f(f)
            # Save coordinates
            self.X[:,mpt[i]:mpt[i+1]] = np.reshape(R, (3,npt[i]))
        # Close the file
        f.close()
    
    # Read big-endian single-precision
    def Read_b4(self, fname):
        """Read a Plot3D grid as a big-endian single-precision file
        
        :Call:
            >>> x.Read_b4(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first record
        R = io.read_record_b4_i(f)
        # Check for single zone or multiple zone
        if len(R) == 1:
            # Number of zones
            self.NG = R[0]
            # Read dimensions
            R = io.read_record_b4_i(f)
        else:
            # Single zone
            self.NG = 1
        # Resize dimensions
        self.dims = np.resize(R, (self.NG, 3))
        self.NJ = self.dims[:,0]
        self.NK = self.dims[:,1]
        self.NL = self.dims[:,2]
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.append([0], np.cumsum(npt))
        # Initialize coordinates
        self.X = np.zeros((3,mpt[-1]))
        # Loop through the grids
        for i in range(self.NG):
            # Read record
            R = io.read_record_b4_f(f)
            # Save coordinates
            self.X[:,mpt[i]:mpt[i+1]] = np.reshape(R, (3,npt[i]))
        # Close the file
        f.close()
    
    # Read little-endian double-precision
    def Read_lb8(self, fname):
        """Read a Plot3D grid as a little-endian double-precision file
        
        :Call:
            >>> x.Read_lb8(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first record
        R = io.read_record_lb4_i(f)
        # Check for single zone or multiple zone
        if len(R) == 1:
            # Number of zones
            self.NG = R[0]
            # Read dimensions
            R = io.read_record_lb4_i(f)
        else:
            # Single zone
            self.NG = 1
        # Resize dimensions
        self.dims = np.resize(R, (self.NG, 3))
        self.NJ = self.dims[:,0]
        self.NK = self.dims[:,1]
        self.NL = self.dims[:,2]
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.append([0], np.cumsum(npt))
        # Initialize coordinates
        self.X = np.zeros((3,mpt[-1]))
        # Loop through the grids
        for i in range(self.NG):
            # Read record
            R = io.read_record_lb8_f(f)
            # Save coordinates
            self.X[:,mpt[i]:mpt[i+1]] = np.reshape(R, (3,npt[i]))
        # Close the file
        f.close()
    
    # Read little-endian single-precision
    def Read_lb4(self, fname):
        """Read a Plot3D grid as a little-endian single-precision file
        
        :Call:
            >>> x.Read_lb4(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first record
        R = io.read_record_lb4_i(f)
        # Check for single zone or multiple zone
        if len(R) == 1:
            # Number of zones
            self.NG = R[0]
            # Read dimensions
            R = io.read_record_lb4_i(f)
        else:
            # Single zone
            self.NG = 1
        # Resize dimensions
        self.dims = np.resize(R, (self.NG, 3))
        self.NJ = self.dims[:,0]
        self.NK = self.dims[:,1]
        self.NL = self.dims[:,2]
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.append([0], np.cumsum(npt))
        # Initialize coordinates
        self.X = np.zeros((3,mpt[-1]))
        # Loop through the grids
        for i in range(self.NG):
            # Read record
            R = io.read_record_lb8_f(f)
            # Save coordinates
            self.X[:,mpt[i]:mpt[i+1]] = np.reshape(R, (3,npt[i]))
        # Close the file
        f.close()
    
    # Read as an ascii file
    def Read_ASCII(self, fname):
        """Read a Plot3D grid as an ASCII file
        
        :Call:
            >>> x.Read_ASCII(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'r')
        # Read the first line
        v = [int(vi) for vi in f.readline().split()]
        # Get the number of elements
        if len(v) == 1:
            # Number of zones
            self.NG = int(v)
            # Read dimensions line
            v = [int(vi) for vi in f.readline().split()]
        else:
            # One zone
            self.NG = 1
        # Initialize grid dimensions
        self.dims = np.ones((self.NG,3), dtype=int)
        # Set first grid
        self.dims[0] = v
        # Loop through remaining grids (if any)
        for i in range(1, self.NG):
            self.dims[i] = np.fromfile(f, sep=" ", count=3, dtype='int')
        # Extract useful parameters
        self.NJ = self.dims[:,0]
        self.NK = self.dims[:,1]
        self.NL = self.dims[:,2]
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.append([0], np.cumsum(npt))
        # Initialize coordinates
        self.X = np.zeros((3,mpt[-1]))
        # Read the grids
        for i in range(self.NG):
            # Global point indices
            ia = mpt[i]
            ib = mpt[i+1]
            ni = npt[i]
            # Read x,y,z coordinates
            self.X[0,ia:ib] = np.fromfile(f, sep=" ", count=ni, dtype='float')
            self.X[1,ia:ib] = np.fromfile(f, sep=" ", count=ni, dtype='float')
            self.X[2,ia:ib] = np.fromfile(f, sep=" ", count=ni, dtype='float')
        # Close the file
        f.close()
    
    # Write as an ASCII file
    def Write_ASCII(self, fname, single=False):
        """Write a multiple-zone ASCII Plot3D file
        
        :Call:
            >>> x.Write_ASCII(fname, single=False)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
            *single*: ``True`` | {``False``}
                If ``True``, write a single-zone file
        :Versions:
            * 2016-10-16 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'w')
        # Check for single grid
        if not (single and self.NG == 1):
            # Write the number of zones
            f.write('%s\n' % self.NG)
        # Write the dimensions of each grid
        for i in range(self.NG):
            # Write NJ, NK, NL
            self.dims[i].tofile(f, sep=' ')
            f.write('\n')
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.append([0], np.cumsum(npt))
        # Loop through the grids to write the nodes
        for i in range(self.NG):
            # Point indices
            ia = mpt[i]
            ib = mpt[i+1]
            # Write the coordinates of grid *i*
            self.X[0,ia:ib].tofile(f, sep=' ')
            f.write('\n')
            self.X[1,ia:ib].tofile(f, sep=' ')
            f.write('\n')
            self.X[2,ia:ib].tofile(f, sep=' ')
            f.write('\n')
        # Close the file.
        f.close()
            
    # Write as a little-endian double-precision file
    def Write_lb8(self, fname, single=False):
        """Write a multiple-zone little-endian, double-precision Plot3D file
        
        :Call:
            >>> x.Write_lb8(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
            *single*: ``True`` | {``False``}
                If ``True``, write a single-zone file
        :Versions:
            * 2016-10-16 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'wb')
        # Check for single grid
        if not(single and self.NG == 1):
            # Write the number of zones
            io.write_record_lb4_i(f, self.NG)
        # Write the dimensions
        io.write_record_lb4_i(f, self.dims)
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.append([0], np.cumsum(npt))
        # Write the coordinates
        for i in range(self.NG):
            # Point indices
            ia = mpt[i]
            ib = mpt[i+1]
            # Put all three coordinates 
            io.write_record_lb8_f(f, self.X[:,ia:ib])
        # Close the file
        f.close()
            
    # Write as a little-endian single-precision file
    def Write_lb4(self, fname, single=False):
        """Write a multiple-zone little-endian, single-precision Plot3D file
        
        :Call:
            >>> x.Write_lb4(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
            *single*: ``True`` | {``False``}
                If ``True``, write a single-zone file
        :Versions:
            * 2016-10-16 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'wb')
        # Check for single grid
        if not(single and self.NG == 1):
            # Write the number of zones
            io.write_record_lb4_i(f, self.NG)
        # Write the dimensions
        io.write_record_lb4_i(f, self.dims)
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.append([0], np.cumsum(npt))
        # Write the coordinates
        for i in range(self.NG):
            # Point indices
            ia = mpt[i]
            ib = mpt[i+1]
            # Put all three coordinates 
            io.write_record_lb8_f(f, self.X[:,ia:ib])
        # Close the file
        f.close()
            
    # Write as a big-endian double-precision file
    def Write_b8(self, fname, single=False):
        """Write a multiple-zone little-endian, double-precision Plot3D file
        
        :Call:
            >>> x.Write_b8(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
            *single*: ``True`` | {``False``}
                If ``True``, write a single-zone file
        :Versions:
            * 2016-10-16 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'wb')
        # Check for single grid
        if not(single and self.NG == 1):
            # Write the number of zones
            io.write_record_b4_i(f, self.NG)
        # Write the dimensions
        io.write_record_b4_i(f, self.dims)
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.append([0], np.cumsum(npt))
        # Write the coordinates
        for i in range(self.NG):
            # Point indices
            ia = mpt[i]
            ib = mpt[i+1]
            # Put all three coordinates 
            io.write_record_b8_f(f, self.X[:,ia:ib])
        # Close the file
        f.close()
            
    # Write as a big-endian single-precision file
    def Write_b4(self, fname, single=False):
        """Write a multiple-zone little-endian, single-precision Plot3D file
        
        :Call:
            >>> x.Write_b4(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
            *single*: ``True`` | {``False``}
                If ``True``, write a single-zone file
        :Versions:
            * 2016-10-16 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'wb')
        # Check for single grid
        if not(single and self.NG == 1):
            # Write the number of zones
            io.write_record_b4_i(f, self.NG)
        # Write the dimensions
        io.write_record_b4_i(f, self.dims)
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.append([0], np.cumsum(npt))
        # Write the coordinates
        for i in range(self.NG):
            # Point indices
            ia = mpt[i]
            ib = mpt[i+1]
            # Put all three coordinates 
            io.write_record_b8_f(f, self.X[:,ia:ib])
        # Close the file
        f.close()
    
    
# class X
