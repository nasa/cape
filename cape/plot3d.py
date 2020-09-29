#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:class:`cape.plot3d`: Python interface to Plot3D files
========================================================

This module provides a generic Plot3D file interface for reading Plot3D
grid files using the class :class:`cape.plot3d.X`.  This class
automatically detects endianness of the grid and can handle IBLANKS in
addition to single-grid or multiple-grid formats.

The :class:`cape.plot3d.Q` interface to solution files also exists, but
it is not reliable since Plot3D solution files are dependent on the
solver used to create the solution file.

"""

# Standard library
import os

# Third-party modules
import numpy as np

# Local modules
from . import io
from . import util
from . import tri
from .filecntl import namelist2


# Default tolerances for mapping triangulations
atoldef = 3e-2
rtoldef = 1e-4
ctoldef = 1e-3
antoldef = 2e-2
rntoldef = 1e-4
cntoldef = 1e-3
aftoldef = 1e-3
rftoldef = 1e-6
cftoldef = 1e-3
anftoldef = 1e-3
rnftoldef = 1e-6
cnftoldef = 1e-3


# Plot3D Multiple-Grid file
class X(object):
  # ========
  # Config
  # ========
  # <
    # Initialization method
    def __init__(self, fname=None):
        r"""Initialization method
        
        :Call:
            >>> x = X(fname=None)
        :Inputs:
            *fname*: :class:`str`
                Name of Plot3D grid file to read
        :Versions:
            * 2016-10-11 ``@ddalle``: Version 1.0
        """
        # Check for a file to read
        if fname is not None:
            self.Read(fname)
    
    # Display method
    def __repr__(self):
        r"""Representation method
        
        :Versions:
            * 2018-01-11 ``@ddalle``: Version 1.0
        """
        return "<plot3d.X NG=%s>" % self.NG
    
    # Display method
    def __str__(self):
        r"""Representation method
        
        :Versions:
            * 2018-01-11 ``@ddalle``: Version 1.0
        """
        return "<plot3d.X NG=%s>" % self.NG
  # >
  
  
  # =======
  # Readers
  # =======
  # <
    def Read(self, fname, **kw):
        r"""Read a Plot3D grid file of any format
        
        :Call:
            >>> x.Read(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
        :Attributes:
            *x.X*: :class:`np.ndarray` (:class:`float` shape=(N,3))
                Array of coordinates of all points in the grid
            *x.NG*: :class:`int`
                Number of grids in the file
            *x.NJ*: :class:`np.ndarray`\ [:class:`int`] *shape*: *x.nG*,
                *J*-dimension of each grid
            *x.NK*: :class:`np.ndarray`\ [:class:`int`] *shape*: *x.nG*,
                *K*-dimension of each grid
            *x.NL*: :class:`np.ndarray`\ [:class:`int`] *shape*: *x.nG*,
                *L*-dimension of each grid
        :Versions:
            * 2016-10-15 ``@ddalle``: Version 1.0
            * 2017-02-07 ``@ddalle``: Version 1.1, updated doc
        """
        # Check for keywords
        if kw.get('lb8'):
            # Read as little-endian double stream
            self.Read_lb8(fname)
            return
        elif kw.get('b8'):
            # Read as big-endian double stream
            self.Read_b8(fname)
            return
        elif kw.get('lb4'):
            # Read as little-endian single stream
            self.Read_lb4(fname)
            return
        elif kw.get('b4'):
            # Read as big-endian single stream
            self.Read_b4(fname)
            return
        elif kw.get('lr8'):
            # Read as little-endian double record
            self.Read_lr8(fname)
            return
        elif kw.get('r8'):
            # Read as big-endian double record
            self.Read_r8(fname)
            return
        elif kw.get('lr4'):
            # Read as little-endian single record
            self.Read_lr4(fname)
            return
        elif kw.get('r4'):
            # Read as big-endian single record
            self.Read_r4(fname)
            return
        elif kw.get('ascii'):
            # Read as an ASCII file
            self.Read_ASCII(fname)
            return
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
        elif ext == 'lr8':
            # Read as little-endian double
            self.Read_lr8(fname)
        elif ext == 'r8':
            # Read as big-endian double
            self.Read_r8(fname)
        elif ext == 'lr4':
            # Read as little-endian single
            self.Read_lr4(fname)
        elif ext == 'r4':
            # Read as big-endian single
            self.Read_r4(fname)
        elif ext == 'ascii':
            # Read as an ASCII file
            self.Read_ASCII(fname)
        
    # Determine file type blindly
    def GetFileType(self, fname):
        r"""Get full file type of a Plot3D grid file
        
        :Call:
            >>> ext = x.GetBasicFileType(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Outputs:
            *ext*: {ascii} | lb8 | lb4 | b8 | b4 | lr4
                File type in the form of a file extension code
        :Attributes:
            *x.byteorder*: ``"little"`` | ``"big"`` | {``None``}
                Endianness of binary file
            *x.filetype*: ``"record"`` | ``"stream"`` | {``"ascii"``}
                Basic file type
            *x.p3dtype*: ``"multiple"`` | {``"single"``}
                Plot3D zone type
        :Versions:
            * 2016-10-14 ``@ddalle``: Version 1.0
        """
        # Get basic file type
        self.GetBasicFileType(fname)
        # Check file type
        if self.filetype == 'ascii':
            # This requires no action
            self.ext = 'ascii'
            return self.ext
        # Open the file
        f = open(fname, 'rb')
        # Check byte order
        if (self.byteorder == 'little') and (self.filetype == "record"):
            # Little-endian with records
            ext = 'lr'
            # Check for number-of-grids-marker
            if self.p3dtype == 'multiple':
                # Read number of grids record
                ng = io.read_record_lr4_i(f)
            # Read first record
            dims = io.read_record_lr4_i(f)
            # Read the record marker
            r, = np.fromfile(f, count=1, dtype='<i4')
        elif (self.byteorder == 'big') and (self.filetype == "record"):
            # Big-endian records
            ext = 'r'
            # Check for number-of-grids-marker
            if self.p3dtype == 'multiple':
                # Read number of grids record
                ng = io.read_record_r4_i(f)
            # Read first record
            dims = io.read_record_r4_i(f)
            # Read record marker for first grid
            r, = np.fromfile(f, count=1, dtype='>i4')
        elif (self.byteorder == "little"):
            # Little-endian stream
            ext = 'lb'
            # Check for number-of-grids type
            if self.p3dtype == "multiple":
                # Read the number of grids
                ng = io.fromfile_lb4_i(f, 1)
                # Read the dimensions
                dims = io.fromfile_lb4_i(f, ng*3).reshape((ng[0],3))
            else:
                # Read the dimensions
                dims = io.fromfile_lb4_i(f, 3).reshape((1,3))
            # Total number of points
            npt = np.sum(np.prod(dims, axis=1))
            # Number of bytes in file if single-precision
            nb4 = f.tell() + npt*12
        else:
            # Big-endian stream
            ext = 'b'
            # Check for number-of-grids type
            if self.p3dtype == "multiple":
                # Read the number of grids
                ng = io.fromfile_b4_i(f, 1)
                # Read the dimensions
                dims = io.fromfile_b4_i(f, ng*3).reshape((ng[0],3))
            else:
                # Read the dimensions
                dims = io.fromfile_b4_i(f, 3).reshape((1,3))
        # Try to determine single/double precision
        if (self.filetype == "record"):
            # Number of points in the first grid
            # (Assume at least one grid)
            npt = dims[0] * dims[1] * dims[2]
            # Use the number of bytes in the record marker to check
            if r/24 == npt:
                # Double-precision
                self.ext = ext + '8'
                # No iblanks
                self.iblank = False
            elif r/12 == npt:
                # Single-precision
                self.ext = ext + '4'
                # No iblanks
                self.iblank = False
            elif r/28 == npt:
                # Double-precision
                self.ext = ext + '8'
                # With iblanks
                self.iblank = True
            elif r/16 == npt:
                # Single-precision
                self.ext = ext + '4'
                # With iblanks
                self.iblank = True
            else:
                raise ValueError("Could not determine precision of" +
                    ("%s file" % ext))
        else:
            # Total number of points
            npt = np.sum(np.prod(dims, axis=1))
            # Number of bytes in file if single-precision
            nb4 = f.tell() + npt*12
            # Go to the end of the file
            f.seek(0, 2)
            # Check current position
            if nb4 == f.tell():
                # Single precision
                self.ext = ext + '4'
            else:
                # Double
                self.ext = ext + '8'
        # Close the file
        f.close()
        
        # Output
        return self.ext
    
    # Determine basic aspects of file type (do not determine single/double)
    def GetBasicFileType(self, fname):
        r"""Determine if a file is ASCII, little-endian, or big-endian
        
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
            *x.filetype*: ``"record"`` | ``"stream"`` | {``"ascii"``}
                Basic file type
            *x.p3dtype*: {``"multiple"``} | ``"single"``
                Plot3D zone type
        :Versions:
            * 2016-10-14 ``@ddalle``: Version 1.0
        """
        # Open file
        f = open(fname, 'rb');
        # Try to interpret the first record
        qr4  = io.check_record_r4(f)
        qlr4 = io.check_record_lr4(f)
        # Read the first four bytes using big/little
        r4,  = np.fromfile(f, count=1, dtype=">i4"); f.seek(0)
        lr4, = np.fromfile(f, count=1, dtype="<i4"); f.seek(0)
        # Process apparent little-endian record
        if qlr4:
            # Actually read the record
            R = io.read_record_lr4_i(f)
            # Try the next record
            if io.check_record_lr4(f):
                # Check the length of the first record
                if (R.size == 1):
                    # Valid multiple grid file
                    self.byteorder = 'little'
                    self.filetype  = 'record'
                    self.p3dtype   = 'multiple'
                else:
                    # Valid single-grid file
                    self.byteorder = 'little'
                    self.filetype  = 'record'
                    self.p3dtype   = 'single'
            else:
                # Odd coincidence of the first record
                self.byteorder = 'little'
                self.filetype  = 'stream'
                self.p3dtype   = 'multiple'
            # Close the file
            f.close()
            return
        elif qr4:
            # Actually read the record
            R = io.read_record_r4_i(f)
            # Try the next record
            if io.check_record_r4(f):
                # Check the length of the first record
                if (R.size == 1):
                    # Valid multiple grid file
                    self.byteorder = 'big'
                    self.filetype  = 'record'
                    self.p3dtype   = 'multiple'
                else:
                    # Valid single-grid file
                    self.byteorder = 'big'
                    self.filetype  = 'record'
                    self.p3dtype   = 'single'
            else:
                # Odd coincidence of the first record
                self.byteorder = 'big'
                self.filetype  = 'stream'
                self.p3dtype   = 'multiple'
            # Close the file
            f.close()
            return
        elif (r4 > 0) and (r4 < 1e6):
            # Appears to be a big-endian stream file
            self.byteorder = 'big'
            self.filetype  = 'stream'
            self.p3dtype   = 'multiple'
            # Close the file
            f.close()
            return
        elif (lr4 > 0) and (lr4 < 1e6):
            # Appears to be a little-endian stream file
            self.byteorder = 'little'
            self.filetype  = 'stream'
            self.p3dtype   = 'multiple'
            # Close the file
            f.close()
            return
        # Apparently it's ASCII
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
        r"""Read Plot3D grid big-endian double-precision file (stream)
        
        :Call:
            >>> x.Read_b8(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first line
        self.NG, = io.fromfile_b4_i(f, 1)
        # Read the dimensions
        R = io.fromfile_b4_i(f, self.NG*3)
        # Resize dimensions
        self.dims = np.resize(R, (self.NG, 3))
        self.NJ = self.dims[:,0]
        self.NK = self.dims[:,1]
        self.NL = self.dims[:,2]
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.sum(npt)
        # Initialize coordinates
        self.X = np.zeros((3,mpt))
        # Read coordinates
        R = io.fromfile_b8_f(f, mpt*3)
        # Save coordinates
        self.X = R.reshape((mpt,3))
        # Close the file
        f.close()
    
    # Read big-endian single-precision
    def Read_b4(self, fname):
        r"""Read a Plot3D grid as a big-endian single-precision file
        
        :Call:
            >>> x.Read_b4(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first line
        self.NG, = io.fromfile_b4_i(f, 1)
        # Read the dimensions
        R = io.fromfile_b4_i(f, self.NG*3)
        # Resize dimensions
        self.dims = np.resize(R, (self.NG, 3))
        self.NJ = self.dims[:,0]
        self.NK = self.dims[:,1]
        self.NL = self.dims[:,2]
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.sum(npt)
        # Initialize coordinates
        self.X = np.zeros((3,mpt))
        # Read coordinates
        R = io.fromfile_b4_f(f, mpt*3)
        # Save coordinates
        self.X = R.reshape((mpt,3))
        # Close the file
        f.close()
    
    # Read little-endian double-precision
    def Read_lb8(self, fname):
        r"""Read a Plot3D grid as a little-endian double-precision file
        
        :Call:
            >>> x.Read_lb8(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first line
        self.NG, = io.fromfile_lb4_i(f, 1)
        # Read the dimensions
        R = io.fromfile_lb4_i(f, self.NG*3)
        # Resize dimensions
        self.dims = np.resize(R, (self.NG, 3))
        self.NJ = self.dims[:,0]
        self.NK = self.dims[:,1]
        self.NL = self.dims[:,2]
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.sum(npt)
        # Initialize coordinates
        self.X = np.zeros((3,mpt))
        # Read coordinates
        R = io.fromfile_lb8_f(f, mpt*3)
        # Save coordinates
        self.X = R.reshape((mpt,3))
        # Close the file
        f.close()
    
    # Read little-endian single-precision
    def Read_lb4(self, fname):
        r"""Read a Plot3D grid as a little-endian single-precision file
        
        :Call:
            >>> x.Read_lb4(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first line
        self.NG, = io.fromfile_lb4_i(f, 1)
        # Read the dimensions
        R = io.fromfile_lb4_i(f, self.NG*3)
        # Resize dimensions
        self.dims = np.resize(R, (self.NG, 3))
        self.NJ = self.dims[:,0]
        self.NK = self.dims[:,1]
        self.NL = self.dims[:,2]
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.sum(npt)
        # Initialize coordinates
        self.X = np.zeros((3,mpt))
        # Read coordinates
        R = io.fromfile_lb4_f(f, mpt*3)
        # Save coordinates
        self.X = R.reshape((mpt,3))
        # Close the file
        f.close()
    
    # Read big-endian double-precision
    def Read_r8(self, fname):
        r"""Read a Plot3D grid as a big-endian double-precision file
        
        :Call:
            >>> x.Read_r8(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first record
        R = io.read_record_r4_i(f)
        # Check for single zone or multiple zone
        if len(R) == 1:
            # Number of zones
            self.NG = R[0]
            # Read dimensions
            R = io.read_record_r4_i(f)
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
        # Initialize IBLANK
        if self.iblank:
            self.IB = np.zeros(mpt[-1], dtype="bool")
        # Loop through the grids
        for i in range(self.NG):
            # Read record marker
            r1, = np.fromfile(f, count=1, dtype=">i4")
            # Number of points
            jpt = npt[i]
            # Read points
            R = np.fromfile(f, count=jpt*3, dtype=">f8")
            # Save coordinates
            self.X[:,mpt[i]:mpt[i+1]] = np.reshape(R, (3,jpt))
            # Check for iblanks
            if self.iblank:
                # Read IBlanks
                IB = np.fromfile(f, count=jpt, dtype=">i4")
                # Save them
                self.IB[mpt[i]:mpt[i+1]] = IB 
            # Read end-of-record marker
            r2, = np.fromfile(f, count=1, dtype=">i4")
            # Check consistency
            if r1 != r2:
                raise ValueError("End-of-record marker does not match start")
        # Close the file
        f.close()
    
    # Read big-endian single-precision
    def Read_r4(self, fname):
        r"""Read a Plot3D grid as a big-endian single-precision file
        
        :Call:
            >>> x.Read_r4(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first record
        R = io.read_record_r4_i(f)
        # Check for single zone or multiple zone
        if len(R) == 1:
            # Number of zones
            self.NG = R[0]
            # Read dimensions
            R = io.read_record_r4_i(f)
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
        # Initialize IBLANK
        if self.iblank:
            self.IB = np.zeros(mpt[-1], dtype="bool")
        # Loop through the grids
        for i in range(self.NG):
            # Read record marker
            r1, = np.fromfile(f, count=1, dtype=">i4")
            # Number of points
            jpt = npt[i]
            # Read points
            R = np.fromfile(f, count=jpt*3, dtype=">f4")
            # Save coordinates
            self.X[:,mpt[i]:mpt[i+1]] = np.reshape(R, (3,jpt))
            # Check for iblanks
            if self.iblank:
                # Read IBlanks
                IB = np.fromfile(f, count=jpt, dtype=">i4")
                # Save them
                self.IB[mpt[i]:mpt[i+1]] = IB 
            # Read end-of-record marker
            r2, = np.fromfile(f, count=1, dtype=">i4")
            # Check consistency
            if r1 != r2:
                raise ValueError("End-of-record marker does not match start")
        # Close the file
        f.close()
    
    # Read little-endian double-precision
    def Read_lr8(self, fname):
        r"""Read a Plot3D grid as a little-endian double-precision file
        
        :Call:
            >>> x.Read_lr8(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first record
        R = io.read_record_lr4_i(f)
        # Check for single zone or multiple zone
        if len(R) == 1:
            # Number of zones
            self.NG = R[0]
            # Read dimensions
            R = io.read_record_lr4_i(f)
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
        # Initialize IBLANK
        if self.iblank:
            self.IB = np.zeros(mpt[-1], dtype="bool")
        # Loop through the grids
        for i in range(self.NG):
            # Read record marker
            r1, = np.fromfile(f, count=1, dtype="<i4")
            # Number of points
            jpt = npt[i]
            # Read points
            R = np.fromfile(f, count=jpt*3, dtype="<f8")
            # Save coordinates
            self.X[:,mpt[i]:mpt[i+1]] = np.reshape(R, (3,jpt))
            # Check for iblanks
            if self.iblank:
                # Read IBlanks
                IB = np.fromfile(f, count=jpt, dtype="<i4")
                # Save them
                self.IB[mpt[i]:mpt[i+1]] = IB 
            # Read end-of-record marker
            r2, = np.fromfile(f, count=1, dtype="<i4")
            # Check consistency
            if r1 != r2:
                raise ValueError("End-of-record marker does not match start")
        # Close the file
        f.close()
    
    # Read little-endian single-precision
    def Read_lr4(self, fname):
        r"""Read a Plot3D grid as a little-endian single-precision file
        
        :Call:
            >>> x.Read_lr4(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first record
        R = io.read_record_lr4_i(f)
        # Check for single zone or multiple zone
        if len(R) == 1:
            # Number of zones
            self.NG = R[0]
            # Read dimensions
            R = io.read_record_lr4_i(f)
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
        # Initialize IBLANK
        if self.iblank:
            self.IB = np.zeros(mpt[-1], dtype="bool")
        # Loop through the grids
        for i in range(self.NG):
            # Read record marker
            r1, = np.fromfile(f, count=1, dtype="<i4")
            # Number of points
            jpt = npt[i]
            # Read points
            R = np.fromfile(f, count=jpt*3, dtype="<f4")
            # Save coordinates
            self.X[:,mpt[i]:mpt[i+1]] = np.reshape(R, (3,jpt))
            # Check for iblanks
            if self.iblank:
                # Read IBlanks
                IB = np.fromfile(f, count=jpt, dtype="<i4")
                # Save them
                self.IB[mpt[i]:mpt[i+1]] = IB 
            # Read end-of-record marker
            r2, = np.fromfile(f, count=1, dtype="<i4")
            # Check consistency
            if r1 != r2:
                raise ValueError("End-of-record marker does not match start")
        # Close the file
        f.close()
    
    # Read as an ascii file
    def Read_ASCII(self, fname):
        r"""Read a Plot3D grid as an ASCII file
        
        :Call:
            >>> x.Read_ASCII(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'r')
        # Read the first line
        v = [int(vi) for vi in f.readline().split()]
        # Get the number of elements
        if len(v) == 1:
            # Number of zones
            self.NG = int(v[0])
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
  # >
  
  # =======
  # Writers
  # =======
  # <
    # Write as an ASCII file
    def Write_ASCII(self, fname, single=False):
        r"""Write a multiple-zone ASCII Plot3D file
        
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
            * 2016-10-16 ``@ddalle``: Version 1.0
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
        r"""Write multi-zone little-endian, double-precision Plot3D file
        
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
            * 2016-10-16 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'wb')
        # Check for single grid
        if not(single and self.NG == 1):
            # Write the number of zones
            io.write_record_lr4_i(f, self.NG)
        # Write the dimensions
        io.write_record_lr4_i(f, self.dims)
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.append([0], np.cumsum(npt))
        # Write the coordinates
        for i in range(self.NG):
            # Point indices
            ia = mpt[i]
            ib = mpt[i+1]
            # Put all three coordinates 
            io.write_record_lr8_f(f, self.X[:,ia:ib])
        # Close the file
        f.close()
            
    # Write as a little-endian single-precision file
    def Write_lb4(self, fname, single=False):
        r"""Write multi-zone little-endian, single-precision Plot3D file
        
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
            * 2016-10-16 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'wb')
        # Check for single grid
        if not(single and self.NG == 1):
            # Write the number of zones
            io.write_record_lr4_i(f, self.NG)
        # Write the dimensions
        io.write_record_lr4_i(f, self.dims)
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.append([0], np.cumsum(npt))
        # Write the coordinates
        for i in range(self.NG):
            # Point indices
            ia = mpt[i]
            ib = mpt[i+1]
            # Put all three coordinates 
            io.write_record_lr8_f(f, self.X[:,ia:ib])
        # Close the file
        f.close()
            
    # Write as a big-endian double-precision file
    def Write_b8(self, fname, single=False):
        r"""Write multi-zone little-endian, double-precision Plot3D file
        
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
            * 2016-10-16 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'wb')
        # Check for single grid
        if not(single and self.NG == 1):
            # Write the number of zones
            io.write_record_r4_i(f, self.NG)
        # Write the dimensions
        io.write_record_r4_i(f, self.dims)
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.append([0], np.cumsum(npt))
        # Write the coordinates
        for i in range(self.NG):
            # Point indices
            ia = mpt[i]
            ib = mpt[i+1]
            # Put all three coordinates 
            io.write_record_r8_f(f, self.X[:,ia:ib])
        # Close the file
        f.close()
            
    # Write as a big-endian single-precision file
    def Write_b4(self, fname, single=False):
        r"""Write multi-zone little-endian, single-precision Plot3D file
        
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
            * 2016-10-16 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'wb')
        # Check for single grid
        if not(single and self.NG == 1):
            # Write the number of zones
            io.write_record_r4_i(f, self.NG)
        # Write the dimensions
        io.write_record_r4_i(f, self.dims)
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.append([0], np.cumsum(npt))
        # Write the coordinates
        for i in range(self.NG):
            # Point indices
            ia = mpt[i]
            ib = mpt[i+1]
            # Put all three coordinates 
            io.write_record_r8_f(f, self.X[:,ia:ib])
        # Close the file
        f.close()
  # >

  # ===================
  # Triangulation
  # ===================
  # <
   # --- Creators ---
    # Divide mesh into simple triangulation
    def make_tri(self):
        r"""Divide a surface grid into a triangulation

        :Call:
            >>> tri = x.make_tri()
        :Versions:
            * 2020-07-06 ``@ddalle``: Version 1.0
        """
        # Select dimensions
        dims = self.dims
        # Get min dimension for each grid
        ndimmin = np.min(dims, axis=1)
        # Get max min dimension
        if np.max(ndimmin) > 1:
            # Find index of first offending grid
            ig = np.where(ndimmin > 1)[0][0]
            # Found at least one volume mesh
            raise IndexError(
                ("Cannot process non-surface grid %i " % ig) +
                ("dimensions %i x %i x %i" % tuple(dims[ig])))
        # Get nodes
        nodes = self.X.copy().T
        # Triangle counter along each dimension
        tridims = np.fmax(1, dims - 1)
        # Number of triangles
        Ntri = 2 * np.prod(tridims, axis=1)
        ntri = 2 * np.sum(Ntri)
        # Initialize tris and component numbers
        tris = np.zeros((ntri, 3), dtype="int")
        compid = np.zeros(ntri, dtype="int")
        # Current triangle counter
        mtri = 0
        # Current node counter
        mnode = 0
        # Loop through meshes
        for i in range(self.NG):
            # Get number of triangles in this grid
            itri = Ntri[i]
            # Dimensions for this triangle
            idims = dims[i]
            # Get minor dimensions
            if idims[2] == 1:
                # Use J and K
                nj = idims[0]
                nk = idims[1]
            elif dims[i][1] == 1:
                # Use J and L
                nj = idims[0]
                nk = idims[2]
            else:
                # Use K and L
                nj = idims[1]
                nk = idims[2]
            # Number of nodes in this grid
            inode = nj * nk
            # Range of nodes
            ia = mnode
            ib = mnode + inode
            # Downselect nodes
            inodes = nodes[ia:ib, :]
            # Boolean maps to select corners
            q1 = np.ones(inode, dtype="bool")
            q2 = np.ones(inode, dtype="bool")
            q3 = np.ones(inode, dtype="bool")
            q4 = np.ones(inode, dtype="bool")
            # Adjust the map for top left corner
            q1[nj-1::nj] = False
            q1[nj*(nk-1):] = False
            # Adjust the map for top right corner
            q2[::nj] = False
            q2[nj*(nk-1):] = False
            # Adjust the map for bottom left corner
            q3[nj-1::nj] = False
            q3[:nj] = False
            # Adjust the map for bottom right corner
            q4[::nj] = False
            q4[:nj] = False
            # Get corners of each each quad
            x1 = inodes[q1, :]
            x2 = inodes[q2, :]
            x3 = inodes[q3, :]
            x4 = inodes[q4, :]
            # Calculate lengths of diagonals
            d14 = np.sqrt(np.sum((x1 - x4)**2, axis=1))
            d23 = np.sqrt(np.sum((x2 - x3)**2, axis=1))
            # Find shorter diagonal
            q14 = (d14 < d23 + 2e-2*d14)
            q23 = np.logical_not(q14)
            # Node numbers
            i1 = np.where(q1)[0]
            i2 = np.where(q2)[0]
            i3 = np.where(q3)[0]
            i4 = np.where(q4)[0]
            # Initialize tris
            itris = np.zeros((itri, 3), dtype="int")
            # Assume 2 -> 3 connection at first
            itris1 = np.stack((i2, i3, i4), axis=1)
            itris2 = np.stack((i1, i3, i2), axis=1)
            # Save 1 -> 4 where appropriate
            itris1[q14, :] = np.stack((i1[q14], i4[q14], i2[q14]), axis=1)
            itris2[q14, :] = np.stack((i1[q14], i3[q14], i4[q14]), axis=1)
            # Save to overall tris
            itris[::2, :] = itris1
            itris[1::2, :] = itris2
            # Save tris (1-based indexing)
            tris[mtri:mtri + itri, :] = itris + mnode + 1
            # Save components
            compid[mtri:mtri + itri] = i + 1
            # Update counters
            mtri += itri
            mnode += inode
        # Create triangulation
        T = tri.Tri(Nodes=nodes, Tris=tris, CompID=compid)
        # Output
        return T
  # >
  
  # ======
  # MIXSUR
  # ======
  # <
    # Map surface grid points to TRI file components
    def MapTriCompID(self, tri, n=1, **kw):
        r"""Create a ``.ovfi`` file using family names from a tri surf
        
        :Call:
            >>> C = x.MapTriOvfi(tri, n=1, **kw)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *tri*: :class:`cape.tri.Tri`
                Triangulation; likely with named components
            *n*: {``1``} | positive :class:`int`
                Grid number to process (1-based index)
        :Outputs:
            *C*: :class:`np.ndarray`\ [:class:`int`]
                * *shape*: (nj, nk, 2)

                Array of component IDs closest to each point in 

        :Versions:
            * 2017-02-08 ``@ddalle``: Version 1.0
        """
        # Check grid number
        if n > self.NG:
            raise ValueError("Cannot process grid %i; only %s grids present"
                % (n, self.NG))
        # Check triangulation type
        tt = type(tri).__name__
        if not tt.startswith("Tri"):
            raise TypeError(
                "Triangulation for mapping must be 'Tri', or 'Triq'")
        # Process primary tolerances
        atol  = kw.get("atol",  kw.get("AbsTol",  atoldef))
        rtol  = kw.get("rtol",  kw.get("RelTol",  rtoldef))
        ctol  = kw.get("ctol",  kw.get("CompTol", ctoldef))
        antol = kw.get("ntol",  kw.get("ProjTol", antoldef))
        antol = kw.get("antol", kw.get("AbsProjTol",  antol))
        rntol = kw.get("rntol", kw.get("RelProjTol",  rntoldef))
        cntol = kw.get("cntol", kw.get("CompProjTol", cntoldef))
        # Process family tolerances
        aftol = kw.get("aftol", kw.get("AbsFamilyTol",  aftoldef))
        rftol = kw.get("rftol", kw.get("RelFamilyTol",  rftoldef))
        cftol = kw.get("cftol", kw.get("CompFamilyTol", cftoldef))
        # Family projection tolerances
        anftol = kw.get("nftol",  kw.get("ProjFamilyTol", anftoldef))
        anftol = kw.get("anftol", kw.get("AbsProjFamilyTol", anftol))
        rnftol = kw.get("rnftol", kw.get("RelProjFamilyTol", rnftoldef))
        cnftol = kw.get("cnftol", kw.get("CompProjFamilyTol", cnftoldef))
        # Get scale of the entire triangulation
        L = tri.GetCompScale()
        # Initialize scales of components
        LC = {}
        # Put together absolute and relative tols
        tol   = atol   + rtol*L
        ntol  = antol  + rntol*L
        ftol  = aftol  + rftol*L
        nftol = anftol + rnftol*L
        # Get number of points in prior grids
        ia = np.prod(self.NJ[:n-1] * self.NK[:n-1] * self.NL[:n-1]) - 1
        # Get number of points
        nj = self.NJ[n-1]
        nk = self.NK[n-1]
        nl = self.NL[n-1]
        # Verbose flag
        v = kw.get("v", False)
        # Initialize components for each surface grid
        C = np.zeros((nj,nk,4), dtype=int)
        # Loop through columns
        for k in range(nk):
            # Status update if verbose
            if v:
                print("  k = %i/%i" % (k+1,nk))
            # Loop through rows of points
            for j in range(nj):
                # Get overall index
                i = ia + k*nj + j
                # Perform search
                T = tri.GetNearestTri(self.X[:,i])
                # Get components
                c1 = T.get("c1")
                c2 = T.get("c2")
                c3 = T.get("c3")
                c4 = T.get("c4")
                # Make sure component scale is present
                if c1 not in LC:
                    LC[c1] = tri.GetCompScale(c1)
                # Make sure secondary component scale is present
                if (c2 is not None) and (c2 not in LC):
                    LC[c2] = tri.GetCompScale(c2)
                # Get overall tolerances
                toli  = tol + ctol*LC[c1]
                ntoli = ntol + cntol*LC[c1]
                # Filter results
                if (T["d1"] > toli) or (T["z1"] > ntoli):
                    # Status update
                    if v:
                        print("   j=%s, k=%s, d1=%.2e/%.2e, z1=%.2e/%.2e"
                            % (j,k, T["d1"],toli, T["z1"],ntoli))
                    continue
                # Check proximity of secondary component
                if (c2 is not None):
                     # Make sure secondary component scale is present
                    if c2 not in LC:
                        LC[c2] = tri.GetCompScale(c2)
                    # Maximum component scale
                    Li = max(LC[c1], LC[c2])
                    # Get overall tolerances
                    ftoli  = ftol  + cftol*Li
                    nftoli = nftol + cnftol*Li
                    # Filter family matches
                    if (T["d2"] > ftoli) or (T["z2"] > nftoli):
                        c2 = None
                # Filter tertiary family proximity
                if (c3 is not None):
                    if (T["d3"]>ftoli) or (T["z3"]>nftoli):
                        c3 = None
                # Filter fourth family proximity
                if (c4 is not None):
                    if (T["d4"]>ftoli) or (T["z4"]>nftoli):
                        c4 = None
                # Save components
                if (c2 is None):
                    # Save primary family only
                    C[j,k,0] = c1
                elif (c3 is None):
                    # Sort primary and secondary families
                    C[j,k,0] = min(c1,c2)
                    C[j,k,1] = max(c1,c2)
                elif (c4 is None):
                    # Sort primary, secondary, tertiary families
                    C[j,k,:3] = np.sort([c1,c2,c3])
                else:
                    # Sort maximum of four families
                    C[j,k,:] = np.sort([c1,c2,c3,c4])
        # Output
        return C
        
    # Map surface grid points to TRI file components
    def MapTriBCs(self, tri, n=1, **kw):
        r"""Find the BC blocks by projecting a mesh to a triangulation
        
        :Call:
            >>> x.MapTriBCs(tri, n=1, **kw)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *tri*: :class:`cape.tri.Tri`
                Triangulation; likely with named components
            *n*: {``1``} | positive :class:`int`
                Grid number to process (1-based index)
        :Versions:
            * 2017-02-08 ``@ddalle``: Version 1.0
        """
        # Get compIDs
        C = self.MapTriCompID(tri, n=n, **kw)
        # Map them
        return MapTriMatchBCs(C)
        
    # Function to edit an OVFI file
    def MapOvfi(self, fi, fo, tri, **kw):
        r"""Edit a ``.ovfi`` file using a triangulation for family names
        
        :Call:
            >>> x.MapOvfi(fi, fo, tri, **kw)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fi*: :class:`str`
                Name of input OVFI file
            *fo*: :class:`str`
                Name of output OVFI file
            *tri*: :class:`caepe.tri.Tri`
                Triangulation with named components
        :Keyword Arguments:
            *v*: ``True`` | {``False``}
                Verbose
            *atol*, *AbsTol*: {``_atol_``} | :class:`float` > 0
                Absolute tolerance for nearest-tri search
            *rtol*, *RelTol*: {``_rtol_``} | :class:`float` >= 0
                Tolerance for nearest-tri relative to scale of triangulation
            *ctol*, *CompTol*: {``_ctol_``} | :class:`float` >= 0
                Tolerance for nearest-tri relative to scale of component
            *antol*, *AbsProjTol*: {``_antol_``} | :class:`float` > 0
                Absolute projection tolerance for near nearest-tri search
            *rntol*, *RelProjTol*: {``_rntol_``} | :class:`float` >= 0
                Projection tolerance relative to scale of triangulation
            *cntol*, *CompProjTol*: {``_cntol_``} | :class:`float` >= 0
                Projection tolerance relative to scale of component
            *aftol*, *AbsFamilyTol*: {``_aftol_``} | :class:`float` > 0
                Absolute tolerance for secondary family search
            *rftol*, *RelFamilyTol*: {``_rftol_``} | :class:`float`
                Secondary family search tol relative to tri scale
            *cftol*, *CompFamilyTol*: {``_cftol_``} | :class:`float`
                Secondary family search tol relative to component scale
            *nftol*, *ProjFamilyTol*: {``_anftol_``} | :class:`float`
                Absolute projection tol for secondary family search
            *anftol*, *AbsProjFamilyTol*: {*nftol*} | :class:`float`
                Absolute projection tol for secondary family search
            *rnftol*, *RelProjFamilyTol*: {``_rnftol_``} | :class:`float`
                Secondary family search projection tol relative to tri scale
            *cnftol*, *CompProjFamilyTol*: {``_cnftol_``} | :class:`float`
                Secondary family search projection tol relative to comp scale
        :Versions:
            * 2017-02-09 ``@ddalle``: Version 1.0
        """
        # Invert families (must be a UH3D for now)
        try:
            # Initialize family list
            faces = {}
            # Loop through faces
            for face in tri.Conf:
                # Get compIDs
                comps = tri.Conf[face]
                # Check for a list
                if type(comps).__name__ in ['list', 'ndarray']:
                    # Loop through list
                    for c in comps:
                        faces[c] = face
                else:
                    # Single component
                    faces[comps] = face
        except AttributeError:
            raise ValueError("Triangulation must have *Conf* attribute\n" +
                "In most cases, the triangulation must be from a UH3D file")
        # Default family name
        # Process boundary conditions
        BCs = self.MapTriBCs(tri, n=1, **kw)
        # Read namelist
        ovfi = namelist2.Namelist2(fi)
        # Read BCINP section
        ibtyp = np.array(ovfi.GetKeyFromGroupName("BCINP","IBTYP")).flatten()
        ibdir = np.array(ovfi.GetKeyFromGroupName("BCINP","IBDIR")).flatten()
        jbcs  = np.array(ovfi.GetKeyFromGroupName("BCINP","JBCS" )).flatten()
        jbce  = np.array(ovfi.GetKeyFromGroupName("BCINP","JBCE" )).flatten()
        kbcs  = np.array(ovfi.GetKeyFromGroupName("BCINP","KBCS" )).flatten()
        kbce  = np.array(ovfi.GetKeyFromGroupName("BCINP","KBCE" )).flatten()
        lbcs  = np.array(ovfi.GetKeyFromGroupName("BCINP","LBCS" )).flatten()
        lbce  = np.array(ovfi.GetKeyFromGroupName("BCINP","LBCE" )).flatten()
        # Filter out non-wall BCs
        qnowall = np.logical_or(ibtyp>9, ibdir!=3)
        # Number of actual BCs
        nBC = len(BCs)
        # Basic types
        ityp = [5]*nBC
        idir = [3]*nBC
        # Indices
        ja = []; jb = []
        ka = []; kb = []
        la = [1]*nBC
        lb = [1]*nBC
        # CompID
        compID = []
        fams = []
        # Loop through BCs
        for i in range(nBC):
            # Append family
            compID.append(BCs[i][0])
            fams.append(faces[BCs[i][0]])
            # Set BCs
            ja.append(BCs[i][1]+1)
            jb.append(BCs[i][2])
            ka.append(BCs[i][3]+1)
            kb.append(BCs[i][4])
        # Append BC types to non-wall BCs
        ibtyp = ityp + list(ibtyp[qnowall])
        ibdir = idir + list(ibdir[qnowall])
        # Append indices
        jbcs = ja + list(jbcs[qnowall])
        jbce = jb + list(jbce[qnowall])
        kbcs = ka + list(kbcs[qnowall])
        kbce = kb + list(kbce[qnowall])
        lbcs = la + list(lbcs[qnowall])
        lbce = lb + list(lbce[qnowall])
        # Set parameters
        ovfi.SetKeyInGroupName('BCINP', 'IBTYP', ibtyp)
        ovfi.SetKeyInGroupName('BCINP', 'IBDIR', ibdir)
        ovfi.SetKeyInGroupName('BCINP', 'JBCS', jbcs)
        ovfi.SetKeyInGroupName('BCINP', 'JBCE', jbce)
        ovfi.SetKeyInGroupName('BCINP', 'KBCS', kbcs)
        ovfi.SetKeyInGroupName('BCINP', 'KBCE', kbce)
        ovfi.SetKeyInGroupName('BCINP', 'LBCS', lbcs)
        ovfi.SetKeyInGroupName('BCINP', 'LBCE', lbce)
        # Create line for list of families
        line = "C Family: %s\n" % (' '.join(fams))
        # Set families
        ovfi.ReplaceLineStartsWith('C Family', line)
        # Write the processed namelist
        ovfi.Write(fo)
    # Fill in docstring
    MapOvfi.__doc__ = MapOvfi.__doc__.replace('_atol_', str(atoldef))
    MapOvfi.__doc__ = MapOvfi.__doc__.replace('_rtol_', str(rtoldef))
    MapOvfi.__doc__ = MapOvfi.__doc__.replace('_ctol_', str(ctoldef))
    MapOvfi.__doc__ = MapOvfi.__doc__.replace('_antol_', str(antoldef))
    MapOvfi.__doc__ = MapOvfi.__doc__.replace('_rntol_', str(rntoldef))
    MapOvfi.__doc__ = MapOvfi.__doc__.replace('_cntol_', str(cntoldef))
    MapOvfi.__doc__ = MapOvfi.__doc__.replace('_aftol_', str(aftoldef))
    MapOvfi.__doc__ = MapOvfi.__doc__.replace('_rftol_', str(rftoldef))
    MapOvfi.__doc__ = MapOvfi.__doc__.replace('_anftol_', str(anftoldef))
    MapOvfi.__doc__ = MapOvfi.__doc__.replace('_cnftol_', str(cnftoldef))
    MapOvfi.__doc__ = MapOvfi.__doc__.replace('_rnftol_', str(rnftoldef))
  # >
# class X

# Plot3D Solution file
class Q(X):
    
    def __init__(self, fname=None):
        """Initialization method
        
        :Call:
            >>> x = X(fname=None)
        :Inputs:
            *fname*: :class:`str`
                Name of Plot3D grid file to read
        :Versions:
            * 2016-10-11 ``@ddalle``: Version 1.0
        """
        # Check for a file to read
        if fname is not None:
            self.Read(fname)
            
  # =======
  # Readers
  # =======
  # <
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
            *ext*: {``ascii``} | ``lb8`` | ``lb4`` | ``b8`` | ``b4`` | ``lr4``
                File type in the form of a file extension code
        :Attributes:
            *x.byteorder*: ``"little"`` | ``"big"`` | {``None``}
                Endianness of binary file
            *x.filetype*: ``"record"`` | ``"stream"`` | {``"ascii"``}
                Basic file type
            *x.p3dtype*: ``"multiple"`` | {``"single"``}
                Plot3D zone type
        :Versions:
            * 2016-10-14 ``@ddalle``: Version 1.0
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
        if (self.byteorder == 'little') and (self.filetype == "record"):
            # Little-endian with records
            ext = 'lr'
            # Check for number-of-grids-marker
            if self.p3dtype == 'multiple':
                # Read number of grids record
                ng = io.read_record_lr4_i(f)
            # Read first record
            dims = io.read_record_lr4_i(f)
            # Read the record marker
            r, = np.fromfile(f, count=1, dtype='<i4')
        elif (self.byteorder == 'big') and (self.filetype == "record"):
            # Big-endian records
            ext = 'r'
            # Check for number-of-grids-marker
            if self.p3dtype == 'multiple':
                # Read number of grids record
                ng = io.read_record_r4_i(f)
            # Read first record
            dims = io.read_record_r4_i(f)
            # Read record marker for first grid
            r, = np.fromfile(f, count=1, dtype='>i4')
        elif (self.byteorder == "little"):
            # Little-endian stream
            ext = 'lb'
            # Check for number-of-grids type
            if self.p3dtype == "multiple":
                # Read the number of grids
                ng = io.fromfile_lb4_i(f, 1)
                # Read the dimensions
                dims = io.fromfile_lb4_i(f, ng*4).reshape((ng[0],4))
            else:
                # Read the dimensions
                dims = io.fromfile_lb4_i(f, 4).reshape((1,4))
            # Total number of points
            npt = np.sum(np.prod(dims, axis=1))
            # Number of bytes in file if single-precision
            nb4 = f.tell() + npt*12
        else:
            # Big-endian stream
            ext = 'b'
            # Check for number-of-grids type
            if self.p3dtype == "multiple":
                # Read the number of grids
                ng = io.fromfile_b4_i(f, 1)
                # Read the dimensions
                dims = io.fromfile_b4_i(f, ng*4).reshape((ng[0],4))
            else:
                # Read the dimensions
                dims = io.fromfile_b4_i(f, 4).reshape((1,4))
        # Try to determine single/double precision
        if (self.filetype == "record"):
            # Number of points in the first grid
            # (Assume at least one grid)
            npt = dims[0] * dims[1] * dims[2]
            nq  = dims[3]
            # Use the number of bytes in the record marker to check
            if r/8 == npt*nq:
                # Double-precision
                self.ext = ext + '8'
            elif r/4 == npt*nq:
                # Single-precision
                self.ext = ext + '4'
        else:
            # Total number of points
            npt = np.sum(np.prod(dims, axis=1))
            # Number of bytes in file if single-precision
            nb4 = f.tell() + npt*4
            # Go to the end of the file
            f.seek(0, 2)
            # Check current position
            if nb4 == f.tell():
                # Single precision
                self.ext = ext + '4'
            else:
                # Double
                self.ext = ext + '8'
        # Close the file
        f.close()
        # Output
        return self.ext
    
    # Read big-endian double-precision
    def Read_b8(self, fname):
        """Read a Plot3D grid as a big-endian double-precision file (stream)
        
        :Call:
            >>> x.Read_b8(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first line
        self.NG, = io.fromfile_b4_i(f, 1)
        # Read the dimensions
        R = io.fromfile_b4_i(f, self.NG*4)
        # Resize dimensions
        self.dims = np.resize(R, (self.NG, 4))
        self.NJ = self.dims[:,0]
        self.NK = self.dims[:,1]
        self.NL = self.dims[:,2]
        self.NQ = self.dims[:,3]
        # Initialize states
        self.Q = []
        # Loop through grids
        for ig in range(self.NG):
            # Dimensions for this grid
            nj, nk, nl, nq = self.dims[ig]
            # Read states
            R = io.fromfile_b8_f(f, nj*nk*nl*nq)
            # Save states
            self.Q.append(R.reshape((nj,nk,nl,nq)))
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
            * 2016-10-15 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first line
        self.NG, = io.fromfile_b4_i(f, 1)
        # Read the dimensions
        R = io.fromfile_b4_i(f, self.NG*4)
        # Resize dimensions
        self.dims = np.resize(R, (self.NG, 4))
        self.NJ = self.dims[:,0]
        self.NK = self.dims[:,1]
        self.NL = self.dims[:,2]
        self.NQ = self.dims[:,3]
        # Initialize states
        self.Q = []
        # Loop through grids
        for ig in range(self.NG):
            # Dimensions for this grid
            nj, nk, nl, nq = self.dims[ig]
            # Read states
            R = io.fromfile_b4_f(f, nj*nk*nl*nq)
            # Save states
            self.Q.append(R.reshape((nj,nk,nl,nq)))
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
            * 2016-10-15 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first line
        self.NG, = io.fromfile_lb4_i(f, 1)
        # Read the dimensions
        R = io.fromfile_lb4_i(f, self.NG*4)
        # Resize dimensions
        self.dims = np.resize(R, (self.NG, 4))
        self.NJ = self.dims[:,0]
        self.NK = self.dims[:,1]
        self.NL = self.dims[:,2]
        self.NQ = self.dims[:,3]
        # Initialize states
        self.Q = []
        # Loop through grids
        for ig in range(self.NG):
            # Dimensions for this grid
            nj, nk, nl, nq = self.dims[ig]
            # Read states
            R = io.fromfile_lb8_f(f, nj*nk*nl*nq)
            # Save states
            self.Q.append(R.reshape((nj,nk,nl,nq)))
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
            * 2016-10-15 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first line
        self.NG, = io.fromfile_lb4_i(f, 1)
        # Read the dimensions
        R = io.fromfile_lb4_i(f, self.NG*4)
        # Resize dimensions
        self.dims = np.resize(R, (self.NG, 4))
        self.NJ = self.dims[:,0]
        self.NK = self.dims[:,1]
        self.NL = self.dims[:,2]
        self.NQ = self.dims[:,3]
        # Initialize states
        self.Q = []
        # Loop through grids
        for ig in range(self.NG):
            # Dimensions for this grid
            nj, nk, nl, nq = self.dims[ig]
            # Read states
            R = io.fromfile_lb4_f(f, nj*nk*nl*nq)
            # Save states
            self.Q.append(R.reshape((nj,nk,nl,nq)))
        # Close the file
        f.close()
    
    # Read big-endian double-precision
    def Read_r8(self, fname):
        """Read a Plot3D grid as a big-endian double-precision file
        
        :Call:
            >>> x.Read_r8(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first record
        R = io.read_record_r4_i(f)
        # Check for single zone or multiple zone
        if len(R) == 1:
            # Number of zones
            self.NG = R[0]
            # Read dimensions
            R = io.read_record_r4_i(f)
        else:
            # Single zone
            self.NG = 1
        # Resize dimensions
        self.dims = np.resize(R, (self.NG, 3))
        self.NJ = self.dims[:,0]
        self.NK = self.dims[:,1]
        self.NL = self.dims[:,2]
        self.NQ = self.dims[:,3]
        # Initialize states
        self.Q = []
        # Loop through the grids
        for i in range(self.NG):
            # Read record
            R = io.read_record_r8_f(f)
            # Save states
            self.Q.append(R.reshape((nj,nk,nl,nq)))
        # Close the file
        f.close()
    
    # Read big-endian single-precision
    def Read_r4(self, fname):
        """Read a Plot3D grid as a big-endian single-precision file
        
        :Call:
            >>> x.Read_r4(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first record
        R = io.read_record_r4_i(f)
        # Check for single zone or multiple zone
        if len(R) == 1:
            # Number of zones
            self.NG = R[0]
            # Read dimensions
            R = io.read_record_r4_i(f)
        else:
            # Single zone
            self.NG = 1
        # Resize dimensions
        self.dims = np.resize(R, (self.NG, 3))
        self.NJ = self.dims[:,0]
        self.NK = self.dims[:,1]
        self.NL = self.dims[:,2]
        self.NQ = self.dims[:,3]
        # Initialize states
        self.Q = []
        # Loop through the grids
        for i in range(self.NG):
            # Read record
            R = io.read_record_r4_f(f)
            # Save states
            self.Q.append(R.reshape((nj,nk,nl,nq)))
        # Close the file
        f.close()
    
    # Read little-endian double-precision
    def Read_lr8(self, fname):
        """Read a Plot3D grid as a little-endian double-precision file
        
        :Call:
            >>> x.Read_lr8(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first record
        R = io.read_record_lr4_i(f)
        # Check for single zone or multiple zone
        if len(R) == 1:
            # Number of zones
            self.NG = R[0]
            # Read dimensions
            R = io.read_record_lr4_i(f)
        else:
            # Single zone
            self.NG = 1
        # Resize dimensions
        self.dims = np.resize(R, (self.NG, 3))
        self.NJ = self.dims[:,0]
        self.NK = self.dims[:,1]
        self.NL = self.dims[:,2]
        self.NQ = self.dims[:,3]
        # Initialize states
        self.Q = []
        # Loop through the grids
        for i in range(self.NG):
            # Read record
            R = io.read_record_lr8_f(f)
            # Save states
            self.Q.append(R.reshape((nj,nk,nl,nq)))
        # Close the file
        f.close()
    
    # Read little-endian single-precision
    def Read_lr4(self, fname):
        """Read a Plot3D grid as a little-endian single-precision file
        
        :Call:
            >>> x.Read_lr4(fname)
        :Inputs:
            *x*: :class:`cape.plot3d.X`
                Plot3D grid interface
            *fname*: :class:`str`
                Name of Plot3D file
        :Versions:
            * 2016-10-15 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'rb')
        # Read the first record
        R = io.read_record_lr4_i(f)
        # Check for single zone or multiple zone
        if len(R) == 1:
            # Number of zones
            self.NG = R[0]
            # Read dimensions
            R = io.read_record_lr4_i(f)
        else:
            # Single zone
            self.NG = 1
        # Resize dimensions
        self.dims = np.resize(R, (self.NG, 3))
        self.NJ = self.dims[:,0]
        self.NK = self.dims[:,1]
        self.NL = self.dims[:,2]
        self.NQ = self.dims[:,3]
        # Initialize states
        self.Q = []
        # Loop through the grids
        for i in range(self.NG):
            # Read record
            R = io.read_record_lr4_f(f)
            # Save states
            self.Q.append(R.reshape((nj,nk,nl,nq)))
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
            * 2016-10-15 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'r')
        # Read the first line
        v = [int(vi) for vi in f.readline().split()]
        # Get the number of elements
        if len(v) == 1:
            # Number of zones
            self.NG = int(v[0])
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
  # >
  
  # =======
  # Writers
  # =======
  # <
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
            * 2016-10-16 ``@ddalle``: Version 1.0
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
            * 2016-10-16 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'wb')
        # Check for single grid
        if not(single and self.NG == 1):
            # Write the number of zones
            io.write_record_lr4_i(f, self.NG)
        # Write the dimensions
        io.write_record_lr4_i(f, self.dims)
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.append([0], np.cumsum(npt))
        # Write the coordinates
        for i in range(self.NG):
            # Point indices
            ia = mpt[i]
            ib = mpt[i+1]
            # Put all three coordinates 
            io.write_record_lr8_f(f, self.X[:,ia:ib])
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
            * 2016-10-16 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'wb')
        # Check for single grid
        if not(single and self.NG == 1):
            # Write the number of zones
            io.write_record_lr4_i(f, self.NG)
        # Write the dimensions
        io.write_record_lr4_i(f, self.dims)
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.append([0], np.cumsum(npt))
        # Write the coordinates
        for i in range(self.NG):
            # Point indices
            ia = mpt[i]
            ib = mpt[i+1]
            # Put all three coordinates 
            io.write_record_lr8_f(f, self.X[:,ia:ib])
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
            * 2016-10-16 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'wb')
        # Check for single grid
        if not(single and self.NG == 1):
            # Write the number of zones
            io.write_record_r4_i(f, self.NG)
        # Write the dimensions
        io.write_record_r4_i(f, self.dims)
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.append([0], np.cumsum(npt))
        # Write the coordinates
        for i in range(self.NG):
            # Point indices
            ia = mpt[i]
            ib = mpt[i+1]
            # Put all three coordinates 
            io.write_record_r8_f(f, self.X[:,ia:ib])
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
            * 2016-10-16 ``@ddalle``: Version 1.0
        """
        # Open the file
        f = open(fname, 'wb')
        # Check for single grid
        if not(single and self.NG == 1):
            # Write the number of zones
            io.write_record_r4_i(f, self.NG)
        # Write the dimensions
        io.write_record_r4_i(f, self.dims)
        # Point counts
        npt = np.prod(self.dims, axis=1)
        mpt = np.append([0], np.cumsum(npt))
        # Write the coordinates
        for i in range(self.NG):
            # Point indices
            ia = mpt[i]
            ib = mpt[i+1]
            # Put all three coordinates 
            io.write_record_r8_f(f, self.X[:,ia:ib])
        # Close the file
        f.close()
  # >
# class Q


# Map surface grid points to TRI file components
def MapTriMatchBCs(C):
    """Create a ``.ovfi`` file using the family names from a triangulation
    
    :Call:
        >>> BCs = MapTriMatchBCs(C)
    :Inputs:
        *x*: :class:`cape.plot3d.X`
            Plot3D grid interface
        *tri*: :class:`cape.tri.Tri`
            Triangulation; likely with named components
        *n*: {``1``} | positive :class:`int`
            Grid number to process (1-based index)
    :Versions:
        * 2017-02-08 ``@ddalle``: Version 1.0
    """
    # Get list of component IDs included
    comps = np.unique(C[C>0])
    # Initialize blocks
    BCs = []
    # Maximum number of blocks
    MaxBlocks = 5000
    # Loop through them
    for comp in comps:
        # Get mask of grid points matching *comp*
        I = np.any(C==comp, axis=2)
        # Loop until we've emptied all the blocks
        n = 0
        while (n<MaxBlocks) and np.any(I):
            # Get the indices of the block
            ja, jb, ka, kb = util.GetBCBlock2(I)
            # Append those indices
            BCs.append([comp, ja, jb, ka, kb])
            # Blank out those grid points to look for next block
            I[ja:jb,ka:kb] = False
    # Output
    return BCs
# end MapTriBCs
