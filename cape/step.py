#!/usr/bin/env python
"""
:mod:`cape.step`: Python interface to STEP Files
===================================================

This module provides an interface to import points, curves, and potentially
other entities in the future from STEP files following ISO 10303-21.  It
provides the class :class:`cape.step.STEP`, which allows the user to read
(preferably simple) STEP files.  Its primary purpose is to read STEP files and
write Plot3D curve files.

See also:

    * :mod:`pc_Step2Crv`
    * :mod:`pc_StepTri2Crv`

"""

# File checking
import os
# Basic numerics
import numpy as np
# B-spline interpolation
from scipy.interpolate import splev
# Detailed string processing
import re

# CAPE input/output module
from . import io

# Class for step files
class STEP(object):
    """Interface for STEP files
    
    :Call:
        >>> stp = STEP(fname)
    :Inputs:
        *fname*: :class:`str`
            Name of STEP file to read (``.stp`` or ``.step``)
    :Outputs:
        *stp*: :class:`cape.step.STEP`
            STEP file interface
        *stp.npt*: :class:`int`
            Number of points defined in file
        *stp.ncrv*: :class:`int`
            Number of curves
        *stp.pts*: :class:`np.ndarray` (:class:`float`) shape=(*npt*,3)
            Array of coordinates of all points defined in file
        *stp.ipt*: :class:`np.ndarray` (:class:`int`) shape=(*npt*,)
            Array of point entity indices; *pts[i]* is entity *pts[ipt[i]]*
        *stp.crvs*: :class:`list` (:class:`np.ndarray` | ``None``)
            List of sampled curves; initialized to ``None``
        *stp.ocrv*: :class:`list` (:class:`int`)
            Order of each spline
        *stp.icrv*: :class:`list` (:class:`np.ndarray` (:class:`int`))
            Array of indices of knots for each spline
    :Versions:
        * 2016-05-10 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self, fname=None, xtol=None, ytol=None, ztol=None):
        """Initialization method
        
        :Versions:
            * 2016-05-10 ``@ddalle``: First version
        """
        # Check for empty
        if fname is None: return
        # Read the file
        self.Read(fname)
        # Check for tolerances
        if xtol is not None: self.pts[np.abs(self.pts[:,0])<=xtol,1] == 0.0
        if ytol is not None: self.pts[np.abs(self.pts[:,1])<=ytol,1] == 0.0
        if ztol is not None: self.pts[np.abs(self.pts[:,2])<=ztol,1] == 0.0
        
    
    # Read a file
    def Read(self, fname):
        """Read a STEP file
        
        :Call:
            >>> stp.Read(fname)
        :Inputs:
            *stp*: :class:`cape.step.STEP`
                STEP file interface
            *fname*: :class:`str`
                Name of file to read
        :Versions:
            * 2016-05-10 ``@ddalle``: First version
        """
        # Check for the file
        if not os.path.isfile(fname):
            raise SystemError("STEP file '%s' could not be found" % fname)
        # Open the file.
        f = open(fname)
        # Save file name
        self.fname = fname
        # Read the first line.
        line = f.readline()
        # Check it.
        if not line.startswith('ISO-10'):
            raise IOError("File is not a recognized STEP file.")
        # Initialize point coordinates and numbers
        R = []
        I = []
        # Initialize directions and numbers
        D = []
        ID = []
        # Initialize vectors and numbers
        V = []
        IV = []
        # Initialize curves
        oC = []
        iC = []
        J = []
        # Set location.
        ftell = 0
        # Loop through contents
        while f.tell() != ftell:
            # Read the next line.
            ftell = f.tell()
            line = f.readline()
            # Check for labeled contents.
            if not line.startswith('#'): continue
            # Split the contents to get the type.
            v = line.split('=')
            # Check contents for a Cartesian point.
            if (len(v) == 2) and (v[1].startswith('CARTESIAN_POINT')):
                # Get the string containing the coordinates.
                g = re.search("\([-+0-9Ee., ]+\)", v[1])
                # Check for a match.
                if not g: continue
                # Read the coordinates.
                coord = re.findall("[-+0-9Ee.]+", g.group(0))
                # Check for a list of three coordinates.
                if len(coord) != 3: continue
                # Convert to floats and append to list of coordinates.
                R.append([float(xi) for xi in coord])
                # Save the index.
                I.append(int(v[0][1:]))
            elif (len(v) == 2) and (v[1].startswith('B_SPLINE_CURVE')):
                # Get the string containing the indices
                g = re.search("\([0-9#, ]+\)", v[1])
                # Check for a match.
                if not g: continue
                # Read the indices.
                jx = re.findall('[0-9]+', g.group(0))
                # Get the order.
                nx = int(line.split(',')[1])
                # Append the list of coordinates.
                oC.append(nx)
                iC.append(np.array([int(j) for j in jx]))
                # Save curve numbers
                J.append(int(v[0][1:]))
            elif (len(v) == 2) and (v[1].startswith('DIRECTION')):
                # Get the string containing the coordinates.
                g = re.search("\([-+0-9Ee., ]+\)", v[1])
                # Check for a match.
                if not g: continue
                # Read the coordinates.
                coord = re.findall("[-+0-9Ee.]+", g.group(0))
                # Check for a list of three coordinates.
                if len(coord) != 3: continue
                # Convert to floats and append to list of coordinates.
                D.append([float(xi) for xi in coord])
                # Save the index.
                ID.append(int(v[0][1:]))
            elif (len(v) == 2) and (v[1].startswith('VECTOR')):
                # Get the numeric strings from the definition
                jv = re.findall('[-0-9][-+0-9Ee.]*', v[1])
                # Get the length and direction
                di = D[ID.index(int(jv[0]))]
                li = float(jv[1])
                # Append the vector
                V.append([dij*li for dij in di])
                # Save the number
                IV.append(int(v[0][1:]))
            elif (len(v) == 2) and (v[1].startswith('LINE')):
                # Get the start point and vector indices
                jl = re.findall('[0-9]+', v[1].split("'")[-1])
                # Get the vector and start point
                vj = V[IV.index(int(jl[1]))]
                xj = R[ I.index(int(jl[0]))]
                # Create a point for the end point
                ij = int(v[0][1:])
                R.append([vj[k]+xj[k] for k in range(len(xj))])
                I.append(ij)
                # Create a curve
                oC.append(1)
                iC.append(np.array([int(jl[0]), ij]))
                # Save curve number
                J.append(ij)
        # Close file.
        f.close()
        # Convert to NumPy.
        self.pts = np.array(R)
        self.ipt = np.array(I)
        # Save the counts
        self.ncrv = len(iC)
        self.npt = self.pts.shape[0]
        # Save the curves
        self.ocrv = oC
        self.icrv = iC
        self.jcrv = J
        # Initialize sampled curves
        self.crvs = [None for j in range(self.ncrv)]
        
    # Get knots of a curve
    def GetCurveKnots(self, j):
        """Get knots of curve *j*
        
        :Call:
            >>> X = stp.GetCurveKnots(j)
        :Inputs:
            *stp*: :class:`cape.step.STEP`
                STEP file interface
            *j*: :class:`int`
                Curve number
        :Outputs:
            *X*: :class:`np.ndarray` (:class:`float`) shape=(*ncrvj*,3)
                Matrix of x, y, and z coordinates of curve *j*
        :Versions:
            * 2016-05-10 ``@ddalle``: First version
        """
        # Get curve list
        I = self.icrv[j]
        # Form array
        return np.array([self.pts[self.ipt==i][0] for i in I])
        
    # Evaluate the spline of a curve
    def EvaluateCurve(self, j, u):
        """Evaluate B-spline of curve *j*
        
        :Call:
            >>> Y = stp.EvaluateCurve(j, u)
        :Inputs:
            *stp*: :class:`cape.step.STEP`
                STEP file interface
            *j*: :class:`int`
                Curve number
            *u*: :class:`np.ndarray` (:class:`float`)
                Values of input parameter to :func:`splev`; like arc length
        :Outputs:
            *Y*: :class:`np.ndarray` (:class:`float`) shape=(*u.size*,3)
                Points along the spline
        :Versions:
            * 2016-05-10 ``@ddalle``: First version
        """
        # Get the knots
        c = np.transpose(self.GetCurveKnots(j))
        # Number of points
        n = len(c[0])
        # Get the order
        k = self.ocrv[j]
        # Number of spline intervals
        N = (n-1) / k
        # Create the parametric values of the knots
        t = np.hstack(([0], np.floor(np.arange(n+k-1)/k), [N]))
        # Evaluate
        Y = splev(u, (t, c, k))
        # Output as array
        return np.transpose([Y[0], Y[1], Y[2]])
        
    # Sample a curve with uniform spacing
    def SampleCurve(self, j, n=None, ds=None, dth=None, da=None):
        """Evaluate a B-spline with uniformly distributed points
        
        :Call:
            >>> y = stp.SampleCurve(j, n=None, ds=None, dth=None)
        :Inputs:
            *stp*: :class:`cape.step.STEP`
                STEP file interface
            *j*: :class:`int`
                Curve number
            *n*: :class:`int`
                (Minimum) number of intervals to use
            *ds*: :class:`float`
                Upper bound of uniform spacing
            *dth*: :class:`float` | {``None``}
                Maximum allowed turning angle in degrees
            *da*: :class:`float` | {``None``}
                Maximum allowed length-weighted turning angle
        :Outputs:
            *y* :class:`np.ndarray` (:class:`float`) shape=(*n*,3)
                Uniformly spaced points along the spline
        :Versions:
            * 2016-05-10 ``@ddalle``: First version
        """
        # Default turning angle
        if dth is None: dth = 180.0
        # Order of the spline
        k = self.ocrv[j]
        # Number of knots
        n0 = len(self.icrv[j])
        # Number of intervals
        N = (n0-1) / k
        # Evaluate curve on a fine grid.
        u = np.linspace(0, N, 100*N+1)
        # Evaluate
        X = self.EvaluateCurve(j, u)
        # Calculate step size of those curves
        dL = np.sqrt(np.sum((X[1:]-X[:-1])**2, 1))
        # Cumulative length array
        L = np.insert(np.cumsum(dL), 0, 0.0)
        # Get requested spacing
        if ds and n:
            # Minimum number of points
            L1 = np.linspace(0, L[-1], max(n,np.ceil(L[-1]/ds))+1)
        elif ds is not None:
            # Uniform lengths based on spacing; *ds* is upper limit
            L1 = np.linspace(0, L[-1], np.ceil(L[-1]/ds)+1)
        elif n is not None:
            # Uniform lengths based on *n* points
            L1 = np.linspace(0, L[-1], n+1)
        else:
            # No spacing
            raise ValueError("Please specify either *ds* or *n*.")
        # Redistribute input parameter in order to get requested spacing
        w = np.interp(L1, L, u)
        # Reevaluate.
        self.crvs[j] = self.EvaluateCurve(j, w)
        # Loop until turning angle criterion is met
        kth = 0
        while dth is not None and kth < 5:
            # Evaluation counter
            kth += 1
            # Get current turning angles
            th = self.GetTurningAngle(j)
            # Find turning angle exceedances
            ith = np.where(th > dth)[0]
            # Exit if no exceedances
            if len(ith) == 0: break
            # Loop through the exceedances in reverse order
            for i in ith[::-1]:
                # Add a point before and after the angle
                w = np.insert(w, i+2, (w[i+1]+w[i+2])/2)
                w = np.insert(w, i+1, (w[i]+w[i+1])/2)
            # Reevaluate.
            self.crvs[j] = self.EvaluateCurve(j, w)
        # Loop until weighted turning angle criterion is met
        kth = 0
        while da is not None and kth < 5:
            # Evaluation counter
            kth += 1
            # Reevaluate.
            self.crvs[j] = self.EvaluateCurve(j, w)
            # Exit if no test
            if dth is None: break
            # Get current turning angles
            th = self.GetWeightedTurningAngle(j)
            # Find turning angle exceedances
            ith = np.where(th > da)[0]
            # Exit if no exceedances
            if len(ith) == 0: break
            # Finer points
            w2 = (w[:-1] + w[1:]) / 2
            # Loop through the exceedances in reverse order
            for i in ith[::-1]:
                # Check if the following segment has already been refined.
                if i+1 not in ith:
                    w = np.insert(w, i+2, (w[i+1]+w[i+2])/2)
                # Add a point before and after the angle
                w = np.insert(w, i+1, (w[i]+w[i+1])/2)
            # Reevaluate.
            self.crvs[j] = self.EvaluateCurve(j, w)
        # Output
        return self.crvs[j]
        
    # Sample multiple curves
    def SampleCurves(self, J=None, n=None, ds=None, dth=None, da=None):
        """Evaluate a list of B-splines with uniformly distributed points
        
        :Call:
            >>> stp.SampleCurves(J=None, n=None, ds=None, dth=None, da=None)
        :Inputs:
            *stp*: :class:`cape.step.STEP`
                STEP file interface
            *J*: {``None``} | :class:`int`
                List of curve numbers (defaults to all curves)
            *n*: :class:`int`
                Number of intervals to use
            *ds*: :class:`float`
                Upper bound of uniform spacing
            *dth*: :class:`float` | {``None``}
                Maximum allowed turning angle in degrees
            *da*: :class:`float` | {``None``}
                Maximum allowed length-weighted turning angle
        :Versions:
            * 2016-05-10 ``@ddalle``: First version
        """
        # Default list: all
        if J is None:
            J = range(self.ncrv)
        # Sample each curve
        for j in J:
            self.SampleCurve(j, n=n, ds=ds, dth=dth, da=da)
            
    # Evaluate turning angle
    def GetWeightedTurningAngle(self, j):
        """Calculate turning angles between each segment of sampled curve *j*
        
        The turning angle is weighted by the sum of the lengths of the
        neighboring segments
        
        :Call:
            >>> a = stp.GetWeightedTurningAngle(j)
        :Inputs:
            *stp*: :class:`cape.step.STEP`
                STEP file interface
            *j*: :class:`int`
                Curve number
        :Outputs:
            *a*: :class:`np.ndarray` (:class:`float`)
                Length-weighted angle for each pair (in degree-inches)
        :Versions:
            * 2016-05-10 ``@ddalle``: First version
        """
        # Check for curve
        if self.crvs[j] is None:
            raise ValueError("Curve %i has not been sampled" % j)
        # Get segments
        dx = self.crvs[j][1:] - self.crvs[j][:-1]
        # Lengths of each segment
        L = np.sqrt(np.sum(dx**2, axis=1))
        # Dot products
        cth = np.sum(dx[:-1]*dx[1:], axis=1) / (L[:-1]*L[1:])
        # Trim
        cth = np.fmin(1, np.fmax(-1, cth))
        # Angles
        return 180/np.pi * np.arccos(cth) * (L[:-1]+L[1:])
        
    # Evaluate turning angle
    def GetTurningAngle(self, j):
        """Calculate turning angles between each segment of sampled curve *j*
        
        :Call:
            >>> theta = stp.GetTurningAngle(j)
        :Inputs:
            *stp*: :class:`cape.step.STEP`
                STEP file interface
            *j*: :class:`int`
                Curve number
        :Outputs:
            *theta*: :class:`np.ndarray` (:class:`float`)
                Angle between each pair of segments in degrees
        :Versions:
            * 2016-05-10 ``@ddalle``: First version
        """
        # Check for curve
        if self.crvs[j] is None:
            raise ValueError("Curve %i has not been sampled" % j)
        # Get segments
        dx = self.crvs[j][1:] - self.crvs[j][:-1]
        # Lengths of each segment
        L = np.sqrt(np.sum(dx**2, axis=1))
        # Dot products
        cth = np.sum(dx[:-1]*dx[1:], axis=1) / (L[:-1]*L[1:])
        # Trim
        cth = np.fmin(1, np.fmax(-1, cth))
        # Angles
        return 180/np.pi * np.arccos(cth)
        
    # Link curves
    def LinkCurves(self, axis='x', ds=1.0):
        """Reorder curves into a single chain
        
        This also ensures that the end of curve *j* is the start of *j*\ +1. 
        
        :Call:
            >>> stp.LinkCurves(axis='x', ds=1.0)
        :Inputs:
            *stp*: :class:`cape.step.STEP`
                STEP file interface
            *axis*: {``x``} | ``y`` | ``z`` | ``-x`` | ``-y`` | ``-z``
                Dominant sorting axis
            *ds*: ``None`` | :class:`float`
                Maximum gap to close between start and end points
        :Versions:
            * 2016-05-10 ``@ddalle``: First version
        """
        # Default list of curves
        J = range(self.ncrv)
        # Check if the curves have been sampled.
        for j in J:
            if self.crvs[j] is None:
                raise ValueError(("Curve %i (and possibly others) " % j) +
                    "has not been sampled")
        # Axis index
        if axis.endswith('y'):
            # Sort based on 'y'
            ia = 1
        elif axis.endswith('z'):
            # Sort based on 'z'
            ia = 2
        else:
            # Sort based on 'x'
            ia = 0
        # Get start and end points
        xs = np.array([self.crvs[j][0]  for j in J])
        xe = np.array([self.crvs[j][-1] for j in J])
        # Check for min/max
        if axis.startswith('-'):
            # Check if we should use start or end point
            if max(xs[:,ia]) > max(xe[:,ia]):
                # Get index of maximum; no flip
                js = np.argmax(xs[:,ia])
                jf = 0
            else:
                # Use end point; flip first curve
                js = np.argmax(xe[:,ia])
                jf = 1
        else:
            # Check if we should use start or end point
            if min(xs[:,ia]) <= min(xe[:,ia]):
                # Get index of maximum; no flip
                js = np.argmin(xs[:,ia])
                jf = 0
            else:
                # Use end point; flip first curve
                js = np.argmin(xe[:,ia])
                jf = 1
        # Initialize sorted indices
        K = [js]
        J.remove(js)
        # Initialize flip flags
        F = [jf]
        # Loop until all curves are gone
        while len(J) > 0:
            # Current end point
            if jf:
                # Flip; take the first point as the end point
                xj = self.crvs[js][0]
            else:
                # No flip; take the last point ad the end point
                xj = self.crvs[js][-1]
            # Get distance to all start and end points
            dxs = np.sqrt((xs[J,0]-xj[0])**2
                + (xs[J,1]-xj[1])**2 + (xs[J,2]-xj[2])**2)
            dxe = np.sqrt((xe[J,0]-xj[0])**2
                + (xe[J,1]-xj[1])**2 + (xe[J,2]-xj[2])**2)
            # Min distances
            dxsj = min(dxs)
            dxej = min(dxe)
            # Check
            if min(dxsj, dxej) > ds:
                for k in K:
                    print("%2i: [%.4f, %.4f]" % (k, xs[k,ia], xe[k,ia]))
                raise ValueError(
                    ("Distance between curve %i " % js) +
                    ("to next curve exceeds tolerance %s\n" % ds) +
                    ("Current coordinate is %s" % (xj[ia])))
            # Check for min/max
            if dxsj <= dxej:
                # Found match at start of next curve
                js = J[np.argmin(dxs)]
                jf = 0
            else:
                # Found match at end of next curve
                js = J[np.argmin(dxe)]
                jf = 1
            # Append
            K.append(js)
            J.remove(js)
            F.append(jf)
        # Perform reordering
        self.ocrv = [self.ocrv[j] for j in K]
        self.icrv = [self.icrv[j] for j in K]
        self.crvs = [self.crvs[j] for j in K]
        # Flip curves as necessary
        for j in range(self.ncrv):
            # Check the "flip" flag
            if F[j]:
                self.crvs[j] = self.crvs[j][::-1,:]
        # Perform linking
        for j in range(self.ncrv-1):
            # Link the shared point
            self.crvs[j+1][0] = self.crvs[j][-1]
    
    # Write curves to Plot3D file
    def WritePlot3DCurves(self, fname, J=None, bin=False, endian=None):
        """Write list of curves to Plot3D format
        
        :Call:
            >>> stp.WritePlot3DCurves(fname, J=None)
        :Inputs:
            *stp*: :class:`cape.step.STEP`
                STEP file interface
            *fname*: :class:`str`
                Name of Plot3D file to create
            *J*: {``None``} | :class:`list` (:class:`int`)
                List of curve indices to write (default is all)
        :Versions:
            * 2016-05-10 ``@ddalle``: First version
        """
        # Check for binary
        if bin:
            # Write binary file
            self.WritePlot3DCurvesBinary(fname, J=J, endian=endian)
        else:
            # Write ASCII
            self.WritePlot3DCurvesASCII(fname, J=J)
        
    
    # Write curves to Plot3D file
    def WritePlot3DCurvesASCII(self, fname, J=None):
        """Write list of curves to ASCII Plot3D format
        
        :Call:
            >>> stp.WritePlot3DCurvesASCII(fname, J=None)
        :Inputs:
            *stp*: :class:`cape.step.STEP`
                STEP file interface
            *fname*: :class:`str`
                Name of Plot3D file to create
            *J*: {``None``} | :class:`list` (:class:`int`)
                List of curve indices to write (default is all)
        :Versions:
            * 2016-05-10 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'w')
        # Default list of curves
        if J is None:
            J = range(self.ncrv)
        # Check if the curves have been sampled.
        for j in J:
            if self.crvs[j] is None:
                raise ValueError(("Curve %i (and possibly others) " % j) +
                    "has not been sampled")
        # Write the number of curves
        f.write('%12i\n' % len(J))
        # Loop through curves to write dimensions.
        for j in J:
            f.write('%8i%8i%8i\n' % (self.crvs[j].shape[0],1,1))
        # Loop through curves to write coordinates
        for j in J:
            # Extract coordinates
            xj, yj, zj = tuple(self.crvs[j].transpose())
            # Write the coordinates.
            xj.tofile(f, sep=" ", format="%14.7E")
            f.write("\n")
            yj.tofile(f, sep=" ", format="%14.7E")
            f.write("\n")
            zj.tofile(f, sep=" ", format="%14.7E")
            f.write("\n")
        # Close the file
        f.close()
    
    # Write curves to Plot3D file
    def WritePlot3DCurvesBin(self, fname, J=None, **kw):
        """Write list of curves to ASCII Plot3D format
        
        :Call:
            >>> stp.WritePlot3DCurvesBin(fname, J=None, endian=None, **kw)
        :Inputs:
            *stp*: :class:`cape.step.STEP`
                STEP file interface
            *fname*: :class:`str`
                Name of Plot3D file to create
            *J*: {``None``} | :class:`list` (:class:`int`)
                List of curve indices to write (default is all)
            *endian*: {``None``} | ``"big"`` | ``"little"``
                Byte order; use system default if not specified
            *single*: ``True`` | {``False``}
                Whether or not to write single-precision
        :Versions:
            * 2016-05-10 ``@ddalle``: First version
        """
        # Default list of curves
        if J is None:
            J = range(self.ncrv)
        # Check if the curves have been sampled.
        for j in J:
            if self.crvs[j] is None:
                raise ValueError(("Curve %i (and possibly others) " % j) +
                    "has not been sampled")
        # Default byte order
        bo = kw.get('endian')
        if bo is None:
            # This checks system byte order and environment variable flags
            bo = io.sbo
        # Check which version to write
        if bo == 'big':
            # Big-endian
            if kw.get('single', False):
                # Single precision
                self.WritePlot3DCurves_r4(fname, J)
            else:
                # Double precision
                self.WritePlot3DCurves_r8(fname, J)
        else:
            # Little-endian
            if kw.get('single', False):
                # Single precision
                self.WritePlot3DCurves_lr4(fname, J)
            else:
                # Double precision
                self.WritePlot3DCurves_lr8(fname, J)
        
    # Write curves to Plot3D file, big-endian double
    def WritePlot3DCurves_r4(self, fname, J):
        """Write list of curves to double-precision big-endian file
        
        :Call:
            >>> stp.WritePlot3DCurves_r4(fname)
        :Inputs:
            *stp*: :class:`cape.step.STEP`
                STEP file interface
            *fname*: :class:`str`
                Name of Plot3D file to create
            *J*: {``None``} | :class:`list` (:class:`int`)
                List of curve indices to write (default is all)
        :Versions:
            * 2016-09-29 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'wb')
        # Number of curves
        io.write_record_r4_i(f, len(J))
        # Assemble curve dimensions
        gdims = np.array([[self.crvs[j].shape[0],1,1] for j in J])
        # Write grid dimensions: JE, KE, LE
        io.write_record_r4_i(f, gdims.flatten())
        # Loop through curves to write coordinates
        for j in J:
            # Get the curve
            X = self.crvs[j]
            # Write coordinates
            io.write_record_r4_f(f, X.transpose())
        # Close the file
        f.close()
        
    # Write curves to Plot3D file, big-endian double
    def WritePlot3DCurves_r8(self, fname, J):
        """Write list of curves to double-precision big-endian file
        
        :Call:
            >>> stp.WritePlot3DCurves_r8(fname)
        :Inputs:
            *stp*: :class:`cape.step.STEP`
                STEP file interface
            *fname*: :class:`str`
                Name of Plot3D file to create
            *J*: {``None``} | :class:`list` (:class:`int`)
                List of curve indices to write (default is all)
        :Versions:
            * 2016-09-29 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'wb')
        # Number of curves
        io.write_record_r4_i(f, len(J))
        # Assemble curve dimensions
        gdims = np.array([[self.crvs[j].shape[0],1,1] for j in J])
        # Write grid dimensions: JE, KE, LE
        io.write_record_r4_i(f, gdims.flatten())
        # Loop through curves to write coordinates
        for j in J:
            # Get the curve
            X = self.crvs[j]
            # Write coordinates
            io.write_record_r8_f(f, X.transpose())
        # Close the file
        f.close()
        
    # Write curves to Plot3D file, little-endian double
    def WritePlot3DCurves_lr8(self, fname, J):
        """Write list of curves to single-precision little-endian file
        
        :Call:
            >>> stp.WritePlot3DCurves_lr4(fname)
        :Inputs:
            *stp*: :class:`cape.step.STEP`
                STEP file interface
            *fname*: :class:`str`
                Name of Plot3D file to create
            *J*: {``None``} | :class:`list` (:class:`int`)
                List of curve indices to write (default is all)
        :Versions:
            * 2016-09-29 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'wb')
        # Number of curves
        io.write_record_lr4_i(f, len(J))
        # Assemble curve dimensions
        gdims = np.array([[self.crvs[j].shape[0],1,1] for j in J])
        # Write grid dimensions: JE, KE, LE
        io.write_record_lr4_i(f, gdims.flatten())
        # Loop through curves to write coordinates
        for j in J:
            # Get the curve
            X = self.crvs[j]
            # Write coordinates
            io.write_record_lr8_f(f, X.transpose())
        # Close the file
        f.close()
        
    # Write curves to Plot3D file, little-endian double
    def WritePlot3DCurves_lr4(self, fname, J):
        """Write list of curves to single-precision little-endian file
        
        :Call:
            >>> stp.WritePlot3DCurves_lr4(fname)
        :Inputs:
            *stp*: :class:`cape.step.STEP`
                STEP file interface
            *fname*: :class:`str`
                Name of Plot3D file to create
            *J*: {``None``} | :class:`list` (:class:`int`)
                List of curve indices to write (default is all)
        :Versions:
            * 2016-09-29 ``@ddalle``: First version
        """
        # Open the file
        f = open(fname, 'wb')
        # Number of curves
        io.write_record_lr4_i(f, len(J))
        # Assemble curve dimensions
        gdims = np.array([[self.crvs[j].shape[0],1,1] for j in J])
        # Write grid dimensions: JE, KE, LE
        io.write_record_lr4_i(f, gdims.flatten())
        # Loop through curves to write coordinates
        for j in J:
            # Get the curve
            X = self.crvs[j]
            # Write coordinates
            io.write_record_lr4_f(f, X.transpose())
        # Close the file
        f.close()
    
# class STEP
