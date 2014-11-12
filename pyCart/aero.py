"""
Run Directory History Interface: :mod:`pyCart.aero`
===================================================

This module contains functions for reading and processing forces, moments, and
other statistics from a run directory.

:Versions:
    * 2014-11-12 ``@ddalle``: Starter version
"""

# File interface
import os
# Basic numerics
import numpy as np
# Advanced text (regular expressions)
import re


# History class
class Aero(dict):
    """
    Aerodynamic history class
    =========================
    
    This class provides an interface to important data from a run directory.  It
    reads force and moment histories for named components, if available, and
    other types of data can also be stored
    
    :Call:
        >>> aero = pyCart.aero.Aero(comps=[])
    :Inputs:
        *comps*: :class:`list` (:class:`str`)
            List of components to read; defaults to all components available
    :Outputs:
        *aero*: :class:`pyCart.aero.Aero`
            Instance of the aero history class
        *aero*: :class:`dict` (:class:`numpy.ndarray`)
            Dictionary of force and/or moment histories
    :Versions:
        * 2014-11-12 ``@ddalle``: Starter version
    """
    
    # Initialization method
    def __init__(self, comps=[]):
        """Initialization method
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Process the best data folder.
        if os.path.islink('BEST'):
            # There's a BEST/ folder; use it as most recent adaptation cycle.
            fdir = 'BEST'
        elif os.path.isdir('adapt00'):
            # It's an adaptive run, but it hasn't gotten far yet.
            fdir = 'adapt00'
        else:
            # This is not an adaptive cycle; use root folder.
            fdir = '.'
        # Read the loadsCC.dat file to see what components are requested.
        self.ReadLoadsCC()
        # Default component list.
        if (not comps):
            # Extract keys from dictionary.
            comps = self.Components.keys()
        # Loop through components.
        for comp in comps:
            # Expected name of the history file.
            fname = os.path.join(fdir, comp+'.dat')
            # Check if it exists.
            if not os.path.isfile(fname):
                # Warn and got to the next component.
                print("Warning: Component '%s' was not found." % comp)
                continue
            # Otherwise, read the file.
            A = np.loadtxt(fname)
            # Extract info from components for readability
            d = self.Components[comp]
            # Make the component.
            self[comp] = FM(d['C'], MRP=d['MRP'], A=A)
            
            
    # Function to read 'loadsCC.dat'
    def ReadLoadsCC(self):
        """Read forces and moments from a :file:`loadsCC.dat` file if possible
        
        :Call:
            >> aero.ReadLoadsCC()
        :Inputs:
            *aero*: :class:`pyCart.aero.Aero`
                Instance of the aero history class
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Initialize list of components.
        self.Components = {}
        # Check for the file.
        if os.path.isfile(os.path.join('BEST', 'loadsCC.dat')):
            # Use the most recent file.
            fCC = os.path.join('BEST', 'loadsCC.dat')
        elif os.path.isfile(os.path.join('adapt00', 'loadsCC.dat')):
            # Most recent adaptation currently running; no file in BEST
            fCC = os.path.join('adapt00', 'loadsCC.dat')
        elif os.path.isfile('loadsCC.dat'):
            # Non-adaptive run
            fCC = 'loadsCC.dat'
        else:
            # Nothing to do.
            return None
        # Read the file.
        linesCC = open(fCC).readlines()
        # Loop through the lines.
        for line in linesCC:
            # Strip line.
            line = line.strip()
            # Check for empty line or comment.
            if (not line) or line.startswith('#'): continue
            # Get name of component.
            comp = line.split()[0]
            # Add line to dictionary if necessary.
            if comp not in self.Components:
                self.Components[comp] = {'C':[], 'MRP':None}
            # Try to get the coefficient name.
            try:
                # Find text like '(C_A)' and return 'C_A'.
                c = re.search('\(([A-Za-z_]+)\)', line).group(1)
            except Exception:
                # Failed to find expected text.
                print("Warning: no coefficient name in line:\n  '%s'" % line) 
                continue
            # Filter the coefficient.
            if c == 'C_A':
                # Axial force
                self.Components[comp]['C'].append('CA')
                continue
            elif c == 'C_Y': 
                # Lateral force
                self.Components[comp]['C'].append('CY')
                continue
            elif c == 'C_N':
                # Normal force
                self.Components[comp]['C'].append('CN')
                continue
            elif c == 'C_M_x':
                # Rolling moment
                self.Components[comp]['C'].append('CLL')
            elif c == 'C_M_y':
                # Pitching moment
                self.Components[comp]['C'].append('CLM')
            elif c == 'C_M_z':
                # Yaw moment
                self.Components[comp]['C'].append('CLN')
            else:
                # Extra coefficient such as lift, drag, etc.
                continue
            # Only process reference point once.
            if self.Components[comp]['MRP'] is not None: continue
            # Try to find reference point.
            try:
                # Search for text like '(17.0, 0, 0)'.
                txt = re.search('\(([0-9., -]+)\)', line).group(1)
                # Split into coordinates.
                MRP = np.array([float(v) for v in txt.split(',')])
                # Save it.
                self.Components[comp]['MRP'] = MRP
            except Exception:
                # Failed to find expected text.
                print("Warning: no reference point in line:\n  '%s'" % line)
                
        
# Individual component force and moment
class FM(object):
    """
    Individual force and moment class
    =================================
    
    This class contains methods for reading data about an individual component. 
    The list of available components comes from a :file:`loadsCC.dat` file if
    one exists.
    
    :Call:
        >>> FM = pyCart.aero.FM(C, MRP=None, A=None)
    :Inputs:
        *aero*: :class:`pyCart.aero.Aero`
            Instance of the aero history class
        *C*: :class:`list` (:class:`str`)
            List of coefficients to initialize
        *MRP*: :class:`numpy.ndarray` (:class:`float`) shape=(3,)
            Moment reference point
        *A*: :class:`numpy.ndarray` shape=(*N*,4) or shape=(*N*,7)
            Matrix of forces and/or moments at *N* iterations
    :Outputs:
        *FM*: :class:`pyCart.aero.FM`
            Instance of the force and moment class
        *FM.C*: :class:`list` (:class:`str`)
            List of coefficients
        *FM.MRP*: :class:`numpy.ndarray` (:class:`float`) shape=(3,)
            Moment reference point
        *FM.i*: :class:`numpy.ndarray` shape=(0,)
            List of iteration numbers
        *FM.CA*: :class:`numpy.ndarray` shape=(0,)
            Axial force coefficient at each iteration
        *FM.CY*: :class:`numpy.ndarray` shape=(0,)
            Lateral force coefficient at each iteration
        *FM.CN*: :class:`numpy.ndarray` shape=(0,)
            Normal force coefficient at each iteration
        *FM.CLL*: :class:`numpy.ndarray` shape=(0,)
            Rolling moment coefficient at each iteration
        *FM.CLM*: :class:`numpy.ndarray` shape=(0,)
            Pitching moment coefficient at each iteration
        *FM.CLN*: :class:`numpy.ndarray` shape=(0,)
            Yaw moment coefficient at each iteration
    :Versions:
        * 2014-11-12 ``@ddalle``: Starter version
    """
    # Initialization method
    def __init__(self, C, MRP=None, A=None):
        """Initialization method
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Save component list.
        self.C = C
        # Initialize iteration list.
        self.i = np.array([])
        # Loop through components.
        for c in C:
            setattr(self, c, np.array([]))
        # Save the MRP.
        self.MRP = np.array(MRP)
        # Check for data.
        if A is not None:
            # Use method to parse.
            self.AddData(A)
            
    # Function to display contents
    def __repr__(self):
        """Representation method
        
        Returns one of the following:
        
            * ``'<aero.FM Force, i=100>'``
            * ``'<aero.FM Moment, i=100, MRP=(0.00, 1.00, 2.00)>'``
            * ``'<aero.FM FM, i=100, MRP=(0.00, 1.00, 2.00)>'``
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Initialize the string.
        txt = '<aero.FM '
        # Check for a moment.
        if ('CA' in self.C) and ('CLL' in self.C):
            # Force and moment.
            txt += ' FM'
        elif ('CA' in self.C):
            # Force only
            txt += ' Force'
        elif ('CLL' in self.C):
            # Moment only
            txt += ' Moment'
        # Add number of iterations.
        txt += (', i=%i' % self.i.size)
        # Add MRP if possible.
        if (self.MRP.size == 3):
            txt += (', MRP=(%.2f, %.2f, %.2f)' % tuple(self.MRP))
        # Finish the string and return it.
        return txt + '>'
        
    # String method
    def __str__(self):
        """String method
        
        Returns one of the following:
        
            * ``'<aero.FM Force, i=100>'``
            * ``'<aero.FM Moment, i=100, MRP=(0.00, 1.00, 2.00)>'``
            * ``'<aero.FM FM, i=100, MRP=(0.00, 1.00, 2.00)>'``
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        return self.__repr__()
        
            
    # Method to add data to instance
    def AddData(self, A):
        """Add iterative force and/or moment history for a component
        
        :Call:
            >>> FM.AddData(A)
        :Inputs:
            *FM*: :class:`pyCart.aero.FM`
                Instance of the force and moment class
            *A*: :class:`numpy.ndarray` shape=(*N*,4) or shape=(*N*,7)
                Matrix of forces and/or moments at *N* iterations
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Get size of A.
        n, m = A.shape
        # Save the iterations.
        self.i = A[:,0]
        # Check size.
        if m == 7:
            # Save all fields.
            self.CA = A[:,1]
            self.CY = A[:,2]
            self.CN = A[:,3]
            self.CLL = A[:,4]
            self.CLM = A[:,5]
            self.CLN = A[:,6]
        elif (self.MRP) and (m == 4):
            # Save only moments.
            self.CLL = A[:,1]
            self.CLM = A[:,2]
            self.CLN = A[:,3]
        elif (m == 4):
            # Save only forces.
            self.CA = A[:,1]
            self.CY = A[:,2]
            self.CN = A[:,3]
            
    
        
