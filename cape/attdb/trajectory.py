#!/usr/bin/env python
"""
---------------------------------------------------------
Flight Condition List Interface: :mod:`attdb.trajectory`
---------------------------------------------------------

The run matrix module provides a single class,
:class:`attdb.trajectory.Trajectory`, which provides an interface to run
matrix conditions using a class based on the built-in :class:`dict` class.

An instance of the :class:`Trajectory` class is initiated using a syntax very
similar to a :class:`dict`.

    .. code-block:: python
    
        # Import the module
        import attdb.trajectory
        
        # Use arrays from standard Python numerics module
        import numpy as np
        
        # Create some vectors of conditions
        mach = np.array([  0.50,   1.05,   1.75])
        aoav = np.array([  0.00,   4.00,   0.00])
        phiv = np.array([  0.00,  30.00,   0.00])
        q    = np.array([280.44, 597.02, 702.83])
        T    = np.array([510.41, 446.83, 377.35])
        # Create the trajectory
        x = attdb.trajectory.Trajectory(
            mach=mach, alpha_t=aoav, phi=phiv, q=q, T=T)
            
Based on the standard run matrix format used by this program, the
:class:`Trajectory` interface works best when exactly these columns are used.
However, any list of independent variables can be used (as long as they are all
arrays with the same number of entries).  The resulting instance of the
:class:`Trajectory` class stores each input variable using the same syntax as
:class:`dict` but also provides Matlab-like :class:`struct` syntax.

    .. code-block:: pycon
    
        >>> x["alpha_t"]
        array([ 0.,  4.,  0.])
        >>> x.alpha_t
        array([ 0.,  4.,  0.])
        
The use of the later syntax means that some variables (most importantly *n*)
are out of bounds.

The interface also provides a variety of other methods that help with
conversions.  For example, users can quickly access the angle of attack
(different from total angle of attack) or the freestream static pressure.  It
also has a method :func:`GetFolderNames` that returns a simple string for each
case.  These strings are used to identify cases in the line load worksheets.

    .. code-block:: pycon
    
        >>> x.GetAlpha()
        array([ 0.        ,  1.73222664,  0.        ])
        >>> x.GetPressure()
        array([ 1602.51428571,   773.59248461,   327.85072886])
        >>> x.GetFolderNames()
        ['m0.50a0.0r000.0', 'm1.05a2.0r030.0', 'm1.75a0.0r000.0']
        
The other important method is :func:`Filter`, which simplifies the process of
finding specially defined subsets of the run matrix.

    .. code-block:: pycon
    
        >>> x.Filter(["mach>1", "alpha_t<5"])
        array([1, 2])

"""

# Numerics
import numpy as np
# Regular expressions
import re

# Utilities
from . import convert

# Trajectory class
class Trajectory(dict):
    """Class to store information about run matrix
    
    :Call:
        >>> x = Trajectory(**kw)
    :Inputs:
        *mach*: {``None``} | :class:`np.ndarray`
            Mach numbers
        *alpha_t*: {``None``} | :class:`np.ndarray`
            Total angles of attack [deg]
        *phi*: {``None``} | :class:`np.ndarray`
            Roll angles [deg]
        *q*: {``None``} | :class:`np.ndarray`
            Dynamic pressure values [psf]
        *T*: {``None``} | :class:`np.ndarray`
            Temperature values [degR]
    :Outputs:
        *x*: :class:`attdb.trajectory.Trajectory`
            Run matrix from XLS interface
        *x.n*: :class:`int`
            Number of cases in the trajectory
        *x.mach*: :class:`np.ndarray`
            Mach numbers copied from *mach*
        *x["mach"]*: :class:`np.ndarray`
            Mach numbers copied from *mach*
    :Versions:
        * 2017-07-20 ``@ddalle``: Started
    """
   # ======
   # Config
   # ======
   # <
    # Initialization method
    def __init__(self, **kw):
        """Initialization method
        
        :Versions:
            * 2017-07-20 ``@ddalle``: First version
        """
        # Initialize count (for null cases)
        self.n = 0
        # Loop through list of keys
        for k in kw:
            # Get value
            v = np.asarray(kw[k])
            # Size
            self.n = len(v)
            # Set value per usual syntax
            self[k] = v
            # Set value as attribute
            setattr(self,k, v)
            
    # Display method
    def __repr__(self):
        """Representation method
        
        :Versions:
            * 2017-07-21 ``@ddalle``: First version
        """
        return "<Trajectory n=%s>" % self.n
            
    # String method
    def __str__(self):
        """Convert to string method
        
        :Versions:
            * 2017-07-21 ``@ddalle``: First version
        """
        return "<Trajectory n=%s>" % self.n
        
    # Copy
    def copy(self):
        """Create a duplicate of a run matrix interface
        
        :Call:
            >>> y = x.copy()
        :Inputs:
            *x*: :class:`trajectory.Trajectory`
                A run matrix interface
        :Outputs:
            *y*: :class:`trajectory.Trajectory`
                Copy of *x* with independent arrays
        :Versions:
            * 2018-10-03 ``@ddalle``: First version
        """
        # Create dictionary of copied arrays
        kw = {}
        # Copy each key
        for k in self.keys():
            kw[k] = self[k].copy()
        # Create output
        y = Trajectory(**kw)
        # Output
        return y
   # >
            
   # ===============
   # Case Management
   # ===============
   # <
    # Get case name
    def GetFolderNames(self, I=None):
        """Construct folder name for one or more cases
        
        :Call:
            >>> frun = x.GetFolderNames(i)
            >>> fruns = x.GetFolderNames(I=None)
        :Inputs:
            *x*: :class:`attdb.trajectory.Trajectory`
                Run matrix conditions interface
            *i*: :class:`int`
                Single case index
            *I*: ``None`` | :class:`np.ndarray` (:class:`int`)
                List of cases, or return for all cases if ``None``
        :Outputs:
            *fdir*: :class:`str`
                Name of case *i*
            *fdirs*: :class:`list` (:class:`str`)
                List of case names
        :Versions:
            * 2017-07-20 ``@ddalle``: First version
        """
        # Number of cases
        self.n = len(self["mach"])
        # Check for default input
        if I is None:
            # Default: all cases
            I = np.arange(self.n)
        # Save original type
        t = I.__class__.__name__
        # Convert to array
        I = np.array(I).flatten()
        # Initialize output
        fruns = []
        # Loop through cases
        for i in I:
            # Create the name
            frun = "m%.2fa%.1fr%05.1f" % (
                self.GetMach(i),
                self.GetAlphaTotal(i),
                self.GetPhi(i))
            # Append to lest
            fruns.append(frun)
        # Check for scalar input
        if t not in ['list', 'ndarray']:
            # Singleton input
            return frun
        else:
            return fruns
            
    # Get case name, alphabeta
    def GetFolderNamesAB(self, I=None):
        """Construct folder name for one or more cases
        
        :Call:
            >>> frun = x.GetFolderNamesAB(i)
            >>> fruns = x.GetFolderNamesAB(I=None)
        :Inputs:
            *x*: :class:`attdb.trajectory.Trajectory`
                Run matrix conditions interface
            *i*: :class:`int`
                Single case index
            *I*: ``None`` | :class:`np.ndarray` (:class:`int`)
                List of cases, or return for all cases if ``None``
        :Outputs:
            *fdir*: :class:`str`
                Name of case *i*
            *fdirs*: :class:`list` (:class:`str`)
                List of case names
        :Versions:
            * 2017-07-20 ``@ddalle``: First version
        """
        # Number of cases
        self.n = len(self["mach"])
        # Check for default input
        if I is None:
            # Default: all cases
            I = np.arange(self.n)
        # Save original type
        t = I.__class__.__name__
        # Convert to array
        I = np.array(I).flatten()
        # Initialize output
        fruns = []
        # Loop through cases
        for i in I:
            # Create the name
            frun = "m%.2fa%.2fb%.2f" % (
                self.GetMach(i),
                self.GetAlpha(i),
                self.GetBeta(i))
            # Append to lest
            fruns.append(frun)
        # Check for scalar input
        if t not in ['list', 'ndarray']:
            # Singleton input
            return frun
        else:
            return fruns
   # >
   
   # =========
   # Filtering
   # =========
   # <
    # Find a value
    def GetKey(self, k, I=None):
        """Get value from a trajectory key, including specially named keys
        
        :Call:
            >>> V = x.GetKey(k)
            >>> V = x.GetKey(k, I)
            >>> v = x.GetKey(k, i)
        :Inputs:
            *x*: :class:`attdb.trajectory.Trajectory`
                Run matrix conditions interface
            *k*: :class:`str`
                Trajectory key name
            *i*: :class:`int`
                Case index
            *I*: :class:`np.ndarray` (:class:`int`)
                Array of case indices
        :Outputs:
            *V*: :class:`np.ndarray`
                Array of values from one or more cases
            *v*: :class:`np.any`
                Value for individual case *i*
        :Versions:
            * 2018-10-03 ``@ddalle``: First version
        """
        if k in self:
            # The key is present directly
            V = self[k]
            # Index input type
            t = I.__class__.__name__
            # Process indices
            if I is None:
                # Return entire array
                pass
            else:
                # Subset
                V = V[I]
        elif k.lower() in ["aoa", "alpha"]:
            # Angle of attack
            V = self.GetAlpha(I)
        elif k.lower() in ["aos", "beta"]:
            # Sideslip angle
            V = self.GetBeta(I)
        elif k.lower() in ["alpha_t", "aoap"]:
            # Total angle of attack
            V = self.GetAlphaTotal(I)
        elif k.lower() in ["phi", "phip"]:
            # Velocity roll angle
            V = self.GetPhi(I)
        elif k.lower() in ["alpha_m", "aoav"]:
            # Total angle of attack
            V = self.GetAlphaManeuver(I)
        elif k.lower() in ["phi_m", "phim", "phiv"]:
            # Velocity roll angle
            V = self.GetPhiManeuver(I)
        elif k in ["q"]:
            # Dynamic pressure
            V = self.GetDynamicPressure(I)
        elif k in ["p", "pinf"]:
            # Static pressure
            V = self.GetPressure(I)
        # Output
        return V
   
    # Find a match
    def FindMatches(self, y, i, keys=None, **kw):
        """Find index or indices of cases matching another trajectory case
        
        :Call:
            >>> I = x.FindMatches(y, i, keys=None)
        :Inputs:
            *x*: :class:`attdb.trajectory.Trajectory`
                Run matrix conditions interface
            *y*: :class:`attdb.trajectory.Trajectory`
                Target run matrix conditions interface
            *i*: :class:`int`
                Case number of case in *y*
            *keys*: {``None``} | :class:`list` (:class:`str`)
                List of keys to test for equivalence
            *tol*: {``1e-8``} | :class:`float` >= 0
                Tolerance for two values to be ideintal
            *machtol*: {*tol*} | :class:`float` >= 0
                Tolerance for *mach* key, for instance
        :Outputs:
            *I*: :class:`np.ndarray` (:class:`int`)
                List of indices matching all constraints
        :Versions:
            * 2017-07-21 ``@ddalle``: First version
        """
        # Key list
        if keys is None:
            # Use all
            keys = self.keys()
        # Default tolerance
        tol = kw.get("tol", 1e-8)
        # Initialize list
        I = np.ones(self.n, dtype="bool")
        # Loop through keys
        for k in keys:
            # Check for specific tolerance
            ktol = kw.get("%stol"%k, tol)
            # Get values from this trajectory
            V = self.GetKey(k)
            # Get values from target trajectory
            v = y.GetKey(k, i)
            # Apply constraints
            I = np.logical_and(I, np.abs(V-v)<=ktol)
        # Output
        return np.where(I)[0]
        
    # Find the first match
    def FindMatch(self, y, i, keys=None, **kw):
        """Find the first case index (if any) matching another trajectory case
        
        :Call:
            >>> j = x.FindMatch(y, i, keys=None)
        :Inputs:
            *x*: :class:`attdb.trajectory.Trajectory`
                Run matrix conditions interface
            *y*: :class:`attdb.trajectory.Trajectory`
                Target run matrix conditions interface
            *i*: :class:`int`
                Case number of case in *y*
            *keys*: {``None``} | :class:`list` (:class:`str`)
                List of keys to test for equivalence
            *tol*: {``1e-8``} | :class:`float` >= 0
                Tolerance for two values to be ideintal
            *machtol*: {*tol*} | :class:`float` >= 0
                Tolerance for *mach* key, for instance
        :Outputs:
            *j*: ``None`` | :class:`int`
                Index of first matching case, if any
        :Versions:
            * 2017-07-21 ``@ddalle``: First version
        """
        # Find all matches
        I = self.FindMatches(y, i, keys=keys, **kw)
        # Check for a match
        if len(I) > 0:
            # Use first match
            return I[0]
        else:
            # Use ``None`` to indicate no match
            return None
            
    # Function to filter cases
    def Filter(self, cons, I=None):
        """Filter cases according to a set of constraints

        The constraints are specified as a list of strings that contain
        inequalities of variables that are in *x.keys*.

        For example, if *m* is the name of a key (presumably meaning Mach
        number), and *a* is a variable presumably representing angle of attack,
        the following example finds the indices of all cases with Mach number
        greater than 1.5 and angle of attack equal to ``2.0``.

            >>> i = x.Filter(['m>1.5', 'a==2.0'])

        A warning will be produces if one of the constraints does not correspond
        to a trajectory variable or cannot be evaluated for any other reason.

        :Call:
            >>> i = x.Filter(cons)
            >>> i = x.Fitler(cons, I)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *cons*: :class:`list` (:class:`str`)
                List of constraints
            *I*: :class:`list` (:class:`int`)
                List of initial indices to consider
        :Outputs:
            *i*: :class:`numpy.ndarray` (:class:`int`)
                List of indices that match constraints
        :Versions:
            * 2014-12-09 ``@ddalle``: First version
        """
        # Initialize the conditions.
        if I is None:
            # Consider all indices
            i = np.arange(self.n) > -1
        else:
            # Start with all indices failed.
            i = np.arange(self.n) < -1
            # Set the specified indices to True
            i[I] = True
        # Check for None
        if cons is None: cons = []
        # Loop through constraints
        for con in cons:
            # Check for empty constraints.
            if len(con.strip()) == 0: continue
            # Check for escape characters
            if re.search('[\n]', con):
                print("Constraint %s contains escape character; skipping")
                continue
            # Substitute '=' -> '==' while leaving '==', '<=', '>=', '!=' 
            con = re.sub("(?<![<>=!~])=(?!=)", "==", con)
            # Replace variable names with calls to GetValue()
            # But don't replace functions
            #     sin(phi)      --> sin(self.GetValue('phi'))
            #     np.sin(phi)   --> np.sin(self.GetValue('phi'))
            #     sin(self.phi) --> sin(self.phi)
            #     user=="@user" --> self.GetValue('user')=="@user"
            con = re.sub(
                r"(?<!['\"@\w.])([A-Za-z_]\w*)(?![\w(.'\"])",
                r"self.GetValue('\1')", con)
            # Replace any raw function calls with numpy ones
            con = re.sub(
                r"(?<![\w.])([A-Za-z_]\w*)(?=\()",
                r"np.\1", con)
            # Constraint may fail with bad input.
            try:
                # Apply the constraint.
                i = np.logical_and(i, eval(con))
            except Exception:
                # Print a warning and move on.
                print("Constraint '%s' failed to evaluate." % con)
        # Output
        return np.where(i)[0]
    
    # Function to get sweep based on constraints
    def GetSweep(self, i0=0, **kw):
        """Return a list of indices meeting sweep constraints
        
        For example, using ``EqCons=['mach']`` will cause the method to return
        points with *x.mach* matching *x.mach[i0]*.
        
        :Call:
            >>> I = x.GetSweep(i0=0, **kw)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *i0*: {``0``} | :class:`int` > 0
                Index to be included in the sweep
            *SortVar*: :class:`str`
                Variable by which to sort each sweep
            *EqCons*: :class:`list` (:class:`str`)
                List of trajectory keys which must match (exactly) the first
                point in the sweep
            *TolCons*: :class:`dict` (:class:`float`)
                Dictionary whose keys are trajectory keys which must match the
                first point in the sweep to a specified tolerance and whose
                values are the specified tolerances
            *IndexTol*: :class:`int`
                If specified, only trajectory points in the range
                ``[i0,i0+IndexTol]`` are considered for the sweep
        :Outputs:
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of trajectory point indices in the sweep
        :Versions:
            * 2015-05-24 ``@ddalle``: First version
            * 2017-06-27 ``@ddalle``: Added special variables
            * 2018-06-22 ``@ddalle``: 
        """
        # Mask
        m = np.arange(self.n) > -1
        # Sort key.
        xk = kw.get('SortVar')
        # Get constraints
        EqCons  = kw.get('EqCons',  [])
        TolCons = kw.get('TolCons', {})
        # Ensure no NoneType
        if EqCons  is None: EqCons = []
        if TolCons is None: TolCons = {}
        # Loop through equality constraints.
        for c in EqCons:
            # Get the key (for instance if matching ``k%10``)
            k = re.split('[^a-zA-Z_]', c)[0]
            # Check for the key.
            if k in self.keys():
                # Get the target value.
                x0 = getattr(self,k)[i0]
                # Get the value
                V = eval('self.%s' % c)
            elif k == "alpha":
                # Get the target value.
                x0 = self.GetAlpha(i0)
                # Extract matrix values
                V = self.GetAlpha()
            elif k == "beta": 
                # Get the target value
                x0 = self.GetBeta(i0)
                # Extract matrix values
                V = self.GetBeta()
            elif k in ["alpha_m", "aoam"]:
                # Get the target value.
                x0 = self.GetAlphaManeuver(i0)
                # Extract matrix values
                V = self.GetAlphaManeuver()
            elif k in ["phi_m", "phim"]:
                # Get the target value.
                x0 = self.GetPhiManeuver(i0)
                # Extract matrix values
                V = self.GetPhiManeuver()
            else:
                raise KeyError(
                    "Could not find trajectory key for constraint '%s'." % c)
            # Evaluate constraint
            m = np.logical_and(m, np.abs(V - x0) <= 1e-10)
        # Loop through tolerance-based constraints.
        for c in TolCons:
            # Get the key (for instance if matching 'i%10', key is 'i')
            k = re.split('[^a-zA-Z_]', c)[0]
            # Get tolerance.
            tol = TolCons[c]
            # Check for the key.
            if k in self.keys():
                # Get the target value.
                x0 = getattr(self,k)[i0]
                # Get the values
                V = eval('self.%s' % c)
            elif k == "alpha":
                # Get the target value.
                x0 = self.GetAlpha(i0)
                # Get trajectory values
                V = self.GetAlpha()
            elif k == "beta":
                # Get the target value
                x0 = self.GetBeta(i0)
                # Get trajectory values
                V = self.GetBeta()
            elif k in ["alpha_m", "aoam"]:
                # Get the target value.
                x0 = self.GetAlphaManeuver(i0)
                # Extract matrix values
                V = self.GetAlphaManeuver()
            elif k in ["phi_m", "phim"]:
                # Get the target value.
                x0 = self.GetPhiManeuver(i0)
                # Extract matrix values
                V = self.GetPhiManeuver()
            else:
                raise KeyError(
                    "Could not find trajectory key for constraint '%s'." % c)
            # Evaluate constraint
            m = np.logical_and(m, np.abs(x0-V) <= tol)
        # Initialize output.
        I = np.arange(self.n)
        # Apply the final mask.
        J = I[m]
        # Check for a sort variable.
        if (xk is not None):
            # Sort based on that key.
            if xk in self.keys:
                # Sort based on trajectory key
                vx = getattr(self,xk)[J]
            elif xk.lower() in ["alpha"]:
                # Sort based on angle of attack
                vx = self.GetAlpha(J)
            elif xk.lower() in ["beta"]:
                # Sort based on angle of sideslip
                vx = self.GetBeta(J)
            elif xk.lower() in ["alpha_t", "aoav"]:
                # Sort based on total angle of attack
                vx = self.GetAlphaTotal(J)
            elif xk.lower() in ["alpha_m", "aoam"]:
                # Sort based on total angle of attack
                vx = self.GetAlphaManeuver(J)
            elif xk.lower() in ["phi_m", "phim"]:
                # Sort based on velocity roll
                vx = self.GetPhiManeuver(J)
            elif xk.lower() in ["phi", "phiv"]:
                # Sort based on velocity roll
                vx = self.GetPhi(J)
            else:
                # Unable to sort
                raise ValueError("Unable to sort based on variable '%s'" % xk)
            # Order
            j = np.argsort(vx)
            # Sort the indices.
            J = J[j]
        # Output
        return J
        
    # Function to get sweep based on constraints
    def GetSweepMask(self, M, **kw):
        """
        Return a list of indices meeting sweep constraints
        
        The sweep uses the index of the first entry of ``True`` in *M*, i.e.
        ``i0=np.where(M)[0][0]``.  Then the sweep contains all other points that
        meet all criteria with respect to trajectory point *i0*.
        
        For example, using ``EqCons=['mach']`` will cause the method to return
        points with *x.mach* matching *x.mach[i0]*.
        
        :Call:
            >>> I = x.GetSweepMask(M, **kw)
        :Inputs:
            *x*: :class:`cape.trajectory.Trajectory`
                Instance of the pyCart trajectory class
            *M*: :class:`numpy.ndarray` (:class:`bool`)
                Mask of which trajectory points should be considered
            *SortVar*: :class:`str`
                Variable by which to sort each sweep
            *EqCons*: :class:`list` (:class:`str`)
                List of trajectory keys which must match (exactly) the first
                point in the sweep
            *TolCons*: :class:`dict` (:class:`float`)
                Dictionary whose keys are trajectory keys which must match the
                first point in the sweep to a specified tolerance and whose
                values are the specified tolerances
            *IndexTol*: :class:`int`
                If specified, only trajectory points in the range
                ``[i0,i0+IndexTol]`` are considered for the sweep
        :Outputs:
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of trajectory point indices in the sweep
        :Versions:
            * 2015-05-24 ``@ddalle``: First version
            * 2017-06-27 ``@ddalle``: Added special variables
        """
        # Check for an *i0* point.
        if not np.any(M): return np.array([])
        # Copy the mask.
        m = M.copy()
        # Sort key.
        xk = kw.get('SortVar')
        # Get constraints
        EqCons  = kw.get('EqCons',  [])
        TolCons = kw.get('TolCons', {})
        # Ensure no NoneType
        if EqCons  is None: EqCons = []
        if TolCons is None: TolCons = {}
        # Get the first index.
        i0 = np.where(M)[0][0]
        # Loop through equality constraints.
        for c in EqCons:
            # Get the key (for instance if matching ``k%10``)
            k = re.split('[^a-zA-Z_]', c)[0]
            # Check for the key.
            if k in self.keys():
                # Get the target value.
                x0 = getattr(self,k)[i0]
                # Get the value
                V = eval('self.%s' % c)
            elif k == "alpha":
                # Get the target value.
                x0 = self.GetAlpha(i0)
                # Extract matrix values
                V = self.GetAlpha()
            elif k == "beta": 
                # Get the target value
                x0 = self.GetBeta(i0)
                # Extract matrix values
                V = self.GetBeta()
            elif k in ["alpha_m", "aoam"]:
                # Get the target value.
                x0 = self.GetAlphaManeuver(i0)
                # Extract matrix values
                V = self.GetAlphaManeuver()
            elif k in ["phi_m", "phim"]:
                # Get the target value.
                x0 = self.GetPhiManeuver(i0)
                # Extract matrix values
                V = self.GetPhiManeuver()
            else:
                raise KeyError(
                    "Could not find trajectory key for constraint '%s'." % c)
            # Evaluate constraint
            m = np.logical_and(m, np.abs(V - x0) <= 1e-10)
        # Loop through tolerance-based constraints.
        for c in TolCons:
            # Get the key (for instance if matching 'i%10', key is 'i')
            k = re.split('[^a-zA-Z_]', c)[0]
            # Get tolerance.
            tol = TolCons[c]
            # Check for the key.
            if k in self.keys():
                # Get the target value.
                x0 = getattr(self,k)[i0]
                # Get the values
                V = eval('self.%s' % c)
            elif k == "alpha":
                # Get the target value.
                x0 = self.GetAlpha(i0)
                # Get trajectory values
                V = self.GetAlpha()
            elif k == "beta":
                # Get the target value
                x0 = self.GetBeta(i0)
                # Get trajectory values
                V = self.GetBeta()
            elif k in ["alpha_m", "aoam"]:
                # Get the target value.
                x0 = self.GetAlphaManeuver(i0)
                # Extract matrix values
                V = self.GetAlphaManeuver()
            elif k in ["phi_m", "phim"]:
                # Get the target value.
                x0 = self.GetPhiManeuver(i0)
                # Extract matrix values
                V = self.GetPhiManeuver()
            else:
                raise KeyError(
                    "Could not find trajectory key for constraint '%s'." % c)
            # Evaluate constraint
            m = np.logical_and(m, np.abs(x0-V) <= tol)
        # Initialize output.
        I = np.arange(self.n)
        # Apply the final mask.
        J = I[m]
        # Check for a sort variable.
        if (xk is not None):
            # Sort based on that key.
            if xk in self.keys:
                # Sort based on trajectory key
                vx = getattr(self,xk)[J]
            elif xk.lower() in ["alpha"]:
                # Sort based on angle of attack
                vx = self.GetAlpha(J)
            elif xk.lower() in ["beta"]:
                # Sort based on angle of sideslip
                vx = self.GetBeta(J)
            elif xk.lower() in ["alpha_t", "aoav"]:
                # Sort based on total angle of attack
                vx = self.GetAlphaTotal(J)
            elif xk.lower() in ["alpha_m", "aoam"]:
                # Sort based on total angle of attack
                vx = self.GetAlphaManeuver(J)
            elif xk.lower() in ["phi_m", "phim"]:
                # Sort based on velocity roll
                vx = self.GetPhiManeuver(J)
            elif xk.lower() in ["phi", "phiv"]:
                # Sort based on velocity roll
                vx = self.GetPhi(J)
            else:
                # Unable to sort
                raise ValueError("Unable to sort based on variable '%s'" % xk)
            # Order
            j = np.argsort(vx)
            # Sort the indices.
            J = J[j]
        # Output
        return J
    
    # Function to get set of sweeps based on criteria
    def GetSweeps(self, **kw):
        """
        Return a list of index sets in which each list contains cases that
        satisfy specified criteria.
        
        For example, using ``EqCons=['mach']`` will cause the method to return
        lists of points with the same Mach number.
        
        :Call:
            >>> J = x.GetSweeps(**kw)
        :Inputs:
            *x*: :class:`attdb.trajectory.Trajectory`
                Run matrix conditions interface
            *cons*: :class:`list` (:class:`str`)
                List of global constraints; only points satisfying these
                constraints will be in one of the output sweeps
            *I*: :class:`numpy.ndarray` (:class:`int`)
                List of indices to restrict to
            *SortVar*: :class:`str`
                Variable by which to sort each sweep
            *EqCons*: :class:`list` (:class:`str`)
                List of trajectory keys which must match (exactly) the first
                point in the sweep
            *TolCons*: :class:`dict` (:class:`float`)
                Dictionary whose keys are trajectory keys which must match the
                first point in the sweep to a specified tolerance and whose
                values are the specified tolerances
            *IndexTol*: :class:`int`
                If specified, only trajectory points in the range
                ``[i0,i0+IndexTol]`` are considered for the sweep
        :Outputs:
            *J*: :class:`list` (:class:`numpy.ndarray` (:class:`int`))
                List of trajectory point sweeps
        :Versions:
            * 2015-05-25 ``@ddalle``: First version
        """
        # Expand global index constraints.
        I0 = self.Filter(cons=kw.get('cons'))
        # Save number of cases
        self.n = len(self.mach)
        # Check for index input
        I = kw.get("I")
        # Check for nontrivial indices
        if I is not None:
            # Combine constraints
            I0 = np.intersect1d(I, I0)
        # Initialize mask (list of ``True`` with *n* entries)
        M = np.arange(self.n) < 0
        # Set the mask to ``True`` for any cases passing global constraints.
        M[I0] = True
        # Initialize output.
        J = []
        # Initialize target output
        JT = []
        # Safety check: no more than *n* sets.
        i = 0
        # Loop through cases.
        while np.any(M) and i<self.n:
            # Increase number of sweeps.
            i += 1
            # Get the current sweep.
            I = self.GetSweepMask(M, **kw)
            # Save the sweep.
            J.append(I)
            # Update the mask.
            M[I] = False
        # Output
        return J
        
   # >
   
   # ==========
   # Conditions
   # ==========
   # <
    # Get the Mach number
    def GetMach(self, I=None):
        """Return Mach number for one or more cases
        
        :Call:
            >>> mach = x.GetMach(I=None)
        :Inputs:
            *x*: :class:`attdb.trajectory.Trajectory`
                Run matrix conditions interface
            *I*: {``None``} | :class:`int` | :class:`np.ndarray`
                Case index or list of indices
        :Outputs:
            *mach*: :class:`float`
                Mach number(s)
        :Versions:
            * 2017-07-20 ``@ddalle``: First version
        """
        # Number of cases
        self.n = len(self.mach)
        # Default input
        if I is None:
            # Default: all cases
            I = np.arange(self.n)
        # Get values
        m = self.get("mach", np.zeros(0))
        # Output
        return m[I]
        
    # Get the total angle of attack
    def GetAlphaTotal(self, I=None):
        """Return total angle of attack for one or more cases
        
        :Call:
            >>> aoav = x.GetAlphaTotal(I=None)
        :Inputs:
            *x*: :class:`attdb.trajectory.Trajectory`
                Run matrix conditions interface
            *I*: {``None``} | :class:`int` | :class:`np.ndarray`
                Case index or list of indices
        :Outputs:
            *aoav*: :class:`float`
                Total angle(s) of attack [deg]
        :Versions:
            * 2017-07-20 ``@ddalle``: First version
        """
        # Number of cases
        self.n = len(self.mach)
        # Default input
        if I is None:
            # Default: all cases
            I = np.arange(self.n)
        # Get values
        a = self.get("alpha_t", np.zeros(0))
        # Output
        return a[I]
        
    # Get the roll angle
    def GetPhi(self, I=None):
        """Return velocity roll angle for one or more cases
        
        :Call:
            >>> phiv = x.GetPhi(I=None)
        :Inputs:
            *x*: :class:`attdb.trajectory.Trajectory`
                Run matrix conditions interface
            *I*: {``None``} | :class:`int` | :class:`np.ndarray`
                Case index or list of indices
        :Outputs:
            *phiv*: :class:`float` | :class:`np.ndarray`
                Roll angle[s] [deg]
        :Versions:
            * 2017-07-20 ``@ddalle``: First version
        """
        # Number of cases
        self.n = len(self.mach)
        # Default input
        if I is None:
            # Default: all cases
            I = np.arange(self.n)
        # Get values
        phiv = self.get("phi", np.zeros(0))
        # Output
        return phiv[I]
        
    # Get the dynamic pressure
    def GetDynamicPressure(self, I=None):
        """Return dynamic pressure for one or more cases
        
        :Call:
            >>> q = x.GetDynamicPressure(I=None)
        :Inputs:
            *x*: :class:`attdb.trajectory.Trajectory`
                Run matrix conditions interface
            *I*: {``None``} | :class:`int` | :class:`np.ndarray`
                Case index or list of indices
        :Outputs:
            *q*: :class:`float` | :class:`np.ndarray`
                Dynamic pressure(s) [psf]
        :Versions:
            * 2017-07-21 ``@ddalle``: First version
        """
        # Number of cases
        self.n = len(self.mach)
        # Default input
        if I is None:
            # Default: all cases
            I = np.arange(self.n)
        # Output
        return self.q[I]
        
    # Get the dynamic pressure
    def GetTemperature(self, I=None):
        """Return static temperature for one or more cases
        
        :Call:
            >>> T = x.GetTemperature(I=None)
        :Inputs:
            *x*: :class:`attdb.trajectory.Trajectory`
                Run matrix conditions interface
            *I*: {``None``} | :class:`int` | :class:`np.ndarray`
                Case index or list of indices
        :Outputs:
            *T*: :class:`float` | :class:`np.ndarray`
                Dynamic pressure(s) [psf]
        :Versions:
            * 2017-07-21 ``@ddalle``: First version
        """
        # Number of cases
        self.n = len(self.mach)
        # Default input
        if I is None:
            # Default: all cases
            I = np.arange(self.n)
        # Output
        return self.T[I]
        
    # Get static pressure
    def GetPressure(self, I=None):
        """Return dynamic pressure for one or more cases
        
        :Call:
            >>> p = x.GetPressure(I=None)
        :Inputs:
            *x*: :class:`attdb.trajectory.Trajectory`
                Run matrix conditions interface
            *I*: {``None``} | :class:`int` | :class:`np.ndarray`
                Case index or list of indices
        :Outputs:
            *p*: :class:`float` | :class:`np.ndarray`
                Static pressure(s) [psf]
        :Versions:
            * 2017-07-21 ``@ddalle``: First version
        """
        # Number of cases
        self.n = len(self.mach)
        # Default input
        if I is None:
            # Default: all cases
            I = np.arange(self.n)
        # Get values
        q = self.q[I]
        M = self.mach[I]
        # Output
        return q / (0.7*M*M)
        
        
    # Get Reynolds number
    def GetReynoldsNumber(self, I=None):
        """Return dynamic pressure for one or more cases
        
        :Call:
            >>> p = x.GetReynoldsNumber(I=None)
        :Inputs:
            *x*: :class:`attdb.trajectory.Trajectory`
                Run matrix conditions interface
            *I*: {``None``} | :class:`int` | :class:`np.ndarray`
                Case index or list of indices
        :Outputs:
            *Re*: :class:`float` | :class:`np.ndarray`
                Reynolds number(s) [1/in]
        :Versions:
            * 2017-07-21 ``@ddalle``: First version
        """
        # Number of cases
        self.n = len(self.mach)
        # Default input
        if I is None:
            # Default: all cases
            I = np.arange(self.n)
        # Get values
        q = self.q[I]
        T = self.T[I]
        M = self.mach[I]
        # Calculate pressure
        p = q / (0.7*M*M)
        # Output
        return convert.ReynoldsPerFoot(p, T, M)/12
        
        
    # Get the angle of attack
    def GetAlpha(self, I=None):
        """Return angle of attack for one or more cases
        
        :Call:
            >>> alpha = x.GetAlpha(I=None)
        :Inputs:
            *x*: :class:`attdb.trajectory.Trajectory`
                Run matrix conditions interface
            *I*: {``None``} | :class:`int` | :class:`np.ndarray`
                Case index or list of indices
        :Outputs:
            *alpha*: :class:`float`
                Angle(s) of attack [deg]
        :Versions:
            * 2017-07-20 ``@ddalle``: First version
        """
        # Number of cases
        self.n = len(self.mach)
        # Default input
        if I is None:
            # Default: all cases
            I = np.arange(self.n)
        # Get values
        aoav = self.alpha_t[I]
        phiv = self.phi[I]
        # Convert
        a, b = convert.AlphaTPhi2AlphaBeta(aoav, phiv)
        # Output
        return a
        
    # Get the angle of attack
    def GetBeta(self, I=None):
        """Return angle of sideslip for one or more cases
        
        :Call:
            >>> beta = x.GetBeta(I=None)
        :Inputs:
            *x*: :class:`attdb.trajectory.Trajectory`
                Run matrix conditions interface
            *I*: {``None``} | :class:`int` | :class:`np.ndarray`
                Case index or list of indices
        :Outputs:
            *eta*: :class:`float`
                Angle(s) of sideslip [deg]
        :Versions:
            * 2017-07-20 ``@ddalle``: First version
        """
        # Number of cases
        self.n = len(self.mach)
        # Default input
        if I is None:
            # Default: all cases
            I = np.arange(self.n)
        # Get values
        aoav = self.alpha_t[I]
        phiv = self.phi[I]
        # Convert
        a, b = convert.AlphaTPhi2AlphaBeta(aoav, phiv)
        # Output
        return b
        
    # Get the total angle of attack
    def GetAlphaManeuver(self, I=None):
        """Return total angle of attack for one or more cases
        
        :Call:
            >>> aoam = x.GetAlphaManeuver(I=None)
        :Inputs:
            *x*: :class:`attdb.trajectory.Trajectory`
                Run matrix conditions interface
            *I*: {``None``} | :class:`int` | :class:`np.ndarray`
                Case index or list of indices
        :Outputs:
            *aoam*: :class:`float`
                Maneuver angle(s) of attack [deg]
        :Versions:
            * 2017-07-21 ``@ddalle``: First version
        """
        # Number of cases
        self.n = len(self.mach)
        # Default input
        if I is None:
            # Default: all cases
            I = np.arange(self.n)
        # Get values
        aoav = self.alpha_t[I]
        phiv = self.phi[I]
        # Convert
        aoam, phim = convert.AlphaTPhi2AlphaMPhi(aoav, phiv)
        # Output
        return aoam
        
    # Get the roll angle
    def GetPhiManeuver(self, I=None):
        """Return velocity roll angle for one or more cases
        
        :Call:
            >>> phim = x.GetPhiManeuver(I=None)
        :Inputs:
            *x*: :class:`attdb.trajectory.Trajectory`
                Run matrix conditions interface
            *I*: {``None``} | :class:`int` | :class:`np.ndarray`
                Case index or list of indices
        :Outputs:
            *phim*: :class:`float`
                Maneuver roll angle[s] [deg]
        :Versions:
            * 2017-07-21 ``@ddalle``: First version
        """
        # Number of cases
        self.n = len(self.mach)
        # Default input
        if I is None:
            # Default: all cases
            I = np.arange(self.n)
        # Get values
        aoav = self.alpha_t[I]
        phiv = self.phi[I]
        # Convert
        aoam, phim = convert.AlphaTPhi2AlphaMPhi(aoav, phiv)
        # Output
        return phim
   # >
  
  # ================
  # Value Extraction
  # ================
  # <
    # Find a value
    def GetValue(self, k, I=None):
        """Get value from a trajectory key, including specially named keys
        
        :Call:
            >>> V = x.GetValue(k)
            >>> V = x.GetValue(k, I)
            >>> v = x.GetValue(k, i)
        :Inputs:
            *x*: :class:`attdb.trajectory.Trajectory`
                Run matrix conditions interface
            *k*: :class:`str`
                Trajectory key name
            *i*: :class:`int`
                Case index
            *I*: :class:`np.ndarray` (:class:`int`)
                Array of case indices
        :Outputs:
            *V*: :class:`np.ndarray`
                Array of values from one or more cases
            *v*: :class:`np.any`
                Value for individual case *i*
        :Versions:
            * 2018-10-03 ``@ddalle``: First version
        """
        if k in self.keys():
            # The key is present directly
            V = getattr(self,k)
            # Index input type
            t = I.__class__.__name__
            # Process indices
            if I is None:
                # Return entire array
                pass
            else:
                # Subset
                V = V[I]
        elif k.lower() in ["aoa", "alpha"]:
            # Angle of attack
            V = self.GetAlpha(I)
        elif k.lower() in ["aos", "beta"]:
            # Sideslip angle
            V = self.GetBeta(I)
        elif k.lower() in ["alpha_t", "aoap"]:
            # Total angle of attack
            V = self.GetAlphaTotal(I)
        elif k.lower() in ["phi", "phip"]:
            # Velocity roll angle
            V = self.GetPhi(I)
        elif k.lower() in ["alpha_m", "aoav"]:
            # Total angle of attack
            V = self.GetAlphaManeuver(I)
        elif k.lower() in ["phi_m", "phim", "phiv"]:
            # Velocity roll angle
            V = self.GetPhiManeuver(I)
        elif k in ["q"]:
            # Dynamic pressure
            V = self.GetDynamicPressure(I)
        elif k in ["p", "pinf"]:
            # Static pressure
            V = self.GetPressure(I)
        else:
            # Evaluate an expression, for example "mach%1.0"
            V = eval('self.' + k)
            # Index input type
            t = I.__class__.__name__
            # Process indices
            if I is None:
                # Return entire array
                pass
            else:
                # Subset
                V = V[I]
        # Output
        return V
  # >
# class Trajectory
        
