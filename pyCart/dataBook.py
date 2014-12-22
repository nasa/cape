"""
Data Book Module: :mod:`pyCart.dataBook`
========================================

This module contains functions for reading and processing forces, moments, and
other statistics from a trajectory

:Versions:
    * 2014-12-20 ``@ddalle``: Started
"""

# File interface
import os
# Basic numerics
import numpy as np
# Advanced text (regular expressions)
import re
# Date processing
from datetime import datetime

# Use this to only update entries with newer iterations.
from .case import GetCurrentIter

#<!--
# ---------------------------------
# I consider this portion temporary

# Get the umask value.
umask = 0027
# Get the folder permissions.
fmask = 0777 - umask
dmask = 0777 - umask

# ---------------------------------
#-->


# Aerodynamic history class
class DataBook(dict):
    """
    This class provides an interface to the data book for a given CFD run
    matrix.
    
    :Call:
        >>> DB = pyCart.dataBook.DataBook(x, opts)
    :Inputs:
        *x*: :class:`pyCart.trajectory.Trajectory`
            The current pyCart trajectory (i.e. run matrix)
        *opts*: :class:`pyCart.options.Options`
            Global pyCart options instance
    :Outputs:
        *DB*: :class:`pyCart.dataBook.DataBook`
            Instance of the pyCart data book class
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
    """
    
    # Initialization method
    def __init__(self, x, opts):
        """Initialization method
        
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Save the root directory.
        self.RootDir = os.getcwd()
        # Save the components
        self.Components = opts.get_DataBookComponents()
        # Save the folder
        self.Dir = opts.get_DataBookDir()
        # Save the trajectory.
        self.x = x
        # Save the options.
        self.opts = opts
        # Make sure the destination folder exists.
        for fdir in self.Dir.split('/'):
            # Check if the folder exists.
            if not os.path.isdir(fdir):
                os.mkdir(fdir, dmask)
            # Go to the folder.
            os.chdir(fdir)
        # Go back to root folder.
        os.chdir(self.RootDir)
        # Loop through the components.
        for comp in self.Components:
            # Initialize the data book.
            self[comp] = DBComp(comp, x, opts, fdir)
        # Initialize targets.
        self.Targets = []
        # Read the targets.
        for targ in opts.get_DataBookTargets():
            # Read the file.
            self.Targets.append(DBTarget(targ, opts))
            
    # Write the data book
    def Write(self):
        """Write the current data book in Python memory to file
        
        :Call:
            >>> DB.Write()
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
        """
        # Loop through the components.
        for comp in self.Components:
            self[comp].Write()
            
    # Update data book
    def UpdateDataBook(self, I):
        """Update the data book for a list of cases from the run matrix
        
        :Call:
            >>> DB.UpdateDataBook(I)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *I*: :class:`list` (:class:`int`)
                List of trajectory indices
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
        """
        # Loop through indices.
        for i in I:
            self.UpdateCase(i)
        
            
    # Update or add an entry.
    def UpdateCase(self, i):
        """Update or add a trajectory to a data book
        
        The history of a run directory is processed if either one of three
        criteria are met.
        
            1. The case is not already in the data book
            2. The most recent iteration is greater than the data book value
            3. The number of iterations used to create statistics has changed
        
        :Call:
            >>> DB.UpdateCase(i)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *i*: :class:`int`
                Trajectory index
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
        """
        # Get the first data book component.
        c0 = self.Components[0]
        # Try to find a match existing in the data book.
        j = self[c0].FindMatch(i)
        # Get the name of the folder.
        frun = self.x.GetFullFolderNames(i)
        # Status update.
        print(frun)
        # Go home.
        os.chdir(self.RootDir)
        # Check if the folder exists.
        if not os.path.isdir(frun):
            # Nothing to do.
            return
        # Go to the folder.
        os.chdir(frun)
        # Get the current iteration number.
        nIter = GetCurrentIter()
        # Get the number of iterations used for stats.
        nStats = self.opts.get_nStats()
        # Process whether or not to update.
        if np.isnan(j):
            # No current entry.
            print("  Adding new databook entry at iteration %s." % nIter)
            q = True
        elif self[c0]['nIter'] < nIter:
            # Update
            print("  Updating from iteration %s to %s."
                % (self[c0]['nIter'], nIter))
            q = True
        elif self[c0]['nStats'] != nStats:
            # Change statistics
            print("  Recomputing statistics using %i iterations." % nStats)
            q = True
        else:
            # Up-to-date
            print("  Databook up to date.")
            q = False
        # Check for an update
        if (not q): return
        # Read the history.
        A = Aero(self.Components)
        # Loop through components.
        for comp in self.Components:
            # Extract the component history and component databook.
            FM = A[comp]
            DC = self[comp]
            # Add to the number of cases.
            DC.n += 1
            # Process the statistics.
            s = FM.GetStats(nStats)
            # This is the part where we do transformations....



            # Save the data.
            if np.isnan(j):
                # Append trajectory values.
                for k in self.x.keys:
                    # I hate the way NumPy does appending.
                    DC[k] = np.hstack((DC[k], [getattr(self.x,k)[i]]))
                # Append values.
                for c in DC.DataCols:
                    DC[c] = np.hstack((DC[c], [s[c]]))
                # Append iteration counts.
                DC['nIter']  = np.hstack((DC['nIter'], [nIter]))
                DC['nStats'] = np.hstack((DC['nStats'], [nStats]))
                # Process the target.
                if len(DC.TargetCols) > 0:
                    # Select one.
                    c = DC.TargetCols[0]
                    # Determine which target is in use.
                    it, ct = self.GetTargetIndex(c)
                    # Select the target.
                    DBT = self.Targets[it]
                    # Find matches
                    jt = DBT.FindMatch(self.x, i)
                    # Check for a match.
                    if len(jt) > 0:
                        # Select the first match.
                        jt = jt[0]
                    else:
                        # No match
                        jt = np.nan
                # Append targets.
                for c in DC.targs:
                    # Determine the target to use.
                    it, ct = self.GetTargetIndex(DC.targs[c])
                    # Target name of the coefficient
                    cc = c + '_t'
                    # Store it.
                    if np.isnan(jt):
                        # No match found
                        DC[cc] = np.hstack((DC[cc], [np.nan]))
                    else:
                        # Append the match.
                        DC[cc] = np.hstack((DC[cc], [DBT[ct][jt]]))
            else:
                # No need to update trajectory values.
                # Update data values.
                for c in DC.DataCols:
                    DC[c][j] = s[c]
                # No reason to update targets, either.
        # Go back.
        os.chdir(self.RootDir)
                    
                    
                    
    # Get target to use based on target name
    def GetTargetIndex(self, ftarg):
        """Get the index of the target to use based on a name
        
        For example, if "UPWT/CAFC" will use the target "UPWT" and the column
        named "CAFC".  If there is no "/" character in the name, the first
        available target is used.
        
        :Call:
            >>> i, c = self.GetTargetIndex(ftarg)
        :Inputs:
            *DB*: :class:`pyCart.dataBook.DataBook`
                Instance of the pyCart data book class
            *targ*: :class:`str`
                Name of the target and column
        :Outputs:
            *i*: :class:`int`
                Index of the target to use
            *c*: :class:`str`
                Name of the column to use from that target
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
        """
        # Check if there's a slash
        if "/" in ftarg:
            # List of target names.
            TNames = [DBT.Name for DBT in self.Targets]
            # Split.
            ctarg = ftarg.split("/")
            # Find the name,
            i = TNames.index(ctarg[0])
            # Column name
            c = ctarg[1]
        else:
            # Use the first target.
            i = 0
            c = ftarg
        # Output
        return i, c
            
                
                
# Individual component data book
class DBComp(dict):
    """
    Individual component data book
    
    :Call:
        >>> DBi = DBComp(comp, x, opts)
    :Inputs:
        *comp*: :class:`str`
            Name of the component
        *x*: :class:`pyCart.trajectory.Trajectory`
            Trajectory for processing variable types
        *opts*: :class:`pyCart.options.Options`
            Global pyCart options instance
        *fdir*: :class:`str`
            Data book folder (forward slash separators)
    :Outputs:
        *DBi*: :class:`pyCart.dataBook.DBComp`
            An individual component data book
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
    """
    # Initialization method
    def __init__(self, comp, x, opts, fdir="data"):
        """Initialization method
        
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Get the list of columns for that coefficient.
        cols = opts.get_DataBookCols(comp)
        # Get the directory.
        fdir = opts.get_DataBookDir()
        
        # Construct the file name.
        fcomp = 'aero_%s.dat' % comp
        # Folder name for compatibility.
        fdir = fdir.replace("/", os.sep)
        # Construct the full file name.
        fname = os.path.join(fdir, fcomp)
        
        # Save relevant information
        self.x = x
        self.opts = opts
        self.comp = comp
        self.cols = cols
        # Save the target translations.
        self.targs = opts.get_CompTargets(comp)
        # Divide columns into parts.
        self.DataCols = opts.get_DataBookDataCols(comp)
        self.TargetCols = opts.get_DataBookTargetCols(comp)
        # Save the file name.
        self.fname = fname
        
        # Read the file or initialize empty arrays.
        self.Read(fname)
        
        
    # Function to read data book files
    def Read(self, fname=None):
        """Read a single data book file or initialize empty arrays
        
        :Call:
            >>> DBi.Read()
            >>> DBi.Read(fname)
        :Inputs:
            *DBi*: :class:`pyCart.dataBook.DBComp`
                An individual component data book
            *fname*: :class:`str`
                Name of file to read (default: ``'aero_%s.dat' % self.comp``)
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Check for default file name
        if fname is None: fname = self.fname
        # Try to read the file.
        try:
            # DataBook delimiter
            delim = self.opts.get_Delimiter()
            # Initialize column number
            nCol = 0
            # Loop through trajectory keys.
            for k in self.x.keys:
                # Get the type.
                t = self.x.defns[k].get('Value', 'float')
                # Read the column
                self[k] = np.loadtxt(fname, 
                    delimiter=delim, dtype=t, usecols=[nCol])
                # Increase the column number
                nCol += 1
            # Loop through the data book columns.
            for c in self.cols:
                # Add the column.
                self[c] = np.loadtxt(fname, delimiter=delim, usecols=[nCol])
                # Increase column number.
                nCol += 1
            # Last iteration number
            self['nIter'] = np.loadtxt(fname, 
                delimiter=delim, dtype=int, usecols=[nCol])
            # Number of iterations used for averaging.
            self['nStats'] = np.loadtxt(fname, 
                delimiter=delim, dtype=int, usecols=[nCol+1])
        except Exception:
            # Initialize empty trajectory arrays.
            for k in self.x.keys:
                # Get the type.
                t = self.x.defns[k].get('Value', 'float')
                # Initialize an empty array.
                self[k] = np.array([], dtype=t)
            # Initialize the data columns.
            for c in self.cols:
                self[c] = np.array([])
            # Last iteration number
            self['nIter'] = np.array([], dtype=int)
            # Number of iterations used for averaging.
            self['nStats'] = np.array([], dtype=int)
        # Set the number of points.
        self.n = len(self[c])
        
    # Function to write data book files
    def Write(self, fname=None):
        """Write a single data book file or initialize empty arrays
        
        :Call:
            >>> DBi.Write()
            >>> DBi.Write(fname)
        :Inputs:
            *DBi*: :class:`pyCart.dataBook.DBComp`
                An individual component data book
            *fname*: :class:`str`
                Name of file to read (default: ``'aero_%s.dat' % self.comp``)
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Check for default file name
        if fname is None: fname = self.fname
        # Check for a previous old file.
        if os.path.isfile(fname+'.old'):
            # Remove it.
            os.remove(fname+'.old')
        # Check for an existing data file.
        if os.path.isfile(fname):
            # Move it to ".old"
            os.rename(fname, fname+'.old')
        # DataBook delimiter
        delim = self.opts.get_Delimiter()
        # Open the file.
        f = open(fname, 'w')
        # Write the header.
        f.write("# aero data for '%s' extracted on %s\n" %
            (self.comp, datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')))
        # Empty line.
        f.write('#\n')
        # Reference quantities
        f.write('# Reference Area = %.6E\n' %
            self.opts.get_RefArea(self.comp))
        f.write('# Reference Length = %.6E\n' %
            self.opts.get_RefLength(self.comp))
        # Get the nominal MRP.
        xMRP = self.opts.get_RefPoint(self.comp)
        # Write it.
        f.write('# Nominal moment reference point:\n')
        f.write('# XMRP = %.6E\n' % xMRP[0])
        f.write('# YMRP = %.6E\n' % xMRP[1])
        # Check for 3D.
        if len(xMRP) > 2:
            f.write('# ZMRP = %.6E\n' % xMRP[2])
        # Empty line and start of variable list.
        f.write('#\n# ')
        # Loop through trajectory keys.
        for k in self.x.keys:
            # Just write the name.
            f.write(k + delim)
        # Loop through coefficients.
        for c in self.cols:
            # Write the name. (represents the means)
            f.write(c + delim)
        # Write the number of iterations and num used for stats.
        f.write('nIter%snStats\n' % delim)
        # Loop through the database entries.
        for i in np.arange(self.n):
            # Write the trajectory points.
            for k in self.x.keys:
                f.write('%s%s' % (self[k][i], delim))
            # Write values.
            for c in self.cols:
                f.write('%.8E%s' % (self[c][i], delim))
            # Write number of iterations.
            f.write('%i%s%i\n' % (self['nIter'][i], delim, self['nStats'][i]))
        # Close the file.
        f.close()
        
        # Find an entry by trajectory variables.
    def FindMatch(self, i):
        """Find an entry by run matrix (trajectory) variables
        
        It is assumed that exact matches can be found.
        
        :Call:
            >>> j = DBi.FindMatch(i)
        :Inputs:
            *DBi*: :class:`pyCart.dataBook.DBComp`
                Instance of the pyCart data book target data carrier
            *i*: :class:`int`
                Index of the case from the trajectory to try match
        :Outputs:
            *j*: :class:`numpy.ndarray` (:class:`int`)
                Array of index that matches the trajectory case or ``NaN``
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
        """
        # Initialize indices (assume all are matches)
        j = np.arange(self.n)
        # Loop through keys requested for matches.
        for k in self.x.keys:
            # Get the target value (from the trajectory)
            v = getattr(self.x,k)[i]
            # Search for matches.
            try:
                # Filter test criterion.
                jk = np.where(self[k] == v)[0]
                # Restrict to rows that match the above.
                j = np.intersect1d(j, jk)
            except Exception:
                # No match found.
                return np.nan
        # Output
        try:
            # There should be exactly one match.
            return j[0]
        except Exception:
            # Return no match.
            return np.nan
        
        
# Data book target instance
class DBTarget(dict):
    """
    Class to handle data from data book target files.  There are more
    constraints on target files than the files that data book creates, and raw
    data books created by pyCart are not valid target files.
    
    :Call:
        >>> DBT = pyCart.dataBook.DBTarget(targ, opts)
    :Inputs:
        *targ*: :class:`pyCart.options.DataBook.DBTarget`
            Instance of a target source options interface
        *opts*: :class:`pyCart.options.Options`
            Global pyCart options instance to determine which fields are useful
    :Outputs:
        *DBT*: :class:`pyCart.dataBook.DBTarget`
            Instance of the pyCart data book target data carrier
    :Versions:
        * 2014-12-20 ``@ddalle``: Started
    """
    
    # Initialization method
    def __init__(self, targ, opts):
        """Initialization method
        
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Save the target options
        self.opts = targ
        # Source file
        fname = targ.get_TargetFile()
        # Name of this target.
        tname = targ.get_TargetName()
        # Check for the file.
        if not os.path.isfile(fname):
            raise IOError(
                "Target source file '%s' could not be found." % fname)
        # Save the name.
        self.Name = tname
        # Delimiter
        delim = targ.get_Delimiter()
        # Comment character
        comchar = targ.get_CommentChar()
        # Open the file again.
        f = open(fname)
        # Loop until finding a line that doesn't begin with comment char.
        line = comchar
        nskip = -1
        while line.strip().startswith(comchar):
            # Save the old line.
            headers = line
            # Read the next line
            line = f.readline()
            nskip += 1
        # Close the file.
        f.close()
        # Translate into headers
        self.headers = headers.strip().split(delim)

        # Read it.
        self.data = np.loadtxt(fname, delimiter=delim, skiprows=nskip)
        # Initialize requested fields with the fields that correspond to
        # trajectory keys
        cols = targ.get_Trajectory().values()
        # Process the required fields.
        for comp in opts.get_DataBookComponents():
            # Loop through the targets.
            for ctarg in opts.get_CompTargets(comp).values():
                # Get the target source for this entry.
                if '/' not in ctarg:
                    # Only one target source; assume it's this one.
                    ti = tname
                    fi = ctarg
                else:
                    # Read the target name.
                    ti = ctarg.split('/')[0]
                    # Name of the column
                    fi = ctarg.split('/')[1]
                # Check if the target is from this target source.
                if ti != tname: continue
                # Add the field if necessary.
                if fi not in cols:
                    # Check if the column is present.
                    if fi not in self.headers:
                        raise IOError("There is not field '%s' in '%s'."
                            % (fi, ti))
                    # Add the column
                    cols.append(fi)
        # Process the columns.
        for col in cols:
            # Find it and save it as a key.
            self[col] = self.data[:,self.headers.index(col)]
        
    # Find an entry by trajectory variables.
    def FindMatch(self, x, i):
        """Find an entry by run matrix (trajectory) variables
        
        Cases will be considered matches by comparing variables specified in 
        the *DataBook* section of :file:`pyCart.json` as cases to compare
        against.  Suppose that the control file contains the following.
        
        .. code-block:: python
        
            "DataBook": {
                "Targets": {
                    "Name": "Experiment",
                    "File": "WT.dat",
                    "Trajectory": {"alpha": "ALPHA", "Mach": "MACH"}
                    "Tolerances": {
                        "alpha": 0.05,
                        "Mach": 0.01
                    }
                }
            }
        
        Then any entry in the data book target that matches the Mach number
        within 0.01 (using a column labeled *MACH*) and alpha to within 0.05 is
        considered a match.  If there are more trajectory variables, they are
        not used for this filtering of matches.
        
        :Call:
            >>> j = DBT.FindMatch(x, i)
        :Inputs:
            *DBT*: :class:`pyCart.dataBook.DBTarget`
                Instance of the pyCart data book target data carrier
            *x*: :class:`pyCart.trajectory.Trajectory`
                The current pyCart trajectory (i.e. run matrix)
            *i*: :class:`int`
                Index of the case from the trajectory to try match
        :Outputs:
            *j*: :class:`numpy.ndarray` (:class:`int`)
                Array of indices that match the trajectory within tolerances
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Initialize indices (assume all are matches)
        j = np.arange(self.data.shape[0])
        # Get the trajectory key translations.   This determines which keys to
        # filter and what those keys are called in the source file.
        tkeys = self.opts.get_Trajectory()
        # Loop through keys requested for matches.
        for k in tkeys:
            # Get the tolerance.
            tol = self.opts.get_Tol(k)
            # Get the target value (from the trajectory)
            v = getattr(x,k)[i]
            # Get the name of the column according to the source file.
            c = tkeys[k]
            # Search for matches.
            try:
                # Filter test criterion.
                jk = np.where(np.abs(self[c] - v) <= tol)[0]
                # Restrict to rows that match the above.
                j = np.intersect1d(j, jk)
            except Exception:
                pass
        # Output
        return j
            
        
        
# Aerodynamic history class
class Aero(dict):
    """
    This class provides an interface to important data from a run directory.  It
    reads force and moment histories for named components, if available, and
    other types of data can also be stored
    
    :Call:
        >>> aero = pyCart.dataBook.Aero(comps=[])
    :Inputs:
        *comps*: :class:`list` (:class:`str`)
            List of components to read; defaults to all components available
    :Outputs:
        *aero*: :class:`pyCart.aero.Aero`
            Instance of the aero history class, similar to dictionary of force
            and/or moment histories
    :Versions:
        * 2014-11-12 ``@ddalle``: Starter version
        * 2014-12-21 ``@ddalle``: Copied from previous `aero.Aero`
    """
    
    # Initialization method
    def __init__(self, comps=[]):
        """Initialization method
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Process the best data folder.
        if os.path.isfile('history.dat'):
            # Use current directory.
            fdir = '.'
        elif os.path.islink('BEST'):
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
        # Read the residuals.
        self.Residual = CaseResid()
        # Default component list.
        if (type(comps).__name__ in ["str", "unicode", "int"]):
            # Make a singleton list.
            comps = [comps]
        elif len(comps) < 1:
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
            lines = open(fname).readlines()
            # Filter comments
            lines = [l for l in lines if not l.startswith('#')]
            # Convert all the values to floats
            # Can't make this an array yet because it's not rectangular.
            V = [[float(v) for v in l.split()] for l in lines]
            # Columns to use: 0 and {-6,-3}:
            n = len(self.Components[comp]['C'])
            # Create an array with the original data.
            A = np.array([v[0:1] + v[-n:] for v in V])
            # Get the number of entries in each row.
            # This will be one larger if a time-accurate iteration.
            # It's a column of zeros, and it's the second column.
            L = np.array([len(v) for v in V])
            # Check for steady-state iterations.
            if np.any(L == n+1):
                # At least one steady-state iteration
                n0 = np.max(A[L==n+1,0])
                # Add that iteration number to the time-accurate steps.
                A[L!=n+1,0] += n0
            # Extract info from components for readability
            d = self.Components[comp]
            # Make the component.
            self[comp] = CaseFM(d['C'], MRP=d['MRP'], A=A)
            
            
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
class CaseFM(object):
    """
    This class contains methods for reading data about an the histroy of an
    individual component for a single case.  The list of available components
    comes from a :file:`loadsCC.dat` file if one exists.
    
    :Call:
        >>> FM = pyCart.dataBook.CaseFM(C, MRP=None, A=None)
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
        * 2014-12-21 ``@ddalle``: Copied from previous `aero.FM`
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
        
            * ``'<dataBook.CaseFM Force, i=100>'``
            * ``'<dataBook.CaseFM Moment, i=100, MRP=(0.00, 1.00, 2.00)>'``
            * ``'<dataBook.CaseFM FM, i=100, MRP=(0.00, 1.00, 2.00)>'``
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        # Initialize the string.
        txt = '<dataBook.CaseFM '
        # Check for a moment.
        if ('CA' in self.C) and ('CLL' in self.C):
            # Force and moment.
            txt += 'FM'
        elif ('CA' in self.C):
            # Force only
            txt += 'Force'
        elif ('CLL' in self.C):
            # Moment only
            txt += 'Moment'
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
        
            * ``'<dataBook.CaseFM Force, i=100>'``
            * ``'<dataBook.CaseFM Moment, i=100, MRP=(0.00, 1.00, 2.00)>'``
            * ``'<dataBook.CaseFM FM, i=100, MRP=(0.00, 1.00, 2.00)>'``
        
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
            *FM*: :class:`pyCart.dataBook.CaseFM`
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
            # Save list of coefficients.
            self.coeffs = ['CA', 'CY', 'CN', 'CLL', 'CLM', 'CLN']
        elif (self.MRP.size==3) and (m == 4):
            # Save only moments.
            self.CLL = A[:,1]
            self.CLM = A[:,2]
            self.CLN = A[:,3]
            # Save list of coefficients.
            self.coeffs = ['CLL', 'CLM', 'CLN']
        elif (m == 4):
            # Save only forces.
            self.CA = A[:,1]
            self.CY = A[:,2]
            self.CN = A[:,3]
            # Save list of coefficients.
            self.coeffs = ['CA', 'CY', 'CN']
        
    # Method to get averages and standard deviations
    def GetStats(self, nAvg=100):
        """Get mean, min, max, and standard deviation for all coefficients
        
        :Call:
            >>> s = FM.GetStats(nAvg=100)
        :Inputs:
            *FM*: :class:`pyCart.aero.FM`
                Instance of the force and moment class
            *nAvg*: :class:`int`
                Number of iterations in window
        :Outputs:
            *s*: :class:`dict` (:class:`float`)
                Dictionary of mean, min, max, std for each coefficient
        :Versions:
            * 2014-12-09 ``@ddalle``: First version
        """
        # Process min indices for plotting and averaging.
        i0 = max(0, self.i.size-nAvg)
        # Initialize output.
        s = {}
        # Loop through coefficients.
        for c in self.coeffs:
            # Get the values
            F = getattr(self, c)
            # Save the mean value.
            s[c] = np.mean(F[i0:])
            # Save the statistics.
            s[c+'_min'] = np.min(F[i0:])
            s[c+'_max'] = np.max(F[i0:])
            s[c+'_std'] = np.std(F[i0:])
        # Output
        return s
            

    

# Aerodynamic history class
class CaseResid(object):
    """
    Iterative history class
    
    This class provides an interface to residuals, CPU time, and similar data
    for a given run directory
    
    :Call:
        >>> hist = pyCart.dataBook.CaseResid()
    :Outputs:
        *hist*: :class:`pyCart.dataBook.CaseResid`
            Instance of the run history class
    :Versions:
        * 2014-11-12 ``@ddalle``: Starter version
    """
    
    # Initialization method
    def __init__(self):
        """Initialization method
        
        :Versions:
            * 2014-11-12 ``@ddalle``: First version
        """
        
        # Process the best data folder.
        if os.path.isfile('history.dat'):
            # Subsequent non-adaptive runs
            fdir = '.'
        elif os.path.islink('BEST'):
            # There's a BEST/ folder; use it as most recent adaptation cycle.
            fdir = 'BEST'
        elif os.path.isdir('adapt00'):
            # It's an adaptive run, but it hasn't gotten far yet.
            fdir = 'adapt00'
        else:
            # This is not an adaptive cycle; use root folder.
            fdir = '.'
        # History file name.
        fhist = os.path.join(fdir, 'history.dat')
        # Read the file.
        lines = open(fhist).readlines()
        # Filter comments.
        lines = [l for l in lines if not l.startswith('#')]
        # Convert all the values to floats.
        A = np.array([[float(v) for v in l.split()] for l in lines])
        # Get the indices of steady-state iterations.
        # (Time-accurate iterations are marked with decimal step numbers.)
        i = np.array(['.' not in l.split()[0] for l in lines])
        # Check for steady-state iterations.
        if np.any(i):
            # Get the last steady-state iteration.
            n0 = np.max(A[i,0])
            # Add this to the time-accurate iteration numbers.
            A[np.logical_not(i),0] += n0
        # Eliminate subiterations.
        A = A[np.mod(A[:,0], 1.0) == 0.0]
        # Save the number of iterations.
        self.nIter = A.shape[0]
        # Save the iteration numbers.
        self.i = A[:,0]
        # Save the CPU time per processor.
        self.CPUtime = A[:,1]
        # Save the maximum residual.
        self.maxResid = A[:,2]
        # Save the global residual.
        self.L1Resid = A[:,3]
        # Check for a 'user_time.dat' file.
        if os.path.isfile('user_time.dat'):
            # Initialize time
            t = 0.0
            # Loop through lines.
            for line in open('user_time.dat').readlines():
                # Check comment.
                if line.startswith('#'): continue
                # Add to the time.
                t += np.sum([float(v) for v in line.split()[1:]])
        else:
            # Find the indices of run break points.
            i = np.where(self.CPUtime[1:] < self.CPUtime[:-1])[0]
            # Sum the end times.
            t = np.sum(self.CPUtime[i]) + self.CPUtime[-1]
        # Save the time.
        self.CPUhours = t / 3600.
        
        
