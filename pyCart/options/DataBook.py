"""Interface for Cart3D meshing settings"""


# Import options-specific utilities
from util import rc0, odict


# Class for autoInputs
class DataBook(odict):
    """Dictionary-based interface for DataBook specifications"""
    
    # Initialization method
    def __init__(self, fname=None, **kw):
        """Data book options initialization method
        
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Store the data in *this* instance
        for k in kw:
            self[k] = kw[k]
        # Upgrade important groups to their own classes.
        self._DBTarget()
    
    # Initialization and confirmation for autoInputs options
    def _DBTarget(self):
        """Initialize data book target options if necessary"""
        # Check for missing entirely.
        if 'Targets' not in self:
            # Empty/default
            self['Targets'] = [DBTarget()]
            return None
        # Read the targets
        targs = self['Targets']
        # Check the type.
        if type(targs).__name__ == 'dict':
            # Convert it to a list of dictionaries
            targs = [targs]
        elif type(targs).__name__ not in ['list', 'ndarray']:
            # Invalid
            raise IOError('Data book targets must be a list')
        # Initialize final state.
        self['Targets'] = []
        # Loop through targets
        for targ in targs:
            # Convert to special class.
            self['Targets'].append(DBTarget(**targ))
    
    # Get the list of components.
    def get_DataBookComponents(self):
        """Get the list of components to be used for the data book
        
        :Call:
            >>> comps = opts.get_DataBookComponents()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *comps*: :class:`list`
                List of components
        :Versions:
            * 2014-12-20 ``@ddalle``: First version
        """
        # Get the value from the dictionary.
        comps = self.get('Components', ['entire'])
        # Make sure it's a list.
        if type(comps).__name__ not in ['list']:
            comps = [comps]
        # Check contents.
        for comp in comps:
            if (type(comp).__name__ not in ['str', 'int', 'unicode']):
                raise IOError("Component '%s' is not a str or int." % comp)
        # Output
        return comps
        
    # Get the number of initial divisions
    def get_nStats(self):
        """Get the number of iterations to be used for collecting statistics
        
        :Call:
            >>> nStats = opts.get_nStats()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *nStats*: :class:`int`
                Number of iterations to be used for statistics
        :Versions:
            * 2014-12-20 ``@ddalle``: First version
        """
        return self.get_key('nStats', 0)
        
    # Set the number of initial mesh divisions
    def set_nStats(self, nStats=rc0('db_stats')):
        """Set the number of divisions in background mesh
        
        :Call:
            >>> opts.set_nDiv(nStats)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *nStats*: :class:`int`
                Number of iterations to be used for statistics
        :Versions:
            * 2014-12-20 ``@ddalle``: First version
        """
        self['nStats'] = nStats
        
    # Get the location
    def get_DataBookDir(self):
        """Get the folder that holds the data book
        
        :Call:
            >>> fdir = opts.get_DataBookDir()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fdir*: :class:`str`
                Relative path to data book folder
        :Versions:
            * 2014-12-20 ``@ddalle``: First version
        """
        return self.get('Folder', 'data')
        
    # Set the location
    def set_DataBookDir(self, fdir=rc0('db_dir')):
        """Set the folder that holds the data book
        
        :Call:
            >>> fdir = opts.get_DataBookDir()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fdir*: :class:`str`
                Relative path to data book folder
        :Versions:
            * 2014-12-20 ``@ddalle``: First version
        """
        self['Folder'] = fdir
        
    # Get the file delimiter
    def get_Delimiter(self):
        """Get the delimiter to use in files
        
        :Call:
            >>> delim = opts.get_Delimiter()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *delim*: :class:`str`
                Delimiter to use in data book files
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        return self.get('Delimiter', rc0('Delimiter'))
        
    # Set the file delimiter.
    def set_Delimiter(self, delim=rc0('Delimiter')):
        """Set the delimiter to use in files
        
        :Call:
            >>> opts.set_Delimiter(delim)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *delim*: :class:`str`
                Delimiter to use in data book files
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        self['Delimiter'] = delim
        
                                                
    # Get the targets
    def get_DataBookTargets(self):
        """Get the list of components to be used for the data book
        
        :Call:
            >>> comps = opts.get_DataBookTargets()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *targets*: :class:`list` (:class:`dict`)
                List of components
        :Versions:
            * 2014-12-20 ``@ddalle``: First version
        """
        # Get the value from the dictionary.
        targets = self.get('Targets', [])
        # Make sure it's a list.
        if type(targets).__name__ not in ['list']:
            targets = [targets]
        # Check contents.
        for targ in targets:
            if (type(targ).__name__ not in ['DBTarget']):
                raise IOError("Target '%s' is not a DBTarget." % targ)
        # Output
        return targets
        
    # Get the coefficients for a specific component
    def get_DataBookCoeffs(self, comp):
        """Get the list of data book coefficients for a specific component
        
        :Call:
            >>> coeffs = opts.get_DataBookCoeffs(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *coeffs*: :class:`list` (:class:`str`)
                List of coefficients for that component
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Get the component options.
        copts = self.get(comp, {})
        # Check for manually-specified coefficients
        coeffs = copts.get("Coefficients", [])
        # Check the type.
        if type(coeffs).__name__ not in ['list']:
            raise IOError(
                "Coefficients for component '%s' must be a list." % comp) 
        # Exit if that exists.
        if len(coeffs) > 0:
            return coeffs
        # Check the type.
        ctype = copts.get("Type", "Force")
        # Default coefficients
        if ctype in ["Force", "force"]:
            # Force only, body-frame
            coeffs = ["CA", "CY", "CN"]
        elif ctype in ["Moment", "moment"]:
            # Moment only, body-frame
            coeffs = ["CLL", "CLM", "CLN"]
        elif ctype in ["FM", "full", "Full"]:
            # Force and moment
            coeffs = ["CA", "CY", "CN", "CLL", "CLM", "CLN"]
        # Output
        return coeffs
        
    # Get the targets for a specific component
    def get_CompTargets(self, comp):
        """Get the list of targets for a specific data book component
        
        :Call:
            >>> targs = opts.get_CompTargets(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *targs*: :class:`list` (:class:`str`)
                List of targets for that component
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Get the component options.
        copts = self.get(comp, {})
        # Get the targets.
        targs = copts.get('Targets', {})
        # Make sure it's a dict.
        if type(targs).__name__ not in ['dict']:
            raise IOError("Targets for component '%s' are not a dict." % comp)
        # Output
        return targs
        
    # Get full list of columns for a specific component
    def get_DataBookCols(self, comp):
        """Get the full list of data book columns for a specific component
        
        This includes the list of coefficients, e.g. ``['CA', 'CY', 'CN']``;
        statistics such as ``'CA_min'`` if *nStats* is greater than 0; and
        targets such as ``'CA
        
        :Call:
            >>> cols = opts.get_DataBookCols(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *cols*: :class:`list` (:class:`str`)
                List of coefficients and other columns for that coefficient
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Get the list of coefficients.
        coeffs = self.get_DataBookCoeffs(comp)
        # Initialize output
        cols = [] + coeffs
        # Get the number of iterations used for statistics
        nStats = self.get_nStats()
        # Options for te
        # Process statistical columns.
        if nStats > 0:
            # Loop through columns.
            for c in coeffs:
                # Append all statistical columns.
                cols += [c+'_min', c+'_max', c+'_std']
        # Process targets.
        targs = self.get_CompTargets(comp)
        # Loop through the targets.
        for c in targs:
            # Append target column
            cols.append(c+'_t')
        # Output
        return cols
        
        
# Class for target data
class DBTarget(odict):
    """Dictionary-based interface for databook targets"""
    
    # Get the maximum number of refinements
    def get_TargetName(self):
        """Get the name/label for a given target
        
        :Call:
            >>> Name = opts.get_TargetName()
        :Inputs:
            *opts*: :class:`pyCart.options.DataBook.DBTarget`
                Options interface
        :Outputs:
            *Name*: :class:`str`
                Name of the component (label for plots)
        :Versions:
            * 2014-08-03 ``@ddalle``: First version
        """
        return self.get('Name', 'Target')
        
    # Get the fie name
    def get_TargetFile(self):
        """Get the file name for the target
        
        :Call:
            >>> fname = opts.get_TargetFile()
        :Inputs:
            *opts*: :class:`pyCart.options.DataBook.DBTarget`
                Options interface
        :Outputs:
            *fname*: :class:`str`
                Name of the file
        :Versions:
            * 2014-12-20 ``@ddalle``: First version
        """
        return self.get('File', 'Target.dat')
        
    # Get the delimiter
    def get_Delimiter(self):
        """Get the delimiter for a target file
        
        :Call:
            >>> delim = opts.get_Delimiter()
        :Inputs:
            *opts*: :class:`pyCart.options.DataBook.DBTarget`
                Options interface
        :Outputs:
            *delim*: :class:`str`
                Delimiter text
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        return self.get('Delimiter', rc0('Delimiter'))
        
    # Get the comment character.
    def get_CommentChar(self):
        """Get the character to used to mark comments
        
        :Call:
            >>> comchar = opts.get_CommentChar()
        :Inputs:
            *opts*: :class:`pyCart.options.DataBook.DBTarget`
                Options interface
        :Outputs:
            *comchar*: :class:`str`
                Comment character (may be multiple characters)
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        return self.get('Comment', '#')
    
    # Get trajectory conversion
    def get_Trajectory(self):
        """Get the trajectory translations
        
        :Call:
            >>> traj = opst.get_Trajectory()
        :Inputs:
            *opts*: :class:`pyCart.options.DataBook.DBTarget`
                Options interface
        :Outputs:
            *comchar*: :class:`str`
                Comment character (may be multiple characters)
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        return self.get('Trajectory', {})
        
    # Get the tolerance for a given trajectory variable
    def get_Tol(self, k=None):
        """
        Get the tolerance to consider a target condition to be a match of the
        run matrix point, either a default value or a tolerance for a specific
        trajectory variable
        
        :Call:
            >>> tol = opts.get_Tol()
            >>> tol = opts.get_Tol(k)
        :Inputs:
            *opts*: :class:`pyCart.options.DataBook.DBTarget`
                Options interface
            *k*: :class:`str`
                Name of a trajectory variable
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Get the entry for "DataBook/Target/Tolerances".
        tols = self.get('Tolerances', 0.01)
        # Check the type.
        if type(tols).__name__ != "dict":
            # Use it as a default and output it.
            return tols
        # Else, get the default value.
        dtol = tols.get('default', 0.01)
        # Check if default was requested.
        if k is None:
            # Return the default.
            return dtol
        else:
            # Try to return the specific tolerance.
            return tols.get(k, dtol)

