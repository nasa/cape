"""Interface for Cart3D meshing settings"""


# Import options-specific utilities
from util import rc0, odict, getel


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
        self._DBPlot()
    
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
            
    # Get the plot options.
    def _DBPlot(self):
        """Initialize plot options"""
        # Check if missing entirely.
        if 'Plot' not in self:
            # Empty.
            self['Plot'] = []
            return None
        # Read the plots
        o_plt = self['Plot']
        # Check the type.
        if type(o_plt).__name__ == 'dict':
            # Convert to singleton list.
            o_plt = [o_plt]
        elif type(o_plt).__name__ not in ['list', 'ndarray']:
            # Invalid
            raise IOError('Data book plot options must be a list.')
        # Initialize the plots.
        self['Plot'] = []
        # Initialize if possible.
        if len(o_plt) > 0:
            # Convert to special class.
            self['Plot'].append(DBPlot(**o_plt[0]))
        # Loop through the plots
        for i in range(1, len(o_plt)):
            # Initialize with previous object.
            self['Plot'].append(DBPlot(defs=self['Plot'][i-1], **o_plt[i]))
    
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
        
    # Get the key on which to sort
    def get_SortKey(self):
        """Get the key to use for sorting the data book
        
        :Call:
            >>> key = opts.get_SortKey()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *key*: :class:`str` | ``None`` | :class:`list` (:class:`str`)
                Name of key to sort with
        :Versions:
            * 2014-12-30 ``@ddalle``: First version
        """
        return self.get('Sort')
        
    # Set the key on which to sort
    def set_SortKey(self, key):
        """Set the key to use for sorting the data book
        
        :Call:
            >>> opts.set_SortKey(key)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *key*: :class:`str` | ``None`` | :class:`list` (:class:`str`)
                Name of key to sort with
        :Versions:
            * 2014-12-30 ``@ddalle``: First version
        """
        self['Sort'] = key
        
                                                
    # Get the targets
    def get_DataBookTargets(self):
        """Get the list of targets to be used for the data book
        
        :Call:
            >>> targets = opts.get_DataBookTargets()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *targets*: :class:`list` (:class:`dict`)
                List of targets
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
        
    # Get the plots
    def get_DataBookPlots(self):
        """Get the list of data book plots
        
        :Call:
            >>> o_plt = opts.get_DataBookPlots()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *o_plt*: :class:`list` (:class:`pyCart.options.DataBook.DBPlot`)
                List of data book plot descriptors
        :Versions:
            * 2014-12-20 ``@ddalle``: First version
        """
        # Get the value from the dictionary.
        o_plt = self.get('Plot', [])
        # Make sure it's a list.
        if type(o_plt).__name__ not in ['list']:
            o_plt = [o_plt]
        # Check contents.
        for o_i in o_plt:
            if (type(o_i).__name__ not in ['DBPlot']):
                raise IOError("Plot '%s' is not a DBPlot." % o_i)
        # Output
        return o_plt
        
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
        
    # Get the transformations for a specific component
    def get_DataBookTransformations(self, comp):
        """
        Get the transformations required to transform a component's data book
        into the body frame of that component.
        
        :Call:
            >>> tlist = opts.get_DataBookTransformations(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *tlist*: :class:`list` (:class:`dict`)
                List of targets for that component
        :Versions:
            * 2014-12-22 ``@ddalle``: First version
        """
        # Get the options for the component.
        copts = self.get(comp, {})
        # Get the value specified, defaulting to an empty list.
        tlist = copts.get('Transformations', [])
        # Make sure it's a list.
        if type(tlist).__name__ not in ['list', 'ndarray']:
            # Probably a single transformation; put it in a list
            tlist = [tlist]
        # Output
        return tlist
        
    # Get full list of columns for a specific component
    def get_DataBookCols(self, comp):
        """Get the full list of data book columns for a specific component
        
        This includes the list of coefficients, e.g. ``['CA', 'CY', 'CN']``;
        statistics such as ``'CA_min'`` if *nStats* is greater than 0; and
        targets such as ``'CA_t`` if there is a target for *CA*.
        
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
        # Data columns (from CFD)
        dcols = self.get_DataBookDataCols(comp)
        # Target columns (for comparison)
        tcols = self.get_DataBookTargetCols(comp)
        # Output
        return dcols + tcols
        
    # Get full list of data columns for a specific component
    def get_DataBookDataCols(self, comp):
        """Get the list of data book columns for a specific component
        
        This includes the list of coefficients, e.g. ``['CA', 'CY', 'CN']``;
        statistics such as ``'CA_min'`` if *nStats* is greater than 0.
        
        :Call:
            >>> cols = opts.get_DataBookDataCols(comp)
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
        # Process statistical columns.
        if nStats > 0:
            # Loop through columns.
            for c in coeffs:
                # Append all statistical columns.
                cols += [c+'_min', c+'_max', c+'_std']
        # Output.
        return cols
        
    # Get list of target data columns for a specific component
    def get_DataBookTargetCols(self, comp):
        """Get the list of data book target columns for a specific component
        
        :Call:
            >>> cols = opts.get_DataBookDataCols(comp)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *comp*: :class:`str`
                Name of component
        :Outputs:
            *cols*: :class:`list` (:class:`str`)
                List of coefficient target values
        :Versions:
            * 2014-12-21 ``@ddalle``: First version
        """
        # Initialize output
        cols = []
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

# Class for target data
class DBPlot(odict):
    """Dictionary-based interface for databook plots
    
    :Call:
        >>> DBP = DBPlot(defs={}, **kw)
    :Inputs:
        *defs*: :class:`dict`
            Dictionary of default options from previous dictionary
        *Label*: :class:`str`
            Label for the plot (defaults to list of components in the plot)
        *XAxis*: :class:`str`
            Name of variable to use for *x* axis
        *XLabel*: :class:`str`
            Label for *x* axis of plots
        *YAxis*: :class:`str`
            Name of variable to plot
        *YLabel*: :class:`str`
            Label for *y* axis of plots
        *Tolerances*: :class:`dict`
            Dict of trajectory keys to hold constant, and their tolerances
        *Restriction*: :class:`str`
            Text label for limited distribution, e.g. "ITAR"
        *PlotOptions*: :class:`dict`
            Dict of options passed to :func:`matplotlib.pyplot.plot`
        *TargetOptions*: :class:`dict`
            Dict of plot options for target lines
        *Output*: :class:`str`  [ 'PDF' | 'PNG' | 'SVG' ]
            Extension for optional individual plots (no output otherwise)
        *StandardDeviation*: :class:`float`
            Multiple of standard deviation to show (only show if >0)
        *MinMax*: :class:`bool`
            Whether or not to plot min and max from iterative history
        *MinMaxOptions*: :class:`dict`
            Options for min/max plot to :func:`matplotlib.pyplot.fill_between`
        *StDevOptions*: :class:`dict`
            Plot options for standard deviation plot 
    :Outputs:
        *DBP*: :class:`pyCart.options.DataBook.DBPlot`
            Instance of databook plot options class
    :Versions:
        * 2014-12-27 ``@ddalle``: First version
    """
    
    # Initialization method
    def __init__(self, defs={}, **kw):
        # Loop through recognized keys.
        for k in ["XAxis", "XLabel", "YAxis", "YLabel", "Restriction",
                "Sweep", "Components", "Output", "StandardDeviation", "MinMax", 
                "Label", "PlotOptions", "TargetOptions",
                "MinMaxOptions", "StDevOptions"]:
            # Save the property, defaulting to the last dict
            self[k] = kw.get(k, defs.get(k))
        # Make sure "Components" is a list.
        if type(self["Components"]).__name__ not in ['list', 'ndarray']:
            self["Components"] = [self["Components"]]
            
    # Function to get the plot options
    def get_PlotOptions(self, i=None):
        """Get the plot options for component *i*
        
        :Call:
            >>> o_plt = DBP.get_PlotOptions(i=None)
        :Inputs:
            *DBP*: :class:`pyCart.options.DataBook.DBPlot`
                Instance of databook plot options class
            *i*: :class:`int` or ``None``
                Plot index to extract options for
        :Outputs:
            *o_plt*: :class:`dict`
                Dictionary of plot options
        :Versions:
            * 2014-12-27 ``@ddalle``: First version
        """
        # Extract option keys
        o_in = self.get("PlotOptions", {})
        # Create dict of non-list
        o_plt = {}
        # Loop through keys because Python <2.7 can't handle dict iterators :(
        for k in o_in:
            # Get the option (list).
            o_k = o_in[k]
            # Check if it's a list.
            if type(o_k).__name__ == 'list':
                # Cycle through list.
                o_plt[k] = o_k[i % len(o_k)]
            else:
                # Use the non-list value.
                o_plt[k] = o_k
        # Set some defaults.
        o_plt.setdefault("color", "k")
        o_plt.setdefault("marker", "^")
        # Output
        return o_plt
        
    # Get the target options.
    def get_TargetOptions(self, i=None):
        """Get the plot options for target *i*
        
        :Call:
            >>> o_plt = DBP.get_TargetOptions(i=None)
        :Inputs:
            *DBP*: :class:`pyCart.options.DataBook.DBPlot`
                Instance of databook plot options class
            *i*: :class:`int` or ``None``
                Plot index to extract options for
        :Outputs:
            *o_plt*: :class:`dict`
                Dictionary of plot options
        :Versions:
            * 2014-12-27 ``@ddalle``: First version
        """
        # Extract option keys
        o_in = self.get("TargetOptions", {})
        # Create dict of non-list
        o_plt = {}
        # Loop through keys because Python <2.7 can't handle dict iterators :(
        for k in o_in:
            # Get the option (list).
            o_k = o_in[k]
            # Check if it's a list.
            if type(o_k).__name__ == 'list':
                # Cycle through list.
                o_plt[k] = o_k[i % len(o_k)]
            else:
                # Use the non-list value.
                o_plt[k] = o_k
        # Set some defaults.
        o_plt.setdefault("color", "r")
        o_plt.setdefault("marker", "o")
        # Output
        return o_plt
        
    # Get the target options.
    def get_MinMaxOptions(self, i=None):
        """Get the plot options for min/max plot *i*
        
        :Call:
            >>> o_plt = DBP.get_MinMaxOptions(i=None)
        :Inputs:
            *DBP*: :class:`pyCart.options.DataBook.DBPlot`
                Instance of databook plot options class
            *i*: :class:`int` or ``None``
                Plot index to extract options for
        :Outputs:
            *o_plt*: :class:`dict`
                Dictionary of plot options
        :Versions:
            * 2014-12-28 ``@ddalle``: First version
        """
        # Extract option keys
        o_in = self.get("MinMaxOptions", {})
        # Create dict of non-list
        o_plt = {}
        # Loop through keys because Python <2.7 can't handle dict iterators :(
        for k in o_in:
            # Get the option (list).
            o_k = o_in[k]
            # Check if it's a list.
            if type(o_k).__name__ == 'list':
                # Cycle through list.
                o_plt[k] = o_k[i % len(o_k)]
            else:
                # Use the non-list value.
                o_plt[k] = o_k
        # Set some defaults.
        o_plt.setdefault("color", self.get_PlotOptions(i).get("color", "k"))
        o_plt.setdefault("facecolor", o_plt["color"])
        o_plt.setdefault("alpha", 0.5)
        o_plt.setdefault("lw", 0.2)
        # Output
        return o_plt
        
    # Get the target options.
    def get_StDevOptions(self, i=None):
        """Get the plot options for standard deviation plot *i*
        
        :Call:
            >>> o_plt = DBP.get_MinMaxOptions(i=None)
        :Inputs:
            *DBP*: :class:`pyCart.options.DataBook.DBPlot`
                Instance of databook plot options class
            *i*: :class:`int` or ``None``
                Plot index to extract options for
        :Outputs:
            *o_plt*: :class:`dict`
                Dictionary of plot options
        :Versions:
            * 2014-12-28 ``@ddalle``: First version
        """
        # Extract option keys
        o_in = self.get("StDevOptions", {})
        # Create dict of non-list
        o_plt = {}
        # Loop through keys because Python <2.7 can't handle dict iterators :(
        for k in o_in:
            # Get the option (list).
            o_k = o_in[k]
            # Check if it's a list.
            if type(o_k).__name__ == 'list':
                # Cycle through list.
                o_plt[k] = o_k[i % len(o_k)]
            else:
                # Use the non-list value.
                o_plt[k] = o_k
        # Set some defaults.
        o_plt.setdefault("color", self.get_PlotOptions(i).get("color", "k"))
        o_plt.setdefault("facecolor", o_plt["color"])
        o_plt.setdefault("alpha", 0.5)
        o_plt.setdefault("lw", 0.2)
        # Output
        return o_plt
            
    # Get the component name.
    def get_Component(self, i=None):
        """Get the component name of plot *i*
        
        :Call:
            >>> comp = DBP.get_Component(i=None)
        :Inputs:
            *DBP*: :class:`pyCart.options.DataBook.DBPlot`
                Instance of databook plot options class
            *i*: :class:`int` or ``None``
                Plot index to extract options for
        :Outputs:
            *comp*: :class:`str`
                Name of component
        :Versions:
            * 2014-12-28 ``@ddalle``: First version
        """
        # Extract component(s)
        comps = self.get("Components", [])
        # Extract safely.
        return getel(comps, i)
    
    
