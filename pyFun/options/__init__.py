"""
Cart3D and pyCart settings module: :mod:`pyCart.options`
========================================================

This module provides tools to read, access, modify, and write settings for
:mod:`pyCart`.  The class is based off of the built-int :class:`dict` class, so
its default behavior, such as ``opts['InputCntl']`` or 
``opts.get('InputCntl')`` are also present.  In addition, many convenience
methods, such as ``opts.set_it_fc(n)``, which sets the number of
:file:`flowCart` iterations,  are provided.

In addition, this module controls default values of each pyCart
parameter in a two-step process.  The precedence used to determine what the
value of a given parameter should be is below.

    *. Values directly specified in the input file, :file:`pyCart.json`
    
    *. Values specified in the default control file,
       :file:`$PYCART/settings/pyCart.default.json`
    
    *. Hard-coded defaults from this module
"""

# Import options-specific utilities (loads :mod:`os`, too)
from util import *

# Import template module
import cape.options

# Import modules for controlling specific parts of Cart3D
from .pbs         import PBS
from .DataBook    import DataBook
from .Report      import Report
from .runControl  import RunControl


# Class definition
class Options(cape.options.Options):
    """
    Options structure, subclass of :class:`dict`
    
    :Call:
        >>> opts = Options(fname=None, **kw)
    :Inputs:
        *fname*: :class:`str`
            File to be read as a JSON file with comments
        *kw*: :class:`dict`
            Dictionary to be transformed into :class:`pyCart.options.Options`
    :Versions:
        * 2014.07.28 ``@ddalle``: First version
    """
    
    # Initialization method
    def __init__(self, fname=None, **kw):
        """Initialization method with optional JSON input"""
        # Check for an input file.
        if fname:
            # Read the input file.
            lines = open(fname).readlines()
            # Expand references to other JSON files and strip comments
            lines = expandJSONFile(lines)
            # Get the equivalent dictionary.
            d = json.loads(lines)
            # Loop through the keys.
            for k in d:
                kw[k] = d[k]
        # Read the defaults.
        defs = getPyFunDefaults()
        
        # Apply the defaults.
        kw = applyDefaults(kw, defs)
        # Store the data in *this* instance
        for k in kw:
            self[k] = kw[k]
        # Upgrade important groups to their own classes.
        self._PBS()
        self._DataBook()
        self._Report()
        self._RunControl()
        # Add extra folders to path.
        self.AddPythonPath()
    
    # ============
    # Initializers
    # ============
   # <
    
    # Initialization and confirmation for PBS options
    def _PBS(self):
        """Initialize PBS options if necessary"""
        # Check status.
        if 'PBS' not in self:
            # Missing entirely
            self['PBS'] = PBS()
        elif type(self['PBS']).__name__ == 'dict':
            # Add prefix to all the keys.
            tmp = {}
            for k in self['PBS']:
                tmp["PBS_"+k] = self['PBS'][k]
            # Convert to special class.
            self['PBS'] = PBS(**tmp)
            
    # Initialization method for overall run control
    def _RunControl(self):
        """Initialize report options if necessary"""
        # Check status.
        if 'RunControl' not in self:
            # Missing entirely.
            self['RunControl'] = Report()
        elif type(self['RunControl']).__name__ == 'dict':
            # Convert to special class
            self['RunControl'] = RunControl(**self['RunControl'])
    
    # Initialization method for databook
    def _DataBook(self):
        """Initialize data book options if necessary"""
        # Check status.
        if 'DataBook' not in self:
            # Missing entirely.
            self['DataBook'] = DataBook()
        elif type(self['DataBook']).__name__ == 'dict':
            # Convert to special class
            self['DataBook'] = DataBook(**self['DataBook'])
            
    # Initialization method for automated report
    def _Report(self):
        """Initialize report options if necessary"""
        # Check status.
        if 'Report' not in self:
            # Missing entirely.
            self['Report'] = Report()
        elif type(self['Report']).__name__ == 'dict':
            # Convert to special class
            self['Report'] = Report(**self['Report'])
   # >
    
    # ==============
    # Global Options
    # ==============
   # <
    
    # Method to get the namelist template
    def get_Namelist(self):
        """Return the name of the master :file:`fun3d.nml` file
        
        :Call:
            >>> fname = opts.get_Namelist()
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
        :Outputs:
            *fname*: :class:`str`
                Name of FUN3D namelist template file
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        return self.get('InputCntl', rc0('InputCntl'))
        
    # Method to set the namelist template
    def set_Namelist(self, fname):
        """Set the name of the master :file:`fun3d.nml` file
        
        :Call:
            >>> opts.set_Namelist(fname)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *fname*: :class:`str`
                Name of FUN3D namelist template file
        :Versions:
            * 2015-10-16 ``@ddalle``: First version
        """
        self['Namelist'] = fname
    
    # Method to determine if groups have common meshes.
    def get_GroupMesh(self):
        """Determine whether or not groups have common meshes
        
        :Call:
            >>> qGM = opts.get_GroupMesh()
        :Inputs:
            *opts* :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *qGM*: :class:`bool`
                True all cases in a group use the same (starting) mesh
        :Versions:
            * 2014-10-06 ``@ddalle``: First version
        """
        # Safely get the trajectory.
        x = self.get('Trajectory', {})
        return x.get('GroupMesh', rc0('GroupMesh'))
        
    # Method to specify that meshes do or do not use the same mesh
    def set_GroupMesh(self, qGM=rc0('GroupMesh')):
        """Specify that groups do or do not use common meshes
        
        :Call:
            >>> opts.get_GroupMesh(qGM)
        :Inputs:
            *opts* :class:`pyCart.options.Options`
                Options interface
            *qGM*: :class:`bool`
                True all cases in a group use the same (starting) mesh
        :Versions:
            * 2014-10-06 ``@ddalle``: First version
        """
        self['Trajectory']['GroupMesh'] = qGM
   # >
    
    # ===================
    # Overall run control
    # ===================
   # <
   
    # Get number of inputs
    def get_nSeq(self):
        self._RunControl()
        return self['RunControl'].get_nSeq()
    # Copy documentation
    get_nSeq.__doc__ = RunControl.get_nSeq.__doc__
    
    # Get input sequence
    def get_InputSeq(self, i=None):
        self._RunControl()
        return self['RunControl'].get_InputSeq(i)
        
    # Set input sequence
    def set_InputSeq(self, InputSeq=rc0('InputSeq'), i=None):
        self._RunControl()
        self['RunControl'].set_InputSeq(InputSeq, i)
        
    # Get iteration break points
    def get_IterSeq(self, i=None):
        self._RunControl()
        return self['RunControl'].get_IterSeq(i)
        
    # Set Iteration break points
    def set_IterSeq(self, IterSeq=rc0('IterSeq'), i=None):
        self._RunControl()
        return self['RunControl'].set_IterSeq(IterSeq, i)
        
    # Get MPI status
    def get_MPI(self, i=None):
        self._RunControl()
        return self['RunControl'].get_mpi_fc(i)
        
    # Set MPI status
    def set_MPI(self, MPI=rc0('MPI'), i=None):
        self._RunControl()
        self['RunControl'].set_mpi_fc(mpi_fc, i)
        
    # Get the number of threads for RunControl
    def get_nProc(self, i=None):
        self._RunControl()
        return self['RunControl'].get_nProc(i)
        
    # Set the number of threads for RunControl
    def set_nProc(self, nProc=rc0('nProc'), i=None):
        self._RunControl()
        self['RunControl'].set_nProc(nProc, i)
        
    # Get the MPI system command
    def get_mpicmd(self, i=None):
        self._RunControl()
        return self['RunControl'].get_mpicmd(i)
        
    # Set the MPI system command
    def set_mpicmd(self, mpicmd=rc0('mpicmd'), i=None):
        self._RunControl()
        self['RunControl'].set_mpicmd(mpicmd, i)
        
    # Get the submittable/nonsubmittalbe status
    def get_qsub(self, i=None):
        self._RunControl()
        return self['RunControl'].get_qsub(i)
        
    # Set the submittable/nonsubmittalbe status
    def set_qsub(self, qsub=rc0('qsub'), i=None):
        self._RunControl()
        self['RunControl'].set_qsub(qsub, i)
        
    # Get the resubmittable/nonresubmittalbe status
    def get_resub(self, i=None):
        self._RunControl()
        return self['RunControl'].get_resub(i)
        
    # Set the resubmittable/nonresubmittalbe status
    def set_resub(self, resub=rc0('resub'), i=None):
        self._RunControl()
        self['RunControl'].set_resub(resub, i)
        
    # Copy over the documentation.
    for k in ['InputSeq', 'IterSeq',  
            'MPI', 'nProc', 'mpicmd', 'qsub', 'resub']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(RunControl,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(RunControl,'set_'+k).__doc__
   # >
   
        
    # ============
    # PBS settings
    # ============
   # <
   # >
    
    
    # =================
    # Folder management
    # =================
   # <
        
        
    ## Copy over the documentation.
    #for k in []:
    #    # Get the documentation for the "get" and "set" functions
    #    eval('get_'+k).__doc__ = getattr(Management,'get_'+k).__doc__
    #    eval('set_'+k).__doc__ = getattr(Management,'set_'+k).__doc__
   # >
   
    
    # =============
    # Configuration
    # =============
   # <
        
    
        
    ## Copy over the documentation.
    #for k in ['ClicForce', 'Xslice', 'Yslice', 'Zslice',
    #        'PointSensor', 'LineSensor']:
    #    # Get the documentation for the "get" and "set" functions
    #    eval('get_'+k+'s').__doc__ = getattr(Config,'get_'+k+'s').__doc__
    #    eval('set_'+k+'s').__doc__ = getattr(Config,'set_'+k+'s').__doc__
    #    eval('add_'+k).__doc__ = getattr(Config,'add_'+k).__doc__
   # >
   
    
    
    
    # =========
    # Data book
    # =========
   # <
    
    # Copy over the documentation.
    for k in []:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(DataBook,'get_'+k).__doc__
   # >
   
    
    # =======
    # Reports
    # =======
   # <
    
    # Copy over the documentation
    for k in []:
        # Get the documentation from the submodule
        eval('get_'+k).__doc__ = getattr(Report,'get_'+k).__doc__
   # >
    
# class Options


