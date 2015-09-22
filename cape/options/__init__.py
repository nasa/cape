"""
CAPE generic settings module: :mod:`cape.options`
=================================================

This module provides tools to read, access, modify, and write settings for
:mod:`cape`.  The class is based off of the built-int :class:`dict` class, so
its default behavior, such as ``opts['InputCntl']`` or 
``opts.get('InputCntl')`` are also present.  In addition, many convenience
methods, such as ``opts.set_it_fc(n)``, which sets the number of
:file:`flowCart` iterations,  are provided.

In addition, this module controls default values of each pyCart
parameter in a two-step process.  The precedence used to determine what the
value of a given parameter should be is below.

    *. Values directly specified in the input file, :file:`cape.json`
    
    *. Values specified in the default control file,
       :file:`$CAPE/settings/cape.default.json`
    
    *. Hard-coded defaults from this module
"""

# Import options-specific utilities (loads :mod:`os`, too)
from util import *

# Import more specific modules for controlling subgroups of options
from .pbs        import PBS
from .Management import Management

# Class definition
class Options(odict):
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
            # Strip comments and join list into a single string.
            lines = stripComments(lines, '#')
            lines = stripComments(lines, '//')
            # Get the equivalent dictionary.
            d = json.loads(lines)
            # Loop through the keys.
            for k in d:
                kw[k] = d[k]
        # Read the defaults.
        defs = getCapeDefaults()
        # Apply the defaults.
        kw = applyDefaults(kw, defs)
        # Store the data in *this* instance
        for k in kw:
            self[k] = kw[k]
        # Upgrade important groups to their own classes.
        self._PBS()
        # Add extra folders to path.
        self.AddPythonPath()
        
    # Function to add to the path.
    def AddPythonPath(self):
        """Add requested locations to the Python path
        
        :Call:
            >>> opts.AddPythonPath()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Get the "PythonPath" option
        lpath = self.get("PythonPath", [])
        # Quit if empty.
        if (not lpath): return
        # Ensure list.
        if type(lpath).__name__ != "list":
            lpath = [lpath]
        # Loop through elements.
        for fdir in lpath:
            # Add absolute path, not relative.
            os.sys.path.append(os.path.abspath(fdir))
    
    # ============
    # Initializers
    # ============
            
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
    
    
    
    # ==============
    # Global Options
    # ==============
    
    # Method to get the max number of jobs to submit.
    def get_nSubmit(self):
        """Return the maximum number of jobs to submit at one time
        
        :Call:
            >>> nSub = opts.get_nSubmit()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *nSub*: :class:`int`
                Maximum number of jobs to submit
        :Versions:
            * 2015-01-24 ``@ddalle``: First version
        """
        return self.get('nSubmit', rc0('nSubmit'))
        
    # Set the max number of jobs to submit.
    def set_nSubmit(self, nSub=rc0('nSubmit')):
        """Set the maximum number of jobs to submit at one time
        
        :Call:
            >>> opts.set_nSubmit(nSub)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *nSub*: :class:`int`
                Maximum number of jobs to submit
        :Versions:
            * 2015-01-24 ``@ddalle``: First version
        """
        self['nSubmit'] = nSub
    
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
        
    
    # ==============
    # Shell Commands
    # ==============
    
    # Function to get the shell commands
    def get_ShellCmds(self):
        """Get shell commands, if any
        
        :Call:
            >>> cmds = opts.get_ShellCmds()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *cmds*: :class:`list`
        """
        # Get the commands.
        cmds = self.get('ShellCmds', [])
        # Turn to a list if not.
        if type(cmds).__name__ != 'list':
            cmds = [cmds]
        # Output
        return cmds
        
        
    # ============
    # PBS settings
    # ============
    
    # Get number of unique PBS scripts
    def get_nPBS(self):
        self._PBS()
        return self['PBS'].get_nPBS()
    get_nPBS.__doc__ = PBS.get_nPBS.__doc__
    
    # Get PBS *join* setting
    def get_PBS_j(self, i=None):
        self._PBS()
        return self['PBS'].get_PBS_j(i)
        
    # Set PBS *join* setting
    def set_PBS_j(self, j=rc0('PBS_j'), i=None):
        self._PBS()
        self['PBS'].set_PBS_j(j, i)
    
    # Get PBS *rerun* setting
    def get_PBS_r(self, i=None):
        self._PBS()
        return self['PBS'].get_PBS_r(i)
        
    # Set PBS *rerun* setting
    def set_PBS_r(self, r=rc0('PBS_r'), i=None):
        self._PBS()
        self['PBS'].set_PBS_r(r, i)
    
    # Get PBS shell setting
    def get_PBS_S(self, i=None):
        self._PBS()
        return self['PBS'].get_PBS_S(i)
        
    # Set PBS shell setting
    def set_PBS_S(self, S=rc0('PBS_S'), i=None):
        self._PBS()
        self['PBS'].set_PBS_S(S, i)
    
    # Get PBS nNodes setting
    def get_PBS_select(self, i=None):
        self._PBS()
        return self['PBS'].get_PBS_select(i)
        
    # Set PBS nNodes setting
    def set_PBS_select(self, n=rc0('PBS_select'), i=None):
        self._PBS()
        self['PBS'].set_PBS_select(n, i)
    
    # Get PBS CPUS/node setting
    def get_PBS_ncpus(self, i=None):
        self._PBS()
        return self['PBS'].get_PBS_ncpus(i)
        
    # Set PBS CPUs/node setting
    def set_PBS_ncpus(self, n=rc0('PBS_ncpus'), i=None):
        self._PBS()
        self['PBS'].set_PBS_ncpus(n, i)
    
    # Get PBS MPI procs/node setting
    def get_PBS_mpiprocs(self, i=None):
        self._PBS()
        return self['PBS'].get_PBS_mpiprocs(i)
        
    # Set PBS *rerun* setting
    def set_PBS_mpiprocs(self, n=rc0('PBS_mpiprocs'), i=None):
        self._PBS()
        self['PBS'].set_PBS_mpiprocs(n, i)
    
    # Get PBS model or arch setting
    def get_PBS_model(self, i=None):
        self._PBS()
        return self['PBS'].get_PBS_model(i)
        
    # Set PBS model or arch setting
    def set_PBS_model(self, s=rc0('PBS_model'), i=None):
        self._PBS()
        self['PBS'].set_PBS_model(s, i)
    
    # Get PBS group setting
    def get_PBS_W(self, i=None):
        self._PBS()
        return self['PBS'].get_PBS_W(i)
        
    # Set PBS group setting
    def set_PBS_W(self, W=rc0('PBS_W'), i=None):
        self._PBS()
        self['PBS'].set_PBS_W(W, i)
    
    # Get PBS queue setting
    def get_PBS_q(self, i=None):
        self._PBS()
        return self['PBS'].get_PBS_q(i)
        
    # Set PBS queue setting
    def set_PBS_q(self, q=rc0('PBS_q'), i=None):
        self._PBS()
        self['PBS'].set_PBS_q(q, i)
    
    # Get PBS walltime setting
    def get_PBS_walltime(self, i=None):
        self._PBS()
        return self['PBS'].get_PBS_walltime(i)
        
    # Set PBS walltime setting
    def set_PBS_walltime(self, t=rc0('PBS_walltime'), i=None):
        self._PBS()
        self['PBS'].set_PBS_walltime(t, i)
        
    # Copy over the documentation.
    for k in ['PBS_j', 'PBS_r', 'PBS_S', 'PBS_select', 'PBS_mpiprocs',
            'PBS_ncpus', 'PBS_model', 'PBS_W', 'PBS_q', 'PBS_walltime']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(PBS,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(PBS,'set_'+k).__doc__
    
    
    # =================
    # Folder management
    # =================
    
    # Get the archive folder
    def get_ArchiveFolder(self):
        self._Management()
        return self['Management'].get_ArchiveFolder()
        
    # Set the archive folder
    def set_ArchiveFolder(self, fdir=rc0('ArchiveFolder')):
        self._Management()
        self['Management'].set_ArchiveFolder(fdir)
        
    # Get the archive format
    def get_ArchiveFormat(self):
        self._Management()
        return self['Management'].get_ArchiveFormat()
        
    # Set the archive format
    def set_ArchiveFormat(self, fmt=rc0('ArchiveFormat')):
        self._Management()
        self['Management'].set_ArchiveFormat(fmt)
        
    # Get the archive type
    def get_ArchiveType(self):
        self._Management()
        return self['Management'].get_ArchiveType()
        
    # Set the archive type
    def set_ArchiveType(self, atype=rc0('ArchiveType')):
        self._Management()
        self['Management'].set_ArchiveType(atype)
        
    # Get the archive action
    def get_ArchiveAction(self):
        self._Management()
        return self['Management'].get_ArchiveAction()
        
    # Set the archive action
    def set_ArchiveAction(self, fcmd=rc0('ArchiveAction')):
        self._Management()
        self['Management'].set_ArchiveAction(fcmd)
        
    # Get the remote copy command
    def get_RemoteCopy(self):
        self._Management()
        return self['Management'].get_RemoteCopy()
        
    # Set the remote copy command
    def set_RemoteCopy(self, fcmd=rc0('RemoteCopy')):
        self._Management()
        self['Management'].set_RemoteCopy(fcmd)
        
    # Copy over the documentation.
    for k in ['ArchiveFolder', 'ArchiveFormat', 'ArchiveAction', 'ArchiveType',
            'RemoteCopy', 'TarPBS']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(Management,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(Management,'set_'+k).__doc__


