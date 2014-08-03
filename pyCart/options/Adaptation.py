"""Interface for Cart3D adaptation settings"""


# Import options-specific utilities
from util import rc0, odict

# Class for flowCart settings
class Adaptation(odict):
    """Dictionary-based interfaced for options for Cart3D adaptation"""
    
    
    # Get number of adaptation cycles
    def get_n_adapt_cycles(self, i=None):
        """Return the number of Cart3D number of adaptation cycles
        
        :Call:
            >>> nAdapt = opts.get_n_adapt_cycles(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *nAdapt*: :class:`int` or :class:`list`(:class:`int`)
                Number of adaptation cycles
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('n_adapt_cycles', i)
        
    # Set adjointCart iteration count
    def set_n_adapt_cycles(self, nAdapt=rc0('n_adapt_cycles'), i=None):
        """Set the number of Cart3D adaptation cycles
        
        :Call:
            >>> opts.set_n_adaptation_cycles(nAdapt, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *nAdapt*: :class:`int` or :class:`list`(:class:`int`)
                Number of iterations for run *i* or all runs if ``i==None``
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('n_adapt_cycles', nAdapt, i)
        
        
    # Get the adaptation tolerance
    def get_etol(self, i=None):
        """Return the target output error tolerance
        
        :Call:
            >>> etol = opts.get_etol(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *etol*: :class:`float` or :class:`list`(:class:`float`)
                Output error tolerance
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('etol', i)
        
    # Set the adaptation tolerance
    def set_etol(self, etol=rc0('etol'), i=None):
        """Set the output error tolerance
        
        :Call:
            >>> opts.set_etol(etol, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *etol*: :class:`float` or :class:`list`(:class:`float`)
                Output error tolerance
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('etol', etol, i)
        
        
    # Get the maximum cell count
    def get_max_nCells(self, i=None):
        """Return the maximum cell count
        
        :Call:
            >>> max_nCells = opts.get_max_nCells(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *etol*: :class:`float` or :class:`list`(:class:`float`)
                Output error tolerance
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('etol', i)
    
    # Set the maximum cell count
    def set_max_nCells(self, max_nCells=rc0('max_nCells'), i=None):
        """Return the maximum cell count
        
        :Call:
            >>> max_nCells = opts.get_max_nCells(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *etol*: :class:`float` or :class:`list`(:class:`float`)
                Output error tolerance
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('max_nCells', max_nCells)
        
        
    # Get the number of flowCart iterations for refined meshes
    def get_ws_it(self, i=None):
        """Get number of `flowCart` iterations on refined mesh *i*
        
        :Call:
            >>> ws_it = opts.get_ws_it(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *ws_it*: :class:`int` or :class:`list`(:class:`int`)
                Number of `flowCart` iterations
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('ws_it', i)
    
    # Set the number of flowcart iterations fore refined meshes
    def set_ws_it(self, ws_it=rc0('ws_it'), i=None):
        """Set number of `flowCart` iterations on refined mesh *i*
        
        :Call:
            >>> opts.set_ws_it(ws_it, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *ws_it*: :class:`int` or :class:`list`(:class:`int`)
                Number of `flowCart` iterations
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('ws_it', ws_it, i)
        
    
    # Get the mesh growth ratio for refinement i
    def get_mesh_growth(self, i=None):
        """Get the refinement cell count ratio
        
        :Call:
            >>> mesh_growth = opts.get_mesh_growth(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *mesh_growth*: :class:`float` or :class:`list`(:class:`float`)
                Refinement mesh growth ratio
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('mesh_growth', i)
    
    # Set the number of flowcart iterations fore refined meshes
    def set_mesh_growth(self, mesh_growth=rc0('mesh_growth'), i=None):
        """Set the refinement cell count ratio
        
        :Call:
            >>> opts.set_mesh_growth(mesh_growth, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *mesh_growth*: :class:`float` or :class:`list`(:class:`float`)
                Refinement mesh growth ratio
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('mesh_growth', mesh_growth, i)
    
    
    # Get the adaptation type
    def get_apc(self, i=None):
        """Get the adaptation type
        
        :Call:
            >>> apc = opts.get_apc(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *apc*: :class:`str` or :class:`list`(:class:`str`)
                Adaptation cycle type, ``"a"`` or ``"p"``
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('apc', i)
    
    # Set the adaptation type
    def set_apc(self, apc=rc0('apc'), i=None):
        """Set the adaptation type
        
        :Call:
            >>> apc = opts.get_apc(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *apc*: :class:`str` or :class:`list`(:class:`str`)
                Adaptation cycle type, ``"a"`` or ``"p"``
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('apc', apc, i)









