"""Interface for options specific to running `flowCart`"""


# Import options-specific utilities
from util import getel, setel, rc, rc0, odict

# Class for flowCart settings
class flowCart(odict):
    """Dictionary-based interfaced for options specific to ``flowCart``"""
    
    # Number of iterations
    def get_it_fc(self, i=None):
        """Return the number of iterations for `flowCart`
        
        :Call:
            >>> it_fc = opts.get_it_fc(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *it_fc*: :class:`int` or :class:`list`(:class:`int`)
                Number of iterations for run *i* or all runs if ``i==None``
        :Versions:
            * 2014.08.01 ``@ddalle``: First version
        """
        return self.get_key('it_fc', i)
        
    # Set flowCart iteration count
    def set_it_fc(self, it_fc=rc0('it_fc'), i=None):
        """Set the number of iterations for `flowCart`
        
        :Call:
            >>> opts.set_it_fc(it_fc)
            >>> opts.set_it_fc(it_fc, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *it_fc*: :class:`int` or :class:`list`(:class:`int`)
                Number of iterations for run *i* or all runs if ``i==None``
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.01 ``@ddalle``: First version
        """
        self.set_key('it_fc', it_fc, i)
        
        
    # Get flowCart multigrd levels
    def get_mg_fc(self, i=None):
        """Return the number of multigrid levels for `flowCart`
        
        :Call:
            >>> mg_fc = opts.get_mg_fc(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *mg_fc*: :class:`int` or :class:`list`(:class:`int`)
                Multigrid levels for run *i* or all runs if ``i==None``
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self_get_key('mg_fc', i)
    
    # Set flowCart iteration levels
    def set_mg_fc(self, mg_fc=rc0('mg_fc'), i=None):
        """Set number of multigrid levels for `flowCart`
        
        :Call:
            >>> opts.set_mg_fc(mg_fc, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *mg_fc*: :class:`int` or :class:`list`(:class:`int`)
                Multigrid levels for run *i* or all runs if ``i==None``
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('mg_fc', mg_fc, i)
        
    
    # Get the CFL number
    def get_cfl(self, i=None):
        """Return the CFL number for `flowCart`
        
        :Call:
            >>> cfl = opts.get_cfl(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *cfl*: :class:`float` or :class:`list`(:class:`float`)
                Multigrid levels for run *i* or all runs if ``i==None``
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('cfl', i)
    
    # Set CFL number
    def set_cfl(self, cfl=rc0('cfl'), i=None):
        """Set number of multigrid levels for `flowCart`
        
        :Call:
            >>> opts.set_mg_fc(mg_fc, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *mg_fc*: :class:`int` or :class:`list`(:class:`int`)
                Multigrid levels for run *i* or all runs if ``i==None``
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('cfl', cfl, i)
        
        