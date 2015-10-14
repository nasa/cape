"""Interface for options specific to running `adjointCart`"""


# Import options-specific utilities
from util import rc0, odict

# Class for flowCart settings
class adjointCart(odict):
    """Dictionary-based interfaced for options specific to ``adjointCart``"""
    
    
    # Number of iterations for adjointCart
    def get_it_ad(self, i=None):
        """Return the number of iterations for `adjointCart`
        
        :Call:
            >>> it_fc = opts.get_it_fc(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *it_ad*: :class:`int` or :class:`list`(:class:`int`)
                Number of iterations for run *i* or all runs if ``i==None``
        :Versions:
            * 2014.08.01 ``@ddalle``: First version
        """
        return self.get_key('it_ad', i)
        
    # Set adjointCart iteration count
    def set_it_ad(self, it_ad=rc0('it_ad'), i=None):
        """Set the number of iterations for `adjointCart`
        
        :Call:
            >>> opts.set_it_ad(it_ad)
            >>> opts.set_it_ad(it_ad, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *it_ad*: :class:`int` or :class:`list`(:class:`int`)
                Number of iterations for run *i* or all runs if ``i==None``
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.01 ``@ddalle``: First version
        """
        self.set_key('it_ad', it_ad, i)
        
        
    # Get adjointCart multigrd levels
    def get_mg_ad(self, i=None):
        """Return the number of multigrid levels for `adjointCart`
        
        :Call:
            >>> mg_fc = opts.get_mg_ad(i=None)
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
        return self.get_key('mg_ad', i)
    
    # Set adjointCart iteration levels
    def set_mg_ad(self, mg_ad=rc0('mg_ad'), i=None):
        """Set number of multigrid levels for `adjointCart`
        
        :Call:
            >>> opts.set_mg_ad(mg_ad, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *mg_ad*: :class:`int` or :class:`list`(:class:`int`)
                Multigrid levels for run *i* or all runs if ``i==None``
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('mg_ad', mg_ad, i)
        
        
    # Adjoint first order
    def get_adj_first_order(self, i=None):
        """Get whether or not to run adjoins in first-order mode
        
        :Call:
            >>> adj = opts.set_adj_first_order(i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *adj*: :class:`bool` or :class:`list`(:class:`int`)
                Whether or not to always run `adjointCart` first-order
        :Versions:
            * 2014-11-17 ``@ddalle``: First version
        """
        return self.get_key('adj_first_order', i)
        
    # Adjoint first order
    def set_adj_first_order(self, adj=rc0('adj_first_order'), i=None):
        """Set whether or not to run adjoins in first-order mode
        
        :Call:
            >>> opts.set_adj_first_order(adj, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *adj*: :class:`bool` or :class:`list`(:class:`int`)
                Whether or not to always run `adjointCart` first-order
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-11-17 ``@ddalle``: First version
        """
        self.set_key('adj_first_order', adj, i)
