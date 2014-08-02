"""Interface for options specific to running `flowCart`"""


# Import options-specific utilities
from util import getel, setel

rc = {
    "it_fc": 200,
    "cfl": 1.1,
    "cflmin": 0.8,
    "mg_fc": 3,
    "limiter": 2,
    "y_is_spanwise": True,
    "binaryIO": True,
    "OMP_NUM_THREADS": 8,
    "tm": False,
}

# Function to ensure scalar from above
def rc0(p):
    """Return default setting, but ensure a scalar"""
    # Use the `getel` function to do this.
    return getel(rc[p], 0)

# Class for flowCart settings
class flowCart(dict):
    
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
        # Return the iteration number
        it_fc = self.get('it_fc', rc["it_fc"])
        # Safe indexing
        return getel(it_fc, i)
        
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
        # Get current setting safely
        IT_FC = self.get('it_fc', rc['it_fc']) 
        # Set the iteration count
        self['it_fc'] = setel(IT_FC, i, it_fc)
        
        
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
        # Safe retrieval
        mg_fc = self.get('mg_fc', rc["mg_fc"])
        # Safe indexing
        return getel(mg_fc, i)
    
    # Set flowCart iteration levels
    def set_mg_fc(self, mg_fc=rc0('mg_fc'), i=None):
        """Set number of multigrid levels for `flowCart`
        
        :Call:
            >>> mg_fc = opts.get_mg_fc(i=None)
            
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
            *mg_fc*: :class:`int` or :class:`list`(:class:`int`)
                Multigrid levels for run *i* or all runs if ``i==None``
                
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        # Ensure flowCart settings
        self.setdefault('flowCart', {})
        # Safe setting
        MG_FC = self['flowCart'].get('mg_fc', rc['mg_fc'])
        # Set the multigrid levels
        self['flowCart']['mg_fc'] = mg_fc
        
        