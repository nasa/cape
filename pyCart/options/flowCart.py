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
        """
        # Return the iteration number
        it_fc = self.get('it_fc', rc["it_fc"])
        # Safe indexing
        return getel(it_fc, i)
        
    # Set flowCart iteration count
    def set_it_fc(self, it_fc=rc0('it_fc'), i=None):
        """Set the number of iterations for `flowCart`"""
        # Get current setting safely
        IT_FC = self.get('it_fc', rc['it_fc']) 
        # Set the iteration count
        self['it_fc'] = setel(IT_FC, i, it_fc)
        
        