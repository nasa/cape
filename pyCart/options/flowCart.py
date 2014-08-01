"""Interface for options specific to running `flowCart`"""


# Import options-specific utilities
from util import getel, setel, rc0, rc


# Class for flowCart settings
class flowCart(dict):
    
    # Number of iterations
    def get_it_fc(self, i=None):
        """Return the number of iterations for `flowCart`"""
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
        
        