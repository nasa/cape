"""
Template for Cape meshing options
=================================

This module contains the skeleton of settings for meshes.  Since most meshing
options are specific to codes, very few options are contained in this overall
template.
"""


# Import options-specific utilities
from util import rc0, odict

# Class for Cart3D mesh settings
class Mesh(odict):
    """Dictionary-based interfaced for options for Cart3D meshing"""
    
    # Initialization method
    def __init__(self, **kw):
        # Store the data in *this* instance
        for k in kw:
            self[k] = kw[k]
