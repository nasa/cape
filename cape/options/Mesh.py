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
    
    # Mesh filenames
    def get_MeshFile(self, i=None):
        """Return the original mesh file names
        
        :Call:
            >>> fname = opts.get_MeshFile(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Optional index
        :Outputs:
            *fname*: :class:`str` | :class:`list` (:class:`str`)
                Mesh file name or list of files
        :Versions:
            * 2015-10-19 ``@ddalle``: First version
        """
        return self.get_key('MeshFile', i)

    # Get the triangulation file(s)
    def get_TriFile(self, i=None):
        """Return the surface triangulation file
        
        :Call:
            >>> fname = opts.get_TriFile(i=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Run index
        :Outputs:
            *fname*: :class:`str` or :class:`list`(:class:`str`)
                Surface triangulation file
        :Versions:
            * 2014-08-03 ``@ddalle``: First version
            * 2016-03-29 ``@ddalle``: Copied from :mod:`pyCart.options`
        """
        return self.get_key('TriFile', i)
