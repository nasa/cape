"""
Cart3D triangulation intersect options
======================================

This module interfaces options for two Cart3D triangulation tools called
``intersect`` and ``verify``.  This allows the user to intersect overlapping
triangulated bodies and create a single water-tight body.  The ``verify`` script
then tests for a well-defined water-tight triangulation.  Intersections, missing
triangles, holes, etc. all cause ``verify`` to fail.

The main option for both ``intersect`` and ``verify`` is simply whether or not
to use them.  The user can specify the following in the ``"RunControl"`` section
of the JSON file.

    .. code-block:: javascript
    
        "RunControl": {
            "intersect": True,
            "verfiy": False
        }
        
There are a variety of input flags to those utilities, which are interfaced via
the :class:`cape.options.intersect.intersect` and
:class:`cape.options.intersect.verify` classes defined here.
"""

# Ipmort options-specific utilities
from util import rc0, odict, getel


# Resource limits class
class intersect(odict):
    """Class for Cart3D ``intersect`` command-line settings
    
    :Call:
        >>> opts = intersect(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of ``intersect`` command-line options
    :Outputs:
        *opts*: :class:`cape.options.intersect.intersect`
            Intersect utility options interface
    :Versions:
        * 2016-04-05 ``@ddalle``: First version
    """
    # Get a intersect input file
    def get_intersect_i(self, j=0):
        """Get the input file for ``intersect``
        
        :Call:
            >>> fname = opts.get_intersect_i(j=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *fname*: :class:`str`
                Name of input file
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        return self.get_key('i', j)
    
    # Set the intersect input file
    def set_intersect_i(self, fname, j=None):
        """Set the input file for ``intersect``
        
        :Call:
            >>> opts.set_intersect_i(fname, j=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fname*: :class:`str`
                Name of input file (usually ``.surf``)
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        self.set_key('i', fname, j)
        
    # Get a intersect output file
    def get_intersect_o(self, j=0):
        """Get the output file for ``intersect``
        
        :Call:
            >>> fname = opts.get_intersect_o(j=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *fname*: :class:`str`
                Name of output file
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        return self.get_key('o', j)
    
    # Set the intersect output file
    def set_intersect_o(self, fname, j=None):
        """Set the output file for ``intersect``
        
        :Call:
            >>> opts.set_intersect_o(fname, j=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fname*: :class:`str`
                Name of output file
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        self.set_key('o', fname, j)
    
    # ASCII setting
    def get_ascii(self, j=0):
        """Get the ASCII setting for the input file
        
        :Call:
            >>> ascii = opts.get_ascii(j)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *ascii*: :class:`bool`
                Whether or not input file is ASCII
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        return self.get_key('ascii', j)
    
    # ASCII setting
    def set_ascii(self, ascii, j=None):
        """Set the ASCII setting for the input file
        
        :Call:
            >>> opts.get_ascii(ascii, j)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *ascii*: :class:`bool`
                Whether or not input file is ASCII
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        self.set_key('ascii', j)
        
    # Subtraction option
    def get_cutout(self, j=0):
        """Get the number of the component to subtract (cut out)
        
        :Call:
            >>> cutout = opts.get_cutout(j=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int`
                Phase number
        :Outputs:
            *cutout*: :class:`int`
                Component ID
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        return self.get_key('cutout', j)
        
    # Subtraction option
    def set_cutout(self, cutout, j=None):
        """Set the component ID to subtract (cut out)
        
        :Call:
            >>> opts.set_cutout(cutout, j=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *cutout*: :class:`int`
                Component ID
            *j*: :class:`int`
                Phase number
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        self.set_key('cutout', cutout, j)
    
    # Get rm small tri option
    def get_intersect_rm(self):
        """Get the option to remove small triangles

        :Call:
            >>> q = opts.get_intersect_rm()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *q*: ``True`` | ``False``
                Whether or not to remove small triangles
        :Versions:
            * 2017-06-24 ``@ddalle``: First version
        """
        return self.get_key('rm', rck='intersect_rm')

    # Set rm small tri option
    def set_intersect_rm(self, q=rc0('intersect_rm')):
        """Set the option to remove small triangles

        :Call:
            >>> opts.set_intersect_rm(q)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *q*: ``True`` | ``False``
                Whether or not to remove small triangles
        :Versions:
            * 2017-06-24 ``@ddalle``: First version
        """
        self.set_key('rm', q)

    # Get small triangle size for intersect
    def get_intersect_smalltri(self):
        """Get the cutoff size for small triangles

        :Call:
            >>> A = opts.get_intersect_smalltri()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *A*: :class:`float`
                Small area cutoff
        :Versions:
            * 2017-06-24 ``@ddalle``: First version
        """
        return self.get_key('smalltri', rck='intersect_smalltri')

    # Set small triangle size
    def set_intersect_smalltri(self, smalltri=rc0('intersect_smalltri')):
        """Get the cutoff size for small triangles

        :Call:
            >>> opts.get_intersect_smalltri(A)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *A*: :class:`float`
                Small area cutoff
        :Versions:
            * 2017-06-24 ``@ddalle``: First version
        """
        self.set_key('smalltri', smalltri)
        
# class intersect


# Resource limits class
class verify(odict):
    """Class for Cart3D ``verify`` command-line settings
    
    :Call:
        >>> opts = verify(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of ``verify`` command-line options
    :Outputs:
        *opts*: :class:`cape.options.intersect.verify`
            Intersect utility options interface
    :Versions:
        * 2016-04-05 ``@ddalle``: First version
    """
    # Get a verify input file
    def get_verify_i(self, j=0):
        """Get the input file for ``verify``
        
        :Call:
            >>> fname = opts.get_verify_i(j=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *fname*: :class:`str`
                Name of input file
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        return self.get_key('i', j)
    
    # Set the intersect input file
    def set_verify_i(self, fname, j=None):
        """Set the input file for ``verify``
        
        :Call:
            >>> opts.set_verify_i(fname, j=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fname*: :class:`str`
                Name of input file
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        self.set_key('i', fname, j)
    
    # ASCII setting
    def get_ascii(self, j=0):
        """Get the ASCII setting for the input file
        
        :Call:
            >>> ascii = opts.get_ascii(j)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *ascii*: :class:`bool`
                Whether or not input file is ASCII
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        return self.get_key('ascii', j)
    
    # ASCII setting
    def set_ascii(self, ascii, j=None):
        """Set the ASCII setting for the input file
        
        :Call:
            >>> opts.get_ascii(ascii, j)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *ascii*: :class:`bool`
                Whether or not input file is ASCII
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        self.set_key('ascii', j)
        
# class verify

