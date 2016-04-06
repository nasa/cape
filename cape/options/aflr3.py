"""
AFLR3 volume mesh generation options
====================================

This module provides a class to access command-line options to the AFLR3
mesh-generation program.  It is specified in the ``"RunControl"`` section for
modules that utilize the solver, which includes FUN3D.

The options in this module are among the command-line options to AFLR3.
"""

# Ipmort options-specific utilities
from util import rc0, odict, getel


# Resource limits class
class aflr3(odict):
    """Class for AFLR3 command-line settings
    
    :Call:
        >>> opts = aflr3(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of AFLR3 command-line options
    :Outputs:
        *opts*: :class:`cape.options.aflr3.aflr3`
            AFLR3 options interface
    :Versions:
        * 2016-04-04 ``@ddalle``: First version
    """
    # Boundary condition file
    def get_aflr3_BCFile(self, j=0):
        """Get the AFLR3 boundary condition file
        
        :Call:
            >>> fname = opts.get_aflr3_BCFile()
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
        :Outputs:
            *fname*: :class:`str`
                Name of file used to map BCs to AFLR3 surface file
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        return getel(self.get('BCFile'), j)
    
    # Boundary condition file
    def set_aflr3_BCFile(self, fname, j=0):
        """Set the AFLR3 boundary condition file
        
        :Call:
            >>> opts.set_aflr3_BCFile(fname)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fname*: :class:`str`
                Name of file used to map BCs to AFLR3 surface file
        :Versions:
            * 2016-04-05 ``@ddalle``: First version
        """
        self.set_key('BCFile', fname, j)
        
    # Get a AFLR3 input file
    def get_aflr3_i(self, j=0):
        """Get the input file for AFLR3
        
        :Call:
            >>> fname = opts.get_aflr3_i(j=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *fname*: :class:`instr`
                Name of input file (usually ``.surf``)
        :Versions:
            * 2016-04-04 ``@ddalle``: First version
        """
        return self.get_key('i', j)
    
    # Set the AFLR3 input file
    def set_aflr3_i(self, fname, j=0):
        """Set the input file for AFLR3
        
        :Call:
            >>> opts.set_aflr3_i(fname, j=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fname*: :class:`instr`
                Name of input file (usually ``.surf``)
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-04-04 ``@ddalle``: First version
        """
        self.set_key('i', fname, j)
        
    # Get a AFLR3 output file
    def get_aflr3_o(self, j=0):
        """Get the output file for AFLR3
        
        :Call:
            >>> fname = opts.get_aflr3_o(j=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *fname*: :class:`instr`
                Name of output file (``.ugrid``)
        :Versions:
            * 2016-04-04 ``@ddalle``: First version
        """
        return self.get_key('o', j)
    
    # Set the AFLR3 output file
    def set_aflr3_o(self, fname, j=0):
        """Set the output file for AFLR3
        
        :Call:
            >>> opts.set_aflr3_o(fname, j=0)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *fname*: :class:`instr`
                Name of output file (``.ugrid``)
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-04-04 ``@ddalle``: First version
        """
        self.set_key('o', fname, j)
        
    # Get the boundary layer stretching ratio
    def get_blr(self, j=None):
        """Get the AFLR3 boundary layer stretching ratio
        
        :Call:
            >>> blr = opts.get_blr(j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *blr*: :class:`float` (1.0 < *blr* < 4.0)
                Stretching ratio
        :Versions:
            * 2016-04-04 ``@ddalle``: First version
        """
        return self.get_key('blr', j)
        
    # Set the boundary layer stretching ratio
    def set_blr(self, blr, j=None):
        """Get the AFLR3 boundary layer stretching ratio
        
        :Call:
            >>> opts.set_blr(blr, j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *blr*: :class:`float` (1.0 < *blr* < 4.0)
                Stretching ratio
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-04-04 ``@ddalle``: First version
        """
        self.set_key('blr', blr, j)
    
    # Get BL grid option
    def get_blc(self, j=None):
        """Get the AFLR3 BL option with prism layers
        
        :Call:
            >>> blc = opts.get_blc(j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *blc*: :class:`bool`
                Whether or not to create BL grid with prisms
        :Versions:
            * 2016-04-04 ``@ddalle``: First version
        """
        return self.get_key('blc', j)
        
    # Set the BL grid option
    def set_blc(self, blc, j=None):
        """Get the AFLR3 BL option with prism layers
        
        :Call:
            >>> opts.set_blc(blc, j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *blc*: :class:`bool`
                Whether or not to create BL grid with prisms
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-04-04 ``@ddalle``: First version
        """
        self.set_key('blc', blc, j)
    
    # Get the initial surface spacing
    def get_blds(self, j=None):
        """Get the initial boundary-layer spacing
        
        :Call:
            >>> blds = opts.get_blds(j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *blds*: :class:`float`
                Initial boundary layer spacing
        :Versions:
            * 2016-04-04 ``@ddalle``: First version
        """
        return self.get_key('blds', j)
        
    # Set the initial surface spacing
    def set_blds(self, blds, j=None):
        """Set the initial boundary-layer spacing
        
        :Call:
            >>> opts.set_blds(blds, j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *blds*: :class:`float`
                Initial boundary layer spacing
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-04-04 ``@ddalle``: First version
        """
        self.set_key('blds', blds, j)
        
    # Maximum angle between BL intersecting faces
    def get_angblisimx(self, j=None):
        """Get the maximum angle between BL intersecting faces
        
        :Call:
            >>> angbli = opts.get_angblisimx(j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *angbli*: :class:`float` (100 <= *angbli* < 180)
                Max BL intersecting face angle
        :Versions:
            * 2016-04-04 ``@ddalle``: First version
        """
        return self.get_key('angblisimx', j)
        
    # Maximum angle between BL intersecting faces
    def set_angblisimx(self, angblisimx, j=None):
        """Set the maximum angle between BL intersecting faces
        
        :Call:
            >>> opts.get_angblisimx(angbli, j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *angbli*: :class:`float` (100 <= *angbli* < 180)
                Max BL intersecting face angle
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-04-04 ``@ddalle``: First version
        """
        return self.get_key('angblisimx', j)
    
# class aflr3

