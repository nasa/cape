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
    def set_aflr3_i(self, fname, j=None):
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
    def set_aflr3_o(self, fname, j=None):
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
        
    # Get the number of prism layers
    def get_bli(self, j=None):
        """Get the number of AFLR3 prism layers
        
        :Call:
            >>> bli = opts.get_bli(j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *bli*: :class:`int` > 0
                Number of prism layers
        :Versions:
            * 2017-04-26 ``@ddalle``: First version
        """
        return self.get_key('bli', j)
        
    # Set the number of prism layers
    def set_bli(self, bli, j=None):
        """Set the number of AFLR3 prism layers
        
        :Call:
            >>> opts.set_bli(bli, j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *bli*: :class:`int` > 0
                Maximum number of prism layers
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2017-04-26 ``@ddalle``: First version
        """
        self.set_key('bli', bli, j)
    
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

    # Growth flag
    def get_grow(self, j=None):
        """Get the growth option for AFLR3

        :Call:
            >>> grow = opts.get_grow(j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *grow*: 1 < :class:`float` <= 2
                Growth parameter
        :Versions:
            * 2017-04-26 ``@ddalle``: First version
        """
        return self.get_key('grow', j)

    # Set growth flag
    def set_grow(self, grow, j=None):
        """Set the growth option for AFLR3

        :Call:
            >>> opts.get_grow(grow, j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *grow*: 1 < :class:`float` <= 2
                Growth parameter
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2017-04-26 ``@ddalle``: First version
        """
        self.set_key('grow', grow, j)
        
    # Get the volume grid stretching
    def get_cdfr(self, j=None):
        """Get the maximum geometric growth rate in the volume region
        
        :Call:
            >>> cdfr = opts.get_cdfr(j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *cdfr*: {``1.1``} | :class:`float`; 1 < *cdfr* <= 3
                Max geometric growth rate
        :Versions:
            * 2016-05-06 ``@ddalle``: First version
        """
        return self.get_key('cdfr', j, rck='aflr3_cdfr')
        
    # Grid spacing exclusion zone
    def get_cdfs(self, j=None):
        """Get the geometric growth exclusion zone
        
        :Call:
            >>> cdfs = opts.get_cdfr(j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *cdfs*: {``None``} | :class:`float`; 0 <= *cdfs* <= 10
                Exclusion zone size
        :Versions:
            * 2017-04-26 ``@ddalle``: First version
        """
        return self.get_key('cdfs', j, rck='aflr3_cdfs')
        
    # Set the max geometric growth rate
    def set_cdfr(self, cdfr, j=None):
        """Set the maximum geometric growth rate in the volume region
        
        :Call:
            >>> opts.set_cdfr(cdfr, j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *cdfr*: :class:`float`; 1 < *cdfr* <= 3
                Max geometric growth rate
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-05-06 ``@ddalle``: First version
        """
        self.set_key('cdfr', cdfr, j)
        
    # Set the geometric exclusion zone
    def set_cdfs(self, cdfs, j=None):
        """Set the exclusion zone for geometric growth
        
        :Call:
            >>> opts.set_cdfs(cdfs, j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *cdfs*: :class:`float`; 0 <= *cdfr* <= 10
                Geometric growth exclusion zone size
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2017-04-26 ``@ddalle``: First version
        """
        self.set_key('cdfs', cdfs, j)
        
    # Get the number of quality improvement passes
    def get_nqual(self, j=None):
        """Get the number of mesh quality improvement passes
        
        :Call:
            >>> nqual = opts.get_nqual(j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *nqual*: {``2``} | :class:`int` 0 <= *nqual* <= 10
                Number of mesh quality improvement passes
        :Versions:
            * 2016-05-07 ``@ddalle``: First version
        """
        return self.get_key('nqual', j, rck='aflr3_nqual')
        
    # Set the number of quality improvement passes
    def set_nqual(self, nqual, j=None):
        """Set the number of mesh quality improvement passes
        
        :Call:
            >>> opts.set_nqual(nqual, j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *nqual*: :class:`int` 0 <= *nqual* <= 10
                Number of mesh quality improvement passes
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-05-07 ``@ddalle``: First version
        """
        self.set_key('nqual', nqual, j)
        
    # Maximum angle between BL intersecting faces
    def get_angqbf(self, j=None):
        """Get the maximum angle on surface triangles
        
        :Call:
            >>> angbli = opts.get_angqbf(j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *angqbf*: :class:`float` (100 <= *angqbf* <= 180)
                Max surface angle
        :Versions:
            * 2017-06-16 ``@ddalle``: First version
        """
        return self.get_key('angqbf', j)
        
    # Maximum angle between BL intersecting faces
    def set_angqbf(self, angqbf, j=None):
        """Set the maximum angle on surface tris
        
        :Call:
            >>> opts.get_angblisimx(angbli, j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *angqbf*: :class:`float` (100 <= *angqbf* <= 180)
                Max surface triangle angle
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2017-06-16 ``@ddalle``: First version
        """
        return self.set_key('angqbf', angqbf, j)
        
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
        self.set_key('angblisimx', angblisimx, j)
        
    # Tolerance for bad surface cells
    def get_angqbf(self, j=None):
        """Get the AFLR3 option *angqbf*
        
        Setting this option to ``0`` will allow for mesh generation from
        lower-quality surface meshes.
        
        :Call:
            >>> angqbf = opts.get_angqbf(j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *angqbf*: {``None``} | :class:`float` >= 0
                AFLR3 option
        :Versions:
            * 2017-06-13 ``@ddalle``: First version
        """
        return self.get_key('angqbf', j)
        
    # Set bad surface cell option
    def set_angqbf(self, angqbf, j=None):
        """Get the AFLR3 option *angqbf*
        
        Setting this option to ``0`` will allow for mesh generation from
        lower-quality surface meshes.
        
        :Call:
            >>> opts.set_angqbf(angqbf, j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *angqbf*: {``None``} | :class:`float` >= 0
                AFLR3 option
            *j*: {``None``} | :class:`int`
                Phase number
        :Versions:
            * 2017-06-13 ``@ddalle``: First version
        """
        return self.set_key('angqbf', angqbf, j)
    
    # Distribution function flag
    def get_mdf(self, j=None):
        """Get the AFLR3 volume grid distribution flag

        :Call:
            >>> mdsblf = opts.get_mdsblf(j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *mdf*: ``1`` | {``2``}
                Volume distribution flag
        :Versions:
            * 2017-04-26 ``@ddalle``: First version
        """
        return self.get_key('mdf', j, rck="aflr3_mdf")
    
    # BL distribution function flag
    def set_mdf(self, mdf, j=None):
        """Set the AFLR3 volume grid distribution flag

        :Call:
            >>> opts.set_mdf(mdf, j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *mdf*: ``1`` | {``2``}
                Volume distribution flag
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2017-04-26 ``@ddalle``: First version
        """
        self.set_key('mdf', mdf, j)
    
    # BL spacing thickness factor option
    def get_mdsblf(self, j=None):
        """Get the BL spacing thickness factor option

        :Call:
            >>> mdsblf = opts.get_mdsblf(j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *mdsblf*: ``0`` | {``1``} | ``2``
                BL spacing flag if *mdl* is not ``0``
        :Versions:
            * 2017-04-26 ``@ddalle``: First version
        """
        return self.get_key('mdsblf', j, rck="aflr3_mdsblf")
    
    # BL spacing thickness factor option
    def set_mdsblf(self, mdsblf, j=None):
        """Set the BL spacing thickness factor option

        :Call:
            >>> opts.set_mdsblf(mdsblf, j=None)
        :Inputs:
            *opts*: :class:`cape.options.Options`
                Options interface
            *mdsblf*: ``0`` | {``1``} | ``2``
                BL spacing flag if *mdl* is not ``0``
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2017-04-26 ``@ddalle``: First version
        """
        self.set_key('mdsblf', mdsblf, j)
# class aflr3

