r"""
:mod:`cape.pyfun.options.runctlopts`: FUN3D un control options
=================================================================

Options interface for aspects of running a case of FUN3D.  The settings
are read from the ``"RunControl"`` of a JSON file, and the contents of
this section are written to ``case.json`` within each run folder.

The FUN3D-specific options include adaptation settings and command-line
arguments for both ``nodet`` and ``dual``.

:See Also:
    * :mod:`cape.cfdx.options.runctlopts`
    * :mod:`cape.cfdx.options.archiveopts`
    * :mod:`cape.pyfun.options.archiveopts`
"""

# Local imports
from .archiveopts import ArchiveOpts
from .util import rc0, odict
from ...cfdx.options import runctlopts
from ...optdict import BOOL_TYPES, INT_TYPES, OptionsDict


# Class for `nodet` inputs
class nodet(odict):
    r"""Class for ``nodet`` command-line inputs"""
    
    # Animation frequency
    def get_nodet_animation_freq(self, i=None):
        """Get animation frequency command-line option
        
        :Call:
            >>> f = opts.get_nodet_animation_freq(i=None)
        :Inputs:
            *opts*: :class:`cape.pyfun.options.Options`
                Options interface
            *i*: :class:`int`
                Run index
        :Outputs:
            *f*: :class:`int`
                Animation frequency; when ``nodet`` outputs are written
        :Versions:
            * 2015-11-24 ``@ddalle``: First version
        """
        return self.get_key('animation_freq', i, rck='nodet_animation_freq')
        
    # Set animation frequency
    def set_nodet_animation_freq(self, f=rc0('nodet_animation_freq'), i=None):
        """Set animation frequency command-line option
        
        :Call:
            >>> opts.set_nodet_animation_freq(f, i=None)
        :Inputs:
            *opts*: :class:`cape.pyfun.options.Options`
                Options interface
            *f*: :class:`int`
                Animation frequency; when ``nodet`` outputs are written
            *i*: :class:`int`
                Run index
        :Versions:
            * 2015-11-24 ``@ddalle``: First version
        """
        self.set_key('animation_freq', f, i)
# class nodet


# Class for ``dual`` inputs
class dual(odict):
    r"""Class for ``dual`` command-line inputs"""
    
    # Helpful convergence flag
    def get_dual_outer_loop_krylov(self, j=None):
        """Get ``--outer_loop_krylov`` setting for ``dual``
        
        :Call:
            >>> f = opts.get_dual_outer_loop_krylov(j=None)
        :Inputs:
            *opts*: :class:`cape.pyfun.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *f*: {``True``} | ``False``
                Whether or not to use this flag
        :Versions:
            * 2016-04-28 ``@ddalle``: First version
        """
        return self.get_key('outer_loop_krylov', j,
            rck='dual_outer_loop_krylov')
    
    # Helpful convergence flag
    def set_dual_outer_loop_krylov(self,f=rc0('dual_outer_loop_krylov'),j=None):
        """Set ``--outer_loop_krylov`` setting for ``dual``
        
        :Call:
            >>> opts.set_dual_outer_loop_krylov(f=True, j=None)
        :Inputs:
            *opts*: :class:`cape.pyfun.options.Options`
                Options interface
            *f*: {``True``} | ``False``
                Whether or not to use this flag
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-04-28 ``@ddalle``: First version
        """
        self.set_key('outer_loop_krylov', f, j)
        
    # Residual adjoint dot-product
    def get_dual_rad(self, j=None):
        """Get command-line setting for residual adjoint dot product
        
        :Call:
            >>> rad = opts.get_dual_rad(j=None)
        :Inputs:
            *opts*: :class:`cape.pyfun.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *rad*: {``True``} | ``False``
                Whether or not to use residual adjoint dot product
        :Versions:
            * 2016-04-28 ``@ddalle``: First version
        """
        return self.get_key('rad', j, rck='dual_rad')
        
    # Residual adjoint dot-product
    def set_dual_rad(self, rad=rc0('dual_rad'), j=None):
        """Set command-line setting for residual adjoint dot product
        
        :Call:
            >>> opts.set_dual_rad(rad=True, j=None)
        :Inputs:
            *opts*: :class:`cape.pyfun.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *rad*: {``True``} | ``False``
                Whether or not to use residual adjoint dot product
        :Versions:
            * 2016-04-28 ``@ddalle``: First version
        """
        self.set_key('rad', rad, j)
        
    # Adapt setting
    def get_dual_adapt(self, j=None):
        """Get command-line setting for adapting when running ``dual``
        
        :Call:
            >>> adapt = opts.get_dual_adapt(j=None)
        :Inputs:
            *opts*: :class:`cape.pyfun.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *adapt*: {``True``} | ``False``
                Whether or not to adapt after running ``dual``
        :Versions:
            * 2016-04-28 ``@ddalle``: First version
        """
        return self.get_key('adapt', j, rck=rc0('dual_adapt'))
        
    # Adapt setting
    def set_dual_adapt(self, adapt=rc0('dual_adapt'), j=None):
        """Set command-line setting for adapting when running ``dual``
        
        :Call:
            >>> opts.set_dual_adapt(adapt=True, j=None)
        :Inputs:
            *opts*: :class:`cape.pyfun.options.Options`
                Options interface
            *adapt*: {``True``} | ``False``
                Whether or not to adapt after running ``dual``
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-04-28 ``@ddalle``: First version
        """
        self.set_key('adapt', adapt, j)
# class dual


# Class for Report settings
class RunControlOpts(runctlopts.RunControlOpts):
    r"""FUN3D-specific "RunControl" options interface
    
    :Call:
        >>> opts = RunControl(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of "RunControl" settings
    :Outputs:
        *opts*: :class:`Options`
            Options interface
    :Versions:
        * 2015-09-28 ``@ddalle``: Version 1.0
        * 2022-10-24 ``@ddalle``: Version 2.0
    """
   # =====================
   # Class attributes
   # =====================
   # <
    # Names of allowed settings
    _optlist = {
        "AdaptPhase",
        "Adaptive",
        "Dual",
        "DualPhase",
        "KeepRestarts",
        "dual",
        "nIterAdjoint",
        "nodet",
    }

    # Option types
    _opttypes = {
        "AdaptPhase": BOOL_TYPES,
        "Adaptive": BOOL_TYPES,
        "Dual": BOOL_TYPES,
        "DualPhase": BOOL_TYPES,
        "KeepRestarts": BOOL_TYPES,
        "nIterAdjoint": INT_TYPES,
    }

    # Default values
    _rc = {
        "AdaptPhase": True,
        "Adaptive": False,
        "Dual": False,
        "DualPhase": True,
        "KeepRestarts": False,
        "nIterAdjoint": 200,
    }

    # Descriptions
    _rst_descriptions = {
        "AdaptPhase": "whether or not to adapt mesh at end of phase",
        "Adaptive": "whether or not to run adaptively",
        "Dual": "whether or not to run all adaptations with adjoint",
        "DualPhase": "whether or not to run phase in dual mode",
        "KeepRestarts": "whether or not to keep restart files",
        "nIterAdjoint": "number of iterations for adjoint solver",
    }

    # Additional sections
    _sec_cls = {
        "Archive": ArchiveOpts,
        "dual": dual,
        "nodet": nodet,
    }

    # Disallow other attributes
    __slots__ = tuple()
   # >
   
   # ==============
   # Local settings
   # ==============
   # <
    # Get current adaptation number
    def get_AdaptationNumber(self, j=None):
        r"""Get the adaptation number for a given phase
        
        :Call:
            >>> nadapt = opts.get_AdaptationNumber(j=None)
        :Inputs:
            *opts*: :class:`cape.pyfun.options.Options`
                Options interface
            *j*: {``None``} | :class:`int`
                Phase number
        :Outputs:
            *nadapt*: :class:`int` | ``None``
                Number of adaptations prior to phase *j*
        :Versions:
            * 2015-12-31 ``@ddalle``: Version 1.0
            * 2022-10-24 ``@ddalle``: Version 1.1; use *PhaseSequence*
        """
        # Check for adaptive case
        if not (self.get_Adaptive() or self.get_Dual()):
            return None
        elif j is None:
            # No phases
            return None
        # Initialize adaptation number
        nadapt = 0
        # Loop through prior phases
        for k in self.get_PhaseSequence():
            # Exit if we've passed phase *j*
            if k >= j:
                break
            # Check if it's an adaptation phase
            if self.get_nIter(k) > 0 and self.get_AdaptPhase(k):
                nadapt += 1
        # Output
        return nadapt
   # >


# Create properties
RunControlOpts.add_properties(RunControlOpts._rst_descriptions)
# Upgrade subsections
RunControlOpts.promote_sections()
