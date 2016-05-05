"""
Interface to FUN3D run control options
======================================

This module provides a class to mirror the Fortran namelist capability.  For
now, nonunique section names are not allowed.
"""

# Import options-specific utilities
from .util import rc0, getel, odict

# Import template module
import cape.options.runControl

# Class for `nodet` inputs
class nodet(odict):
    """Class for ``nodet`` command-line inputs"""
    
    # Animation frequency
    def get_nodet_animation_freq(self, i=None):
        """Get animation frequency command-line option
        
        :Call:
            >>> f = opts.get_nodet_animation_freq(i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
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
            *opts*: :class:`pyFun.options.Options`
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
    """Class for ``dual`` command-line inputs"""
    
    # Helpful convergence flag
    def get_dual_outer_loop_krylov(self, j=None):
        """Get ``--outer_loop_krylov`` setting for ``dual``
        
        :Call:
            >>> f = opts.get_dual_outer_loop_krylov(j=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
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
            *opts*: :class:`pyFun.options.Options`
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
            *opts*: :class:`pyFun.options.Options`
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
            *opts*: :class:`pyFun.options.Options`
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
            *opts*: :class:`pyFun.options.Options`
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
            *opts*: :class:`pyFun.options.Options`
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
class RunControl(cape.options.runControl.RunControl):
    """Dictionary-based interface for automated reports
    
    :Call:
        >>> opts = RunControl(**kw)
    :Versions:
        * 2015-09-28 ``@ddalle``: Subclassed to CAPE
    """
    
    # Initialization method
    def __init__(self, fname=None, **kw):
        # Store the data in *this* instance
        for k in kw:
            self[k] = kw[k]
        # Upgrade important groups to their own classes.
        self._Environ()
        self._ulimit()
        self._aflr3()
        self._intersect()
        self._verify()
        self._Archive()
        self._nodet()
        self._dual()
        
    # ============ 
    # Initializers
    # ============
   # <
   
    # Initialization and confirmation for ``nodet`` options
    def _nodet(self):
        """Initialize ``nodet`` options if necessary"""
        if 'nodet' not in self:
            # Empty/default
            self['nodet'] = nodet()
        elif type(self['nodet']).__name__ == 'dict':
            # Convert to special class
            self['nodet'] = nodet(**self['nodet'])
   
    # Initialization and confirmation for ``dual`` options
    def _dual(self):
        """Initialize ``dual`` options if necessary"""
        if 'dual' not in self:
            # Empty/default
            self['dual'] = dual()
        elif type(self['dual']).__name__ == 'dict':
            # Convert to special class
            self['dual'] = dual(**self['dual'])
            
   # >
   
    # ============== 
    # Local settings
    # ==============
   # <
   
    # Keep Restart files?
    def get_KeepRestarts(self, i=None):
        """Return whether or not to keep restart files
        
        :Call:
            >>> qr = opts.get_KeepRestarts(i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *qr*: :class:`bool`
                Whether or not to copy flow solution files
        :Versions:
            * 2015-12-31 ``@ddalle``: First version
        """
        return self.get_key('KeepRestarts', i)
        
    # Force to keep restart files
    def set_KeepRestarts(self, qr=rc0("KeepRestarts"), i=None):
        """Set whether or not to keep restart files
        
        :Call:
            >>> opts.get_KeepRestarts(qr, i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *qr*: :class:`bool`
                Whether or not to copy flow solution files
            *i*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2015-12-31 ``@ddalle``: First version
        """
        self.set_key('KeepRestarts', qr, i)
        
    # Get adaptive status
    def get_Adaptive(self, i=None):
        """Return whether or not to run adaptively
        
        :Call:
            >>> ac = opts.get_Adaptive(i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *ac*: :class:`bool` | :class:`list` (:class:`bool`)
                Whether or not to use `aero.csh`
        :Versions:
            * 2015-12-30 ``@ddalle``: First version
        """
        return self.get_key('Adaptive', i)
        
    # Set adaptive status
    def set_Adaptive(self, ac=rc0('Adaptive'), i=None):
        """Return whether or not to run adaptively
        
        :Call:
            >>> opts.set_Adaptive(ac, i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *ac*: :class:`bool` | :class:`list` (:class:`bool`)
                Whether or not to use `aero.csh`
            *i*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2015-12-30 ``@ddalle``: First version
        """
        self.set_key('Adaptive', ac, i)
        
    # Get Dual status
    def get_Dual(self, i=None):
        """Return whether or not to run in dual-mode with an adjoint
        
        This applies to the whole case, not to individual phases
        
        :Call:
            >>> d = opts.get_Dual(i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *d*: :class:`bool` | :class:`list` (:class:`bool`)
                Whether or not to run the case with dual mode
        :Versions:
            * 2015-12-30 ``@ddalle``: First version
        """
        return self.get_key('Dual', i)
        
    # Set Dual status
    def set_Dual(self, d=rc0('Dual'), i=None):
        """Set whether or not to run in dual-mode with an adjoint
        
        This applies to the whole case, not to individual phases
        
        :Call:
            >>> opts.get_Dual(d=False, i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *d*: :class:`bool` | :class:`list` (:class:`bool`)
                Whether or not to run the case with dual mode
            *i*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2015-12-30 ``@ddalle``: First version
        """
        self.set_key('Dual', d, i)
        
    # Get status of phase adaptive
    def get_AdaptPhase(self, i=None):
        """Determine whether or not a phase is adaptive
        
        :Call:
            >>> qa = opts.get_AdaptPhase(i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *qa*: :class:`bool` | :class:`list` (:class:`bool`)
                Whether or not phase ends with an adaptation
        :Versions:
            * 2015-12-30 ``@ddalle``: First version
        """
        return self.get_key('AdaptPhase', i)
        
    # Set status of phase adaptive
    def set_AdaptPhase(self, qa=rc0('AdaptPhase'), i=None):
        """Set whether or not a phase is adaptive
        
        :Call:
            >>> opts.set_AdaptPhase(qa, i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *qa*: :class:`bool` | :class:`list` (:class:`bool`)
                Whether or not phase ends with an adaptation
            *i*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2015-12-30 ``@ddalle``: First version
        """
        self.set_key('AdaptPhase', qa, i)
        
    # Get status of dual phase
    def get_DualPhase(self, i=None):
        """Determine whether or not a phase is run with an adjoint
        
        :Call:
            >>> qd = opts.get_DualPhase(i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *i*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *qd*: :class:`bool` | :class:`list` (:class:`bool`)
                Whether or not phase ends with an adjoint computation
        :Versions:
            * 2015-12-30 ``@ddalle``: First version
        """
        return self.get_key('DualPhase', i)
        
    # Set status of dual phase
    def set_DualPhase(self, qa=rc0('DualPhase'), i=None):
        """Set whether or not a phase is run with an adjoint
        
        :Call:
            >>> opts.set_DualPhase(qd, i=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *qd*: :class:`bool` | :class:`list` (:class:`bool`)
                Whether or not phase ends with an adjoint computation
            *i*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2015-12-30 ``@ddalle``: First version
        """
        self.set_key('DualPhase', qa, i)
        
    # Get number of iterations for adjoint solver
    def get_nIterAdjoint(self, j=None):
        """Get number of iterations for ``dual``, the adjoint solver
        
        :Call:
            >>> n = opts.get_nIterAdjoint(j=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *j*: :class:`int` | ``None``
                Phase number
        :Outputs:
            *n*: :class:`int`
                Number of iterations for ``dual``
        :Versions:
            * 2016-04-28 ``@ddalle``: First version
        """
        return self.get_key('nIterAdjoint', j)
        
    # Set number of iterations for adjoint solver
    def set_nIterAdjoint(self, n=rc0('nIterAdjoint'), j=None):
        """Set number of iterations for ``dual``, the adjoint solver
        
        :Call:
            >>> n = opts.get_nIterAdjoint(n=200, j=None)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *n*: :class:`int`
                Number of iterations for ``dual``
            *j*: :class:`int` | ``None``
                Phase number
        :Versions:
            * 2016-04-28 ``@ddalle``: First version
        """
        self.set_key('nIterAdjoint', n, j)
        
        
    # Get current adaptation number
    def get_AdaptationNumber(self, i):
        """Get the adaptation number for a given phase
        
        :Call:
            >>> j = opts.get_AdaptationNumber(i)
        :Inputs:
            *opts*: :class:`pyFun.options.Options`
                Options interface
            *i*: :class:`int`
                Phase number
        :Outputs:
            *j*: :class:`int` | ``None``
                Number of adaptation prior to phase *i*
        :Versions:
            * 2015-12-31 ``@ddalle``: First version
        """
        # Check for adaptive case
        if not (self.get_Adaptive() or self.get_Dual()):
            return None
        # Initialize adaptation number
        j = 0
        # Loop through prior phases
        for k in range(i):
            # Check if it's an adaptation phase
            if self.get_nIter(k)>0 and self.get_AdaptPhase(k):
                j += 1
        # Output
        return j
        
   # >
    
# class RunControl


