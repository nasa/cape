"""Interface for options specific to running `flowCart`"""


# Import options-specific utilities
from util import rc0, odict

# Class for flowCart settings
class flowCart(odict):
    """Dictionary-based interfaced for options specific to ``flowCart``"""
    
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
        return self.get_key('it_fc', i)
        
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
        self.set_key('it_fc', it_fc, i)
        
        
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
        return self.get_key('mg_fc', i)
    
    # Set flowCart iteration levels
    def set_mg_fc(self, mg_fc=rc0('mg_fc'), i=None):
        """Set number of multigrid levels for `flowCart`
        
        :Call:
            >>> opts.set_mg_fc(mg_fc, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *mg_fc*: :class:`int` or :class:`list`(:class:`int`)
                Multigrid levels for run *i* or all runs if ``i==None``
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('mg_fc', mg_fc, i)
        
    
    # Get the CFL number
    def get_cfl(self, i=None):
        """Return the nominal CFL number for `flowCart`
        
        :Call:
            >>> cfl = opts.get_cfl(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *cfl*: :class:`float` or :class:`list`(:class:`float`)
                Nominal CFL number for run *i*
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('cfl', i)
    
    # Set CFL number
    def set_cfl(self, cfl=rc0('cfl'), i=None):
        """Set nominal CFL number `flowCart`
        
        :Call:
            >>> opts.set_mg_fc(mg_fc, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *cfl*: :class:`float` or :class:`list`(:class:`float`)
                Nominal CFL number for run *i*
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('cfl', cfl, i)
        
    
    # Get the minimum CFL number
    def get_cflmin(self, i=None):
        """Return the minimum CFL number for `flowCart`
        
        :Call:
            >>> cflmin = opts.get_cflmin(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *cfl*: :class:`float` or :class:`list`(:class:`float`)
                Minimum CFL number for run *i*
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('cflmin', i)
    
    # Set minimum CFL number
    def set_cflmin(self, cflmin=rc0('cflmin'), i=None):
        """Set minimum CFL number for `flowCart`
        
        :Call:
            >>> opts.set_cflmin(cflmin, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *cflmin*: :class:`float` or :class:`list`(:class:`float`)
                Minimum CFL number for run *i*
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('cflmin', cflmin, i)
        
        
    # Get the flowCart limiter
    def get_limiter(self, i=None):
        """Return the limiter `flowCart`
        
        :Call:
            >>> limiter = opts.get_limiter(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *limiter*: :class:`int` or :class:`list`(:class:`int`)
                Limiter ID for `flowCart`
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('limiter', i)
    
    # Set the flowCart limiter
    def set_limiter(self, limiter=rc0('limiter'), i=None):
        """Set limiter for `flowCart`
        
        :Call:
            >>> opts.set_limiter(limiter, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *limiter*: :class:`int` or :class:`list`(:class:`int`)
                Limiter ID for `flowCart`
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('limiter', limiter, i)
        
        
    # Get the y-spanwise status
    def get_y_is_spanwise(self, i=None):
        """Return whether or not *y* is the spanwise axis for `flowCart`
        
        :Call:
            >>> y_is_spanwise = opts.get_y_is_spanwise(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *y_is_spanwise*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not *y* is the spanwise index `flowCart`
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('y_is_spanwise', i)
    
    # Set the y-spanwise status
    def set_y_is_spanwise(self, y_is_spanwise=rc0('y_is_spanwise'), i=None):
        """Set limiter for `flowCart`
        
        :Call:
            >>> opts.set_y_is_spanwise(y_is_spanwise, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *y_is_spanwise*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not *y* is the spanwise index `flowCart`
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('y_is_spanwise', y_is_spanwise, i)
        
        
    # Get the y-spanwise status
    def get_binaryIO(self, i=None):
        """Return whether or not `flowCart` is set for binary I/O
        
        :Call:
            >>> binaryIO = opts.get_binaryIO(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *binaryIO*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not `flowCart` is for binary I/O
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('y_is_spanwise', i)
    
    # Set the y-spanwise status
    def set_binaryIO(self, binaryIO=rc0('binaryIO'), i=None):
        """Set limiter for `flowCart`
        
        :Call:
            >>> opts.set_binaryIO(binaryIO, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *binaryIO*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not `flowCart` is set for binary I/O
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('binaryIO', binaryIO, i)
        
        
    # Get the cut cell gradient status
    def get_tm(self, i=None):
        """Return whether or not `flowCart` is set for cut cell gradient
        
        :Call:
            >>> tm = opts.get_tm(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *tm*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not `flowCart` is set for cut cell gradient
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('tm', i)
    
    # Set the y-spanwise status
    def set_tm(self, tm=rc0('tm'), i=None):
        """Set cut cell gradient status for `flowCart`
        
        :Call:
            >>> opts.set_tm(tm, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *tm*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not `flowCart` is set for cut cell gradient
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('tm', tm, i)
        
    
    # Get the number of threads for `flowCart`
    def get_OMP_NUM_THREADS(self, i=None):
        """Return the number of threads used for `flowCart`
        
        :Call:
            >>> cflmin = opts.get_OMP_NUM_THREADS(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *nThreads*: :class:`int` or :class:`list`(:class:`int`)
                Number of threads for `flowCart`
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('OMP_NUM_THREADS', i)
    
    # Set number of threads for `flowCart`
    def set_OMP_NUM_THREADS(self, nThreads=rc0('cflmin'), i=None):
        """Set minimum CFL number for `flowCart`
        
        :Call:
            >>> opts.set_OMP_NUM_THREADS(nThreads, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *nThreads*: :class:`int` or :class:`list`(:class:`int`)
                Number of threads for `flowCart`
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('OMP_NUM_THREADS', OMP_NUM_THREADS, i)
        
        
    
        
        