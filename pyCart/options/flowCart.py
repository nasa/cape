"""Interface for options specific to running `flowCart`"""


# Import options-specific utilities
from util import rc0, odict

# Class for flowCart settings
class flowCart(odict):
    """Dictionary-based interfaced for options specific to ``flowCart``"""
    
    # Run input sequence
    def get_InputSeq(self, i=None):
        """Return the input sequence for `flowCart`
        
        :Call:
            >>> InputSeq = opts.get_InputSeq(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *InputSeq*: :class:`int` or :class:`list`(:class:`int`)
                Sequence of input run index(es)
        :Versions:
            * 2014.10.02 ``@ddalle``: First version
        """
        return self.get_key('InputSeq', i)
        
    # Set run input sequence.
    def set_InputSeq(self, InputSeq=rc0('InputSeq'), i=None):
        """Set the input sequence for `flowCart`
        
        :Call:
            >>> opts.get_InputSeq(InputSeq, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *InputSeq*: :class:`int` or :class:`list`(:class:`int`)
                Sequence of input run index(es)
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.10.02 ``@ddalle``: First version
        """
        self.set_key('InputSeq', InputSeq, i)
        
    
    # Get minimum cumulative iteration count
    def get_IterSeq(self, i=None):
        """
        Get the break points for run *i*.  Input *i* will be repeated until the
        cumulative iteration count is greater than or equal to *IterSeq[i]*.
        
        :Call:
            >>> IterSeq = opts.get_IterSeq(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *IterSeq*: :class:`int` or :class:`list`(:class:`int`)
                Sequence of iteration break points
        :Versions:
            * 2014.10.02 ``@ddalle``: First version
        """
        return self.get_key('IterSeq', i)
        
    # Set minimum cumulative iteration count
    def set_IterSeq(self, IterSeq, i=None):
        """
        Get the break points for run *i*.  Input *i* will be repeated until the
        cumulative iteration count is greater than or equal to *IterSeq[i]*.
        
        :Call:
            >>> opts.get_IterSeq(IterSeq, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *IterSeq*: :class:`int` or :class:`list`(:class:`int`)
                Sequence of iteration break points
        :Versions:
            * 2014.10.02 ``@ddalle``: First version
        """
        self.set_key('IterSeq', IterSeq, i)
        
    
    # Number of iterations
    def get_nSeq(self):
        """Return the number of input sets in the sequence
        
        :Call:
            >>> nSeq = opts.get_nSeq()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *nSeq*: :class:`int`
                Number of input sets in the sequence
        :Versions:
            * 2014.10.02 ``@ddalle``: First version
        """
        # Get the input sequence.
        InputSeq = self.get_InputSeq()
        # Check if it's a list.
        if type(InputSeq).__name__ == "list":
            # Use the length.
            return len(InputSeq)
        else:
            # Something is messed up.
            return 1
            
    # Minimum required number of iterations
    def get_LastIter(self):
        """Return the minimum number of iterations for case to be done
        
        :Call:
            >>> nIter = opts.get_LastIter()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *nIter*: :class:`int`
                Number of required iterations for case
        :Versions:
            * 2014.10.02 ``@ddalle``: First version
        """
        return self.get_IterSeq(self.get_nSeq())
            
        
    # Get first-order status
    def get_first_order(self, i=None):
        """Return whether or not `flowCart` should be run first-order
        
        :Call:
            >>> fo = opts.get_first_order(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *fo*: :class:`int` or :class:`list`(:class:`int`)
                Switch for running `flowCart` in first-order mode
        :Versions:
            * 2014.10.02 ``@ddalle``: First version
        """
        return self.get_key('first_order', i)
        
    # Set first-order status
    def set_first_order(self, i=None):
        """Set whether or not `flowCart` should be run first-order
        
        :Call:
            >>> opts.set_first_order(fo)
            >>> opts.set_first_order(fo, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fo*: :class:`int` or :class:`list`(:class:`int`)
                Switch for running `flowCart` in first-order mode
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.10.02 ``@ddalle``: First version
        """
        self.set_key('first_order', fo, i)
            
        
    # Get first-order status
    def get_robust_mode(self, i=None):
        """Return whether or not `flowCart` should be run in robust mode
        
        :Call:
            >>> rm = opts.get_robust_mode(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *rm*: :class:`int` or :class:`list`(:class:`int`)
                Switch for running `flowCart` in robust mode
        :Versions:
            * 2014-11-21 ``@ddalle``: First version
        """
        return self.get_key('robust_mode', i)
    
    # Set robust-mode status
    def set_robust_mode(self, i=None):
        """Return whether or not `flowCart` should be run in robust mode
        
        :Call:
            >>> opts.get_robust_mode(rm, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *rm*: :class:`int` or :class:`list`(:class:`int`)
                Switch for running `flowCart` in robust mode
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-11-21 ``@ddalle``: First version
        """
        self.set_key('robust_mode', rm, i)
        
        
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
        
        
    # Get MPI status
    def get_mpi_fc(self, i=None):
        """Return whether or not to use `mpi_flowCart`
        
        :Call:
            >>> mpi_fc = opts.get_mpi_fc(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *mpi_fc*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to use `mpi_flowCart`
        :Versions:
            * 2014.10.02 ``@ddalle``: First version
        """
        return self.get_key('mpi_fc', i)
    
    # Set MPI status
    def set_mpi_fc(self, mpi_fc=rc0('mpi_fc'), i=None):
        """Set whether or not to use `mpi_flowCart`
        
        :Call:
            >>> opts.set_mpi_fc(mpi_fc, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *mpi_fc*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to use `mpi_flowCart`
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.10.02 ``@ddalle``: First version
        """
        self.set_key('mpi_fc', mpi_fc, i)
        
    
    # Get unsteady status
    def get_unsteady(self, i=None):
        """Return whether or not to use time-domain `td_flowCart`
        
        :Call:
            >>> td_fc = opts.get_unsteady(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *td_fc*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to use ``td_flowCart -unsteady``
        :Versions:
            * 2014-11-28 ``@ddalle``: First version
        """
        return self.get_key('unsteady', i)
    
    # Set unsteady status
    def set_unsteady(self, td_fc=rc0('unsteady'), i=None):
        """Set whether or not to use time-domain `td_flowCart`
        
        :Call:
            >>> opts.set_unsteady(td_fc, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *td_fc*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to use ``td_flowCart -unsteady``
        :Versions:
            * 2014-11-28 ``@ddalle``: First version
        """
        self.set_key('unsteady', td_fc, i)
        
        
    # Get aero.csh status
    def get_use_aero_csh(self, i=None):
        """Return whether or not to use `aero.csh`
        
        :Call:
            >>> ac = opts.get_use_aero_csh(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *ac*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to use `aero.csh`
        :Versions:
            * 2014-10-03 ``@ddalle``: First version
        """
        return self.get_key('use_aero_csh', i)
    
    # Set aero.csh status
    def set_use_aero_csh(self, ac=rc0('use_aero_csh'), i=None):
        """Set whether or not to use `aero.csh`
        
        :Call:
            >>> opts.set_use_aero_csh(ac, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *ac*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to use `aero.csh`
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-10-03 ``@ddalle``: First version
        """
        self.set_key('use_aero_csh', ac, i)
        
    
    # Get jumpstart status
    def get_jumpstart(self, i=None):
        """
        Return whether or not to "jump start", i.e. create meshes before running
        :file:`aero.csh`.
        
        :Call:
            >>> js = opts.get_jumpstart()
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *js*: :class:`bool`
                Whether or not to jumpstart
        :Versions:
            * 2014-12-04 ``@ddalle``: First version
        """
        return self.get_key('restart', i)
        
    # Set jumpstart status
    def set_jumpstart(self, js=rc0('jumpstart'), i=None):
        """
        Set whether or not to "jump start", i.e. create meshes before running
        :file:`aero.csh`.
        
        :Call:
            >>> opts.get_jumpstart(js)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *js*: :class:`bool`
                Whether or not to jumpstart
        :Versions:
            * 2014-12-04 ``@ddalle``: First version
        """
        self.set_key('jumpstart', js, i)
        
    
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
        
    
    # Get the time step
    def get_dt(self, i=None):
        """Return the time-accurate nondimensional physical time step
        
        :Call:
            >>> dt = opts.get_dt(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *dt*: :class:`float` or :class:`list`(:class:`float`)
                Nondimensional physical time step
        :Versions:
            * 2014-11-28 ``@ddalle``: First version
        """
        return self.get_key('dt', i)
        
    # Set the physical time step
    def set_dt(self, dt=rc0('dt'), i=None):
        """Set the time-accurate nondimensional physical time step
        
        :Call:
            >>> opts.set_dt(dt, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *dt*: :class:`float` or :class:`list`(:class:`float`)
                Nondimensional physical time step
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-11-28 ``@ddalle``: First version
        """
        self.set_key('dt', dt, i)
        
    
    # Get the number of unsteady steps
    def get_nSteps(self, i=None):
        """Return the number of unsteady time steps
        
        :Call:
            >>> nSteps = opts.get_nSteps(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *nSteps*: :class:`int` or :class:`list`(:class:`int`)
                Number of unsteady time steps to advance
        :Versions:
            * 2014-11-28 ``@ddalle``: First version
        """
        return self.get_key('nSteps', i)
        
    # Set the number of unsteady time steps
    def set_nSteps(self, nSteps=rc0('nSteps'), i=None):
        """Set the number of unsteady time steps
        
        :Call:
            >>> opts.set_nSteps(nSteps, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *nSteps*: :class:`int` or :class:`list`(:class:`int`)
                Number of unsteady time steps to advance
        :Versions:
            * 2014-11-28 ``@ddalle``: First version
        """
        self.set_key('nSteps', nSteps, i)
        
    
    # Get the unsteady checkpoint interval
    def get_checkptTD(self, i=None):
        """Return the number of steps between unsteady checkpoints
        
        :Call:
            >>> checkptTD = opts.get_checkptTD(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *checkptTD*: :class:`int` or :class:`list`(:class:`int`)
                Number of unsteady time steps between checkpoints
        :Versions:
            * 2014-11-28 ``@ddalle``: First version
        """
        return self.get_key('checkptTD', i)
        
    # Set the unsteady checkpoint interval
    def set_checkptTD(self, checkptTD=rc0('checkptTD'), i=None):
        """Set the number of steps between unsteady checkpoints
        
        :Call:
            >>> opts.set_checkptTD(checkptTD, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *checkptTD*: :class:`int` or :class:`list`(:class:`int`)
                Number of unsteady time steps between checkpoints
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-11-28 ``@ddalle``: First version
        """
        self.set_key('checkptTD', checkptTD, i)
        
    
    # Get the unsteady checkpoint interval
    def get_vizTD(self, i=None):
        """Return the number of steps between visualization outputs
        
        :Call:
            >>> vizTD = opts.get_vizTD(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *vizTD*: :class:`int` or :class:`list`(:class:`int`)
                Number of unsteady time steps between visualization outputs
        :Versions:
            * 2014-11-28 ``@ddalle``: First version
        """
        return self.get_key('vizTD', i)
        
    # Set the unsteady checkpoint interval
    def set_vizTD(self, vizTD=rc0('vizTD'), i=None):
        """Set the number of steps between unsteady checkpoints
        
        :Call:
            >>> opts.set_vizTD(vizTD, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *vizTD*: :class:`int` or :class:`list`(:class:`int`)
                Number of unsteady time steps between visualization outputs
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-11-28 ``@ddalle``: First version
        """
        self.set_key('vizTD', vizTD, i)
        
    
    # Get the time-accurate initial relaxation status
    def get_fc_clean(self, i=None):
        """
        Return whether or not to run an initial relaxation step before starting
        time-accurate solution
        
        :Call:
            >>> fc_clean = opts.get_fc_clean(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *fc_clean*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to run relaxation step
        :Versions:
            * 2014-12-01 ``@ddalle``: First version
        """
        return self.get_key('fc_clean', i)
        
    # Set the time-accurate initial relaxation status
    def set_fc_clean(self, fc_clean=rc0('fc_clean'), i=None):
        """
        Set whether or not to run an initial relaxation step before starting
        time-accurate solution
        
        :Call:
            >>> opts.set_fc_clean(fc_clean, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fc_clean*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to run relaxation step
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-12-01 ``@ddalle``: First version
        """
        self.set_key('fc_clean', fc_clean, i)
        
    
    # Get the number of iterations to use for iterative or time averaging
    def get_fc_stats(self, i=None):
        """Get number of iterations to use for iterative or time averaging
        
        :Call:
            >>> nstats = opts.get_fc_stats(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *nstats*: :class:`int` or :class:`list`(:class:`int`)
                Number of iterations to use for averaging (off if ``0``)
        :Versions:
            * 2014-12-01 ``@ddalle``: First version
        """
        return self.get_key('fc_stats', i)
        
    # Set the number of iterations to use for iterative or time averaging
    def set_fc_stats(self, nstats=rc0('fc_stats'), i=None):
        """Get number of iterations to use for iterative or time averaging
        
        :Call:
            >>> opts.set_fc_stats(nstats, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *nstats*: :class:`int` or :class:`list`(:class:`int`)
                Number of iterations to use for averaging (off if ``0``)
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-12-01 ``@ddalle``: First version
        """
        self.set_key('fc_stats', nstats, i)
        
        
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
        
        
    # Get the full multigrid status
    def get_fmg(self, i=None):
        """Return whether or not `flowCart` is set to run full multigrid
        
        :Call:
            >>> fmg = opts.get_fmg(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *fmg*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not `flowCart` is set to run full multigrid
        :Versions:
            * 2014-11-14 ``@ddalle``: First version
        """
        return self.get_key('fmg', i)
    
    # Set the full multigrid status
    def set_fmg(self, fmg=rc0('fmg'), i=None):
        """Set full multigrid status for `flowCart`
        
        :Call:
            >>> opts.set_fmg(fmg, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fmg*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not `flowCart` is set to run full multigrid
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-11-14 ``@ddalle``: First version
        """
        self.set_key('fmg', fmg, i)
        
        
    # Get the poly multigrid status
    def get_pmg(self, i=None):
        """Return whether or not `flowCart` is set to run poly multigrid
        
        :Call:
            >>> fmg = opts.get_pmg(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *pmg*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not `flowCart` is set to run poly multigrid
        :Versions:
            * 2014-11-14 ``@ddalle``: First version
        """
        return self.get_key('pmg', i)
    
    # Set the full multigrid status
    def set_pmg(self, pmg=rc0('pmg'), i=None):
        """Set poly multigrid status for `flowCart`
        
        :Call:
            >>> opts.set_pmg(pmg, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *pmg*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not `flowCart` is set to run poly multigrid
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-11-14 ``@ddalle``: First version
        """
        self.set_key('pmg', pmg, i)
        
        
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
        return self.get_key('binaryIO', i)
    
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
        
        
    # Get the y-spanwise status
    def get_tecO(self, i=None):
        """Return whether or not `flowCart` dumps Tecplot triangulations
        
        :Call:
            >>> tecO = opts.get_tecO(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *tecO*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not `flowCart` dumps Tecplot triangulations
        :Versions:
            * 2014.09.07 ``@ddalle``: First version
        """
        return self.get_key('tecO', i)
    
    # Set the y-spanwise status
    def set_tecO(self, tecO=rc0('tecO'), i=None):
        """Set whether `flowCart` dumps Tecplot triangulations
        
        :Call:
            >>> opts.set_tecO(tecO, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *tecO*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not `flowCart` dumps Tecplot triangulations
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.09.07 ``@ddalle``: First version
        """
        self.set_key('tecO', tecO, i)
        
        
    # Get the buffer limit setting
    def get_buffLim(self, i=None):
        """Return whether or not to use buffer limits
        
        :Call:
            >>> buffLim = opts.get_buffLim(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *buffLim*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to use `flowCart` buffer limits
        :Versions:
            * 2014-11-21 ``@ddalle``: First version
        """
        return self.get_key('tm', i)
    
    # Set the buffer limit setting
    def set_buffLim(self, buffLim=rc0('buffLim'), i=None):
        """Set `flowCart` buffer limit setting
        
        :Call:
            >>> opts.set_buffLim(buffLim, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *buffLim*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to use `flowCart` buffer limits
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-11-21 ``@ddalle``: First version
        """
        self.set_key('buffLim', buffLim, i)
        
        
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
    def get_nProc(self, i=None):
        """Return the number of threads used for `flowCart`
        
        :Call:
            >>> nProc = opts.get_nProc(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *nProc*: :class:`int` or :class:`list`(:class:`int`)
                Number of threads for `flowCart`
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
            * 2014.10.02 ``@ddalle``: Switched to "nProc"
        """
        return self.get_key('nProc', i)
    
    # Set number of threads for `flowCart`
    def set_nProc(self, nThreads=rc0('cflmin'), i=None):
        """Set minimum CFL number for `flowCart`
        
        :Call:
            >>> opts.set_nProc(nProc, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *nProc*: :class:`int` or :class:`list`(:class:`int`)
                Number of threads for `flowCart`
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
            * 2014.10.02 ``@ddalle``: Switched to "nProc"
        """
        self.set_key('nProc', nProc, i)
        
        
    # Get the command name for "mpirun" or "mpiexec"
    def get_mpicmd(self, i=None):
        """Return either ``'mpirun'`` or ``'mpiexec``
        
        :Call:
            >>> mpicmd = opts.get_mpicmd(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *mpicmd*: :class:`str`
                System command to call MPI
        :Versions:
            * 2014.10.02 ``@ddalle``: First version
        """
        return self.get_key('mpicmd', i)
    
    # Set the command name for "mpirun" or "mpiexec"
    def set_mpicmd(self, mpicmd=rc0('mpicmd'), i=None):
        """Set minimum CFL number for `flowCart`
        
        :Call:
            >>> opts.set_mpicmd(mpicmd, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *mpicmd*: :class:`str`
                System command to call MPI
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.10.02 ``@ddalle``: First version
        """
        self.set_key('mpicmd', mpicmd, i)
    
    # Get the submittable-job status
    def get_qsub(self, i=None):
        """Determine whether or not to submit jobs
        
        :Call:
            >>> qsub = opts.get_qsub(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *qsub*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to submit case to PBS
        :Versions:
            * 2014.10.05 ``@ddalle``: First version
        """
        return self.get_key('qsub', i)
    
    # Set the submittable-job status
    def set_qsub(self, qsub=rc0('qsub'), i=None):
        """Set jobs as submittable or nonsubmittable
        
        :Call:
            >>> opts.set_qsub(qsub, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *qsub*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to submit case to PBS
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.10.05 ``@ddalle``: First version
        """
        self.set_key('qsub', qsub, i)
        
    
    # Get the resubmittable-job status
    def get_resub(self, i=None):
        """Determine whether or not a job should restart or resubmit itself
        
        :Call:
            >>> resub = opts.get_resub(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *resub*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to resubmit/restart a case
        :Versions:
            * 2014.10.05 ``@ddalle``: First version
        """
        return self.get_key('resub', i)
    
    # Set the resubmittable-job status
    def set_resub(self, resub=rc0('resub'), i=None):
        """Set jobs as resubmittable or nonresubmittable
        
        :Call:
            >>> opts.set_resub(resub, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *resub*: :class:`bool` or :class:`list`(:class:`bool`)
                Whether or not to resubmit/restart a case
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.10.05 ``@ddalle``: First version
        """
        self.set_key('resub', resub, i)
        
        