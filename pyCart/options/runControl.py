"""Interface for options specific to running `flowCart`"""


# Import options-specific utilities
from util import rc0, odict, isArray
# Run control class
import cape.options.runControl

# Function to test if something is an acceptable Runge-Kutta object
def isRK(RK):
    """
    Determine if a variable has a value that can be interpreted as a single set
    of Runge-Kutta stage inputs
    
    :Call:
        >>> q = isRK(RK)
    :Inputs:
        *RK*: any
            Any variable
    :Outputs:
        *q*: :class:`bool`
            ``True`` unless *RK* is a list with depth not equal to 2
    :Versions:
        * 2014-12-17 ``@ddalle``: First version
    """
    # Check if the input is an array.
    if not isArray(RK): return True
    # Check the the first element.
    if (len(RK)<1) or (not isArray(RK[0])): return False
    # The first element is now a list
    RK0 = RK[0]
    # Check the first element of RK0
    if (len(RK0)==2) and (not isArray(RK0[0])):
        # This looks like an Nx2 input.
        return True
    else:
        # The above case is the only type of acceptable list.
        return False
# def isRK

        
# Class for flowCart inputs
class flowCart(odict):
    """Class for flowCart settings"""
    
    # Get the Runge-Kutta scheme
    def get_RKScheme(self, i=None):
        """Return the Runge-Kutta scheme for a run
        
        :Call:
            >>> RK = opts.get_RKScheme(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *RK*: :class:`str` or :class:`list` ([:class:`float`,:class:`int`])
                Named Runge-Kutta scheme or list of coefficients and gradient
                evaluation flags
        :See also:
            * :func:`pyCart.inputCntl.InputCntl.SetRungeKutta`
        :Versions:
            * 2014-12-17 ``@ddalle``: First version
        """
        # Get the value.
        RK = self.get('RKScheme', rc0('RKScheme'))
        # Check the type.
        if i is None:
            # Just output
            return RK
        if isRK(RK):
            # No list.
            return RK
        else:
            # Check for empty input.
            if len(RK) == 0:
                return None
            # Array-like
            if i:
                # Check the length.
                if i >= len(RK):
                    # Take the last element.
                    return RK[-1]
                else:
                    # Take the *i*th element.
                    return RK[i]
            else:
                # Use the first element.
                return RK[0]
                
    # Set the Runge-Kutta scheme for a certain run
    def set_RKScheme(self, RK=rc0('RKScheme'), i=None):
        """Set the Runge-Kutta scheme for a run
        
        :Call:
            >>> opts.set_RKScheme(RK, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *RK*: :class:`str` or :class:`list` ([:class:`float`,:class:`int`])
                Named Runge-Kutta scheme or list of coefficients and gradient
                evaluation flags
            *i*: :class:`int` or ``None``
                Run index
        :See also:
            * :func:`pyCart.inputCntl.InputCntl.SetRungeKutta`
        :Versions:
            * 2014-12-17 ``@ddalle``: First version
        """
        # Get the current value.
        x = self.get('RKScheme', rc0('RKScheme'))
        # Check the index input.
        if i is None:
            # Scalar output
            self['RKScheme'] = RK
        else:
            # Ensure list.
            if isRK(x):
                # Make a list.
                y = [x]
            else:
                # Already a list.
                y = list(x)
            # Make sure *y* is long enough.
            for j in range(len(y), i):
                y.append(y[-1])
            # Check if we are setting an element or appending it.
            if i >= len(y):
                # Append
                y.append(RK)
            else:
                # Set the value.
                y[i] = RK
            # Output
            self['RKScheme'] = y
            
        
        
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
            * 2014-10-02 ``@ddalle``: First version
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
            * 2014-10-02 ``@ddalle``: First version
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
            * 2014-08-01 ``@ddalle``: First version
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
            * 2014-08-01 ``@ddalle``: First version
        """
        self.set_key('it_fc', it_fc, i)
        
    
    # Number of iterations between averaging operation
    def get_it_avg(self, i=None):
        """
        Return the number of iterations between writing ``triq`` file for
        cumulative averaging.  If ``0``, do not perform averaging.
        
        Not available during ``aero.csh`` runs.
        
        :Call:
            >>> it_avg = opts.get_it_avg(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *it_avg*: :class:`int` or :class:`list`(:class:`int`)
                Stopping interval between averaging for run *i*
        :Versions:
            * 2015-09-14 ``@ddalle``: First version
        """
        return self.get_key('it_avg', i)
        
    # Set flowCart iteration averaging interval count
    def set_it_avg(self, it_avg=rc0('it_avg'), i=None):
        """
        Set the number of iterations between writing ``triq`` file for
        cumulative averaging.  If ``0``, do not perform averaging.
        
        Not available during ``aero.csh`` runs.
        
        :Call:
            >>> opts.set_it_avg(it_avg)
            >>> opts.set_it_avg(it_avg, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *it_avg*: :class:`int` or :class:`list`(:class:`int`)
                Stopping interval between averaging for run *i*
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2015-09-14 ``@ddalle``: First version
        """
        self.get_key('it_avg', i)
        
    
    # Number orders of convergence to terminate early at
    def get_nOrders(self, i=None):
        """Get the number of orders of convergence for early termination
        
        :Call:
            >>> nOrders = opts.get_nOrders(i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *nOrders*: :class:`int`
                Number of orders for convergence
        :Versions:
            * 2014-12-12 ``@ddalle``: First version
        """
        return self.get_key('nOrders', i)
        
    # Set number orders of convergence to terminate early at
    def set_nOrders(self, nOrders=rc0('nOrders'), i=None):
        """Set the number of orders of convergence for early termination
        
        :Call:
            >>> opts.set_nOrders(nOrders, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *nOrders*: :class:`int`
                Number of orders for convergence
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-12-12 ``@ddalle``: First version
        """
        self.set_key('nOrders', nOrders, i)
        
        
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
            *dt*: :class:`float` | :class:`list` (:class:`float`)
                Nondimensional physical time step
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-11-28 ``@ddalle``: First version
        """
        self.set_key('dt', dt, i)
        
    
    # Get the number of subiterations for time-accurate inputs
    def get_it_sub(self, i=None):
        """Return the number of subiterations to perform at each time step
        
        :Call:
            >>> it_sub = opts.get_it_sub(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *it_sub*: :class:`int` | :class:`list` (:class:`int`)
                Number of subiterations
        :Versions:
            * 2014-11-28 ``@ddalle``: First version
            * 2015-11-09 ``@ddalle``: ``nSteps`` --> ``it_sub``
        """
        return self.get_key('it_sub', i)
        
    # Set the number of subiterations for time-accurate inputs
    def set_it_sub(self, it_sub=rc0('it_sub'), i=None):
        """Set the number of subiterations to perform at each time step
        
        :Call:
            >>> opts.set_it_sub(it_sub, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
            *it_sub*: :class:`int` | :class:`list` (:class:`int`)
                Number of subiterations
        :Versions:
            * 2014-11-28 ``@ddalle``: First version
            * 2015-11-09 ``@ddalle``: ``nSteps`` --> ``it_sub``
        """
        self.set_key('it_sub', it_sub, i)
        
    
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
        return self.get_key('buffLim', i)
    
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
        
    
# class flowCart

# Class for flowCart settings
class adjointCart(odict):
    """Dictionary-based interfaced for options specific to ``adjointCart``"""
    
    
    # Number of iterations for adjointCart
    def get_it_ad(self, i=None):
        """Return the number of iterations for `adjointCart`
        
        :Call:
            >>> it_fc = opts.get_it_fc(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *it_ad*: :class:`int` or :class:`list`(:class:`int`)
                Number of iterations for run *i* or all runs if ``i==None``
        :Versions:
            * 2014.08.01 ``@ddalle``: First version
        """
        return self.get_key('it_ad', i)
        
    # Set adjointCart iteration count
    def set_it_ad(self, it_ad=rc0('it_ad'), i=None):
        """Set the number of iterations for `adjointCart`
        
        :Call:
            >>> opts.set_it_ad(it_ad)
            >>> opts.set_it_ad(it_ad, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *it_ad*: :class:`int` or :class:`list`(:class:`int`)
                Number of iterations for run *i* or all runs if ``i==None``
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.01 ``@ddalle``: First version
        """
        self.set_key('it_ad', it_ad, i)
        
        
    # Get adjointCart multigrd levels
    def get_mg_ad(self, i=None):
        """Return the number of multigrid levels for `adjointCart`
        
        :Call:
            >>> mg_fc = opts.get_mg_ad(i=None)
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
        return self.get_key('mg_ad', i)
    
    # Set adjointCart iteration levels
    def set_mg_ad(self, mg_ad=rc0('mg_ad'), i=None):
        """Set number of multigrid levels for `adjointCart`
        
        :Call:
            >>> opts.set_mg_ad(mg_ad, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *mg_ad*: :class:`int` or :class:`list`(:class:`int`)
                Multigrid levels for run *i* or all runs if ``i==None``
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('mg_ad', mg_ad, i)
        
        
    # Adjoint first order
    def get_adj_first_order(self, i=None):
        """Get whether or not to run adjoins in first-order mode
        
        :Call:
            >>> adj = opts.set_adj_first_order(i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *adj*: :class:`bool` or :class:`list`(:class:`int`)
                Whether or not to always run `adjointCart` first-order
        :Versions:
            * 2014-11-17 ``@ddalle``: First version
        """
        return self.get_key('adj_first_order', i)
        
    # Adjoint first order
    def set_adj_first_order(self, adj=rc0('adj_first_order'), i=None):
        """Set whether or not to run adjoins in first-order mode
        
        :Call:
            >>> opts.set_adj_first_order(adj, i)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *adj*: :class:`bool` or :class:`list`(:class:`int`)
                Whether or not to always run `adjointCart` first-order
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-11-17 ``@ddalle``: First version
        """
        self.set_key('adj_first_order', adj, i)
# class adjointCart

# Class for flowCart settings
class Adaptation(odict):
    """Dictionary-based interfaced for options for Cart3D adaptation"""
    
    
    # Get number of adaptation cycles
    def get_n_adapt_cycles(self, i=None):
        """Return the number of Cart3D number of adaptation cycles
        
        :Call:
            >>> nAdapt = opts.get_n_adapt_cycles(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *nAdapt*: :class:`int` or :class:`list`(:class:`int`)
                Number of adaptation cycles
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('n_adapt_cycles', i)
        
    # Set adjointCart iteration count
    def set_n_adapt_cycles(self, nAdapt=rc0('n_adapt_cycles'), i=None):
        """Set the number of Cart3D adaptation cycles
        
        :Call:
            >>> opts.set_n_adaptation_cycles(nAdapt, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *nAdapt*: :class:`int` or :class:`list`(:class:`int`)
                Number of iterations for run *i* or all runs if ``i==None``
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('n_adapt_cycles', nAdapt, i)
    
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
        return self.get_key('jumpstart', i)
        
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
        
        
    # Get the adaptation tolerance
    def get_etol(self, i=None):
        """Return the target output error tolerance
        
        :Call:
            >>> etol = opts.get_etol(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *etol*: :class:`float` or :class:`list`(:class:`float`)
                Output error tolerance
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('etol', i)
        
    # Set the adaptation tolerance
    def set_etol(self, etol=rc0('etol'), i=None):
        """Set the output error tolerance
        
        :Call:
            >>> opts.set_etol(etol, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *etol*: :class:`float` or :class:`list`(:class:`float`)
                Output error tolerance
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('etol', etol, i)
        
        
    # Get the maximum cell count
    def get_max_nCells(self, i=None):
        """Return the maximum cell count
        
        :Call:
            >>> max_nCells = opts.get_max_nCells(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *etol*: :class:`float` or :class:`list`(:class:`float`)
                Output error tolerance
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('etol', i)
    
    # Set the maximum cell count
    def set_max_nCells(self, max_nCells=rc0('max_nCells'), i=None):
        """Return the maximum cell count
        
        :Call:
            >>> max_nCells = opts.get_max_nCells(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *etol*: :class:`float` or :class:`list`(:class:`float`)
                Output error tolerance
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('max_nCells', max_nCells)
        
        
    # Get the number of flowCart iterations for refined meshes
    def get_ws_it(self, i=None):
        """Get number of `flowCart` iterations on refined mesh *i*
        
        :Call:
            >>> ws_it = opts.get_ws_it(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *ws_it*: :class:`int` or :class:`list`(:class:`int`)
                Number of `flowCart` iterations
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('ws_it', i)
    
    # Set the number of flowcart iterations fore refined meshes
    def set_ws_it(self, ws_it=rc0('ws_it'), i=None):
        """Set number of `flowCart` iterations on refined mesh *i*
        
        :Call:
            >>> opts.set_ws_it(ws_it, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *ws_it*: :class:`int` or :class:`list`(:class:`int`)
                Number of `flowCart` iterations
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('ws_it', ws_it, i)
        
    
    # Get the mesh growth ratio for refinement i
    def get_mesh_growth(self, i=None):
        """Get the refinement cell count ratio
        
        :Call:
            >>> mesh_growth = opts.get_mesh_growth(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *mesh_growth*: :class:`float` or :class:`list`(:class:`float`)
                Refinement mesh growth ratio
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('mesh_growth', i)
    
    # Set the number of flowcart iterations fore refined meshes
    def set_mesh_growth(self, mesh_growth=rc0('mesh_growth'), i=None):
        """Set the refinement cell count ratio
        
        :Call:
            >>> opts.set_mesh_growth(mesh_growth, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *mesh_growth*: :class:`float` or :class:`list`(:class:`float`)
                Refinement mesh growth ratio
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('mesh_growth', mesh_growth, i)
    
    
    # Get the adaptation type
    def get_apc(self, i=None):
        """Get the adaptation type
        
        :Call:
            >>> apc = opts.get_apc(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *apc*: :class:`str` or :class:`list`(:class:`str`)
                Adaptation cycle type, ``"a"`` or ``"p"``
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        return self.get_key('apc', i)
    
    # Set the adaptation type
    def set_apc(self, apc=rc0('apc'), i=None):
        """Set the adaptation type
        
        :Call:
            >>> opts.set_apc(apc, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *apc*: :class:`str` or :class:`list`(:class:`str`)
                Adaptation cycle type, ``"a"`` or ``"p"``
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014.08.02 ``@ddalle``: First version
        """
        self.set_key('apc', apc, i)


    # Get the number of buffer layers
    def get_abuff(self, i=None):
        """Get the number of buffer layers
        
        :Call:
            >>> buf = opts.get_abuff(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *buf*: :class:`int` or :class:`list`(:class:`int`)
                Number of buffer layers
        :Versions:
            * 2014-11-14 ``@ddalle``: First version
        """
        return self.get_key('buf', i)
        
    # Set the number of buffer layers.
    def set_abuff(self, buf=rc0('buf'), i=None):
        """Set the number of buffer layers
        
        :Call:
            >>> opts.set_abuff(buf, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *buf*: :class:`int` or :class:`list`(:class:`int`)
                Number of buffer layers
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-11-14 ``@ddalle``: First version
        """
        self.set_key('buf', buf, i)


    # Get the number of additional adaptations using same error map
    def get_final_mesh_xref(self, i=None):
        """Get the number additional adaptations to perform on final error map
        
        :Call:
            >>> xref = opts.get_final_mesh_xref(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *xref*: :class:`int` or :class:`list`(:class:`int`)
                Number of additional adaptations
        :Versions:
            * 2014-11-19 ``@ddalle``: First version
        """
        return self.get_key('final_mesh_xref', i)

    # Set the number of additional adaptations using same error map
    def set_final_mesh_xref(self, xref=rc0('final_mesh_xref'), i=None):
        """Set the number additional adaptations to perform on final error map
        
        :Call:
            >>> opts.set_final_mesh_xref(xref, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *xref*: :class:`int` or :class:`list`(:class:`int`)
                Number of additional adaptations
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-11-19 ``@ddalle``: First version
        """
        self.set_key('final_mesh_xref', xref, i)
# class Adaptation

# Class for autoInputs
class autoInputs(odict):
    """Dictionary-based interface for `autoInputs` options"""
    
    # Get the nominal mesh radius
    def get_r(self, i=None):
        """Get the nominal mesh radius
        
        :Call:
            >>> r = opts.get_r(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *r*: :class:`float` or :class:`list`(:class:`float`)
                Nominal mesh radius
        :Versions:
            * 2014-08-03 ``@ddalle``: First version
        """
        return self.get_key('r', i)
        
    # Set the nominal mesh radius
    def set_r(self, r=rc0('r'), i=None):
        """Set the nominal mesh radius
        
        :Call:
            >>> opts.set_r(r, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *r*: :class:`float` or :class:`list`(:class:`float`)
                Nominal mesh radius
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-08-03 ``@ddalle``: First version
        """
        self.set_key('r', r, i)
        
    # Get the number of initial divisions
    def get_nDiv(self, i=None):
        """Get the number of divisions in background mesh
        
        :Call:
            >>> nDiv = opts.get_nDiv(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *nDiv*: :class:`int` or :class:`list`(:class:`int`)
                Number of background mesh divisions
        :Versions:
            * 2014-12-02 ``@ddalle``: First version
        """
        return self.get_key('nDiv', i)
        
    # Set the number of initial mesh divisions
    def set_nDiv(self, nDiv=rc0('nDiv'), i=None):
        """Set the number of divisions in background mesh
        
        :Call:
            >>> opts.set_nDiv(nDiv, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *nDiv*: :class:`int` or :class:`list`(:class:`int`)
                Number of background mesh divisions
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-12-02 ``@ddalle``: First version
        """
        self.set_key('nDiv', nDiv, i)
# class autoInputs

# Class for cubes
class cubes(odict):
    """Dictionary-based interface for `cubes` options"""
    
    # Get the maximum number of refinements
    def get_maxR(self, i=None):
        """Get the number of refinements
        
        :Call:
            >>> maxR = opts.get_maxR(i=None):
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *maxR*: :class:`int` or :class:`list`(:class:`int`)
                (Maximum) number of refinements for initial mesh
        :Versions:
            * 2014-08-03 ``@ddalle``: First version
        """
        return self.get_key('maxR', i)
        
    # Set the maximum number of refinements
    def set_maxR(self, maxR=rc0('maxR'), i=None):
        """Get the number of refinements
        
        :Call:
            >>> opts.set_maxR(maxR, i=None):
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *maxR*: :class:`int` or :class:`list`(:class:`int`)
                (Maximum) number of refinements for initial mesh
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-08-03 ``@ddalle``: First version
        """
        self.set_key('maxR', maxR, i)
        
    # Get the 'cubes_a' parameter
    def get_cubes_a(self, i=None):
        """Get the "cubes_a" parameter
        
        :Call:
            >>> cubes_a = opts.get_cubes_a(i=None):
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *cubes_a*: :class:`int` or :class:`list`(:class:`int`)
                Customizable parameter for `cubes`
        :Versions:
            * 2014-08-03 ``@ddalle``: First version
        """
        return self.get_key('cubes_a', i)
        
    # Set the 'cubes_a' parameter
    def set_cubes_a(self, cubes_a=rc0('cubes_a'), i=None):
        """Set the "cubes_a" parameter
        
        :Call:
            >>> opts.set_cubes_a(cubes_a, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *cubes_a*: :class:`int` or :class:`list`(:class:`int`)
                Customizable parameter for `cubes`
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-08-03 ``@ddalle``: First version
        """
        self.set_key('cubes_a', cubes_a, i)
        
    # Get the 'cubes_b' parameter
    def get_cubes_b(self, i=None):
        """Get the "cubes_b" parameter
        
        :Call:
            >>> cubes_b = opts.get_cubes_b(i=None):
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *cubes_b*: :class:`int` or :class:`list`(:class:`int`)
                Customizable parameter for `cubes`
        :Versions:
            * 2014-08-03 ``@ddalle``: First version
        """
        return self.get_key('cubes_b', i)
        
    # Set the 'cubes_b' parameter
    def set_cubes_b(self, cubes_b=rc0('cubes_b'), i=None):
        """Set the "cubes_b" parameter
        
        :Call:
            >>> opts.set_cubes_b(cubes_b, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *cubes_b*: :class:`int` or :class:`list`(:class:`int`)
                Customizable parameter for `cubes`
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-08-03 ``@ddalle``: First version
        """
        self.set_key('cubes_b', cubes_b, i)
        
    # Get the reorder setting
    def get_reorder(self, i=None):
        """Get the `cubes` reordering status
        
        :Call:
            >>> reorder = opts.get_reorder(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *reorder*: :class:`bool` or :class:`list`(:class:`bool`)
                Reorder status
        :Versions:
            * 2014-08-03 ``@ddalle``: First version
        """
        return self.get_key('reorder', i)
        
    # Set the reorder setting
    def set_reorder(self, reorder=rc0('reorder'), i=None):
        """Set the `cubes` reordering status
        
        :Call:
            >>> opts.set_reorder(reorder, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *reorder*: :class:`bool` or :class:`list`(:class:`bool`)
                Reorder status
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-08-03 ``@ddalle``: First version
        """
        self.set_key('reorder', reorder, i)
        
    # Get the number of initial refinements at sharp edges
    def get_sf(self, i=None):
        """Get the number of additional refinements around sharp edges
        
        :Call:
            >>> sf = opts.get_sf(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *i*: :class:`int` or ``None``
                Run index
        :Outputs:
            *sf*: :class:`int` or :class:`list` (:class:`int`)
                Number of additional refinements at sharp edges
        :Versions:
            * 2014-12-02 ``@ddalle``: First version
        """
        return self.get_key('sf', i)
        
    # Set the number of additional refinements at sharp edges
    def set_sf(self, sf=rc0('sf'), i=None):
        """Set the number of additional refinements around sharp edges
        
        :Call:
            >>> opts.set_sf(sf, i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *sf*: :class:`int` or :class:`list` (:class:`int`)
                Number of additional refinements at sharp edges
            *i*: :class:`int` or ``None``
                Run index
        :Versions:
            * 2014-12-02 ``@ddalle``: First version
        """
        self.set_key('sf', sf, i)
    
    # Get the mesh prespecification file
    def get_preSpecCntl(self):
        """Return the template :file:`preSpec.c3d.cntl` file
        
        :Call:
            >>> fpre = opts.get_preSpecCntl(i=None)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
        :Outputs:
            *fpre*: :class:`str`
                Mesh prespecification file
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        # Get the value
        fpre = self.get_key('pre', 0)
        # Check for ``None``
        if fpre is None:
            # Use default
            return rc0('pre')
        else:
            # Specified value
            return fpre

    # Set the mesh prespecification file
    def set_preSpecCntl(self, fpre=rc0('pre')):
        """Set the template :file:`preSpec.c3d.cntl` file
        
        :Call:
            >>> opts.set_preSpecCntl(fpre)
        :Inputs:
            *opts*: :class:`pyCart.options.Options`
                Options interface
            *fpre*: :class:`str`
                Mesh prespecification file
        :Versions:
            * 2014-10-08 ``@ddalle``: First version
        """
        self.set_key('pre', fpre)
# class cubes
        

# Class for flowCart settings
class RunControl(cape.options.runControl.RunControl):
    """Dictionary-based interface for options specific to ``flowCart``"""
    
    # Initialization method
    def __init__(self, fname=None, **kw):
        # Store the data in *this* instance
        for k in kw:
            self[k] = kw[k]
        # Upgrade important groups to their own classes.
        self._flowCart()
        self._adjointCart()
        self._Adaptation()
        self._autoInputs()
        self._cubes()
    
    # Initialization and confirmation for flowCart options
    def _flowCart(self):
        """Initialize `flowCart` optiosn if necessary"""
        if 'flowCart' not in self:
            # Empty/default
            self['flowCart'] = flowCart()
        elif type(self['flowCart']).__name__ == 'dict':
            # Convert to special class
            self['flowCart'] = flowCart(**self['flowCart'])
    
    # Initialization and confirmation for adjointCart options
    def _adjointCart(self):
        """Initialize `adjointCart` optiosn if necessary"""
        if 'adjointCart' not in self:
            # Empty/default
            self['adjointcart'] = adjointCart()
        elif type(self['adjointCart']).__name__ == 'dict':
            # Convert to special class
            self['adjointCart'] = adjointCart(**self['adjointCart'])
    
    # Initialization and confirmation for autoInputs options
    def _Adaptation(self):
        """Initialize adaptive options if necessary"""
        # Check for missing entirely.
        if 'Adaptation' not in self:
            # Empty/default
            self['Adaptation'] = Adaptation()
        elif type(self['Adaptation']).__name__ == 'dict':
            # Convert to special class.
            self['Adaptation'] = Adaptation(**self['Adaptation'])
    
    # Initialization and confirmation for autoInputs options
    def _autoInputs(self):
        """Initialize `autoInputs` options if necessary"""
        # Check for missing entirely.
        if 'autoInputs' not in self:
            # Empty/default
            self['autoInputs'] = autoInputs()
        elif type(self['autoInputs']).__name__ == 'dict':
            # Convert to special class.
            self['autoInputs'] = autoInputs(**self['autoInputs'])
    
    # Initialization and confirmation for cubes options
    def _cubes(self):
        """Initialize `cubes` options if necessary"""
        # Check for missing entirely.
        if 'cubes' not in self:
            # Empty/default
            self['cubes'] = cubes()
        elif type(self['cubes']).__name__ == 'dict':
            # Convert to special class.
            self['cubes'] = cubes(**self['cubes'])
        
    
    # ============== 
    # Local settings
    # ==============
   # <
    # Get aero.csh status
    def get_Adaptive(self, i=None):
        """Return whether or not to use `aero.csh`
        
        :Call:
            >>> ac = opts.get_Adaptive(i=None)
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
        # Make sure adaptation settings are present
        self._Adaptation()
        # Check the number of cycles
        if self['Adaptation'].get_key('n_adapt_cycles', i) > 0:
            # At least one cycle
            return True
        else:
            # ``None`` or ``0``
            return False
   # >
    
    # ===================
    # flowCart parameters
    # ===================
   # <
    
    # Get flowCart order
    def get_first_order(self, i=None):
        self._flowCart()
        return self['flowCart'].get_first_order(i)
        
    # Set flowCart order
    def set_first_order(self, fo=rc0('first_order'), i=None):
        self._flowCart()
        self['flowCart'].set_first_order(fo, i)
    
    # Get flowCart robust mode
    def get_robust_mode(self, i=None):
        self._flowCart()
        return self['flowCart'].get_robust_mode(i)
        
    # Set flowCart robust mode
    def set_robust_mode(self, rm=rc0('robust_mode'), i=None):
        self._flowCart()
        self['flowCart'].set_robust_mode(rm, i)
    
    # Number of iterations
    def get_it_fc(self, i=None):
        self._flowCart()
        return self['flowCart'].get_it_fc(i)
        
    # Set flowCart iteration count
    def set_it_fc(self, it_fc=rc0('it_fc'), i=None):
        self._flowCart()
        self['flowCart'].set_it_fc(it_fc, i)
    
    # Averaging interval
    def get_it_avg(self, i=None):
        self._flowCart()
        return self['flowCart'].get_it_avg(i)
        
    # Get the number of subiterations
    def get_it_sub(self, i=None):
        self._flowCart()
        return self['flowCart'].get_it_sub(i)
        
    # Set the number of subiterations
    def set_it_sub(self, it_sub=rc0('it_sub'), i=None):
        self._flowCart()
        self['flowCart'].set_it_sub(it_sub, i)
        
    # Set flowCart averaging interval
    def set_it_avg(self, it_avg=rc0('it_avg'), i=None):
        self._flowCart()
        self['flowCart'].set_it_fc(it_avg, i)
        
    # Get number of orders for early termination
    def get_nOrders(self, i=None):
        self._flowCart()
        return self['flowCart'].get_nOrders(i)
        
    # Set number of orders for early termination
    def set_nOrders(self, nOrders=rc0('nOrders'), i=None):
        self._flowCart()
        self['flowCart'].set_nOrders(nOrders, i)
        
    # Get flowCart iteration count
    def get_mg_fc(self, i=None):
        self._flowCart()
        return self['flowCart'].get_mg_fc(i)
        
    # Set flowCart iteration count
    def set_mg_fc(self, mg_fc=rc0('mg_fc'), i=None):
        self._flowCart()
        self['flowCart'].set_mg_fc(mg_fc, i)
        
    # Get flowCart full multigrid setting
    def get_fmg(self, i=None):
        self._flowCart()
        return self['flowCart'].get_fmg(i)
        
    # Set flowCart multigrid
    def set_fmg(self, fmg=rc0('fmg'), i=None):
        self._flowCart()
        self['flowCart'].set_fmg(fmg, i)
        
    # Get flowCart ploy multigrid setting
    def get_pmg(self, i=None):
        self._flowCart()
        return self['flowCart'].get_pmg(i)
        
    # Set flowCart multigrid
    def set_pmg(self, pmg=rc0('pmg'), i=None):
        self._flowCart()
        self['flowCart'].set_pmg(pmg, i)
        
    # Get unsteady status
    def get_unsteady(self, i=None):
        self._flowCart()
        return self['flowCart'].get_unsteady(i)
        
    # Set unsteady status
    def set_unsteady(self, td_fc=rc0('unsteady'), i=None):
        self._flowCart()
        self['flowCart'].set_unsteady(td_fc, i)
        
    # Get the nominal CFL number
    def get_cfl(self, i=None):
        self._flowCart()
        return self['flowCart'].get_cfl(i)
        
    # Set the nominal CFL number
    def set_cfl(self, cfl=rc0('cfl'), i=None):
        self._flowCart()
        self['flowCart'].set_cfl(cfl, i)
        
    # Get the minimum CFL number
    def get_cflmin(self, i=None):
        self._flowCart()
        return self['flowCart'].get_cflmin(i)
    
    # Set the minimum CFL number
    def set_cflmin(self, cflmin=rc0('cflmin'), i=None):
        self._flowCart()
        self['flowCart'].set_cflmin(cflmin, i)
        
    # Get the nondimensional physical time step
    def get_dt(self, i=None):
        self._flowCart()
        return self['flowCart'].get_dt(i)
        
    # Set the nondimensional physical time step
    def set_dt(self, dt=rc0('dt'), i=None):
        self._flowCart()
        self['flowCart'].set_dt(dt, i)
        
    # Get cut-cell gradient flag
    def get_tm(self, i=None):
        self._flowCart()
        return self['flowCart'].get_tm(i)
        
    # Set cut-cell gradient flag
    def set_tm(self, tm=rc0('tm'), i=None):
        self._flowCart()
        self['flowCart'].set_tm(tm, i)
        
    # Get buffer limiter switch
    def get_buffLim(self, i=None):
        self._flowCart()
        return self['flowCart'].get_buffLim(i)
        
    # Set buffer limiter switch.
    def set_buffLim(self, buffLim=rc0('buffLim'), i=None):
        self._flowCart()
        self['flowCart'].set_buffLim(buffLim, i)
        
    # Get the number of time steps between checkpoints
    def get_checkptTD(self, i=None):
        self._flowCart()
        return self['flowCart'].get_checkptTD(i)
        
    # Set the number of time steps between checkpoints
    def set_checkptTD(self, checkptTD=rc0('checkptTD'), i=None):
        self._flowCart()
        self['flowCart'].set_checkptTD(checkptTD, i)
        
    # Get the number of time steps between visualization outputs
    def get_vizTD(self, i=None):
        self._flowCart()
        return self['flowCart'].get_vizTD(i)
        
    # Set the number of time steps visualization outputs
    def set_vizTD(self, vizTD=rc0('vizTD'), i=None):
        self._flowCart()
        self['flowCart'].set_vizTD(vizTD, i)
        
    # Get the relaxation step command
    def get_fc_clean(self, i=None):
        self._flowCart()
        return self['flowCart'].get_fc_clean(i)
        
    # Set the relaxation step command
    def set_fc_clean(self, fc_clean=rc0('fc_clean'), i=None):
        self._flowCart()
        self['flowCart'].set_fc_clean(fc_clean, i)
        
    # Get the number of iterations to average over
    def get_fc_stats(self, i=None):
        self._flowCart()
        return self['flowCart'].get_fc_stats(i)
    
    # Set the number of iterations to average over
    def set_fc_stats(self, nstats=rc0('fc_stats'), i=None):
        self._flowCart()
        self['flowCart'].set_fc_stats(nstats, i)
        
    # Get the limiter
    def get_limiter(self, i=None):
        self._flowCart()
        return self['flowCart'].get_limiter(i)
    
    # Set the limiter
    def set_limiter(self, limiter=rc0('limiter'), i=None):
        self._flowCart()
        self['flowCart'].set_limiter(limiter, i)
        
    # Get the y_is_spanwise status
    def get_y_is_spanwise(self, i=None):
        self._flowCart()
        return self['flowCart'].get_y_is_spanwise(i)
        
    # Set the y_is_spanwise status
    def set_y_is_spanwise(self, y_is_spanwise=rc0('y_is_spanwise'), i=None):
        self._flowCart()
        self['flowCart'].set_y_is_spanwise(y_is_spanwise, i)
        
    # Get the binary I/O status
    def get_binaryIO(self, i=None):
        self._flowCart()
        return self['flowCart'].get_binaryIO(i)
        
    # Set the binary I/O status
    def set_binaryIO(self, binaryIO=rc0('binaryIO'), i=None):
        self._flowCart()
        self['flowCart'].set_binaryIO(binaryIO, i)
        
    # Get the Tecplot output status
    def get_tecO(self, i=None):
        self._flowCart()
        return self['flowCart'].get_tecO(i)
        
    # Set the Tecplot output status
    def set_tecO(self, tecO=rc0('tecO'), i=None):
        self._flowCart()
        self['flowCart'].set_tecO(tecO, i)
        
    # Get the current Runge-Kutta scheme
    def get_RKScheme(self, i=None):
        self._flowCart()
        return self['flowCart'].get_RKScheme(i)
        
    # Set the Runge-Kutta scheme
    def set_RKScheme(self, RK=rc0('RKScheme'), i=None):
        self._flowCart()
        self['flowCart'].set_RKScheme(RK, i)
        
    # Copy over the documentation.
    for k in ['it_fc', 'it_sub', 'it_avg', 'dt',
            'unsteady', 'first_order', 'robust_mode', 'RKScheme',
            'tm', 'mg_fc', 'cfl', 'cflmin', 'limiter', 'fmg', 'pmg', 
            'checkptTD', 'vizTD', 'fc_clean', 'fc_stats',
            'nOrders', 'buffLim', 'y_is_spanwise',
            'binaryIO', 'tecO']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(flowCart,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(flowCart,'set_'+k).__doc__
   # >
   
    # ====================
    # adjointCart settings
    # ====================
   # <
    
    # Number of iterations
    def get_it_ad(self, i=None):
        self._adjointCart()
        return self['adjointCart'].get_it_ad(i)
        
    # Set adjointCart iteration count
    def set_it_ad(self, it_ad=rc0('it_ad'), i=None):
        self._adjointCart()
        self['adjointCart'].set_it_ad(it_ad, i)
    
    # Get adjointCart iteration count
    def get_mg_ad(self, i=None):
        self._adjointCart()
        return self['adjointCart'].get_mg_ad(i)
        
    # Set adjointCart iteration count
    def set_mg_ad(self, mg_ad=rc0('mg_ad'), i=None):
        self._adjointCart()
        self['adjointCart'].set_mg_ad(mg_ad, i)
        
    # First-order adjoint
    def get_adj_first_order(self, i=None):
        self._adjointCart()
        return self['adjointCart'].get_adj_first_order(i)
        
    # First-order adjoint
    def set_adj_first_order(self, adj=rc0('adj_first_order'), i=None):
        self._adjointCart()
        self['adjointCart'].set_adj_first_order(adj, i)
        
    # Copy over the documentation.
    for k in ['it_ad', 'mg_ad', 'adj_first_order']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(adjointCart,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(adjointCart,'set_'+k).__doc__
   # >
    
    # ===================
    # Adaptation settings
    # ===================
   # <
    
    # Get number of adapt cycles
    def get_n_adapt_cycles(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_n_adapt_cycles(i)
        
    # Set number of adapt cycles
    def set_n_adapt_cycles(self, nAdapt=rc0('n_adapt_cycles'), i=None):
        self._Adaptation()
        self['Adaptation'].set_n_adapt_cycles(nAdapt, i)
        
    # Get jumpstart status
    def get_jumpstart(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_jumpstart(i)
        
    # Jumpstart status
    def set_jumpstart(self, js=rc0('jumpstart'), i=None):
        self._Adaptation()
        self['Adaptation'].set_jumpstart(js, i)
    
    # Get error tolerance
    def get_etol(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_etol(i)
        
    # Set error tolerance
    def set_etol(self, etol=rc0('etol'), i=None):
        self._Adaptation()
        self['Adaptation'].set_etol(etol, i)
    
    # Get maximum cell count
    def get_max_nCells(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_max_nCells(i)
        
    # Set maximum cell count
    def set_max_nCells(self, etol=rc0('max_nCells'), i=None):
        self._Adaptation()
        self['Adaptation'].set_max_nCells(etol, i)
    
    # Get flowCart iterations on refined meshes
    def get_ws_it(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_ws_it(i)
        
    # Set flowCart iterations on refined meshes
    def set_ws_it(self, ws_it=rc0('ws_it'), i=None):
        self._Adaptation()
        self['Adaptation'].set_ws_it(ws_it, i)
        
    # Get mesh growth ratio
    def get_mesh_growth(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_mesh_growth(i)
        
    # Set mesh growth ratio
    def set_mesh_growth(self, mesh_growth=rc0('mesh_growth'), i=None):
        self._Adaptation()
        self['Adaptation'].set_mesh_growth(mesh_growth, i)
        
    # Get mesh refinement cycle type
    def get_apc(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_apc(i)
        
    # Set mesh refinement cycle type
    def set_apc(self, apc=rc0('apc'), i=None):
        self._Adaptation()
        self['Adaptation'].set_apc(apc, i)
        
    # Get number of buffer layers
    def get_abuff(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_abuff(i)
        
    # Set number of buffer layers
    def set_abuff(self, buf=rc0('buf'), i=None):
        self._Adaptation()
        self['Adaptation'].set_abuff(abuff, i)
    
    # Get number of additional adaptations on final error map
    def get_final_mesh_xref(self, i=None):
        self._Adaptation()
        return self['Adaptation'].get_final_mesh_xref(i)
    
    # Set number of additional adaptations on final error map
    def set_final_mesh_xref(self, xref=rc0('final_mesh_xref'), i=None):
        self._Adaptation()
        self['Adaptation'].set_final_mesh_xref(xref, i)
        
    # Copy over the documentation.
    for k in ['n_adapt_cycles', 'etol', 'max_nCells', 'ws_it',
            'mesh_growth', 'apc', 'abuff', 'final_mesh_xref']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(Adaptation,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(Adaptation,'set_'+k).__doc__
   # >
    
    # ==========
    # autoInputs
    # ==========
   # <
    
    # Get the nominal mesh radius
    def get_r(self, i=None):
        self._autoInputs()
        return self['autoInputs'].get_r(i)
        
    # Set the nominal mesh radius
    def set_r(self, r=rc0('r'), i=None):
        self._autoInputs()
        self['autoInputs'].set_r(r, i)
        
    # Get the background mesh divisions
    def get_nDiv(self, i=None):
        self._autoInputs()
        return self['autoInputs'].get_nDiv(i)
    
    # Set the background mesh divisions
    def set_nDiv(self, nDiv=rc0('nDiv'), i=None):
        self._autoInputs()
        self['autoInputs'].set_nDiv(nDiv, i)
        
    # Copy over the documentation.
    for k in ['r', 'nDiv']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(autoInputs,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(autoInputs,'set_'+k).__doc__
   # >
        
    # =====
    # cubes
    # =====
   # <
    
    # Get the number of refinements
    def get_maxR(self, i=None):
        self._cubes()
        return self['cubes'].get_maxR(i)
        
    # Set the number of refinements
    def set_maxR(self, maxR=rc0('maxR'), i=None):
        self._cubes()
        self['cubes'].set_maxR(maxR, i)
        
    # Get the 'cubes_a' parameter
    def get_cubes_a(self, i=None):
        self._cubes()
        return self['cubes'].get_cubes_a(i)
        
    # Set the 'cubes_a' parameter
    def set_cubes_a(self, cubes_a=rc0('cubes_a'), i=None):
        self._cubes()
        self['cubes'].set_cubes_a(cubes_a, i)
        
    # Get the 'cubes_b' parameter
    def get_cubes_b(self, i=None):
        self._cubes()
        return self['cubes'].get_cubes_b(i)
        
    # Set the 'cubes_a' parameter
    def set_cubes_b(self, cubes_b=rc0('cubes_b'), i=None):
        self._cubes()
        self['cubes'].set_cubes_b(cubes_b, i)
        
    # Get the mesh reordering status
    def get_reorder(self, i=None):
        self._cubes()
        return self['cubes'].get_reorder(i)
        
    # Set the mesh reordering status
    def set_reorder(self, reorder=rc0('reorder'), i=None):
        self._cubes()
        self['cubes'].set_reorder(reorder, i)
        
    # Get the additional refinements around sharp edges
    def get_sf(self, i=None):
        self._cubes()
        return self['cubes'].get_sf(i)
        
    # Set the additional refinements around sharp edges
    def set_sf(self, sf=rc0('sf'), i=None):
        self._cubes()
        self['cubes'].set_sf(sf, i)
    
    # Get preSpec file
    def get_preSpecCntl(self):
        self._cubes()
        return self['cubes'].get_preSpecCntl()
        
    # Set preSpec file
    def set_preSpecCntl(self, fpre=rc0('pre')):
        self._cubes()
        self['cubes'].set_preSpecCntl(fpre)


    # Copy over the documentation.
    for k in ['maxR', 'cubes_a', 'cubes_b', 'reorder', 'sf', 'preSpecCntl']:
        # Get the documentation for the "get" and "set" functions
        eval('get_'+k).__doc__ = getattr(cubes,'get_'+k).__doc__
        eval('set_'+k).__doc__ = getattr(cubes,'set_'+k).__doc__
   # >
# class RunControl
        

