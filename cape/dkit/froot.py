# -*- coding: utf-8 -*-
r"""
This module provides a function :func:`froot1` to solve nonlinear
equations

    .. math::
    
        f(x) = 0

where both *x* and the output of *f* can be either scalars or vectors.
The function operates without attempting to use exact derivative or
gradient information.

"""

# Third-party
import numpy as np
    
    
# Basic scalar root finder
def froot1(f, x0, *a, **kw):
    r"""Find a root of a real-valued scalar nonlinear function
    
    :Call:
        >>> x = froot1(f, x0, **kw)
        >>> x = froot1(f, x0, x1, **kw)
    :Inputs:
        *f*: :class:`func`
            Function which can evaluate ``f(x0)``
        *x0*: :class:`float`
            Initial guess, scalar or vector
        *x1*: ``class(x0)``
            (Optional) secondary guess to scale initial step or bracket
        *xtol*, *TolX*: {``1e-14``} | :class:`float`
            Tolerance in *x*
        *ytol*, *TolFun*: {``1e-14``} | :class:`float`
            Tolerance for absolute value of ``f(x)``
        *imax*, *MaxIter*: {``80``} | :class:`int`
            Maximum iterations
        *fmax*, *MaxFunEvals*: {``161``} | :class:`int`
            Maximum function evaluations
        *y0*: {``None``} | :class:`float`
            Optional evaluation of ``f(x0)`` to reduce func evals by one
        *v*, *Display* : ``True`` | {``False``}
            Display iterative histories to STDOUT
    :Outputs:
        *x*: :class:`float`
            Solution such that ``f(x)`` is approximately zero
    :Versions:
        * 2010-03-13 ``@ddalle``: Optimized scalar root search
        * 2018-07-24 ``@ddalle``: Ported to Python (from MATLAB)
    """
    # Process optional args
    if len(a) > 1:
        # Too many inputs
        raise IOError(
            "Too many arguments; function requires 2 or 3 " +
            ("non-keyword arguments; %s received" % 2+len(a)))
    elif len(a) > 0:
        # Secondary step given
        x1 = a[0]
    else:
        # No secondary step
        x1 = None
    # Process keyword secondary guess
    x1 = kw.get("x1")
    # Verbosity
    v = kw.get("v", kw.get("Display", False))
    # Other keyword arguments
    xtol = kw.get("xtol", kw.get("TolX", 1e-14))
    ytol = kw.get("ytol", kw.get("TolFun", 1e-14))
    imax = kw.get("imax", kw.get("MaxIter", 80))
    fmax = kw.get("fmax", kw.get("MaxFunEvals", 161))
    # Get initial output
    y0 = kw.get("y0")
    # Evaluate if needed
    if y0 is None:
        # Evaluate initial guess
        try:
            # Attempt evaluation at initial point
            y0 = np.real(f(x0))
        except Exception:
            # Could not evaluate
            raise ValueError(
                "Could not evaluate function at initial conditions.")
    else:
        # Only use real part
        y0 = np.real(y0)
    # Redefine function
    fn = lambda x: np.real(f(x))
    # Initialize counters
    i_iter = 0
    i_eval = 1
    # Check for secondary input
    if x1 is None:
        # Set both values to the initial guess
        a, fa = x0, y0
        b, fb = x0, y0
    else:
        # use the secondary point as counterpoint
        a = x1
        b, fb = x0, y0
        # Try to evaluate *a*
        try:
            fa = fn(a)
        except Exception:
            fa = np.nan
        # If evaluation failed (or returned NaN), use *b*
        if np.isnan(fa):
            a, fb = b, fa
        # Evaluation counter
        i_eval += 1
    # Make sure *b* is a closer estimate of the solution than *a*
    if np.abs(fa) < np.abs(fb):
        a, fa, b, fb = b, fb, a, fa
    # Assign *a* to previous estimates
    c, fc = a, fa
    d, fd = a, fa
    # Test for initial bracket
    q_bracket = fa*fb < 0
    # Initial convergence test
    q_cont = np.abs(fb) > ytol or np.abs(b-a) > xtol
    # Display headers
    if v:
        print(" Iter  Evals       x          |f(x)|       |dx|     Method")
    # Default search length
    x_search = max(1.0, 100*np.abs(b-c))
    # Iterations
    while q_cont:
        # Iteration counter
        i_iter += 1
        # Best estimate of next point
        if q_bracket:
           # -- Bracketing methods --
            # Check if there are three unique points.
            if np.min(np.abs([c-b, c-a])) > xtol:
                # Divided differences
                z01 = (fb-fa)/(b-a)
                z02 = (fb-fc)/(b-c)
                z12 = (fa-fc)/(a-c)
                z012 = (z12 - z01) / (c - b)
                # Derivatives
                w = z01 + z02 - z12
                # Discriminant
                q = w*w - 4*fb*z012
                # Check the discriminant (should be positive)
                if q > 0 and w > 0:
                    # Quadratic interpolation
                    s = b - 2*fb/(w + np.sqrt(q))
                elif q > 0:
                    # Quadratic interpolation
                    s = b - 2*fb/(w - np.sqrt(q))
                else:
                    # Linear interpolation
                    s = b - fb / z01
                # Method name
                if v:
                    s_method = 'bracket-mueller'
            elif np.abs(fc - fb) > ytol:
                # Root of the secant
                s = b - (c - b) * fb / (fc - fb)
                # Method name
                if v:
                    s_method = 'bracket-newton'
            else:
                # Pick a point outside the interval
                s = np.min([a, b]) - 1
                # Method name
                if v:
                    s_method = "open-step-left"
            # Check if the root is strictly between *a* and *b*
            if (s-a)*(s-b) >= 0:
                # Root of the linear interpolation.
                s = b - (a - b) * fb / (fa - fb)
                # Method name
                if v:
                    s_method = 'bracket-linear'
            # Check the function convergence criterion
            if abs(fb) <= ytol and np.abs(fa) > 10*np.abs(fb):
                # Direction to go
                xdir = -np.sign((b-a)*(fb-fa)*fb)
                # Try to go just across the root
                s = s + 0.5 * xtol * xdir
                # Method name
                if v:
                    s_method = "root-cross"
            # Evaluation
            try:
                fs = fn(s)
            except Exception:
                # Failed evaluation; set to NaN.
                fs = np.nan
            # Evaluation counter
            i_eval += 1
            # Save the current interval width and function value
            dx = np.abs(b-a)
            # Pick the side containing the root
            if fs == 0:
                # Exact root found
                a, fa, b, fb = s, fs, s, fs
            elif fs * fb < 0:
                # Root between *b* and *s*
                a, fa, c, fc = s, fs, a, fa
            elif not np.isnan(fs):
                # Root between *a* and *s*
                b, fb, c, fc = s, fs, b, fb
            # Check if it was a good update
            if abs(b-a) > dx / 2 and min(abs(fa/fc), abs(fb/fc)) > 0.75:
                # Perform a bisection
                m = (a + b) / 2
                # Check for convergence below machine epsilon
                if np.abs(m-a) == 0 or np.abs(m-b) == 0:
                    break
                # Evaluation
                try:
                    fm = fn(m)
                except Exception:
                    # Failed evaluation; set to NaN.
                    fm = np.nan
                # Evaluation counter
                i_eval += 1
                # Pick the correct side
                if fm == 0:
                    # Exact root found
                    a, fa, b, fb = m, fm, m, fm
                elif fm * fb < 0:
                    # Root between *b* and *s*
                    a, fa = m, fm
                elif fm * fa < 0:
                    # Root between *a* and *s*
                    b, fb = m, fm
                # Method name
                if v:
                    s_method = 'bracket-bisection'
            # Ensure that |f(b)| <= |f(a)|.
            if np.abs(fa) < np.abs(fb):
                a, fa, b, fb = b, fb, a, fa
        else:
           # -- Open methods --
            if np.max(np.abs([b-c, b-d])) <= xtol:
                # Same three points; move over by a small amount
                s = b + 10*xtol*max(1.0, np.abs(b))
                # Method name
                if v:
                    s_method = 'open-step'
            elif np.max(np.abs([fb-fc, fb-fd])) <= ytol:
                # Matching function evaluations; search for something else
                s = b + x_search
                # Change the search value
                x_search = -2 * x_search
                # Method name
                if v:
                    s_method = 'open-search'
            elif np.abs(b-c) <= xtol:
                # Use Newton's method with previous old estimate.
                s = b - fb*(b - d)/(fb - fd)
                # Method name
                if v:
                    s_method = 'open-newton'
            elif np.abs(c-d) <= xtol:
                # Use Newton's method
                s = b - fb*(b - c)/(fb - fc)
                # Method name
                if v:
                    s_method = 'open-newton'
            else:
                # Use Mueller's full method
                # Divided differences.
                z01 = (fb-fc)/(b-c)
                z02 = (fb-fd)/(b-d)
                z12 = (fc-fd)/(c-d)
                z012 = (z12 - z01) / (d - b)
                # Derivatives
                w = z01 + z02 - z12
                # Discriminant
                q = w*w - 4*fb*z012
                r = np.sqrt(q)
                # Check the discriminant.
                if q > 0 and w > 0:
                    # Real quadratic interpolation
                    s = b - 2*fb/(w + r)
                elif q > 0:
                    # Real quadratic interpolation
                    s = b - 2*fb/(w - r)
                elif np.abs(z012) <= ytol:
                    # Use Newton's method
                    s = b - fb / z01
                else:
                    # Use the minimum of the parabola
                    s = b - w / (2*z012)
                # Method name
                if v:
                    s_method = 'open-mueller'
            # Evaluate the new point.
            while True:
                try:
                    # First attempt at evaluation
                    fs = fn(s)
                except Exception:
                    # Failed evaluation: set to NaN
                    fs = np.nan
                # Evaluation counter
                i_eval += 1
                # Check if it was successful
                if not np.isnan(fs):
                    # Succesful evaluation
                    break
                elif i_eval >= fmax:
                    # Forced to quit by evaluation criteria
                    break
                elif np.abs(s-b) > xtol:
                    # Move closer to *b*
                    s = 0.9*b + 0.1*s
                else:
                    # Move to the other side of *b*
                    s = 11*b - 10*s
            # Update points
            c, fc, d, fd = b, fb, c, fc
            a, fa, b, fb = b, fb, s, fs
        # --- Iteration checks ---
        # Check if a bracket has been found
        q_bracket = not np.any(np.isnan([a, b, fa, fb])) and fa*fb < 0
        # Test for exact solution
        q_cont = fb != 0
        # Test for convergence
        q_cont = q_cont and np.abs(fb) > ytol
        # Test interval width
        q_cont = q_cont and np.abs(b-a)*max(1.0, np.abs(b)) > xtol
        # Test if iterations or function evaluations have been exhausted
        q_cont = q_cont and i_iter < imax and i_eval+1 < fmax
        # Check for a function with no root
        q_cont = q_cont and (q_bracket or i_iter < 3 or np.abs(b-c) > ytol)
        # Iteration display
        if v:
            print(' %3i  %4i    %+12.4e %11.4e %11.4e  %s' % (
                i_iter, i_eval, b, np.abs(fb), np.abs(b-a), s_method))
    # Output
    return b

