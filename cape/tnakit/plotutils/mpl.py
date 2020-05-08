#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------
:mod:`cape.tnakit.plotutils.mpl`: Matplotlib/Pyplot Interfaces
--------------------------------------------------------------------

This module contains handles to various :mod:`matplotlib` plotting
methods.  In particular, it centralizes some of the multiple-line
instructions that are used in multiple places.  An example is placing
labels on one or more corners of the axis window.

It also includes syntax to import modules without raising ``ImportError``.
"""

# Standard library modules
import os

# Standard third-party modules
import numpy as np

# Statistics
try:
    from scipy.stats import t as student
except ImportError:
    pass

# Local modules
from . import genopts
from . import mplopts

# TNA toolkit modules
from .. import optitem
from .. import typeutils
from .. import statutils as stats

# Get a variable to hold the "type" of "module"
mod = os.__class__

# Initialize handle for modules
mpl = object()
plt = object()


# Import :mod:`matplotlib`
def import_matplotlib():
    """Function to import Matplotlib if possible

    This function checks if the global variable *mpl* is already a
    module.  If so, the function exits without doing anything.
    Otherwise it imports :mod:`matplotlib` as *mpl*.  If the operating
    system is not Windows, and there is no environment variable
    *DISPLAY*, the backend is set to ``"Agg"``.

    :Call:
        >>> import_matplotlib()
    :Versions:
        * 2019-08-22 ``@ddalle``: Documented first version
    """
    # Make global variables
    global mpl
    # Exit if already imported
    if isinstance(mpl, mod):
        return
    # Import module
    try:
        import matplotlib as mpl
    except ImportError:
        return
    # Check for no-display
    if (os.name != "nt") and (os.environ.get("DISPLAY") is None):
        # Not on Windows and no display: no window to create fig
        mpl.use("Agg")


# Import :mod:`matplotlib`
def import_pyplot():
    """Function to import Matplotlib's PyPlot if possible

    This function checks if the global variable *plt* is already a
    module.  If so, the function exits without doing anything.
    Otherwise it imports :mod:`matplotlib.pyplot` as *plt* after
    calling :func:`import_matplotlib`.

    :Call:
        >>> import_pyplot()
    :See also:
        * :func:`import_matplotlib`
    :Versions:
        * 2019-08-22 ``@ddalle``: Documented first version
    """
    # Make global variables
    global plt
    # Exit if already imported
    if isinstance(plt, mod):
        return
    # Otherwise, import matplotlib first
    import_matplotlib()
    # Import module
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return


# Primary plotter
def plot(xv, yv, *a, **kw):
    """Plot connected points with many options

    :Call:
        >>> h, kw = plot_base(xv, yv, *a, **kw)
    :Inputs:
        *xv*: :class:`np.ndarray` (:class:`float`)
            Array of values for *x*-axis
        *yv*: :class:`np.ndarray` (:class:`float`)
            Array of values for *y*-axis
    :Outputs:
        *h*: :class:`dict`
            Dictionary of plot handles
    :Versions:
        * 2019-03-01 ``@ddalle``: First (independent) version
    """
   # --- Prep ---
    # Ensure plot() is loaded
    import_pyplot()
    # Initialize output
    h = {}
    # Number of args and args used
    na = len(a)
    ia = 0
   # --- Default Control ---
    # Min/Max
    if ("ymin" in kw) and ("ymax" in kw):
        # If min/max values are specified, turn on *PlotMinMax*
        qmmx = True
    else:
        # Otherwise false
        qmmx = False
    # UQ
    if ("yerr" in kw) or ("xerr" in kw):
        # If min/max values are specified, turn on *PlotUncertainty*
        quq = True
    else:
        # Otherwise false
        quq = False
   # --- Control Options ---
    # Options to plot min/max
    qline = kw.pop("PlotLine", True)
    qmmx = kw.pop("PlotMinMax", qmmx)
    qerr = kw.pop("PlotError", False)
    quq = kw.pop("PlotUncertainty", kw.pop("PlotUQ", quq and (not qerr)))
   # --- Universal Options ---
    # Label for legend
    lbl = kw.pop("Label", "")
    # Index
    i = kw.pop("i", kw.pop("Index", 0))
    # Rotation
    r = kw.pop("Rotate", False)
    # Universal options
    kw_u = {
        "i": i,
        "Rotate": r,
    }
    # Font options
    kw_font = mplopts.font_options(kw)
   # --- Figure Setup ---
    # Process figure options
    kw_f = mplopts.figure_options(kw)
    # Get/create figure
    h["fig"] = figure(**kw_f)
   # --- Axis Setup ---
    # Process axis options
    kw_a = mplopts.axes_options(kw)
    # Get/create axis
    ax = axes(**kw_a)
    # Save it
    h["ax"] = ax
   # --- Primary Plot ---
    # Process plot options
    kw_p = mplopts.plot_options(kw, kw_u)
    # Add label
    kw_p.setdefault("label", lbl)
    # Call plot method
    if qline:
        # Check *a[0]*
        if (na > 0) and typeutils.isstr(a[0]):
            # Format given
            fmt = (a[0],)
            # One arg used
            ia += 1
        else:
            # No format option
            fmt = tuple()
        # Plot call
        h['line'] = plot_line(xv, yv, *fmt, **kw_p)
   # --- Min/Max ---
    # Process min/max options
    t_mmax, kw_mmax = mplopts.minmax_options(kw, kw_p, kw_u)
    # Plot it
    if qmmx:
        # Min/max values
        if na >= ia+2:
            # Get parameter values
            ymin = a[ia]
            ymax = a[ia+1]
            # Increase count
            ia += 2
        else:
            # Pop values
            ymin = kw.pop("ymin", None)
            ymax = kw.pop("ymax", None)
        # Plot call
        if t_mmax == "FillBetween":
            # Do a :func:`fill_between` plot
            h['minmax'] = fill_between(xv, ymin, ymax, **kw_mmax)
        elif t_mmax == "ErrorBar":
            # Convert to error bar widths
            yerr = minmax_to_errorbar(yv, ymin, ymax, **kw_mmax)
            # Do a :func:`errorbar` plot
            h['minmax'] = errorbar(xv, yv, yerr)
   # --- Error ---
    # Process min/max options
    t_err, kw_err = mplopts.error_options(kw, kw_p, kw_u)
    # Plot it
    if qerr:
        # Min/max values
        if na >= ia+1:
            # Get parameter values
            yerr = a[ia]
            # Increase count
            ia += 1
        else:
            # Pop values
            yerr = kw.pop("yerr", None)
        # Check for horizontal error bars
        xerr = kw.pop("xerr", None)
        # Plot call
        if t_err == "FillBetween":
            # Convert to min/max values
            ymin, ymax = errorbar_to_minmax(yv, yerr)
            # Do a :func:`fill_between` plot
            h['error'] = fill_between(xv, ymin, ymax, **kw_err)
        elif t_err == "ErrorBar":
            # Do a :func:`errorbar` plot
            h['error'] = errorbar(xv, yv, yerr, **kw_err)
   # --- UQ ---
    # Process min/max options
    t_uq, kw_uq = mplopts.uq_options(kw, kw_p, kw_u)
    # Plot it
    if quq:
        # Min/max values
        if na >= ia+1:
            # Get parameter values
            yerr = a[ia]
            # Increase count
            ia += 1
        else:
            # Pop values
            yerr = kw.pop("yerr", kw.pop("uy", None))
        # Check for horizontal error bars
        xerr = kw.pop("xerr", kw.pop("ux", None))
        # Plot call
        if t_uq == "FillBetween":
            # Convert to min/max values
            ymin, ymax = errorbar_to_minmax(yv, yerr)
            # Do a :func:`fill_between` plot
            h['uq'] = fill_between(xv, ymin, ymax, **kw_uq)
        elif t_uq == "ErrorBar":
            # Do a :func:`errorbar` plot
            h['uq'] = errobar(xv, yv, yerr, **kw_uq)
   # --- Axis formatting ---
    # Process grid lines options
    kw_gl = mplopts.grid_options(kw, kw_p, kw_u)
    # Apply grid lines
    grid(ax, **kw_gl)
    # Process spine options
    kw_sp = mplopts.spine_options(kw, kw_p, kw_u)
    # Apply options relating to spines
    h["spines"] = format_spines(ax, **kw_sp)
    # Process axes format options
    kw_axfmt = mplopts.axformat_options(kw, kw_p, kw_u)
    # Apply formatting
    h['xlabel'], h['ylabel'] = format_axes(ax, **kw_axfmt)
   # --- Legend ---
    # Process options for legend
    kw_leg = mplopts.legend_options(kw, kw_font)
    # Create legend
    h['legend'] = legend(ax, **kw_leg)
   # --- Cleanup ---
    # Any unused keys?
    genopts.display_remaining("plot", "    ", **kw)
    # Output
    return h


# Primary Histogram plotter
def hist(v, **kw):
    """Plot histograms with many options

    :Call:
        >>> h, kw = plot_base(xv, yv, *a, **kw)
    :Inputs:
        *v*: :class:`np.ndarray` (:class:`float`)
            List of values for which to create histogram
    :Outputs:
        *h*: :class:`dict`
            Dictionary of plot handles
    :Versions:
        * 2019-03-11 ``@jmeeroff``: function created
    """
   # --- Prep ---
    # Ensure plot() is loaded
    import_pyplot()
    # Initialize output
    h = {}
    # Filter out non-numeric entries
    v = v[np.logical_not(np.isnan(v))]
    # define coefficient for labeling
    coeff = kw.pop("coeff", "c")
   # --- Statistics ---
    # Calculate the mean
    vmu = np.mean(v)
    # Calculate StdDev
    vstd = np.std(v)
    # Coverage Intervals
    cov = kw.pop("Coverage", kw.pop("cov", 0.99))
    cdf = kw.pop("CoverageCDF", kw.pop("cdf", cov))
    # Nominal bounds (like 3-sigma for 99.5% coverage, etc.)
    kcdf = student.ppf(0.5+0.5*cdf, v.size)
    # Check for outliers ...
    fstd = kw.pop('FilterSigma', 2.0*kcdf)
    # Remove values from histogram
    if fstd:
        # Find indices of cases that are within outlier range
        j = np.abs(v-vmu)/vstd <= fstd
        # Filter values
        v = v[j]
    # Calculate interval
    acov, bcov = stats.get_cov_interval(v, cov, cdf=cdf, **kw)
   # --- Universal Options ---
    # dictionary for universal options
    # 1 -- Orientation
    orient = kw.pop("orientation", "vertical")
    kw_u = {
        "orientation": orient,
    }
   # --- Figure Setup ---
    # Process figure options
    kw_f = mplopts.figure_options(kw)
    # Get/create figure
    h["fig"] = figure(**kw_f)
   # --- Axis Setup ---
    # Process axis options
    kw_a = mplopts.axes_options(kw)
    # Get/create axis
    ax = axes(**kw_a)
    # Save it
    h["ax"] = ax
   # --- Histogram Plot ---
    # Process histogram plot options
    kw_p = mplopts.hist_options(kw, kw_u)
    # Plot
    h['hist'] = _hist(v, **kw_p)
   # --- Plot Gaussian ---
    normed = kw_p['density']
    # Need to pop the gaussian command regardless
    qgaus = kw.pop("PlotGaussian", True)
    # Only plot gaussian if the historgram is 'normed'
    qgaus = normed and qgaus
    # Process gaussian options
    kw_gauss = mplopts.gauss_options(kw, kw_p, kw_u)
    # Plot gaussian
    if qgaus:
        h['gauss'] = plot_gaussian(ax, vmu, vstd, **kw_gauss)
   # --- Axis formatting ---
    # Process grid lines options
    kw_gl = mplopts.grid_options(kw, kw_p, kw_u)
    # Apply grid lines
    grid(ax, **kw_gl)
    # Process spine options
    kw_sp = mplopts.spine_options(kw, kw_p, kw_u)
    # Apply options relating to spines
    h["spines"] = format_spines(ax, **kw_sp)
    # Process axes format options
    kw_axfmt = mplopts.axformat_options(kw, kw_p, kw_u)
    # Force y=0 or x=0 depending on orientation
    if orient == "vertical":
        kw_axfmt['YMin'] = 0
    else:
        kw_axfmt['XMin'] = 0
    # Apply formatting
    h['xlabel'], h['ylabel'] = format_axes(h['ax'], **kw_axfmt)
   # --- Plot an Interval ---
    # Get flag for manual interval plot
    qint = kw.pop("PlotInterval", False)
    # Process options
    kw_interval = mplopts.interval_options(kw, kw_p, kw_u)
    # Plot if necessary
    if qint:
        h['interval'] = plot_interval(ax, acov, bcov, **kw_interval)
   # --- Plot Mean ---
    qmu = kw.pop("PlotMean", True)
    # Process mean options
    kw_mu = mplopts.mu_options(kw, kw_p, kw_u)
    # Plot the mean
    if qmu:
        h['mean'] = plot_mean(ax, vmu, **kw_mu)
   # --- Plot Standard Deviation ---
    qsig = kw.pop("PlotSigma", False)
    # Process sigma options
    kw_s = mplopts.std_options(kw, kw_p, kw_u)
    # Plot the sigma
    if qsig:
        h['sigma'] = plot_lines_std(ax, vmu, vstd, **kw_s)
   # --- Delta Plot ----
    qdel = kw.pop("PlotDelta", False)
    # Process delta options
    kw_delta = mplopts.delta_options(kw, kw_p, kw_u)
    # Plot the delta
    if qdel:
        h['delta'] = plot_delta(ax, vmu, **kw_delta)
   # --- Mean labels ---
    # Process mean labeling options
    opts = kw.pop("MeanLabelOptions", {})
    kw_mu_lab = mplopts.histlabel_options(opts, kw_p, kw_u)
    c = kw_mu_lab.pop('clabel', 'mu')
    # Do the label
    if qmu:
        # Formulate the label
        # Format code
        flbl = kw_mu_lab.pop("MuFormat", "%.4f")
        # First part
        klbl = (u'%s' % c)
        # Insert value
        lbl = ('%s = %s' % (klbl, flbl)) % vmu
        h['MeanLabel'] = histlab_part(lbl, 0.99, True, ax, **kw_mu_lab)
   # --- StdDev labels ---
    # Process mean labeling options
    opts = kw.pop("SigmaLabelOptions", {})
    # Make sure alignment is correct
    opts.setdefault('horizontalalignment', 'left')
    kw_sig_lab = mplopts.histlabel_options(opts, kw_p, kw_u)
    c = kw_sig_lab.pop('clabel', coeff)
    # Do the label
    if qsig:
        # Formulate the label
        # Format code
        flbl = kw_mu_lab.pop("SigFormat", "%.4f")
        # First part
        klbl = (u'\u03c3(%s)' % c)
        # Insert value
        lbl = ('%s = %s' % (klbl, flbl)) % vstd
        h['SigmaLabel'] = histlab_part(lbl, 0.01, True, ax, **kw_sig_lab)
   # --- Interval Labels ---
    # Process mean labeling options
    opts = kw.pop("IntLabelOptions", {})
    kw_int_lab = mplopts.histlabel_options(opts, kw_p, kw_u)
    c = kw_int_lab.pop('clabel', cov)
    if qint:
        flbl = kw_sig_lab.get("IntervalFormat", "%.4f")
        # Form
        klbl = "I(%.1f%%%%)" % (100*c)
        # Get ionterval values
        a = kw_interval.get('imin', None)
        b = kw_interval.get('imax', None)
        # Insert value
        lbl = ('%s = [%s,%s]' % (klbl, flbl, flbl)) % (a, b)
        h['IntLabel'] = histlab_part(lbl, 0.99, False, ax, **kw_int_lab)
   # --- Delta Labels ---
    # Process Delta labeling options
    opts = kw.pop("DeltaLabelOptions", {})
    opts.setdefault('horizontalalignment', 'left')
    kw_del_lab = mplopts.histlabel_options(opts, kw_p, kw_u)
    c = kw_del_lab.pop('clabel', 'coeff')
    if qdel:
        flbl = kw_del_lab.get("DeltaFormat", "%.4f")
        # Get delta values
        dc = kw_delta.get('Delta', 0.0)
        # Insert value
        if type(dc).__name__ in ['ndarray', 'list', 'tuple']:
            lbl = (u'\u0394%s = (%s, %s)' % (c, flbl, flbl)) % (dc[0], dc[1])
        else:
            lbl = (u'\u0394%s = %s' % (c, flbl)) % dc
        h['delta'] = histlab_part(lbl, 0.01, False, ax, **kw_del_lab)
   # --- Cleanup ---
    # Any unused keys?
    genopts.display_remaining("plot_hist", "    ", **kw)
    # Output
    return h


# Figure part
def figure(**kw):
    """Create new figure or edit one if necessary

    :Call:
        >>> fig = figure_part(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of figure-like options
        *fig*: ``None`` | :class:`matplotlib.figure.Figure`
            Optional figure handle
        *FigHeight*, *hfig*: ``None`` | :class:`float`
            Figure height in inches
        *FigWidth*, *wfig*: ``None`` | :class:`float`
            Figure width in inches
    :Outputs:
        *fig*: :class:`matplotlib.figure.Figure`
            Figure handle
    :Versions:
        * 2019-03-06 ``@ddalle``: First version
    """
    # Import PyPlot
    import_pyplot()
    # Get figure handle
    fig = kw.pop("fig", None)
    # Figure size
    hfig = kw.pop("FigHeight", kw.pop("hfig", None))
    wfig = kw.pop("FigWidth",  kw.pop("wfig", None))
    # Create figure if needed
    if fig is None:
        # Use most recent figure (can be new one)
        fig = plt.gcf()
    # Apply sizes
    if hfig: fig.set_figheight(hfig)
    if wfig: fig.set_figwidth(wfig)
    # Loop through any remaining options
    for k in kw:
        print("  Warning: unused figure option '%s'" % k)
    # Output
    return fig


# Axis part (initial)
def axes(**kw):
    """Create new axes or edit one if necessary

    :Call:
        >>> fig = axes_part(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of figure-like options
        *ax*: ``None`` | :class:`matplotlib.axes._subplots.AxesSubplot`
            Optional axes handle
    :Outputs:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
    :Versions:
        * 2019-03-06 ``@ddalle``: First version
    """
    # Import PyPlot
    import_pyplot()
    # Get figure handle
    ax = kw.pop("ax", None)
    # Create figure if needed
    if ax is None:
        # Use most recent figure (can be new one)
        ax = plt.gca()
    # Loop through any remaining options
    for k in kw:
        print("  Warning: unused axes option '%s'" % k)
    # Output
    return ax


# Plot part
def plot_line(xv, yv, fmt=None, **kw):
    """Call the :func:`plot` function with cycling options

    :Call:
        >>> h = plot_line(xv, yv, **kw)
        >>> h = plot_line(xv, yv, fmt, **kw)
    :Inputs:
        *xv*: :class:`np.ndarray`
            Array of *x*-coordinates
        *yv*: :class:`np.ndarray`
            Array of *y*-coordinates
        *fmt*: :class:`str`
            Optional format option
        *i*, *Index*: {``0``} | :class:`int`
            Phase number to cycle through plot options
        *Rotate*: ``True`` | {``False``}
            Option to plot independent variable on vertical axis
    :Keyword Arguments:
        * See :func:`matplotlib.pyplot.plot`
    :Outputs:
        *h*: :class:`list` (:class:`matplotlib.lines.Line2D`)
            List of line instances
    :Versions:
        * 2019-03-04 ``@ddalle``: First version
    """
    # Ensure plot() is available
    import_pyplot()
    # Get index
    i = kw.pop("i", kw.pop("Index", 0))
    # Get rotation option
    r = kw.pop("Rotate", False)
    # Flip inputs
    if r:
        yv, xv = xv, yv
    # Initialize plot options
    kw_p = mplopts.select_plotopts(kw, i)
    # Call plot
    if typeutils.isstr(fmt):
        # Call with extra format argument
        h = plt.plot(xv, yv, fmt, **kw_p)
    else:
        # No format argument
        h = plt.plot(xv, yv, **kw_p)
    # Output
    return h


# Region plot
def fill_between(xv, ymin, ymax, **kw):
    """Call the :func:`fill_between` or :func:`fill_betweenx` function

    :Call:
        >>> h = fill_between(xv, ymin, ymax, **kw)
    :Inputs:
        *xv*: :class:`np.ndarray`
            Array of independent variable values
        *ymin*: :class:`np.ndarray` | :class:`float`
            Array of values or single value for lower bound of window
        *ymax*: :class:`np.ndarray` | :class:`float`
            Array of values or single value for upper bound of window
        *i*, *Index*: {``0``} | :class:`int`
            Phase number to cycle through plot options
        *Rotate*: ``True`` | {``False``}
            Option to plot independent variable on vertical axis
    :Versions:
        * 2019-03-04 ``@ddalle``: First version
        * 2019-08-22 ``@ddalle``: Renamed from :func:`fillbetween`
    """
   # --- Values ---
    # Convert possible scalar for *ymin*
    if isinstance(ymin, float):
        # Expand scalar to array
        yl = ymin*np.ones_like(xv)
    elif typeutils.isarray(ymin):
        # Ensure array (not list, etc.)
        yl = np.asarray(ymin)
    else:
        raise TypeError(
            ("'ymin' value must be either float or array ") +
            ("(got %s)" % ymin.__class__))
    # Convert possible scalar for *ymax*
    if isinstance(ymax, float):
        # Expand scalar to array
        yu = ymax*np.ones_like(xv)
    elif typeutils.isarray(ymax):
        # Ensure array (not list, etc.)
        yu = np.asarray(ymax)
    else:
        raise TypeError(
            ("'ymax' value must be either float or array ") +
            ("(got %s)" % ymax.__class__))
   # --- Main ---
    # Ensure fill_between() is available
    import_pyplot()
    # Get index
    i = kw.pop("i", kw.pop("Index", 0))
    # Get rotation option
    r = kw.pop("Rotate", False)
    # Check for vertical or horizontal
    if r:
        # Vertical function
        fnplt = plt.fill_betweenx
    else:
        # Horizontal function
        fnplt = plt.fill_between
    # Initialize plot options
    kw_fb = mplopts.select_plotopts(kw, i)
    # Call the plot method
    h = fnplt(xv, yl, yu, **kw_fb)
    # Output
    return h


# Error bar plot
def errorbar(xv, yv, yerr=None, xerr=None, **kw):
    """Call the :func:`errobar`function

    :Call:
        >>> h = errorbar(xv, yv, yerr=None, xerr=None, **kw
    :Inputs:
        *xv*: :class:`np.ndarray`
            Array of independent variable values
        *yv*: :class:`np.ndarray` | :class:`float`
            Array of values for center of error bar
        *yerr*: {``None``} | :class:`np.ndarray` | :class:`float`
            Array or constant error bar half-heights; shape(2,N) array
            for distinct above- and below-widths
        *xerr*: {``None``} | :class:`np.ndarray` | :class:`float`
            Array or constant error bar half-widths; shape(2,N) array
            for distinct above- and below-widths
        *i*, *Index*: {``0``} | :class:`int`
            Phase number to cycle through plot options
        *Rotate*: ``True`` | {``False``}
            Option to plot independent variable on vertical axis
    :Versions:
        * 2019-03-04 ``@ddalle``: First version
        * 2019-08-22 ``@ddalle``: Renamed from :func:`errorbar_part`
    """
   # --- Values ---
    # Convert possible scalar for *yerr*
    if yerr is None:
        # No vertical uncertainty (?)
        uy = None
    elif isinstance(yerr, float):
        # Expand scalar to array
        uy = yerr*np.ones_like(yv)
    elif typeutils.isarray(yerr):
        # Ensure array (not list, etc.)
        uy = np.asarray(yerr)
    else:
        raise TypeError(
            ("'yerr' value must be either float or array ") +
            ("(got %s)" % yerr.__class__))
    # Convert possible scalar for *ymax*
    if xerr is None:
        # No horizontal uncertainty
        ux = None
    elif isinstance(xerr, float):
        # Expand scalar to array
        ux = xerr*np.ones_like(xv)
    elif typeutils.isarray(xerr):
        # Ensure array (not list, etc.)
        ux = np.asarray(xerr)
    else:
        raise TypeError(
            ("'xerr' value must be either None, float, or array ") +
            ("(got %s)" % yerr.__class__))
   # --- Main ---
    # Ensure fill_between() is available
    import_pyplot()
    # Get index
    i = kw.pop("i", kw.pop("Index", 0))
    # Get rotation option
    r = kw.pop("Rotate", False)
    # Check for vertical or horizontal
    if r:
        # Vertical function
        xv, yv, xerr, yerr = yv, xv, yerr, xerr
    # Initialize plot options
    kw_eb = mplopts.select_plotopts(kw, i)
    # Call the plot method
    h = plt.errorbar(xv, yv, yerr=uy, xerr=ux, **kw_eb)
    # Output
    return h


# Histogram
def _hist(v, **kw):
    """Call the :func:`hist` function

    :Call:
        >>> h = _hist(V, **kw))
    :Inputs:
        *v*: :class:`np.ndarray` (:class:`float`)
            List of values for which to create histogram
    :Keyword Arguments:
        * See :func:`matplotlib.pyplot.hist`
    :Outputs:
        *h*: :class: `tuple`
            Tuple of (n, bins, patches)
    :Versions:
        * 2019-03-11 ``@jmeeroff``: First version
        * 2019-08-22 ``@ddalle``: From :func:`Part.hist_part`
    """
    # Ensure hist() is available
    import_pyplot()
    # Call plot
    h = plt.hist(v, **kw)
    # Output
    return h


# Mean plotting part
def plot_mean(ax, vmu, **kw):
    """Plot the mean on the histogram

    :Call:
        >>> h = plot_mean(V, ax **kw))
    :Inputs:
        *ax*: ``None`` | :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
        *vmu*: :class: 'float'
            Desired mean of distribution (`np.meam()`)
    :Keyword Arguments:
        *orientation*: {``"horizontal"``} | ``"vertical"``
            Option to flip *x* and *y* axes
    :Outputs:
        *h*: :class:`matplotlib.lines.Line2D`
            Handle to options for simple line
    :Versions:
        * 2019-03-11 ``@jmeeroff``: First version
    """
    # Ensure pyplot loaded
    import_pyplot()
    # Get horizontal/vertical option
    orient = kw.pop('orientation', "")
    # Check orientation
    if orient == 'vertical':
        # Vertical: get vertical limits of axes window
        pmin, pmax = ax.get_ylim()
        # Plot a vertical mean line
        h = plt.plot([vmu, vmu], [pmin, pmax], **kw)

    else:
        # Horizontal: get horizontal limits of axes window
        pmin, pmax = ax.get_xlim()
        # Plot a horizontal range bar
        h = plt.plot([pmin, pmax], [vmu, vmu], **kw)
    # Return
    return h[0]


# Show an interval on a histogram
def plot_interval(ax, vmin, vmax, **kw):
    """Plot the mean on the histogram

    :Call:
        >>> h = plot_interval(ax, vmin, vmax, **kw))
    :Inputs:
        *ax*: ``None`` | :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
        *vmin*: :class:`float`
            Minimum value of interval to plot
        *vmax*: :class:`float`
            Maximum value of interval to plot
    :Keyword Arguments:
        *orientation*: {``"horizontal"``} | ``"vertical"``
            Option to flip *x* and *y* axes
    :Outputs:
        *h*: :class:`matplotlib.collections.PolyCollection`
            Handle to interval plot
    :Versions:
        * 2019-03-11 ``@jmeeroff``: First version
        * 2019-08-22 ``@ddalle``: Added *vmin*, *vmax* inputs
    """
    # Ensure pyplot loaded
    import_pyplot()
    # Get horizontal/vertical option
    orient = kw.pop('orientation', "")
    # Check orientation
    if orient == 'vertical':
        # Vertical: get vertical limits of axes window
        pmin, pmax = ax.get_ylim()
        # Plot a vertical range bar
        h = plt.fill_betweenx([pmin, pmax], vmin, vmax, **kw)

    else:
        # Horizontal: get horizontal limits of axes window
        pmin, pmax = ax.get_xlim()
        # Plot a horizontal range bar
        h = plt.fill_between([pmin, pmax], vmin, vmax, **kw)
    # Return
    return h


# Gaussian plotting part
def plot_gaussian(ax, vmu, vstd, **kw):
    """Plot a Gaussian distribution (on a histogram)

    :Call:
        >>> h = plot_gaussian(ax, vmu, vstd, **kw))
    :Inputs:
        *ax*: ``None`` | :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
        *vmu*: :class: 'float'
            Mean of distribution
        *vstd*: :class: `float`
            Standard deviation of distribution
    :Keyword Arguments:
        *orientation*: {``"horizontal"``} | ``"vertical"``
            Option to flip *x* and *y* axes
        *ngauss*: {``151``} | :class:`int` > 2
            Number of points to use in Gaussian curve
    :Outputs:
        *h*: :class:`matplotlib.lines.Line2D`
            Handle to options for Gaussian curve
    :Versions:
        * 2019-03-11 ``@jmeeroff``: First version
        * 2019-08-22 ``@ddalle``: Added *ngauss*
    """
    # Ensure pyplot loaded
    import_pyplot()
    # Get horizontal/vertical option
    orient = kw.pop('orientation', "")
    # Get axis limits
    if orient == "vertical":
        # Get existing horizontal limits
        xmin, xmax = ax.get_xlim()
    else:
        # Get existing
        xmin, xmax = ax.get_ylim()
    # Number of points to plot
    ngauss = kw.pop("ngauss", 151)
    # Create points at which to plot
    xval = np.linspace(xmin, xmax, ngauss)
    # Compute Gaussian distribution
    yval = 1/(vstd*np.sqrt(2*np.pi))*np.exp(-0.5*((xval-vmu)/vstd)**2)
    # Plot
    if orient == "vertical":
        # Plot vertical dist with bump pointing to the right
        h = plt.plot(xval, yval, **kw)
    else:
        # Plot horizontal dist with vertical bump
        h = plt.plot(yval, xval, **kw)
    # Return
    return h[0]


# Interval using two lines
def plot_lines_std(ax, vmu, vstd, **kw):
    """Use two lines to show standard deviation window

    :Call:
        >>> h = plot_lines_std(ax, vmu, vstd, **kw))
    :Inputs:
        *ax*: ``None`` | :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
        *vmu*: :class: 'float'
            Desired mean of distribution (`np.meam()`)
        *vstd*: :class: `float`
            Desired standard deviation of distribution( `np.std()`)
    :Keyword Arguments:
        * "StdOptions" : :dict: of line `LineOptions`
    :Outputs:
        *h*: ``None`` | :class:`list`\ [:class:`matplotlib.Line2D`]
            List of line instances
    :Versions:
        * 2019-03-14 ``@jmeeroff``: First version
        * 2019-08-22 ``@ddalle``: From :func:`Part.std_part`
    """
    # Ensure pyplot loaded
    import_pyplot()
    # Get orientation option
    orient = kw.pop('orientation', None)
    # Check multiplier
    ksig = kw.pop('StDev', None)
    # Exit if no standard deviation to show
    if not ksig:
        return
    # Plot lines
    if type(ksig).__name__ in ['ndarray', 'list', 'tuple']:
        # Separate lower and upper limits
        vmin = vmu - ksig[0]*vstd
        vmax = vmu + ksig[1]*vstd
    else:
        # Use as a single number
        vmin = vmu - ksig*vstd
        vmax = vmu + ksig*vstd
    # Check orientation
    if orient == 'vertical':
        # Get vertical limits
        pmin, pmax = ax.get_ylim()
        # Plot a vertical line for the min and max
        h = (
            plt.plot([vmin, vmin], [pmin, pmax], **kw) +
            plt.plot([vmax, vmax], [pmin, pmax], **kw))
    else:
        # Get horizontal limits
        pmin, pmax = ax.get_xlim()
        # Plot a horizontal line for the min and max
        h = (
            plt.plot([pmin, pmax], [vmin, vmin], **kw) +
            plt.plot([pmin, pmax], [vmax, vmax], **kw))
    # Return
    return h


# Delta Plotting
def plot_delta(ax, vmu, **kw):
    """Plot deltas on the histogram

    :Call:
        >>> h = plot_delta(ax, vmu, **kw))
    :Inputs:
        *ax*: ``None`` | :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
        *vmu*: :class: 'float'
            Desired mean of distribution (`np.meam()`)
    :Keyword Arguments:
        * "DeltaOptions" : :dict: of line `LineOptions`
    :Outputs:
        *h*: :class:`list` (:class:`matplotlib.lines.Line2D`)
            List of line instances
    :Versions:
        * 2019-03-14 ``@jmeeroff``: First version
    """
    # Ensure pyplot loaded
    import_pyplot()
    # Check orientation
    orient = kw.pop('orientation', None)
    # Reference delta
    dc = kw.pop('Delta', 0.0)
    # Check for single number or list
    if type(dc).__name__ in ['ndarray', 'list', 'tuple']:
        # Separate lower and upper limits
        cmin = vmu - dc[0]
        cmax = vmu + dc[1]
    else:
        # Use as a single number
        cmin = vmu - dc
        cmax = vmu + dc
    # Check orientation
    if orient == 'vertical':
        pmin, pmax = ax.get_ylim()
        # Plot a vertical line for the min and max
        h = (
            plt.plot([cmin, cmin], [pmin, pmax], **kw) +
            plt.plot([cmax, cmax], [pmin, pmax], **kw))
    else:
        pmin, pmax = ax.get_xlim()
        # Plot a horizontal line for the min and max
        h = (
            plt.plot([pmin, pmax], [cmin, cmin], **kw) +
            plt.plot([pmin, pmax], [cmax, cmax], **kw))
    # Return
    return h


# Histogram Labels Plotting
def histlab_part(lbl, pos1, pos2, ax, **kw):
    """Plot the mean on the histogram

    :Call:
        >>> h = histlab_part(lbl, pos, ax, **kw))
    :Inputs:
        *lbl*: :class:`string`
            Label title (i.e. coefficient)
        *ax*: ``None`` | :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
        *pos1*: :class:`float`
            label locations (left to right alignment)
        *pos2*: :class:`bool`
            above(True) below(False) top spine alignment
    :Keyword Arguments:
        * "*Options" : :dict: of `LabelOptions`
    :Outputs:
        *h*: :class:`list` (:class:`matplotlib.lines.Line2D`)
            List of label instances
    :Versions:
        * 2019-03-14 ``@jmeeroff``: First version
    """
    # Ensure pyplot loaded
    import_pyplot()
    # Remove orientation orientation
    kw.pop('orientation', None)
    # get figure handdle
    f = plt.gcf()
    # Y-coordinates of the current axes w.r.t. figure scale
    ya = ax.get_position().get_points()
    ha = ya[1, 1] - ya[0, 1]
    # Y-coordinates above and below the box
    yf = 2.5 / ha / f.get_figheight()
    yu = 1.0 + 0.065*yf
    yl = 1.0 - 0.04*yf
    # Above/Below Spine location
    if pos2:
        y = yu
    else:
        y = yl
    # plot the label
    h = plt.text(pos1, y, lbl, transform=ax.transAxes, **kw)
    return h


# Legend
def legend(ax=None, **kw):
    """Create/update a legend

    :Call:
        >>> leg = legend(ax=None, **kw)
    :Inputs:
        *ax*: {``None``} | :class:`matplotlib.axes._subplots.AxesSubplot`
            Axis handle (default is ``plt.gca()``
    :Outputs:
        *leg*: :class:`matplotlib.legend.Legend`
            Legend handle
    :Versions:
        * 2019-03-07 ``@ddalle``: First version
        * 2019-08-22 ``@ddalle``: From :func:`Part.legend_part`
    """
   # --- Setup ---
    # Import modules if needed
    import_pyplot()
    # Get font properties
    opts_prop = kw.pop("prop", {})
    # Default axis: most recent
    if ax is None:
        ax = plt.gca()
    # Get figure
    fig = ax.get_figure()
   # --- Initial Attempt ---
    # Initial legend attempt
    try:
        # Use the options with specified font
        leg = ax.legend(prop=opts_prop, **kw)
    except Exception:
        # Remove font
        opts_prop.pop("family", None)
        # Do it again
        leg = ax.legend(prop=opts_prop, **kw)
   # --- Font ---
    # Number of entries in the legend
    ntext = len(leg.get_texts())
    # If no legends, remove it
    if ntext < 1:
        # Delete legend
        leg.remove()
        # Exit function
        return leg
    # Get number of columns, and apply default based on number of labels
    ncol = kw.setdefault("ncol", ntext // 5 + 1)
    # Get Number of rows
    nrow = (ntext // ncol) + (ntext % ncol > 0)
    # Default font size
    if nrow > 5:
        # Smaller font
        fsize = 7
    else:
        # Larger font (still pretty small)
        fsize = 9
    # Apply default font size
    opts_prop.setdefault("size", fsize)
   # --- Spacing ---
    # Penultimate legend
    leg = ax.legend(prop=opts_prop, **kw)
    # We have to draw the figure in order to get legend box
    fig.canvas.draw()
    # Get bounds of the legend box
    bbox = leg.get_window_extent()
    # Transformation so we convert the bounds of the legend into data space
    trans = ax.transData.inverted()
    # Convert to data space
    (xlmin, ylmin), (xlmax, ylmax) = trans.transform(bbox)
    # Get data limits
    xmin, xmax = get_xlim(ax, pad=0.0)
    ymin, ymax = get_ylim(ax, pad=0.0)
    # Get location
    loc = kw.get("loc")
    # Check for special cases
    if loc in ["upper center", 9]:
        # Check bottom of legend box
        if ylmin < ymax:
            # Get axes limits
            ya, yb = ax.get_ylim()
            # Extend the upper limit of *y*-axis to make room
            yu = yb + max(0.0, ymax-ylmin)
            # Set new limits
            ax.set_ylim(ya, yu)
    elif loc in ["lower center", 8]:
        # Check top of legend box
        if ylmax > ymin:
            # Get axes limits
            ya, yb = ax.get_ylim()
            # Extend the lower limit of *y*-axis to make room
            yl = ya - max(0.0, ylmax-ymin)
            # Set new limits
            ax.set_ylim(yl, yb)
   # --- Cleanup ---
    # Output
    return leg


# Axes format
def format_axes(ax, **kw):
    """Format and label axes

    :Call:
        >>> xl, yl = format_axes(ax, **kw)
    :Inputs:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
    :Outputs:
        *xl*: :class:`matplotlib.text.Text`
            X label
        *yl*: :class:`matplotlib.text.Text`
            Y label
    :Versions:
        * 2019-03-06 ``@jmeeroff``: First version
    """
   # --- Prep ---
    # Make sure pyplot loaded
    import_pyplot()
   # --- Labels ---
    # Get user-specified axis labels
    xlbl = kw.pop("XLabel", None)
    ylbl = kw.pop("YLabel", None)
    # Check for rotation kw
    rot = kw.pop("Rotate", False)
    # Switch args if needed
    if rot:
        # Flip labels
        xlbl, ylbl = ylbl, xlbl
    # Check for orientation kw (histogram)
    orient = kw.pop('orientation', None)
    if orient == 'vertical':
        # Flip labels
        xlbl, ylbl = ylbl, xlbl
    # Apply *x* label
    if xlbl is None:
        # Get empty label
        xl = ax.xaxis.label
    else:
        # Apply label
        xl = plt.xlabel(xlbl)
    # Apply *y* label
    if ylbl is None:
        # Get handle to empty label
        yl = ax.yaxis.label
    else:
        # Create non-empty label
        yl = plt.ylabel(ylbl)
   # --- Data Limits ---
    # Get pad parameter
    pad = kw.pop("Pad", kw.pop("pad", 0.05))
    # Specific pad parameters
    xpad = kw.pop("XPad", kw.pop("xpad", pad))
    ypad = kw.pop("YPad", kw.pop("ypad", pad))
    # Get limits that include all data (and not extra).
    xmin, xmax = get_xlim(ax, pad=xpad)
    ymin, ymax = get_ylim(ax, pad=ypad)
    # Remove remaining ``None`` imputs
    kw = genopts.denone(kw)
    # Check for specified limits
    xmin = kw.pop("XMin", xmin)
    xmax = kw.pop("XMax", xmax)
    ymin = kw.pop("YMin", ymin)
    ymax = kw.pop("YMax", ymax)
    # Make sure data is included.
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
   # --- Canvas ---
    # Attempt to apply tight axes
    try:
        plt.tight_layout()
    except Exception:
        pass
    # User-specified margins
    adj_l = kw.pop("AdjustLeft",   None)
    adj_r = kw.pop("AdjustRight",  None)
    adj_t = kw.pop("AdjustTop",    None)
    adj_b = kw.pop("AdjustBottom", None)
    # Set margins
    if adj_l is not None: plt.subplots_adjust(left=adj_l)
    if adj_r is not None: plt.subplots_adjust(right=adj_r)
    if adj_t is not None: plt.subplots_adjust(top=adj_t)
    if adj_b is not None: plt.subplots_adjust(bottom=adj_b)
   # --- Cleanup ---
    # Output
    return xl, yl


# Creation and formatting of grid lines
def grid(ax, **kw):
    """Add grid lines to an axis and format them

    :Call:
        >>> grid(ax, **kw)
    :Inputs:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
    :Effects:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Grid lines added to axes
    :Versions:
        * 2019-03-07 ``@jmeeroff``: First version
    """
    # Make sure pyplot loaded
    import_pyplot()
    # Get grid option
    ogrid = kw.pop("MajorGrid", kw.pop("Grid", None))
    # Check value
    if ogrid is None:
        # Leave it as it currently is
        pass
    elif ogrid:
        # Get grid style
        kw_g = kw.pop("GridOptions", {})
        kw_m = kw.pop("MajorGridOptions", {})
        # Combine
        kw_g = dict(kw_g, **kw_m)
        # Ensure that the axis is below
        ax.set_axisbelow(True)
        # Add the grid
        ax.grid(True, **kw_g)
    else:
        # Turn the grid off, even if previously turned on
        ax.grid(False)
    # Get minor grid option
    ogrid = kw.pop("MinorGrid", None)
    # Check value
    if ogrid is None:
        # Leave it as it currently is
        pass
    elif ogrid:
        # Get grid style
        kw_g = kw.pop("MinorGridOptions", {})
        # Ensure that the axis is below
        ax.set_axisbelow(True)
        # Minor ticks are required
        ax.minorticks_on()
        # Add the grid
        ax.grid(which="minor", **kw_g)
    else:
        # Turn the grid off, even if previously turned on
        ax.grid(False)


# Single spine: extents
def format_spine1(spine, opt, vmin, vmax):
    """Apply formatting to a single spine

    :Call:
        >>> format_spine1(spine, opt, vmin, vmax)
    :Inputs:
        *spine*: :clas:`matplotlib.spines.Spine`
            A single spine (left, right, top, or bottom)
        *opt*: ``None`` | ``True`` | ``False`` | ``"clipped"``
            Option for this spine
        *vmin*: :class:`float`
            If using clipped spines, minimum value for spine (data space)
        *vmax*: :class:`float`
            If using clipped spines, maximum value for spine (data space)
    :Versions:
        * 2019-03-08 ``@ddalle``: First version
    """
    # Process it
    if opt is None:
        # Leave as is
        pass
    elif opt is True:
        # Turn on
        spine.set_visible(True)
    elif opt is False:
        # Turn off
        spine.set_visible(False)
    elif typeutils.isstr(opt):
        # Check value
        if opt == "on":
            # Same as ``True``
            spine.set_visible(True)
        elif opt == "off":
            # Same as ``False``
            spine.set_visible(False)
        elif opt in ["clip", "clipped", "truncate", "truncated"]:
            # Set limits
            spine.set_bounds(vmin, vmax)
        else:
            raise ValueError("Could not process spine option '%s'" % opt)
    else:
        raise TypeError("Could not process spine option " +
                        ("of type '%s'" % opt.__class__))


# Spine formatting
def format_spines(ax, **kw):
    """Manipulate spines (ticks, extents, etc.)

    :Call:
        >>> h = format_spines(ax, **kw)
    :Inputs:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
    :Effects:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Grid lines added to axes
    :Versions:
        * 2019-03-07 ``@jmeeroff``: First version
    """
   # --- Setup ---
    # Make sure pyplot loaded
    import_pyplot()
    # Get spine handles
    spineL = ax.spines["left"]
    spineR = ax.spines["right"]
    spineT = ax.spines["top"]
    spineB = ax.spines["bottom"]
    # Set default flags to "clipped" if certain other options are present
    # In particular, if manual bounds are specified, set the default spine
    # option to "clipped" unless otherwise specified
    for d in ["X", "Left", "Right", "Y", "Top", "Bottom"]:
        # Spine flag
        k = d + "Spine"
        # Key for min and max spine values
        kmin = k + "Min"
        kmax = k + "Max"
        # Check
        if kmin in kw:
            kw.setdefault(k, "clipped")
        elif kmax in kw:
            kw.setdefault(k, "clipped")
   # --- Data/Spine Bounds ---
    # Get existing data limits
    xmin, xmax = get_xlim(ax, pad=0.0)
    ymin, ymax = get_ylim(ax, pad=0.0)
    # Process manual limits for min and max spines
    xa = kw.pop("XSpineMin", xmin)
    xb = kw.pop("XSpineMax", xmax)
    ya = kw.pop("YSpineMin", ymin)
    yb = kw.pop("YSpineMax", ymax)
    # Process manual limits for individual spines
    yaL = kw.pop("LeftSpineMin", ya)
    ybL = kw.pop("LeftSpineMax", yb)
    yaR = kw.pop("RightSpineMin", ya)
    ybR = kw.pop("RightSpineMax", yb)
    xaB = kw.pop("BottomSpineMin", xa)
    xbB = kw.pop("BottomSpineMax", xb)
    xaT = kw.pop("TopSpineMin", xa)
    xbT = kw.pop("TopSpineMax", xb)
   # --- Overall Spine Options ---
    # Option to turn off all spines
    qs = kw.pop("Spines", None)
    # Only valid options are ``None`` and ``False``
    if qs is not False: qs = None
    # Spine pairs options
    qX = kw.pop("XSpine", qs)
    qY = kw.pop("YSpine", qs)
    # Left spine options
    qL = kw.pop("LeftSpine",   qY)
    qR = kw.pop("RightSpine",  qY)
    qT = kw.pop("TopSpine",    qX)
    qB = kw.pop("BottomSpine", qX)
   # --- Spine On/Off Extents ---
    # Process these options
    format_spine1(spineL, qL, yaL, ybL)
    format_spine1(spineR, qR, yaR, ybR)
    format_spine1(spineT, qT, xaT, xbT)
    format_spine1(spineB, qB, xaB, xbB)
   # --- Spine Formatting ---
    # Paired options
    spopts = kw.pop("SpineOptions", {})
    xsopts = kw.pop("XSpineOptions", {})
    ysopts = kw.pop("YSpineOptions", {})
    # Individual spines
    lsopts = kw.pop("LeftSpineOptions", {})
    rsopts = kw.pop("RightSpineOptions", {})
    bsopts = kw.pop("BottomSpineOptions", {})
    tsopts = kw.pop("TopSpineOptions", {})
    # Combine settings
    xsopts = dict(spopts, **xsopts)
    ysopts = dict(spopts, **ysopts)
    lsopts = dict(ysopts, **lsopts)
    rsopts = dict(ysopts, **rsopts)
    bsopts = dict(xsopts, **bsopts)
    tsopts = dict(xsopts, **tsopts)
    # Apply spine options
    spineL.set(**lsopts)
    spineR.set(**rsopts)
    spineB.set(**bsopts)
    spineT.set(**tsopts)
   # --- Tick Settings ---
    # Option to turn off all ticks
    qt = kw.pop("Ticks", None)
    # Only valid options are ``None`` and ``False``
    if qt is not False: qt = None
    # Options for ticks on each axis
    qtL = kw.pop("LeftSpineTicks",   qt and qL)
    qtR = kw.pop("RightSpineTicks",  qt and qR)
    qtB = kw.pop("BottomSpineTicks", qt and qB)
    qtT = kw.pop("TopSpineTicks",    qt and qT)
    # Turn on/off
    if qtL is not None: ax.tick_params(left=qtL)
    if qtR is not None: ax.tick_params(right=qtR)
    if qtB is not None: ax.tick_params(bottom=qtB)
    if qtT is not None: ax.tick_params(top=qtT)
   # --- Tick label settings ---
    # Option to turn off all tick labels
    qtl = kw.pop("TickLabels", None)
    # Only valid options are ``None`` and ``False``
    if qtl is not False: qtl = None
    # Options for labels on each spine
    qtlL = kw.pop("LeftTickLabels",   qtl)
    qtlR = kw.pop("RightTickLabels",  qtl)
    qtlB = kw.pop("BottomTickLabels", qtl)
    qtlT = kw.pop("TopTickLabels",    qtl)
    # Turn on/off labels
    if qtlL is not None: ax.tick_params(labelleft=qtlL)
    if qtlR is not None: ax.tick_params(labelright=qtlR)
    if qtlB is not None: ax.tick_params(labelbottom=qtlB)
    if qtlT is not None: ax.tick_params(labeltop=qtlT)
   # --- Tick Formatting ---
    # Directions
    tkdir = kw.pop("TickDirection", "out")
    xtdir = kw.pop("XTickDirection", tkdir)
    ytdir = kw.pop("YTickDirection", tkdir)
    # Apply tick directions (can be overridden below)
    ax.tick_params(axis="x", direction=xtdir)
    ax.tick_params(axis="y", direction=ytdir)
    # Universal and paired options
    tkopts = kw.pop("TickOptions", {})
    xtopts = kw.pop("XTickOptions", {})
    ytopts = kw.pop("YTickOptions", {})
    # Inherit options from spines
    tkopts = dict(spopts, **tkopts)
    xtopts = dict(xsopts, **xtopts)
    ytopts = dict(ysopts, **ytopts)
    # Apply
    ax.tick_params(axis="both", **tkopts)
    ax.tick_params(axis="x",    **xtopts)
    ax.tick_params(axis="y",    **ytopts)
   # --- Output ---
    # Return all spine handles
    return {
        "left":   spineL,
        "right":  spineR,
        "top":    spineT,
        "bottom": spineB,
    }


# Convert min/max to error bar widths
def minmax_to_errorbar(yv, ymin, ymax):
    """Convert min/max values to error bar below/above widths

    :Call:
        >>> yerr = minmax_to_errorbar(yv, ymin, ymax)
    :Inputs:
        *yv*: :class:`np.ndarray` shape=(*N*,)
            Nominal dependent axis values
        *ymin*: :class:`float` | :class:`np.ndarray` shape=(*N*,)
            Minimum values of region plot
        *ymax*: :class:`float` | :class:`np.ndarray` shape=(*N*,)
            Maximum values of region plot
    :Outputs:
        *yerr*: :class:`np.ndarray` shape=(2,*N*)
            Array of lower, upper error bar widths
    :Versions:
        * 2019-03-06 ``@ddalle/@jmeeroff``: First version
    """
    # Widths of *ymax* above *yv*
    yu = ymax - np.asarray(yv)
    # Widths of *ymin* below *yv*
    yl = np.asarray(yv) - ymin
    # Separate error bars
    yerr = np.array([yl, yu])
    # Output
    return yerr

# Convert min/max to error bar widths
def errorbar_to_minmax(yv, yerr):
    """Convert min/max values to error bar below/above widths

    :Call:
        >>> ymin, ymax = minmax_to_errorbar(yv, yerr)
    :Inputs:
        *yv*: :class:`np.ndarray` shape=(*N*,)
            Nominal dependent axis values
        *yerr*: :class:`float` | :class:`np.ndarray` shape=(2,*N*) | (*N*,)
            Array of lower, upper error bar widths
    :Outputs:
        *ymin*: :class:`np.ndarray` shape=(*N*,)
            Minimum values of region plot
        *ymax*: :class:`np.ndarray` shape=(*N*,)
            Maximum values of region plot
    :Versions:
        * 2019-03-06 ``@ddalle/@jmeeroff``: First version
    """
    # Ensure arrays
    yv = np.asarray(yv)
    yerr = np.asarray(yerr)
    # Length of *yv*
    N = yv.size
    # Check number of dimensions of *yerr*
    if yerr.ndim == 0:
        # Scalar: simple arithmetic
        ymax = yv + yerr
        ymin = yv - yerr
    elif yerr.ndim == 1:
        # Single array: check size
        if yerr.size != N:
            raise ValueError(
                "Error bar width vector does not match size of main data")
        # Simple arithmetic
        ymax = yv + yerr
        ymin = yv - yerr
    elif yerr.ndim == 2:
        # 2D array: separate below/above widths
        if yerr.shape[0] != 2:
            raise ValueError(
                ("2D error bar width must have shape (2,N), ") +
                ("got %s" % repr(yerr.shape)))
        elif yerr.shape[1] != N:
            raise ValueError(
                "Error bar width vector does not match size of main data")
        # Separate above/below deltas
        ymax = yv + yerr[1]
        ymin = yv - yerr[0]
    else:
        # 3D or more array
        raise ValueError(
            "Cannot convert %i-D array to min/max values" % yerr.ndim)
    # Output
    return ymin, ymax


# Function to automatically get inclusive data limits.
def get_ylim(ax, pad=0.05):
    """Calculate appropriate *y*-limits to include all lines in a plot

    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` and
    :class:`matplotlib.collections.PolyCollection` are checked.

    :Call:
        >>> ymin, ymax = get_ylim(ax, pad=0.05)
    :Inputs:
        *ax*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *pad*: :class:`float`
            Extra padding to min and max values to plot.
    :Outputs:
        *ymin*: :class:`float`
            Minimum *y* coordinate including padding
        *ymax*: :class:`float`
            Maximum *y* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: First version
        * 2019-03-07 ``@ddalle``: Added ``"LineCollection"``
    """
    # Initialize limits.
    ymin = np.inf
    ymax = -np.inf
    # Loop through all children of the input axes.
    for h in ax.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Get the y data for this line
            ydata = h.get_ydata()
            # Check the min and max data
            if len(ydata) > 0:
                ymin = min(ymin, min(h.get_ydata()))
                ymax = max(ymax, max(h.get_ydata()))
        elif t in ['PathCollection', 'PolyCollection', 'LineCollection']:
            # Loop through paths
            for P in h.get_paths():
                # Get the coordinates
                ymin = min(ymin, min(P.vertices[:, 1]))
                ymax = max(ymax, max(P.vertices[:, 1]))
        elif t in ["Rectangle"]:
            # Skip if invisible
            if h.axes is None: continue
            # Get bounding box
            bbox = h.get_bbox().extents
            # Combine limits
            ymin = min(ymin, bbox[1])
            ymax = max(ymax, bbox[3])
    # Check for identical values
    if ymax - ymin <= 0.1*pad:
        # Expand by manual amount
        ymax += pad*abs(ymax)
        ymin -= pad*abs(ymin)
    # Add padding.
    yminv = (1+pad)*ymin - pad*ymax
    ymaxv = (1+pad)*ymax - pad*ymin
    # Output
    return yminv, ymaxv


# Function to automatically get inclusive data limits.
def get_xlim(ax, pad=0.05):
    """Calculate appropriate *x*-limits to include all lines in a plot

    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` are
    checked.

    :Call:
        >>> xmin, xmax = get_xlim(ax, pad=0.05)
    :Inputs:
        *ax*: :class:`matplotlib.axes.AxesSubplot`
            Axis handle
        *pad*: :class:`float`
            Extra padding to min and max values to plot.
    :Outputs:
        *xmin*: :class:`float`
            Minimum *x* coordinate including padding
        *xmax*: :class:`float`
            Maximum *x* coordinate including padding
    :Versions:
        * 2015-07-06 ``@ddalle``: First version
        * 2019-03-07 ``@ddalle``: Added ``"LineCollection"``
    """
    # Initialize limits.
    xmin = np.inf
    xmax = -np.inf
    # Loop through all children of the input axes.
    for h in ax.get_children()[:-1]:
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Get data
            xdata = h.get_xdata()
            # Check the min and max data
            if len(xdata) > 0:
                xmin = min(xmin, np.min(h.get_xdata()))
                xmax = max(xmax, np.max(h.get_xdata()))
        elif t in ['PathCollection', 'PolyCollection', 'LineCollection']:
            # Loop through paths
            for P in h.get_paths():
                # Get the coordinates
                xmin = min(xmin, np.min(P.vertices[:, 0]))
                xmax = max(xmax, np.max(P.vertices[:, 0]))
        elif t in ["Rectangle"]:
            # Skip if invisible
            if h.axes is None: continue
            # Get bounding box
            bbox = h.get_bbox().extents
            # Combine limits
            xmin = min(xmin, bbox[0])
            xmax = max(xmax, bbox[2])
    # Check for identical values
    if xmax - xmin <= 0.1*pad:
        # Expand by manual amount
        xmax += pad*abs(xmax)
        xmin -= pad*abs(xmin)
    # Add padding.
    xminv = (1+pad)*xmin - pad*xmax
    xmaxv = (1+pad)*xmax - pad*xmin
    # Output
    return xminv, xmaxv

