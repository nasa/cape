#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------
:mod:`cape.tnakit.plot_mpl`: Matplotlib/Pyplot Interfaces
--------------------------------------------------------------------

This module contains handles to various :mod:`matplotlib` plotting
methods.  In particular, it centralizes some of the multiple-line
instructions that are used in multiple places.  An example is placing
labels on one or more corners of the axis window.

It also includes syntax to import modules without raising ``ImportError``.
"""

# Standard library modules
import os

# Required third-party modules
import numpy as np

# TNA toolkit modules
import cape.tnakit.kwutils as kwutils
import cape.tnakit.optitem as optitem
import cape.tnakit.rstutils as rstutils
import cape.tnakit.statutils as statutils
import cape.tnakit.typeutils as typeutils

# Local modules
from . import mpl

# Local direct imports
from .mplopts import MPLOpts
from .mpl import (
    axes, axes_adjust, axes_adjust_col, axes_adjust_row, axes_format,
    figure, grid, imshow, spine, spines)


# Preprocess kwargs
def _preprocess_kwargs(**kw):
    r"""Process options and output handle from input options

    :Call:
        >>> opts, h = _preprocess_kwargs(**kw)
    :Inputs:
        *opts*: {``None``} | :class:`MPLOpts`
            Options instance, updated by remaining *kw* items
        *optscls*: {``MPLOpts``} | :class:`type`
            If *opts* is ``None``, this class is used to process options
        *handle*: {``None``} | :class:`MPLHandle`
            Optional preexisting handle to save plot objects to
        *kw*: :class:`dict`
            Other kwargs processed by *optscls* or *opts*
    :Outputs:
        *opts*: :class:`MPLOpts`
            Options with all *kw* checked and applied
        *h*: *handle* | :class:`MPLHandle`
            Container to save Matplotlib plot handles
    :Versions:
        * 2020-01-25 ``@ddalle``: First version
    """
    # Process options
    opts = kw.pop("opts", None)
    # Process options class
    optscls = kw.pop("optscls", MPLOpts)
    # Check if that resulted in anything
    if isinstance(opts, MPLOpts):
        # Blend in any other options
        opts.update(**kw)
    else:
        # Get options class
        opts = optscls(**kw)
    # Get output
    h = kw.pop("handle", None)
    # Initialize output if necessary
    if h is None:
        h = MPLHandle()
    # Save options
    h.opts = opts
    # Output
    return opts, h


# Primary plotter
def plot(xv, yv, *a, **kw):
    r"""Plot connected points with many options

    :Call:
        >>> h, kw = plot(xv, yv, *a, **kw)
    :Inputs:
        *xv*: :class:`np.ndarray` (:class:`float`)
            Array of values for *x*-axis
        *yv*: :class:`np.ndarray` (:class:`float`)
            Array of values for *y*-axis
    :Outputs:
        *h*: :class:`cape.tnakit.plto_mpl.MPLHandle`
            Dictionary of plot handles
    :Versions:
        * 2019-03-01 ``@ddalle``: First (independent) version
        * 2019-12-23 ``@ddalle``: Object-oriented options and output
    """
   # --- Prep ---
    # Process options
    opts, h = _preprocess_kwargs(**kw)
    # Process plot format
    if len(a) == 0:
        # No primary plot specifier
        pass
    elif len(a) == 1:
        # Check one arg for type
        if not typeutils.isstr(a[0]):
            raise TypeError(
                "Extra plot arg must be string (got %s)" % type(a[0]))
        # Use format from user
        opts.set_option("fmt", a[0])
    else:
        # Too many args
        raise TypeError(
            "plot() takes at most 3 args (%i given)" % (len(a) + 2))
    # Save values
    opts.set_option("x", xv)
    opts.set_option("y", yv)
   # --- Control Options ---
    # Defaults to plot different parts
    opts.setdefault_option("ShowLine", True)
    opts.setdefault_option("ShowMinMax", ("ymin" in opts) and ("ymax" in opts))
    opts.setdefault_option("ShowError", ("yerr" in opts))
    opts.setdefault_option("ShowUncertainty", ("uy" in opts))
   # --- Axes Setup ---
    # Figure, then axes
    _part_init_figure(opts, h)
    _part_init_axes(opts, h)
   # --- Primary Plot ---
    # Plot, then others
    _part_plot(opts, h)
    _part_minmax(opts, h)
    _part_error(opts, h)
    _part_uq(opts, h)
   # --- Axis formatting ---
    # Format grid, spines, extents, and window
    _part_axes_grid(opts, h)
    _part_axes_spines(opts, h)
    _part_axes_format(opts, h)
    _part_axes_adjust(opts, h)
   # --- Labeling ---
    # Legend
    _part_legend(opts, h)
   # --- Cleanup ---
    # Final margin adjustment
    _part_axes_adjust(opts, h)
    # Output
    return h


# Partial function: prepare figure
def _part_init_figure(opts, h):
    # Process figure options
    kw_fig = opts.figure_options()
    # Get/create figure
    h.fig = mpl._figure(**kw_fig)


# Partial function: prepare axes
def _part_init_axes(opts, h):
    # Process axis options
    kw_ax = opts.axes_options()
    # Get/create axis
    h.ax = mpl._axes(**kw_ax)


# Partial function: plot()
def _part_plot(opts, h):
    # Call plot method
    if opts.get_option("ShowLine", True):
        # Process plot options
        kw = opts.plot_options()
        # Get values
        xv = opts.get_option("x")
        yv = opts.get_option("y")
        # Get format
        fmt = opts.get_option("PlotFormat")
        # Create format args
        if fmt:
            a = tuple(fmt)
        else:
            a = tuple()
        # Plot call
        lines = mpl._plot(xv, yv, *a, **kw)
        # Save lines
        h.save("lines", lines)


# Partial function: minmax()
def _part_minmax(opts, h):
    # Plot it
    if opts.get_option("ShowMinMax"):
        # Process min/max options
        opts_mmax = opts.minmax_options()
        # Get type
        minmax_type = opts_mmax.get("MinMaxPlotType", "ErrorBar")
        # Options for the plot function
        kw = opts_mmax.get("MinMaxOptions", {})
        # Get values
        xv = opts.get_option("x")
        yv = opts.get_option("y")
        # Min/max values
        ymin = opts.get_option("ymin")
        ymax = opts.get_option("ymax")
        # Plot call
        if minmax_type == "FillBetween":
            # Do a :func:`fill_between` plot
            hi = mpl._fill_between(xv, ymin, ymax, **kw)
        elif minmax_type == "ErrorBar":
            # Convert to error bar widths
            yerr = minmax_to_errorbar(yv, ymin, ymax, **kw)
            # Do a :func:`errorbar` plot
            hi = mpl._errorbar(xv, yv, yerr, **kw)
        # Save result
        h.save("minmax", hi)


# Partial function: error()
def _part_error(opts, h):
    # Plot it
    if opts.get_option("ShowError"):
        # Process "error" options
        opts_error = opts.error_options()
        # Get type
        error_type = opts_error.get("ErrorPlotType", "FillBetween")
        # Get options for plot function
        kw = opts_error.get("ErrorOptions", {})
        # Get values
        xv = opts.get_option("x")
        yv = opts.get_option("y")
        # Error magnitudes
        yerr = kw.get_option("yerr")
        # Plot call
        if error_type == "FillBetween":
            # Convert to min/max values
            ymin, ymax = errorbar_to_minmax(yv, yerr)
            # Do a :func:`fill_between` plot
            hi = mpl._fill_between(xv, ymin, ymax, **kw)
        elif t_err == "ErrorBar":
            # Do a :func:`errorbar` plot
            hi = mpl._errorbar(xv, yv, yerr, **kw)
        # Save error handles
        h.save("error", hi)
    

# Partial function: uq()
def _part_uq(opts, h):
    # Plot it
    if opts.get_option("ShowUncertainty"):
        # Process uncertainty quantification options
        opts_uq = opts.uq_options()
        # Plot type
        uq_type = opts.get("UncertaintyPlotType", "FillBetween")
        # Get options for plot function
        kw = opts_uq.get("UncertaintyOptions", {})
        # Get values
        xv = opts.get_option("x")
        yv = opts.get_option("y")
        # Uncertainty magnitudes
        yerr = opts.get_option("yerr")
        # Plot call
        if uq_type == "FillBetween":
            # Convert to min/max values
            ymin, ymax = errorbar_to_minmax(yv, yerr)
            # Do a :func:`fill_between` plot
            hi = mpl._fill_between(xv, ymin, ymax, **kw)
        elif uq_type == "ErrorBar":
            # Do a :func:`errorbar` plot
            hi = mpl._errorbar(xv, yv, yerr, **kw)
        # Save UQ handles
        h.save("uq", hi)


# Partial function: axes_adjust()
def _part_axes_adjust(opts, h):
    # Process axes_adjust() options
    kw = opts.axadjust_options()
    # Apply margin adjustments
    mpl._axes_adjust(h.fig, ax=h.ax, **kw)


# Partial function: grid()
def _part_axes_grid(opts, h):
    # Process grid lines options
    kw = opts.grid_options()
    # Apply grid lines
    h.grid = mpl._grid(h.ax, **kw)


# Partial function: spines()
def _part_axes_spines(opts, h):
    # Process spines options
    kw = opts.spine_options()
    # Format the spines
    h.spines = mpl._spines(h.ax, **kw)


# Partial function: formatting for axes
def _part_axes_format(opts, h):
    # Process axes format options
    kw = opts.axformat_options()
    # Format the axes
    xl, yl = mpl._axes_format(h.ax, **kw)
    # Save
    h.save("xlabel", xl)
    h.save("ylabel", yl)


# Partial function: legend()
def _part_legend(opts, h):
    # Process legend options
    kw = opts.legend_options()
    # Setup legend
    leg = mpl._legend(h.ax, **kw)
    # Save
    h.save("legend", leg)


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
    mpl._import_pyplot()
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
    h['xlabel'], h['ylabel'] = axes_format(h['ax'], **kw_axfmt)
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
        h['sigma'] = _plots_std(ax, vmu, vstd, **kw_s)
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


# Move axes all the way to one side
def move_axes(ax, loc, margin=0.0):
    r"""Move an axes object's plot region to one side

    :Call:
        >>> move_axes(ax, loc, margin=0.0)
    :Inputs:
        *ax*: :class:`Axes`
            Axes handle
        *loc*: :class:`str` | :class:`int`
            Direction to move axes

                ===========================  ============
                String                       Code
                ===========================  ============
                ``"top"`` | ``"up"``         ``1``
                ``"right"``                  ``2``
                ``"bottom"`` | ``"down"``    ``3``
                ``"left"``                   ``4``
                ===========================  ============

        *margin*: {``0.0``} | :class:`float`
            Margin to leave outside of axes and tick labels
    :Versions:
        * 2020-01-10 ``@ddalle``: First version
    """
    # Import plot modules
    mpl._import_pyplot()
    # Check inputs
    if not isinstance(loc, (int, typeutils.strlike)):
        raise TypeError("Location must be int or str (got %s)" % type(loc))
    elif isinstance(loc, int) and (loc < 1 or loc > 10):
        raise TypeError("Location int must be in [1 .. 4] (got %i)" % loc)
    elif not isinstance(margin, float):
        raise TypeError("Margin must be float (got %s)" % type(margin))
    # Get axes
    if ax is None:
        ax = mpl.plt.gca()
    # Get current position
    xmin, ymin, w, h = ax.get_position().bounds
    # Max positions
    xmax = xmin + w
    ymax = ymin + h
    # Get extents occupied by labels
    wa, ha, wb, hb = mpl.get_axes_label_margins(ax)
    # Filter location
    if loc in [1, "top", "up"]:
        # Get shift directions to top edge
        dx = 0.0
        dy = 1.0 - hb - margin - ymax
    elif loc in [2, "right"]:
        # Get shift direction to right edge
        dx = 1.0 - wb - margin - xmax
        dy = 0.0
    elif loc in [3, "bottom", "down"]:
        # Get shift direction to bottom
        dx = 0.0
        dy = margin + ha - ymin
    elif loc in [4, "left"]:
        # Get shift direction to bottom and right
        dx = margin + wa - xmin
        dy = 0.0 
    else:
        # Unknown string
        raise ValueError("Unknown location string '%s'" % loc)
    # Set new position
    ax.set_position([xmin + dx, ymin + dy, w, h])


# Nudge axes without resizing
def nudge_axes(ax, dx=0.0, dy=0.0):
    r"""Move an axes object's plot region to one side

    :Call:
        >>> nudge_axes(ax, dx=0.0, dy=0.0)
    :Inputs:
        *ax*: :class:`Axes`
            Axes handle
        *dx*: {``0.0``} | :class:`float`
            Figure fraction to move axes to the right
        *dy*: {``0.0``} | :class:`float`
            Figure fraction to move axes upward
    :Versions:
        * 2020-01-10 ``@ddalle``: First version
    """
    # Import plot modules
    mpl._import_pyplot()
    # Check inputs
    if not isinstance(dx, float):
        raise TypeError("dx must be float (got %s)" % type(dx))
    if not isinstance(dy, float):
        raise TypeError("dy must be float (got %s)" % type(dy))
    # Get axes
    if ax is None:
        ax = mpl.plt.gca()
    # Get current position
    xmin, ymin, w, h = ax.get_position().bounds
    # Set new position
    ax.set_position([xmin + dx, ymin + dy, w, h])


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
    mpl._import_pyplot()
    # Call plot
    h = mpl.plt.hist(v, **kw)
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
    mpl._import_pyplot()
    # Get horizontal/vertical option
    orient = kw.pop('orientation', "")
    # Check orientation
    if orient == 'vertical':
        # Vertical: get vertical limits of axes window
        pmin, pmax = ax.get_ylim()
        # Plot a vertical mean line
        h = mpl.plt.plot([vmu, vmu], [pmin, pmax], **kw)

    else:
        # Horizontal: get horizontal limits of axes window
        pmin, pmax = ax.get_xlim()
        # Plot a horizontal range bar
        h = mpl.plt.plot([pmin, pmax], [vmu, vmu], **kw)
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
    mpl._import_pyplot()
    # Get horizontal/vertical option
    orient = kw.pop('orientation', "")
    # Check orientation
    if orient == 'vertical':
        # Vertical: get vertical limits of axes window
        pmin, pmax = ax.get_ylim()
        # Plot a vertical range bar
        h = mpl.plt.fill_betweenx([pmin, pmax], vmin, vmax, **kw)

    else:
        # Horizontal: get horizontal limits of axes window
        pmin, pmax = ax.get_xlim()
        # Plot a horizontal range bar
        h = mpl.plt.fill_between([pmin, pmax], vmin, vmax, **kw)
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
    mpl._import_pyplot()
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
        h = mpl.plt.plot(xval, yval, **kw)
    else:
        # Plot horizontal dist with vertical bump
        h = mpl.plt.plot(yval, xval, **kw)
    # Return
    return h[0]


# Interval using two lines
def _plots_std(ax, vmu, vstd, **kw):
    """Use two lines to show standard deviation window

    :Call:
        >>> h = _plots_std(ax, vmu, vstd, **kw))
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
    mpl._import_pyplot()
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
            mpl.plt.plot([vmin, vmin], [pmin, pmax], **kw) +
            mpl.plt.plot([vmax, vmax], [pmin, pmax], **kw))
    else:
        # Get horizontal limits
        pmin, pmax = ax.get_xlim()
        # Plot a horizontal line for the min and max
        h = (
            mpl.plt.plot([pmin, pmax], [vmin, vmin], **kw) +
            mpl.plt.plot([pmin, pmax], [vmax, vmax], **kw))
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
    mpl._import_pyplot()
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
            mpl.plt.plot([cmin, cmin], [pmin, pmax], **kw) +
            mpl.plt.plot([cmax, cmax], [pmin, pmax], **kw))
    else:
        pmin, pmax = ax.get_xlim()
        # Plot a horizontal line for the min and max
        h = (
            mpl.plt.plot([pmin, pmax], [cmin, cmin], **kw) +
            mpl.plt.plot([pmin, pmax], [cmax, cmax], **kw))
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
        *pos1*: :class: `float`
            label locations (left to right alignment)
        *pos2*: :class: `boolean'
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
    mpl._import_pyplot()
    # Remove orientation orientation
    kw.pop('orientation', None)
    # get figure handdle
    f = mpl.plt.gcf()
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
    h = mpl.plt.text(pos1, y, lbl, transform=ax.transAxes, **kw)
    return h


# Convert min/max to error bar widths
def minmax_to_errorbar(yv, ymin, ymax):
    r"""Convert min/max values to error bar below/above widths

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
    r"""Convert min/max values to error bar below/above widths

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
    r"""Calculate appropriate *y*-limits to include all lines in a plot

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
    return mpl.get_ylim()


# Function to automatically get inclusive data limits.
def get_xlim(ax, pad=0.05):
    r"""Calculate appropriate *x*-limits to include all lines in a plot

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
    return mpl.get_xlim()


# Output class
class MPLHandle(object):
    r"""Container for handles from :mod:`matplotlib.pyplot`

    :Versions:
        * 2019-12-20 ``@ddalle``: First version
    """
    # Initialization method
    def __init__(self, **kw):
        r"""Initialization method

        :Call:
            >>> h = MPLHandle(**kw)
        :Attributes:
            *h.fig*: {``None``} | :class:`matplotlib.figure.Figure`
                Figure handle
            *h.ax*: {``None``} | :class:`matplotlib.axes.Axes`
                Axes handle
            *h.lines*: {``[]``} | :class:`list`\ [:class:`Line2D`]
                List of line handles
        :Versions:
            * 2019-12-20 ``@ddalle``: First version
        """
        # Initialize simple handles
        self.fig = kw.get("fig")
        self.ax = kw.get("ax")

        # Initialize handles
        self.lines = kw.get("lines", [])

    # Combine two handles
    def add(self, h1):
        r"""Combine two plot handle objects

        Attributes of *h1* will override those of *h* except where the
        attribute is a list.  For example, *h1.lines* will be added to
        *h.lines* to get handles to lines from both instances.

        :Call:
            >>> h.add(h1)
        :Inputs:
            *h*: :class:`MPLHandle`
                Parent handle
            *h1*: :class:`MPLHandle`
                Second handle for collecting all objects
        :Versions:
            * 2019-12-26 ``@ddalle``: First version
        """
        # Check inputs
        if not isinstance(h1, MPLHandle):
            raise TypeError("Second handle must be MPLHandle object")
        # Loop through data attributes
        for (k, v) in h1.__dict__.items():
            # Save using dedicated method
            self.save(k, v)

    # Save one attribute
    def save(self, attr, v):
        r"""Save one plot object without erasing others

        This method provides a simple method to save multiple instances
        of a certain kind of plot object without having to perform
        checks manually.  For example if *h.lines* already contains
        several lines, and *L* contains several new ones, then
        ``h.save("lines", L)`` will effectively run ``h.lines += L``.

        Conversions from single object to :class:`list` are handled
        automatically, and items are only added to the list if not
        already present.  Thus running :func:`save` several times in a
        row with the same arguments will not create the illusion of many
        objects for a given attribute.

        :Call:
            >>> h.save(attr, v)
        :Inputs:
            *h*: :class:`MPLHandle`
                Matplotlib object handle
            *attr*: :class:`str`
                Name of attribute to save
            *v*: :class:`any`
                Value to save/add for that attribute
        :Versions:
            * 2020-01-25 ``@ddalle``: First version
        """
        # Get current value
        v0 = self.__dict__.get(attr)
        # Check for case with no current value
        if v0 is None:
            # Save as given
            self.__dict__[attr] = v
            # No more actions
            return
        # Special attributes that should not be a list
        if attr in {"ax", "fig", "name"}:
            # Only one value allowed for these
            self.__dict__[attr] = v
            return
        # Otherwise process list combinations
        if isinstance(v0, list):
            # Check type of new value
            if not isinstance(v, list):
                # Check if *v* is already present
                if v not in v0:
                    v0.append(v)
                return
            # Otherwise loop through *v* entries
            for vi in v:
                # Check if *vi* is already present
                if vi not in v0:
                    v0.append(vi)
        else:
            # Check type of new value
            if not isinstance(v, list):
                # Check if *v* and *v0* are the same
                if v is not v0:
                    self.__dict__[attr] = [v0, v]
                return
            # Otherwise convert current value to list
            v0 = [v0]
            self.__dict__[attr] = v0
            # Loop through *v* entries
            for vi in v:
                # Check if *vi* is already present
                if vi not in v0:
                    v0.append(vi)
        
