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
    axes_autoscale_height,
    axlabel, auto_xlim, auto_ylim, get_figure,
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


# Contour plotter
def contour(xv, yv, zv, *a, **kw):
    r"""Plot contours many options

    :Call:
        >>> h, kw = plot(xv, yv, zv, *a, **kw)
    :Inputs:
        *xv*: :class:`np.ndarray` (:class:`float`)
            Array of values for *x*-axis
        *yv*: :class:`np.ndarray` (:class:`float`)
            Array of values for *y*-axis
        *zv*: :class:`np.ndarray` (:class:`float`)
            Array of values for contour levels
    :Outputs:
        *h*: :class:`cape.tnakit.plto_mpl.MPLHandle`
            Dictionary of plot handles
    :Versions:
        * 2020-03-26 ``@jmeeroff``: First version
    """
   # --- Prep ---
    # Process options
    opts, h = _preprocess_kwargs(**kw)
    # Save values
    opts.set_option("x", xv)
    opts.set_option("y", yv)
    opts.set_option("z", zv)
   # --- Axes Setup ---
    # Figure, then axes
    _part_init_figure(opts, h)
    _part_init_axes(opts, h)
   # --- Primary Plot ---
    # Plot, then others
    _part_contour(opts, h)
   # --- Axis formatting ---
    # Format grid, spines, extents, and window
    _part_axes_grid(opts, h)
    _part_axes_spines(opts, h)
    _part_axes_format(opts, h)
    _part_axes_adjust(opts, h)
   # --- Labeling ---
    # Colorbar
    _part_colorbar(opts, h)
   # --- Cleanup ---
    # Final margin adjustment
    _part_axes_adjust(opts, h)
    # Output
    return h


# Histogram plotter
def hist(v, *a, **kw):
    r"""Plot histogram with many options

    :Call:
        >>> h, kw = hist(v,*a, **kw)
    :Inputs:
        *v*: :class:`np.ndarray` (:class:`float`)
            Array of values to plot in historgram
    :Outputs:
        *h*: :class:`cape.tnakit.plto_mpl.MPLHandle`
            Dictionary of plot handles
    :Versions:
        * 2020-04-23 ``@jmeeroff``: First version
    """
   # --- Prep ---
    # Process options
    opts, h = _preprocess_kwargs(**kw)
   # --- Statistics ---
    # Filter out non-numeric entries
    v = v[np.logical_not(np.isnan(v))]
    # Calculate the mean
    vmu = np.mean(v)
    # Calculate StdDev
    vstd = np.std(v)
    # Coverage Intervals
    cov = kw.pop("Coverage", kw.pop("cov", 0.99))
    cdf = kw.pop("CoverageCDF", kw.pop("cdf", cov))
    # Nominal bounds (like 3-sigma for 99.5% coverage, etc.)
    kcdf = statutils.student.ppf(0.5+0.5*cdf, v.size)
    # Check for outliers ...
    fstd = kw.pop('FilterSigma', 2.0*kcdf)
    # Remove values from histogram
    if fstd:
        # Find indices of cases that are within outlier range
        j = np.abs(v-vmu)/vstd <= fstd
        # Filter values
        v = v[j]
    # Calculate interval
    acov, bcov = statutils.get_cov_interval(v, cov, cdf=cdf, **kw)
    # Save values
    opts.set_option("v", v)
    # Save stats
    opts.set_option("mu", vmu)
    opts.set_option("std", vstd)
    # Save Interval
    opts.set_option('acov', acov)
    opts.set_option('bcov', bcov)
   # --- Control Options ---
    opts.setdefault_option("ShowGauss", False) 
    opts.setdefault_option("ShowInterval", False) 
    opts.setdefault_option("ShowMean", False) 
   # --- Axes Setup ---
    # Figure, then axes
    _part_init_figure(opts, h)
    _part_init_axes(opts, h)
   # --- Primary Plot ---
    # Histogram, then others
    _part_hist(opts, h)
    # Gaussian
    _part_gauss(opts, h)
    # Interval
    _part_interval(opts, h)
    # Mean
    _part_mean(opts, h)
   # --- Axis formatting ---
    # Format grid, spines, and window
    _part_axes_grid(opts, h)
    _part_axes_spines(opts, h)
    _part_axes_format(opts, h)
    _part_axes_adjust(opts, h)
   # --- Labeling ---
    # --- Mean labels ---
    # Process mean labeling options
    _part_mean_label(opts, h)
    # Readjust axes
    _part_axes_adjust(opts, h)
    # Output
    return h


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


# Partial function: contour()
def _part_contour(opts, h):
    # Call contour method
    # Process contour options
    kw = opts.contour_options()
    # Get values
    xv = opts.get_option("x")
    yv = opts.get_option("y")
    zv = opts.get_option("z")
    # Contour plot call
    contour, lines = mpl._contour(xv, yv, zv, **kw)
    # Save contour and lines
    h.save("contour", contour)
    h.save("lines", lines)


# Partial function: contour()
def _part_hist(opts, h):
    # Call histogram method
    # Process histogram options
    kw = opts.hist_options()
    # Get values
    v = opts.get_option("v")
    # Histogram plot call
    hist = mpl._hist(v, **kw)
    # Save histogram
    h.save("hist", hist)

# Partial function: gaussian()
def _part_gauss(opts, h):
    if opts.get_option("ShowGauss"):
        # Process gaussian options
        kw = opts.gauss_options()
        # Get mu and std
        vmu = opts.get_option('mu')
        vstd = opts.get_option('std')
        # Get orientation
        rotate = opts.get_option('Rotate')
        if rotate:
            orient = "horizontal"
        else:
            orient = "vertical"
        # Number of points to plot
        ngauss = kw.pop("NGauss", 151)
        # Get axis limits
        ax = h.ax
        if orient == "vertical":
            # Get existing horizontal limits
            xmin, xmax = ax.get_xlim()
        else:
            # Get existing
            xmin, xmax = ax.get_ylim()
        # Create points at which to plot
        xval = np.linspace(xmin, xmax, ngauss)
        # Compute Gaussian distribution
        yval = 1/(vstd*np.sqrt(2*np.pi))*np.exp(-0.5*((xval-vmu)/vstd)**2)
        # Plot
        if orient == "vertical":
            # Plot vertical dist with bump pointing to the right
            gauss = mpl._plot(xval, yval, **kw)
        else:
            # Plot horizontal dist with vertical bump
            gauss = mpl._plot(yval, xval, **kw)
        # Save
        h.save("gauss", gauss)

# Partial function: intervale()
def _part_interval(opts, h):
    if opts.get_option("ShowInterval"):
        # Process interval options
        kw = opts.interval_options()
        # Get interval 
        acov = opts.get_option('acov')
        bcov = opts.get_option('bcov')
        # Get orientation
        rotate = opts.get_option('Rotate')
        # Pop out rotate keyword
        r2 = kw.pop('Rotate', None)
        if rotate:
            orient = "horizontal"
        else:
            orient = "vertical"
        # Get axis limits
        ax = h.ax
        if orient == 'vertical':
            # Vertical: get vertical limits of axes window
            pmin, pmax = ax.get_ylim()
            # Plot a vertical range bar
            interval = mpl._fill_between([pmin, pmax], acov, bcov, Rotate=True, **kw)

        else:
            # Horizontal: get horizontal limits of axes window
            pmin, pmax = ax.get_xlim()
            # Plot a horizontal range bar
            interval = mpl._fill_between([pmin, pmax], acov, bcov, **kw)
        # Return
        h.save("interval", interval)

# Partial function: mean()
def _part_mean(opts, h):
    if opts.get_option("ShowMean"):
        # Get mu
        vmu = opts.get_option('mu')
        # Get orientation
        rotate = opts.get_option('Rotate')
        if rotate:
            orient = "horizontal"
        else:
            orient = "vertical"
        # Process mean options
        opts_mean = opts.mean_options()
        # Options for the plot function
        kw = {}
        kw['PlotOptions'] = opts_mean
        # Get axis limits
        ax = h.ax
        if orient == 'vertical':
            # Vertical: get vertical limits of axes window
            pmin, pmax = ax.get_ylim()
            # Plot a vertical mean line
            mean = mpl.plot([vmu, vmu], [pmin, pmax], **kw)
        else:
            # Horizontal: get horizontal limits of axes window
            pmin, pmax = ax.get_xlim()
            # Plot a horizontal range bar
            mean = mpl.plot([pmin, pmax], [vmu, vmu], **kw)
        # Return
        h.save('mean', mean)
    
# Partial function: mean_label()
def _part_mean_label(opts, h):
    if opts.get_option("ShowMean"):
        # Process mean labeling options
        kw = opts.histlabel_options()
        # Get mu
        vmu = opts.get_option('mu')
        # Do the label
        # Formulate the label
        c = 'mu'
        # Format code
        flbl = kw.pop("MuFormat", "%.4f")
        # First part
        klbl = (u'%s' % c)
        # Get axis limits
        ax = h.ax
        # Insert value
        lbl = ('%s = %s' % (klbl, flbl)) % vmu
        meanlabel = histlab_part(lbl, 0.99, True, ax, **kw)
        # Return
        h.save('meanlabel', meanlabel)    


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
        uy = opts.get_option("uy")
        # Plot call
        if uq_type == "FillBetween":
            # Convert to min/max values
            ymin, ymax = errorbar_to_minmax(yv, uy)
            # Do a :func:`fill_between` plot
            hi = mpl._fill_between(xv, ymin, ymax, **kw)
        elif uq_type == "ErrorBar":
            # Do a :func:`errorbar` plot
            hi = mpl._errorbar(xv, yv, uy, **kw)
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


# Partial function: colorbar()
def _part_colorbar(opts, h):
    # Process colorbar options [None yet]
    kw = {}
    # Setup colobar
    cbar = mpl._colorbar(**kw)
    # Save
    h.save("colorbar", cbar)


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
        *h*: ``None`` | :class:`list`| [:class:`matplotlib.Line2D`]
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
