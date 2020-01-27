#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------
:mod:`cape.tnakit.plot_mpl.mpl`: Direct PyPlot Interface
--------------------------------------------------------------------

This module contains handles to various :mod:`matplotlib` plotting
methods.  It contains the direct calls to functions like
:func:`plt.plot`, both with options checks (:func:`mpl.plot`) and
without checks (:func:`mpl._plot`).

It also includes syntax to import modules without raising
``ImportError``.
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
from .mplopts import MPLOpts

# Get a variable to hold the "type" of "module"
mod = os.__class__

# Initialize handle for modules
plt = object()
mpl = object()
mplax = object()
mplfig = object()

# Specific types
Axes = object()
Figure = object()


# Import :mod:`matplotlib`
def _import_matplotlib():
    """Function to import Matplotlib if possible

    This function checks if the global variable *mpl* is already a
    module.  If so, the function exits without doing anything.
    Otherwise it imports :mod:`matplotlib` as *mpl*.  If the operating
    system is not Windows, and there is no environment variable
    *DISPLAY*, the backend is set to ``"Agg"``.

    :Call:
        >>> _import_matplotlib()
    :Versions:
        * 2019-08-22 ``@ddalle``: Documented first version
    """
    # Make global variables
    global mpl
    global mplax
    global mplfig
    global Axes
    global Figure
    # Exit if already imported
    if isinstance(mpl, mod):
        return
    # Import module
    try:
        import matplotlib as mpl
        import matplotlib.axes as mplax
        import matplotlib.figure as mplfig
        # Access types
        Axes = mplax._subplots.Axes
        Figure = mplfig.Figure
    except ImportError:
        return
    # Check for no-display
    if (os.name != "nt") and (os.environ.get("DISPLAY") is None):
        # Not on Windows and no display: no window to create fig
        mpl.use("Agg")


# Import :mod:`matplotlib`
def _import_pyplot():
    """Function to import Matplotlib's PyPlot if possible

    This function checks if the global variable *plt* is already a
    module.  If so, the function exits without doing anything.
    Otherwise it imports :mod:`matplotlib.pyplot` as *plt* after
    calling :func:`import_matplotlib`.

    :Call:
        >>> _import_pyplot()
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
    _import_matplotlib()
    # Import module
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return


# Figure part
def figure(**kw):
    r"""Get or create figure handle and format it

    :Call:
        >>> fig = figure(**kw)
    :Inputs:
        *fig*: {``None``} | :class:`matplotlib.figure.Figure`
            Optional figure handle
        *FigOptions*: {``None``} | :class:`dict`
            Options to apply to figure handle using :func:`fig.set`
    :Outputs:
        *fig*: :class:`matplotlib.figure.Figure`
            Figure handle
    :Versions:
        * 2019-03-06 ``@ddalle``: First version
        * 2020-01-24 ``@ddalle``: Added options checks
    """
    # Process options
    opts = MPLOpts(_section="figure", **kw)
    # Get figure options
    kw_fig = opts.figure_options()
    # Call root function
    return _figure(**kw_fig)


# Axis part (initial)
def axes(**kw):
    r"""Create new axes or edit one if necessary

    :Call:
        >>> ax = axes(**kw)
    :Inputs:
        *ax*: ``None`` | :class:`AxesSubplot`
            Optional axes handle
        *AxesOptions*: {``None``} | :class:`dict`
            Options to apply to figure handle using :func:`ax.set`
    :Outputs:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
    :Versions:
        * 2019-03-06 ``@ddalle``: First version
        * 2020-01-24 ``@ddalle``: Moved to :mod:`plot_mpl.mpl`
    """
    # Process options
    opts = MPLOpts(_section="axes", **kw)
    # Get figure options
    kw_ax = opts.axes_options()
    # Call root function
    return _axes(**kw_ax)


# Plot function with options check
def plot(xv, yv, fmt=None, **kw):
    r"""Call the :func:`plot` function with cycling options

    :Call:
        >>> h = plot(xv, yv, **kw)
        >>> h = plot(xv, yv, fmt, **kw)
    :Inputs:
        *xv*: :class:`np.ndarray`
            Array of *x*-coordinates
        *yv*: :class:`np.ndarray`
            Array of *y*-coordinates
        *fmt*: :class:`str`
            Optional format option
        *i*, *Index*: {``0``} | :class:`int`
            Phase number to cycle through plot options
        *rotate*, *Rotate*: ``True`` | {``False``}
            Plot independent variable on vertical axis
    :Keyword Arguments:
        * See :func:`matplotlib.pyplot.plot`
    :Outputs:
        *h*: :class:`list` (:class:`matplotlib.lines.Line2D`)
            List of line instances
    :Versions:
        * 2019-03-04 ``@ddalle``: First version
        * 2020-01-24 ``@ddalle``: Moved to :mod:`plot_mpl.mpl`
    """
    # Process options
    opts = MPLOpts(_section="plot", **kw)
    # Get plot options
    kw_p = opts.plot_options()
    # Call root function
    return _plot(xv, yv, fmt=fmt, **kw_p)


# Error bar plot
def errorbar(xv, yv, yerr=None, xerr=None, **kw):
    r"""Call the :func:`errorbar` function with options checks

    :Call:
        >>> h = errorbar(xv, yv, yerr=None, xerr=None, **kw)
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
        * 2020-01-24 ``@ddalle``: Moved to :mod:`plot_mpl.mpl`
    """
    # Process options
    opts = MPLOpts(_section="errorbar", **kw)
    # Get plot options
    kw_eb = opts.errorbar_options()
    # Call root function
    return _errorbar(xv, yv, yerr, xerr, **kw_eb)


# Region plot
def fill_between(xv, ymin, ymax, **kw):
    r"""Call the :func:`fill_between` or :func:`fill_betweenx` function

    :Call:
        >>> h = _fill_between(xv, ymin, ymax, **kw)
    :Inputs:
        *xv*: :class:`np.ndarray`
            Array of independent variable values
        *ymin*: :class:`np.ndarray` | :class:`float`
            Array of values or single value for lower bound of window
        *ymax*: :class:`np.ndarray` | :class:`float`
            Array of values or single value for upper bound of window
        *Index*: {``0``} | :class:`int`
            Phase number to cycle through plot options
        *Rotate*: ``True`` | {``False``}
            Option to plot independent variable on vertical axis
    :Versions:
        * 2019-03-04 ``@ddalle``: First version
        * 2019-08-22 ``@ddalle``: Renamed from :func:`fillbetween`
        * 2020-01-24 ``@ddalle``: Moved to :mod:`plot_mpl.mpl`
    """
    # Process options
    opts = MPLOpts(_section="fillbetween", **kw)
    # Get plot options
    kw_fb = opts.fillbetween_options()
    # Call root function
    return _fill_between(xv, ymin, ymax, **kw_fb)


# Manage single subplot extents
def axes_adjust(fig=None, **kw):
    r"""Manage margins of one axes handle

    This function provides two methods for adjusting the margins of an
    axes handle.  The first is to automatically detect all the space
    taken up outside of the plot region by both tick labels and axes
    labels and then expand the plot to fill the figure up to amounts
    specified by the *Margin* parameters.  The second is to directly
    specify the figure coordinates of the figure edges using the
    *Adjust* parameters, which override any *Margin* specifications. It
    is possible, however, to use *Adjust* for the top and *Margin* for
    the bottom, for example.

    :Call:
        >>> ax = axes_adjust(fig=None, **kw)
    :Inputs:
        *fig*: {``None``} | :class:`Figure` | :class:`int`
            Figure handle or number (default from :func:`plt.gcf`)
        *ax*: {``None``} | :class:`AxesSubplot`
            Axes handle, if specified, *Subplot* is ignored
        *Subplot*: {``None``} | :class:`int` > 0
            Subplot index; if ``None``, use :func:`plt.gca`; adds a
            new subplot if *Subplot* is greater than the number of
            existing subplots in *fig* (1-based index)
        *SubplotRows*: {*Subplot*} | :class:`int` > 0
            Number of subplot rows if creating new subplot
        *SubplotCols*: {*Subplot*} | :class:`int` > 0
            Number of subplot columns if creating new subplot
        *MarginBottom*: {``0.02``} | :class:`float`
            Figure fraction from bottom edge to bottom label
        *MarginLeft*: {``0.02``} | :class:`float`
            Figure fraction from left edge to left-most label
        *MarginRight*: {``0.015``} | :class:`float`
            Figure fraction from right edge to right-most label
        *MarginTop*: {``0.015``} | :class:`float`
            Figure fraction from top edge to top-most label
        *AdjustBottom*: ``None`` | :class:`float`
            Figure coordinate for bottom edge of axes
        *AdjustLeft*: ``None`` | :class:`float`
            Figure coordinate for left edge of axes
        *AdjustRight*: ``None`` | :class:`float`
            Figure coordinate for right edge of axes
        *AdjustTop*: ``None`` | :class:`float`
            Figure coordinate for top edge of axes
        *KeepAspect*: {``None``} | ``True`` | ``False``
            Keep aspect ratio; default is ``True`` unless
            ``ax.get_aspect()`` is ``"auto"``
    :Outputs:
        *ax*: :class:`AxesSubplot`
            Handle to subplot directed to use from these options
    :Versions:
        * 2020-01-03 ``@ddalle``: First version
        * 2010-01-10 ``@ddalle``: Add support for ``"equal"`` aspect
    """
    # Get options
    opts = MPLOpts(_section="axformat", **kw)
    # Call root function
    return _axes_adjust(fig, **opts)


# Axes format
def axes_format(ax, **kw):
    r"""Format and label axes

    :Call:
        >>> xl, yl = axes_format(ax, **kw)
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
        * 2020-01-08 ``@ddalle``: 2.0, removed margin adjustment
        * 2020-01-08 ``@ddalle``: 2.1, from :func:`axes_format`
        * 2020-01-27 ``@ddalle``: Options checks
    """
    # Get options
    opts = MPLOpts(_section="axformat", **kw)
    # Call root function
    return _axes_format(ax, **opts)


# Grid
def grid(ax, **kw):
    r"""Add grid lines to an axis and format them

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
        * 2019-12-23 ``@ddalle``: Updated from :mod:`plotutils`
        * 2020-01-27 ``@ddalle``: Options checker
    """
    # Process options
    opts = MPLOpts(_section="grid", **kw)
    # Call root function
    return _grid(ax, **kw)


# Single spine: extents
def spine(spine, opt, vmin, vmax):
    r"""Apply visibility options to a single spine

    :Call:
        >>> spine(spine, opt, vmin, vmax)
    :Inputs:
        *spine*: :clas:`matplotlib.spines.Spine`
            A single spine (left, right, top, or bottom)
        *opt*: ``None`` | ``True`` | ``False`` | ``"clipped"``
            Option for this spine
        *vmin*: :class:`float`
            If using clipped spines, minimum value for spine
        *vmax*: :class:`float`
            If using clipped spines, maximum value for spine
    :Versions:
        * 2019-03-08 ``@ddalle``: First version
        * 2020-01027 ``@ddalle``: From :func:`plot_mpl.format_spine1`
    """
    _spine(spine, opt, vmin, vmax)


# Spine formatting
def spines(ax, **kw):
    r"""Format Matplotlib axes spines and ticks

    :Call:
        >>> h = spines(ax, **kw)
    :Inputs:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
    :Effects:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Grid lines added to axes
    :Versions:
        * 2019-03-07 ``@jmeeroff``: First version
        * 2019-12-23 ``@ddalle``: From :mod:`tnakit.plotutils`
        * 2020-01-27 ``@ddalle``: From :func:`plot_mpl.format_spines`
    """
    # Get options
    opts = MPLOpts(_section="spines", **kw)
    # Call root function
    return _spines(ax, **opts)


# Axis part (initial)
def _axes(**kw):
    r"""Create new axes or edit one if necessary

    :Call:
        >>> ax = axes(**kw)
    :Inputs:
        *ax*: ``None`` | :class:`AxesSubplot`
            Optional axes handle
        *AxesOptions*: {``None``} | :class:`dict`
            Options to apply to figure handle using :func:`ax.set`
    :Outputs:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
    :Versions:
        * 2019-03-06 ``@ddalle``: First version
        * 2020-01-24 ``@ddalle``: Moved to :mod:`plot_mpl.mpl`
    """
    # Import PyPlot
    _import_pyplot()
    # Get figure handle and other options
    ax = kw.get("ax", None)
    axopts = kw.get("AxesOptions", {})
    # Check for a subplot description
    axnum = axopts.pop("subplot", None)
    # Create figure if needed
    if not isinstance(ax, Axes):
        # Check for specified figure
        if axnum is None:
            # Use most recent figure (can be new one)
            ax = plt.gca()
        else:
            # Get specified figure
            ax = plt.subplot(axnum)
    # Loop through options
    for (k, v) in axopts.items():
        # Check for None
        if not v:
            continue
        # Get setter function
        fn = getattr(ax, "set_" + k, None)
        # Check property
        if fn is None:
            sys.stderr.write("No axes property '%s'\n" % k)
            sys.stderr.flush()
        # Apply setter
        fn(v)
    # Output
    return ax


# Manage single subplot extents
def _axes_adjust(fig=None, **kw):
    r"""Manage margins of one axes handle

    This function provides two methods for adjusting the margins of an
    axes handle.  The first is to automatically detect all the space
    taken up outside of the plot region by both tick labels and axes
    labels and then expand the plot to fill the figure up to amounts
    specified by the *Margin* parameters.  The second is to directly
    specify the figure coordinates of the figure edges using the
    *Adjust* parameters, which override any *Margin* specifications. It
    is possible, however, to use *Adjust* for the top and *Margin* for
    the bottom, for example.

    :Call:
        >>> ax = axes_adjust(fig=None, **kw)
    :Inputs:
        *fig*: {``None``} | :class:`Figure` | :class:`int`
            Figure handle or number (default from :func:`plt.gcf`)
        *ax*: {``None``} | :class:`AxesSubplot`
            Axes handle, if specified, *Subplot* is ignored
        *Subplot*: {``None``} | :class:`int` > 0
            Subplot index; if ``None``, use :func:`plt.gca`; adds a
            new subplot if *Subplot* is greater than the number of
            existing subplots in *fig* (1-based index)
        *SubplotRows*: {*Subplot*} | :class:`int` > 0
            Number of subplot rows if creating new subplot
        *SubplotCols*: {*Subplot*} | :class:`int` > 0
            Number of subplot columns if creating new subplot
        *MarginBottom*: {``0.02``} | :class:`float`
            Figure fraction from bottom edge to bottom label
        *MarginLeft*: {``0.02``} | :class:`float`
            Figure fraction from left edge to left-most label
        *MarginRight*: {``0.015``} | :class:`float`
            Figure fraction from right edge to right-most label
        *MarginTop*: {``0.015``} | :class:`float`
            Figure fraction from top edge to top-most label
        *AdjustBottom*: ``None`` | :class:`float`
            Figure coordinate for bottom edge of axes
        *AdjustLeft*: ``None`` | :class:`float`
            Figure coordinate for left edge of axes
        *AdjustRight*: ``None`` | :class:`float`
            Figure coordinate for right edge of axes
        *AdjustTop*: ``None`` | :class:`float`
            Figure coordinate for top edge of axes
        *KeepAspect*: {``None``} | ``True`` | ``False``
            Keep aspect ratio; default is ``True`` unless
            ``ax.get_aspect()`` is ``"auto"``
    :Outputs:
        *ax*: :class:`AxesSubplot`
            Handle to subplot directed to use from these options
    :Versions:
        * 2020-01-03 ``@ddalle``: First version
        * 2010-01-10 ``@ddalle``: Add support for ``"equal"`` aspect
    """
    # Make sure pyplot is present
    _import_pyplot()
    # Default figure
    if fig is None:
        # Get most recent figure or create
        fig = plt.gcf()
    elif isinstance(fig, int):
        # Get figure handle from number
        fig = plt.figure(fig)
    elif not isinstance(fig, Figure):
        # Not a figure or number
        raise TypeError(
            "'fig' arg expected 'int' or 'Figure' (got %s)" % type(fig))
    # Get axes from figure
    ax_list = fig.get_axes()
    # Minimum number of axes
    nmin_ax = min(1, len(ax_list))
    # Get "axes" option
    ax = kw.get("ax")
    # Get subplot number options
    subplot_i = kw.get("Subplot")
    subplot_m = kw.get("SubplotRows", nmin_ax)
    subplot_n = kw.get("SubplotCols", nmin_ax // subplot_m)
    # Check for axes
    if ax is None:
        # Check for index
        if subplot_i is None:
            # Get most recent axes
            ax = plt.gca()
            # Reset axes list
            ax_list = fig.get_axes()
            # Get index
            subplot_i = ax_list.index(ax) + 1
        elif not isinstance(subplot_i, int):
            # Must be an integer
            raise TypeError(
                "'Subplot' keyword must be 'int' (got %s)" % type(subplot_i))
        elif subplot_i <= 0:
            # Must be *positive*
            raise ValueError(
                "'Subplot' index must be positive (1-based) (got %i)" %
                subplot_i)
        elif subplot_i > len(ax_list):
            # Create new subplot
            ax = fig.add_subplot(subplot_m, subplot_n, subplot_i)
        else:
            # Get existing subplot (1-based indexing for consistency)
            ax = ax_list[subplot_i - 1]
    elif ax not in ax_list:
        # Axes from different figure!
        raise ValueError("Axes handle 'ax' is not in current figure")
    else:
        # Get subplot index (1-based index)
        subplot_i = ax_list.index(ax) + 1
    # Figure out subplot column and row index from counts (0-based)
    subplot_j = (subplot_i - 1) // subplot_n
    subplot_k = (subplot_i - 1) % subplot_n
    # Get sizes of all tick and axes labels
    labelw_l, labelh_b, labelw_r, labelh_t = get_axes_label_margins(ax)
    # Process width and height
    ax_w = 1.0 - labelw_r - labelw_l
    ax_h = 1.0 - labelh_t - labelh_b
    # Process row and column space available
    ax_rowh = ax_h / float(subplot_m)
    ax_colw = ax_w / float(subplot_n)
    # Default margins (no tight_layout yet)
    adj_b = labelh_b + subplot_j * ax_rowh
    adj_l = labelw_l + subplot_k * ax_colw
    adj_r = adj_l + ax_colw
    adj_t = adj_b + ax_rowh
    # Get extra margins
    margin_b = kw.get("MarginBottom", 0.02)
    margin_l = kw.get("MarginLeft", 0.02)
    margin_r = kw.get("MarginRight", 0.015)
    margin_t = kw.get("MarginTop", 0.015)
    # Apply to minimum margins
    adj_b += margin_b
    adj_l += margin_l
    adj_r -= margin_r
    adj_t -= margin_t
    # Get user options
    adj_b = kw.get("AdjustBottom", adj_b)
    adj_l = kw.get("AdjustLeft", adj_l)
    adj_r = kw.get("AdjustRight", adj_r)
    adj_t = kw.get("AdjustTop", adj_t)
    # Get current position of axes
    x0, y0, w0, h0 = ax.get_position().bounds
    # Keep same bottom edge if not specified
    if adj_b is None:
        adj_b = y0
    # Keep same left edge if not specified
    if adj_l is None:
        adj_l = x0
    # Keep same right edge if not specified
    if adj_r is None:
        adj_r = x0 + w0
    # Keep same top edge if not specified
    if adj_t is None:
        adj_t = y0 + h0
    # Aspect ratio option
    keep_ar = kw.get("KeepAspect")
    # Default aspect ratio option
    if keep_ar is None:
        # True unless current aspect is "equal" (which is usual case)
        keep_ar = ax.get_aspect() != "auto"
    # Turn off axis("equal") option if necessary
    if (not keep_ar) and (ax.get_aspect() != "auto"):
        # Can only adjust aspect ratio if this is off
        ax.set_aspect("auto")
    # Process aspect ratio
    if keep_ar:
        # Get the width and height of adjusted figure w/ cur margins
        w1 = adj_r - adj_l
        h1 = adj_t - adj_b
        # Currently expected expansion ratios
        rw = w1 / w0
        rh = h1 / h0
        # We can only use the smaller expansion
        if rw > rh:
            # Get current horizontal center
            xc = 0.5 * (adj_l + adj_r)
            # Reduce the horizontal expansion
            w1 = w0 * rh
            # New edge locations
            adj_l = xc - 0.5*w1
            adj_r = xc + 0.5*w1
        elif rh > rw:
            # Get current vertical center
            yc = 0.5 * (adj_b + adj_t)
            # Reduce vertical expansion
            h1 = h0 * rw
            # New edge locations
            adj_b = yc - 0.5*h1
            adj_t = yc + 0.5*h1
    # Set new position
    ax.set_position([adj_l, adj_b, adj_r-adj_l, adj_t-adj_b])
    # Output
    return ax


# Axes format
def _axes_format(ax, **kw):
    r"""Format and label axes

    :Call:
        >>> xl, yl = _axes_format(ax, **kw)
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
        * 2020-01-08 ``@ddalle``: 2.0, removed margin adjustment
        * 2020-01-08 ``@ddalle``: 2.1, from :func:`axes_format`
        * 2020-01-27 ``@ddalle``: 2.2, from :mod:`plot_mpl`
    """
   # --- Prep ---
    # Make sure pyplot loaded
    _import_pyplot()
   # --- Labels ---
    # Get user-specified axis labels
    xlbl = kw.get("XLabel", None)
    ylbl = kw.get("YLabel", None)
    # Check for rotation kw
    rot = kw.get("Rotate", False)
    # Switch args if needed
    if rot:
        # Flip labels
        xlbl, ylbl = ylbl, xlbl
    # Check for orientation kw (histogram)
    orient = kw.get('orientation', None)
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
    pad = kw.get("Pad", 0.05)
    # Specific pad parameters
    xpad = kw.get("XPad", pad)
    ypad = kw.get("YPad", pad)
    # Get limits that include all data (and not extra).
    xmin, xmax = get_xlim(ax, pad=xpad)
    ymin, ymax = get_ylim(ax, pad=ypad)
    # Check for specified limits
    xmin = kw.get("XLimMin", xmin)
    xmax = kw.get("XLimMax", xmax)
    ymin = kw.get("YLimMin", ymin)
    ymax = kw.get("YLimMax", ymax)
    # Check for typles
    xmin, xmax = kw.get("XLim", (xmin, xmax))
    ymin, ymax = kw.get("YLim", (ymin, ymax))
    # Make sure data is included.
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
   # --- Cleanup ---
    # Output
    return xl, yl


# Error bar plot
def _errorbar(xv, yv, yerr=None, xerr=None, **kw):
    r"""Call the :func:`errorbar` function

    :Call:
        >>> h = _errorbar(xv, yv, yerr=None, xerr=None, **kw)
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
        * 2020-01-24 ``@ddalle``: Moved to :mod:`plot_mpl.mpl`
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
    _import_pyplot()
    # Get index
    i = kw.pop("Index", kw.pop("i", 0))
    # Get rotation option
    r = kw.pop("Rotate", False)
    # Check for vertical or horizontal
    if r:
        # Vertical function
        xv, yv, xerr, yerr = yv, xv, yerr, xerr
    # Initialize plot options
    kw_eb = MPLOpts.select_phase(kw, i)
    # Call the plot method
    h = plt.errorbar(xv, yv, yerr=uy, xerr=ux, **kw_eb)
    # Output
    return h


# Figure part
def _figure(**kw):
    r"""Get or create figure handle and format it

    :Call:
        >>> fig = _figure(**kw)
    :Inputs:
        *fig*: {``None``} | :class:`matplotlib.figure.Figure`
            Optional figure handle
        *FigOptions*: {``None``} | :class:`dict`
            Options to apply to figure handle using :func:`fig.set`
    :Outputs:
        *fig*: :class:`matplotlib.figure.Figure`
            Figure handle
    :Versions:
        * 2019-03-06 ``@ddalle``: First version
    """
    # Import PyPlot
    _import_pyplot()
    # Get figure handle and other options
    fig = kw.get("fig", None)
    figopts = kw.get("FigOptions", {})
    # Check for a figure number
    fignum = figopts.pop("num", None)
    # Create figure if needed
    if not isinstance(fig, Figure):
        # Check for specified figure
        if fignum is None:
            # Use most recent figure (can be new one)
            fig = plt.gcf()
        else:
            # Get specified figure
            fig = plt.figure(fignum)
    # Loop through options
    for (k, v) in figopts.items():
        # Check for None
        if not v:
            continue
        # Get setter function
        fn = getattr(fig, "set_" + k, None)
        # Check property
        if fn is None:
            sys.stderr.write("No figure property '%s'\n" % k)
            sys.stderr.flush()
        # Apply setter
        fn(v)
    # Output
    return fig


# Region plot
def _fill_between(xv, ymin, ymax, **kw):
    r"""Call the :func:`fill_between` or :func:`fill_betweenx` function

    :Call:
        >>> h = _fill_between_nocheck(xv, ymin, ymax, **kw)
    :Inputs:
        *xv*: :class:`np.ndarray`
            Array of independent variable values
        *ymin*: :class:`np.ndarray` | :class:`float`
            Array of values or single value for lower bound of window
        *ymax*: :class:`np.ndarray` | :class:`float`
            Array of values or single value for upper bound of window
        *Index*: {``0``} | :class:`int`
            Phase number to cycle through plot options
        *Rotate*: ``True`` | {``False``}
            Option to plot independent variable on vertical axis
    :Versions:
        * 2019-03-04 ``@ddalle``: First version
        * 2019-08-22 ``@ddalle``: Renamed from :func:`fillbetween`
        * 2020-01-24 ``@ddalle``: Renamed from :func:`fill_between`
    """
   # --- Values ---
    # Convert possible scalar for *ymin*
    if isinstance(ymin, float):
        # Expand scalar to array
        yl = ymin * np.ones_like(xv)
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
        yu = ymax * np.ones_like(xv)
    elif typeutils.isarray(ymax):
        # Ensure array (not list, etc.)
        yu = np.asarray(ymax)
    else:
        raise TypeError(
            ("'ymax' value must be either float or array ") +
            ("(got %s)" % ymax.__class__))
   # --- Main ---
    # Ensure fill_between() is available
    _import_pyplot()
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
    kw_fb = MPLOpts.select_phase(kw, i)
    # Call the plot method
    h = fnplt(xv, yl, yu, **kw_fb)
    # Output
    return h


# Creation and formatting of grid lines
def _grid(ax, **kw):
    r"""Add grid lines to an axis and format them

    :Call:
        >>> _grid(ax, **kw)
    :Inputs:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
    :Effects:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Grid lines added to axes
    :Versions:
        * 2019-03-07 ``@jmeeroff``: First version
        * 2019-12-23 ``@ddalle``: Updated from :mod:`plotutils`
        * 2020-01-27 ``@ddalle``: From :func:`plot_mpl.grid`
    """
    # Make sure pyplot loaded
    _import_pyplot()
    # Get major grid option
    major_grid = kw.get("Grid", None)
    # Check value
    if major_grid is None:
        # Leave it as it currently is
        pass
    elif major_grid:
        # Get grid style
        kw_major = kw.get("GridOptions", {})
        # Ensure that the axis is below
        ax.set_axisbelow(True)
        # Add the grid
        ax.grid(True, **kw_major)
    else:
        # Turn the grid off, even if previously turned on
        ax.grid(False)
    # Get minor grid option
    minor_grid = kw.get("MinorGrid", None)
    # Check value
    if minor_grid is None:
        # Leave it as it currently is
        pass
    elif minor_grid:
        # Get grid style
        kw_minor = kw.get("MinorGridOptions", {})
        # Ensure that the axis is below
        ax.set_axisbelow(True)
        # Minor ticks are required
        ax.minorticks_on()
        # Add the grid
        ax.grid(which="minor", **kw_minor)
    else:
        # Turn the grid off, even if previously turned on
        ax.grid(False, which="minor")


# Plot part
def _plot(xv, yv, fmt=None, **kw):
    r"""Call the :func:`plot` function with cycling options

    :Call:
        >>> h = _plot(xv, yv, **kw)
        >>> h = _plot(xv, yv, fmt, **kw)
    :Inputs:
        *xv*: :class:`np.ndarray`
            Array of *x*-coordinates
        *yv*: :class:`np.ndarray`
            Array of *y*-coordinates
        *fmt*: :class:`str`
            Optional format option
        *i*, *Index*: {``0``} | :class:`int`
            Phase number to cycle through plot options
        *rotate*, *Rotate*: ``True`` | {``False``}
            Plot independent variable on vertical axis
    :Keyword Arguments:
        * See :func:`matplotlib.pyplot.plot`
    :Outputs:
        *h*: :class:`list` (:class:`matplotlib.lines.Line2D`)
            List of line instances
    :Versions:
        * 2019-03-04 ``@ddalle``: First version
        * 2020-01-24 ``@ddalle``: Moved to :mod:`plot_mpl.mpl`
    """
    # Ensure plot() is available
    _import_pyplot()
    # Get index
    i = kw.pop("Index", kw.pop("i", 0))
    # Get rotation option
    r = kw.pop("Rotate", kw.pop("rotate", False))
    # Flip inputs
    if r:
        yv, xv = xv, yv
    # Initialize plot options
    kw_p = MPLOpts.select_phase(kw, i)
    # Call plot
    if typeutils.isstr(fmt):
        # Call with extra format argument
        h = plt.plot(xv, yv, fmt, **kw_p)
    else:
        # No format argument
        h = plt.plot(xv, yv, **kw_p)
    # Output
    return h


# Single spine: extents
def _spine(spine, opt, vmin, vmax):
    r"""Apply visibility options to a single spine

    :Call:
        >>> _spine(spine, opt, vmin, vmax)
    :Inputs:
        *spine*: :clas:`matplotlib.spines.Spine`
            A single spine (left, right, top, or bottom)
        *opt*: ``None`` | ``True`` | ``False`` | ``"clipped"``
            Option for this spine
        *vmin*: :class:`float`
            If using clipped spines, minimum value for spine
        *vmax*: :class:`float`
            If using clipped spines, maximum value for spine
    :Versions:
        * 2019-03-08 ``@ddalle``: First version
        * 2020-01027 ``@ddalle``: From :func:`plot_mpl.format_spine1`
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
def _spines(ax, **kw):
    r"""Format Matplotlib axes spines and ticks

    :Call:
        >>> h = _spines(ax, **kw)
    :Inputs:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
    :Effects:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Grid lines added to axes
    :Versions:
        * 2019-03-07 ``@jmeeroff``: First version
        * 2019-12-23 ``@ddalle``: From :mod:`tnakit.plotutils`
        * 2020-01-27 ``@ddalle``: From :func:`plot_mpl.format_spines`
    """
   # --- Setup ---
    # Make sure pyplot loaded
    _import_pyplot()
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
    xa = kw.get("XSpineMin", xmin)
    xb = kw.get("XSpineMax", xmax)
    ya = kw.get("YSpineMin", ymin)
    yb = kw.get("YSpineMax", ymax)
    # Process manual limits for individual spines
    yaL = kw.get("LeftSpineMin", ya)
    ybL = kw.get("LeftSpineMax", yb)
    yaR = kw.get("RightSpineMin", ya)
    ybR = kw.get("RightSpineMax", yb)
    xaB = kw.get("BottomSpineMin", xa)
    xbB = kw.get("BottomSpineMax", xb)
    xaT = kw.get("TopSpineMin", xa)
    xbT = kw.get("TopSpineMax", xb)
   # --- Overall Spine Options ---
    # Option to turn off all spines
    qs = kw.get("Spines", None)
    # Only valid options are ``None`` and ``False``
    if qs is not False:
        qs = None
    # Spine pairs options
    qX = kw.get("XSpine", qs)
    qY = kw.get("YSpine", qs)
    # Left spine options
    qL = kw.get("LeftSpine",   qY)
    qR = kw.get("RightSpine",  qY)
    qT = kw.get("TopSpine",    qX)
    qB = kw.get("BottomSpine", qX)
   # --- Spine On/Off Extents ---
    # Process these options
    _spine(spineL, qL, yaL, ybL)
    _spine(spineR, qR, yaR, ybR)
    _spine(spineT, qT, xaT, xbT)
    _spine(spineB, qB, xaB, xbB)
   # --- Spine Formatting ---
    # Paired options
    spopts = kw.get("SpineOptions", {})
    xsopts = kw.get("XSpineOptions", {})
    ysopts = kw.get("YSpineOptions", {})
    # Individual spines
    lsopts = kw.get("LeftSpineOptions", {})
    rsopts = kw.get("RightSpineOptions", {})
    bsopts = kw.get("BottomSpineOptions", {})
    tsopts = kw.get("TopSpineOptions", {})
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
    qt = kw.get("Ticks", None)
    # Only valid options are ``None`` and ``False``
    if qt is not False:
        qt = None
    # Options for ticks on each axis
    qtL = kw.get("LeftSpineTicks",   qt and qL)
    qtR = kw.get("RightSpineTicks",  qt and qR)
    qtB = kw.get("BottomSpineTicks", qt and qB)
    qtT = kw.get("TopSpineTicks",    qt and qT)
    # Turn on/off
    if qtL is not None:
        ax.tick_params(left=qtL)
    if qtR is not None:
        ax.tick_params(right=qtR)
    if qtB is not None:
        ax.tick_params(bottom=qtB)
    if qtT is not None:
        ax.tick_params(top=qtT)
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
    if qtlL is not None:
        ax.tick_params(labelleft=qtlL)
    if qtlR is not None:
        ax.tick_params(labelright=qtlR)
    if qtlB is not None:
        ax.tick_params(labelbottom=qtlB)
    if qtlT is not None:
        ax.tick_params(labeltop=qtlT)
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
    return ax.spines


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
    # Initialize limits
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
            if h.axes is None:
                continue
            # Get bounding box
            bbox = h.get_bbox().extents
            # Combine limits
            ymin = min(ymin, bbox[1])
            ymax = max(ymax, bbox[3])
        elif t in ["AxesImage"]:
            # Get bounds
            bbox = h.get_extent()
            # Update limits
            xmin = min(xmin, min(bbox[2], bbox[3]))
            xmax = max(xmax, max(bbox[2], bbox[3]))
    # Check for identical values
    if ymax - ymin <= 0.1*pad:
        # Expand by manual amount
        ymax += pad*abs(ymax)
        ymin -= pad*abs(ymin)
    # Add padding
    yminv = (1+pad)*ymin - pad*ymax
    ymaxv = (1+pad)*ymax - pad*ymin
    # Output
    return yminv, ymaxv


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
    # Initialize limits
    xmin = np.inf
    xmax = -np.inf
    # Loop through all children of the input axes.
    for h in ax.get_children()[:-1]:
        # Get the type's name string
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
        elif t in ["AxesImage"]:
            # Get bounds
            bbox = h.get_extent()
            # Update limits
            xmin = min(xmin, min(bbox[0], bbox[1]))
            xmax = max(xmax, max(bbox[0], bbox[1]))
    # Check for identical values
    if xmax - xmin <= 0.1*pad:
        # Expand by manual amount
        xmax += pad*abs(xmax)
        xmin -= pad*abs(xmin)
    # Add padding
    xminv = (1+pad)*xmin - pad*xmax
    xmaxv = (1+pad)*xmax - pad*xmin
    # Output
    return xminv, xmaxv


# Get extents of axes in figure fraction coordinates
def get_axes_plot_extents(ax=None):
    r"""Get extents of axes plot in figure fraction coordinates

    :Call:
        >>> xmin, ymin, xmax, ymax = get_axes_plot_extents(ax)
    :Inputs:
        *ax*: {``None``} | :class:`Axes`
            Axes handle (defaults to ``plt.gca()``)
    :Outputs:
        *xmin*: :class:`float`
            Horizontal coord of plot region left edge, 0 is figure left
        *ymin*: :class:`float`
            Vertical coord of plot region bottom edge, 0 is fig bottom
        *xmax*: :class:`float`
            Horizontal coord of plot region right edge, 1 is fig right
        *ymax*: :class:`float`
            Vertical coord of plot region top edge, 1 is figure's top
    :Versions:
        * 2020-01-08 ``@ddalle``: First version
    """
    # Import modules
    _import_pyplot()
    # Default axes
    if ax is None:
        ax = plt.gca()
    # Get figure
    fig = ax.figure
    # Draw the figure once to ensure the extents can be calculated
    ax.draw(fig.canvas.get_renderer())
    # Size of figure in pixels
    _, _, ifig, jfig = fig.get_window_extent().bounds
    # Get pixel count for axes extents
    ia, ja, ib, jb = _get_axes_extents(ax)
    # Convert to fractions
    xmin = ia / ifig
    ymin = ja / jfig
    xmax = ib / ifig
    ymax = jb / jfig
    # Output
    return xmin, ymin, xmax, ymax


# Get extents of axes with labels
def get_axes_full_extents(ax=None):
    r"""Get extents of axes including labels in figure fraction coords

    :Call:
        >>> xmin, ymin, xmax, ymax = get_axes_full_extents(ax)
    :Inputs:
        *ax*: {``None``} | :class:`Axes`
            Axes handle (defaults to ``plt.gca()``)
    :Outputs:
        *xmin*: :class:`float`
            Horizontal coord of plot region left edge, 0 is figure left
        *ymin*: :class:`float`
            Vertical coord of plot region bottom edge, 0 is fig bottom
        *xmax*: :class:`float`
            Horizontal coord of plot region right edge, 1 is fig right
        *ymax*: :class:`float`
            Vertical coord of plot region top edge, 1 is figure's top
    :Versions:
        * 2020-01-08 ``@ddalle``: First version
    """
    # Import modules
    _import_pyplot()
    # Default axes
    if ax is None:
        ax = plt.gca()
    # Get figure
    fig = ax.figure
    # Draw the figure once to ensure the extents can be calculated
    ax.draw(fig.canvas.get_renderer())
    # Size of figure in pixels
    _, _, ifig, jfig = fig.get_window_extent().bounds
    # Get pixel count for axes extents
    ia, ja, ib, jb = _get_axes_full_extents(ax)
    # Convert to fractions
    xmin = ia / ifig
    ymin = ja / jfig
    xmax = ib / ifig
    ymax = jb / jfig
    # Output
    return xmin, ymin, xmax, ymax


# Get extents of axes with labels
def get_axes_label_margins(ax=None):
    r"""Get margins occupied by axis and tick labels on axes' four sides

    :Call:
        >>> wl, hb, wr, ht = get_axes_label_margins(ax)
    :Inputs:
        *ax*: {``None``} | :class:`Axes`
            Axes handle (defaults to ``plt.gca()``)
    :Outputs:
        *wl*: :class:`float`
            Figure fraction beyond plot of labels on left
        *hb*: :class:`float`
            Figure fraction beyond plot of labels below
        *wr*: :class:`float`
            Figure fraction beyond plot of labels on right
        *ht*: :class:`float`
            Figure fraction beyond plot of labels above
    :Versions:
        * 2020-01-08 ``@ddalle``: First version
    """
    # Import modules
    _import_pyplot()
    # Default axes
    if ax is None:
        ax = plt.gca()
    # Get figure
    fig = ax.figure
    # Draw the figure once to ensure the extents can be calculated
    ax.draw(fig.canvas.get_renderer())
    # Size of figure in pixels
    _, _, ifig, jfig = fig.get_window_extent().bounds
    # Get pixel count for axes extents
    wa, ha, wb, hb = _get_axes_label_margins(ax)
    # Convert to fractions
    margin_l = wa / ifig
    margin_b = ha / jfig
    margin_r = wb / ifig
    margin_t = hb / jfig
    # Output
    return margin_l, margin_b, margin_r, margin_t


# Get extents of axes in pixels
def _get_axes_plot_extents(ax):
    r"""Get extents of axes plot in figure fraction coordinates

    :Call:
        >>> ia, ja, ib, jb = _get_axes_plot_extents(ax)
    :Inputs:
        *ax*: :class:`Axes`
            Axes handle
    :Outputs:
        *ia*: :class:`float`
            Pixel count of plot region left edge, 0 is figure left
        *ja*: :class:`float`
            Pixel count of plot region bottom edge, 0 is fig bottom
        *ib*: :class:`float`
            Pixel count of plot region right edge
        *jb*: :class:`float`
            Pixel count of plot region top edge
    :Versions:
        * 2020-01-08 ``@ddalle``: First version
    """
    # Get pixel count for axes extents
    ia, ja, iw, jh = ax.get_window_extent().bounds
    # Add width and height
    ib = ia + iw
    jb = ja + jh
    # Output
    return ia, ja, ib, jb


# Get extents of axes with labels
def _get_axes_full_extents(ax):
    r"""Get extents of axes including labels in pixels

    :Call:
        >>> ia, ja, ib, jb = _get_axes_full_extents(ax)
    :Inputs:
        *ax*: :class:`Axes`
            Axes handle
    :Outputs:
        *ia*: :class:`float`
            Pixel count of plot plus labels left edge
        *ja*: :class:`float`
            Pixel count of plot plus labels bottom edge
        *ib*: :class:`float`
            Pixel count of plot plus labels right edge
        *jb*: :class:`float`
            Pixel count of plot plus labels top edge
    :Versions:
        * 2020-01-08 ``@ddalle``: First version
    """
    # Get pixel count for axes extents
    ia, ja, ib, jb = _get_axes_plot_extents(ax)
    # Get plot window bounds to check for valid tick labels
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    # Get axis label extents
    ia_x, ja_x, iw_x, jw_x = ax.xaxis.label.get_window_extent().bounds
    ia_y, ja_y, iw_y, jw_y = ax.yaxis.label.get_window_extent().bounds
    # Initialize bounds of labels and axes, if the extent is nonzero
    if iw_x*jw_x > 0:
        # Get axes bounds
        ib_x = ia_x + iw_x
        jb_x = ja_x + jw_x
        # Note: could be logical here and just use xlabel to set bottom
        # bounds, but why not be conservative?
        ia = min(ia, ia_x)
        ib = max(ib, ib_x)
        ja = min(ja, ja_x)
        jb = max(jb, jb_x)
    # Process ylabel
    if iw_y*jw_y > 0:
        # Get axes bounds
        ib_y = ia_y + iw_y
        jb_y = ja_y + jw_y
        # Note: could be logical here and just use xlabel to set bottom
        # bounds, but why not be conservative?
        ia = min(ia, ia_y)
        ib = max(ib, ib_y)
        ja = min(ja, ja_y)
        jb = max(jb, jb_y)
    # Loop through xtick labels
    for tick in ax.get_xticklabels():
        # Get position in data coordinates
        xtick, _ = tick.get_position()
        # Check if it's clipped
        if (xtick < xmin) or (xtick > xmax):
            continue
        # Get window extents
        ia_t, ja_t, iw_t, jw_t = tick.get_window_extent().bounds
        # Check for null tick
        if iw_t*jw_t == 0.0:
            continue
        # Translate to actual bounds
        ib_t = ia_t + iw_t
        jb_t = ja_t + jw_t
        # Update bounds
        ia = min(ia, ia_t)
        ib = max(ib, ib_t)
        ja = min(ja, ja_t)
        jb = max(jb, jb_t)
    # Loop through ytick labels
    for tick in ax.get_yticklabels():
        # Get position in data coordinates
        _, ytick = tick.get_position()
        # Check if it's clipped
        if (ytick < ymin) or (ytick > ymax):
            continue
        # Get window extents
        ia_t, ja_t, iw_t, jw_t = tick.get_window_extent().bounds
        # Check for null tick
        if iw_t*jw_t == 0.0:
            continue
        # Translate to actual bounds
        ib_t = ia_t + iw_t
        jb_t = ja_t + jw_t
        # Update bounds
        ia = min(ia, ia_t)
        ib = max(ib, ib_t)
        ja = min(ja, ja_t)
        jb = max(jb, jb_t)
    # Deal with silly scaling factors for both axes
    for tick in [ax.xaxis.offsetText, ax.yaxis.offsetText]:
        # Get window extents
        ia_t, ja_t, iw_t, jw_t = tick.get_window_extent().bounds
        # Check for null tick
        if iw_t*jw_t == 0.0:
            continue
        # Translate to actual bounds
        ib_t = ia_t + iw_t
        jb_t = ja_t + jw_t
        # Update bounds
        ia = min(ia, ia_t)
        ib = max(ib, ib_t)
        ja = min(ja, ja_t)
        jb = max(jb, jb_t)
    # Output
    return ia, ja, ib, jb


# Get extents of axes with labels
def _get_axes_label_margins(ax):
    r"""Get pixel counts of tick and axes labels on all four sides

    :Call:
        >>> wa, ha, wb, hb = _get_axes_label_margins(ax)
    :Inputs:
        *ax*: :class:`Axes`
            Axes handle
    :Outputs:
        *wa*: :class:`float`
            Pixel count beyond plot of labels on left
        *ha*: :class:`float`
            Pixel count beyond plot of labels below
        *wb*: :class:`float`
            Pixel count beyond plot of labels on right
        *hb*: :class:`float`
            Pixel count beyond plot of labels above
    :Versions:
        * 2020-01-08 ``@ddalle``: First version
    """
    # Get pixel count for plot extents
    ia_ax, ja_ax, ib_ax, jb_ax = _get_axes_plot_extents(ax)
    # Get pixel count for plot extents
    ia, ja, ib, jb = _get_axes_full_extents(ax)
    # Margins
    wa = ia_ax - ia
    ha = ja_ax - ja
    wb = ib - ib_ax
    hb = jb - jb_ax
    # Output
    return wa, ha, wb, hb
