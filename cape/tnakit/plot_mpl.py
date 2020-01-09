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
import cape.tnakit.rstutils as rstutils
import cape.tnakit.statutils as statutils
import cape.tnakit.typeutils as typeutils

# TNA toolkit direct imports
from cape.tnakit.optutils import optitem

# Get a variable to hold the "type" of "module"
mod = os.__class__

# Initialize handle for modules
plt = object()
mpl = object()
mplax = object()
mplfig = object()


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
    global mplax
    global mplfig
    # Exit if already imported
    if isinstance(mpl, mod):
        return
    # Import module
    try:
        import matplotlib as mpl
        import matplotlib.axes as mplax
        import matplotlib.figure as mplfig
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
        *h*: :class:`cape.tnakit.plto_mpl.MPLHandle`
            Dictionary of plot handles
    :Versions:
        * 2019-03-01 ``@ddalle``: First (independent) version
        * 2019-12-23 ``@ddalle``: Object-oriented options and output
    """
   # --- Prep ---
    # Ensure plot() is loaded
    import_pyplot()
    # Process options
    opts = MPLOpts(**kw)
    # Initialize output
    h = MPLHandle()
    # Number of args and args used
    na = len(a)
    ia = 0
   # --- Default Control ---
    # Min/Max
    if ("ymin" in opts) and ("ymax" in opts):
        # If min/max values are specified, turn on *PlotMinMax*
        qmmx = True
    else:
        # Otherwise false
        qmmx = False
    # UQ
    if ("yerr" in opts) or ("xerr" in opts):
        # If min/max values are specified, turn on *PlotUncertainty*
        quq = True
    else:
        # Otherwise false
        quq = False
   # --- Control Options ---
    # Options to plot min/max
    qline = opts.get("ShowLine", True)
    qmmx = opts.get("ShowMinMax", qmmx)
    qerr = opts.get("ShowError", False)
    quq = opts.get("ShowUncertainty", quq and (not qerr))
   # --- Universal Options ---
    # Font options
    kw_font = opts.font_options()
   # --- Figure Setup ---
    # Process figure options
    kw_fig = opts.figure_options()
    # Get/create figure
    h.fig = figure(**kw_fig)
   # --- Axis Setup ---
    # Process axis options
    kw_ax = opts.axes_options()
    # Get/create axis
    h.ax = axes(**kw_ax)
   # --- Primary Plot ---
    # Process plot options
    kw_plot = opts.plot_options()
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
        h.lines += _plot(xv, yv, *fmt, **kw_plot)
   # --- Min/Max ---
    # Process min/max options
    minmax_type, kw_mmax = opts.minmax_options()
    # Plot it
    if qmmx:
        # Min/max values
        if na >= ia + 2:
            # Get parameter values
            ymin = a[ia]
            ymax = a[ia+1]
            # Increase count
            ia += 2
        else:
            # Pop values
            ymin = opts.get("ymin", None)
            ymax = opts.get("ymax", None)
        # Plot call
        if minmax_type == "FillBetween":
            # Do a :func:`fill_between` plot
            h.minmax = fill_between(xv, ymin, ymax, **kw_mmax)
        elif minmax_type == "ErrorBar":
            # Convert to error bar widths
            yerr = minmax_to_errorbar(yv, ymin, ymax, **kw_mmax)
            # Do a :func:`errorbar` plot
            h.minmax = errorbar(xv, yv, yerr, **kw_mmax)
   # --- Error ---
    # Process min/max options
    error_type, kw_err = opts.error_options()
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
            yerr = kw.get("yerr", None)
        # Check for horizontal error bars
        xerr = kw.get("xerr", None)
        # Plot call
        if error_type == "FillBetween":
            # Convert to min/max values
            ymin, ymax = errorbar_to_minmax(yv, yerr)
            # Do a :func:`fill_between` plot
            h.error = fill_between(xv, ymin, ymax, **kw_err)
        elif t_err == "ErrorBar":
            # Do a :func:`errorbar` plot
            h.error = errorbar(xv, yv, yerr, **kw_err)
   # --- UQ ---
    # Process min/max options
    uq_type, kw_uq = opts.uq_options()
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
            yerr = kw.get("uy", kw.get("yerr", None))
        # Check for horizontal error bars
        xerr = kw.get("ux", kw.get("ux", None))
        # Plot call
        if uq_type == "FillBetween":
            # Convert to min/max values
            ymin, ymax = errorbar_to_minmax(yv, yerr)
            # Do a :func:`fill_between` plot
            h.uq = fill_between(xv, ymin, ymax, **kw_uq)
        elif uq_type == "ErrorBar":
            # Do a :func:`errorbar` plot
            h.uq = errobar(xv, yv, yerr, **kw_uq)
   # --- Axis formatting ---
    ## Process grid lines options
    kw_grid = opts.grid_options()
    # Apply grid lines
    grid(h.ax, **kw_grid)
    # Process spine options
    kw_spines = opts.spine_options()
    # Apply options relating to spines
    h.spines = format_spines(h.ax, **kw_spines)
    # Process axes format options
    kw_axfmt = opts.axformat_options()
    # Apply formatting
    h.xlabel, h.ylabel = axes_format(h.ax, **kw_axfmt)
   # --- Margin adjustment ---
    # Process axes margin/adjust options
    kw_axadj = opts.axadjust_options()
    # Adjust extents
    axes_adjust(h.fig, ax=h.ax, **kw_axadj)
   # --- Legend ---
    ## Process options for legend
    kw_legend = opts.legend_options()
    # Create legend
    h.legend = legend(h.ax, **kw_legend)
   # --- Cleanup ---
    # Save options
    h.opts = opts
    # Output
    return h


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
    :Outputs:
        *ax*: :class:`AxesSubplot`
            Handle to subplot directed to use from these options
    :Versions:
        * 2020-01-03 ``@ddalle``: First version
    """
    # Make sure pyplot is present
    import_pyplot()
    # Default figure
    if fig is None:
        # Get most recent figure or create
        fig = plt.gcf()
    elif isinstance(fig, int):
        # Get figure handle from number
        fig = plt.figure(fig)
    elif not isinstance(fig, mplfig.Figure):
        # Not a figure or number
        raise TypeError(
            "'fig' arg expected 'int' or 'Figure' (got %s)" % type(fig))
    # Process options
    opts = MPLOpts(**kw)
    # Get axes from figure
    ax_list = fig.get_axes()
    # Minimum number of axes
    nmin_ax = min(1, len(ax_list))
    # Get "axes" option
    ax = opts.get("ax")
    # Get subplot number options
    subplot_i = opts.get("Subplot")
    subplot_m = opts.get("SubplotRows", nmin_ax)
    subplot_n = opts.get("SubplotCols", nmin_ax // subplot_m)
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
    margin_b = opts.get("MarginBottom", 0.02)
    margin_l = opts.get("MarginLeft", 0.02)
    margin_r = opts.get("MarginRight", 0.015)
    margin_t = opts.get("MarginTop", 0.015)
    # Apply to minimum margins
    adj_b += margin_b
    adj_l += margin_l
    adj_r -= margin_r
    adj_t -= margin_t
    # Get user options
    adj_b = opts.get("AdjustBottom", adj_b)
    adj_l = opts.get("AdjustLeft", adj_l)
    adj_r = opts.get("AdjustRight", adj_r)
    adj_t = opts.get("AdjustTop", adj_t)
    # Update bottom margin
    if adj_b is not None:
        # Current
        (xmin, ymin), (xmax, ymax) = ax.get_position().get_points()
        xmax = min(1.0, xmax)
        ymax = min(1.0, ymax)
        # Update bottom and top
        ax.set_position([xmin, adj_b, xmax-xmin, ymax-adj_b])
    # Update left margin
    if adj_l is not None:
        # Current
        (xmin, ymin), (xmax, ymax) = ax.get_position().get_points()
        xmax = min(1.0, xmax)
        ymax = min(1.0, ymax)
        # Update left and right
        ax.set_position([adj_l, ymin, xmax-adj_l, ymax-ymin])
    # Update right margin
    if adj_r is not None:
        # Current
        (xmin, ymin), (xmax, ymax) = ax.get_position().get_points()
        ymax = min(1.0, ymax)
        # Update left and right
        ax.set_position([xmin, ymin, adj_r-xmin, ymax-ymin])
    # Update top margin
    if adj_t is not None:
        # Current
        (xmin, ymin), (xmax, ymax) = ax.get_position().get_points()
        xmax = min(1.0, xmax)
        # Update top
        ax.set_position([xmin, ymin, xmax-xmin, adj_t-ymin])
    # Output
    return ax


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
    import_pyplot()
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
    import_pyplot()
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
    import_pyplot()
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
    """
    # Import PyPlot
    import_pyplot()
    import_matplotlib()
    # Get figure handle and other options
    fig = kw.get("fig", None)
    figopts = kw.get("FigOptions", {})
    # Check for a figure number
    fignum = figopts.pop("num", None)
    # Create figure if needed
    if not isinstance(fig, mplfig.Figure):
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


# Axis part (initial)
def axes(**kw):
    """Create new axes or edit one if necessary

    :Call:
        >>> fig = axes(**kw)
    :Inputs:
        *ax*: ``None`` | :class:`matplotlib.axes._subplots.AxesSubplot`
            Optional axes handle
        *AxesOptions*: {``None``} | :class:`dict`
            Options to apply to figure handle using :func:`ax.set`
    :Outputs:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
    :Versions:
        * 2019-03-06 ``@ddalle``: First version
    """
    # Import PyPlot
    import_pyplot()
    # Get figure handle and other options
    ax = kw.get("ax", None)
    axopts = kw.get("AxesOptions", {})
    # Check for a subplot description
    axnum = axopts.pop("subplot", None)
    # Create figure if needed
    if not isinstance(ax, mplax._subplots.Axes):
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
    """
    # Ensure plot() is available
    import_pyplot()
    # Get index
    i = kw.pop("Index", kw.pop("i", 0))
    # Get rotation option
    r = kw.pop("Rotate", kw.pop("rotate", False))
    # Flip inputs
    if r:
        yv, xv = xv, yv
    # Initialize plot options
    kw_p = MPLOpts.select_plotphase(kw, i)
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
    r"""Call the :func:`fill_between` or :func:`fill_betweenx` function

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
    kw_fb = MPLOpts.select_plotphase(kw, i)
    # Call the plot method
    h = fnplt(xv, yl, yu, **kw_fb)
    # Output
    return h


# Error bar plot
def errorbar(xv, yv, yerr=None, xerr=None, **kw):
    r"""Call the :func:`errobar` function

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
    kw_eb = MPLOpts.select_plotphase(kw, i)
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
    # Check basic option
    # Import modules if needed
    import_pyplot()
    # Get overall "Legend" option
    show_legend = kw.pop("ShowLegend", None)
    # Exit immediately if explicit
    if show_legend is False:
        return
    # Get font properties (copy)
    opts_prop = dict(kw.pop("prop", {}))
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
        # Repeat plot
        leg = ax.legend(prop=opts_prop, **kw)
   # --- Font ---
    # Number of entries in the legend
    ntext = len(leg.get_texts())
    # If no legends, remove it
    if (ntext < 1) or not show_legend:
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
    """
   # --- Prep ---
    # Make sure pyplot loaded
    import_pyplot()
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


# Creation and formatting of grid lines
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
    """
    # Make sure pyplot loaded
    import_pyplot()
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


# Single spine: extents
def format_spine1(spine, opt, vmin, vmax):
    r"""Apply visibility options to a single spine

    :Call:
        >>> format_spine1(spine, opt, vmin, vmax)
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
    r"""Format Matplotlib axes spines and ticks

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
        * 2019-12-23 ``@ddalle``: From :mod:`tnakit.plotutils`
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
    if qs is not False: qs = None
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
    format_spine1(spineL, qL, yaL, ybL)
    format_spine1(spineR, qR, yaR, ybR)
    format_spine1(spineT, qT, xaT, xbT)
    format_spine1(spineB, qB, xaB, xbB)
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
    if qt is not False: qt = None
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
            # Check for list (combine)
            if isinstance(v, list):
                # Get attribute from parent
                v0 = self.__dict__.get(k)
                # Check if both are lists
                if isinstance(v0, list):
                    # Combine two lists
                    self.__dict__[k] = v0 + v
                else:
                    # Replace non-list value
                    self.__dict__[k] = v
            elif v.__class__.__module__.startswith("matplotlib"):
                # Some other plot handle; replace
                self.__dict__[k] = v
        


# Standard type strings
_rst_boolf = """```True`` | {``False``}"""
_rst_booln = """{``None``} | ``True`` | ``False``"""
_rst_boolt = """{``True``} | ``False``"""
_rst_dict = """{``None``} | :class:`dict`"""
_rst_float = """{``None``} | :class:`float`"""
_rst_floatpos = """{``None``} | :class:`float` > 0.0"""
_rst_int = """{``None``} | :class:`int`"""
_rst_intpos = """{``None``} | :class:`int` > 0"""
_rst_num = """{``None``} | :class:`int` | :class:`float`"""
_rst_numpos = """{``None``} | :class:`int` > 0 | :class:`float` > 0.0"""
_rst_str = """{``None``} | :class:`str`"""
_rst_strnum = """{``None``} | :class:`str` | :class:`int` | :class:`float`"""


# Options interface
class MPLOpts(dict):
  # ====================
  # Class Attributes
  # ====================
  # <
   # --- Global Options ---
    # Lists of options
    _optlist = [
        "AdjustBottom",
        "AdjustLeft",
        "AdjustRight",
        "AdjustTop",
        "BottomSpine",
        "BottomSpineMax",
        "BottomSpineMin",
        "BottomSpineOptions",
        "BottomSpineTicks",
        "BottomTickLabels",
        "Density",
        "ErrorBarOptions",
        "ErrorBarMarker",
        "ErrorOptions",
        "ErrorPlotType",
        "FigDPI",
        "FigHeight",
        "FigNumber",
        "FigOptions",
        "FigWidth",
        "FillBetweenOptions",
        "FontOptions",
        "FontName",
        "FontSize",
        "FontStretch",
        "FontStyle",
        "FontVariant",
        "FontWeight",
        "Grid",
        "GridOptions",
        "GridStyle",
        "Index",
        "Label",
        "LeftSpine",
        "LeftSpineMax",
        "LeftSpineMin",
        "LeftSpineOptions",
        "LeftSpineTicks",
        "LeftTickLabels",
        "Legend",
        "LegendFontName",
        "LegendFontSize",
        "LegendFontStretch",
        "LegendFontStyle",
        "LegendFontVariant",
        "LegendFontWeight",
        "LegendOptions",
        "MajorGrid",
        "MarginBottom",
        "MarginLeft",
        "MarginRight",
        "MarginTop",
        "MinMaxOptions",
        "MinMaxPlotType",
        "MinorGrid",
        "MinorGridOptions",
        "Pad",
        "PlotColor",
        "PlotLineStyle",
        "PlotLineWidth",
        "PlotOptions",
        "RightSpine",
        "RightSpineMax",
        "RightSpineMin",
        "RightSpineOptions",
        "RightSpineTicks",
        "RightTickLabels",
        "Rotate",
        "ShowError",
        "ShowLegend",
        "ShowLine",
        "ShowMinMax",
        "ShowUncertainty",
        "SpineOptions",
        "Spines",
        "Subplot",
        "SubplotCols",
        "SubplotRows",
        "TickDirection",
        "TickFontSize",
        "TickLabels",
        "TickOptions",
        "TickRotation",
        "TickSize",
        "Ticks",
        "TopSpine",
        "TopSpineMax",
        "TopSpineMin",
        "TopSpineOptions",
        "TopSpineTicks",
        "TopTickLabels",
        "UncertaintyPlotType",
        "UncertaintyOptions",
        "XLabel",
        "XLim",
        "XPad",
        "XSpine",
        "XSpineMax",
        "XSpineMin",
        "XSpineOptions",
        "XTickDirection",
        "XTickFontSize",
        "XTickLabels",
        "XTickOptions",
        "XTickRotation",
        "XTickSize",
        "XTicks",
        "YLabel",
        "YLim",
        "YPad",
        "YSpine",
        "YSpineMax",
        "YSpineMin",
        "YSpineOptions",
        "YTickDirection",
        "YTickFontSize",
        "YTickLabels",
        "YTickOptions",
        "YTickRotation",
        "YTickSize",
        "YTicks",
        "ax",
        "fig",
        "xerr",
        "yerr",
        "ymin",
        "ymax",
    ]

    # Options for which a singleton is a list
    _optlist_list = [
        "dashes",
        "XLim",
        "YLim",
        "XTickLabels",
        "XTicks",
        "YTickLabels",
        "YTicks"
    ]

    # Alternate names
    _optmap = {
        "Axes": "ax",
        "BottomTicks": "BottomSpineTicks",
        "ErrorBarOpts": "ErrorBarOptions",
        "Figure": "fig",
        "FillBetweenOpts": "FillBetweenOptions",
        "Font": "FontName",
        "FontFamily": "FontName",
        "GridOpts": "GridOptions",
        "LeftTicks": "LeftSpineTicks",
        "MajorGridOpts": "GridOptions",
        "MajorGridOptions": "GridOptions",
        "MinMaxOpts": "MinMaxOptions",
        "PlotOpts": "PlotOptions",
        "RightTicks": "RightSpineTicks",
        "ShowUQ": "ShowUncertainty",
        "TopTicks": "TopSpineTicks",
        "UQOptions": "UncertaintyOptions",
        "UQOpts": "UncertaintyOptions",
        "UncertaintyOpts": "UncertaintyOptions",
        "UQPlotType": "UncertaintyPlotType",
        "density": "Density",
        "grid": "Grid",
        "hfig": "FigHeight",
        "i": "Index",
        "label": "Label",
        "lbl": "Label",
        "nfig": "FigNumber",
        "numfig": "FigNumber",
        "rotate": "Rotate",
        "subplot": "Subplot",
        "wfig": "FigWidth",
        "xlabel": "XLabel",
        "xlim": "XLim",
        "ylabel": "YLabel",
        "ylim": "YLim",
    }

   # --- Option Sublists ---
    _optlist_axes = [
        "ax",
        "AxesOptions"
    ]
    _optlist_axadjust = [
        "AdjustBottom",
        "AdjustLeft",
        "AdjustRight",
        "AdjustTop",
        "MarginBottom",
        "MarginLeft",
        "MarginRight",
        "MarginTop",
        "Subplot",
        "SubplotCols",
        "SubplotRows"
    ]
    _optlist_axformat = [
        "Density",
        "Index",
        "Pad",
        "Rotate",
        "XLabel",
        "XLim",
        "XLimMax",
        "XLimMin",
        "XPad",
        "YLabel",
        "YLim",
        "YLimMax",
        "YLimMin",
        "YPad"
    ]
    _optlist_fig = [
        "fig",
        "FigOptions",
        "FigNumber",
        "FigWidth",
        "FigHeight",
        "FigDPI"
    ]
    _optlist_font = [
        "FontOptions",
        "FontName",
        "FontSize",
        "FontStretch",
        "FontStyle",
        "FontVariant",
        "FontWeight"
    ]
    _optlist_grid = [
        "Grid",
        "GridOptions",
        "MajorGrid",
        "MinorGrid",
        "MinorGridOptions",
    ]
    _optlist_plot = [
        "Index",
        "Rotate",
        "PlotOptions",
        "PlotColor",
        "PlotLineStyle",
        "PlotLineWidth",
    ]
    _optlist_errobar = [
        "Index",
        "Rotate",
        "ErrorBarOptions",
        "ErrorBarMarker"
    ]
    _optlist_fillbetween = [
        "Index",
        "Rotate",
        "FillBetweenOptions"
    ]
    _optlist_minmax = [
        "Index",
        "Rotate",
        "MinMaxOptions",
        "MinMaxPlotType",
        "ErrorBarOptions",
        "ErrorBarMarker",
        "FillBetweenOptions"
    ]
    _optlist_error = [
        "Index",
        "Rotate",
        "ErrorOptions",
        "ErrorPlotType",
        "ErrorBarOptions",
        "ErrorBarMarker",
        "FillBetweenOptions"
    ]
    _optlist_uq = [
        "Index",
        "Rotate",
        "ErrorBarMarker",
        "ErrorBarOptions",
        "FillBetweenOptions",
        "UncertaintyPlotType",
        "UncertaintyOptions"
    ]
    _optlist_spines = [
        "Spines",
        "SpineOptions",
        "Ticks",
        "TickDirection",
        "TickFontSize",
        "TickLabels",
        "TickOptions",
        "TickRotation",
        "TickSize",
        "BottomSpine",
        "BottomSpineMax",
        "BottomSpineMin",
        "BottomSpineOptions",
        "BottomSpineTicks",
        "BottomTickLabels",
        "LeftSpine",
        "LeftSpineMax",
        "LeftSpineMin",
        "LeftSpineOptions",
        "LeftSpineTicks",
        "LeftTickLabels",
        "RightSpine",
        "RightSpineMax",
        "RightSpineMin",
        "RightSpineOptions",
        "RightSpineTicks",
        "RightTickLabels",
        "TopSpine",
        "TopSpineMax",
        "TopSpineMin",
        "TopSpineOptions",
        "TopSpineTicks",
        "TopTickLabels",
        "XSpine",
        "XSpineMax",
        "XSpineMin",
        "XSpineOptions",
        "XTicks",
        "XTickDirection",
        "XTickFontSize",
        "XTickLabels",
        "XTickOptions",
        "XTickRotation",
        "XTickSize",
        "YSpine",
        "YSpineMax",
        "YSpineMin",
        "YSpineOptions",
        "YTicks",
        "YTickDirection",
        "YTickFontSize",
        "YTickLabels",
        "YTickOptions",
        "YTickRotation",
        "YTickSize",
    ]
    _optlist_legend = [
        "Legend",
        "LegendAnchor",
        "LegendFontName",
        "LegendFontSize",
        "LegendFontStretch",
        "LegendFontStyle",
        "LegendFontVariant",
        "LegendFontWeight",
        "LegendLocation",
        "LegendOptions"
    ]
    _optlist_subplot = [
        "AdjustBottom",
        "AdjustLeft",
        "AdjustRight",
        "AdjustTop",
        "Subplot",
        "SubplotCols",
        "SubplotRows"
    ]

   # --- Types ---
    # Types
    _opttypes = {
        "AdjustBottom": float,
        "AdjustLeft": float,
        "AdjustRight": float,
        "AdjustTop": float,
        "AxesOptions": dict,
        "BottomSpine": (bool, typeutils.strlike),
        "BottomSpineMax": float,
        "BottomSpineMin": float,
        "BottomSpineOptions": dict,
        "BottomSpineTicks": bool,
        "BottomTickLabels": bool,
        "Density": bool,
        "ErrorBarMarker": typeutils.strlike,
        "ErrorBarOptions": dict,
        "ErrorOptions": dict,
        "ErrorPlotType": typeutils.strlike,
        "FigDPI": (float, int),
        "FigHeight": float,
        "FigNumber": int,
        "FigOptions": dict,
        "FigWidth": float,
        "FillBetweenOptions": dict,
        "FontName": typeutils.strlike,
        "FontOptions": dict,
        "FontSize": (int, float, typeutils.strlike),
        "FontStretch": (int, float, typeutils.strlike),
        "FontStyle": typeutils.strlike,
        "FontVariant": typeutils.strlike,
        "FontWeight": (float, int, typeutils.strlike),
        "Grid": int,
        "GridOptions": dict,
        "Index": int,
        "Label": typeutils.strlike,
        "LeftSpine": (bool, typeutils.strlike),
        "LeftSpineMax": float,
        "LeftSpineMin": float,
        "LeftSpineOptions": dict,
        "LeftSpineTicks": bool,
        "LeftTickLabels": bool,
        "Legend": bool,
        "LegendAnchor": (tuple, list),
        "LegendFontName": typeutils.strlike,
        "LegendFontOptions": dict,
        "LegendFontSize": (int, float, typeutils.strlike),
        "LegendFontStretch": (int, float, typeutils.strlike),
        "LegendFontStyle": typeutils.strlike,
        "LegendFontVariant": typeutils.strlike,
        "LegendFontWeight": (float, int, typeutils.strlike),
        "LegendLocation": (int, typeutils.strlike),
        "LegendOptions": dict,
        "MajorGrid": bool,
        "MarginBottom": float,
        "MarginLeft": float,
        "MarginRight": float,
        "MarginTop": float,
        "MinorGrid": bool,
        "MinorGridOptions": dict,
        "MinMaxOptions": dict,
        "MinMaxPlotType": typeutils.strlike,
        "Pad": float,
        "PlotColor": (tuple, typeutils.strlike),
        "PlotLineStyle": typeutils.strlike,
        "PlotLineWidth": (float, int),
        "PlotOptions": dict,
        "Rotate": bool,
        "ShowError": bool,
        "ShowLine": bool,
        "ShowMinMax":bool,
        "ShowUncertainty": bool,
        "SpineOptions": dict,
        "Spines": bool,
        "SubplotCols": int,
        "SubplotRows": int,
        "RightSpine": (bool, typeutils.strlike),
        "RightSpineMax": float,
        "RightSpineMin": float,
        "RightSpineOptions": dict,
        "RightSpineTicks": bool,
        "RightTickLabels": bool,
        "TickDirection": typeutils.strlike,
        "TickFontSize": (int, float, typeutils.strlike),
        "TickLabels": bool,
        "TickOptions": dict,
        "TickRotation": (int, float),
        "TickSize": (int, float, typeutils.strlike),
        "Ticks": bool,
        "TopSpine": (bool, typeutils.strlike),
        "TopSpineMax": float,
        "TopSpineMin": float,
        "TopSpineOptions": dict,
        "TopSpineTicks": bool,
        "TopTickLabels": bool,
        "UncertaintyOptions": dict,
        "UncertaintyPlotType": typeutils.strlike,
        "XLabel": typeutils.strlike,
        "XLim": (tuple, list),
        "XLimMax": float,
        "XLimMin": float,
        "XPad": float,
        "XTickDirection": typeutils.strlike,
        "XTickFontSize": (int, float, typeutils.strlike),
        "XTickLabels": (bool, list),
        "XTickOptions": dict,
        "XTickRotation": (int, float),
        "XTickSize": (int, float, typeutils.strlike),
        "XTicks": (bool, list),
        "YLabel": typeutils.strlike,
        "YLim": (tuple, list),
        "YLimMax": float,
        "YLimMin": float,
        "YPad": float,
        "YTickDirection": typeutils.strlike,
        "YTickFontSize": (int, float, typeutils.strlike),
        "YTickLabels": (bool, list),
        "YTickOptions": dict,
        "YTickRotation": (int, float),
        "YTickSize": (int, float, typeutils.strlike),
        "YTicks": (bool, list),
        "ax": object,
        "fig": object,
        "xerr": typeutils.arraylike,
        "yerr": typeutils.arraylike,
        "ymax": typeutils.arraylike,
        "ymin": typeutils.arraylike,
    }

   # --- Cascading Options
    # Global options mapped to subcategory options
    _kw_submap = {
        "AxesOptions": {},
        "ErrorOptions": {},
        "FigOptions": {
            "FigNumber": "num",
            "FigDPI": "dpi",
            "FigHeight": "figheight",
            "FigWidth": "figwidth",
        },
        "FontOptions": {
            "FontName":    "family",
            "FontSize":    "size",
            "FontStretch": "stretch",
            "FontStyle":   "style",
            "FontVariant": "variant",
            "FontWeight":  "weight",
        },
        "PlotOptions": {
            "Index": "Index",
            "Rotate": "Rotate",
            "Label": "label",
            "PlotColor": "color",
            "PlotLineWidth": "lw",
            "PlotLineStyle": "ls"
        },
        "LegendOptions": {
            "LegendAnchor": "bbox_to_anchor",
            "LegendLocation": "loc",
            "LegendNCol": "ncol",
            "ShowLegend": "ShowLegend",
        },
        "MinMaxOptions": {},
        "FillBetweenOptions": {
            "Index": "Index",
            "Rotate": "Rotate",
        },
        "ErrorBarOptions": {
            "Index": "Index",
            "Rotate": "Rotate",
            "ErrorBarMarker": "marker",
        },
        "GridOptions": {
            "GridColor": "color",
        },
        "UncertaintyOptions": {},
        "SpineOptions": {
            "TickFontSize": "labelsize",
            "TickRotation": "rotation",
            "TickSize": "size",
            "XTickFontSize": "labelsize",
            "XTickRotation": "rotation",
            "XTickSize": "size",
            "YTickFontSize": "labelsize",
            "YTickRotation": "rotation",
            "YTickSize": "size",
        },
    }

    # Options to inherit from elsewhere
    _kw_cascade = {
        "ErrorBarOptions": {
            "plot.color": "color",
        },
        "FillBetweenOptions": {
            "plot.color": "color",
        },
    }

   # --- Conflicting Options ---
    # Aliases to merge for subcategory options
    _kw_subalias = {
        "PlotOptions": {
            "linewidth": "lw",
            "linestyle": "ls",
            "c": "color",
        },
        "ErrorBarOptions": {
            "linewidth": "lw",
            "linestyle": "ls",
            "c": "color",
            "mec": "markeredgecolor",
            "mew": "mergeredgewidth",
            "mfc": "markerfacecolor",
            "ms": "markersize",
        },
        "FillBetweenOptions": {
            "linewidth": "lw",
            "linestyle": "ls",
            "c": "color",
        },
        "GridOptions": {
            "linewidth": "lw",
            "linestyle": "ls",
            "c": "color",
        },
    }

   # --- Documentation Data ---
    # Type strings
    _rst_types = {
        "AdjustBottom": _rst_float,
        "AdjustLeft": _rst_float,
        "AdjustRight": _rst_float,
        "AdjustTop": _rst_float,
        "AxesOptions": _rst_dict,
        "BottomSpine": """{``None``} | ``True`` | ``False`` | ``"clipped"``""",
        "BottomSpineMax": _rst_float,
        "BottomSpineMin": _rst_float,
        "BottomSpineOptions": _rst_dict,
        "BottomSpineTicks": _rst_booln,
        "BottomTickLabels": _rst_booln,
        "Density": _rst_boolt,
        "ErrorBarMarker": _rst_str,
        "ErrorBarOptions": _rst_dict,
        "ErrorOptions": _rst_dict,
        "ErrorPlotType": """``"FillBetween"`` | {``"ErrorBar"``}""",
        "FigDPI": _rst_numpos,
        "FigHeight": _rst_floatpos,
        "FigNumber": _rst_intpos,
        "FigOptions": _rst_dict,
        "FigWidth": _rst_floatpos,
        "FillBetweenOptions": _rst_dict,
        "FontName": _rst_str,
        "FontOptions": _rst_dict,
        "FontSize": _rst_strnum,
        "FontStretch": _rst_strnum,
        "FontStyle": ("""{``None``} | ``"normal"`` | """ +
            """``"italic"`` | ``"oblique"``"""),
        "FontVariant": """{``None``} | ``"normal"`` | ``"small-caps"``""",
        "FontWeight": _rst_strnum,
        "Grid": _rst_boolt,
        "GridOptions": _rst_dict,
        "Index": """{``0``} | :class:`int` >=0""",
        "Label": _rst_str,
        "LeftSpine": """{``None``} | ``True`` | ``False`` | ``"clipped"``""",
        "LeftSpineMax": _rst_float,
        "LeftSpineMin": _rst_float,
        "LeftSpineOptions": _rst_dict,
        "LeftSpineTicks": _rst_booln,
        "LeftTickLabels": _rst_booln,
        "Legend": _rst_booln,
        "LegendAnchor": r"""{``None``} | :class:`tuple`\ (*x*, *y*)""",
        "LegendFontName": _rst_str,
        "LegendFontOptions": _rst_dict,
        "LegendFontSize": _rst_strnum,
        "LegendFontStretch": _rst_strnum,
        "LegendFontStyle": ("""{``None``} | ``"normal"`` | """ +
            """``"italic"`` | ``"oblique"``"""),
        "LegendFontVariant": '{``None``} | ``"normal"`` | ``"small-caps"``',
        "LegendFontWeight": _rst_strnum,
        "LegendLocation": """{``None``} | :class:`str` | :class:`int`""",
        "LegendOptions": _rst_dict,
        "MajorGrid": _rst_boolt,
        "MarginBottom": _rst_float,
        "MarginLeft": _rst_float,
        "MarginRight": _rst_float,
        "MarginTop": _rst_float,
        "MinMaxPlotType": """{``"FillBetween"``} | ``"ErrorBar"``""",
        "MinMaxOptions": _rst_dict,
        "MinorGrid": _rst_boolf,
        "MinorGridOptions": _rst_dict,
        "Pad": _rst_float,
        "PlotColor": """{``None``} | :class:`str` | :class:`tuple`""",
        "PlotLineStyle": ('``":"`` | ``"-"`` | ``"none"`` | ' +
            '``"-."`` | ``"--"``'),
        "PlotLineWidth": _rst_numpos,
        "PlotOptions": _rst_dict,
        "RightSpine": """{``None``} | ``True`` | ``False`` | ``"clipped"``""",
        "RightSpineMax": _rst_float,
        "RightSpineMin": _rst_float,
        "RightSpineOptions": _rst_dict,
        "RightSpineTicks": _rst_booln,
        "RightTickLabels": _rst_booln,
        "Rotate": _rst_boolt,
        "ShowError": _rst_booln,
        "ShowMinMax": _rst_booln,
        "ShowUncertainty": _rst_booln,
        "Subplot": """{``None``} | :class:`Axes` | :class:`int`""",
        "SubplotCols": _rst_intpos,
        "SubplotRows": _rst_intpos,
        "TopSpine": """{``None``} | ``True`` | ``False`` | ``"clipped"``""",
        "TopSpineMax": _rst_float,
        "TopSpineMin": _rst_float,
        "TopSpineOptions": _rst_dict,
        "TopSpineTicks": _rst_booln,
        "TopTickLabels": _rst_booln,
        "UncertaintyOptions": _rst_dict,
        "UncertaintyPlotType": """{``"FillBetween"``} | ``"ErrorBar"``""",
        "XLabel": _rst_str,
        "XLim": r"""{``None``} | (:class:`float`, :class:`float`)""",
        "XLimMax": _rst_float,
        "XLimMin": _rst_float,
        "XPad": """{*Pad*} | :class:`float`""",
        "YLabel": _rst_str,
        "YLim": r"""{``None``} | (:class:`float`, :class:`float`)""",
        "YLimMax": _rst_float,
        "YLimMin": _rst_float,
        "YPad": """{*Pad*} | :class:`float`""",
        "ax": """{``None``} | :class:`matplotlib.axes._subplots.Axes`""",
        "fig": """{``None``} | :class:`matplotlib.figure.Figure`""",
    }
    # Option descriptions
    _rst_descriptions = {
        "AdjustBottom": """Figure-scale coordinates of bottom of axes""",
        "AdjustLeft": """Figure-scale coordinates of left side of axes""",
        "AdjustRight": """Figure-scale coordinates of right side of axes""",
        "AdjustTop": """Figure-scale coordinates of top of axes""",
        "AxesOptions": """Options to :class:`AxesSubplot`""",
        "BottomSpine": "Turn on/off bottom plot spine",
        "BottomSpineMax": "Maximum *x* coord for bottom plot spine",
        "BottomSpineMin": "Minimum *x* coord for bottom plot spine",
        "BottomSpineTicks": "Turn on/off labels on bottom spine",
        "BottomSpineOptions": "Additional options for bottom spine",
        "BottomTickLabels": "Turn on/off tick labels on bottom spine",
        "Density": """Option to scale histogram plots""",
        "ErrorBarMarker": """Marker for :func:`errorbar` plots""",
        "ErrorBarOptions": """Options for :func:`errorbar` plots""",
        "ErrorOptions": """Options for error plots""",
        "ErrorPlotType": """Plot type for "error" plots""",
        "FigDPI": "Figure resolution in dots per inch",
        "FigHeight": "Figure height [inches]",
        "FigNumber": "Figure number",
        "FigOptions": """Options to :class:`matplotlib.figure.Figure`""",
        "FigWidth": "Figure width [inches]",
        "FillBetweenOptions": """Options for :func:`fill_between` plots""",
        "FontName": """Font name (categories like ``sans-serif`` allowed)""",
        "FontOptions": """Options to :class:`FontProperties`""",
        "FontSize": """Font size (options like ``"small"`` allowed)""",
        "FontStretch": ("""Stretch, numeric in range 0-1000 or """ +
            """string such as ``"condensed"``, ``"extra-condensed"``, """ +
            """``"semi-expanded"``"""),
        "FontStyle": """Font style/slant""",
        "FontVariant": """Font capitalization variant""",
        "FontWeight": ("""Numeric font weight 0-1000 or ``"normal"``, """ +
            """``"bold"``, etc."""),
        "Grid": """Option to turn on/off axes grid""",
        "GridOptions": """Plot options for major grid""",
        "Index": """Index to select specific option from lists""",
        "Label": """Label passed to :func:`plt.legend`""",
        "LeftSpine": "Turn on/off left plot spine",
        "LeftSpineMax": "Maximum *y* coord for left plot spine",
        "LeftSpineMin": "Minimum *y* coord for left plot spine",
        "LeftSpineTicks": "Turn on/off labels on left spine",
        "LeftSpineOptions": "Additional options for left spine",
        "LeftTickLabels": "Turn on/off tick labels on left spine",
        "Legend": "Turn on/off (auto) legend",
        "LegendAnchor": "Location passed to *bbox_to_anchor*",
        "LegendFontName": "Font name (categories like ``sans-serif`` allowed)",
        "LegendFontOptions": "Options to :class:`FontProperties`",
        "LegendFontSize": 'Font size (options like ``"small"`` allowed)',
        "LegendFontStretch": ("""Stretch, numeric in range 0-1000 or """ +
            """string such as ``"condensed"``, ``"extra-condensed"``, """ +
            """``"semi-expanded"``"""),
        "LegendFontStyle": """Font style/slant""",
        "LegendFontVariant": """Font capitalization variant""",
        "LegendFontWeight": ("""Numeric font weight 0-1000 or """ +
            """``"normal"``, ``"bold"``, etc."""),
        "LegendLocation": """Numeric location or abbreviation""",
        "LegendOptions": """Options to :func:`plt.legend`""",
        "MajorGrid": """Option to turn on/off grid at main ticks""",
        "MarginBottom": "Figure fraction from bottom edge to bottom label",
        "MarginLeft": "Figure fraction from left edge to left-most label",
        "MarginRight": "Figure fraction from right edge to right-most label",
        "MarginTop": "Figure fraction from top edge to top-most label",
        "MinMaxOptions": "Options for error-bar or fill-between min/max plot",
        "MinMaxPlotType": """Plot type for min/max plot""",
        "MinorGrid": """Turn on/off grid at minor ticks""",
        "MinorGridOptions": """Plot options for minor grid""",
        "Pad": "Padding to add to both axes, *ax.set_xlim* and *ax.set_ylim*",
        "PlotColor": """Color option to :func:`plt.plot` for primary curve""",
        "PlotOptions": """Options to :func:`plt.plot` for primary curve""",
        "PlotLineStyle": """Line style for primary :func:`plt.plot`""",
        "PlotLineWidth": """Line width for primary :func:`plt.plot`""",
        "RightSpine": "Turn on/off right plot spine",
        "RightSpineMax": "Maximum *y* coord for right plot spine",
        "RightSpineMin": "Minimum *y* coord for right plot spine",
        "RightSpineTicks": "Turn on/off labels on right spine",
        "RightSpineOptions": "Additional options for right spine",
        "RightTickLabels": "Turn on/off tick labels on right spine",
        "Rotate": """Option to flip *x* and *y* axes""",
        "ShowError": """Show "error" plot using *xerr*""",
        "ShowMinMax": """Plot *ymin* and *ymax* at each point""",
        "ShowUncertainty": """Plot uncertainty bounds""",
        "Subplot": "Subplot index (1-based)",
        "SubplotCols": "Expected number of subplot columns",
        "SubplotRows": "Expected number of subplot rows",
        "TopSpine": "Turn on/off top plot spine",
        "TopSpineMax": "Maximum *x* coord for top plot spine",
        "TopSpineMin": "Minimum *x* coord for top plot spine",
        "TopSpineTicks": "Turn on/off labels on top spine",
        "TopSpineOptions": "Additional options for top spine",
        "TopTickLabels": "Turn on/off tick labels on top spine",
        "UncertaintyOptions": """Options for UQ plots""",
        "UncertaintyPlotType": """Plot type for UQ plots""",
        "XLabel": """Label to put on *x* axis""",
        "XLim": """Limits for min and max value of *x*-axis""",
        "XLimMax": """Min value for *x*-axis in plot""",
        "XLimMin": """Max value for *x*-axis in plot""",
        "XPad": """Extra padding to add to *x* axis limits""",
        "YLabel": """Label to put on *y* axis""",
        "YLim": """Limits for min and max value of *y*-axis""",
        "YLimMax": """Min value for *y*-axis in plot""",
        "YLimMin": """Max value for *y*-axis in plot""",
        "YPad": """Extra padding to add to *y* axis limits""",
        "ax": """Handle to existing axes""",
        "fig": """Handle to existing figure""",
    }
    
   # --- RC ---
    # Default values
    _rc = {
        "ShowLine": True,
        "ShowError": False,
        "Index": 0,
        "Rotate": False,
        "MinMaxPlotType": "FillBetween",
    }

    # Default figure options
    _rc_figopts = {
        "figwidth": 5.5,
        "figheight": 4.4,
    }
    _rc_figure = {}

    # Default axes options
    _rc_axopts = {}
    _rc_axes = {}
    _rc_axformat = {
        "Pad": 0.05,
    }
    _rc_axadjust = {}

    # Default options for plot
    _rc_plot = {
        "color": ["b", "k", "darkorange", "g"],
        "ls": "-",
        "zorder": 8,
    }
    # Default options for errobar()/fill_between() plots
    _rc_error = {}
    _rc_minmax = {}
    _rc_uq = {}
    # Options for fill_between
    _rc_fillbetween = {
        "alpha": 0.2,
        "lw": 0,
        "zorder": 4,
    }
    # Options for errobar()
    _rc_errorbar = {
        "capsize": 1.5,
        "lw": 0.5,
        "elinewidth": 0.8,
        "zorder": 6,
    }

    # Options for grid()
    _rc_grid = {
        "Grid": True,
        "MajorGrid": True,
    }
    # Formatting for grid lines
    _rc_majorgrid = {
        "ls": ":",
        "lw": 0.5,
        "color": "#a0a0a0",
    }
    _rc_minorgrid = {}

    # Default options for histogram
    rc_hist = {
        "facecolor": 'c',
        "zorder": 2,
        "bins": 20,
        "density": True,
        "edgecolor": 'k',
    }
    
    # Default legend options
    _rc_legend = {
        "loc": "upper center",
        "labelspacing": 0.5,
        "framealpha": 1.0,
    }
    
    # Default font properties
    _rc_font = {
        "family": "DejaVu Sans",
    }
    
    # Font properties for legend
    _rc_legend_font = dict(
        _rc_font, size=None)

    # Default options for spines
    _rc_spines = {
        "Spines": True,
        "Ticks": True,
        "TickDirection": "out",
        "RightSpineTicks": False,
        "TopSpineTicks": False,
        "RightSpine": False,
        "TopSpine": False,
    }
    
    # Default options for mean plot
    rc_mu = {
        "color": 'k',
        "lw": 2,
        "zorder": 6,
        "label": "Mean value",
    }
    
    # Default options for gaussian plot
    rc_gauss = {
        "color": "navy",
        "lw": 1.5,
        "zorder": 7,
        "label": "Normal Distribution",
    }
    
    # Default options for interval plot
    rc_interval = {
        "color": "b",
        "lw": 0,
        "zorder": 1,
        "alpha": 0.2,
        "imin": 0.,
        "imax": 5.,
    }
    
    # Default options for standard deviation plot
    rc_std = {
        'color': 'navy',
        'lw': 2,
        'zorder': 5,
        "dashes": [4, 2],
        'StDev': 3,
    }
    # Default options for delta plot on histograms
    rc_delta = {
        'color': "r",
        'ls': "--",
        'lw': 1.0,
        'zorder': 3,
    }
    
    # Default histogram label options
    rc_histlbl = {
        'color': 'k',
        'horizontalalignment': 'right',
        'verticalalignment': 'top',
    }
    kw_figure = [
        "FigHeight",
        "FigWidth"
    ]
  # >
  
  # ============
  # Config
  # ============
  # <
    # Initialization method
    def __init__(self, optsdict=None, warnmode=1, **kw):
        r"""Initialization method

        :Call:
            >>> opts.__init__(optsdict=None, warnmode=1, **kw)
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
            *optsdict*: {``None``} | :class:`dict`
                Dictionary of previous options (overwritten by *kw*)
            *warnmode*: ``0`` | {``1``} | ``2``
                Warning mode from :mod:`kwutils`
        :Versions:
            * 2019-12-19 ``@ddalle``: First version
        """
        # Get class
        cls = self.__class__
        # Initialize an unfiltered dict
        if isinstance(optsdict, dict):
            # Initialize with dictionary
            optsdict = dict(optsdict, **kw)
        else:
            # Initialize from just keywords
            optsdict = kw
        # Remove anything that's ``None``
        opts = cls.denone(optsdict)

        # Check keywords
        opts = kwutils.check_kw_eltypes(
            cls._optlist,
            cls._optmap,
            cls._opttypes,
            {}, warnmode, **opts)

        # Copy entries
        for (k, v) in opts.items():
            self[k] = v
  # >

  # ============
  # Utilities
  # ============
  # <
    # Remove ``None`` keys
    @staticmethod
    def denone(opts):
        """Remove any keys whose value is ``None``
    
        :Call:
            >>> opts = denone(opts)
        :Inputs:
            *opts*: :class:`dict`
                Any dictionary
        :Outputs:
            *opts*: :class:`dict`
                Input with any keys whose value is ``None`` removed;
                returned for convenience but input is also affected
        :Versions:
            * 2019-03-01 ``@ddalle``: First version
            * 2019-12-19 ``@ddalle``: From :mod:`mplopts`
        """
        # Loop through keys
        for (k, v) in dict(opts).items():
            # Check if ``None``
            if v is None:
                opts.pop(k)
        # Output
        return opts

    # Select options for phase *i*
    @classmethod
    def select_plotphase(cls, kw, i=0):
        r"""Select option *i* for each option in *kw*
    
        This cycles through lists of options for named options such as *color* and
        repeats if *i* is longer than the list of options.  Special options like
        *dashes* that accept a list as a value are handled automatically.  If the
        value of the option is a single value, it is returned regardless of the
        value of *i*
    
        :Call:
            >>> kw_p = select_plotphase(kw, i=0)
        :Inputs:
            *kw*: :class:`dict`
                Dictionary of options for one or more graphics features
            *i*: {``0``} | :class:`int`
                Index
        :Outputs:
            *kw_p*: :class:`dict`
                Dictionary of options with lists replaced by scalar values
        :Versions:
            * 2019-03-04 ``@ddalle``: First version
        """
        # Initialize plot options
        kw_p = {}
        # Loop through options
        for (k, V) in kw.items():
            # Check if this is a "list" option
            if k in cls._optlist_list:
                # Get value as a list
                v = optitem.getringel_list(V, i)
            else:
                # Get value as a scalar
                v = optitem.getringel(V, i)
            # Set option
            kw_p[k] = v
        # Output
        return kw_p
  # >
  
  # ================
  # Categories
  # ================
  # <
    # Figure options
    def figure_options(self):
        r"""Process options specific to Matplotlib figure

        :Call:
            >>> kw = figure_options()
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`plt.figure`
        :Versions:
            * 2019-03-06 ``@ddalle``: First version
            * 2019-12-20 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
        """
        # Class
        cls = self.__class__
        # Submap
        kw_map = cls._kw_submap["FigOptions"]
        # Get top-level options
        kw_fig = self.get("FigOptions", {})
        # Apply defaults
        kw_fig = dict(cls._rc_figopts, **kw_fig)
        # Individual options
        for (k, kp) in kw_map.items():
            # Check if present
            if k not in self:
                continue
            # Remove option and save it under shortened name
            kw_fig[kp] = self[k]
        # Save figure options
        kw = dict(cls._rc_figure, FigOptions=cls.denone(kw_fig))
        # Loop through other options
        for k in cls._optlist_fig:
            # Check applicability
            if k not in self:
                # Not present
                continue
            elif k in kw_map:
                # Already mapped to fig() opts
                continue
            # Otherwise, assign the value
            kw[k] = self[k]
        # Output
        return cls.denone(kw)

    # Axes options
    def axes_options(self):
        r"""Process options for axes handle

        :Call:
            >>> kw = opts.axes_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`plt.axes`
        :Versions:
            * 2019-03-07 ``@ddalle``: First version
            * 2019-12-20 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
        """
        # Class
        cls = self.__class__
        # Submap
        kw_map = cls._kw_submap["AxesOptions"]
        # Get top-level options
        kw_ax = self.get("AxesOptions", {})
        # Apply defaults
        kw_ax = dict(cls._rc_axopts, **kw_ax)
        # Individual options
        for (k, kp) in kw_map.items():
            # Check if present
            if k not in self:
                continue
            # Remove option and save it under shortened name
            kw_ax[kp] = self[k]
        # Save figure options
        kw = dict(cls._rc_axes, AxesOptions=cls.denone(kw_ax))
        # Loop through other options
        for k in cls._optlist_axes:
            # Check applicability
            if k not in self:
                # Not present
                continue
            elif k in kw_map:
                # Already mapped to fig() opts
                continue
            # Otherwise, assign the value
            kw[k] = self[k]
        # Output
        return cls.denone(kw)

    # Global font options
    def font_options(self):
        r"""Process global font options

        :Call:
            >>> kw = opts.font_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of font property options
        :Versions:
            * 2019-03-07 ``@ddalle``: First version
            * 2019-12-19 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
        """
        # Class
        cls = self.__class__
        # Submap
        kw_map = cls._kw_submap["FontOptions"]
        # Get top-level options
        kw_font = self.get("FontOptions", {})
        # Apply defaults
        kw = dict(cls._rc_font, **kw_font)
        # Individual options
        for (k, kp) in kw_map.items():
            # Check if present
            if k not in self:
                continue
            # Remove option and save it under shortened name
            kw[kp] = self[k]
        # Remove "None"
        return cls.denone(kw)

    # Primary options
    def plot_options(self):
        r"""Process options to primary plot curve

        :Call:
            >>> kw = opts.plot_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`plot`
        :Versions:
            * 2019-03-07 ``@ddalle``: First version
            * 2019-12-19 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
        """
        # Class
        cls = self.__class__
        # Submap
        kw_map = cls._kw_submap["PlotOptions"]
        # Aliases
        kw_alias = cls._kw_subalias["PlotOptions"]
        # Get top-level options
        kw_plt = self.get("PlotOptions", {})
        # Apply aliases
        kw_plot = {
            kw_alias.get(k, k): v
            for (k, v) in kw_plt.items()
        }
        # Apply defaults
        kw = dict(cls._rc_plot, **kw_plot)
        # Individual options
        for (k, kp) in kw_map.items():
            # Check if present
            if k not in self:
                continue
            # Remove option and save it under shortened name
            kw[kp] = self[k]
        # Remove "None"
        return cls.denone(kw)

    # Process options for min/max plot
    def minmax_options(self):
        r"""Process options for min/max plots

        :Call:
            >>> minmax_type, kw = opts.minmax_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *minmax_type*: {``"FillBetween"``} | ``"ErrorBar"``
                Plot type for min/max plot
            *kw*: :class:`dict`
                Dictionary of options to :func:`plot`
        :Versions:
            * 2019-03-04 ``@ddalle``: First version
            * 2019-12-20 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
        """
        # Get min/max plot options
        opts = self.get("MinMaxOptions", {})
        # Class
        cls = self.__class__
        # Default type
        tmmx = cls._rc.get("MinMaxPlotType", "FillBetween")
        # Specified type
        tmmx = self.get("MinMaxPlotType", tmmx)
        # Simplify case for comparison
        t = tmmx.lower().replace("_", "")
        # Submap
        kw_map = cls._kw_submap["MinMaxOptions"]
        # Get top-level options
        kw_mmx = self.get("MinMaxOptions", {})
        # Apply defaults
        kw = dict(cls._rc_minmax, **kw_mmx)
        # Individual options
        for (k, kp) in kw_map.items():
            # Check if present
            if k not in self:
                continue
            # Remove option and save it under shortened name
            kw[kp] = self[k]
        # Fitler type
        if t == "fillbetween":
            # Region plot
            minmax_type = "FillBetween"
            # Get options for :func:`fill_between`
            kw_plt = self.fillbetween_options()
        elif t == "errorbar":
            # Error bars
            minmax_type = "ErrorBar"
            # Get options for :func:`errorbar`
            kw_plt = self.errorbar_options()
        else:
            raise ValueError("Unrecognized min/max plot type '%s'" % tmmx)
        # MinMaxOptions overrides
        kw = dict(kw_plt, **kw)
        # Output
        return minmax_type, cls.denone(kw)

    # Process options for "error" plot
    def error_options(self):
        r"""Process options for error plots

        :Call:
            >>> error_type, kw = opts.error_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *error_type*: ``"FillBetween"`` | {``"ErrorBar"``}
                Plot type for error plot
            *kw*: :class:`dict`
                Dictionary of options to :func:`plot`
        :Versions:
            * 2019-03-04 ``@ddalle``: First version
            * 2019-12-23 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
        """
        # Get min/max plot options
        opts = self.get("ErrorOptions", {})
        # Class
        cls = self.__class__
        # Default type
        terr = cls._rc.get("ErrorPlotType", "ErrorBar")
        # Specified type
        terr = self.get("ErrorPlotType", terr)
        # Simplify case for comparison
        t = terr.lower().replace("_", "")
        # Submap
        kw_map = cls._kw_submap["ErrorOptions"]
        # Get top-level options
        kw_err = self.get("ErrorOptions", {})
        # Apply defaults
        kw = dict(cls._rc_error, **kw_err)
        # Individual options
        for (k, kp) in kw_map.items():
            # Check if present
            if k not in self:
                continue
            # Remove option and save it under shortened name
            kw[kp] = self[k]
        # Fitler type
        if t == "fillbetween":
            # Region plot
            error_type = "FillBetween"
            # Get options for :func:`fill_between`
            kw_plt = self.fillbetween_options()
        elif t == "errorbar":
            # Error bars
            error_type = "ErrorBar"
            # Get options for :func:`errorbar`
            kw_plt = self.errorbar_options()
        else:
            raise ValueError("Unrecognized min/max plot type '%s'" % terr)
        # MinMaxOptions overrides
        kw = dict(kw_plt, **kw)
        # Output
        return error_type, cls.denone(kw)

    # Process options for UQ plot
    def uq_options(self):
        r"""Process options for uncertainty quantification plots

        :Call:
            >>> uq_type, kw = opts.uq_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *uq_type*: {``"FillBetween"``} | ``"ErrorBar"``
                Plot type for UQ plot
            *kw*: :class:`dict`
                Dictionary of options to plot function
        :Versions:
            * 2019-03-04 ``@ddalle``: First version
            * 2019-12-23 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
        """
        # Get min/max plot options
        opts = self.get("UncertaintyOptions", {})
        # Class
        cls = self.__class__
        # Default type
        tuq = cls._rc.get("UncertaintyPlotType", "FillBetween")
        # Specified type
        tuq = self.get("UncertaintyPlotType", tuq)
        # Simplify case for comparison
        t = tuq.lower().replace("_", "")
        # Submap
        kw_map = cls._kw_submap["UncertaintyOptions"]
        # Get top-level options
        kw_uq = self.get("UncertaintyOptions", {})
        # Apply defaults
        kw = dict(cls._rc_uq, **kw_uq)
        # Individual options
        for (k, kp) in kw_map.items():
            # Check if present
            if k not in self:
                continue
            # Remove option and save it under shortened name
            kw[kp] = self[k]
        # Fitler type
        if t == "fillbetween":
            # Region plot
            uq_type = "FillBetween"
            # Get options for :func:`fill_between`
            kw_plt = self.fillbetween_options()
        elif t == "errorbar":
            # Error bars
            uq_type = "ErrorBar"
            # Get options for :func:`errorbar`
            kw_plt = self.errorbar_options()
        else:
            raise ValueError("Unrecognized min/max plot type '%s'" % tuq)
        # MinMaxOptions overrides
        kw = dict(kw_plt, **kw)
        # Output
        return uq_type, cls.denone(kw)

    # Options for errorbar() plots
    def errorbar_options(self):
        r"""Process options for :func:`errorbar` calls

        :Call:
            >>> kw = opts.errorbar_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`errorbar`
        :Versions:
            * 2019-03-05 ``@ddalle``: First version
            * 2019-12-21 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
        """
        # Class
        cls = self.__class__
        # Submap (global options mapped to errorbar() opts)
        kw_map = cls._kw_submap["ErrorBarOptions"]
        # Aliases for errorbar() opts to avoid conflict
        kw_alias = cls._kw_subalias["ErrorBarOptions"]
        # Options to cascade css-style from PlotOptions
        kw_css = cls._kw_cascade["ErrorBarOptions"]
        # Get directly specified
        kw_eb = self.get("ErrorBarOptions", {})
        # Apply aliases
        kw = {
            kw_alias.get(k, k): v
            for (k, v) in kw_eb.items()
        }
        # Get :func:`plot` options
        kw_plt = self.plot_options()
        # loop through cascading options
        for (k2, k1) in kw_css.items():
            # Split "from" name into part and option
            ka, kb = k2.split(".", 1)
            # Confirm it comes from "plot"
            if ka == "plot":
                v = kw_plt.get(kb)
            else:
                raise ValuError(
                    "ErrorBarOptions cannot inherit from '%s'" % ka)
            # Check for valid value
            if v is not None:
                # Don't override specified value
                kw.setdefault(k1, v)
        # Apply defaults
        kw = dict(cls._rc_errorbar, **kw)
        # Individual options
        for (k, kp) in kw_map.items():
            # Check if present
            if k not in self:
                continue
            # Remove option and save it under shortened name
            kw[kp] = self[k]
        # Remove "None"
        return cls.denone(kw)

    # Options for fill_between() plots
    def fillbetween_options(self):
        r"""Process options for :func:`fill_between` calls

        :Call:
            >>> kw = opts.fillbetween_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`fill_between`
        :Versions:
            * 2019-03-05 ``@ddalle``: First version
            * 2019-12-21 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
        """
        # Class
        cls = self.__class__
        # Submap (global options mapped to errorbar() opts)
        kw_map = cls._kw_submap["FillBetweenOptions"]
        # Aliases for errorbar() opts to avoid conflict
        kw_alias = cls._kw_subalias["FillBetweenOptions"]
        # Options to cascade css-style from PlotOptions
        kw_css = cls._kw_cascade["FillBetweenOptions"]
        # Get directly specified
        kw_eb = self.get("FillBetweenOptions", {})
        # Apply aliases
        kw = {
            kw_alias.get(k, k): v
            for (k, v) in kw_eb.items()
        }
        # Get :func:`plot` options
        kw_plt = self.plot_options()
        # loop through cascading options
        for (k2, k1) in kw_css.items():
            # Split "from" name into part and option
            ka, kb = k2.split(".", 1)
            # Confirm it comes from "plot"
            if ka == "plot":
                v = kw_plt.get(kb)
            else:
                raise ValuError(
                    "FillBetweenOptions cannot inherit from '%s'" % ka)
            # Check for valid value
            if v is not None:
                # Don't override specified value
                kw.setdefault(k1, v)
        # Apply defaults
        kw = dict(cls._rc_fillbetween, **kw)
        # Individual options
        for (k, kp) in kw_map.items():
            # Check if present
            if k not in self:
                continue
            # Remove option and save it under shortened name
            kw[kp] = self[k]
        # Remove "None"
        return cls.denone(kw)

    # Process axes formatting options
    def axformat_options(self):
        r"""Process options for axes format

        :Call:
            >>> kw = opts.axformat_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`axes_format`
        :Versions:
            * 2019-03-07 ``@jmeeroff``: First version
        """
        # Class
        cls = self.__class__
        # Initialize output
        kw = {}
        # Loop through other options
        for k in cls._optlist_axformat:
            # Check applicability
            if k not in self:
                # Not present
                continue
            # Otherwise, assign the value
            kw[k] = self[k]
        # Get rotation option
        rotate = kw.get("Rotate", False)
        # Get density option
        density = kw.get("Density")
        # Different defaults for histograms
        if density is None:
            # No default label
            ylbl = None
        elif density:
            # Default label for PDF
            ylbl = "Probability Density"
        else:
            # Raw histogram option
            ylbl = "Count"
        # Process which axis this default applies to
        if rotate:
            # Default
            xlbl = None
        else:
            # Data on horizontal axis
            xlbl = ylbl
            ylbl = None
        # Apply defaults
        kw = dict(cls._rc_axformat, **kw)
        # Return
        return cls.denone(kw)

    # Process axes formatting options
    def axadjust_options(self):
        r"""Process options for axes margin adjustment

        :Call:
            >>> kw = opts.axadjust_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`axes_adjust`
        :Versions:
            * 2020-01-08 ``@ddalle``: First version
        """
        # Class
        cls = self.__class__
        # Initialize output
        kw = {}
        # Loop through other options
        for k in cls._optlist_axadjust:
            # Check applicability
            if k not in self:
                # Not present
                continue
            # Otherwise, assign the value
            kw[k] = self[k]
        # Apply defaults
        kw = dict(cls._rc_axadjust, **kw)
        # Return
        return cls.denone(kw)

    # Grid options
    def grid_options(self):
        r"""Process options to axes :func:`grid` command

        :Call:
            >>> kw = opts.grid_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`grid`
            *kw["MajorGrid"]*: ``True`` | ``False``
                Plot major grid
            *kw["MajorGridOptions"]*: :class:`dict`
                Options to :func:`plt.grid` for major grid
            *kw["MinorGrid"]*: ``True`` | ``False``
                Plot minor grid
            *kw["MinorGridOptions"]*: :class:`dict`
                Options to :func:`plt.grid` for minor grid
        :Versions:
            * 2019-03-07 ``@jmeeroff``: First version
            * 2019-12-23 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
        """
        # Class
        cls = self.__class__
        # Submap
        kw_map = cls._kw_submap["GridOptions"]
        # Aliases
        kw_alias = cls._kw_subalias["GridOptions"]
        # Get top-level options
        kw_maj = self.get("GridOptions", {})
        kw_min = self.get("MinorGridOptions", {})
        # Apply aliases
        kw_major = {
            kw_alias.get(k, k): v
            for (k, v) in kw_maj.items()
        }
        kw_minor = {
            kw_alias.get(k, k): v
            for (k, v) in kw_min.items()
        }
        # Individual options
        for (k, kp) in kw_map.items():
            # Check if present
            if k not in self:
                continue
            # Remove option and save it under shortened name
            kw_major[kp] = self[k]
        # Initialize output
        kw = {}
        # Loop through primary options
        for k in cls._optlist_grid:
            # Check applicability
            if k not in self:
                # Not present
                continue
            elif k in kw_map:
                # Already mapped to fig() opts
                continue
            # Otherwise, assign the value
            kw[k] = self[k]
        # Apply defaults
        kw_minor = dict(cls._rc_minorgrid, **kw_minor)
        kw_major = dict(cls._rc_majorgrid, **kw_major)
        # Apply overall defaults
        kw = dict(cls._rc_grid, **kw)
        # Ensure major and minor oprionts
        kw["MajorGridOptions"] = cls.denone(kw_major)
        kw["MinorGridOptions"] = cls.denone(kw_minor)
        # Remove "None"
        return cls.denone(kw)

    # Spine options
    def spine_options(self):
        r"""Process options for axes "spines"

        :Call:
            >>> kw = opts.spine_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to each of four spines
        :Versions:
            * 2019-03-07 ``@jmeeroff``: First version
            * 2019-12-20 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
        """
        # Class
        cls = self.__class__
        # Submap (global options -> AxesOptions)
        kw_map = cls._kw_submap["SpineOptions"]
        # Initialize
        kw = {}
        # Loop through other options
        for k in cls._optlist_spines:
            # Check applicability
            if k not in self:
                # Not present
                continue
            elif k in kw_map:
                # Already mapped to fig() opts
                continue
            # Otherwise, assign the value
            kw[k] = self[k]
        # Apply defaults
        kw = dict(cls._rc_spines, **kw)
        # Loop through map options
        for (k1, k2) in kw_map.items():
            # Check if the option is specified by the user
            if k1 not in self:
                continue
            # Check prefix
            if k.startswith("XTick"):
                optgroup = "XTickOptions"
            elif k.startswith("YTick"):
                optgroup = "YTickOptions"
            elif k.startswith("Tick"):
                optgroup = "TickOptions"
            else:
                continue
            # Get appropriate subgroup (creating if necessary)
            opts = kw.setdefault(optgroup, {})
            # Apply the option
            opts[k2] = self[k1]
        # Output
        return cls.denone(kw)

    # Legend options
    def legend_options(self):
        r"""Process options for :func:`legend`
    
        :Call:
            >>> kw = opts.legend_options(kw, kwp={})
        :Inputs:
            *kw*: :class:`dict`
                Dictionary of options to parent function
            *kwp*: {``{}``}  | :class:`dict`
                Dictionary of options from which to inherit
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Options to :func:`legend`
        :Versions:
            * 2019-03-07 ``@ddalle``: First version
            * 2019-12-23 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
        """
        # Class
        cls = self.__class__
        # Submap (global options -> LegendOptions)
        kw_map = cls._kw_submap["LegendOptions"]
        # Font submapt (global FontOptions -> LegendOptions["prop"])
        kw_fontmap = cls._kw_submap["FontOptions"]
        # Get overall options
        kw_font = self.font_options()
        # Initialize the font properties
        prop = self.get("LegendFontOptions", {})
        # Apply defaults
        prop = dict(cls._rc_legend_font, **prop)
        # Loop through font options
        for (k1, k2) in kw_fontmap.items():
            # Prepend "Legend" to name
            ka = "Legend" + k1
            # Check if present
            if ka not in self:
                continue
            # Otherwise assign it
            prop[k2] = self[ka]
        # Get *LegendOptions*
        kw = self.get("LegendOptions", {})
        # Apply defaults
        kw = dict(cls._rc_legend, **kw)
        # Set font properties
        kw["prop"] = cls.denone(prop)
        # Individual options
        for (k, kp) in kw_map.items():
            # Check if present
            if k not in self:
                continue
            # Remove option and save it under shortened name
            kw[kp] = self[k]
        # Global on/off option
        kw["ShowLegend"] = self.get("ShowLegend")
        # Specific location options
        loc = kw.get("loc")
        # Check it
        if loc in ["upper center", 9]:
            # Bounding box location on top spine
            kw.setdefault("bbox_to_anchor", (0.5, 1.05))
        elif loc in ["lower center", 8]:
            # Bounding box location on bottom spine
            kw.setdefault("bbox_to_anchor", (0.5, -0.05))
        # Output
        return cls.denone(kw)
  # >

  # =========================
  # Docstring Manipulation
  # =========================
  # <
    # Loop through functions to rename
    for (fn, optlist) in [
        (axes_options, _optlist_axes),
        (error_options, _optlist_error),
        (errorbar_options, _optlist_errobar),
        (figure_options, _optlist_fig),
        (fillbetween_options, _optlist_fillbetween),
        (font_options, _optlist_font),
        (grid_options, _optlist_grid),
        (legend_options, _optlist_legend),
        (plot_options, _optlist_plot),
        (uq_options, _optlist_uq)
    ]:
        # Create string to replace "%(keys)s" with
        _doc_rst = rstutils.rst_param_list(
            optlist,
            _rst_types,
            _rst_descriptions,
            _optmap,
            indent=12)
        # Apply text to the docstring
        fn.__doc__ = fn.__doc__ % {"keys": _doc_rst}
  # >

# Delete local variables during documentation process
delattr(MPLOpts, "fn")
delattr(MPLOpts, "optlist")
delattr(MPLOpts, "_doc_rst")
