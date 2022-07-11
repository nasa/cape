#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
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


# Close a figure
def close(fig=None):
    _import_pyplot()
    plt.close(fig)


# Figure part
def figure(**kw):
    r"""Get or create figure handle and format it

    :Call:
        >>> fig = figure(**kw)
    :Inputs:
        %(keys)s
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
        %(keys)s
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


# Label axes in one of 12 special positions
def axlabel(lbl, pos=None, **kw):
    r"""Create a label for an axes object

    The *pos* integer implies positions represented by the following
    text diagram::

          0            6            1
        +-----------------------------+
        | 2            7            3 | 10
        |                             |
        | 17           9           16 | 15
        |                             |
        | 4            8            5 | 11
        +-----------------------------+
          12          14           13

    :Call:
        >>> h = axlabel(lbl, pos=None, **kw):
    :Inputs:
        *lbl*: :class:`str`
            Text of label to add
        *pos*: :class:`int`
            Index for label position
    :Keyword Arguments:
        %(keys)s
    :Outputs:
        *h*: :class:`matplotlib.text.Text`
            Matplotlib ``Text`` instance
        *h._label*: :class:`str`
            Set to ``"<axlabel>"`` for automatic detection
    :Versions:
        * 2020-04-29 ``@ddalle``: First version
    """
    # Process options
    opts = MPLOpts(_section="axlabel", **kw)
    # Get axlabel options
    kw_lbl = opts.get_option("AxesLabelOptions")
    # Set default position
    kw_lbl.setdefault("pos", pos)
    # Remove it
    ax = kw_lbl.pop("ax", None)
    # Call root function
    return _axlabel(ax, lbl, **kw_lbl)


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
    :Keyword Arguments:
        %(keys)s
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


# Semilogy function with options check
def semilogy(xv, yv, fmt=None, **kw):
    r"""Call the :func:`semilogy` function with cycling options

    :Call:
        >>> h = semilogy(xv, yv, **kw)
        >>> h = semilogy(xv, yv, fmt, **kw)
    :Inputs:
        *xv*: :class:`np.ndarray`
            Array of *x*-coordinates
        *yv*: :class:`np.ndarray`
            Array of *y*-coordinates
        *fmt*: :class:`str`
            Optional format option
    :Keyword Arguments:
        %(keys)s
    :Outputs:
        *h*: :class:`list` (:class:`matplotlib.lines.Line2D`)
            List of line instances
    :Versions:
        * 2021-01-05 ``@ddalle``: Version 1.0; fork from plot()
    """
    # Process options
    opts = MPLOpts(_section="plot", **kw)
    # Get plot options
    kw_p = opts.plot_options()
    # Call root function
    return _semilogy(xv, yv, fmt=fmt, **kw_p)


# Contour function with options check
def contour(xv, yv, zv, **kw):
    r"""Call the :func:`contour` function with cycling options

    :Call:
        >>> hc, hl = contour(xv, yv, zv, **kw)
    :Inputs:
        *xv*: :class:`np.ndarray`
            Array of *x*-coordinates
        *yv*: :class:`np.ndarray`
            Array of *y*-coordinates
        *zv*: :class:`np.ndarray`
            Array of contour levels
    :Keyword Arguments:
        %(keys)s
    :Outputs:
        *hc*: :class:`matplotlib.tri.tricontour.TriContourSet`
            Unstructured contour handles
        *hl*: :class:`list`\ [:class:`matplotlib.lines.Line2D`]
            List of line instances
    :Versions:
        * 2020-03-26 ``@jmeeroff``: First version
    """
    # Process options
    opts = MPLOpts(_section="contour", **kw)
    # Get contour options
    kw_p = opts.contour_options()
    # Call root function
    return _contour(xv, yv, zv, **kw_p)


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
    :Keyword Arguments:
        %(keys)s
    :Outputs:
        *h*: :class:`matplotlib.container.ErrorbarContainer`
            Errorbar plot handle
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
        >>> h = fill_between(xv, ymin, ymax, **kw)
    :Inputs:
        *xv*: :class:`np.ndarray`
            Array of independent variable values
        *ymin*: :class:`np.ndarray` | :class:`float`
            Array of values or single value for lower bound of window
        *ymax*: :class:`np.ndarray` | :class:`float`
            Array of values or single value for upper bound of window
    :Keyword Arguments:
        %(keys)s
    :Outputs:
        *h*: :class:`matplotlib.collections.PolyCollection`
            Region plot handle
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




# Histogram function with options check
def hist(v, **kw):
    r"""Call the :func:`hist` function with cycling options

    :Call:
        >>> h = contour(v, **kw)
    :Inputs:
        *v*: :class:`np.ndarray`
            Array of values
    :Keyword Arguments:
        %(keys)s
    :Outputs:
        *h*: :class:`tuple` 
            Tuple of values, bins, patches
    :Versions:
        * 2020-04-23 ``@jmeeroff``: First version
    """
    # Process options
    opts = MPLOpts(_section="hist", **kw)
    # Get hist options
    kw_p = opts.hist_options()
    # Call root function
    return _hist(v, **kw_p)

# Show an image
def imshow(png, **kw):
    r"""Display an image

    :Call:
        >>> img = imshow(fpng, **kw)
        >>> img = imshow(png, **kw)
    :Inputs:
        *fpng*: :class:`str`
            Name of PNG file
        *png*: :class:`np.ndarray`
            Image array from :func:`plt.imread`
    :Keyword Arguments:
        %(keys)s
    :Outputs:
        *img*: :class:`matplotlib.image.AxesImage`
            Image handle
    :Versions:
        * 2020-01-09 ``@ddalle``: First version
        * 2020-01-27 ``@ddalle``: From :mod:`plot_mpl`
    """
    # Process opts
    opts = MPLOpts(_section="imshow", **kw)
    # Get opts for imshow
    kw = opts.imshow_options()
    # Use basic function
    return _imshow(png, **kw)

# Scatter function with options check
def scatter(xv, yv, s=None, c=None, **kw):
    r"""Call the :func:`scatter` function with cycling options

    :Call:
        >>> h = scatter(xv, yv, **kw)
        >>> h = scatter(xv, yv, s=None, c=None, **kw)
    :Inputs:
        *xv*: :class:`np.ndarray`
            Array of *x*-coordinates
        *yv*: :class:`np.ndarray`
            Array of *y*-coordinates
        *s*: :class:`np.ndarray` | :class:`float`
            Size of marker for each data point, in points^2
        *c*: :class:`np.ndarray` | :class:`list`
            Color or color description to use for each data point;
            usually an array of floats that maps into color map
    :Keyword Arguments:
        %(keys)s
    :Outputs:
        *h*: :class:`list` (:class:`matplotlib.lines.Line2D`)
            List of line instances
    :Versions:
        * 2020-02-14 ``@ddalle``: First version
    """
    # Apply optional positional arguments
    if s:
        kw["ScatterSize"] = s
    if c.any():
        kw["ScatterColor"] = c
    # Process options
    opts = MPLOpts(_section="scatter", **kw)
    # Get plot options
    kw_p = opts.get_option("ScatterOptions")
    # Call root function
    return _scatter(xv, yv, **kw_p)


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
    :Keyword Arguments:
        %(keys)s
    :Outputs:
        *ax*: :class:`AxesSubplot`
            Handle to subplot directed to use from these options
    :Versions:
        * 2020-01-03 ``@ddalle``: First version
        * 2010-01-10 ``@ddalle``: Add support for ``"equal"`` aspect
    """
    # Get options
    opts = MPLOpts(_section="axadjust", **kw)
    # Call root function
    return _axes_adjust(fig, **opts)


# Co-align a column of axes
def axes_adjust_col(fig, **kw):
    r"""Adjust a column of axes with shared horizontal extents

    :Call:
        >>> axes_adjust_col(fig, **kw)
    :Inputs:
        *fig*: {``None``} | :class:`Figure` | :class:`int`
            Figure handle or number (default from :func:`plt.gcf`)
        %(keys)s
    :Versions:
        * 2020-01-10 ``@ddalle``: First version
        * 2020-01-27 ``@ddalle``: Added options checks
    """
    # Get options
    opts = MPLOpts(_section="axadjust_col", **kw)
    # Call root function
    return _axes_adjust_col(fig, **kw)


# Co-align a row of axes
def axes_adjust_row(fig, **kw):
    r"""Adjust a row of axes with shared vertical extents

    :Call:
        >>> axes_adjust_row(fig, **kw)
    :Inputs:
        *fig*: {``None``} | :class:`Figure` | :class:`int`
            Figure handle or number (default from :func:`plt.gcf`)
        %(keys)s
    :Versions:
        * 2020-01-10 ``@ddalle``: First version
        * 2020-01-27 ``@ddalle``: Added options checks
    """
    # Get options
    opts = MPLOpts(_section="axadjust_row", **kw)
    # Call root function
    return _axes_adjust_row(fig, **kw)


# Autoscale axes plot window height
def axes_autoscale_height(ax=None, **kw):
    r"""Autoscale height of axes plot window

    For scaling purposes, the *x* limits are taken from
    :func:`ax.get_xlim`.

    :Call:
        >>> axes_autoscale_height(ax, **kw)
    :Inputs:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
        *YPad*, *Pad*: {``0.07``} | :class:`float`
            Padding for *y* axis
        *YMin*, *ymin*: {``None``} | :class:`float`
            Override automatic *ymin* coordinate
        *YMax*, *ymax*: {``None``} | :class:`float`
            Override automatic *ymax* coordinate
    :Versions:
        * 2020-03-16 ``@ddalle``: First version
    """
    # Get options
    opts = MPLOpts(_section="axheight", **kw)
    # Call root function
    return _axes_autoscale_height(ax, **kw)


# Autoscale axes plot window width
def axes_autoscale_width(ax=None, **kw):
    r"""Autoscale width of axes plot window

    For scaling purposes, the *y* limits are taken from
    :func:`ax.get_ylim`.

    :Call:
        >>> axes_autoscale_width(ax, **kw)
    :Inputs:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
        *XPad*, *Pad*: {``0.07``} | :class:`float`
            Padding for *y* axis
        *XMin*, *xmin*: {``None``} | :class:`float`
            Override automatic *xmin* coordinate
        *XMax*, *xmax*: {``None``} | :class:`float`
            Override automatic *xmax* coordinate
    :Versions:
        * 2020-03-16 ``@ddalle``: First version
    """
    # Get options
    opts = MPLOpts(_section="axwidth", **kw)
    # Call root function
    return _axes_autoscale_height(ax, **kw)


# Axes format
def axes_format(ax, **kw):
    r"""Format and label axes

    :Call:
        >>> xl, yl = axes_format(ax, **kw)
    :Inputs:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
    :Keyword Arguments:
        %(keys)s
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
    :Keyword Arguments:
        %(keys)s
    :Versions:
        * 2019-03-07 ``@jmeeroff``: First version
        * 2019-12-23 ``@ddalle``: Updated from :mod:`plotutils`
        * 2020-01-27 ``@ddalle``: Options checker
    """
    # Process options
    opts = MPLOpts(_section="grid", **kw)
    # Call root function
    return _grid(ax, **kw)


# Legend
def legend(ax=None, **kw):
    r"""Create/update a legend

    :Call:
        >>> leg = legend(ax=None, **kw)
    :Inputs:
        *ax*: {``None``} | :class:`matplotlib.axes._subplots.AxesSubplot`
            Axis handle (default is ``plt.gca()``
    :Keyword Arguments:
        %(keys)s
    :Outputs:
        *leg*: :class:`matplotlib.legend.Legend`
            Legend handle
    :Versions:
        * 2019-03-07 ``@ddalle``: First version
        * 2019-08-22 ``@ddalle``: From :func:`Part.legend_part`
        * 2020-01-27 ``@ddalle``: From :func:`plot_mpl.legend`
    """
    # Process options
    opts = MPLOpts(_section="legend", **kw)
    # Call root function
    return _legend(ax, **kw)


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
    :Keyword Arguments:
        %(keys)s
    :Versions:
        * 2019-03-07 ``@jmeeroff``: First version
        * 2019-12-23 ``@ddalle``: From :mod:`tnakit.plotutils`
        * 2020-01-27 ``@ddalle``: From :func:`plot_mpl.format_spines`
    """
    # Get options
    opts = MPLOpts(_section="spines", **kw)
    # Call root function
    return _spines(ax, **opts)


# Get axes handle based on inputs
def get_axes(ax=None):
    r"""Get axes handle from default, handle, or figure 

    :Call:
        >>> ax = get_axes(ax=None)
        >>> ax = get_axes(fig)
    :Inputs:
        *ax*: {``None``} | :class:`Axes`
            Optional prespecified axes handle
        *fig*: :class:`Figure` | :class:`int`
            Figure handle or number
    :Outputs:
        *ax*: :class:`Axes`
            Converted :class:`matplotlib` axes handle
    :Versions:
        * 2021-12-16 ``@ddalle``: Version 1.0
    """
    # Make sure pyplot is present
    _import_pyplot()
    # Default figure
    if ax is None:
        # Get most recent figure or create
        return plt.gca()
    elif isinstance(ax, (int, Figure)):
        # Got a figure instead
        fig = get_figure(fig)
        # Get recent axes from *fig*
        return fig.gca()
    elif isinstance(ax, Axes):
        # Already an axes
        return ax
    # Not an axes, figure, or number
    raise TypeError(
        "'ax' arg expected 'Axes', 'int', or 'Figure' (got %s)" % type(ax))


# Get figure handle based on inputs
def get_figure(fig=None):
    r"""Get figure handle from default, handle, or number

    :Call:
        >>> fig = get_figure(fig=None)
    :Inputs:
        *fig*: {``None``} | :class:`Figure` | :class:`int`
            Figure handle or number
    :Outputs:
        *fig*: :class:`Figure`
            Converted :class:`matplotlib` figure handle
    :Versions:
        * 2020-04-02 ``@ddalle``: First version
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
    # Output
    return fig


# Axis part (initial)
def _axes(**kw):
    r"""Create new axes or edit one if necessary

    :Call:
        >>> ax = axes(**kw)
    :Keyword Arguments:
        %(keys)s
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
    :Keyword Arguments:
        %(keys)s
    :Outputs:
        *ax*: :class:`AxesSubplot`
            Handle to subplot directed to use from these options
    :Versions:
        * 2020-01-03 ``@ddalle``: First version
        * 2020-01-10 ``@ddalle``: Add support for ``"equal"`` aspect
        * 2020-04-23 ``@ddalle``: Process neighboring axes
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
    nmin_ax = max(1, len(ax_list))
    # Get "axes" option
    ax = kw.get("ax")
    # Get subplot number option
    subplot_i = kw.get("Subplot")
    # Get number of rows and columns of figures
    if subplot_i is None:
        # Use existing number of axes
        subplot_m = kw.get("SubplotRows", nmin_ax)
        subplot_n = kw.get("SubplotCols", (nmin_ax+subplot_m-1) // subplot_m)
    else:
        # Allow for *Subplot* to be greater than current count
        subplot_m = kw.get("SubplotRows", max(nmin_ax, subplot_i))
        subplot_n = kw.get("SubplotCols", (nmin_ax+subplot_m-1) // subplot_m)
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
    label_wl, label_hb, label_wr, label_ht = get_axes_label_margins(ax)
    # Get information on neighboring axes
    ax_margins, neighbors = get_axes_neighbors(ax)
    # Unpack axes margins
    axes_wl, axes_hb, axes_wr, axes_ht = ax_margins 
    # Process width and height
    ax_w = 1.0 - label_wr - label_wl - axes_wr - axes_wl
    ax_h = 1.0 - label_ht - label_hb - axes_ht - axes_hb
    # Process row and column space available
    ax_rowh = ax_h
    ax_colw = ax_w
    # Default margins (no tight_layout yet)
    adj_b = label_hb + subplot_j * ax_rowh
    adj_l = label_wl + subplot_k * ax_colw
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
    # Check for multiple axes
    if len(ax_list) > 1:
        # Note that there is ample code for subplots below.
        # It doesn't seem to be relevant but might be removed.
        return
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
    # Calculate shifts of each plot margin
    dxa = adj_l - x0
    dxb = adj_r - x0 - w0
    dya = adj_b - y0
    dyb = adj_t - y0 - h0
    # New positions
    x1 = adj_l
    y1 = adj_b
    w1 = adj_r - adj_l
    h1 = adj_t - adj_b
    # Set new position
    ax.set_position([x1, y1, w1, h1])
    # Loop through neighbors
    for k, axk in enumerate(fig.get_axes()):
        # Skip main axes
        if axk is ax:
            continue
        # Get current bounds
        xk, yk, wk, hk = axk.get_position().bounds
        # Get neighbor information
        neighbor = neighbors[k]
        # Determine horizontal shift (prior to link considerations)
        if neighbor["lshift"]:
            # Shift leftward
            dxk = dxa
        elif neighbor["rshift"]:
            # Shift to the right
            dxk = dxb
        else:
            # No shift
            dxk = 0.0
        # Determine vertical shift (prior to link considerations)
        if neighbor["dshift"]:
            # Shift downward
            dyk = dya
        elif neighbor["ushift"]:
            # Shift upward
            dyk = dyb
        else:
            # No vertical shift
            dyk = 0.0
        # Apply shifts to current bounds
        xk1 = xk + dxk
        yk1 = yk + dyk
        # Check for horizontal linking
        if neighbor["xlink"]:
            # Copy horizontal from *ax*, shift vertical
            print("XLINK")
            axk.set_position([x1, yk1, w1, hk])
        elif neighbor["ylink"]:
            # Copy vertical from *ax*, shift horizontal
            axk.set_position([xk1, y1, wk, h1])
        else:
            # Apply nominal shifts
            axk.set_position([xk1, yk1, wk, hk])
    # Output
    return ax


# Co-align a column of axes
def _axes_adjust_col(fig, **kw):
    r"""Adjust a column of axes with shared horizontal extents

    :Call:
        >>> _axes_adjust_col(fig, **kw)
    :Inputs:
        *fig*: {``None``} | :class:`Figure` | :class:`int`
            Figure handle or number (default from :func:`plt.gcf`)
    :Keyword Arguments:
        %(keys)s
    :Versions:
        * 2020-01-10 ``@ddalle``: First version
        * 2020-01-27 ``@ddalle``: Added options checks
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
    # Process options
    opts = MPLOpts(**kw)
    # Get axes from figure
    ax_all = fig.get_axes()
    # Get list of figures
    subplot_list = kw.get("SubplotList")
    # Default order
    if subplot_list is None:
        # Get current extents
        extents = [ax.get_position().bounds for ax in ax_all]
        # Get middle *y* coordinate for sorting
        y0 = [extent[1] + 0.5*extent[3] for extent in extents]
        # Sort by descending *y0*
        subplot_list = list(np.argsort(y0) + 1)
    # Get the list of axes
    ax_list = [ax_all[i-1] for i in subplot_list]
    # Get index of ax to use for vertical rubber
    subplot_rubber = kw.get("SubplotRubber")
    # Default flexible subplot
    if subplot_rubber is None:
        # Get the ones with ``axis("equal")`` turned on
        ax_auto = [
            j for (j, ax) in enumerate(ax_list)
            if ax.get_aspect() == "auto"
        ]
        # Check for any such figures
        if len(ax_auto) == 0:
            # Nothing to work with; oh boy!
            subplot_rubber = len(ax_list) - 1
        else:
            # Use the last one that's not ``axis("equal")``
            subplot_rubber = ax_auto[-1]
        # Get handle
        ax_rubber = subplot_rubber
    elif isinstance(subplot_rubber, type(ax_list[0])):
        # Get handle
        ax_rubber = subplot_rubber
        # Check if it's in the list
        if ax_rubber not in ax_list:
            raise ValueError("Rubber subplot not in figure's subplot list")
        # Use axis directly
        subplot_rubber = ax_list.index(ax_rubber)
    else:
        # Use index from global list
        if subplot_rubber > 0:
            # Shift from 1-based to 0-based
            ax_rubber = ax_all[subplot_rubber - 1]
        else:
            # Negative indexing from end of list
            ax_rubber = ax_all[subplot_rubber]
        # Check if it's in the list
        if ax_rubber not in ax_list:
            raise ValueError("Rubber subplot not in figure's subplot list")
        # Use axis directly
        subplot_rubber = ax_list.index(ax_rubber)
    # Number of axes in col
    nrows = len(ax_list)
    # Get the margins occupied by tick and axes labels
    margins = [get_axes_label_margins(ax) for ax in ax_list]
    # Extract the sides
    margins_l = [margin[0] for margin in margins]
    margins_b = [margin[1] for margin in margins]
    margins_r = [margin[2] for margin in margins]
    margins_t = [margin[3] for margin in margins]
    # Use the maximum margin for left and right
    wa = max(margins_l)
    wb = max(margins_r)
    # Get extra margins
    margin_b = opts.get("MarginBottom", 0.02)
    margin_l = opts.get("MarginLeft", 0.02)
    margin_r = opts.get("MarginRight", 0.015)
    margin_t = opts.get("MarginTop", 0.015)
    margin_v = opts.get("MarginVSpace", 0.02)
    # Default extents
    adj_b = margin_b + margins_b[0]
    adj_l = margin_l + wa
    adj_r = 1.0 - margin_r - wb
    adj_t = 1.0 - margin_t - margins_t[-1]
    # Get user options
    adj_b = opts.get("AdjustBottom", adj_b)
    adj_l = opts.get("AdjustLeft", adj_l)
    adj_r = opts.get("AdjustRight", adj_r)
    adj_t = opts.get("AdjustTop", adj_t)
    # Shared axes width
    w_all = adj_r - adj_l
    # Get current extents
    extents = [ax.get_position().bounds for ax in ax_list]
    # Deal with any axis("equal") subplots
    for (j, ax) in enumerate(ax_list):
        # Check for rubber axes
        if ax is ax_rubber:
            # Remember index
            j_rubber = j
        # Check for aspect ratio
        if ax.get_aspect() == "auto":
            # No adjustments necessary
            continue
        # Otherwise, get current position spec
        xminj, yminj, wj, hj = extents[j]
        # Expand (or shrink) current height
        hj = hj * (w_all / wj)
        # Recreate extents (can't change existing tuple)
        extents[j] = (xminj, yminj, wj, hj)
    # Measure all the current figure heights
    h_list = [pos[3] for pos in extents]
    # Total vertical space occupied by fixed plots
    h_fixed = sum(h_list) - h_list[j_rubber]
    # Add in required vertical text space
    if nrows > 1:
        h_fixed += sum(margins_b[1:]) + sum(margins_t[:-1])
    # Add in vertical margins between subplots
    h_fixed += margin_v * (nrows-1)
    # Calculate vertical extent for the rubber plot
    h_rubber = adj_t - adj_b - h_fixed
    # Initialize cumulative vertical coordinate
    ymin = adj_b - margins_b[0]
    # Loop through axes
    for (j, ax) in enumerate(ax_list):
        # Check if it's the rubber plot
        if ax is ax_rubber:
            # Use the previously calculated height
            hj = h_rubber
        else:
            # Use the current extent
            hj = extents[j][3]
        # Add bottom text margin
        ymin += margins_b[j]
        # Set position
        ax.set_position([adj_l, ymin, w_all, hj])
        # Add top text margin and vspace
        ymin += margins_t[j] + margin_v
        # Add plot extent
        ymin += hj


# Co-align a row of axes
def _axes_adjust_row(fig, **kw):
    r"""Adjust a row of axes with shared vertical extents

    :Call:
        >>> _axes_adjust_row(fig, **kw)
    :Inputs:
        *fig*: {``None``} | :class:`Figure` | :class:`int`
            Figure handle or number (default from :func:`plt.gcf`)
    :Keyword Arguments:
        %(keys)s
    :Versions:
        * 2020-01-10 ``@ddalle``: First version
        * 2020-01-27 ``@ddalle``: Added options checks
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
    # Process options
    opts = MPLOpts(**kw)
    # Get axes from figure
    ax_list = fig.get_axes()
    # Number of axes
    nax = len(ax_list)
    # Get list of figures
    subplot_list = kw.get("SubplotList")
    # Default order
    if subplot_list is None:
        # Get current extents
        extents = [ax_list[i].get_position().bounds for i in range(nax)]
        # Get middle *y* coordinate for sorting
        y0 = [extent[0] + 0.5*extent[2] for extent in extents]
        # Sort
        subplot_list = list(np.argsort(y0) + 1)
    # Get index of ax to use for vertical rubber
    subplot_rubber = kw.get("SubplotRubber", -1)
    # Adjust for 1-based index
    if subplot_rubber > 0:
        subplot_rubber -= 1
    # Get handle
    ax_rubber = ax_list[subplot_list[subplot_rubber]]
    # Number of axes in row
    ncols = len(subplot_list)
    # Get the margins occupied by tick and axes labels
    margins = [get_axes_label_margins(ax_list[i-1]) for i in subplot_list]
    # Extract the sides
    margins_l = [margin[0] for margin in margins]
    margins_b = [margin[1] for margin in margins]
    margins_r = [margin[2] for margin in margins]
    margins_t = [margin[3] for margin in margins]
    # Use the maximum margin for left and right
    ha = max(margins_b)
    hb = max(margins_t)
    # Get extra margins
    margin_b = opts.get("MarginBottom", 0.02)
    margin_l = opts.get("MarginLeft", 0.02)
    margin_r = opts.get("MarginRight", 0.015)
    margin_t = opts.get("MarginTop", 0.015)
    margin_h = opts.get("MarginHSpace", 0.02)
    # Default extents
    adj_b = margin_b + ha
    adj_l = margin_l + margins_l[0]
    adj_r = 1.0 - margin_r - margins_r[-1]
    adj_t = 1.0 - margin_t - hb
    # Get user options
    adj_b = opts.get("AdjustBottom", adj_b)
    adj_l = opts.get("AdjustLeft", adj_l)
    adj_r = opts.get("AdjustRight", adj_r)
    adj_t = opts.get("AdjustTop", adj_t)
    # Shared axes height
    h_all = adj_r - adj_l
    # Get current extents
    extents = [ax_list[i-1].get_position().bounds for i in subplot_list]
    # Deal with any axis("equal") subplots
    for (j, i) in enumerate(subplot_list):
        # Get axes
        ax = ax_list[i-1]
        # Check for rubber axes
        if ax is ax_rubber:
            # Remember index
            j_rubber = j
        # Check for aspect ratio
        if ax.get_aspect() == "auto":
            # No adjustments necessary
            continue
        # Otherwise, get current position spec
        xminj, yminj, wj, hj = extents[j]
        # Expand (or shrink) current height
        wj = wj * (h_all / hj)
        # Recreate extents (can't change existing tuple)
        extents[j] = (xminj, yminj, wj, hj)
    # Measure all the current figure widths
    w_list = [pos[2] for pos in extents]
    # Total vertical space occupied by fixed plots
    w_fixed = sum(w_list) - w_list[j_rubber]
    # Add in required vertical text space
    if ncols > 1:
        w_fixed += sum(margins_l[1:]) + sum(margins_r[:-1])
    # Add in vertical margins between subplots
    w_fixed += margin_h * (ncols-1)
    # Calculate vertical extent for the rubber plot
    w_rubber = adj_r - adj_l - w_fixed
    # Initialize cumulative vertical coordinate
    xmin = adj_l - margins_l[0]
    # Loop through axes
    for (j, i) in enumerate(subplot_list):
        # Get axes
        ax = ax_list[i-1]
        # Check if it's the rubber plot
        if ax is ax_rubber:
            # Use the previously calculated height
            wj = w_rubber
        else:
            # Use the current extent
            wj = extents[j][2]
        # Add bottom text margin
        xmin += margins_l[j]
        # Set position
        ax.set_position([xmin, adj_b, wj, h_all])
        # Add top text margin and vspace
        xmin += margins_r[j] + margin_h
        # Add plot extent
        xmin += wj


# Autoscale axes
def _axes_autoscale_height(ax=None, **kw):
    r"""Autoscale height of axes

    For scaling purposes, the *x* limits are taken from
    :func:`ax.get_xlim`.

    :Call:
        >>> _axes_autoscale_height(ax, **kw)
    :Inputs:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
        *YPad*, *Pad*: {``0.07``} | :class:`float`
            Padding for *y* axis
        *YMin*, *ymin*: {``None``} | :class:`float`
            Override automatic *ymin* coordinate
        *YMax*, *ymax*: {``None``} | :class:`float`
            Override automatic *ymax* coordinate
    :Versions:
        * 2020-03-16 ``@ddalle``: First version
    """
    # Default axes
    if ax is None:
        ax = plt.gca()
    # Get figure
    fig = ax.get_figure()
    # Aspect ratio of the figure
    ar_fig = fig.get_figheight() / fig.get_figwidth()
    # Get current axes xlims
    xmin, xmax = ax.get_xlim()
    # Find appropriate y limits
    ymin, ymax = auto_ylim(ax, pad=kw.get("YPad", kw.get("Pad", 0.07)))
    # Check for overrides
    ymin = kw.get("YMin", ymin)
    ymax = kw.get("YMax", ymax)
    # Get current axes position
    x0, y0, x1, _ = get_axes_plot_extents(ax)
    # Axes width
    w_ax = x1 - x0
    # Calculate appropriate axes height
    h_ax = (ymax - ymin) / (xmax - xmin) * w_ax / ar_fig
    # Set new position
    ax.set_position([x0, y0, w_ax, h_ax])


# Autoscale axes
def _axes_autoscale_width(ax=None, **kw):
    r"""Autoscale width of axes

    For scaling purposes, the *y* limits are taken from
    :func:`ax.get_ylim`.

    :Call:
        >>> _axes_autoscale_width(ax, **kw)
    :Inputs:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
        *XPad*, *Pad*: {``0.07``} | :class:`float`
            Padding for *y* axis
        *XMin*, *xmin*: {``None``} | :class:`float`
            Override automatic *xmin* coordinate
        *XMax*, *xmax*: {``None``} | :class:`float`
            Override automatic *xmax* coordinate
    :Versions:
        * 2020-03-16 ``@ddalle``: First version
    """
    # Default axes
    if ax is None:
        ax = plt.gca()
    # Get figure
    fig = ax.get_figure()
    # Aspect ratio of the figure
    ar_fig = fig.get_figheight() / fig.get_figwidth()
    # Get current axes xlims
    ymin, ymax = ax.get_ylim()
    # Find appropriate y limits
    xmin, xmax = auto_xlim(ax, pad=kw.get("XPad", kw.get("Pad", 0.07)))
    # Check for overrides
    xmin = kw.get("XMin", xmin)
    xmax = kw.get("XMax", xmax)
    # Get current axes position
    x0, y0, _, y1 = get_axes_plot_extents(ax)
    # Axes width
    h_ax = y1 - y0
    # Calculate appropriate axes height
    w_ax = (xmax - xmin) / (ymax - ymin) * h_ax * ar_fig
    # Set new position
    ax.set_position([x0, y0, w_ax, h_ax])


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
    xmin, xmax = auto_xlim(ax, pad=xpad)
    ymin, ymax = auto_ylim(ax, pad=ypad)
    # Check for specified limits
    xmin = kw.get("XLimMin", xmin)
    xmax = kw.get("XLimMax", xmax)
    ymin = kw.get("YLimMin", ymin)
    ymax = kw.get("YLimMax", ymax)
    # Test for log scale
    if ax.yaxis.get_scale() == "log":
        # Prevent negative limits
        if ymin <= 0:
            ymin = 1.0
        if ymax <= 0:
            ymax = max(1.0, ymin+1.0)
    # Check for typles
    xmin, xmax = kw.get("XLim", (xmin, xmax))
    ymin, ymax = kw.get("YLim", (ymin, ymax))
    # Make sure data is included.
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
   # --- Cleanup ---
    # Output
    return xl, yl


# Label axes in one of 12 special positions
def _axlabel(ax, lbl, pos=None, **kw):
    r"""Create a label for an axes object

    The *pos* integer implies positions represented by the following
    text diagram::

          0            6            1
        +-----------------------------+
        | 2            7            3 | 10
        |                             |
        | 17           9           16 | 15
        |                             |
        | 4            8            5 | 11
        +-----------------------------+
          12          14           13

    :Call:
        >>> h = _axlabel(ax, lbl, pos=None, **kw):
    :Inputs:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
        *lbl*: :class:`str`
            Text of label to add
        *pos*: :class:`int`
            Index for label position
        *xgap*: {``0.05``} | :class:`float`
            Horizontal spacing in inches
        *ygap*: {``0.05``} | :class:`float`
            Vertical spacing in inches
        *x*: {``None``} | :class:`float`
            Override default *x*-coordinate in *ax.transAxes* scale
        *y*: {``None``} | :class:`float`
            Override default *y*-coordinate in *ax.transAxes* scale
    :Outputs:
        *h*: :class:`matplotlib.text.Text`
            Matplotlib ``Text`` instance
        *h._label*: :class:`str`
            Set to ``"<axlabel>"`` for automatic detection
    :Versions:
        * 2020-04-29 ``@ddalle``: First version
    """
    # Import module
    _import_pyplot()
    # Default axes
    if ax is None:
        ax = plt.gca()
    # Get figure
    fig = ax.figure
    # Options for spacing as figure fraction [inches]
    xgap = 0.05
    ygap = 0.05
    # Dimensions of figure [inches]
    wfig = fig.get_figwidth()
    hfig = fig.get_figheight()
    # Get axes-relative positions
    xa, ya, wa, ha = ax.get_position().bounds
    # Convert gap to axes units
    dx = (xgap / wfig) * (1 / wa)
    dy = (ygap / hfig) * (1 / ha)
    # Default position
    if pos is None:
        # Create mask of which positions are available
        mask = np.ones(18, dtype="bool")
        # Loop through children
        for hi in ax.get_children():
            # Check if it's a :class:`Text` object
            if hi.__class__.__name__ != "Text":
                continue
            # Check its labe
            if hi.get_label() != "<axlabel>":
                continue
            # Get position
            xi, yi = hi.get_position()
            # Filter position
            if 0.0 <= xi < 0.1 and 1.0 <= yi < 1.1:
                # Upper left (above)
                mask[0] = False
            elif 0.9 < xi < 1.0 and 1.0 <= yi < 1.1:
                # Upper right (above)
                mask[1] = False
            elif 0.0 < xi < 0.1 and 0.9 < yi < 1.0:
                # Upper left (below)
                mask[2] = False
            elif 0.9 < xi < 1.0 and 0.9 < yi < 1.0:
                # Upper right (below)
                mask[3] = False
            elif 0.0 <= xi < 0.1 and 0.0 <= yi < 0.1:
                # Lower left (above)
                mask[4] = False
            elif 0.9 < xi < 1.0 and 0.0 <= yi < 0.1:
                # Lower right (above)
                mask[5] = False
            elif 0.38 < xi < 0.55 and 1.0 <= yi < 1.1:
                # Upper center (above)
                mask[6] = False
            elif 0.38 < xi < 0.55 and 0.9 < yi < 1.0:
                # Upper center (below)
                mask[7] = False
            elif 0.38 < xi < 0.55 and 0.0 <= yi < 0.1:
                # Lower center (above)
                mask[8] = False
            elif 0.38 < xi < 0.55 and 0.38 < yi < 0.6:
                # Center
                mask[9] = False
            elif 1.0 <= xi < 1.1 and 0.9 < yi < 1.0:
                # Upper right (outside)
                mask[10] = False
            elif 1.0 <= xi < 1.1 and 0.0 <= yi < 0.1:
                # Lower right (outside)
                mask[11] = False
            elif 0.0 <= xi < 0.1 and -0.1 < yi < 0.0:
                # Lower left (below)
                mask[12] = False
            elif 0.9 < xi < 1.0 and -0.1 < yi < 0.0:
                # Lower right (below)
                mask[13] = False
            elif 0.38 < xi < 0.55 and -0.1 < yi < 0.0:
                # Lower center (below)
                mask[14] = False
            elif 1.0 <= xi < 1.1 and 0.38 < yi < 0.6:
                # Center right (outside)
                mask[15] = False
            elif 0.9 < xi < 1.0 and 0.38 < yi < 0.6:
                # Center right (inside)
                mask[16] = False
            elif 0.0 <= xi < 0.1 and 0.38 < yi < 0.6:
                # Center left (inside)
                mask[17] = False
        # Get available positions
        pos_avail = np.where(mask)[0]
        # Check for any available positions
        if len(pos_avail) == 0:
            # Just go to the beginning
            pos = 0
        else:
            # Use first available position
            pos = int(pos_avail[0])
    # Filter position
    if not isinstance(pos, int):
        # Bad type
        raise TypeError("*pos* parameter must be 'int'; got '%s'" % type(pos))
    elif pos == 0:
        # Upper left (above)
        kw.setdefault("horizontalalignment", "left")
        kw.setdefault("verticalalignment", "bottom")
        # Default positions
        x = dx
        y = 1 + 0.65*dy
    elif pos == 1:
        # Upper right (above)
        kw.setdefault("horizontalalignment", "right")
        kw.setdefault("verticalalignment", "bottom")
        # Default positions
        x = 1 - dx
        y = 1 + 0.65*dy
    elif pos == 2:
        # Upper left (below)
        kw.setdefault("horizontalalignment", "left")
        kw.setdefault("verticalalignment", "top")
        # Default positions
        x = dx
        y = 1 - dy
    elif pos == 3:
        # Upper right (below)
        kw.setdefault("horizontalalignment", "right")
        kw.setdefault("verticalalignment", "top")
        # Default positions
        x = 1 - dx
        y = 1 - dy
    elif pos == 4:
        # Lower left (above)
        kw.setdefault("horizontalalignment", "left")
        kw.setdefault("verticalalignment", "bottom")
        # Default positions
        x = dx
        y = 0.65*dy
    elif pos == 5:
        # Lower right (above)
        kw.setdefault("horizontalalignment", "right")
        kw.setdefault("verticalalignment", "bottom")
        # Default positions
        x = 1 - dx
        y = 0.65*dy
    elif pos == 6:
        # Top middle (above)
        kw.setdefault("horizontalalignment", "center")
        kw.setdefault("verticalalignment", "bottom")
        # Default positions
        x = 0.5
        y = 1 + 0.65*dy
    elif pos == 7:
        # Top middle (below)
        kw.setdefault("horizontalalignment", "center")
        kw.setdefault("verticalalignment", "top")
        # Default positions
        x = 0.5
        y = 1 - dy
    elif pos == 8:
        # Bottom middle (above)
        kw.setdefault("horizontalalignment", "center")
        kw.setdefault("verticalalignment", "bottom")
        # Default positions
        x = 0.5
        y = 0.65*dy
    elif pos == 9:
        # Center
        kw.setdefault("horizontalalignment", "center")
        kw.setdefault("verticalalignment", "center")
        # Default positions
        x = 0.5
        y = 0.5
    elif pos == 10:
        # Top right (right)
        kw.setdefault("horizontalalignment", "left")
        kw.setdefault("verticalalignment", "top")
        # Default positions
        x = 1 + dx
        y = 1 - dy
    elif pos == 11:
        # Bottom right (right)
        kw.setdefault("horizontalalignment", "left")
        kw.setdefault("verticalalignment", "bottom")
        # Default positions
        x = 1 + dx
        y = 0.65*dy
    elif pos == 12:
        # Bottom left (below)
        kw.setdefault("horizontalalignment", "left")
        kw.setdefault("verticalalignment", "top")
        # Default positions
        x = dx
        y = -dy
    elif pos == 13:
        # Bottom right (below)
        kw.setdefault("horizontalalignment", "right")
        kw.setdefault("verticalalignment", "top")
        # Default positions
        x = 1 - dx
        y = -dy
    elif pos == 14:
        # Bottom center (below)
        kw.setdefault("horizontalalignment", "center")
        kw.setdefault("verticalalignment", "top")
        # Default positions
        x = 0.5
        y = -dy
    elif pos == 15:
        # Center right (outside)
        kw.setdefault("horizontalalignment", "left")
        kw.setdefault("verticalalignment", "center")
        # Default rotation
        kw.setdefault("rotation", -90)
        # Default positions
        x = 1 + dx
        y = 0.5
    elif pos == 16:
        # Center right (inside)
        kw.setdefault("horizontalalignment", "right")
        kw.setdefault("verticalalignment", "center")
        # Default rotation
        kw.setdefault("rotation", -90)
        # Default positions
        x = 1 - dx
        y = 0.5
    elif pos == 17:
        # Center left (inside)
        kw.setdefault("horizontalalignment", "left")
        kw.setdefault("verticalalignment", "center")
        # Default rotation
        kw.setdefault("rotation", 90)
        # Default positions
        x = dx
        y = 0.5
    else:
        # Unrecognized
        raise ValueError("Unrecognized *pos*; must be in range [0, 17]")
    # Override coordinates
    x = kw.pop("x", x)
    y = kw.pop("y", y)
    # Set transformation unless overridden
    kw.setdefault("transform", ax.transAxes)
    # Create label
    h = plt.text(x, y, lbl, **kw)
    # Set a special label (used for automatic *pos*)
    h.set_label("<axlabel>")
    # Output
    return h


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
    :Keyword Arguments:
        * See :func:`matplotlib.pyplot.errorbar`
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
    :Keyword Arguments:
        %(keys)s
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


# Colorbars
def _colorbar(ax=None, **kw):
    r"""Add colorbar to a contour plot

    Note that any previous colorbar axes will be removed by default.  To
    keep old colorbars and add a new one, set the label of the
    preexisting colorbar axes to something other than ``"<colorbar>"``.

    :Call:
        >>> _colorbar(**kw)
        >>> _colorbar(ax, **kw)
    :Inputs:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
    :Keyword Arguments:
        %(keys)s
    :Versions:
        * 2020-03-27 ``@jmeeroff``: First version
        * 2020-04-23 ``@ddalle``: Add checks for previous colorbars
        * 2020-05-06 ``@ddalle``: Forked :func:`_colorbar_rm`
    """
    # Make sure pyplot loaded
    _import_pyplot()
    # Gte axes
    if ax is None:
        # Assume current figure can be used
        ax = plt.gca()
    # Get figure handle
    fig = ax.figure
    # Remove any existing colorbars
    _colorbar_rm(ax)
    # Create the new colorbar
    h = plt.colorbar(**kw)
    # Return colorbar
    return h


# Colorbar remover
def _colorbar_rm(ax=None):
    r"""Remove any colorbars from a figure

    :Call:
        >>> _colorbar_rm(ax=None)
    :Inputs:
        *ax*: :class:`matplotlib.axes._subplots.AxesSubplot`
            Axes handle
    :Versions:
        * 2020-05-06 ``@ddalle``: First version
    """
    # Make sure pyplot loaded
    _import_pyplot()
    # Gte axes
    if ax is None:
        # Assume current figure can be used
        ax = plt.gca()
    # Get figure handle
    fig = ax.figure
    # Loop through all axes to check for other color bars
    for axk in list(fig.get_axes()):
        # Check the label
        if axk.get_label() == "<colorbar>":
            # If it's a colorbar already... remove it
            axk.remove()
        elif int(mpl.__version__.split(".")[0]) < 3:
            # No <colorbar> label; try checking size
            if axk.get_position().bounds[0] > 0.87:
                # Remove it...
                axk.remove()
                # Update figure
                plt.draw()


# Region plot
def _fill_between(xv, ymin, ymax, **kw):
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
    :Keyword Arguments:
        * See :func:`matplotlib.pyplot.fill_between`
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
    :Keyword Arguments:
        %(keys)s
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


# Show an image
def _imshow(png, **kw):
    r"""Display an image

    :Call:
        >>> img = _imshow(fpng, **kw)
        >>> img = _imshow(png, **kw)
    :Inputs:
        *fpng*: :class:`str`
            Name of PNG file
        *png*: :class:`np.ndarray`
            Image array from :func:`plt.imread`
    :Keyword Arguments:
        %(keys)s
    :Outputs:
        *img*: :class:`matplotlib.image.AxesImage`
            Image handle
    :Versions:
        * 2020-01-09 ``@ddalle``: First version
        * 2020-01-27 ``@ddalle``: From :mod:`plot_mpl`
    """
    # Make sure modules are loaded
    _import_pyplot()
    # Process input
    if typeutils.isstr(png):
        # Check if file exists
        if not os.path.isfile(png):
            raise SystemError("No PNG file '%s'" % png)
        # Read it
        png = plt.imread(png)
    elif not isinstance(png, np.ndarray):
        # Bad type
        raise TypeError("Image array must be NumPy array")
    elif png.nd not in [2, 3]:
        # Bad dimension
        raise ValueError("Image array must be 2D or 3D (got %i dims)" % png.nd)
    # Process image size
    png_rows = png.shape[0]
    png_cols = png.shape[1]
    # Aspect ratio
    png_ar = float(png_rows) / float(png_cols)
    # Process input coordinates
    xmin = kw.get("ImageXMin")
    xmax = kw.get("ImageXMax")
    ymin = kw.get("ImageYMin")
    ymax = kw.get("ImageYMax")
    # Middle coordinates if filling in
    xmid = kw.get("ImageXCenter")
    ymid = kw.get("ImageYCenter")
    # Check both axes if either side is specified
    x_nospec = (xmin is None) and (xmax is None)
    y_nospec = (ymin is None) and (ymax is None)
    # Fill in defaults
    if x_nospec and y_nospec:
        # All defaults
        extent = (0, png_cols, png_rows, 0)
    elif y_nospec:
        # Check which side(s) specified
        if xmin is None:
            # Read *xmin* from right edge
            xmin = xmax - png_cols
        elif xmax is None:
            # Read *xmax* from left edge
            xmax = xmin + png_cols
        # Specify default *y* values
        yhalf = 0.5 * (xmax - xmin) * png_ar
        # Create *y* window
        if ymid is None:
            # Use ``0``
            ymin = -yhalf
            ymax = yhalf
        else:
            # Use center
            ymin = ymid - yhalf
            ymax = ymid + yhalf
        # Extents
        extent = (xmin, xmax, ymin, ymax)
    elif x_nospec:
        # Check which side(s) specified
        if ymin is None:
            # Read *xmin* from right edge
            ymin = ymax - png_rows
        elif xmax is None:
            # Read *xmax* from left edge
            ymax = ymin + png_rows
        # Scale *x* width
        xwidth = (ymax - ymin) / png_ar
        # Check for a center
        if xmid is None:
            # Use ``0`` for left edge
            xmin = 0.0
            xmax = xwidth
        else:
            # Use specified center
            xmin = xmid - 0.5*xwidth
            xmax = xmid + 0.5*xwidth
        # Extents
        extent = (xmin, xmax, ymin, ymax)
    else:
        # Check which side(s) of *x* is(are) specified
        if xmin is None:
            # Read *xmin* from right edge
            xmin = xmax - png_cols
        elif xmax is None:
            # Read *xmax* from left edge
            xmax = xmin + png_cols
        # Check which side(s) of *y* is(are) specified
        if ymin is None:
            # Read *xmin* from right edge
            ymin = ymax - png_rows
        elif xmax is None:
            # Read *xmax* from left edge
            ymax = ymin + png_rows
        # Extents
        extent = (xmin, xmax, ymin, ymax)
    # Check for directly specified width
    kw_extent = kw.get("ImageExtent")
    # Override ``None``
    if kw_extent is not None:
        extent = kw_extent
    # Show the image
    img = plt.imshow(png, extent=extent)
    # Output
    return img


# Legend
def _legend(ax=None, **kw):
    """Create/update a legend

    :Call:
        >>> leg = _legend(ax=None, **kw)
    :Inputs:
        *ax*: {``None``} | :class:`matplotlib.axes._subplots.AxesSubplot`
            Axis handle (default is ``plt.gca()``
    :Keyword Arguments:
        %(keys)s
    :Outputs:
        *leg*: :class:`matplotlib.legend.Legend`
            Legend handle
    :Versions:
        * 2019-03-07 ``@ddalle``: First version
        * 2019-08-22 ``@ddalle``: From :func:`Part.legend_part`
    """
   # --- Setup ---
    # Import modules if needed
    _import_pyplot()
    # Default axis: most recent
    if ax is None:
        ax = plt.gca()
    # Assume no legend until a label is found, or override
    show_legend = False
    # Check for labels
    for line in ax.get_lines():
        # Check for a label
        lbl = line.get_label()
        # If it starts with an _, it's "not a label"
        if not lbl.startswith("_"):
            # Found first label
            show_legend = True
            break
    # Get overall "Legend" option
    show_legend = kw.pop("ShowLegend", show_legend)
    # Exit immediately if explicit
    if show_legend is False:
        # Get current legend if any
        leg = ax.get_legend()
        # If present; delete it
        if leg:
            leg.remove()
        # Nothing left to do
        return leg
    # Get font properties (copy)
    opts_prop = dict(kw.pop("prop", {}))
    # Get figure
    fig = ax.get_figure()
   # --- Initial Attempt ---
    # Initial legend attempt
    try:
        # Use the options with specified font
        leg = ax.legend(prop=opts_prop, **kw)
    except Exception:
        # Remove font; possibly not found
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
    if nrow > 5 or ncol > 2:
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
    xmin, xmax = auto_xlim(ax, pad=0.0)
    ymin, ymax = auto_ylim(ax, pad=0.0)
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
        *h*: :class:`list`\ [:class:`matplotlib.lines.Line2D`]
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


# Plot part
def _semilogy(xv, yv, fmt=None, **kw):
    r"""Call the :func:`semilogy` function with cycling options

    :Call:
        >>> h = _semilogy(xv, yv, **kw)
        >>> h = _semilogy(xv, yv, fmt, **kw)
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
        *h*: :class:`list`\ [:class:`matplotlib.lines.Line2D`]
            List of line instances
    :Versions:
        * 2021-01-05 ``@ddalle``: Version 1.0; fork from _plot()
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
        h = plt.semilogy(xv, yv, fmt, **kw_p)
    else:
        # No format argument
        h = plt.semilogy(xv, yv, **kw_p)
    # Output
    return h


# contour part
def _contour(xv, yv, zv, **kw):
    r"""Call the :func:`contour` function with cycling options

    :Call:
        >>> hc, hl = _plot(xv, yv, zv, **kw)
    :Inputs:
        *xv*: :class:`np.ndarray`
            Array of *x*-coordinates
        *yv*: :class:`np.ndarray`
            Array of *y*-coordinates
        *ContourType*: 
            :class:`str`:{"tricontourf"} | "tricontour" | "tripcolor"
        *i*, *Index*: {``0``} | :class:`int`
            Phase number to cycle through plot options
        *rotate*, *Rotate*: ``True`` | {``False``}
            Plot independent variable on vertical axis
    :Keyword Arguments:
        * See :func:`matplotlib.pyplot.tricontourf`
    :Outputs:
        *hc*: :class:`matplotlib.tri.tricontour.TriContourSet`
            Unstructured contour handles
        *hl*: :class:`list`\ [:class:`matplotlib.lines.Line2D`]
            List of line instances
    :Versions:
        * 2019-03-04 ``@ddalle``: First version
        * 2020-01-24 ``@ddalle``: Moved to :mod:`plot_mpl.mpl`
        * 2020-03-20 ``@jmeeroff``: Adapted from mpl._plot
    """
    # Ensure plot() is available
    _import_pyplot()
    # Get Contour Type
    ctyp = kw.pop("ContourType", kw.pop("ctyp", "tricontourf"))
    # Get index
    i = kw.pop("Index", kw.pop("i", 0))
    # Get rotation option
    r = kw.pop("Rotate", kw.pop("rotate", False))
    # Options for marker
    kw_p = kw.pop("MarkerOptions", {})
    # Flip inputs
    if r:
        yv, xv = xv, yv
    # Initialize plot options
    kw_c = MPLOpts.select_phase(kw, i)
    # Option to mark the data points
    mark = kw_c.pop("MarkPoints", kw_c.pop("mark", True))
    # Get levels
    levels = kw_c.pop("levels", None)
    # Put together args
    if levels is None:
        # No specified levels
        a = xv, yv, zv
    else:
        # Number of values of levels specified
        a = xv, yv, zv, levels
    # Remove any existing colorbars
    _colorbar_rm(kw.get("ax"))
    # Filter the contour type
    if ctyp == "tricontourf":
        # Filled contour
        h = plt.tricontourf(*a, **kw_c)
    elif ctyp == "tricontour":
        # Contour lines
        h = plt.tricontour(*a, **kw_c)
    elif ctyp == "tripcolor":
        # Triangulated 
        h = plt.tripcolor(*a, **kw_c)
    else:
         # Unrecognized
        raise ValueError("Unrecognized ContourType '%s'" % ctyp)
    # Check for marker options
    if mark:
        # Select phase for markers
        kw_p = MPLOpts.select_phase(kw_p, i)
        # Plot
        hline = plt.plot(xv, yv, **kw_p)
    else:
        # No lines
        hline = []
    # Output
    return h, hline


# Scatter part
def _scatter(xv, yv, **kw):
    r"""Call the :func:`plot` function with cycling options

    :Call:
        >>> h = _scatter(xv, yv, **kw)
    :Inputs:
        *xv*: :class:`np.ndarray`
            Array of *x*-coordinates
        *yv*: :class:`np.ndarray`
            Array of *y*-coordinates
        *rotate*, *Rotate*: ``True`` | {``False``}
            Plot independent variable on vertical axis
    :Keyword Arguments:
        * See :func:`matplotlib.pyplot.scatter`
    :Outputs:
        *h*: :class:`list` (:class:`matplotlib.lines.Line2D`)
            List of line instances
    :Versions:
        * 2020-02-14 ``@ddalle``: First version
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
    # Get Color and Size options
    color = kw.pop('c', None)
    size = kw.pop('s', None)
    # Initialize plot options
    kw_p = MPLOpts.select_phase(kw, i)
    # Call scatter
    h = plt.scatter(xv, yv, s=size, c=color, **kw_p)
    # Output
    return h


# Histogram part
def _hist(v, **kw):
    r"""Call the :func:`hist` function with cycling options

    :Call:
        >>> h = _hist(v, **kw)
    :Inputs:
        *v*: :class:`np.ndarray`
            Array of values
        *i*, *Index*: {``0``} | :class:`int`
            Phase number to cycle through plot options
    :Keyword Arguments:
        * See :func:`matplotlib.pyplot.hist`
    :Outputs:
        *h*: :class:`tuple` 
            Tuple of values, bins, patches
    :Versions:
        * 2020-04-21 ``@jmeeroff``: First version
    """
    # Ensure plot() is available
    _import_pyplot()
    # Get index
    i = kw.pop("Index", kw.pop("i", 0))
    # Get rotation option
    r = kw.pop("Rotate", kw.pop("rotate", False))
    # Initialize plot options
    kw_p = MPLOpts.select_phase(kw, i)
    # Flip inputs
    if r:
        # Call scatter
        h = plt.hist(v, orientation="horizontal", **kw_p)
    else:
        h = plt.hist(v, **kw_p)
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
    :Keyword Arguments:
        %(keys)s
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
    xmin, xmax = auto_xlim(ax, pad=0.0)
    ymin, ymax = auto_ylim(ax, pad=0.0)
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
    if qtl is not False:
        qtl = None
    # Option for both x and y
    qtlX = kw.pop("XTickLabels", qtl)
    qtlY = kw.pop("YTickLabels", qtl)
    # Options for labels on each spine
    qtlL = kw.pop("LeftTickLabels", qtlY)
    qtlR = kw.pop("RightTickLabels", qtlY)
    qtlB = kw.pop("BottomTickLabels", qtlX)
    qtlT = kw.pop("TopTickLabels",  qtlX)
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
def auto_xlim(ax, pad=0.05):
    r"""Calculate appropriate *x*-limits to include all lines in a plot

    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` are
    checked.

    :Call:
        >>> xmin, xmax = auto_xlim(ax, pad=0.05)
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
            # Filter Nans
            xdata = xdata[np.isfinite(xdata)]
            # Check the min and max data
            if len(xdata) > 0:
                try:
                    # Get min/max values
                    xmin = min(xmin, np.min(xdata))
                    xmax = max(xmax, np.max(xdata))
                except Exception:
                    # For some data types (like date arrays),
                    # we can't use np.min(), but we can use slower min()
                    if np.isfinite(xmin):
                        xmin = min(xmin, min(xdata))
                    else:
                        xmin = min(xdata)
                    if np.isfinite(xmax):
                        xmax = max(xmax, max(xdata))
                    else:
                        xmax = max(xdata)
        elif t in ['PathCollection']:
            # Get bounds
            bbox = h.get_datalim(ax.transData).extents
            # Update limits
            xmin = min(xmin, min(bbox[0], bbox[2]))
            xmax = max(xmax, max(bbox[0], bbox[2]))
        elif t in ['PolyCollection', 'LineCollection']:
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
    # Only add padding for floats
    if not isinstance(xmin, float):
        return xmin, xmax
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


# Function to automatically get inclusive data limits.
def auto_ylim(ax, pad=0.05):
    r"""Calculate appropriate *y*-limits to include all lines in a plot

    Plotted objects in the classes :class:`matplotlib.lines.Lines2D` and
    :class:`matplotlib.collections.PolyCollection` are checked.

    :Call:
        >>> ymin, ymax = auto_ylim(ax, pad=0.05)
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
    # Not a log plot at first 
    islog = 0
    # Loop through all children of the input axes.
    for h in ax.get_children():
        # Get the type.
        t = type(h).__name__
        # Check the class.
        if t == 'Line2D':
            # Get the y data for this line
            ydata = h.get_ydata()
            # Filter Nans
            ydata = ydata[np.isfinite(ydata)]
            # Check if log scale
            if h.axes.get_yscale() == "log":
                islog = 1
            # Check the min and max data
            if len(ydata) > 0:
                ymin = min(ymin, np.min(ydata))
                ymax = max(ymax, np.max(ydata))
        elif t in ['PathCollection']:
            # Get bounds
            bbox = h.get_datalim(ax.transData).extents
            # Update limits
            ymin = min(ymin, min(bbox[1], bbox[3]))
            ymax = max(ymax, max(bbox[1], bbox[3]))
        elif t in ['PolyCollection', 'LineCollection']:
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
            ymin = min(ymin, min(bbox[1], bbox[3]))
            ymax = max(ymax, max(bbox[1], bbox[3]))
    # Check for identical values
    if ymax - ymin <= 0.1*pad:
        # Expand by manual amount
        ymax += pad*abs(ymax)
        ymin -= pad*abs(ymin)
    # Add padding
    # Modify for log scale - only pad max
    if islog == 1:
        ymaxv = (1+pad)*ymax - pad*ymin
        yminv = ymin    
    else:
        yminv = (1+pad)*ymin - pad*ymax
        ymaxv = (1+pad)*ymax - pad*ymin
    # Output
    return yminv, ymaxv


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
    ia, ja, ib, jb = _get_axes_plot_extents(ax)
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


# Get extents of other axes
def get_axes_fig_margins(ax=None):
    r"""Get margins due to other axes in figure

    :Call:
        >>> wl, hb, wr, ht = get_axes_fig_margins(ax)
    :Inputs:
        *ax*: {``None``} | :class:`Axes`
            Axes handle (defaults to ``plt.gca()``)
    :Outputs:
        *wl*: :class:`float`
            Figure fraction beyond plot of axes on left
        *hb*: :class:`float`
            Figure fraction beyond plot of axes below
        *wr*: :class:`float`
            Figure fraction beyond plot of axes on right
        *ht*: :class:`float`
            Figure fraction beyond plot of axes above
    :Versions:
        * 2020-04-23 ``@ddalle``: First version
    """
    # Import modules
    _import_pyplot()
    # Default axes
    if ax is None:
        ax = plt.gca()
    # Get figure
    fig = ax.figure
    # List of current axes
    axs = fig.get_axes()
    # Check for trivial case
    if len(axs) == 1:
        # No extra margins
        return 0.0, 0.0, 0.0, 0.0
    # Draw the figure once to ensure the extents can be calculated
    ax.draw(fig.canvas.get_renderer())
    # Size of figure in pixels
    _, _, ifig, jfig = fig.get_window_extent().bounds
    # Get pixel count for main axes
    ia, ja, ib, jb = _get_axes_full_extents(ax)
    # Initialize margins
    wa = 0.0
    ha = 0.0
    wb = 0.0
    hb = 0.0
    # Loop through axes
    for ax1 in axs:
        # Skip main axes
        if ax1 is ax:
            continue
        # Draw the axes once to ensure the extents can be calculated
        ax1.draw(fig.canvas.get_renderer())
        # Get extents of this figure
        ia1, ja1, ib1, jb1 = _get_axes_full_extents(ax1)
        # Expand margins if needed
        wa = max(wa, ia - ia1)
        ha = max(ha, ja - ja1)
        wb = max(wb, ib1 - ib)
        hb = max(hb, jb1 - jb)
    # Convert to fractions
    margin_l = wa / ifig
    margin_b = ha / jfig
    margin_r = wb / ifig
    margin_t = hb / jfig
    # Output
    return margin_l, margin_b, margin_r, margin_t


# Get extents of other axes
def get_axes_neighbors(ax=None):
    r"""Get information on neighboring axes

    :Call:
        >>> margins, neighbors = get_axes_neighbors(ax)
    :Inputs:
        *ax*: {``None``} | :class:`Axes`
            Axes handle (defaults to ``plt.gca()``)
    :Outputs:
        *wl*: :class:`float`
            Figure fraction beyond plot of axes on left
        *hb*: :class:`float`
            Figure fraction beyond plot of axes below
        *wr*: :class:`float`
            Figure fraction beyond plot of axes on right
        *ht*: :class:`float`
            Figure fraction beyond plot of axes above
    :Versions:
        * 2020-04-23 ``@ddalle``: First version
    """
    # Import modules
    _import_pyplot()
    # Default axes
    if ax is None:
        ax = plt.gca()
    # Get figure
    fig = ax.figure
    # List of current axes
    axs = fig.get_axes()
    # Check for trivial case
    if len(axs) == 1:
        # No extra margins
        return (0.0, 0.0, 0.0, 0.0), [None]
    # Draw the figure once to ensure the extents can be calculated
    ax.draw(fig.canvas.get_renderer())
    # Size of figure in pixels
    _, _, ifig, jfig = fig.get_window_extent().bounds
    # Get pixel count for main axes (plot only)
    plot_ia, plot_ja, plot_ib, plot_jb = _get_axes_plot_extents(ax)
    full_ia, full_ja, full_ib, full_jb = _get_axes_full_extents(ax)
    # Bounds for up/down, left/right margins
    x1 = plot_ia + 0.1*(plot_ib - plot_ia)
    x2 = plot_ia + 0.9*(plot_ib - plot_ia)
    y1 = plot_ja + 0.1*(plot_jb - plot_ja)
    y2 = plot_ja + 0.9*(plot_jb - plot_ja)
    # Tolerance for linked axes
    tol = 0.05
    # Initialize neighbors
    neighbors = []
    # Initialize margins
    wa = 0.0
    ha = 0.0
    wb = 0.0
    hb = 0.0
    # Loop through axes
    for k, axk in enumerate(axs):
        # Check for main axes
        if axk is ax:
            # No margins needed
            neighbors.append(None)
            # Go to next axes
            continue
        # Initialize information for this neighbor
        neighbor = {}
        # Draw the axes once to ensure the extents can be calculated
        axk.draw(fig.canvas.get_renderer())
        # Get extents of this figure
        ia1, ja1, ib1, jb1 = _get_axes_plot_extents(axk)
        ia2, ja2, ib2, jb2 = _get_axes_full_extents(axk)
        # Expand margins if needed
        wa = max(wa, full_ia - ia2)
        ha = max(ha, full_ja - ja2)
        wb = max(wb, ib2 - full_ib)
        hb = max(hb, jb2 - full_jb)
        # Midpoint
        xk = 0.5 * (ia1 + ib1)
        yk = 0.5 * (ja1 + jb1)
        # Check horizontal orientation
        if xk < x1:
            # To the left
            lshift = True
            rshift = False
        elif xk <= x2:
            # No horizontal shift
            lshift = False
            rshift = False
        else:
            # Shift right
            lshift = False
            rshift = True
        # Check vertical orientation
        if yk < y1:
            # Downward
            dshift = True
            ushift = False
        elif yk <= y2:
            # No vertical shift
            dshift = False
            ushift = False
        else:
            # Shift up
            dshift = False
            ushift = True
        # Check horizontal links
        if abs(ia1 - plot_ia) / ifig <= tol:
            if abs(ib1 - plot_ib) / ifig <= tol:
                # Linked
                xlink = True
            else:
                # Not linked
                xlink = False
        else:
            # Not linked
            xlink = False
        # Check vertical links
        if abs(ja1 - plot_ja) / jfig <= tol:
            if abs(jb1 - plot_jb) / jfig <= tol:
                # Linked
                ylink = True
            else:
                # Not linked
                ylink = False
        else:
            # Not linked
            ylink = False
        # Save neighbor information
        neighbor = {
            "xlink": xlink,
            "ylink": ylink,
            "lshift": lshift,
            "rshift": rshift,
            "dshift": dshift,
            "ushift": ushift,
        }
        # Add neighbor to list
        neighbors.append(neighbor)
    # Convert to fractions
    margin_l = wa / ifig
    margin_b = ha / jfig
    margin_r = wb / ifig
    margin_t = hb / jfig
    # Combine into tuple
    margins = margin_l, margin_b, margin_r, margin_t
    # Output
    return margins, neighbors


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
    # Deal with legend
    leg = ax.get_legend()
    # Check for null legend
    if leg:
        # Get window extents
        ia_t, ja_t, iw_t, jw_t = leg.get_window_extent().bounds
        # Check for null tick
        if iw_t*jw_t > 0.0:
            # Translate to actual bounds
            ib_t = ia_t + iw_t
            jb_t = ja_t + jw_t
            # Update bounds
            ia = min(ia, ia_t)
            ib = max(ib, ib_t)
            ja = min(ja, ja_t)
            jb = max(jb, jb_t)
    # Deal with children
    for h in ax.get_children():
        # Only process certain types
        typ = h.__class__.__name__
        # Check if it's an object we want to consider
        if typ == "Text":
            # Check for text
            if h.get_text().strip() == "":
                continue
        else:
            # Don't process this kind of Matplotlib object
            continue
        # Get window extents
        ia_t, ja_t, iw_t, jw_t = h.get_window_extent().bounds
        # Check for null window
        if iw_t*jw_t > 0.0:
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


# Automatic documentation
MPLOpts._doc_keys_fn(axes_adjust, "axadjust")
MPLOpts._doc_keys_fn(axes_adjust_col, "axadjust_col")
MPLOpts._doc_keys_fn(axes_adjust_row, "axadjust_row")
MPLOpts._doc_keys_fn(axes, "axes")
MPLOpts._doc_keys_fn(axes_format, "axformat")
MPLOpts._doc_keys_fn(figure, "fig")
MPLOpts._doc_keys_fn(grid, "grid")
MPLOpts._doc_keys_fn(imshow, "imshow")
MPLOpts._doc_keys_fn(legend, "legend")
MPLOpts._doc_keys_fn(spines, "spines")

# Document plotters and direct option users
MPLOpts._doc_keys_fn(errorbar, ["ErrorBarOptions"])
MPLOpts._doc_keys_fn(fill_between, ["FillBetweenOptions"])
MPLOpts._doc_keys_fn(plot, ["PlotOptions"])
MPLOpts._doc_keys_fn(axlabel, ["AxesLabelOptions"])

# Document private functions
MPLOpts._doc_keys_fn(_axes_adjust, "axadjust", submap=False)
MPLOpts._doc_keys_fn(_axes_adjust_col, "axadjust_col", submap=False)
MPLOpts._doc_keys_fn(_axes_adjust_row, "axadjust_row", submap=False)
MPLOpts._doc_keys_fn(_axes, "axes", submap=False)
MPLOpts._doc_keys_fn(_figure, "fig", submap=False)
MPLOpts._doc_keys_fn(_grid, "grid", submap=False)
MPLOpts._doc_keys_fn(_imshow, "imshow", submap=False)
MPLOpts._doc_keys_fn(_legend, "legend", submap=False)
MPLOpts._doc_keys_fn(_spines, "spines", submap=False)
