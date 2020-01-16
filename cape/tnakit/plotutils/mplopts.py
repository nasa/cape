#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------
:mod:`cape.tnakit.plotutils.mplopts`: Options for Matplotlib/Pyplot
--------------------------------------------------------------------

This is a module that stores the defaults for developed interfaces to
common :mod:`matplotlib` plotting functions in addition to a variety
of functions to process options and apply the defaults.

"""

# Local module imports
from . import genopts

# TNA toolkit submodules
from .. import optitem


# List of options whose scalar values are lists
listopts_plot = [
    "dashes"
]

# Default figure options
rc_figure = {
    "wfig": 5.5,
    "hfig": 4.4,
}

# Default axes options
rc_axes = {}

# Default options for plot
rc_plot = {
    "color": ["b", "k", "darkorange", "g"],
    "ls": "-",
    "zorder": 8,
}

rc_hist = {
    "facecolor": 'c',
    "zorder": 2,
    "bins": 20,
    "density": True,
    "edgecolor": 'k',
}
# Options for fill_between
rc_fillbetween = {
    "alpha": 0.2,
    "lw": 0,
    "zorder": 4,
}

# Options for errobar()
rc_errorbar = {
    "capsize": 1.5,
    "elinewidth": 0.8,
    "zorder": 6,
}

# Default legend options
rc_legend = {
    "loc": "upper center",
    "labelspacing": 0.5,
    "framealpha": 1.0,
}

# Default font properties
rc_font = {
    "family": "DejaVu Sans",
}

# Font properties for legend
rc_legend_font = dict(
    rc_font, size=None)

# Mapping of font property names
rc_font_keys = {
    "Font":        "family",
    "FontName":    "family",
    "FontFamily":  "family",
    "FontSize":    "size",
    "FontStretch": "stretch",
    "FontStyle":   "style",
    "FontVariant": "variant",
    "FontWeight":  "weight",
}

# Default options for axis formatting
rc_axfmt = {
    "XLabel": None,
    "YLabel": None,
    "Pad": 0.05,
}

# Default options for grid lines
rc_grid = {
    "MajorGrid": True,
}

# Formatting for grid lines
rc_majorgrid = {
    "ls": ":",
    "color": "#a0a0a0",
}
rc_minorgrid = {}

# default options for spines
rc_spine = {
    "Spines": True,
    "Ticks": True,
    "TickDirection": "out",
}

# default options for mean plot
rc_mu = {
    "color": 'k',
    "lw": 2,
    "zorder": 6,
    "label": "Mean value",
}

# default options for gaussian plot
rc_gauss = {
    "color": "navy",
    "lw": 1.5,
    "zorder": 7,
    "label": "Normal Distribution",
}

# default options for interval plot
rc_interval = {
    "color": "b",
    "lw": 0,
    "zorder": 1,
    "alpha": 0.2,
    "imin": 0.,
    "imax": 5.,
}

# default options for standard deviation plot
rc_std = {
    'color': 'navy',
    'lw': 2,
    'zorder': 5,
    "dashes": [4, 2],
    'StDev': 3,
}
# default options for delta plot on histograms
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


# Combine options and defaults
def process_options(defs, **kw):
    """Process options for some function, using defaults and overrides

    Any key whose value is ``None`` is removed.  The ``None`` keys are removed
    *after* applying the defaults, so setting a key to ``None`` in *kw* will
    remove it from the output.

    :Call:
        >>> opts = process_plot_options(opts, **kw)
    :Inputs:
        *defs*: :class:`dict`
            Dictionary of options
        *kw*: :class:`dict`
            Dictionary of specific options
    :Outputs:
        *opts*: :class:`dict`
            Copied options based on *kw* with fallback to *defs*
    :Versions:
        * 2019-03-01 ``@ddalle``: First version
    """
    # Combine options
    opts = dict(defs, **kw)
    # Remove ``None``s
    return genopts.denone(opts)


# Select options *i*
def select_plotopts(kw, i=0):
    """Select option *i* for each plot option

    This cycles through lists of options for named options such as *color* and
    repeats if *i* is longer than the list of options.  Special options like
    *dashes* that accept a list as a value are handled automatically.  If the
    value of the option is a single value, it is returned regardless of the
    value of *i*

    :Call:
        >>> kw_p = select_plotopts(kw, i=0)
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
        if k in listopts_plot:
            # Get value as a list
            v = optitem.getringel_list(V, i)
        else:
            # Get value as a scalar
            v = optitem.getringel(V, i)
        # Set option
        kw_p[k] = v
    # Output
    return kw_p


# Figure options
def figure_options(kw):
    """Process options for the figure handle

    :Call:
        >>> opts = figure_options(kw, kwu=None)
    :Inputs:
        *fig*: ``None`` | :class:`matplotlib.figure.Figure`
            Optional figure handle
        *FigHeight*, *hfig*: ``None`` | :class:`float`
            Figure height in inches
        *FigWidth*, *wfig*: ``None`` | :class:`float`
            Figure width in inches
    :Outputs:
        *opts*: :class:`dict`
            Options w/ ``None`` removed and defaulted to *rc_figure*
    :Effects:
        *kw*: :class:`dict`
            Options listed above are removed
    :Versions:
        * 2019-03-06 ``@ddalle``: First version
    """
    # Initialize output
    opts = {}
    # Pop options used for this purpose
    opts["fig"] = kw.pop("fig", kw.pop("Figure", None))
    # Figure size
    hfig = opts.pop("FigHeight", opts.pop("hfig", rc_figure.get("hfig")))
    wfig = opts.pop("FigWidth", opts.pop("wfig",  rc_figure.get("wfig")))
    # Save fig sizes
    opts["hfig"] = hfig
    opts["wfig"] = wfig
    # Output
    return process_options(rc_figure, **opts)


# Axes options
def axes_options(kw):
    """Process options for the figure handle

    :Call:
        >>> opts = figure_options(kw, kwu=None)
    :Inputs:
        *ax*: ``None`` | :class:`matplotlib.axes._subplots.AxesSubplot`
            Optional axes handle
    :Outputs:
        *opts*: :class:`dict`
            Options w/ ``None`` removed and defaulted to *rc_figure*
    :Effects:
        *kw*: :class:`dict`
            Options listed above are removed
    :Versions:
        * 2019-03-06 ``@ddalle``: First version
    """
    # Initialize output
    opts = {}
    # Pop options used for this purpose
    opts["ax"] = kw.pop("ax", kw.pop("Axes", None))
    # Output
    return process_options(rc_axes, **opts)


# Global font options
def font_options(kw):
    """Process global font options

    :Call:
        >>> opts = font_options(kw)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of options to parent function
        *kwp*: {``{}``}  | :class:`dict`
            Dictionary of options from which to inherit
    :Keys:
        *FontOptions*: {``{}``} | :class:`dict`
            Options to
            :class:`matplotlib.font_manager.FontProperties`
        *Font*, *FontName*, *FontFamily*: {``None``} | :class:`str`
            Name of font (categories such ``sans-serif`` included)
        *FontSize*: {``None``} | :class:`int` | :class:`str`
            Font size (options such as ``"small"`` included)
        *FontStretch*: {``None``} | :class:`int` | :class:`str`
            Stretch, either numeric in range 0-1000 or options such as
            ``"condensed"``, ``"extra-condensed"``, ``"semi-expanded"``
        *FontStyle*: {``None``} | |MPLFontStyles|
            Font style/slant
        *FontVariant*: {``None``} | ``"normal"`` | ``"small-caps"``
            Font variant
        *FontWeight*: {``None``} | :class:`float` | :class:`str`
            Numeric font weight 0-1000 or ``normal``,  ``bold``, etc.
    :Effects:
        *kw*: :class:`dict`
            Options mentioned above are removed from *kw*
    :Versions:
        * 2019-03-07 ``@ddalle``: First version

    .. |MPLFontStyles| replace::
        ``"normal"`` | ``"italic"`` | ``"oblique"``
    """
    # Get top-level options
    kw_font = kw.pop("FontOptions", {})
    # Apply defaults
    opts = dict(rc_font, **kw_font)
    # Individual options
    for k, kp in rc_font_keys.items():
        # Check if present
        if k not in kw: continue
        # Remove option and save it under shortened name
        opts[kp] = kw.pop(k)
    # Remove "None"
    return genopts.denone(opts)


# Process primary plot options
def plot_options(kw, kwu=None):
    """Process options for the primary plot, including defaults

    :Call:
        >>> opts = plot_options(kw, kwu=None)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of options to parent function
        *kw["PlotOptions"]*: :class:`dict`
            Options for :func:`plot` method
        *kw["LineOptions"]*: :class:`dict`
            Alternative name for :func:`plot` options
        *kwu*: ``None`` | :class:`dict`
            Fallback options not affected by this function
    :Outputs:
        *opts*: :class:`dict`
            Processed options with ``None`` removed and
            defaulted to *rc_plot*
    :Effects:
        *kw*: :class:`dict`
            *LineOptions* and/or *PlotOptions* keys are removed
    :Versions:
        * 2019-03-01 ``@ddalle``: First version
    """
    # Get all possible options
    opts_l = kw.pop("LineOptions", {})
    opts_p = kw.pop("PlotOptions", {})
    # Combine these in a preferred order
    opts = dict(opts_l, **opts_p)
    # Apply defaults
    if isinstance(kwu, dict):
        # Apply "universal" options
        opts = dict(kwu, **opts)
    # Defaults
    return process_options(rc_plot, **opts)


# Process options for min/max plot
def minmax_options(kw, kwp={}, kwu=None):
    """Process options for min/max plots

    :Call:
        >>> tmmx, opts = minmax_options(kw, kwp={}, kwu=None)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of options to parent function
        *kw["MinMaxOptions"]*: :class:`dict`
            Options for :func:`fill_between` or :func:`fill_betweenx`
        *kwp*: :class:`dict`
            Dictionary of options to :func:`plot`
        *kwu*: ``None`` | :class:`dict`
            Fallback options not affected by this function
    :Outputs:
        *tmmx*: {``"FillBetween"``} | ``"ErrorBar"``
            Plot type for min/max plot
        *opts*: :class:`dict`
            Options for :func:`fill_between` or :func:`fill_betweenx`
    :Versions:
        * 2019-03-04 ``@ddalle``: First version
    """
    # Get min/max plot options
    opts = kw.pop("MinMaxOptions", {})
    # Get type
    tmmx = kw.pop("PlotTypeMinMax", "FillBetween")
    # Filter it for checking
    t = tmmx.lower().replace("_", "")
    # Check it
    if t == "fillbetween":
        # Region plot
        tmmx = "FillBetween"
        # Get options for :func:`fill_between`
        opts = fillbetween_options(opts, kwp, kwu)
    elif t == "errorbar":
        # Error bars
        tmmx = "ErrorBar"
        # Get options for :func:`errorbar`
        opts = errorbar_options(opts, kwp, kwu)
    # Output
    return tmmx, opts


# Process options for min/max plot
def uq_options(kw, kwp={}, kwu=None):
    """Process options for uncertainty quantification plots

    :Call:
        >>> tuq, opts = uq_options(kw, kwp={}, kwu=None)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of options to parent function
        *kw["UncertaintyOptions"]*, *kw["UQOptions"]*: :class:`dict`
            Options for :func:`fill_between` or :func:`fill_betweenx`
        *kwp*: :class:`dict`
            Dictionary of options to :func:`plot`
        *kwu*: ``None`` | :class:`dict`
            Fallback options not affected by this function
    :Outputs:
        *tuq*: {``"FillBetween"``} | ``"ErrorBar"``
            Plot type for UQ plot
        *opts*: :class:`dict`
            Options for :func:`fill_between` or :func:`fill_betweenx`
    :Versions:
        * 2019-03-04 ``@ddalle``: First version
    """
    # Get uq plot options
    opts_u = kw.pop("UncertaintyOptions", {})
    opts_q = kw.pop("UQOptions", {})
    # Combine
    opts = dict(opts_q, **opts_u)
    # Get type
    tuq = kw.pop("PlotTypeUncertainty",
                 kw.pop("PlotTypeUQ", "FillBetween"))
    # Filter it for checking
    t = tuq.lower().replace("_", "")
    # Check it
    if t == "fillbetween":
        # Region plot
        tuq = "FillBetween"
        # Get options for :func:`fill_between`
        opts = fillbetween_options(opts, kwp, kwu)
    elif t == "errorbar":
        # Error bars
        tuq = "ErrorBar"
        # Get options for :func:`errorbar`
        opts = errorbar_options(opts, kwp, kwu)
    # Output
    return tuq, opts


# Process options for min/max plot
def error_options(kw, kwp={}, kwu=None):
    """Process options for "error" plots

    :Call:
        >>> terr, opts = error_options(kw, kwp={}, kwu=None)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of options to parent function
        *kw["ErrorOptions"]*: :class:`dict`
            Options for :func:`fill_between` or :func:`fill_betweenx`
        *kwp*: :class:`dict`
            Dictionary of options to :func:`plot`
        *kwu*: ``None`` | :class:`dict`
            Fallback options not affected by this function
    :Outputs:
        *terr*: {``"FillBetween"``} | ``"ErrorBar"``
            Plot type for error plot
        *opts*: :class:`dict`
            Options for :func:`fill_between` or :func:`fill_betweenx`
    :Versions:
        * 2019-03-04 ``@ddalle``: First version
    """
    # Get error plot options
    opts = kw.pop("ErrorOptions", {})
    # Get type
    terr = kw.pop("PlotTypeError", "ErrorBar")
    # Filter it for checking
    t = terr.lower().replace("_", "")
    # Check it
    if t == "fillbetween":
        # Region plot
        terr = "FillBetween"
        # Get options for :func:`fill_between`
        opts = fillbetween_options(opts, kwp, kwu)
    elif t == "errorbar":
        # Error bars
        terr = "ErrorBar"
        # Get options for :func:`errorbar`
        opts = errorbar_options(opts, kwp, kwu)
    # Output
    return terr, opts


# Legend options
def legend_options(kw, kwp={}, kwu=None):
    """Process options for :func:`legend`

    :Call:
        >>> opts = legend_options(kw, kwp={})
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of options to parent function
        *kwp*: {``{}``}  | :class:`dict`
            Dictionary of options from which to inherit
    :Keys:
        *LegendOptions*: {``{}``} | :class:`dict`
            Options to :func:`matplotlib.pyplot.legend`
        *LegendFontOptions*: {``{}``} | :class:`dict`
            Options to :class:`matplotlib.font_manager.FontProperties`
        *LegendFont*: {``None``} | :class:`str`
            Name of font (categories such ``sans-serif`` included)
        *LegendFontSize*: {``None``} | :class:`int` | :class:`str`
            Font size (options such as ``"small"`` included)
        *LegendFontStretch*: {``None``} | :class:`int` | :class:`str`
            Stretch, either numeric in range 0-1000 or options such as
            ``"condensed"``, ``"extra-condensed"``, ``"semi-expanded"``
        *LegendFontStyle*: {``None``} | :class:`str`
            Font style/slant
        *LegendFontVariant*: {``None``} | ``"normal"`` | ``"small-caps"``
            Font variant
        *LegendFontWeight*: {``None``} | :class:`float` | :class:`str`
            Numeric font weight 0-1000 or ``normal``,  ``bold``, etc.
    :Effects:
        *kw*: :class:`dict`
            Options mentioned above are removed from *kw*
    :Versions:
        * 2019-03-07 ``@ddalle``: First version
    """
   # --- Top Level ---
    # Get *LegendOptions*
    opts = kw.pop("LegendOptions", {})
   # --- Font Options ---
    # Check for global font options, legend font options, and opts["prop"]
    kw_font = kwp.get("FontOptions", {})
    kw_lfnt = kw.pop("LegendFontOptions", {})
    kw_prop = opts.get("prop", {})
    # Combine legend font options
    kw_lfnt = dict(kw_font, **kw_lfnt)
    kw_prop = dict(kw_lfnt, **kw_prop)
    # Apply defaults
    opts_prop = process_options(rc_font, **kw_prop)
    # Check for specific *LegendFont* properties
    for k, kp in rc_font_keys.items():
        # Name of legend key
        kl = "Legend" + k
        # Check if present
        if kl not in kw: continue
        # Remove option and save it under shortened name
        opts_prop[kp] = kw.pop(kl)
   # --- Legend Options ---
    # Combine default options
    opts = process_options(rc_legend, **opts)
    # Get location option
    loc = opts.get("loc")
    # Check it
    if loc in ["upper center", 9]:
        # Bounding box location on top spine
        opts.setdefault("bbox_to_anchor", (0.5, 1.05))
    elif loc in ["lower center", 8]:
        # Bounding box location on bottom spine
        opts.setdefault("bbox_to_anchor", (0.5, -0.05))
   # --- Cleanup ---
    # Save font properties
    opts["prop"] = genopts.denone(opts_prop)
    # Output
    return genopts.denone(opts)


# Process fill_between() options
def fillbetween_options(opts, kwp={}, kwu=None):
    """Process options for :func:`fill_between` plots

    :Call:
        >>> kwfb = fillbetween_options(opts, kwp={}, kwu=None)
    :Inputs:
        *opts*: :class:`dict`
            Options for :func:`fill_between` or :func:`fill_betweenx`
        *kwp*: :class:`dict`
            Dictionary of options to :func:`plot`
        *kwu*: ``None`` | :class:`dict`
            Fallback options not affected by this function
    :Outputs:
        *kwfb*: :class:`dict`
            Processed options with defaults applied
    :Versions:
        * 2019-03-05 ``@ddalle``: First version
    """
    # Apply universal properties
    if isinstance(kwu, dict):
        # Apply "universal" options
        opts = dict(kwu, **opts)
    # Defaults for this plot function
    kwfb = process_options(rc_fillbetween, **opts)
    # Apply default options from plot
    kwfb.setdefault("color", kwp.get("color"))
    # Output
    return kwfb


# Process errobar() options
def errorbar_options(opts, kwp={}, kwu=None):
    """Process options for :func:`fill_between` plots

    :Call:
        >>> kweb = errorbar_options(opts, kwp={}, kwu=None)
    :Inputs:
        *opts*: :class:`dict`
            Options for :func:`errorbar`
        *kwp*: :class:`dict`
            Dictionary of options to :func:`plot`
        *kwu*: ``None`` | :class:`dict`
            Fallback options not affected by this function
    :Outputs:
        *kweb*: :class:`dict`
            Processed options with defaults applied
    :Versions:
        * 2019-03-05 ``@ddalle``: First version
    """
    # Apply universal properties
    if isinstance(kwu, dict):
        # Apply "universal" options
        opts = dict(kwu, **opts)
    # Defaults for this plot function
    kweb = process_options(rc_errorbar, **opts)
    # Apply default options from plot
    kweb.setdefault("ecolor", kwp.get("color"))
    # Turn off main line unless requested
    kweb.setdefault("ls", "")
    # Output
    return kweb


# Parse general interval options
def interval_options(kw, kwp={}, kwu=None):
    """Process options for :func:`interval` plots

    :Call:
        >>> kwfb = interval_options(opts, kwp={}, kwu=None)
    :Inputs:
        *opts*: :class:`dict`
            Options for :func:`interval`
        *kwp*: :class:`dict`
            Dictionary of options to :func:`plot`
        *kwu*: ``None`` | :class:`dict`
            Fallback options not affected by this function
    :Outputs:
        *kwfb*: :class:`dict`
            Processed options with defaults applied
    :Versions:
        * 2019-03-05 ``@ddalle``: First version
    """
    # Get dictionary of overall plot options for interval
    opts = kw.pop('IntervalOptions', {})
    # Apply universal properties
    if isinstance(kwu, dict):
        # Apply "universal" options
        opts = dict(kwu, **opts)
    # Defaults for this plot function
    return process_options(rc_interval, **opts)


# Process axes formatting options
def axformat_options(kw, kwp={}, kwu=None):
    """Process options for axes format

    :Call:
        >>> opts = axformat_options(kw, kwp={} kwu=None)
    :Inputs:
        *opts*: :class:`dict`
            Options for :func:`errorbar`
        *kwp*: :class:`dict`
            Dictionary of options to :func:`plot`
        *kwu*: ``None`` | :class:`dict`
            Fallback options not affected by this function
    :Keywords:
        *XLabel*: ``None`` | :class:`str` | :class:`unicode`
            Label for independent data axis
        *YLabel*: ``None`` | :class:`str` | :class:`unicode`
            Label for dependent data axis
        *XMin*: ``None`` | :class:`float`
            Manual minimum value for *x*-axis
        *XMax*: ``None`` |  :class:`float`
            Manual maximum value for *x*-axis
        *YMin*: ``None`` | :class:`float`
            Manual minimum value for *y*-axis
        *YMax*: ``None`` |  :class:`float`
            Manual maximum value for *y*-axis
        *AdjustLeft*: ``None`` | 0 <= :class:`float` < 1
            Manual left margin from figure to axis window
        *AdjustRight*: ``None`` | 0 < :class:`float` <= 1
            Manual right margin from figure to axis window
        *AdjustTop*: ``None`` | 0 < :class:`float` <= 1
            Manual top margin from figure to axis window
        *AdjustBottom*: ``None`` | 0 <= :class:`float` < 1
            Manual bottom margin from figure to axis window
    :Outputs:
        *opts*: :class:`dict`
            Processed options with defaults applied
    :Versions:
        * 2019-03-07 ``@jmeeroff``: First version
    """
    # Initialize output
    opts = {}
    # Get rotation option
    if kwu is None:
        # Not rotated
        r = False
    else:
        # Process rotation option
        r = kwu.get("Rotate", False)
    # Get density option
    o_density = kwp.get("density")
    # Different defaults for histograms
    if o_density is None:
        # No default label
        ylbl = None
    elif o_density:
        # Default label for PDF
        ylbl = "Probability Density"
    else:
        # Raw histogram option
        ylbl = "Count"
    # Process which axis this default applies to
    if r:
        # Default
        xlbl = None
    else:
        # Data on horizontal axis
        xlbl = ylbl
        ylbl = None
    # Process label axes
    opts["XLabel"] = kw.pop("XLabel", xlbl)
    opts["YLabel"] = kw.pop("YLabel", ylbl)
    # Data limits
    opts["XMin"] = kw.pop("XMin", None)
    opts["XMax"] = kw.pop("XMax", None)
    opts["YMin"] = kw.pop("YMin", None)
    opts["YMax"] = kw.pop("YMax", None)
    # Padding
    opts["Pad"]  = kw.pop("Pad", None)
    opts["XPad"] = kw.pop("XPad", None)
    opts["YPad"] = kw.pop("YPad", None)
    # User-set limits
    opts["AdjustLeft"]   = kw.pop("AdjustLeft", None)
    opts["AdjustRight"]  = kw.pop("AdjustRight", None)
    opts["AdjustTop"]    = kw.pop("AdjustTop", None)
    opts["AdjustBottom"] = kw.pop("AdjustBottom", None)
    # Check for universal
    if isinstance(kwu, dict):
        # Apply "universal" options
        opts = dict(kwu, **opts)
    # Remove ``None``
    opts = genopts.denone(opts)
    # Defaults for this plot function
    kwfmt = process_options(rc_axfmt, **opts)
    # Return
    return kwfmt


# Process grid lines formatting options
def grid_options(kw, kwp={}, kwu=None):
    """Process options for grid lines

    :Call:
        >>> opts = grid_options(kw,kwp={} kwu=None)
    :Inputs:
        *opts*: :class:`dict`
            Options for :func:`errorbar`
        *kwp*: :class:`dict`
            Dictionary of options to :func:`plot`
        *kwu*: ``None`` | :class:`dict`
            Fallback options not affected by this function
    :Keywords:
        *MajorGrid*, *Grid*: ``True`` | ``False`` | {``None``}
            Option to turn on/off or leave alone major grid
        *MinorGrid*: ``True`` | ``False`` | {``None``}
            Option to turn on/off or leave alone minor grid
        *MajorGridOptions*, *GridOptions*: :class:`dict`
            Format options for major grid lines
        *MinorGridOptions*: :class:`dict`
            Format options for minor grid lines
    :Outputs:
        *kwgl*: :class:`dict`
            Processed options with defaults applied
    :Versions:
        * 2019-03-07 ``@jmeeroff``: First version
    """
    # Initialize output
    opts = {}
    # Pop major and minor grid options
    genopts.transfer_key(kw, opts, "Grid")
    genopts.transfer_key(kw, opts, "MajorGrid")
    genopts.transfer_key(kw, opts, "MinorGrid")
    # Pop grid line options, accounting for old 'Style' tags
    opts_mg = kw.pop("MajorGridOptions", {})
    opts_ms = kw.pop("MajorGridStyle", {})
    opts_g = kw.pop("GridOptions", {})
    opts_s = kw.pop("GridStyle", {})
    # Combine
    opts1 = dict(opts_g, **opts_mg)
    opts2 = dict(opts_s, **opts_ms)
    optsM = dict(opts2, **opts1)
    # Apply defaults
    opts['GridOptions'] = process_options(rc_majorgrid, **optsM)
    # Minor grid styles
    opts_mg = kw.pop("MinorGridOptions", {})
    opts_ms = kw.pop("MinorGridStyle", {})
    # Combine
    optsm = dict(opts_ms, **opts_mg)
    # Apply defaults
    opts['MinorGridOptions'] = process_options(rc_minorgrid, **optsm)
    # Check for "niversal" options
    if isinstance(kwu, dict):
        # Apply "universal" options
        opts = dict(kwu, **opts)
    # Defaults for this plot function
    kwgl = process_options(rc_grid, **opts)
    # Output
    return kwgl


# Process gir lines formatting options
def spine_options(kw, kwp={}, kwu=None):
    """Process options for spines

    :Call:
        >>> opts = spine_options(kw,kwp={} kwu=None)
    :Inputs:
        *opts*: :class:`dict`
            Options for :func:`errorbar`
        *kwp*: :class:`dict`
            Dictionary of options to :func:`plot`
        *kwu*: ``None`` | :class:`dict`
            Fallback options not affected by this function
    :Global Keywords:
        *Spines* {``None``} | ``False``
            Optional parameter to turn off all spines
        *SpineOptions*: :class:`dict`
            Formatting options applied to all spines
        *Ticks*: {``None``} | ``False``
            Optional parameter to turn off all ticks
        *TickDirection*: ``None`` | ``"in"`` | ``"out"`` | ``"inout"``
            Global tick direction
        *TickLabels*: {``None``} | ``False``
            Optional parameter to turn off all tick labels
        *TickOptions*: :class:`dict`
            Formatting options applied to ticks on all spines
        *TickRotation*: ``None`` | :class:`float` | :class:`int`
            Rotation for all tick labels
        *TickSize*: ``None`` | :class:`float` | :class:`int`
            Size of actual ticks on all axes
    :LeftSpine:
        *LeftSpine*: {``None``} | ``True`` | ``False`` | ``"clipped"``
            Option to turn on/off/leave alone left spine
        *LeftSpineMin*: {``None``} | :class:`float`
            Manual lower bound for extent of left spine
        *LeftSpineMax*: {``None``} | :class:`float`
            Manual upper bound for extent of left spine
        *LeftSpineOptions*: {``{}``} | :class:`dict`
            Options to apply to left spine
        *LeftSpineTicks*: ``None`` | ``True`` | ``False``
            Turn on/off ticks on left spine
        *LeftTickLabels*: ``None`` | ``True`` | ``False``
            Turn on/off labels on left spine ticks
    :RightSpine:
        *RightSpine*: {``None``} | ``True`` | ``False`` | ``"clipped"``
            Option to turn on/off/leave alone right spine
    :BottomSpine:
        *BottomSpine*: {``None``} | ``True`` | ``False`` | ``"clipped"``
            Option to turn on/off/leave alone bottom spine
    :TopSpine:
        *TopSpine*: {``None``} | ``True`` | ``False`` | ``"clipped"``
            Option to turn on/off/leave alone top spine
    :XSpine:
        *XSpine*: {``None``} | ``True`` | ``False`` | ``"clipped"``
            Option to turn on/off/leave alone both horizontal spines
        *XSpineMin*: {``None``} | :class:`float`
            Manual lower bound for extent of both horizontal spines
        *XSpineMax*: {``None``} | :class:`float`
            Manual upper bound for extent of both horizontal spines
        *XSpineOptions*: :class:`dict`
            Formatting options to apply to both horizontal spines
        *XTickDirection*: ``None`` | ``"in"`` | ``"out"`` | ``"inout"``
            Tick direction for horizontal spines
        *XTickLabelSize*, *XTickFontSize*: ``None`` | :class:`float`
            Font size for horizontal-axis labels
        *XTickOptions*: {``{}``} | :class:`dict`
            Formatting options for ticks on horizontal spines
        *XTickRotation*: ``None`` | :class:`float` | :class:`int`
            Rotation for horizontal-axis tick labels
        *XTickSize*: ``None`` | :class:`float` | :class:`int`
            Size of actual ticks on horizontal axes
    :YSpine:
        *YSpine*: {``None``} | ``True`` | ``False`` | ``"clipped"``
            Option to turn on/off/leave alone both vertical spines
        *YSpineMin*: {``None``} | :class:`float`
            Manual lower bound for extent of both vertical spines
        *YSpineMax*: {``None``} | :class:`float`
            Manual upper bound for extent of both vertical spines
        *YSpineOptions*: :class:`dict`
            Formatting options to apply to both vertical spines
        *YTickDirection*: ``None`` | ``"in"`` | ``"out"`` | ``"inout"``
            Tick direction for vertical spines
        *YTickLabelSize*, *YTickFontSize*: ``None`` | :class:`float`
            Font size for vertical-axis labels
        *YTickOptions*: {``{}``} | :class:`dict`
            Formatting options for ticks on vertical spines
        *YTickRotation*: ``None`` | :class:`float` | :class:`int`
            Rotation for vertical-axis tick labels
        *XTickSize*: ``None`` | :class:`float` | :class:`int`
            Size of actual ticks on vertical axes
    :Outputs:
        *kwsp*: :class:`dict`
            Processed options with defaults applied
    :Versions:
        * 2019-03-07 ``@jmeeroff``: First version
    """
    # Initialize output
    opts = {}
    # Universal directions
    udirs = ["X", "Y"]
    # List of principal directions
    pdirs = ["Left", "Right", "Bottom", "Top"]
    # Universal keys
    uskeys = ["", "Min", "Max", "Options"]
    ukeys  = ["TickOptions"]
    # Keys for principal direction keys
    pskeys = ["", "Min", "Max", "Ticks", "Options"]
    pkeys  = ["TickLabels"]
    # Loop through paired directions
    for d in udirs:
        # Loop through appropriate keys
        for k in uskeys:
            # Name of key to transfer
            kd = d + "Spine" + k
            # Transfer from *kw* to *opts*
            genopts.transfer_key(kw, opts, kd)
        # Loop through appropriate keys
        for k in ukeys:
            # Name of key to transfer
            kd = d + k
            # Transfer from *kw* to *opts*
            genopts.transfer_key(kw, opts, kd)
    # Loop through principal directions
    for d in pdirs:
        # Loop through appropriate keys
        for k in pskeys:
            # Name of key to transfer
            kd = d + "Spine" + k
            # Transfer from *kw* to *opts*
            genopts.transfer_key(kw, opts, kd)
        # Loop through appropriate keys without "Spine" infix
        for k in pkeys:
            # Name of key to transfer
            kd = d + k
            # Transfer from *kw* to *opts*
            genopts.transfer_key(kw, opts, kd)
    # Global settings
    genopts.transfer_key(kw, opts, "Spines")
    genopts.transfer_key(kw, opts, "SpineOptions")
    genopts.transfer_key(kw, opts, "Ticks")
    genopts.transfer_key(kw, opts, "TickDirection")
    genopts.transfer_key(kw, opts, "TickLabels")
    genopts.transfer_key(kw, opts, "TickOptions")
    # Ensure *TickOptions* for both axis paris
    opts.setdefault("TickOptions", {})
    opts.setdefault("XTickOptions", {})
    opts.setdefault("YTickOptions", {})
    # Create special options to apply to formatting
    tick_master_options = {
        "TickLabelRotation": "rotation",
        "TickRotation":      "rotation",
        "TickFontSize":      "labelsize",
        "TickLabelSize":     "labelsize",
        "TickSize":          "size",
    }
    # Loop through special options
    for k1, k2 in tick_master_options.items():
        # Combine "X" and "Y" names
        kx1 = "X" + k1
        ky1 = "Y" + k1
        # Transfer value
        genopts.transfer_key(kw, opts["TickOptions"], k1, k2)
        genopts.transfer_key(kw, opts["XTickOptions"], kx1, k2)
        genopts.transfer_key(kw, opts["YTickOptions"], ky1, k2)
    # Ensure quality dictionaries
    for k, v in opts.items():
        # Check for a dictionary
        if not isinstance(v, dict): continue
        # Remove ``None`` entries
        opts[k] = genopts.denone(v)
    # Defaults for this plot function
    kwsp = process_options(rc_spine, **opts)
    # Output
    return kwsp


# Process histogram plot options
def hist_options(kw, kwu=None):
    """Process options for the histogram plot, including defauls

    :Call:
        >>> opts = plot_options(kw, kwu=None)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of options to parent function
        *kw["HistOptions"]*: :class:`dict`
            Options for :func: histogram `plot` method
        *kwu*: ``None`` | :class:`dict`
            Fallback options not affected by this function
    :Outputs:
        *opts*: :class:`dict`
            Processed options with ``None`` removed and
            defaulted to *rc_hist*
    :Effects:
        *kw*: :class:`dict`
            *HistOptions* keys are removed
    :Versions:
        * 2019-03-01 ``@ddalle``: First version
        * 2019-03-11 ``@jmeeroff`` : modified :func: `plot_options`
    """
    # Ensure matplotlib
    import_matplotlib()
    # Determine name of density/normed key
    if int(mpl.__version__[0]) < 2:
        # Old input: "normed"
        k_density = "normed"
    else:
        # New input: "density"
        k_density = "density"
    # Get all possible options
    opts = kw.pop("HistOptions", {})
    # Fix *rc_hist* accordingly
    rc_hist[k_density] = rc_hist.pop(
        "density", rc_hist.pop("normed", None))
    # Get global normalization option
    o_density = kw.pop("density", kw.pop("normed", rc_hist[k_density]))
    # Get option from *HistOptions*
    o_density = opts.pop("density", opts.pop("normed", o_density))
    # Save to dict using appropriate name
    opts[k_density] = o_density
    # Apply defaults
    if isinstance(kwu, dict):
        # Apply "universal" options
        opts = dict(kwu, **opts)
    # Apply defaults
    return process_options(rc_hist, **opts)


# Process gaussian plot options
def gauss_options(kw, kw_p, kwu=None):
    """Process options for plotting the mean, including defauls

    :Call:
        >>> opts = gauss_options(kw, kwu=None)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of options to parent function
        *kw["HistOptions"]*: :class:`dict`
            Options for :func: histogram `plot` method
        *kwp*: :class:`dict`
            Dictionary of options to :func:`plot`
        *kwu*: ``None`` | :class:`dict`
            Fallback options not affected by this function
    :Outputs:
        *opts*: :class:`dict`
            Processed options with ``None`` removed and
            defaulted to *rc_gauss*
    :Effects:
        *kw*: :class:`dict`
            *HistOptions* keys are removed
    :Versions:
        * 2019-03-01 ``@ddalle``: First version
        * 2019-03-11 ``@jmeeroff`` : modified :func: `plot_options`
    """
    # Get all possible options
    opts = kw.pop("GaussianOptions", {})
    # Apply defaults
    if isinstance(kwu, dict):
        # Apply "universal" options
        opts = dict(kwu, **opts)
    # Defaults
    return process_options(rc_gauss, **opts)


# Process mean plot options
def mu_options(kw, kw_p, kwu=None):
    """Process options for plotting the mean, including defauls

    :Call:
        >>> opts = gauss_options(kw, kwu=None)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of options to parent function
        *kw["HistOptions"]*: :class:`dict`
            Options for :func: histogram `plot` method
        *kwp*: :class:`dict`
            Dictionary of options to :func:`plot`
        *kwu*: ``None`` | :class:`dict`
            Fallback options not affected by this function
    :Outputs:
        *opts*: :class:`dict`
            Processed options with ``None`` removed and
            defaulted to *rc_mu*
    :Effects:
        *kw*: :class:`dict`
            *HistOptions* keys are removed
    :Versions:
        * 2019-03-01 ``@ddalle``: First version
        * 2019-03-11 ``@jmeeroff`` : modified :func: `plot_options`
    """
    # Get all possible options
    opts = kw.pop("MeanOptions", {})
    # Apply defaults
    if isinstance(kwu, dict):
        # Apply "universal" options
        opts = dict(kwu, **opts)
    # Defaults
    return process_options(rc_mu, **opts)


# Process options for standard deviation plots
def std_options(kw, kwp={}, kwu=None):
    """Process options for :func:`std` plots

    :Call:
        >>> kwfb = std_options(opts, kwp={}, kwu=None)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of options to parent function
        *kwp*: :class:`dict`
            Dictionary of options to :func:`plot`
        *kwu*: ``None`` | :class:`dict`
            Fallback options not affected by this function
    :Outputs:
        *opts*: :class:`dict`
            Processed options with ``None`` removed
            and defaulted to *rc_std*
    :Versions:
        * 2019-03-05 ``@ddalle``: First version
    """
    opts = kw.pop('StDevOptions', {})
    # Apply universal properties
    if isinstance(kwu, dict):
        # Apply "universal" options
        opts = dict(kwu, **opts)
    # Defaults for this plot function
    return process_options(rc_std, **opts)


# Process options for delta plots on historgrams
def delta_options(kw, kwp={}, kwu=None):
    """Process options for :func:`delta` plots

    :Call:
        >>> kwfb = delta_options(kw, kwp={}, kwu=None)
    :Inputs:
        *kw*: :class:`dict`
            Dictionary of options to parent function
        *kwp*: :class:`dict`
            Dictionary of options to :func:`plot`
        *kwu*: ``None`` | :class:`dict`
            Fallback options not affected by this function
    :Outputs:
    :Outputs:
        *opts*: :class:`dict`
            Processed options with ``None`` removed
            and defaulted to *rc_delta*
    :Versions:
        * 2019-03-05 ``@ddalle``: First version
    """
    opts = kw.pop('DeltaOptions', {})
    # Apply universal properties
    if isinstance(kwu, dict):
        # Apply "universal" options
        opts = dict(kwu, **opts)
    # Defaults for this plot function
    return process_options(rc_delta, **opts)


# Process histlabel options
def histlabel_options(opts, kwp={}, kwu=None):
    """Process options for :func:`fill_between` plots

    :Call:
        >>> kweb = errorbar_options(opts, kwp={}, kwu=None)
    :Inputs:
        *opts*: :class:`dict`
            Options for histogram labels
        *kwp*: :class:`dict`
            Dictionary of options to :func:`plot`
        *kwu*: ``None`` | :class:`dict`
            Fallback options not affected by this function
    :Outputs:
        *kweb*: :class:`dict`
            Processed options with defaults applied
    :Versions:
        * 2019-03-05 ``@ddalle``: First version
    """
    # Apply universal properties
    if isinstance(kwu, dict):
        # Apply "universal" options
        opts = dict(kwu, **opts)
    # Defaults for this plot function
    return process_options(rc_histlbl, **opts)
