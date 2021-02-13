#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------
:mod:`cape.tnakit.plot_mpl.mplopts`: Matplotlib/Pyplot Options
--------------------------------------------------------------------

This module creates the class :class:`MPLOpts`, which contains all
options recognized by the primary :mod:`cape.tnakit.plot_mpl` module
during any type of plot.

A generic dictionary of keyword arguments can be transformed into
:class:`MPLOpts` using the ``**`` operator.

    .. code-block:: python

        # Declare some plot options
        kw = {
            "PlotColor": "c",
            "PlotOptions": {
                "linewidth": 2,
                "dashes": [3, 1],
            },
            "i": 2,
        }
        # Filter options, declare defaults, etc.
        opts = MPLOpts(**kw)

This operation performs several steps including

    * renaming options based on abbrevs; e.g. "i" to "Index",
    * removing any unrecognized options,
    * applying mapped options, e.g. "PlotColor" -> "PlotOptions.color",
    * applying cascading options, e.g. "PlotOptions.color" to
      "ErrorBarOptions.color",
    * declare default values for any options, and
    * remove any options whose value is ``None``.

In addition, the :class:`MPLOpts` instance has several methods to access
only the options relevant to a particular plot function.

"""

# TNA toolkit modules
import cape.tnakit.kwutils as kwutils
import cape.tnakit.typeutils as typeutils


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
class MPLOpts(kwutils.KwargHandler):
    r"""Options class for all plot methods in :mod:`plot_mpl` module

    :Call:
        >>> opts = MPLOpts(**kw)
    :Inputs:
        *kw*: :class:`dict`
            Keyword options to be filtered and mapped
    :Outputs:
        *opts*: :class:`plot_mpl.MPLOpts`
            Options from kwargs with defaults applied
    :Versions:
        * 2020-01-23 ``@ddalle``: Version 2.0 based on KwargHandler
    """
  # ====================
  # Class Attributes
  # ====================
  # <
   # --- Global Options ---
    # Lists of options
    _optlist = {
       # A-E
        "AdjustBottom",
        "AdjustLeft",
        "AdjustRight",
        "AdjustTop",
        "AxesLabelColor",
        "AxesLabelFontOptions",
        "AxesLabelFontName",
        "AxesLabelFontSize",
        "AxesLabelFontStretch",
        "AxesLabelFontStyle",
        "AxesLabelFontVariant",
        "AxesLabelFontWeight",
        "AxesLabelOptions",
        "AxesLabelPosition",
        "BottomSpine",
        "BottomSpineMax",
        "BottomSpineMin",
        "BottomSpineOptions",
        "BottomSpineTicks",
        "BottomTickLabels",
        "ContourColorMap",
        "ContourLevels",
        "ContourOptions",
        "ContourType",
        "Coverage",
        "CoverageCDF",
        "CoverageOptions",
        "Delta",
        "Density",
        "ErrorBarOptions",
        "ErrorBarMarker",
        "ErrorOptions",
        "ErrorPlotType",
       # F-J
        "FigDPI",
        "FigHeight",
        "FigNumber",
        "FigOptions",
        "FigWidth",
        "FillBetweenOptions",
        "FilterSigma",
        "FontOptions",
        "FontName",
        "FontSize",
        "FontStretch",
        "FontStyle",
        "FontVariant",
        "FontWeight",
        "Grid",
        "GridColor",
        "GridOptions",
        "GridStyle",
        "HistBins",
        "HistColor",
        "HistDeltaLabelColor",
        "HistDeltaLabelName",
        "HistDeltaLabelOptions",
        "HistDeltaLabelPosition",
        "HistDeltaOptions",
        "HistGaussianOptions",
        "HistIntervalLabelColor",
        "HistIntervalLabelOptions",
        "HistIntervalLabelPosition",
        "HistIntervalOptions",
        "HistMeanLabelColor",
        "HistMeanLabelOptions",
        "HistMeanLabelPosition",
        "HistMeanOptions",
        "HistSigmaLabelColor",
        "HistSigmaLabelOptions",
        "HistSigmaLabelPosition",
        "HistSigmaOptions",
        "HistOptions",
        "ImageExtent",
        "ImageXCenter",
        "ImageXMax",
        "ImageXMin",
        "ImageYCenter",
        "ImageYMax",
        "ImageYMin",
        "Index",
       # K-O
        "KeepAspect",
        "Label",
        "LeftSpine",
        "LeftSpineMax",
        "LeftSpineMin",
        "LeftSpineOptions",
        "LeftSpineTicks",
        "LeftTickLabels",
        "Legend",
        "LegendFontName",
        "LegendFontOptions",
        "LegendFontSize",
        "LegendFontStretch",
        "LegendFontStyle",
        "LegendFontVariant",
        "LegendFontWeight",
        "LegendOptions",
        "MajorGrid",
        "MarginBottom",
        "MarginHSpace",
        "MarginLeft",
        "MarginRight",
        "MarginTop",
        "MarginVSpace",
        "MarkPoints",
        "MarkerColor",
        "MarkerOptions",
        "MarkerSize",
        "MeanOptions",
        "MinMaxOptions",
        "MinMaxPlotType",
        "MinorGrid",
        "MinorGridOptions",
        "NGauss",
       # P-T
        "Pad",
        "PlotColor",
        "PlotLineStyle",
        "PlotLineWidth",
        "PlotFormat",
        "PlotOptions",
        "RightSpine",
        "RightSpineMax",
        "RightSpineMin",
        "RightSpineOptions",
        "RightSpineTicks",
        "RightTickLabels",
        "Rotate",
        "ScatterColor",
        "ScatterSize",
        "ScatterOptions",
        "ShowError",
        "ShowHistDelta",
        "ShowHistDeltaLabel",
        "ShowHistGaussian",
        "ShowHistInterval",
        "ShowHistIntervalLabel",
        "ShowHistMean",
        "ShowHistMeanLabel",
        "ShowHistSigma",
        "ShowHistSigmaLabel",
        "ShowLegend",
        "ShowLine",
        "ShowMinMax",
        "ShowUncertainty",
        "SeamDY",
        "SpineOptions",
        "Spines",
        "StDev",
        "Subplot",
        "SubplotCols",
        "SubplotList",
        "SubplotRows",
        "SubplotRubber",
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
       # U-Z
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
       # a-z
        "acov",
        "ax",
        "bcov",
        "fig",
        "mu",
        "std",
        "ux",
        "uy",
        "v",
        "x",
        "xerr",
        "y",
        "yerr",
        "ymin",
        "ymax",
        "z"
    }

    # Options for which a singleton is a list
    _optlist_list = {
        "dashes",
        "ContourLevels",
        "XLim",
        "YLim",
        "XTickLabels",
        "XTicks",
        "YTickLabels",
        "YTicks"
    }

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
        "IntOpts" : "IntervalOptions",
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
        "axlabelopts": "AxesLabelOptions",
        "cdf": "CoverageCDF",
        "cov": "Coverage",
        "density": "Density",
        "dy": "SeamDY",
        "fmt": "PlotFormat",
        "grid": "Grid",
        "hfig": "FigHeight",
        "hspace": "MarginHSpace",
        "i": "Index",
        "label": "Label",
        "lbl": "Label",
        "normed": "density",
        "nfig": "FigNumber",
        "ngauss" : "NGauss",
        "numfig": "FigNumber",
        "rotate": "Rotate",
        "subplot": "Subplot",
        "vspace": "MarginVSpace",
        "wfig": "FigWidth",
        "xlabel": "XLabel",
        "xlim": "XLim",
        "ylabel": "YLabel",
        "ylim": "YLim",
    }

   # --- Option Sublists ---
    _optlists = {
        "axes": [
            "ax",
            "AxesOptions"
        ],
        "axadjust": [
            "MarginBottom",
            "MarginLeft",
            "MarginRight",
            "MarginTop",
            "AdjustBottom",
            "AdjustLeft",
            "AdjustRight",
            "AdjustTop",
            "KeepAspect",
            "Subplot",
            "SubplotCols",
            "SubplotRows"
        ],
        "axadjust_col": [
            "MarginBottom",
            "MarginLeft",
            "MarginRight",
            "MarginTop",
            "MarginVSpace",
            "AdjustBottom",
            "AdjustLeft",
            "AdjustRight",
            "AdjustTop",
            "SubplotList",
            "SubplotRubber"
        ],
        "axadjust_row": [
            "MarginBottom",
            "MarginLeft",
            "MarginRight",
            "MarginTop",
            "MarginHSpace",
            "AdjustBottom",
            "AdjustLeft",
            "AdjustRight",
            "AdjustTop",
            "SubplotList",
            "SubplotRubber"
        ],
        "axformat": [
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
        ],
        "axheight": [
            "Pad",
            "YPad",
            "YMax",
            "YMin"
        ],
        "axlabel": [
            "ax",
            "AxesLabelColor",
            "AxesLabelOptions",
            "AxesLabelFontOptions",
            "AxesLabelFontName",
            "AxesLabelFontSize",
            "AxesLabelFontStretch",
            "AxesLabelFontStyle",
            "AxesLabelFontVariant",
            "AxesLabelFontWeight",
            "AxesLabelPosition"
        ],
        "axwidth": [
            "Pad",
            "XPad",
            "XMax",
            "XMin"
        ],
        "contour": [
            "Index",
            "Rotate",
            "ContourType",
            "ContourLevels",
            "ContourOptions",
            "MarkPoints",
            "MarkerColor",
            "MarkerOptions",
            "MarkerSize"
        ],
        "coverage":[
            "Coverage",
            "CoverageCDF",
            "CoverageOptions",
            "FilterSigma"
        ],
        "error": [
            "Index",
            "Rotate",
            "ErrorOptions",
            "ErrorPlotType",
            "ErrorBarOptions",
            "ErrorBarMarker",
            "FillBetweenOptions"
        ],
        "errorbar": [
            "Index",
            "Rotate",
            "ErrorBarOptions",
            "ErrorBarMarker"
        ],
        "fig": [
            "fig",
            "FigOptions",
            "FigNumber",
            "FigWidth",
            "FigHeight",
            "FigDPI"
        ],
        "fillbetween": [
            "Index",
            "Rotate",
            "FillBetweenOptions"
        ],
        "font": [
            "FontOptions",
            "FontName",
            "FontSize",
            "FontStretch",
            "FontStyle",
            "FontVariant",
            "FontWeight"
        ],
        "grid": [
            "Grid",
            "GridOptions",
            "MinorGrid",
            "MinorGridOptions",
        ],
        "hist" : [
            "HistBins",
            "HistColor",
            "HistOptions",
            "Rotate"
        ],
        "histdelta": [
            "Delta",
            "HistDeltaOptions"
        ],
        "histdeltalabel": [
            "HistDeltaLabelOptions"
        ],
        "histgaussian" : [
            "NGauss",
            "HistGaussianOptions",
        ],
        "histinterval": [
            "Index",
            "Rotate",
            "HistIntervalOptions"
        ],
        "histintervallabel": [
            "HistIntervalLabelOptions"
        ],
        "histmean": [
            "HistMeanOptions",
            "Rotate"
        ],
        "histmeanlabel": [
            "HistMeanLabelOptions"
        ],
        "histsigma" : [
            "HistSigmaOptions",
            "StDev"
        ],
        "histsigmalabel": [
            "HistSigmaLabelOptions"
        ],
        "imshow": [
            "ImageXMin",
            "ImageXMax",
            "ImageXCenter",
            "ImageYMin",
            "ImageYMax",
            "ImageYCenter",
            "ImageExtent"
        ],
        "legend": [
            "Legend",
            "LegendAnchor",
            "LegendFontOptions",
            "LegendLocation",
            "LegendOptions"
        ],
        "legendfont": [
            "LegendFontName",
            "LegendFontSize",
            "LegendFontStretch",
            "LegendFontStyle",
            "LegendFontVariant",
            "LegendFontWeight"
        ],
        "plot": [
            "Index",
            "Rotate",
            "PlotOptions",
            "PlotFormat"
        ],
        "minmax": [
            "Index",
            "Rotate",
            "MinMaxOptions",
            "MinMaxPlotType",
            "ErrorBarOptions",
            "ErrorBarMarker",
            "FillBetweenOptions"
        ],
        "scatter": [
            "Index",
            "Rotate",
            "ScatterOptions",
            "ScatterColor",
            "ScatterSize"
        ],
        "seam": [
            "SeamDY"
        ],
        "spines": [
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
        ],
        "subplot": [
            "AdjustBottom",
            "AdjustLeft",
            "AdjustRight",
            "AdjustTop",
            "Subplot",
            "SubplotCols",
            "SubplotRows"
        ],
        "uq": [
            "Index",
            "Rotate",
            "ErrorBarMarker",
            "ErrorBarOptions",
            "FillBetweenOptions",
            "UncertaintyPlotType",
            "UncertaintyOptions"
        ],
    }

   # --- Types ---
    # Types
    _opttypes = {
       # A-E
        "AdjustBottom": float,
        "AdjustLeft": float,
        "AdjustRight": float,
        "AdjustTop": float,
        "AxesLabelColor": (tuple, typeutils.strlike),
        "AxesLabelOptions": dict,
        "AxesLabelFontOptions": dict,
        "AxesLabelFontName": typeutils.strlike,
        "AxesLabelFontSize": (int, float, typeutils.strlike),
        "AxesLabelFontStretch": (int, float, typeutils.strlike),
        "AxesLabelFontStyle": typeutils.strlike,
        "AxesLabelFontVariant": typeutils.strlike,
        "AxesLabelFontWeight": (float, int, typeutils.strlike),
        "AxesLabelPosition": int,
        "AxesOptions": dict,
        "BottomSpine": (bool, typeutils.strlike),
        "BottomSpineMax": float,
        "BottomSpineMin": float,
        "BottomSpineOptions": dict,
        "BottomSpineTicks": bool,
        "BottomTickLabels": bool,
        "ContourColorMap" : typeutils.strlike,
        "ContourType": typeutils.strlike,
        "ContourLevels": (typeutils.intlike, list),
        "ContourOptions": dict,
        "Coverage": float,
        "CoverageCDF": float,
        "CoverageOptions": dict,
        "Delta": float,
        "Density": bool,
        "ErrorBarMarker": typeutils.strlike,
        "ErrorBarOptions": dict,
        "ErrorOptions": dict,
        "ErrorPlotType": typeutils.strlike,
       # F-J
        "FigDPI": (float, int),
        "FigHeight": float,
        "FigNumber": int,
        "FigOptions": dict,
        "FigWidth": float,
        "FillBetweenOptions": dict,
        "FilterSigma" : float,
        "FontName": typeutils.strlike,
        "FontOptions": dict,
        "FontSize": (int, float, typeutils.strlike),
        "FontStretch": (int, float, typeutils.strlike),
        "FontStyle": typeutils.strlike,
        "FontVariant": typeutils.strlike,
        "FontWeight": (float, int, typeutils.strlike),
        "Grid": int,
        "GridOptions": dict,
        "HistBins": int,
        "HistColor": (tuple, typeutils.strlike),
        "HistDeltaOptions": dict,
        "HistDeltaLabelOptions": dict,
        "HistDeltaLabelColor": typeutils.strlike,
        "HistDeltaLabelName": typeutils.strlike,
        "HistDeltaLabelPosition": int,
        "HistGaussianOptions": dict,
        "HistIntervalOptions": dict,
        "HistIntervalLabelOptions": dict,
        "HistIntervalLabelColor": typeutils.strlike,
        "HistIntervalLabelPosition": int,
        "HistMeanOptions": dict,
        "HistMeanLabelOptions": dict,
        "HistMeanLabelColor": typeutils.strlike,
        "HistMeanLabelPosition": int,
        "HistOptions" : dict,
        "HistSigmaOptions": dict,
        "HistSigmaLabelOptions": dict,
        "HistSigmaLabelColor": typeutils.strlike,
        "HistSigmaLabelPosition": int,
        "ImageExtent": (tuple, dict),
        "ImageXCenter": float,
        "ImageXMax": float,
        "ImageXMin": float,
        "ImageYCenter": float,
        "ImageYMax": float,
        "ImageYMin": float,
        "Index": int,
       # K-O
        "KeepAspect": bool,
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
        "MarginHSpace": float,
        "MarginLeft": float,
        "MarginRight": float,
        "MarginTop": float,
        "MarginVSpace": float,
        "MarkPoints": bool,
        "MarkerColor": (tuple, typeutils.strlike),
        "MarkerOptions": dict,
        "MarkerSize": (int, float),
        "MinorGrid": bool,
        "MinorGridOptions": dict,
        "MinMaxOptions": dict,
        "MinMaxPlotType": typeutils.strlike,
        "NGauss" : int,
       # P-T
        "Pad": float,
        "PlotColor": (tuple, typeutils.strlike),
        "PlotFormat": typeutils.strlike,
        "PlotLineStyle": typeutils.strlike,
        "PlotLineWidth": (float, int),
        "PlotOptions": dict,
        "Rotate": bool,
        "ScatterOptions": dict,
        "SeamDY": (float, int),
        "ShowError": bool,
        "ShowHistDelta": bool,
        "ShowHistDeltaLabel": bool,
        "ShowHistInterval": bool,
        "ShowHistIntervalLabel": bool,
        "ShowHistGaussian": bool,
        "ShowHistMean": bool,
        "ShowHistMeanLabel": bool,
        "ShowHistSigma": bool,
        "ShowHistSigmaLabel": bool,
        "ShowLine": bool,
        "ShowMinMax": bool,
        "ShowUncertainty": bool,
        "SpineOptions": dict,
        "Spines": bool,
        "StDev": (int, float, list, tuple, typeutils.arraylike),
        "SubplotCols": int,
        "SubplotList": list,
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
       # U-Z
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
       # a-z
        "ax": object,
        "acov": float,
        "bcov": float,
        "fig": object,
        "mu": float,
        "std": float,
        "ux": typeutils.arraylike,
        "uy": typeutils.arraylike,
        "v": typeutils.arraylike,
        "x": typeutils.arraylike,
        "xerr": typeutils.arraylike,
        "y": typeutils.arraylike,
        "yerr": typeutils.arraylike,
        "ymax": typeutils.arraylike,
        "ymin": typeutils.arraylike,
        "z": typeutils.arraylike,
    }

   # --- Cascading Options ---
    # Global options mapped to subcategory options
    _kw_submap = {
        "AxesLabelFontOptions": {
            "FontOptions.family": "family",
            "FontOptions.size": "size",
            "FontOptions.stretch": "stretch",
            "FontOptions.style": "style",
            "FontOptions.variant": "variant",
            "FontOptions.weight": "weight",
            "AxesLabelFontName": "family",
            "AxesLabelFontSize": "size",
            "AxesLabelFontStretch": "stretch",
            "AxesLabelFontStyle": "style",
            "AxesLabelFontVariant": "variant",
            "AxesLabelFontWeight": "weight",
        },
        "AxesLabelOptions": {
            "AxesLabelColor": "color",
            "AxesLabelFontOptions": "fontdict",
            "AxesLabelPosition": "pos",
        },
        "AxesOptions": {},
        "ContourOptions": {
            "Index": "Index",
            "Rotate": "Rotate",
            "Label": "label",
            "ContourColorMap": "cmap",
            "ContourLevels": "levels",
            "MarkPoints": "MarkPoints",
            "MarkerOptions": "MarkerOptions",
        },
        "CoverageOptions": {
            "Coverage": "Coverage",
            "CoverageCDF": "CoverageCDF",
        },
        "ErrorOptions": {},
        "FigOptions": {
            "FigNumber": "num",
            "FigDPI": "dpi",
            "FigHeight": "figheight",
            "FigWidth": "figwidth",
        },
        "ErrorBarOptions": {
            "Index": "Index",
            "Rotate": "Rotate",
            "ErrorBarMarker": "marker",
            "PlotOptions.color": "color",
        },
        "FillBetweenOptions": {
            "Index": "Index",
            "Rotate": "Rotate",
            "PlotOptions.color": "color",
        },
        "FontOptions": {
            "FontName": "family",
            "FontSize": "size",
            "FontStretch": "stretch",
            "FontStyle": "style",
            "FontVariant": "variant",
            "FontWeight": "weight",
        },
        "GridOptions": {
            "GridColor": "color",
        },
        "HistOptions": {
            "HistBins": "bins",
            "HistColor": "color",
            "Rotate": "Rotate"
        },
        "HistDeltaOptions":{
            "Delta": "Delta",
            "Index": "Index",
            "Rotate": "Rotate"
        },
        "HistDeltaLabelOptions": {
            "AxesLabelFontOptions": "fontdict",
            "HistDeltaLabelColor": "color",
            "HistDeltaLabelPosition": "pos",
        },
        "HistGaussianOptions": {
            "Index": "Index",
            "Rotate": "Rotate",
            "PlotOptions.color": "color",
            "NGauss": "NGauss",
        },
        "HistIntervalOptions": {
            "Index" : "Index",
            "Rotate" : "Rotate",
        },
        "HistIntervalLabelOptions": {
            "AxesLabelFontOptions": "fontdict",
            "HistIntervalLabelColor": "color",
            "HistIntervalLabelPosition": "pos",
        },
        "HistMeanOptions" : {
            "Rotate": "Rotate",
            "PlotOptions.color": "color",
            "PlotOptions.lw": "lw",
            "PlotOptions.ls": "ls",
        },
        "HistMeanLabelOptions": {
            "AxesLabelFontOptions": "fontdict",
            "HistMeanLabelColor": "color",
            "HistMeanLabelPosition": "pos",
        },
        "HistSigmaOptions": {
            "Index": "Index",
            "Rotate": "Rotate",
            "PlotOptions.color": "color",
            "StDev": "StDev",
        },
        "HistSigmaLabelOptions": {
            "AxesLabelFontOptions": "fontdict",
            "HistSigmaLabelColor": "color",
            "HistSigmaLabelPosition": "pos",
        },
        "LegendOptions": {
            "LegendAnchor": "bbox_to_anchor",
            "LegendFontOptions": "prop",
            "LegendLocation": "loc",
            "LegendNCol": "ncol",
            "ShowLegend": "ShowLegend",
        },
        "LegendFontOptions": {
            "FontOptions.family": "family",
            "FontOptions.size": "size",
            "FontOptions.stretch": "stretch",
            "FontOptions.style": "style",
            "FontOptions.variant": "variant",
            "FontOptions.weight": "weight",
            "LegendFontName": "family",
            "LegendFontSize": "size",
            "LegendFontStretch": "stretch",
            "LegendFontStyle": "style",
            "LegendFontVariant": "variant",
            "LegendFontWeight": "weight",
        },
        "MarkerOptions": {
            "MarkerColor": "color",
            "MarkerSize": "markersize",
        },
        "MinMaxOptions": {},
        "PlotOptions": {
            "Index": "Index",
            "Rotate": "Rotate",
            "Label": "label",
            "PlotColor": "color",
            "PlotLineWidth": "lw",
            "PlotLineStyle": "ls",
        },
        "ScatterOptions": {
            "Index": "Index",
            "Rotate": "Rotate",
            "ScatterColor": "c",
            "ScatterSize": "s",
        },
        "TickOptions": {
            "TickFontSize": "labelsize",
            "TickRotation": "rotation",
            "TickSize": "size",
        },
        "XTickOptions": {
            "TickOptions.labelsize": "labelsize",
            "TickOptions.rotation": "rotation",
            "TickOptions.size": "size",
            "XTickFontSize": "labelsize",
            "XTickRotation": "rotation",
            "XTickSize": "size",
        },
        "YTickOptions": {
            "TickOptions.labelsize": "labelsize",
            "TickOptions.rotation": "rotation",
            "TickOptions.size": "size",
            "YTickFontSize": "labelsize",
            "YTickRotation": "rotation",
            "YTickSize": "size",
        },
        "UncertaintyOptions": {},
    }

   # --- Conflicting Options ---
    # Aliases to merge for subcategory options
    _kw_subalias = {
        "ContourOptions": {
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
        "HistGaussianOptions": {
            "linewidth": "lw",
            "linestyle": "ls",
            "c": "color",
        },
        "HistOptions": {
            "linewidth": "lw",
            "linestyle": "ls",
            "c": "color",
        },
        "HistDeltaOptions": {
            "linewidth": "lw",
            "linestyle": "ls",
            "c": "color",
        },
        "HistMeanOptions": {
            "linewidth": "lw",
            "linestyle": "ls",
            "c": "color",
        },
        "HistIntervalOptions": {
            "linewidth": "lw",
            "linestyle": "ls",
            "c": "color",
        },
        "PlotOptions": {
            "linewidth": "lw",
            "linestyle": "ls",
            "c": "color",
        },
        "MarkerOptions": {
            "ms": "markersize",
            "linewidth": "lw",
            "linestyle": "ls",
            "c": "color",
        },
        "HistSigmaOptions": {
            "linewidth": "lw",
            "linestyle": "ls",
            "c": "color",
        },
    }

   # --- Documentation Data ---
    # Type strings
    _rst_types = {
        # A-E
        "AdjustBottom": _rst_float,
        "AdjustLeft": _rst_float,
        "AdjustRight": _rst_float,
        "AdjustTop": _rst_float,
        "AxesOptions": _rst_dict,
        "AxesLabelColor": """{``None``} | :class:`str` | :class:`tuple`""",
        "AxesLabelFontName": _rst_str,
        "AxesLabelFontOptions": _rst_dict,
        "AxesLabelFontSize": _rst_strnum,
        "AxesLabelFontStretch": _rst_strnum,
        "AxesLabelFontStyle": ("""{``None``} | ``"normal"`` | """ +
            """``"italic"`` | ``"oblique"``"""),
        "AxesLabelFontVariant": '{``None``} | ``"normal"`` | ``"small-caps"``',
        "AxesLabelFontWeight": _rst_strnum,
        "AxesLabelOptions": _rst_dict,
        "AxesLabelPosition": _rst_int,
        "BottomSpine": """{``None``} | ``True`` | ``False`` | ``"clipped"``""",
        "BottomSpineMax": _rst_float,
        "BottomSpineMin": _rst_float,
        "BottomSpineOptions": _rst_dict,
        "BottomSpineTicks": _rst_booln,
        "BottomTickLabels": _rst_booln,
        "ContourColorMap": _rst_str,
        "ContourLevels": "{``None``} | :class:`int` | :class:`np.ndarray`",
        "ContourOptions": _rst_dict,
        "ContourType": "{``tricontourf``} | ``tricontour`` | ``tripcolor``",
        "Coverage": _rst_float,
        "CoverageCDF": _rst_float,
        "CoverageOptions": _rst_dict,
        "Delta": _rst_float,
        "Density": _rst_boolt,
        "ErrorBarMarker": _rst_str,
        "ErrorBarOptions": _rst_dict,
        "ErrorOptions": _rst_dict,
        "ErrorPlotType": """``"FillBetween"`` | {``"ErrorBar"``}""",
       # F-J
        "FigDPI": _rst_numpos,
        "FigHeight": _rst_floatpos,
        "FigNumber": _rst_intpos,
        "FigOptions": _rst_dict,
        "FigWidth": _rst_floatpos,
        "FillBetweenOptions": _rst_dict,
        "FilterSigma": _rst_float,
        "FontName": _rst_str,
        "FontOptions": _rst_dict,
        "FontSize": _rst_strnum,
        "FontStretch": _rst_strnum,
        "FontStyle": ("""{``None``} | ``"normal"`` | """ +
            """``"italic"`` | ``"oblique"``"""),
        "FontVariant": """{``None``} | ``"normal"`` | ``"small-caps"``""",
        "FontWeight": _rst_strnum,
        "Grid": _rst_boolt,
        "GridColor": """{``None``} | :class:`str` | :class:`tuple`""",
        "GridOptions": _rst_dict,
        "HistBins": _rst_intpos,
        "HistColor": """{``None``} | :class:`str` | :class:`tuple`""",
        "HistDeltaLabelOptions": _rst_dict,
        "HistDeltaLabelColor": """{``None``} | :class:`str` | :class:`tuple`""",     
        "HistDeltaLabelName": """{``None``} | :class:`str`""",
        "HistDeltaLabelPosition":_rst_strnum,
        "HistDeltaOptions": _rst_dict,
        "HistGaussianOptions": _rst_dict,
        "HistIntervalLabelOptions": _rst_dict,
        "HistIntervalLabelColor": """{``None``} | :class:`str` | :class:`tuple`""",     
        "HistIntervalLabelPosition":_rst_strnum,
        "HistIntervalOptions": _rst_dict,
        "HistMeanOptions": _rst_dict,
        "HistMeanLabelOptions": _rst_dict,
        "HistMeanLabelColor": """{``None``} | :class:`str` | :class:`tuple`""",     
        "HistMeanLabelPosition":_rst_strnum,
        "HistSigmaOptions": _rst_dict,
        "HistSigmaLabelOptions": _rst_dict,
        "HistSigmaLabelColor": """{``None``} | :class:`str` | :class:`tuple`""",     
        "HistSigmaLabelPosition":_rst_strnum,
        "HistOptions": _rst_dict,
        "ImageExtent": """{``None``} | :class:`tuple` | :class:`list`""",
        "ImageXCenter": _rst_float,
        "ImageXMax": _rst_float,
        "ImageXMin": """{``0.0``} | :class:`float`""",
        "ImageYCenter": """{``0.0``} | :class:`float`""",
        "ImageYMax": _rst_float,
        "ImageYMin": _rst_float,
        "Index": """{``0``} | :class:`int` >=0""",
       # K-O
        "KeepAspect": _rst_booln,
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
        "MarginHSpace": _rst_float,
        "MarginLeft": _rst_float,
        "MarginRight": _rst_float,
        "MarginTop": _rst_float,
        "MarginVSpace": _rst_float,
        "MarkPoints": _rst_boolt,
        "MarkerColor": """{``None``} | :class:`str` | :class:`tuple`""",
        "MarkerOptions": _rst_dict,
        "MarkerSize": """{``None``} | :class:`int` | :class:`float`""",
        "MeanOptions": _rst_dict,
        "MinMaxPlotType": """{``"FillBetween"``} | ``"ErrorBar"``""",
        "MinMaxOptions": _rst_dict,
        "MinorGrid": _rst_boolf,
        "MinorGridOptions": _rst_dict,
        "NGauss": """{``151``} | :class:`int`""",
       # P-T
        "Pad": _rst_float,
        "PlotColor": """{``None``} | :class:`str` | :class:`tuple`""",
        "PlotFormat": _rst_str,
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
        "ScatterColor": "{``None``} | :class:`np.ndarray` | :class:`list`",
        "ScatterOptions": _rst_dict,
        "ScatterSize": "{``None``} | :class:`np.ndarray` | :class:`float`",
        "SeamDY": _rst_float,
        "ShowError": _rst_booln,
        "ShowHistDelta": _rst_booln,
        "ShowHistDeltaLabel": _rst_booln,
        "ShowHistGaussian": _rst_booln,
        "ShowHistInterval": _rst_booln,
        "ShowHistIntervalLabel": _rst_booln,
        "ShowHistMean": _rst_booln,
        "ShowHistMeanLabel": _rst_booln,
        "ShowHistSigma": _rst_booln,
        "ShowHistSigmaLabel": _rst_booln,
        "ShowMinMax": _rst_booln,
        "ShowUncertainty": _rst_booln,
        "SpineOptions": _rst_dict,
        "StDev":  "{``None``} | :class:`np.ndarray` | :class:`float` | :class:`list` | :class:`tuple`",
        "Subplot": """{``None``} | :class:`Axes` | :class:`int`""",
        "SubplotCols": _rst_intpos,
        "SubplotList": r"""{``None``} | :class:`list`\ [:class:`int`]""",
        "SubplotRows": _rst_intpos,
        "SubplotRubber": _rst_int,
        "TopSpine": """{``None``} | ``True`` | ``False`` | ``"clipped"``""",
        "TopSpineMax": _rst_float,
        "TopSpineMin": _rst_float,
        "TopSpineOptions": _rst_dict,
        "TopSpineTicks": _rst_booln,
        "TopTickLabels": _rst_booln,
       # U-Z
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
       # a-z
        "ax": """{``None``} | :class:`matplotlib.axes._subplots.Axes`""",
        "acov": r""":class:`float`""",
        "bcov": r""":class:`float`""",
        "fig": """{``None``} | :class:`matplotlib.figure.Figure`""",
        "mu": r""":class:`float`""",
        "std": r""":class:`float`""",
        "uy": r""":class:`np.ndarray`\ [:class:`float`]""",
        "ux": r""":class:`np.ndarray`\ [:class:`float`]""",
        "uz": r""":class:`np.ndarray`\ [:class:`float`]""",
        "v": r""":class:`np.ndarray`""",
        "x": r""":class:`np.ndarray`\ [:class:`float`]""",
        "xerr": r""":class:`np.ndarray`\ [:class:`float`]""",
        "y": r""":class:`np.ndarray`\ [:class:`float`]""",
        "yerr": r""":class:`np.ndarray`\ [:class:`float`]""",
        "ymax": r""":class:`np.ndarray`\ [:class:`float`]""",
        "ymin": r""":class:`np.ndarray`\ [:class:`float`]""",
        "z": r""":class:`np.ndarray`\ [:class:`float`]""",
    }
    # Option descriptions
    _rst_descriptions = {
       # A-E
        "AdjustBottom": """Figure-scale coordinates of bottom of axes""",
        "AdjustLeft": """Figure-scale coordinates of left side of axes""",
        "AdjustRight": """Figure-scale coordinates of right side of axes""",
        "AdjustTop": """Figure-scale coordinates of top of axes""",
        "AxesLabelColor": "Color option to :func:`plt.text` for axlabel",
        "AxesLabelFontName": "Font name (types like ``sans-serif`` allowed)",
        "AxesLabelFontOptions": "*fontdict* for :func:`plt.text`",
        "AxesLabelFontSize": 'Font size (options like ``"small"`` allowed)',
        "AxesLabelFontStretch": "Stretch option for :func:`axlabel`",
        "AxesLabelFontStyle": """Font style/slant""",
        "AxesLabelFontVariant": """Font capitalization variant""",
        "AxesLabelFontWeight": 'Numeric weight 0-1000 or ``"bold"``, etc.',
        "AxesLabelOptions": """Options for axes labels""",
        "AxesLabelPosition": """Position code for axes label""",
        "AxesOptions": """Options to :class:`AxesSubplot`""",
        "BottomSpine": "Turn on/off bottom plot spine",
        "BottomSpineMax": "Maximum *x* coord for bottom plot spine",
        "BottomSpineMin": "Minimum *x* coord for bottom plot spine",
        "BottomSpineTicks": "Turn on/off labels on bottom spine",
        "BottomSpineOptions": "Additional options for bottom spine",
        "BottomTickLabels": "Turn on/off tick labels on bottom spine",
        "ContourColorMap": """Color map for contour plots""",
        "ContourLevels": """Number or list of levels for contour plots""",
        "ContourOptions": """Options to :func:`plt.tricontour` and variants""",
        "ContourType": """Contour type specifier""",
        "Coverage": """Coverage interval""",
        "CoverageCDF": """Coverage interval cumulative density function""",
        "CoverageOptions": """Coverage interval options""",
        "Delta": """Reference delta for histogram delta plot""",
        "DeltaOptions": """Options for histogram delta plot""",
        "Density": """Option to scale histogram plots""",
        "ErrorBarMarker": """Marker for :func:`errorbar` plots""",
        "ErrorBarOptions": """Options for :func:`errorbar` plots""",
        "ErrorOptions": """Options for error plots""",
        "ErrorPlotType": """Plot type for "error" plots""",
       # F-J
        "FigDPI": "Figure resolution in dots per inch",
        "FigHeight": "Figure height [inches]",
        "FigNumber": "Figure number",
        "FigOptions": """Options to :class:`matplotlib.figure.Figure`""",
        "FigWidth": "Figure width [inches]",
        "FillBetweenOptions": """Options for :func:`fill_between` plots""",
        "FilterSigma": """Sigma multiplier for outlier sigma""",
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
        "GridColor": """Color passed to *GridOptions*""",
        "GridOptions": """Plot options for major grid""",
        "HistBins" : """Number of histogram bins passed to *HistOptions*""",
        "HistColor": """Histogram passed to *HistOptions*""",
        "HistDeltaLabelOptions": """Options for delta labeling on histograms""",
        "HistDeltaLabelColor": """Color for histogram delta label""",
        "HistDeltaLabelName": """Name of coefficient for histogram delta label""",
        "HistDeltaLabelPosition": """Position for histogram delta label""",
        "HistDeltaOptions": """Options for delta plotting on histograms""",
        "HistGaussianOptions": """Plot options for gaussian plot on histogram""",
        "HistIntervalLabelOptions": """Options for interval labeling on histograms""",
        "HistIntervalLabelColor": """Color for histogram interval label""",
        "HistIntervalLabelPosition": """Position for histogram interval label""",
        "HistIntervalOptions": """Options for interval plotting on histograms""",
        "HistMeanOptions": """Options for mean plotting on histograms""",
        "HistMeanLabelOptions": """Options for mean labeling on histograms""",
        "HistMeanLabelColor": """Color for histogram mean label""",
        "HistMeanLabelPosition": """Position for histogram mean label""",
        "HistOptions": """Plot options for histograms""",
        "HistSigmaOptions": """Options for mean plotting on histograms""",
        "HistSigmaLabelOptions": """Options for mean labeling on histograms""",
        "HistSigmaLabelColor": """Color for histogram mean label""",
        "HistSigmaLabelPosition": """Position for histogram mean label""",
        "ImageXMin": "Coordinate for left edge of image",
        "ImageXMax": "Coordinate for right edge of image",
        "ImageXCenter": "Horizontal center coord if *x* edges not specified",
        "ImageYMin": "Coordinate for bottom edge of image",
        "ImageYMax": "Coordinate for top edge of image",
        "ImageYCenter": "Vertical center coord if *y* edges not specified",
        "ImageExtent": ("Spec for *ImageXMin*, *ImageXMax*, " +
            "*ImageYMin*, *ImageYMax*"),
        "Index": """Index to select specific option from lists""",
   
       # K-O
        "KeepAspect": ("""Keep aspect ratio; default is ``True`` unless""" +
            """``ax.get_aspect()`` is ``"auto"``"""),
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
        "MarginHSpace": "Figure fraction for horizontal space between axes",
        "MarginLeft": "Figure fraction from left edge to left-most label",
        "MarginRight": "Figure fraction from right edge to right-most label",
        "MarginTop": "Figure fraction from top edge to top-most label",
        "MarginVSpace": "Figure fraction for vertical space between axes",
        "MarkPoints": "Put a marker at contributing data points",
        "MarkerColor": "Color for markers in *MarkerOptions*",
        "MarkerOptions": "Options for markers on non-plot() functions",
        "MarkerSize": "*markersize* passed to *MarkerOptions*",
        "MeanOptions": "Options for plotting mean on histogram plots",
        "MinMaxOptions": "Options for error-bar or fill-between min/max plot",
        "MinMaxPlotType": """Plot type for min/max plot""",
        "MinorGrid": """Turn on/off grid at minor ticks""",
        "MinorGridOptions": """Plot options for minor grid""",
        "NGauss": """Number of points to sample Gaussian plot""",
       # P-T
        "Pad": "Padding to add to both axes, *ax.set_xlim* and *ax.set_ylim*",
        "PlotColor": """Color option to :func:`plt.plot` for primary curve""",
        "PlotFormat": """Format specifier as third arg to :func:`plot`""",
        "PlotLineStyle": """Line style for primary :func:`plt.plot`""",
        "PlotLineWidth": """Line width for primary :func:`plt.plot`""",
        "PlotOptions": """Options to :func:`plt.plot` for primary curve""",
        "RightSpine": "Turn on/off right plot spine",
        "RightSpineMax": "Maximum *y* coord for right plot spine",
        "RightSpineMin": "Minimum *y* coord for right plot spine",
        "RightSpineTicks": "Turn on/off labels on right spine",
        "RightSpineOptions": "Additional options for right spine",
        "RightTickLabels": "Turn on/off tick labels on right spine",
        "Rotate": """Option to flip *x* and *y* axes""",
        "ScatterColor": (
            "Color or color description to use for each data point; " +
            "usually an array of floats that maps into color map"),
        "ScatterOptions": """Options to :func:`plt.scatter`""",
        "ScatterSize": "Size [pt^2] of marker for each data point",
        "SeamDY": "Vertical offset for seam plots",
        "ShowError": """Show "error" plot using *xerr*""",
        "ShowHistDelta": """"Show delta bounds on histrogram plot""",
        "ShowHistDeltaLabel": """Show delta label on histogram plot""",
        "ShowHistGaussian": """Show Gaussian plot on histogram""",
        "ShowHistInterval": """Show interval plot on histogram""",
        "ShowHistIntervalLabel": """Show interval mabel on histogram""",
        "ShowHistMean": """Plot *mu* on histogram""",
        "ShowHistMeanLabel": """Show *mu* mabel on histogram""",
        "ShowHistSigma": """Plot *sigma* bounds on histogram""",
        "ShowHistSigmaLabel": """Plot *sigma* label on histogram""",
        "ShowMinMax": """Plot *ymin* and *ymax* at each point""",
        "ShowUncertainty": """Plot uncertainty bounds""",
        "SpineOptions": """Options for all spines""",
        "StDev": """*sigma* multipliers for sigma bounds""",
        "Subplot": "Subplot index (1-based)",
        "SubplotCols": "Expected number of subplot columns",
        "SubplotList": "List of subplots to put in row/column",
        "SubplotRows": "Expected number of subplot rows",
        "SubplotRubber": "Index of subplot to expand",
        "TopSpine": "Turn on/off top plot spine",
        "TopSpineMax": "Maximum *x* coord for top plot spine",
        "TopSpineMin": "Minimum *x* coord for top plot spine",
        "TopSpineTicks": "Turn on/off labels on top spine",
        "TopSpineOptions": "Additional options for top spine",
        "TopTickLabels": "Turn on/off tick labels on top spine",
       # U-Z
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
       # a-z
        "ax": """Handle to existing axes""",
        "acov": """First edge of coverage interval""",
        "bcov": """Second edge of coverage interval""",
        "fig": """Handle to existing figure""",
        "mu": """Sample mean""",
        "std": "Sample standard deviation""",
        "ux": """UQ *x* magintudes""",
        "uy": """UQ magnitudes""",
        "v": """Values for histogram plot""",
        "x": """Nominal *x* values to plot""",
        "xerr": """*x* widths for error plots""",
        "y": """Nominal *y* values to plot""",
        "yerr": """Error magnitudes""",
        "ymin": """Max values for min/max plots""",
        "ymax": """Max values for min/max plots""",
        "z": """Contour levels to plot""",
    }
    
   # --- RC ---
    # Default values
    _rc = {
        "ShowLine": True,
        "ShowError": False,
        "ShowHistDelta": False,
        "ShowHistDeltaLabel": False, 
        "ShowHistInterval": False,
        "ShowHistIntervalLabel": False, 
        "ShowHistGaussian" : False,
        "ShowHistMean": False,
        "ShowHistMeanLabel": False,
        "ShowHistSigma": False,
        "ShowHistSigmaLabel": False,
        "Index": 0,
        "Rotate": False,
        "AxesLabelOptions": {},
        "AxesLabelFontOptions": {},
        "AxesOptions": {},
        "ContourOptions": {
            "cmap": "viridis",
            "alpha": 1,
        },
        "CoverageOptions":{
            "Coverage":.99,
        },
        "ErrorBarOptions": {},
        "ErrorOptions": {},
        "FigOptions": {
            "figwidth": 5.5,
            "figheight": 4.4,
        },
        "FillBetweenOptions": {
            "alpha": 0.2,
            "lw": 0,
            "zorder": 4,
        },
        "FontOptions": {
            "family": "DejaVu Sans",
        },
        "GridOptions": {
            "ls": ":",
            "lw": 0.9,
            "color": "#a0a0a0",
        },
        "HistDeltaOptions":{
            "Delta": 0.0,
            'color': "r",
            'ls': "--",
            'lw': 1.0,
            'zorder': 3,
        },
        "HistDeltaLabelOptions": {
            "HistDeltaLabelColor": 'navy',
            "HistDeltaLabelName": 'coeff'
        },
        "HistGaussianOptions": {
            "color": "navy",
            "lw": 1.5,
            "zorder": 7,
            "label": "Normal Distribution",
        },
        "HistOptions": {
            "bins" : 20,
            "color" : "c",
            "zorder" : 2,
            "edgecolor" : "k",
            "lw" : 1, 
            "density" : True,
        },
        "HistMeanOptions": {
            "color": 'k',
            "lw": 2,
            "zorder": 6,
            "label": "Mean value",
        },
        "HistMeanLabelOptions": {
            "HistMeanLabelColor": 'k'
        },
        "HistIntervalOptions": {
            "lw": 0,
            "zorder": 1,
            "alpha": 0.2,
        },
        "HistIntervalLabelOptions": {
            "HistIntervalLabelColor": 'k'
        },
        "HistSigmaOptions": {
            'color': 'navy',
            'lw': 2,
            'zorder': 5,
            "dashes": [4, 2],
            'StDev': 3,
        },
        "HistSigmaLabelOptions": {
            "HistSigmaLabelColor": 'r'
        },
        "LegendFontOptions": {},
        "LegendOptions": {
            "loc": "upper center",
            "labelspacing": 0.5,
            "framealpha": 1.0,
        },
        "MarkerOptions": {
            "color": "k",
            "marker": "o",
            "markersize": 4,
            "ls": "-",
            "lw": 0,
            "zorder": 7,
        },
        "MinMaxOptions": {},
        "MinMaxPlotType": "FillBetween",
        "MinorGridOptions": {
            "ls": ":",
            "lw": 0.5,
            "color": "#b0b0b0",
        },
        "ScatterOptions": {
            "marker": "d",
            "zorder": 7,
        },
        "PlotOptions": {
            "color": ["b", "k", "darkorange", "g"],
            "ls": "-",
            "zorder": 8,
        },
        "TickOptions": {},
        "XTickOptions": {},
        "YTickOptions": {},
    }

    # Options for sections
    _rc_sections = {
        "figure": {},
        "axes": {},
        "axformat": {
            "Pad": 0.05,
        },
        "axadjust": {},
        "contour": {},
        "plot": {},
        "error": {},
        "minmax": {},
        "uq": {},
        "fillbetween": {},
        "errorbar": {
            "capsize": 1.5,
            "lw": 0.5,
            "elinewidth": 0.8,
            "zorder": 6,
        },
        "imshow": {},
        "grid": {
            "Grid": True,
            "MajorGrid": True,
        },
        "hist": {},
        "legend": {
            "loc": "upper center",
            "labelspacing": 0.5,
            "framealpha": 1.0,
        },
        "font": {},
        "spines": {
            "Spines": True,
            "Ticks": True,
            "TickDirection": "out",
            "RightSpineTicks": False,
            "TopSpineTicks": False,
            "RightSpine": False,
            "TopSpine": False,
        },
        "histlbl": {
            'color': 'k',
            'horizontalalignment': 'right',
            'verticalalignment': 'top',
        },
    }
  # >

  # ==================
  # Categories
  # ==================
  # <
    # Subplot column options
    def axadjust_col_options(self):
        r"""Process options for axes margin adjustment

        :Call:
            >>> kw = opts.axadjust_col_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`axes_adjust_col`
        :Versions:
            * 2020-01-10 ``@ddalle``: First version
            * 2020-01-18 ``@ddalle``: Using :class:`KwargHandler`
        """
        return self.section_options("axadjust_col")

    # Subplot row options
    def axadjust_row_options(self):
        r"""Process options for axes margin adjustment

        :Call:
            >>> kw = opts.axadjust_row_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`axes_adjust_row`
        :Versions:
            * 2020-01-10 ``@ddalle``: First version
            * 2020-01-18 ``@ddalle``: Using :class:`KwargHandler`
        """
        return self.section_options("axadjust_row")

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
        :Outputs:            "CoverageCDF": CoverageOptions.Coverage,
            * 2020-01-08 ``@ddalle``: First version
            * 2020-01-18 ``@ddalle``: Using :class:`KwargHandler`
        """
        return self.section_options("axadjust")

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
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Use the "axes" section
        return self.section_options("axes")

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
            * 2020-01-18 ``@ddalle``: Using :class:`KwargHandler`
        """
        return self.section_options("axformat")

    # Options for coverage intervals
    def coverage_options(self):
        r"""Process options for coverage intervals

        :Call:
            >>> kw = opts.coverage_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options for coverage intervals
        :Versions:
            * 2019-03-05 ``@ddalle``: First version
            * 2019-12-21 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Specific options
        return self.get_option("CoverageOptions")

    # Options for delta plots
    def delta_options(self):
        r"""Process options for delta plots

        :Call:
            >>> kw = opts.delta_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options for mpl._plot
        :Versions:
            * 2019-03-05 ``@ddalle``: First version
            * 2019-12-21 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Specific options
        return self.get_option("HistDeltaOptions")

    def deltalabel_options(self):
        r"""Process options for delta labels

        :Call:
            >>> kw = opts.deltalabel_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options for mpl.axlabel
        :Versions:
            * 2019-03-05 ``@ddalle``: First version
            * 2019-12-21 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Specific options
        return self.get_option("HistDeltaLabelOptions")

    # Process options for "error" plot
    def error_options(self):
        r"""Process options for error plots

        :Call:
            >>> kw = opts.error_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`error`
        :Versions:
            * 2019-03-04 ``@ddalle``: First version
            * 2019-12-23 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Use the "minmax" section
        kw = self.section_options("error")
        # Save section name
        mainopt = "ErrorOptions"
        # Get type-specific options removed
        kw_eb = kw.pop("ErrorBarOptions", {})
        kw_fb = kw.pop("FillBetweenOptions", {})
        kw_mm = kw.get(mainopt, {})
        # Get the plot type
        mmax_type = kw.get("MinMaxPlotType", "fillbetween").lower()
        mmax_type = mmax_type.replace("_", "")
        # Check type
        if mmax_type == "errorbar":
            # Combine ErrorBar options into main options
            kw[mainopt] = dict(kw_eb, **kw_mm)
        else:
            # Combine FillBetween options into main options
            kw[mainopt] = dict(kw_fb, **kw_mm)
        # Output
        return kw

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
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Specific options
        return self.get_option("ErrorBarOptions")

    # Figure creation and manipulation
    def figure_options(self):
        r"""Process options specific to Matplotlib figure

        :Call:
            >>> kw = figure_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`plt.figure`
        :Versions:
            * 2019-03-06 ``@ddalle``: First version
            * 2019-12-20 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Use the "figure" section
        return self.section_options("fig")

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
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Specific options
        return self.get_option("FillBetweenOptions")

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
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Use the "font" section and only return "FontOptions"
        return self.section_options("font", "FontOptions")

    # Gauss options
    def gauss_options(self):
        r"""Process options to gaussian plot curve

        :Call:
            >>> kw = opts.gauss_options()
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
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Get default options for HistGaussianOptions
        kw = self.get_option("HistGaussianOptions")
        return kw

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
        :Versions:
            * 2019-03-07 ``@jmeeroff``: First version
            * 2019-12-23 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        return self.section_options("grid")

    # Histogram label options
    def histlabel_options(self):
        r"""Process options to gaussian plot curve

        :Call:
            >>> kw = opts.gauss_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`plot`
        :Versions:
            * 2019-03-07 ``@ddalle``: First ver        "gauss": {
            "color": "navy",
            "lw": 1.5,
            "zorder": 7,
            "label": "Normal Distribution",
        },:class:`KwargHandler`
        """
        # Use the "gauss" section
        kw = self.section_options("histlbl")
        return kw

    # Process imshow() options
    def imshow_options(self):
        r"""Process options for image display calls

        :Call:
            >>> kw = opts.imshow_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`imshow`
        :Versions:
            * 2020-01-09 ``@ddalle``: First version
            * 2020-01-18 ``@ddalle``: Using :class:`KwargHandler`
        """
        return self.section_options("imshow")

    # Options for intervals on histogram plots
    def interval_options(self):
        r"""Process options for :func:`interval` calls

        :Call:
            >>> kw = opts.interval_options()
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
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Specific options
        kw = self.get_option("HistIntervalOptions")
        return kw


    def intervallabel_options(self):
        r"""Process options for interval label calls

        :Call:
            >>> kw = opts.intervallabel_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options for interval labels
        :Versions:
            * 2019-03-05 ``@ddalle``: First version
            * 2019-12-21 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Specific options
        kw = self.get_option("HistIntervalLabelOptions")
        return kw

    # Options for font in legend
    def legend_font_options(self):
        r"""Process font options for :func:`legend` calls

        :Call:
            >>> kw = opts.legend_font_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`errorbar`
        :Versions:
            * 2020-01-19 ``@ddalle``: First version
        """
        # Specific options
        return self.get_option("LegendFontOptions")

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
            * 2020-01-98 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Get *LegendOptions* options
        kw = self.get_option("LegendOptions")
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
        return kw

    # Mean options
    def mean_options(self):
        r"""Process options plot mean on histogram

        :Call:
            >>> kw = opts.mean_options()
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
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Use mean options section
        kw = self.get_option("HistMeanOptions")
        return kw

    def meanlabel_options(self):
        r"""Process options to histogram mean labels

        :Call:
            >>> kw = opts.meanlabe;_options()
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
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Use the "gauss" section
        kw = self.get_option("HistMeanLabelOptions")
        return kw

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
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Use the "plot" section and only return "PlotOptions"
        return self.section_options("plot", "PlotOptions")

    # Process options for contour plots
    def contour_options(self):
        r"""Process options for contour plots

        :Call:
            >>> kw = opts.contour_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`contour`
        :Versions:
            * 2020-03-26 ``@jmeeroff``: First version
        """
        # Use the "contour" section and only return "ContourOptions"
        return self.section_options("contour", "ContourOptions")

    # Process options for contour plots
    def hist_options(self):
        r"""Process options for histogram plots

        :Call:
            >>> kw = opts.hist_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`hist`
        :Versions:
            * 2020-04-23 ``@jmeeroff``: First version
        """
        # Use the "hist" section and only return "HistOptions"
        return self.section_options("hist", "HistOptions")


    # Process options for min/max plot
    def minmax_options(self):
        r"""Process options for min/max plots

        :Call:
            >>> kw = opts.minmax_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`minmax`
        :Versions:
            * 2019-03-04 ``@ddalle``: First version
            * 2019-12-20 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Use the "minmax" section
        kw = self.section_options("minmax")
        # Save section name
        mainopt = "MinMaxOptions"
        # Get type-specific options removed
        kw_eb = kw.pop("ErrorBarOptions", {})
        kw_fb = kw.pop("FillBetweenOptions", {})
        kw_mm = kw.get(mainopt, {})
        # Get the plot type
        mmax_type = kw.get("MinMaxPlotType", "fillbetween").lower()
        mmax_type = mmax_type.replace("_", "")
        # Check type
        if mmax_type == "errorbar":
            # Combine ErrorBar options into main options
            kw[mainopt] = dict(kw_eb, **kw_mm)
        else:
            # Combine FillBetween options into main options
            kw[mainopt] = dict(kw_fb, **kw_mm)
        # Output
        return kw

    # Process options for UQ plot
    def uq_options(self):
        r"""Process options for uncertainty quantification plots

        :Call:
            >>> kw = opts.uq_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`uq`
        :Versions:
            * 2019-03-04 ``@ddalle``: First version
            * 2019-12-23 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Use the "uq" section
        kw = self.section_options("uq")
        # Save section name
        mainopt = "UncertaintyOptions"
        # Get type-specific options removed
        kw_eb = kw.pop("ErrorBarOptions", {})
        kw_fb = kw.pop("FillBetweenOptions", {})
        kw_mm = kw.get(mainopt, {})
        # Get the plot type
        mmax_type = kw.get("UncertaintyPlotType", "fillbetween").lower()
        mmax_type = mmax_type.replace("_", "")
        # Check type
        if mmax_type == "errorbar":
            # Combine ErrorBar options into main options
            kw[mainopt] = dict(kw_eb, **kw_mm)
        else:
            # Combine FillBetween options into main options
            kw[mainopt] = dict(kw_fb, **kw_mm)
        # Output
        return kw

   # Options for scatter plots
    def scatter_options(self):
        r"""Process options for :func:`scatter` calls

        :Call:
            >>> kw = opts.scatter_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`scatter`
        :Versions:
            * 2019-03-05 ``@ddalle``: First version
            * 2019-12-21 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Specific options
        kw = self.get_option("ScatterOptions")
        return kw

    # Options for sigma bounds on histogram plots
    def sigma_options(self):
        r"""Process options for :func:`sigma` calls

        :Call:
            >>> kw = opts.sigma_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`_plot`
        :Versions:
            * 2019-03-05 ``@ddalle``: First version
            * 2019-12-21 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Specific options
        kw = self.get_option("HistSigmaOptions")
        return kw

    def sigmalabel_options(self):
        r"""Process options for :func:`sigma` labels

        :Call:
            >>> kw = opts.sigma_options()
        :Inputs:
            *opts*: :class:`MPLOpts`
                Options interface
        :Keys:
            %(keys)s
        :Outputs:
            *kw*: :class:`dict`
                Dictionary of options to :func:`axlabel``
        :Versions:
            * 2019-03-05 ``@ddalle``: First version
            * 2019-12-21 ``@ddalle``: From :mod:`tnakit.mpl.mplopts`
            * 2020-01-17 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Specific options
        kw = self.get_option("HistSigmaLabelOptions")
        return kw

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
            * 2020-01-20 ``@ddalle``: Using :class:`KwargHandler`
        """
        # Use the "spines" section, cascading opts handled internally
        return self.section_options("spines")
  # >


# Document sublists
MPLOpts._doc_keys("axadjust_options", "axadjust")
MPLOpts._doc_keys("axadjust_col_options", "axadjust_col")
MPLOpts._doc_keys("axadjust_row_options", "axadjust_row")
MPLOpts._doc_keys("axformat_options", "axformat")
MPLOpts._doc_keys("axes_options", "axes")
MPLOpts._doc_keys("contour_options", "contour")
MPLOpts._doc_keys("error_options", "error")
MPLOpts._doc_keys("figure_options", "fig")
MPLOpts._doc_keys("gauss_options", "histgaussian")
MPLOpts._doc_keys("grid_options", "grid")
MPLOpts._doc_keys("imshow_options", "imshow")
MPLOpts._doc_keys("minmax_options", "minmax")
MPLOpts._doc_keys("plot_options", "plot")
MPLOpts._doc_keys("spine_options", "spines")
MPLOpts._doc_keys("uq_options", "uq")

# Special categories
MPLOpts._doc_keys("errorbar_options", ["ErrorBarOptions"])
MPLOpts._doc_keys("fillbetween_options", ["FillBetweenOptions"])
MPLOpts._doc_keys("legend_font_options", ["LegendFontOptions"])
MPLOpts._doc_keys("legend_options", ["LegendOptions"])
