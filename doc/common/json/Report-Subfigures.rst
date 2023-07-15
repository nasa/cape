----------------------------------
Options for ``Subfigures`` section
----------------------------------


Unique options for default subfigure
====================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"


Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._TableSubfigOpts`
* :mod:`cape.cfdx.options.reportopts.SubfigOpts`

Unique options for ``CoeffTable`` subfigure
===========================================

*CA*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CA"
*CLL*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLL"
*CLM*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLM"
*CLN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLN"
*CN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CN"
*CY*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CY"
*Coefficients*: {``None``} | :class:`list`\ [:class:`str`]
    list of coefficients to detail in table
*Components*: {``None``} | :class:`list`\ [:class:`str`]
    list of components for which to report coefficients
*EpsFormat*: {``None``} | :class:`str`
    printf-style text format for sampling error
*Iteration*: {``None``} | :class:`int`
    specific iteration at which to sample results
*MuFormat*: {``None``} | :class:`str`
    printf-style text format for mean value
*SigmaFormat*: {``None``} | :class:`str`
    printf-sylte text format for standard deviation


Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._TableSubfigOpts`
* :mod:`cape.cfdx.options.reportopts.SubfigOpts`

Unique options for ``Conditions`` subfigure
===========================================

*SkipVars*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys to leave out of table
*SpecialVars*: {``None``} | :class:`list`\ [:class:`str`]
    keys not in run matrix to attempt to calculate


Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._TableSubfigOpts`
* :mod:`cape.cfdx.options.reportopts.SubfigOpts`

Unique options for ``Conditions`` subfigure
===========================================

*SkipVars*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys to leave out of table
*SpecialVars*: {``None``} | :class:`list`\ [:class:`str`]
    keys not in run matrix to attempt to calculate


Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._MPLSubfigOpts`
* :mod:`cape.cfdx.options.reportopts.SubfigOpts`

Unique options for ``ContourCoeff`` subfigure
=============================================

*AxisEqual*: {``True``} | ``False``
    option to scale x and y axes with common scale
*ColorBar*: {``True``} | ``False``
    option to turn on color bar (scale)
*ContourColorMap*: {``'jet'``} | :class:`str`
    name of color map to use w/ contour plots
*ContourOptions*: {``None``} | :class:`dict`
    options passed to contour plot function
*ContourType*: ``'tricontour'`` | {``'tricontourf'``} | ``'tripcolor'``
    contour plotting function/type to use
*LineType*: {``'plot'``} | ``'triplot'``
    plot function to use to mark data points
*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis
*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis

Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``PlotOptions``
----------------------------------




Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._TableSubfigOpts`
* :mod:`cape.cfdx.options.reportopts.SubfigOpts`

Unique options for ``CoeffTable`` subfigure
===========================================

*CA*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CA"
*CLL*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLL"
*CLM*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLM"
*CLN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLN"
*CN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CN"
*CY*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CY"
*Coefficients*: {``None``} | :class:`list`\ [:class:`str`]
    list of coefficients to detail in table
*Components*: {``None``} | :class:`list`\ [:class:`str`]
    list of components for which to report coefficients
*EpsFormat*: {``None``} | :class:`str`
    printf-style text format for sampling error
*Iteration*: {``None``} | :class:`int`
    specific iteration at which to sample results
*MuFormat*: {``None``} | :class:`str`
    printf-style text format for mean value
*SigmaFormat*: {``None``} | :class:`str`
    printf-sylte text format for standard deviation


Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts.SubfigOpts`

Unique options for ``Image`` subfigure
======================================

*ImageFile*: {``'export.png'``} | :class:`str`
    name of image file to copy from case folder


Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts.SubfigOpts`

Unique options for ``Paraview`` subfigure
=========================================

*Command*: {``'pvpython'``} | :class:`str`
    name of Python/Paraview executable to call
*Format*: {``'png'``} | :class:`str`
    image file format
*ImageFile*: {``'export.png'``} | :class:`str`
    name of image file created by *Layout*
*Layout*: {``'layout.py'``} | :class:`str`
    name of Python file to execute with Paraview


Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._IterSubfigOpts`
* :mod:`cape.cfdx.options.reportopts._MPLSubfigOpts`
* :mod:`cape.cfdx.options.reportopts.SubfigOpts`
* :mod:`cape.cfdx.options.reportopts._PlotCoeffSubfigOpts`

Unique options for ``PlotCoeff`` subfigure
==========================================

*CaptionComponent*: {``None``} | :class:`str`
    explicit text for component portion of caption
*Delta*: {``0.0``} | :class:`float`
    specified interval(s) to plot above and below mean
*DeltaFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowDelta* value
*DeltaPlotOptions*: {``None``} | :class:`PlotCoeffIterDeltaPlotOpts`
    plot options for fixed-width above and below mu
*EpsilonFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowEpsilon* value
*EpsilonPlotOptions*: {``None``} | :class:`PlotCoeffIterEpsilonPlotOpts`
    value of option "EpsilonPlotOptions"
*KEpsilon*: {``0.0``} | :class:`float`
    multiple of iterative error to plot
*MuFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowMu* value
*MuPlotOptions*: {``None``} | :class:`PlotCoeffIterMuPlotOpts`
    plot options for horizontal line showing mean
*NAverage*: {``None``} | :class:`int`
    value of option "NAverage"
*ShowDelta*: {``True``} | ``False``
    option to print value of *Delta*
*ShowEpsilon*: ``True`` | {``False``}
    option to print value of iterative sampling error
*ShowMu*: {``True``} | ``False``
    option to print value of mean over window
*ShowSigma*: {``True``} | ``False``
    option to print value of standard deviation
*SigmaFormat*: {``'%.4f'``} | :class:`object`
    printf-style flag for *ShowSigma* value

Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``DeltaPlotOptions``
---------------------------------------



Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``EpsilonPlotOptions``
-----------------------------------------



Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``MuPlotOptions``
------------------------------------



Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``PlotOptions``
----------------------------------




Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._IterSubfigOpts`
* :mod:`cape.cfdx.options.reportopts._MPLSubfigOpts`
* :mod:`cape.cfdx.options.reportopts.SubfigOpts`
* :mod:`cape.cfdx.options.reportopts._PlotCoeffSubfigOpts`

Unique options for ``PlotCoeff`` subfigure
==========================================

*CaptionComponent*: {``None``} | :class:`str`
    explicit text for component portion of caption
*Delta*: {``0.0``} | :class:`float`
    specified interval(s) to plot above and below mean
*DeltaFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowDelta* value
*DeltaPlotOptions*: {``None``} | :class:`PlotCoeffIterDeltaPlotOpts`
    plot options for fixed-width above and below mu
*EpsilonFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowEpsilon* value
*EpsilonPlotOptions*: {``None``} | :class:`PlotCoeffIterEpsilonPlotOpts`
    value of option "EpsilonPlotOptions"
*KEpsilon*: {``0.0``} | :class:`float`
    multiple of iterative error to plot
*MuFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowMu* value
*MuPlotOptions*: {``None``} | :class:`PlotCoeffIterMuPlotOpts`
    plot options for horizontal line showing mean
*NAverage*: {``None``} | :class:`int`
    value of option "NAverage"
*ShowDelta*: {``True``} | ``False``
    option to print value of *Delta*
*ShowEpsilon*: ``True`` | {``False``}
    option to print value of iterative sampling error
*ShowMu*: {``True``} | ``False``
    option to print value of mean over window
*ShowSigma*: {``True``} | ``False``
    option to print value of standard deviation
*SigmaFormat*: {``'%.4f'``} | :class:`object`
    printf-style flag for *ShowSigma* value

Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``DeltaPlotOptions``
---------------------------------------



Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``EpsilonPlotOptions``
-----------------------------------------



Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``MuPlotOptions``
------------------------------------



Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``PlotOptions``
----------------------------------




Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._MPLSubfigOpts`
* :mod:`cape.cfdx.options.reportopts.SubfigOpts`
* :mod:`cape.cfdx.options.reportopts._PlotCoeffSubfigOpts`

Unique options for ``PlotCoeffSweep`` subfigure
===============================================

*MinMax*: ``True`` | {``False``}
    option to plot min/max of value over iterative window
*MinMaxOptions*: {``None``} | :class:`PlotCoeffSweepMinMaxPlotOpts`
    plot options for *MinMax* plot
*Target*: {``None``} | :class:`str`
    name of target databook to co-plot
*TargetOptions*: {``None``} | :class:`PlotCoeffSweepTargetPlotOpts`
    plot options for optional target

Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``MinMaxOptions``
------------------------------------



Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``PlotOptions``
----------------------------------



Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``TargetOptions``
------------------------------------




Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._MPLSubfigOpts`
* :mod:`cape.cfdx.options.reportopts.SubfigOpts`

Unique options for ``ContourCoeff`` subfigure
=============================================

*AxisEqual*: {``True``} | ``False``
    option to scale x and y axes with common scale
*ColorBar*: {``True``} | ``False``
    option to turn on color bar (scale)
*ContourColorMap*: {``'jet'``} | :class:`str`
    name of color map to use w/ contour plots
*ContourOptions*: {``None``} | :class:`dict`
    options passed to contour plot function
*ContourType*: ``'tricontour'`` | {``'tricontourf'``} | ``'tripcolor'``
    contour plotting function/type to use
*LineType*: {``'plot'``} | ``'triplot'``
    plot function to use to mark data points
*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis
*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis

Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``PlotOptions``
----------------------------------




Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._MPLSubfigOpts`
* :mod:`cape.cfdx.options.reportopts.SubfigOpts`

Unique options for ``ContourCoeff`` subfigure
=============================================

*AxisEqual*: {``True``} | ``False``
    option to scale x and y axes with common scale
*ColorBar*: {``True``} | ``False``
    option to turn on color bar (scale)
*ContourColorMap*: {``'jet'``} | :class:`str`
    name of color map to use w/ contour plots
*ContourOptions*: {``None``} | :class:`dict`
    options passed to contour plot function
*ContourType*: ``'tricontour'`` | {``'tricontourf'``} | ``'tripcolor'``
    contour plotting function/type to use
*LineType*: {``'plot'``} | ``'triplot'``
    plot function to use to mark data points
*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis
*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis

Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``PlotOptions``
----------------------------------




Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts.ResidualSubfigOpts`
* :mod:`cape.cfdx.options.reportopts._IterSubfigOpts`
* :mod:`cape.cfdx.options.reportopts._MPLSubfigOpts`
* :mod:`cape.cfdx.options.reportopts.SubfigOpts`

Unique options for ``PlotL1`` subfigure
=======================================



Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._IterSubfigOpts`
* :mod:`cape.cfdx.options.reportopts._MPLSubfigOpts`
* :mod:`cape.cfdx.options.reportopts.SubfigOpts`

Unique options for ``PlotL2`` subfigure
=======================================

*PlotOptions0*: {``None``} | :class:`ResidPlot0Opts`
    plot options for initial residual
*Residual*: {``'L2'``} | :class:`str`
    name of residual field or type to plot

Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``PlotOptions``
----------------------------------



Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``PlotOptions0``
-----------------------------------




Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._MPLSubfigOpts`
* :mod:`cape.cfdx.options.reportopts.SubfigOpts`

Unique options for ``PlotLineLoad`` subfigure
=============================================

*AdjustBottom*: {``0.1``} | :class:`float`
    margin from axes to bottom of figure
*AdjustLeft*: {``0.12``} | :class:`float`
    margin from axes to left of figure
*AdjustRight*: {``0.97``} | :class:`float`
    margin from axes to right of figure
*AdjustTop*: {``0.97``} | :class:`float`
    margin from axes to top of figure
*AutoUpdate*: {``True``} | ``False``
    option to create line loads if not in databook
*Coefficient*: {``None``} | :class:`str`
    coefficient to plot
*Component*: {``None``} | :class:`str`
    config component tp plot
*Orientation*: ``'horizontal'`` | {``'vertical'``}
    orientation of vehicle in line load plot
*SeamCurve*: ``'smy'`` | ``'smz'``
    name of seam curve, if any, to show w/ line loads
*SeamLocation*: ``'bottom'`` | ``'left'`` | ``'right'`` | ``'top'``
    location for optional seam curve plot
*SeamOptions*: {``None``} | :class:`PlotLineLoadSeamPlotOpts`
    plot options for optional seam curve
*SubplotMargin*: {``0.015``} | :class:`float`
    margin between line load and seam curve subplots
*XPad*: {``0.03``} | :class:`float`
    additional padding from data to xmin and xmax w/i axes
*YPad*: {``0.03``} | :class:`float`
    additional padding from data to ymin and ymax w/i axes

Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``PlotOptions``
----------------------------------



Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``SeamOptions``
----------------------------------




Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._IterSubfigOpts`
* :mod:`cape.cfdx.options.reportopts._MPLSubfigOpts`
* :mod:`cape.cfdx.options.reportopts.SubfigOpts`

Unique options for ``PlotL2`` subfigure
=======================================

*PlotOptions0*: {``None``} | :class:`ResidPlot0Opts`
    plot options for initial residual
*Residual*: {``'L2'``} | :class:`str`
    name of residual field or type to plot

Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``PlotOptions``
----------------------------------



Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``PlotOptions0``
-----------------------------------




Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._TableSubfigOpts`
* :mod:`cape.cfdx.options.reportopts.SubfigOpts`

Unique options for ``CoeffTable`` subfigure
===========================================

*CA*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CA"
*CLL*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLL"
*CLM*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLM"
*CLN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLN"
*CN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CN"
*CY*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CY"
*Coefficients*: {``None``} | :class:`list`\ [:class:`str`]
    list of coefficients to detail in table
*Components*: {``None``} | :class:`list`\ [:class:`str`]
    list of components for which to report coefficients
*EpsFormat*: {``None``} | :class:`str`
    printf-style text format for sampling error
*Iteration*: {``None``} | :class:`int`
    specific iteration at which to sample results
*MuFormat*: {``None``} | :class:`str`
    printf-style text format for mean value
*SigmaFormat*: {``None``} | :class:`str`
    printf-sylte text format for standard deviation


Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts.ConditionsTableSubfigOpts`
* :mod:`cape.cfdx.options.reportopts._TableSubfigOpts`
* :mod:`cape.cfdx.options.reportopts.SubfigOpts`

Unique options for ``SweepCases`` subfigure
===========================================



Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._MPLSubfigOpts`
* :mod:`cape.cfdx.options.reportopts.SubfigOpts`
* :mod:`cape.cfdx.options.reportopts._PlotCoeffSubfigOpts`

Unique options for ``PlotCoeffSweep`` subfigure
===============================================

*MinMax*: ``True`` | {``False``}
    option to plot min/max of value over iterative window
*MinMaxOptions*: {``None``} | :class:`PlotCoeffSweepMinMaxPlotOpts`
    plot options for *MinMax* plot
*Target*: {``None``} | :class:`str`
    name of target databook to co-plot
*TargetOptions*: {``None``} | :class:`PlotCoeffSweepTargetPlotOpts`
    plot options for optional target

Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``MinMaxOptions``
------------------------------------



Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``PlotOptions``
----------------------------------



Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts._PlotOptsOpts`

Unique options for ``TargetOptions``
------------------------------------




Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts.ConditionsTableSubfigOpts`
* :mod:`cape.cfdx.options.reportopts._TableSubfigOpts`
* :mod:`cape.cfdx.options.reportopts.SubfigOpts`

Unique options for ``SweepCases`` subfigure
===========================================



Also accepts the options from these classes:

* :mod:`cape.cfdx.options.reportopts.SubfigOpts`

Unique options for ``Tecplot`` subfigure
========================================

*ColorMaps*: {``[]``} | :class:`list`\ [:class:`dict`]
    customized Tecplot colormap
*ContourLevels*: {``None``} | :class:`list`\ [:class:`dict`]
    customized settings for Tecplot contour levels
*FieldMap*: {``None``} | :class:`list`\ [:class:`int`]
    list of zone numbers for Tecplot layout group boundaries
*FigWidth*: {``1024``} | :class:`int`
    width of output image in pixels
*Keys*: {``None``} | :class:`dict`
    dict of Tecplot layout statements to customize
*Layout*: {``None``} | :class:`str`
    template Tecplot layout file
*VarSet*: {``{}``} | :class:`dict`
    variables and their values to define in Tecplot layout


