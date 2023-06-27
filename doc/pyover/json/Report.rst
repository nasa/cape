
.. _pyover-json-report:

******************************
Options for ``Report`` Section
******************************
The options below are the available options in the ``Report`` Section of the ``pyover.json`` control file


*Reports*: {``None``} | :class:`list`\ [:class:`str`]
    list of reports



*Sweeps*: {``None``} | :class:`dict`
    collection of sweep definitions



*Subfigures*: {``None``} | :class:`dict`
    collection of subfigure definitions



*Figures*: {``None``} | :class:`dict`
    collection of figure definitions


Options for all ``Subfigures``
==============================
PlotLInf Options
================

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis



*Caption*: {``None``} | :class:`str`
    subfigure caption



*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values



*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes



*NPlotLast*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotLast"



*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels



*Residual*: {``'Linf'``} | :class:`str`
    name of residual field or type to plot



*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*NPlotFirst*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    iteration at which to start figure



*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits



*XLabel*: {``None``} | :class:`str`
    manual label for x-axis



*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction



*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits



*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction



*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches



*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits



*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches



*NPlotIters*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotIters"



*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes



*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis



*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels



*YLabel*: {``'$L_\\infty$ residual'``} | :class:`str`
    manual label for y-axis



*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits



*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"



*PlotOptions*: {``None``} | :class:`ResidPlotOpts`
    options for main line(s) of plot



*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text



*PlotOptions0*: {``None``} | :class:`ResidPlot0Opts`
    plot options for initial residual



*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label



*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"



*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label



*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits



*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure



*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"



*Caption*: {``None``} | :class:`str`
    subfigure caption



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"


CoeffTable Options
------------------

*CN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CN"



*Coefficients*: {``None``} | :class:`list`\ [:class:`str`]
    list of coefficients to detail in table



*Caption*: {``None``} | :class:`str`
    subfigure caption



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*CY*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CY"



*Components*: {``None``} | :class:`list`\ [:class:`str`]
    list of components for which to report coefficients



*Header*: {``''``} | :class:`str`
    subfigure header



*Iteration*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to sample results



*CLL*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLL"



*CLM*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLM"



*EpsFormat*: {``None``} | :class:`str`
    printf-style text format for sampling error



*MuFormat*: {``None``} | :class:`str`
    printf-style text format for mean value



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*CLN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLN"



*SigmaFormat*: {``None``} | :class:`str`
    printf-sylte text format for standard deviation



*CA*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CA"



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"


Conditions Options
------------------

*SkipVars*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys to leave out of table



*Caption*: {``None``} | :class:`str`
    subfigure caption



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*Header*: {``''``} | :class:`str`
    subfigure header



*SpecialVars*: {``None``} | :class:`list`\ [:class:`str`]
    keys not in run matrix to attempt to calculate



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"


ConditionsTable Options
-----------------------

*SkipVars*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys to leave out of table



*Caption*: {``None``} | :class:`str`
    subfigure caption



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*Header*: {``''``} | :class:`str`
    subfigure header



*SpecialVars*: {``None``} | :class:`list`\ [:class:`str`]
    keys not in run matrix to attempt to calculate



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"


ContourCoeff Options
--------------------

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis



*Caption*: {``None``} | :class:`str`
    subfigure caption



*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values



*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*ContourOptions*: {``None``} | :class:`dict`
    options passed to contour plot function



*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes



*ContourType*: ``'tricontour'`` | {``'tricontourf'``} | ``'tripcolor'``
    contour plotting function/type to use



*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels



*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure



*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits



*XLabel*: {``None``} | :class:`str`
    manual label for x-axis



*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction



*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits



*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction



*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches



*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits



*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis



*AxisEqual*: {``True``} | :class:`bool` | :class:`bool_`
    option to scale x and y axes with common scale



*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches



*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes



*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis



*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels



*YLabel*: {``None``} | :class:`str`
    manual label for y-axis



*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits



*ContourColorMap*: {``'jet'``} | :class:`str`
    name of color map to use w/ contour plots



*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"



*PlotOptions*: {``None``} | :class:`ContourCoeffPlotOpts`
    options for main line(s) of plot



*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text



*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label



*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"



*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label



*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits



*ColorBar*: {``True``} | :class:`bool` | :class:`bool_`
    option to turn on color bar (scale)



*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis



*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure



*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction



*LineType*: {``'plot'``} | ``'triplot'``
    plot function to use to mark data points


FMTable Options
---------------

*CN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CN"



*Coefficients*: {``None``} | :class:`list`\ [:class:`str`]
    list of coefficients to detail in table



*Caption*: {``None``} | :class:`str`
    subfigure caption



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*CY*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CY"



*Components*: {``None``} | :class:`list`\ [:class:`str`]
    list of components for which to report coefficients



*Header*: {``''``} | :class:`str`
    subfigure header



*Iteration*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to sample results



*CLL*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLL"



*CLM*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLM"



*EpsFormat*: {``None``} | :class:`str`
    printf-style text format for sampling error



*MuFormat*: {``None``} | :class:`str`
    printf-style text format for mean value



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*CLN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLN"



*SigmaFormat*: {``None``} | :class:`str`
    printf-sylte text format for standard deviation



*CA*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CA"



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"


Image Options
-------------

*ImageFile*: {``'export.png'``} | :class:`str`
    name of image file to copy from case folder



*Caption*: {``None``} | :class:`str`
    subfigure caption



*Width*: {``0.5``} | :class:`float` | :class:`float32`
    value of option "Width"



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"


Paraview Options
----------------

*Caption*: {``None``} | :class:`str`
    subfigure caption



*Layout*: {``'layout.py'``} | :class:`str`
    name of Python file to execute with Paraview



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"



*ImageFile*: {``'export.png'``} | :class:`str`
    name of image file created by *Layout*



*Format*: {``'png'``} | :class:`str`
    image file format



*Width*: {``0.5``} | :class:`float` | :class:`float32`
    value of option "Width"



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*Command*: {``'pvpython'``} | :class:`str`
    name of Python/Paraview executable to call


PlotCoeff Options
-----------------

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis



*Caption*: {``None``} | :class:`str`
    subfigure caption



*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values



*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes



*NPlotLast*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotLast"



*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels



*MuFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowMu* value



*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*NPlotFirst*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    iteration at which to start figure



*EpsilonPlotOptions*: {``None``} | :class:`PlotCoeffIterEpsilonPlotOpts`
    value of option "EpsilonPlotOptions"



*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits



*EpsilonFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowEpsilon* value



*MuPlotOptions*: {``None``} | :class:`PlotCoeffIterMuPlotOpts`
    plot options for horizontal line showing mean



*XLabel*: {``None``} | :class:`str`
    manual label for x-axis



*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction



*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits



*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction



*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches



*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits



*Coefficient*: {``None``} | :class:`object`
    value of option "Coefficient"



*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches



*ShowDelta*: {``[True, False]``} | :class:`bool` | :class:`bool_`
    option to print value of *Delta*



*DeltaFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowDelta value



*KEpsilon*: {``0.0``} | :class:`float` | :class:`float32`
    multiple of iterative error to plot



*NPlotIters*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotIters"



*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes



*KSigma*: {``None``} | :class:`object`
    value of option "KSigma"



*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis



*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels



*ShowMu*: {``[True, False]``} | :class:`bool` | :class:`bool_`
    option to print value of mean over window



*YLabel*: {``None``} | :class:`str`
    manual label for y-axis



*Component*: {``None``} | :class:`object`
    value of option "Component"



*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits



*SigmaPlotOptions*: {``None``} | :class:`object`
    value of option "SigmaPlotOptions"



*DeltaPlotOptions*: {``None``} | :class:`PlotCoeffIterDeltaPlotOpts`
    plot options for fixed-width above and below mu



*ShowEpsilon*: {``False``} | :class:`bool` | :class:`bool_`
    option to print value of iterative sampling error



*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"



*PlotOptions*: {``None``} | :class:`PlotCoeffIterPlotOpts`
    options for main line(s) of plot



*Delta*: {``0.0``} | :class:`float` | :class:`float32`
    specified interval(s) to plot above and below mean



*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text



*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label



*SigmaFormat*: {``'%.4f'``} | :class:`object`
    printf-style flag for *ShowSigma* value



*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*ShowSigma*: {``[True, False]``} | :class:`bool` | :class:`bool_`
    option to print value of standard deviation



*NAverage*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NAverage"



*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"



*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label



*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits



*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure



*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image



*CaptionComponent*: {``None``} | :class:`str`
    explicit text for component portion of caption



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"


PlotCoeffIter Options
---------------------

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis



*Caption*: {``None``} | :class:`str`
    subfigure caption



*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values



*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes



*NPlotLast*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotLast"



*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels



*MuFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowMu* value



*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*NPlotFirst*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    iteration at which to start figure



*EpsilonPlotOptions*: {``None``} | :class:`PlotCoeffIterEpsilonPlotOpts`
    value of option "EpsilonPlotOptions"



*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits



*EpsilonFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowEpsilon* value



*MuPlotOptions*: {``None``} | :class:`PlotCoeffIterMuPlotOpts`
    plot options for horizontal line showing mean



*XLabel*: {``None``} | :class:`str`
    manual label for x-axis



*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction



*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits



*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction



*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches



*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits



*Coefficient*: {``None``} | :class:`object`
    value of option "Coefficient"



*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches



*ShowDelta*: {``[True, False]``} | :class:`bool` | :class:`bool_`
    option to print value of *Delta*



*DeltaFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowDelta value



*KEpsilon*: {``0.0``} | :class:`float` | :class:`float32`
    multiple of iterative error to plot



*NPlotIters*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotIters"



*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes



*KSigma*: {``None``} | :class:`object`
    value of option "KSigma"



*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis



*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels



*ShowMu*: {``[True, False]``} | :class:`bool` | :class:`bool_`
    option to print value of mean over window



*YLabel*: {``None``} | :class:`str`
    manual label for y-axis



*Component*: {``None``} | :class:`object`
    value of option "Component"



*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits



*SigmaPlotOptions*: {``None``} | :class:`object`
    value of option "SigmaPlotOptions"



*DeltaPlotOptions*: {``None``} | :class:`PlotCoeffIterDeltaPlotOpts`
    plot options for fixed-width above and below mu



*ShowEpsilon*: {``False``} | :class:`bool` | :class:`bool_`
    option to print value of iterative sampling error



*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"



*PlotOptions*: {``None``} | :class:`PlotCoeffIterPlotOpts`
    options for main line(s) of plot



*Delta*: {``0.0``} | :class:`float` | :class:`float32`
    specified interval(s) to plot above and below mean



*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text



*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label



*SigmaFormat*: {``'%.4f'``} | :class:`object`
    printf-style flag for *ShowSigma* value



*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*ShowSigma*: {``[True, False]``} | :class:`bool` | :class:`bool_`
    option to print value of standard deviation



*NAverage*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NAverage"



*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"



*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label



*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits



*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure



*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image



*CaptionComponent*: {``None``} | :class:`str`
    explicit text for component portion of caption



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"


PlotCoeffSweep Options
----------------------

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis



*Caption*: {``None``} | :class:`str`
    subfigure caption



*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values



*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes



*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels



*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure



*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits



*Coefficient*: {``None``} | :class:`object`
    value of option "Coefficient"



*XLabel*: {``None``} | :class:`str`
    manual label for x-axis



*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction



*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits



*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction



*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches



*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits



*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches



*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes



*KSigma*: {``None``} | :class:`object`
    value of option "KSigma"



*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis



*MinMax*: {``False``} | :class:`bool` | :class:`bool_`
    option to plot min/max of value over iterative window



*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels



*YLabel*: {``None``} | :class:`str`
    manual label for y-axis



*Component*: {``None``} | :class:`object`
    value of option "Component"



*Target*: {``None``} | :class:`str`
    name of target databook to co-plot



*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits



*SigmaPlotOptions*: {``None``} | :class:`object`
    value of option "SigmaPlotOptions"



*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"



*PlotOptions*: {``None``} | :class:`PlotCoeffSweepPlotOpts`
    options for main line(s) of plot



*TargetOptions*: {``None``} | :class:`PlotCoeffSweepTargetPlotOpts`
    plot options for optional target



*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text



*MinMaxOptions*: {``None``} | :class:`PlotCoeffSweepMinMaxPlotOpts`
    plot options for *MinMax* plot



*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label



*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"



*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label



*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits



*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure



*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"


PlotContour Options
-------------------

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis



*Caption*: {``None``} | :class:`str`
    subfigure caption



*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values



*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*ContourOptions*: {``None``} | :class:`dict`
    options passed to contour plot function



*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes



*ContourType*: ``'tricontour'`` | {``'tricontourf'``} | ``'tripcolor'``
    contour plotting function/type to use



*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels



*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure



*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits



*XLabel*: {``None``} | :class:`str`
    manual label for x-axis



*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction



*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits



*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction



*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches



*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits



*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis



*AxisEqual*: {``True``} | :class:`bool` | :class:`bool_`
    option to scale x and y axes with common scale



*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches



*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes



*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis



*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels



*YLabel*: {``None``} | :class:`str`
    manual label for y-axis



*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits



*ContourColorMap*: {``'jet'``} | :class:`str`
    name of color map to use w/ contour plots



*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"



*PlotOptions*: {``None``} | :class:`ContourCoeffPlotOpts`
    options for main line(s) of plot



*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text



*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label



*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"



*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label



*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits



*ColorBar*: {``True``} | :class:`bool` | :class:`bool_`
    option to turn on color bar (scale)



*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis



*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure



*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction



*LineType*: {``'plot'``} | ``'triplot'``
    plot function to use to mark data points


PlotContourSweep Options
------------------------

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis



*Caption*: {``None``} | :class:`str`
    subfigure caption



*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values



*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*ContourOptions*: {``None``} | :class:`dict`
    options passed to contour plot function



*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes



*ContourType*: ``'tricontour'`` | {``'tricontourf'``} | ``'tripcolor'``
    contour plotting function/type to use



*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels



*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure



*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits



*XLabel*: {``None``} | :class:`str`
    manual label for x-axis



*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction



*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits



*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction



*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches



*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits



*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis



*AxisEqual*: {``True``} | :class:`bool` | :class:`bool_`
    option to scale x and y axes with common scale



*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches



*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes



*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis



*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels



*YLabel*: {``None``} | :class:`str`
    manual label for y-axis



*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits



*ContourColorMap*: {``'jet'``} | :class:`str`
    name of color map to use w/ contour plots



*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"



*PlotOptions*: {``None``} | :class:`ContourCoeffPlotOpts`
    options for main line(s) of plot



*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text



*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label



*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"



*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label



*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits



*ColorBar*: {``True``} | :class:`bool` | :class:`bool_`
    option to turn on color bar (scale)



*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis



*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure



*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction



*LineType*: {``'plot'``} | ``'triplot'``
    plot function to use to mark data points


PlotL1 Options
--------------

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis



*Caption*: {``None``} | :class:`str`
    subfigure caption



*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values



*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes



*NPlotLast*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotLast"



*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels



*Residual*: {``'L1'``} | :class:`str`
    name of residual field or type to plot



*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*NPlotFirst*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    iteration at which to start figure



*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits



*XLabel*: {``None``} | :class:`str`
    manual label for x-axis



*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction



*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits



*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction



*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches



*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits



*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches



*NPlotIters*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotIters"



*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes



*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis



*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels



*YLabel*: {``None``} | :class:`str`
    manual label for y-axis



*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits



*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"



*PlotOptions*: {``None``} | :class:`ResidPlotOpts`
    options for main line(s) of plot



*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text



*PlotOptions0*: {``None``} | :class:`ResidPlot0Opts`
    plot options for initial residual



*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label



*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"



*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label



*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits



*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure



*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"


PlotL2 Options
--------------

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis



*Caption*: {``None``} | :class:`str`
    subfigure caption



*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values



*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes



*NPlotLast*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotLast"



*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels



*Residual*: {``'L2'``} | :class:`str`
    name of residual field or type to plot



*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*NPlotFirst*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    iteration at which to start figure



*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits



*XLabel*: {``None``} | :class:`str`
    manual label for x-axis



*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction



*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits



*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction



*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches



*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits



*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches



*NPlotIters*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotIters"



*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes



*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis



*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels



*YLabel*: {``None``} | :class:`str`
    manual label for y-axis



*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits



*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"



*PlotOptions*: {``None``} | :class:`ResidPlotOpts`
    options for main line(s) of plot



*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text



*PlotOptions0*: {``None``} | :class:`ResidPlot0Opts`
    plot options for initial residual



*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label



*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"



*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label



*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits



*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure



*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"


PlotLineLoad Options
--------------------

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis



*Caption*: {``None``} | :class:`str`
    subfigure caption



*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values



*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*AutoUpdate*: {``True``} | :class:`bool` | :class:`bool_`
    option to create line loads if not in databook



*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes



*SeamLocation*: ``'bottom'`` | ``'left'`` | ``'right'`` | ``'top'``
    location for optional seam curve plot



*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels



*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*SubplotMargin*: {``0.015``} | :class:`float` | :class:`float32`
    margin between line load and seam curve subplots



*SeamCurve*: ``'smy'`` | ``'smz'``
    name of seam curve, if any, to show w/ line loads



*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure



*Coefficient*: {``None``} | :class:`str`
    coefficient to plot



*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits



*XLabel*: {``None``} | :class:`str`
    manual label for x-axis



*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction



*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits



*SeamOptions*: {``None``} | :class:`PlotLineLoadSeamPlotOpts`
    plot options for optional seam curve



*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction



*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches



*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits



*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches



*XPad*: {``0.03``} | :class:`float` | :class:`float32`
    additional padding from data to xmin and xmax w/i axes



*AdjustRight*: {``0.97``} | :class:`float` | :class:`float32`
    margin from axes to right of figure



*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes



*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis



*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels



*YLabel*: {``None``} | :class:`str`
    manual label for y-axis



*Component*: {``None``} | :class:`str`
    config component tp plot



*Orientation*: ``'horizontal'`` | {``'vertical'``}
    orientation of vehicle in line load plot



*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits



*YPad*: {``0.03``} | :class:`float` | :class:`float32`
    additional padding from data to ymin and ymax w/i axes



*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"



*PlotOptions*: {``None``} | :class:`PlotLineLoadPlotOpts`
    options for main line(s) of plot



*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text



*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label



*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"



*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label



*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits



*AdjustTop*: {``0.97``} | :class:`float` | :class:`float32`
    margin from axes to top of figure



*AdjustBottom*: {``0.1``} | :class:`float` | :class:`float32`
    margin from axes to bottom of figure



*AdjustLeft*: {``0.12``} | :class:`float` | :class:`float32`
    margin from axes to left of figure



*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure



*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"


PlotResid Options
-----------------

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis



*Caption*: {``None``} | :class:`str`
    subfigure caption



*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values



*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes



*NPlotLast*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotLast"



*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels



*Residual*: {``'L2'``} | :class:`str`
    name of residual field or type to plot



*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*NPlotFirst*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    iteration at which to start figure



*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits



*XLabel*: {``None``} | :class:`str`
    manual label for x-axis



*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction



*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits



*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction



*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches



*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits



*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches



*NPlotIters*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotIters"



*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes



*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis



*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels



*YLabel*: {``None``} | :class:`str`
    manual label for y-axis



*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits



*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"



*PlotOptions*: {``None``} | :class:`ResidPlotOpts`
    options for main line(s) of plot



*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text



*PlotOptions0*: {``None``} | :class:`ResidPlot0Opts`
    plot options for initial residual



*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label



*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"



*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label



*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits



*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure



*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"


Summary Options
---------------

*CN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CN"



*Coefficients*: {``None``} | :class:`list`\ [:class:`str`]
    list of coefficients to detail in table



*Caption*: {``None``} | :class:`str`
    subfigure caption



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*CY*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CY"



*Components*: {``None``} | :class:`list`\ [:class:`str`]
    list of components for which to report coefficients



*Header*: {``''``} | :class:`str`
    subfigure header



*Iteration*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to sample results



*CLL*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLL"



*CLM*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLM"



*EpsFormat*: {``None``} | :class:`str`
    printf-style text format for sampling error



*MuFormat*: {``None``} | :class:`str`
    printf-style text format for mean value



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*CLN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLN"



*SigmaFormat*: {``None``} | :class:`str`
    printf-sylte text format for standard deviation



*CA*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CA"



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"


SweepCases Options
------------------

*SkipVars*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys to leave out of table



*Caption*: {``None``} | :class:`str`
    subfigure caption



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*Header*: {``''``} | :class:`str`
    subfigure header



*SpecialVars*: {``None``} | :class:`list`\ [:class:`str`]
    keys not in run matrix to attempt to calculate



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"


SweepCoeff Options
------------------

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis



*Caption*: {``None``} | :class:`str`
    subfigure caption



*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values



*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes



*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels



*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure



*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits



*Coefficient*: {``None``} | :class:`object`
    value of option "Coefficient"



*XLabel*: {``None``} | :class:`str`
    manual label for x-axis



*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction



*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits



*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction



*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches



*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits



*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches



*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes



*KSigma*: {``None``} | :class:`object`
    value of option "KSigma"



*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis



*MinMax*: {``False``} | :class:`bool` | :class:`bool_`
    option to plot min/max of value over iterative window



*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels



*YLabel*: {``None``} | :class:`str`
    manual label for y-axis



*Component*: {``None``} | :class:`object`
    value of option "Component"



*Target*: {``None``} | :class:`str`
    name of target databook to co-plot



*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits



*SigmaPlotOptions*: {``None``} | :class:`object`
    value of option "SigmaPlotOptions"



*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"



*PlotOptions*: {``None``} | :class:`PlotCoeffSweepPlotOpts`
    options for main line(s) of plot



*TargetOptions*: {``None``} | :class:`PlotCoeffSweepTargetPlotOpts`
    plot options for optional target



*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text



*MinMaxOptions*: {``None``} | :class:`PlotCoeffSweepMinMaxPlotOpts`
    plot options for *MinMax* plot



*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label



*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values



*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"



*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label



*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits



*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure



*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"


SweepConditions Options
-----------------------

*SkipVars*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys to leave out of table



*Caption*: {``None``} | :class:`str`
    subfigure caption



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*Header*: {``''``} | :class:`str`
    subfigure header



*SpecialVars*: {``None``} | :class:`list`\ [:class:`str`]
    keys not in run matrix to attempt to calculate



*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"


Tecplot Options
---------------

*ColorMaps*: {``[]``} | :class:`list`\ [:class:`dict`]
    customized Tecplot colormap



*Caption*: {``None``} | :class:`str`
    subfigure caption



*Layout*: {``None``} | :class:`str`
    template Tecplot layout file



*Type*: {``None``} | :class:`str`
    subfigure type or parent



*VarSet*: {``{}``} | :class:`dict`
    variables and their values to define in Tecplot layout



*FieldMap*: {``None``} | :class:`list`\ [:class:`int` | :class:`int32` | :class:`int64`]
    list of zone numbers for Tecplot layout group boundaries



*Width*: {``0.5``} | :class:`float` | :class:`float32`
    value of option "Width"



*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment



*FigWidth*: {``1024``} | :class:`int` | :class:`int32` | :class:`int64`
    width of output image in pixels



*Keys*: {``None``} | :class:`dict`
    dict of Tecplot layout statements to customize



*ContourLevels*: {``None``} | :class:`list`\ [:class:`dict`]
    customized settings for Tecplot contour levels



*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"


Options for all ``Figures``
===========================

*Header*: {``''``} | :class:`str`
    optional header for a figure



*Parent*: {``None``} | :class:`str`
    name of report from which to inherit options



*Subfigures*: {``None``} | :class:`list`\ [:class:`str`]
    value of option "Subfigures"



*Alignment*: ``'left'`` | {``'center'``} | ``'right'``
    horizontal alignment for subfigs in a figure


Options for all ``Sweeps``
==========================

*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis of sweep contour plots



*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis of sweep plots



*Figures*: {``None``} | :class:`list`\ [:class:`str`]
    list of figures in sweep report



*TolCons*: {``None``} | :class:`dict`
    tolerances for run matrix keys to be in same sweep



*CarpetEqCons*: {``None``} | :class:`list`\ [:class:`str`]
    run matrix keys that are constant on carpet subsweep



*GlobalCons*: {``None``} | :class:`list`\ [:class:`str`]
    list of global constraints for sweep



*MinCases*: {``3``} | :class:`int` | :class:`int32` | :class:`int64`
    minimum number of data points in a sweep to include plot



*Indices*: {``None``} | :class:`list`\ [:class:`int` | :class:`int32` | :class:`int64`]
    explicit list of run matrix/databook indices to include



*EqCons*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys that must be constant on a sweep



*IndexTol*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max delta of run matrix/databook index for single sweep



*RunMatrixOnly*: {``False``} | :class:`bool` | :class:`bool_`
    option to restrict databook to current run matrix


