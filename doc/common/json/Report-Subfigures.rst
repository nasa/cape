----------------------------------
Options for ``Subfigures`` section
----------------------------------


Options for default subfigure
=============================

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


Options for ``CoeffTable`` subfigure
====================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
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
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Coefficients*: {``None``} | :class:`list`\ [:class:`str`]
    list of coefficients to detail in table
*Components*: {``None``} | :class:`list`\ [:class:`str`]
    list of components for which to report coefficients
*EpsFormat*: {``None``} | :class:`str`
    printf-style text format for sampling error
*Header*: {``''``} | :class:`str`
    subfigure header
*Iteration*: {``None``} | :class:`int`
    specific iteration at which to sample results
*MuFormat*: {``None``} | :class:`str`
    printf-style text format for mean value
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*SigmaFormat*: {``None``} | :class:`str`
    printf-sylte text format for standard deviation
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"


Options for ``Conditions`` subfigure
====================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Header*: {``''``} | :class:`str`
    subfigure header
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*SkipVars*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys to leave out of table
*SpecialVars*: {``None``} | :class:`list`\ [:class:`str`]
    keys not in run matrix to attempt to calculate
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"


Options for ``Conditions`` subfigure
====================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Header*: {``''``} | :class:`str`
    subfigure header
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*SkipVars*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys to leave out of table
*SpecialVars*: {``None``} | :class:`list`\ [:class:`str`]
    keys not in run matrix to attempt to calculate
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"


Options for ``ContourCoeff`` subfigure
======================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*AxisEqual*: {``True``} | ``False``
    option to scale x and y axes with common scale
*Caption*: {``None``} | :class:`str`
    subfigure caption
*ColorBar*: {``True``} | ``False``
    option to turn on color bar (scale)
*ContourColorMap*: {``'jet'``} | :class:`str`
    name of color map to use w/ contour plots
*ContourOptions*: {``None``} | :class:`dict`
    options passed to contour plot function
*ContourType*: ``'tricontour'`` | {``'tricontourf'``} | ``'tripcolor'``
    contour plotting function/type to use
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Format*: ``'jpeg'`` | ``'jpg'`` | {``'pdf'``} | ``'png'`` | ``'svg'``
    image file format
*LineType*: {``'plot'``} | ``'triplot'``
    plot function to use to mark data points
*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure
*PlotOptions*: {``None``} | :class:`ContourCoeffPlotOpts`
    options for main line(s) of plot
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower left'`` | ``'lower right'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values


Options for ``CoeffTable`` subfigure
====================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
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
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Coefficients*: {``None``} | :class:`list`\ [:class:`str`]
    list of coefficients to detail in table
*Components*: {``None``} | :class:`list`\ [:class:`str`]
    list of components for which to report coefficients
*EpsFormat*: {``None``} | :class:`str`
    printf-style text format for sampling error
*Header*: {``''``} | :class:`str`
    subfigure header
*Iteration*: {``None``} | :class:`int`
    specific iteration at which to sample results
*MuFormat*: {``None``} | :class:`str`
    printf-style text format for mean value
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*SigmaFormat*: {``None``} | :class:`str`
    printf-sylte text format for standard deviation
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"


Options for ``Image`` subfigure
===============================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*ImageFile*: {``'export.png'``} | :class:`str`
    name of image file to copy from case folder
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``0.5``} | :class:`float`
    value of option "Width"


Options for ``Paraview`` subfigure
==================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Command*: {``'pvpython'``} | :class:`str`
    name of Python/Paraview executable to call
*Format*: {``'png'``} | :class:`str`
    image file format
*ImageFile*: {``'export.png'``} | :class:`str`
    name of image file created by *Layout*
*Layout*: {``'layout.py'``} | :class:`str`
    name of Python file to execute with Paraview
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``0.5``} | :class:`float`
    value of option "Width"


Options for ``PlotCoeff`` subfigure
===================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*CaptionComponent*: {``None``} | :class:`str`
    explicit text for component portion of caption
*Coefficient*: {``None``} | :class:`object`
    value of option "Coefficient"
*Component*: {``None``} | :class:`object`
    value of option "Component"
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
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
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Format*: ``'jpeg'`` | ``'jpg'`` | {``'pdf'``} | ``'png'`` | ``'svg'``
    image file format
*KEpsilon*: {``0.0``} | :class:`float`
    multiple of iterative error to plot
*KSigma*: {``None``} | :class:`object`
    value of option "KSigma"
*MuFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowMu* value
*MuPlotOptions*: {``None``} | :class:`PlotCoeffIterMuPlotOpts`
    plot options for horizontal line showing mean
*NAverage*: {``None``} | :class:`int`
    value of option "NAverage"
*NPlotFirst*: {``1``} | :class:`int`
    iteration at which to start figure
*NPlotIters*: {``None``} | :class:`int`
    value of option "NPlotIters"
*NPlotLast*: {``None``} | :class:`int`
    value of option "NPlotLast"
*PlotOptions*: {``None``} | :class:`PlotCoeffIterPlotOpts`
    options for main line(s) of plot
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower left'`` | ``'lower right'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction
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
*SigmaPlotOptions*: {``None``} | :class:`object`
    value of option "SigmaPlotOptions"
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values


Options for ``PlotCoeff`` subfigure
===================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*CaptionComponent*: {``None``} | :class:`str`
    explicit text for component portion of caption
*Coefficient*: {``None``} | :class:`object`
    value of option "Coefficient"
*Component*: {``None``} | :class:`object`
    value of option "Component"
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
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
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Format*: ``'jpeg'`` | ``'jpg'`` | {``'pdf'``} | ``'png'`` | ``'svg'``
    image file format
*KEpsilon*: {``0.0``} | :class:`float`
    multiple of iterative error to plot
*KSigma*: {``None``} | :class:`object`
    value of option "KSigma"
*MuFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowMu* value
*MuPlotOptions*: {``None``} | :class:`PlotCoeffIterMuPlotOpts`
    plot options for horizontal line showing mean
*NAverage*: {``None``} | :class:`int`
    value of option "NAverage"
*NPlotFirst*: {``1``} | :class:`int`
    iteration at which to start figure
*NPlotIters*: {``None``} | :class:`int`
    value of option "NPlotIters"
*NPlotLast*: {``None``} | :class:`int`
    value of option "NPlotLast"
*PlotOptions*: {``None``} | :class:`PlotCoeffIterPlotOpts`
    options for main line(s) of plot
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower left'`` | ``'lower right'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction
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
*SigmaPlotOptions*: {``None``} | :class:`object`
    value of option "SigmaPlotOptions"
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values


Options for ``PlotCoeffSweep`` subfigure
========================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Coefficient*: {``None``} | :class:`object`
    value of option "Coefficient"
*Component*: {``None``} | :class:`object`
    value of option "Component"
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Format*: ``'jpeg'`` | ``'jpg'`` | {``'pdf'``} | ``'png'`` | ``'svg'``
    image file format
*KSigma*: {``None``} | :class:`object`
    value of option "KSigma"
*MinMax*: ``True`` | {``False``}
    option to plot min/max of value over iterative window
*MinMaxOptions*: {``None``} | :class:`PlotCoeffSweepMinMaxPlotOpts`
    plot options for *MinMax* plot
*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure
*PlotOptions*: {``None``} | :class:`PlotCoeffSweepPlotOpts`
    options for main line(s) of plot
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower left'`` | ``'lower right'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction
*SigmaPlotOptions*: {``None``} | :class:`object`
    value of option "SigmaPlotOptions"
*Target*: {``None``} | :class:`str`
    name of target databook to co-plot
*TargetOptions*: {``None``} | :class:`PlotCoeffSweepTargetPlotOpts`
    plot options for optional target
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values


Options for ``ContourCoeff`` subfigure
======================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*AxisEqual*: {``True``} | ``False``
    option to scale x and y axes with common scale
*Caption*: {``None``} | :class:`str`
    subfigure caption
*ColorBar*: {``True``} | ``False``
    option to turn on color bar (scale)
*ContourColorMap*: {``'jet'``} | :class:`str`
    name of color map to use w/ contour plots
*ContourOptions*: {``None``} | :class:`dict`
    options passed to contour plot function
*ContourType*: ``'tricontour'`` | {``'tricontourf'``} | ``'tripcolor'``
    contour plotting function/type to use
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Format*: ``'jpeg'`` | ``'jpg'`` | {``'pdf'``} | ``'png'`` | ``'svg'``
    image file format
*LineType*: {``'plot'``} | ``'triplot'``
    plot function to use to mark data points
*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure
*PlotOptions*: {``None``} | :class:`ContourCoeffPlotOpts`
    options for main line(s) of plot
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower left'`` | ``'lower right'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values


Options for ``ContourCoeff`` subfigure
======================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*AxisEqual*: {``True``} | ``False``
    option to scale x and y axes with common scale
*Caption*: {``None``} | :class:`str`
    subfigure caption
*ColorBar*: {``True``} | ``False``
    option to turn on color bar (scale)
*ContourColorMap*: {``'jet'``} | :class:`str`
    name of color map to use w/ contour plots
*ContourOptions*: {``None``} | :class:`dict`
    options passed to contour plot function
*ContourType*: ``'tricontour'`` | {``'tricontourf'``} | ``'tripcolor'``
    contour plotting function/type to use
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Format*: ``'jpeg'`` | ``'jpg'`` | {``'pdf'``} | ``'png'`` | ``'svg'``
    image file format
*LineType*: {``'plot'``} | ``'triplot'``
    plot function to use to mark data points
*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure
*PlotOptions*: {``None``} | :class:`ContourCoeffPlotOpts`
    options for main line(s) of plot
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower left'`` | ``'lower right'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values


Options for ``PlotL1`` subfigure
================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Format*: ``'jpeg'`` | ``'jpg'`` | {``'pdf'``} | ``'png'`` | ``'svg'``
    image file format
*NPlotFirst*: {``1``} | :class:`int`
    iteration at which to start figure
*NPlotIters*: {``None``} | :class:`int`
    value of option "NPlotIters"
*NPlotLast*: {``None``} | :class:`int`
    value of option "NPlotLast"
*PlotOptions*: {``None``} | :class:`ResidPlotOpts`
    options for main line(s) of plot
*PlotOptions0*: {``None``} | :class:`ResidPlot0Opts`
    plot options for initial residual
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*Residual*: {``'L1'``} | :class:`str`
    name of residual field or type to plot
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower left'`` | ``'lower right'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values


Options for ``PlotL2`` subfigure
================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Format*: ``'jpeg'`` | ``'jpg'`` | {``'pdf'``} | ``'png'`` | ``'svg'``
    image file format
*NPlotFirst*: {``1``} | :class:`int`
    iteration at which to start figure
*NPlotIters*: {``None``} | :class:`int`
    value of option "NPlotIters"
*NPlotLast*: {``None``} | :class:`int`
    value of option "NPlotLast"
*PlotOptions*: {``None``} | :class:`ResidPlotOpts`
    options for main line(s) of plot
*PlotOptions0*: {``None``} | :class:`ResidPlot0Opts`
    plot options for initial residual
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*Residual*: {``'L2'``} | :class:`str`
    name of residual field or type to plot
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower left'`` | ``'lower right'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values


Options for ``PlotLineLoad`` subfigure
======================================

*AdjustBottom*: {``0.1``} | :class:`float`
    margin from axes to bottom of figure
*AdjustLeft*: {``0.12``} | :class:`float`
    margin from axes to left of figure
*AdjustRight*: {``0.97``} | :class:`float`
    margin from axes to right of figure
*AdjustTop*: {``0.97``} | :class:`float`
    margin from axes to top of figure
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*AutoUpdate*: {``True``} | ``False``
    option to create line loads if not in databook
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Coefficient*: {``None``} | :class:`str`
    coefficient to plot
*Component*: {``None``} | :class:`str`
    config component tp plot
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Format*: ``'jpeg'`` | ``'jpg'`` | {``'pdf'``} | ``'png'`` | ``'svg'``
    image file format
*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure
*Orientation*: ``'horizontal'`` | {``'vertical'``}
    orientation of vehicle in line load plot
*PlotOptions*: {``None``} | :class:`PlotLineLoadPlotOpts`
    options for main line(s) of plot
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower left'`` | ``'lower right'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction
*SeamCurve*: ``'smy'`` | ``'smz'``
    name of seam curve, if any, to show w/ line loads
*SeamLocation*: ``'bottom'`` | ``'left'`` | ``'right'`` | ``'top'``
    location for optional seam curve plot
*SeamOptions*: {``None``} | :class:`PlotLineLoadSeamPlotOpts`
    plot options for optional seam curve
*SubplotMargin*: {``0.015``} | :class:`float`
    margin between line load and seam curve subplots
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*XPad*: {``0.03``} | :class:`float`
    additional padding from data to xmin and xmax w/i axes
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*YPad*: {``0.03``} | :class:`float`
    additional padding from data to ymin and ymax w/i axes
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values


Options for ``PlotL2`` subfigure
================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Format*: ``'jpeg'`` | ``'jpg'`` | {``'pdf'``} | ``'png'`` | ``'svg'``
    image file format
*NPlotFirst*: {``1``} | :class:`int`
    iteration at which to start figure
*NPlotIters*: {``None``} | :class:`int`
    value of option "NPlotIters"
*NPlotLast*: {``None``} | :class:`int`
    value of option "NPlotLast"
*PlotOptions*: {``None``} | :class:`ResidPlotOpts`
    options for main line(s) of plot
*PlotOptions0*: {``None``} | :class:`ResidPlot0Opts`
    plot options for initial residual
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*Residual*: {``'L2'``} | :class:`str`
    name of residual field or type to plot
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower left'`` | ``'lower right'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values


Options for ``CoeffTable`` subfigure
====================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
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
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Coefficients*: {``None``} | :class:`list`\ [:class:`str`]
    list of coefficients to detail in table
*Components*: {``None``} | :class:`list`\ [:class:`str`]
    list of components for which to report coefficients
*EpsFormat*: {``None``} | :class:`str`
    printf-style text format for sampling error
*Header*: {``''``} | :class:`str`
    subfigure header
*Iteration*: {``None``} | :class:`int`
    specific iteration at which to sample results
*MuFormat*: {``None``} | :class:`str`
    printf-style text format for mean value
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*SigmaFormat*: {``None``} | :class:`str`
    printf-sylte text format for standard deviation
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"


Options for ``SweepCases`` subfigure
====================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Header*: {``''``} | :class:`str`
    subfigure header
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*SkipVars*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys to leave out of table
*SpecialVars*: {``None``} | :class:`list`\ [:class:`str`]
    keys not in run matrix to attempt to calculate
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"


Options for ``PlotCoeffSweep`` subfigure
========================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Coefficient*: {``None``} | :class:`object`
    value of option "Coefficient"
*Component*: {``None``} | :class:`object`
    value of option "Component"
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Format*: ``'jpeg'`` | ``'jpg'`` | {``'pdf'``} | ``'png'`` | ``'svg'``
    image file format
*KSigma*: {``None``} | :class:`object`
    value of option "KSigma"
*MinMax*: ``True`` | {``False``}
    option to plot min/max of value over iterative window
*MinMaxOptions*: {``None``} | :class:`PlotCoeffSweepMinMaxPlotOpts`
    plot options for *MinMax* plot
*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure
*PlotOptions*: {``None``} | :class:`PlotCoeffSweepPlotOpts`
    options for main line(s) of plot
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower left'`` | ``'lower right'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction
*SigmaPlotOptions*: {``None``} | :class:`object`
    value of option "SigmaPlotOptions"
*Target*: {``None``} | :class:`str`
    name of target databook to co-plot
*TargetOptions*: {``None``} | :class:`PlotCoeffSweepTargetPlotOpts`
    plot options for optional target
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values


Options for ``SweepCases`` subfigure
====================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Header*: {``''``} | :class:`str`
    subfigure header
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*SkipVars*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys to leave out of table
*SpecialVars*: {``None``} | :class:`list`\ [:class:`str`]
    keys not in run matrix to attempt to calculate
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Width*: {``None``} | :class:`float`
    value of option "Width"


Options for ``Tecplot`` subfigure
=================================

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
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
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*VarSet*: {``{}``} | :class:`dict`
    variables and their values to define in Tecplot layout
*Width*: {``0.5``} | :class:`float`
    value of option "Width"


