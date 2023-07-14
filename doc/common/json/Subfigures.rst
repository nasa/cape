--------------------
"Subfigures" section
--------------------


--------------------------
Unique options for default
--------------------------

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Width*: {``None``} | :class:`float`
    value of option "Width"
*Type*: {``None``} | :class:`str`
    subfigure type or parent


----------------------------------------
Unique options for *Type*\ ="CoeffTable"
----------------------------------------

*CLL*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLL"
*CN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CN"
*CLM*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLM"
*SigmaFormat*: {``None``} | :class:`str`
    printf-sylte text format for standard deviation
*CLN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLN"
*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*EpsFormat*: {``None``} | :class:`str`
    printf-style text format for sampling error
*Iteration*: {``None``} | :class:`int`
    specific iteration at which to sample results
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Coefficients*: {``None``} | :class:`list`\ [:class:`str`]
    list of coefficients to detail in table
*Width*: {``None``} | :class:`float`
    value of option "Width"
*CA*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CA"
*CY*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CY"
*Components*: {``None``} | :class:`list`\ [:class:`str`]
    list of components for which to report coefficients
*Header*: {``''``} | :class:`str`
    subfigure header
*MuFormat*: {``None``} | :class:`str`
    printf-style text format for mean value
*Type*: {``None``} | :class:`str`
    subfigure type or parent


----------------------------------------
Unique options for *Type*\ ="Conditions"
----------------------------------------

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Width*: {``None``} | :class:`float`
    value of option "Width"
*SkipVars*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys to leave out of table
*Header*: {``''``} | :class:`str`
    subfigure header
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*SpecialVars*: {``None``} | :class:`list`\ [:class:`str`]
    keys not in run matrix to attempt to calculate


----------------------------------------
Unique options for *Type*\ ="Conditions"
----------------------------------------

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Width*: {``None``} | :class:`float`
    value of option "Width"
*SkipVars*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys to leave out of table
*Header*: {``''``} | :class:`str`
    subfigure header
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*SpecialVars*: {``None``} | :class:`list`\ [:class:`str`]
    keys not in run matrix to attempt to calculate


------------------------------------------
Unique options for *Type*\ ="ContourCoeff"
------------------------------------------

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*ContourColorMap*: {``'jet'``} | :class:`str`
    name of color map to use w/ contour plots
*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*PlotOptions*: {``None``} | :class:`ContourCoeffPlotOpts`
    options for main line(s) of plot
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Caption*: {``None``} | :class:`str`
    subfigure caption
*ContourType*: ``'tricontour'`` | {``'tricontourf'``} | ``'tripcolor'``
    contour plotting function/type to use
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis
*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format
*LineType*: {``'plot'``} | ``'triplot'``
    plot function to use to mark data points
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*AxisEqual*: {``True``} | ``False``
    option to scale x and y axes with common scale
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*ColorBar*: {``True``} | ``False``
    option to turn on color bar (scale)
*ContourOptions*: {``None``} | :class:`dict`
    options passed to contour plot function
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure
*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction


----------------------------------------
Unique options for *Type*\ ="CoeffTable"
----------------------------------------

*CLL*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLL"
*CN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CN"
*CLM*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLM"
*SigmaFormat*: {``None``} | :class:`str`
    printf-sylte text format for standard deviation
*CLN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLN"
*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*EpsFormat*: {``None``} | :class:`str`
    printf-style text format for sampling error
*Iteration*: {``None``} | :class:`int`
    specific iteration at which to sample results
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Coefficients*: {``None``} | :class:`list`\ [:class:`str`]
    list of coefficients to detail in table
*Width*: {``None``} | :class:`float`
    value of option "Width"
*CA*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CA"
*CY*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CY"
*Components*: {``None``} | :class:`list`\ [:class:`str`]
    list of components for which to report coefficients
*Header*: {``''``} | :class:`str`
    subfigure header
*MuFormat*: {``None``} | :class:`str`
    printf-style text format for mean value
*Type*: {``None``} | :class:`str`
    subfigure type or parent


-----------------------------------
Unique options for *Type*\ ="Image"
-----------------------------------

*ImageFile*: {``'export.png'``} | :class:`str`
    name of image file to copy from case folder
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Width*: {``0.5``} | :class:`float`
    value of option "Width"


--------------------------------------
Unique options for *Type*\ ="Paraview"
--------------------------------------

*Command*: {``'pvpython'``} | :class:`str`
    name of Python/Paraview executable to call
*Layout*: {``'layout.py'``} | :class:`str`
    name of Python file to execute with Paraview
*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Format*: {``'png'``} | :class:`str`
    image file format
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Width*: {``0.5``} | :class:`float`
    value of option "Width"
*ImageFile*: {``'export.png'``} | :class:`str`
    name of image file created by *Layout*
*Type*: {``None``} | :class:`str`
    subfigure type or parent


---------------------------------------
Unique options for *Type*\ ="PlotCoeff"
---------------------------------------

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*PlotOptions*: {``None``} | :class:`PlotCoeffIterPlotOpts`
    options for main line(s) of plot
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Caption*: {``None``} | :class:`str`
    subfigure caption
*KSigma*: {``None``} | :class:`object`
    value of option "KSigma"
*KEpsilon*: {``0.0``} | :class:`float`
    multiple of iterative error to plot
*NAverage*: {``None``} | :class:`int`
    value of option "NAverage"
*MuFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowMu* value
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*DeltaFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowDelta value
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format
*ShowSigma*: {``True``} | ``False``
    option to print value of standard deviation
*Delta*: {``0.0``} | :class:`float`
    specified interval(s) to plot above and below mean
*DeltaPlotOptions*: {``None``} | :class:`PlotCoeffIterDeltaPlotOpts`
    plot options for fixed-width above and below mu
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*NPlotLast*: {``None``} | :class:`int`
    value of option "NPlotLast"
*MuPlotOptions*: {``None``} | :class:`PlotCoeffIterMuPlotOpts`
    plot options for horizontal line showing mean
*EpsilonPlotOptions*: {``None``} | :class:`PlotCoeffIterEpsilonPlotOpts`
    value of option "EpsilonPlotOptions"
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*CaptionComponent*: {``None``} | :class:`str`
    explicit text for component portion of caption
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*SigmaPlotOptions*: {``None``} | :class:`object`
    value of option "SigmaPlotOptions"
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*Component*: {``None``} | :class:`object`
    value of option "Component"
*EpsilonFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowEpsilon* value
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*Coefficient*: {``None``} | :class:`object`
    value of option "Coefficient"
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*ShowEpsilon*: ``True`` | {``False``}
    option to print value of iterative sampling error
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*NPlotIters*: {``None``} | :class:`int`
    value of option "NPlotIters"
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*SigmaFormat*: {``'%.4f'``} | :class:`object`
    printf-style flag for *ShowSigma* value
*NPlotFirst*: {``1``} | :class:`int`
    iteration at which to start figure
*ShowDelta*: {``True``} | ``False``
    option to print value of *Delta*
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*ShowMu*: {``True``} | ``False``
    option to print value of mean over window
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction


---------------------------------------
Unique options for *Type*\ ="PlotCoeff"
---------------------------------------

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*PlotOptions*: {``None``} | :class:`PlotCoeffIterPlotOpts`
    options for main line(s) of plot
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Caption*: {``None``} | :class:`str`
    subfigure caption
*KSigma*: {``None``} | :class:`object`
    value of option "KSigma"
*KEpsilon*: {``0.0``} | :class:`float`
    multiple of iterative error to plot
*NAverage*: {``None``} | :class:`int`
    value of option "NAverage"
*MuFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowMu* value
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*DeltaFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowDelta value
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format
*ShowSigma*: {``True``} | ``False``
    option to print value of standard deviation
*Delta*: {``0.0``} | :class:`float`
    specified interval(s) to plot above and below mean
*DeltaPlotOptions*: {``None``} | :class:`PlotCoeffIterDeltaPlotOpts`
    plot options for fixed-width above and below mu
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*NPlotLast*: {``None``} | :class:`int`
    value of option "NPlotLast"
*MuPlotOptions*: {``None``} | :class:`PlotCoeffIterMuPlotOpts`
    plot options for horizontal line showing mean
*EpsilonPlotOptions*: {``None``} | :class:`PlotCoeffIterEpsilonPlotOpts`
    value of option "EpsilonPlotOptions"
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*CaptionComponent*: {``None``} | :class:`str`
    explicit text for component portion of caption
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*SigmaPlotOptions*: {``None``} | :class:`object`
    value of option "SigmaPlotOptions"
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*Component*: {``None``} | :class:`object`
    value of option "Component"
*EpsilonFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowEpsilon* value
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*Coefficient*: {``None``} | :class:`object`
    value of option "Coefficient"
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*ShowEpsilon*: ``True`` | {``False``}
    option to print value of iterative sampling error
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*NPlotIters*: {``None``} | :class:`int`
    value of option "NPlotIters"
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*SigmaFormat*: {``'%.4f'``} | :class:`object`
    printf-style flag for *ShowSigma* value
*NPlotFirst*: {``1``} | :class:`int`
    iteration at which to start figure
*ShowDelta*: {``True``} | ``False``
    option to print value of *Delta*
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*ShowMu*: {``True``} | ``False``
    option to print value of mean over window
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction


--------------------------------------------
Unique options for *Type*\ ="PlotCoeffSweep"
--------------------------------------------

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*PlotOptions*: {``None``} | :class:`PlotCoeffSweepPlotOpts`
    options for main line(s) of plot
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Caption*: {``None``} | :class:`str`
    subfigure caption
*KSigma*: {``None``} | :class:`object`
    value of option "KSigma"
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*Target*: {``None``} | :class:`str`
    name of target databook to co-plot
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*SigmaPlotOptions*: {``None``} | :class:`object`
    value of option "SigmaPlotOptions"
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*Component*: {``None``} | :class:`object`
    value of option "Component"
*MinMaxOptions*: {``None``} | :class:`PlotCoeffSweepMinMaxPlotOpts`
    plot options for *MinMax* plot
*TargetOptions*: {``None``} | :class:`PlotCoeffSweepTargetPlotOpts`
    plot options for optional target
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*Coefficient*: {``None``} | :class:`object`
    value of option "Coefficient"
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*MinMax*: ``True`` | {``False``}
    option to plot min/max of value over iterative window
*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction


------------------------------------------
Unique options for *Type*\ ="ContourCoeff"
------------------------------------------

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*ContourColorMap*: {``'jet'``} | :class:`str`
    name of color map to use w/ contour plots
*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*PlotOptions*: {``None``} | :class:`ContourCoeffPlotOpts`
    options for main line(s) of plot
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Caption*: {``None``} | :class:`str`
    subfigure caption
*ContourType*: ``'tricontour'`` | {``'tricontourf'``} | ``'tripcolor'``
    contour plotting function/type to use
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis
*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format
*LineType*: {``'plot'``} | ``'triplot'``
    plot function to use to mark data points
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*AxisEqual*: {``True``} | ``False``
    option to scale x and y axes with common scale
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*ColorBar*: {``True``} | ``False``
    option to turn on color bar (scale)
*ContourOptions*: {``None``} | :class:`dict`
    options passed to contour plot function
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure
*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction


------------------------------------------
Unique options for *Type*\ ="ContourCoeff"
------------------------------------------

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*ContourColorMap*: {``'jet'``} | :class:`str`
    name of color map to use w/ contour plots
*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*PlotOptions*: {``None``} | :class:`ContourCoeffPlotOpts`
    options for main line(s) of plot
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Caption*: {``None``} | :class:`str`
    subfigure caption
*ContourType*: ``'tricontour'`` | {``'tricontourf'``} | ``'tripcolor'``
    contour plotting function/type to use
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis
*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format
*LineType*: {``'plot'``} | ``'triplot'``
    plot function to use to mark data points
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*AxisEqual*: {``True``} | ``False``
    option to scale x and y axes with common scale
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*ColorBar*: {``True``} | ``False``
    option to turn on color bar (scale)
*ContourOptions*: {``None``} | :class:`dict`
    options passed to contour plot function
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure
*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction


------------------------------------
Unique options for *Type*\ ="PlotL1"
------------------------------------

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*PlotOptions*: {``None``} | :class:`ResidPlotOpts`
    options for main line(s) of plot
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Caption*: {``None``} | :class:`str`
    subfigure caption
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*NPlotLast*: {``None``} | :class:`int`
    value of option "NPlotLast"
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*Width*: {``None``} | :class:`float`
    value of option "Width"
*Residual*: {``'L1'``} | :class:`str`
    name of residual field or type to plot
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*NPlotIters*: {``None``} | :class:`int`
    value of option "NPlotIters"
*PlotOptions0*: {``None``} | :class:`ResidPlot0Opts`
    plot options for initial residual
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*NPlotFirst*: {``1``} | :class:`int`
    iteration at which to start figure
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction


------------------------------------
Unique options for *Type*\ ="PlotL2"
------------------------------------

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*PlotOptions*: {``None``} | :class:`ResidPlotOpts`
    options for main line(s) of plot
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Caption*: {``None``} | :class:`str`
    subfigure caption
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*NPlotLast*: {``None``} | :class:`int`
    value of option "NPlotLast"
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*Width*: {``None``} | :class:`float`
    value of option "Width"
*Residual*: {``'L2'``} | :class:`str`
    name of residual field or type to plot
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*NPlotIters*: {``None``} | :class:`int`
    value of option "NPlotIters"
*PlotOptions0*: {``None``} | :class:`ResidPlot0Opts`
    plot options for initial residual
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*NPlotFirst*: {``1``} | :class:`int`
    iteration at which to start figure
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction


------------------------------------------
Unique options for *Type*\ ="PlotLineLoad"
------------------------------------------

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*AutoUpdate*: {``True``} | ``False``
    option to create line loads if not in databook
*SeamCurve*: ``'smy'`` | ``'smz'``
    name of seam curve, if any, to show w/ line loads
*XPad*: {``0.03``} | :class:`float`
    additional padding from data to xmin and xmax w/i axes
*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*PlotOptions*: {``None``} | :class:`PlotLineLoadPlotOpts`
    options for main line(s) of plot
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Caption*: {``None``} | :class:`str`
    subfigure caption
*SeamLocation*: ``'bottom'`` | ``'left'`` | ``'right'`` | ``'top'``
    location for optional seam curve plot
*AdjustTop*: {``0.97``} | :class:`float`
    margin from axes to top of figure
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*AdjustRight*: {``0.97``} | :class:`float`
    margin from axes to right of figure
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format
*AdjustLeft*: {``0.12``} | :class:`float`
    margin from axes to left of figure
*AdjustBottom*: {``0.1``} | :class:`float`
    margin from axes to bottom of figure
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*SeamOptions*: {``None``} | :class:`PlotLineLoadSeamPlotOpts`
    plot options for optional seam curve
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*YPad*: {``0.03``} | :class:`float`
    additional padding from data to ymin and ymax w/i axes
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*Component*: {``None``} | :class:`str`
    config component tp plot
*Orientation*: ``'horizontal'`` | {``'vertical'``}
    orientation of vehicle in line load plot
*Coefficient*: {``None``} | :class:`str`
    coefficient to plot
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*SubplotMargin*: {``0.015``} | :class:`float`
    margin between line load and seam curve subplots
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction


------------------------------------
Unique options for *Type*\ ="PlotL2"
------------------------------------

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*PlotOptions*: {``None``} | :class:`ResidPlotOpts`
    options for main line(s) of plot
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Caption*: {``None``} | :class:`str`
    subfigure caption
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*NPlotLast*: {``None``} | :class:`int`
    value of option "NPlotLast"
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*Width*: {``None``} | :class:`float`
    value of option "Width"
*Residual*: {``'L2'``} | :class:`str`
    name of residual field or type to plot
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*NPlotIters*: {``None``} | :class:`int`
    value of option "NPlotIters"
*PlotOptions0*: {``None``} | :class:`ResidPlot0Opts`
    plot options for initial residual
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*NPlotFirst*: {``1``} | :class:`int`
    iteration at which to start figure
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction


----------------------------------------
Unique options for *Type*\ ="CoeffTable"
----------------------------------------

*CLL*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLL"
*CN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CN"
*CLM*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLM"
*SigmaFormat*: {``None``} | :class:`str`
    printf-sylte text format for standard deviation
*CLN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLN"
*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*EpsFormat*: {``None``} | :class:`str`
    printf-style text format for sampling error
*Iteration*: {``None``} | :class:`int`
    specific iteration at which to sample results
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Coefficients*: {``None``} | :class:`list`\ [:class:`str`]
    list of coefficients to detail in table
*Width*: {``None``} | :class:`float`
    value of option "Width"
*CA*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CA"
*CY*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CY"
*Components*: {``None``} | :class:`list`\ [:class:`str`]
    list of components for which to report coefficients
*Header*: {``''``} | :class:`str`
    subfigure header
*MuFormat*: {``None``} | :class:`str`
    printf-style text format for mean value
*Type*: {``None``} | :class:`str`
    subfigure type or parent


----------------------------------------
Unique options for *Type*\ ="SweepCases"
----------------------------------------

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Width*: {``None``} | :class:`float`
    value of option "Width"
*SkipVars*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys to leave out of table
*Header*: {``''``} | :class:`str`
    subfigure header
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*SpecialVars*: {``None``} | :class:`list`\ [:class:`str`]
    keys not in run matrix to attempt to calculate


--------------------------------------------
Unique options for *Type*\ ="PlotCoeffSweep"
--------------------------------------------

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction
*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*PlotOptions*: {``None``} | :class:`PlotCoeffSweepPlotOpts`
    options for main line(s) of plot
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*FigureWidth*: {``6``} | :class:`float`
    width of subfigure graphics in inches
*Caption*: {``None``} | :class:`str`
    subfigure caption
*KSigma*: {``None``} | :class:`object`
    value of option "KSigma"
*TickLabels*: {``None``} | ``True`` | ``False``
    common value(s) for ticks of both axes
*XLabel*: {``None``} | :class:`str`
    manual label for x-axis
*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels
*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values
*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits
*YMax*: {``None``} | :class:`float`
    explicit upper limit for y-axis limits
*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values
*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format
*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure
*XMax*: {``None``} | :class:`float`
    explicit upper limit for x-axis limits
*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label
*Target*: {``None``} | :class:`str`
    name of target databook to co-plot
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis
*YMin*: {``None``} | :class:`float`
    explicit lower limit for y-axis limits
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label
*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis
*SigmaPlotOptions*: {``None``} | :class:`object`
    value of option "SigmaPlotOptions"
*Width*: {``None``} | :class:`float`
    value of option "Width"
*XMin*: {``None``} | :class:`float`
    explicit lower limit for x-axis limits
*Component*: {``None``} | :class:`object`
    value of option "Component"
*MinMaxOptions*: {``None``} | :class:`PlotCoeffSweepMinMaxPlotOpts`
    plot options for *MinMax* plot
*TargetOptions*: {``None``} | :class:`PlotCoeffSweepTargetPlotOpts`
    plot options for optional target
*YLabel*: {``None``} | :class:`str`
    manual label for y-axis
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*Coefficient*: {``None``} | :class:`object`
    value of option "Coefficient"
*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits
*Ticks*: {``None``} | ``True`` | ``False``
    value of option "Ticks"
*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes
*MinMax*: ``True`` | {``False``}
    option to plot min/max of value over iterative window
*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure
*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text
*RestrictionXPosition*: {``None``} | :class:`float`
    explicit x-coord of restriction
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values
*RestrictionYPosition*: {``None``} | :class:`float`
    explicit y-coord of restriction


----------------------------------------
Unique options for *Type*\ ="SweepCases"
----------------------------------------

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Width*: {``None``} | :class:`float`
    value of option "Width"
*SkipVars*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys to leave out of table
*Header*: {``''``} | :class:`str`
    subfigure header
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*SpecialVars*: {``None``} | :class:`list`\ [:class:`str`]
    keys not in run matrix to attempt to calculate


-------------------------------------
Unique options for *Type*\ ="Tecplot"
-------------------------------------

*FieldMap*: {``None``} | :class:`list`\ [:class:`int`]
    list of zone numbers for Tecplot layout group boundaries
*FigWidth*: {``1024``} | :class:`int`
    width of output image in pixels
*Layout*: {``None``} | :class:`str`
    template Tecplot layout file
*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*Width*: {``0.5``} | :class:`float`
    value of option "Width"
*ContourLevels*: {``None``} | :class:`list`\ [:class:`dict`]
    customized settings for Tecplot contour levels
*Keys*: {``None``} | :class:`dict`
    dict of Tecplot layout statements to customize
*VarSet*: {``{}``} | :class:`dict`
    variables and their values to define in Tecplot layout
*ColorMaps*: {``[]``} | :class:`list`\ [:class:`dict`]
    customized Tecplot colormap
*Type*: {``None``} | :class:`str`
    subfigure type or parent


