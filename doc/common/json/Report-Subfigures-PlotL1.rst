--------------------------------
Options for ``PlotL1`` subfigure
--------------------------------

**Option aliases:**

* *LineOptions0* -> *PlotOptions0*
* *nPlotFirst* -> *NPlotFirst*
* *nFirst* -> *NPlotFirst*
* *nPlotIters* -> *NPlotIters*
* *nPlotLast* -> *NPlotLast*
* *FigHeight* -> *FigureHeight*
* *FigWidth* -> *FigureWidth*
* *GridStyle* -> *GridPlotOptions*
* *LineOptions* -> *PlotOptions*
* *MinorGridStyle* -> *MinorGridPlotOptions*
* *RestrictionLocation* -> *RestrictionLoc*
* *RestrictionX* -> *RestrictionXPosition*
* *RestrictionY* -> *RestrictionYPosition*
* *dpi* -> *DPI*
* *Parent* -> *Type*
* *parent* -> *Type*
* *pos* -> *Position*
* *type* -> *Type*
* *width* -> *Width*

**Recognized options:**

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
*XTickLabels*: {``None``} | ``True`` | ``False`` | :class:`float` | :class:`str`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | ``True`` | ``False`` | :class:`float`
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
*YTickLabels*: {``None``} | ``True`` | ``False`` | :class:`float` | :class:`str`
    option to turn off x-axis tick labels or set values
*YTicks*: {``None``} | ``True`` | ``False`` | :class:`float`
    option to turn off y-axis ticks or set values

