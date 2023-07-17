--------------------------------------
Options for ``ContourCoeff`` subfigure
--------------------------------------

**Option aliases:**

* *ColorMap* -> *ContourColorMap*
* *PlotType* -> *LineType*
* *XAxis* -> *XCol*
* *YAxis* -> *YCol*
* *xcol* -> *XCol*
* *xk* -> *XCol*
* *ycol* -> *YCol*
* *yk* -> *YCol*
* *FigHeight* -> *FigureHeight*
* *FigWidth* -> *FigureWidth*
* *GridStyle* -> *GridPlotOptions*
* *LineOptions* -> *PlotOptions*
* *MinorGridStyle* -> *MinorGridPlotOptions*
* *RestrictionLocation* -> *RestrictionLoc*
* *RestrictionX* -> *RestrictionXPosition*
* *RestrictionY* -> *RestrictionYPosition*
* *dpi* -> *DPI*
* *nPlotFirst* -> *NPlotFirst*
* *nFirst* -> *NPlotFirst*
* *Parent* -> *Type*
* *parent* -> *Type*
* *pos* -> *Position*
* *type* -> *Type*
* *width* -> *Width*

**Recognized options:**

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
*XTickLabels*: {``None``} | ``True`` | ``False`` | :class:`float` | :class:`str`
    option to turn off x-axis tick labels or set values
*XTicks*: {``None``} | ``True`` | ``False`` | :class:`float`
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
*YTickLabels*: {``None``} | ``True`` | ``False`` | :class:`float` | :class:`str`
    option to turn off x-axis tick labels or set values
*YTicks*: {``None``} | ``True`` | ``False`` | :class:`float`
    option to turn off y-axis ticks or set values

