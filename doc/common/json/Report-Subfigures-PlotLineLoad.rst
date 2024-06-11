--------------------------------------
Options for ``PlotLineLoad`` subfigure
--------------------------------------

**Option aliases:**

* *SeamCurves* -> *SeamCurve*
* *SeamLocations* -> *SeamLocation*
* *SeamCurveOptions* -> *SeamOptions*
* *Targets* -> *Target*
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
    config component to plot
*DPI*: {``150``} | :class:`int`
    dots per inch if saving as rasterized image
*FigureHeight*: {``4.5``} | :class:`float`
    height of subfigure graphics in inches
*FigureWidth*: {``6.0``} | :class:`float`
    width of subfigure graphics in inches
*Format*: ``'jpeg'`` | ``'jpg'`` | {``'pdf'``} | ``'png'`` | ``'svg'``
    image file format
*Grid*: {``None``} | ``True`` | ``False``
    whether to show axes grid in background
*GridPlotOptions*: {``{}``} | :class:`dict`
    plot options for major grid, if shown
*MinorGrid*: {``None``} | ``True`` | ``False``
    whether to show axes minor grid
*MinorGridPlotOptions*: {``{}``} | :class:`dict`
    plot options for minor grid, if shown
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
*Width*: {``0.33``} | :class:`float`
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
*YPad*: {``0.03``} | :class:`float`
    additional padding from data to ymin and ymax w/i axes
*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels
*YTickLabels*: {``None``} | ``True`` | ``False`` | :class:`float` | :class:`str`
    option to turn off x-axis tick labels or set values
*YTicks*: {``None``} | ``True`` | ``False`` | :class:`float`
    option to turn off y-axis ticks or set values

