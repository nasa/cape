
.. _pyfun-json-report:

**********************
Report Section Options
**********************
The options below are the available options in the Report Section of the ``pyfun.json`` control file

..
    start-Report-subfigures

*Subfigures*: {``None``} | :class:`dict`
    collection of subfigure definitions

..
    end-Report-subfigures

..
    start-Report-sweeps

*Sweeps*: {``None``} | :class:`dict`
    collection of sweep definitions

..
    end-Report-sweeps

..
    start-Report-reports

*Reports*: {``None``} | :class:`list`\ [:class:`str`]
    list of reports

..
    end-Report-reports

..
    start-Report-figures

*Figures*: {``None``} | :class:`dict`
    collection of figure definitions

..
    end-Report-figures

Figures Options
===============
Figure Options
--------------
..
    start-Figure-alignment

*Alignment*: ``'left'`` | ``'right'`` | {``'center'``}
    horizontal alignment for subfigs in a figure

..
    end-Figure-alignment

..
    start-Figure-parent

*Parent*: {``None``} | :class:`str`
    name of report from which to inherit options

..
    end-Figure-parent

..
    start-Figure-header

*Header*: {``''``} | :class:`str`
    optional header for a figure

..
    end-Figure-header

..
    start-Figure-subfigures

*Subfigures*: {``None``} | :class:`list`\ [:class:`str`]
    value of option "Subfigures"

..
    end-Figure-subfigures

Subfigures Options
==================
Subfig Options
--------------
..
    start-Subfig-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-Subfig-alignment

..
    start-Subfig-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-Subfig-caption

..
    start-Subfig-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-Subfig-position

..
    start-Subfig-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-Subfig-width

..
    start-Subfig-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-Subfig-type

CoeffTable Options
------------------
..
    start-CoeffTable-iteration

*Iteration*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to sample results

..
    end-CoeffTable-iteration

..
    start-CoeffTable-coefficients

*Coefficients*: {``None``} | :class:`list`\ [:class:`str`]
    list of coefficients to detail in table

..
    end-CoeffTable-coefficients

..
    start-CoeffTable-cn

*CN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CN"

..
    end-CoeffTable-cn

..
    start-CoeffTable-header

*Header*: {``''``} | :class:`str`
    subfigure header

..
    end-CoeffTable-header

..
    start-CoeffTable-epsformat

*EpsFormat*: {``None``} | :class:`str`
    printf-style text format for sampling error

..
    end-CoeffTable-epsformat

..
    start-CoeffTable-cy

*CY*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CY"

..
    end-CoeffTable-cy

..
    start-CoeffTable-cll

*CLL*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLL"

..
    end-CoeffTable-cll

..
    start-CoeffTable-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-CoeffTable-alignment

..
    start-CoeffTable-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-CoeffTable-caption

..
    start-CoeffTable-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-CoeffTable-position

..
    start-CoeffTable-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-CoeffTable-width

..
    start-CoeffTable-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-CoeffTable-type

..
    start-CoeffTable-sigmaformat

*SigmaFormat*: {``None``} | :class:`str`
    printf-sylte text format for standard deviation

..
    end-CoeffTable-sigmaformat

..
    start-CoeffTable-clm

*CLM*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLM"

..
    end-CoeffTable-clm

..
    start-CoeffTable-components

*Components*: {``None``} | :class:`list`\ [:class:`str`]
    list of components for which to report coefficients

..
    end-CoeffTable-components

..
    start-CoeffTable-ca

*CA*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CA"

..
    end-CoeffTable-ca

..
    start-CoeffTable-cln

*CLN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLN"

..
    end-CoeffTable-cln

..
    start-CoeffTable-muformat

*MuFormat*: {``None``} | :class:`str`
    printf-style text format for mean value

..
    end-CoeffTable-muformat

Conditions Options
------------------
..
    start-Conditions-header

*Header*: {``''``} | :class:`str`
    subfigure header

..
    end-Conditions-header

..
    start-Conditions-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-Conditions-alignment

..
    start-Conditions-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-Conditions-caption

..
    start-Conditions-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-Conditions-position

..
    start-Conditions-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-Conditions-width

..
    start-Conditions-skipvars

*SkipVars*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys to leave out of table

..
    end-Conditions-skipvars

..
    start-Conditions-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-Conditions-type

..
    start-Conditions-specialvars

*SpecialVars*: {``None``} | :class:`list`\ [:class:`str`]
    keys not in run matrix to attempt to calculate

..
    end-Conditions-specialvars

ConditionsTable Options
-----------------------
..
    start-ConditionsTable-header

*Header*: {``''``} | :class:`str`
    subfigure header

..
    end-ConditionsTable-header

..
    start-ConditionsTable-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-ConditionsTable-alignment

..
    start-ConditionsTable-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-ConditionsTable-caption

..
    start-ConditionsTable-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-ConditionsTable-position

..
    start-ConditionsTable-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-ConditionsTable-width

..
    start-ConditionsTable-skipvars

*SkipVars*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys to leave out of table

..
    end-ConditionsTable-skipvars

..
    start-ConditionsTable-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-ConditionsTable-type

..
    start-ConditionsTable-specialvars

*SpecialVars*: {``None``} | :class:`list`\ [:class:`str`]
    keys not in run matrix to attempt to calculate

..
    end-ConditionsTable-specialvars

ContourCoeff Options
--------------------
..
    start-ContourCoeff-contourtype

*ContourType*: ``'tricontour'`` | {``'tricontourf'``} | ``'tripcolor'``
    contour plotting function/type to use

..
    end-ContourCoeff-contourtype

..
    start-ContourCoeff-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-ContourCoeff-width

..
    start-ContourCoeff-axisequal

*AxisEqual*: {``True``} | :class:`bool` | :class:`bool_`
    option to scale x and y axes with common scale

..
    end-ContourCoeff-axisequal

..
    start-ContourCoeff-nplotfirst

*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure

..
    end-ContourCoeff-nplotfirst

..
    start-ContourCoeff-plotoptions

*PlotOptions*: {``None``} | :class:`ContourCoeffPlotOpts`
    options for main line(s) of plot

..
    end-ContourCoeff-plotoptions

..
    start-ContourCoeff-xlimmax

*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits

..
    end-ContourCoeff-xlimmax

..
    start-ContourCoeff-ylabeloptions

*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label

..
    end-ContourCoeff-ylabeloptions

..
    start-ContourCoeff-xmax

*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits

..
    end-ContourCoeff-xmax

..
    start-ContourCoeff-restrictionoptions

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction

..
    end-ContourCoeff-restrictionoptions

..
    start-ContourCoeff-xmin

*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits

..
    end-ContourCoeff-xmin

..
    start-ContourCoeff-yticklabeloptions

*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels

..
    end-ContourCoeff-yticklabeloptions

..
    start-ContourCoeff-figurewidth

*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches

..
    end-ContourCoeff-figurewidth

..
    start-ContourCoeff-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-ContourCoeff-caption

..
    start-ContourCoeff-restrictionxposition

*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction

..
    end-ContourCoeff-restrictionxposition

..
    start-ContourCoeff-yticklabels

*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-ContourCoeff-yticklabels

..
    start-ContourCoeff-ticklabels

*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes

..
    end-ContourCoeff-ticklabels

..
    start-ContourCoeff-restrictionyposition

*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction

..
    end-ContourCoeff-restrictionyposition

..
    start-ContourCoeff-xlabeloptions

*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label

..
    end-ContourCoeff-xlabeloptions

..
    start-ContourCoeff-yticks

*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values

..
    end-ContourCoeff-yticks

..
    start-ContourCoeff-ymax

*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits

..
    end-ContourCoeff-ymax

..
    start-ContourCoeff-ymin

*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"

..
    end-ContourCoeff-ymin

..
    start-ContourCoeff-ylabel

*YLabel*: {``None``} | :class:`str`
    manual label for y-axis

..
    end-ContourCoeff-ylabel

..
    start-ContourCoeff-xlabel

*XLabel*: {``None``} | :class:`str`
    manual label for x-axis

..
    end-ContourCoeff-xlabel

..
    start-ContourCoeff-xticks

*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values

..
    end-ContourCoeff-xticks

..
    start-ContourCoeff-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-ContourCoeff-alignment

..
    start-ContourCoeff-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-ContourCoeff-position

..
    start-ContourCoeff-xticklabeloptions

*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels

..
    end-ContourCoeff-xticklabeloptions

..
    start-ContourCoeff-dpi

*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image

..
    end-ContourCoeff-dpi

..
    start-ContourCoeff-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-ContourCoeff-type

..
    start-ContourCoeff-restriction

*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure

..
    end-ContourCoeff-restriction

..
    start-ContourCoeff-linetype

*LineType*: {``'plot'``} | ``'triplot'``
    plot function to use to mark data points

..
    end-ContourCoeff-linetype

..
    start-ContourCoeff-ylimmax

*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits

..
    end-ContourCoeff-ylimmax

..
    start-ContourCoeff-ticks

*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"

..
    end-ContourCoeff-ticks

..
    start-ContourCoeff-xcol

*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis

..
    end-ContourCoeff-xcol

..
    start-ContourCoeff-ylim

*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis

..
    end-ContourCoeff-ylim

..
    start-ContourCoeff-ticklabeloptions

*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes

..
    end-ContourCoeff-ticklabeloptions

..
    start-ContourCoeff-xlim

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis

..
    end-ContourCoeff-xlim

..
    start-ContourCoeff-colorbar

*ColorBar*: {``True``} | :class:`bool` | :class:`bool_`
    option to turn on color bar (scale)

..
    end-ContourCoeff-colorbar

..
    start-ContourCoeff-figureheight

*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches

..
    end-ContourCoeff-figureheight

..
    start-ContourCoeff-xticklabels

*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-ContourCoeff-xticklabels

..
    start-ContourCoeff-contouroptions

*ContourOptions*: {``None``} | :class:`dict`
    options passed to contour plot function

..
    end-ContourCoeff-contouroptions

..
    start-ContourCoeff-restrictionloc

*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text

..
    end-ContourCoeff-restrictionloc

..
    start-ContourCoeff-format

*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format

..
    end-ContourCoeff-format

..
    start-ContourCoeff-contourcolormap

*ContourColorMap*: {``'jet'``} | :class:`str`
    name of color map to use w/ contour plots

..
    end-ContourCoeff-contourcolormap

..
    start-ContourCoeff-ycol

*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis

..
    end-ContourCoeff-ycol

FMTable Options
---------------
..
    start-FMTable-iteration

*Iteration*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to sample results

..
    end-FMTable-iteration

..
    start-FMTable-coefficients

*Coefficients*: {``None``} | :class:`list`\ [:class:`str`]
    list of coefficients to detail in table

..
    end-FMTable-coefficients

..
    start-FMTable-cn

*CN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CN"

..
    end-FMTable-cn

..
    start-FMTable-header

*Header*: {``''``} | :class:`str`
    subfigure header

..
    end-FMTable-header

..
    start-FMTable-epsformat

*EpsFormat*: {``None``} | :class:`str`
    printf-style text format for sampling error

..
    end-FMTable-epsformat

..
    start-FMTable-cy

*CY*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CY"

..
    end-FMTable-cy

..
    start-FMTable-cll

*CLL*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLL"

..
    end-FMTable-cll

..
    start-FMTable-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-FMTable-alignment

..
    start-FMTable-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-FMTable-caption

..
    start-FMTable-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-FMTable-position

..
    start-FMTable-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-FMTable-width

..
    start-FMTable-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-FMTable-type

..
    start-FMTable-sigmaformat

*SigmaFormat*: {``None``} | :class:`str`
    printf-sylte text format for standard deviation

..
    end-FMTable-sigmaformat

..
    start-FMTable-clm

*CLM*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLM"

..
    end-FMTable-clm

..
    start-FMTable-components

*Components*: {``None``} | :class:`list`\ [:class:`str`]
    list of components for which to report coefficients

..
    end-FMTable-components

..
    start-FMTable-ca

*CA*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CA"

..
    end-FMTable-ca

..
    start-FMTable-cln

*CLN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLN"

..
    end-FMTable-cln

..
    start-FMTable-muformat

*MuFormat*: {``None``} | :class:`str`
    printf-style text format for mean value

..
    end-FMTable-muformat

Image Options
-------------
..
    start-Image-imagefile

*ImageFile*: {``'export.png'``} | :class:`str`
    name of image file to copy from case folder

..
    end-Image-imagefile

..
    start-Image-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-Image-alignment

..
    start-Image-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-Image-caption

..
    start-Image-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-Image-position

..
    start-Image-width

*Width*: {``0.5``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-Image-width

..
    start-Image-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-Image-type

Paraview Options
----------------
..
    start-Paraview-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-Paraview-alignment

..
    start-Paraview-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-Paraview-caption

..
    start-Paraview-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-Paraview-position

..
    start-Paraview-width

*Width*: {``0.5``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-Paraview-width

..
    start-Paraview-command

*Command*: {``'pvpython'``} | :class:`str`
    name of Python/Paraview executable to call

..
    end-Paraview-command

..
    start-Paraview-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-Paraview-type

..
    start-Paraview-imagefile

*ImageFile*: {``'export.png'``} | :class:`str`
    name of image file created by *Layout*

..
    end-Paraview-imagefile

..
    start-Paraview-format

*Format*: {``'png'``} | :class:`str`
    image file format

..
    end-Paraview-format

..
    start-Paraview-layout

*Layout*: {``'layout.py'``} | :class:`str`
    name of Python file to execute with Paraview

..
    end-Paraview-layout

PlotCoeff Options
-----------------
..
    start-PlotCoeff-nplotlast

*NPlotLast*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotLast"

..
    end-PlotCoeff-nplotlast

..
    start-PlotCoeff-coefficient

*Coefficient*: {``None``} | :class:`object`
    value of option "Coefficient"

..
    end-PlotCoeff-coefficient

..
    start-PlotCoeff-deltaplotoptions

*DeltaPlotOptions*: {``None``} | :class:`PlotCoeffIterDeltaPlotOpts`
    plot options for fixed-width above and below mu

..
    end-PlotCoeff-deltaplotoptions

..
    start-PlotCoeff-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-PlotCoeff-width

..
    start-PlotCoeff-showdelta

*ShowDelta*: {``[True, False]``} | :class:`bool` | :class:`bool_`
    option to print value of *Delta*

..
    end-PlotCoeff-showdelta

..
    start-PlotCoeff-nplotfirst

*NPlotFirst*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    iteration at which to start figure

..
    end-PlotCoeff-nplotfirst

..
    start-PlotCoeff-plotoptions

*PlotOptions*: {``None``} | :class:`PlotCoeffIterPlotOpts`
    options for main line(s) of plot

..
    end-PlotCoeff-plotoptions

..
    start-PlotCoeff-xlimmax

*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits

..
    end-PlotCoeff-xlimmax

..
    start-PlotCoeff-ylabeloptions

*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label

..
    end-PlotCoeff-ylabeloptions

..
    start-PlotCoeff-xmax

*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits

..
    end-PlotCoeff-xmax

..
    start-PlotCoeff-restrictionoptions

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction

..
    end-PlotCoeff-restrictionoptions

..
    start-PlotCoeff-xmin

*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits

..
    end-PlotCoeff-xmin

..
    start-PlotCoeff-yticklabeloptions

*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels

..
    end-PlotCoeff-yticklabeloptions

..
    start-PlotCoeff-component

*Component*: {``None``} | :class:`object`
    value of option "Component"

..
    end-PlotCoeff-component

..
    start-PlotCoeff-showepsilon

*ShowEpsilon*: {``False``} | :class:`bool` | :class:`bool_`
    option to print value of iterative sampling error

..
    end-PlotCoeff-showepsilon

..
    start-PlotCoeff-figurewidth

*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches

..
    end-PlotCoeff-figurewidth

..
    start-PlotCoeff-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-PlotCoeff-caption

..
    start-PlotCoeff-restrictionxposition

*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction

..
    end-PlotCoeff-restrictionxposition

..
    start-PlotCoeff-deltaformat

*DeltaFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowDelta value

..
    end-PlotCoeff-deltaformat

..
    start-PlotCoeff-delta

*Delta*: {``0.0``} | :class:`float` | :class:`float32`
    specified interval(s) to plot above and below mean

..
    end-PlotCoeff-delta

..
    start-PlotCoeff-yticklabels

*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-PlotCoeff-yticklabels

..
    start-PlotCoeff-ticklabels

*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes

..
    end-PlotCoeff-ticklabels

..
    start-PlotCoeff-ksigma

*KSigma*: {``None``} | :class:`object`
    value of option "KSigma"

..
    end-PlotCoeff-ksigma

..
    start-PlotCoeff-restrictionyposition

*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction

..
    end-PlotCoeff-restrictionyposition

..
    start-PlotCoeff-muformat

*MuFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowMu* value

..
    end-PlotCoeff-muformat

..
    start-PlotCoeff-xlabeloptions

*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label

..
    end-PlotCoeff-xlabeloptions

..
    start-PlotCoeff-yticks

*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values

..
    end-PlotCoeff-yticks

..
    start-PlotCoeff-nplotiters

*NPlotIters*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotIters"

..
    end-PlotCoeff-nplotiters

..
    start-PlotCoeff-ymax

*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits

..
    end-PlotCoeff-ymax

..
    start-PlotCoeff-ymin

*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"

..
    end-PlotCoeff-ymin

..
    start-PlotCoeff-showsigma

*ShowSigma*: {``[True, False]``} | :class:`bool` | :class:`bool_`
    option to print value of standard deviation

..
    end-PlotCoeff-showsigma

..
    start-PlotCoeff-ylabel

*YLabel*: {``None``} | :class:`str`
    manual label for y-axis

..
    end-PlotCoeff-ylabel

..
    start-PlotCoeff-xlabel

*XLabel*: {``None``} | :class:`str`
    manual label for x-axis

..
    end-PlotCoeff-xlabel

..
    start-PlotCoeff-xticks

*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values

..
    end-PlotCoeff-xticks

..
    start-PlotCoeff-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-PlotCoeff-alignment

..
    start-PlotCoeff-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-PlotCoeff-position

..
    start-PlotCoeff-xticklabeloptions

*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels

..
    end-PlotCoeff-xticklabeloptions

..
    start-PlotCoeff-dpi

*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image

..
    end-PlotCoeff-dpi

..
    start-PlotCoeff-sigmaplotoptions

*SigmaPlotOptions*: {``None``} | :class:`object`
    value of option "SigmaPlotOptions"

..
    end-PlotCoeff-sigmaplotoptions

..
    start-PlotCoeff-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-PlotCoeff-type

..
    start-PlotCoeff-restriction

*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure

..
    end-PlotCoeff-restriction

..
    start-PlotCoeff-naverage

*NAverage*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NAverage"

..
    end-PlotCoeff-naverage

..
    start-PlotCoeff-epsilonplotoptions

*EpsilonPlotOptions*: {``None``} | :class:`PlotCoeffIterEpsilonPlotOpts`
    value of option "EpsilonPlotOptions"

..
    end-PlotCoeff-epsilonplotoptions

..
    start-PlotCoeff-captioncomponent

*CaptionComponent*: {``None``} | :class:`str`
    explicit text for component portion of caption

..
    end-PlotCoeff-captioncomponent

..
    start-PlotCoeff-ylimmax

*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits

..
    end-PlotCoeff-ylimmax

..
    start-PlotCoeff-ticks

*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"

..
    end-PlotCoeff-ticks

..
    start-PlotCoeff-ylim

*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis

..
    end-PlotCoeff-ylim

..
    start-PlotCoeff-ticklabeloptions

*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes

..
    end-PlotCoeff-ticklabeloptions

..
    start-PlotCoeff-kepsilon

*KEpsilon*: {``0.0``} | :class:`float` | :class:`float32`
    multiple of iterative error to plot

..
    end-PlotCoeff-kepsilon

..
    start-PlotCoeff-xlim

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis

..
    end-PlotCoeff-xlim

..
    start-PlotCoeff-figureheight

*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches

..
    end-PlotCoeff-figureheight

..
    start-PlotCoeff-sigmaformat

*SigmaFormat*: {``'%.4f'``} | :class:`object`
    printf-style flag for *ShowSigma* value

..
    end-PlotCoeff-sigmaformat

..
    start-PlotCoeff-xticklabels

*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-PlotCoeff-xticklabels

..
    start-PlotCoeff-epsilonformat

*EpsilonFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowEpsilon* value

..
    end-PlotCoeff-epsilonformat

..
    start-PlotCoeff-showmu

*ShowMu*: {``[True, False]``} | :class:`bool` | :class:`bool_`
    option to print value of mean over window

..
    end-PlotCoeff-showmu

..
    start-PlotCoeff-format

*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format

..
    end-PlotCoeff-format

..
    start-PlotCoeff-muplotoptions

*MuPlotOptions*: {``None``} | :class:`PlotCoeffIterMuPlotOpts`
    plot options for horizontal line showing mean

..
    end-PlotCoeff-muplotoptions

..
    start-PlotCoeff-restrictionloc

*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text

..
    end-PlotCoeff-restrictionloc

PlotCoeffIter Options
---------------------
..
    start-PlotCoeffIter-nplotlast

*NPlotLast*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotLast"

..
    end-PlotCoeffIter-nplotlast

..
    start-PlotCoeffIter-coefficient

*Coefficient*: {``None``} | :class:`object`
    value of option "Coefficient"

..
    end-PlotCoeffIter-coefficient

..
    start-PlotCoeffIter-deltaplotoptions

*DeltaPlotOptions*: {``None``} | :class:`PlotCoeffIterDeltaPlotOpts`
    plot options for fixed-width above and below mu

..
    end-PlotCoeffIter-deltaplotoptions

..
    start-PlotCoeffIter-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-PlotCoeffIter-width

..
    start-PlotCoeffIter-showdelta

*ShowDelta*: {``[True, False]``} | :class:`bool` | :class:`bool_`
    option to print value of *Delta*

..
    end-PlotCoeffIter-showdelta

..
    start-PlotCoeffIter-nplotfirst

*NPlotFirst*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    iteration at which to start figure

..
    end-PlotCoeffIter-nplotfirst

..
    start-PlotCoeffIter-plotoptions

*PlotOptions*: {``None``} | :class:`PlotCoeffIterPlotOpts`
    options for main line(s) of plot

..
    end-PlotCoeffIter-plotoptions

..
    start-PlotCoeffIter-xlimmax

*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits

..
    end-PlotCoeffIter-xlimmax

..
    start-PlotCoeffIter-ylabeloptions

*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label

..
    end-PlotCoeffIter-ylabeloptions

..
    start-PlotCoeffIter-xmax

*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits

..
    end-PlotCoeffIter-xmax

..
    start-PlotCoeffIter-restrictionoptions

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction

..
    end-PlotCoeffIter-restrictionoptions

..
    start-PlotCoeffIter-xmin

*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits

..
    end-PlotCoeffIter-xmin

..
    start-PlotCoeffIter-yticklabeloptions

*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels

..
    end-PlotCoeffIter-yticklabeloptions

..
    start-PlotCoeffIter-component

*Component*: {``None``} | :class:`object`
    value of option "Component"

..
    end-PlotCoeffIter-component

..
    start-PlotCoeffIter-showepsilon

*ShowEpsilon*: {``False``} | :class:`bool` | :class:`bool_`
    option to print value of iterative sampling error

..
    end-PlotCoeffIter-showepsilon

..
    start-PlotCoeffIter-figurewidth

*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches

..
    end-PlotCoeffIter-figurewidth

..
    start-PlotCoeffIter-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-PlotCoeffIter-caption

..
    start-PlotCoeffIter-restrictionxposition

*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction

..
    end-PlotCoeffIter-restrictionxposition

..
    start-PlotCoeffIter-deltaformat

*DeltaFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowDelta value

..
    end-PlotCoeffIter-deltaformat

..
    start-PlotCoeffIter-delta

*Delta*: {``0.0``} | :class:`float` | :class:`float32`
    specified interval(s) to plot above and below mean

..
    end-PlotCoeffIter-delta

..
    start-PlotCoeffIter-yticklabels

*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-PlotCoeffIter-yticklabels

..
    start-PlotCoeffIter-ticklabels

*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes

..
    end-PlotCoeffIter-ticklabels

..
    start-PlotCoeffIter-ksigma

*KSigma*: {``None``} | :class:`object`
    value of option "KSigma"

..
    end-PlotCoeffIter-ksigma

..
    start-PlotCoeffIter-restrictionyposition

*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction

..
    end-PlotCoeffIter-restrictionyposition

..
    start-PlotCoeffIter-muformat

*MuFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowMu* value

..
    end-PlotCoeffIter-muformat

..
    start-PlotCoeffIter-xlabeloptions

*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label

..
    end-PlotCoeffIter-xlabeloptions

..
    start-PlotCoeffIter-yticks

*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values

..
    end-PlotCoeffIter-yticks

..
    start-PlotCoeffIter-nplotiters

*NPlotIters*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotIters"

..
    end-PlotCoeffIter-nplotiters

..
    start-PlotCoeffIter-ymax

*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits

..
    end-PlotCoeffIter-ymax

..
    start-PlotCoeffIter-ymin

*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"

..
    end-PlotCoeffIter-ymin

..
    start-PlotCoeffIter-showsigma

*ShowSigma*: {``[True, False]``} | :class:`bool` | :class:`bool_`
    option to print value of standard deviation

..
    end-PlotCoeffIter-showsigma

..
    start-PlotCoeffIter-ylabel

*YLabel*: {``None``} | :class:`str`
    manual label for y-axis

..
    end-PlotCoeffIter-ylabel

..
    start-PlotCoeffIter-xlabel

*XLabel*: {``None``} | :class:`str`
    manual label for x-axis

..
    end-PlotCoeffIter-xlabel

..
    start-PlotCoeffIter-xticks

*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values

..
    end-PlotCoeffIter-xticks

..
    start-PlotCoeffIter-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-PlotCoeffIter-alignment

..
    start-PlotCoeffIter-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-PlotCoeffIter-position

..
    start-PlotCoeffIter-xticklabeloptions

*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels

..
    end-PlotCoeffIter-xticklabeloptions

..
    start-PlotCoeffIter-dpi

*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image

..
    end-PlotCoeffIter-dpi

..
    start-PlotCoeffIter-sigmaplotoptions

*SigmaPlotOptions*: {``None``} | :class:`object`
    value of option "SigmaPlotOptions"

..
    end-PlotCoeffIter-sigmaplotoptions

..
    start-PlotCoeffIter-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-PlotCoeffIter-type

..
    start-PlotCoeffIter-restriction

*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure

..
    end-PlotCoeffIter-restriction

..
    start-PlotCoeffIter-naverage

*NAverage*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NAverage"

..
    end-PlotCoeffIter-naverage

..
    start-PlotCoeffIter-epsilonplotoptions

*EpsilonPlotOptions*: {``None``} | :class:`PlotCoeffIterEpsilonPlotOpts`
    value of option "EpsilonPlotOptions"

..
    end-PlotCoeffIter-epsilonplotoptions

..
    start-PlotCoeffIter-captioncomponent

*CaptionComponent*: {``None``} | :class:`str`
    explicit text for component portion of caption

..
    end-PlotCoeffIter-captioncomponent

..
    start-PlotCoeffIter-ylimmax

*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits

..
    end-PlotCoeffIter-ylimmax

..
    start-PlotCoeffIter-ticks

*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"

..
    end-PlotCoeffIter-ticks

..
    start-PlotCoeffIter-ylim

*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis

..
    end-PlotCoeffIter-ylim

..
    start-PlotCoeffIter-ticklabeloptions

*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes

..
    end-PlotCoeffIter-ticklabeloptions

..
    start-PlotCoeffIter-kepsilon

*KEpsilon*: {``0.0``} | :class:`float` | :class:`float32`
    multiple of iterative error to plot

..
    end-PlotCoeffIter-kepsilon

..
    start-PlotCoeffIter-xlim

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis

..
    end-PlotCoeffIter-xlim

..
    start-PlotCoeffIter-figureheight

*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches

..
    end-PlotCoeffIter-figureheight

..
    start-PlotCoeffIter-sigmaformat

*SigmaFormat*: {``'%.4f'``} | :class:`object`
    printf-style flag for *ShowSigma* value

..
    end-PlotCoeffIter-sigmaformat

..
    start-PlotCoeffIter-xticklabels

*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-PlotCoeffIter-xticklabels

..
    start-PlotCoeffIter-epsilonformat

*EpsilonFormat*: {``'%.4f'``} | :class:`str`
    printf-style flag for *ShowEpsilon* value

..
    end-PlotCoeffIter-epsilonformat

..
    start-PlotCoeffIter-showmu

*ShowMu*: {``[True, False]``} | :class:`bool` | :class:`bool_`
    option to print value of mean over window

..
    end-PlotCoeffIter-showmu

..
    start-PlotCoeffIter-format

*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format

..
    end-PlotCoeffIter-format

..
    start-PlotCoeffIter-muplotoptions

*MuPlotOptions*: {``None``} | :class:`PlotCoeffIterMuPlotOpts`
    plot options for horizontal line showing mean

..
    end-PlotCoeffIter-muplotoptions

..
    start-PlotCoeffIter-restrictionloc

*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text

..
    end-PlotCoeffIter-restrictionloc

PlotCoeffSweep Options
----------------------
..
    start-PlotCoeffSweep-minmaxoptions

*MinMaxOptions*: {``None``} | :class:`PlotCoeffSweepMinMaxPlotOpts`
    plot options for *MinMax* plot

..
    end-PlotCoeffSweep-minmaxoptions

..
    start-PlotCoeffSweep-coefficient

*Coefficient*: {``None``} | :class:`object`
    value of option "Coefficient"

..
    end-PlotCoeffSweep-coefficient

..
    start-PlotCoeffSweep-target

*Target*: {``None``} | :class:`str`
    name of target databook to co-plot

..
    end-PlotCoeffSweep-target

..
    start-PlotCoeffSweep-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-PlotCoeffSweep-width

..
    start-PlotCoeffSweep-nplotfirst

*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure

..
    end-PlotCoeffSweep-nplotfirst

..
    start-PlotCoeffSweep-plotoptions

*PlotOptions*: {``None``} | :class:`PlotCoeffSweepPlotOpts`
    options for main line(s) of plot

..
    end-PlotCoeffSweep-plotoptions

..
    start-PlotCoeffSweep-xlimmax

*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits

..
    end-PlotCoeffSweep-xlimmax

..
    start-PlotCoeffSweep-ylabeloptions

*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label

..
    end-PlotCoeffSweep-ylabeloptions

..
    start-PlotCoeffSweep-xmax

*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits

..
    end-PlotCoeffSweep-xmax

..
    start-PlotCoeffSweep-restrictionoptions

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction

..
    end-PlotCoeffSweep-restrictionoptions

..
    start-PlotCoeffSweep-xmin

*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits

..
    end-PlotCoeffSweep-xmin

..
    start-PlotCoeffSweep-yticklabeloptions

*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels

..
    end-PlotCoeffSweep-yticklabeloptions

..
    start-PlotCoeffSweep-component

*Component*: {``None``} | :class:`object`
    value of option "Component"

..
    end-PlotCoeffSweep-component

..
    start-PlotCoeffSweep-targetoptions

*TargetOptions*: {``None``} | :class:`PlotCoeffSweepTargetPlotOpts`
    plot options for optional target

..
    end-PlotCoeffSweep-targetoptions

..
    start-PlotCoeffSweep-figurewidth

*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches

..
    end-PlotCoeffSweep-figurewidth

..
    start-PlotCoeffSweep-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-PlotCoeffSweep-caption

..
    start-PlotCoeffSweep-restrictionxposition

*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction

..
    end-PlotCoeffSweep-restrictionxposition

..
    start-PlotCoeffSweep-yticklabels

*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-PlotCoeffSweep-yticklabels

..
    start-PlotCoeffSweep-ticklabels

*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes

..
    end-PlotCoeffSweep-ticklabels

..
    start-PlotCoeffSweep-ksigma

*KSigma*: {``None``} | :class:`object`
    value of option "KSigma"

..
    end-PlotCoeffSweep-ksigma

..
    start-PlotCoeffSweep-restrictionyposition

*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction

..
    end-PlotCoeffSweep-restrictionyposition

..
    start-PlotCoeffSweep-minmax

*MinMax*: {``False``} | :class:`bool` | :class:`bool_`
    option to plot min/max of value over iterative window

..
    end-PlotCoeffSweep-minmax

..
    start-PlotCoeffSweep-xlabeloptions

*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label

..
    end-PlotCoeffSweep-xlabeloptions

..
    start-PlotCoeffSweep-yticks

*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values

..
    end-PlotCoeffSweep-yticks

..
    start-PlotCoeffSweep-ymax

*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits

..
    end-PlotCoeffSweep-ymax

..
    start-PlotCoeffSweep-ymin

*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"

..
    end-PlotCoeffSweep-ymin

..
    start-PlotCoeffSweep-ylabel

*YLabel*: {``None``} | :class:`str`
    manual label for y-axis

..
    end-PlotCoeffSweep-ylabel

..
    start-PlotCoeffSweep-xlabel

*XLabel*: {``None``} | :class:`str`
    manual label for x-axis

..
    end-PlotCoeffSweep-xlabel

..
    start-PlotCoeffSweep-xticks

*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values

..
    end-PlotCoeffSweep-xticks

..
    start-PlotCoeffSweep-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-PlotCoeffSweep-alignment

..
    start-PlotCoeffSweep-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-PlotCoeffSweep-position

..
    start-PlotCoeffSweep-xticklabeloptions

*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels

..
    end-PlotCoeffSweep-xticklabeloptions

..
    start-PlotCoeffSweep-dpi

*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image

..
    end-PlotCoeffSweep-dpi

..
    start-PlotCoeffSweep-sigmaplotoptions

*SigmaPlotOptions*: {``None``} | :class:`object`
    value of option "SigmaPlotOptions"

..
    end-PlotCoeffSweep-sigmaplotoptions

..
    start-PlotCoeffSweep-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-PlotCoeffSweep-type

..
    start-PlotCoeffSweep-restriction

*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure

..
    end-PlotCoeffSweep-restriction

..
    start-PlotCoeffSweep-ylimmax

*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits

..
    end-PlotCoeffSweep-ylimmax

..
    start-PlotCoeffSweep-ticks

*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"

..
    end-PlotCoeffSweep-ticks

..
    start-PlotCoeffSweep-ylim

*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis

..
    end-PlotCoeffSweep-ylim

..
    start-PlotCoeffSweep-ticklabeloptions

*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes

..
    end-PlotCoeffSweep-ticklabeloptions

..
    start-PlotCoeffSweep-xlim

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis

..
    end-PlotCoeffSweep-xlim

..
    start-PlotCoeffSweep-figureheight

*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches

..
    end-PlotCoeffSweep-figureheight

..
    start-PlotCoeffSweep-xticklabels

*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-PlotCoeffSweep-xticklabels

..
    start-PlotCoeffSweep-format

*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format

..
    end-PlotCoeffSweep-format

..
    start-PlotCoeffSweep-restrictionloc

*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text

..
    end-PlotCoeffSweep-restrictionloc

PlotContour Options
-------------------
..
    start-PlotContour-contourtype

*ContourType*: ``'tricontour'`` | {``'tricontourf'``} | ``'tripcolor'``
    contour plotting function/type to use

..
    end-PlotContour-contourtype

..
    start-PlotContour-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-PlotContour-width

..
    start-PlotContour-axisequal

*AxisEqual*: {``True``} | :class:`bool` | :class:`bool_`
    option to scale x and y axes with common scale

..
    end-PlotContour-axisequal

..
    start-PlotContour-nplotfirst

*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure

..
    end-PlotContour-nplotfirst

..
    start-PlotContour-plotoptions

*PlotOptions*: {``None``} | :class:`ContourCoeffPlotOpts`
    options for main line(s) of plot

..
    end-PlotContour-plotoptions

..
    start-PlotContour-xlimmax

*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits

..
    end-PlotContour-xlimmax

..
    start-PlotContour-ylabeloptions

*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label

..
    end-PlotContour-ylabeloptions

..
    start-PlotContour-xmax

*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits

..
    end-PlotContour-xmax

..
    start-PlotContour-restrictionoptions

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction

..
    end-PlotContour-restrictionoptions

..
    start-PlotContour-xmin

*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits

..
    end-PlotContour-xmin

..
    start-PlotContour-yticklabeloptions

*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels

..
    end-PlotContour-yticklabeloptions

..
    start-PlotContour-figurewidth

*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches

..
    end-PlotContour-figurewidth

..
    start-PlotContour-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-PlotContour-caption

..
    start-PlotContour-restrictionxposition

*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction

..
    end-PlotContour-restrictionxposition

..
    start-PlotContour-yticklabels

*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-PlotContour-yticklabels

..
    start-PlotContour-ticklabels

*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes

..
    end-PlotContour-ticklabels

..
    start-PlotContour-restrictionyposition

*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction

..
    end-PlotContour-restrictionyposition

..
    start-PlotContour-xlabeloptions

*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label

..
    end-PlotContour-xlabeloptions

..
    start-PlotContour-yticks

*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values

..
    end-PlotContour-yticks

..
    start-PlotContour-ymax

*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits

..
    end-PlotContour-ymax

..
    start-PlotContour-ymin

*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"

..
    end-PlotContour-ymin

..
    start-PlotContour-ylabel

*YLabel*: {``None``} | :class:`str`
    manual label for y-axis

..
    end-PlotContour-ylabel

..
    start-PlotContour-xlabel

*XLabel*: {``None``} | :class:`str`
    manual label for x-axis

..
    end-PlotContour-xlabel

..
    start-PlotContour-xticks

*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values

..
    end-PlotContour-xticks

..
    start-PlotContour-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-PlotContour-alignment

..
    start-PlotContour-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-PlotContour-position

..
    start-PlotContour-xticklabeloptions

*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels

..
    end-PlotContour-xticklabeloptions

..
    start-PlotContour-dpi

*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image

..
    end-PlotContour-dpi

..
    start-PlotContour-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-PlotContour-type

..
    start-PlotContour-restriction

*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure

..
    end-PlotContour-restriction

..
    start-PlotContour-linetype

*LineType*: {``'plot'``} | ``'triplot'``
    plot function to use to mark data points

..
    end-PlotContour-linetype

..
    start-PlotContour-ylimmax

*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits

..
    end-PlotContour-ylimmax

..
    start-PlotContour-ticks

*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"

..
    end-PlotContour-ticks

..
    start-PlotContour-xcol

*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis

..
    end-PlotContour-xcol

..
    start-PlotContour-ylim

*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis

..
    end-PlotContour-ylim

..
    start-PlotContour-ticklabeloptions

*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes

..
    end-PlotContour-ticklabeloptions

..
    start-PlotContour-xlim

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis

..
    end-PlotContour-xlim

..
    start-PlotContour-colorbar

*ColorBar*: {``True``} | :class:`bool` | :class:`bool_`
    option to turn on color bar (scale)

..
    end-PlotContour-colorbar

..
    start-PlotContour-figureheight

*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches

..
    end-PlotContour-figureheight

..
    start-PlotContour-xticklabels

*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-PlotContour-xticklabels

..
    start-PlotContour-contouroptions

*ContourOptions*: {``None``} | :class:`dict`
    options passed to contour plot function

..
    end-PlotContour-contouroptions

..
    start-PlotContour-restrictionloc

*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text

..
    end-PlotContour-restrictionloc

..
    start-PlotContour-format

*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format

..
    end-PlotContour-format

..
    start-PlotContour-contourcolormap

*ContourColorMap*: {``'jet'``} | :class:`str`
    name of color map to use w/ contour plots

..
    end-PlotContour-contourcolormap

..
    start-PlotContour-ycol

*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis

..
    end-PlotContour-ycol

PlotContourSweep Options
------------------------
..
    start-PlotContourSweep-contourtype

*ContourType*: ``'tricontour'`` | {``'tricontourf'``} | ``'tripcolor'``
    contour plotting function/type to use

..
    end-PlotContourSweep-contourtype

..
    start-PlotContourSweep-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-PlotContourSweep-width

..
    start-PlotContourSweep-axisequal

*AxisEqual*: {``True``} | :class:`bool` | :class:`bool_`
    option to scale x and y axes with common scale

..
    end-PlotContourSweep-axisequal

..
    start-PlotContourSweep-nplotfirst

*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure

..
    end-PlotContourSweep-nplotfirst

..
    start-PlotContourSweep-plotoptions

*PlotOptions*: {``None``} | :class:`ContourCoeffPlotOpts`
    options for main line(s) of plot

..
    end-PlotContourSweep-plotoptions

..
    start-PlotContourSweep-xlimmax

*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits

..
    end-PlotContourSweep-xlimmax

..
    start-PlotContourSweep-ylabeloptions

*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label

..
    end-PlotContourSweep-ylabeloptions

..
    start-PlotContourSweep-xmax

*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits

..
    end-PlotContourSweep-xmax

..
    start-PlotContourSweep-restrictionoptions

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction

..
    end-PlotContourSweep-restrictionoptions

..
    start-PlotContourSweep-xmin

*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits

..
    end-PlotContourSweep-xmin

..
    start-PlotContourSweep-yticklabeloptions

*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels

..
    end-PlotContourSweep-yticklabeloptions

..
    start-PlotContourSweep-figurewidth

*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches

..
    end-PlotContourSweep-figurewidth

..
    start-PlotContourSweep-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-PlotContourSweep-caption

..
    start-PlotContourSweep-restrictionxposition

*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction

..
    end-PlotContourSweep-restrictionxposition

..
    start-PlotContourSweep-yticklabels

*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-PlotContourSweep-yticklabels

..
    start-PlotContourSweep-ticklabels

*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes

..
    end-PlotContourSweep-ticklabels

..
    start-PlotContourSweep-restrictionyposition

*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction

..
    end-PlotContourSweep-restrictionyposition

..
    start-PlotContourSweep-xlabeloptions

*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label

..
    end-PlotContourSweep-xlabeloptions

..
    start-PlotContourSweep-yticks

*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values

..
    end-PlotContourSweep-yticks

..
    start-PlotContourSweep-ymax

*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits

..
    end-PlotContourSweep-ymax

..
    start-PlotContourSweep-ymin

*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"

..
    end-PlotContourSweep-ymin

..
    start-PlotContourSweep-ylabel

*YLabel*: {``None``} | :class:`str`
    manual label for y-axis

..
    end-PlotContourSweep-ylabel

..
    start-PlotContourSweep-xlabel

*XLabel*: {``None``} | :class:`str`
    manual label for x-axis

..
    end-PlotContourSweep-xlabel

..
    start-PlotContourSweep-xticks

*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values

..
    end-PlotContourSweep-xticks

..
    start-PlotContourSweep-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-PlotContourSweep-alignment

..
    start-PlotContourSweep-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-PlotContourSweep-position

..
    start-PlotContourSweep-xticklabeloptions

*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels

..
    end-PlotContourSweep-xticklabeloptions

..
    start-PlotContourSweep-dpi

*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image

..
    end-PlotContourSweep-dpi

..
    start-PlotContourSweep-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-PlotContourSweep-type

..
    start-PlotContourSweep-restriction

*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure

..
    end-PlotContourSweep-restriction

..
    start-PlotContourSweep-linetype

*LineType*: {``'plot'``} | ``'triplot'``
    plot function to use to mark data points

..
    end-PlotContourSweep-linetype

..
    start-PlotContourSweep-ylimmax

*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits

..
    end-PlotContourSweep-ylimmax

..
    start-PlotContourSweep-ticks

*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"

..
    end-PlotContourSweep-ticks

..
    start-PlotContourSweep-xcol

*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis

..
    end-PlotContourSweep-xcol

..
    start-PlotContourSweep-ylim

*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis

..
    end-PlotContourSweep-ylim

..
    start-PlotContourSweep-ticklabeloptions

*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes

..
    end-PlotContourSweep-ticklabeloptions

..
    start-PlotContourSweep-xlim

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis

..
    end-PlotContourSweep-xlim

..
    start-PlotContourSweep-colorbar

*ColorBar*: {``True``} | :class:`bool` | :class:`bool_`
    option to turn on color bar (scale)

..
    end-PlotContourSweep-colorbar

..
    start-PlotContourSweep-figureheight

*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches

..
    end-PlotContourSweep-figureheight

..
    start-PlotContourSweep-xticklabels

*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-PlotContourSweep-xticklabels

..
    start-PlotContourSweep-contouroptions

*ContourOptions*: {``None``} | :class:`dict`
    options passed to contour plot function

..
    end-PlotContourSweep-contouroptions

..
    start-PlotContourSweep-restrictionloc

*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text

..
    end-PlotContourSweep-restrictionloc

..
    start-PlotContourSweep-format

*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format

..
    end-PlotContourSweep-format

..
    start-PlotContourSweep-contourcolormap

*ContourColorMap*: {``'jet'``} | :class:`str`
    name of color map to use w/ contour plots

..
    end-PlotContourSweep-contourcolormap

..
    start-PlotContourSweep-ycol

*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis

..
    end-PlotContourSweep-ycol

PlotL1 Options
--------------
..
    start-PlotL1-nplotlast

*NPlotLast*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotLast"

..
    end-PlotL1-nplotlast

..
    start-PlotL1-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-PlotL1-width

..
    start-PlotL1-nplotfirst

*NPlotFirst*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    iteration at which to start figure

..
    end-PlotL1-nplotfirst

..
    start-PlotL1-residual

*Residual*: {``'L1'``} | :class:`str`
    name of residual field or type to plot

..
    end-PlotL1-residual

..
    start-PlotL1-plotoptions

*PlotOptions*: {``None``} | :class:`ResidPlotOpts`
    options for main line(s) of plot

..
    end-PlotL1-plotoptions

..
    start-PlotL1-xlimmax

*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits

..
    end-PlotL1-xlimmax

..
    start-PlotL1-ylabeloptions

*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label

..
    end-PlotL1-ylabeloptions

..
    start-PlotL1-xmax

*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits

..
    end-PlotL1-xmax

..
    start-PlotL1-restrictionoptions

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction

..
    end-PlotL1-restrictionoptions

..
    start-PlotL1-xmin

*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits

..
    end-PlotL1-xmin

..
    start-PlotL1-yticklabeloptions

*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels

..
    end-PlotL1-yticklabeloptions

..
    start-PlotL1-figurewidth

*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches

..
    end-PlotL1-figurewidth

..
    start-PlotL1-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-PlotL1-caption

..
    start-PlotL1-restrictionxposition

*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction

..
    end-PlotL1-restrictionxposition

..
    start-PlotL1-yticklabels

*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-PlotL1-yticklabels

..
    start-PlotL1-ticklabels

*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes

..
    end-PlotL1-ticklabels

..
    start-PlotL1-restrictionyposition

*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction

..
    end-PlotL1-restrictionyposition

..
    start-PlotL1-xlabeloptions

*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label

..
    end-PlotL1-xlabeloptions

..
    start-PlotL1-yticks

*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values

..
    end-PlotL1-yticks

..
    start-PlotL1-nplotiters

*NPlotIters*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotIters"

..
    end-PlotL1-nplotiters

..
    start-PlotL1-ymax

*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits

..
    end-PlotL1-ymax

..
    start-PlotL1-plotoptions0

*PlotOptions0*: {``None``} | :class:`ResidPlot0Opts`
    plot options for initial residual

..
    end-PlotL1-plotoptions0

..
    start-PlotL1-ymin

*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"

..
    end-PlotL1-ymin

..
    start-PlotL1-ylabel

*YLabel*: {``None``} | :class:`str`
    manual label for y-axis

..
    end-PlotL1-ylabel

..
    start-PlotL1-xlabel

*XLabel*: {``None``} | :class:`str`
    manual label for x-axis

..
    end-PlotL1-xlabel

..
    start-PlotL1-xticks

*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values

..
    end-PlotL1-xticks

..
    start-PlotL1-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-PlotL1-alignment

..
    start-PlotL1-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-PlotL1-position

..
    start-PlotL1-xticklabeloptions

*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels

..
    end-PlotL1-xticklabeloptions

..
    start-PlotL1-dpi

*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image

..
    end-PlotL1-dpi

..
    start-PlotL1-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-PlotL1-type

..
    start-PlotL1-restriction

*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure

..
    end-PlotL1-restriction

..
    start-PlotL1-ylimmax

*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits

..
    end-PlotL1-ylimmax

..
    start-PlotL1-ticks

*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"

..
    end-PlotL1-ticks

..
    start-PlotL1-ylim

*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis

..
    end-PlotL1-ylim

..
    start-PlotL1-ticklabeloptions

*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes

..
    end-PlotL1-ticklabeloptions

..
    start-PlotL1-xlim

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis

..
    end-PlotL1-xlim

..
    start-PlotL1-figureheight

*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches

..
    end-PlotL1-figureheight

..
    start-PlotL1-xticklabels

*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-PlotL1-xticklabels

..
    start-PlotL1-format

*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format

..
    end-PlotL1-format

..
    start-PlotL1-restrictionloc

*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text

..
    end-PlotL1-restrictionloc

PlotL2 Options
--------------
..
    start-PlotL2-nplotlast

*NPlotLast*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotLast"

..
    end-PlotL2-nplotlast

..
    start-PlotL2-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-PlotL2-width

..
    start-PlotL2-nplotfirst

*NPlotFirst*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    iteration at which to start figure

..
    end-PlotL2-nplotfirst

..
    start-PlotL2-residual

*Residual*: {``'L2'``} | :class:`str`
    name of residual field or type to plot

..
    end-PlotL2-residual

..
    start-PlotL2-plotoptions

*PlotOptions*: {``None``} | :class:`ResidPlotOpts`
    options for main line(s) of plot

..
    end-PlotL2-plotoptions

..
    start-PlotL2-xlimmax

*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits

..
    end-PlotL2-xlimmax

..
    start-PlotL2-ylabeloptions

*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label

..
    end-PlotL2-ylabeloptions

..
    start-PlotL2-xmax

*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits

..
    end-PlotL2-xmax

..
    start-PlotL2-restrictionoptions

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction

..
    end-PlotL2-restrictionoptions

..
    start-PlotL2-xmin

*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits

..
    end-PlotL2-xmin

..
    start-PlotL2-yticklabeloptions

*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels

..
    end-PlotL2-yticklabeloptions

..
    start-PlotL2-figurewidth

*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches

..
    end-PlotL2-figurewidth

..
    start-PlotL2-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-PlotL2-caption

..
    start-PlotL2-restrictionxposition

*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction

..
    end-PlotL2-restrictionxposition

..
    start-PlotL2-yticklabels

*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-PlotL2-yticklabels

..
    start-PlotL2-ticklabels

*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes

..
    end-PlotL2-ticklabels

..
    start-PlotL2-restrictionyposition

*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction

..
    end-PlotL2-restrictionyposition

..
    start-PlotL2-xlabeloptions

*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label

..
    end-PlotL2-xlabeloptions

..
    start-PlotL2-yticks

*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values

..
    end-PlotL2-yticks

..
    start-PlotL2-nplotiters

*NPlotIters*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotIters"

..
    end-PlotL2-nplotiters

..
    start-PlotL2-ymax

*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits

..
    end-PlotL2-ymax

..
    start-PlotL2-plotoptions0

*PlotOptions0*: {``None``} | :class:`ResidPlot0Opts`
    plot options for initial residual

..
    end-PlotL2-plotoptions0

..
    start-PlotL2-ymin

*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"

..
    end-PlotL2-ymin

..
    start-PlotL2-ylabel

*YLabel*: {``None``} | :class:`str`
    manual label for y-axis

..
    end-PlotL2-ylabel

..
    start-PlotL2-xlabel

*XLabel*: {``None``} | :class:`str`
    manual label for x-axis

..
    end-PlotL2-xlabel

..
    start-PlotL2-xticks

*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values

..
    end-PlotL2-xticks

..
    start-PlotL2-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-PlotL2-alignment

..
    start-PlotL2-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-PlotL2-position

..
    start-PlotL2-xticklabeloptions

*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels

..
    end-PlotL2-xticklabeloptions

..
    start-PlotL2-dpi

*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image

..
    end-PlotL2-dpi

..
    start-PlotL2-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-PlotL2-type

..
    start-PlotL2-restriction

*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure

..
    end-PlotL2-restriction

..
    start-PlotL2-ylimmax

*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits

..
    end-PlotL2-ylimmax

..
    start-PlotL2-ticks

*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"

..
    end-PlotL2-ticks

..
    start-PlotL2-ylim

*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis

..
    end-PlotL2-ylim

..
    start-PlotL2-ticklabeloptions

*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes

..
    end-PlotL2-ticklabeloptions

..
    start-PlotL2-xlim

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis

..
    end-PlotL2-xlim

..
    start-PlotL2-figureheight

*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches

..
    end-PlotL2-figureheight

..
    start-PlotL2-xticklabels

*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-PlotL2-xticklabels

..
    start-PlotL2-format

*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format

..
    end-PlotL2-format

..
    start-PlotL2-restrictionloc

*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text

..
    end-PlotL2-restrictionloc

PlotLineLoad Options
--------------------
..
    start-PlotLineLoad-coefficient

*Coefficient*: {``None``} | :class:`str`
    coefficient to plot

..
    end-PlotLineLoad-coefficient

..
    start-PlotLineLoad-autoupdate

*AutoUpdate*: {``True``} | :class:`bool` | :class:`bool_`
    option to create line loads if not in databook

..
    end-PlotLineLoad-autoupdate

..
    start-PlotLineLoad-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-PlotLineLoad-width

..
    start-PlotLineLoad-adjustbottom

*AdjustBottom*: {``0.1``} | :class:`float` | :class:`float32`
    margin from axes to bottom of figure

..
    end-PlotLineLoad-adjustbottom

..
    start-PlotLineLoad-orientation

*Orientation*: ``'horizontal'`` | {``'vertical'``}
    orientation of vehicle in line load plot

..
    end-PlotLineLoad-orientation

..
    start-PlotLineLoad-nplotfirst

*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure

..
    end-PlotLineLoad-nplotfirst

..
    start-PlotLineLoad-plotoptions

*PlotOptions*: {``None``} | :class:`PlotLineLoadPlotOpts`
    options for main line(s) of plot

..
    end-PlotLineLoad-plotoptions

..
    start-PlotLineLoad-xlimmax

*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits

..
    end-PlotLineLoad-xlimmax

..
    start-PlotLineLoad-ylabeloptions

*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label

..
    end-PlotLineLoad-ylabeloptions

..
    start-PlotLineLoad-xmax

*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits

..
    end-PlotLineLoad-xmax

..
    start-PlotLineLoad-component

*Component*: {``None``} | :class:`str`
    config component tp plot

..
    end-PlotLineLoad-component

..
    start-PlotLineLoad-restrictionoptions

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction

..
    end-PlotLineLoad-restrictionoptions

..
    start-PlotLineLoad-xmin

*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits

..
    end-PlotLineLoad-xmin

..
    start-PlotLineLoad-yticklabeloptions

*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels

..
    end-PlotLineLoad-yticklabeloptions

..
    start-PlotLineLoad-adjustright

*AdjustRight*: {``0.97``} | :class:`float` | :class:`float32`
    margin from axes to right of figure

..
    end-PlotLineLoad-adjustright

..
    start-PlotLineLoad-adjusttop

*AdjustTop*: {``0.97``} | :class:`float` | :class:`float32`
    margin from axes to top of figure

..
    end-PlotLineLoad-adjusttop

..
    start-PlotLineLoad-adjustleft

*AdjustLeft*: {``0.12``} | :class:`float` | :class:`float32`
    margin from axes to left of figure

..
    end-PlotLineLoad-adjustleft

..
    start-PlotLineLoad-figurewidth

*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches

..
    end-PlotLineLoad-figurewidth

..
    start-PlotLineLoad-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-PlotLineLoad-caption

..
    start-PlotLineLoad-restrictionxposition

*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction

..
    end-PlotLineLoad-restrictionxposition

..
    start-PlotLineLoad-xpad

*XPad*: {``0.03``} | :class:`float` | :class:`float32`
    additional padding from data to xmin and xmax w/i axes

..
    end-PlotLineLoad-xpad

..
    start-PlotLineLoad-seamlocation

*SeamLocation*: ``'bottom'`` | ``'left'`` | ``'right'`` | ``'top'``
    location for optional seam curve plot

..
    end-PlotLineLoad-seamlocation

..
    start-PlotLineLoad-yticklabels

*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-PlotLineLoad-yticklabels

..
    start-PlotLineLoad-ticklabels

*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes

..
    end-PlotLineLoad-ticklabels

..
    start-PlotLineLoad-restrictionyposition

*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction

..
    end-PlotLineLoad-restrictionyposition

..
    start-PlotLineLoad-xlabeloptions

*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label

..
    end-PlotLineLoad-xlabeloptions

..
    start-PlotLineLoad-yticks

*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values

..
    end-PlotLineLoad-yticks

..
    start-PlotLineLoad-ypad

*YPad*: {``0.03``} | :class:`float` | :class:`float32`
    additional padding from data to ymin and ymax w/i axes

..
    end-PlotLineLoad-ypad

..
    start-PlotLineLoad-ymax

*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits

..
    end-PlotLineLoad-ymax

..
    start-PlotLineLoad-ymin

*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"

..
    end-PlotLineLoad-ymin

..
    start-PlotLineLoad-ylabel

*YLabel*: {``None``} | :class:`str`
    manual label for y-axis

..
    end-PlotLineLoad-ylabel

..
    start-PlotLineLoad-xlabel

*XLabel*: {``None``} | :class:`str`
    manual label for x-axis

..
    end-PlotLineLoad-xlabel

..
    start-PlotLineLoad-xticks

*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values

..
    end-PlotLineLoad-xticks

..
    start-PlotLineLoad-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-PlotLineLoad-alignment

..
    start-PlotLineLoad-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-PlotLineLoad-position

..
    start-PlotLineLoad-xticklabeloptions

*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels

..
    end-PlotLineLoad-xticklabeloptions

..
    start-PlotLineLoad-dpi

*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image

..
    end-PlotLineLoad-dpi

..
    start-PlotLineLoad-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-PlotLineLoad-type

..
    start-PlotLineLoad-subplotmargin

*SubplotMargin*: {``0.015``} | :class:`float` | :class:`float32`
    margin between line load and seam curve subplots

..
    end-PlotLineLoad-subplotmargin

..
    start-PlotLineLoad-restriction

*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure

..
    end-PlotLineLoad-restriction

..
    start-PlotLineLoad-ylimmax

*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits

..
    end-PlotLineLoad-ylimmax

..
    start-PlotLineLoad-ticks

*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"

..
    end-PlotLineLoad-ticks

..
    start-PlotLineLoad-ylim

*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis

..
    end-PlotLineLoad-ylim

..
    start-PlotLineLoad-seamcurve

*SeamCurve*: ``'smy'`` | ``'smz'``
    name of seam curve, if any, to show w/ line loads

..
    end-PlotLineLoad-seamcurve

..
    start-PlotLineLoad-ticklabeloptions

*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes

..
    end-PlotLineLoad-ticklabeloptions

..
    start-PlotLineLoad-xlim

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis

..
    end-PlotLineLoad-xlim

..
    start-PlotLineLoad-figureheight

*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches

..
    end-PlotLineLoad-figureheight

..
    start-PlotLineLoad-xticklabels

*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-PlotLineLoad-xticklabels

..
    start-PlotLineLoad-seamoptions

*SeamOptions*: {``None``} | :class:`PlotLineLoadSeamPlotOpts`
    plot options for optional seam curve

..
    end-PlotLineLoad-seamoptions

..
    start-PlotLineLoad-format

*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format

..
    end-PlotLineLoad-format

..
    start-PlotLineLoad-restrictionloc

*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text

..
    end-PlotLineLoad-restrictionloc

PlotResid Options
-----------------
..
    start-PlotResid-nplotlast

*NPlotLast*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotLast"

..
    end-PlotResid-nplotlast

..
    start-PlotResid-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-PlotResid-width

..
    start-PlotResid-nplotfirst

*NPlotFirst*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    iteration at which to start figure

..
    end-PlotResid-nplotfirst

..
    start-PlotResid-residual

*Residual*: {``'L2'``} | :class:`str`
    name of residual field or type to plot

..
    end-PlotResid-residual

..
    start-PlotResid-plotoptions

*PlotOptions*: {``None``} | :class:`ResidPlotOpts`
    options for main line(s) of plot

..
    end-PlotResid-plotoptions

..
    start-PlotResid-xlimmax

*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits

..
    end-PlotResid-xlimmax

..
    start-PlotResid-ylabeloptions

*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label

..
    end-PlotResid-ylabeloptions

..
    start-PlotResid-xmax

*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits

..
    end-PlotResid-xmax

..
    start-PlotResid-restrictionoptions

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction

..
    end-PlotResid-restrictionoptions

..
    start-PlotResid-xmin

*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits

..
    end-PlotResid-xmin

..
    start-PlotResid-yticklabeloptions

*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels

..
    end-PlotResid-yticklabeloptions

..
    start-PlotResid-figurewidth

*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches

..
    end-PlotResid-figurewidth

..
    start-PlotResid-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-PlotResid-caption

..
    start-PlotResid-restrictionxposition

*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction

..
    end-PlotResid-restrictionxposition

..
    start-PlotResid-yticklabels

*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-PlotResid-yticklabels

..
    start-PlotResid-ticklabels

*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes

..
    end-PlotResid-ticklabels

..
    start-PlotResid-restrictionyposition

*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction

..
    end-PlotResid-restrictionyposition

..
    start-PlotResid-xlabeloptions

*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label

..
    end-PlotResid-xlabeloptions

..
    start-PlotResid-yticks

*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values

..
    end-PlotResid-yticks

..
    start-PlotResid-nplotiters

*NPlotIters*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "NPlotIters"

..
    end-PlotResid-nplotiters

..
    start-PlotResid-ymax

*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits

..
    end-PlotResid-ymax

..
    start-PlotResid-plotoptions0

*PlotOptions0*: {``None``} | :class:`ResidPlot0Opts`
    plot options for initial residual

..
    end-PlotResid-plotoptions0

..
    start-PlotResid-ymin

*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"

..
    end-PlotResid-ymin

..
    start-PlotResid-ylabel

*YLabel*: {``None``} | :class:`str`
    manual label for y-axis

..
    end-PlotResid-ylabel

..
    start-PlotResid-xlabel

*XLabel*: {``None``} | :class:`str`
    manual label for x-axis

..
    end-PlotResid-xlabel

..
    start-PlotResid-xticks

*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values

..
    end-PlotResid-xticks

..
    start-PlotResid-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-PlotResid-alignment

..
    start-PlotResid-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-PlotResid-position

..
    start-PlotResid-xticklabeloptions

*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels

..
    end-PlotResid-xticklabeloptions

..
    start-PlotResid-dpi

*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image

..
    end-PlotResid-dpi

..
    start-PlotResid-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-PlotResid-type

..
    start-PlotResid-restriction

*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure

..
    end-PlotResid-restriction

..
    start-PlotResid-ylimmax

*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits

..
    end-PlotResid-ylimmax

..
    start-PlotResid-ticks

*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"

..
    end-PlotResid-ticks

..
    start-PlotResid-ylim

*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis

..
    end-PlotResid-ylim

..
    start-PlotResid-ticklabeloptions

*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes

..
    end-PlotResid-ticklabeloptions

..
    start-PlotResid-xlim

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis

..
    end-PlotResid-xlim

..
    start-PlotResid-figureheight

*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches

..
    end-PlotResid-figureheight

..
    start-PlotResid-xticklabels

*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-PlotResid-xticklabels

..
    start-PlotResid-format

*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format

..
    end-PlotResid-format

..
    start-PlotResid-restrictionloc

*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text

..
    end-PlotResid-restrictionloc

Summary Options
---------------
..
    start-Summary-iteration

*Iteration*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    specific iteration at which to sample results

..
    end-Summary-iteration

..
    start-Summary-coefficients

*Coefficients*: {``None``} | :class:`list`\ [:class:`str`]
    list of coefficients to detail in table

..
    end-Summary-coefficients

..
    start-Summary-cn

*CN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CN"

..
    end-Summary-cn

..
    start-Summary-header

*Header*: {``''``} | :class:`str`
    subfigure header

..
    end-Summary-header

..
    start-Summary-epsformat

*EpsFormat*: {``None``} | :class:`str`
    printf-style text format for sampling error

..
    end-Summary-epsformat

..
    start-Summary-cy

*CY*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CY"

..
    end-Summary-cy

..
    start-Summary-cll

*CLL*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLL"

..
    end-Summary-cll

..
    start-Summary-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-Summary-alignment

..
    start-Summary-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-Summary-caption

..
    start-Summary-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-Summary-position

..
    start-Summary-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-Summary-width

..
    start-Summary-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-Summary-type

..
    start-Summary-sigmaformat

*SigmaFormat*: {``None``} | :class:`str`
    printf-sylte text format for standard deviation

..
    end-Summary-sigmaformat

..
    start-Summary-clm

*CLM*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLM"

..
    end-Summary-clm

..
    start-Summary-components

*Components*: {``None``} | :class:`list`\ [:class:`str`]
    list of components for which to report coefficients

..
    end-Summary-components

..
    start-Summary-ca

*CA*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CA"

..
    end-Summary-ca

..
    start-Summary-cln

*CLN*: {``['mu', 'std']``} | :class:`list`\ [:class:`str`]
    value of option "CLN"

..
    end-Summary-cln

..
    start-Summary-muformat

*MuFormat*: {``None``} | :class:`str`
    printf-style text format for mean value

..
    end-Summary-muformat

SweepCases Options
------------------
..
    start-SweepCases-header

*Header*: {``''``} | :class:`str`
    subfigure header

..
    end-SweepCases-header

..
    start-SweepCases-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-SweepCases-alignment

..
    start-SweepCases-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-SweepCases-caption

..
    start-SweepCases-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-SweepCases-position

..
    start-SweepCases-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-SweepCases-width

..
    start-SweepCases-skipvars

*SkipVars*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys to leave out of table

..
    end-SweepCases-skipvars

..
    start-SweepCases-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-SweepCases-type

..
    start-SweepCases-specialvars

*SpecialVars*: {``None``} | :class:`list`\ [:class:`str`]
    keys not in run matrix to attempt to calculate

..
    end-SweepCases-specialvars

SweepCoeff Options
------------------
..
    start-SweepCoeff-minmaxoptions

*MinMaxOptions*: {``None``} | :class:`PlotCoeffSweepMinMaxPlotOpts`
    plot options for *MinMax* plot

..
    end-SweepCoeff-minmaxoptions

..
    start-SweepCoeff-coefficient

*Coefficient*: {``None``} | :class:`object`
    value of option "Coefficient"

..
    end-SweepCoeff-coefficient

..
    start-SweepCoeff-target

*Target*: {``None``} | :class:`str`
    name of target databook to co-plot

..
    end-SweepCoeff-target

..
    start-SweepCoeff-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-SweepCoeff-width

..
    start-SweepCoeff-nplotfirst

*NPlotFirst*: {``None``} | :class:`object`
    iteration at which to start figure

..
    end-SweepCoeff-nplotfirst

..
    start-SweepCoeff-plotoptions

*PlotOptions*: {``None``} | :class:`PlotCoeffSweepPlotOpts`
    options for main line(s) of plot

..
    end-SweepCoeff-plotoptions

..
    start-SweepCoeff-xlimmax

*XLimMax*: {``None``} | :class:`object`
    outer limits for min and max x-axis limits

..
    end-SweepCoeff-xlimmax

..
    start-SweepCoeff-ylabeloptions

*YLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis label

..
    end-SweepCoeff-ylabeloptions

..
    start-SweepCoeff-xmax

*XMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for x-axis limits

..
    end-SweepCoeff-xmax

..
    start-SweepCoeff-restrictionoptions

*RestrictionOptions*: {``{}``} | :class:`dict`
    additional opts to ``text()`` for restriction

..
    end-SweepCoeff-restrictionoptions

..
    start-SweepCoeff-xmin

*XMin*: {``None``} | :class:`float` | :class:`float32`
    explicit lower limit for x-axis limits

..
    end-SweepCoeff-xmin

..
    start-SweepCoeff-yticklabeloptions

*YTickLabelOptions*: {``None``} | :class:`dict`
    text options for y-axis tick labels

..
    end-SweepCoeff-yticklabeloptions

..
    start-SweepCoeff-component

*Component*: {``None``} | :class:`object`
    value of option "Component"

..
    end-SweepCoeff-component

..
    start-SweepCoeff-targetoptions

*TargetOptions*: {``None``} | :class:`PlotCoeffSweepTargetPlotOpts`
    plot options for optional target

..
    end-SweepCoeff-targetoptions

..
    start-SweepCoeff-figurewidth

*FigureWidth*: {``6``} | :class:`float` | :class:`float32`
    width of subfigure graphics in inches

..
    end-SweepCoeff-figurewidth

..
    start-SweepCoeff-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-SweepCoeff-caption

..
    start-SweepCoeff-restrictionxposition

*RestrictionXPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit x-coord of restriction

..
    end-SweepCoeff-restrictionxposition

..
    start-SweepCoeff-yticklabels

*YTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-SweepCoeff-yticklabels

..
    start-SweepCoeff-ticklabels

*TickLabels*: {``None``} | :class:`bool` | :class:`bool_`
    common value(s) for ticks of both axes

..
    end-SweepCoeff-ticklabels

..
    start-SweepCoeff-ksigma

*KSigma*: {``None``} | :class:`object`
    value of option "KSigma"

..
    end-SweepCoeff-ksigma

..
    start-SweepCoeff-restrictionyposition

*RestrictionYPosition*: {``None``} | :class:`float` | :class:`float32`
    explicit y-coord of restriction

..
    end-SweepCoeff-restrictionyposition

..
    start-SweepCoeff-minmax

*MinMax*: {``False``} | :class:`bool` | :class:`bool_`
    option to plot min/max of value over iterative window

..
    end-SweepCoeff-minmax

..
    start-SweepCoeff-xlabeloptions

*XLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis label

..
    end-SweepCoeff-xlabeloptions

..
    start-SweepCoeff-yticks

*YTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off y-axis ticks or set values

..
    end-SweepCoeff-yticks

..
    start-SweepCoeff-ymax

*YMax*: {``None``} | :class:`float` | :class:`float32`
    explicit upper limit for y-axis limits

..
    end-SweepCoeff-ymax

..
    start-SweepCoeff-ymin

*Ymin*: {``None``} | :class:`object`
    value of option "Ymin"

..
    end-SweepCoeff-ymin

..
    start-SweepCoeff-ylabel

*YLabel*: {``None``} | :class:`str`
    manual label for y-axis

..
    end-SweepCoeff-ylabel

..
    start-SweepCoeff-xlabel

*XLabel*: {``None``} | :class:`str`
    manual label for x-axis

..
    end-SweepCoeff-xlabel

..
    start-SweepCoeff-xticks

*XTicks*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis ticks or set values

..
    end-SweepCoeff-xticks

..
    start-SweepCoeff-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-SweepCoeff-alignment

..
    start-SweepCoeff-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-SweepCoeff-position

..
    start-SweepCoeff-xticklabeloptions

*XTickLabelOptions*: {``None``} | :class:`dict`
    text options for x-axis tick labels

..
    end-SweepCoeff-xticklabeloptions

..
    start-SweepCoeff-dpi

*DPI*: {``150``} | :class:`int` | :class:`int32` | :class:`int64`
    dots per inch if saving as rasterized image

..
    end-SweepCoeff-dpi

..
    start-SweepCoeff-sigmaplotoptions

*SigmaPlotOptions*: {``None``} | :class:`object`
    value of option "SigmaPlotOptions"

..
    end-SweepCoeff-sigmaplotoptions

..
    start-SweepCoeff-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-SweepCoeff-type

..
    start-SweepCoeff-restriction

*Restriction*: {``''``} | :class:`str`
    data restriction to place on figure

..
    end-SweepCoeff-restriction

..
    start-SweepCoeff-ylimmax

*YLimMax*: {``None``} | :class:`object`
    outer limits for min and max y-axis limits

..
    end-SweepCoeff-ylimmax

..
    start-SweepCoeff-ticks

*Ticks*: {``None``} | :class:`bool` | :class:`bool_`
    value of option "Ticks"

..
    end-SweepCoeff-ticks

..
    start-SweepCoeff-ylim

*YLim*: {``None``} | :class:`object`
    explicit min and max limits for y-axis

..
    end-SweepCoeff-ylim

..
    start-SweepCoeff-ticklabeloptions

*TickLabelOptions*: {``None``} | :class:`dict`
    common options for ticks of both axes

..
    end-SweepCoeff-ticklabeloptions

..
    start-SweepCoeff-xlim

*XLim*: {``None``} | :class:`object`
    explicit min and max limits for x-axis

..
    end-SweepCoeff-xlim

..
    start-SweepCoeff-figureheight

*FigureHeight*: {``4.5``} | :class:`float` | :class:`float32`
    height of subfigure graphics in inches

..
    end-SweepCoeff-figureheight

..
    start-SweepCoeff-xticklabels

*XTickLabels*: {``None``} | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`bool` | :class:`bool_`
    option to turn off x-axis tick labels or set values

..
    end-SweepCoeff-xticklabels

..
    start-SweepCoeff-format

*Format*: {``'pdf'``} | ``'svg'`` | ``'png'`` | ``'jpg'`` | ``'jpeg'``
    image file format

..
    end-SweepCoeff-format

..
    start-SweepCoeff-restrictionloc

*RestrictionLoc*: ``'bottom'`` | ``'bottom left'`` | ``'bottom right'`` | ``'left'`` | ``'lower right'`` | ``'lower left'`` | ``'right'`` | {``'top'``} | ``'top left'`` | ``'top right'`` | ``'upper left'`` | ``'upper right'``
    location for subfigure restriction text

..
    end-SweepCoeff-restrictionloc

SweepConditions Options
-----------------------
..
    start-SweepConditions-header

*Header*: {``''``} | :class:`str`
    subfigure header

..
    end-SweepConditions-header

..
    start-SweepConditions-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-SweepConditions-alignment

..
    start-SweepConditions-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-SweepConditions-caption

..
    start-SweepConditions-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-SweepConditions-position

..
    start-SweepConditions-width

*Width*: {``None``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-SweepConditions-width

..
    start-SweepConditions-skipvars

*SkipVars*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys to leave out of table

..
    end-SweepConditions-skipvars

..
    start-SweepConditions-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-SweepConditions-type

..
    start-SweepConditions-specialvars

*SpecialVars*: {``None``} | :class:`list`\ [:class:`str`]
    keys not in run matrix to attempt to calculate

..
    end-SweepConditions-specialvars

Tecplot Options
---------------
..
    start-Tecplot-keys

*Keys*: {``None``} | :class:`dict`
    dict of Tecplot layout statements to customize

..
    end-Tecplot-keys

..
    start-Tecplot-fieldmap

*FieldMap*: {``None``} | :class:`list`\ [:class:`int` | :class:`int32` | :class:`int64`]
    list of zone numbers for Tecplot layout group boundaries

..
    end-Tecplot-fieldmap

..
    start-Tecplot-alignment

*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"

..
    end-Tecplot-alignment

..
    start-Tecplot-colormaps

*ColorMaps*: {``[]``} | :class:`list`\ [:class:`dict`]
    customized Tecplot colormap

..
    end-Tecplot-colormaps

..
    start-Tecplot-caption

*Caption*: {``None``} | :class:`str`
    subfigure caption

..
    end-Tecplot-caption

..
    start-Tecplot-position

*Position*: ``'t'`` | ``'c'`` | {``'b'``}
    subfigure vertical alignment

..
    end-Tecplot-position

..
    start-Tecplot-width

*Width*: {``0.5``} | :class:`float` | :class:`float32`
    value of option "Width"

..
    end-Tecplot-width

..
    start-Tecplot-type

*Type*: {``None``} | :class:`str`
    subfigure type or parent

..
    end-Tecplot-type

..
    start-Tecplot-figwidth

*FigWidth*: {``1024``} | :class:`int` | :class:`int32` | :class:`int64`
    width of output image in pixels

..
    end-Tecplot-figwidth

..
    start-Tecplot-contourlevels

*ContourLevels*: {``None``} | :class:`list`\ [:class:`dict`]
    customized settings for Tecplot contour levels

..
    end-Tecplot-contourlevels

..
    start-Tecplot-varset

*VarSet*: {``{}``} | :class:`dict`
    variables and their values to define in Tecplot layout

..
    end-Tecplot-varset

..
    start-Tecplot-layout

*Layout*: {``None``} | :class:`str`
    template Tecplot layout file

..
    end-Tecplot-layout

Sweeps Options
==============
Sweep Options
-------------
..
    start-Sweep-eqcons

*EqCons*: {``None``} | :class:`list`\ [:class:`str`]
    list of run matrix keys that must be constant on a sweep

..
    end-Sweep-eqcons

..
    start-Sweep-figures

*Figures*: {``None``} | :class:`list`\ [:class:`str`]
    list of figures in sweep report

..
    end-Sweep-figures

..
    start-Sweep-xcol

*XCol*: {``None``} | :class:`str`
    run matrix key to use for *x*-axis of sweep plots

..
    end-Sweep-xcol

..
    start-Sweep-runmatrixonly

*RunMatrixOnly*: {``False``} | :class:`bool` | :class:`bool_`
    option to restrict databook to current run matrix

..
    end-Sweep-runmatrixonly

..
    start-Sweep-globalcons

*GlobalCons*: {``None``} | :class:`list`\ [:class:`str`]
    list of global constraints for sweep

..
    end-Sweep-globalcons

..
    start-Sweep-mincases

*MinCases*: {``3``} | :class:`int` | :class:`int32` | :class:`int64`
    minimum number of data points in a sweep to include plot

..
    end-Sweep-mincases

..
    start-Sweep-indices

*Indices*: {``None``} | :class:`list`\ [:class:`int` | :class:`int32` | :class:`int64`]
    explicit list of run matrix/databook indices to include

..
    end-Sweep-indices

..
    start-Sweep-ycol

*YCol*: {``None``} | :class:`str`
    run matrix key to use for *y*-axis of sweep contour plots

..
    end-Sweep-ycol

..
    start-Sweep-tolcons

*TolCons*: {``None``} | :class:`dict`
    tolerances for run matrix keys to be in same sweep

..
    end-Sweep-tolcons

..
    start-Sweep-carpeteqcons

*CarpetEqCons*: {``None``} | :class:`list`\ [:class:`str`]
    run matrix keys that are constant on carpet subsweep

..
    end-Sweep-carpeteqcons

..
    start-Sweep-indextol

*IndexTol*: {``None``} | :class:`int` | :class:`int32` | :class:`int64`
    max delta of run matrix/databook index for single sweep

..
    end-Sweep-indextol

