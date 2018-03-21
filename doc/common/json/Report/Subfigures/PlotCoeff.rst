    

.. _cape-json-ReportPlotCoeff:
            
Iterative Force or Moment Plot
------------------------------
To plot iterative histories of force and/or moment coefficients on one or more
component, use the ``"PlotCoeff"`` subfigure. There are many options for this
class of subfigure. In addition to standard alignment and caption options,
there are options for which component(s) and coefficient(s) to plot, options
for how the plots are presented, which iterations to use, output format, and
figure sizes.

The default caption, which is placed in sans-serif font below the figure, is
*Component*/*Coefficient*, which may be confusing if two components are
included.  For example, a caption such as ``"[LeftWing, RightWing]/CY"`` could
be generated automatically.

The full list of options is shown below.

    *P*: :class:`dict`
        Dictionary of settings for *PlotCoeff* subfigures
        
        *Type*: {``"PlotCoeff"``} | :class:`str`
            Subfigure type
            
        *Header*: {``""``} | :class:`str`
            Heading placed above subfigure (bold, italic)
            
        *Position*: ``"t"`` | ``"c"`` | {``"b"``}
            Vertical alignment of subfigure
            
        *Alignment*: ``"left"`` | {``"center"``}
            Horizontal alignment
            
        *Width*: {``0.5``} | :class:`float`
            Width of subfigure as a fraction of page text width
            
        *nPlotFirst*: {``0``} | :class:`int`
            First iteration to plot; often useful to eliminate startup
            transients from the plot which may have a much larger scale than the
            final value
            
        *nPlotLast*: {``null``} | :class:`int`
            If specified, only plot up to this iteration
            
        *nPlot*: {``null``} | :class:`int`
            If specified, plot at most this many iterations; alternative method
            to hide startup transients
            
        *nStats*: :class:`int`
            Number of iterations to use for statistics; defaults to data book
            option
            
        *nMinStats*: :class:`int`
            First iteration to allow to be used for mean calculation
            
        *nMaxStats*: :class:`int`
            Maximum number of iterations to allow to be used in statistics
            
        *FigWidth*: {``6.0``} | :class:`float`
            Width of figure internally to Python; affects aspect ratio of
            figure and font size when integrated into report; decrease this
            parameter to make text appear larger in report
            
        *FigHeight*: {``4.5``} | :class:`float`
            Similar to *FigWidth* and primarily used to set aspect ratio
        
        *Component*: {``"entire"``} | :class:`str` | :class:`list`
            Component or list of components to plot, must be name(s) of
            components defined in :file:`Config.xml`
            
        *Coefficient*: ``"CA"`` | ``"CY"`` | {``"CN"``} | ``"CLL"`` | ``"CLM"``
        | ``"CLN"`` | :class:`list`
        
            Force or moment coefficient(s) to plot, any of ``"CA"``
            
        *Delta*: {``0.0``} | :class:`float`
            If nonzero, plot a horizontal line this value above and below the
            iterative mean, by default with a dashed red line
            
        *StandardDeviation*: {``0.0``} | :class:`float`
            If nonzero, plot a rectangular box centered on the iterative mean
            value and spanning vertically above and below the mean this number
            times the iterative standard deviation; the width of the box shows
            the iteration window used to compute the statistics
            
        *IterativeError*: {``0.0``} | :class:`float`
            If nonzero, plot a rectangular box centered on the iterative mean
            value and spanning vertically above and below the mean this number
            times the iterative uncertainty; the width of the box shows
            the iteration window used to compute the statistics
            
        *ShowMu*: {``[true, false]``} | ``true`` | ``false`` | :class:`list`
            Whether or not to print the mean value in the upper right corner of
            the plot; by default show the value of the first
            component/coefficient and not for the others
            
        *ShowSigma*: {``[true, false]``} | ``true`` | ``false`` | :class:`list`
            Whether or not to print the standard deviation in the upper left
            
        *ShowDelta*: {``[true, false]``} | ``true`` | ``false`` | :class:`list`
            Whether or not to print the fixed delta value in the upper right
            
        *ShowEpsilon*: ``true`` | {``false``}
            Whether or not to print iterative uncertainty in upper left
            
        *Format*: {``"pdf"``} | ``"svg"`` | ``"png"`` | :class:`str`
            Format of graphic file to save
            
        *DPI*: {``150``} | :class:`int`
            Resolution (dots per inch) if saved as a raster format
            
        *LineOptions*: {``{"color":["k","g","c","m","b","r"]}``} | :class:`dict`
            Plot options for the primary iterative plot; options are passed to
            :func:`matplotlib.pyplot.plot`, and lists are cycled through, so the
            default plots the first history in black, the second in green, etc.
            
        *MeanOptions*: {``{"ls": null}``} | :class:`dict`
            Plot options for the iterative mean value; most options are
            inherited from *LineOptions*, and setting *ls* to ``None`` as in the
            default creates a dotted line that is dashed for the iterations used
            to compute the mean
            
        *StDevOptions*: {``{"facecolor": "b", "alpha": 0.35, "ls": "none"}``} |
        :class:`dict`
        
            Plot options for standard deviation plots; options are passed to
            :func:`matplotlib.pyplot.fill_between`
            
        *ErrPlotOptions*: {``{"facecolor": "g", "alpha": 0.4, "ls": "none"}``} |
        :class:`dict`
        
            Plot options for iterative uncertainty window, passed to
            :func:`matplotlib.pyplot.fill_between`
            
        *DeltaOptions*: {``{"color": null}``} | :class:`dict`
            Plot options for fixed interval plot, passed to
            :func:`matplotlib.pyplot.plot`
        
A typical application of this subfigure involves plotting multiple
coefficients, and it is often advantageous to define a new subfigure class and
allow most of the plotting options to "cascade." Consider the following example
used to define plots of the force coefficients on left and right wings of some
geometry.

    .. code-block:: javascript
    
        "Subfigures": {
            "WingCA": {
                "Type": "PlotCoeff",
                "Component": ["LeftWing", "RightWing"],
                "Coefficient": "CA",
                "LineOptions": {"color": ["k", "g"]},
                "StandardDeviation": 1.0
            },
            "WingCN": {
                "Type": "WingCA",
                "Coefficient": "CN"
            },
            "LeftWingCY": {
                "Type": "WingCA",
                "Coefficient": "CY",
                "Component": "LeftWing"
            },
            "RightWingCY": {
                "Type": "LeftWingCY",
                "Component": "RightWing",
                "LineOptions": {"color": "g"}
            }
            
The example creates four iterative history plots without having to repeat all
the options for each subfigure.  All four plots will use a *StandardDeviation*
value of ``1.0``.  Also note how multiple levels of recursion are allowed as
shown by the last subfigure which uses ``"LeftWingCY" --> "WingCA" -->
"PlotCoeff"`` for the *Type* specification.

