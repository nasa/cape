

.. _cape-json-ReportSweepCoeff:

Sweep Plots of Coefficients
----------------------------------------

The ``"SweepCoeff"`` class of subfigure is used to plot one or more force or
moment coefficients (or other database coefficient) for a sweep of cases. For
example, it can be used to plot normal force coefficient versus Mach number for
cases having the same angle of attack and sideslip angle. If the sweep
including this subfigure have *CarpetEqCons* or *CarpetTolCons* specified, the
plots on each page (i.e. for each sweep) will be divided into multiple lines as
divided by the carpet constraints.

    *S*: :class:`dict`
        Dictionary of settings for *SweepCoeff* subfigure
        
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
            
        *FigWidth*: {``6.0``} | :class:`float`
            Width of figure internally to Python; affects aspect ratio of
            figure and font size when integrated into report; decrease this
            parameter to make text appear larger in report
            
        *FigHeight*: {``4.5``} | :class:`float`
            Similar to *FigWidth* and primarily used to set aspect ratio
        
        *Component*: {``"entire"``} | :class:`str` | :class:`list`
            Component or list of components to plot, must be name(s) of
            components defined in :file:`Config.xml`
            
        *Coefficient*: ``"CA"`` | ``"CY"`` | {``"CN"``} | ``"CLL"`` 
        | ``"CLM"`` | ``"CLN"`` | :class:`list`
        
            Force or moment coefficient(s) to plot, any database value
            
        *StandardDeviation*: {``0.0``} | :class:`float`
            If nonzero, plot the value *StandardDeviation* above and below the
            mean value at each point
            
        *MinMax*: ``true`` | {``false``}
            Whether or not to plot minimum and maximum value over iterative
            history
            
        *Format*: {``"pdf"``} | ``"svg"`` | ``"png"`` | :class:`str`
            Format of graphic file to save
            
        *DPI*: {``150``} | :class:`int`
            Resolution (dots per inch) if saved as a raster format
            
        *LineOptions*: {``{"color": "k", "marker": ["^","s","o"]}``} |
        :class:`dict`
        
            Plot options for the plot value; options are passed to
            :func:`matplotlib.pyplot.plot`, and lists are cycled through, so
            the default plots the first line with a ``"^"`` marker, etc.
            
        *TargetOptions*: {``{"color": "r", "marker": ["^","s","o"]}``} |
        :class:`dict`
        
            Plot options for target value plot if the data book contains a
            target for the current component and coefficient
            
        *MinMaxOptions*: {``{"facecolor":"g", "alpha":0.4, "lw":0}``} |
        :class:`dict`
        
            Options for plot of min/max value over iterative window, passed to
            :func:`matplotlib.pyplot.fill_between`
            
        *StDevOptions*: {``{"facecolor":"b", "alpha":0.35, "lw":0}``} |
        :class:`dict`
        
            Plot options for standard deviation plots; options are passed to
            :func:`matplotlib.pyplot.fill_between`
            
        *PlotTypeMinMax*: {``"FillBetween"``} | ``"ErrorBar"``
            Method for plotting range of values for min/max values
            
        *PlotTypeUncertainty*: ``"FillBetween"`` | {``"ErrorBar"``}
            Method for plotting range of values for sampling error
            
        *PlotTypeStDev*: {``"FillBetween"``} | ``"ErrorBar"``
            Method for plotting range of values for standard deviation plot
            
        *ShowMu*: {``True``} | ``False``
            Prints value of mean value in upper right corner of plot
            
        *ShowSigma*: ``True`` | {``False``}
            Prints value of standard deviation in upper left corner
            
        *ShowDelta*: ``True`` | {``False``}
            Prints value of fixed width in upper right corner
            
        *MuFormat*: {``"%.4f"``} | :class:`str`
            Format flag for value of mean printed via *ShowMu*
            
        *SigmaFormat*: {``"%.4f"``} | :class:`str`
            Format flag for value of mean printed via *ShowSigma*
            
        *DeltaFormat*: {``"%.4f"``} | :class:`str`
            Format flag for value of mean printed via *ShowDelta*

