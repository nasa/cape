

.. _cape-json-ReportSweepCoeffHist:

Database Coefficient Histogram Plots
----------------------------------------

The ``"SweepCoeffHist"`` class of subfigure is used to plot histograms or range
histograms of any database coefficient.  Users can create a histogram of the
raw data or a histogram of the deltas to another database.  A "range" histogram
is only defined when there is a *Target* database, and it plots a histogram of
the absolute values of the deltas.

    *S*: :class:`dict`
        Dictionary of settings for *SweepCoeffHist* subfigure
        
        *Type*: {``"SweepCoeffHist"``} | :class:`str`
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
            
        *Coefficient*: ``"CA"`` | ``"CY"`` | {``"CN"``} | :class:`str`
            Name of coefficient to plot, any database value
            
        *Target*: {``None``} | :class:`str`
            Name of target database to which to compare each point
        
        *TargetLabel*: :class:`str`
            Name of the target to use in legend
            
        *Format*: {``"pdf"``} | ``"svg"`` | ``"png"`` | :class:`str`
            Format of graphic file to save
            
        *DPI*: {``150``} | :class:`int`
            Resolution (dots per inch) if saved as a raster format
            
        *StandardDeviation*: {``3.0``} | :class:`float`
            If nonzero, plot the value *StandardDeviation* above and below the
            mean value on the histogram
            
        *OutlierSigma*: {``4.0``} | :class:`float`
            Multiple of standard deviation to use as filter for outlier data
        
        *Range*: {``4.0``} | ``None`` | :class:`float`
            Multiple of standard deviation; manually-specified plot range
            
        *Delta*: {``0.0``} | :class:`float`
            Fixed value to plot for scale reference between histograms
            
        *PlotMean*: {``True``} | ``False``
            Whether or not to plot vertical line
            
        *PlotGaussian*: ``True`` | {``False``}
            Whether or not to plot curve representing idealized normal
            distribution
            
        *HistOptions*: {``{"facecolor": "c", "bins": 20}``} | :class:`dict`
            Options passed to :func:`pyplot.hist`
            
        *MeanOptions*: {``{"color": "k", "lw": 2}``} | :class:`dict`
            Plot options for the vertical line of the mean value
            
        *DeltaOptions*: {``{"color": "r", "ls": "--"}``} | :class:`dict`
            Plot options for vertical fixed-range lines
            
        *GaussianOptions*: {``{"color":"navy", "lw": 1.5}``} | :class:`dict`
            Options for plot of ideal normal distribution
            
        *StDevOptions*: {``{"color":"b"}``} | :class:`dict`
            Plot options for vertical line showing multiple of standard
            deviation
            
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
            
        *SigmaLabel*: {``"σ(%s)" % coeff``} | ``"σ(Δ%s)"`` | :class:`unicode`
            Label for standard deviation printed using *ShowSigma*

