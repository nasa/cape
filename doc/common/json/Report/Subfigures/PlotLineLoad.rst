            
            
.. _cape-json-ReportPlotLineLoad:

Single-Case Line Load Plots
------------------------------
The ``"PlotLineLoad"`` subfigure is used to plot one sectional load profile for
one component of one CFD solution and possibly compare it to another line load
from a target database.

    *S*: :class:`dict`
        Dictionary of settings for *PlotLineLoad* subfigure
        
        *Type*: {``"PlotLineLoad"``} | :class:`str`
            Subfigure type
            
        *Header*: {``""``} | :class:`str`
            Heading placed above subfigure (bold, italic)
            
        *Position*: ``"t"`` | ``"c"`` | {``"b"``}
            Vertical alignment of subfigure
            
        *Alignment*: ``"left"`` | {``"center"``}
            Horizontal alignment
            
        *Width*: {``0.5``} | :class:`float`
            Width of subfigure as a fraction of page text width
            
        *Component*: {``"entire"``} | :class:`str`
            DataBook component to plot
            
        *Coefficient*: ``"CA"`` | ``"CY"`` | {``"CN"``}
            Sectional load coefficient to plot
            
        *Format*: {``"pdf"``} | ``"svg"`` | ``"png"`` | :class:`str`
            Format of graphic file to save
            
        *DPI*: {``150``} | :class:`int`
            Resolution (dots per inch) if saved as a raster format
            
        *FigWidth*: {``6.0``} | :class:`float`
            Width of figure internally to Python; affects aspect ratio of
            figure and font size when integrated into report; decrease this
            parameter to make text appear larger in report
            
        *FigHeight*: {``4.5``} | :class:`float`
            Similar to *FigWidth* and primarily used to set aspect ratio
            
        *LineOptions*: {``{"color": "k"}``} | :class:`dict`
            Options for sectional load plot to :func:`pyplot.plot`
            
        *SeamOptions*: {*LineOptions*} | :class:`dict`
            Optional separate options for seam plot
            
        *TargetOptions*: {``{"color": "r", "zorder": 2}``} | :class:`dict`
            Plot options for target line load plot
            
        *SeamCurves*: {``"smy"``} | ``"smz"`` | ``None``
            Seam curve or list of seam curves to plot
        
        *SeamLocations*: {``None``} | ``"bottom"`` | ``"top"`` | ``"left"`` 
        | ``"right"`` | :class:`list`
            
            List of locations where to plot seam curves
            
        *AdjustLeft*: {``0.12``} | :class:`float`
            Manual adjustment of left edge of axes on figure
            
        *AdjustRight*: {``0.97``} | :class:`float`
            Manual adjustment of right edge of axes on figure
            
        *AdjustBottom*: {``0.1``} | :class:`float`
            Manual adjustment of lower edge of axes on figure
            
        *AdjustTop*: {``0.97``} | :class:`float`
            Manual adjustment of upper edge of axes on figure
        
        *SubplotMargin*: {``0.015``} | :class:`float`
            Margin between seam curve and data subplots
            
        *XPad*: {``0.03``} | :class:`float`
            Extra padding of *x*-axis to data range
            
        *YPad*: {``0.03``} | :class:`float`
            Extra padding of *y*-axis to data range

