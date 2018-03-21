    

.. _cape-json-ReportPlotL2:
            
Iterative Residual History Subfigure
-------------------------------------
To plot iterative histories of residual histories, the user can select the
specific ``"PlotL1"`` type (for Cart3D), the ``"PlotL2"`` type (for most CFD
solvers), or the more user-controlled ``"PlotResid"`` subfigure type.

The full list of options is shown below.

    *P*: :class:`dict`
        Dictionary of settings for *PlotResid* subfigures
        
        *Type*: {``"PlotResid"``} | ``"PlotL1"`` | ``"PlotL2"`` | :class:`str`
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
            
        *Residual*: {``"R_1"``} | :class:`str`
            Name of other residual coefficient to plot
            
        *YLabel*: {``"L2 residual"``} | :class:`str`
            Axis label for *y*-axis

