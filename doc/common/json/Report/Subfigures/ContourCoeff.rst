

.. _cape-json-ReportContourCoeff:

Sweep Coefficient Contour Plots
--------------------------------
The ``"ContourCoeff"`` class of subfigure is used to create a contour plot of
unstructured 2D data for one or more force & moment or other database
coefficients (such as a ``TriqFM`` load).  The user must specify the *XAxis*
and *YAxis* of the contour plot; in most applications these are angle of
sidelsip and angle of attack, respectively.  Sweep constraints for
*CarpetEqCons* and *CarpetTolCons* are ignored.

    *S*: :class:`dict`
        Dictionary of settings for *ContourCoeff* subfigure
        
        *Type*: {``"ContourCoeff"``} | :class:`str`
            Subfigure type
            
        *ContourType*: {``"tricontourf"``} | ``"tricontour"`` | ``"tripcolor"``
            Contour plot, filled, not filled, or triangulated
        
        *LineType*: {``"plot"``} | ``"triplot"``
            Type for plotting CFD data points; ``"triplot"`` adds a
            triangulation of the CFD data
            
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
            Force or moment coefficient(s) to plot, any database value
            
        *XAxis*: ``"beta"`` | :class:`str`
            Name of trajectory key or *beta* to use as *x*-axis
            
        *XAxis*: ``"alpha"`` | :class:`str`
            Name of trajectory key or *alpha* to use as *y*-axis
            
        *ContourOptions*: {``{}``} | :class:`dict`
            Dictionary of plotting options to :func:`pyplot.tricontour`
            
        *LineOptions*: {``{"color": "k", "marker": "o"}``} | :class:`dict`
            Options for plotting actual values; default plots black dots
            
        *AxisEqual*: {``True``} | ``False``
            If ``True``, scale *x* and *y* axes to have same scale
            
        *ColorMap*: {``"jet"``} | :class:`str`
            Name of color map to use
        


