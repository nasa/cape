        
            
.. _cape-json-ReportSweepCases:

List of Cases in a Sweep
------------------------
The ``"SweepCases"`` subfigure class, which is only available for sweeps, shows
the list of cases in a sweep. The method of displaying the sweeps is to list
the names of each case, i.e. the group/case folder name, in some monospace
format. The list of options is below.

    *C*: :class:`dict`
        Dictionary of settings for *SweepCases* type subfigure
        
        *Type*: {``"SweepCases"``} | :class:`str`
            Subfigure type
            
        *Header*: {``"Sweep Cases"``} | :class:`str`
            Heading placed above subfigure (bold, italic)
        
        *Position*: {``"t"``} | ``"c"`` | ``"b"``
            Vertical alignment of subfigure
            
        *Alignment*: {``"left"``} | ``"center"``
            Horizontal alignment
            
        *Width*: {``0.6``} | :class:`float`
            Width of subfigure as a fraction of page text width

