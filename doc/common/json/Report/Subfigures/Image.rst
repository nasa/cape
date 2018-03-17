    

.. _cape-json-ReportImage:
            
Insert Image Subfigure Type
----------------------------
the simplest type of subfigure simply copies an image and adds it to a report.

The full list of options is shown below.

    *P*: :class:`dict`
        Dictionary of settings for *Image* subfigures
        
        *Type*: {``"Image"``} | :class:`str`
            Subfigure type
            
        *Header*: {``""``} | :class:`str`
            Heading placed above subfigure (bold, italic)
            
        *Position*: ``"t"`` | ``"c"`` | {``"b"``}
            Vertical alignment of subfigure
            
        *Alignment*: ``"left"`` | {``"center"``}
            Horizontal alignment
            
        *Width*: {``0.5``} | :class:`float`
            Width of subfigure as a fraction of page text width
            
        *Image*: {``"export.png"``} | :class:`str`
            Name of file relative to solution folder; this file should exist in
            each run folder for best results

