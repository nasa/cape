

.. _cape-json-ReportFigure:

Figure Definitions
==================

Each figure contains a small number of options used to define the figure.  The
primary option is a list of subfigures, and the others are also defined below.

    *Figures*: {``{}``} | ``{fig: F}`` | :class:`dict` (:class:`dict`)
        Dictionary of figure definitions
        
        *fig*: :class:`str`
            Name of figure
        
        *F*: :class:`dict`
            Dictionary of settings for figure called ``"F"``
            
            *Header*: :class:`str`
                Title to be placed at the top of the figure
                
            *Alignment*: {``"left"``} | ``"center"`` | ``"right"``
                Horizontal alignment for the figure
                
            *Subfigures*: ``[]`` | :class:`list` (:class:`str`)
                List of subfigures


