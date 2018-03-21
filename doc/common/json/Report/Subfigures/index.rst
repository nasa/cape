
                
.. _cape-json-ReportSubfigure:

Subfigure Definitions
=====================

Each subfigure contains several key options including heading and caption and
two alignment options.  The key option is *Type*, which categorizes which kind
of subfigure is being generated, and it must be traceable to one of several
defined subfigure types.

The list of options common to each subfigure is shown below.

    *Subfigures*: {``{}``} | ``{sfig: U}`` | :class:`dict` (:class:`dict`)
        Dictionary of subfigure definitions
        
        *sfig*: :class:`str`
            Name of subfigure
            
        *U*: :class:`dict`
            Dictionary of settings for subfigure *sfig*
            
            *Type*: ``"Conditions"`` | ``"SweepConditions"`` |
            ``"SweepCases"`` | ``"Summary"`` | ``"PlotCoeff"`` |
            ``"PlotLineLoad"`` | ``"SweepCoeff"`` |
            ``"SweepCoeffHist"`` | ``"ContourCoeff"`` |
            ``"PlotL1"`` | ``"PlotL2"`` | ``"PlotResid"`` |
            ``"Tecplot3View"`` | ``"Tecplot"`` | ``"Paraview"`` |
            ``"Image"`` | :class:`str`
                    
                Subfigure type
            
            *Header*: {``""``} | :class:`str`
                Heading to be placed above the subfigure (bold, italic)
                
            *Caption*: {``""``} | :class:`str`
                Caption to be placed below figure
                
            *Position*: {``"t"``} | ``"c"`` | ``"b"``
                Vertical alignment of subfigure; top or bottom
                
            *Alignment*: ``"left"`` | {``"center"``}
                Horizontal alignment of subfigure
                
            *Width*: :class:`float`
                Width of subfigure as a fraction of text width
                
            *Grid*: {``None``} | ``True`` | ``False``
                Whether or not to plot major grid on 2D subfigure plots
                
            *GridStyle*: {``{}``} | :class:`dict`
                PyPlot formatting options for major grid on 2D plots
                
            *MinorGrid*: {``None``} | ``True`` | ``False``
                Whether or not to plot minor grid on 2D subfigure plots
                
            *MinorGridStyle*: {``{}``} | :class:`dict`
                PyPlot formatting options for minor grid on 2D plots
                
            
However, the *Type* value does not always have to be from the list of possible
values above.  Another option is to define one subfigure and use that
subfigure's options as the basis for another one.  An example of this is below.

    .. code-block:: javascript
    
        "Subfigures": {
            "Wing": {
                "Type": "PlotCoeff",
                "Component": "wing",
            },
            "CN": {
                "Type": "Wing",
                "Coefficient": "CN"
            },
            "CLM": {
                "Type": "Wing",
                "Coefficient": "CLM"
            }
        }

This defines two coefficient plots, which both use the *Component* named 
``"wing"``.  When using a previous template subfigure is used as *Type*, all of
the options from that subfigure are used as defaults, which can save many lines
in the JSON file when there are several similar figures defined.

The subsections that follow describe options that correspond to options for
each base type of subfigure.

.. toctree::
    :maxdepth: 2
    
    Conditions
    SweepConditions
    SweepCases
    Summary
    PlotCoeff
    PlotL2
    SweepCoeff
    SweepCoeffHist
    ContourCoeff
    PlotLineLoad
    Tecplot
    Paraview
    Image
    
