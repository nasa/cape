
--------------------------------------
Automated Report Generation with LaTeX
--------------------------------------

The automated report syntax is almost entirely general to all Cape solvers, and
so most of the documentation is in the :ref:`Cape Report page
<cape-json-Report>`.  Some sample JSON syntax is provided below.

The section in :file:`pyCart.json` labeled "Report" is for generating automated
reports of results.  It requires a fairly complete installation of `pdfLaTeX`.
Further, an installation of Tecplot 360 enhances the capability of the report
generation.

    .. code-block:: javascript
    
        "Report": {
            "Archive": false,
            "Reports": ["case"],
            "case": {
                "Title": "Automated Cart3D Report",
                "Figures": ["Summary", "Forces"],
                "FailFigures": ["Summary", "Surface"],
                "ZeroFigures": ["Summary", "Surface"]
            },
            "Figures": {
                "Summary": {
                    "Alignment": "left",
                    "Subfigures": ["Conditions", "Summary"]
                },
                "Forces": {
                    "Alignment": "center",
                    "Header": "Force, moment, \\& residual histories",
                    "Subfigures": ["CA", "CY", "CN", "L1"]
                }
            },
            "Subfigures": {
                "Conditions": {
                    "Type": "Conditions",
                    "Alignment": "left",
                    "Width": 0.35,
                    "SkipVars": []
                },
                "Summary": {
                    "Type": "Summary"
                },
                "Surface": {
                    "Type": "Tecplot",
                    "Layout": "surface.lay"
                },
                "wingFM": {
                    "Type": "PlotCoeff",
                    "Component": "CA",
                    "Width": 0.5,
                    "StandardDeviation": 1.0, 
                    "nStats": 200
                },
                "CA": {"Type": "wingFM" "Coefficient": "CA"},
                "CY": {"Type": "wingFM" "Coefficient": "CY"},
                "CN": {"Type": "wingFM" "Coefficient": "CN"},
                "L1": {"Type": "PlotL1"}
            }
        }
        
The :ref:`basic report definitions <cape-json-ReportReport>`, :ref:`sweep
definitions <cape-json-ReportSweep>`, and :ref:`figure definitions
<cape-json-ReportFigure>` are completely unchanged. While most of the content of
the :ref:`subfigure definition section <cape-json-ReportSubfigure>` also
applies, there are some additional specific capabilities.

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
            ``"SweepCoeff"`` | ``"PlotL1"`` | ``"Tecplot3View"`` |
            ``"Tecplot"`` | :class:`str`
                    
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

The subsections that follow describe options that correspond to options for each
base type of subfigure.

.. _pyCart-json-ReportTecplot:

Tecplot Layout Figure
---------------------
The capability to create an image using a Tecplot layout file is provided by the
``"Tecplot"`` subfigure type.  It has pretty generic subfigure options except
that it has an additional parameter ``"Mach"`` that pyCart needs to know in
order to calculate pressure coefficient.  The ``plt`` files created by Cart3D do
not include the freestream Mach number, so pyCart needs to read it from the run
matrix.

    *P*: :class:`dict`
        Dictionary of settings for *Tecplot* subfigure
        
        *Type*: {``"Tecplot"``} | :class:`str`
            Subfigure type
            
        *Header*: {``""``} | :class:`str`
            Heading placed above subfigure (bold, italic)
            
        *Position*: ``"t"`` | ``"c"`` | {``"b"``}
            Vertical alignment of subfigure
            
        *Alignment*: ``"left"`` | {``"center"``}
            Horizontal alignment
            
        *Width*: {``0.5``} | :class:`float`
            Width of subfigure as a fraction of page text width
            
        *Caption*: {``""``} | :class:`str`
            Caption text
            
        *Layout*: {``"layout.lay"``} | :class:`str`
            Name of Tecplot layout file
            
        *Mach*: {``"mach"``} | :class:`str`
            Name of trajectory key that determines freestream Mach number


.. _pyCart-json-ReportTecplot3View:

Tecplot 3-View Figure
---------------------
The pyCart report also contains a special type of subfigure that shows a 3-view
for a specific component.  The component can be either a name (as defined in the
config file) or a component ID number.

    *P*: :class:`dict`
        Dictionary of settings for *Tecplot* subfigure
        
        *Type*: {``"Tecplot3View"``} | :class:`str`
            Subfigure type
            
        *Header*: {``""``} | :class:`str`
            Heading placed above subfigure (bold, italic)
            
        *Position*: ``"t"`` | ``"c"`` | {``"b"``}
            Vertical alignment of subfigure
            
        *Alignment*: ``"left"`` | {``"center"``}
            Horizontal alignment
            
        *Width*: {``0.66``} | :class:`float`
            Width of subfigure as a fraction of page text width
            
        *Component*: {``"entire"``} | :class:`str` | :class:`int`
            Name or number of component to show

            
.. _pyCart-json-ReportParaview:

Paraview Figure
----------------
A capability similar to the Tecplot layout is provided for the Paraview
visualization software.  This is a somewhat different capability but has many
similarities.  For some typical installations of Paraview, reading ``plt`` files
is not supported, so Cart3D must be called with the *tecIO* option turned off.

    *P*: :class:`dict`
        Dictionary of settings for *Tecplot* subfigure
        
        *Type*: {``"Paraview"``} | :class:`str`
            Subfigure type
            
        *Header*: {``""``} | :class:`str`
            Heading placed above subfigure (bold, italic)
            
        *Position*: ``"t"`` | ``"c"`` | {``"b"``}
            Vertical alignment of subfigure
            
        *Alignment*: ``"left"`` | {``"center"``}
            Horizontal alignment
            
        *Width*: {``0.5``} | :class:`float`
            Width of subfigure as a fraction of page text width
            
        *Caption*: {``""``} | :class:`str`
            Caption text
            
        *Layout*: {``"layout.py"``} | :class:`str`
            Name of Paraview python script
            
        *ImageFile*: {``"export.png"``} | :class:`str`
            Name of file produced by the Paraview python script
            
        *Format*: {``"png"``} | :class:`str`
            Format of the image produced by the Paraview python script
            
        *Mach*: {``"mach"``} | :class:`str`
            Name of trajectory key that determines freestream Mach number
            
        *Command*: {``"pvpython"``} | :class:`str`
            Command-line command used to run Python script
        
