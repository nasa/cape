
        
Figure Definitions
------------------
Figures are essentially a collection of subfigures.  They are used to divide
the report into aesthetically pleasing groups and to make sure the subfigures
fit onto pages appropriately.

    .. code-block:: javascript
    
        "Figures": {
            // Figure to show table of case parameters and integrated FM
            "CaseSummary": {
                "Alignment": "left",
                "Subfigures": ["CaseConds", "TableFM"]
            },
            // Figure to show iterative histories
            "IterFM": {
                "Header": "Iterative Force, Moment, and Residual Histories",
                "Alignment": "center",
                "Subfigures": [
                    "body_CA", "body_CY", "body_CN",
                    "body_CLL", "body_CLN", "body_CLN",
                    "protub1_CA", "protub2_CA", "L2"
                ]
            }
        }
        
Each figure primarily consists of the list of subfigures, but there are two
additional options.  The ``"Alignment"`` option simply tells LaTeX how to align
the subfigures when the subfigure widths do not add up to the full text width.
The ``"Header"`` specifies text that is printed as bold italic text immediately
above the set of subfigures.

Further descriptions of the available can be found in the :ref:`JSON "Figures"
section <cape-json-ReportFigure>`.


Subfigure Definitions
---------------------
A "subfigure" is essentially an image, sizing information, and a caption.  Each
subfigure is defined by a dictionary in the ``"Subfigures"`` subsection of
``"Report"`` in a JSON file.  There are many types of subfigures, but there are
several options that are common to each.  In addition, the notion of cascading
options applies to all subfigures and can be a time-saving measure.

    .. code-block:: javascript
    
        "Subfigures": {
            // Example showing generic options
            "body": {
                "Type": "PlotCoeff",
                "Width": 0.33,
                "FigWidth": 5.5,
                "FigHeight": 4.25,
                "Caption": "Forces and moments on \\texttt{body}"
            },
            // Example showing cascading options
            "body_CA": {
                "Type": "body",
                "Caption": "body/CA",
                "Coefficient": "CA"
            }
        }

Each subfigure must have a ``"Type"``, which must either be from a list of
defined subfigure types or the name of another subfigure.  The ``"Width"``
parameter specifies the width of the image within the PDF document as a
fraction of the text region of the PDF document (i.e. the ``\textwidth``
parameter from LaTeX).  The ``"Caption"`` parameter is also available to all
subfigures, although many subfigures will provide a default caption.

This example shows how an iterative force and moment plot called ``"body_CA"``
can be defined based on a template ``"body"``.  It is very likely that any user
wanting to see a plot of *CA* may also want to see a plot of *CY* or *CN*.  The
ability to cascade options by using some of the options from ``"body"`` reduces
the amount of lines needed to define the series of figures and also decreases
how much work is needed to modify settings for the whole set of subfigures.

A much more in-depth description of the available options for each type of
subfigure can be found in the :ref:`JSON "Subfigure" section
<cape-json-ReportSubfigure>`.
