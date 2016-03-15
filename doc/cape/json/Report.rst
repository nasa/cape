
.. _cape-json-Report:

--------------------------------------
Automated Report Generation with LaTeX
--------------------------------------

The section in :file:`pyCart.json` labeled "Report" is for generating automated
reports of results.  It requires a fairly complete installation of `pdfLaTeX`.
Further, an installation of Tecplot 360 or ParaView enhances the capability of
the report generation.

    .. code-block:: javascript
    
        "Report": {
            "Archive": false,
            "Reports": ["case", "mach"],
            "case": {
                "Title": "Automated Cart3D Report",
                "Subtitle": "Forces, Moments, \\& Residuals",
                "Author": "Cape Developers",
                "Affiliation": "NASA Ames",
                "Logo": "NASA_logo.pdf",
                "Frontispiece": "NASA_logo.pdf",
                "Restriction": "Distribution Unlimited",
                "Figures": ["Summary", "Forces"],
                "FailFigures": ["Summary", "Surface"],
                "ZeroFigures": ["Summary", "Surface"]
            },
            "mach": {
                "Title": "Results for Mach Sweeps",
                "Sweeps": "mach"
            },
            "Sweeps": {
                "mach": {
                    "Figures": ["SweepCond", "SweepCoeff"],
                    "EqCons": ["alpha", "beta"],
                    "XAxis": "mach"
                }
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
                },
                "SweepCond": {
                    "Subfigures": ["SweepConds", "SweepCases"],
                },
                "SweepCoeff": {
                    "Subfigures": ["mach_CA", "mach_CN"],
                },
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
                "CA": {
                    "Type": "PlotCoeff",
                    "Component": "wing",
                    "Coefficient": "CA",
                    "Width": 0.5,
                    "StandardDeviation": 1.0, 
                    "nStats": 200
                },
                "CY": {"Type": "CA" "Coefficient": "CY"},
                "CN": {"Type": "CA" "Coefficient": "CN"},
                "L1": {"Type": "PlotL1"}
                "mach_CA": {
                    "Type": "SweepCoeff",
                    "Width": 0.5,
                    "Component": "wing",
                    "Coefficient": "CA"
                },
                "mach_CN": {"Type": "mach_CA", "Coefficient": "CN"}
            }
        }

These sections are put into action by calls of ``cape --report``, where ``cape``
can be replaced by ``pycart``, ``pyfun``, or ``pyover``, as appropriate.  This
section enables some powerful capabilities, and it is often the longest section
of the JSON file.

There are three primary fields: "Sweeps", "Figures", and "Subfigures", along
with two minor settings of "Archive" and "Reports". The example above has a
report named "case" that produces one page for each solution and a report called
"mach" that creates sweeps of results from the data book. Users may build a
specific report with a command such as ``cape --report case`` (assuming there
is a report called ``"case"``).  With no value (i.e. ``cape --report``), the
first report in the *Reports* field is created.

Because this section often becomes very long, a useful tool is to separate the
definitions into multiple JSON files.  Using the example above may allow the
user to replace that section with the following syntax.

    .. code-block:: javascript
    
        "Report": {
            
            "Archive": false,
            "Reports": ["case", "mach"],
            "case": JSONFile("Report-case.json")
            "mach": {
                "Title": "Results for Mach Sweeps",
                "Sweeps": "mach"
            },
            "Sweeps": JSONFile("Report-Sweeps.json")
            "Figures": JSONFile("Report-Figures.json")
            "Subfigures": JSONFile("Report-Subfigures.json")
        }

The base level option names for this parameter are described in dictionary
format below.

The description of the available options is shown below.  If *Reports* is not
defined, the list of reports is 

    *Reports*: :class:`list` (:class:`str`) | ``["R1", "R2"]``
        List of reports defined in this JSON file
        
    *Archive*: {``true``} | ``false``
        Whether or not to tar folders in the report folder in order to reduce
        file count
        
    *Sweeps*: ``{}`` | ``{[S]}`` | :class:`dict` (:class:`dict`)
        Dictionary of sweep definitions (combined plots of subsets of cases)
        
    *Figures*: ``{}`` | ``{[F]}`` | :class:`dict` (:class:`dict`)
        Dictionary if figure definitions
        
    *Subfigures*: ``{}`` | ``{[U]}`` | :class:`dict` (:class:`dict`)
        Dictionary of subfigure definitions to be used by the figures 
    
    *R1*: :class:`dict`
        Definition of report named ``"R1"``
        
    *R2*: :class:`dict`
        Definition of report named ``"R2"``
        
.. _cape-json-ReportReport:

Report Definitions
==================

Each report is defined with a :class:`dict` containing several options.  The
name of the key is the name of the report, so for example ``pycart --report
case`` will look for a definition under ``"case"`` in the ``"Report"`` section. 
If ``pycart --report`` is called without a report name, cape will update the
first report definition it finds.  Reports named any of ``"Reports"``,
``"Archive"``, ``"Sweeps"``, ``"Figures"``, or ``"Subfigures"`` (case-sensitive)
are not allowed.

The options used to describe a single report are listed below.

    *case*: :class:`dict`
        Definition of report named ``"case"``
        
        *Title*: :class:`str`
            Title placed on title page and PDF title
            
        *Subtitle*: :class:`str`
            Subtitle placed on title page
            
        *Author*: :class:`str`
            LaTeX string of author(s) printed on title page
            
        *Affiliation*: :class:`str`
            Name of institution or otherwise to be placed below author
            
        *Logo*: :class:`str`
            File name (relative to ``report/`` folder) of logo to place on
            bottom left of each report page
            
        *Frontispiece*: :class:`str`
            File name (relative to ``report/`` folder) of image to be placed on
            title page
            
        *Restriction*: {``""``} | ``"SBU - ITAR"`` | :class:`str`
            Data release restriction to place in center footnote if applicable
            
        *Sweeps*: {``[]``} | :class:`list` (:class:`str`)
            List of names of sweeps (plots of run matrix subsets) to include
            
        *Figures*: {``[]``} | :class:`list` (:class:`str`)
            List of figures for analysis of each case that has been run at least
            one iteration
        
        *ErrorFigures*: {``[]``} | :class:`list` (:class:`str`)
            List of figures for each case with ``"ERROR"`` status, defaults to
            value of *Figures*
            
        *ZeroFigures*: {``[]``} | :class:`list` (:class:`str`)
            List of figures for each cases that have a folder but have not run
            for any iterations yet and are not marked ``"ERROR"``, defaults to
            value of *Figures*
            
.. _cape-json-ReportSweep:

Sweep Definitions
=================

Each sweep has a definition that is similar to a report but with additional
options to divide the run matrix into subsets.  For example, if the run matrix
has three independent variables (which pyCart calls trajectory keys) of
``"Mach"``, ``"alpha"``, and ``"beta"``, then a common sweep would be to plot
results as a function of Mach number for constant *alpha* and *beta*.  To do
that, one would put ``"EqCons": ["alpha", "beta"]`` in the sweep definition.

The full list of available options is below.

    *Sweeps*: ``{}`` | ``{[S]}`` | :class:`dict` (:class:`dict`)
        Dictionary of sweep definitions (combined plots of subsets of cases)
        
        *S*: :class:`dict`
            Dictionary of sweep definitions for sweep named ``"S"``
            
            *Figures*: {``[]``} | :class:`list` (:class:`str`)
                List of figures to include for each sweep subset
                
            *EqCons*: {``[]``} | :class:`list` (:class:`str`)
                List of trajectory keys to hold constant for each subset
                
            *TolCons*: {``{}``} | :class:`dict` (:class:`float`)
                Dictionary of trajectory keys to hold within a certain tolerance
                from the value of that key for the first case in the subset
                
            *IndexTol*: {``None``} | :class:`int`
                If used, only allow the index of the first and last cases in a
                subset to differ by this value
                
            *XAxis*: {``None``} | :class:`str`
                Name of trajectory key used to sort subset; if ``None``, sort by
                data book index
                
            *TrajectoryOnly*: ``true`` | {``false``}
                By default, the data book is the source for sweep plots; this
                option can restrict the plots to points in the current run
                matrix
                
            *GlobalCons*: {``[]``} | :class:`list` (:class:`str`)
                List of global constraints to only divide part of the run matrix
                into subsets
                
            *Indices*: {``None``} | :class:`list` (:class:`int`)
                If used, list of indices to divide into subsets
                
            *MinCases*: {``1``} | :class:`int`
                Minimum number of cases for a sweep to be reported
                
            *CarpetEqCons*: ``[]`` | :class:`list` (:class:`str`)
                Some sweep subfigures allow a sweep to be subdivided into
                subsweeps; this could be used to create plots of *CN* versus
                *Mach* with several lines each having constant *alpha*
                
            *CarpetTolCons*: ``{}`` | :class:`dict` (:class:`float`)
                Tolerance constraints for subdividing sweeps

The subsets are defined so that each case meeting the *GlobalCons* is placed
into exactly one subset.  For each subset, pyCart begins with the first
available case and applies the constraints using that point as a reference.
                
Constraints can be defined in more complex ways than the example given prior to
the list of options.  For relatively simple run matrices, grouping cases by
constant values of one or more trajectory keys (i.e. using *EqCons*) may be
adequate, but other run matrices may require more advanced settings.

For example, wind tunnel data often is collected at conditions that are not
exactly constant, i.e. the angle of attack may fluctuate slightly.  Instead of
using *EqCons*, a better option in this case would be to include ``"TolCons":
{"alpha": 0.02}``.  Then all cases in a subset would have an angle of attack
within ``0.02`` of the angle of attack of the first point of the subset.

Another advanced capability is to use *EqCons* such as ``["k%10"]`` or
``["k/10%10"]``.  This could be used to require each case to have the same ones
digit or the same tens digit of some trajectory variable called *k*.

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

.. _cape-json-ReportConditions:

Run Conditions Table Subfigure
------------------------------
The ``"Conditions"`` subfigure creates a table of conditions for the independent
variables.  The primary purpose is to list the run conditions for each case for
the observer to quickly reference which case is being analyzed.  It creates a
table with three three columns: name of trajectory key, abbreviation for the
key, and the value of that key for the case being reported.

If the subfigure is used as part of a sweep report, then the "Value" column will
show either the value of the first case in the sweep (if all cases in the sweep
have the same value) or an entry with the format ``v0, [vmin, vmax]`` where *v0*
is the value at the first point in the sweep, *vmin* is the minimum value for
that independent variable for each point in the sweep, and *vmax* is the maximum
value.

The options are listed below.
    
    *C*: :class:`dict`
        Dictionary of settings for *Conditions* type subfigure
        
        *Type*: {``"Conditions"``} | :class:`str`
            Subfigure type
        
        *Header*: {``"Conditions"``} | :class:`str`
            Heading placed above subfigure (bold, italic)
            
        *Position*: {``"t"``} | ``"c"`` | ``"b"``
            Vertical position in row of subfigures
            
        *Alignment*: {``"left"``} | ``"center"``
            Horizontal alignment
            
        *Width*: {``0.4``} | :class:`float`
            Width of subfigure as a fraction of page text width
        
        *SkipVars*: {``[]``} | :class:`list` (:class:`str`)
            List of trajectory keys to not include in conditions table


.. _cape-json-ReportSweepConditions:

Sweep Conditions Table Subfigure
--------------------------------
The ``"SweepConditions"`` subfigure class, which is only available for sweeps
(i.e. cannot be included in reports for individual cases), shows the list of
constraints that define a sweep.  It creates a three-column table with the first
column the name of the variable, the second column the value of the variable
(i.e. trajectory key or derived key such as ``k%10``) for the first case in the
sweep, and the third column a description of the constraint.  The constraint
description is either ``=``, meaning that all cases in the sweep have the same
value for that variable, or ``±tol`` if all the cases in the sweep are
constrained to be within a tolerance *tol* of the first point in the sweep.
            
    *C*: :class:`dict`
        Dictionary of settings for *SweepConditions* type subfigure
        
        *Type*: {``"SweepConditions"``} | :class:`str`
            Subfigure type
            
        *Header*: {``"Sweep Constraints"``} | :class:`str`
            Heading placed above subfigure (bold, italic)
        
        *Position*: {``"t"``} | ``"c"`` | ``"b"``
            Vertical alignment of subfigure
            
        *Alignment*: {``"left"``} | ``"center"``
            Horizontal alignment
            
        *Width*: {``0.4``} | :class:`float`
            Width of subfigure as a fraction of page text width

            
.. _cape-json-ReportSweepCases:

List of Cases in a Sweep
------------------------
The ``"SweepCases"`` subfigure class, which is only available for sweeps, shows
the list of cases in a sweep.  The method of displaying the sweeps is to list
the names of each case, i.e. the group/case folder name, in some monospace
format.  The list of options is below.

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
            
            
.. _cape-json-ReportSummary:

Tabular Force & Moment Results
------------------------------
The ``"FMTable"`` subfigure class presents a table of textual force and/or
moment coefficients for an individual case.  The user can specify a list of
components and a list of coefficients.  For each coefficient, the user may
choose to display the mean value, the iterative standard deviation, and/or the
iterative uncertainty estimate.

Aliases for this subfigure are ``"ForceTable`` and ``"Summary"``.

Each component (for example left wing, right wing, fuselage) has its own column,
and the coefficients form rows.  This subfigure class is only available for case
reports; it cannot be used on a sweep.

    *S*: :class:`dict`
        Dictionary of settings for *Summary* subfigures
        
        *Type*: {``"Summary"``} | :class:`str`
            Subfigure type
            
        *Header*: {``"Force \\& moment summary"``} | :class:`str`
            Heading placed above subfigure (bold, italic)
            
        *Position*: {``"t"``} | ``"c"`` | ``"b"``
            Vertical alignment of subfigure
            
        *Alignment*: {``"left"``} | ``"center"``
            Horizontal alignment
            
        *Width*: {``0.6``} | :class:`float`
            Width of subfigure as a fraction of page text width
            
        *Iteration*: {``0``} | :class:`int`
            If nonzero, display results from specified iteration number
            
        *Components*: {``["entire"]``} | :class:`list` (:class:`str`)
            List of components
            
        *Coefficients*: {``["CA", "CY", "CN"]``} | :class:`list` (:class:`str`)
            List of coefficients to display
            
        *CA*: {``["mu", "std"]``} | :class:`list` (:class:`str`)
            Quantities to report for *CA*; mean, standard deviation, and error
            
        *CY*: {``["mu", "std"]``} | :class:`list` (:class:`str`)
            Quantities to report for *CY*; mean, standard deviation, and error
            
        *CN*: {``["mu", "std"]``} | :class:`list` (:class:`str`)
            Quantities to report for *CN*; mean, standard deviation, and error
            
        *CLL*: {``["mu", "std"]``} | :class:`list` (:class:`str`)
            Quantities to report for *CLL*; mean, standard deviation, and error
            
        *CLM*: {``["mu", "std"]``} | :class:`list` (:class:`str`)
            Quantities to report for *CLM*; mean, standard deviation, and error
            
        *CLN*: {``["mu", "std"]``} | :class:`list` (:class:`str`)
            Quantities to report for *CLN*; mean, standard deviation, and error
    

.. _cape-json-ReportPlotCoeff:
            
Iterative Force or Moment Plot
------------------------------
To plot iterative histories of force and/or moment coefficients on one or more
component, use the ``"PlotCoeff"`` subfigure.  There are many options for this
class of subfigure.  In addition to standard alignment and caption options,
there are options for which component(s) and coefficient(s) to plot, options for
how the plots are presented, which iterations to use, output format, and figure
sizes.

The default caption, which is placed in sans-serif font below the figure, is
*Component*/*Coefficient*, which may be confusing if two components are
included.  For example, a caption such as ``"[LeftWing, RightWing]/CY"`` could
be generated automatically.

The full list of options is shown below.

    *P*: :class:`dict`
        Dictionary of settings for *PlotCoeff* subfigures
        
        *Type*: {``"PlotCoeff"``} | :class:`str`
            Subfigure type
            
        *Header*: {``""``} | :class:`str`
            Heading placed above subfigure (bold, italic)
            
        *Position*: ``"t"`` | ``"c"`` | {``"b"``}
            Vertical alignment of subfigure
            
        *Alignment*: ``"left"`` | {``"center"``}
            Horizontal alignment
            
        *Width*: {``0.5``} | :class:`float`
            Width of subfigure as a fraction of page text width
            
        *nPlotFirst*: {``0``} | :class:`int`
            First iteration to plot; often useful to eliminate startup
            transients from the plot which may have a much larger scale than the
            final value
            
        *nPlotLast*: {``null``} | :class:`int`
            If specified, only plot up to this iteration
            
        *nPlot*: {``null``} | :class:`int`
            If specified, plot at most this many iterations; alternative method
            to hide startup transients
            
        *nStats*: :class:`int`
            Number of iterations to use for statistics; defaults to data book
            option
            
        *nMinStats*: :class:`int`
            First iteration to allow to be used for mean calculation
            
        *nMaxStats*: :class:`int`
            Maximum number of iterations to allow to be used in statistics
            
        *FigWidth*: {``6.0``} | :class:`float`
            Width of figure internally to Python; affects aspect ratio of figure
            and font size when integrated into report; decrease this parameter
            to make text appear larger in report
            
        *FigHeight*: {``4.5``} | :class:`float`
            Similar to *FigWidth* and primarily used to set aspect ratio
        
        *Component*: {``"entire"``} | :class:`str` | :class:`list`
            Component or list of components to plot, must be name(s) of
            components defined in :file:`Config.xml`
            
        *Coefficient*: ``"CA"`` | ``"CY"`` | {``"CN"``} | ``"CLL"`` | ``"CLM"``
        | ``"CLN"`` | :class:`list`
        
            Force or moment coefficient(s) to plot, any of ``"CA"``
            
        *Delta*: {``0.0``} | :class:`float`
            If nonzero, plot a horizontal line this value above and below the
            iterative mean, by default with a dashed red line
            
        *StandardDeviation*: {``0.0``} | :class:`float`
            If nonzero, plot a rectangular box centered on the iterative mean
            value and spanning vertically above and below the mean this number
            times the iterative standard deviation; the width of the box shows
            the iteration window used to compute the statistics
            
        *IterativeError*: {``0.0``} | :class:`float`
            If nonzero, plot a rectangular box centered on the iterative mean
            value and spanning vertically above and below the mean this number
            times the iterative uncertainty; the width of the box shows
            the iteration window used to compute the statistics
            
        *ShowMu*: {``[true, false]``} | ``true`` | ``false`` | :class:`list`
            Whether or not to print the mean value in the upper right corner of
            the plot; by default show the value of the first
            component/coefficient and not for the others
            
        *ShowSigma*: {``[true, false]``} | ``true`` | ``false`` | :class:`list`
            Whether or not to print the standard deviation in the upper left
            
        *ShowDelta*: {``[true, false]``} | ``true`` | ``false`` | :class:`list`
            Whether or not to print the fixed delta value in the upper right
            
        *ShowEpsilon*: ``true`` | {``false``}
            Whether or not to print iterative uncertainty in upper left
            
        *Format*: {``"pdf"``} | ``"svg"`` | ``"png"`` | :class:`str`
            Format of graphic file to save
            
        *DPI*: {``150``} | :class:`int`
            Resolution (dots per inch) if saved as a raster format
            
        *LineOptions*: {``{"color":["k","g","c","m","b","r"]}``} | :class:`dict`
            Plot options for the primary iterative plot; options are passed to
            :func:`matplotlib.pyplot.plot`, and lists are cycled through, so the
            default plots the first history in black, the second in green, etc.
            
        *MeanOptions*: {``{"ls": null}``} | :class:`dict`
            Plot options for the iterative mean value; most options are
            inherited from *LineOptions*, and setting *ls* to ``None`` as in the
            default creates a dotted line that is dashed for the iterations used
            to compute the mean
            
        *StDevOptions*: {``{"facecolor": "b", "alpha": 0.35, "ls": "none"}``} |
        :class:`dict`
        
            Plot options for standard deviation plots; options are passed to
            :func:`matplotlib.pyplot.fill_between`
            
        *ErrPlotOptions*: {``{"facecolor": "g", "alpha": 0.4, "ls": "none"}``} |
        :class:`dict`
        
            Plot options for iterative uncertainty window, passed to
            :func:`matplotlib.pyplot.fill_between`
            
        *DeltaOptions*: {``{"color": null}``} | :class:`dict`
            Plot options for fixed interval plot, passed to
            :func:`matplotlib.pyplot.plot`
        
A typical application of this subfigure involves plotting multiple coefficients,
and it is often advantageous to define a new subfigure class and allow most of
the plotting options to "cascade."  Consider the following example used to
define plots of the force coefficients on left and right wings of some geometry.

    .. code-block:: javascript
    
        "Subfigures": {
            "WingCA": {
                "Type": "PlotCoeff",
                "Component": ["LeftWing", "RightWing"],
                "Coefficient": "CA",
                "LineOptions": {"color": ["k", "g"]},
                "StandardDeviation": 1.0
            },
            "WingCN": {
                "Type": "WingCA",
                "Coefficient": "CN"
            },
            "LeftWingCY": {
                "Type": "WingCA",
                "Coefficient": "CY",
                "Component": "LeftWing"
            },
            "RightWingCY": {
                "Type": "LeftWingCY",
                "Component": "RightWing",
                "LineOptions": {"color": "g"}
            }
            
The example creates four iterative history plots without having to repeat all
the options for each subfigure.  All four plots will use a *StandardDeviation*
value of ``1.0``.  Also note how multiple levels of recursion are allowed as
shown by the last subfigure which uses ``"LeftWingCY" --> "WingCA" -->
"PlotCoeff"`` for the *Type* specification.


.. _cape-json-ReportSweepCoeff:

Force and Moment Coefficient Sweep Plots
----------------------------------------
The ``"SweepCoeff"`` class of subfigure is used to plot one or more force or
moment coefficients for a sweep of cases.  For example, it can be used to plot
normal force coefficient versus Mach number for cases having the same angle of
attack and sideslip angle.  If the sweep including this subfigure have
*CarpetEqCons* or *CarpetTolCons* specified, the plots on each page (i.e. for
each sweep) will be divided into multiple lines as divided by the carpet
constraints.

    *S*: :class:`dict`
        Dictionary of settings for *SweepCoeff* subfigure
        
        *Type*: {``"PlotCoeff"``} | :class:`str`
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
            Width of figure internally to Python; affects aspect ratio of figure
            and font size when integrated into report; decrease this parameter
            to make text appear larger in report
            
        *FigHeight*: {``4.5``} | :class:`float`
            Similar to *FigWidth* and primarily used to set aspect ratio
        
        *Component*: {``"entire"``} | :class:`str` | :class:`list`
            Component or list of components to plot, must be name(s) of
            components defined in :file:`Config.xml`
            
        *Coefficient*: ``"CA"`` | ``"CY"`` | {``"CN"``} | ``"CLL"`` | ``"CLM"``
        | ``"CLN"`` | :class:`list`
        
            Force or moment coefficient(s) to plot, any of ``"CA"``
            
        *StandardDeviation*: {``0.0``} | :class:`float`
            If nonzero, plot the value *StandardDeviation* above and below the
            mean value at each point
            
        *MinMax*: ``true`` | {``false``}
            Whether or not to plot minimum and maximum value over iterative
            history
            
        *Format*: {``"pdf"``} | ``"svg"`` | ``"png"`` | :class:`str`
            Format of graphic file to save
            
        *DPI*: {``150``} | :class:`int`
            Resolution (dots per inch) if saved as a raster format
            
        *LineOptions*: {``{"color": "k", "marker": ["^","s","o"]}``} |
        :class:`dict`
        
            Plot options for the plot value; options are passed to
            :func:`matplotlib.pyplot.plot`, and lists are cycled through, so the
            default plots the first line with a ``"^"`` marker, etc.
            
        *TargetOptions*: {``{"color": "r", "marker": ["^","s","o"]}``} |
        :class:`dict`
        
            Plot options for target value plot if the data book contains a
            target for the current component and coefficient
            
        *MinMaxOptions*: {``{"facecolor":"g", "alpha":0.4, "lw":0}``} |
        :class:`dict`
        
            Options for plot of min/max value over iterative window, passed to
            :func:`matplotlib.pyplot.fill_between`
            
        *StDevOptions*: {``{"facecolor":"b", "alpha":0.35, "lw":0}``} |
        :class:`dict`
        
            Plot options for standard deviation plots; options are passed to
            :func:`matplotlib.pyplot.fill_between`
            
        *PlotTypeMinMax*: {``"FillBetween"``} | ``"ErrorBar"``
            Method for plotting range of values for min/max values
            
        *PlotTypeUncertainty*: ``"FillBetween"`` | {``"ErrorBar"``}
            Method for plotting range of values for sampling error
            
        *PlotTypeStDev*: {``"FillBetween"``} | ``"ErrorBar"``
            Method for plotting range of values for standard deviation plot

