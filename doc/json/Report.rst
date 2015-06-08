
--------------------------------------
Automated Report Generation with LaTeX
--------------------------------------

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

These sections are put into action by calls of ``pycart --report``.  There are
three primary fields: "Sweeps", "Figures", and "Subfigures", along with two
minor settings of "Archive" and "Reports".  Remaining options are identified as
the names of reports.  The example above has one report named "Report", but the
JSON file may have more reports with different names.  Users may build a
specific report with a command such as ``pycart --report case`` (assuming there
is a report called ``"case"``).

Each report contains a "Title", a list of figures for nominal cases, a list of
figures for cases that fail (optional), and a list of figures for cases that
have not been started yet (optional).  Each case has one or more pages (but each
case starts on a new page) that contains the appropriate list of figures from
these three possibilities.

In addition, a report can contain so-called "Sweeps," which report status and
results from subsets of the run matrix.  For example, one can create a page of a
report for cases having the same angle of attack and sideslip, and the page
could contain plots versus Mach number.

Each figure contains an alignment, a heading, and a list of subfigures.  The
subfigure definitions contain the real information about the report.

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
        
        
Report Definitions
==================

Each report is defined with a :class:`dict` containing several options.  The
name of the key is the name of the report, so for example ``pycart --report
case`` will look for a definition under ``"case"`` in the ``"Report"`` section. 
If ``pycart --report`` is called without a report name, pyCart will update the
first report definition it finds.  Reports named any of ``"Reports"``,
``"Archive"``, ``"Sweeps"``, ``"Figures"``, or ``"Subfigures"`` (case-sensitive)
are not allowed.

The options used to describe a single report are listed below.

    *case*: :class:`dict`
        Definition of report named ``"case"``
        
        *Title*: :class:`str`
            Title placed on title page and PDF title
            
        *Author*: :class:`str`
            LaTeX string of author(s) printed on title page
            
        *Logo*: :class:`str`
            File name (relative to ``report/`` folder if relative) of logo to
            place on bottom left of each report page
            
        *Restriction*: {``""``} | ``"SBU - ITAR"`` | ``"SECRET"`` | :class:`str`
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
            
            *Figures*: ``[]`` | :class:`list` (:class:`str`)
                List of figures to include for each sweep subset
                
            *EqCons*: ``[]`` | :class:`list` (:class:`str`)
                List of trajectory keys to hold constant for each subset
                
            *TolCons*: ``{}`` | :class:`dict` (:class:`float`)
                Dictionary of trajectory keys to hold within a certain tolerance
                from the value of that key for the first case in the subset
                
            *IndexTol*: ``None`` | :class:`int`
                If used, only allow the index of the first and last cases in a
                subset to differ by this value
                
            *XAxis*: ``None`` | :class:`str`
                Name of trajectory key used to sort subset; if ``None``, sort by
                data book index
                
            *GlobalCons*: ``[]`` | :class:`list` (:class:`str`)
                List of global constraints to only divide part of the run matrix
                into subsets
                
            *Indices*: ``None`` | :class:`list` (:class:`int`)
                If used, list of indices to divide into subsets

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


Figure Definitions
==================

Each figure contains a small number of options used to define the figure.  The
primary option is a list of subfigures, and the others are also defined below.

    *Figures*: ``{}`` | ``{[F]}`` | :class:`dict` (:class:`dict`)
        Dictionary of figure definitions
        
        *F*: :class:`dict`
            Dictionary of settings for figure called ``"F"``
            
            *Heading*: :class:`str`
                Title to be placed at the top of the figure
