
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
                    "Tecplot": "surface.lay"
                },
                "CA": {
                    "Width": 0.5
                }
            }, 
            "Report": {
                "Title": "Automated Cart3D Report",
                "Figures": ["Summary", "Forces"],
                "FailFigures": ["Summary", "Surface"],
                "ZeroFigures": ["Summary", "Surface"]
            }
        }

These sections are put into action by calls of ``pycart --report``.  There are
three recognized settings: "Archive", "Figures", and "Subfigures".  Remaining
options are identified as the names of reports.  The example above has one
report named "Report", but the JSON file may have more reports with different
names.  Users may build a specific report with a command such as ``pycart
--report case`` (assuming there is a report called "case").

Each report contains a "Title", a list of figures for nominal cases, a list of
figures for cases that fail (optional), and a list of figures for cases that
have not been started yet (optional).  Each case has one or more pages (but each
case starts on a new page) that contains the appropriate list of figures from
these three possibilities.

Each figure contains an alignment, a heading, and a list of subfigures.  The
subfigure definitions contain the real information about the report.

The description of the available options is shown below.

    *Components*: {``["entire"]``} | :class:`list` (:class:`str`)
        List of components to plot
        
    *Coefficients*: {``["CA","CY","CN","L1"]``} | :class:`list` (``"CA"`` |
    ``"CY"`` | ``"CN"`` | ``"CLL"`` | ``"CLM"`` | ``"CLN"`` | ``"L1"``)
            
        List of coefficients to plot (for that component)
        
    *nPlot*: {``1000``} | :class:`int` >0
        Plot the force/moment coefficients or residual for the last *nPlot*
        iterations
        
    *nAverage*: {``100``} | :class:`int` >=0
        Use the last *nAverage* iterations to compute an average of each
        coefficient.  If ``0``, do not compute average.
        
    *nRow*: {``2``} | :class:`int` >0
        Number of rows of plots
        
    *nCol*: {``2``} | :class:`int` >0
        Number of columns of plots
        
    *Restriction*: {``""``} | ``"SBU - ITAR"`` | ``"SECRET"`` | :class:`str`
        String of text to place at bottom center of plot that displays any
        limitations on distribution of the plot
        
    *Deltas*: {``{}``} | :class:`dict` (:class:`float` >=0)
        Dict of deltas to plot above and below the mean with a red line for each
        coefficient; overridden by *dCA*, etc.
        
    *dCA*: {``0.01``} | :class:`float` >=0
        Deltas to plot above and below *CA* mean
        
    *dCY*: {``0.01``} | :class:`float` >=0
        Deltas to plot above and below *CY* mean
        
    *dCN*: {``0.01``} | :class:`float` >=0
        Deltas to plot above and below *CN* mean
        
    *dCLL*: {``0.01``} | :class:`float` >=0
        Deltas to plot above and below *CLL* mean
        
    *dCLM*: {``0.01``} | :class:`float` >=0
        Deltas to plot above and below *CLM* mean
        
    *dCLN*: {``0.01``} | :class:`float` >=0
        Deltas to plot above and below *CLN* mean
    
