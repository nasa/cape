
--------------------------------------
Automated Report Generation with LaTeX
--------------------------------------

The section in :file:`pyCart.json` labeled "Report" is for generating automated
reports of results.  It requires a fairly complete installation of `pdfLaTeX`.
Further, an installation of Tecplot 360 enhances the capability of the report
generation.

    .. code-block:: javascript
    
        "Report": {
            "Report": {
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
                
                },
                "CA": {
                    "Width": 0.5
                },
            }
        }

It is also possible to specify special rules for given components, for example
if one component has more unsteadiness than others, or if the user wants to plot
moments on one coefficient.  See the following example.

    .. code-block:: javascript
    
        "Plot": {
            "Components": ["LeftWing", "LeftAileron", "Fuselage"],
            "Coefficients": ["CA", "CY", "CN", "L1"],
            "Deltas": {"CA": 0.005, "CY": 0.01, "CN": 0.01},
            "LeftAileron": {
                "Coefficients": ["CLL", "CLM", "CLN"],
                "nRow": 3,
                "nCol": 1,
                "dCLL": 0.002,
                "dCLM": 0.005,
                "dCLN": 0.005
            }
        }
        
What this example shows is that the user can specify options that by default
apply to all the component plots, but the user can specify specific options for
each individual component plot that override these defaults.

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
    
