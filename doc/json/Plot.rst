
--------------------------
Iterative History Plotting
--------------------------

The section in :file:`pyCart.json` labeled "Plot" is for plotting iterative
histories of certain components along with (optionally) the L1 residual history.
The list of available options and defaults is shown below.

    .. code-block:: javascript
    
        "Plot": {
            "Components": ["entire"],
            "Coefficients": ["CA", "CY", "CN", "L1"],
            "nPlot": 1000,
            "nAverage": 100,
            "nRow": 2,
            "nCol": 2,
            "Restriction": ""
            "Deltas": {},
            "dCA": 0.01,
            "dCY": 0.01,
            "dCN": 0.01,
            "dCLL": 0.01,
            "dCLM": 0.01,
            "dCLN": 0.01
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
    
