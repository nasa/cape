"""
Command-line plotting interface: :mod:`cape.pycart.pc_Plot`
===========================================================


"""

# pyCart modules
from .dataBook import Aero
from .case import ReadCaseJSON
from .options.Plot import Plot
# System interface
import os
# Reading JSON files
import json
# Plotting
import matplotlib.pyplot as plt

# Function to read the local settings file.
def ReadPlotJSON(fname='plot.json'):
    """Read plot options from JSON file
    
    :Call:
        >>> opts = pc_Plot.ReadPlotJSON(fname='plot.json')
    :Inputs:
        *fname*: :class:`str`
            Name of file to read
    :Outputs:
        *opts*: :class:`pyCart.options.Plot.Plot
            Options interface for iterative history plots
    :Versions:
        * 2015-02-17 ``@ddalle``: First version
    """
    # Check for the file.
    if not os.path.isfile(fname):
        # Empty settings object.
        return Plot()
    # Read the file, fail if not present.
    f = open(fname)
    # Read the settings.
    opts = json.load(f)
    # Close the file.
    f.close()
    # Convert to a flowCart object.
    return Plot(**opts)
    
# Function to process command-line inputs.
def ProcessCLI(opts, C=[], **kw):
    """Process command-line overrides to options
    
    :Call:
        >>> ProcessCLI(opts, C, **kw)
    :Inputs:
        *opts*: :class:`pyCart.options.Plot.Plot
            Options interface for iterative history plots
        *C*: :class:`list`\ [:class:`str`]
            List of coefficients to plot
        *n*: :class:`int`
            Number of iterations to plot
        *nAve*: :class:`int`
            Number of iterations to use for statistics
        *nFirst*: :class:`int`
            
        *f*: {``'plot.json'``} | :class:`str`
            Name of JSON formatting file to read
        *p*: {``'entire'``} | :class:`str`
            Name of component to plot 
        *nRow*: :class:`int` or :class:`str`
            Number of rows to use in plot
        *nCol*: :class:`int` or :class:`str`
            Number of columns to use in plot
        *ext*: {``'pdf'``} | ``'pdf'`` | ``'png'``
            File extension 
        *dpi*: :class:`int`
            Dots per inch for output file if raster format is used
        *o*: :class:`str`
            File name to use (overrides default: ``aero_$comp.$ext``
        *i*: :class:`bool`
            Whether or not to display plot interactively
        *tag*: :class:`str`
            Tag to put in uppler left corner of plot
        *dCA*: :class:`float` or :class:`str` or ``None``
            Delta to display on C_A plots
        *dCY*: :class:`float` or :class:`str` or ``None``
            Delta to display on C_Y plots
        *dCN*: :class:`float` or :class:`str` or ``None``
            Delta to display on C_N plots
        *dCLL*: :class:`float` or :class:`str` or ``None``
            Delta to display on C_l plots
        *dCLM*: :class:`float` or :class:`str` or ``None``
            Delta to display on C_m plots
        *dCLN*: :class:`float` or :class:`str` or ``None``
            Delta to display on C_n plots
        *restriction*: :class:`str`
            Distribution restriction text to display at bottom of plot
        *SBU*: :class:`bool`
            Use the restriction text ``'SBU - ITAR'``
    :Versions:
        * 2015-02-17 ``@ddalle``: First version
    """
    # Coefficient list
    if (type(C).__name__ == 'list') and (len(C) > 0):
        # Set coefficients.
        opts['Coefficients'] = C
    # Number of average iterations.
    if kw.get('nAve'): opts['nAverage'] = kw['nAve']
    # Number of iterations to plot.
    if kw.get('n'): opts['nAverage'] = kw['n']
    # Start and end iterations
    if kw.get('nLast'):  opts['nLast']  = kw['nLast']
    if kw.get('nFirst'): opts['nFirst'] = kw['nFirst']
    # Check for component plot.
    if kw.get('p'): opts.set_PlotComponents([kw['p']])
    # Number of rows and columns.
    if kw.get('nRow'): opts['nRow'] = kw['nRow']
    if kw.get('nCol'): opts['nCol'] = kw['nCol']
    # Process restriction flag
    if 'SBU' in kw:
        # Sensitive but unclassified.
        opts['Restriction'] = 'SBU - ITAR'
    elif 'ITAR' in kw:
        # ITAR
        opts['Restriction'] = 'ITAR'
    elif 'FOUO' in kw:
        # U/FOUO, For Official Use Only
        opts['Restriction'] = 'U/FOUO'
    elif 'SECRET' in kw:
        # Classified, secret
        opts['Restriction'] = 'SECRET'
    else:
        # Use the --restriction option
        opts['Restriction'] = kw.get('restriction', '')
    # Check for deltas.
    if kw.get('dCA'):  opts['Deltas']['CA'] = kw['dCA']
    if kw.get('dCY'):  opts['Deltas']['CY'] = kw['dCY']
    if kw.get('dCN'):  opts['Deltas']['CN'] = kw['dCN']
    if kw.get('dCLL'): opts['Deltas']['CLL'] = kw['dCLL']
    if kw.get('dCLM'): opts['Deltas']['CLM'] = kw['dCLM']
    if kw.get('dCLN'): opts['Deltas']['CLN'] = kw['dCLN']

# Main function
def pc_Plot(C=[], **kw):
    """Plot a specified list of coefficients or residuals
    
    :Call:
        >>> h = pc_Plot(C=[], **kw)
    :Inputs:
        *C*: :class:`list`\ [:class:`str`]
            List of coefficients to plot
        *f*: {``'plot.json'``} | :class:`str`
            Name of JSON formatting file to read
        *p*: {``'entire'``} | :class:`str`
            Name of component to plot 
        *ext*: {``'pdf'``} | ``'pdf'`` | ``'png'`` | ``'svg'``
            File extension 
        *dpi*: :class:`int`
            Dots per inch for output file if raster format is used
        *o*: :class:`str`
            File name to use (overrides default: ``aero_$comp.$ext``
        *i*: :class:`bool`
            Whether or not to display plot interactively
        *tag*: :class:`str`
            Tag to put in uppler left corner of plot
        *restriction*: :class:`str`
            Distribution restriction text to display at bottom of plot
        *SBU*: :class:`bool`
            Use the restriction text ``'SBU - ITAR'``
    :Versions:
        * 2014-11-13 ``@ddalle``: First version
    """
    # Process file name.
    fjson = kw.get('f', 'plot.json')
    # Read the options.
    opts = ReadPlotJSON(fjson)
    # Process the options.
    ProcessCLI(opts, C, **kw)
    # Get the components.
    comps = opts.get_PlotComponents()
    
    # Name of this folder.
    fpwd = os.path.split(os.getcwd())[-1]
    # Get extension.
    ext = kw.get('ext', 'pdf')
    # Get file name.
    f0 = kw.get('o', 'aero')
    
    # Check for 'case.json' flag.
    if os.path.isfile('case.json'):
        fc = ReadCaseJSON()

    # Read the data.
    AP = Aero(comps)
    
    # Loop through components.
    for comp in comps:
        # New figure.
        plt.figure()
        # Default tag
        ftag = '%s\nComponent=%s' % (fpwd, comp)
        
        # Coefficients to plot.
        coeffs = opts.get_PlotCoeffs(comp)
        
        # Initialize dictionary of plot deltas.
        d = {}
        # Loop through coefficients.
        for coeff in coeffs:
            d[coeff] = opts.get_PlotDelta(coeff, comp)
        
        # Plot keyword arguments
        kwp = {
            'nRow': opts.get_nPlotRows(comp),
            'nCol': opts.get_nPlotCols(comp),
            'n': opts.get_nPlotIter(comp),
            'nFirst': opts.get_nPlotFirst(comp),
            'nLast': opts.get_nPlotLast(comp),
            'nAvg': opts.get_nAverage(comp),
            'tag': kw.get('tag', ftag),
            'restriction': opts.get_PlotRestriction(comp),
            'd': d
        }
            
        # Plot the components.
        h = AP.Plot(comp, coeffs, **kwp)
        
        # Check the pass flag.
        if kw.get('PASS'): h['pass'].set_text('PASS')
        # Check for ability to put iteration number.
        if os.path.isfile('case.json'):
            # Set the iteration flag.
            h['iter'].set_text('%i/%i' %
                (AP[comp].i[-1], fc.get_LastIter()))
        
        # Output file name.
        fname = '%s_%s.%s' % (f0, comp, ext)
        
        # Save the file.
        if ext in ['png', 'jpg']:
            # Get the dots per inch setting.
            idpi = int(kw.get('dpi', 150))
            # Save the figure.
            h['fig'].savefig(fname, dpi=idpi)
        else:
            # Save the figure to vector format.
            h['fig'].savefig(fname)
        # Check for interactive mode.
        if kw.get('i'):
            plt.show()
        
        
