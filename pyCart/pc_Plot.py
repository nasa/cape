"""
Command-line plotting interface: :mod:`pyCart.pc_Plot`
======================================================


"""

# Modules
from . import aero
# System interface
import os

# Main function
def pc_Plot(C, **kw):
    """Plot a specified list of coefficients or residuals
    
    :Call:
        >>> h = pc_Plot(C, **kw)
    :Inputs:
        *C*: :class:`list` (:class:`str`)
            List of coefficients to plot
        *p*: :class:`str`
            Name of component to plot [{``'entire'``} | :class:`str`]
        *nRow*: :class:`int` or :class:`str`
            Number of rows to use in plot
        *nCol*: :class:`int` or :class:`str`
            Number of columns to use in plot
        *ext*: :class:`str`
            File extension [{``'pdf'``} | ``'pdf'`` | ``'png'``]
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
        * 2014-11-13 ``@ddalle``: First version
    """
    # Number of components
    nC = len(C)
    # Get the component.
    comp = kw.get('p', 'entire')
    # Get the number or rows and columns.
    nRow = kw.get('nRow')
    nCol = kw.get('nCol')
    # Process defaults.
    if nCol is None:
        # Two columns only for 2x2 or 3x2
        if nC in [4, 6]:
            # Use two columns.
            nCol = 2
            nRow = nC / nCol
        else:
            # Use a single column.
            nCol = 1
            nRow = nC
    elif nRow is None:
        # Figure out the right number of rows.
        nRow = pyCart.aero.np.ceil(float(nC)/float(nRow))
    # Ensure integers.
    nRow = int(nRow)
    nCol = int(nCol)
    # Get extension.
    ext = kw.get('ext', 'pdf')
    # Get file name.
    if 'o' in kw:
        # Manual file name.
        fname = kw['o']
        # Check for extension.
        if '.' not in fname:
            # Append extension.
            fname = fname + '.' + ext
    else:
        # Default file name.
        fname = 'aero_%s.%s' % (comp, ext)
    # Name of this folder.
    fpwd = os.path.split(os.getcwd())[-1]
    # Default tag
    ftag = '%s\nComponent=%s' % (fpwd, comp)
    # Process restriction flag
    if 'SBU' in kw:
        # Sensitive but unclassified.
        fsbu = 'SBU - ITAR'
    elif 'ITAR' in kw:
        # ITAR
        fsbu = 'ITAR'
    elif 'FOUO' in kw:
        # U/FOUO, For Official Use Only
        fsbu = 'U/FOUO'
    elif 'SECRET' in kw:
        # Classified, secret
        fsbu = 'SECRET'
    else:
        # Use the --restriction option
        fsbu = kw.get('restriction', '')
    # Process default delta
    d0 = kw.get('d', 0.01)
    # Convert it to proper format.
    if d0 is True:
        # Default is OFF
        d0 = None
    else:
        # Convert to float
        d0 = float(d0)
    # Form dictionary of all coefficient deltas.
    d = {}
    # Loop through coeffs.
    for c in ['CA', 'CY', 'CN', 'CLL', 'CLM', 'CLN']:
        # Read the option.
        dc = kw.get('d'+c, d0)
        # Convert to float.
        if dc is not None: dc = float(dc)
        d[c] = dc
    # Plot keyword arguments
    kwp = {
        'n': int(kw.get('nShow', kw.get('n', 1000))),
        'nAvg': int(kw.get('nAvg', 100)),
        'tag': kw.get('tag', ftag),
        'restriction': fsbu,
        'd': d
    }
        
    # Read the component.
    FM = aero.Aero([comp])[comp]
    # Plot the components.
    h = FM.Plot(nRow, nCol, C, **kwp)
    
    # Save the file.
    if fname.endswith('png') or fname.endswith('jpg'):
        # Get the dots per inch setting.
        idpi = int(kw.get('dpi', 150))
        # Save the figure.
        h['fig'].savefig(fname, dpi=idpi)
    else:
        # Save the figure to vector format.
        h['fig'].savefig(fname)
    # Check for interactive mode.
    if kw.get('i'):
        aero.plt.show()
        
        
