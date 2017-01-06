#!/usr/bin/env python
"""
Run Tecplot on a Layout File and Export an Image
================================================

Run Tecplot using a macro that automatically opens a layout and exports an
image with specified image size.

:Usage:
    .. code-block:: bash
    
        $ pc_ExportLayout.py $LAY [$PNG] [OPTIONS]
        
:Inputs:
    
    *LAY*: Name of the Tecplot layout file
    *PNG*: Name of output file 
    
:Options:

    -h, --help
        Display this help message and quit
        
    --fmt FMT
        Export image with format *FMT*; must be valid format understood by
        Tecplot; also affects default output file name [Default is ``png``]
        
    -w, --width WIDTH
        Export PNG image with width of *WIDTH* pixels

:Versions:
    * 2017-01-05 ``@ddalle``: First version
"""

# System interface
import os
# Input parsing
import cape.argread
import cape.tecplot

# Main function
def ExportLayout(*a, **kw):
    """Run a special macro that exports an image based on a Tecplot layout
    
    :Call:
        >>> main(lay, png, w=None, fmt="png")
        >>> main(i=lay, o=png, w=None, fmt="png")
    :Inputs:
        *lay*: :class:`str`
            Name of input Tecplot layout file
        *png*: {``None``} | :class:`str`
            Output image file name; defaults to *lay* with extension changed
        *fmt*: {``"png"``} | :class:`str`
            Valid export image format
        *w*: {``None``} | :class:`int`
            Image width
    :Versions:
        * 2017-01-05 ``@ddalle``: First versoin
    """
    # Get the input
    if len(a) == 0:
        # Need at least one input
        raise IOError("Function 'ExportLayout' must have at least one input")
    else:
        # Read the first input
        flay = a[0]
    # Get extension
    fmt = kw.get('fmt', 'png')
    # Form default file name
    fpng = '.'.join(flay.split('.')[:-1])
    fpng = '%s.%s' % (fpng, fmt)
    # Check for second input
    if len(a) > 1: fpng = a[1]
    # Override with '-o' flag
    fpng = kw.get('o', fpng)
    
    # Get the width
    w = kw.get('w', kw.get('width'))
    # Convert to integer
    if w is not None: w = int(w)
    
    # Run the functions
    cape.tecplot.ExportLayout(flay, fname=fpng, fmt=fmt.upper(), w=w)
    
    
    
# Check if run as a script.
if __name__ == "__main__":
    # Parse inputs.
    a, kw = cape.argread.readkeys(os.sys.argv)
    
    # Check for a help flag.
    if kw.get('h') or kw.get('help'):
        print(__doc__)
        os.sys.exit()
        
    # Run the main function
    ExportLayout(*a, **kw)
