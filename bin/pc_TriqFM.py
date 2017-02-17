#!/usr/bin/env python
"""
Calculate Forces and Moments on a TRIQ File
===========================================

Calculate the integrated forces and moments on a triangulated surface

:Call:

    .. code-block:: console
    
        $ pc_TriqFM.py $TRIQ $COMP1 $COMP2 [...] [OPTIONS]
        $ pc_TriqFM.py -h
        $ triqfm $TRIQ $COMP1 $COMP2 [...] [OPTIONS]
        $ triqfm -h

:Inputs:
    * *TRIQ*: Name of annotated triangulation (TRIQ) file
    * *COMP1*: Name or 1-based index of first component to analyze
    * *COMP2*: Name or 1-based index of second component to analyze
    
:Options:
    -h, --help
        Display this help message and exit
        
    --json *JSON*
        Read options from commented JSON file called *JSON*
        
    --triq *TRIQ*
        Use *TRIQ* as unmapped input file {"grid.i.triq"}
        
    --comps *COMPS*
        List of components to analyze separated by commas
        
    --tri *TRI*, --map *TRI*
        Use a separate triangulation (Cart3D tri, UH3D, AFLR3 surf, IDEAS unv,
        or other TRIQ format) as a map for which triangles to extract; if used,
        the component list *COMPS* and config file *CONFIG* apply to this file
        
    -m, --momentum
        Include momentum forces in total
        
    --xMRP *XMRP*
        Set *x*-coordinate of moment reference point {0.0}
        
    --yMRP *YMRP*
        Set *x*-coordinate of moment reference point {0.0}
        
    --zMRP *ZMRP*
        Set *x*-coordinate of moment reference point {0.0}
        
    --MRP "*MRP*"
        Moment reference point {"*XMRP*, *YMRP*, *ZMRP*"} 
        
:Versions:
    * 2017-02-16 ``@ddalle``: First version
"""

# Module to handle inputs and os interface
import sys
# Get the modules for tri files and surface grids
import cape.tri
# Command-line input parser
import cape.argread
# Read JSON file with comments
from cape.options.util import loadJSONFile

# Main function
def TriqFM(*a, **kw):
    """Extract forces and moments on one or more subsets of a TRIQ file
    
    Note that both *triq* and *comps* can be overwritten by keyword arguments
    even if specified using sequential arguments.
    
    :Call:
        >>> C = TriqFM(triq, *comps, **kw)
    :Inputs:
        *triq*: {``"grid.i.triq"``} | :class:`str`
            Name of TRIQ annotated triangulation file
        *comps*: {``None``} | :class:`list` (:class:`str` | :class:`int`)
            List of components to analyze (default is to entire surface)
        *tri*, *map*: {``None``} | :class:`str`
            Separate triangulation file to use for identifying *triq* subset
        *c*: {``None``} | :class:`str`
            Configuration (XML or JSON format) file to use
        *m*, *momentum*: ``True`` | {``False``}
            Include momentum (flow-through) forces in total
        *xMRP*: {``0.0``} | :class:`float`
            *x*-coordinate of moment reference point
        *yMRP*: {``0.0``} | :class:`float`
            *y*-coordinate of moment reference point
        *zMRP*: {``0.0``} | :class:`float`
            *z*-coordinate of moment reference point
        *MRP* {[*XMRP*, *YMRP*, *ZMRP*]} | :class:`list`
            Moment reference point
    :Outputs:
        *C*: :class:`dict` (:class:`float`)
            Dictionary of force and moment coefficients
    :Versions:
        * 2017-02-16 ``@ddalle``: First version
    """
   # -----------------
   # Sequential Inputs
   # -----------------
    # Get TRIQ file name
    if len(a) < 1:
        # Default
        ftriq = "grid.i.triq"
    else:
        # User-specified
        ftriq = a[0]
    # Get component list
    if len(a) < 2:
        # No components
        comps = None
    else:
        # List of components
        comps = a[1:]
   # --------------
   # JSON Settings
   # --------------
    # Check for JSON file
    fjson = kw.get("json")
    # Read if appropriate
    if fjson is None:
        # No JSON settings
        opts = {}
    else:
        # Read options from file
        opts = loadJSONFile(fjson)
    # Load settings from JSON file
    ftriq = opts.get("triq", opts.get("TriqFile", ftriq))
    ftri  = opts.get("tri",  opts.get("TriFile", opts.get("map")))
    fcfg  = opts.get("c",    opts.get("ConfigFile"))
    comps = opts.get("comps" opts.get("Components", opts.get("CompID",comps)))
    qm    = opts.get("m",    opts.get("Momentum", False))
    # Process moment reference point
    xMRP = opts.get("xMRP", 0.0)
    yMRP = opts.get("yMRP", 0.0)
    zMRP = opts.get("zMRP", 0.0)
    xMRP = kw.get("xMRP", xMRP)
    yMRP = kw.get("yMRP", yMRP)
    zMRP = kw.get("zMRP", zMRP)
    # Construct total MRP
    MRP = opts.get("MRP", [xMRP,yMRP,zMRP])
    MRP = kw.get("MRP", MRP)
   # --------------
   # Keyword Inputs
   # --------------
    # Read settings
    ftriq = kw.get("triq",  ftriq)
    ftri  = kw.get("tri",   ftri)
    ftri  = kw.get("map",   ftri)
    fcfg  = kw.get("c",     fcfg)
    comps = kw.get("comps", comps)
    qm    = kw.get("m",     kw.get("momentum", False))
    # Ensure list of components
    if type(comps).__name__ != "list": comps = [comps]
   # ------
   # Input
   # ------
    # Check for a tri file
    if ftri is None:
        # Read the input TRIQ file
        triq = cape.tri.Triq(ftriq, c=fcfg)
    else:
        # Read the TRIQ file
        triq = cape.tri.Triq(ftriq)
        # Read the TRI file
        tri = cape.tri.Tri(ftri, c=fcfg)
    # Initialize output
    FM = {}
   # ----------
   # Processing
   # ----------
   
        
    

# Main function
def MapTriqTri(*a, **kw):
    """Use a UH3D file to determine the family of each surface grid point
    
    :Call:
        >>> MapTriqTri(fjson, **kw)
        >>> MapTriqTri(**kw)
    :Sequential Inputs:
        *fjson*: :class:`str`
            Name of JSON file from which to read settings
    :Keyword Inputs:
        *triq*: {``"grid.i.triq"``} | :class:`str`
            Name of TRI/TRIQ file to read as input
        *uh3d*, *tri*: :class:`str`
            Name of input TRI/UH3D file to use for mapping components
        *ext*: {``None``} | i.triq | i.tri | triq | tri | uh3d
            File extension for outputs; by default copy from *triq*
        *label*: {``None``} | :class:`str`
            Infix to add to output file names
        *fmt*: {``None``} | ascii | b4 | b8 | lb4 | lb8
            File format; by default copy from *triq*
        *c*: :class:`str`
            (Optional) name of configuration file for labeling *tri* faces
        *v*: ``True`` | {``False``}
            Verbosity option
    :Versions:
        * 2017-02-10 ``@ddalle``: First version
    """
    # -----------------
    # Sequential Inputs
    # -----------------
    # Get the JSON file name
    if len(a) < 1:
        # No JSON file
        opts = {}
    else:
        # JSON file given
        fjson = a[0]
        # Read options
        opts = loadJSONFile(fjson)
    # Load settings from JSON file
    ftriq = opts.get("triq", opts.get("TriqFile", "grid.i.triq"))
    fuh3d = opts.get("uh3d", opts.get("tri", opts.get("TriFile")))
    fcfg  = opts.get("c",    opts.get("ConfigFile"))
    fout  = opts.get("o",    opts.get("OutputFile"))
    comps = opts.get("comps", opts.get("Components"))
    join  = opts.get("join",  opts.get("Join", False))
    ext = opts.get("ext",   opts.get("Extension"))
    lbl = opts.get("label", opts.get("Label", opts.get("Infix")))
    fmt = opts.get("fmt",   opts.get("Format"))
    v   = opts.get("v",     opts.get("Verbose", False))
    # --------------
    # Keyword Inputs
    # --------------
    # Check inputs that override sequential inputs
    ftriq = kw.get('triq', ftriq)
    fuh3d = kw.get('uh3d', fuh3d)
    fcfg  = kw.get('c',    fcfg)
    ext   = kw.get('ext',   ext)
    lbl   = kw.get('label', lbl)
    fmt   = kw.get('fmt',   fmt)
    ext   = kw.get('ext',   ext)
    join  = kw.get('join',  join)
    # Apply JSON settings passed through JSON
    kw.setdefault('join', join)
    kw.setdefault('ext', ext)
    kw.setdefault('v', v)
    # Process components
    kwcomps = kw.get("comps")
    # Process command-line component list if necessary
    if kwcomps is not None:
        # Split by commas
        comps = [comp.strip() for comp in kwcomps.split(",")]
    # Set components
    kw["comps"] = comps
    # --------------
    # Read Tri Files
    # --------------
    # Check for UH3D file
    if fuh3d is None:
        raise ValueError("No mapping triangulation specified")
    # Read the files
    triq = cape.tri.Tri(ftriq)
    tric = cape.tri.Tri(fuh3d, c=fcfg)
    # Process extension
    if ext is None:
        # Get parts of *triq* file name split by '.'
        prts = ftriq.split(".")
        # Process
        if prts[-1] not in ['triq', 'tri']:
            # Unusual file; write tri
            ext = "i.tri"
        elif len(prts) == 1:
            # I guess the file name is either "tri" or "triq" to get here
            ext = prts[-1]
        elif prts[-2] == "i":
            # Use the ".i" infix
            ext = '.'.join(prts[-2:])
        else:
            # Use either "tri" or "triq" without the "i"
            ext = prts[-1]
    # -------
    # Mapping
    # -------
    # Check for --join flag
    if join:
        # Check for an output name
        if fout is None:
            return ValueError("No file name for combined output file")
        # Extract all components
        triu = triq.ExtractMappedComps(tric, **kw)
        # Write file
        triu.Write(fout, **kw)
    else:
        # Extract individual components
        tris = triq.ExtractMappedComps(tric, **kw)
        # Loop through files
        for comp in tris:
            # Get output file name
            if lbl is None:
                # No infix
                fo = '%s.%s' % (comp, ext)
            else:
                # Add infix to output file name
                fo = '%s.%s.%s' % (comp, lbl, ext)
            # Write the file
            tris[comp].Write(fo, **kw)
    

# Only process inputs if called as a script!
if __name__ == "__main__":
    # Process the command-line interface inputs.
    a, kw = cape.argread.readkeys(sys.argv)
    # Check for a help option.
    if kw.get('h',False) or kw.get('help',False):
        import cape.text
        print(cape.text.markdown(__doc__)
        sys.exit()
    # Run the main function.
    MapTriqTri(*a, **kw)
    
