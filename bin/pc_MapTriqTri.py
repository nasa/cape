#!/usr/bin/env python
"""
Extract Mapped Components of TRIQ: ``pc_MapTriqTri.py``
=======================================================

Use the family names from a ``.uh3d`` file to extract components from a
``.tri`` (triangulation) or ``.triq`` (annotated triangulation) and write them
to file.  This can be used to extract a panel from a solution ``.triq`` file
that does not have that panel labeled as a separate face.

:Usage:

    .. code-block:: console
    
        $ pc_MapTriqTri.py $json [OPTIONS]
        $ pc_MapTriqTri.py -h

:Inputs:
    * *JSON*: Name of JSON file from which to read settings
    
:Options:
    -h, --help
        Display this help message and exit
        
    -v
        Run verbose version
        
    --json JSON
        Use commented JSON file called *JSON*
        
    --triq TRIQ
        Use *TRIQ* as unmapped input file {``grid.i.triq``}
        
    --tri UH3D, --uh3d UH3D
        Use *UH3D* file as the triangulation for mapping
        
    --join
        Write single combined output file instead of one for each component
        
    -o $FOUT
        If using the ``-join`` flag, write output as *FOUT*
        
    --ext EXT
        File extension, by default copied from *TRIQ*
        
    --label LBL
        Infix to use in file names create; files will be either
        ``"$COMP.$LBL.$EXT"`` or ``"$COMP.$EXT"`` {""}
        
    --fmt FMT
        Output file format, ``{} | ascii | b4 | lb4``; if not specified, copy
        format from *TRIQ*
        
    -b4
        Write single-precision big-endian output (equiv. to ``--ext b4``)
        
    -lb4
        Write single-precision little-endian output (equiv. to ``--ext lb4``)
        
    --atol ATOL, --AbsTol ATOL
        Absolute tolerance for nearest-tri search {_atol_}
        
    --rtol RTOL, --RelTol RTOL
        Tolerance for nearest-tri search relative to scale of tri {_rtol_}
        
    --ctol CTOL, --CompTol CTOL
        Tolerance for nearest-tri search relative to scale of comp {_ctol_}
        
    --antol ANTOL, --AbsProjTol ANTOL
        Absolute projection tolerance for nearest-tri search {_antol_}
        
    --rntol RNTOL, --RelProjTol RNTOL
        Projection tolerance relative to tri scale {_rntol_}
        
    --cntol CNTOL, --CompProjTol CNTOL
        Projection tolerance relative to component scale {_cntol_}
        
    --aftol AFTOL, --AbsFamilyTol AFTOL
        Absolute distance tol for secondary family {_aftol_}
        
    --rftol RFTOL, --RelFamilyTol RFTOL
        Distance tol for secondary family relative to tri scale {_rftol_}
    
    --cftol CFTOL, --CompFamilyTol CFTOL
        Distance tol for secondary family relative to comp scale {_cftol_}
        
    --anftol ANFTOL, --nftol, --ProjFamilyTol, --AbsProjFamilyTol ANFTOL
        Absolute projection tol for secondary family {_anftol_}
        
    --rnftol RNFTOL, --RelProjFamilyTol RNFTOL
        Projection tol for secondary family relative to tri scale {_rnftol_}
        
    --cnftol CNFTOL, --CompProjFamilyTol CNFTOL
        Projection tol for secondary family relative to comp scale {_cnftol_}

:Versions:
    * 2017-02-10 ``@ddalle``: First version
"""

# Module to handle inputs and os interface
import sys
# Get the modules for tri files and surface grids
import cape.tri
import cape.plot3d
# Command-line input parser
import cape.argread
import cape.text
# Read JSON file with comments
from cape.options.util import loadJSONFile

# Edit docstring with actual default tolerances
__doc__ = cape.text.setdocvals(__doc__, {
    "atol":   cape.plot3d.atoldef,
    "rtol":   cape.plot3d.rtoldef,
    "ctol":   cape.plot3d.ctoldef,
    "antol":  cape.plot3d.antoldef,
    "rntol":  cape.plot3d.rntoldef,
    "cntol":  cape.plot3d.cntoldef,
    "aftol":  cape.plot3d.aftoldef,
    "rftol":  cape.plot3d.rftoldef,
    "cftol":  cape.plot3d.cftoldef,
    "anftol": cape.plot3d.anftoldef,
    "rnftol": cape.plot3d.rnftoldef,
    "cnftol": cape.plot3d.cnftoldef
})

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
        print(cape.text.markdown(__doc__))
        sys.exit()
    # Run the main function.
    MapTriqTri(*a, **kw)
    
