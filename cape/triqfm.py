#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.triqfm`: Command-line interface to ``triqfm`` tool
==============================================================

This function provides the Python function :func:`triqfm` that executes
the TriqFM functionality and the function :func:`main` that accesses
this function from the command line.

"""

# Standard library
import json
import sys

# Third-party modules
import numpy as np

# Local imports
from . import argread
from . import text as textutils
from .tri import Tri, Triq
from .cfdx.options.util import loadJSONFile


HELP_TRIQFM = r"""
``triqfm``: Calculate forces and moments on a ``.triq`` file
=============================================================

Calculate the integrated forces and moments on a triangulated surface.
This can calculate the forces and moments on one or many subcomponents
of the annotated surface triangulation and write all the results to
parametrically named files.

:Usage:
    .. code-block:: console
    
        $ triqfm TRIQ COMP1 COMP2 [...] [OPTIONS]
        $ triqfm -h

:Inputs:
    * *TRIQ*: Name of annotated triangulation (TRIQ) file
    * *COMP1*: Name or 1-based index of first component to analyze
    * *COMP2*: Name or 1-based index of second component to analyze
    
:Examples:
    
    Calculate the forces and moments on the full geometry in
    ``"grid.i.triq"``, which is a solution calculated at Mach 0.75.
    
        .. code-block:: console
        
            $ triqfm grid.i.triq --mach 0.75
            $ triqfm -m 0.75
    
:Options:
    -h, --help
        Display this help message and exit
        
    --json JSON
        Read options from commented JSON file called *JSON*
        
    --triq TRIQ
        Use *TRIQ* as unmapped input file {"grid.i.triq"}
        
    --comps COMPS
        List of components to analyze separated by commas
        
    --tri TRI, --map TRI
        Use a separate triangulation (Cart3D tri, UH3D, AFLR3 surf,
        IDEAS unv, or other TRIQ format) as a map for which triangles to
        extract; if used, the component list *COMPS* and config file
        *CONFIG* apply to this file
        
    --momentum
        Include momentum forces in total
        
    -m MACH, --mach MACH
        Use *MACH* as the freestream Mach number {1.0}
        
    --Re REY, --Rey REY
        Reynolds number per grid unit {1.0}
    
    --Aref AREF, --RefArea AREF
        Reference area for coefficient computation {1.0}
        
    --Lref LREF, --RefLength LREF
        Reference length for coefficient computation {1.0}
        
    --bref BREF, --RefSpan BREF
        Reference span for coefficient computation {*LREF*}
        
    --xMRP XMRP
        Set *x*-coordinate of moment reference point {0.0}
        
    --yMRP YMRP
        Set *y*-coordinate of moment reference point {0.0}
        
    --zMRP ZMRP
        Set *z*-coordinate of moment reference point {0.0}
        
    --MRP MRP
        Moment reference point {"*XMRP*, *YMRP*, *ZMRP*"} 
        
:Versions:
    * 2017-02-17 ``@ddalle``: Version 1.0
"""


# Main function
def triqfm(*a, **kw):
    r"""Extract forces and moments from a TRIQ file
    
    Note that both *triq* and *comps* can be overwritten by keyword
    arguments even if specified using positional arguments.
    
    :Call:
        >>> C = triqfm(triq, *comps, **kw)
    :Inputs:
        *triq*: {``"grid.i.triq"``} | :class:`str`
            Name of TRIQ annotated triangulation file
        *comps*: {``None``} | :class:`list` (:class:`str` | :class:`int`)
            List of components to analyze (default is to entire surface)
        *tri*, *map*: {``None``} | :class:`str`
            Separate triangulation file to use for identifying *triq* subset
        *c*: {``None``} | :class:`str`
            Configuration (XML or JSON format) file to use
        *incm*, *momentum*: ``True`` | {``False``}
            Include momentum (flow-through) forces in total
        *m*, *mach*: {``1.0``} | :class:`float`
            Freestream Mach number
        *RefArea*, *Aref*: {``1.0``} | :class:`float`
            Reference area
        *RefLength*, *Lref*: {``1.0``} | :class:`float`
            Reference length (longitudinal)
        *RefSpan*, *bref*: {*Lref*} | :class:`float`
            Reference span (for rolling and yawing moments)
        *Re*, *Rey*: {``1.0``} | :class:`float`
            Reynolds number per grid unit (units same as *triq.Nodes*)
        *gam*, *gamma*: {``1.4``} | :class:`float` > 1
            Freestream ratio of specific heats
        *xMRP*: {``0.0``} | :class:`float`
            *x*-coordinate of moment reference point
        *yMRP*: {``0.0``} | :class:`float`
            *y*-coordinate of moment reference point
        *zMRP*: {``0.0``} | :class:`float`
            *z*-coordinate of moment reference point
        *MRP* {[*XMRP*, *YMRP*, *ZMRP*]} | :class:`list`
            Moment reference point
    :Outputs:
        *FM*: :class:`dict`\ [:class:`float`]
            Dictionary of force and moment coefficients
    :Versions:
        * 2017-02-16 ``@ddalle``: Version 1.0; :func:`TriqFM`
        * 2021-10-14 ``@ddalle``: Version 1.1; in :mod:`cape.triqfm`
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
    ftriq = opts.get("triq",  opts.get("TriqFile", ftriq))
    ftri  = opts.get("tri",   opts.get("TriFile", opts.get("map")))
    fcfg  = opts.get("c",     opts.get("ConfigFile"))
    fo    = opts.get("o",     opts.get("OutputFile"))
    comps = opts.get(
        "comps", opts.get("Components", opts.get("CompID", comps)))
    incm  = opts.get("incm",  opts.get("Momentum", False))
    # Freestream conditions
    mach = opts.get("m",    opts.get("mach",  opts.get("Mach",     1.0)))
    Rey  = opts.get("Re",   opts.get("Rey",   opts.get("Reynolds", 1.0)))
    gam  = opts.get("gam",  opts.get("gamma", opts.get("Gamma",    1.4)))
    # Scale
    Aref = opts.get("Aref", opts.get("RefArea",   1.0))
    Lref = opts.get("Lref", opts.get("RefLength", 1.0))
    bref = opts.get("bref", opts.get("RefSpan"))
    # Process moment reference point
    xMRP = float(opts.get("xMRP", kw.get("xMRP", 0.0)))
    yMRP = float(opts.get("yMRP", kw.get("yMRP", 0.0)))
    zMRP = float(opts.get("zMRP", kw.get("zMRP", 0.0)))
    # Construct total MRP
    MRP = opts.get("MRP", [xMRP, yMRP, zMRP])
    # If given as string input, we have to do some work
    if "MRP" in kw:
        # Split CLI text by comma
        MRP = [float(x) for x in kw["MRP"].split(",")]
   # --------------
   # Keyword Inputs
   # --------------
    # Read settings
    ftriq = kw.get("triq",  ftriq)
    ftri  = kw.get("tri",   ftri)
    ftri  = kw.get("map",   ftri)
    fcfg  = kw.get("c",     fcfg)
    fo    = kw.get("o",     fo)
    incm  = kw.get("incm",  kw.get("momentum", incm))
    # Check for components from kwargs
    kwcomps = kw.get("comps", kw.get("comps", "")).split(",")
    if len(kwcomps) > 0:
        comps = kwcomps
    # Default output file name
    if fo is None:
        # Strip suffixes
        if ftriq.endswith(".i.triq"):
            # grid.i.triq -> grid.json
            fo = ftriq[:-6] + "json"
        elif ftriq.endswith(".triq"):
            # grid.triq -> grid.json
            fo = ftriq[:-4] + "json"
        else:
            # grid.uh3d -> grid.uh3d.json
            fo = ftriq + ".json"
    # Read conditions
    mach = kw.get("m",   kw.get("mach",  mach))
    Rey  = kw.get("Re",  kw.get("Rey",   Rey))
    gam  = kw.get("gam", kw.get("gamma", gam))
    # Read scales
    Aref = kw.get("Aref", kw.get("RefArea",   Aref))
    Lref = kw.get("Lref", kw.get("RefLength", Lref))
    bref = kw.get("bref", kw.get("RefSpan",   bref))
    # Fallback reference span
    if bref is None:
        bref = Lref
    # Ensure list of components
    if not isinstance(comps, list):
        comps = [comps]
   # ------
   # Input
   # ------
    # Check for a tri file
    if ftri is None:
        # Read the input TRIQ file
        triq = Triq(ftriq, c=fcfg)
        # No component map
        compmap = {}
    else:
        # Read the unmapped TRIQ file
        triq = Triq(ftriq)
        # Read the TRI file
        tri = Tri(ftri, c=fcfg)
        # Map the component IDs
        compmap = triq.MapTriCompID(tri, v=True)
    # Initialize output
    FM = {}
   # ----------
   # Processing
   # ----------
    # Set inputs for TriqForces
    kwfm = {
        "m":    float(mach),
        "Re":   float(Rey),
        "gam":  float(gam),
        "Aref": float(Aref),
        "Lref": float(Lref),
        "bref": float(bref),
        "incm": incm
    }
    # Loop through components
    for comp in comps:
        # Process component
        if isinstance(comp, (list, np.ndarray)):
            # Make up a name
            cname = str(comp[0])
            # Translate component numbers if needed
            comp = [compmap.get(k, k) for k in comp]
        elif (comp is None) or (comp == ""):
            # Name of component
            cname = "entire"
            comp = None
            # Which components to process
            if ftri is not None:
                # Get the list of components from the mapping tri
                comp = compmap.values()
        else:
            # Use the name directly
            cname = str(comp)
            # If the component is an integer, make sure we use the map
            comp = compmap.get(comp, comp)
        # Read the forces and moments right from the TRIQ file
        FMc = triq.GetTriForces(comp, **kwfm)
        # Save it
        FM[cname] = FMc
   # ------
   # Output
   # ------
    # Open the output file
    with open(fo, 'w') as fp:
        # Dump the results
        json.dump(FM, fp, indent=1)
    # Output
    return FM
        

# Only process inputs if called as a script!
def main():
    # Process the command-line interface inputs.
    a, kw = argread.readkeys(sys.argv)
    # Check for a help option.
    if kw.get('h', False) or kw.get("help", False):
        print(textutils.markdown(HELP_TRIQFM))
        return
    # Run the main function
    triqfm(*a, **kw)
    
