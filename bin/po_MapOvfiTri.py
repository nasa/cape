#!/usr/bin/env python
"""
Edit OVFI Namelist Family Names: ``po_MapOvfiTri.py``
=====================================================

Use the family names from a ``.uh3d`` file to determine which family

:Call:

    .. code-block:: console
    
        $ po_MapOvfiTri.py $grid $uh3d [OPTIONS]
        $ po_MapOvfiTri.py $grid $uh3d $ext [OPTIONS]
        $ po_MapOvfiTri.py [OPTIONS]
        $ pc_MapOvfiTri.py -h

:Inputs:
    * *grid*: Name of input grid (without) extensions
    * *uh3d*: Name of input '.uh3d' or '.tri' file
    * *ext*: Infix to add to edited '.ovfi' file {'ByPatch'}
    
:Options:
    -h, --help
        Display this help message and exit
        
    -v
        Run verbose version
        
    -grid $GRID, -pre $GRID
        Name of grid without extensions
        
    -tri UH3D, -uh3d UH3D
        Use *UH3D* file as input triangulation
       
    -c XML
        Use file *XML* to map component ID numbers
        
    -ext EXT
        Infix to add to edited ``.ovfi`` file {ByPatch}
        
    -i OVFI_IN
        Use *OVFI_IN* as input ``.ovfi`` namelist {$GRID.ovfi}
        
    -o OVFI_OUT
        Use *OVFI_OUT* as edited ``.ovfi`` namelist {$GRID.$EXT.ovfi}
        
    -srf SRF
        Use *SRF* as the surface mesh
        
    -atol *ATOL*, -AbsTol *ATOL*
        Absolute tolerance for nearest-tri search {_atol_}
        
    -rtol *RTOL*, -RelTol *RTOL*
        Tolerance for nearest-tri search relative to scale of tri {_rtol_}
        
    -ctol *CTOL*, -CompTol *CTOL*
        Tolerance for nearest-tri search relative to scale of comp {_ctol_}
        
    -antol *ANTOL*, -AbsProjTol *ANTOL*
        Absolute projection tolerance for nearest-tri search {_antol_}
        
    -rntol *RNTOL*, -RelProjTol *RNTOL*
        Projection tolerance relative to tri scale {_rntol_}
        
    -cntol *CNTOL*, -CompProjTol *CNTOL*
        Projection tolerance relative to component scale {_cntol_}
        
    -aftol *AFTOL*, -AbsFamilyTol *AFTOL*
        Absolute distance tol for secondary family {_aftol_}
        
    -rftol *RFTOL*, -RelFamilyTol *RFTOL*
        Distance tol for secondary family relative to tri scale {_rftol_}
    
    -cftol *CFTOL*, -CompFamilyTol *CFTOL*
        Distance tol for secondary family relative to comp scale {_cftol_}
        
    -anftol *ANFTOL*, -nftol, -ProjFamilyTol, -AbsProjFamilyTol *ANFTOL*
        Absolute projection tol for secondary family {_anftol_}
        
    -rnftol *RNFTOL*, -RelProjFamilyTol *RNFTOL*
        Projection tol for secondary family relative to tri scale {_rnftol_}
        
    -cnftol *CNFTOL*, -CompProjFamilyTol *CNFTOL*
        Projection tol for secondary family relative to comp scale {_cnftol_}

:Versions:
    * 2017-02-09 ``@ddalle``: First version
"""

# Module to handle inputs and os interface
import sys
# Get the modules for tri files and surface grids
import cape.tri
import cape.plot3d
# Command-line input parser
import cape.argread

# Edit docstring with actual default tolerances
__doc__.replace("_atol_",   str(cape.plot3d.atoldef))
__doc__.replace("_rtol_",   str(cape.plot3d.rtoldef))
__doc__.replace("_ctol_",   str(cape.plot3d.ctoldef))
__doc__.replace("_antol_",  str(cape.plot3d.antoldef))
__doc__.replace("_rntol_",  str(cape.plot3d.rntoldef))
__doc__.replace("_cntol_",  str(cape.plot3d.cntoldef))
__doc__.replace("_aftol_",  str(cape.plot3d.aftoldef))
__doc__.replace("_rftol_",  str(cape.plot3d.rftoldef))
__doc__.replace("_cftol_",  str(cape.plot3d.cftoldef))
__doc__.replace("_anftol_", str(cape.plot3d.anftoldef))
__doc__.replace("_rnftol_", str(cape.plot3d.rnftoldef))
__doc__.replace("_cnftol_", str(cape.plot3d.cnftoldef))

# Main function
def MapOvfiTri(*a, **kw):
    """Use a UH3D file to determine the family of each surface grid point
    
    :Call:
        >>> MapOvfiTri(grid, uh3d, ext='ByPatch', **kw)
        >>> MapOvfiTri(grid=None, uh3d=None, **kw)
    :Sequential Inputs:
        *grid*: :class:`str`
            Name of grid without file extension
        *uh3d*: :class:`str`
            Name of input tri file
        *ext*: {``"ByPatch"``} | :class:`str`
            Infix to add to output ``.ovfi`` file
    :Keyword Inputs:
        *grid*, *pre*: {``None``} | :class:`str`
            Name of grid without file extension (overrides sequential input)
        *uh3d*, *tri*: {``None``} | :class:`str`
            Name of triangulation file (overrides sequential input)
        *ext*: {``"ByPatch"``} | :class:`str`
            Infix to add to output ``.ovfi`` file (overrides sequential input)
        *c*: :class:`str`
            (Optional) name of configuration file to apply
        *v*: ``True`` | {``False``}
            Verbosity option
        *i*: {``"$grid.ovfi"``} | :class:`str`
            Full file name of input ``.ovfi`` namelist
        *o*: {``"$grid.$ext.ovfi"``} | :class:`str`
            Full file name of output ``.ovfi`` namelist
        *srf*: {``"$grid.srf"``} | :class:`str`
            Full file name of Plot3D surface mesh
    :Versions:
        * 2017-02-09 ``@ddalle``: First version
    """
    # -----------------
    # Sequential Inputs
    # -----------------
    # Get the grid name
    if len(a) < 1:
        grid = None
    else:
        grid = a[0]
    # Get the tri file
    if len(a) < 2:
        fuh3d = None
    else:
        fuh3d = a[1]
    # Get the infix
    if len(a) < 3:
        ext = "ByPatch"
    else:
        ext = a[2]
    # --------------
    # Keyword Inputs
    # --------------
    # Check inputs that override sequential inputs
    grid  = kw.get('grid', kw.get('pre', grid))
    fuh3d = kw.get('uh3d', kw.get('tri', fuh3d))
    ext   = kw.get('ext', ext)
    # Process file names based on grid
    srf = "%s.srf" % grid
    fi = "%s.ovfi" % grid
    fo = "%s.%s.ovfi" % (grid, ext)
    # Check for hard-coded file names
    srf = kw.get('srf', kw.get('surf', srf))
    fi = kw.get('i', fi)
    fo = kw.get('o', fo)
    # Configuration
    fxml = kw.get('c')
    # (No need to process tolerances)
    # ----------
    # Read Files
    # ----------
    # Read triangulation
    tri = cape.tri.Tri(fuh3d, c=fxml)
    # Read surface mesh
    srf = cape.plot3d.X(srf)
    # Map
    srf.MapOvfi(fi, fo, tri, **kw)
    

# Only process inputs if called as a script!
if __name__ == "__main__":
    # Process the command-line interface inputs.
    a, kw = cape.argread.readkeys(sys.argv)
    # Check for a help option.
    if kw.get('h',False) or kw.get('help',False):
        print __doc__
        sys.exit()
    # Run the main function.
    MapOvfiTri(*a, **kw)
    
