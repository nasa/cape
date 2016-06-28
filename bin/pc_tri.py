#!/usr/bin/env python
"""
Convert UH3D triangulation to Cart3D format: :mod:`pc_UH3D2Tri`
===============================================================

Convert a '.uh3d' file to a Cart3D triangulation format.

:Call:

    .. code-block:: console
    
        $ pc_UH3D2Tri.py $uh3d
        $ pc_UH3D2Tri.py -i $uh3d
        $ pc_UH3D2Tri.py $uh3d $tri
        $ pc_UH3D2Tri.py -i $uh3d -o $tri
        $ pc_UH3D2Tri.py -h

:Inputs:
    * *uh3d*: Name of input '.uh3d' file
    * *tri*: Name of output '.tri' file
    
:Options:
    -h, --help
        Display this help message and exit
        
    -i UH3D
        Use *UH3D* as input file
        
    -o TRI
        Use *TRI* as name of created output file
       
    -c XML
        Use file *XML* to map component ID numbers
        
    -xtol XTOL
        Truncate nodal coordinates within *XTOL* of x=0 plane to zero
        
    -ytol YTOL
        Truncate nodal coordinates within *YTOL* of y=0 plane to zero
        
    -ztol ZTOL
        Truncate nodal coordinates within *ZTOL* of z=0 plane to zero
    
If the name of the output file is not specified, it will just add '.tri' as the
extension to the input (deleting '.uh3d' if possible).

:Versions:
    * 2016-02-29 ``@ddalle``: First version
"""

# Module to handle inputs and os interface
import sys
# Modules from Cape
import cape.tri
import cape.argread as argr

# Read a tri file
def ReadTri(**kw):
    """Read a triangulation file according to command-line inputs
    
    :Call:
        >>> tri = ReadTri(i=None, fmt=None, c=None, bc=None)
    :Inputs:
        *i*: {``None``} | :class:`str`
            Input file specified with ``-i`` flag, takes priority over *fname*
        *fmt*: {``None``} | "tri" | "uh3d" | "surf" | "unv" | "triq"
            Format, by default, determine from the extension
        *c*: {``None``} | :class:`str`
            (Optional) name of configuration file to apply
        *bc*: {``None``} | :class:`str`
            (Optional) Name of AFLR3 boundary condition file
    :Outputs:
        *tri*: :class:`cape.tri.Tri`
            Triangulation interface
    :Versions:
        * 2016-06-28 ``@ddalle``: First version
    """
    # Prioritize a "-i" input.
    fname = kw.get('i')
    # Must have a file name.
    if fname is None:
        # Required input.
        raise IOError("Input file could not be determined.")
        
    # Get format
    fmt = kw.get('fmt')
    # Read the tri file
    tri = cape.tri.ReadTriFile(fname, fmt=fmt)
    
    # Get configuration file option
    fxml = kw.get('c')
    # Check for a config file.
    if fxml is None:
        # Nothing to read
        cfg = None
    else:
        # Read config file
        cfg = cape.tri.Config(fxml)
    # Apply configuration
    if cfg is not None:
        tri.ApplyConfig(cfg)
    
    # Check for AFLR3 bcs
    fbc = kw.get('bc')
    # If present, map it.
    if fbc is not None:
        # Map boundary conditions
        self.tri.ReadBCs_AFLR3(fbc)
    
    # Output
    return tri
        
# Manipulation
def TransformTri(tri, **kw):
    """Perform manipulations to the geometry file
    
    :Call:
        >>> TransformTri(tri, **kw)
    :Inputs:
        *tri*: :class:`cape.tri.Tri`
            Triangulation instance
        *dx*: {``None``} | :class:`float`
            Shift the coordinates in the *x* direction
        *dy*: {``None``} | :class:`float`
            Shift the coordinates in the *y* direction
        *dz*: {``None``} | :class:`float`
            Shift the coordinates in the *z* direction
        *x0*: {``0.0``} | :class:`float`
            Target value of *x* to use for trimming
        *y0*: {``0.0``} | :class:`float`
            Target value of *y* to use for trimming
        *z0*: {``0.0``} | :class:`float`
            Target value of *z* to use for trimming
        *xtol*: {``None``} | :class:`float`
            Force all points within *xtol* of *x0* to be at exactly *x0*
        *ytol*: {``None``} | :class:`float`
            Force all points within *ytol* of *y0* to be at exactly *y0*
        *ztol*: {``None``} | :class:`float`
            Force all points within *ztol* of *z0* to be at exactly *z0*
    :Versions:
        * 2016-06-28 ``@ddalle``: First version
    """
    # Get shift values
    dx = kw.get('dx')
    dy = kw.get('dy')
    dz = kw.get('dz')
    # Apply translations
    if dx is not None: tri.Nodes[:,0] += float(dx)
    if dy is not None: tri.Nodes[:,1] += float(dy)
    if dz is not None: tri.Nodes[:,2] += float(dz)
    
    # Check for tolerances
    xtol = kw.get('xtol')
    ytol = kw.get('ytol')
    ztol = kw.get('ztol')
    # Target valuex
    x0 = kw.get('x0', 0.0)
    y0 = kw.get('y0', 0.0)
    z0 = kw.get('z0', 0.0)
    # Apply tolerances
    if xtol is not None:
        tri.Nodes[abs(tri.Nodes[:,0]-x0)<=float(xtol), 0] = x0
    if ytol is not None:
        tri.Nodes[abs(tri.Nodes[:,1]-y0)<=float(ytol), 1] = y0
    if ztol is not None:
        tri.Nodes[abs(tri.Nodes[:,2]-z0)<=float(ztol), 2] = z0
    
    
# Write triangulation
def WriteTri(tri, **kw):
    """Write manipulated triangulation file
    
    The default format is determined by the output file name if specified.  If
    no output is specified, the input file is used; this may result in
    overwriting th
    
    :Call:
        >>> tri = WriteTri(tri, i=None, fmt=None, c=None)
    :Inputs:
        *i*: {``None``} | :class:`str`
            Input file specified with ``-i`` flag, takes priority over *fin*
        *o*: {``None``} | :class:`str`
            Input file specified with ``-i`` flag, takes priority over *fout*
        *fmt*: {``None``} | "tri" | "uh3d" | "surf" | "stl" | "triq"
            Format, by default, determine from the extension
        *c*: {``None``} | :class:`str`
            (Optional) name of configuration file to apply
        *bc*: {``None``} | :class:`str`
            (Optional) Name of AFLR3 boundary condition file
    :Outputs:
        *tri*: :class:`cape.tri.Tri`
            Triangulation interface
    :Versions:
        * 2016-06-28 ``@ddalle``: First version
    """
    # Get input and output file names
    fin = kw.get('i')
    fout = kw.get('o', fin)
    
    # Check for directly-set format
    fmt = kw.get('fmt')
    # Determine final output format and file name
    if fmt is None:
        # Use the file name; split based on '.'
        fpart = fout.split('.')
        # Get the extension
        if len(fpart) < 2:
            # Odd case, no extension given
            fmt = 'tri'
        else:
            # Get the extension
            fmt = fpart[-1].lower()
    else:
        # Use specified format in file name
        if len(a) < 2 and 'o' not in kw:
            # Strip extension from current output file name
            fout = '.'.join(fout.split('.')[:-1])
            # Add the given extension
            fout = '%s.%s' % (fout, fmt)
    # Check for overwrite
    if fout == fin:
        raise IOError("Conversion would overwrite original triangulation")
    # Write
    if fmt == 'stl':
        # Write a simple STL file
        tri.WriteSTL(fout)
    elif fmt == 'uh3d':
        # Write a UH3D file
        tri.WriteUH3D(fout)
    elif fmt == 'surf':
        # Write a AFLR3 .surf file
        tri.WriteSurf(fout)
    elif fmt == 'triq':
        # Write a triq file with solution
        tri.WriteTriq(fout)
    else:
        # Write a tri file
        tri.Write(fout)
    

# Only process inputs if called as a script!
if __name__ == "__main__":
    # Process the command-line interface inputs.
    (a, kw) = argr.readkeys(sys.argv)
    # Check for a help option.
    if kw.get('h',False) or kw.get('help',False):
        print __doc__
        sys.exit()
        
    # Set input files
    if len(a) > 0: kw.setdefault('i', a[0])
    if len(a) > 1: kw.setdefault('o', a[1])
        
    # Read
    tri = ReadTri(**kw)
    # Convert/manipulate
    TransformTri(tri, **kw)
    # Write
    WriteTri(tri, **kw)
    
