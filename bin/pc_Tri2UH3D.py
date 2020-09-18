#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Convert Cart3D Triangulation to UH3D Format: ``pc_Tri2UH3D.py``
===================================================================

Convert a Cart3D triangulation ``.tri`` file to a UH3D file.  The most
common purpose for this task is to inspect triangulations with moving
bodies with alternative software such as ANSA.

:Usage:
    .. code-block:: console
    
        $ pc_Tri2UH3D.py TRI [OPTIONS]
        $ pc_Tri2UH3D.py TRI UH3D [OPTIONS]
        $ pc_Tri2UH3D.py -tri TRI [-c CFG -uh3d UH3D]
        $ pc_Tri2UH3D.py -h

:Inputs:
    * *TRI*: Name of output ``.tri`` file
    * *UH3D*: Name of input ``.uh3d`` file
    * *CFG*: Name of configuration file: XML, JSON, or MIXSUR

:Options:
    -h, --help
        Display this help message and exit

    --tri TRI
        Use *TRI* as name of created output file

    -c CFG
        Use *CFG* as configuration file (defaults to :file:`Config.xml`)

    --uh3d UH3D
        Use *UH3D* as input file

If the name of the output file is not specified, the script will just
add ``.uh3d`` as the extension to the input (deleting ``.tri`` if
possible).

:Versions:
    * 2015-04-17 ``@ddalle``: Version 1.0
    * 2017-04-06 ``@ddalle``: Version 1.1: JSON and MIXSUR config files
"""

# Standard library
import os.path
import sys

# CAPE modules
import cape.argread
import cape.config
import cape.tri


# Main function
def Tri2UH3D(*a, **kw):
    r"""Convert a UH3D triangulation file to Cart3D tri format
    
    :Call:
        >>> Tri2UH3D(tri, uh3d, c='Config.xml', h=False)
        >>> Tri2UH3D(**kw)
    :Inputs:
        *tri*: :class:`str`
            Name of input file
        *uh3d*: :class:`str`
            Name of output file
        *c*: :class:`str`
            Name of configuration 
        *h*: ``True`` | {``False``}
            Display help and exit if ``True``
    :Versions:
        * 2015-04-17 ``@ddalle``: Version 1.0
    """
    # Get the file pyCart settings file name.
    if len(a) == 0:
        # Defaults
        ftri = None
    else:
        # Use the first general input.
        ftri = a[0]
    # Prioritize a "-i" input.
    ftri = kw.get('tri', kw.get('i', ftri))
    # Must have a file name.
    if ftri is None:
        # Required input.
        import cape.text
        print(cape.text.markdown(__doc__))
        raise IOError("At least one input required.")
    
    # Get the file pyCart settings file name.
    if len(a) <= 2:
        # Defaults
        fuh3d = ftri.rstrip('tri') + 'uh3d'
    else:
        # Use the first general input.
        fuh3d = a[1]
    # Prioritize a "-i" input.
    fuh3d = kw.get('uh3d', kw.get('o', fuh3d))
    # Config file
    fcfg = kw.get('c', "Config.xml")
        
    # Read in the TRI file
    if os.path.isfile(fcfg):
        # Apply the configuration file
        tri = cape.tri.Tri(ftri, c=fcfg)
    else:
        # No configuration file
        tri = cape.tri.Tri(ftri)
    
    # Write it.
    tri.WriteUH3D(fuh3d)


# Only process inputs if called as a script!
if __name__ == "__main__":
    # Process the command-line interface inputs.
    (a, kw) = cape.argread.readkeys(sys.argv)
    # Check for a help option.
    if kw.get('h',False) or kw.get('help',False):
        import cape.text
        print(cape.text.markdown(__doc__))
        sys.exit()
    # Run the main function.
    Tri2UH3D(*a, **kw)

