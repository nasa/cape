#!/usr/bin/env python
"""
Write SPLITMQ Input File Using OVERFLOW Namelist
================================================

Create a ``splitmq`` input file for extracting the L=1 surface from all walls
using an OVERFLOW namelist as grid information about surface grids.

By changing the first two lines of the resulting file, an input file for
``splitmx`` can also be created.

:Call:

    .. code-block:: console
    
        $ po_WriteSplitmq.py $NML $SPLITMQ [OPTIONS]

:Inputs:
    * *NML*: Name of OVERFLOW namelist file
    * *SPLITMQ*: Name of output ``splitmq`` input file
    
:Options:
    -h, --help
        Display this help message and exit
        
    -i NML
        Use *NML* as input namelist file {``overflow.inp``}
        
    -o SPLITMQ
        Create *SPLITMQ* as output {``splitmq.i``}
        
    --wall
        Only include walls (by default, thrust BCs are also included)

:Versions:
    * 2016-12-19 ``@ddalle``: First version
"""

# Module to handle inputs and os interface
import sys
import numpy as np
# Command-line input parser
import cape.argread as argr
# Get the Tecplot module.
import pyOver.overNamelist

# Main function
def Nml2Splitmq(*a, **kw):
    """Convert a Tecplot PLT file to a Cart3D annotated triangulation (TRIQ)
    
    :Call:
        >>> Nml2Splitmq(fnml)
        >>> Nml2Splitmq(fnml, fsplitmq)
        >>> Nml2Splitmq(i=fnml, o=fsplitmq)
    :Inputs:
        *fnml*: {``"overflow.inp"``} | :class:`str`
            Name of OVERFLOW namelist input file
        *spitmq*: {``"splitmq.i"``} | :class:`str`
            Name of ``splitmq`` input file
        *wall*: ``True`` | {``False``}
            Only include walls if ``True``; otherwise include thrust BCs, too
    :Versions:
        * 2017-01-04 ``@ddalle``: First version
    """
    # Get the file name.
    if len(a) == 0:
        # Defaults
        fnml = "overflow.inp"
    else:
        # Use the first general input.
        fnml = a[0]
    # Prioritize a "-i" input.
    fnml = kw.get('i', fnml)
    # Get the output file name if given as second input
    if len(a) < 2:
        # Default
        fsplitmq = "splitmq.i"
    else:
        # Use the second general input.
        fsplitmq = a[1]
    # Prioritize a "-i" input.
    fsplitmq = kw.get('o', fsplitmq)
    # Read wall input
    wall = kw.get('wall', False)
    
    # Read the namelist
    nml = pyOver.overNamelist.OverNamelist(fnml)
    
    # Write a splitmq file
    nml.WriteSplitmqI(fsplitmq, wall=wall)
    

# Only process inputs if called as a script!
if __name__ == "__main__":
    # Process the command-line interface inputs.
    (a, kw) = argr.readkeys(sys.argv)
    # Check for a help option.
    if kw.get('h',False) or kw.get('help',False):
        print __doc__
        sys.exit()
    # Run the main function.
    Nml2Splitmq(*a, **kw)
    
