#!/usr/bin/env python
"""
Create command strings for binary modules
=========================================

This module creates system commands as lists of strings for binaries or scripts
that require multiple command-line options.  It is closely tied to
:mod:`cape.bin`.
"""

# Import getel feature
from cape.options.util import getel

# Function get aflr3 commands
def aflr3(opts=None, j=0, **kw):
    """Create AFLR3 system command as a list of strings
    
    :Call:
        >>> cmdi = aflr3(opts=None, j=0, **kw)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface, either global, "RunControl", or "aflr3"
        *j*: :class:`int` | ``None``
            Phase number
        *blc*: :class:`bool`
            Whether or not to generate prism layers
        *blr*: :class:`float`
            Boundary layer stretching option
        *blds*: :class:`float`
            Initial surface stretching
        *angblisimx*: :class:`float`
            Max BL intersection angle
    :Outputs:
        *cmdi*: :class:`list` (:class:`str`)
            System command created as list of strings
    :Versions:
        * 2016-04-04 ``@ddalle``: First version
    """
    # Check the input type.
    if opts is not None:
        # Get values
        fi         = opts.get_aflr3_i(j)
        fo         = opts.get_aflr3_o(j)
        blc        = opts.get_blc(j)
        blr        = opts.get_blr(j)
        blds       = opts.get_blds(j)
        angblisimx = opts.get_angblisimx(j)
    else:
        fi         = getel(kw.get('i'), j)
        fo         = getel(kw.get('o'), j)
        blc        = getel(kw.get('blc'), j)
        blr        = getel(kw.get('blr'), j)
        blds       = getel(kw.get('blds'), j)
        angblisimx = getel(kw.get('angblisimx'), j)
    # Initialize command
    cmdi = ['aflr3']
    # Start with input and output files
    if fi is None:
        raise ValueError("Input file to aflr3 required")
    else:
        cmdi += ['-i', fi]
    # Check for output file
    if fo is None:
        raise ValueError("Output file for aflr3 required")
    else:
        cmdi += ['-o', fo]
    # Process boolean settings
    if blc:    cmdi.append('-blc')
    # Process flag/value options
    if blr:    cmdi += ['-blr',  str(blr)]
    if blds:   cmdi += ['-blds', str(blds)]
    # Process options that come with an equal sign
    if angblisimx: cmdi += ['angblisimx=%s' % angblisimx]
    # Output
    return cmdi
