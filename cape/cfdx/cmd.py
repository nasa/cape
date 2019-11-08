#!/usr/bin/env python
"""
:mod:`cape.cfdx.cmd`: Creating system commands 
=================================================

This module creates system commands as lists of strings for binaries or scripts
that require multiple command-line options.  It is closely tied to
:mod:`cape.bin`.

Commands are created in the form of a list of strings.  This is the format used
in :mod:`subprocess` commands (with *shell*\ =``False``\ ) and also
:func:`cape.bin.calli`, etc.  As a very simple example, the system command
``"ls -lh"`` becomes the list ``["ls", "-lh"]``. 

Inputs to the functions in this module take one of two forms:

    * A :class:`cape.options.Options` object or subset thereof
    * Keyword arguments
    
The first method allows any appropriate :class:`Options` interface and then
extracts the appropriate portion. For instance :func:`verify` can be given
either a top-level options object, a ``"RunControl"`` options interface (e.g.
from :mod:`cape.options.runControl`), or a :class:`dict` of options specific to
``verify``.  It does this using the following Python commands:

    .. code-block:: python
    
        opts = opts.get("RunControl", opts)
        opts = opts.get("verify", opts)
        
Other functions, such as :func:`aflr3`, rely on the built-in methods of the
:class:`cape.options.Options` class.  For example, ``opts.get_aflr3_i(j)``
returns the ``aflr3`` input file name for phase *j* if *opts* is in any of the
following classes:

    ============================================  =======================
    Class                                         Description
    ============================================  =======================
    :class:`cape.options.Options`                 All Cape settings
    :class:`cape.options.runControl.RunControl`   All run settings
    :class:`cape.options.aflr3.aflr3`             AFLR3 settings
    ============================================  =======================

"""

# Import getel feature
from cape.cfdx.options.util import getel

# Functions to get system command names
from ..util import GetTecplotCommand


# Function to run tecplot
def tecmcr(mcr="export-lay.mcr", **kw):
    """
    Run a Tecplot macro
    
    :Call:
        >>> cmd = pyCart.cmd.tecmcr(mcr="export-lay.mcr")
    :Inputs:
        *mcr*: :class:`str`
            File name of Tecplot macro
    :Outputs:
        *cmd*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2015-03-10 ``@ddalle``: First version
    """
    # Get tecplot command
    t360 = GetTecplotCommand()
    # Form the command
    cmd = [t360, '-b', '-p', mcr]
    # Output
    return cmd


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
        *cdfr*: :class:`float`
            Maximum geometric stretching
        *cdfs*: {``None``} | 0 <= :class:`float` <=10
            Distribution function exclusion zone
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
        blc        = opts.get_aflr3_blc(j)
        blr        = opts.get_aflr3_blr(j)
        bli        = opts.get_aflr3_bli(j)
        blds       = opts.get_aflr3_blds(j)
        grow       = opts.get_aflr3_grow(j)
        cdfr       = opts.get_aflr3_cdfr(j)
        cdfs       = opts.get_aflr3_cdfs(j)
        nqual      = opts.get_aflr3_nqual(j)
        mdsblf     = opts.get_aflr3_mdsblf(j)
        angqbf     = opts.get_aflr3_angqbf(j)
        angblisimx = opts.get_aflr3_angblisimx(j)
        # Extra options
        flags = opts.get_aflr3_flags()
        keys  = opts.get_aflr3_keys()
    else:
        # Get command-line options
        fi         = getel(kw.get('i'), j)
        fo         = getel(kw.get('o'), j)
        bli        = getel(kw.get('bli'), j)
        blc        = getel(kw.get('blc'), j)
        blr        = getel(kw.get('blr'), j)
        blds       = getel(kw.get('blds'), j)
        grow       = getel(kw.get('grow'), j)
        cdfr       = getel(kw.get('cdfr'), j)
        cdfs       = getel(kw.get('cdfs'), j)
        nqual      = getel(kw.get('nqual'), j)
        mdsblf     = getel(kw.get('mdsblf'), j)
        angqbf     = getel(kw.get('angqbf'), j)
        angblisimx = getel(kw.get('angblisimx'), j)
        # Extra options
        flags = kw.get('flags', {})
        keys  = kw.get('keys',  {})
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
    if bli:    cmdi += ['-bli',  str(bli)]
    if blr:    cmdi += ['-blr',  str(blr)]
    if blds:   cmdi += ['-blds', str(blds)]
    if grow:   cmdi += ['-grow', '%i' % grow]
    # Process options that come with an equal sign
    if cdfs       is not None: cmdi += ['cdfs=%s' % cdfs]
    if cdfr       is not None: cmdi += ['cdfr=%s' % cdfr]
    if angqbf     is not None: cmdi += ['angqbf=%s' % angqbf]
    if angblisimx is not None: cmdi += ['angblisimx=%s' % angblisimx]
    # Options that can be None
    if mdsblf is not None: cmdi += ['-mdsblf', str(mdsblf)]
    if nqual  is not None: cmdi += ['nqual=%i' % nqual]
    # Loop through flags
    for k, v in flags.items():
        # Check type
        if isinstance(v, bool):
            # Check Boolean value
            if v:
                # Add '-$k' to command
                cmdi += ['-%s' % k]
        elif v is not None:
            # Convert to string, '-$k $v'
            cmdi += ['-%s' % k, '%s' % v]
    # Loop though keys
    for k, v in keys.items():
        # Skip if None
        if v is None:
            continue
        # Append to command
        cmdi += ['%s=%s' % (k, v)]
    # Output
    return cmdi


# Function to call verify
def intersect(opts=None, j=0, **kw):
    """Interface to Cart3D binary ``intersect``
    
    :Call:
        >>> cmd = cape.cmd.intesect(opts=None, **kw)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface
        *i*: :class:`str`
            Name of input ``tri`` file
        *o*: :class:`str`
            Name of output ``tri`` file
    :Outputs:
        *cmd*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2015-02-13 ``@ddalle``: First version
    """
    # Check input type
    if opts is not None:
        # Check for "RunControl"
        opts = opts.get("RunControl", opts)
        opts = opts.get("intersect",  opts)
        # Get settings
        fin    = opts.get_key('i', j)
        fout   = opts.get_key('o', j)
        cutout = opts.get_key('cutout', j)
        ascii  = opts.get_key('ascii', j)
        v      = opts.get_key('v', j)
        T      = opts.get_key('T', j)
        iout   = opts.get_key('intersect', j)
        # Defaults
        if fin    is None: fin    = 'Components.tri'
        if fout   is None: fout   = 'Components.i.tri'
        if cutout is None: cutout = None
        if ascii  is None: ascii  = True
        if v      is None: v      = False
        if T      is None: T      = False
        if iout   is None: iout   = False
    else:
        # Get settings from inputs
        fin    = kw.get('i', 'Components.tri')
        fout   = kw.get('o', 'Components.i.tri')
        cutout = kw.get('cutout')
        ascii  = kw.get('ascii', True)
        v      = kw.get('v', False)
        T      = kw.get('T', False)
        iout   = kw.get('intersect', False)
    # Build the command.
    cmdi = ['intersect', '-i', fin, '-o', fout]
    # Type option
    if cutout: cmdi += ['-cutout', str(cutout)]
    if ascii:  cmdi.append('-ascii')
    if v:      cmdi.append('-v')
    if T:      cmdi.append('-T')
    if iout:   cmdi.append('-intersections')
    # Output
    return cmdi


# Function to call verify
def verify(opts=None, **kw):
    """Generate command for Cart3D executable ``verify``
    
    :Call:
        >>> cmdi = cape.cmd.verify(opts=None, **kw)
    :Inputs:
        *opts*: :class:`cape.options.Options`
            Options interface
        *i*: {``"Components.i.tri"`` | :class:`str`
            Name of *tri* file to test
        *ascii*: {``True``} | ``False``
            Flag to consider input file ASCII
        *binary*: ``True`` | {``False``}
            Flag to consider input file binary
    :Outputs:
        *cmdi*: :class:`list` (:class:`str`)
            Command split into a list of strings
    :Versions:
        * 2015-02-13 ``@ddalle``: First version
    """
    # Check input type
    if opts is not None:
        # Check for "RunControl"
        opts = opts.get("RunControl", opts)
        opts = opts.get("verify", opts)
        # Get settings
        ftri  = getel(opts.get('i', 'Components.i.tri'), 0)
        binry = getel(opts.get('binary', False), 0)
        ascii = getel(opts.get('ascii', not binry), 0)
    else:
        # Get settings from inputs
        ftri  = kw.get('i', 'Components.i.tri')
        binry = kw.get('binary', False)
        ascii = kw.get('ascii', not binry) 
    # Build the command.
    cmdi = ['verify', ftri]
    # Type
    if binry:
        # Binary triangulation
        cmdi.append('-binary')
    elif ascii:
        # ASCII triangulation
        cmdi.append('-ascii')
    # Output
    return cmdi

