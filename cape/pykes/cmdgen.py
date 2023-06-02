r"""
:mod:`cape.pykes.cmd`: Create commands for Kestrel executables 
=================================================================

This module creates system commands as lists of strings for executable
binaries or scripts for Kestrel. The command-line interface to Kestrel
is quite simple, so commands just take the form

    .. code-block:: console

        $ csi -p NUMCORES -i XMLFILE

"""

# Standard library
import os.path as op


# Function to create ``nodet`` or ``nodet_mpi`` command
def csi(opts=None, i=0, **kw):
    r"""Create commands for main Kestrel executable

    :Call:
        >>> cmdi = csi(opts, i=0)
        >>> cmdi = csi(**kw)
    :Inputs:
        *opts*: :class:`Options` | :class:`RunControl`
            Global pykes options interface or "RunControl" interface
        *i*: :class:`int`
            Phase number
        *nProc*: {``None``} | :class:`int`
            Directly specified number of cores
        *XMLPrefix*: {``"kestrel"``} | :class:`str`
            Directly specified XML prefix
        *xmlfile*: {``None``} | :class:`str`
            Name of XML file; usually from *proj*
    :Outputs:
        *cmdi*: :class:`list`\ [:class:`str`]
            Command split into a list of strings
    :Versions:
        * 2021-10-20 ``@ddalle``: Version 1.0
    """
    # Check for options input
    if opts is None:
        # Defaults
        n_proc = 1
    else:
        # Get values from *RunControl* section
        n_proc  = opts.get_nProc(i)
    # Get values from keyword arguments
    n_proc  = int(kw.get("nProc", n_proc))
    # Job name
    prefix = kw.get("XMLPrefix", "kestrel")
    # Default XML file
    fxml = "%s.xml" % prefix
    fxmli = "%s.%02i.xml" % (prefix, i)
    # Check if we should use phase
    if op.isfile(fxmli) and not op.isfile(fxml):
        # Use phase number in default
        fxml = fxmli
    # Check for override
    fxml = kw.get("xmlfile", fxml)
    # Form command
    cmdi = ["csi", "-p", str(n_proc), "-i", fxml]
    # Output
    return cmdi

