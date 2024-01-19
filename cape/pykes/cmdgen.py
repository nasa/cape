r"""
:mod:`cape.pykes.cmdgen`: Create commands for Kestrel executables
=================================================================

This module creates system commands as lists of strings for executable
binaries or scripts for Kestrel. The command-line interface to Kestrel
is quite simple, so commands just take the form

    .. code-block:: console

        $ csi -p NUMCORES -i XMLFILE

"""

# Standard library
import os.path as op

# Local imports
from .options import Options
from ..cfdx.cmdgen import isolate_subsection


# Function to create ``nodet`` or ``nodet_mpi`` command
def csi(opts=None, j=0, **kw):
    r"""Create commands for main Kestrel executable

    :Call:
        >>> cmdi = csi(opts, j=0)
        >>> cmdi = csi(**kw)
    :Inputs:
        *opts*: :class:`Options` | :class:`RunControl`
            Global pykes options interface or "RunControl" interface
        *j*: {``0``} | :class:`int`
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
    # Isolate options
    opts = isolate_subsection(opts, Options, ("RunControl",))
    # Apply kwargs
    opts.set_opts(kw)
    # Get number of cores
    n_proc = opts.get_opt("nProc", j=j)
    # Job name
    prefix = opts.get_opt("XMLPrefix", j=j)
    # Default XML file
    fxml = "%s.xml" % prefix
    fxmli = "%s.%02i.xml" % (prefix, j)
    # Check if we should use phase
    if op.isfile(fxmli) and not op.isfile(fxml):
        # Use phase number in default
        fxml = fxmli
    # Form command
    cmdi = ["csi", "-p", str(n_proc), "-i", fxml]
    # Output
    return cmdi

