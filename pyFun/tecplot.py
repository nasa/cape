"""
Module to interface with Tecplot scripts: :mod:`pyFun.tecplot`
==============================================================

This is a module built off of the :mod:`pyFun.fileCntl` module customized for
manipulating Tecplot layout files and macros.
"""

# Import the base file control class.
import cape.tecplot


# Stand-alone function to run a Tecplot layout file
def ExportLayout(lay="layout.lay", fname="export.png", fmt="PNG", w=None):
    """Stand-alone function to open a layout and export an image
    
    :Call:
        >>> ExportLayout(lay="layout.lay", fname="export.png", fmt="PNG", w=None)
    :Inputs:
        *lay*: {``"layout.lay"``} | :class:`str`
            Name of Tecplot layout file
        *fname*: {``"export.png"``} | :class:`str`
            Name of image file to export
        *fmt*: {``"PNG"``} | ``"JPG"`` | :class:`str`
            Valid image format for Tecplot export
        *w*: {``None``} | :class:`float`
            Image width in pixels
    :Versions:
        * 2015-03-10 ``@ddalle``: First version
    """
    cape.tecplot.ExportLayout(lay=lay, fname=fname, fmt=fmt, w=w)
    
# Aerodynamic history class
class Tecscript(cape.tecplot.Tecscript):
    """
    File control class for Tecplot script files
    
    :Call:
        >>> tec = pyFun.tecplot.Tecscript()
        >>> tec = pyFun.tecplot.Tecscript(fname="layout.lay")
    :Inputs:
        *fname*: :class:`str`
            Name of Tecplot script to read
    :Outputs:
        *tec*: :class:`pyFun.tecplot.Tecscript` or derivative
            Instance of Tecplot script base class
    :Versions:
        * 2015-02-26 ``@ddalle``: Started
        * 2015-03-10 ``@ddalle``: First version
    """
    pass

# class Tecscript


# Tecplot macro
class TecMacro(cape.tecplot.TecMacro):
    """
    File control class for Tecplot macr files
    
    :Call:
        >>> tec = pyFun.tecplot.TecMacro()
        >>> tec = pyFun.tecplot.TecMacro(fname="export.mcr")
    :Inputs:
        *fname*: :class:`str`
            Name of Tecplot script to read
    :Outputs:
        *tec*: :class:`pyFun.tecplot.TecMacro`
            Instance of Tecplot macro interface
    :Versions:
        * 2015-03-10 ``@ddalle``: First version
    """
    pass

# class TecMacro

