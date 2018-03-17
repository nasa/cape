

.. _cape-json-ReportParaview:

Paraview Subfigure
-------------------
The ``"Paraview"`` class of subfigure is used to set up *Paraview*,
specifically ``pvpython``, to create figures for each solution.  This
``pvpython`` program, which is part of *Paraview*, executes a Python script
within the solution folder.

The script has access to the file :file:`conditions.json` within each solution
folder, which can be used to determine the freestream Mach number, etc. within
the Python script.

The full list of options is shown below.

    *S*: :class:`dict`
        Dictionary of settings for *Paraview* subfigure
        
        *Type*: {``"Tecplot"``} | :class:`str`
            Subfigure type
            
        *Header*: {``""``} | :class:`str`
            Heading placed above subfigure (bold, italic)
            
        *Position*: ``"t"`` | ``"c"`` | {``"b"``}
            Vertical alignment of subfigure
            
        *Alignment*: ``"left"`` | {``"center"``}
            Horizontal alignment
            
        *Width*: {``0.5``} | :class:`float`
            Width of subfigure as a fraction of page text width
        
        *Layout*: {``"layout.py"``} | :class:`str`
            Name of Python script to copy to folder and run with ``pvpython``
            
        *ImageFile*: {``"export.png"``} | :class:`str`
            Name of file created by executing script
            
        *Format*: {``"png"``} | :class:`str`
            Format of image created
            
        *Command*: {``"pvpython"``} | :class:`str`
            Command used to open layout; can be set to ``"python"`` to just run
            a raw Python script to create a figure

