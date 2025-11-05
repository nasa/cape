-------------------------------------
``Tecplot``: tecplot layout subfigure
-------------------------------------

**Option aliases:**

* *ActiveFields* → *ActiveFieldMaps*
* *ActiveFieldMap* → *ActiveFieldMaps*
* *Parent* → *Type*
* *parent* → *Type*
* *pos* → *Position*
* *type* → *Type*
* *width* → *Width*

**Recognized options:**

*ActiveFieldMaps*: {``None``} | :class:`str`
    string of active Tecplot fields(zones)
*Alignment*: {``'center'``} | :class:`object`
    value of option "Alignment"
*Caption*: {``None``} | :class:`str`
    subfigure caption
*ColorMaps*: {``[]``} | :class:`list`\ [:class:`dict`]
    customized Tecplot colormap
*ContourLevels*: {``None``} | :class:`list`\ [:class:`dict`]
    customized settings for Tecplot contour levels
*FieldMap*: {``None``} | :class:`int` | :class:`str`
    list of zone numbers for Tecplot layout group boundaries
*FigWidth*: {``1024``} | :class:`int`
    width of output image in pixels
*Keys*: {``None``} | :class:`dict`
    dict of Tecplot layout statements to customize
*Layout*: {``None``} | :class:`str`
    template Tecplot layout file
*Position*: {``'b'``} | ``'c'`` | ``'t'``
    subfigure vertical alignment
*Type*: {``None``} | :class:`str`
    subfigure type or parent
*VTKFiles*: {``None``} | :class:`list`\ [:class:`object`]
    files to convert .vtk -> .plt
*VarSet*: {``{}``} | :class:`dict`
    variables and their values to define in Tecplot layout
*Width*: {``0.5``} | :class:`float`
    value of option "Width"

