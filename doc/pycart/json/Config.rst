
.. _pycart-json-config:

******************************
Options for ``Config`` Section
******************************
The options below are the available options in the ``Config`` Section of the ``pycart.json`` control file


*ConfigFile*: {``'Config.xml'``} | :class:`str`
    configuration file name



*Xslices*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`str`
    *x*\ -slice(s) to export



*Components*: {``[]``} | :class:`list`\ [:class:`str`]
    list of components to request from solver



*Zslices*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`str`
    *z*\ -slice(s) to export



*RefSpan*: {``None``} | :class:`dict` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128`
    value of option "RefSpan"



*LineSensors*: {``None``} | :class:`dict`
    dictionary of line sensor definitions



*RefLength*: {``1.0``} | :class:`dict` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128`
    value of option "RefLength"



*RefPoint*: {``[0.0, 0.0, 0.0]``} | :class:`dict` | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128`
    value of option "RefPoint"



*RefArea*: {``1.0``} | :class:`dict` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128`
    reference area [for a component]



*Force*: {``None``} | :class:`list`\ [:class:`str`]
    components to only report force (not moment)



*Points*: {``{}``} | :class:`dict`
    dictionary of reference point locations



*Yslices*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`str`
    *y*\ -slice(s) to export


