
.. _pycart-json-config:

**********************
Config Section Options
**********************
The options below are the available options in the Config Section of the ``pycart.json`` control file

..
    start-Config-refarea

*RefArea*: {``1.0``} | :class:`dict` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128`
    reference area [for a component]

..
    end-Config-refarea

..
    start-Config-reflength

*RefLength*: {``1.0``} | :class:`dict` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128`
    value of option "RefLength"

..
    end-Config-reflength

..
    start-Config-linesensors

*LineSensors*: {``None``} | :class:`dict`
    dictionary of line sensor definitions

..
    end-Config-linesensors

..
    start-Config-force

*Force*: {``None``} | :class:`list`\ [:class:`str`]
    components to only report force (not moment)

..
    end-Config-force

..
    start-Config-zslices

*Zslices*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`str`
    *z*\ -slice(s) to export

..
    end-Config-zslices

..
    start-Config-xslices

*Xslices*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`str`
    *x*\ -slice(s) to export

..
    end-Config-xslices

..
    start-Config-components

*Components*: {``[]``} | :class:`list`\ [:class:`str`]
    list of components to request from solver

..
    end-Config-components

..
    start-Config-refpoint

*RefPoint*: {``[0.0, 0.0, 0.0]``} | :class:`dict` | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128`
    value of option "RefPoint"

..
    end-Config-refpoint

..
    start-Config-refspan

*RefSpan*: {``None``} | :class:`dict` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128`
    value of option "RefSpan"

..
    end-Config-refspan

..
    start-Config-points

*Points*: {``{}``} | :class:`dict`
    dictionary of reference point locations

..
    end-Config-points

..
    start-Config-configfile

*ConfigFile*: {``'Config.xml'``} | :class:`str`
    configuration file name

..
    end-Config-configfile

..
    start-Config-yslices

*Yslices*: {``None``} | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128` | :class:`str`
    *y*\ -slice(s) to export

..
    end-Config-yslices

