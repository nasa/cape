
.. _pyfun-json-config:

******************************
Options for ``Config`` Section
******************************
The options below are the available options in the ``Config`` Section of the ``pyfun.json`` control file


*ConfigFile*: {``'Config.xml'``} | :class:`str`
    configuration file name



*SpeciesThermoDataFile*: {``'inputs/species_thermo_data'``} | :class:`str`
    template ``species_thermo_data`` file



*KineticDataFile*: {``'inputs/kinetic_data'``} | :class:`str`
    template ``kinetic_data`` file



*RubberDataFile*: {``'inputs/rubber.data'``} | :class:`str`
    template for ``rubber.data`` file



*TDataFile*: {``'inputs/tdata'``} | :class:`str`
    template for ``tdata`` file



*Components*: {``[]``} | :class:`list`\ [:class:`str`]
    list of components to request from solver



*MovingBodyInputFile*: {``'inputs/moving_body.input'``} | :class:`str`
    template ``moving_body.input`` file



*Inputs*: {``None``} | :class:`dict`
    dictionary of component indices for named comps



*RefSpan*: {``None``} | :class:`dict` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128`
    value of option "RefSpan"



*RefLength*: {``1.0``} | :class:`dict` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128`
    value of option "RefLength"



*RefPoint*: {``[0.0, 0.0, 0.0]``} | :class:`dict` | :class:`str` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128`
    value of option "RefPoint"



*RefArea*: {``1.0``} | :class:`dict` | :class:`float` | :class:`float16` | :class:`float32` | :class:`float64` | :class:`float128`
    reference area [for a component]



*Points*: {``{}``} | :class:`dict`
    dictionary of reference point locations


