
.. _pycart-json-functional:

**********************************
Options for ``Functional`` Section
**********************************
The options below are the available options in the ``Functional`` Section of the ``pycart.json`` control file


*target*: {``0.0``} | :class:`float` | :class:`float32`
    target value; functional is ``weight*(F-target)**N``



*force*: {``0``} | ``1`` | ``2``
    axis number of force to use (0-based)



*moment*: {``0``} | ``1`` | ``2``
    axis number of moment to use (0-based)



*index*: {``0``} | ``1`` | ``2``
    index of moment reference point to use (0-based)



*N*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "N"



*J*: {``0``} | ``1``
    value of option "J"



*type*: {``'optForce'``} | ``'optMoment'`` | ``'optSensor'``
    output type



*compID*: {``None``} | :class:`str`
    name of component from which to calculate force/moment



*parent*: {``None``} | :class:`str`
    name of parent coefficient



*frame*: {``0``} | ``1``
    force frame; ``0`` for body axes and ``1`` for stability



*weight*: {``1.0``} | :class:`float` | :class:`float32`
    weight multiplier for force's contribution to total


