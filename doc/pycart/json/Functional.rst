
.. _pycart-json-functional:

**************************
Functional Section Options
**************************
The options below are the available options in the Functional Section of the ``pycart.json`` control file

FunctionalCoeff Options
-----------------------
..
    start-FunctionalCoeff-frame

*frame*: {``0``} | ``1``
    force frame; ``0`` for body axes and ``1`` for stability

..
    end-FunctionalCoeff-frame

..
    start-FunctionalCoeff-parent

*parent*: {``None``} | :class:`str`
    name of parent coefficient

..
    end-FunctionalCoeff-parent

..
    start-FunctionalCoeff-compid

*compID*: {``None``} | :class:`str`
    name of component from which to calculate force/moment

..
    end-FunctionalCoeff-compid

..
    start-FunctionalCoeff-target

*target*: {``0.0``} | :class:`float` | :class:`float32`
    target value; functional is ``weight*(F-target)**N``

..
    end-FunctionalCoeff-target

..
    start-FunctionalCoeff-j

*J*: {``0``} | ``1``
    value of option "J"

..
    end-FunctionalCoeff-j

..
    start-FunctionalCoeff-index

*index*: {``0``} | ``1`` | ``2``
    index of moment reference point to use (0-based)

..
    end-FunctionalCoeff-index

..
    start-FunctionalCoeff-type

*type*: {``'optForce'``} | ``'optMoment'`` | ``'optSensor'``
    output type

..
    end-FunctionalCoeff-type

..
    start-FunctionalCoeff-n

*N*: {``1``} | :class:`int` | :class:`int32` | :class:`int64`
    value of option "N"

..
    end-FunctionalCoeff-n

..
    start-FunctionalCoeff-force

*force*: {``0``} | ``1`` | ``2``
    axis number of force to use (0-based)

..
    end-FunctionalCoeff-force

..
    start-FunctionalCoeff-weight

*weight*: {``1.0``} | :class:`float` | :class:`float32`
    weight multiplier for force's contribution to total

..
    end-FunctionalCoeff-weight

..
    start-FunctionalCoeff-moment

*moment*: {``0``} | ``1`` | ``2``
    axis number of moment to use (0-based)

..
    end-FunctionalCoeff-moment

