
.. _json-Functional:

----------------------------
Output Functional Definition
----------------------------

For cases utilizing Cart3D's adjoint capabilities, either for output-based mesh
adaptation or for evaluating derivatives and sensitivities, an output functional
must be defined.  This can be done by defining the functional in the template
:file:`input.cntl` file or by using the "Functional" section of
:file:`pyCart.json`.  The default for this section is simply ``"Functional":
{}``, so the following example shows how to define two forces that are added as
a weighted sum to define the output functional.

    .. code-block:: javascript
    
        "Functional": {
            "CN": {
                "type": "optForce",
                "force": 2,
                "frame": 0,
                "weight": 1.0,
                "compID": "wings",
                "J": 0,
                "N": 1,
                "target": 0.0
            },
            "CY": {
                "type": "optForce",
                "force": 1,
                "frame": 0,
                "weight": 0.5,
                "compID": "entire"
            }
        }
        
This defines the output function as
:math:`1.0C_{N,\mathit{wings}}+0.5C_{Y,\mathit{entire}}`, i.e. the normal force
coefficient on the component called ``wings`` plus 0.5 times the side force
coefficient on the entire surface.  For this to work, there must be a component
in :file:`Config.xml` called ``"wings"`` (``"entire"`` is defined
automatically), and both "wings" and "entire" must be requested forces in the
"Config" section of :file:`pyCart.json`.

The names of the output forces (``"CN"`` and ``"CY"`` in the example above) can
be whatever the user chooses.  These names will be used in the processed
:file:`input.cntl` files used to run Cart3D, so coherent names are always
recommended.  The only actual limitation (which is a limitation if pyCart and
not Cart3D itself) is that no two forces can have the same name.  For example,
if you want to use the normal force on two components, their forces cannot both
be named ``"CN"``.  Instead use something like ``"CN_L"`` and ``"CN_R"``.

In addition, the *J*, *N*, and *target* options, which are described below, can
be omitted in most cases.  They can be used to define special nonlinear output
functions.

The full dictionary of "Functional" options and their possible values is shown
below.

    *Functional*: {``{}``} | ``{"C": C, "C2": C2}`` | :class:`dict`
        Dictionary of output force(s)
        
        *C*: :class:`dict`
            An individual output force description :class:`dict`
            
            *type*: {``"optForce"``} | ``"optMoment"`` | ``"optSensor"``
                Output type; if ``"optSensor"``, the name of the key is the
                name of the point/line sensor to reference
                
            *compID*: :class:`str` | :class:`int`
                Name of component from which to calculate force/moment
                
            *force*: {``0``} | ``1`` | ``2``
                Component of force to use, i.e. ``0`` for *x*-axis (*CA* or
                *CLL*), ``1`` for *y*-axis, ``2`` for *z*-axis; does not apply
                to ``"optSensor"``
                
            *frame*: {``0``} | ``1``
                Force frame; ``0`` for body axes and ``1`` for stability axes;
                does not apply to ``"optSensor"``
                
            *weight*: {``1.0``} | :class:`float`
                Weight multiplier for force's contribution to total
                
            *J*: {``0``} | ``1``
                Modifier for the force; not normally used
                
            *N*: {``1``} | :class:`int`
                Exponent on force coefficient
                
            *target*: {``0.0``} | :class:`float`
                Target value; functional is ``weight*(F-target)**N``
                
            
