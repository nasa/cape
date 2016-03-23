
.. _pyfun-json-Config:

-----------------------------
FUN3D Component Configuration
-----------------------------

FUN3D force and moment tracking is defined in the ``"Config"`` section of the
:file:`pyFun.json` file.  This section informs pyFun about component
definitions, which components will have iterative force and moment histories,
and can be used to rotate/translate surfaces.  Settings here edit the
``component_parameters`` section of the FUN3D namelist.

The generic options dictionary for this section can be found in the :ref:`Cape
section <cape-json-Config>`.  Specific syntax for pyFun is shown below.

    .. code-block:: javascript
    
        "Config": {
            // List of force & moment components
            "Components": ["total", "wing"],
            // Component definitions, based on MAPBC file
            "Inputs": {
                "total": "1-14",
                "wing": "2-4,6-8"
            },
            // Reference values
            "RefArea": 1.0,
            "RefLength": {
                "total": 0.5,
                "wing": 1.0
            },
            // Moment history requests with MRP
            "RefPoint": "MPR"
            
The *Inputs* option defines components that are not spelled out in the ``mapbc``
file.  This is the way to define a component that has triangles with different
component IDs.  In other words, it is used to group components.

The full dictionary of FUN3D "Config" options is shown below.
        
    *Components*: :class:`list` (:class:`str`)
        List of components on which to request force history
        
    *Inputs*: {``{}``} | :class:`dict` (:class:`str`)
        Dictionary of component numbers
        
    *Points*: {``{}``} | :class:`dict` (:class:`list`)
        Dictionary of named points and their coordinates
        
    *RefArea*: {``1.0``} | :class:`float` | :class:`dict` (:class:`float`)
        Reference area or :class:`dict` of reference areas for different
        components
        
    *RefLength*: {``1.0``} | :class:`float` | :class:`dict` (:class:`float`)
        Reference length or :class:`dict` of reference lengths for different
        components
        
    *RefPoint*: {``[0.0, 0.0, 0.0]``} | :class:`dict` | :class:`list`
        Three-dimensional float specifying global reference point or
        :class:`dict` of components and their moment reference points

