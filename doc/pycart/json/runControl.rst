
.. _pycart-json-RunControl:

------------------------------------
Run Sequence and Cart3D Mode Control
------------------------------------

The ``"RunControl"`` section of :file:`pyCart.json` contains settings that
control how many iterations to run Cart3D, what mode to use for each subset of
iterations, and command-line inputs to the various Cart3D binaries. The entire
contents of this section, with all default options applied where necessary, is
written to a file called :file:`case.json` in the folder created for each case.

Several subsections of this section are described separately.

The options for this section have the following JSON syntax.

    .. code-block:: javascript
    
        "RunControl": {
            // Phase description
            "PhaseSequence": [0, 1],
            "PhaseIters": [200, 400],
            // Job type
            "MPI": false,
            "qsub": true,
            "Resubmit": false,
            "Continue": true,
            "nProc": 8,
            "Adaptive": false,
            // Environment variable settings
            "Environ": { },
            // System-wide resource settings
            "ulimit": {
                "s": 4194304
            },
            
            // Surface mesh preparation options
            "intersect": {},
            "verify": {},
            
            // Mesh setup options
            "autoInputs": {
                "r": 8,
                "nDiv": 4,
            },
            // Volume mesh generation options
            "cubes": {
                "maxR": 10,
                "cubes_a": 10,
                "cubes_b": 2,
                "reorder": true,
                "sf": 0
            },
            
            // Adaptation settings
            "Adaptation": { },
            // Main flow solver inputs
            "flowCart": { },
            // Adjoint solver inputs
            "adjointCart": { },
            // Archive settings
            "Archive": { }
        }

Some of these options are common to all solvers, and the full description of
such settings can be found on the :ref:`corresponding Cape page
<cape-json-RunControl>`.  Separate sections for :ref:`flowCart
<pycart-json-flowCart>`, :ref:`adjointCart <pycart-json-adjointCart>`, and
:ref:`Adaptation <pycart-json-Adaptation>` are provided for some of the subset
dictionaries.

The :ref:`CAPE intersect <cape-json-intersect>` section is also relevant.

.. _pycart-json-autoInputs:

Options for `autoInputs`
========================

The description of ``autoInputs`` controls are shown below. The user can also
tell pyCart not to use ``autoInputs`` (and use a premade :file:`input.c3d`
instead) by adding ``"autoInputs": {}`` to :file:`pyCart.json`.

    *r*: {``8``} | :class:`float`
        Mesh radius.  This defines the dimensions of the flow domain;
        specifically the limits are set *r* times the largest dimension of the
        surface away from the surface.
        
    *nDiv*: {``4``} | :class:`int`
        Number of divisions in the initial mesh.  An exponent of 2 is highly
        recommended for efficiency purposes.

.. _pycart-json-cubes:        

Options for `cubes`
===================

The program that actually creates the volume mesh in Cart3D is called
``cubes``. The options that apply to it directly are shown below.

    *maxR*: {``10``} | :class:`int`
        Maximum number of refinements in volume mesh before *XLevs*
        
    *cubes_a*: {``10``} | :class:`float`
        Angle criterion for cut cell refinement
        
    *cubes_b*: {``2``} | :class:`int`
        Number of additional buffer layers
        
    *reorder*: {``true``} | ``false``
        Whether or not to reorder cells for optimal performance
        
    *sf*: {``0``} | :class:`int`
        Number of additional refinements for sharp features

