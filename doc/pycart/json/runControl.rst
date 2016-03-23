
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
            // Adaptation settings
            "Adaptation": { },
            // Input generation settings
            "autoInputs": { },
            // Volume mesh generation
            "cubes": { },
            // Main flow solver inputs
            "flowCart": { },
            // Adjoint solver inputs
            "adjointCart": { },
            // Archive settings
            "Archive": { }
        }

Some of these options are common to all solvers, and the full description of
such settings can be found on the :ref:`corresponding Cape page
<cape-json-RunControl>`.

