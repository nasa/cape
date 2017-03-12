
.. _pycart-ex-lineload-arrow:

Demo 6: Line Loads on the Arrow Example
=======================================

NOTE: This example requires `Chimera Grid Tools 
<https://www.nas.nasa.gov/publications/software/docs/chimera/index.html>`_ to
calculate sectional loads.  Specifically, the `triload
<https://www.nas.nasa.gov/publications/software/docs/chimera/pages/triload.html>`_
command is used.  To acquire Chimera Grid Tools, free software from NASA, use
the `NASA Software Catalog <https://software.nasa.gov/software/ARC-16025-1A>`_.

Using the geometry from :ref:`Example 2 <pycart-ex-arrow>`, this cases
continues the analysis and adds computation of sectional loads.  The example is
located in ``$PYCART/examples/pycart/06_lineload_arrow``.

The geometry used for this shape is a capped cylinder with four fins and 9216
faces and seven components.  The surface triangulation, :file:`arrow.tri`, is
shown below.

    .. figure:: ../arrow01.png
        :width: 4in
        
        Simple bullet shape triangulation with four fins
        
This example is set up with a small run matrix to demonstrate line loads on a
few related cases.

    .. code-block:: console
    
        $ pycart -c
        Case Config/Run Directory   Status  Iterations  Que CPU Time 
        ---- ---------------------- ------- ----------- --- --------
        0    poweroff/m1.25a0.0b0.0 ---     /           .            
        1    poweroff/m1.25a2.0b0.0 ---     /           .            
        2    poweroff/m1.25a0.0b2.0 ---     /           .            
        
Running Cases
-------------
When doing post-processing of Cart3D results, it is often desirable to perform
time-averaging or iteration-averaging before doing analysis.  When Cart3D exits
(either ``flowCart`` or ``mpix_flowCart`` called with the ``-clic`` option), it
writes a ``Components.i.triq`` file.  This is a surface triangulation with some
results from the last iteration.  It contains the pressure coefficient and the
five native Cart3D state variables at each node of the triangulation.

To get a ``triq`` file with averaged results, we have to run ``flowCart`` a few
iterations at a time and manually perform averaging.  The ``-stats`` option
performs a similar task, but it is not quite consistent with what's needed for
an averaged line load.  To get pyCart to perform this unusual task, we have the
following ``"RunControl"`` section in :file:`pyCart.json`.

    .. code-block:: javascript
    
        // Iteration control and command-line inputs
        "RunControl": {
            // Run sequence
            "PhaseSequece": [0],
            "PhaseIters": [200],
            // System configuration
            "nProc": 4,
            "MPI": 0,
            "Adaptive": 0,
            // Options for ``flowCart``
            "flowCart": {
                "it_fc": 200,
                "it_avg": 10,
                "it_start": 100,
                "cfl": 1.1,
                "mg_fc": 3,
                "y_is_spanwise": true
            },
            // Defines the flow domain automatically
            "autoInputs": {"r": 6},
            // Volume mesh options
            "cubes": {
                "maxR": 8,
                "pre": "preSpec.c3d.cntl",
                "cubes_a": 8,
                "cubes_b": 2,
                "reorder": true
            }
        }

As previously, the *RunControl* > *flowCart* > *it_fc* option controls how many
iterations ``flowCart`` runs for.  The *it_avg* and *it_start* are new options.
The idea is that Cart3D will be run for *it_avg* iterations at a time.  pyCart
then calculates a cumulative average ``triq`` file that updates after each
*it_avg* iterations.  However, it first runs *it_start* iterations before
initiating this start-stop behavior.  This prevents initial iterations from
corrupting the average.

If we run one case, there is a lot of output printed to STDOUT, and it looks
something like this.  **Note:** This is set up to run

    .. code-block:: console
    
        $ pycart -I 0
        Case Config/Run Directory   Status  Iterations  Que CPU Time 
        ---- ---------------------- ------- ----------- --- --------
        0    poweroff/m1.25a0.0b0.0 ---     /           .            
          Group name: 'poweroff' (index 0)
          Preparing surface triangulation...
          Reading tri file(s) from root directory.
         > autoInputs -r 6 -t Components.i.tri -maxR 8 -nDiv 4
         > cubes -pre preSpec.c3d.cntl -maxR 8 -reorder -a 8 -b 2
         > mgPrep -n 3
             Starting case 'poweroff/m1.25a0.0b0.0'
         > flowCart -his -clic -N 100 ...
         > flowCart -his -clic -restart -N 110 ...
         > flowCart -his -clic -restart -N 120 ...
         > flowCart -his -clic -restart -N 130 ...
         > flowCart -his -clic -restart -N 140 ...
         > flowCart -his -clic -restart -N 150 ...
         > flowCart -his -clic -restart -N 160 ...
         > flowCart -his -clic -restart -N 170 ...
         > flowCart -his -clic -restart -N 180 ...
         > flowCart -his -clic -restart -N 190 ...
         > flowCart -his -clic -restart -N 200 ...
             Writing triangulation: 'Components.11.100.200.triq'
        
        Submitted or ran 1 job(s).
        
        ---=1, 
        
This lengthy output explains more clearly what is meant by running ``flowCart``
10 iterations at a time.  The iteration-averaged surface file that gets created
at the end, ``Components.11.100.200.triq``, explains the contents of the file. 
Specifically, it says that the file contains input from 11 iterations between
100 and 200.

