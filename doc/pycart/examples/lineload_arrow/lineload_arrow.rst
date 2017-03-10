
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
continues the analysis and adds computation of sectional loads.

The geometry used for this shape is a capped cylinder with four fins and 9216
faces and seven components.  The surface triangulation, :file:`arrow.tri`, is
shown below.

    .. figure:: ../arrow01.png
        :width: 4in
        
        Simple bullet shape triangulation with four fins
        
This example is set up with a larger run matrix in order to demonstrate more of
the features of pyCart, including databook sweeps.

    .. code-block:: none
    
        $ pycart -c
        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m1.25a0.0    ---     /           .            
        1    poweroff/m1.25a1.0    ---     /           .            
        2    poweroff/m1.25a2.0    ---     /           .            
        3    poweroff/m1.25a5.0    ---     /           .            
        4    poweroff/m1.50a0.0    ---     /           .            
        5    poweroff/m1.50a1.0    ---     /           .            
        6    poweroff/m1.50a2.0    ---     /           .            
        7    poweroff/m1.50a5.0    ---     /           .            
        8    poweroff/m1.75a0.0    ---     /           .            
        9    poweroff/m1.75a1.0    ---     /           .            
        10   poweroff/m1.75a2.0    ---     /           .            
        11   poweroff/m1.75a5.0    ---     /           .            
        12   poweroff/m2.00a0.0    ---     /           .            
        13   poweroff/m2.00a1.0    ---     /           .            
        14   poweroff/m2.00a2.0    ---     /           .            
        15   poweroff/m2.00a5.0    ---     /           .            
        16   poweroff/m2.50a0.0    ---     /           .            
        17   poweroff/m2.50a1.0    ---     /           .            
        18   poweroff/m2.50a2.0    ---     /           .            
        19   poweroff/m2.50a5.0    ---     /           .            
        
        ---=20, 
