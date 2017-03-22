
.. _pycart-ex-data-arrow:

Demo 7: Data Book Plots and Reports
===================================

Using the geometry from :ref:`Example 2 <pycart-ex-arrow>` and 
:ref:`Example 7 <pycart-ex-lineload-arrow>`, this example computes forces and
moments on a larger number of cases in order to

continues the analysis and adds computation of sectional loads.  The example is
located in ``$PYCART/examples/pycart/07_data_arrow``.

The geometry used for this shape is a capped cylinder with four fins and 9216
faces and seven components.  This example is set to run 30 cases with a square
matrix of Mach number and angle of attack.

    .. code-block:: console
    
        $ pycart -c
        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m0.50a00.0   ---     /           .            
        1    poweroff/m0.50a01.0   ---     /           .            
        2    poweroff/m0.50a02.0   ---     /           .            
        3    poweroff/m0.50a05.0   ---     /           .            
        4    poweroff/m0.50a10.0   ---     /           .            
        5    poweroff/m0.80a00.0   ---     /           .            
        6    poweroff/m0.80a01.0   ---     /           .            
        7    poweroff/m0.80a02.0   ---     /           .            
        8    poweroff/m0.80a05.0   ---     /           .            
        9    poweroff/m0.80a10.0   ---     /           .            
        10   poweroff/m0.95a00.0   ---     /           .            
        11   poweroff/m0.95a01.0   ---     /           .            
        12   poweroff/m0.95a02.0   ---     /           .            
        13   poweroff/m0.95a05.0   ---     /           .            
        14   poweroff/m0.95a10.0   ---     /           .            
        15   poweroff/m1.10a00.0   ---     /           .            
        16   poweroff/m1.10a01.0   ---     /           .            
        17   poweroff/m1.10a02.0   ---     /           .            
        18   poweroff/m1.10a05.0   ---     /           .            
        19   poweroff/m1.10a10.0   ---     /           .            
        20   poweroff/m1.40a00.0   ---     /           .            
        21   poweroff/m1.40a01.0   ---     /           .            
        22   poweroff/m1.40a02.0   ---     /           .            
        23   poweroff/m1.40a05.0   ---     /           .            
        24   poweroff/m1.40a10.0   ---     /           .            
        25   poweroff/m2.20a00.0   ---     /           .            
        26   poweroff/m2.20a01.0   ---     /           .            
        27   poweroff/m2.20a02.0   ---     /           .            
        28   poweroff/m2.20a05.0   ---     /           .            
        29   poweroff/m2.20a10.0   ---     /           .            
        
        ---=30,
        
Setup
-----
The interesting part of the JSON setup to get the data we need for this example
is in the ``"Config"`` and ``"DataBook"`` sections.  The ``"Config"`` section
tells Cart3D which things for which to track iterative histories.  In addition,
it defines the reference area and any reference points (such as the Moment
reference point).

    .. code-block:: javascript
    
        "Config": {
            // File to name each integer CompID and group them into families 
            "File": "arrow.xml",
            // Declare forces and moments
            "Force": [
                "cap",
                "body",
                "fins",
                "arrow_no_base",
                "arrow_total",
                "fin1",
                "fin2",
                "fin3",
                "fin4"
            ],
            "RefPoint": {
                "arrow_no_base": "MRP",
                "arrow_total": "MRP", 
                "fins": "MRP",
                "fin1": "MRP",
                "fin2": "MRP",
                "fin3": "MRP",
                "fin4": "MRP"
            },
            // Define some points for easier reference
            "Points": {
                "MRP": [3.5, 0.0, 0.0]
            },
            // Reference quantities
            "RefArea": 3.14159,
            "RefLength": 2.0
        }
    
    