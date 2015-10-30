
Demo 2: Closer Analysis of Simple Arrow Shape
=============================================

The second example is similar to the first pyCart demo except that four fins
have been added and more details of the input files are explained.  The example
is found in ``$PYCART/examples/pycart/arrow`` where ``$PYCART`` is the
installation folder.

The geometry used for this shape is a capped cylinder with four fins and 9216
faces and seven components.  The surface triangulation, :file:`arrow.tri`, is
shown below.

    .. figure:: arrow01.png
        :width: 4in
        
        Simple bullet shape triangulation with four fins
        
This example is set up with a larger run matrix in order to demonstrate more of
the features of pyCart.

    .. code-block:: none
    
        $ pycart -c
        Case Config/Run Directory    Status  Iterations  Que 
        ---- ----------------------- ------- ----------- ---
        0    poweroff/m1.25a0.0r0.0  ---     /           .   
        1    poweroff/m1.25a1.0r0.0  ---     /           .   
        2    poweroff/m1.25a1.0r15.0 ---     /           .   
        3    poweroff/m1.25a1.0r30.0 ---     /           .   
        4    poweroff/m1.25a1.0r45.0 ---     /           .   
        5    poweroff/m1.5a1.0r0.0   ---     /           .   
        6    poweroff/m1.5a1.0r15.0  ---     /           .   
        7    poweroff/m1.5a1.0r30.0  ---     /           .   
        8    poweroff/m1.5a1.0r45.0  ---     /           .   
        9    poweroff/m1.75a1.0r0.0  ---     /           .   
        10   poweroff/m1.75a1.0r15.0 ---     /           .   
        11   poweroff/m1.75a1.0r30.0 ---     /           .   
        12   poweroff/m1.75a1.0r45.0 ---     /           .   
        13   poweroff/m2.0a1.0r0.0   ---     /           .   
        14   poweroff/m2.0a1.0r15.0  ---     /           .   
        15   poweroff/m2.0a1.0r30.0  ---     /           .   
        16   poweroff/m2.0a1.0r45.0  ---     /           .   
        17   poweroff/m2.5a1.0r0.0   ---     /           .   
        18   poweroff/m2.5a1.0r15.0  ---     /           .   
        19   poweroff/m2.5a1.0r30.0  ---     /           .   
        20   poweroff/m2.5a1.0r45.0  ---     /           .   
        
        ---=21,  

