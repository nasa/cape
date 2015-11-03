
Moving Components Example: Fins
===============================

The following 

similar to :ref:`pycart-ex-arrow`


    .. code-block:: none
    
        $ pycart -f fins.json -I 7
        Case Config/Run Directory                   Status  Iterations  Que 
        ---- -------------------------------------- ------- ----------- ---
        0    poweroff_d2+05.0d4-15.0/m1.75a1.0r30.0 ---     /           .   
          Group name: 'poweroff_d2+05.0d4-15.0' (index 4)
          Preparing surface triangulation...
          Reading tri file(s) from root directory.
             Writing triangulation: 'Components.tri'
             Writing triangulation: 'Components.c.tri'
         > autoInputs -r 8 -t Components.tri -maxR 10
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/fins/poweroff_d2+05.0d4-15.0')
             (STDOUT = 'autoInputs.out')
         > intersect -i Components.tri -o Components.o.tri
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/fins/poweroff_d2+05.0d4-15.0')
             (STDOUT = 'intersect.out')
             Writing triangulation: 'Components.i.tri'
         > cubes -pre preSpec.c3d.cntl -maxR 10 -reorder -a 10 -b 2
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/fins/poweroff_d2+05.0d4-15.0')
             (STDOUT = 'cubes.out')
         > mgPrep -n 3
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/fins/poweroff_d2+05.0d4-15.0')
             (STDOUT = 'mgPrep.out')
        Using template for 'input.cntl' file
             Starting case 'poweroff_d2+05.0d4-15.0/m1.75a1.0r30.0'.
         > flowCart -his -clic -N 200 -y_is_spanwise -limiter 2 -T -cfl 1.1 -mg 3 -binaryIO -tm 0
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/fins/poweroff_d2+05.0d4-15.0/m1.75a1.0r30.0')
             (STDOUT = 'flowCart.out')
        
        Submitted or ran 1 job(s).
        
        ---=1,


