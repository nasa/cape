
Demo 4: Business Jet, Data Book, and Automated Reports
======================================================


    .. code-block:: none
    
        $ pycart -n 1
        Case Config/Run Directory   Status  Iterations  Que 
        ---- ---------------------- ------- ----------- ---
        0    poweroff/m0.84a0.0b0.0 ---     /           .   
          Case name: 'poweroff/m0.84a0.0b0.0' (index 0)
          Preparing surface triangulation...
          Reading tri file(s) from root directory.
             Writing triangulation: 'Components.i.tri'
         > autoInputs -r 8 -t Components.i.tri -maxR 11 -nDiv 4
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/bJet/poweroff/m0.84a0.0b0.0')
             (STDOUT = 'autoInputs.out')
         > cubes -pre preSpec.c3d.cntl -maxR 11 -reorder -a 10 -b 2
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/bJet/poweroff/m0.84a0.0b0.0')
             (STDOUT = 'cubes.out')
         > mgPrep -n 3
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/bJet/poweroff/m0.84a0.0b0.0')
             (STDOUT = 'mgPrep.out')
             Starting case 'poweroff/m0.84a0.0b0.0'.
         > flowCart -his -clic -N 200 -y_is_spanwise -limiter 2 -T -cfl 1.1 -mg 3 -tm 0
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/bJet/poweroff/m0.84a0.0b0.0')
             (STDOUT = 'flowCart.out')
         > flowCart -his -clic -restart -N 300 -y_is_spanwise -limiter 2 -T -cfl 1.1 -mg 3 -tm 1
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/bJet/poweroff/m0.84a0.0b0.0')
             (STDOUT = 'flowCart.out')
        
        Submitted or ran 1 job(s).
        
        ---=1, 


    .. figure:: bJet01.png
        :width: 5in
    
        Business jet triangulation colored by pressure coefficient from
        ``poweroff/m0.84a0.0b0.0`` solution
