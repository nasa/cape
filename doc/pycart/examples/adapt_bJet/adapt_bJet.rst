
Demo 5: Mesh Adaptation on a Business Jet
=========================================

The following example uses a more complex geometry in combination with Cart3D's
adaptive meshing capability.  It can be found in the
``$PYCART/examples/pycart/adapt_bJet`` folder.  Let's run the first case.

    .. code-block:: none
    
        Case Config/Run Directory   Status  Iterations  Que CPU Time 
        ---- ---------------------- ------- ----------- --- --------
        0    poweroff/m0.82a0.0b0.0 ---     /           .            
          Group name: 'poweroff' (index 0)
          Preparing surface triangulation...
          Reading tri file(s) from root directory.
            Writing triangulation: 'Components.i.tri'
         > autoInputs -r 8 -t Components.i.tri -maxR 8 -nDiv 4
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/adapt_bJet/poweroff')
             (STDOUT = 'autoInputs.out')
             Starting case 'poweroff/m0.82a0.0b0.0'.
         > ./aero.csh
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/adapt_bJet/poweroff/m0.82a0.0b0.0')
             (STDOUT = 'flowCart.out')
        adapt00 --> adapt00.tar
        adapt01 --> adapt01.tar
         > ./aero.csh restart
             (PWD = '/u/wk/ddalle/usr/pycart/examples/pycart/adapt_bJet/poweroff/m0.82a0.0b0.0')
             (STDOUT = 'flowCart.out')
        adapt03 --> adapt03.tar
        adapt02 --> adapt02.tar
        
        Submitted or ran 1 job(s).
        
        ---=1, 

A sample graphic of the surface pressure made with Tecplot is shown below.

    .. figure:: adapt_bJet01.png
        :width: 5in
    
        Business jet solution showing pressure with slice at *y* = 0 for
        ``poweroff/m0.84a0.0b0.0`` solution
        
Phase Control
-------------
The ``"RunControl"`` section of :file:`pyCart.json` contains additional
information compared to previous examples for control of the mesh adaptation
settings.
