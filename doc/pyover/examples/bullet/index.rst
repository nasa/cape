
.. _pyover-example-bullet:

------------------------
OVERFLOW Bullet Example
------------------------

This pyOver example looks at the process from grid generation to execution and
post-processing for a simple bullet geometry.  The surface grid system is shown
in :numref:`tab-pyover-bullet-01`, and one view of the volume grid is shown in
:numref:`fig-pyover-bullet-01`.

    .. _tab-pyover-bullet-01:
    .. table:: OVERFLOW surface grid for bullet example
    
        +------------------------------+------------------------------+
        |.. image:: bullet-surf-01.png |.. image:: bullet-surf-02.png |
        |    :width: 3.1in             |    :width: 3.1in             |
        |                              |                              |
        |Front and side view           |Aft view                      |
        +------------------------------+------------------------------+
        
    .. _fig-pyover-bullet-01:
    .. figure:: bullet-vol-01.png
        :width: 4.0in
        
        OVERFLOW volume grid slice for bullet example
        
This is a three-grid system that demonstrates a lot of the basic features of
running OVERFLOW and creating a simple grid system.  
        
Grid generation
----------------



Execution
----------

    .. code-block:: console
    
        $ pyover -I 1
        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        1    poweroff/m0.8a4.0b0.0 ---     /           .            
          Case name: 'poweroff/m0.8a4.0b0.0' (index 1)
             Starting case 'poweroff/m0.8a4.0b0.0'
         > overrunmpi -np 6 run 01
             (PWD = '/examples/pyover/01_bullet/poweroff/m0.8a4.0b0.0')
             (STDOUT = 'overrun.out')
           Wall time used: 0.07 hrs (phase 0)
           Wall time used: 0.07 hrs
           Previous phase: 0.07 hrs
         > overrunmpi -np 6 run 02
             (PWD = '/examples/pyover/01_bullet/poweroff/m0.8a4.0b0.0')
             (STDOUT = 'overrun.out')
           Wall time used: 0.08 hrs (phase 1)
           Wall time used: 0.14 hrs
           Previous phase: 0.08 hrs
         > overrunmpi -np 6 run 03
             (PWD = /examples/pyover/01_bullet/poweroff/m0.8a4.0b0.0')
             (STDOUT = 'overrun.out')
           Wall time used: 0.05 hrs (phase 2)
        
        Submitted or ran 1 job(s).
        
        ---=1, 

