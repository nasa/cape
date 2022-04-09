
.. This documentation written by TestDriver()
   on 2022-04-09 at 01:40 PDT

Test ``02_cli``: PASS
=======================

This test PASSED on 2022-04-09 at 01:40 PDT

This test is run in the folder:

    ``test/cape/02_cli/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ cape -c
        $ cape -c --filter b2
        $ cape -c --cons 'beta==2,Mach%1==0.5'
        $ cape -c --glob 'poweroff/m0*'
        $ cape -c --re 'm.\.5.*b2'
        $ cape -c -I 2:5,7,18:
        $ cape -c -I 15: --cons Mach%1=0.5 --re b2

Command 1: Status (PASS)
-------------------------

:Command:
    .. code-block:: console

        $ cape -c

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.59 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m0.5a0.0b0.0 ---     /           .            
        1    poweroff/m0.5a2.0b0.0 ---     /           .            
        2    poweroff/m0.5a0.0b2.0 ---     /           .            
        3    poweroff/m0.5a2.0b2.0 ---     /           .            
        4    poweroff/m0.8a0.0b0.0 ---     /           .            
        5    poweroff/m0.8a2.0b0.0 ---     /           .            
        6    poweroff/m0.8a0.0b2.0 ---     /           .            
        7    poweroff/m0.8a2.0b2.0 ---     /           .            
        8    poweroff/m1.1a0.0b0.0 ---     /           .            
        9    poweroff/m1.1a2.0b0.0 ---     /           .            
        10   poweroff/m1.1a0.0b2.0 ---     /           .            
        11   poweroff/m1.1a2.0b2.0 ---     /           .            
        12   poweroff/m1.5a0.0b0.0 ---     /           .            
        13   poweroff/m1.5a2.0b0.0 ---     /           .            
        14   poweroff/m1.5a0.0b2.0 ---     /           .            
        15   poweroff/m1.5a2.0b2.0 ---     /           .            
        16   poweroff/m2.5a0.0b0.0 ---     /           .            
        17   poweroff/m2.5a2.0b0.0 ---     /           .            
        18   poweroff/m2.5a0.0b2.0 ---     /           .            
        19   poweroff/m2.5a2.0b2.0 ---     /           .            
        
        ---=20, 
        

:STDERR:
    * **PASS**

Command 2: Filter (PASS)
-------------------------

:Command:
    .. code-block:: console

        $ cape -c --filter b2

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.56 seconds
    * Cumulative time: 1.16 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        2    poweroff/m0.5a0.0b2.0 ---     /           .            
        3    poweroff/m0.5a2.0b2.0 ---     /           .            
        6    poweroff/m0.8a0.0b2.0 ---     /           .            
        7    poweroff/m0.8a2.0b2.0 ---     /           .            
        10   poweroff/m1.1a0.0b2.0 ---     /           .            
        11   poweroff/m1.1a2.0b2.0 ---     /           .            
        14   poweroff/m1.5a0.0b2.0 ---     /           .            
        15   poweroff/m1.5a2.0b2.0 ---     /           .            
        18   poweroff/m2.5a0.0b2.0 ---     /           .            
        19   poweroff/m2.5a2.0b2.0 ---     /           .            
        
        ---=10, 
        

:STDERR:
    * **PASS**

Command 3: Constraints (PASS)
------------------------------

:Command:
    .. code-block:: console

        $ cape -c --cons 'beta==2,Mach%1==0.5'

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.47 seconds
    * Cumulative time: 1.63 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        2    poweroff/m0.5a0.0b2.0 ---     /           .            
        3    poweroff/m0.5a2.0b2.0 ---     /           .            
        14   poweroff/m1.5a0.0b2.0 ---     /           .            
        15   poweroff/m1.5a2.0b2.0 ---     /           .            
        18   poweroff/m2.5a0.0b2.0 ---     /           .            
        19   poweroff/m2.5a2.0b2.0 ---     /           .            
        
        ---=6, 
        

:STDERR:
    * **PASS**

Command 4: Glob (PASS)
-----------------------

:Command:
    .. code-block:: console

        $ cape -c --glob 'poweroff/m0*'

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.47 seconds
    * Cumulative time: 2.10 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m0.5a0.0b0.0 ---     /           .            
        1    poweroff/m0.5a2.0b0.0 ---     /           .            
        2    poweroff/m0.5a0.0b2.0 ---     /           .            
        3    poweroff/m0.5a2.0b2.0 ---     /           .            
        4    poweroff/m0.8a0.0b0.0 ---     /           .            
        5    poweroff/m0.8a2.0b0.0 ---     /           .            
        6    poweroff/m0.8a0.0b2.0 ---     /           .            
        7    poweroff/m0.8a2.0b2.0 ---     /           .            
        
        ---=8, 
        

:STDERR:
    * **PASS**

Command 5: Regular Expression (PASS)
-------------------------------------

:Command:
    .. code-block:: console

        $ cape -c --re 'm.\.5.*b2'

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.47 seconds
    * Cumulative time: 2.57 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        2    poweroff/m0.5a0.0b2.0 ---     /           .            
        3    poweroff/m0.5a2.0b2.0 ---     /           .            
        14   poweroff/m1.5a0.0b2.0 ---     /           .            
        15   poweroff/m1.5a2.0b2.0 ---     /           .            
        18   poweroff/m2.5a0.0b2.0 ---     /           .            
        19   poweroff/m2.5a2.0b2.0 ---     /           .            
        
        ---=6, 
        

:STDERR:
    * **PASS**

Command 6: Index List (PASS)
-----------------------------

:Command:
    .. code-block:: console

        $ cape -c -I 2:5,7,18:

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.59 seconds
    * Cumulative time: 3.16 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        2    poweroff/m0.5a0.0b2.0 ---     /           .            
        3    poweroff/m0.5a2.0b2.0 ---     /           .            
        4    poweroff/m0.8a0.0b0.0 ---     /           .            
        7    poweroff/m0.8a2.0b2.0 ---     /           .            
        18   poweroff/m2.5a0.0b2.0 ---     /           .            
        19   poweroff/m2.5a2.0b2.0 ---     /           .            
        
        ---=6, 
        

:STDERR:
    * **PASS**

Command 7: Compound Subsets (PASS)
-----------------------------------

:Command:
    .. code-block:: console

        $ cape -c -I 15: --cons Mach%1=0.5 --re b2

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.63 seconds
    * Cumulative time: 3.79 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        15   poweroff/m1.5a2.0b2.0 ---     /           .            
        18   poweroff/m2.5a0.0b2.0 ---     /           .            
        19   poweroff/m2.5a2.0b2.0 ---     /           .            
        
        ---=3, 
        

:STDERR:
    * **PASS**

