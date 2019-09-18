
.. This documentation written by TestDriver()
   on 2019-09-18 at 12:34 PDT

Test ``02_cli``
=================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/cape/02_cli/``

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

Command 1: Status
------------------

:Command:
    .. code-block:: console

        $ cape -c

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.530888 seconds
    * Cumulative time: 0.530888 seconds
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

Command 2: Filter
------------------

:Command:
    .. code-block:: console

        $ cape -c --filter b2

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.544177 seconds
    * Cumulative time: 1.07506 seconds
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

Command 3: Constraints
-----------------------

:Command:
    .. code-block:: console

        $ cape -c --cons 'beta==2,Mach%1==0.5'

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.554111 seconds
    * Cumulative time: 1.62918 seconds
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

Command 4: Glob
----------------

:Command:
    .. code-block:: console

        $ cape -c --glob 'poweroff/m0*'

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.518918 seconds
    * Cumulative time: 2.14809 seconds
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

Command 5: Regular Expression
------------------------------

:Command:
    .. code-block:: console

        $ cape -c --re 'm.\.5.*b2'

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.53187 seconds
    * Cumulative time: 2.67996 seconds
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

Command 6: Index List
----------------------

:Command:
    .. code-block:: console

        $ cape -c -I 2:5,7,18:

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.535938 seconds
    * Cumulative time: 3.2159 seconds
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

Command 7: Compound Subsets
----------------------------

:Command:
    .. code-block:: console

        $ cape -c -I 15: --cons Mach%1=0.5 --re b2

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.546682 seconds
    * Cumulative time: 3.76258 seconds
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

