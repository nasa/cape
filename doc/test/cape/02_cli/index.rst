
.. This documentation written by TestDriver()
   on 2019-07-11 at 12:54 PDT

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
    * Command took 0.443961 seconds
    * Cumulative time: 0.443961 seconds
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
    * Command took 0.374665 seconds
    * Cumulative time: 0.818626 seconds
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
    * Command took 0.352112 seconds
    * Cumulative time: 1.17074 seconds
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
    * Command took 0.384191 seconds
    * Cumulative time: 1.55493 seconds
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
    * Command took 0.401355 seconds
    * Cumulative time: 1.95628 seconds
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
    * Command took 0.388858 seconds
    * Cumulative time: 2.34514 seconds
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
    * Command took 0.364444 seconds
    * Cumulative time: 2.70959 seconds
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

