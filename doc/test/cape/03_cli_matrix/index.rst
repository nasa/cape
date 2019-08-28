
.. This documentation written by TestDriver()
   on 2019-08-28 at 13:13 PDT

Test ``03_cli_matrix``
========================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/cape/03_cli_matrix/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ cape -c
        $ cape -c -f cape-json.json
        $ cape -c -f cape-json.json --re a0
        $ cape -c -f cape-mixed.json
        $ cape -c -f cape-mixed.json --re a2

Command 1: Missing JSON File
-----------------------------

:Command:
    .. code-block:: console

        $ cape -c

:Return Code:
    * **PASS**
    * Output: ``1``
    * Target: ``1``
:Time Taken:
    * **PASS**
    * Command took 0.671469 seconds
    * Cumulative time: 0.671469 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case Config/Run Directory   Status  Iterations  Que CPU Time 
        ---- ---------------------- ------- ----------- --- --------
        0    poweroff/m0.80a0.0b0.0 ---     /           .            
        1    poweroff/m0.80a4.0b0.0 ---     /           .            
        2    poweroff/m0.90a0.0b0.0 ---     /           .            
        3    poweroff/m0.90a4.0b0.0 ---     /           .            
        4    poweroff/m1.05a0.0b0.0 ---     /           .            
        5    poweroff/m1.05a4.0b0.0 ---     /           .            
        
        ---=6, 
        

:STDERR:
    * **PASS**
    * Target:

      .. code-block:: pytb

        ValueError: No cape control file 'cape.json' found
        


Command 2: JSON-only Matrix
----------------------------

:Command:
    .. code-block:: console

        $ cape -c -f cape-json.json

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.425357 seconds
    * Cumulative time: 1.09683 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case Config/Run Directory   Status  Iterations  Que CPU Time 
        ---- ---------------------- ------- ----------- --- --------
        0    poweroff/m0.80a0.0b0.0 ---     /           .            
        1    poweroff/m0.80a4.0b0.0 ---     /           .            
        2    poweroff/m0.90a0.0b0.0 ---     /           .            
        3    poweroff/m0.90a4.0b0.0 ---     /           .            
        4    poweroff/m1.05a0.0b0.0 ---     /           .            
        5    poweroff/m1.05a4.0b0.0 ---     /           .            
        
        ---=6, 
        

:STDERR:
    * **PASS**

Command 3: JSON-only with RegEx
--------------------------------

:Command:
    .. code-block:: console

        $ cape -c -f cape-json.json --re a0

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.412908 seconds
    * Cumulative time: 1.50973 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case Config/Run Directory   Status  Iterations  Que CPU Time 
        ---- ---------------------- ------- ----------- --- --------
        0    poweroff/m0.80a0.0b0.0 ---     /           .            
        2    poweroff/m0.90a0.0b0.0 ---     /           .            
        4    poweroff/m1.05a0.0b0.0 ---     /           .            
        
        ---=3, 
        

:STDERR:
    * **PASS**

Command 4: Mixed CSV and JSON
------------------------------

:Command:
    .. code-block:: console

        $ cape -c -f cape-mixed.json

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.381979 seconds
    * Cumulative time: 1.89171 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m0.5a0.0b4.0 ---     /           .            
        1    poweroff/m0.5a2.0b4.0 ---     /           .            
        2    poweroff/m0.8a0.0b4.0 ---     /           .            
        3    poweroff/m0.8a2.0b4.0 ---     /           .            
        4    poweroff/m1.1a0.0b4.0 ---     /           .            
        5    poweroff/m1.1a2.0b4.0 ---     /           .            
        6    poweroff/m1.5a0.0b4.0 ---     /           .            
        7    poweroff/m1.5a2.0b4.0 ---     /           .            
        8    poweroff/m2.5a0.0b4.0 ---     /           .            
        9    poweroff/m2.5a2.0b4.0 ---     /           .            
        
        ---=10, 
        

:STDERR:
    * **PASS**

Command 5: Mixed CSV and JSON with RegEx
-----------------------------------------

:Command:
    .. code-block:: console

        $ cape -c -f cape-mixed.json --re a2

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.406806 seconds
    * Cumulative time: 2.29852 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        1    poweroff/m0.5a2.0b4.0 ---     /           .            
        3    poweroff/m0.8a2.0b4.0 ---     /           .            
        5    poweroff/m1.1a2.0b4.0 ---     /           .            
        7    poweroff/m1.5a2.0b4.0 ---     /           .            
        9    poweroff/m2.5a2.0b4.0 ---     /           .            
        
        ---=5, 
        

:STDERR:
    * **PASS**

