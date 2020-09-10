
.. This documentation written by TestDriver()
   on 2020-09-10 at 12:10 PDT

Test ``02_bullet_py3``
========================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/pycart/02_bullet_py3/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ pycart3 -c
        $ pycart3 -I 0
        $ pycart3 -I 0 --aero
        $ python3 test_databook.py

Command 1: Run Matrix Status
-----------------------------

:Command:
    .. code-block:: console

        $ pycart3 -c

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.66429 seconds
    * Cumulative time: 0.66429 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m1.5a0.0b0.0 ---     /           .            
        1    poweroff/m2.0a0.0b0.0 ---     /           .            
        2    poweroff/m2.0a2.0b0.0 ---     /           .            
        3    poweroff/m2.0a2.0b2.0 ---     /           .            
        
        ---=4, 
        

:STDERR:
    * **PASS**

Command 2: Run Case 0
----------------------

:Command:
    .. code-block:: console

        $ pycart3 -I 0

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 7.10261 seconds
    * Cumulative time: 7.7669 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 3: Collect Aero Data
-----------------------------

:Command:
    .. code-block:: console

        $ pycart3 -I 0 --aero

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.635031 seconds
    * Cumulative time: 8.40193 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        Force component 'bullet_no_base'...
        poweroff/m1.5a0.0b0.0
          Adding new databook entry at iteration 200.
        Writing 1 new or updated entries
        

:STDERR:
    * **PASS**

Command 4: Test DataBook Value
-------------------------------

:Command:
    .. code-block:: console

        $ python3 test_databook.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.714622 seconds
    * Cumulative time: 9.11655 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        CA = 0.746
        

    * Target:

      .. code-block:: none

        CA = <valint>[0.744,0.746]
        

:STDERR:
    * **PASS**

