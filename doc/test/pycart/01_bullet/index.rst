
.. This documentation written by TestDriver()
   on 2019-09-18 at 13:05 PDT

Test ``01_bullet``
====================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/pycart/01_bullet/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ pycart -c
        $ pycart -I 0
        $ pycart -I 0 --aero
        $ python2 test_databook.py
        $ python3 test_databook.py

Command 1: Run Matrix Status
-----------------------------

:Command:
    .. code-block:: console

        $ pycart -c

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.552373 seconds
    * Cumulative time: 0.552373 seconds
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

        $ pycart -I 0

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 8.04344 seconds
    * Cumulative time: 8.59582 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 3: Collect Aero Data
-----------------------------

:Command:
    .. code-block:: console

        $ pycart -I 0 --aero

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.594554 seconds
    * Cumulative time: 9.19037 seconds
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

        $ python2 test_databook.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.540842 seconds
    * Cumulative time: 9.73121 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        CA = 0.745
        

    * Target:

      .. code-block:: none

        CA = <valint>[0.744,0.746]
        

:STDERR:
    * **PASS**

Command 5: Test DataBook Value
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
    * Command took 0.906264 seconds
    * Cumulative time: 10.6375 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        CA = 0.745
        

    * Target:

      .. code-block:: none

        CA = <valint>[0.744,0.746]
        

:STDERR:
    * **PASS**

