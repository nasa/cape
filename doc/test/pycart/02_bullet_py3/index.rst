
.. This documentation written by TestDriver()
   on 2021-10-16 at 01:40 PDT

Test ``02_bullet_py3``: PASS
==============================

This test PASSED on 2021-10-16 at 01:40 PDT

This test is run in the folder:

    ``test/pycart/02_bullet_py3/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python3 -m cape.pycart -c
        $ python3 -m cape.pycart -I 0
        $ python3 -m cape.pycart -I 0 --aero
        $ python3 test_databook.py

Command 1: Run Matrix Status (PASS)
------------------------------------

:Command:
    .. code-block:: console

        $ python3 -m cape.pycart -c

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.79 seconds
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

Command 2: Run Case 0 (PASS)
-----------------------------

:Command:
    .. code-block:: console

        $ python3 -m cape.pycart -I 0

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 6.21 seconds
    * Cumulative time: 6.99 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 3: Collect Aero Data (PASS)
------------------------------------

:Command:
    .. code-block:: console

        $ python3 -m cape.pycart -I 0 --aero

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.58 seconds
    * Cumulative time: 7.58 seconds
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

Command 4: Test DataBook Value (PASS)
--------------------------------------

:Command:
    .. code-block:: console

        $ python3 test_databook.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.57 seconds
    * Cumulative time: 8.14 seconds
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

