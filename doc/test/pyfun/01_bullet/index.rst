
.. This documentation written by TestDriver()
   on 2021-12-01 at 01:41 PST

Test ``01_bullet``: PASS
==========================

This test PASSED on 2021-12-01 at 01:41 PST

This test is run in the folder:

    ``test/pyfun/01_bullet/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ pyfun -I 8
        $ pyfun -I 8 -c
        $ pyfun -I 8 --fm
        $ python2 test_databook.py
        $ python3 test_databook.py

Command 1: Run Case 8 (PASS)
-----------------------------

:Command:
    .. code-block:: console

        $ pyfun -I 8

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 39.24 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        8    bullet/m1.10a0.0b0.0  ---     /           .            
          Case name: 'bullet/m1.10a0.0b0.0' (index 8)
             Starting case 'bullet/m1.10a0.0b0.0'
         > nodet --animation_freq 100
             (PWD = 'bullet/m1.10a0.0b0.0/')
             (STDOUT = 'fun3d.out')
         > nodet --animation_freq 100
             (PWD = 'bullet/m1.10a0.0b0.0/')
             (STDOUT = 'fun3d.out')
        
        Submitted or ran 1 job(s).
        
        ---=1, 
        

:STDERR:
    * **PASS**

Command 2: Show DONE Status (PASS)
-----------------------------------

:Command:
    .. code-block:: console

        $ pyfun -I 8 -c

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.70 seconds
    * Cumulative time: 39.94 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        8    bullet/m1.10a0.0b0.0  DONE    200/200     .        0.1 
        
        DONE=1, 
        

:STDERR:
    * **PASS**

Command 3: Collect Aero (PASS)
-------------------------------

:Command:
    .. code-block:: console

        $ pyfun -I 8 --fm

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.98 seconds
    * Cumulative time: 40.92 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 4: Check DataBook (Python 2) (PASS)
--------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test_databook.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.62 seconds
    * Cumulative time: 41.55 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        CA = 0.461
        

    * Target:

      .. code-block:: none

        CA = <valint>[0.460,0.462]
        

:STDERR:
    * **PASS**

Command 5: Check DataBook (Python 3) (PASS)
--------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test_databook.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.79 seconds
    * Cumulative time: 42.33 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        CA = 0.461
        

    * Target:

      .. code-block:: none

        CA = <valint>[0.460,0.462]
        

:STDERR:
    * **PASS**

