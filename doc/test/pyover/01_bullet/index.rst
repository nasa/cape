
.. This documentation written by TestDriver()
   on 2019-08-30 at 14:31 PDT

Test ``01_bullet``
====================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/pyover/01_bullet/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ pyover -I 1
        $ pyover -I 1 -c
        $ pyover -I 1 --fm
        $ python2 test_databook.py
        $ python3 test_databook.py

Command 1: Run Case 1
----------------------

:Command:
    .. code-block:: console

        $ pyover -I 1

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 128.235 seconds
    * Cumulative time: 128.235 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        1    poweroff/m0.8a4.0b0.0 ---     /           .            
          Case name: 'poweroff/m0.8a4.0b0.0' (index 1)
             Starting case 'poweroff/m0.8a4.0b0.0'
         > overrunmpi -np 24 run 01
             (PWD = '/u/wk/ddalle/usr/pycart/test/pyover/01_bullet/work/poweroff/m0.8a4.0b0.0')
             (STDOUT = 'overrun.out')
           Wall time used: 0.01 hrs (phase 0)
           Wall time used: 0.01 hrs
           Previous phase: 0.01 hrs
         > overrunmpi -np 24 run 02
             (PWD = '/u/wk/ddalle/usr/pycart/test/pyover/01_bullet/work/poweroff/m0.8a4.0b0.0')
             (STDOUT = 'overrun.out')
           Wall time used: 0.01 hrs (phase 1)
           Wall time used: 0.03 hrs
           Previous phase: 0.01 hrs
         > overrunmpi -np 24 run 03
             (PWD = '/u/wk/ddalle/usr/pycart/test/pyover/01_bullet/work/poweroff/m0.8a4.0b0.0')
             (STDOUT = 'overrun.out')
           Wall time used: 0.01 hrs (phase 2)
        
        Submitted or ran 1 job(s).
        
        ---=1, 
        

:STDERR:
    * **PASS**

Command 2: Show DONE Status
----------------------------

:Command:
    .. code-block:: console

        $ pyover -I 1 -c

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.397998 seconds
    * Cumulative time: 128.633 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        1    poweroff/m0.8a4.0b0.0 DONE    1500/1500   .        0.8 
        
        DONE=1, 
        

    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        1    poweroff/m0.8a4.0b0.0 DONE    1750/1000   .        0.8 
        
        DONE=1, 
        

:STDERR:
    * **PASS**

Command 3: Collect Aero
------------------------

:Command:
    .. code-block:: console

        $ pyover -I 1 --fm

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.664736 seconds
    * Cumulative time: 129.297 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 4: Check DataBook (Python 2)
-------------------------------------

:Command:
    .. code-block:: console

        $ python2 test_databook.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.428887 seconds
    * Cumulative time: 129.726 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        CN = 0.190
        

    * Target:

      .. code-block:: none

        CN = <valint>[0.190,0.206]
        

:STDERR:
    * **PASS**

Command 5: Check DataBook (Python 3)
-------------------------------------

:Command:
    .. code-block:: console

        $ python3 test_databook.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.740356 seconds
    * Cumulative time: 130.467 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        CN = 0.190
        

    * Target:

      .. code-block:: none

        CN = <valint>[0.190,0.206]
        

:STDERR:
    * **PASS**

