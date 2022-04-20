
.. This documentation written by TestDriver()
   on 2022-04-20 at 01:43 PDT

Test ``01_bullet``: PASS
==========================

This test PASSED on 2022-04-20 at 01:43 PDT

This test is run in the folder:

    ``test/pyover/01_bullet/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ pyover -I 1
        $ pyover -I 1 -c
        $ pyover -I 1 --fm
        $ python2 test_databook.py
        $ python3 test_databook.py

Command 1: Run Case 1 (PASS)
-----------------------------

:Command:
    .. code-block:: console

        $ pyover -I 1

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 101.96 seconds
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
             (PWD = 'poweroff/m0.8a4.0b0.0/')
             (STDOUT = 'overrun.out')
           Wall time used: 0.01 hrs (phase 0)
           Wall time used: 0.01 hrs
           Previous phase: 0.01 hrs
         > overrunmpi -np 24 run 02
             (PWD = 'poweroff/m0.8a4.0b0.0/')
             (STDOUT = 'overrun.out')
           Wall time used: 0.01 hrs (phase 1)
           Wall time used: 0.02 hrs
           Previous phase: 0.01 hrs
         > overrunmpi -np 24 run 03
             (PWD = 'poweroff/m0.8a4.0b0.0/')
             (STDOUT = 'overrun.out')
           Wall time used: 0.01 hrs (phase 2)
        
        Submitted or ran 1 job(s).
        
        ---=1, 
        

:STDERR:
    * **PASS**

Command 2: Show DONE Status (PASS)
-----------------------------------

:Command:
    .. code-block:: console

        $ pyover -I 1 -c

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.54 seconds
    * Cumulative time: 102.51 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        1    poweroff/m0.8a4.0b0.0 DONE    1500/1500   .        0.7 
        
        DONE=1, 
        

    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        1    poweroff/m0.8a4.0b0.0 DONE    1500/1500   .   ...
        
        DONE=1, 
        

:STDERR:
    * **PASS**

Command 3: Collect Aero (PASS)
-------------------------------

:Command:
    .. code-block:: console

        $ pyover -I 1 --fm

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.89 seconds
    * Cumulative time: 103.40 seconds
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
    * Command took 0.64 seconds
    * Cumulative time: 104.04 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        CN = 0.221
        

    * Target:

      .. code-block:: none

        CN = <valint>[0.190,0.226]
        

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
    * Command took 0.81 seconds
    * Cumulative time: 104.85 seconds
:STDOUT:
    * **PASS**
    * Actual:

      .. code-block:: none

        CN = 0.221
        

    * Target:

      .. code-block:: none

        CN = <valint>[0.190,0.226]
        

:STDERR:
    * **PASS**

