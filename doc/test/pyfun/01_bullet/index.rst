
.. This documentation written by TestDriver()
   on 2019-08-29 at 14:36 PDT

Test ``01_bullet``
====================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/pyfun/01_bullet/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ pyfun -I 8
        $ pyfun -I 8 -c
        $ pyfun -I 8 --fm
        $ python2 test_databook.py
        $ python3 test_databook.py

Command 1: Run Case 8
----------------------

:Command:
    .. code-block:: console

        $ pyfun -I 8

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 46.6189 seconds
    * Cumulative time: 46.6189 seconds
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
             (PWD = '/u/wk/ddalle/usr/pycart/test/pyfun/01_bullet/work/bullet/m1.10a0.0b0.0')
             (STDOUT = 'fun3d.out')
         > nodet --animation_freq 100
             (PWD = '/u/wk/ddalle/usr/pycart/test/pyfun/01_bullet/work/bullet/m1.10a0.0b0.0')
             (STDOUT = 'fun3d.out')
        
        Submitted or ran 1 job(s).
        
        ---=1, 
        

:STDERR:
    * **PASS**

Command 2: Show DONE Status
----------------------------

:Command:
    .. code-block:: console

        $ pyfun -I 8 -c

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.656991 seconds
    * Cumulative time: 47.2759 seconds
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

Command 3: Collect Aero
------------------------

:Command:
    .. code-block:: console

        $ pyfun -I 8 --fm

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.901045 seconds
    * Cumulative time: 48.1769 seconds
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
    * Command took 0.425219 seconds
    * Cumulative time: 48.6021 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        CA = <valint>[0.460,0.462]
        

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
    * Command took 0.897933 seconds
    * Cumulative time: 49.5001 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        CA = <valint>[0.460,0.462]
        

:STDERR:
    * **PASS**

