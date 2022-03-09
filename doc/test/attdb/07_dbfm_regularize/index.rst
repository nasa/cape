
.. This documentation written by TestDriver()
   on 2022-03-09 at 01:45 PST

Test ``07_dbfm_regularize``: PASS
===================================

This test PASSED on 2022-03-09 at 01:45 PST

This test is run in the folder:

    ``test/attdb/07_dbfm_regularize/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_griddata.py
        $ python3 test01_griddata.py
        $ python2 test02_rbf.py
        $ python3 test02_rbf.py

Command 1: Regularize using griddata: Python 2 (PASS)
------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_griddata.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 1.44 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        regularized cols:
            reg.bullet.CA : 578  reg.bullet.CY : 578  reg.bullet.CN : 578
            reg.bullet.CLL: 578  reg.bullet.CLM: 578  reg.bullet.CLN: 578
            reg.q         : 578  reg.T         : 578  reg.mach      : 578
            reg.alpha     : 578  reg.beta      : 578
        

:STDERR:
    * **PASS**

Command 2: Regularize using griddata: Python 3 (PASS)
------------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_griddata.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 1.10 seconds
    * Cumulative time: 2.54 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        regularized cols:
            reg.bullet.CA : 578  reg.bullet.CY : 578  reg.bullet.CN : 578
            reg.bullet.CLL: 578  reg.bullet.CLM: 578  reg.bullet.CLN: 578
            reg.mach      : 578  reg.q         : 578  reg.T         : 578
            reg.alpha     : 578  reg.beta      : 578
        

:STDERR:
    * **PASS**

Command 3: Regularize using RBF: Python 2 (PASS)
-------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_rbf.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.41 seconds
    * Cumulative time: 2.95 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        regularized cols:
            reg.bullet.CA : 578  reg.bullet.CY : 578  reg.bullet.CN : 578
            reg.bullet.CLL: 578  reg.bullet.CLM: 578  reg.bullet.CLN: 578
            reg.alpha     : 578  reg.beta      : 578  reg.mach      : 578
        

:STDERR:
    * **PASS**

Command 4: Regularize using RBF: Python 3 (PASS)
-------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_rbf.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.79 seconds
    * Cumulative time: 3.74 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        regularized cols:
            reg.bullet.CA : 578  reg.bullet.CY : 578  reg.bullet.CN : 578
            reg.bullet.CLL: 578  reg.bullet.CLM: 578  reg.bullet.CLN: 578
            reg.alpha     : 578  reg.beta      : 578  reg.mach      : 578
        

:STDERR:
    * **PASS**

