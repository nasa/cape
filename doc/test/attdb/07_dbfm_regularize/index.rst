
.. This documentation written by TestDriver()
   on 2021-04-28 at 13:25 PDT

Test ``07_dbfm_regularize``
=============================

This test is run in the folder:

    ``/u/wk/ddalle/usr/cape/test/attdb/07_dbfm_regularize/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_griddata.py
        $ python3 test01_griddata.py
        $ python2 test02_rbf.py
        $ python3 test02_rbf.py

Command 1: Regularize using griddata: Python 2
-----------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_griddata.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 1.24511 seconds
    * Cumulative time: 1.24511 seconds
:STDOUT:
    * **FAIL**
    * Actual:

      .. code-block:: none

        regularized cols:
            reg.bullet.CA : 578  reg.bullet.CY : 578  reg.bullet.CN : 578
            reg.bullet.CLL: 578  reg.bullet.CLM: 578  reg.bullet.CLN: 578
            reg.q         : 578  reg.T         : 578  reg.mach      : 578
            reg.alpha     : 578  reg.beta      : 578
        

    * Target:

      .. code-block:: none

        regularized cols:
            reg.alpha     : 578  reg.beta      : 578  reg.bullet.CA : 578
            reg.bullet.CY : 578  reg.bullet.CN : 578  reg.bullet.CLL: 578
            reg.bullet.CLM: 578  reg.bullet.CLN: 578  reg.q         : 578
            reg.T         : 578  reg.mach      : 578
        

:STDERR:
    * **PASS**

