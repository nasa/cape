
.. This documentation written by TestDriver()
   on 2020-03-26 at 09:06 PDT

Test ``07_dbfm_regularize``
=============================

This test is run in the folder:

    ``/home/dalle/usr/pycart/test/attdb/07_dbfm_regularize/``

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
    * Command took 0.475306 seconds
    * Cumulative time: 0.475306 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        regularized cols:
            reg.alpha     : 578  reg.beta      : 578  reg.bullet.CA : 578
            reg.bullet.CY : 578  reg.bullet.CN : 578  reg.bullet.CLL: 578
            reg.bullet.CLM: 578  reg.bullet.CLN: 578  reg.q         : 578
            reg.T         : 578  reg.mach      : 578
        

:STDERR:
    * **PASS**

Command 2: Regularize using griddata: Python 3
-----------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_griddata.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 1.10313 seconds
    * Cumulative time: 1.57843 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        regularized cols:
            reg.alpha     : 578  reg.beta      : 578  reg.bullet.CA : 578
            reg.bullet.CY : 578  reg.bullet.CN : 578  reg.bullet.CLL: 578
            reg.bullet.CLM: 578  reg.bullet.CLN: 578  reg.q         : 578
            reg.T         : 578  reg.mach      : 578
        

:STDERR:
    * **PASS**

Command 3: Regularize using RBF: Python 2
------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_rbf.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.274116 seconds
    * Cumulative time: 1.85255 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        regularized cols:
            reg.alpha     : 578  reg.beta      : 578  reg.bullet.CA : 578
            reg.bullet.CY : 578  reg.bullet.CN : 578  reg.bullet.CLL: 578
            reg.bullet.CLM: 578  reg.bullet.CLN: 578  reg.q         : 578
            reg.T         : 578  reg.mach      : 578
        

:STDERR:
    * **PASS**

Command 4: Regularize using RBF: Python 3
------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_rbf.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.546057 seconds
    * Cumulative time: 2.39861 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        regularized cols:
            reg.alpha     : 578  reg.beta      : 578  reg.bullet.CA : 578
            reg.bullet.CY : 578  reg.bullet.CN : 578  reg.bullet.CLL: 578
            reg.bullet.CLM: 578  reg.bullet.CLN: 578  reg.q         : 578
            reg.T         : 578  reg.mach      : 578
        

:STDERR:
    * **PASS**

