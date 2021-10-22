
.. This documentation written by TestDriver()
   on 2021-10-22 at 01:45 PDT

Test ``05_rdb_llreg``: PASS
=============================

This test PASSED on 2021-10-22 at 01:45 PDT

This test is run in the folder:

    ``test/attdb/05_rdb_llreg/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_griddata.py
        $ python3 test01_griddata.py

Command 1: Regularize line load using griddata: Python 2 (PASS)
----------------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_griddata.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.47 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        reg.bullet.dCN.shape = [51, 578]
        

:STDERR:
    * **PASS**

Command 2: Regularize line load using griddata: Python 3 (PASS)
----------------------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_griddata.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.56 seconds
    * Cumulative time: 1.03 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        reg.bullet.dCN.shape = [51, 578]
        

:STDERR:
    * **PASS**

