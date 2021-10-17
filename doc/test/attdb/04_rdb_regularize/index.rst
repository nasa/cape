
.. This documentation written by TestDriver()
   on 2021-10-17 at 01:45 PDT

Test ``04_rdb_regularize``: PASS
==================================

This test PASSED on 2021-10-17 at 01:45 PDT

This test is run in the folder:

    ``test/attdb/04_rdb_regularize/``

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
    * Command took 0.64 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        max error(regalpha) = 0.00
        max error(regbeta)  = 0.00
        monotonic(regCN): True
        

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
    * Command took 0.61 seconds
    * Cumulative time: 1.25 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        max error(regalpha) = 0.00
        max error(regbeta)  = 0.00
        monotonic(regCN): True
        

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
    * Command took 0.45 seconds
    * Cumulative time: 1.70 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        max error(regalpha) = 0.00
        max error(regbeta)  = 0.00
        monotonic(regCN): True
        

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
    * Command took 0.55 seconds
    * Cumulative time: 2.26 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        max error(regalpha) = 0.00
        max error(regbeta)  = 0.00
        monotonic(regCN): True
        

:STDERR:
    * **PASS**

