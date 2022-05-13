
.. This documentation written by TestDriver()
   on 2022-05-13 at 15:17 PDT

Test ``02_rdb_io``: PASS
==========================

This test PASSED on 2022-05-13 at 15:17 PDT

This test is run in the folder:

    ``test/attdb/02_rdb_io/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_csv_to_mat.py
        $ python3 test01_csv_to_mat.py

Command 1: CSV read; MAT write/read: Python 2 (PASS)
-----------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_csv_to_mat.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.41 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case 13: m0.95a5.00 CA=0.526
        Case 13: m0.95a5.00 CA=0.526
        

:STDERR:
    * **PASS**

Command 2: CSV read; MAT write/read: Python 3 (PASS)
-----------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_csv_to_mat.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.52 seconds
    * Cumulative time: 0.93 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case 13: m0.95a5.00 CA=0.526
        Case 13: m0.95a5.00 CA=0.526
        

:STDERR:
    * **PASS**

