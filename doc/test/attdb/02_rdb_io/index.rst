
.. This documentation written by TestDriver()
   on 2021-04-28 at 13:51 PDT

Test ``02_rdb_io``
====================

This test is run in the folder:

    ``/u/wk/ddalle/usr/cape/test/attdb/02_rdb_io/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_csv_to_mat.py
        $ python3 test01_csv_to_mat.py

Command 1: CSV read; MAT write/read: Python 2
----------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_csv_to_mat.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.581909 seconds
    * Cumulative time: 0.581909 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case 13: m0.95a5.00 CA=0.526
        Case 13: m0.95a5.00 CA=0.526
        

:STDERR:
    * **PASS**

Command 2: CSV read; MAT write/read: Python 3
----------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_csv_to_mat.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.616326 seconds
    * Cumulative time: 1.19824 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case 13: m0.95a5.00 CA=0.526
        Case 13: m0.95a5.00 CA=0.526
        

:STDERR:
    * **PASS**

