
.. This documentation written by TestDriver()
   on 2021-04-28 at 13:25 PDT

Test ``11_rdb_writecsv``
==========================

This test is run in the folder:

    ``/u/wk/ddalle/usr/cape/test/attdb/11_rdb_writecsv/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_csv_dense.py
        $ python3 test01_csv_dense.py
        $ python2 test02_csv_default.py
        $ python3 test02_csv_default.py

Command 1: Simple dense CSV writer: Python 2
---------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_csv_dense.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.548811 seconds
    * Cumulative time: 0.548811 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

:Compare Files:
    * **PASS**
    * Target:
        - :download:`FILE-target-00-00.txt`

Command 2: Simple dense CSV writer: Python 3
---------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_csv_dense.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.764456 seconds
    * Cumulative time: 1.31327 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

:Compare Files:
    * **PASS**
    * Target:
        - :download:`FILE-target-01-00.txt`

Command 3: CSV writer with defaults: Python 2
----------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_csv_default.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.488855 seconds
    * Cumulative time: 1.80212 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

:Compare Files:
    * **PASS**
    * Target:
        - :download:`FILE-target-02-00.txt`

Command 4: CSV writer with defaults: Python 3
----------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_csv_default.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.57547 seconds
    * Cumulative time: 2.37759 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

:Compare Files:
    * **PASS**
    * Target:
        - :download:`FILE-target-03-00.txt`

