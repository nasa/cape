:Compare Files:
    * **PASS**
    * Target:
        - :download:`FILE-target-00-00.txt`

:Compare Files:
    * **PASS**
    * Target:
        - :download:`FILE-target-01-00.txt`

:Compare Files:
    * **PASS**
    * Target:
        - :download:`FILE-target-02-00.txt`

:Compare Files:
    * **PASS**
    * Target:
        - :download:`FILE-target-03-00.txt`


.. This documentation written by TestDriver()
   on 2022-04-16 at 01:45 PDT

Test ``11_rdb_writecsv``: PASS
================================

This test PASSED on 2022-04-16 at 01:45 PDT

This test is run in the folder:

    ``test/attdb/11_rdb_writecsv/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_csv_dense.py
        $ python3 test01_csv_dense.py
        $ python2 test02_csv_default.py
        $ python3 test02_csv_default.py

Command 1: Simple dense CSV writer: Python 2 (PASS)
----------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_csv_dense.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.62 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 2: Simple dense CSV writer: Python 3 (PASS)
----------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_csv_dense.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.80 seconds
    * Cumulative time: 1.42 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 3: CSV writer with defaults: Python 2 (PASS)
-----------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_csv_default.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.41 seconds
    * Cumulative time: 1.83 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 4: CSV writer with defaults: Python 3 (PASS)
-----------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_csv_default.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.51 seconds
    * Cumulative time: 2.34 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

