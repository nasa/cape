
.. This documentation written by TestDriver()
   on 2022-02-08 at 01:46 PST

Test ``16_rdb_read_tsv``: PASS
================================

This test PASSED on 2022-02-08 at 01:46 PST

This test is run in the folder:

    ``test/attdb/16_rdb_read_tsv/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_tsv_dense.py
        $ python3 test01_tsv_dense.py
        $ python2 test02_tsv_default.py
        $ python3 test02_tsv_default.py

Command 1: Simple dense TSV reader: Python 2 (PASS)
----------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_tsv_dense.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.59 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        alpha: float64
        beta: float64
        CN: float64
        

:STDERR:
    * **PASS**

Command 2: Simple dense TSV reader: Python 3 (PASS)
----------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_tsv_dense.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.78 seconds
    * Cumulative time: 1.37 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        alpha: float64
        beta: float64
        CN: float64
        

:STDERR:
    * **PASS**

Command 3: Main TSV reader: Python 2 (PASS)
--------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_tsv_default.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.42 seconds
    * Cumulative time: 1.78 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        alpha: float64
        beta: float64
        CN: float64
        

:STDERR:
    * **PASS**

Command 4: Main TSV reader: Python 3 (PASS)
--------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_tsv_default.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.51 seconds
    * Cumulative time: 2.30 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        alpha: float64
        beta: float64
        CN: float64
        

:STDERR:
    * **PASS**

