
.. This documentation written by TestDriver()
   on 2021-04-28 at 13:25 PDT

Test ``16_rdb_read_tsv``
==========================

This test is run in the folder:

    ``/u/wk/ddalle/usr/cape/test/attdb/16_rdb_read_tsv/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_tsv_dense.py
        $ python3 test01_tsv_dense.py
        $ python2 test02_tsv_default.py
        $ python3 test02_tsv_default.py

Command 1: Simple dense TSV reader: Python 2
---------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_tsv_dense.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.605356 seconds
    * Cumulative time: 0.605356 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        alpha: float64
        beta: float64
        CN: float64
        

:STDERR:
    * **PASS**

Command 2: Simple dense TSV reader: Python 3
---------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_tsv_dense.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.572166 seconds
    * Cumulative time: 1.17752 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        alpha: float64
        beta: float64
        CN: float64
        

:STDERR:
    * **PASS**

Command 3: Main TSV reader: Python 2
-------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_tsv_default.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.591782 seconds
    * Cumulative time: 1.7693 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        alpha: float64
        beta: float64
        CN: float64
        

:STDERR:
    * **PASS**

Command 4: Main TSV reader: Python 3
-------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_tsv_default.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.804888 seconds
    * Cumulative time: 2.57419 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        alpha: float64
        beta: float64
        CN: float64
        

:STDERR:
    * **PASS**

