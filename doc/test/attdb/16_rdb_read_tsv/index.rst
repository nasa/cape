
.. This documentation written by TestDriver()
   on 2021-01-14 at 23:14 PST

Test ``16_rdb_read_tsv``
==========================

This test is run in the folder:

    ``/home/dalle/usr/cape/test/attdb/16_rdb_read_tsv/``

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
    * Command took 0.180236 seconds
    * Cumulative time: 0.180236 seconds
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
    * Command took 0.27769 seconds
    * Cumulative time: 0.457926 seconds
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
    * Command took 0.180847 seconds
    * Cumulative time: 0.638773 seconds
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
    * Command took 0.280217 seconds
    * Cumulative time: 0.91899 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        alpha: float64
        beta: float64
        CN: float64
        

:STDERR:
    * **PASS**
