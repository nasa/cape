
.. This documentation written by TestDriver()
   on 2021-04-28 at 13:51 PDT

Test ``15_rdb_write_tsv``
===========================

This test is run in the folder:

    ``/u/wk/ddalle/usr/cape/test/attdb/15_rdb_write_tsv/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_tsv_dense.py
        $ python3 test01_tsv_dense.py
        $ python2 test02_tsv_default.py
        $ python3 test02_tsv_default.py

Command 1: Simple dense TSV writer: Python 2
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
    * Command took 0.502562 seconds
    * Cumulative time: 0.502562 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

:Compare Files:
    * **PASS**
    * Target:
        - :download:`FILE-target-00-00.txt`

Command 2: Simple dense TSV writer: Python 3
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
    * Command took 0.784196 seconds
    * Cumulative time: 1.28676 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

:Compare Files:
    * **PASS**
    * Target:
        - :download:`FILE-target-01-00.txt`

Command 3: TSV writer with defaults: Python 2
----------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_tsv_default.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.411685 seconds
    * Cumulative time: 1.69844 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

:Compare Files:
    * **PASS**
    * Target:
        - :download:`FILE-target-02-00.txt`

Command 4: TSV writer with defaults: Python 3
----------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_tsv_default.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.513966 seconds
    * Cumulative time: 2.21241 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

:Compare Files:
    * **PASS**
    * Target:
        - :download:`FILE-target-03-00.txt`

