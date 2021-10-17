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
   on 2021-10-17 at 01:45 PDT

Test ``15_rdb_write_tsv``: PASS
=================================

This test PASSED on 2021-10-17 at 01:45 PDT

This test is run in the folder:

    ``test/attdb/15_rdb_write_tsv/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_tsv_dense.py
        $ python3 test01_tsv_dense.py
        $ python2 test02_tsv_default.py
        $ python3 test02_tsv_default.py

Command 1: Simple dense TSV writer: Python 2 (PASS)
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
    * Command took 0.56 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 2: Simple dense TSV writer: Python 3 (PASS)
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
    * Command took 0.59 seconds
    * Cumulative time: 1.15 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 3: TSV writer with defaults: Python 2 (PASS)
-----------------------------------------------------

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
    * Cumulative time: 1.57 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 4: TSV writer with defaults: Python 3 (PASS)
-----------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_tsv_default.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.50 seconds
    * Cumulative time: 2.07 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

