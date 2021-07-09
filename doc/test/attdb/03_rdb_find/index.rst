
.. This documentation written by TestDriver()
   on 2021-04-28 at 13:51 PDT

Test ``03_rdb_find``
======================

This test is run in the folder:

    ``/u/wk/ddalle/usr/cape/test/attdb/03_rdb_find/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_find.py
        $ python3 test01_find.py
        $ python2 test02_multi.py
        $ python3 test02_multi.py

Command 1: Basic DataKit search: Python 2
------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_find.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.438374 seconds
    * Cumulative time: 0.438374 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case 12: CT=0.0
        Case 15: CT=1.4
        Case 17: CT=1.4
        

:STDERR:
    * **PASS**

Command 2: Basic DataKit search: Python 3
------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_find.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.511946 seconds
    * Cumulative time: 0.95032 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case 12: CT=0.0
        Case 15: CT=1.4
        Case 17: CT=1.4
        

:STDERR:
    * **PASS**

Command 3: Multiple-condition search: Python 2
-----------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_multi.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.487851 seconds
    * Cumulative time: 1.43817 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Select alpha,beta, display all
          Case 02: m0.50a2.0b0.0_CT0.0
          Case 06: m0.70a2.0b0.0_CT0.0
          Case 09: m0.70a2.0b0.0_CT2.1
          Case 12: m0.90a2.0b0.0_CT0.0
          Case 15: m0.90a2.0b0.0_CT1.4
          Case 17: m0.90a2.0b0.0_CT1.4
        Select alpha,beta, match once
          Case 02, match 00: m0.50a2.0b0.0_CT0.0
          Case 06, match 01: m0.70a2.0b0.0_CT0.0
          Case 12, match 02: m0.90a2.0b0.0_CT0.0
        Select alpha,beta, map all matches
          Case 02, match 00: m0.50a2.0b0.0_CT0.0
          Case 06, match 01: m0.70a2.0b0.0_CT0.0
          Case 09, match 01: m0.70a2.0b0.0_CT2.1
          Case 12, match 02: m0.90a2.0b0.0_CT0.0
          Case 15, match 02: m0.90a2.0b0.0_CT1.4
          Case 17, match 02: m0.90a2.0b0.0_CT1.4
        

:STDERR:
    * **PASS**

Command 4: Multiple-condition search: Python 3
-----------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_multi.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.566599 seconds
    * Cumulative time: 2.00477 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Select alpha,beta, display all
          Case 02: m0.50a2.0b0.0_CT0.0
          Case 06: m0.70a2.0b0.0_CT0.0
          Case 09: m0.70a2.0b0.0_CT2.1
          Case 12: m0.90a2.0b0.0_CT0.0
          Case 15: m0.90a2.0b0.0_CT1.4
          Case 17: m0.90a2.0b0.0_CT1.4
        Select alpha,beta, match once
          Case 02, match 00: m0.50a2.0b0.0_CT0.0
          Case 06, match 01: m0.70a2.0b0.0_CT0.0
          Case 12, match 02: m0.90a2.0b0.0_CT0.0
        Select alpha,beta, map all matches
          Case 02, match 00: m0.50a2.0b0.0_CT0.0
          Case 06, match 01: m0.70a2.0b0.0_CT0.0
          Case 09, match 01: m0.70a2.0b0.0_CT2.1
          Case 12, match 02: m0.90a2.0b0.0_CT0.0
          Case 15, match 02: m0.90a2.0b0.0_CT1.4
          Case 17, match 02: m0.90a2.0b0.0_CT1.4
        

:STDERR:
    * **PASS**
