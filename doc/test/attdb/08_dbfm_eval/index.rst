
.. This documentation written by TestDriver()
   on 2021-04-28 at 13:25 PDT

Test ``08_dbfm_eval``
=======================

This test is run in the folder:

    ``/u/wk/ddalle/usr/cape/test/attdb/08_dbfm_eval/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_fm.py
        $ python3 test01_fm.py
        $ python2 test02_CLMX.py
        $ python3 test02_CLMX.py
        $ python2 test03_q.py
        $ python3 test03_q.py
        $ python2 test04_aoap.py
        $ python3 test04_aoap.py

Command 1: Evaluate FM cols: Python 2
--------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_fm.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.459351 seconds
    * Cumulative time: 0.459351 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        mach: 0.90
        alpha: 1.50
        beta: 0.50
        bullet.CA: 0.24
        bullet.CY: 0.02
        bullet.CN: 0.07
        bullet.CLL: 0.00
        bullet.CLM: 0.22
        bullet.CLN: 0.07
        

:STDERR:
    * **PASS**

Command 2: Evaluate FM cols: Python 3
--------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_fm.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.517067 seconds
    * Cumulative time: 0.976418 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        mach: 0.90
        alpha: 1.50
        beta: 0.50
        bullet.CA: 0.24
        bullet.CY: 0.02
        bullet.CN: 0.07
        bullet.CLL: 0.00
        bullet.CLM: 0.22
        bullet.CLN: 0.07
        

:STDERR:
    * **PASS**

Command 3: Evaluate *CLMX* and *CLNX*: Python 2
------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_CLMX.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.419217 seconds
    * Cumulative time: 1.39563 seconds
:STDOUT:
    * **FAIL**
    * Actual:

      .. code-block:: none

        mach : 0.90
        alpha: 1.50
        beta : 0.50
        xMRP : 2.00
        bullet.CLM : 0.218
        bullet.CLMX: 0.352
        bullet.CLN : 0.073
        bullet.CLNX: 0.117
        

    * Target:

      .. code-block:: none

        mach : 0.90
        alpha: 1.50
        beta : 0.50
        xMRP : 2.00
        bullet.CLM : 0.219
        bullet.CLMX: 0.353
        bullet.CLN : 0.073
        bullet.CLNX: 0.118
        

:STDERR:
    * **PASS**

