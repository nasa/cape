
.. This documentation written by TestDriver()
   on 2022-01-11 at 01:45 PST

Test ``09_rdb_lleval``: PASS
==============================

This test PASSED on 2022-01-11 at 01:45 PST

This test is run in the folder:

    ``test/attdb/09_rdb_lleval/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_eval.py
        $ python3 test01_eval.py

Command 1: Interpolate line loads: Python 2 (PASS)
---------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_eval.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.61 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        mach: 0.90
        alpha: 1.50
        beta: 0.50
        bullet.dCN.size: 51
        bullet.dCN.xargs: ['bullet.x']
        

:STDERR:
    * **PASS**

Command 2: Interpolate line loads: Python 3 (PASS)
---------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_eval.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 1.01 seconds
    * Cumulative time: 1.62 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        mach: 0.90
        alpha: 1.50
        beta: 0.50
        bullet.dCN.size: 51
        bullet.dCN.xargs: ['bullet.x']
        

:STDERR:
    * **PASS**

