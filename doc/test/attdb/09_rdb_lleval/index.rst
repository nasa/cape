
.. This documentation written by TestDriver()
   on 2021-03-19 at 09:48 PDT

Test ``09_rdb_lleval``
========================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/attdb/09_rdb_lleval/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_eval.py
        $ python3 test01_eval.py

Command 1: Interpolate line loads: Python 2
--------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_eval.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 13.7837 seconds
    * Cumulative time: 13.7837 seconds
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

Command 2: Interpolate line loads: Python 3
--------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_eval.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 5.00335 seconds
    * Cumulative time: 18.7871 seconds
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

