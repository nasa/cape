
.. This documentation written by TestDriver()
   on 2020-03-27 at 11:08 PDT

Test ``09_rdb_lleval``
========================

This test is run in the folder:

    ``/home/dalle/usr/pycart/test/attdb/09_rdb_lleval/``

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
    * Command took 0.513976 seconds
    * Cumulative time: 0.513976 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        mach: 0.90
        alpha: 1.50
        beta: 0.50
        bullet.dCN.size: 51
        

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
    * Command took 0.85598 seconds
    * Cumulative time: 1.36996 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        mach: 0.90
        alpha: 1.50
        beta: 0.50
        bullet.dCN.size: 51
        

:STDERR:
    * **PASS**

