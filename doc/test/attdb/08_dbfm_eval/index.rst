
.. This documentation written by TestDriver()
   on 2021-04-28 at 13:51 PDT

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
    * Command took 0.559291 seconds
    * Cumulative time: 0.559291 seconds
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
    * Command took 0.718369 seconds
    * Cumulative time: 1.27766 seconds
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
    * Command took 0.446528 seconds
    * Cumulative time: 1.72419 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        mach : 0.90
        alpha: 1.50
        beta : 0.50
        xMRP : 2.00
        bullet.CLM : 0.218
        bullet.CLMX: 0.352
        bullet.CLN : 0.073
        bullet.CLNX: 0.117
        

:STDERR:
    * **PASS**

Command 4: Evaluate *CLMX* and *CLNX*: Python 3
------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test02_CLMX.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.558975 seconds
    * Cumulative time: 2.28316 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        mach : 0.90
        alpha: 1.50
        beta : 0.50
        xMRP : 2.00
        bullet.CLM : 0.218
        bullet.CLMX: 0.352
        bullet.CLN : 0.073
        bullet.CLNX: 0.117
        

:STDERR:
    * **PASS**

Command 5: Evaluate *q* and *T*: Python 2
------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test03_q.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.440592 seconds
    * Cumulative time: 2.72376 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        mach: 0.90
        q: 1250.00
        T: 475.33
        

:STDERR:
    * **PASS**

Command 6: Evaluate *q* and *T*: Python 3
------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test03_q.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.537341 seconds
    * Cumulative time: 3.2611 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        mach: 0.90
        q: 1250.00
        T: 475.33
        

:STDERR:
    * **PASS**

Command 7: Process *aoap* and *phip*: Python 2
-----------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test04_aoap.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.439367 seconds
    * Cumulative time: 3.70046 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        mach: 0.90
        aoa : 1.50
        beta: 0.50
        aoap: 1.5811
        phip: 18.4373
        bullet.CA : 0.241 0.241 0.241
        bullet.CY : 0.022 0.022 0.022
        bullet.CN : 0.067 0.067 0.067
        bullet.CLL: 0.000 0.000 0.000
        bullet.CLM: 0.218 0.218 0.218
        bullet.CLN: 0.073 0.073 0.073
        aoap: size=578, dtype=float64
        phip: size=578, dtype=float64
        

:STDERR:
    * **PASS**

Command 8: Process *aoap* and *phip*: Python 3
-----------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test04_aoap.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.53397 seconds
    * Cumulative time: 4.23443 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        mach: 0.90
        aoa : 1.50
        beta: 0.50
        aoap: 1.5811
        phip: 18.4373
        bullet.CA : 0.241 0.241 0.241
        bullet.CY : 0.022 0.022 0.022
        bullet.CN : 0.067 0.067 0.067
        bullet.CLL: 0.000 0.000 0.000
        bullet.CLM: 0.218 0.218 0.218
        bullet.CLN: 0.073 0.073 0.073
        aoap: size=578, dtype=float64
        phip: size=578, dtype=float64
        

:STDERR:
    * **PASS**

