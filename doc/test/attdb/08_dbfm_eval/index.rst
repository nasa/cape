
.. This documentation written by TestDriver()
   on 2021-03-19 at 09:48 PDT

Test ``08_dbfm_eval``
=======================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/attdb/08_dbfm_eval/``

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
    * Command took 0.398279 seconds
    * Cumulative time: 0.398279 seconds
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
    * Command took 0.561863 seconds
    * Cumulative time: 0.960142 seconds
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
    * Command took 0.226174 seconds
    * Cumulative time: 1.18632 seconds
:STDOUT:
    * **PASS**
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
    * Command took 0.520876 seconds
    * Cumulative time: 1.70719 seconds
:STDOUT:
    * **PASS**
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

Command 5: Evaluate *q* and *T*: Python 2
------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test03_q.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.319079 seconds
    * Cumulative time: 2.02627 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        mach: 0.90
        q: 1250.00
        T: 475.33
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test03_q.py", line 34, in <module>
            print("%s: %.2f" % (col, db(col, mach)))
          File "/u/wk/ddalle/usr/pycart/cape/attdb/rdb.py", line 1594, in __call__
            return self.rcall(col, *a, **kw)
          File "/u/wk/ddalle/usr/pycart/cape/attdb/rdb.py", line 1702, in rcall
            v = f(col, args_col, *x, **kw_fn)
          File "/u/wk/ddalle/usr/pycart/cape/attdb/rdb.py", line 3709, in rcall_multilinear
            return self._rcall_multilinear(col, args, x, **kw)
          File "/u/wk/ddalle/usr/pycart/cape/attdb/rdb.py", line 3792, in _rcall_multilinear
            ("but total size of args %s is %i." % (args, np.prod(N))))
        ValueError: ("Column 'q' has size 578, ", "but total size of args ['mach'] is 2.")
        


