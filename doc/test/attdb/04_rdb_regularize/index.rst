
.. This documentation written by TestDriver()
   on 2021-03-19 at 09:48 PDT

Test ``04_rdb_regularize``
============================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/attdb/04_rdb_regularize/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_griddata.py
        $ python3 test01_griddata.py
        $ python2 test02_rbf.py
        $ python3 test02_rbf.py

Command 1: Regularize using griddata: Python 2
-----------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_griddata.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.482347 seconds
    * Cumulative time: 0.482347 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        max error(regalpha) = 0.00
        max error(regbeta)  = 0.00
        monotonic(regCN): True
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_griddata.py", line 25, in <module>
            db.regularize_by_griddata("CN", ["alpha", "beta"], prefix="reg")
          File "/u/wk/ddalle/usr/pycart/cape/attdb/rdb.py", line 10604, in regularize_by_griddata
            W = self.genr8_griddata_weights(args, *x, **kw)
          File "/u/wk/ddalle/usr/pycart/cape/attdb/rdb.py", line 6878, in genr8_griddata_weights
            W1 = sciint.griddata(x, kmode, y, method, rescale=rescale)
        TypeError: griddata() got an unexpected keyword argument 'rescale'
        


