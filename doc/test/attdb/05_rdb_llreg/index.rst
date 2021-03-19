
.. This documentation written by TestDriver()
   on 2021-03-19 at 09:48 PDT

Test ``05_rdb_llreg``
=======================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/attdb/05_rdb_llreg/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_griddata.py
        $ python3 test01_griddata.py

Command 1: Regularize line load using griddata: Python 2
---------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_griddata.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.48794 seconds
    * Cumulative time: 0.48794 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        reg.bullet.dCN.shape = [51, 578]
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_griddata.py", line 29, in <module>
            prefix="reg.")
          File "/u/wk/ddalle/usr/pycart/cape/attdb/rdb.py", line 10650, in regularize_by_griddata
            iargs, *x, I=masks[i], **kw)
          File "/u/wk/ddalle/usr/pycart/cape/attdb/rdb.py", line 6878, in genr8_griddata_weights
            W1 = sciint.griddata(x, kmode, y, method, rescale=rescale)
        TypeError: griddata() got an unexpected keyword argument 'rescale'
        


