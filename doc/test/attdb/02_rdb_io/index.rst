
.. This documentation written by TestDriver()
   on 2021-03-19 at 09:48 PDT

Test ``02_rdb_io``
====================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/attdb/02_rdb_io/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_csv_to_mat.py
        $ python3 test01_csv_to_mat.py

Command 1: CSV read; MAT write/read: Python 2
----------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_csv_to_mat.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.627647 seconds
    * Cumulative time: 0.627647 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case 13: m0.95a5.00 CA=0.526
        Case 13: m0.95a5.00 CA=0.526
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_csv_to_mat.py", line 22, in <module>
            db.write_mat("aero_arrow_no_base.mat")
          File "/u/wk/ddalle/usr/pycart/cape/attdb/rdb.py", line 1538, in write_mat
            dbmat.write_mat(fname, cols=cols, attrs=attrs)
          File "/u/wk/ddalle/usr/pycart/cape/attdb/ftypes/matfile.py", line 428, in write_mat
            sio.savemat(fname, dbmat, oned_as="column")
          File "/usr/lib64/python2.7/site-packages/scipy/io/matlab/mio.py", line 270, in savemat
            MW.put_variables(mdict)
          File "/usr/lib64/python2.7/site-packages/scipy/io/matlab/mio5.py", line 866, in put_variables
            self._matrix_writer.write_top(var, asbytes(name), is_global)
          File "/usr/lib64/python2.7/site-packages/scipy/io/matlab/mio5.py", line 617, in write_top
            self.write(arr)
          File "/usr/lib64/python2.7/site-packages/scipy/io/matlab/mio5.py", line 638, in write
            % (arr, type(arr)))
        TypeError: Could not convert <scipy.io.matlab.mio5_params.mat_struct object at 0x7f504a365390> (type <class 'scipy.io.matlab.mio5_params.mat_struct'>) to array
        


