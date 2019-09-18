
.. This documentation written by TestDriver()
   on 2019-09-18 at 12:35 PDT

Test ``01_bullet``
====================

This test is run in the folder:

    ``/u/wk/ddalle/usr/pycart/test/pycart/01_bullet/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ pycart -c
        $ pycart -I 0
        $ pycart -I 0 --aero
        $ python2 test_databook.py
        $ python3 test_databook.py

Command 1: Run Matrix Status
-----------------------------

:Command:
    .. code-block:: console

        $ pycart -c

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.823873 seconds
    * Cumulative time: 0.823873 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Case Config/Run Directory  Status  Iterations  Que CPU Time 
        ---- --------------------- ------- ----------- --- --------
        0    poweroff/m1.5a0.0b0.0 ---     /           .            
        1    poweroff/m2.0a0.0b0.0 ---     /           .            
        2    poweroff/m2.0a2.0b0.0 ---     /           .            
        3    poweroff/m2.0a2.0b2.0 ---     /           .            
        
        ---=4, 
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "/u/wk/ddalle/usr/pycart/bin/pycart", line 31, in <module>
            cntl = cape.pycart.Cntl(fname)
          File "/u/wk/ddalle/usr/pycart/cape/pycart/cntl.py", line 169, in __init__
            self.opts = options.Options(fname=fname)
          File "/u/wk/ddalle/usr/pycart/cape/pycart/options/__init__.py", line 96, in __init__
            defs = getPyCartDefaults()
          File "/u/wk/ddalle/usr/pycart/cape/pycart/options/util.py", line 172, in getPyCartDefaults
            return loadJSONFile(fname)
          File "/u/wk/ddalle/usr/pycart/cape/options/util.py", line 441, in loadJSONFile
            txt, fnames, linenos = expandJSONFile(fname)
          File "/u/wk/ddalle/usr/pycart/cape/options/util.py", line 341, in expandJSONFile
            txt = io.open(fname, mode="r", encoding="utf-8").read()
        IOError: [Errno 2] No such file or directory: '/u/wk/ddalle/usr/pycart/cape/pycart/../settings/pyCart.default.json'
        


