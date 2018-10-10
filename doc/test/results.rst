
.. _test-results:

Results of doctest builder
==========================

This test was run on 2018-10-08 at time 14:54:20.

**Summary**:
    *    29 tests
    *     1 failure in tests
    *     0 failures in setup code
    *     0 failures in cleanup code


Document: :doc:`test/cape/cli </test/cape/cli>`
-----------------------------------------------
**Document summary**:

  * 1 items passed all tests:
      - 7 tests in default

  * 7 tests in 1 items.
  * 7 passed and 0 failed.
  * Test passed.


Document: :doc:`test/cape/cmd </test/cape/cmd>`
-----------------------------------------------
**Document summary**:

  * 1 items passed all tests:
      - 2 tests in default

  * 2 tests in 1 items.
  * 2 passed and 0 failed.
  * Test passed.


Document: :doc:`test/cape/atm </test/cape/atm>`
-----------------------------------------------
**Document summary**:

  * 1 items passed all tests:
      - 2 tests in default

  * 2 tests in 1 items.
  * 2 passed and 0 failed.
  * Test passed.


Document: :doc:`test/cape/subset </test/cape/subset>`
-----------------------------------------------------
**Document summary**:

  * 1 items passed all tests:
      - 7 tests in default

  * 7 tests in 1 items.
  * 7 passed and 0 failed.
  * Test passed.


Document: :doc:`test/pycart/index </test/pycart/index>`
-------------------------------------------------------
**Document summary**:

  * 1 items passed all tests:
      - 2 tests in default

  * 2 tests in 1 items.
  * 2 passed and 0 failed.
  * Test passed.


Document: :doc:`test/cape/case </test/cape/case>`
-------------------------------------------------
Failure in file "test/cape/case.rst", line 197, in default

**Failed example**:

    .. code-block:: python

        # Case test folder
        fcape = os.path.join(cape.test.ftcape, "case")
        # Go to this folder.
        os.chdir(fcape)
    
        # Example file names
        fstrt = "cape_start.dat"
        ftime = "cape_time.dat"
    
        # Delete test files if present
        for fn in [fstrt, ftime]:
            if os.path.isfile(fn):
                os.remove(fn)
    
        # Create initial time
        tic = cape.case.datetime.now()
    
        # Read settings
        rc = cape.case.ReadCaseJSON()
    
        # Write a flag for starting a program
        cape.case.WriteStartTimeProg(tic, rc, 0, fstrt, "prog")
    
        # Read it
        nProc, t0 = cape.case.ReadStartTimeProg(fstrt)
    
        # Calculate delta time
        dt = tic - t0
    
        # Test output
        print(nProc - rc.get_nProc())
        print(dt.seconds > 1)
    
        # Write output file
        cape.case.WriteUserTimeProc(tic, rc, 0, ftime, "cape")

**Exception raised**:

    .. code-block:: pytb

        Traceback (most recent call last):
          File "/usr/lib/python2.7/doctest.py", line 1315, in __run
            compileflags, 1) in test.globs
          File "<doctest default[0]>", line 35, in <module>
            cape.case.WriteUserTimeProc(tic, rc, 0, ftime, "cape")
        AttributeError: 'module' object has no attribute 'WriteUserTimeProc'

**Document summary**:

  * 1 items had failures:
      - 1 of   3 in default

  * 3 tests in 1 items.
  * 2 passed and 1 failed.
  * *Test Failed* 1 failures.


Document: :doc:`test/cape/argread </test/cape/argread>`
-------------------------------------------------------
**Document summary**:

  * 1 items passed all tests:
      - 6 tests in default

  * 6 tests in 1 items.
  * 6 passed and 0 failed.
  * Test passed.


