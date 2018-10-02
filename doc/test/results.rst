
.. _test-results:

Results of doctest builder
==========================

This test was run on 2018-10-01 at time 20:19:07.

**Summary**:
    *    22 tests
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
Failure in file "test/pycart/index.rst", line 105, in default

**Failed example**:

    .. code-block:: python

        # Bullet test folder
        fbullet = os.path.join(cape.test.ftpycart, "bullet")
        # Go to this folder.
        os.chdir(fbullet)
    
        # Clean up if necessary.
        if os.path.isdir('poweroff'): shutil.rmtree('poweroff')
        if os.path.isdir('data'):     shutil.rmtree('data')
        # Remove log file if necessary
        if os.path.isfile('test.out'): os.remove('test.out')
    
        # Show status
        cape.test.shell('pycart -c')
    
        # Run one case
        cape.test.shell('pycart -I 0')
    
        # Assemble the data book
        cape.test.shell('pycart -I 0 --aero')
    
        # Read the interface
        cart3d = pyCart.Cart3d()
        # Show it
        print(cart3d)
        # Read the databook
        cart3d.ReadDataBook()
        # Get the value.
        CA = cart3d.DataBook['bullet_no_base']['CA'][0]
        # Test it
        print(abs(CA - 0.745) <= 0.02)

**Exception raised**:

    .. code-block:: pytb

        Traceback (most recent call last):
          File "/usr/lib/python2.7/doctest.py", line 1315, in __run
            compileflags, 1) in test.globs
          File "<doctest default[0]>", line 16, in <module>
            cape.test.shell('pycart -I 0')
          File "/home/dalle/usr/pycart/cape/test.py", line 107, in shell
            raise ValueError("See status in 'test.out' file")
        ValueError: See status in 'test.out' file

**Document summary**:

  * 1 items had failures:
      - 1 of   2 in default

  * 2 tests in 1 items.
  * 1 passed and 1 failed.
  * *Test Failed* 1 failures.


Document: :doc:`test/cape/argread </test/cape/argread>`
-------------------------------------------------------
**Document summary**:

  * 1 items passed all tests:
      - 6 tests in default

  * 6 tests in 1 items.
  * 6 passed and 0 failed.
  * Test passed.


