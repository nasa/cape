
.. _test-results:

Results of doctest builder
==========================

This test was run on 2018-03-26 at time 15:35:56.

**Summary**:
    *    16 tests
    *     6 failures in tests
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
Failure in file "test/cape/subset.rst", line 177, in default

**Failed example**:

    .. code-block:: python

        # Bullet test folder
        fcape = os.path.join(cape.test.ftcape, "cape")
        # Go to this folder.
        os.chdir(fcape)
    
        # Read the interface
        cntl = cape.Cntl()
        # Run the API --filter command
        I = cntl.x.FilterString("b2")
        # Print it
        print(I)

**Expected**:

    .. code-block:: none

        array([ 2,  3,  6,  7, 10, 11, 14, 15, 18, 19])

**Got**:

    .. code-block:: none

        [ 2  3  6  7 10 11 14 15 18 19]

Failure in file "test/cape/subset.rst", line 227, in default

**Failed example**:

    .. code-block:: python

        # Bullet test folder
        fcape = os.path.join(cape.test.ftcape, "cape")
        # Go to this folder.
        os.chdir(fcape)
    
        # Read the interface
        cntl = cape.Cntl()
        # Run the API --glob command
        I = cntl.x.FilterWildcard("poweroff/m0*")
        # Print it
        print(I)

**Expected**:

    .. code-block:: none

        array([0, 1, 2, 3, 4, 5, 6, 7])

**Got**:

    .. code-block:: none

        [0 1 2 3 4 5 6 7]

Failure in file "test/cape/subset.rst", line 283, in default

**Failed example**:

    .. code-block:: python

        # Bullet test folder
        fcape = os.path.join(cape.test.ftcape, "cape")
        # Go to this folder.
        os.chdir(fcape)
    
        # Read the interface
        cntl = cape.Cntl()
        # Run the API --re command
        I = cntl.x.FilterRegex("m.\.5.*b2")
        # Print it
        print(I)

**Expected**:

    .. code-block:: none

        array([ 2,  3, 14, 15, 18, 19])

**Got**:

    .. code-block:: none

        [ 2  3 14 15 18 19]

Failure in file "test/cape/subset.rst", line 343, in default

**Failed example**:

    .. code-block:: python

        # Bullet test folder
        fcape = os.path.join(cape.test.ftcape, "cape")
        # Go to this folder.
        os.chdir(fcape)
    
        # Read the interface
        cntl = cape.Cntl()
        # Run the API --cons command
        I = cntl.x.Filter(["beta==2", "Mach%1==0.5"])
        # Print it
        print(I)

**Expected**:

    .. code-block:: none

        array([ 2,  3, 14, 15, 18, 19])

**Got**:

    .. code-block:: none

        [ 2  3 14 15 18 19]

Failure in file "test/cape/subset.rst", line 459, in default

**Failed example**:

    .. code-block:: python

        # Bullet test folder
        fcape = os.path.join(cape.test.ftcape, "cape")
        # Go to this folder.
        os.chdir(fcape)
    
        # Read the interface
        cntl = cape.Cntl()
        # Run a compound API command
        I = cntl.x.GetIndices(I=np.arange(15,20), cons=["Mach%1==0.5"], re="b2")
        # Print it
        print(I)

**Expected**:

    .. code-block:: none

        array([15, 18, 19])

**Got**:

    .. code-block:: none

        [15 18 19]

**Document summary**:

  * 1 items had failures:
      - 5 of   7 in default

  * 7 tests in 1 items.
  * 2 passed and 5 failed.
  * *Test Failed* 5 failures.


Document: :doc:`test/pycart/index </test/pycart/index>`
-------------------------------------------------------
Failure in file "test/pycart/index.rst", line 107, in default

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
          File "/usr/lib64/python2.7/doctest.py", line 1289, in __run
            compileflags, 1) in test.globs
          File "<doctest default[0]>", line 13, in <module>
            cape.test.shell('pycart -c')
          File "/u/wk/ddalle/usr/pycart/cape/test.py", line 98, in shell
            raise ValueError("See status in 'test.out' file")
        ValueError: See status in 'test.out' file

**Document summary**:

  * 1 items had failures:
      - 1 of   2 in default

  * 2 tests in 1 items.
  * 1 passed and 1 failed.
  * *Test Failed* 1 failures.


