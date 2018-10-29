
.. _test-cape-dataBook:

Testing :mod:`cape.dataBook`: Baseline Data Interface
=====================================================

This section tests the :mod:`cape.dataBook` module that forms the templates for
interacting with the various data books.  Each flow solver module has a separate
:mod:`dataBook` module
forms command) and :mod:`bin` (which calls them).

.. testsetup:: *

    # System modules
    import os
    import numpy as np
    
    # Modules to import
    import cape.test
    import cape
    import cape.dataBook


.. _test-cape-dataBook-DataBook:

Test DataBook Class
--------------------
This test reads a basic force & moment data book and tests some of the
properties.

.. testcode::

    # Case test folder
    fcape = os.path.join(cape.test.ftcape, "dataBook")
    # Go to this folder.
    os.chdir(fcape)
    
    # Read settings
    cntl = cape.Cntl()
    
    # Read data book
    DB = cape.dataBook.DataBook(cntl.x, cntl.opts)
    
    # Display the data book
    print(DB)
    # List the components
    for comp in DB.Components:
        print(comp)
        
    # Extract a component
    DBc = DB["fin1"]
    # Display that
    print(DBc)
    
    # Match the trajectory to the actual data
    DBc.UpdateTrajectory()
    # Filter cases at alpha=2
    I = DBc.x.Filter(["alpha==2"])
    # Show those cases
    for frun in DBc.x.GetFullFolderNames(I):
        print(frun)
        
.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    <DataBook nComp=10, nCase=30>
    cap
    body
    fins
    arrow_no_base
    arrow_total
    fuselage
    fin1
    fin2
    fin3
    fin4
    <DBComp fin1, nCase=30>
    poweroff/m0.50a02.0
    poweroff/m0.80a02.0
    poweroff/m0.95a02.0
    poweroff/m1.10a02.0
    poweroff/m1.40a02.0
    poweroff/m2.20a02.0


.. _test-cape-dataBook-CaseFM:

Test CaseFM Class
------------------
The :class:`cape.dataBook.CaseFM` class is used for processing iterative
histories of force and moment components.  This test creates an instance
manually using random numbers.  


.. testcode::

    # Case test folder
    fcape = os.path.join(cape.test.ftcape, "dataBook")
    # Go to this folder.
    os.chdir(fcape)
    
    # Create a force & moment history
    FM = cape.dataBook.CaseFM("fin")
    # Some iterations
    n = 500
    # Create some iterations
    FM.i = np.arange(n)
    
    # Seed the random number generator
    np.random.seed(450)
    # Create some random numbers
    FM.CN = 1.4 + 0.3*np.random.randn(n)
    
    # Save properties
    FM.cols = ["i", "CN"]
    FM.coeffs = ["CN"]
    
    # Display it
    print(FM)
    
    # Calculate statistics
    S = FM.GetStatsN(100)
    
    # Show values
    print("Mean value: %(CN).4f" % S)
    print("Min value: %(CN_min).4f" % S)
    print("Max value: %(CN_max).4f" % S)
    print("Standard deviation: %(CN_std).4f" % S)
    print("Sampling error: %(CN_err).4f" % S)
    
.. testoutput::
    :options: +NORMALIZE_WHITESPACE
    
    <dataBook.CaseFM('fin', i=500)>
    Mean value: 1.4149
    Min value: 0.6555
    Max value: 2.0462
    Standard deviation: 0.3095
    Sampling error: 0.0190

