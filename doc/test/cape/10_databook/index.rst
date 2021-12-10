
.. This documentation written by TestDriver()
   on 2021-12-10 at 01:40 PST

Test ``10_databook``: PASS
============================

This test PASSED on 2021-12-10 at 01:40 PST

This test is run in the folder:

    ``test/cape/10_databook/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_db.py
        $ python3 test01_db.py

**Included file:** ``test01_db.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import numpy
        import numpy as np
        
        # Import cape modules
        import cape
        import cape.cfdx.dataBook
        
        # Test DataBook CLass
        # Read settings
        cntl = cape.Cntl()
        
        # Read data book
        DB = cape.cfdx.dataBook.DataBook(cntl.x, cntl.opts)
        
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
        DBc.UpdateRunMatrix()
        # Filter cases at alpha=2
        I = DBc.x.Filter(["alpha==2"])
        # Show those cases
        for frun in DBc.x.GetFullFolderNames(I):
            print(frun)
        
        # Test CaseFM Class
        # Create a force & moment history
        FM = cape.cfdx.dataBook.CaseFM("fin")
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

Command 1: Python 2 (PASS)
---------------------------

:Command:
    .. code-block:: console

        $ python2 test01_db.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.48 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

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
        <dataBook.CaseFM('fin', i=500)>
        Mean value: 1.4149
        Min value: 0.6555
        Max value: 2.0462
        Standard deviation: 0.3095
        Sampling error: 0.0190
        

:STDERR:
    * **PASS**

Command 2: Python 3 (PASS)
---------------------------

:Command:
    .. code-block:: console

        $ python3 test01_db.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.59 seconds
    * Cumulative time: 1.07 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

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
        <dataBook.CaseFM('fin', i=500)>
        Mean value: 1.4149
        Min value: 0.6555
        Max value: 2.0462
        Standard deviation: 0.3095
        Sampling error: 0.0190
        

:STDERR:
    * **PASS**

