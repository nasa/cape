
.. This documentation written by TestDriver()
   on 2022-01-28 at 01:40 PST

Test ``04_subset_api``: PASS
==============================

This test PASSED on 2022-01-28 at 01:40 PST

This test is run in the folder:

    ``test/cape/04_subset_api/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_subset.py
        $ python3 test01_subset.py

**Included file:** ``test01_subset.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import cape module
        import cape
        
        # Load interface
        cntl = cape.Cntl()
        
        # Display subsets
        print(list(cntl.x.FilterString("b2")))
        print(list(cntl.x.FilterWildcard("poweroff/m0*")))
        print(list(cntl.x.FilterRegex("m.\.5.*b2")))
        print(list(cntl.x.GetIndices(I=range(15,20), cons=["Mach%1==0.5"])))
        

Command 1: Python 2 (PASS)
---------------------------

:Command:
    .. code-block:: console

        $ python2 test01_subset.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.60 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]
        [0, 1, 2, 3, 4, 5, 6, 7]
        [2, 3, 14, 15, 18, 19]
        [15, 16, 17, 18, 19]
        

:STDERR:
    * **PASS**

Command 2: Python 3 (PASS)
---------------------------

:Command:
    .. code-block:: console

        $ python3 test01_subset.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.72 seconds
    * Cumulative time: 1.32 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]
        [0, 1, 2, 3, 4, 5, 6, 7]
        [2, 3, 14, 15, 18, 19]
        [15, 16, 17, 18, 19]
        

:STDERR:
    * **PASS**

