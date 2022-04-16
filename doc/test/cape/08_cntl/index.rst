
.. This documentation written by TestDriver()
   on 2022-04-16 at 01:40 PDT

Test ``08_cntl``: **FAIL** (command 1)
========================================

This test **FAILED** (command 1) on 2022-04-16 at 01:40 PDT

This test is run in the folder:

    ``test/cape/08_cntl/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_cntl.py
        $ python3 test01_cntl.py

**Included file:** ``test01_cntl.py``

    .. code-block:: python

        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        
        # Import cape module
        import cape
        
        # Initiate object
        cntl = cape.Cntl()
        
        # print results
        print(cntl)

Command 1: Python 2 (**FAIL**)
-------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_cntl.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.57 seconds
:STDOUT:
    * **FAIL**
    * Actual:

      .. code-block:: none

        Importing module 'dac'
          InitFunction: dac.InitCntl
        <cape.Cntl(nCase=20)>
        

    * Target:

      .. code-block:: none

        Importing module 'dac'
          InitFunction: dac.InitCntl()
        <cape.Cntl(nCase=20)>
        

:STDERR:
    * **PASS**

