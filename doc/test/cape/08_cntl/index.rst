
.. This documentation written by TestDriver()
   on 2022-04-19 at 01:40 PDT

Test ``08_cntl``: PASS
========================

This test PASSED on 2022-04-19 at 01:40 PDT

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

Command 1: Python 2 (PASS)
---------------------------

:Command:
    .. code-block:: console

        $ python2 test01_cntl.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.53 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Importing module 'dac'
          InitFunction: dac.InitCntl()
        <cape.Cntl(nCase=20)>
        

:STDERR:
    * **PASS**

Command 2: Python 3 (PASS)
---------------------------

:Command:
    .. code-block:: console

        $ python3 test01_cntl.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.70 seconds
    * Cumulative time: 1.24 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Importing module 'dac'
          InitFunction: dac.InitCntl()
        <cape.Cntl(nCase=20)>
        

:STDERR:
    * **PASS**

