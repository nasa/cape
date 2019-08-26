
.. This documentation written by TestDriver()
   on 2019-08-26 at 15:25 PDT

Test ``08_cntl``
==================

This test is run in the folder:

    ``/u/wk/jmeeroff/usr/pycart/test/cape/08_cntl/``

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

Command 1: Python 2
--------------------

:Command:
    .. code-block:: console

        $ python2 test01_cntl.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.332198 seconds
    * Cumulative time: 0.332198 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Importing module 'dac'
          InitFunction: dac.InitCntl()
        <cape.Cntl(nCase=20)>
        

:STDERR:
    * **PASS**

Command 2: Python 3
--------------------

:Command:
    .. code-block:: console

        $ python3 test01_cntl.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.40579 seconds
    * Cumulative time: 0.737988 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Importing module 'dac'
          InitFunction: dac.InitCntl()
        <cape.Cntl(nCase=20)>
        

:STDERR:
    * **PASS**

