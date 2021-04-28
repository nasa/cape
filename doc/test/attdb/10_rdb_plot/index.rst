
.. This documentation written by TestDriver()
   on 2021-04-28 at 13:25 PDT

Test ``10_rdb_plot``
======================

This test is run in the folder:

    ``/u/wk/ddalle/usr/cape/test/attdb/10_rdb_plot/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_pre.py
        $ python3 test01_pre.py
        $ python2 test02_ll.py
        $ python3 test02_ll.py

Command 1: Prep plot args: Python 2
------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_pre.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.560805 seconds
    * Cumulative time: 0.560805 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Version 1: scalar by scalar args
          col  = 'bullet.CN'
          I    = []
          mach =  0.90
          aoa  =  1.90
          beta = -0.10
        Version 2: scalar by array indices
          col  = 'bullet.CN'
          I    = 200, 238
          mach =  0.80,  0.80
          aoa  =  0.75,  1.50
          beta =  1.25, -2.00
        Version 3:
          col = 'bullet.dCN'
          I    = 240, 251
          mach =  0.80,  0.80
          aoa  =  1.50,  1.50
          beta = -1.50,  1.25
        

:STDERR:
    * **PASS**

Command 2: Prep plot args: Python 3
------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_pre.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.622255 seconds
    * Cumulative time: 1.18306 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Version 1: scalar by scalar args
          col  = 'bullet.CN'
          I    = []
          mach =  0.90
          aoa  =  1.90
          beta = -0.10
        Version 2: scalar by array indices
          col  = 'bullet.CN'
          I    = 200, 238
          mach =  0.80,  0.80
          aoa  =  0.75,  1.50
          beta =  1.25, -2.00
        Version 3:
          col = 'bullet.dCN'
          I    = 240, 251
          mach =  0.80,  0.80
          aoa  =  1.50,  1.50
          beta = -1.50,  1.25
        

:STDERR:
    * **PASS**

Command 3: Plot line load: Python 2
------------------------------------

:Command:
    .. code-block:: console

        $ python2 test02_ll.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.788593 seconds
    * Cumulative time: 1.97165 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        Index: 535
        

:STDERR:
    * **PASS**

:PNG:
    * **FAIL**
    * Difference fraction: 0.0105
    * Actual:

        .. image:: PNG-02-00.png
            :width: 4.5in

    * Target:

        .. image:: PNG-target-02-00.png
            :width: 4.5in

