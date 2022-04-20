
.. This documentation written by TestDriver()
   on 2022-04-20 at 01:46 PDT

Test ``02_pmpl_opts``: PASS
=============================

This test PASSED on 2022-04-20 at 01:46 PDT

This test is run in the folder:

    ``test/tnakit/02_pmpl_opts/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_PlotOptions.py
        $ python3 test01_PlotOptions.py

Command 1: :class:`MPLOpts` *PlotOptions*: Python 2 (PASS)
-----------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_PlotOptions.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.49 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        PlotOptions:
            :Index: 2
            :Rotate: ``False``
            :color:
                +------------------+
                | ``"b"``          |
                +------------------+
                | ``"k"``          |
                +------------------+
                | ``"darkorange"`` |
                +------------------+
                | ``"g"``          |
                +------------------+
            :ls: ``"--"``
            :zorder: 8
        

:STDERR:
    * **PASS**

Command 2: :class:`MPLOpts` *PlotOptions*: Python 3 (PASS)
-----------------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_PlotOptions.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.50 seconds
    * Cumulative time: 0.99 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        PlotOptions:
            :Index: 2
            :Rotate: ``False``
            :color:
                +------------------+
                | ``"b"``          |
                +------------------+
                | ``"k"``          |
                +------------------+
                | ``"darkorange"`` |
                +------------------+
                | ``"g"``          |
                +------------------+
            :ls: ``"--"``
            :zorder: 8
        

:STDERR:
    * **PASS**

