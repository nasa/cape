
.. This documentation written by TestDriver()
   on 2021-10-13 at 15:05 PDT

Test ``02_pmpl_opts``: **FAIL** (command 2)
=============================================

This test **FAILED** (command 2) on 2021-10-13 at 15:05 PDT

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
    * Command took 0.17 seconds
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
            :zorder: 8
            :ls: ``"--"``
        

:STDERR:
    * **PASS**

Command 2: :class:`MPLOpts` *PlotOptions*: Python 3 (**FAIL**)
---------------------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_PlotOptions.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.27 seconds
    * Cumulative time: 0.44 seconds
:STDOUT:
    * **FAIL**
    * Actual:

      .. code-block:: none

        PlotOptions:
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
            :Index: 2
            :Rotate: ``False``
        

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
            :zorder: 8
            :ls: ``"--"``
        

:STDERR:
    * **PASS**

