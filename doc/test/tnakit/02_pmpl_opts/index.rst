
.. This documentation written by TestDriver()
   on 2020-03-30 at 10:48 PDT

Test ``02_pmpl_opts``
=======================

This test is run in the folder:

    ``/home/dalle/usr/pycart/test/tnakit/02_pmpl_opts/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_PlotOptions.py
        $ python3 test01_PlotOptions.py

Command 1: :class:`MPLOpts` *PlotOptions*: Python 2
----------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_PlotOptions.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.234808 seconds
    * Cumulative time: 0.234808 seconds
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

Command 2: :class:`MPLOpts` *PlotOptions*: Python 3
----------------------------------------------------

:Command:
    .. code-block:: console

        $ python3 test01_PlotOptions.py

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.442529 seconds
    * Cumulative time: 0.677337 seconds
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

