
.. This documentation written by TestDriver()
   on 2022-05-11 at 01:41 PDT

Test ``02_pmpl_opts``: **FAIL** (command 1)
=============================================

This test **FAILED** (command 1) on 2022-05-11 at 01:41 PDT

This test is run in the folder:

    ``test/tnakit/02_pmpl_opts/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_PlotOptions.py
        $ python3 test01_PlotOptions.py

Command 1: :class:`MPLOpts` *PlotOptions*: Python 2 (**FAIL**)
---------------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_PlotOptions.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.14 seconds
:STDOUT:
    * **FAIL**
    * Actual: (empty)
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
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_PlotOptions.py", line 5, in <module>
            from cape.tnakit.plot_mpl import MPLOpts, rstutils
          File "/u/wk/ddalle/usr/cape/cape/__init__.py", line 87
        SyntaxError: Non-ASCII character '\xc2' in file /u/wk/ddalle/usr/cape/cape/__init__.py on line 88, but no encoding declared; see http://www.python.org/peps/pep-0263.html for details
        


