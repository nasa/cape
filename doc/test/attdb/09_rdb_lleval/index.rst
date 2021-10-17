
.. This documentation written by TestDriver()
   on 2021-10-17 at 01:45 PDT

Test ``09_rdb_lleval``: **FAIL** (command 1)
==============================================

This test **FAILED** (command 1) on 2021-10-17 at 01:45 PDT

This test is run in the folder:

    ``test/attdb/09_rdb_lleval/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_eval.py
        $ python3 test01_eval.py

Command 1: Interpolate line loads: Python 2 (**FAIL**)
-------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_eval.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.93 seconds
:STDOUT:
    * **FAIL**
    * Actual:

      .. code-block:: none

        mach: 0.90
        alpha: 1.50
        beta: 0.50
        

    * Target:

      .. code-block:: none

        mach: 0.90
        alpha: 1.50
        beta: 0.50
        bullet.dCN.size: 51
        bullet.dCN.xargs: ['bullet.x']
        

:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        Traceback (most recent call last):
          File "test01_eval.py", line 37, in <module>
            plt.plot(db["bullet.x"], dCN)
          File "/u/wk/ddalle/.local/lib/python2.7/site-packages/matplotlib/pyplot.py", line 3352, in plot
            ax = gca()
          File "/u/wk/ddalle/.local/lib/python2.7/site-packages/matplotlib/pyplot.py", line 969, in gca
            return gcf().gca(**kwargs)
          File "/u/wk/ddalle/.local/lib/python2.7/site-packages/matplotlib/pyplot.py", line 586, in gcf
            return figure()
          File "/u/wk/ddalle/.local/lib/python2.7/site-packages/matplotlib/pyplot.py", line 533, in figure
            **kwargs)
          File "/u/wk/ddalle/.local/lib/python2.7/site-packages/matplotlib/backend_bases.py", line 161, in new_figure_manager
            return cls.new_figure_manager_given_figure(num, fig)
          File "/u/wk/ddalle/.local/lib/python2.7/site-packages/matplotlib/backends/_backend_tk.py", line 1046, in new_figure_manager_given_figure
            window = Tk.Tk(className="matplotlib")
          File "/usr/lib64/python2.7/lib-tk/Tkinter.py", line 1745, in __init__
            self.tk = _tkinter.create(screenName, baseName, className, interactive, wantobjects, useTk, sync, use)
        _tkinter.TclError: no display name and no $DISPLAY environment variable
        


