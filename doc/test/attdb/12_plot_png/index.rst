
.. This documentation written by TestDriver()
   on 2021-10-17 at 01:45 PDT

Test ``12_plot_png``: **FAIL** (command 1)
============================================

This test **FAILED** (command 1) on 2021-10-17 at 01:45 PDT

This test is run in the folder:

    ``test/attdb/12_plot_png/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ python2 test01_plot_png.py
        $ python3 test01_plot_png.py

Command 1: Line load plot with PNG: Python 2 (**FAIL**)
--------------------------------------------------------

:Command:
    .. code-block:: console

        $ python2 test01_plot_png.py

:Return Code:
    * **FAIL**
    * Output: ``1``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.73 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **FAIL**
    * Actual:

      .. code-block:: pytb

        /u/wk/ddalle/usr/cape/cape/tnakit/plot_mpl/mpl.py:84: UserWarning: 
        This call to matplotlib.use() has no effect because the backend has already
        been chosen; matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
        or matplotlib.backends is imported for the first time.
        
        The backend was *originally* set to 'TkAgg' by the following code:
          File "test01_plot_png.py", line 8, in <module>
            import matplotlib.pyplot as plt
          File "/u/wk/ddalle/.local/lib/python2.7/site-packages/matplotlib/pyplot.py", line 71, in <module>
            from matplotlib.backends import pylab_setup
          File "/u/wk/ddalle/.local/lib/python2.7/site-packages/matplotlib/backends/__init__.py", line 17, in <module>
            line for line in traceback.format_stack()
        
        
          mpl.use("Agg")
        Traceback (most recent call last):
          File "test01_plot_png.py", line 26, in <module>
            h = db.plot("bullet.dCN", 1, XLabel="x/Lref", YLabel="dCN/d(x/Lref)")
          File "/u/wk/ddalle/usr/cape/cape/attdb/rdb.py", line 10407, in plot
            return self.plot_linear(col, *a, **kw)
          File "/u/wk/ddalle/usr/cape/cape/attdb/rdb.py", line 10717, in plot_linear
            hi = pmpl.plot(X, V, **opts)
          File "/u/wk/ddalle/usr/cape/cape/tnakit/plot_mpl/__init__.py", line 263, in plot
            _part_init_figure(opts, h)
          File "/u/wk/ddalle/usr/cape/cape/tnakit/plot_mpl/__init__.py", line 414, in _part_init_figure
            h.fig = mpl._figure(**kw_fig)
          File "/u/wk/ddalle/usr/cape/cape/tnakit/plot_mpl/mpl.py", line 1890, in _figure
            fig = plt.gcf()
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
        


:PNG:
    * **FAIL**
    * Actual:

        .. image:: PNG-00-00.png
            :width: 4.5in

    * Target:

        .. image:: PNG-target-00-00.png
            :width: 4.5in

