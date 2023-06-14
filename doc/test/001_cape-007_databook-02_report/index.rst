
--------------------------------------------------------------
Test results: :mod:`test.001_cape.007_databook.test_02_report`
--------------------------------------------------------------

This page documents from the folder

    ``test/001_cape/007_databook``

using the file

    ``test_02_report.py``

.. literalinclude:: _test_02_report.py
    :caption: test_02_report.py
    :language: python

Test case: :func:`test_01_sweep`
--------------------------------
This test case runs the function:

.. literalinclude:: _test_02_report.py
    :caption: test_01_sweep
    :language: python
    :pyobject: test_01_sweep

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, TEST_FILES)
        def test_01_sweep():
            # Read settings
            cntl = cape.cntl.Cntl()
            # Read data book
            db = databook.DataBook(cntl)
            # Read reports
            rp = report.Report(cntl, 'mach')
            # Add databook to cntl
            cntl.DataBook = db
            # Define subfig
            sfig = "mach_arrow_CA"
            # Define sweep
            swp = "mach"
            # Figure name
            fpdf = '%s.png' % sfig
            # File names
            fabs = os.path.abspath(fpdf)
            fdir = os.path.dirname(os.getcwd())
            ftarg = os.path.join(fdir, fpdf)
            # Generate the Figure
    >       rp.SubfigSweepCoeff(sfig, swp, [0], True)
    
    test/001_cape/007_databook/test_02_report.py:41: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/cfdx/report.py:3633: in SubfigSweepCoeff
        h = DBc.PlotCoeff(
    cape/cfdx/dataBook.py:5134: in PlotCoeff
        return self.PlotCoeffBase(coeff, I, **kw)
    cape/cfdx/dataBook.py:4998: in PlotCoeffBase
        h['line'] = plt.plot(xv, yv, **kw_p)
    /home5/ddalle/.local/lib/python3.9/site-packages/matplotlib/pyplot.py:2757: in plot
        return gca().plot(
    /home5/ddalle/.local/lib/python3.9/site-packages/matplotlib/axes/_axes.py:1632: in plot
        lines = [*self._get_lines(*args, data=data, **kwargs)]
    /home5/ddalle/.local/lib/python3.9/site-packages/matplotlib/axes/_base.py:312: in __call__
        yield from self._plot_args(this, kwargs)
    /home5/ddalle/.local/lib/python3.9/site-packages/matplotlib/axes/_base.py:538: in _plot_args
        return [l[0] for l in result]
    /home5/ddalle/.local/lib/python3.9/site-packages/matplotlib/axes/_base.py:538: in <listcomp>
        return [l[0] for l in result]
    /home5/ddalle/.local/lib/python3.9/site-packages/matplotlib/axes/_base.py:531: in <genexpr>
        result = (make_artist(x[:, j % ncx], y[:, j % ncy], kw,
    /home5/ddalle/.local/lib/python3.9/site-packages/matplotlib/axes/_base.py:351: in _makeline
        seg = mlines.Line2D(x, y, **kw)
    /home5/ddalle/.local/lib/python3.9/site-packages/matplotlib/lines.py:370: in __init__
        self.set_color(color)
    /home5/ddalle/.local/lib/python3.9/site-packages/matplotlib/lines.py:1028: in set_color
        mcolors._check_color_like(color=color)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    kwargs = {'color': ['b', 'g', 'm', 'darkorange', 'purple']}, k = 'color'
    v = ['b', 'g', 'm', 'darkorange', 'purple']
    
        def _check_color_like(**kwargs):
            """
            For each *key, value* pair in *kwargs*, check that *value* is color-like.
            """
            for k, v in kwargs.items():
                if not is_color_like(v):
    >               raise ValueError(f"{v!r} is not a valid value for {k}")
    E               ValueError: ['b', 'g', 'm', 'darkorange', 'purple'] is not a valid value for color
    
    /home5/ddalle/.local/lib/python3.9/site-packages/matplotlib/colors.py:130: ValueError

