
----------------------------------------------------------------
Test results: :mod:`test.902_pyfun.001_bullet.test_001_pyfuncli`
----------------------------------------------------------------

This page documents from the folder

    ``test/902_pyfun/001_bullet``

using the file

    ``test_001_pyfuncli.py``

.. literalinclude:: _test_001_pyfuncli.py
    :caption: test_001_pyfuncli.py
    :language: python

Test case: :func:`test_01_run`
------------------------------
This test case runs the function:

.. literalinclude:: _test_001_pyfuncli.py
    :caption: test_01_run
    :language: python
    :pyobject: test_01_run

PASS

Test case: :func:`test_02_c`
----------------------------
This test case runs the function:

.. literalinclude:: _test_001_pyfuncli.py
    :caption: test_02_c
    :language: python
    :pyobject: test_02_c

PASS

Test case: :func:`test_03_fm`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_001_pyfuncli.py
    :caption: test_03_fm
    :language: python
    :pyobject: test_03_fm

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, fresh=False)
        def test_03_fm():
            # Instantiate
            cntl = cape.pyfun.cntl.Cntl()
            # Collect aero
    >       cntl.cli(fm=True, I="8")
    
    test/902_pyfun/001_bullet/test_001_pyfuncli.py:52: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/pyfun/cntl.py:210: in cli
        cmd = self.cli_cape(*a, **kw)
    cape/cntl.py:955: in cli_cape
        self.UpdateFM(**kw)
    cape/cntl.py:107: in wrapper_func
        v = func(self, *args, **kwargs)
    cape/cntl.py:4172: in UpdateFM
        self.DataBook.UpdateDataBook(I, comp=comp)
    cape/cfdx/dataBook.py:739: in UpdateDataBook
        n += self.UpdateCaseComp(i, comp)
    cape/cfdx/dataBook.py:1017: in UpdateCaseComp
        s = FM.GetStats(nStats, nMax)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    self = <dataBook.CaseFM('base', i=0)>, nStats = 50, nMax = None, kw = {}
    
        def GetStats(self, nStats=100, nMax=None, **kw):
            """Get mean, min, max, and standard deviation for all coefficients
        
            :Call:
                >>> s = FM.GetStats(nStats, nMax=None, nLast=None)
            :Inputs:
                *FM*: :class:`cape.cfdx.dataBook.CaseFM`
                    Instance of the force and moment class
                *coeff*: :class:`str`
                    Name of coefficient to process
                *nStats*: {``100``} | :class:`int`
                    Minimum number of iterations in window to use for statistics
                *dnStats*: {*nStats*} | :class:`int`
                    Interval size for candidate windows
                *nMax*: (*nStats*} | :class:`int`
                    Maximum number of iterations to use for statistics
                *nMin*: {``0``} | :class:`int`
                    First usable iteration number
                *nLast*: {*FM.i[-1]*} | :class:`int`
                    Last iteration to use for statistics
            :Outputs:
                *s*: :class:`dict`\ [:class:`float`]
                    Dictionary of mean, min, max, std, err for each coefficient
            :Versions:
                * 2017-09-29 ``@ddalle``: Version 1.0
            """
            # Check for empty instance
            if self.i.size == 0:
    >           raise ValueError("No history found for comp '%s'\n" % self.comp)
    E           ValueError: No history found for comp 'base'
    
    cape/cfdx/dataBook.py:10421: ValueError

