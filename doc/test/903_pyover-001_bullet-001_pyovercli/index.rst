
------------------------------------------------------------------
Test results: :mod:`test.903_pyover.001_bullet.test_001_pyovercli`
------------------------------------------------------------------

This page documents from the folder

    ``test/903_pyover/001_bullet``

using the file

    ``test_001_pyovercli.py``

.. literalinclude:: _test_001_pyovercli.py
    :caption: test_001_pyovercli.py
    :language: python

Test case: :func:`test_01_run`
------------------------------
This test case runs the function:

.. literalinclude:: _test_001_pyovercli.py
    :caption: test_01_run
    :language: python
    :pyobject: test_01_run

PASS

Test case: :func:`test_02_c`
----------------------------
This test case runs the function:

.. literalinclude:: _test_001_pyovercli.py
    :caption: test_02_c
    :language: python
    :pyobject: test_02_c

PASS

Test case: :func:`test_03_fm`
-----------------------------
This test case runs the function:

.. literalinclude:: _test_001_pyovercli.py
    :caption: test_03_fm
    :language: python
    :pyobject: test_03_fm

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, fresh=False)
        def test_03_fm():
            # Instantiate
            cntl = cape.pyover.cntl.Cntl()
            # Collect aero
    >       cntl.cli(fm=True, I="1")
    
    test/903_pyover/001_bullet/test_001_pyovercli.py:50: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/pyover/cntl.py:164: in cli
        cmd = self.cli_cape(*a, **kw)
    cape/cntl.py:945: in cli_cape
        self.UpdateFM(**kw)
    cape/cntl.py:107: in wrapper_func
        v = func(self, *args, **kwargs)
    cape/cntl.py:4215: in UpdateFM
        self.DataBook.UpdateDataBook(I, comp=comp)
    cape/cfdx/dataBook.py:733: in UpdateDataBook
        n += self.UpdateCaseComp(i, comp)
    cape/cfdx/dataBook.py:939: in UpdateCaseComp
        H = self.ReadCaseResid()
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    self = <[AttributeError("'NoneType' object has no attribute 'n'") raised in repr()] DataBook object at 0x74414c0>
    
        def ReadCaseResid(self):
            """Read a :class:`CaseResid` object
        
            :Call:
                >>> H = DB.ReadCaseResid()
            :Inputs:
                *DB*: :class:`cape.cfdx.dataBook.DataBook`
                    Instance of data book class
            :Outputs:
                *H*: :class:`pyOver.dataBook.CaseResid`
                    Residual history class
            :Versions:
                * 2017-04-13 ``@ddalle``: First separate version
            """
            # Get the phase number
    >       rc = case.ReadCaseJSON()
    E       AttributeError: module 'cape.pyover.case' has no attribute 'ReadCaseJSON'
    
    cape/pyover/dataBook.py:567: AttributeError

