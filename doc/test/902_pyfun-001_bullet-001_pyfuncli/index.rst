
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

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, TEST_FILES)
        def test_01_run():
            # Instantiate
            cntl = cape.pyfun.cntl.Cntl()
            # Run first case
    >       cntl.SubmitJobs(I="8")
    
    test/902_pyfun/001_bullet/test_001_pyfuncli.py:29: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/cntl.py:1308: in SubmitJobs
        self.PrepareCase(i)
    cape/pyfun/cntl.py:1366: in PrepareCase
        self.PrepareNamelist(i)
    cape/pyfun/cntl.py:1477: in PrepareNamelist
        nopts = self.opts.select_namelist(j)
    cape/optdict/__init__.py:3667: in wrapper
        v = f(*a, **kw)
    cape/pyfun/options/fun3dnmlopts.py:178: in select_namelist
        return self.sample_dict(self, j=j, **kw)
    cape/optdict/__init__.py:1600: in sample_dict
        assert_isinstance(i, INT_TYPES, "case index")
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    obj = None
    cls_or_tuple = (<class 'int'>, <class 'numpy.int8'>, <class 'numpy.int16'>, <class 'numpy.int32'>, <class 'numpy.int64'>, <class 'numpy.uint8'>, ...)
    desc = 'case index'
    
        def assert_isinstance(obj, cls_or_tuple, desc=None):
            r"""Conveniently check types
        
            Applies ``isinstance(obj, cls_or_tuple)`` but also constructs
            a :class;`TypeError` and appropriate message if test fails
        
            :Call:
                >>> assert_isinstance(obj, cls, desc=None)
                >>> assert_isinstance(obj, cls_tuple, desc=None)
            :Inputs:
                *obj*: :class:`object`
                    Object whose type is checked
                *cls*: :class:`type`
                    Single permitted class
                *cls_tuple*: :class:`tuple`\ [:class:`type`]
                    Tuple of allowed classes
            :Raises:
                :class:`OptdictTypeError`
            :Versions:
                * 2022-09-17 ``@ddalle``: Version 1.0
            """
            # Special case for ``None``
            if cls_or_tuple is None:
                return
            # Check for passed test
            if isinstance(obj, cls_or_tuple):
                return
            # Generate type error message
            msg = _genr8_type_error(obj, cls_or_tuple, desc)
            # Raise
    >       raise OptdictTypeError(msg)
    E       cape.optdict.opterror.OptdictTypeError: For case index: got type 'NoneType'; expected 'int' | 'int8' | 'int16' | 'int32' | 'int64' | 'uint8' | 'uint16' | 'uint32' | 'uint64'
    
    cape/optdict/opterror.py:106: OptdictTypeError

Test case: :func:`test_02_c`
----------------------------
This test case runs the function:

.. literalinclude:: _test_001_pyfuncli.py
    :caption: test_02_c
    :language: python
    :pyobject: test_02_c

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, fresh=False)
        def test_02_c():
            # Split command and add `-m` prefix
            cmdlist = [sys.executable, "-m", "cape.pyfun", "-c", "-I", "8"]
            # Run the command
            stdout, _, _ = testutils.call_o(cmdlist)
            # Check outout
            result = testutils.compare_files(stdout, "test.02.out", ELLIPSIS=True)
    >       assert result.line1 == result.line2
    E       AssertionError: assert '8    bullet/...           \n' == '8    bullet/...       0...\n'
    E         - 8    bullet/m1.10a0.0b0.0  DONE    200/200     .        0...
    E         + 8    bullet/m1.10a0.0b0.0  ---     /           .
    
    test/902_pyfun/001_bullet/test_001_pyfuncli.py:41: AssertionError

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
            cntl.cli(fm=True, I="8")
            # Read databook
            cntl.ReadDataBook()
            # Get value
    >       CA = cntl.DataBook["bullet_no_base"]["CA"][0]
    E       IndexError: index 0 is out of bounds for axis 0 with size 0
    
    test/902_pyfun/001_bullet/test_001_pyfuncli.py:54: IndexError

