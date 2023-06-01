
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

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
        def test_01_run():
            # Instantiate
            cntl = cape.pyover.cntl.Cntl()
            # Run first case
    >       cntl.SubmitJobs(I="1")
    
    test/903_pyover/001_bullet/test_001_pyovercli.py:29: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/cntl.py:1308: in SubmitJobs
        self.PrepareCase(i)
    cape/pyover/cntl.py:360: in PrepareCase
        self.PrepareNamelist(i)
    cape/pyover/cntl.py:447: in PrepareNamelist
        nopts = self.opts.select_namelist(j)
    cape/optdict/__init__.py:3667: in wrapper
        v = f(*a, **kw)
    cape/pyover/options/overnmlopts.py:48: in select_namelist
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

.. literalinclude:: _test_001_pyovercli.py
    :caption: test_02_c
    :language: python
    :pyobject: test_02_c

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, fresh=False)
        def test_02_c():
            # Split command and add `-m` prefix
            cmdlist = [sys.executable, "-m", "cape.pyover", "-c", "-I", "1"]
            # Run the command
            stdout, _, _ = testutils.call_o(cmdlist)
            # Check outout
            result = testutils.compare_files(stdout, "test.02.out", ELLIPSIS=True)
    >       assert result.line1 == result.line2
    E       AssertionError: assert '1    powerof...           \n' == '1    powerof...0   .   ...\n'
    E         - 1    poweroff/m0.8a4.0b0.0 DONE    1500/1500   .   ...
    E         + 1    poweroff/m0.8a4.0b0.0 ---     /           .
    
    test/903_pyover/001_bullet/test_001_pyovercli.py:41: AssertionError

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
            cntl.cli(fm=True, I="1")
            # Read databook
            cntl.ReadDataBook()
            # Get value
    >       CN = cntl.DataBook["bullet_no_base"]["CN"][0]
    E       IndexError: index 0 is out of bounds for axis 0 with size 0
    
    test/903_pyover/001_bullet/test_001_pyovercli.py:54: IndexError

