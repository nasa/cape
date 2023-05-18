
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
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/cntl.py:1167: in SubmitJobs
        self.PrepareCase(i)
    cape/pyfun/cntl.py:1435: in PrepareCase
        self.PrepareNamelist(i)
    cape/pyfun/cntl.py:1493: in PrepareNamelist
        self.PrepareNamelistConfig()
    cape/pyfun/cntl.py:1787: in PrepareNamelistConfig
        RefP = self.opts.get_RefPoint(comp)
    cape/optdict/__init__.py:3651: in wrapper
        v = f(*a, **kw)
    cape/cfdx/options/configopts.py:483: in get_RefPoint
        return self.expand_Point(x)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    self = {'ConfigFile': 'bullet.xml', 'RefArea': 3.14159, 'RefLength': 2.0, 'RefPoint': {'entire': 'MRP', 'bullet_no_base': 'MR... 0.0], 'MRP': [1.75, 0.0, 0.0]}, 'Components': ['bullet_no_base', 'bullet_total', 'cap', 'body', 'base'], 'Inputs': {}}
    x = 1.0
    
        def expand_Point(self, x):
            r"""Expand points that are specified by name instead of value
        
            :Call:
                >>> x = opts.expand_Point(x)
                >>> x = opts.expand_Point(s)
                >>> X = opts.expand_Point(d)
            :Inputs:
                *opts*: :class:`cape.cfdx.options.Options`
                    Options interface
                *x*: :class:`list`\ [:class:`float`]
                    Point
                *s*: :class:`str`
                    Point name
                *d*: :class:`dict`
                    Dictionary of points and point names
            :Outputs:
                *x*: [:class:`float`, :class:`float`, :class:`float`]
                    Point
                *X*: :class:`dict`
                    Dictionary of points
            :Versions:
                * 2015-09-12 ``@ddalle``: Version 1.0
            """
            # Check input type.
            if isinstance(x, str):
                # Single point name
                return self.get_Point(x)
            elif isinstance(x, ARRAY_TYPES):
                # Check length
                n = len(x)
                # Check length
                if n in (2, 3):
                    # Check first entry
                    if isinstance(x[0], FLOAT_TYPES):
                        # Already a point
                        return x
                # Otherwise, this is a list of points
                return [self.get_Point(xk) for xk in x]
            elif not isinstance(x, dict):
                # Unrecognized
    >           raise TypeError(
                    "Cannot expand points of type '%s'"
                    % type(x).__name__)
    E           TypeError: Cannot expand points of type 'float'
    
    cape/cfdx/options/configopts.py:453: TypeError

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
    E       AssertionError: assert '8    bullet/...           \n' == '8    bullet/...       0.1 \n'
    E         - 8    bullet/m1.10a0.0b0.0  DONE    200/200     .        0.1 
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
    >       cntl.cli(fm=True, I="8")
    
    test/902_pyfun/001_bullet/test_001_pyfuncli.py:50: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/pyfun/cntl.py:242: in cli
        cmd = self.cli_cape(*a, **kw)
    cape/cntl.py:804: in cli_cape
        self.UpdateFM(**kw)
    cape/cntl.py:99: in wrapper_func
        v = func(self, *args, **kwargs)
    cape/cntl.py:4057: in UpdateFM
        comp = self.opts.get_DataBookByGlob(["FM", "Force", "Moment"], comp)
    cape/optdict/__init__.py:3651: in wrapper
        v = f(*a, **kw)
    cape/cfdx/options/databookopts.py:1273: in get_DataBookByGlob
        comps_all = self.get_DataBookByType(typ)
    cape/cfdx/options/databookopts.py:1239: in get_DataBookByType
        self.validate_DataBookType(typ)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    self = {'Components': ['bullet_no_base', 'bullet_total', 'cap', 'body', 'base'], 'Folder': 'data/bullet', 'NStats': 50, 'NMin...tal': {'Type': 'Force'}, 'cap': {'Type': 'Force'}, 'body': {'Type': 'Force'}, 'base': {'Type': 'Force'}, 'Targets': {}}
    typ = ['FM', 'Force', 'Moment']
    
        def validate_DataBookType(self, typ: str):
            r"""Ensure that *typ* is a recognized DataBook *Type*
        
            :Call:
                >>> opts.validate_DataBookType(typ)
            :Inputs:
                *opts*: :class:`cape.cfdx.options.Options`
                    Options interface
                *typ*: ``"FM"`` | :class:`str`
                    Target value for ``"Type"`` of matching components
            :Raises:
                :class:`ValueError`
            :Versions:
                * 2023-03-09 ``@ddalle``: v1.0
            """
            # Check value
    >       if typ not in self.__class__._sec_cls_optmap:
    E       TypeError: unhashable type: 'list'
    
    cape/cfdx/options/databookopts.py:1315: TypeError

