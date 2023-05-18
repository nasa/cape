
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
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/cntl.py:1167: in SubmitJobs
        self.PrepareCase(i)
    cape/pyover/cntl.py:360: in PrepareCase
        self.PrepareMesh(i)
    cape/pyover/cntl.py:548: in PrepareMesh
        fcfg = self.GetConfigDir(i)
    cape/pyover/cntl.py:886: in GetConfigDir
        fcfg = self.opts.get_MeshConfigDir(config)
    cape/optdict/__init__.py:3651: in wrapper
        v = f(*a, **kw)
    cape/optdict/__init__.py:3129: in func
        return self.get_opt(opt, j=j, i=i, **kw)
    cape/optdict/__init__.py:1530: in get_opt
        val = self._sample_val(v, j, i, **kw)
    cape/optdict/__init__.py:1606: in _sample_val
        return optitem.getel(v, j=j, i=i, **kw)
    cape/optdict/optitem.py:126: in getel
        vj = _getel_phase(v, j=j, **kw)
    cape/optdict/optitem.py:180: in _getel_phase
        j = _check_phase(j)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    j = 'poweroff'
    
        def _check_phase(j):
            # Check phase input
            if j is None:
                return
            if not isinstance(j, int):
    >           raise TypeError(
                    "Expected 'int' for input 'j'; got '%s'" %
                    type(j).__name__)
    E           TypeError: Expected 'int' for input 'j'; got 'str'
    
    cape/optdict/optitem.py:545: TypeError

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
    >       cntl.cli(fm=True, I="1")
    
    test/903_pyover/001_bullet/test_001_pyovercli.py:50: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/pyover/cntl.py:189: in cli
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
    
    self = {'Components': ['bullet_no_base', 'bullet_cap', 'bullet_body', 'bullet_base'], 'NMin': 2100, 'NStats': 200, 'Folder': 'data', 'fomo': 'common/fomo', 'Targets': {}}
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

