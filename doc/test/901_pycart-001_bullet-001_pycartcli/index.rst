
------------------------------------------------------------------
Test results: :mod:`test.901_pycart.001_bullet.test_001_pycartcli`
------------------------------------------------------------------

This page documents from the folder

    ``test/901_pycart/001_bullet``

using the file

    ``test_001_pycartcli.py``

.. literalinclude:: _test_001_pycartcli.py
    :caption: test_001_pycartcli.py
    :language: python

Test case: :func:`test_01_c`
----------------------------
This test case runs the function:

.. literalinclude:: _test_001_pycartcli.py
    :caption: test_01_c
    :language: python
    :pyobject: test_01_c

PASS

Test case: :func:`test_02_run`
------------------------------
This test case runs the function:

.. literalinclude:: _test_001_pycartcli.py
    :caption: test_02_run
    :language: python
    :pyobject: test_02_run

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, fresh=False)
        def test_02_run():
            # Instantiate
            cntl = cape.pycart.cntl.Cntl()
            # Run first case
            cntl.SubmitJobs(I="0")
            # Collect aero
    >       cntl.cli(fm=True, I="0")
    
    test/901_pycart/001_bullet/test_001_pycartcli.py:42: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/pycart/cntl.py:178: in cli
        cmd = self.cli_cape(*a, **kw)
    cape/cntl.py:804: in cli_cape
        self.UpdateFM(**kw)
    cape/cntl.py:99: in wrapper_func
        v = func(self, *args, **kwargs)
    cape/cntl.py:4057: in UpdateFM
        comp = self.opts.get_DataBookByGlob("FM", comp)
    cape/optdict/__init__.py:3651: in wrapper
        v = f(*a, **kw)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    self = {'Components': ['bullet_no_base'], 'NStats': 50, 'NMin': 140, 'Folder': 'data', 'Targets': {}, 'bullet_no_base': {'Type': 'FM'}}, typ = 'FM'
    pat = True
    
        def get_DataBookByGlob(self, typ, pat=None):
            r"""Get list of components by type and list of wild cards
        
            :Call:
                >>> comps = opts.get_DataBookByGlob(typ, pat=None)
            :Inputs:
                *opts*: :class:`cape.cfdx.options.Options`
                    Options interface
                *typ*: ``"FM"`` | :class:`str`
                    Target value for ``"Type"`` of matching components
                *pat*: {``None``} | :class:`str` | :class:`list`
                    List of component name patterns
            :Outputs:
                *comps*: :class:`str`
                    All components meeting one or more wild cards
            :Versions:
                * 2017-04-25 ``@ddalle``: v1.0
                * 2023-02-06 ``@ddalle``: v1.1; improved naming
                * 2023-03-09 ``@ddalle``: v1.2; validate *typ*
            """
            # Get list of all components with matching type
            comps_all = self.get_DataBookByType(typ)
            # Check for default option
            if pat is None:
                return comps_all
            # Initialize output
            comps = []
            # Ensure input is a list
            if isinstance(pat, ARRAY_TYPES):
                # Already a list
                pats = pat
            else:
                # Read as string: comma-separated list
    >           pats = pat.split(",")
    E           AttributeError: 'bool' object has no attribute 'split'
    
    cape/cfdx/options/databookopts.py:1285: AttributeError

