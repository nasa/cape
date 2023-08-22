
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
    >       cntl.SubmitJobs(I="0")
    
    test/901_pycart/001_bullet/test_001_pycartcli.py:40: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/cntl.py:1312: in SubmitJobs
        self.PrepareCase(i)
    cape/cntl.py:108: in wrapper_func
        v = func(self, *args, **kwargs)
    cape/pycart/cntl.py:337: in PrepareCase
        self.PrepareMesh(i)
    cape/cntl.py:108: in wrapper_func
        v = func(self, *args, **kwargs)
    cape/pycart/cntl.py:532: in PrepareMesh
        runner.run_cubes(0)
    cape/cfdx/case.py:122: in wrapper_func
        v = func(self, *args, **kwargs)
    cape/pycart/case.py:177: in run_cubes
        cmdrun.cubes(opts=rc, j=j)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    opts = {'PhaseSequence': [0], 'PhaseIters': [200], 'qsub': False, 'nProc': 128, 'MPI': False, 'flowCart': {'first_order': Fal...': False}, 'Environ': {}, 'aflr3': {'run': False}, 'intersect': {'run': False}, 'ulimit': {}, 'verify': {'run': False}}
    j = 0, kwargs = {}
    
        def cubes(opts=None, j=0, **kwargs):
            # Required file
            _assertfile('input.c3d')
            # Get command
    >       cmdi = cmdgen.cubes(cntl=cntl, opts=opts, j=j, **kwargs)
    E       NameError: name 'cntl' is not defined
    
    cape/pycart/cmdrun.py:42: NameError

