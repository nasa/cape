
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
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/cntl.py:1167: in SubmitJobs
        self.PrepareCase(i)
    cape/pycart/cntl.py:578: in PrepareCase
        self.PrepareMesh(i)
    cape/pycart/cntl.py:769: in PrepareMesh
        case.CaseCubes(rc, j=0)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    rc = {'PhaseSequence': [0], 'PhaseIters': [200], 'nProc': 4, 'MPI': False, 'flowCart': {'first_order': False, 'it_fc': 200,...': False}, 'Environ': {}, 'aflr3': {'run': False}, 'intersect': {'run': False}, 'ulimit': {}, 'verify': {'run': False}}
    j = 0
    
        def CaseCubes(rc, j=0):
            """Run ``cubes`` and ``mgPrep`` to create multigrid volume mesh
        
            :Call:
                >>> CaseCubes(rc, j=0)
            :Inputs:
                *rc*: :class:`cape.options.runControl.RunControl`
                    Case options interface from ``case.json``
                *j*: {``0``} | :class:`int`
                    Phase number
            :Versions:
                * 2016-04-06 ``@ddalle``: Version 1.0
            """
            # Check for previous iterations
            # TODO: This will need an edit for 'remesh'
            if GetRestartIter() > 0: return
            # Check for mesh file
            if os.path.isfile('Mesh.mg.c3d'): return
            # Check for cubes option
    >       if not rc.get_cubes_run():
    E       AttributeError: 'RunControlOpts' object has no attribute 'get_cubes_run'
    
    cape/pycart/case.py:198: AttributeError

