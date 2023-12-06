
--------------------------------------------------------------------
Test results: :mod:`test.902_pyfun.002_ellipsoid.test_001_aflr3mesh`
--------------------------------------------------------------------

This page documents from the folder

    ``test/902_pyfun/002_ellipsoid``

using the file

    ``test_001_aflr3mesh.py``

.. literalinclude:: _test_001_aflr3mesh.py
    :caption: test_001_aflr3mesh.py
    :language: python

Test case: :func:`test_01_aflr3`
--------------------------------
This test case runs the function:

.. literalinclude:: _test_001_aflr3mesh.py
    :caption: test_01_aflr3
    :language: python
    :pyobject: test_01_aflr3

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, TEST_FILES)
        def test_01_aflr3():
            # Instantiate
            cntl = cape.pyfun.cntl.Cntl()
            # Run a case
            cntl.SubmitJobs(I="6")
            # Get case folder
            case_folder = cntl.x.GetFullFolderNames(6)
            # Full path to ``aflr3.out`` and others
            log_file = os.path.join(case_folder, LOG_FILE)
            mesh_file = os.path.join(case_folder, f"{MESH_PREFIX}.ugrid")
            # Check if files exist
            assert os.path.isfile(log_file)
    >       assert os.path.isfile(mesh_file)
    E       AssertionError: assert False
    E        +  where False = <function isfile at 0x7fe66d2ef280>('Ellipsoid/m0.95a10.0b0.0/ellipsoid.ugrid')
    E        +    where <function isfile at 0x7fe66d2ef280> = <module 'posixpath' from '/usr/lib64/python3.9/posixpath.py'>.isfile
    E        +      where <module 'posixpath' from '/usr/lib64/python3.9/posixpath.py'> = os.path
    
    test/902_pyfun/002_ellipsoid/test_001_aflr3mesh.py:40: AssertionError

