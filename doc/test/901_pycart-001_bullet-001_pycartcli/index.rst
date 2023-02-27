
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
    cape/cntl.py:1155: in SubmitJobs
        self.PrepareCase(i)
    cape/pycart/cntl.py:578: in PrepareCase
        self.PrepareMesh(i)
    cape/pycart/cntl.py:769: in PrepareMesh
        case.CaseCubes(rc, j=0)
    cape/pycart/case.py:203: in CaseCubes
        bin.cubes(opts=rc, j=j)
    cape/pycart/bin.py:52: in cubes
        callf(cmdi, f='cubes.out', v=v)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    cmdi = ['cubes', '-pre', 'preSpec.c3d.cntl', '-maxR', '8', '-reorder', ...]
    f = 'cubes.out', e = None, shell = None, v = False, check = True
    
        def callf(cmdi, f=None, e=None, shell=None, v=True, check=True):
            r"""Call a command with alternate STDOUT by filename
        
            :Call:
                >>> callf(cmdi, f=None, e=None, shell=None, v=True, check=True)
            :Inputs:
                *cmdi*: :class:`list` (:class:`str`)
                    List of strings as for :func:`subprocess.call`
                *f*: :class:`str`
                    File name to which to store STDOUT
                *e*: {*f*} | :class:`str`
                    Separate file name for STDERR
                *shell*: :class:`bool`
                    Whether or not a shell is needed
                *v*: {``True``} | :class:`False`
                    Verbose option; display *PWD* and *STDOUT* values
            :Versions:
                * 2014-08-30 ``@ddalle``: Version 1.0
                * 2015-02-13 ``@ddalle``: Version 2.0; rely on :func:`calli`
                * 2017-03-12 ``@ddalle``: Version 2.1; add *v* option
                * 2019-06-10 ``@ddalle``: Version 2.2; add *e* option
            """
            # Call the command with output status
            ierr = calli(cmdi, f, e, shell, v=v)
            # Check the status.
            if ierr and check:
                # Remove RUNNING file.
                if os.path.isfile('RUNNING'):
                    # Delete it.
                    os.remove('RUNNING')
                # Exit with error notifier.
    >           raise SystemError("Command failed with status %i." % ierr)
    E           SystemError: Command failed with status 255.
    
    cape/cfdx/bin.py:150: SystemError

