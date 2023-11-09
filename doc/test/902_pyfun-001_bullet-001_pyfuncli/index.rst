
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
    cape/cntl.py:1163: in SubmitJobs
        self.StartCase(i)
    cape/cntl.py:96: in wrapper_func
        v = func(self, *args, **kwargs)
    cape/cntl.py:1456: in StartCase
        pbs = self.CaseStartCase()
    cape/pyfun/cntl.py:3047: in CaseStartCase
        return case.StartCase()
    cape/pyfun/case.py:412: in StartCase
        run_fun3d()
    cape/pyfun/case.py:114: in run_fun3d
        RunPhase(rc, i)
    cape/pyfun/case.py:234: in RunPhase
        bin.callf(cmdi, f='fun3d.out')
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    cmdi = ['mpiexec', '-np', '50', 'nodet_mpi', '--animation_freq', '100', ...]
    f = 'fun3d.out', e = None, shell = None, v = True, check = True
    
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
    E           SystemError: Command failed with status 1.
    
    cape/cfdx/bin.py:148: SystemError

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
    E       AssertionError: assert '8    bullet/...           \n' == '8    bullet/...       ... \n'
    E         - 8    bullet/m1.10a0.0b0.0  DONE    200/200     .        ... 
    E         ?                            ^ ^^^^^^^^                   ^^^
    E         + 8    bullet/m1.10a0.0b0.0  INCOMP  0/200       .            
    E         ?                            ^^^ ^^^^          ++         ^^^
    
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

