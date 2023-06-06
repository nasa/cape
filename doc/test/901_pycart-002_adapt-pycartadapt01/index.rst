
-----------------------------------------------------------------
Test results: :mod:`test.901_pycart.002_adapt.test_pycartadapt01`
-----------------------------------------------------------------

This page documents from the folder

    ``test/901_pycart/002_adapt``

using the file

    ``test_pycartadapt01.py``

.. literalinclude:: _test_pycartadapt01.py
    :caption: test_pycartadapt01.py
    :language: python

Test case: :func:`test_02_run`
------------------------------
This test case runs the function:

.. literalinclude:: _test_pycartadapt01.py
    :caption: test_02_run
    :language: python
    :pyobject: test_02_run

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, TEST_FILES)
        def test_02_run():
            # Instantiate
            cntl = cape.pycart.cntl.Cntl()
            # Run first case
    >       cntl.SubmitJobs(I="0")
    
    test/901_pycart/002_adapt/test_pycartadapt01.py:29: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    cape/cntl.py:1155: in SubmitJobs
        self.StartCase(i)
    cape/cntl.py:96: in wrapper_func
        v = func(self, *args, **kwargs)
    cape/cntl.py:1448: in StartCase
        pbs = self.CaseStartCase()
    cape/pycart/cntl.py:321: in CaseStartCase
        return case.StartCase()
    cape/pycart/case.py:589: in StartCase
        run_flowCart()
    cape/pycart/case.py:107: in run_flowCart
        RunPhase(rc, i)
    cape/pycart/case.py:305: in RunPhase
        RunAdaptive(rc, i)
    cape/pycart/case.py:344: in RunAdaptive
        bin.callf(cmdi, f='flowCart.out', v=v_fc)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    
    cmdi = ['./aero.csh'], f = 'flowCart.out', e = None, shell = None, v = False, check = True
    
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

