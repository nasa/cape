
--------------------------------------------------------------
Test results: :mod:`test.001_cape.051_cli.test_001_cli_matrix`
--------------------------------------------------------------

This page documents from the folder

    ``test/001_cape/051_cli``

using the file

    ``test_001_cli_matrix.py``

.. literalinclude:: _test_001_cli_matrix.py
    :caption: test_001_cli_matrix.py
    :language: python

Test case: :func:`test_c`
-------------------------
This test case runs the function:

.. literalinclude:: _test_001_cli_matrix.py
    :caption: test_c
    :language: python
    :pyobject: test_c

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, TEST_FILES)
        def test_c():
            # Loop through commands
            for j, cmdj in enumerate(CMD_LIST):
                # Split command and add `-m` prefix
                cmdlistj = [sys.executable, "-m"] + shlex.split(cmdj)
                # Run the command
                stdout, stderr, ierr = testutils.call_oe(cmdlistj)
                # Name of file with target output
                if ierr:
                    # Something went wrong
                    ftarg = "test.%02i.err" % (j + 1)
                    # Check STDERR target exists
    >               assert os.path.isfile(ftarg)
    E               AssertionError: assert False
    E                +  where False = <function isfile at 0x7e0340>('test.02.err')
    E                +    where <function isfile at 0x7e0340> = <module 'posixpath' from '/nasa/pkgsrc/toss4/2022Q1-rome/views/python/3.9.12/bin/../../../../lib/python3.9/posixpath.py'>.isfile
    E                +      where <module 'posixpath' from '/nasa/pkgsrc/toss4/2022Q1-rome/views/python/3.9.12/bin/../../../../lib/python3.9/posixpath.py'> = os.path
    
    test/001_cape/051_cli/test_001_cli_matrix.py:44: AssertionError

