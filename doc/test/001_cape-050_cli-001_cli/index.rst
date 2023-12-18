
-------------------------------------------------------
Test results: :mod:`test.001_cape.050_cli.test_001_cli`
-------------------------------------------------------

This page documents from the folder

    ``test/001_cape/050_cli``

using the file

    ``test_001_cli.py``

.. literalinclude:: _test_001_cli.py
    :caption: test_001_cli.py
    :language: python

Test case: :func:`test_c`
-------------------------
This test case runs the function:

.. literalinclude:: _test_001_cli.py
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
                stdout, _, _ = testutils.call_o(cmdlistj)
                # Name of file with target output
                ftarg = "test.%02i.out" % (j + 1)
                # Check outout
                result = testutils.compare_files(stdout, ftarg)
    >           assert result.line1 == result.line2
    E           AssertionError: assert '' == '0    powerof...           \n'
    E             - 0    poweroff/m0.5a0.0b0.0 ---     /           .
    
    test/001_cape/050_cli/test_001_cli.py:42: AssertionError

