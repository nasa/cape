
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

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, TEST_FILES)
        def test_01_c():
            # Split command and add `-m` prefix
            cmdlist = [sys.executable, "-m", "cape.pycart", "-c"]
            # Run the command
            stdout, _, _ = testutils.call_o(cmdlist)
            # Check outout
            result = testutils.compare_files(stdout, "test.01.out")
    >       assert result.line1 == result.line2
    E       AssertionError: assert '' == '0    powerof...           \n'
    E         - 0    poweroff/m1.5a0.0b0.0 ---     /           .
    
    test/901_pycart/001_bullet/test_001_pycartcli.py:31: AssertionError

Test case: :func:`test_02_run`
------------------------------
This test case runs the function:

.. literalinclude:: _test_001_pycartcli.py
    :caption: test_02_run
    :language: python
    :pyobject: test_02_run

PASS

