
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

PASS

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
    E       AssertionError: assert '8    bullet/...           \n' == '8    bullet/...       0...\n'
    E         - 8    bullet/m1.10a0.0b0.0  DONE    200/200     .        0...
    E         ?                            ^ ^^^^^^^^                   ^^^^
    E         + 8    bullet/m1.10a0.0b0.0  ERROR   0/200       .            
    E         ?                            ^^^ ^^^^          ++         ^^^^
    
    test/902_pyfun/001_bullet/test_001_pyfuncli.py:43: AssertionError

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
    
    test/902_pyfun/001_bullet/test_001_pyfuncli.py:56: IndexError

