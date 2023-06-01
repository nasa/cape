
--------------------------------------------------------
Test results: :mod:`test.001_cape.005_cntl.test_01_cntl`
--------------------------------------------------------

This page documents from the folder

    ``test/001_cape/005_cntl``

using the file

    ``test_01_cntl.py``

.. literalinclude:: _test_01_cntl.py
    :caption: test_01_cntl.py
    :language: python

Test case: :func:`test_01_cntl`
-------------------------------
This test case runs the function:

.. literalinclude:: _test_01_cntl.py
    :caption: test_01_cntl
    :language: python
    :pyobject: test_01_cntl

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, TEST_FILES, TEST_DIRS)
        def test_01_cntl():
            # Instatiate
            cntl = cape.cntl.Cntl()
            # Test __repr__
    >       assert str(cntl) == "<cape.Cntl(nCase=20)>"
    E       AssertionError: assert '<cape.cntl.Cntl(nCase=20)>' == '<cape.Cntl(nCase=20)>'
    E         - <cape.Cntl(nCase=20)>
    E         + <cape.cntl.Cntl(nCase=20)>
    E         ?      +++++
    
    test/001_cape/005_cntl/test_01_cntl.py:28: AssertionError

