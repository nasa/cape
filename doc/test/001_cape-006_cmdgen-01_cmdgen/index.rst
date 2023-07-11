
------------------------------------------------------------
Test results: :mod:`test.001_cape.006_cmdgen.test_01_cmdgen`
------------------------------------------------------------

This page documents from the folder

    ``test/001_cape/006_cmdgen``

using the file

    ``test_01_cmdgen.py``

.. literalinclude:: _test_01_cmdgen.py
    :caption: test_01_cmdgen.py
    :language: python

Test case: :func:`test_01_aflr3`
--------------------------------
This test case runs the function:

.. literalinclude:: _test_01_cmdgen.py
    :caption: test_01_aflr3
    :language: python
    :pyobject: test_01_aflr3

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_testdir(__file__)
        def test_01_aflr3():
            # Read settings
    >       rc = case.ReadCaseJSON()
    E       AttributeError: module 'cape.cfdx.case' has no attribute 'ReadCaseJSON'
    
    test/001_cape/006_cmdgen/test_01_cmdgen.py:14: AttributeError

Test case: :func:`test_02_intersect`
------------------------------------
This test case runs the function:

.. literalinclude:: _test_01_cmdgen.py
    :caption: test_02_intersect
    :language: python
    :pyobject: test_02_intersect

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_testdir(__file__)
        def test_02_intersect():
            # Read settings
    >       rc = case.ReadCaseJSON()
    E       AttributeError: module 'cape.cfdx.case' has no attribute 'ReadCaseJSON'
    
    test/001_cape/006_cmdgen/test_01_cmdgen.py:40: AttributeError

