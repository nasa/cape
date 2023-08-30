
-------------------------------------------------------------
Test results: :mod:`test.005_cfdx.01_options.test_ulimitopts`
-------------------------------------------------------------

This page documents from the folder

    ``test/005_cfdx/01_options``

using the file

    ``test_ulimitopts.py``

.. literalinclude:: _test_ulimitopts.py
    :caption: test_ulimitopts.py
    :language: python

Test case: :func:`test_rcopts1`
-------------------------------
This test case runs the function:

.. literalinclude:: _test_ulimitopts.py
    :caption: test_rcopts1
    :language: python
    :pyobject: test_rcopts1

FAIL

Failure contents:

.. code-block:: none

    def test_rcopts1():
            # Initialize options
            opts = ULimitOpts(OPTS1)
            # Get values
    >       assert opts.get_ulimit("u") == ULimitOpts._rc["u"]
    E       KeyError: 'u'
    
    test/005_cfdx/01_options/test_ulimitopts.py:16: KeyError

