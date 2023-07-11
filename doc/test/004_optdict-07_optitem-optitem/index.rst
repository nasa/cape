
------------------------------------------------------------------
Test results: :mod:`test.004_optdict.test_07_optitem.test_optitem`
------------------------------------------------------------------

This page documents from the folder

    ``test/004_optdict/test_07_optitem``

using the file

    ``test_optitem.py``

.. literalinclude:: _test_optitem.py
    :caption: test_optitem.py
    :language: python

Test case: :func:`test_getel01`
-------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_getel01
    :language: python
    :pyobject: test_getel01

PASS

Test case: :func:`test_getel02`
-------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_getel02
    :language: python
    :pyobject: test_getel02

PASS

Test case: :func:`test_getel03`
-------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_getel03
    :language: python
    :pyobject: test_getel03

PASS

Test case: :func:`test_getel04`
-------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_getel04
    :language: python
    :pyobject: test_getel04

PASS

Test case: :func:`test_getel05`
-------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_getel05
    :language: python
    :pyobject: test_getel05

PASS

Test case: :func:`test_getel06`
-------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_getel06
    :language: python
    :pyobject: test_getel06

PASS

Test case: :func:`test_getel07`
-------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_getel07
    :language: python
    :pyobject: test_getel07

PASS

Test case: :func:`test_getel_error01`
-------------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_getel_error01
    :language: python
    :pyobject: test_getel_error01

PASS

Test case: :func:`test_setel01`
-------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_setel01
    :language: python
    :pyobject: test_setel01

PASS

Test case: :func:`test_setel02`
-------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_setel02
    :language: python
    :pyobject: test_setel02

PASS

Test case: :func:`test_setel03`
-------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_setel03
    :language: python
    :pyobject: test_setel03

PASS

Test case: :func:`test_setel04`
-------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_setel04
    :language: python
    :pyobject: test_setel04

PASS

Test case: :func:`test_setel05`
-------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_setel05
    :language: python
    :pyobject: test_setel05

PASS

Test case: :func:`test_setel06`
-------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_setel06
    :language: python
    :pyobject: test_setel06

PASS

Test case: :func:`test_setel07`
-------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_setel07
    :language: python
    :pyobject: test_setel07

PASS

Test case: :func:`test_getel_expr01`
------------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_getel_expr01
    :language: python
    :pyobject: test_getel_expr01

FAIL

Failure contents:

.. code-block:: none

    def test_getel_expr01():
            # Working example
            assert abs(optitem.getel(MYEXPR, x=X, i=0) - 0.25) <= 1e-6
            # Test not-a-string
            try:
                optitem.getel({"@expr": 2})
            except OptdictTypeError:
                pass
            else:
    >           assert False
    E           assert False
    
    test/004_optdict/test_07_optitem/test_optitem.py:152: AssertionError

Test case: :func:`test_getel_map01`
-----------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_getel_map01
    :language: python
    :pyobject: test_getel_map01

PASS

Test case: :func:`test_getel_cons01`
------------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_getel_cons01
    :language: python
    :pyobject: test_getel_cons01

PASS

Test case: :func:`test_getel_raw01`
-----------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_getel_raw01
    :language: python
    :pyobject: test_getel_raw01

FAIL

Failure contents:

.. code-block:: none

    def test_getel_raw01():
    >       assert optitem.getel({"@raw": [0, 1]}, j=0) == [0, 1]
    E       AssertionError: assert {'@raw': [0, 1]} == [0, 1]
    E         Use -v to get more diff
    
    test/004_optdict/test_07_optitem/test_optitem.py:210: AssertionError

Test case: :func:`test_getel_dict01`
------------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_getel_dict01
    :language: python
    :pyobject: test_getel_dict01

PASS

Test case: :func:`test_getel_special01`
---------------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_getel_special01
    :language: python
    :pyobject: test_getel_special01

FAIL

Failure contents:

.. code-block:: none

    def test_getel_special01():
            # Use a @map with a bad key
            try:
                optitem.getel({"@expr": "$mach", "bad": True})
            except OptdictKeyError:
                pass
            else:
    >           assert False
    E           assert False
    
    test/004_optdict/test_07_optitem/test_optitem.py:224: AssertionError

Test case: :func:`test_getel_compound01`
----------------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_getel_compound01
    :language: python
    :pyobject: test_getel_compound01

PASS

Test case: :func:`test_getel_x01`
---------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_getel_x01
    :language: python
    :pyobject: test_getel_x01

PASS

Test case: :func:`test_getel_x02`
---------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_getel_x02
    :language: python
    :pyobject: test_getel_x02

FAIL

Failure contents:

.. code-block:: none

    def test_getel_x02():
            # Test an expression whose run matrix value is a scalar
    >       assert optitem.getel({"@expr": "$aoap"}, x=X) == 0.0
    E       AssertionError: assert {'@expr': '$aoap'} == 0.0
    E        +  where {'@expr': '$aoap'} = <function getel at 0x269a5d0>({'@expr': '$aoap'}, x={'aoap': 0.0, 'arch': ['sky', 'sky', 'cas', 'cas', 'rom'], 'mach': array([0.5 , 0.75, 1.  , 1.25, 1.5 ])})
    E        +    where <function getel at 0x269a5d0> = optitem.getel
    
    test/004_optdict/test_07_optitem/test_optitem.py:243: AssertionError

Test case: :func:`test_getel_i01`
---------------------------------
This test case runs the function:

.. literalinclude:: _test_optitem.py
    :caption: test_getel_i01
    :language: python
    :pyobject: test_getel_i01

FAIL

Failure contents:

.. code-block:: none

    def test_getel_i01():
            # No index
            mach = optitem.getel({"@expr": "$mach"}, x=X)
    >       assert np.max(np.abs(mach - X["mach"])) <= 1e-6
    E       TypeError: unsupported operand type(s) for -: 'dict' and 'float'
    
    test/004_optdict/test_07_optitem/test_optitem.py:258: TypeError

