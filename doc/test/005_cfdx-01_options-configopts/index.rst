
-------------------------------------------------------------
Test results: :mod:`test.005_cfdx.01_options.test_configopts`
-------------------------------------------------------------

This page documents from the folder

    ``test/005_cfdx/01_options``

using the file

    ``test_configopts.py``

.. literalinclude:: _test_configopts.py
    :caption: test_configopts.py
    :language: python

Test case: :func:`test_ConfigOpts1`
-----------------------------------
This test case runs the function:

.. literalinclude:: _test_configopts.py
    :caption: test_ConfigOpts1
    :language: python
    :pyobject: test_ConfigOpts1

FAIL

Failure contents:

.. code-block:: none

    def test_ConfigOpts1():
            # Initialize options
            opts = ConfigOpts(**OPTS1)
            # Test point
            x1 = [4.0, 0.0, 0.1]
            # Set span
            opts.set_RefSpan(0.25)
            opts.set_RefSpan(1.0, comp="comp2")
            # Test reference quantities
            assert opts.get_RefArea() == OPTS1["RefArea"]
            assert opts.get_RefLength("comp1") == 1.0
            assert opts.get_RefLength("comp2") == 2.0
            assert opts.get_RefSpan("comp1") == 0.25
            # Test other setters
            opts.set_RefArea(1.5, comp="comp2")
            opts.set_RefLength(2.5, comp="comp2")
            # Delete the spane
            opts.pop("RefSpan")
            # Test fall-back: RefSpan -> RefLength
            assert opts.get_RefSpan("comp1") == 1.0
            # Reference point
            opts.set_RefPoint(x1, "comp2")
            assert opts.get_RefPoint("comp1") == OPTS1["Points"]["MRP"]
    >       assert opts.get_RefPoint("comp2") == x1
    
    test/005_cfdx/01_options/test_configopts.py:52: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/cfdx/options/configopts.py:490: in get_RefPoint
        return self.expand_Point(x)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    self = {'Components': ['comp1', 'comp2'], 'ConfigFile': 'Config.xml', 'Points': {'MRP': [10.0, 0.0, 0.0]}, 'RefArea': {'_defa...RefLength': {'comp1': 1.0, '_default_': 2.0, 'comp2': 2.5}, 'RefPoint': {'_default_': 'MRP', 'comp2': [4.0, 0.0, 0.1]}}
    x = 4.0
    
        def expand_Point(self, x):
            r"""Expand points that are specified by name instead of value
        
            :Call:
                >>> x = opts.expand_Point(x)
                >>> x = opts.expand_Point(s)
                >>> X = opts.expand_Point(d)
            :Inputs:
                *opts*: :class:`cape.cfdx.options.Options`
                    Options interface
                *x*: :class:`list`\ [:class:`float`]
                    Point
                *s*: :class:`str`
                    Point name
                *d*: :class:`dict`
                    Dictionary of points and point names
            :Outputs:
                *x*: [:class:`float`, :class:`float`, :class:`float`]
                    Point
                *X*: :class:`dict`
                    Dictionary of points
            :Versions:
                * 2015-09-12 ``@ddalle``: Version 1.0
            """
            # Check input type.
            if isinstance(x, str):
                # Single point name
                return self.get_Point(x)
            elif x is None:
                # Null input
                return []
            elif isinstance(x, ARRAY_TYPES):
                # Check length
                n = len(x)
                # Check length
                if n in (2, 3):
                    # Check first entry
                    if isinstance(x[0], FLOAT_TYPES):
                        # Already a point
                        return x
                # Otherwise, this is a list of points
                return [self.get_Point(xk) for xk in x]
            elif not isinstance(x, dict):
                # Unrecognized
    >           raise TypeError(
                    "Cannot expand points of type '%s'"
                    % type(x).__name__)
    E           TypeError: Cannot expand points of type 'float'
    
    cape/cfdx/options/configopts.py:460: TypeError

Test case: :func:`test_ConfigOpts2`
-----------------------------------
This test case runs the function:

.. literalinclude:: _test_configopts.py
    :caption: test_ConfigOpts2
    :language: python
    :pyobject: test_ConfigOpts2

PASS

