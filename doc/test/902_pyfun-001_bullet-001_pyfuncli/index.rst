
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

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_sandbox(__file__, TEST_FILES)
        def test_01_run():
            # Instantiate
            cntl = cape.pyfun.cntl.Cntl()
            # Run first case
    >       cntl.SubmitJobs(I="8")
    
    test/902_pyfun/001_bullet/test_001_pyfuncli.py:29: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    cape/cntl.py:1168: in SubmitJobs
        self.PrepareCase(i)
    cape/pyfun/cntl.py:1435: in PrepareCase
        self.PrepareNamelist(i)
    cape/pyfun/cntl.py:1493: in PrepareNamelist
        self.PrepareNamelistConfig()
    cape/pyfun/cntl.py:1787: in PrepareNamelistConfig
        RefP = self.opts.get_RefPoint(comp)
    cape/optdict/__init__.py:3651: in wrapper
        v = f(*a, **kw)
    cape/cfdx/options/configopts.py:487: in get_RefPoint
        return self.expand_Point(x)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
    
    self = {'ConfigFile': 'bullet.xml', 'RefArea': 3.14159, 'RefLength': 2.0, 'RefPoint': {'entire': 'MRP', 'bullet_no_base': 'MR... 0.0], 'MRP': [1.75, 0.0, 0.0]}, 'Components': ['bullet_no_base', 'bullet_total', 'cap', 'body', 'base'], 'Inputs': {}}
    x = 1.0
    
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
    
    cape/cfdx/options/configopts.py:457: TypeError

