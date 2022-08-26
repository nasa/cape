
---------------------------------------------------------------------
Test results: :mod:`test.002_attdb.003_regularize.test_01_regularize`
---------------------------------------------------------------------

This page documents from the folder

    ``test/002_attdb/003_regularize``

using the file

    ``test_01_regularize.py``

.. literalinclude:: _test_01_regularize.py
    :caption: test_01_regularize.py
    :language: python

Test case: :func:`test_01_griddata`
-----------------------------------
This test case runs the function:

.. literalinclude:: _test_01_regularize.py
    :caption: test_01_griddata
    :language: python
    :pyobject: test_01_griddata

FAIL

Failure contents:

.. code-block:: none

    @testutils.run_testdir(__file__)
        def test_01_griddata():
            db = rdb.DataKit(MAT_FILE)
            # Number of break points
            n = 17
            # Reference points for regularization
            A0 = np.linspace(-2, 2, n)
            B0 = np.linspace(-2, 2, n)
            # Save break points
            db.bkpts = {"alpha": A0, "beta": B0}
            # Regularize
    >       db.regularize_by_griddata("CN", ["alpha", "beta"], prefix="reg")
    
    test/002_attdb/003_regularize/test_01_regularize.py:24: 
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    cape/attdb/rdb.py:12383: in regularize_by_griddata
        argreg = self._translate_colname(arg, *tr_args)
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
    
    self = <rdb.DataKit('CN-alpha-beta.mat', cols=['alpha', 'beta', 'CN', 'regCN'])>
    col = None, trans = {}, prefix = 'reg', suffix = None
    
        def _translate_colname(self, col, trans, prefix, suffix):
            r"""Translate column name
        
            :Call:
                >>> dbcol = db._translate_colname(col, |args|)
            :Inputs:
                *db*: :class:`cape.attdb.ftypes.basefile.BaseFile`
                    Data file interface
                *col*: :class:`str`
                    "Original" column name, e.g. from file
                *trans*: :class:`dict`\ [:class:`str`]
                    Alternate names; *col* -> *trans[col]*
                *prefix*: :class:`str` | :class:`dict`
                    Universal prefix or *col*-specific prefixes
                *suffix*: :class:`str` | :class:`dict`
                    Universal suffix or *col*-specific suffixes
            :Outputs:
                *dbcol*: :class:`str`
                    Column names as stored in *db*
            :Versions:
                * 2019-12-04 ``@ddalle``: Version 1.0
                * 2020-02-22 ``@ddalle``: Single-column version
        
            .. |args| replace:: trans, prefix, suffix
            """
            # Get substitution (default is no substitution)
            dbcol = trans.get(col, col)
            # Get prefix
            if isinstance(prefix, dict):
                # Get specific prefix
                pre = prefix.get(col, prefix.get("_", ""))
            elif prefix:
                # Universal prefix
                pre = prefix
            else:
                # No prefix (type-safe)
                pre = ""
            # Get suffix
            if isinstance(suffix, dict):
                # Get specific suffix
                suf = suffix.get(col, suffix.get("_", ""))
            elif suffix:
                # Universal suffix
                suf = suffix
            else:
                # No suffix (type-safe)
                suf = ""
            # Combine fixes
    >       return pre + dbcol + suf
    E       TypeError: must be str, not NoneType
    
    cape/attdb/ftypes/basedata.py:1543: TypeError

Test case: :func:`test_02_rbf`
------------------------------
This test case runs the function:

.. literalinclude:: _test_01_regularize.py
    :caption: test_02_rbf
    :language: python
    :pyobject: test_02_rbf

PASS

