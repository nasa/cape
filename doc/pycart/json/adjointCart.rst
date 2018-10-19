
.. _pycart-json-adjointCart:

-----------------------------
Run Options for `adjointCart`
-----------------------------

The "adjointCart" section of :file:`pyCart.json` controls settings for executing
the adjoint solver.  This is a short section that affects settings in the
:file:`aero.00.csh`, :file:`aero.01.csh`, etc. files.  The default options
within a sample :file:`pyCart.json` file are shown below.

    .. code-block:: javascript
    
        "adjointCart": {
            "it_ad": 120,
            "mg_ad": 3
        }


The dictionary of options is shown below.

    *it_ad*: {``120``} | :class:`int` | :class:`list` (:class:`int`)
        Number of iterations of `adjointCart` to run
        
    *mg_ad*: {``3``} | :class:`int` | :class:`list` (:class:`int`)
        Number of multigrid levels to use while running `adjointCart`
        
