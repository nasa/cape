
.. _pycart-json-Adaptation:

------------------
Adaptation Options
------------------

The "Adaptation" section of :file:`pyCart.json` except for the one that tells
pyCart whether or not to run adaptively.  Below is a sample section of
:file:`pyCart.json` that contains the default values for the relevant options.

    .. code-block:: javascript
    
        "Adaptation": {
            "n_adapt_cycles": 0,
            "etol": 0.000001,
            "max_nCells": 50000000,
            "ws_it": [50, 50],
            "mesh_growth": [1.5, 1.5, 2.0, 2.0, 2.0, 2.0, 2.5],
            "apc": ["p", "a"],
            "abuff": 1,
            "final_mesh_xref": 0
        }

The full description of these options is given in the list below.  Most of these
options are almost exactly analogous to the similarly named variables in
:file:`aero.csh`.

    *n_adapt_cycles*: {``0``} | :class:`int` | :class:`list` (:class:`int`)
        Number of adaptation cycles to run
        
    *etol*: {``0.000001``} | :class:`float`
        Target error tolerance.  The adaptation cycles will terminate if this
        error level (in absolute value) in the output function is reached.
        
    *max_nCells*: {``50000000``} | :class:`int`
        Maximum number of volume mesh cells.  If an adaptation results in more
        cells than this number, analysis terminates.
        
    *ws_it*: {``[50, 50]``} | :class:`list` (:class:`int`)
        Number of `flowCart` iterations to run during each adaptation cycle.
        When setting up adaptation cycles, pyCart assumes the last entry in this
        list is repeated indefinitely.
        
    *mesh_growth*: :class:`list` (:class:`float`)
        Mesh growth factor for each adaptation cycle.  The volume mesh will
        increase the number of volume cells by approximately this number after a
        given adaptation cycle.  When setting up adaptation cycles, pyCart
        assumes that the last entry in this list is repeated indefinitely.
        
    *apc*: {``["p", "a"]``} | :class:`list` (``"p"`` | ``"a"``)
        Sets whether each adaptation cycle is a *propagate* ("p") cycle or
        *adapt* ("a") cycle.  Propagate cycles cannot decrease the size of the
        smallest cell.  When setting up adaptation cycles, pyCart assumes that
        the last entry in this list is repeated indefinitely.
        
    *abuff*: {``1``} | :class:`int` | :class:`list` (:class:`int`)
        Number of buffer layers to use when refining a mesh.  Increasing this
        parameter decreases the size of the smallest clump of cells that can be
        refined.  Increasing it results in less targeted refinement.
        
    *final_mesh_xref*: {``0``} | :class:`int`
        Number of extra refinements to compute using the final error map.  New
        adjoints are not computed for these additional *xref* adaptations.
