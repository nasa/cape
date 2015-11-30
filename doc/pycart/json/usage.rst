-----------------------------------
General Description of JSON Entries
-----------------------------------

The master settings file is loaded in one of two ways: a command-line call to
the script `pycart` or loading an instance of the :class:`pyCart.cart3d.Cart3d`
class.  In both cases, the default name of the file is :file:`pyCart.json`, but
it is possible to use other file names.  The following two examples show the
status of the run matrix; the first load the inputs from :file:`pyCart.json`,
and the second loads the inputs from :file:`run/poweron.json`.

    .. code-block:: bash
    
        $ pycart -c
        $ pycart -f run/poweron.json -c
        
Within a Python script, the settings can be loaded with the following code.

    .. code-block:: python
    
        import pyCart
        
        # Loads pyCart.json
        c1 = pyCart.Cart3d()
        # Loads run/poweron.json
        c2 = pyCart.Cart3d('run/poweron.json')

The location from which either of these two methods is called (i.e., the current
working directory) is remembered as the root directory for the run.  Locations
of other files are relative to this directory.
        
pyCart uses a "run sequence," which causes Cart3D to be run in multiple parts
where each part uses different settings. Here are a number of reasons that a
user may want to do this.

    #. You want to run unsteady analysis but want to get an adapted mesh first. 
       This requires running `aero.csh` and then running `flowCart` manually.
    #. The desire is to run 6 adaptation cycles, but running the first three in
       first-order mode will save time and increase robustness.
    #.  Some cases are particularly difficult to start, and it can be
        advantageous to run these with a more robust set of inputs for some
        number of iterations before continuing to the desired inputs.
        
The way that a user specifies these run sequences is by making some of the
input keys in :file:`pyCart.json` lists instead of single entries.  Two keys
in the "RunControl" section are mandatory.

    .. code-block:: javascript
    
        "RunControl": {
            "PhaseSequence": [0, 1, 2],
            "PhaseIters": [0, 0, 1500]
        }

This tells pyCart to run Cart3D with input set 0 once, then with input set 1
once, and finally with input set 2 until at least 1500 total iterations have
been run.  To define these different input sets, the user can change *any*
other setting to a list.  However, it is *not* required that the user must
make *all* settings into a list.  Furthermore, the lists do not have to a
length of at least 3; if the input sequence number is greater than the length
of the list, the last element of the list is used.  Consider the following more
complete example.

    .. code-block:: javascript
    
        "RunSequence": {
            "InputSeq": [0, 1, 2],
            "IterSeq": [0, 0, 2000],
            "Adaptive": [true, true, false],
            "flowCart": {
                "it_fc": 500,
                "first_order": [true, false],
                "unsteady": [false, false, true]
            },
        
            "Adaptation": {
                "n_adapt_cycles": [3, 5]
            }
        },
        
There are three input sequences (``0``, ``1``, and ``2``), which can be
described as follows.

    0. Run three adaptation cycles in first-order mode
    1. Run two more adaptation cycles in second-order mode
    2. Run more iterations of `flowCart` without adaptation but in
       time-accurate (unsteady) mode.
       
The fact that ``"first_order": [true, false]`` has only two entries means that
the last run will just use the last value, ``false``, as the option.
Similarly, since *it_fc* is not specified as a list, the same value of ``500``
is used for all three runs.


