

----------------------------
Uncategorized pyCart Options
----------------------------

Several inputs in :file:`pyCart.json` do not go in any category.  They are
relatively simple inputs, and many of them have usable defaults, although it's
still a good idea to enter them into the pyCart control file to avoid ambiguity
and improve traceability.

Specific Cart3D Input File Templates
====================================

All of the options in :ref:`the Cape JSON description <cape-json-uncategorized>`
also apply, and there are only a few additional options, which are described
here.  The JSON syntaxis shown below with the default values.

    .. code-block:: javascript
    
        "InputCntl": "input.cntl",
        "AeroCsh": "aero.csh"
        
These are the names and locations of two *template* files required for running
Cart3D.  In the case of :file:`aero.csh`, it is only required if it is an
adaptive run.  It is sometimes useful to use template files with different
names, for example if multiple sets of inputs are being tested.  A recommended
practice for very involved Cart3D analyses is to put input files into a folder. 
Consider the following partial file tree.

    .. code-block:: none
    
        run/
            poweroff.json
            poweron.json
        inputs/
            input.cntl
            aero.csh
            Config.xml
            Components.i.tri
            
In this case, there are two pyCart control files, :file:`poweroff.json` and
:file:`poweron.json`, and they should contain the following.

    .. code-block:: javascript
    
        "InputCntl": "inputs/input.cntl",
        "AeroCsh": "inputs/aero.csh"
        
Obviously, there are a couple more files shown in the partial file tree above,
and these will come into play in other sections.  Then the call to pyCart will
be

    .. code-block:: bash
    
        $ pycart -f run/poweroff.json
        $ pycart -f run/poweron.json
        
for the two input files.

For both of these two files, pyCart provides its own template if the file cannot
befound in the local directory tree.

        
Miscellaneous Options Dictionary
================================

The full list of miscellaneous options specific to pyCart and their allowed
values are listed below.  These options are in addition to the options listed in
the :ref:`Miscellaneous Cape Options <cape-json-misc-dict>` section.

    *InputCntl*: {``"input.cntl"``} | :class:`str`
        Name of Cart3D input file template
 
    *AeroCsh*: {``"aero.csh"``} | :class:`str`
        Adaptive run script template

