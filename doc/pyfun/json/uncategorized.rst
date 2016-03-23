
.. _pyfun-json-uncategorized:

---------------------------
Uncategorized pyFun Options
---------------------------

Several inputs in :file:`pyFun.json` do not go in any category.  They are
relatively simple inputs, and many of them have usable defaults, although it's
still a good idea to enter them into the pyFun control file to avoid ambiguity
and improve traceability.  Additional miscellaneous options can be found in
:ref:`the corresponding Cape JSON section <cape-json-uncategorized>`.

Specific FUN3D Input File Templates
===================================

All of the options in :ref:`the Cape JSON description <cape-json-uncategorized>`
also apply, and there are only a few additional options, which are described
here.  The JSON syntax is shown below with the default values.

    .. code-block:: javascript
    
        "Namelist": "fun3d.nml"
        
This is the name of the FUN3D Fortran namelist file.  Before starting a case,
pyFun will copy this file to the case folder and edit it appropriately, but it
is often useful to set universal settings that will apply to each case directly
in the namelist template as one would when not using a tool like pyFun.

        
Miscellaneous Options Dictionary
================================

The full list of miscellaneous options specific to pyCart and their allowed
values are listed below.  These options are in addition to the options listed in
the :ref:`Miscellaneous Cape Options <cape-json-misc-dict>` section.

    *Namelist*: {``"fun3d.nml"``} | :class:`str`
        Name of FUN3D namelist template

