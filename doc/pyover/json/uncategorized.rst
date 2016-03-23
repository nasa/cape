
.. _pyover-json-uncategorized:

----------------------------
Uncategorized pyOver Options
----------------------------

Several inputs in :file:`pyOver.json` do not go in any category.  They are
relatively simple inputs, and many of them have usable defaults, although it's
still a good idea to enter them into the pyOver control file to avoid ambiguity
and improve traceability.  Additional miscellaneous options can be found in
:ref:`the corresponding Cape JSON section <cape-json-uncategorized>`.

Specific OVERFLOW Input File Templates
======================================

All of the options in :ref:`the Cape JSON description <cape-json-uncategorized>`
also apply, and there are only a few additional options, which are described
here.  The JSON syntax is shown below with the default values.

    .. code-block:: javascript
    
        "OverNamelist": "overflow.inp"
        
This is the name of the OVERFLOW Fortran namelist file.  Before starting a case,
pyOver will copy this file to the case folder and edit it appropriately, but it
is often useful to set universal settings that will apply to each case directly
in the namelist template as one would when not using a tool like pyOver.

        
Miscellaneous Options Dictionary
================================

The full list of miscellaneous options specific to pyCart and their allowed
values are listed below.  These options are in addition to the options listed in
the :ref:`Miscellaneous Cape Options <cape-json-misc-dict>` section.

    *OverNamelist*: {``"overflow.inp"``} | :class:`str`
        Name of OVERFLOW namelist template

