
.. _pyover-json-mesh:

****************************
Options for ``Mesh`` Section
****************************
The options below are the available options in the ``Mesh`` Section of the ``pyover.json`` control file


*ConfigDir*: {``None``} | :class:`str`
    folder from which to copy/link mesh files



*Type*: ``'dcf'`` | ``'peg5'``
    overall meshing stragety



*LinkFiles*: {``['grid.in', 'xrays.in', 'fomo/grid.ibi', 'fomo/grid.nsf', 'fomo/grid.ptv', 'fomo/mixsur.fmp']``} | :class:`list`\ [:class:`str`]
    list of files to link into case folder



*CopyFiles*: {``[]``} | :class:`list`\ [:class:`str`]
    list of files to copy into case folder


