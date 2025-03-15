
.. _configjson-syntax:

Syntax for ConfigJSON Files
============================

Configuration JSON files are a CAPE-specific file using the
`JSON <https://www.json.org>`_ format to describe a surface mesh configuration.
Its main goal is to define two things:

*   Families of components, e.g. that ``"wings"`` includes ``"wing_left"`` and
    ``"wing_right"``
*   Component ID and other properties for each component.

There are two sections of the ConfigJSON file, which in CAPE is read and
interfaced by the class :class:`cape.config.ConfigJSON`. The first section is
the ``"Tree"`` which defines families of components, and the second section is
``"Properties"``, which optionally can be used to define  various properties
for each component or family.

Here's an example file:

    .. code-block:: javascript

        {
            "Tree": {
                "entire": [
                    "arrow_no_base",
                    "base"
                ],
                "arrow_no_base": [
                    "fuselage",
                    "fins"
                ],
                "fuselage": [
                    "cap",
                    "body"
                ],
                "fins": [
                    "fin1",
                    "fin2",
                    "fin3",
                    "fin4"
                ]
            },
            "Properties": {
                "farfield": {
                    "CompID": 1,
                    "aflr3_bc": 1,
                    "fun3d_bc": 5050
                },
                "entire": {
                    "aflr3_bc": -1,
                    "blds": 5.0e-4
                },
                "cap": {
                    "CompID": 2
                },
                "body": {
                    "CompID": 3
                },
                "fin1": {
                    "CompID": 11
                },
                "fin2": {
                    "CompID": 12
                },
                "fin3": {
                    "CompID": 13
                },
                "fin4": {
                    "CompID": 14
                },
                "base": {
                    "CompID": 20
                }
            }
        }
