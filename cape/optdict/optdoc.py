r"""
``optdict.optdoc``: Documentation writers for ``OptionsDict`` subclasses
========================================================================

This module provides functions to write reStructuredText (``.rst``)
files listing all available information for a given subclass of
:class:`OptionsDict`.
"""

# Standard library


# Sequence of title markers for reST, (character, overline option)
RST_SECTION_CHARS = (
    ("-", True),
    ("=", False),
    ("-", False),
    ("^", False),
    ("*", False),
)
