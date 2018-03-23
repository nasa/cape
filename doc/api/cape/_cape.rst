
Compiled Cape functions
=======================

The :mod:`cape._cape` module contains methods that were written in C to
increase speed and efficiency. The most commonly utilized methods enable Cape,
pyCart, etc. to write surface triangulations faster. Because these are simple
tasks that may need to be repeated many times, creating a compiled version can
save considerable time. However, Python versions of these methods are also
included in the main modules.

.. automodule:: cape._cape
    :members:
