#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
``gitutils``: Python interface to git repositories
=================================================================

This package provides the class :class:`GitRepo`. Instances of this
class can run common git commands on the repository from which they
were initiated and perform other git-related actions such as modifying
the configuration.

To instantiate a repo in the current folder, simply run

.. code-block:: python

    from gitutils import GitRepo

    repo = GitRepo()
"""

# Local imports
from .gitrepo import GitRepo
