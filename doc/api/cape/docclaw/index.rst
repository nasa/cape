***************************************************
:mod:`cape.docclaw`: Small-corpus document retrieval
***************************************************

.. automodule:: cape.docclaw
    :members:

DocClaw provides bounded search over solver user manuals (a super-powered
grep, not an answer generator) using SQLite FTS5 lexical search, exact
vector search, and reciprocal-rank fusion. CAPE ships prebuilt corpus
databases under the package ``docs/`` folder, one corpus per solver.
Search from the command line with:

.. code-block:: console

    $ python3 -m cape.docclaw --root "$(python3 -c \
        'import cape, os; print(os.path.dirname(cape.__file__))')/docs" \
        search 'turbulence models' --corpus fun3d

All CLI responses are JSON. The ``DOCCLAW_ROOT`` environment variable can
replace ``--root``.

.. toctree::
    :maxdepth: 1

    api
    chunking
    cli
    corpus
    embeddings
    models
