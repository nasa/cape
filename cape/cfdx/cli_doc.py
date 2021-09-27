#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
:mod:`cape.cfdx.cli_doc`: Template help messages for CAPE executables
======================================================================

This module provides a template docstring *template* that can be used as
a template for each of the executables (e.g. ``pyfun`` and ``pycart``).

This is the template for what users see when they execute

    .. code-block:: console

        $ pyfun -h

or

    .. code-block:: console

        $ pyover --help

:Versions:
    * 2014-10-06 ``@ddalle``: Version 1.0 (of ``pycart``)
    * 2015-10-16 ``@ddalle``: Version 1.1
    * 2020-09-11 ``@ddalle``: Version 2.0; separate doc module
"""

# Template docstrint
template = r"""
Python %(pyver)sinterface for %(solver)s: **%(cmd)s**
======================================================

This function provides a master interface for %(title)s.  All of the
functionality from this script is also accessible from the
%(mod)s module using relatively simple commands.

:Usage:
    .. code-block:: console

        $ %(cmd)s [OPTIONS]

:Examples:

    The basic call submits all (up to 10) jobs prescribed in the file
    ``%(title)s.json``

        .. code-block:: console

            $ %(cmd)s

    This command uses the inputs from ``poweron.json`` and only displays
    statuses.  No jobs are submitted.

        .. code-block:: console

            $ %(cmd)s -f poweron.json -c

    This command submits at most 5 jobs, but only cases with "Mach"
    greater than 1.5 and "alpha" equal to 0.0 are considered as
    candidates.

        .. code-block:: console

            $ %(cmd)s -n 5 --cons "Mach>0.5, alpha==0.0"

:Options:

    -h, --help
        Display this help message and quit

    -c
        Check status and don't submit new jobs

    -j
        Show the PBS job numbers as well

    -f FJSON
        Use %(name)s input file *FJSON* (default: ``"%(title)s.json"``)

    -x FPY
        Executes Python script *FPY* using ``execfile()`` prior to
        taking other actions; can be stacked using
        ``-x FPY1 -x FPY2 -x FPY3``, etc.

    -n NJOB
        Submit at most *NJOB* PBS scripts (default: 10)

    -q QUEUE
        Submit to a specific queue, overrides value in JSON file

    --kill, --qdel
         Remove jobs from the queue and stop them abruptly

    --cons CNS
        Only consider cases that pass a list of inequalities separated
        by commas.  Constraints must use variable names (not
        abbreviations) from the trajectory described in *FJSON*.

    -I INDS
        Specify a list of cases to consider directly by index

    --re REGEX
        Restrict to cases whose name matches regular expression *REGEX*

    --filter TXT
        Restrict to cases whose name contains the exact text *TXT*

    --glob TXT
        Restrict to cases whose name matches the glob *TXT*

    --report RP
        Update report named *RP* (default: first report in JSON file)

    --archive
        Create archive according to options in "Archive" section of
        *FJSON* and clean up run folder if case is marked PASS

    --clean
        Delete any files as described by *ProgressDeleteFiles* in
        "Archive" section of *FJSON*; can be run at any time

    --exec, -e CMD
        Execute command *CMD* in each folder

    --aero, --fm, --fm GLOB
        Loop through cases and extract force and moment coefficients and
        statistics for force & moment components described in the
        "DataBook" section of *FJSON*; only process components whose
        names wildcard *GLOB* if used

    --ll, --ll GLOB
        Loop through cases and extract force and moment coefficients and
        statistics for LineLoad components described in the "DataBook"
        section of *FJSON*; only process components whose names match
        wildcard *GLOB* if used

    --triqfm, --triqfm GLOB
        Loop through cases and extract force and moment coefficients and
        statistics for TriqFM components described in the "DataBook"
        section of *FJSON*; only process components whose names match
        wildcard *GLOB* if used

    --PASS
        Mark cases with "p" in run matrix to denote completion

    --ERROR
        Mark cases with "E" in run matrix to denote that they cannot run

    --unmark
        Remove PASS or ERROR markings from selected cases

    --unmarked
        Find only cases that are not marked PASS or ERROR

    --rm
        Delete a cases folder [``--no-prompt`` w/o interactive prompt]

    --apply
        Apply the settings in *FJSON* to all cases; way to quickly
        change settings for one or more runs

    --extend, --extend E
        Add another run of the current last phase *E* times (default: 1)

    --imax M
        Do not extend a case (when using --extend) beyond iteration *M*

    --no-start
        When running a command that would otherwise submit jobs, set
        them up but do not start (or submit) them

    --no-restart
        When submitting new jobs, only submit new cases (status '---')
"""


# Print help if appropriate.
if __name__ == "__main__":
    print(__doc__)
