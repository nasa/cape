#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System interface and other standard modules
import os
import re
import subprocess as sp

# Local modules
from .. import typeutils


# Convert a results.txt file
def convert_doctest_results(fdoctest, fdoc=None):
    """Convert ``results.txt`` file from Sphinx ``doctest`` to reST
    
    In general this document will convert
    
        * ``_build/doctest/output.txt``  ==> ``results
    
    :Call:
        >>> convert_doctest_results(fdoctest, fdoc=None)
    :Inputs:
        *fdoctest*: :class:`str`
            Folder to ``doctest`` results
        *fdoc*: {``None``} | :class:`str`
            Sphinx documentation root dir
            (defaults to ``../..`` from *fdoctest*)
    :Versions:
        * 2018-03-27 ``@ddalle``: First version
        * 2019-04-05 ``@ddalle``: Forked from :mod:`cape.test`
    """
    # Test type of input
    if not typeutils.isstr(fdoctest):
        raise TypeError("doctest folder must be a string")
    # Get name of file
    ftxt = os.path.join(fdoctest, "output.txt")
    # Check if file exists
    if not os.path.isfile(ftxt):
        print("  No doctest file 'output.txt' file to interpret!")
        return
    # Get documentation folder
    if fdoc is None:
        # Use ../.. from of *fdoctest*
        fdoc = os.path.abspath(os.path.join(fdoctest, "..", ".."))
    # Converted file
    frst = os.path.join(fdoc, "test", "results.rst")
    # Open input and create output
    fi = open(ftxt, 'r')
    fo = open(frst, 'w')
    # Start output file
    fo.write("\n.. _test-results:\n\n")
    fo.write("Results of doctest builder\n")
    fo.write("==========================\n\n")
    # Read the input file
    lines = fi.readlines()
    # Close input file
    fi.close()
    # Split to dates
    tdate, ttime = lines[0].strip().split()[-2:]
    # Write compiled date and time
    fo.write("This test was run on %s at time %s.\n\n" % (tdate, ttime))
    # Write the summary
    fo.write("**Summary**:\n")
    # Write the last four lines (not counting empty line)
    for i in range(-4, 0):
        # Write with a bullet
        fo.write("    * %s" % lines[i])
    # Blank line
    fo.write("\n\n")
    # Start looping through documents
    i = 3
    # Test following line
    while lines[i].startswith("Document"):
        # Process the line
        i = convert_doctest_document(lines, fo, i)
    
    # Close output file
    fo.close()


# Interpret one result
def convert_doctest_document(lines, fo, i):
    """Convert the results of one document test
    
    :Call:
        >>> j = convert_test_document(lines, fo, i)
    :Inputs:
        *lines*: :class:`list` (:class:`str`)
            Lines from input file
        *fo*: :class:`file`
            Output file of interpreted reST, open ``'w'``
        *i*: :class:`int`
            Index of starting line for this document
    :Outputs:
        *j*: :class:`int`
            Index of starting line for next content
    :Versions:
        * 2018-03-27 ``@ddalle``: First version
    """
    # Get the name of the doucment
    txt, fdoc = lines[i].strip().split()
    # Make sure this is a document test
    if not txt.startswith("Document:"):
        return
    # Create section header
    txt = "Document: :doc:`%s </%s>`" % (fdoc, fdoc)
    # Write it
    fo.write(txt + "\n")
    fo.write("-"*len(txt) + "\n")
    # Discard header underlines
    i += 2
    # Loop through failed tests
    while lines[i].startswith("******"):
        # Read the next line
        i += 1
        # Check for summary line (dealing with extra '*******' line)
        if re.match("[0-9]+ items", lines[i]):
            break
        # Write failure line summary (and extra blank line)
        fo.write("Failure in f%s\n" % lines[i][1:])
        # Go to next line
        i += 1
        # Loop until next '******' line found
        while not lines[i].startswith("******"):
            # Read key (strip colon at end)
            hdr = lines[i].strip()[:-1]
            # Write the header
            fo.write("**%s**:\n\n" % hdr)
            # Check type
            if hdr == "Failed example":
                # Write a Python code block
                fo.write("    .. code-block:: python\n\n")
            elif hdr == "Exception raised":
                # Write a Python Traceback code block
                fo.write("    .. code-block:: pytb\n\n")
            else:
                # Write raw text
                fo.write("    .. code-block:: none\n\n")
            # Loop until unindent
            while (not lines[i+1].strip()) or lines[i+1].startswith("    "):
                # Go to next line
                i += 1
                # Write the content
                fo.write("    %s" % lines[i])
            # Clean up
            fo.write("\n")
            # Go to next line
            i += 1
    # Process summary
    fo.write("**Document summary**:\n\n")
    # Set previous indent
    indent = 0
    # Loop through lines
    while lines[i].strip():
        # Check if line has a summary
        if lines[i].startswith("   "):
            # Indent
            indent = 4
            # Write line
            fo.write("      - %s" % lines[i].lstrip())
        else:
            # Check for previous indent
            if indent == 4:
                fo.write("\n")
            # Unset indeint
            indent = 0
            # Write line
            fo.write("  * %s" % ('*'.join(lines[i].split("***"))))
        # Move to next line
        i += 1
    # Clear some space
    fo.write("\n\n")
    # End; skip blank line
    return i + 1
