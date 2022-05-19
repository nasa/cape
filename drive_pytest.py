#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Run ``pytest`` and monitor results

:Versions:
    * 2021-10-15 ``@ddalle``: Version 1.0
"""

# Standard library
import os
import socket
import sys

# Third-party
import testutils
from testutils.sphinxreport import JUnitXMLReport


# Folder containing this document
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Pytest files
COVERAGE_DIR = os.path.join("test", "htmlcov")
JUNIT_FILE = os.path.join("test", "junit.xml")
TEST_DOCDIR = os.path.join("doc", "test")
LAST_COMMIT_FILE = os.path.join("test", "last-commit")


# API version
def main():
    r"""Run crawler in all folders and accumulate results

    :Call:
        >>> stats = crawl()
    :Outputs:
        *stats*: :class:`dict`\ [:class:`dict`\ [:class:`int`]]
            Dictionary of tests results, e.g ``stats["cape"]["PASS"]``
    :Versions:
        * 2021-10-15 ``@ddalle``: Version 1.0
    """
    # Only run on linux281
    if socket.gethostname() != "linux281.nas.nasa.gov":
        return
    # Remember current location
    fpwd = os.getcwd()
    # Enter test folder
    os.chdir(THIS_DIR)
    # Find current branch
    branch, _, _ = testutils.call_o(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"])
    branch = branch.strip()
    # Get up to date
    testutils.call(["git", "pull", "hub", branch])
    # Test current commit
    sha1, _, _ = testutils.call_o(["git", "rev-parse", "HEAD"])
    # Read last commit
    with open(LAST_COMMIT_FILE, "r") as fp:
        sha1_last = fp.read()
    # Test if test necessary
    if sha1.strip() == sha1_last.strip():
        return
    # Form test command (for pytest)
    cmdlist = [
        "python3", "-m", "pytest",
        "--ignore-glob", "test/[a-z]*",
        "--junitxml=%s" % JUNIT_FILE,
        "--cov=cape",
        "--cov-report", "html:%s" % COVERAGE_DIR
    ]
    # Execute the tests
    ierr = testutils.call(cmdlist)
    # Track the coverage report
    os.remove(os.path.join(COVERAGE_DIR, ".gitignore"))
    # Read test results
    report = JUnitXMLReport(JUNIT_FILE)
    # Write report
    report.write_rst()
    # Extract results
    testsuite, = report.tree.findall("testsuite")
    # Count tests, failures, and errors
    ntest = int(testsuite.attrib["tests"])
    nerr = int(testsuite.attrib["errors"])
    nfail = int(testsuite.attrib["failures"])
    # Initialize commit message
    msg = "Auto-commit of all tests:"
    # Write commit message
    if nerr or nfail:
        # Count up the failures
        if nerr:
            msg += (" %i ERRORS" % nerr)
        if nfail:
            # Check for both
            if nerr:
                msg += " and"
            # Add failure count
            msg += (" %i FAILURES" % nfail)
    else:
        # PASS all
        msg += " PASS"
    # Add test results
    os.system("git add %s" % TEST_DOCDIR)
    os.system("git add %s" % COVERAGE_DIR)
    os.system("git commit -m '%s'" % msg)
    # Find current branch
    branch, _, _ = testutils.call_o(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"])
    # Share results
    testutils.call(["git", "push", "hub", branch])
    # Get current SHA-1
    sha1_new, _, _ = testutils.call_o(["git", "rev-parse", "HEAD"])
    # Write commit
    with open(os.path.join("test", "last-commit"), "w") as fp:
        fp.write(sha1_new)
    # Return to original folder
    os.chdir(fpwd)


# Test if called as script
if __name__ == "__main__":
    main()

