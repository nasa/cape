#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Run ``pytest`` and monitor results

:Versions:
    * 2021-10-15 ``@ddalle``: Version 1.0
"""

# Standard library
import os
import sys
import socket

# Third-party
import testutils
from testutils.sphinxreport import JUnitXMLReport


# Folder containing this document
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Python version
PYTHON_VERSION = "%i.%i" % (sys.version_info.major, sys.version_info.minor)

# Pytest files
COVERAGE_DIR = os.path.join("test", "htmlcov")
JUNIT_FILE = os.path.join("test", "junit.xml")
TEST_DOCDIR = os.path.join("doc", "test")
LAST_COMMIT_FILE = os.path.join("test", "last-commit-%s" % PYTHON_VERSION)


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
    # Name of test file to write
    test_index_file = "index-%s.rst" % PYTHON_VERSION.replace(".", "-")
    # Get host name
    hostname = socket.gethostname()
    # Don't run on linux281 or pfe
    if hostname == "linux281.nas.nasa.gov" or hostname.startswith("pfe"):
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
    if os.path.isfile(LAST_COMMIT_FILE):
        with open(LAST_COMMIT_FILE, "r") as fp:
            sha1_last = fp.read()
    else:
        sha1_last = ''
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
    # Read test results
    report = JUnitXMLReport(JUNIT_FILE)
    # Manually set the package name
    report.pkg = "cape"
    # Write report
    report.write_rst(toctree=False, fname=test_index_file)
    # Extract results
    testsuite, = report.tree.findall("testsuite")
    # Count tests, failures, and errors
    ntest = int(testsuite.attrib["tests"])
    nerr = int(testsuite.attrib["errors"])
    nfail = int(testsuite.attrib["failures"])
    # Initialize commit message
    msg = "Auto-commit Python %s test results:" % PYTHON_VERSION
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
    # Total number of tests
    msg += "\n\nTotal tests run: %i" % ntest
    # Add test results
    os.system("git add %s" % TEST_DOCDIR)
    os.system("git commit -m '%s'" % msg)
    # Get current SHA-1
    sha1_new, _, _ = testutils.call_o(["git", "rev-parse", "HEAD"])
    # Write commit
    with open(LAST_COMMIT_FILE, "w") as fp:
        fp.write(sha1_new)
    # Share results
    if ierr or ("push" in sys.argv):
        testutils.call(["git", "push", "hub-ssh", f"{branch}:{branch}"])
    # Return to original folder
    os.chdir(fpwd)
    # Exit status
    return ierr


# Test if called as script
if __name__ == "__main__":
    set.exit(main())

