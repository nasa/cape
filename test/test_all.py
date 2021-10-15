#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Run test crawler in all folders and report results

:Versions:
    * 2021-10-15 ``@ddalle``: Version 1.0
"""

# Standard library
import os
import socket
import sys

# CAPE modules
from cape.testutils import crawler


# Folder containing this document
THIS_DIR = os.path.dirname(__file__)
# Parent folder
CAPE_DIR = os.path.dirname(THIS_DIR)


# API version
def crawl():
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
    # Get up to date
    os.system("git pull hub main")
    # Start container
    stats = {}
    # Keep track of failures
    fail_n = 0
    fail_fams = []
    # Loop through test families
    for family in os.listdir("."):
        # Only folders
        if not os.path.isdir(family):
            continue
        # Enter folder
        os.chidr(os.path.join(THIS_DIR, family))
        # Run the tests
        stats[family] = crawler.cli()
        # Add up failures
        if stats[family]["FAIL"]:
            # Add to counter
            fail_n += stats[family]["FAIL"]
            # Track family
            fail_fams.append(family)
    # Go to CAPE root folder
    os.chdir(CAPE_DIR)
    # Write commit message
    if fail_n:
        # Count up the failures
        msg = "Auto-commit of all tests: %i FAILURES\n\n"
        msg += "Failures occurred in these folders:\n"
        # Loop through families
        for family, stats_fam in stats.items():
            # Get failure count
            n_fam = stats_fam.get("FAIL", 0)
            # Report if any
            if n_fam:
                msg += "    %s: %i\n" % (family, n_fam)
    else:
        # PASS all
        msg = "Auto-commit of all tests: PASS"
    # Add test results
    os.system("git add test")
    os.system("git commit -a -m '%s'" % msg)
    # Share results
    #os.system("git push hub main")
    # Return to original folder
    os.chdir(fpwd)
    # Output
    return stats


# Main loop
def main():
    r"""Run crawler in all folders and accumulate results

    :Versions:
        * 2021-10-15 ``@ddalle``: Version 1.0
    """
    crawl()


# Test if called as script
if __name__ == "__main__":
    main()

