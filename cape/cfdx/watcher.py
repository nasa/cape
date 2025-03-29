r"""
:mod:`cape.cfdx.watcher`: Watcher utilities for CAPE
=====================================================

This module provides the class :class:`CaseWatcher` that is used to
provide runtime "watcjer" function calling for individual cases.
"""

# Standard library modules
import multiprocessing
import os
import time
from typing import Optional

# Third-party

# Local imports

WATCHDIR = "cape"
WAITTIME = 1
TIMELIMIT = 3600


# Base watcher class
class BaseWatcher(object):
    r"""Template watcher class

    :Call:
        >>> watcher = BaseWatcher()
    :Inputs:
        *rootdir*: {``None``} | :class:`str`
            Absolute path to root folder of case/run matrix
    :Outputs:
        *watcher*: :class:`BaseWatcher`
            Watcher instance for one case
    """
   # --- Class attributes ---
    # Instance attributes
    __slots__ = (
        "defns",
        "root_dir",
        "names",
        "processes"
    )

    _watchdir = WATCHDIR

    # Initialization
    def __init__(self, rootdir: Optional[str] = None):
        r"""Initialization method

        :Versions:
            * 2025-03-03 ``@aburkhea``: v1.0
        """
        # Check for default
        if rootdir is None:
            # Save current path
            self.root_dir = os.getcwd()
        elif isinstance(rootdir, str):
            # Save absolute path
            self.root_dir = os.path.abspath(rootdir)
        else:
            # Bad type
            raise TypeError(
                "Watcher *rootdir*: expected 'str' " +
                f"but got '{type(rootdir).__name}'")
        # Initialize process handles
        self.processes = []
        self.names = []
        self.defns = {}

    # Add process
    def add_process(self, pname, process, *a, **kw):
        r"""Add named process to watcher

        :Versions:
            * 2025-03-03 ``@aburkhea``: v1.0
        """
        # Form process
        mprocess = multiprocessing.Process(
            target=process,
            args=a,
            kwargs=kw)
        # Save to processes
        self.processes.append((pname, mprocess))
        # Record process if new defn
        if pname not in self.defns.keys():
            self.defns[pname] = {
                "proc": process,
                "args": a,
                "kwargs": kw
            }

    def refresh_processes(self):
        r"""Refresh processes to allow for re-run

        :Versions:
            * 2025-03-04 ``@aburkhea``: v1.0
        """
        _processes = self.processes[:]
        # For each process
        for i, (name, proc) in enumerate(_processes):
            # Ensure process is closed
            proc.close()
            # Get definitions of process
            defnsj = self.defns[name]
            # Remove old process
            _ = self.processes.pop(i)
            # Add new copy of process
            self.add_process(
                name,
                defnsj["proc"],
                *defnsj["args"],
                **defnsj["kwargs"])

    # Run processes
    def run_processes(self):
        r"""Run all processes in watcher

        :Versions:
            * 2025-03-03 ``@aburkhea``: v1.0
        """
        for name, process in self.processes:
            self.run_process(name, process)

    # Run individual process
    def run_process(self, name, process):
        r"""Run one processes in watcher

        :Versions:
            * 2025-03-03 ``@aburkhea``: v1.0
        """
        # Start process
        process.start()
        process.join()
        # Return exit code
        return process.exitcode

    # Run process group
    def run_process_group(self):
        r"""Run all processes in watcher as joined

        :Versions:
            * 2025-03-03 ``@aburkhea``: v1.0
        """
        # Check if running already
        for _, process in self.processes:
            if process.is_alive():
                continue
            else:
                process.start()
                process.join()

    # Run time-limited individual process
    def run_timed_process(self, name, process):
        r"""Run one processes in watcher with time limit *TIMELIMIT*

        :Versions:
            * 2025-03-03 ``@aburkhea``: v1.0
        """
        # Start process
        process.start()
        # Get start time
        tstart = 0
        # Sleep while process running
        while process.is_alive():
            time.sleep(WAITTIME)
            # Cumulative time
            tstart += WAITTIME
            if tstart > TIMELIMIT:
                # Kill
                process.kill()
                # Force close
                process.close()
        # Return exit code
        return process.exitcode


# Case watcher class
class CaseWatcher(BaseWatcher):
    r"""Watcher for an individual CAPE case

    :Call:
        >>> watcher = CaseWatcher(rootdir)
    :Inputs:
        *rootdir*: {``None``} | :class:`str`
            Absolute path to root folder of case
    :Outputs:
        *watcher*: :class:`CaseWatcher`
            Watcher instance for one case
    :Versions:
        * 2025-03-03 ``@aburkhea``: v1.0
    """
   # --- Class attributes ---
    # Instance attributes
    __slots__ = (
        "watched_process",
    )

   # --- __dunder__ ---

    def set_watched_process(self, wname, process, *a, **kw):
        r"""Set main watched process

        :Versions:
            * 2025-03-03 ``@aburkhea``: v1.0
        """
        # Initialize run_phase process
        wprocess = multiprocessing.Process(
            target=process,
            name=wname,
            args=a,
            kwargs=kw)
        # Set watched process
        self.watched_process = ((wname, wprocess))

    def run_watched_process(self, wname, wproc):
        r"""Run watched process, and periodic watcher processes

        :Versions:
            * 2025-03-03 ``@aburkhea``: v1.0
        """
        # Start proc
        wproc.start()
        # Sleep a little before first alive check
        time.sleep(1)
        cnt = 0
        # While process running call watchers every *WAITTIME* sec
        while wproc.is_alive():
            for name, proc in self.processes:
                self.run_process(name, proc)
            # Sleep
            time.sleep(WAITTIME)
            self.refresh_processes()

        # Check for successful watched process exitcode
        if wproc.exitcode == 0:
            # Call watchers final time
            for name, process in self.processes:
                self.run_process(name, process)
            time.sleep(WAITTIME)
        # Return watched process exit code
        return wproc.exitcode

    def run_processes(self):
        r"""Run all processes with watched process

        :Versions:
            * 2025-03-03 ``@aburkhea``: v1.0
        """
        # Start watched process
        wname = self.watched_process[0]
        wproc = self.watched_process[1]
        wextcode = self.run_watched_process(wname, wproc)
        print(wextcode)
        # If failed watched process, raise error
        if wextcode != 0:
            raise SystemError(f"Watch process {wname} failed")
