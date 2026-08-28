r"""
:mod:`cape.agent.agentutils`: Extra utilities for agentic interface
=====================================================================

This module provides tools used for the main CAPE agentic interface
that would unnecessarily clutter the main :mod:`cape_agentic.agents`
source code.
"""

# Standard library
import json
import shutil
import sys
import threading
import time

# Third-party
import numpy as np

# Local imports
from ..ui.promptutils import sprintf_color


# Spinner class
class ThinkingSpinner:
    """Display an animated spinner while LLM is thinking"""
    def __init__(self, message: str = "Thinking"):
        self.message = message
        self.spinner_chars = [
            "\\", "|", "/", "−",
        ]
        self._stop_event = threading.Event()
        self._thread = None
        # Save current time
        self.tic = time.time()

    def __enter__(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        # Get timer
        dt = time.time() - self.tic
        # Final message
        msg = sprintf_color(
            f"{self.message}\nThought for {float(dt):.1f} seconds\n", "purple")
        # Length of current line
        linelen = shutil.get_terminal_size().columns - 1
        # Clear the line
        sys.stdout.write("\r" + " " * (linelen) + "\r")
        sys.stdout.write(msg)
        sys.stdout.flush()

    def _spin(self):
        idx = 0
        while not self._stop_event.is_set():
            msg = sprintf_color(
                f"{self.message} {self.spinner_chars[idx]}", "purple")
            sys.stdout.write(f"\r{msg}")
            sys.stdout.flush()
            idx = (idx + 1) % len(self.spinner_chars)
            time.sleep(0.25)


# Customize JSON serializer
class _NPEncoder(json.JSONEncoder):
    r"""Encoder for :mod:`json` that can handle NumPy objects"""
    def default(self, obj):
        # Check for array
        if isinstance(obj, np.ndarray):
            # Check for scalar
            if obj.ndim > 0:
                # Convert to list
                return list(obj)
            elif np.issubdtype(obj.dtype, np.integer):
                # Convert to integer
                return int(obj)
            else:
                # Convert to float
                return float(obj)
        elif isinstance(obj, np.integer):
            # Convert to integer
            return int(obj)
        elif isinstance(obj, np.floating):
            # Convert to float
            return float(obj)
        # Otherwise use the default
        return super().default(obj)
