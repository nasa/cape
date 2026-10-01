# Analysis objective

Describe what this analysis is intended to determine.


# Procedure

Describe anything special about the run procedure you want the agent to follow.
The agent will already know how to do basic CAPE procedures, but if you have
any special instructions for handling errors, what settings can be altered,
etc., place them here.

# Completion criteria

Describe what constitutes a successful analysis that can be approved.

Examples:
- Converge to a steady state up to 15000 iterations for every case
- Accept a limit cycle oscillation with steady or shrinking amplitude for cases
  with unsteady/time-accurate inputs **only**; do not go beyond 20,000
  iterations.
- Accept the recommendations of `cape get-col-state` for each reported
  subfigure; approve cases that are close after 20,000 iterations and use
  30,000 iterations as a hard cutoff.
