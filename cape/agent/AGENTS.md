# CAPE

CAPE manages CFD analysis using a run matrix and a main JSON file with a
variety of commands and API functions accessible from the ``cape`` executable
and ``cape`` module.

If the user has not specified which JSON file to work with run

    cape find-json

to find the valid candidates, listing solver module and file name. Other JSON
files in the project may be either "included" JSON files or have syntax errors
causing `cape find-json` to miss them.

The standard procedure for CAPE commands is roughly:

* Run `cape wait`: waits until some number of cases require action
* Take action
    - Submit more cases
    - Fix errors
    - Evaluate cases marked `DONE`
        * A human method is to run `cape report`, which generates a PDF
        * An agent can look at each subfigure:

            cape list-report-subfigs
            cape open subfig {SUBFIG_NAME} -I {CASE}

          Ignore text-only subfigures

        * `cape get-case-state` provides a detailed recommendation if other
          methods fail or the decision is ambiguous.

    - Extend cases if not converged
    - Approve and extract data

The `cape wait` command, and also `cape check`, return a status for each case
in the run matrix, which will be one of these values.

* `---` means the case has not been started or set up yet.
* `INCOMP` means the case is set up but has not completed the minimum
  required iterations and is not running.
* `QUEUE` is an `INCOMP` case that has a PBS/Slurm job currently in the queue.
* `RUNNING` means the case is currently running (in progress).
* `ZOMBIE` means the case appears to be running but has not had any recent
  updates.
* `FAIL`: The case encountered a failure while attempting to run CFD.
* `ERROR`: The user has marked this case a failure, and the status is final.
* `DONE` means the case has completed all required iterations and phases
   and is awaiting disposition by the user or agent.
* `PASS`: The case is `DONE` and marked as final by the user.
* `PASS*`: The case is marked `PASS` by the user but does not meet the
  requirements for `DONE`.
  
## Rules

* **Strongly** prefer CAPE's CLI and Python APIs or manually modifying
  generated cases. Frequent commits to template files or the CAPE JSON file are
  traceable whereas manual edits to individual cases' input files are not.
* The project-local `AGENTS.md` and `ANALYSIS.md` are authoritative
* Use `cape -h -v` and command-specific help like `cape check -h` to discover
  current interfaces.
* You can edit the JSON file to modify the `wait` behavior, improve reporting,
  fix bugs/errors, and extract missing data, but do not change the CFD approach
  without consulting the user.
* Create and commit small run matrices before operating on the full run matrix
  (if large). For example `run/poweron01.json`, `run/poweron02.json`,
  `run/poweron03.json` is a perfectly acceptable development trace. Delete or
  suggest deletion of outright failure CFD runs.

An excellent way to learn what an arbitrary JSON option does is to run

cape inspect-json .RunControl.PhaseIters [-f JSONFILE]

It uses jq-like syntax and follows a max-depth option.



