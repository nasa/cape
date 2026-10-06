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

    - Extend cases if not converged: `cape perform extend -I {CASES}`
    - Approve and extract data: `cape perform approve -I {CASES}`

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

## Case disposition: `cape perform`

Prefer `cape perform ACTION -I {CASES}` over the single-purpose commands.
`ACTION` is a named list of steps from the *Actions* section of the JSON
file. Built-in defaults exist for these names, and each can be redefined:

| Action     | Default steps                         | Instead of          |
|------------|---------------------------------------|---------------------|
| `approve`  | `MarkPASS`, then `update_dex` (all)   | `cape approve` + `cape extract` |
| `extend`   | `ExtendCases(qsub=True)`              | `cape extend`       |
| `extend2`  | `ExtendCases(qsub=True, extend=2)`    | `cape extend` x2    |
| `defail`   | `Defail`, then `SubmitJobs`           | `cape defail`       |
| `dezombie` | `Dezombie`, then `SubmitJobs`         | `cape dezombie`     |

In particular, use `cape perform approve` rather than `cape approve`: the
latter only marks cases `PASS` and leaves data unextracted, while the
project's `approve` action may also run project-specific post-processing.
Always pass `-I` (or other subset options); with no subset `cape perform`
acts on the whole run matrix. Check which actions exist and what each will
do first with

    cape perform --list [-f JSONFILE]

which shows every action's steps, marks built-in defaults `(default)`,
simultaneous steps `[index=N]`, and `AddFileName` steps `[+f]`.

### Action syntax

Each action name maps to one step or a list of steps. A step is either a
shell command string or a dict:

* `"type"`: `"shell"` (default) | `"cntl"` | `"cli"`
    - `shell`: run `"function"` as a shell command from the root folder;
      `{I}` is replaced by the case indices, otherwise `-I {CASES}` is
      appended
    - `cntl`: call method `"function"` of the `Cntl` instance with `I=...`
      (e.g. `MarkPASS`, `update_dex`, `UpdateLL`, `SubmitJobs`)
    - `cli`: call `cape.cfdx.cli.<function>` (e.g. `cape_extract_dex`), which
      re-reads the JSON file
* `"function"`: command, method, or function name (alias `"command"`)
* `"args"`, `"kwargs"`: extra positional/keyword args (`cntl`/`cli` only)
* `"AddFileName"`: append `-f {JSON}` (shell) or `f={JSON}` (cli) so the
  step uses the same JSON file as the `cape perform` call. Use this whenever
  the project has more than one JSON file or a script has a hard-coded
  default such as `pyFun.json`.
* `"index"`: steps run in order of `index` (default: list position); steps
  that share an `index` run **simultaneously** in forked processes

Steps are sequential and a failing step raises, so put state-changing steps
(e.g. `MarkPASS`) first and expensive extraction after.

### Adding custom actions (project tools)

Projects often have scripts in `tools/` (e.g. `tools/bumplabel.py`,
`tools/bumpphase.py`, `tools/bumpta.py`) that take `-f JSONFILE` and
`-I INDICES`, modify the run matrix or case settings for failed cases, and
resubmit. Expose these as named actions rather than calling them ad hoc:

    "Actions": {
        "UserTools": ["bumplabel", "bumpphase", "bumpta"],
        "bumplabel": {"function": "./tools/bumplabel.py", "AddFileName": true},
        "bumpphase": {"function": "./tools/bumpphase.py", "AddFileName": true},
        "bumpta": {"function": "./tools/bumpta.py --force", "AddFileName": true}
    }

Then `cape perform bumplabel -I 14,27` runs
`./tools/bumplabel.py -f run/x.json -I 14,27` from the root folder.
Listing names in `"UserTools"` also offers them as choices in `cape review`
and `cape dispatch`. Before wiring up a script, read it: confirm its option
names (`-I`, `-f`, `--force`), what statuses it skips, any built-in limits
(e.g. refusing more than 50 cases), and whether it submits jobs itself.
Custom actions can also chain steps, e.g. a retry action that runs a tool
then `{"type": "cntl", "function": "SubmitJobs"}`.

New tool scripts should follow the same pattern: accept `-f` and `-I`, use
`cntl.x.GetIndices(**kw)`, skip `PASS`/`ERROR`/running cases, print one line
per case, and commit the script with the JSON change.

### Parallel extraction in `approve`

The default `approve` extracts every DataBook component serially, which can
be slow when there are many components or expensive types (line loads,
TriqFM, surface CP). Redefine `approve` to split extraction into groups that
share an `index`:

    "Actions": {
        "approve": [
            {"type": "cntl", "function": "MarkPASS"},
            {"type": "cntl", "function": "update_dex", "index": 2,
             "kwargs": {"dex": "[A-L]*"}},
            {"type": "cntl", "function": "update_dex", "index": 2,
             "kwargs": {"dex": "[M-Z]*"}},
            {"type": "cntl", "function": "update_dex", "index": 2,
             "kwargs": {"dex": "LL_*"}},
            {"function": "./tools/writesurfcp.py", "index": 3,
             "AddFileName": true}
        ]
    }

Guidelines:

* List the components first (`cape inspect-json .DataBook.Components`)
  and choose `dex` glob patterns that are **disjoint** and together
  cover every component. Two parallel steps must never write the same DataBook
  component; overlapping or missing patterns are the main way to break this.
* Balance groups by cost (isolate slow components such as line loads into
  their own group) rather than by count; a handful of groups is enough.
* Keep `MarkPASS` (which rewrites the run matrix) in its own earlier step,
  never in a parallel group.
* `type: cntl` steps reuse the already-loaded `Cntl` (no `-f` issue);
  `type: cli` steps such as `cape_extract_dex` re-read the JSON and need
  `"AddFileName": true` unless the JSON file is the default.
* Parallel steps' STDOUT goes to `log/cape-perform.{index}.{k}`; STDERR and a
  status board go to the terminal. Check those logs if a group fails, and
  re-run a failed piece directly, e.g. `cape extract "LL_*" -I {CASES}`.
* Test on one or two cases before using a new `approve` on a large batch, and
  commit the JSON change.

## Solver user manuals

The CAPE package includes markdown conversions of CFD solver user manuals
under each solver module's folder:

    cape/py*/manuals/*/{manual,chapter_[0-9]*,appendix_[a-z],table_of_contents}.md

Each `manuals/{VERSION}/` folder holds one manual, either as a single
`manual.md` or split into `chapter_{N}.md` / `appendix_{L}.md` files. To
locate the package folder, run

    python3 -c "import cape, os; print(os.path.dirname(cape.__file__))"

Consult the manual matching the solver and version folder closest to the
version in use for authoritative details on solver input options before
changing them. These are text-only conversions: figures are not included and
tables may be imperfect.

If the `CAPE_MANUAL_PATH` environment variable is set, treat it as a
colon-separated list of directories containing additional user-supplied
manuals (markdown), searched in addition to the package copies.



