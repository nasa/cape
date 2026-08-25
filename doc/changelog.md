# Changelog

## Release 2.3.0

### New Features

-   *More flexible command-line interface*

    CAPE calls, while still preserving backward compatibility, can now be
    simplified. Instead of calling `pyfun`, `pycart`, `pyover`, etc. most
    calls can now be done by just using `cape`. CAPE will automatically
    determine which solver module to use by quickly looking at the contents of
    the JSON file. In combination with the implicit or explicit sub-command,
    this means that most CAPE calls have four different ways to be invoked:

    ```console
    $ cape c
    $ cape -c
    $ pyfun check
    $ pyfun -c
    ```

    If the `cape` executable determines the wrong solver (e.g. it thinks your
    JSON file is a Cart3D JSON file when it's actually for LAVA), you can use
    the explicit call (`pylava` in the previous example).

-   **pylava**: CAPE has added support for a fifth solver, NASA's new LAVA
    package. For this release only the LAVA Cartesian solver is fully
    supported.

-   *Additional CAPE sub-commands and options*

    - Use `cape open-pdf report/report-$REPORT.pdf --pull` to automatically
      transfer a report file from an HPC node and open it locally
    - Use `cape find` to just return the indices of cases that match a set of
      constraints.
    - The `cape edit-json` command is an option to edit the contents of the
      main JSON file through a command-line interface
    - Case subsets can now be constrained by status, e.g

        ```console
        $ cape c --status DONE
        ```

      However, using this constraint interferes with the speed-up from
      parallelization (see below).
    - For shared run matrices, you can use the new option `--me` instead of
      the much longer `--cons "user=='$USER'"`. The `--me` option will do
      nothing unless you have a run matrix key named `"user"` or another
      *user*-type key defined.

-   *Much faster run matrix checking*

    Checking on a large run matrix, especially with `pyfun`, can be
    time-consuming. This is mostly because determining how many iterations have
    been run can be surprisingly complicated for some of the CFD solvers.
    `cape -c` is now automatically parallelized, so that it can analyze
    several cases in parallel, controlled by the new `--nproc` option.

-   *Smarter error handling*

    Many errors caused by incorrect input (and also `KeyboardInterrupt` or
    `Ctrl-C`) will now produce a small error message rather than a full
    Python traceback. Progress will continue to be made on this front in future
    versions of CAPE.

-   *Logging*: most command-line CAPE calls are now recorded in a log file.

    Whenever you run a CAPE command that involves a `cape.cfdx.cntl.Cntl`
    instance, it will now log that command the folder

    `log/{BASENAME_OF_JSON_FILE}/`

    where `$BASENAME_OF_JSON_FILE` is the name of the JSON/YAML control file
    with the `.json` stripped. This can include sub-folders. In this folder
    for logging a single CAPE file, you will see

    - `cmd.log`
    - `hash.log`
    - `last.json`

    For example if you run `pyfun -c` and `pyFun.json` is linked to
    `run/dac3-asc01.json`, you'll get a folder (or append to it if it already
    exists) called `log/cmd/run/dac3-asc01/` with those three files in it.
    The `cmd.log` file will contain a line such as

    `CMD,2026-01-07 14:55:33,pyfun check -f run/dac3-asc01.json`

    The commands are "canonical-ized" so that the actual name of the JSON file
    is always included, and it will use the new two-word format for the logged
    commands. The line preceding this `CMD` line will be a SHA-256 hash of
    the current expanded (and comment-stripped) JSON file for tracing.

    You can turn this off by setting

    ```json
    "LogLevel": 0
    ```

    in the JSON file or by the environment variable `$CAPE_LOG_LEVEL`.

    ```console
    export CAPE_LOG_LEVEL=0
    ```

-   *Advanced case naming*

    New settings in the `"RunMatrix"` section named `"Replace"` and
    `"RegexSubs"`. These allow you to replace one string with another
    (*Replace*) or apply general regular expression replacements using
    `re.sub()`. For example, if you don't want decimal points in your case
    names, you could use:

    ```json
    "RunMatrix": {
        "Replace": {
            ".": "p"
        }
    }
    ```

    to get conversions like `m2.50a2.0` → `m2p50a2p0` or use

    ```json
    "RunMatrix": {
        "Replace": {
            ".": ""
        }
    }
    ```

    to get conversions like `m2.50a2.0` → `m250a20`. One more example:

    ```json
    "RunMatrix": {
        "Replace": {
            ".0": "",
            ".": ""
        }
    }
    ```

    to get `m2.50a2.0` → `m250a2`.
