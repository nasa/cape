

# List of parameters common to **all** run-matrix commands
SUBSET_PARAMS = {
    "f": {
        "description": (
            "Name of JSON file to use. CAPE will find the most appropriate "
            "file if left empty. If users specify a file, continue to use "
            "that file until user specifically requests a different one."
            "Synonyms: file, json."
        ),
        "type": ["string", "null"],
    },
    "I": {
        "description": (
            "Indices of cases to consider. Indexing follows Python syntax. "
            "This can be a single case "
            "like '14', a comma-separated list like '14,19,20', "
            "a range such as '14:20', or a combination like 14,17:20. "
            "Examples:\n* Case 8 -> '8'\n* Cases 5-10 -> '5:11'"
        ),
        "type": ["string", "null"],
    },
    "re": {
        "description": (
            "Consider cases containing matches for this regular expression."
            "Synonym: regex."
        ),
        "type": ["string", "null"]
    },
    "cons": {
        "description": (
            "Add constraints on the run matrix keys. The user specifies the "
            "key and constraint. These can be specified as logical operators."
            "Example: Run cases where run matrix key called 'mach' is "
            "greater than 1.0 -> 'mach>1.0'. Protect the right-hand side with "
            "quotes if searching for a string value."
        ),
        "type": ["string", "null"]
    },
    "filter": {
        "description": (
            "Limit command to cases containing a string of text specified "
            "by the user. "
            "Example: Only show cases containing 'm3': filter='m3'"
        ),
        "type": ["string", "null"]
    },
    "user": {
        "description": (
            "Limit to cases owned by this specific user"
        ),
        "type": ["string", "null"]
    },
    "me": {
        "description": (
            "Limit to cases owned by the current user, equivalent to "
            "user='$USER'"
        ),
        "type": ["boolean", "null"]
    },
    "unmarked": {
        "description": (
            "Only consider cases with no PASS/ERROR markings. This should be "
            "used if a user asks to only see unmarked cases or cases that are"
            "not passed or is looking for cases with a specific status other "
            "than PASS/PASS*/ERROR."
        ),
        "type": ["boolean", "null"]
    },
    "marked": {
        "description": (
            "Only consider cases with PASS/ERROR markings. This should be "
            "used if a user asks to only see marked cases or cases that are "
            "passed."
        ),
        "type": ["boolean", "null"]
    },
}


OTHER_CAPE_PARAMS = {
    "add_cols": {
        "description": (
            "Additional columns to show in case status output. Multiple "
            "additional columns can be specified as comma-separated list."
        ),
        "type": ["string", "null"]
    },
    "h": {
        "description": (
            "Display help message and exit. The help message describes all "
            "commands a user has access to and can call through CAPE. "
            "Synonym: help."
        ),
        "type": ["boolean", "null"]
    },
    "n": {
        "description": (
            "Submit at most n cases. The user specifies the number of cases "
            "to run. This should run all avaiable cases available up to the "
            "number specified. Cases cannot be in status PASS or DONE."
            "Synonym: N."
        ),
        "type": ["integer", "null"]
    },
    "j": {
        "description": (
            "List the PBS/Slurm job ID. This is used when the user checks "
            "the status of one or more cases."
        ),
        "type": ["boolean", "null"]
    },
    "batch": {
        "description": (
            "Submit PBS/Slurm job and run this command."
        ),
        "type": ["boolean", "null"]
    },
    "e": {
        "description": (
            "Execute the command EXEC."
            "Synonym: exec."
        ),
        "type": ["string", "null"]
    },
    "no-restart": {
        "description": (
            "Only submit new cases when submitting jobs."
        ),
        "type": ["boolean", "null"]
    },
    "no-start": {
        "description": (
            "Only set up cases. Do not start or submit cases to run."
        ),
        "type": ["boolean", "null"]
    },
    "q": {
        "description": (
            "Submit to a specific PBS/Slurm queue. The target queue is "
            "specified by the user. This command overrides the queue value "
            "in the input JSON file."
            "Synonyms: queue."
        ),
        "type": ["string", "null"]
    },
    "u": {
        "description": (
            "Pretend to be the user UID. The original user is able to act "
            "as the user of the UID they specify."
        ),
        "type": ["string", "null"]
    },
    "x": {
        "description": (
            "Execute a Python script after reading the JSON file. The script "
            "is specified by the user and can only run after the JSON."
        ),
        "type": ["string", "null"]
    },
}
