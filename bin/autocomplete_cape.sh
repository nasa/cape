#!/bin/bash

_cape_complete() {
    # Generate completions
    COMPREPLY=( $(python3 -m cape.cfdx.autocomplete "$COMP_LINE") )
}

complete -F _cape_complete pyfun

