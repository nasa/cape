
_cape_complete() {
    local cur
    cur="${COMP_WORDS[COMP_CWORD]}"
    COMPREPLY=( $(python3 -m cape.autocomplete "$cur") )
}

complete -F _cape_complete pyfun
