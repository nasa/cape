#!/usr/bin/env tclsh

global Par

# Source folder stuff
set ScriptFile [file normalize [info script]]
set ScriptDir  [file dirname $ScriptFile]
set GeomDir    [file join {*}[lrange [file split $ScriptDir] 0 end-1]]

set Par(ScriptFile) $ScriptFile
set Par(ScriptDir)  $ScriptDir
set Par(GeomDir)    $GeomDir

# Some sort of global sswitch
set ovfi_inputs "ssor"

# List of parts included
set IncludeBullet    1

# Grid scaling parameter
set GlobalScaleFactor 1.0
