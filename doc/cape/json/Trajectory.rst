
.. _cape-json-Trajectory:

------------------------
Trajectory or Run Matrix
------------------------

A required section of :file:`cape.json` describes the variables that define
the run matrix and the values of those variables.  In many cases, these will
be familiar input variables like Mach number, angle of attack, and sideslip
angle; but in other cases they will require additional description within the
:file:`cape.json` file.  In still other cases the defaults may be acceptable,
but the user wants to customize the folder names of some other aspect of the run
matrix.
        
Links to additional options for each specific solver are found below.

    * :ref:`Cart3D <pycart-json-Trajectory>`
    * :ref:`FUN3D <pyfun-json-Trajectory>`
    * :ref:`OVERFLOW <pyover-json-Trajectory>`

.. _cape-json-Trajectory-examples:

Example Trajectories
====================
The possible run matrix (i.e. "Trajectory") definitions range from a few
self-contained lines to hundreds of lines and pointers to external Python
modules.  The following examples demonstrate a few of the possibilities.

Simple Self-Contained Trajectory
--------------------------------
The following example is essentially the simplest type of run matrix that can be
defined.  Fortunately it is also quite common; there are three input variables,
which are Mach number, angle of attack, and sideslip angle.  This matrix has
few enough runs to just list the conditions directly in the JSON file.

    .. code-block:: javascript
    
        "Trajectory": {
            "Keys": ["Mach", "alpha", "beta"],
            "Mach": [2.00, 2.00, 2.00, 2.00, 2.00, 2.50],
            "alpha": [-2.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            "beta": [0.0, 0.0, 0.0, -2.0, 2.0, 0.0]
        }
        
Without going into all of the details, this will create the following folders.

    .. code-block:: none
    
        Grid/m2.0a-2.0b0.0
        Grid/m2.0a0.0b0.0
        Grid/m2.0a2.0b0.0
        Grid/m2.0a0.0b-2.0
        Grid/m2.0a0.0b2.0
        Grid/m2.5a0.0b0.0
        
By default, Cape uses ``Grid`` as the name of the group folder, which is a
folder that contains one or more run folders.  The entire solution for whichever
CFD solver is being used is contained within a single run folder, e.g.
``m2.0a0.0b0.0``.

By default, each folder uses as many digits as necessary to print the value of
each of the variables (i.e. ``m2.0`` instead of ``m2.0000``). Because
``"Mach"``, ``"alpha"``, and ``"beta"`` are recognized variables, Cape knows
to edit the appropriate lines in the input files.  For example, pyCart edits in
:file:`input.cntl` in each of the run folders.

However, all of this behavior can be controlled using more advanced options in
the trajectory ``"Definitions"`` section.


Simple Trajectory with External File
------------------------------------
A similar run matrix description to the one above simply moves the variable
values to an external file.

    .. code-block:: javascript
    
        "Trajectory": {
            "Keys": ["Mach", "alpha", "beta", "Re", "T"],
            "File": "Trajectory.dat",
            "GroupPrefix": "poweroff"
        }
        
The external file :file:`Trajectory.dat` contains the following.

    .. code-block:: none
    
        # Mach, alpha, beta, Re,      T
        2.00    -2.0   0.0   62506.0  445.0
        2.00    0.0    0.0   62506.0  445.0
        2.00    2.0    0.0   62506.0  445.0
        2.00    0.0    -2.0  62506.0  445.0
        2.00    0.0    2.0   62506.0  445.0
        2.50    0.0    0.0   62506.0  445.0
        
In this case, since the ``"GroupPrefix"`` key was set to a non-default value,
the folder names would be the following.  In addition, the folder names will
contain as many digits as the values in the file when using this method.  The
last two variables are Reynolds number per grid unit and static temperature in
Rankine.  These are required parameters for FUN3D and OVERFLOW, but they are not
included in the folder name by default.

    .. code-block:: none
    
        poweroff/m2.00a-2.0b0.0
        poweroff/m2.00a0.0b0.0
        poweroff/m2.00a2.0b0.0
        poweroff/m2.00a0.0b-2.0
        poweroff/m2.00a0.0b2.0
        poweroff/m2.50a0.0b0.0

        
.. _cape-json-TrajectoryCustom:

Standard Variables with Customizations
--------------------------------------
There are several advanced options that do not define any new run variables but
can be used to help organize a run matrix.  The example above shows one such
option with the ``"GroupPrefix"`` usage.  However, there are additional
customizations that can be done.  The following example demonstrates most of
them.

    .. code-block:: javascript
    
        "Trajectory": {
            "Keys": ["n", "Mach", "alpha", "beta", "config", "Label"],
            "File": "Trajectory.dat",
            "Definitions": {
                "n": {
                    "Group": false,
                    "Type": "prefix",
                    "Value": "int",
                    "Label": true,
                    "Format": "%02i_",
                    "Abbreviation": "n"
                },
                "Mach": {
                    "Group": false,
                    "Format": "%.2f",
                    "Abbreviation": "mach"
                },
                "alpha": {
                    "Format": "%+.1f"
                },
                "beta": {
                    "Format": "%+.1f"
                },
            }
        }
        
Then suppose the contents of :file:`Trajectory.dat` is

    .. code-block:: none
    
        # n, Mach, alpha, beta,  config,   label
        1    2.00    -2.0   0.0  poweroff
        2    2.00    0.0    0.0  poweroff
        2    2.00    0.0    0.0  poweroff  try2
        3    2.00    2.0    0.0  poweroff
        4    2.00    0.0    -2.0 poweroff
        5    2.00    0.0    2.0  poweroff
        6    2.50    0.0    0.0  poweroff
        6    2.50    0.0    0.0  poweroff  try2
        6    2.50    0.0    0.0  poweroff  try3     
        2    2.00    0.0    0.0  poweron
        6    2.50    0.0    0.0  poweron
        
In this example, we have done three things:

    #. Added a variable to just keep track of the job number, *n*.
    #. Slightly modified the *Mach*, *alpha*, *beta* keys, but not functionally.
    #. Added built-in variables called *config* and *Label* to help organize.
    
The run directories for these cases, corresponding to the lines of
:file:`Trajectory.dat` above, are the following.

    .. code-block:: none
    
        poweroff/n01_m2.00a-2.0b+0.0
        poweroff/n02_m2.00a+0.0b+0.0
        poweroff/n02_m2.00a+0.0b+0.0_try2
        poweroff/n03_m2.00a+2.0b+0.0
        poweroff/n04_m2.00a+0.0b-2.0
        poweroff/n05_m2.00a+0.0b+2.0
        poweroff/n06_m2.50a+0.0b+0.0
        poweroff/n06_m2.50a+0.0b+0.0_try2
        poweroff/n06_m2.50a+0.0b+0.0_try3
        poweron/n02_m2.00a+0.0b+0.0
        poweron/n06_m2.50a+0.0b+0.0

The folder names are directly affected by the ``"Format"`` key in the definition
of that variable, but the actual variables used to set the case up are not
truncated.  There is the possibility of conflicting folder names, however.

The ``"config"`` variable is convenient for run matrices where different jobs
might require slightly different inputs, and the ``"Label"`` key is convenient
when a particular case may need further analysis or another attempt to reach
satisfactory convergence.


.. _cape-json-TrajectoryAdvanced:

Advanced Example
----------------
There are many more tasks that a user may want to control with the run matrix
input variables.  For example, the user may want to deflect a control surface,
set a thrust level, translate a store that is being separated, rotate two
non-intersecting components relative to each other, or many more.

In Cape, these basically fall into a single category: advanced tasks that need
to be defined by the user.  An example of a ``"Trajectory"`` section in such a
control file might look something like the following.

    .. code-block:: javascript
    
        "Trajectory": {
            "Keys": ["alpha", "beta", "dx", "CT"],
            "File": "Trajectory.dat",
            "GroupPrefix": "poweron",
            "GroupMesh": true,
            "Definitions": {
                "dx": {
                    "Group": true,
                    "Type": "TriFunction",
                    "Value": "float",
                    "Format": "%.2f",
                    "Function": "self.mymod.TranslateDX"
                },
                "CT": {
                    "Group": false,
                    "Type": "CaseFunction",
                    "Value": "float",
                    "Format": "%.2f",
                    "Function": "self.mymod.SetThrust"
                }
            }
        }
        
These "Function" values that are referenced above must be defined by the user in
a Python module called :mod:`mymod`.  The simplest way to do this would be to
define the functions in a file called :file:`mymod.py`.  Of course, the module
does not have to be called :mod:`mymod`; any module can be used, and it requires
the user to have

    .. code-block:: javascript
    
        "Modules": ["mymod"]
        
in the main part of a JSON file to tell Cape to load that module before setting
up cases. It is often a good practice in cases such as these to use inputs in
the following form.

    .. code-block:: javascript
    
        "PythonPath": ["tools/"],
        "Modules": ["mymod"],
        
        "Trajectory": {
            // Rest of trajectory description here
        }
        
Then the module containing the special functions is in :file:`tools/mymod.py`.

Now suppose the trajectory file contains the following.

    .. code-block:: none
    
        # alpha, beta, dx, CT
        0.0, 0.0, 0.00, 0.75
        0.0, 0.0, 0.00, 1.00
        2.0, 0.0, 0.00, 1.00
        0.0, 2.0, 0.00, 1.00
        0.0, 0.0, 0.35, 0.75
        0.0, 0.0, 0.35, 1.00
        
For this case, the run directories will be

    .. code-block:: none
    
        poweron_dx0.00/a0.0b0.0CT0.75
        poweron_dx0.00/a0.0b0.0CT1.00
        poweron_dx0.00/a2.0b0.0CT1.00
        poweron_dx0.00/a0.0b2.0CT1.00
        poweron_dx0.35/a0.0b0.0CT0.75
        poweron_dx0.35/a0.0b0.0CT1.00

The reader may have noticed one last additional capability here, which is the
concept of a "group". The idea is that cases with different values of *alpha*,
*beta*, and *CT* can share the same mesh (or at least initial mesh), but cases
with different values of *dx* cannot. So group cases with the same value of *dx*
together and allow them to use a common mesh. In architectures where disk space
is severely limited, this can be very convenient, but this is not required
practice. Furthermore, enabling this feature requires the *GroupMesh* key
to be set to *true*, although that is the default.


.. _cape-json-TrajectoryDict:
        
Trajectory Option Dictionary
============================
The following is the full list of options for the "Trajectory" section.

    *Keys*: {*required*} | ``["k1", "k2"]`` | :class:`list` (:class:`str`)
        List of input variables for the run matrix
        
    *File*: {``""``} | ``"Trajectory.dat"`` | :class:`str`
        Name of text file (comma-separated, space-separated, or mixed) to use to
        read input variable values
        
    *GroupPrefix*: {``"Grid"``} | :class:`str`
        Prefix to each group folder, overridden by *config* if used
        
    *GroupMesh*: {``true``} | false
        Whether or not runs in the same group should share volume meshes
        
    *k1*: {``[]``} | :class:`list` | :class:`any`
        Fixed value or list of values for key *k1*, supersedes *File*
        
    *k2*: {``[]``} | :class:`list`
        Fixed value or list of values for key *k2*, supersedes *File*
        
    *Definitions*: {``{}``} | ``{"k1": d1, "k2": d1}`` | :class:`dict`
        Dictionary of variable definitions or variable customizations
            
        *d1*: :class:`dict`
            Dictionary of definitions or modifications for key *k1*
                
            *Abbreviation*: :class:`str`
                The label for the variable (e.g. "a" for "alpha") that gets used
                in the folder name
            
            *Group*: {``false``} | ``true``
                If ``false``, then two cases with different values of *k1* can
                still be in the same group
            
            *Format*: {``"%s"``} | ``"%i"`` | ``"%.2f"`` | :class:`str`
                Format string for the variable value in the folder name
            
            *Function*: {``""``} | :class:`str`
                For "TriFunction" and "CaseFunction" keys, the name of the
                function that gets invoked with the value of *k1* as its first
                argument
                
            *Label*: {``true``} | ``false``
                Whether or not the variable and value should be included in the
                folder name
                
            *PBSFormat*: {``"%s"``} | :class:`str`
                Format to use when putting the value in the PBS job name
                
            *PBSLabel*: {``true``} | ``false``
                Whether or not to use the variable in the PBS job name
                
            *SkipIfZero*: ``true`` | {``false``}
                Whether or not to omit a variable from a case name if the value
                of the variable is zero
                
            *Type*: {``"Label"``} | ``"mach"`` | ``"alpha"`` | ``"beta"`` |
            ``"alpha_t"`` | ``"phiv"`` | ``"Prefix"`` | ``"GroupLabel"`` |
            ``"TriFunction"`` | ``"CaseFunction"``
            
                General purpose of the variable
                
            *Value*: {``"float"``} | ``"int"`` | ``"str"`` | ``"bin"`` |
            ``"oct"`` | ``"hex"``
            
                Class of the values of the variable
    
                
.. _cape-json-TrajectoryNames:

Case Name Options
-----------------
Several of the key definition keys, specifically *Abbreviation*, *Format*,
*Label*, and *SkipIfZero* are used for deciding on what the name of the folder
containing the solution will be.  Furthermore, *Abbreviation*, *PBSFormat*, and
*PBSLabel* determine the name of the PBS job that is submitted (if appropriate).
Together, these keys allow for very robust folder naming practices, which is
useful for advanced configurations.  For example, a user may choose to fix the
number of leading zeros for a variable, not include a variable in a name,
include or not include the sign of a variable if it's positive, insert
underscores or other characters between variables, or customize any other aspect
of the parametric folder names.

Consider a set cases defined in a run matrix file :file:`inputs/matrix.csv` with
the following contents.

    .. code-block:: none
    
        # mach, alpha, beta, k,   CT,    config
        4.0,    0.0,   -2.0,   2, 1.075, poweron
        4.0,    0.0,    0.0,  27,   0.1, poweron
        4.0,    0.0,    2.0, 138,   0.0, poweroff
        4.0,    2.0,    0.0,  13,  1.22, poweron

The default or basic JSON definitions to go with this section are below.

    .. code-block:: javascript
    
        "Trajectory": {
            "Keys": ["mach", "alpha", "beta", "k", "CT", "config"],
            "File": "inputs/matrix.csv",
            "Definitions": {
                "k": {
                    "Type": "Value",
                    "Value": "int"
                },
                "CTMain": {
                    "Type": "CaseFunction",
                    "Value": "float",
                    "Function": "self.mymod.SetThrust"
                }
            }
            
Four of the trajectory keys, *mach*, *alpha*, *beta*, and *config*, are built-in
and in this example rely completely on default Cape definitions.  We have
defined *k* as some sort of case index here that doesn't have much effect on the
CFD setup but appears in the folder names for tracking purposes, and *CTMain* is
meant to represent a thrust setting variable that relies on a user-defined
function to adjust boundary conditions appropriately for each case.  The case
names originating from this setup will be the following.

    .. code-block:: none
    
        poweron/m4.0a0.0b-2.0k2CTMain1.075/
        poweron/m4.0a0.0b0.0k27CTMain0.1/
        poweroff/m4.0a0.0b2.0k138CTMain0.0/
        poweron/m4.0a2.0b0.0k13CTMain1.22/

These folder names are adequate in that they are unique and contain enough
information about each case to identify it, but they are clearly a little bit
disappointing.

For one thing, it would be nice to have some visual separation before the *k*
variable to make it easier to identify the individual variables.  Second,
``"CTMain"`` is an unnecessarily long label to include.  The following JSON
syntax will apply these two changes.

    .. code-block:: javascript
    
        "Definitions": {
            "beta": {
                "Format": "%s_"
            },
            "k": {
                "Type": "Value",
                "Value": "int"
            },
            "CTMain": {
                "Abbreviation": "CT",
                "Type": "CaseFunction",
                "Value": "float",
                "Function": "self.mymod.SetThrust"
            }
        }

The resulting folder names are below.

    .. code-block:: none
    
        poweron/m4.0a0.0b-2.0_k2CT1.075/
        poweron/m4.0a0.0b0.0_k27CT0.1/
        poweroff/m4.0a0.0b2.0_k138CT0.0/
        poweron/m4.0a2.0b0.0_k13CT1.22/

This is a little better, but it's still a little difficult to read because the
values don't always have the same number of characters.  The following example
pads the *k* value with leading zeros so that the label has the same number of
digits, includes the sign in the value of *beta*, and modifies the *CT* label.

    .. code-block:: javascript
    
        "Definitions": {
            "beta": {
                "Format": "%+.1f_"
            },
            "k": {
                "Type": "Value",
                "Value": "int",
                "Format": "%03i"
            },
            "CTMain": {
                "Abbreviation": "CT",
                "Type": "CaseFunction",
                "Value": "float",
                "Format": "%.2f",
                "Function": "self.mymod.SetThrust"
            }
        }

    .. code-block:: none
    
        poweron/m4.0a0.0b-2.0_k002_CT1.08/
        poweron/m4.0a0.0b+0.0_k027_CT0.10/
        poweroff/m4.0a0.0b+2.0_k138_CT0.00/
        poweron/m4.0a2.0b+0.0_k013_CT1.22/
    
Finally, it is often the case that some of the variables are not necessary to
include in the case name (because the cases are already unique).  In this case,
suppose we do not need *CT* to be in the case name, and we will have a nicer
looking set of cases.  We might also choose to omit the Mach number from the
case here since all four cases have the same Mach number.  Some users also like
to omit certain variables from the label if the value happens to be zero.

    .. code-block:: javascript
    
        "Definitions": {
            "mach": {
                "Label": false
            },
            "beta": {
                "SkipIfZero": true
            },
            "k": {
                "Abbreviation": "_k",
                "Type": "Value",
                "Value": "int",
                "Format": "%03i"
            },
            "CTMain": {
                "Abbreviation": "CT",
                "Type": "CaseFunction",
                "Value": "float",
                "Label": false,
                "Function": "self.mymod.SetThrust"
            }
        }

    .. code-block:: none
    
        poweron/a0.0b-2.0_k002/
        poweron/a0.0_k027/
        poweroff/a0.0b2.0_k138/
        poweron/a2.0_k013/

To explicitly explain how these options are translated into a case name, each
variable is translated into text using the following Python code.

    .. code-block:: python
    
        Abbreviation + (Format % v)
        
where *Abbreviation* is the content of the ``"Abbreviation"`` option above,
likewise for *Format*, and *v* is the value of that variable.  Cape makes a
folder name by first looping through all trajectory keys with *Type* of
``"Prefix"``, then looping through all other non-string trajectory keys, and
then looping through any other string trajectory keys.  Any key with the *Label*
option set to false is not included in the folder name.  If not defined, the
*Abbreviation* defaults to the name of the trajectory key.

Likewise, the same basic process is used for PBS job names with the additional
rule that PBS job names are limited to 15 characters.  Cape automatically trims
the PBS job name if a longer name would be generated by the user input
parameters.


.. _cape-json-TrajectoryGroups:

Group Options
-------------
Identifying a trajectory key as a "Group" parameter means that all cases sharing
the same value for that key can be grouped together for some reason or another.
Using the example from the previous subsection, we can make *alpha* a group key
and get the results that follow.

    .. code-block:: javascript
    
        "Definitions": {
            "mach": {
                "Label": false
            },
            "alpha": {
                "Group": true
            },
            "k": {
                "Abbreviation": "_k",
                "Type": "Value",
                "Value": "int",
                "Format": "%03i"
            },
            "CTMain": {
                "Abbreviation": "CT",
                "Type": "CaseFunction",
                "Value": "float",
                "Label": false,
                "Function": "self.mymod.SetThrust"
            }
        }

    .. code-block:: none
    
        poweron_a0.0/b-2.0_k002/
        poweron_a0.0/b0.0_k027/
        poweroff_a0.0/b2.0_k138/
        poweron_a2.0/b0.0_k013/


.. _cape-json-TrajectoryKeys:
        
List of Built-In Variables
==========================
The following trajectory variables have built-in definitions that are recognized
by name.  If you use a variable with one of these names, they are automatically
assigned the corresponding default definition (which can then be modified by the
user).  Most of these built-in definitions can be invoked by more than one name,
which is why the headers are lists.  Any name in that list gets the default
definition described.  Finally, all these definitions have ``"Format": "%s"`` as
part of their definition, so that is omitted from this list.

[*M*, *m*, *Mach*, *mach*]:
Mach number; sets the Mach number in appropriate input file(s) for each case

    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "Mach",
            "Value": "float",
            "Abbreviation": "m"
        }

[*Alpha*, *alpha*, *aoa*]:
Angle of attack; sets the angle of attack in input file(s) for each case

    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "alpha",
            "Value": "float",
            "Abbreviation": "a"
        }
        
[*Beta*, *beta*, *aos*]:
Sideslip angle; sets the sideslip angle in input file(s) for each case

    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "beta",
            "Value": "float",
            "Abbreviation": "b"
        }
        
[*alpha_t*, *alphav*, *alphat*, *alpha_total*]:
Total angle of attack; affects angle of attack and sideslip angle

    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "alpha_t",
            "Value": "float",
            "Abbreviation": "a"
        }
        
[*phi*, *phiv*]:
Velocity roll angle; affects angle of attack and sideslip angle

    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "phi",
            "Value": "float",
            "Abbreviation": "r"
        }
        
[*Re*, *Rey*, *Reynolds*, *Reynolds_number*]:
Reynolds number per grid unit

    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "Re",
            "Value": "float",
            "Label": false,
            "Abbreviation": "Re"
        }
        
[*T*, *Tinf*, *temperature*]:
Freestream static temperature
    
    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "T",
            "Value": "float",
            "Label": false,
            "Abbreviation": "T"
        }
        
[*p*, *P*, *pinf*, *pressure*]:
Freestream static pressure

    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "p",
            "Value": "float",
            "Label": false,
            "Abbreviation": "p"
        }
        
[*q*, *qinf*, *qbar*]:
Freestream dynamic pressure

    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "q",
            "Value": "float",
            "Label": false,
            "Abbreviation": "q"
        }
        
[*gamma*]:
Freestream ratio of specific heats

    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "gamma",
            "Value": "float",
            "Label": false,
            "Abbreviation": "g"
        }
        
[*p0*, *p_total*, *total_pressure*]:
Thrust boundary condition total pressure setting

    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "SurfBC",
            "Value": "float",
            "Label": false,
            "Abbreviation": "p0",
            "TotalPressure": null,
            "TotalTemperature": "T0",
            "RefPressure": 1.0,
            "RefTemperature": 1.0,
            "CompID": [],
            "Mach": 0.2,
            "Blend": 0.9
        }
        
[*T0*, *T_total*, *total_temperature*]:
Thrust boundary condition total temperature setting

    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "value",
            "Value": "float",
            "Label": false,
            "Abbreviation": "T0"
        }

[*CT*]:
Thrust coefficient or dimensional thrust

    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "SurfCT",
            "Value": "float",
            "Label": true,
            "Abbreviation": "CT",
            "TotalTemperature": "T0",
            "RefDynamicPressure": null,
            "RefPressure": null,
            "RefTemperature": null,
            "RefArea": null,
            "Mach": 1.0,
            "AreaRatio": 4.0,
            "ExitArea": null,
            "ExitMach": null,
            "CompID": [],
            "Blend": 0.9
        }
        
[*Label*, *label*, *suffix*]:
Label a case with otherwise similar name

    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "Label",
            "Value": "str",
            "Abbreviation": ""
        }
        
[*config*, *GroupPrefix*]:
Prefix added to group folder name (often the entire folder name)

    .. code-block:: javascript
    
        {
            "Group": true,
            "Type": "Prefix",
            "Value": "str",
            "Abbreviation": ""
        }
        
[*GroupLabel*, *GroupSuffix*]:
Suffix added to group folder name

    .. code-block:: javascript
    
        {
            "Group": true,
            "Type": "GroupLabel",
            "Value": "str",
            "Abbreviation": ""
        }
        
[*CaseFunction*]:
Function to apply after writing mesh files and just before writing input files

    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "CaseFunction",
            "Value": "float",
            "Function": ""
        }

