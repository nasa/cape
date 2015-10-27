
------------------------
Trajectory or Run Matrix
------------------------

A required section of :file:`pyCart.json` describes the variables that define
the run conditions and the values of those variables.  In many cases, these will
be familiar input variables like Mach number, angle of attack, and sideslip
angle; but in other cases they will require additional description within the
:file:`pyCart.json` file.

Example Trajectories
====================
These possibilities can be roughly categorized into several types of
trajectories, which range in complexity.  The simplest descriptions are
self-contained and contain just a few lines, and the most complex may contain
more than a hundred lines and a pointer to an external file that sets the actual
values for thousands of jobs.

Simple Self-Contained Trajectory
--------------------------------
The following example is essentially the simplest type of run matrix that can be
defined.  Fortunately it is also quite common; there are three input variables,
which are Mach number, angle of attack, and sideslip angle; and few enough runs
to just list them directly in the :file:`pyCart.json` file.

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
        
By default, pyCart uses ``Grid`` as the name of the group folder, and it uses as
many digits as necessary to print the value of each of the variables (i.e.
``m2.0`` instead of ``m2.0000``).  Because ``"Mach"``, ``"alpha"``, and
``"beta"`` are recognized variables, pyCart knows to set the appropriate values
in :file:`input.cntl` in each of the run folders.

However, all of this behavior can be controlled using more advanced options in
:file:`pyCart.json`.

Simple Trajectory with External File
------------------------------------
A similar run matrix description to the one above simply moves the variable
values to an external file.

    .. code-block:: javascript
    
        "Trajectory": {
            "Keys": ["Mach", "alpha", "beta"],
            "File": "Trajectory.dat",
            "GroupPrefix": "poweroff"
        }
        
The external file :file:`Trajectory.dat` contains the following.

    .. code-block:: none
    
        # Mach, alpha, beta
        2.00    -2.0   0.0
        2.00    0.0    0.0
        2.00    2.0    0.0
        2.00    0.0    -2.0
        2.00    0.0    2.0
        2.50    0.0    0.0
        
In this case, since the ``"GroupPrefix"`` key was set to a non-default value,
the folder names would be the following.  In addition, the folder names will
contain as many digits as the values in the file when using this method.

    .. code-block:: none
    
        poweroff/m2.00a-2.0b0.0
        poweroff/m2.00a0.0b0.0
        poweroff/m2.00a2.0b0.0
        poweroff/m2.00a0.0b-2.0
        poweroff/m2.00a0.0b2.0
        poweroff/m2.50a0.0b0.0
        
Standard Variables with Customizations
--------------------------------------
There are several advanced options that do not define any new run variables but
can be used to help organize a run matrix.  The example above shows one such
option with the ``"GroupPrefix"`` usage.  However, there are additional
customizations that can be done.  The following example demonstrates most of
them.

    .. code-block:: javascript
    
        "Trajectory": {
            "Keys": ["n", "Mach", "alpha", "beta", "config", "label"],
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

Advanced Example
----------------
There are many more tasks that a user may want to control with the run matrix
input variables.  For example, the user may want to deflect a control surface,
set a thrust level, translate a store that is being separated, rotate two
non-intersecting components relative to each other, or many more.

In pyCart, these basically fall into a single category: advanced tasks that need
to be defined by the user.  An example of a ``"Trajectory"`` section in such a
pyCart control file might look something like the following.

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
        
in the main part of :file:`pyCart.json` to tell pyCart to load that module
before setting up cases.  It is often a good practice in cases such as these to
use inputs in the following form.

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

You may have noticed one last additional capability here, which is the concept
of a "group".  The idea is that cases with different values of *alpha*, *beta*,
and *CT*, can share the same mesh (or at least initial mesh), but cases with
different values of *dx* cannot.  So group cases with the same value of *dx*
together and allow them to use a common mesh.  In architectures where disk space
is severely limited, this can be very convenient, but pyCart never requires this
type of usage.  Furthermore, enabling this feature requires the *GroupMesh* key
to be set to *true*, although that is the default.
        
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
        
    *k1*: {``[]``} | :class:`list`
        List of values for trajectory key *k1*, ignored if *File* is used
        
    *k2*: {``[]``} | :class:`list`
        List of values for trajectory key *k2*, ignored if *File* is used
        
    *Definitions*: {``{}``} | ``{"k1": d1, "k2": d1}`` | :class:`dict`
        Dictionary of variable definitions or variable customizations
            
        *d1*: :class:`dict`
            Dictionary of definitions or modifications for key *k1*
            
            *Group*: {``false``} | ``true``
                If ``false``, then two cases with different values of *k1* can
                still be in the same group
                
            *Type*: {``"Label"``} | ``"Mach"`` | ``"alpha"`` | ``"beta"`` |
            ``"alpha_t"`` | ``"phiv"`` | ``"Prefix"`` | ``"GroupLabel"`` |
            ``"TriFunction"`` | ``"CaseFunction"``
            
                General purpose of the variable
                
            *Value*: {``"float"``} | ``"int"`` | ``"str"``
                Class of the values of the variable
                
            *Label*: {``true``} | ``false``
                Whether or not the variable and value should be included in the
                folder name
            
            *Format*: {``"%s"``} | ``"%i"`` | ``"%.2f"`` | :class:`str`
                Format string for the variable value in the folder name
                
            *Abbreviation*: :class:`str`
                The label for the variable (e.g. "a" for "alpha") that gets used
                in the folder name
            
            *Function*: {``""``} | :class:`str`
                For "TriFunction" and "CaseFunction" keys, the name of the
                function that gets invoked with the value of *k1* as its first
                argument
                


List of Built-In Variables
==========================
The following trajectory variables have built-in definitions that are recognized
by name.  If you use a variable with one of these names, they are automatically
assigned the corresponding default definition.  Most of these built-in
definitions can be invoked by more than one name, which is why the headers are
lists.  Any name in that list gets the default definition described.  Finally,
all these definitions have ``"Format": "%s"`` as part of their definition, so
that is omitted.

[*M*, *m*, *Mach*, *mach*]:
Mach number; sets the Mach number in :file:`input.cntl` for each case

    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "Mach",
            "Value": "float",
            "Abbreviation": "m"
        }

[*Alpha*, *alpha*, *aoa*]:
Angle of attack; sets the angle of attack in :file:`input.cntl`

    .. code-block:: javascript
    
        {
            "Group": false,
            "Type": "alpha",
            "Value": "float",
            "Abbreviation": "a"
        }
        
[*Beta*, *beta*, *aos*]:
Sideslip angle; sets the sideslip angle in :file:`input.cntl`

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
        
[*label*, *suffix*]:
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

For example, that's why the first example above has no "Definitions" section at
all.  In addition, the user can choose to modify only a part of these 
definitions and leave the remaining defaults in place.
