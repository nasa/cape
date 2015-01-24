

----------------------------
Uncategorized pyCart Options
----------------------------

Several inputs in :file:`pyCart.json` do not go in any category.  They are
relatively simple inputs, and many of them have usable defaults, although it's
still a good idea to enter them into the pyCart control file to avoid ambiguity
and improve traceability.

Cart3D Input Control Files
==========================
The first two
of these are names of files.  They are shown below with their default values.

    .. code-block:: javascript
    
        "InputCntl": "input.cntl",
        "AeroCsh": "aero.csh"
        
These are the names and locations of two *template* files required for running
Cart3D.  In the case of :file:`aero.csh`, it is only required if it is an
adaptive run.  It is sometimes useful to use template files with different
names, for example if multiple sets of inputs are being tested.  A recommended
practice for very involved Cart3D analyses is to put input files into a folder. 
Consider the following partial file tree.

    .. code-block:: none
    
        run/
            poweroff.json
            poweron.json
        inputs/
            input.cntl
            aero.csh
            Config.xml
            Components.i.tri
            
In this case, there are two pyCart control files, :file:`poweroff.json` and
:file:`poweron.json`, and they should contain the following.

    .. code-block:: javascript
    
        "InputCntl": "inputs/input.cntl",
        "AeroCsh": "inputs/aero.csh"
        
Obviously, there are a couple more files shown in the partial file tree above,
and these will come into play in other sections.  Then the call to pyCart will
be

    .. code-block:: bash
    
        $ pycart -f run/poweroff.json
        $ pycart -f run/poweron.json
        
for the two input files.

Maximum Number of Jobs to Submit
================================
This parameter sets the maximum number of jobs that pyCart will submit with a
single call to *pycart*.

    .. code-block:: javascript
    
        "nSubmit": 10
        
However, this value can be overridden from the command line using the ``-n``
option.

    .. code-block:: bash
    
        $ pycart -n 20

Startup Shell Commands
======================

An important miscellaneous option, especially for cases submitted as PBS jobs,
lists commands to run within the shell before running any Cart3D commands.
This is a list of strings that will be placed at the top of the run script in
each directory.  By default, this is an empty list, which is probably not
adequate to successfully run Cart3D.

    .. code-block:: javascript
    
        "ShellCmds": []
        
When pyCart sets up a case, it creates a run script :file:`run_ cart3d.pbs` in
each folder (or, if there is a nontrivial run sequence,
:file:`run_cart3d.00.pbs`, :file:`run_cart3d.01.pbs`, etc.).  The run script
can use BASH, ``csh``, or any other shell, and this is set in the "PBS" section
of :file:`pyCart.json`.  The default is BASH (that is, ``"/bin/bash"``), but
many Cart3D users prefer ``csh``.

If your rc file for your selected shell contains the necessary commands to run
Cart3D, a possible option is to use the following.

    .. code-block:: javascript
    
        "ShellCmds": [". ~/.cshrc"]
        
(or ``". ~/.bashrc"``, as appropriate)  This is *highly* discouraged unless
Cart3D is basically the only software you ever use.  A better option is to put
the commands that are needed in the :file:`pyCart.json` file, which makes that
file portable and less subject to later errors or changes.  Here is an example
that I use to run Cart3D on NASA's Pleiades supercomputer.

    .. code-block:: javascript
    
        "ShellCmds": [
            ". $MODULESHOME/init/bash",
            "module use -a /u/ddalle/share/modulefiles",
            "module load cart3d",
            "module load pycart",
            "module load mpt",
            "ulimit -S -s 4194304"
        ]
        
The first command is necessary because PBS jobs are started with very few
environment variables set.  For running cases in parallel, this command (or
sourcing a premade :file:`.*shrc` file) is necessary.  Another thing to note
here is that you also need to tell the interpreter where the pyCart commands
are---hence the ``"module load pycart"`` line.


The Group Mesh Option
=====================

The final miscellaneous option controls a peculiar setting that allows meshes
to share common meshes.  For example, many CFD run matrices are sweeps of Mach
number, angle of attack, and sideslip angle.  All of them could conceivably use
the same mesh, and this option allows that if set to ``true``.  The default is
also ``true``.

    .. code-block:: javascript
    
        "GroupMesh": true
        
If this option is set to ``true``, then the volume mesh is created once for
each "group" of cases, and only links are created in the individual case
folders.  For adaptive inputs, only the initial mesh is shared, and thus the
savings can be relatively minimal.

An example where this setting can be useful where there are also multiple
groups is the deflections of control surface.  Each position of, for example,
an elevator requires its own mesh, but changing the angle of attack does not
require a new mesh.  Thus grouping cases by elevator deflection angle can be
useful for saving mesh preparation time and file storage.

For cases where each case requires its own mesh, set this option to ``false``.
