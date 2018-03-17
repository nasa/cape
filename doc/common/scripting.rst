
.. _cape-scripting:

Advanced Setup: Python Scripting
=================================
In addition to the many settings which can be typed into the main JSON file,
users can create Python scripts and/or modules which can either add settings to
what's in the JSON file or change settings depending on the run matrix.  For
example, a complex run configuration with 100 or more databook components could
define them in a Python loop rather than with many lines of ASCII text in the
JSON file.

Use of Python can be done in two main ways: linking to modules via certain
options within the JSON file that point to Python modules and functions or
executing scripts which modify the settings via the command-line option ``-x``. 
The JSON options that result in the use of Python functions are *PythonPath*,
*Modules*, *InitFunction*, and *CaseFunction*.

The *PythonPath* key is an option simply appends folder names to the environment
variable ``$PYTHONPATH`` so that ``.py`` files in that folder can be directly
imported.  Next, *Modules* simply gives a list of Python files to import as
modules whenever the JSON file is read.  Once these modules have been imported,
the *InitFunction* specifies one or more functions that is called as soon as
pyCart is done interpreting the JSON file.  Thus *InitFunction* is usually most
useful to determine settings that are universal to the whole run matrix but for
whatever reason are easier to program in Python than write out in JSON text.
The last option, *CaseFunction*, specifies one or more functions to call before
submitting a new case (or running the ``--apply`` command).  These functions
take the case number as an argument and can be used to change any setting
depending on what the conditions for that case are.

.. _cape-scripting-x:

Using Scripts to Modify or Add Settings
---------------------------------------
Probably the simplest but least reliable method to move some of the work of
creating the settings from JSON to Python is the ``-x`` option.  When a
command-line call sees ``-x SCRIPT``, it first reads the relevant JSON file and
then uses :func:`execfile` to execute that script.

Consider the following example.

    .. code-block:: bash
    
        $ pyfun -f poweroff.json -x poweroff.py -x report.py
        
When the executable script ``pyfun`` sees this command, it first loads the
settings from :file:`poweroff.json` and then runs the scripts
:file:`poweroff.py` and :file:`report.py` (in that order) using
:func:`execfile`.  To do the same thing with a manually-created script, the
following syntax would work.

    .. code-block:: python
    
        # Load relevant module
        import pyFun
        # Read JSON file
        fun3d = pyFun.Fun3d('poweroff.json')
        # Apply scripts
        execfile('poweroff.py')
        execfile('report.py')
        
As the example somewhat indirectly shows (at least to those familiar with
:func:`execfile`), the global variable *fun3d* is accessible within the two
``.py`` scripts, and modifications to the settings must reference the variable
*fun3d*.  For example, if the goal of :file:`poweroff.py` is to set some extra
data book components, the following example might work.

    .. code-block:: python
    
        # List of components to add
        comps = ["nozzle1', "nozzle2", "nozzle3", "attachment"]
        # Add them as default force and moment components
        fun3d.opts["DataBook"]["Components"] += comps
        
        # Loop through four boosters
        for k in range(4):
            # Name of booster
            name = "booster%s" % (k+1)
            # Name of line load component
            comp = "%s_ll" % name
            # Clocking angle so that *CN* points away from core rocket
            clock = k*90.0
            # Add the component
            fun3d.opts["DataBook"]["Components"].append(comp)
            # Add the definition
            fun3d.opts["DataBook"][comp] = {
                "Type": "LineLoad",
                "CompID": name,
                "Transformations": [{"Type": "Euler321", "phi":clock}]
            }
            
As this example might make clear, these scripting capabilities have a tendency
to require or at least benefit from knowledge of the pyCart API.  However, while
API functions may be useful for generating these scripts, all settings *can* be
accessed as if they were in a standard Python :class:`dict`.  The JSON file is
basically read in as a dictionary that is saved in *fun3d.opts*.  For example,
the ``"Report"`` section of the JSON file becomes ``fun3d.opts["Report"]``, and
the *Report>Subfigures* subsection is ``fun3d.opts["Report"]["Subfigures"]``,
etc.

Finally, because this ``-x`` command uses global variables, the scripts must use
the correct variable name for the global settings handle.  In the examples above
it was *fun3d*, but below is a table of which variable name to use for each
primary executable.

    =================   ====================
    Command             Variable Name
    =================   ====================
    ``pycart``          *cart3d*
    ``pyfun``           *fun3d*
    ``pyover``          *ofl*
    =================   ====================


.. _cape-scripting-InitFunction:

Global Settings from Python Modules
-------------------------------------
A more reliable method for altering settings from Python instead of JSON
(reliable in the sense that users might forget to add all the correct ``-x``
options while *InitFunction* contains the proper information right within the
JSON file) is to use initialization functions.

In a way, using modules instead of scripts is slightly more work than using
scripts with the ``-x`` option because it requires creating a module (which is
similar to creating a script) and also setting a few options within the JSON
file.

These functions are controlled in the JSON file using the global
``"InitFunction"`` setting.  Consider the following example from a master JSON
input file.

    .. code-block:: javascript
    
        "PythonPath": ["tools"],
        "Modules": ["tfm"],
        "InitFunction": ["tfm.InitArchive"],
        
These options import a module called :mod:`tfm`, which may be defined in the
file :file:`tools/tfm.py` or :file:`tools/tfm/__init__.py`.  The function
:func:`tfm.InitArchive` is called immediately after the JSON file is
interpreted, and it can be used to change the settings otherwise specified in
the JSON file.

Example contents of the :mod:`tfm` module defined in :file:`tools/tfm.py` are
below:

    .. code-block:: python
    
        # System modules
        import os
        
        # Define an initialization function
        def InitArchive(cntl):
            """Change archive folder to appropriate value for calling user"""
            # Set the *ArchiveFolder* option using an environment variable
            cntl.opts.set_ArchiveFolder(os.path.join('/u/',
                os.environ['USER'], 'sls', '10008', 'f3_tfm'))
                
This function is not terribly complex but accomplishes a task that cannot be
performed directly in the JSON file.  The example is useful under the
assumption that multiple users are running cases from the same input file, and
this allows each user to set the archive location to their own home folder even
while using identical input files.

The input to the *InitFunction* is the :class:`cape.cntl.Cntl`,
:class:`pyCart.cart3d.Cart3d`, :class:`pyFun.fun3d.Fun3d`, etc. global object
that contains all of the settings from the JSON file and an interface to act on
them.  The JSON settings are stored within *cntl.opts*.

The *InitFunction* can be used to accomplish many more complex tasks.  For
example you may have many different databook components that have their own
definitions, which may or may not be simple.  It is usually possible to define
these within the JSON file, but it may require many lines of repetitive text.
The *InitFunction* allows the user define these components within a loop using
Python code and the Cape API.


.. _cape-scripting-CaseFunction:

Special Settings for Individual Runs from Python Modules
----------------------------------------------------------
A related capability to the *InitFunction* is the so-called *CaseFunction*.
This specifies one or more Python functions to be executed from a user-defined
module(s) after all other setup tasks for a case have been performed.  This
allows the user to customize CFD settings or any other aspect of the inputs in
ways that might not be possible with other Cape capabilities.  In short, this
can be used as a method of last resort if the environment does not support the
level of customization required by the user.

Some common examples of *CaseFunction* include:

    * Specifying a different CFL number or some other CFD input as a function
      of freestream Mach number or angle of attack
    * Using different meshes according to Mach number or, for example, a
      control surface deflection setting
    * Adding an extra *Label* or *tag* input variable that allows the user to
      select from various options using the run matrix
    * Translating and/or rotating part of the mesh according to values in the
      run matrix (although this is typically better supported with the
      *TriFunction*, *translation*, *rotation*, or *ConfigFunction* keys)
    * Defining a run matrix variable and using it to turn on/off powered
      boundary conditions
    * Creating a function to alter the PBS setting s for each case
    
An example input from the JSON file is shown below.

    .. code-block:: javascript
        
        "PythonPath": ["tools"],
        "Modules": ["tfm", "freeair"],
        "InitFunction": ["tfm.InitArchive"],
        "CaseFunction": ["freeair.ApplyLabel", "freeair.ApplyTag"]
        
This instructs Cape to run the functions :func:`freeair.ApplyLabel` and then
:func:`freeair.ApplyTag` after the case has been mostly set up but CFD input
files have not been written yet.  Some of the Python syntax for an example for
FUN3D is shown below.

    .. code-block:: python
    
        # Filter options based on the *Label* trajectory key
        def ApplyLabel(cntl, i):
            # Get nahdle to FUN3D namelist
            f3d = cntl.opts["Fun3D"]
            # Key section names
            nsp = "nonlinear_solver_parameters"
            lsp = "linear_solver_parameers"
            ifm = "inviscid_flux_method"
            # Get *Label* value from the run matrix for case *i*
            lbl = cntl.x.Label[i]
            # Select default label if empty
            if lbl:
                # Status update
                print("    Using Label '%s'" % lbl)
            else:
                # Check Mach number for this case
                mach = cntl.x.mach[i]
                # Filter the default value based on Mach
                if mach < 1.4:
                    # Select label to run uRANS
                    lbl = 'd'
                else:
                    # Different label to run RANS
                    lbl = 'b'
            # Actually change the settings ...
            
As can be seen from this example, the inputs to a *CaseFunction* are the
overall control object (:class:`pyCart.cart3d.Cart3d`,
:class:`pyOver.overflow.Overflow`, or :class:`pyfun.fun3d.Fun3d`) and the case
number *i*.

