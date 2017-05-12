
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

