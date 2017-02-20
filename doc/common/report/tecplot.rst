
.. _common-report-tecplot:

Automated Tecplot Figures
--------------------------
One special type of subfigure is the ``"Tecplot"`` subfigure.  This type
applies a Tecplot layout file (``*.lay``), modifies it slightly for each case,
and uses it to export an image.

The process for making a Tecplot subfigure is more complicated than other
subfigures.  Unlike other subfigures it requires quite a bit of setup other
than editing the JSON file.  The process can be outlined as follows.

    1. Go into a case folder with a solution.
    2. Launch Tecplot in this folder using the desired files and create the
       desired image.   Then save the layout file (or several layout files).
    3. Perform minor edits to the text of the layout file.
    4. Copy the file common location in the parent folder.
    5. Edit the JSON file to use the layout.
    
Of these, most steps are pretty straightforward and brief, but task 3 is a
little bit particular.  The process has the complexity it has because it has to
be useful for a range of conditions.  For example, the most up-to-date solution
file may have the iteration number in it, so the file  name in the layout file
would have to change (or something else happen).

Furthermore, there is an issue that's particular to Cart3D in that the cut
planes file (``cutPlanes.plt``) do not have pressure coefficient, and
calculating it requires the freestream Mach number (also not contained in the
file).  So Tecplot layouts in pyCart have the capability of setting the
freestream Mach number to that of the actual case.  As an example, the pressure
coefficient is defined in a typical Cart3D layout file using the following
text.

    .. code-block:: none
    
        $!ALTERDATA
          EQUATION = '{C_p} = ({Pressure} - 1/1.4) / (0.5*1.5*1.5)'
          
In order to make this calculation work for all cases (including those in which
the Mach number is not 1.5), it should be changed to the following.

    .. code-block:: none
    
        $!VarSet |Minf| = 1.5
        $!ALTERDATA
          EQUATION = '{C_p} = ({Pressure} - 1/1.4) / (0.5*|Minf|*|Minf|)'
          
Once this change has been made, pyCart will automatically change the Mach
number for each case.

Another task that is somewhat particular can be handled in either step 2 or 3. 
Before pyCart launches Tecplot, it creates some file links depending on the
solution files in a folder.  Specifically, it will link the most recent version
of a recurring PLT or other solution file to a fixed file name. These depend on
the flow solver, but here is an incomplete list.


    +----------------------------------------+----------------------------+
    | Source glob                            | Link Created               |
    +========================================+============================+
    | *Cart3D*                                                            |
    +----------------------------------------+----------------------------+
    | ``cutPlanes.[0-9]*.plt``               | ``cutPlanes.plt``          |
    +----------------------------------------+----------------------------+
    | ``Components.[0-9]*.plt``              | ``Components.i.plt``       |
    +----------------------------------------+----------------------------+
    | *FUN3D*                                                             |
    +----------------------------------------+----------------------------+
    | ``pyfun_tec_boundary_timestep*.plt``   | ``pyfun_tec_boundary.plt`` |
    +----------------------------------------+----------------------------+
    | ``pyfun_${surf}_timestep*.plt``        | ``pyfun_${surf}.plt``      |
    +----------------------------------------+----------------------------+
    | *OVERFLOW*                                                          |
    +----------------------------------------+----------------------------+
    | ``q.{avg,save,restart,[0-9]*}``        | ``q.pyover.p3d``           |
    +----------------------------------------+----------------------------+
    | ``q.[0-9]*[0-9]``                      | ``q.pyover.vol``           |
    +----------------------------------------+----------------------------+
    | ``q.[0-9]*.avg``                       | ``q.pyover.avg``           |
    +----------------------------------------+----------------------------+
    | ``q.[0-9]*.srf``                       | ``q.pyover.srf``           |
    +----------------------------------------+----------------------------+
    | ``x.{save,restart,[0-9]*}``            | ``x.pyover.p3d``           |
    +----------------------------------------+----------------------------+
    | ``x.{[0-9]*.srf,srf}``                 | ``x.pyover.srf``           |
    +----------------------------------------+----------------------------+
    
