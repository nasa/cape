
.. This documentation written by TestDriver()
   on 2022-01-20 at 01:41 PST

Test ``02_adiabatic``: PASS
=============================

This test PASSED on 2022-01-20 at 01:41 PST

This test is run in the folder:

    ``test/pyfun/02_adiabatic/``

and the working folder for the test is

    ``work/``

The commands executed by this test are

    .. code-block:: console

        $ pyfun -f pyFun01.json -I 0 --no-start
        $ cat bullet/m0.80a0.0b0.0/fun3d.00.nml
        $ pyfun -f pyFun02.json -I 1 --no-start
        $ cat bullet/m0.80a4.0b0.0/fun3d.00.nml
        $ pyfun -f pyFun03.json -I 2 --no-start
        $ cat bullet/m0.80a10.0b0.0/fun3d.00.nml

Command 1: Create Input Files (PASS)
-------------------------------------

:Command:
    .. code-block:: console

        $ pyfun -f pyFun01.json -I 0 --no-start

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.67 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 2: Check Adiabatic BC Setting : all boundaries (PASS)
--------------------------------------------------------------

:Command:
    .. code-block:: console

        $ cat bullet/m0.80a0.0b0.0/fun3d.00.nml

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.01 seconds
    * Cumulative time: 0.68 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        &project
            project_rootname = "arrow"
        /
        &raw_grid
          grid_format = 'aflr3'
          data_format = 'ascii'
        /
        &governing_equations
          eqn_type = 'compressible'
          viscous_terms = 'inviscid'
        /
        &reference_physical_properties
            mach_number = 0.8
            dim_input_type = "nondimensional"
            angle_of_attack = 0.0
            angle_of_yaw = 0.0
            reynolds_number = 698345.1757942521
            temperature = 264.07222222222225
        /
        &inviscid_flux_method
            first_order_iterations = 50
            flux_construction = "roe"
            freeze_limiter_iteration = 150
            flux_construction_lhs = "vanleer"
            flux_limiter = "hvanalbada"
        /
        &nonlinear_solver_parameters
            schedule_iteration = 1 100
            schedule_cfl = 0.1 100.0
        /
        &linear_solver_parameters
          linear_projection = .true.
          meanflow_sweeps   = 5
        /
        &code_run_control
            steps = 100
          stopping_tolerance = 1.0e-20
            restart_read = "off"
        /
        &global
          boundary_animation_freq = 100
            volume_animation_freq = -1
        /
        &sampling_parameters
          number_of_geometries = 2
          sampling_frequency(1) = 200
          sampling_frequency(2) = 200
          type_of_geometry(1) = 'plane'
          label(1) = 'plane-y0'
          plane_center(1:3,1) = 0.0 0.0 0.0
          plane_normal(1:3,1) = 0.0 1.0 0.0
          type_of_geometry(2) = 'plane'
          label(2) = 'plane-z0'
          plane_center(1:3,2) = 0.0 0.0 0.0
          plane_normal(1:3,2) = 0.0 0.0 1.0
        /
        &sampling_output_variables
          mach = .true.
        /
        &boundary_output_variables
          number_of_boundaries = -1
            cp = .true.
            boundary_list = "7-9"
            ptot = .true.
        /
         &component_parameters
            component_input(1) = "7-8"
            component_sref(1) = 3.14159
            component_xmc(1) = 1.75
            component_ymc(1) = 0.0
            component_zmc(1) = 0.0
            component_cref(1) = 2.0
            component_bref(1) = 2.0
            component_name(1) = "bullet_no_base"
            component_count(1) = -1
            component_input(2) = "7-9"
            component_sref(2) = 3.14159
            component_xmc(2) = 1.75
            component_ymc(2) = 0.0
            component_zmc(2) = 0.0
            component_cref(2) = 2.0
            component_bref(2) = 2.0
            component_name(2) = "bullet_total"
            component_count(2) = -1
            component_input(3) = "7"
            component_sref(3) = 3.14159
            component_xmc(3) = 1.75
            component_ymc(3) = 0.0
            component_zmc(3) = 0.0
            component_cref(3) = 2.0
            component_bref(3) = 2.0
            component_name(3) = "cap"
            component_count(3) = -1
            component_input(4) = "8"
            component_sref(4) = 3.14159
            component_xmc(4) = 1.75
            component_ymc(4) = 0.0
            component_zmc(4) = 0.0
            component_cref(4) = 2.0
            component_bref(4) = 2.0
            component_name(4) = "body"
            component_count(4) = -1
            component_input(5) = "9"
            component_sref(5) = 3.14159
            component_xmc(5) = 1.75
            component_ymc(5) = 0.0
            component_zmc(5) = 0.0
            component_cref(5) = 2.0
            component_bref(5) = 2.0
            component_name(5) = "base"
            component_count(5) = -1
            number_of_components = 5
         /
        
         &boundary_conditions
            wall_temp_flag(7) = .true.
            wall_temperature(7) = -1
            wall_temp_flag(8) = .true.
            wall_temperature(8) = -1
            wall_temp_flag(9) = .true.
            wall_temperature(9) = -1
         /
        
         &special_parameters
            large_angle_fix = "on"
         /
        
        

:STDERR:
    * **PASS**

Command 3: Create Input Files (PASS)
-------------------------------------

:Command:
    .. code-block:: console

        $ pyfun -f pyFun02.json -I 1 --no-start

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.70 seconds
    * Cumulative time: 1.38 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 4: Check Adiabatic BC Setting : specified wall selection (PASS)
------------------------------------------------------------------------

:Command:
    .. code-block:: console

        $ cat bullet/m0.80a4.0b0.0/fun3d.00.nml

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.01 seconds
    * Cumulative time: 1.39 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        &project
            project_rootname = "arrow"
        /
        &raw_grid
          grid_format = 'aflr3'
          data_format = 'ascii'
        /
        &governing_equations
          eqn_type = 'compressible'
          viscous_terms = 'inviscid'
        /
        &reference_physical_properties
            mach_number = 0.8
            dim_input_type = "nondimensional"
            angle_of_attack = 4.0
            angle_of_yaw = 0.0
            reynolds_number = 698345.1757942521
            temperature = 264.07222222222225
        /
        &inviscid_flux_method
            first_order_iterations = 50
            flux_construction = "roe"
            freeze_limiter_iteration = 150
            flux_construction_lhs = "vanleer"
            flux_limiter = "hvanalbada"
        /
        &nonlinear_solver_parameters
            schedule_iteration = 1 100
            schedule_cfl = 0.1 100.0
        /
        &linear_solver_parameters
          linear_projection = .true.
          meanflow_sweeps   = 5
        /
        &code_run_control
            steps = 100
          stopping_tolerance = 1.0e-20
            restart_read = "off"
        /
        &global
          boundary_animation_freq = 100
            volume_animation_freq = -1
        /
        &sampling_parameters
          number_of_geometries = 2
          sampling_frequency(1) = 200
          sampling_frequency(2) = 200
          type_of_geometry(1) = 'plane'
          label(1) = 'plane-y0'
          plane_center(1:3,1) = 0.0 0.0 0.0
          plane_normal(1:3,1) = 0.0 1.0 0.0
          type_of_geometry(2) = 'plane'
          label(2) = 'plane-z0'
          plane_center(1:3,2) = 0.0 0.0 0.0
          plane_normal(1:3,2) = 0.0 0.0 1.0
        /
        &sampling_output_variables
          mach = .true.
        /
        &boundary_output_variables
          number_of_boundaries = -1
            cp = .true.
            boundary_list = "7-9"
            ptot = .true.
        /
         &component_parameters
            component_input(1) = "7-8"
            component_sref(1) = 3.14159
            component_xmc(1) = 1.75
            component_ymc(1) = 0.0
            component_zmc(1) = 0.0
            component_cref(1) = 2.0
            component_bref(1) = 2.0
            component_name(1) = "bullet_no_base"
            component_count(1) = -1
            component_input(2) = "7-9"
            component_sref(2) = 3.14159
            component_xmc(2) = 1.75
            component_ymc(2) = 0.0
            component_zmc(2) = 0.0
            component_cref(2) = 2.0
            component_bref(2) = 2.0
            component_name(2) = "bullet_total"
            component_count(2) = -1
            component_input(3) = "7"
            component_sref(3) = 3.14159
            component_xmc(3) = 1.75
            component_ymc(3) = 0.0
            component_zmc(3) = 0.0
            component_cref(3) = 2.0
            component_bref(3) = 2.0
            component_name(3) = "cap"
            component_count(3) = -1
            component_input(4) = "8"
            component_sref(4) = 3.14159
            component_xmc(4) = 1.75
            component_ymc(4) = 0.0
            component_zmc(4) = 0.0
            component_cref(4) = 2.0
            component_bref(4) = 2.0
            component_name(4) = "body"
            component_count(4) = -1
            component_input(5) = "9"
            component_sref(5) = 3.14159
            component_xmc(5) = 1.75
            component_ymc(5) = 0.0
            component_zmc(5) = 0.0
            component_cref(5) = 2.0
            component_bref(5) = 2.0
            component_name(5) = "base"
            component_count(5) = -1
            number_of_components = 5
         /
        
         &boundary_conditions
            wall_temp_flag(8) = .true.
            wall_temperature(8) = -1
            wall_temp_flag(9) = .true.
            wall_temperature(9) = -1
         /
        
         &special_parameters
            large_angle_fix = "on"
         /
        
        

:STDERR:
    * **PASS**

Command 5: Create Input Files (PASS)
-------------------------------------

:Command:
    .. code-block:: console

        $ pyfun -f pyFun03.json -I 2 --no-start

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.69 seconds
    * Cumulative time: 2.07 seconds
:STDOUT:
    * **PASS**
:STDERR:
    * **PASS**

Command 6: Check Adiabatic BC Setting : no boundaries (PASS)
-------------------------------------------------------------

:Command:
    .. code-block:: console

        $ cat bullet/m0.80a10.0b0.0/fun3d.00.nml

:Return Code:
    * **PASS**
    * Output: ``0``
    * Target: ``0``
:Time Taken:
    * **PASS**
    * Command took 0.01 seconds
    * Cumulative time: 2.08 seconds
:STDOUT:
    * **PASS**
    * Target:

      .. code-block:: none

        &project
            project_rootname = "arrow"
        /
        &raw_grid
          grid_format = 'aflr3'
          data_format = 'ascii'
        /
        &governing_equations
          eqn_type = 'compressible'
          viscous_terms = 'inviscid'
        /
        &reference_physical_properties
            mach_number = 0.8
            dim_input_type = "nondimensional"
            angle_of_attack = 10.0
            angle_of_yaw = 0.0
            reynolds_number = 698345.1757942521
            temperature = 264.07222222222225
        /
        &inviscid_flux_method
            first_order_iterations = 50
            flux_construction = "roe"
            freeze_limiter_iteration = 150
            flux_construction_lhs = "vanleer"
            flux_limiter = "hvanalbada"
        /
        &nonlinear_solver_parameters
            schedule_iteration = 1 100
            schedule_cfl = 0.1 100.0
        /
        &linear_solver_parameters
          linear_projection = .true.
          meanflow_sweeps   = 5
        /
        &code_run_control
            steps = 100
          stopping_tolerance = 1.0e-20
            restart_read = "off"
        /
        &global
          boundary_animation_freq = 100
            volume_animation_freq = -1
        /
        &sampling_parameters
          number_of_geometries = 2
          sampling_frequency(1) = 200
          sampling_frequency(2) = 200
          type_of_geometry(1) = 'plane'
          label(1) = 'plane-y0'
          plane_center(1:3,1) = 0.0 0.0 0.0
          plane_normal(1:3,1) = 0.0 1.0 0.0
          type_of_geometry(2) = 'plane'
          label(2) = 'plane-z0'
          plane_center(1:3,2) = 0.0 0.0 0.0
          plane_normal(1:3,2) = 0.0 0.0 1.0
        /
        &sampling_output_variables
          mach = .true.
        /
        &boundary_output_variables
          number_of_boundaries = -1
            cp = .true.
            boundary_list = "7-9"
            ptot = .true.
        /
         &component_parameters
            component_input(1) = "7-8"
            component_sref(1) = 3.14159
            component_xmc(1) = 1.75
            component_ymc(1) = 0.0
            component_zmc(1) = 0.0
            component_cref(1) = 2.0
            component_bref(1) = 2.0
            component_name(1) = "bullet_no_base"
            component_count(1) = -1
            component_input(2) = "7-9"
            component_sref(2) = 3.14159
            component_xmc(2) = 1.75
            component_ymc(2) = 0.0
            component_zmc(2) = 0.0
            component_cref(2) = 2.0
            component_bref(2) = 2.0
            component_name(2) = "bullet_total"
            component_count(2) = -1
            component_input(3) = "7"
            component_sref(3) = 3.14159
            component_xmc(3) = 1.75
            component_ymc(3) = 0.0
            component_zmc(3) = 0.0
            component_cref(3) = 2.0
            component_bref(3) = 2.0
            component_name(3) = "cap"
            component_count(3) = -1
            component_input(4) = "8"
            component_sref(4) = 3.14159
            component_xmc(4) = 1.75
            component_ymc(4) = 0.0
            component_zmc(4) = 0.0
            component_cref(4) = 2.0
            component_bref(4) = 2.0
            component_name(4) = "body"
            component_count(4) = -1
            component_input(5) = "9"
            component_sref(5) = 3.14159
            component_xmc(5) = 1.75
            component_ymc(5) = 0.0
            component_zmc(5) = 0.0
            component_cref(5) = 2.0
            component_bref(5) = 2.0
            component_name(5) = "base"
            component_count(5) = -1
            number_of_components = 5
         /
        
         &special_parameters
            large_angle_fix = "on"
         /
        
        

:STDERR:
    * **PASS**

