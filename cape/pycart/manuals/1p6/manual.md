# Cart3D User Manual

> Source: [https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/cart3Dhome.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/cart3Dhome.html)
>
> Retrieved October 2, 2026. Images have been omitted. Links to pages
> included in this document have been converted to internal links.

---

<a id="page-cart3dhome-html"></a>

## Main Page

*Original page: [cart3Dhome.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/cart3Dhome.html)*

**Cart3D** v1.6   |  |  | | --- | --- | | **What is Cart3D?** | [**Remove extra frames**](http://www.nas.nasa.gov/publications/software/docs/cart3d/pages/index.html) |   **Cart3D** is a high-fidelity inviscid analysis package for conceptual and preliminary aerodynamic design. It allows users to perform automated CFD analysis on complex geometry and supports steady and time-dependent simulations. The package features robust and fully-integrated adjoint-driven mesh adaptation and includes utilities for geometry import, surface modeling and intersection, mesh generation, and post-processing of results. The main flow solvers run in parallel using either shared memory (OpenMP) or distributed memory (MPI) with excellent scalability. The package is highly automated so that geometry acquisition and mesh generation can usually be performed within a few minutes on most current desktop computers.     Geometry enters into Cart3D in the form of surface triangulations. These may be generated from within a CAD packages, from legacy surface triangulations or from structured surface grids. **Cart3D** uses [adaptively refined Cartesian grids](#page-cart3d-images-html) to discretize the space around a geometry and  cuts the geometry out of the set of "cut-cells" which actually intersect the surface triangulation. With adjoint-based adaptation, meshes automatically refine as simulations evolve to reduce error in aerodynamic outputs. Flow solvers run in parallel and to take full advantage of multi-core and multi-cpu hardware.     The **current release** **of Cart3D is** **v1.6.0.** This release can be downloaded from the [NASA Software Catalog](https://software.nasa.gov/) and includes significant improvements and new tools. There are upgrades to the [propulsion boundary conditions](#page-howto-samples-power-readme-html) and mass-flow objective functions for adaptation and design. The solvers are faster, more robust and have lower dissipation. **Cart3D's** integrated [adjoint-driven error-estimation and mesh refinement module](#page-adjoint-index-html) has also been enhanced with full integration of the propulsion boundary conditions and support for farfield objective functions when coupling with NASA's [sBOOM propagation code](https://software.nasa.gov/search/multi/aw/software/9/sboom). Also included are new tools for [simplifying deflection of control surfaces](#page-trix-index-html) and other geometric manipulations. "Extras" include tools for automating setup of sonic-boom simulations a driver for 6-DOF simulations and scripts and examples for integrating with NASA's QUEST Uncertainty Quantification package..  **What Platforms are Supported?**  Currently supported platforms include Linux (X86\_64, Linux 8) and MacOS (arm64, Apple Silicon). For information on other platforms, [contact us](mailto:michael.aftosmis@nasa.gov).    ---       [quick reference guide,](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/COMMAND_SUMMARY.pdf) [Images](#page-cart3d-images-html), [publications](#page-publications-publications-html), [adjoint-based mesh adaptation](#page-adjoint-index-html), [Power BCs](#page-howto-samples-power-readme-html), [geometry intersection](#page-surfacemodeling-html--auxprogs),   [mesh generation](#page-meshgeneration-html), [flowCart](#page-flowcart-html), [file formats,](#page-flowcart-files-html) [geometry import](#page-surfacemodeling-html--geomimport), [triangle formats](#page-cart3dtriangulations-html), [input/output files,](#page-flowcart-files-html) [HowTo's](#page-cart3d-howtos-html),  [boolean polytope intersection](#page-bool-intersection-html), [adaptive precision floating point arithmetic](#page-degeneracy-html), [algorithmic tie-breaking](#page-degeneracy-html--virtualpurturbationsos)    ---   **To request information on this page in a Section 508 accessible format, please contact  [access@mail.arc.nasa.gov](mailto:access@mail.arc.nasa.gov)  last update December 2024, [M. Aftosmis](https://www.nas.nasa.gov/aboutnas/staff/staff_maftosmis.html)** |

---

<a id="page-cart3d-news-html"></a>

## News

*Original page: [cart3d_news.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/cart3d_news.html)*

News  ---   **2026** **2025.02  New equilibrium gas models:** Released a major overhaul to the equilibrium solver, which now supports atmosphere tables from external sources like  [**Mutation++**](https://github.com/mutationpp/)  or [**CEA**](https://www.nasa.gov/glenn/research/chemical-equilibrium-with-applications/). The code comes with tables for both Earth and Mars. The update improves simulations of terrestrial hypersonic flow and allows modeling of entry and flight on the Red Planet. Other extraterrestrial atmospheres also are also available.  ---   **2025** **2025.02  Cart3D Design v1.0 Released:** This release introduces new capabilities for ground noise-based design of low-boom vehicles and enables symbolic manipulation of geometry. In addition to its native gradient-based optimizer and built-in SNOPT support, this release also allows for external optimizers such as SciPy or CAPS/ESP.  ---   **2024** **2024.12  Cart3D v1.6.0 Released:**  This release includes solver improvements in for the flow and adjoint solvers. Tool improvement allow more flexible manipulation of component-based geometry and tighter integration with external tools. These include better integration with NASA’s sBOOM atmospheric propagation code and it allows mesh adaptation and design to be driven using ground-noise outputs in sonic boom applications. Supported platforms include Linux (AMD and Intel) and MacOS (Apple Silicon). See the listing in the [NASA Software Catalog](https://software.nasa.gov) for download.     **2024.11  NASA SC|24 feature on Supercomputing in Supersonics:** The article ["Supercomputing in Supersonics: Analyzing Noise Predictions from High-Speed Aircraft"](https://www.nas.nasa.gov/SC24/research/project02.php) focuses on the role of uncertainty quantification in Certification-by-Analysis for low-boom aircraft. Uncertainty in the boom carpet of NASA's Quesst (X-59) aircraft has [been extensively studied using Cart3D.](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2024-4671.pdf)     **2024.02  SpaceWorks releases Cart3D Plus:**  SpaceWorks Enterprises Inc. released their commercial product "Cart3D Plus" with support for HPC nodes and clusters. This product includes integration with their "Scorch" aeroheating code. SpaceWorks is an Atlanta, GA based company and releases Cart3D under license from the NASA Ames Technology Partnerships Office.   ---   **2023**  **2023.04  NASA news feature:** *"Supercomputers Aid Quesst Researchers in Predicting X-59’s Sound**"*, Cart3D simulations of the X-59 are highlighted in [this outreach article written by John Gould](https://www.nasa.gov/aeroresearch/supercomputers-aid-quesst-researchers-in-predicting-x-59s-sound) that was featured on both the main NASA home page and that of the [ARMD's Quesst mission](https://www.nasa.gov/X59).   ---   **2022**  ****2022.11 Cart3D v1.5.9 Released**** with a host of improvements and new tools. These include updates to the boundary conditions for modeling propulsion, target Mach number steering and new farfield objectives for adaptation and design when solving sonic boom problems. There are also a host of minor solver improvements for greater robustness, accuracy and speed. See the listing in the [NASA Software Catalog](https://software.nasa.gov/software/ARC-14275-1) for download.    ******2022.11 Cart3D IO libraries approved for release****** under NASA's Open Source license. The NASA chief council and Ames' Technology Partnerships division has approved release of the Cart3D IO libraries under version 1.3 of the [NASA Open Source Agreement](https://en.wikipedia.org/wiki/NASA_Open_Source_Agreement). These will be available for download from the [NASA Software Catalog](https://software.nasa.gov) before the end of the calendar year.    **2022.11 NASA SC|22 feature**  on CFD tools for design of low-boom supersonic aircraft *["Advancing NASA Software for Sonic Boom Prediction"](https://www.nas.nasa.gov/SC22/research/project7.html)* featuring discussion of contributions made by Cart3D and other NASA CFD tools.    **2022.08 NASA CAPE software released with Cart3D support.** CAPE is a python-based run manager for Cart3D (pycart) and other CFD tools which can be used to create large aero databases. The aero database for the NASA's Artemis 1 mission used over 20,000 Cart3D simulations. Click [here](https://github.com/nasa/cape/blob/main/README.rst) for general information on CAPE and see a simple [Cart3D Tutorial here](https://github.com/nasa-ddalle/pycart01-bullet).    **2022.03** **NASA News item** on "Ames' Contributions to the X-59 Quiet SuperSonic Technology Aircraft" [See the full article.](https://www.nasa.gov/feature/ames/x-59)    **2022.01 Cover of Aviation Week:** Wind tunnel tests and CFD simulations of the X-59 made the cover of [Aviation Week and Space Technology](https://aviationweek.com) January 24, 2022. The article "[Wind-Tunnel Tests Validate Low-Boom X-59 Computational Fluid Dynamics](https://aviationweek.com/aerospace/emerging-technologies/wind-tunnel-tests-validate-low-boom-x-59-computational-fluid)" discusses the agreement between CFD predictions and the tunnel. A special feature within the article used numerous images from Cart3D simulations showing details of the flow and surface pressure contours.      ---    **2021**  ****2021.12 [Aerospace America's](https://aerospaceamerica.aiaa.org)****annual review issue had a great section on emerging  [low boom technologies](https://aerospaceamerica.aiaa.org/year-in-review/low-boom-technologies-gain-ground-with-improved-measurement-tools-and-more-research/). The lead feature identified supersonic design methodologies developed using Cart3D as one of the most promising low-boom technologies of 2021**.**    ****2021.06 NAS Software Link.**** The NASA Advanced Supercomputing Center posted a general information link to Cart3D on the software tab of[the NAS home page](https://www.nas.nasa.gov).  The info bar describes some recent in-house  work and documentation as well as information for getting Cart3D through NASA's software portal. see: [*https://www.nas.nasa.gov/software/cart3d.html*](https://www.nas.nasa.gov/software/cart3d.html)    **2021.04 NASA Feature "Saving the Earth from Asteroids".** NASA [published a feature article](https://www.nasa.gov/feature/saving-earth-from-asteroids) on planetary defense featuring many simulations done using Cart3D on the main NASA home page as part of an information campaign for Asteroid Day 2021.  **2021.03 NASA supercomputing feature**  on CFD tools for design of low-boom supersonic aircraft with a description of [some of the key analyses done using Cart3D](https://www.nas.nasa.gov/pubs/stories/2021/feature_X-59.html) and other NASA CFD tools.    **2021.02 NAS Researchers Bring Asteroid Simulation Down to Earth.** NASA Advanced Supercomputing Division [press release on asteroid threat assessment and the role of HPC simulation.](https://www.nas.nasa.gov/publications/articles/feature_asteroid_threat_assessment_part_1.html) This article includes a simulation of a ground-impacting asteroid done using Cart3D.       ---    **2020** **2020.12 [Featured article on NASA](https://www.nasa.gov/aeroresearch/aeronautical-artwork-computer-simulation-or-both)**[**Aeronautics** home page](https://www.nasa.gov/aeroresearch/aeronautical-artwork-computer-simulation-or-both). The X-59 is getting a lot of press these days, and some work done by the crew at Ames on the NAS supercomputers found its way to NASA's press department.    ********2020.11 NASA's show at SC|20******** generated in a lot of interest this year due in large part to ARMD's upcoming X-59 low-boom flight demonstrator. Cart3D simulations have played a central role in the X-59's development. [Read about it here.](https://www.nas.nasa.gov/SC20/demos/demo3.html)    ****2020.11 Cart3D Design Framework Update.****Cart3D\_Design\_v0.9.8 was released to go with the Cart3D v1.5.7 update of Cart3D. This update to the design framework recognizes and supports multiple inlets/exists with all the power boundary options in Cart3D, it also has support for mass flow objective functions and constraints. see the [announcement in the Cart3D discussion group](https://groups.google.com/g/cart3d/c/vkbUbU-KUkw).     **2020.10 Released Cart3D v1.5.7** with a several significant improvements and new tools. These include updates to the boundary conditions for modeling propulsion, mass flow rate steering and  mass flow rate objectives for adaptation and design. Also included are new tools simplifying deflection of control surfaces and other geometric manipulations. There are also a host of minor solver improvements for greater robustness, accuracy and speed. See the listing in the [NASA Software Catalog](https://software.nasa.gov/software/ARC-14275-1) for download.     **2020.09 Cart3D** simulations were featured on **NASA Aeronautics** official Facebook page, get the full story at <https://go.nasa.gov/2T3fghs>.     ---    **2019**  ****2019.11 NASA's show at SC|19**** featured a theater-style presentation describing Cart3D's role in design and analysis of the X-59 low-boom flight demonstrator. The NASA's advanced supercomputing division (NAS) ran a short [feature on its official SC|19 website.](https://www.nas.nasa.gov/SC19/demos/demo20.html) There are also some really cool visualizations of the full [3-dimensional flow field around a prototype of the X-59](https://www.nas.nasa.gov/SC19/gallery.html#prettyPhoto[aero]/14/) design and [some other aircraft](https://www.nas.nasa.gov/SC19/gallery.html#prettyPhoto[aero]/15/) using a newly developed computational schlieren visualization technique.       **2019.06 Inside HPC ran a story** titled "Supercomputing Asteroid Impacts for Planetary Defenses" [discussing Cart3D's role in predicting blast damage](https://insidehpc.com/2019/07/supercomputing-asteroid-impacts-for-planetary-defenses/) from the impact of large asteroids as part of NASA's Asteroid Threat Assessment Program. This discussion is related to work presented at SC|18 which is on-line [here](https://www.nas.nasa.gov/SC18/demos/demo8.html) and also [here](https://www.nas.nasa.gov/assets/pdf/ams/2016/AMS_20160922_Robertson.pdf). Earlier work on this topic had been discussed in articles in TechTimes [here](https://www.techtimes.com/articles/210931/20170701/nasa-supercomputer-simulations-may-help-reduce-damage-caused-by-asteroid-impacts.htm) and [elsewhere on the web](https://www.google.com/search?safe=off&client=safari&rls=en&ei=qAomXeXdEtmDtQaow7C4Ag&q=nasa+cart3d+asteroids&oq=nasa+cart3d+asteroids&gs_l=psy-ab.3...5779.5779..5947...0.0..0.136.136.0j1......0....1..gws-wiz.55u1OGt1JaU).   ---    **2018**  **2018.05 Released Cart3D v1.5.5** with a host of major upgrades. Chief among these are new, simpler boundary conditions for modeling propulsion, including mass flow rate steering and new mass flow rate objectives for adaptation and design. Lower dissipation in the solver, tighter error control with adaptation and much more.    2018.05 [AIAA 2017-3255](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa_2017-3255.pdf) named **AIAA 2018 Applied Aerodynamics Best Paper** from the 2017 AIAA AVIATION forum in Denver, Colorado. This paper discussed the application of Cart3D to various cases the second AIAA Sonic Boom Prediction Workshop which was held in January of 2017.    2018.01 Cart3D listed as "Featured Software" in NASA's 2017/2018 [Software Catalog](https://technology.nasa.gov/NASA_Software_Catalog_2017-18.pdf)     ---    **2017**     **2017.11** **Aviation Week and Space Technology** ran an article titled "NASA Supercomputers Unlock Flow Secrets" in their "This Week in Technology" that mentioned some of the asteroid work going on with Cart3D. See the article [here](http://awin.aviationweek.com/ArticlesStory.aspx?keyWord=NASA%20Supercomputers%20Unlock%20Flow%20Secrets:&id=e3185fd8-f971-49ee-a502-cc6994455b78) (requires on-line access to Av.Week). If you don't have on-line access, you can still checkout the [video that Av.Week posted on YouTube](https://www.youtube.com/watch?v=pc37BvEU73g&feature=youtu.be).    **2017.11** One of the **featured demo**'s at the NASA booth at SC|17 was a short presentation titled "Simulating Atmospheric Impacts: From Pebble-to Mountain-Size Meteoroids" which showed some of the ways Cart3D simulations are being used to model meteoroid entry. [Here is a quick summary](https://www.nas.nasa.gov/SC17/demos/demo22.html).    **2017.07** "**NASA Researchers Rock the World of Asteroid Simulation**" Feature story on meteor simulation with Cart3D. The full article is [here.](https://www.nas.nasa.gov/publications/articles/feature_asteroid_simulations.html)  The full article as well as a shorter version was picked up by many news outlets, see it [here.](https://www.nasa.gov/ames/feature/nasa-simulates-asteroid-impacts-to-help-identify-possible-life-threatening-events)    ****2017.03**** Cart3D used to predict the sonic boom resulting from the atmospheric entry of **[fist-sized meteoroids](https://www.nas.nasa.gov/assets/pdf/papers/1-s2.0-S0032063316304469-main.pdf)****.**    2017.01 [NASA Spinoff article](https://spinoff.nasa.gov/Spinoff2017/t_2.html) "Design Software Shapes Future Sonic Booms."    2017.01 Cart3D is #6 on NASA's list of "[Top 20 Requested Software Titles](https://software.nasa.gov/NASA_Software_Catalog_2015-16.pdf)" in the 2015/2016 Software Catalog       ---     **2016**    2016.09 [NASA TV](https://www.nasa.gov/ames/image-feature/nasa-aerodynamics-simulations-reshape-flight-for-fuel-efficiency)[**feature**](https://www.nasa.gov/ames/image-feature/nasa-aerodynamics-simulations-reshape-flight-for-fuel-efficiency) "NASA Aerodynamics Simulations Reshape Flight for Fuel Efficiency"    2016.08 **[Feature story](http://www.nas.nasa.gov/publications/articles/feature_Flaps_VCCTEF_Aftosmis.html)** on multidisciplinary shape optimization including static aeroelastic effects using Cart3D headlines the NASA Advanced Supercomputing Division's font page.    2016.08 Initiated alpha-test program for "*c3d\_Aeroblender*" with [Blender](https://www.blender.org) Plugins for Cart3D. These permit both I/O for Cart3D triangulations & basic parametric shape manipulation.    2016.07 Released Cart3D v1.5.1 with minor bug fixes and a short list of new features     2016.03 Hosted the Cart3D Advanced Design Workshop  at NASA Ames Research Center from 8-9 March covering advanced topics using design framework release v0.9.5 and Cart3D v1.5.   ---    **2015**  2015.09 Released  Cart3D Design Framework v0.95 for beta testing including native aerodynamic constraints, progressive optimization and multiple adjoints.  2015.09 Released Cart3D v1.5 with major upgrades including simulation of unsteady flow, automatic mesh growth for error-driven adaptation, equivalent area functionals, buoyancy driven atmospheric flows and more.    2015.03 Cart3D recognized in TechBriefs' list of  "[Hot 100](http://www.techbriefs.com/component/content/article/ntb/features/technologies/21122) technologies". [Click here](http://www.techbriefs.com/dl/HOT100/Cart3D.pdf) to read more!     ---    **2014**   2014.01 Participated in the [1st AIAA Sonic Boom Prediction Workshop](http://lbpw.larc.nasa.gov/sbpw1/). Click [here](http://arc.aiaa.org/doi/10.2514/6.2014-0558) for more.   ---     **2013**    2013.06 Released major upgrade to [Cart3D v1.4.9](https://groups.google.com/d/msg/cart3d/yY2h3AushKY/TWOy7NkX8hcJ),  with integrated adjoint-based mesh adaptation and error-estimation.     2013.06 Released update to Cart3D Design Framework for consistency with v1.4.9 release     2013.03 Updated beta release of Cart3D design framework Cart3D\_Design (v0.9.1)     2013.01 Article in Spinoff 2012 see <http://spinoff.nasa.gov/Spinoff2012/> around [page 115](http://spinoff.nasa.gov/Spinoff2012/it_1.html)   ---     **2012**  [AIAA-2011-3500](https://www.nas.nasa.gov/assets/pdf/staff/Aftosmis_A_Adjoint-Based_Low-Boom_Design_with_Cart3D.pdf) Awarded "Best Technical Paper" at the AIAA Applied Aerodynamics Conference (held in June 2011, Honolulu, HI). This paper discusses the use of Cart3D Adjoint-Based Design Framework for shaping supersonic aircraft with low sonic boom.    Release of [Cart3D\_v1.4.7](https://groups.google.com/forum/?fromgroups#%21topic/cart3d/5E5lWW664YQ) with integrated adjoint-based mesh adaptation and error-estimation    Close-Out Beta Testing of Cart3D\_AERO (v0.95) module and slate for full-scale release   ---     **2011**  Cart3D Workshop: Getting Started with the Cart3D Adjoint-Based Design Framework.  Hosted at NASA Ames, 2-3 November 2011. This is a kick-off workshop for the Cart3D design framework     Open Beta Testing of Cart3D\_AERO (v0.95) module for adjoint-based mesh adaptation and error-estimation    Release of Cart3D\_v1.4.5   ---    **2010** [Desktop Aeronautics](http://www.desktop.aero/index.php) has acquired a license to sell Cart3D for commercial use through the [NASA Ames Technology Partnerships Division.](http://technology.arc.nasa.gov/index.cfm) Here is a link to their [Cart3D product page](http://www.desktop.aero/cart3d.php).   ---    **2009** Cooperative agreement with Optimal Solutions Software LLC who's Sculptor software product now supports native Cart3D Geometry. See a video at: <http://www.youtube.com/watch?v=hUlFmq0xIK4&feature=related>    Workshop II: **Getting Started with Adjoint-Based Mesh Adaptation and Error Estimation in Cart3D**. Hosted at NASA Ames, Feb, 2009. This is a kick-off workshop for [phase II of the beta program](http://groups.google.com/group/cart3d/browse_thread/thread/dbc68041226d7407) for the Cart3D Adjoint-module.   ---     **2008**   Workshop I: **Getting Started with Adjoint-Based Mesh Adaptation and Error Estimation in Cart3D** Hosted at NASA Ames, Dec. 18. This was a government-only workshop for the [phase I  beta release](http://groups.google.com/group/cart3d/browse_thread/thread/dbc68041226d7407) of the Cart3D Adjoint-module.    **Cart3D v 1.4 released**, Dec 2008  Performance improvements, support for on-the-fly component-wise force extraction, point and line sensors, [see the full announcement](http://groups.google.com/group/cart3d/browse_thread/thread/ec24efcc6ad572d6).    **Beta Release of Cart3D Adjoint-module** Dec 2008  Output functionals drive mesh adaptation to meet user-specified error bounds. See [details of the beta program](http://groups.google.com/group/cart3d/browse_thread/thread/dbc68041226d7407). Get started with background reading [here](https://www.nas.nasa.gov/assets/pdf/staff/Aftosmis_A_Adjoint-Based_Adaptive_Mesh_Refinement_for_Complex_Geometries.pdf).    Cart3D meets sonic-boom propagation milestones in under 2hrs on a desktop workstation. Read about it [here](https://www.nas.nasa.gov/assets/pdf/staff/Aftosmis_A_Adjoint-Based_Adaptive_Mesh_Refinement_for_Sonic_Boom_Prediction.pdf).     Cart3D's importance to NASA's future discussed in "[Petascale Computing: Algorithms and Applications](http://www.amazon.com/Petascale-Computing-Algorithms-Applications-Computational/dp/1584889098)", (Chapman & Hall/CRC Computational Science Series) ed. D. Bader, Jan. 2008.   ---    **2007** **flowCart's hybrid parallel programming paradigm** written-up and analyzed in the OpenMP textbook "[Using OpenMP: Portable Shared Memory Parallel Programming](http://www.amazon.com/Using-OpenMP-Programming-Engineering-Computation/dp/0262533022/ref=sr_1_1?ie=UTF8&s=books&qid=1228421788&sr=1-1)" by Chapman, Jost and van der Pas. Oct. 2007 Cart3D used for analysis of [Orion Launch Abort Vehicle](http://en.wikipedia.org/wiki/Orion_%28spacecraft%29). See the [technical paper](https://www.nas.nasa.gov/assets/pdf/staff/Aftosmis_A_Effects_of_Jet-Interaction_on_Pitch_Control_of_a_Launch_Abort_Vehicle.pdf) on this work. Cart3D used in producing **Orion Aero database** for **Constellation** by NASA's Exploration Systems Mission Directorate   ---     **2006** **Launch of [Cart3D-Discuss](http://groups.google.com/group/cart3d) discussion forum.** Users and observers invited to browse on-line newsgroup, Feb, 2006    **Software for Automated Generation of Cartesian Grids** NASA Tech Briefs, Vol.30 No. 1, Jan. 2006 **Cart3D Hits 2.5 TFLOP/s on 2016 CPUs of NASA's Columbia Supercomputer** [Compare performance](https://www.nas.nasa.gov/assets/pdf/staff/Aftosmis_A_A_Detailed_Performance_Characterization_of_Columbia_using_Aeronautics_Benchmarks_and_Applications.pdf) with that of several top NASA solvers, Jan 2006.   ---     **2005** **Supercomputing SC|05,** Seattle, WA, Nov 2005  Best technical paper award "2005 News and Highlights"  [SC|05 Press release](http://sc05.supercomputing.org/news/press_releases_11172005.php)      **Cart3D v1.3.5 released,** Aug 2004  Improved limiters in flowCart, user-specified ratio of specific heats.  ---    **2004** **Aerodynamic Performance Databases,** Oct 2004  **Cart3D v1.3 released,** Jun 2004  reorder within "cubes", inlet/exit BCs, faster codes and more.   ---    **2003** **Aviation Week & Space Technology**  "Correspondence". Sept. 15, 2003    Contours   Plot of the month, Issue 21, 2003  **Aviation Week & Space Technology**  "NASA Recouping after Columbia Board's Criticism", p. 22. Sept. 1, 2003  The Columbia Accident Investigation Board Report - Volume 1, Chap 3, Fig 3.4-6   Debris trajectory showing strike on RCC 8. **[Gridpoints Magazine,](http://www.nas.nasa.gov/About/Gridpoints/gridpoints.html)**  winter 2003 issue **Aviation Week & Space Technology**  2003 Aerospace Source Book, p. 426. Jan 2003  --- **2002** **Ames Astrogram News Article**,   September 2002 **Cart3D Named *2002 NASA Software of the Year***     NASA Ames Press Release, 2 Aug.  2002    [NASA HQ Press Release, 2 Aug. 2002](ftp://ftp.hq.nasa.gov/pub/pao/pressrel/2002/02-148.txt)    ANSYS Press Release  **AIAA Best Paper Award**  for AIAA Paper 2002-0863  **Release** of cart3d.v1.1, Jan. 2002   ---    **2****001** **NAS News Article (Weekly)**  News item 12/07/2001 **Release** of cart3d.v1.0b, Oct. 2001  **NASA Ames CTO** [awards exclusive and non-exclusive licenses](#page-licensing-html--who) for ***Cart3D***  **Release** of cart3d.v1.0a, Feb. 2001     ---    **2000** **[Gridpoints Magazine (Quarterly)](http://www.nas.nasa.gov/About/Gridpoints/pastgridpoints.html)**  Feature Story, Summer 2000, Vol.1, No.3 **Release** of cart3d.v1.0, Aug. 2000  **A New Generation of Grids**  NAS Front page feature story. June 2000   ---     **1999** **Release** of cart3d.beta.21.09.99, Sep., 1999 **Robust and Efficient Generation of Cartesian Meshes for CFD**  *ARC Commercial Technology Office, Technical Oppurtunities, Aug 1999*   ---     **1998** **Release** of cart3d.beta.09.09.98   ---     **1995** **A Rapid-Deployment Force For CFD: Cartesian Grids**  *SIAM News, Vol. 28, No. 10, December 1995* **Cartesian Mesh Simulations for Complex Geometry**  *1995 NAS Technical Summary*   ---    *last update, December, 2024.* |

---

<a id="page-cart3d-images-html"></a>

## Examples

*Original page: [cart3d_images.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/cart3d_images.html)*

Images from various projects  This page illustrates some applications of Cart3D. These are in no particular order, and include both validation & in-house projects as well as some examples from users. Click on any image to see a larger version.   ---   |  |  | | --- | --- | | SLS and Artemis | X-59 QueSST at cruise | | Time-dependent flow | Hypersonic formation flight- raycart viz | | Low-boom design and analysis | Unsteady flow simulation | | Computational schlieren of T38s   at Mach 1.5 using raycart | Boom-propagation with running  engines and inlets | | Adjoint-based adaptation. 2D NACA 0012  Mach 0.95 | LAV with ACMs firing, Mach 4.0 | | Unsteady Blast wave past LAV geometry | 2D Unsteady flow example | | Mach-alpha sweep on LAV  with 3 different thrust settings | 121 Components, 5.61M cells | | 18 Components, 1.65M cells | 82 Components, 5.81M cells | | 1 Component, 1.2M cells | Isobars in discrete solution | | Component based shuttle geometry | Mesh partitions, using SFC partitioner |  ---     *last update November 2022,   [M. Aftosmis](https://www.nas.nasa.gov/about/staff/maftosmis.html)* |

---

<a id="page-publications-publications-html"></a>

## Publications

*Original page: [publications/publications.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/publications.html)*

---   this is a selected listing,  [click here](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aftosmis_pubs_2025.09.pdf) for a more complete list  ---      Selected Publications    - [*"J. of Aircraft #(#) Direct Optimization of Sonic Boom Ground Noise Via a   Coupled-Adjoint Method."*](https://ntrs.nasa.gov/api/citations/20250009072/downloads/2025_JournalPaper_DLR.pdf)       – Rodriguez, Aftosmis, Nemec, Rallabhandi, and Spurlock (4.1Mb, \*pdf).   Mar. 2026. - [*AIAA 2025-0772.     "Sonic Boom Ground Noise Minimization via the Adjoint Method."*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2025-0772.pdf)           – Rodriguez, D., Aftosmis, M.J., Nemec, M. and Rallabhandi, S. (7.2Mb, \*pdf).     Jan. 2025. - [*ICCFD12-[10-A-01]       "Implicit preconditioning for explicit multigrid solvers on cut-cell Cartesian meshes."*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/iccfd_2024-10-A-01.pdf)               – Chiew, J., Aftosmis, M.J. (1.4Mb, \*pdf).       July 2024. - [*AIAA 2024-4669         "Discretization error estimation and control for farfield acoustic signatures."*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2024-4669.pdf)                   – Rallabhandi, S., Nemec, M, Aftosmis, M.J. (1.9Mb, \*pdf).         July 2024. - [*AIAA 2024-4671           "Acoustic signature uncertainty quantification for quiet supersonic aircraft."*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2024-4671.pdf)                       –  Nemec, M., Bedonian, G., Aftosmis, M.J. (5.3Mb, \*pdf).           July 2024. - [*AIAA 2023-3727             "Recent enhancements to modeling sonic boom propagation using augmented Burgers' equation."*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2023-3727.pdf)                           – Rallabhandi, S., Nemec, M, Aftosmis, M.J. (6.9Mb, \*pdf).             June 2023. - [*AIAA 2023-1792 "Parallel mesh               adaptation for unsteady blast simulations on Cartesian meshes"*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA-2023-1792_400dpi.pdf)                – Spurlock, W., Aftosmis, M., Chiew, J., Nemec, M. (9.2Mb, \*pdf).               Jan. 2023. - [*AIAA                 2022-4085 "Goal-Oriented Discretization Error Control in Coupled                 Nearfield-Farfield Low-Boom Simulations"*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2022-4085_c3d_sBOOM.pdf)                  – Nemec, M, Aftosmis, M.J., Rallabhandi, S. (8.7Mb, \*pdf).                 June                 2022. - [*VFS 2022:                   "Medium-fidelity CFD modeling of multi-copter wakes for airborne                   sensor measurements."*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/VFS2022_ChiewAftosmis_final.pdf)                    – Chiew, J.J., Aftosmis, M.J., Manies, K.L. (5.7Mb,                   \*pdf),                   78th VFS, May 2022- [*AIAA 2021-2624,                     "Integral Velocity Sampling for Unsteady Rotor Models on Cartesian                     Meshes"*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2021-2624_ChiewAftosmis.pdf)                      – Chiew, J.J., Aftosmis, M.J., (4.1Mb,                     \*pdf), Jan                     2022 - [*J. of Aircraft, 59(3) "Cartesian Mesh Simulations for the                       Third AIAA Sonic Boom Prediction Workshop.*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/2021_JofA_SBPW3.pdf)"                        – Spurlock, W.M., Aftosmis, M.J., and Nemec, M.                       (9.8Mb, \*pdf), Oct.                       2021- [*AIAA 2021-2753.                         "Uncertainty Estimates for Sonic-Boom Pressure Signatures and                         Loudness Carpets"*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2021-2753_nemec.pdf)                          – Nemec, M., Aftosmis, M.J., Smith, L., (4.4Mb, \*pdf), Aug.                         2021- [*AIAA 2021-3030,                           "Adjoint-Based Minimization of X-59 Sonic Boom Noise via Control                           Surfaces"*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2021-3030_grndNoise.pdf)                            – Rodriguez, D., Aftosmis, M.J.,                           Nemec, M.J., and                           Spurlock, W., (1.7Mb,                           \*pdf), Aug. 2021- [*AIAA 2019-3488,                             "Adjoint-Based Mesh Adaptation and Shape Optimization for                             Simulations with Propulsion"*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2019-3488_mn_dlr_ma.pdf)                              – Nemec, M., Rodriguez, D., and Aftosmis, M.J. (4.4Mb,                             \*pdf), June 2019- [*J. of Aircraft, 56(3) "Nearfield summary and statistical                               analysis of the second AIAA Sonic Boom Prediction Workshop.*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/JoA_Park_Nemec_2018.pdf)"                                – M. Park and M. Nemec, May, 2018. (2.4Mb, \*pdf),- *["A Conservative,                                 Scalable, Space-Time Blade Element Rotor Model for Multi-rotor                                 Vehicles"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AHS_2018.01_chiew.pdf)*                                   – Chiew, J., and Aftosmis, M. *[AHS 2018,](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AHS_2018.01_chiew.pdf)* (3.4Mb, \*pdf), Jan.                                 2018 - [*AIAA Paper                                   2018-0334,  "Formulation and implementation of inflow/outflow                                   boundary conditions to simulate propulsive                                   effects."*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2018-0334.pdf)                                    – Rodriguez, D. L., Aftosmis, M.J., & Nemec, M.                                   (5.3Mb, \*pdf), Jan. 20180- *"[Simulation-based                                     height of burst map for asteroid airburst damage                                     prediction.](https://doi.org/10.1016/j.actaastro.2017.12.021)"*                                      – Aftosmis, M.J., Mathias, D.L., and Tarano, A.M.,                                     [Acta Astronautica,](https://arc.aiaa.org/doi/10.2514/6.2017-0528)                                       (1.8Mb, \*pdf), 2017- *["An ODE-Based Wall Model                                       for Turbulent Flow Simulations"](https://arc.aiaa.org/doi/10.2514/1.J056151)*                                        – Berger, M., and Aftosmis, M.,                                       *[AIAA                                       Jol. **56**(2)](https://arc.aiaa.org/doi/10.2514/1.J056151)*, (10.6mb \*.pdf),                                       2017- *["Numerical Prediction of Meteoric Infrasound Signatures"](https://doi.org/10.1016/j.pss.2017.03.003)*                                          – Nemec, M., Aftosmis, M.J, Brown, P.G.,                                         [*Planetary and Space Science,*](https://doi.org/10.1016/j.pss.2017.03.003),   (2.5Mb, \*.pdf), 2017- *[AIAA 2017-3255, "Cart3D                                           Simulations for the Second AIAA Sonic Boom Prediction                                           Workshop"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa_2017-3255.pdf)*                                            – Anderson, G.R., Aftosmis, M.J., Nemec, M., (6.3Mb,                                           \*pdf)- *[AIAA 2017-3189,                                             "Computational Modeling of Meteor-Generated Ground Pressure                                             Signatures"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa_2017-3189.pdf)*                                              – Nemec, M., Aftosmis, M.J, Brown, P.G.,  (2.5mb                                             \*.pdf), Jan 2016- *[AIAA 2017-0528, "An                                               ODE-based Wall Model for Turbulent Flow Simulations"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2017-0528_berger_aftosmis.pdf)*                                                – Berger M.J. and  Aftosmis, M.J, (6mb \*.pdf), Jan                                               2016 - *[NASA/TP-2016- 219422,                                                 JANNAF/GL-2016-0001 Advances in Verification, Validation, and                                                 Uncertainty Quantification](http://hdl.handle.net/2060/20160013550)*                                                  – Nemec, M., and Aftosmis M.J, 2016. - *[J. of Aircraft, 53(6).                                                   "Optimization of Flexible Wings with Distributed Flaps at                                                   Off-Design Conditions"](http://arc.aiaa.org/doi/10.2514/1.C033535)**,*                                                    – Rodriguez, D. L., Aftosmis, M.J., Nemec, M.,                                                   and Anderson, G.R., - *[AIAA 2016-0998,                                                     "Numerical Simulation of Bolide Entry with Ground Footprint                                                     Prediction",](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2016-0998_aftosmis.pdf)*                                                      – Aftosmis, M.J., Nemec, M., Mathias, D.L., and Berger, M.J.,                                                     (5Mb, \*.pdf), Jan. 2016 - *[AIAA 2015-3605, "Skylon                                                       Aerodynamics and SABRE plumes](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2015-3605_Skylon_final.pdf)*                                                        – Mehta, U., Aftosmis, M.J., Bowles, J.V., and Pandya, S.A.                                                       (7Mb, \*.pdf), Jul. 2015.- [AIAA 2015-0398, "Adaptive Shape Control for Aerodynamic                                                         Design."](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2015-0398_final.pdf)                                                          – Anderson, G.R., and Aftosmis, M.J. (3.6Mb, \*.pdf), Jan.                                                         2015 - [AIAA 2015-1409,                                                           "Optimized Off-Design Performance of Flexible Wings with Continuous                                                           Trailing-Edge Flaps."](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2015-1409_dlr.pdf)                                                            – Rodriguez, D. L., Aftosmis, M.J., Nemec, M., and Anderson,                                                           G.R., (4.5Mb, \*.pdf), Jan. 2014 - [AIAA                                                             2015-1719, "Aerodynamic Shape Optimization Benchmarks with Error                                                             Control and Automatic Parameterization."](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA-2015-1719_final.pdf)                                                              – Anderson, G.R., Nemec,                                                             M., and Aftosmis, M. J., (10Mb, \*.pdf), Jan.                                                             2014  - [NASA/TM 2014-218386 "Toward automatic                                                               verification of goal-oriented flow                                                               simulations"](http://hdl.handle.net/2060/20150000864)                                                                – Nemec, M., and Aftosmis,                                                               M.J.,                                                               Aug.                                                               2014 also [NAS Technical Report NAS-2014-04,](http://www.nas.nasa.gov/assets/pdf/papers/NAS_Technical_Report_NAS-2014-04.pdf) Aug. 2014 - *[AIAA 2014-0558,                                                                 "Cart3D Simulations for the First AIAA Sonic Boom Prediction Workshop."](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa_2014-0558_aftosmis.pdf)*                                                                   – Aftosmis, M.J. and Nemec, M., (6.3Mb, \*.pdf), Jan. 2014- *[AIAA 2014-0836, "Static Aeroelastic Analysis                                                                   with an Inviscid Cartesian Method](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa_2014-0836_Rodriguez.pdf)*                                                                    – Rodriguez, D.L., Aftosmis, M.J., Nemec, M., and                                                                   Smith, S.C. (2.5Mb, \*.pdf), Jan 2014 - *[AIAA 2013-0865, "Output Error Estimates and                                                                     Mesh Refinement in Aerodynamic Shape                                                                     Optimization."](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa2013_0865_nemec_aftosmis.pdf)*                                                                      – Nemec, M., and Aftosmis, M. J., (4.3Mb, \*.pdf), Jan. 2013- *[AIAA 2013-0649, "Summary of the 2008 NASA                                                                       Fundamental Aeronautics Program Sonic Boom Prediction                                                                       Workshop.](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa2013-0649_park_aftosmis.pdf)*                                                                        – Park, M., Aftosmis, M..J., Campbell, R., Carter, M.,                                                                       Cliff, S., and Bangert, L. (6Mb, \*.pdf), Jan 2013- *[ICCFD7-4306,                                                                         "Inviscid Analysis of Extended Formation Flight"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/ICCFD7-4306_final.pdf)*                                                                          – Kless, J., Aftosmis, M.J., Ning, S.A., and Nemec,                                                                         M., (11Mb .pdf), Jul. 2012- *[ICCFD7-2001,                                                                           "Constraint-based Shape Parameterization for Aerodynamic                                                                           Design"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/ICCFD7-2001_final.pdf)*                                                                            – Anderson, G.R., Aftosmis, M.J., and Nemec, M.,                                                                           (\*.pdf 11Mb), Jul.. 2012- *[ICCFD7-2005,                                                                             "Conceptual Design of Low Sonic Boom Aircraft Using Adjoint-Based                                                                             CFD"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/ICCFD7-2005_final.pdf)*                                                                              – Wintzer, M., Kroo, I., Aftosmis, M., and Nemec, M.,                                                                             (6.6Mb, \*.pdf), Jul. 2012- *[ICCFD7-2006, "Design and Evaluation of a                                                                               Pressure Rail for Sonic Boom Measurement in Wind                                                                               Tunnels"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/ICCFD7-2006_final.pdf)*                                                                                – Cliff, S.E., Elmiligui, A., Aftosmis, M.J.,                                                                               et.al, (6.6Mb, \*.pdf), Jan. 2012- *[AIAA 2012-1301, "Progress                                                                                 Towards a Cartesian Cut-Cell Method for Viscous Compressible                                                                                 Flow"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2012_1301_BergerAftosmis.pdf)*                                                                                  – Berger, M.J., and Aftosmis, M.J., with Allmaras, S.R.                                                                                 (7.3Mb, \*.pdf), Jan. 2012- *[AIAA                                                                                   2012-0965, "Parametric Deformation of Discrete Geometry for                                                                                   Aerodynamic Shape Design"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2012_0965_final.pdf)*                                                                                    – Anderson, G.R., Aftosmis, M.J. and Nemec, M. (4.1Mb,                                                                                   \*.pdf), Jan. 2012- *[AIAA                                                                                     2011-3500, "Adjoint-Based Low-Boom Design with                                                                                     Cart3D"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa2011-3500_aftosmis.pdf)*                                                                                      – Aftosmis, M.J., Nemec, M., and Cliff, S.E., (3.4Mb,                                                                                     \*.pdf), Jun. 2011- *[AIAA 2011-3194, "Analysis of                                                                                       Inviscid Simulations for the Study of Supersonic                                                                                       Retropropulsion"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA2011-3194_submitted_1.pdf)*                                                                                        – Bakhtian, N.,M., and Aftosmis, M.J.,                                                                                       (8.6Mb, \*.pdf), Jun. 2011- *[AIAA                                                                                         2011-3666, "Analysis of Grid Fins for Launch Abort Vehicle Using a                                                                                         Cartesian Euler Solver"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa11_3666_final_1.pdf)*                                                                                          – Kless, J., and Aftosmis, M.J., (5.5 Mb, \*.pdf), Jun.                                                                                         2011- *[AIAA                                                                                           2011-3496, "Sonic Boom Computations for a Mach 1.6 Cruise Low Boom                                                                                           Configuration and Comparisons with Wind Tunnel Data"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa11_3496_1.pdf)*                                                                                            – Elmiligui, Ciff, Wilcox, Bangert,                                                                                           Nemec, Aftosmis and Parlette (9.3Mb, \*.pdf), Jun. 2011 - *[IPPW-8 2011 "Maximum Attainable Drag Limits                                                                                             for Atmospheric Entry via Supersonic                                                                                             Retropropulsion"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/IPPW8_Bakhtian_aftosmis_final.pdf)*                                                                                              – Bakhtian, N., and Aftosmis, M.J., (7.1Mb, \*.pdf),                                                                                             Jun. 2011- *[AIAA 2011-1249,                                                                                               "Parallel Adjoint Framework for Aerodynamic Shape Optimization of                                                                                               Component-Based Geometry"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa2011_1249_1.pdf)*                                                                                                – Nemec, M., and Aftosmis, M.J., (2.2Mb, \*.pdf), Jan.                                                                                               2011. - *[AIAA 2010-1239,                                                                                                 "Parametric Study of Peripheral Nozzle Configurations for                                                                                                 Supersonic Retropropulsion"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA2010-1239.pdf)*>                                                                                                  – Bakhtian, N.M., and Aftosmis, M.J., (3.5Mb, \*.pdf), Jan.                                                                                                 2010.- *EG-CAP-09-140: Analysis of the ALAS at                                                                                                   High-Alpha with ACM Jets Using Cart3D.*                                                                                                    – Aftosmis, M.J., Schwing, A., and Stewart, P.C.,                                                                                                   NASA CEV Aerosciences Project Technical Brief, Nov. 2009- *[ParCFD2009\_"Exploring discretization                                                                                                     error in simulation-based aerodynamic databases"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/parCFD2009_aftosmis_sm.pdf)*                                                                                                      – Aftosmis, M.J., and Nemec, M., 21st Internat. Conf. on Parallel CFD,                                                                                                     Mountain View CA, (2.2Mb, \*.pdf), May 2009- *[AIAA                                                                                                       2008-6026, "Structure-preserving parametric deformation of legacy                                                                                                       geometry"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa2008-6026.pdf)*                                                                                                        – Berkenstock, D.C., and Aftosmis, M.J. (7Mb, \*.pdf),                                                                                                       Sep. 2008- *[AIAA                                                                                                         2008-6593, "Adjoint-based adaptive mesh refinement for sonic-boom                                                                                                         prediction,"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa2008-6593.pdf)*                                                                                                          – Wintzer, M, Nemec, M., and Aftosmis, M.J.(4Mb,                                                                                                         \*.pdf), Jun. 2008- *EG-CAP-08-99: Analysis of the ALAS                                                                                                           LAM/CM separation using Cart3D*                                                                                                            – Aftosmis, M.J., NASA CEV Aerosciences Project                                                                                                           Technical Brief, May 2008- *[AIAA 2008-1281,                                                                                                             "Effects of Jet-Interaction on Pitch Control of a Launch Abort                                                                                                             Vehicle,"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2008-1281.pdf)*                                                                                                              – Aftosmis, M.J. and Rogers, S.E. (12Mb, \*.pdf), Jan.                                                                                                             2008 - *[AIAA 2008-0725,                                                                                                               "Adjoint-based adaptive mesh refinement for complex                                                                                                               geometries,"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2008-0725.pdf)*                                                                                                                – Nemec, M., Aftosmis, M.J., and Wintzer, M., (10Mb,                                                                                                               \*.pdf), Jan. 2008 - *[AIAA 2008-0923,                                                                                                                 "Automatic creation of quadrilateral patches on boundary                                                                                                                 representations"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2008-0923.pdf)*                                                                                                                     - Dannenhoffer, J.F., Aftosmis, M.J., (3Mb, \*.pdf),                                                                                                                 Jan. 2008- *EG-CAP-07-107: LAV-068 ACM Assessment using Cart3D*                                                                                                                    – Aftosmis, M.J., NASA CEV Aerosciences Project                                                                                                                   Technical Brief, Aug. 2007- *EG-CAP-07-55: LAV-054 ACM Assessment using Cart3D*                                                                                                                      – Aftosmis, M.J., NASA CEV Aerosciences Project                                                                                                                     Technical Brief, Jun. 2007- *"2007:                                                                                                                       Petascale Computing: Impact on Future NASA Missions."*                                                                                                                        – Biswas, R., Aftosmis, M.J., Kiris, C., and Shen,                                                                                                                       B.-W., In Petascale Computing: Arch. and Algs.                                                                                                                       (D. Bader, ed.),Chapman and Hall / CRC Press, Dec.2007. - *[AIAA 2007-4187 "Adjoint error estimation and                                                                                                                         adaptive refinement for embedded-boundary Cartesian                                                                                                                         meshes"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa2007-4187.pdf)*                                                                                                                           – Nemec, M.N., and                                                                                                                         Aftosmis, M.J., (5Mb \*.pdf) Jun. 2007- *[AIAA 2007-0074 "Dynamic analysis of                                                                                                                           atmospheric-entry probes and capsules",](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA2007_0074.pdf)*                                                                                                                            – Murman, S.M., and Aftosmis, M.J. (\*.pdf) Jan. 2007- *["On the use of Loop subdivision surfaces for                                                                                                                             surrogate geometry"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/15IMR_submitted_06.04.17.pdf)*                                                                                                                              – Persson, P-O.,                                                                                                                             Aftosmis, M.J., and Haimes, R., 15th Internat. Meshing Roundable,                                                                                                                             (\*.pdf) Oct. 2007- *["Adjoint sensitivity computations for an                                                                                                                               embedded-boundary Cartesian mesh method and CAD                                                                                                                               geometry"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/ICCFD4_nemec_aftosmis.pdf)*                                                                                                                                – Nemec, M. and                                                                                                                               Aftosmis, M.J. 4th                                                                                                                               International Conf. on Comp. Fluid Dynam, Ghent Belgium, (1Mb                                                                                                                               \*.pdf) July 2006- *[AIAA 2006-3456 "Aerodynamic shape                                                                                                                                 optimization using a Cartesian adjoint method for CAD                                                                                                                                 geometry"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa2006_3456.pdf)*                                                                                                                                  – Nemec, M. and Aftosmis, M.J. 24th AIAA                                                                                                                                 Applied Aerodynamics Conference, (1Mb \*.pdf) June 2006- *[AIAA 2006-0652 "Applications of a Cartesian                                                                                                                                   mesh boundary-layer approach for complex                                                                                                                                   configurations"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2006-0652.pdf)*                                                                                                                                      – Aftosmis, M.J., Berger, M.J., Alonso, J.J.,                                                                                                                                    (4.2Mb \*.pdf) Jan. 2006- *[AIAA                                                                                                                                     2006-0084 "A detailed performance characterization of Columbia                                                                                                                                     using aeronautics benchmarks and applications",](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA2006_0084.pdf)*                                                                                                                                      – Aftosmis, M.J., Berger, M.J., Biswas, R.,                                                                                                                                     Djomehri, M.J., Hood, R., Jin, H., and Kiris, C., (3.3Mb \*.pdf), Jan.                                                                                                                                     2006- *[SC|05 "High Resolution Aerospace Applications                                                                                                                                       using the NASA Columbia Supercomputer",](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/sc05_paper_submitted.pdf)*                                                                                                                                        – Mavriplis, D.J., Aftosmis, M.J., and Berger, M.J., Supercomputing                                                                                                                                       2005, Best Paper Award, Seattle, WA., Nov. 2005- *[AIAA 2005-4987,                                                                                                                                         "Adjoint algorithm for CAD-based optimization using a Cartesian                                                                                                                                         method"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa2005_4987.pdf)*                                                                                                                                          – Nemec, M., and                                                                                                                                         Aftosmis, M.J., (400k, \*.pdf) Jun.2005- *[AIAA                                                                                                                                           2005-1223 "Characterization of Space Shuttle Ascent debris using a                                                                                                                                           Cartesian method"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA2005_1223.pdf)*                                                                                                                                            – Murman, S.M.,                                                                                                                                           Aftosmis, M.J., and Rogers, S.E. (4.6Mb \*.pdf), Jan. 2005- *[AIAA                                                                                                                                             2005-0877 "Adjoint formulation for an embedded-boundary Cartesian                                                                                                                                             method"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa2005_0877.pdf)*                                                                                                                                              – Nemec, M., Aftosmis, M.J., Murman, S.M.,                                                                                                                                             and Pulliam T.AIAA Paper 2005-0877 (750kb pdf), Jan. 2005- *[AIAA                                                                                                                                               2005-0490 "Analysis of slope limiters on irregular                                                                                                                                               grids](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa2005_0490.pdf)*                                                                                                                                                – Berger, M.J.,                                                                                                                                               Aftosmis, M.J., and Murman, S.E. (2.1Mb pdf), Jan. 2005- *[AIAA 2004-2226 "STS-107 Investigation Ascent                                                                                                                                                 CFD Support"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2004-2226_STS107.pdf)*                                                                                                                                                  – AIAA Paper 2004-2226 (2.1Mb pdf), Jan. 2005- *[NAS-04-015                                                                                                                                                   Automated parameter studies using a Cartesian                                                                                                                                                   method.](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/nas-04-015.pdf)*                                                                                                                                                    – Murman, S.M.,                                                                                                                                                   Aftosmis, M.J., and Nemec, M., NAS Technical Report NAS-04-015 ,                                                                                                                                                   Nov. 2004- *[AIAA-Paper 2004-6274,                                                                                                                                                     Intelligent control for the BEES Flyer](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/bees_aiaa2004-6274.pdf)*                                                                                                                                                      – Krishnakumar,                                                                                                                                                     K., Gundy-Burlet, K.G., Aftosmis, M.J., Nemec, M., Limes, G.,                                                                                                                                                     Berry, M., and Logan, M., Sept. 2004- *[AIAA Paper 2004-5076,                                                                                                                                                       Automated parameter studies using a Cartesian method.](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA2004_5076.pdf)*                                                                                                                                                         – Murman, S.M., Aftosmis, M.J.,                                                                                                                                                       and Nemec, M., Aug. 2004.- *[AIAA Paper 2004-4837,                                                                                                                                                         Validation of inlet and exit boundary conditions for                                                                                                                                                         a Carteisan method.](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2004_4837.pdf)*                                                                                                                                                          – Pandya, S. A., Murman, S.M., and Aftosmis, M.J., Aug.                                                                                                                                                         2004.- *"Simulations of store                                                                                                                                                           separation from an F/A–18 with a Cartesian                                                                                                                                                           method."*                                                                                                                                                            – Murman, S.M., Aftosmis, M.J.,                                                                                                                                                           and Berger, M.J., *Jol. of Aircraft*,  **41**(4):870-879, Jul-Aug                                                                                                                                                           2004.- *[Automated Euler and Navier-Stokes database                                                                                                                                                             generation for a glide-back booster.](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/iccfd3_chadAftosmis.pdf)"*                                                                                                                                                              –  Chaderjan, N.M.,                                                                                                                                                             Rogers, S.E., Aftosmis, M.J., Pandya, S.A., Ahmad, J.U., and                                                                                                                                                             Tejnil, E.,  Proc. 3rd Intl. Conf. on CFD, July 2004.- *"[On the use of CAD and Cartesian                                                                                                                                                               methods for  aerodynamic optimization](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/iccfd3_nemecAftosmis.pdf)."*                                                                                                                                                                – Nemec, M., Aftosmis, M.J., Pulliam,                                                                                                                                                               T.H., Proceedings of the 3rd International Conference on                                                                                                                                                               Computational Fluid Dynamics, July 2004.- *[Performance of a new CFD solver using a hybrid programming                                                                                                                                                                 paradigm,](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aftosmis_siam_pp04.pdf)*                                                                                                                                                                    – Aftosmis, M.J., Berger, M.J., SIAM                                                                                                                                                                 Conf. on Parallel Processing in Scientific Computing 2004. Feb.                                                                                                                                                                 2004 (invited). ([\*.pdf                                                                                                                                                                 2.5Mb](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aftosmis_siam_pp04.pdf)).- [*AIAA Paper 2004-1232, Applications of                                                                                                                                                                   space-filling curves to Cartesian methods for                                                                                                                                                                   CFD.*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa2004_1232.pdf)                                                                                                                                                                    – Aftosmis, M.J.,                                                                                                                                                                   Berger, M.J., Murman, S.M., (2.8Mb)- [*AIAA Paper 2004-0113, CAD-based                                                                                                                                                                     aerodynamic design of complex configurations using a Cartesian                                                                                                                                                                     method.*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa2004_0113.pdf)                                                                                                                                                                       – Nemec, M.,                                                                                                                                                                     Aftosmis, M.J., and Pulliam, T.H., 42nd AIAA Aerospace Sciences                                                                                                                                                                     Meeting and Exhibit, Jan. 2004 (2.6Mb) - [*AIAA Paper 2003-3670 Cartesian-Grid                                                                                                                                                                       Simulations of a Canard-Controlled Missile with a Spinning                                                                                                                                                                       Tail.*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA2003_3670.pdf)                                                                                                                                                                        –  S. Murman, M. Aftosmis(\*pdf                                                                                                                                                                       4.7Mb)- *[AIAA Paper 2003-4229 , Automated CFD                                                                                                                                                                         Parameter Studies on Distributed Parallel Computers](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA2003_4229.pdf)*                                                                                                                                                                          – S. Rogers, M. Aftosmis, S. Pandya and N.                                                                                                                                                                         Chaderjian, June 2003- *[AIAA-2003-1119 Implicit Approaches for Moving                                                                                                                                                                           Boundaries in a 3-D Cartesian Method,](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA2003-1119.pdf)*                                                                                                                                                                             – S. M. Murman,                                                                                                                                                                           M. J. Aftosmis and M. J. Berger (\*.pdf 1.6Mb) Jan. 2003- *[AIAA 2003-1246, Simulations of                                                                                                                                                                             6-DOF motion with a Cartesian method.](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA2003-1246.pdf)*                                                                                                                                                                              – S. M. Murman, M.J. Aftosmis and M. J. Berger (\*.pdf 4.6Mb).- *[AIAA 2003-1237, An interface                                                                                                                                                                               for specifying rigid-body motions for CFD applications](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA2003-1237.pdf)*                                                                                                                                                                                – S. M. Murman, M. J. Aftosmis and M. J. Berger, (\*.pdf 1.6Mb).- *[AIAA                                                                                                                                                                                 2002-2798, Numerical Simulation of Rolling-Airframes Using a                                                                                                                                                                                 Multilevel Cartesian Method](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA2002_2798.pdf)*                                                                                                                                                                                  – S. M.                                                                                                                                                                                 Murman, M. J. Aftosmis and M. J. Berger (\*.pdf 850Kb).- *[On                                                                                                                                                                                   Generating High Quality "Water Tight" Triangulations Directly From                                                                                                                                                                                   CAD](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/isgg02.pdf)*                                                                                                                                                                                    –  R. Haimes and M. J. Aftosmis, (ISGG) 2002,                                                                                                                                                                                   Honolulu, HI. Jun. 2002. (\*.pdf 1.4Mb).- *[AIAA 2002-0863, Multilevel Error Estimation and                                                                                                                                                                                     Adaptive h-Refinement for Cartesian Meshes with Embedded                                                                                                                                                                                     Boundaries](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa2002-0863.pdf)*                                                                                                                                                                                       – M. J.                                                                                                                                                                                     Aftosmis and M. J. Berger, (best paper award)- *[AIAA 2001-0997, Computation of external                                                                                                                                                                                       aerodynamics for a canard rotor/wing                                                                                                                                                                                       aircraft(820kb).](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa2001-0997.pdf)*                                                                                                                                                                                         – S. A.                                                                                                                                                                                       Pandya and M. J. Aftosmis, (\*.pdf)- *[Parallel multigrid on Cartesian meshes                                                                                                                                                                                         with complex geometry (725kb).](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/parCFD_00.ps.gz)*                                                                                                                                                                                           – Proceedings                                                                                                                                                                                         of the 8th International Conference on Parallel CFD, Trondheim                                                                                                                                                                                         Norway. (\*.ps.gz)- *[AIAA 2000-0808,  A parallel multilevel                                                                                                                                                                                           method for adaptively refined Cartesian grids with embedded                                                                                                                                                                                           boundaries (610kb).](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa2000-0808.pdf)*                                                                                                                                                                                              –  38th AIAA                                                                                                                                                                                           Aerospace Sciences Meeting and Exhibit, Reno NV Jan. 2000. gzipped                                                                                                                                                                                           postscript format- *[A parallel Cartesian approach for external                                                                                                                                                                                             Aerodynamics of vehicles with complex geometry (410kb)](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/tfaws99.ps.gz).*  – Aftosmis, M.                                                                                                                                                                                             J., Berger, M. J., and Adomavicius, G., (\*.ps.gz)- *[On the use of CAD-native                                                                                                                                                                                               predicates and geometry in surface meshing (580kb)](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/nasaTM_1999_208782.pdf)*                                                                                                                                                                                                 –                                                                                                                                                                                                NASA/TM-1999-208782, Ames Research Center, Aug. 1999. acrobat                                                                                                                                                                                               format- *[AIAA 99-0776, Automatic Generation of                                                                                                                                                                                                 CFD-Ready Surface Triangulations from CAD Geometry                                                                                                                                                                                                 (750kb)](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa99-0776.ps.gz)*  – 37th                                                                                                                                                                                                 AIAA Aerospace Sciences Meeting, Reno NV Jan. 1999., (\*.ps.gz)- *[AIAA 99-0777, Automatic                                                                                                                                                                                                   Hybrid-Cartesian Grid Generation for High-Reynolds Number Flows                                                                                                                                                                                                   around Complex Geometries (2.4Mb)](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aiaa99-0777.ps.gz)*  – 37th                                                                                                                                                                                                   AIAA Aerospace Sciences Meeting, Reno NV Jan. 1999., (\*.ps.gz)- *[Adaptive Cartesian Mesh Generation                                                                                                                                                                                                     (1Mb)](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/crcChapterDraft.ps.gz)*                                                                                                                                                                                                      – M. Aftosmis, Contributed Chapter, *CRC Handbook of Mesh Generation, 1998* - *[Aspects (and aspect ratios) of Cartesian                                                                                                                                                                                                       Mesh Methods](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/16ICNMFD_98.pdf)*                                                                                                                                                                                                        – M. Berger & M. Aftosmis,                                                                                                                                                                                                       *Proceedings of the 16th International Conf. on Num. Meth. in Fluid                                                                                                                                                                                                       Dynamics* (\*pdf 360Kb). Jul 1998- *["Lecture notes on Solution adaptive                                                                                                                                                                                                         Cartesian Grid Methods"](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/aftosmis_97vkiNotes.pdf)*                                                                                                                                                                                                           – M. Aftosmis, von Karman                                                                                                                                                                                                         Institute for Fluid Dynamics, Lecture Series 1997-02, March 1997- *[AIAA 97-0196: Robust and Efficient                                                                                                                                                                                                           Cartesian Mesh Generation for Component-Based Geometry, 1997                                                                                                                                                                                                           (1.7Mb)](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA-97-0196.pdf)*– Aftosmis, Berger & Melton, also                                                                                                                                                                                                           [*AIAA Jol.*                                                                                                                                                                                                           **36**(6), 1998](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAAJol_97_v36n6_aftosmis.pdf)- *[AIAA 94-0415: On the Accuracy, Stability, and Monotonicity                                                                                                                                                                                                             of Various Reconstruction Schemes for the Navier-Stokes                                                                                                                                                                                                             Equations](https://arc.aiaa.org/doi/pdf/10.2514/3.12945)*                                                                                                                                                                                                              – M. Aftosmis, D. Gaitonde, T. Tavares, 1994.                                                                                                                                                                                                             Also [*AIAA Jol.***33**(11) 1995](https://arc.aiaa.org/doi/pdf/10.2514/3.12945)- *Emerging CFD Technologies and Aerospace Vehicle Design*                                                                                                                                                                                                                – NASA Wkshp on Surf. Modeling, Grid Generation and                                                                                                                                                                                                               Related issues,NASA LERC May, 1995- *AIAA 95-1725 Adaptation and Surface Modeling for Cartesian                                                                                                                                                                                                                 Mesh Methods*                                                                                                                                                                                                                   – 12th AIAA Computational Fluid Dynamics Conference, June 1995- *AIAA 95-0853 3D Applications of a Cartesian Grid                                                                                                                                                                                                                   Euler Method*                                                                                                                                                                                                                    – 33rd AIAA Aerospace Sciences Meeting, Reno NV, Jan. 1995      ---  *last update February, 2026.* |

---

<a id="page-cart3d-team-html"></a>

## Contact

*Original page: [cart3d_team.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/cart3d_team.html)*

The Cart3D Development Team  Cart3D is an outgrowth of ongoing research in Cartesian mesh methods for CFD performed jointly by NASA Ames and the Courant Institute at NYU.  It supports NASA's [Exploration Systems,](http://www.nasa.gov/exploration/about/esmd_mission.html)  [Aeronautics Research](http://www.aeronautics.nasa.gov/), [Science,](http://science.nasa.gov/missions/) [Space Operations,](https://www.nasa.gov/directorates/space-operations-mission-directorate) and [Space Technology](https://www.nasa.gov/directorates/spacetech/home/index.html) Mission Directorates. Work at NYU has been funded through NASA, the AFOSR and the Department of Energy.     ---  - [Michael Aftosmis](https://www.nas.nasa.gov/aboutnas/staff/staff_maftosmis.html), NASA Ames   Research Center, Moffett Field, CA - [Marian Nemec](https://www.nas.nasa.gov/aboutnas/staff/staff_nemec.html), NASA Ames   Research Center, Moffett Field, CA - [Marsha Berger](https://cs.nyu.edu/berger/),   Courant Institute, NYU, New York, NY & Flatiron Institute, New York, NY - Jonathan   Chiew, NASA Ames Research Center, Moffett   Field, CA - Wade   Spurlock, NASA Ames Research Center, Moffett   Field, CA - David   Rodriguez, Science & Technology Corp., NASA Ames,   Moffett Field, CA   ---   last update December, 2024. |

---

<a id="page-surfacemodeling-html"></a>

## Surface Modeling

*Original page: [surfaceModeling.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/surfaceModeling.html)*

Surface Modeling and Geometry Import  ---   **[Component based approach](#page-surfacemodeling-html--componentbasedapproach)**  **[The Process](#page-surfacemodeling-html--cart3dprocess)**   **[Geometry import into Cart3D](#page-surfacemodeling-html--geomimport)** - fem2tri, off2tri, dxf2tri, & triangulate  **[Auxiliary Programs](#page-surfacemodeling-html--auxprogs)** - intersect, trix, comp2tri & diagnoseGeom  **[A Few Technical Topics and Computational Geometry](#page-surfacemodeling-html--techtalk)**    ---  **Component based approach** |  |  | | --- | --- | | Loosely speaking, all geometry comes into Cart3D as a collection of *components*, which we call a *configuration*. In the exploded view of a space shuttle at the right, there are 20 "[component triangulations](#page-cart3dtriangulations-html--1-component-file-forma)"  which comprise this configuration. By treating each of these components as its own entity, we're free to move or modify an individual component to alter the configuration. Using this *Component Based Approach* you can  do a study in which components move or change (for example a  control surface gets deflected, or the wing is modified), without having to re-generate a completely new surface triangulation for the modified geometry. |  |  In more specific language, each component is a *simplicial polytope*. Think of this as a solid whose surface is a triangulation. Thus, each component must be represented as a single, watertight, triangulation, (no internal geometry). By "watertight" I mean that this triangulation must be locally manifold, so that one can traverse the entire surface by starting in any triangle, and traversing (recursively) over that triangle's edges into the next triangle. If one paints each triangle when it is visited, all the triangles on the component will get painted, regardless of where one begins the process. By this definition, every edge in the triangulation will have exactly 2 neighboring triangles.  A "component" is one such simplicial polytope, and a "configuration" is a collection of components which describe the geometry you want to grid and run a flow simulation on. A configuration may have any number of components. Cart3D has been used on configurations with up to 1800 components, and there is (in principle) no upper limit. The package comes with tools which will diagnose problems with incomming geometry and help you to correct them. Follow this link for [details about Cart3D triangulation formats](#page-cart3dtriangulations-html).   (*[top](#page-surfacemodeling-html--top)*)  **The Process**  1. Import    the components that make up your configuration 2. Extract    the wetted surface (using **intersect**) 3. Build    the Cartesian mesh (using **[cubes](#page-meshgeneration-html)**) 4. Launch    the flow solver (**[flowCart](#page-flowcart-html)**)   Modifying your configuration and then repeating steps 2-4 allows you to conduct parametric studies on a given configuration.   (*[top](#page-surfacemodeling-html--top)*) **Geometry Import** Tools are available for importing both geometry from both unstructured and structured sources. These are summarized in the table below.    |  |  |  | | --- | --- | --- | | **Unstructured Sources** | **Typical source** | **import path** | | STL triangulations | CAD programs, or www | **admesh**->**off2tri** | | OFF triangulations | CAD programs, or www | **off2tri** | | DXF triangulations | AutoCad, other PC based CAD | **dxf2STL**->**admesh**->**off2tri** | | FNF triangulations | CAD programs (esp. ProE) | **fem2tri** | | **Structured Sources** |  |  | | plot3d surface grids | CFD codes/mesh generators | **triangulate** | | LaWGS networks | various | **net2p3d**->**triangulate** |   **Import Tools:**  **fem2tri:**  takes a "finite element neutral" format triangulation and converts it into an individual component triangulation (which you can feed to "comp2tri" to build a configuration). The perl script \*$CART3D/perl/fem2tri.pl" is extensively documented.  **Usage: fem2tri infile.fem > outfile.tri**  (*[top](#page-surfacemodeling-html--top)*) **STL:** Stereolithography files (\***STL**) should be first converted to OFF format triangulations and then converted into Cart3D triangulations using **off2tri** (below). **off2tri:** takes a "OFF triangulation" format triangulation and converts it into an individual component triangulation (which you can feed to "comp2tri" to build a configuration).The perl script \*$CART3D/perl/off2tri.pl" is extensively documented.  OFF format triangulations are relatively common. Of particular note is that "admesh" which is a code that tries to produce manifold triangulations from StereoLithography (STL) files, produces "OFF" format triangulations. This is particularly useful for geometry acquisition from CAD since most CAD systems now produce \*STL triangulations.    **Usage: off2tri infile.fem > outfile.tri**  (*[top](#page-surfacemodeling-html--top)*)    tri2dat.csh:  a simple shell script for converting an ascii formatted Cart3D surface triangulation to a format readable by tecplot or paraview.  **Usage: tri2dat.csh my\_file.tri**  Produces: **my\_file.dat**  **dxf2STL:** takes a "dxf triangulation" format triangulation and converts it into an STL triangulation.The perl script $CART3D/perl/dxf2tri.pl is extensively documented. DXF format triangulations are put out by autoCAD and some other popular CAD programs.     **Usage: dxf2STL infile.dxf > outfile**  1. Generate an    STL file with dxf2STL 2. Use admesh to    convert this to an OFF format triangulation 3. Convert the    \*off file to an ascii format  component    triangulation (\*.a.tri) with off2tri  (*[top](#page-surfacemodeling-html--top)*) **triangulate:** triangulate a Multiple-grid plot3d file. Triangulate takes a multiple-grid plot3d format configuration and triangulates it component-by-component. Points with duplicate geometry (same point in physical space) are removed. It is used for converting components specified from structured geometry sources into  intersection-ready triangulations. Component information is retained for each triangulation. To prevent problems downstream, The default  behavior includes slightly (10x machine zero) perturbing vertices which are shared between separate components. This behavior can be suppressed with the "**-n**" option. *Extremely* degenerate geometry can be inflated (slightly - 10xMachineZero) around component centers in an attempt to avoid excessive tie-breaking down-stream. Normally the "**-r**" flag is set when triangulate is run to remove duplicate vertices from the geometry. Component lists generated by **net2p3d** (or by hand) may be fed into triangulate with the -C flag. The output of triangulate is a [Cart3D component triangulation](#page-cart3dtriangulations-html--1-component-file-forma) or a  [Cart3D configuration triangulation](#page-cart3dtriangulations-html--2-configuration-file-format) (**\*.tri**) file which has a component-by-component triangulation of the configuration, ready for intersection.   |  | | --- | | **% triangulate -**  **Usage: triangulate [-i infile -o outfile -T -v -r -n**  **-C comp\_file -inward -inflate ]**  **Options:**  **-i .....** Input  file name, def:<mgrid.unf>  **-o .....** Output file name, def:<Components.tri>  **-T .....** Output to tecplot file "tec.dat"  **-fast ..** Output to unformatted FAST file "Components.fast"  **-C .....** Component list for infile, def:<Component.list>  **-v .....** Verbose mode  **-r .....** Remove duplicate nodes in input file  **-lex ...** Lexicographically sort the triangle verts (needs -r)  **-n .....** Dont perturb identical pts on diff components  **-inward.** Norm vectors on input geom face inward. def:<outward>  **-ascii .** Input file is ascii fmt -  (output still unformatted)  **-dp ....** Double precision I/O (Output trix format) unformatted)  **-zero ..** set (data < ZERO) to 0.0000 - (param ZERO = 1.E-12)  **-inflate.**Inflate geometry by 10\*Eps to break  degeneracies |   (*[top](#page-surfacemodeling-html--top)*)  **net2p3d** :  Converts a LaWGS net into multiple grid plot3d format. The LaWGS format consists of any number of structured  patches with header information between each. Currently **net2p3d** does not parse scaling, translations or rotations in the LaWGS header. When a single watertight component is described by several meshes within the LaWGS file, these are identified by the component number in the LaWGS file. (In the LaWGS file, this is the first integer following the string which is the component's name, and comes just before the dimensions of the network. In addition to the multiple-grid plot3d representation of the mesh, **net2p3d** outputs a (human readable) **Components.list** file which describes the mapping of LaWGS networks to component numbers. This file can be suppressed with the "-no" option when this mapping is 1-to-1 (the default for **triangulate**).   |  | | --- | | **% net2p3d -**  **Usage: net2p3d [-i infile -o outfile  -v -m**  **-C Compfile -no -ascii]**  **Options:**  **-i ........ Input  file name, def:<LaWGS.net>**  **-o ........ Output file name, def:<mgrid.unf>**  **-v ........ Verbose mode**  **-m ........ "memory verbose" report malloc/free**  **-C ........ Component list file name,def:<Component.list>**  **-no ....... Dont write out a Component list**  **-ascii..... Output file ascii format** |  (*[top](#page-surfacemodeling-html--top)*) **Other Auxiliary Programs** **comp2tri: (New Features!)** combines any number of individual components into a cart3d "configuration" (generally \*.tri file extension). Original component numbers of triangles are retained. The output of comp2tri is a Cart3D [configuration triangulation](#page-cart3dtriangulations-html--2-configuration-file-format) (**\*.tri**) file ready for intersection.   |  | | --- | | **% comp2tri -**  **Usage: comp2tri [OPTIONS] file0.tri file1.tri file2.tri ...** **Options:** **-v ............. Verbose more ON** **-inflate ....... Inflate geometry to break degeneracies    -o %s .......... Output filename <Components.tri>    -trix .......... Output file will be eXtended-TRI (trix) format    -ascii ......... Output file will be ASCII (if not trix)    -keepComps ..... Preserve 'intersect' component tags    -makeGMPtags ... Create GMPtags from volume indexes** **-gmp2comp ...... Copy GMPtags to IntersectComponents    -config ........ Write Config.xml using component tags    -dp ............ Use double precision vert-coordinates <FALSE>****-gmpTagOffset %d Renumber GMPtags by adding "1 x offset" to tags                     in file1.tri, '2 x offset' to tags in file2.tri, etc.                     First file, file0.tri, gets no offset (****tags left unchanged)** |   *[top](#page-surfacemodeling-html--top)* **intersect: (New Features!)** Extracts the wetted surface of a configuration. "Configurations" are collections of components output either by triangulate, comp2tri, or made by some other method. intersect is extensively documented in AIAA 97-0197. The wetted surface extracted by intersect is in the form of a [Cart3D wetted surface triangulation](#page-cart3dtriangulations-html--3-wetted-surface-triangulation-format) and is watertight.  Component information is retained. By convention, output files are generally named **\*.i.tri** to indicate that they are *post-intersection* and do not contain any internal geometry. If "intersect" ever fails, it drops an "Error.dat" file which is a tecplotable file containing geometry local to the problem which caused it to fail. One may then  view the complete geometry and overlay the geometry contained in "Error.dat" to diagnose the source of the problem. Intersect is quite robust, and it begins and ends with a geometry verification phase. If intersect stops during the initial geometry verification it will suggest possible problems in the input geometry (e.g.  Component N is not closed", "Component X is non-manifold" etc These checks are topological in nature and do not depend on floating point math. They are therefore robust, and I've never seen a case where they were incorrect. In verbose mode ("-v"), this verification phase is step #4. Try to diagnose any geometry verification problems by viewing the Error.dat geometry against overlaid with the input geometry. Check in KNOWN\_BUGS for a current listing. Intersect is based on [boolean intersection predicates](#page-bool-intersection-html) and uses adaptive precision floating point math with [automatic tie-breaking to resolve degeneracies](#page-degeneracy-html).   |  | | --- | | **% intersect -**     **Usage:**  **intersect [ -i infile -o outfile -T -v -intersections** **-ascii -mem]**  **Options:**  **-i ............ Input  file name, def:<Components.tri>**  **-o ............ Output file name, def:<Components.i.tri>**  **-ascii ........ Input  geometry file is ASCII**  **-T ............ Also output tecplot file "Components.i.plt"**  **-fast.......... Also output unformatted FAST file** **"Components.i.fast"**  **-v ............ Verbose Mode**  **-mem .......... Report memory useage (auto on with "-v")**  **-cutout %d..... Perform boolean subtraction (A-B) for component <%d>** (details!)     **-overlap %d.... Perform boolean intersection of comp <%d> with others** (details!)    **-intersections. write tecplot file of intersections** **<intersect.dat>** |   (*[top](#page-surfacemodeling-html--top)*)    trix:  The swiss-army knife of triangulations: format converter, translations, rotations, symbolic shape manipulation, component tagging and much much more. Its default action is to convert the input files to Cart3D's extended triangulation (VTK) format. Shape sensitivities (if present) are automatically adjusted to reflect any geometry manipulation. Numerical parameters can be specified as expressions using operators +, -, \*, /, ^ (exponent), unary +, unary -, sqrt, exp, log, pow, sin (radians), con, tan, asin, acos, atan, atan2 (2 args), sind (degrees), cosd, tand, dasin, dacos, datan, datan2, floor, ceil, and constants pi and e.  For help use:   ***% trix -h, --help*** See the full usage statement   ***% trix -helpSymbolic*** Operators and functions for symbolic deformation or component tagging   ***% trix -helpVariable*** List of pre-defined variables for symbolic deformation and tagging   See *trix* in the Cart3D  [*COMMAND\_SUMMARY.pdf*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/COMMAND_SUMMARY.pdf) and advanced features in [*New tricks with trix*](#page-trix-index-html)  Some basic options are listed here    **Usage:     trix [-OPTIONS] [file(s) ...]  *Options are listed in the order they are applied:***  -select ...         Select component(s) to mirror/scale/translate/rotate,                      e.g. -select 1 2 5, omit to select all  ***Scale Geometry:***  -mirror %c          Mirror the X, Y or Z coordinate <none>  -sx %F              Scale geometry in X <1.>  -sy %F              Scale geometry in Y <1.>  -sz %F              Scale geometry in Z <1.>  ***Translate geometry:***  -x %F               Translate geometry in x-direction <0.>  -y %F               Translate geometry in y-direction <0.>  -z %F               Translate geometry in z-direction <0.>  ***Rotate geometry (order rx-ry-rz):***  -cx %F              X center of rotation <0.>  -cy %F              Y center of rotation <0.>  -cz %F              Z center of rotation <0.>  -rx %F              Rotate geometry around x-axis (deg) <0.>  -ry %F              Rotate geometry around y-axis (deg) <0.>  -rz %F              Rotate geometry around z-axis (deg) <0.>  -rg %F %F %F %F     Rotate geometry around vector with tail <cx, cy, cz>                      and head < 0., 0., 0. > (deg) <0.>  ***IO options:***  -v                  Be verbose <FALSE>  -dp                 Use double precision vert-coordinates <FALSE>  -o %s               Output filename prefix <Components> (in VTU format)  -T                  Output a Tecplot file for each component (.dat)  -tri                Output all files as traditional Cart3D tri-files  -noVTK              Do not write an extended triangulation (VTK) file <FALSE>  ***Tag manipulations:***  -comp2gmp           Overwrite or create GMPtags from component tags  -tagRegion %F %F %F %F %F %F                      Tag rectangular region inside XMIN XMAX YMIN YMAX ZMIN ZMAX  -add2comp %d        Increment or decrement component tags by this amount <0>  -add2gmp %d         Increment or decrement GMPtags by this amount <0>  ***Linearization:***  -dx                 Linearize with respect to x translation <FALSE>  -dy                 Linearize with respect to y translation <FALSE>  -dz                 Linearize with respect to z translation <FALSE>  ***Input file list:***   ...                file1.tri file2.triq (extensions not important)   (*[top](#page-surfacemodeling-html--top)*)   **diagnoseGeom**: diagnoseGeom is a utility aimed at helping you diagnose [configuration](#page-cart3dtriangulations-html--2-configuration-file-format) geometries that intersect rejects. It verifies that components in your configuration are all valid and then performs intersection checks on all combinations of components in the configuration. When errors are found, they are reported and logged into a (newly created) subdirectory called "diagnosis". From this information you can quickly identify the offending component and make a repair.  **Usage:**  **diagnoseGeom [-ascii -base=basename -split -verbose]** **Example:** (Tell me what's wrong with the ascii configuration triangulation "myConfig.a.tri")  **% diagnoseGeom -ascii myConfig** **A Few Technical Topics and Computational Geometry**  - **[Boolean polytope   intersection](#page-bool-intersection-html)** - **[Degeneracies in geometric   data - and what **intersect** does with   them](#page-degen-ex-html)** - **[An   algorithmic approach to tie breaking](#page-degeneracy-html--virtualpurturbationsos)** - **[Adaptive precision   floating point arithmetic](#page-degeneracy-html)**   (*[top](#page-surfacemodeling-html--top)*)   ---   Questions?   Visit [Cart3D Discuss](https://groups.google.com/forum/#%21forum/cart3d),   or   [Contact Us](#page-cart3d-team-html)  last update December 2024   --- |

---

<a id="page-meshgeneration-html"></a>

## Mesh Generation

*Original page: [meshGeneration.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/meshGeneration.html)*

Mesh Generation  ---   **[What is "cubes"?](#page-meshgeneration-html--what-is)**  **[What is new in cubes?](#page-meshgeneration-html--whats-new-in-cubes)**   **[What input files does cubes require?](#page-meshgeneration-html--tips)**   **[What is "autoInputs"?](#page-meshgeneration-html--autoinputs)**  **[Usage - cubes](#page-meshgeneration-html--use)**  **[How do I run a 2D case?](#page-meshgeneration-html--2d)**   **[Tell me more about prespecified adaptation regions.](#page-prespecnote-html)**  **[File Format of meshes produced by cubes](#page-meshgeneration-html--fileformat)**  **[Further Documentation](#page-meshgeneration-html--doc)**   ---  **What is "cubes"?**    |  |  | | --- | --- | | **Cubes**  is a mesh generation program which produces topologically unstructured, adaptively refined, Cartesian  meshes around any geometry or configuration that may be described by a collection of simplicial polyhedra (closed surface triangulations - the output of **intersect** works fine) The source is written in ANSI C and makes extensive use of bitwise operators to minimize memory requirements and maximize speed.  **Features**:   - Gross   memory usage typically ~50Mb/million   cells - Speed   is well in excess of 10,000,000 cells/min   on a typical (2013-era) laptop or   desktop. - Geometry   based adaptive cell division - Option   for users to limit total # cells & max   allowable memory - Multiple   region (split-cells) handled   automatically - The   ability to [prespecify   mesh adaptation](#page-prespecnote-html) in certain regions   or on particular components - [Remesh using   previously](#page-meshgeneration-html--whats-new-in-cubes) created or adapted meshes   as a cell-density map while adding in   additional refinement in particular   regions or on certain components.               [(top)](#page-meshgeneration-html--top) | 5M cell mesh around a HSCT configuration    side view - zoom box on nacelle region    closeup near nacelle. | **What's new in cubes?**   In v1.5 we've introduced a "-remesh" option that allows cubes to use previously created meshes as a meshing template for relative cell densities. For example, if you have an [adapted mesh from ***aero.csh***](#page-adjoint-index-html), You can simply remesh it to make the whole thing finer, change the domain size, put in more background divisions, or even manually add specific refinement regions. This is also handy if you've used Cart3D's adjoint-based adaptation to build an adapted mesh on a small shared-memory system and you want to use that mesh as the basis for a large distributed memory run (using MPI). Using the adapted mesh as a cell-density map, you easily create a much finer mesh that harnesses all the knowledge that the adjoint provided about relative cell density requirements, and then launch a large distributed memory simulation using MPI.  This is also a great tool for adding in a refined region in the wake when an initially steady simulation starts to become unsteady as the mesh gets adaptively refined. This example shows you more detail on remeshing. **A few tips****?**  Experienced users can make use of a couple of handy features in **cubes** to for both greater control and a more streamlined meshing process. All command line options are listed in the [usage statement](#page-meshgeneration-html--use) below.      Automatic reordering:  "% cubes -reorder" will reorder the mesh before it ever hits your disk. This is completely equivalent to running the standalone reorder program, but you skip ever writing the un-reordered mesh, and go straight to Mesh.R.c3d. Since CPU speeds are improving faster than disk speed, writing the mesh has been taking an increasing amount (relative) time in the past few years, Writing out  the mesh at the end of cubes just to read it back in in the first step of reorder seemed like a complete waste of time,  so now you can do it direct. Doing the reorder on-the-fly within cubes does mean that the whole mesh needs to reside in memory at the same time as all the other data structures that cubes needs, so this option does increase the memory requirements of cubes by a bit, but this consumption is not severe, and nobody's complained yet, since cubes has a very tiny memory footprint to start with.      Sharp Feature Refinement: You can additional refinement sweeps that just target sharp features in your geometry (sharp edges, ridges, etc). This argument takes an integer which says how many additional refinement sweeps to do. Invoke it with "%cubes -sf %d" where %d is an integer. Usually "-sf 1" or (at most) "-sf 2" is sufficient.   [(top)](#page-meshgeneration-html--top)      **What input files does cubes require?**  **Cubes** requires 2 files to get going:  1. An input file controlling the mesh    you want to make (only 7 entries! ). 2. Input surface triangulations are    read in as a Cart3D "[wetted    surface triangulation](#page-cart3dtriangulations-html--3-wetted-surface-triangulation-format)" which may contain any    number of components.  Entry #1 in the [cubes input file](#page-input-c3d-html) takes    the name of the triangulation you're gonna    mesh.   Optionally, you can include a "[prespecified adaptation region](#page-prespecnote-html)" file, usually called [preSpec.c3d.cntl](#page-prespec-c3d-cntl-html). Entries in this file can be used to direct cubes to refine more in certain regions of the domain click **[here](#page-prespecnote-html)** to learn more about prespecified adaptation regions, or [here](#page-prespec-c3d-cntl-html) to see a sample.  [(top)](#page-meshgeneration-html--top) **What is "autoInputs"?** The "autoInputs" utility automatically generates the [input.c3d](#page-input-c3d-html) and [preSpec files](#page-prespec-c3d-cntl-html) that cubes needs to get going. This utility benefited greatly from beta-testing and we strongly recommend its use by both novice and experienced users alike.  Basically you hand autoInputs your geometry and a rough idea of how big you want the mesh to be and it will produce an [input.c3d](#page-input-c3d-html) and [preSpec.c3d.cntl](#page-prespec-c3d-cntl-html) file that you can use to mesh. Overall, autoInputs is very handy for getting going in short order. The [preSpec.c3d.cntl](#page-prespec-c3d-cntl-html) files that autoInputs produces are quite detailed, and include preSpec boxes for the entire configuration, and smaller boxes for each component. Like the rest of the utility programs in Cart3D, the executable for autoInputs is located in $CART3D/bin/$CART3D\_ARCH/. To get a usage statement, type "% autoInputs -" (the trailing "‐" will get you the usage statement, as with all executables in the package).  Usage - autoInputs  |  | | --- | | **% autoInputs -**  **Usage: autoInputs [ argument list ]** **Options:**      -t %s       Input '\*.tri' file name <Components.i.tri>      -r %F       nominal mesh radius <default = 30.>      -nDiv %d    nominal # of divisions in background mesh <default: 5>      -maxR %d    Max Num of cell refinements to perform <default: 11>      -symmX      Halfbody mesh in X      -symmY      Halfbody mesh in Y      -symmZ      Halfbody mesh in Z      -halfBody   Input geometry is a half-body      -pre %s     file name for preSpec control file <preSpec.c3d.cntl>      -i   %s     file name for mesh input control file <input.c3d> |  [(top)](#page-meshgeneration-html--top)  **Usage - Cubes**  |  | | --- | | **% cubes -**  **Usage: cubes [ argument list ]**    **Options:****-- Meshing Options--**        -maxR %d    Max number of refinements (overrides input file)        -a %f       Angle threshold (deg) for geom refinement <25 deg>        -b %d       Number of layers of buffer cells <3>        -pre %s     Prespecified adapt region filename <None>        -d %f       Angle (deg) for directional refinement (aniso only) <10 deg>        -Internal   Mesh the \*interior\* of the geometry        -mesh2d     Make 2D mesh in X-Y, no refinement in Z        -I          Restrict cells to isotropic division only        -remesh     Set cell density from "refMesh.{mg.c3d,c3d.Info}"        -reorder    Reorder output mesh in space-filling curve order        -sfc %c     Mesh order: H=Peano-Hilbert, M=Morton <H>        -verify     Verify that all cut cells close    **-- Advanced Options--**  -sf %d      Number of additional levels at sharp edges        -weight     Area-weight triangles in divide criteria        -TPC %d     Adapt based on # of triangles-per-cutCell        -vtest      Special vtest for buffering        -lin %s     Linearize cut-cells for specified design variable        -try\_exact  (if -N specified) Use Newton solve to hit -N exactly    **-- I/O Options--**        -i %s       Input file name <input.c3d>        -o %s       Output file name <Mesh.c3d>        -no\_file    Suppress output file        -v          Verbose mode ON        -quiet      Don't make excessive noise        -mem        Report memory usage (auto on with -v)        -h          Print history of # of cells with refinement (auto on with -v)        -Dunset     Dump Tecplot file <unset.dat>        -Dcut       Dump Tecplot file <cutcells.dat>        -Dflow      Dump Tecplot file <flowcells.dat>        -Dsolid     Dump Tecplot file <solidcells.dat>        -Dsplit     Dump Tecplot file <splitcells.dat>        -Daniso     Dump Tecplot file <anisocells.dat>        -Xcut %d    Number of X=const cut planes <cutPlanes.dat>        -Ycut %d    Number of Y=const cut planes <cutPlanes.dat>        -Zcut %d    Number of Z=const cut planes <cutPlanes.dat>    **-- Memory Options--**        -memLim %f  Target memory usage (in MB) of final mesh        -N %d       Target final number of Cartesian cells        -no\_est     Don't estimate memory requirements (slower, less compact)    **-- (Deprecated)--**        -P                          -T                          -ascii        -STARS      Record intermediate refinement history |  [(top)](#page-meshgeneration-html--top) **How do I run a 2D case?** 2D cases can be run by making a "slab-like" domain in x-y that is only 1-cell deep (in z). Then use the "-mesh2d" command-line flag to suppress cell refinement in z. The result is a mesh which is adapted in x-y only, and is only 1-cell across. The NACA 0012 example in $CART3D/cases/samples/naca0012/ is designed to illustrate the setup and running of 2D problems.     **File Format of Meshes produced by Cubes** The file format used by Cart3D to store meshes relies heavily upon the fact that the meshes are Cartesian to store meshes in a highly compressed format. This produces quite small mesh files, but the format takes a little explanation. For users who wish to use Cart3D meshes in other applications, or wish to write their own visualization software for these meshes, this format is described in the \*pdf document [fileformat.pdf (50kb)](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/fileformat.pdf) and an ansi-C I/O library is available for use.  If you're thinking of writing code to read Cart3D meshes (or flowCart's checkpoint files) we generally encourage you to download a copy of the **Cart3D I/O library** (NASA OpenSource) from the  [NASA Software Catalog](https://software.nasa.gov)  which makes working with these files much easier.   [(top)](#page-meshgeneration-html--top) **Further Documentation** Many of the algorithms in cubes are described in:  *[AIAA Paper 97-0196: Robust and Efficient Cartesian Mesh Generation for Component-Based Geometry](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA-97-0196.pdf)*  35th AIAA Aerospace Sciences Meeting, Reno NV Jan, 1997.  (1.7Mb, acrobat format) [(top)](#page-meshgeneration-html--top)  |  |  | | --- | --- | | ---   Questions?   Visit  [Cart3D Discuss](https://groups.google.com/forum/#%21forum/cart3d),   or  [Contact Us](#page-cart3d-team-html)  Last update December, 2024.   --- |  | |

---

<a id="page-flowsolvers-html"></a>

## Flow Solvers

*Original page: [flowSolvers.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/flowSolvers.html)*

Cart3D's Flow Solvers  ---  **[Tell me a little about flowCart?](#page-flowsolvers-html--flowcart)**  **[What is the difference between "flowCart" and "mpix\_flowCart"?](#page-flowsolvers-html--what)**    ---     **Tell me a little about FlowCart****?**  **[flowCart](#page-flowcart-html)**/[mpix\_flowCart](#page-flowcart-html) is the basic inviscid solver being released with Cart3D. It is a scalable, multilevel,  linearly-exact upwind solver and uses domain-decomposition to achieve very good scalability for both steady and time-dependent flows. On most modern machines it converges well over 1 million cells per-core in under 10 minutes. [flowCart](#page-flowcart-html) is very tightly integrated into Cart3D and all of our  automation tools are built around it. Since it is a multilevel code, it converges very quickly and includes our latest work on low-dissipation approaches, solid wall boundaries, mesh interfaces and limiters. Both the parallelization and multigrid are completely transparent to the user and are turned on by simple command line arguments to encourage their use.  [(top)](#page-flowsolvers-html--top)   **[What is the difference between "flowCart" and "mpix\_flowCart"](#page-flowsolvers-html--what)** It's basically a choice between shared and distributed memory. Cart3D's default solver module is called "flowCart" which is a shared-memory build and can be run in parallel within a single compute node. "mpix\_flowCart" is a distributed memory build. Both builds use the same strategy for domain-decomposition and have very similar scalability on most systems. So...which build should you use? We recommend all users start with flowCart, and then, after warming up, start experimenting with mpix\_flowCart if they happen to be in a distributed computing environment. With multi-core CPUs, shared memory is more relevant than ever, but if you're in a cluster-computing environment or have access to [large parallel systems](http://www.nas.nasa.gov/hecc/resources/pleiades.html), distributed memory is the fastest way to kill large problems. So, if you have a cluster of 8 machines, each of which has two 8-core cpu's (16 processing cores per cluster node), you can run flowCart on up to 16 threads in each box. If you need to run more threads, or if you need more memory than you have in a single box you can choose mpix\_flowCart to distribute your job over multiple cluster nodes. flowCart has been run in shared memory on over 1000 cores here in [NASA's Supercomputing division.](http://www.nas.nasa.gov/) We routinely run long, unsteady, distributed-memory simulations with mpix\_flowCart on many thousands of cores. [(top)](#page-flowsolvers-html--top)   ---   Questions?   Visit [Cart3D Discuss](https://groups.google.com/forum/#%21forum/cart3d),   or  [Contact Us](#page-cart3d-team-html)  last update December 2024   --- |

---

<a id="page-postprocess-html"></a>

## Pre- and Post-Processing

*Original page: [postprocess.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/postprocess.html)*

Pre- and Post-Processing  ---  **[What external tools are available for pre-processing Cart3D triangulations?](#page-postprocess-html--preprocess)**   **[Post processing Options?](#page-postprocess-html--postprocess)**   **[Force and Moment Extraction using CLiC](#page-postprocess-html--clic)**     **[On-the-fly component-based force and moment extraction](#page-postprocess-html--fomo)**  [**Point and line "sensors" for data collection**](#page-postprocess-html--sensors)  **[What about CGNS?](#page-postprocess-html--cgns)**     --- **What external tools are available for pre-processing Cart3D triangulations?** **File Import Tools:**   Cart3D comes with several tools for for importing data. Most of these are documented on the Surface Modeling pages. These tools allow you to import geometry from STL, DXF, OFF, FNF triangulations, or from structured surface meshes in LaWGS or Plot3D format. Using these tools you can easily convert your input geometry into Cart3D as a [single component](#page-cart3dtriangulations-html--1-component-file-forma), [configuration](#page-cart3dtriangulations-html--2-configuration-file-format), or a [final wetted surface](#page-cart3dtriangulations-html--3-wetted-surface-triangulation-format). In addition to those distributed with the package, other geometry import tools are available through [the discussion group](#page-maillist-html).     **Native Triangulations:**   A number of packages can write Cart3D triangulations natively, these include (among others) [Pointwise](https://www.cadence.com/en_US/home/tools/system-analysis/computational-fluid-dynamics/pointwise.html), [OpenVSP](https://openvsp.org), [ESP/OpenCSM](https://acdl.mit.edu/ESP/), [Tecplot](https://www.tecplot.com). We also have plugins for the popular [Blender](https://www.blender.org/features/) animation and modeling package, Contact us if you're interested in this. In addition to the plugins and native support in existing packages, an open source software library is available through the [NASA software catalog](https://software.nasa.gov) for natively reading and writing Cart3D triangulations which you can link to from any source code package.  **Overgrid**:   But what about actually viewing and manipulating these components? Fortunately, several users and collaborators have developed tools to allow you to do this. Of those that have found their way back to us, William Chan's Overgrid package is one of the best and the most accessible. **Overgrid** is actually part of NASA's Chimera Grid Tools CGT) package, but thanks to a happy collaboration with the CGT team, the software can read, manipulate and write Cart3D [single component](#page-cart3dtriangulations-html--1-component-file-forma), [configuration](#page-cart3dtriangulations-html--2-configuration-file-format), or  [wetted surface](#page-cart3dtriangulations-html--3-wetted-surface-triangulation-format) triangulations. For viewing triangulations, and moving components, its extremely useful. CGT's "triged" command line tool is also very powerful. Here are a couple of screen shots showing overgrid's triangulation editor: |  |  | | --- | --- | |  |  | | click to enlarge either image |  |   One final note. Although within **Cart3D**, we don't enforce file naming conventions, **Overgrid** does expect triangulations to conform to the naming conventions listed [here](#page-cart3dtriangulations-html). **Overgrid** is available through the [NASA Software Catalog](https://software.nasa.gov)    autoInputs     Once you have an surface triangulation, [autoInputs](#page-meshgeneration-html--autoinputs)  is a new utility that automatically generates the [input.c3d](#page-input-c3d-html) and [preSpec files](#page-prespec-c3d-cntl-html) that the mesher ([cubes](#page-meshgeneration-html)) needs to get going. This utility befitted greatly from beta-testing and we strongly recommend its use by both novice and experienced users alike. Read all about autoInputs on the [mesh generation page](#page-meshgeneration-html).    **trix**      The aptly-named **trix** utility performs format convertion, component/config translations, rotations, and more. The default action is to convert the input files to the Cart3D extended triangulation (VTK) format. Shape sensitivties (if present) are automatically adjusted to reflect any geometry manipulations. **trix** can also be used to move/rotate selected disconnected components with respect to others in the configuration.  *([top](#page-postprocess-html--top))* **Postprocessing Options?** **Volume/Surface Visualization:**   Cart3D solutions are plottable using a variety of commercially available (and free) scientific visualization packages. The most popular are Tecplot Inc.'s Tecplot 360 and Fieldview products along with Kitware's Paraview package ([see](#page-notes-html--note-2) **Note 2**). Here is a table summarizing the postprocessing options supported by **cubes,** **flowCart** and some of the auxiliary programs:      |  |  |  | | --- | --- | --- | | **program** | **dataset type** | **commercial post-processor supported** | | **intersect** | surface triangulation | Tecplot, Fieldview, Paraview | | **cubes** | cutting planes | Tecplot, Fieldview, Paraview | | **mgPrep** | cutting planes | Tecplot, Paraview | | **c3dvis** | 3D volume mesh + flow variables | Tecplot, Fieldview, Ensight, Paraview | | **flowCart, adjointCart** | surface triangulations + [cutting Planes](#page-flowcart-io-html--cutplanes) + flow variables | Tecplot, Fieldview, Paraview |   *([top)](#page-postprocess-html--top)*  **CLiC: Force and Moment Extraction using CLiC (Post-Process)**  **[CLiC](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/clic/clic.html)** is a Component based force and moment module developed as a post-processor and data-extractor for **Cart3D**. Its an extremely flexible and powerful package. Using **clic**, you can extract Cp cuts on any component or group of components in your configuration, you can compute LDM (lift, drag, moment) for components, component groups or configurations. You can also use it to extract the usual bevy of point-moments, line-moments ("hinge moments") etc.. If you want to see some of what it was designed to do, take a look at the original ISO software project plan ([here](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/clic-spp.pdf), 64kb acrobat format). **Clic** can be run as a postprocessor, or called directly through an API. The [clic home page](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/clic/clic.html) will get you started with the package.    *([top](#page-postprocess-html--top))* **On-the-fly Component-Based Force and Moment Extraction** Starting with version Cart3D v1.4, you can now extract force and moment histories of any component or component group while the simulation is in progress. flowCart's "-his" flag has always given "forces.dat" and "moments.dat" with load evolutions on the entire geometry in the simulation. However, using the new "[$\_\_Force\_Moment\_Pro...](#page-input-cntl-html--force-moment)" section in [input.cntl](#page-input-cntl-html) permits much more flexibility. This section relies upon a [GMP-style component hierarchy](#page-config-xml-html) and automatically outputs a clic-like loads summary table in "loadsTri.dat" at the end of the run. [*(*](#page-postprocess-html--top)*[top](#page-postprocess-html--top))* **Point and Line Sensors for Data Extraction**    Also starting with v1.4, you can extract data along any number of discrete points or lines within the computational domain. Need to do a wake survey? Want to compare with off-body data or irregularly placed pressure taps? This is by far the most accurate way of instrumenting specific locations in the flow field. You specify point or line sensors using 1-line additions to the "[$\_\_Post\_Processing:](#page-input-cntl-html--postprocess)" section in [input.cntl](#page-input-cntl-html) and at the end of your run you'll see data files ID'd using the sensor names "lineSensor<Name>.dat". Data from an arbitrary list of point-sensors gets collected in "pointSensors.dat".    Point sensors can also be used in the  the "[$\_\_Convergence His...:](#page-input-cntl-html--postprocess)" section in [input.cntl](#page-input-cntl-html). In this case, they work like "strip-chart" recorders of data at a particular point in the flow. This allows you to monitor convergence at discrete off-body locations to help understand what's going on in your simulation. When used as convergence monitors, each point sensor produces a "pointSensor\_<Name>.dat" file with the state vector and Cp's as a function of iteration count.  *([top](#page-postprocess-html--top))* **What about CGNS?** Good question. There is currently no **[CGNS](http://www.cgns.org/)** standard for Cartesian grids, but it is something that we'd very much like to support and we're working with the **CGNS** team to develop this standard. When this is done, we'll add **CGNS** support into **Cart3D**. Of course The CGNS system already does support unstructured surface triangulations and we do have translators for Cart3D surface triangulations (all types) and **CGNS**. These were written by the [CGNS team](http://www.cgns.org/) and you should drop a note to *[CGNS-Support@CGNS.org](mailto:CGNS-Support@CGNS.org)* to get a copy of these translators.    *([top](#page-postprocess-html--top))*  ---  *last update December, 2024.* |

---

<a id="page-maillist-html"></a>

## Mailing List and Discussion Group

*Original page: [mailList.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/mailList.html)*

The Cart3D Mailing List & Discussion Group  ---      Is there a Cart3D Discussion Group?  Yes. After quite a bit of tinkering, we've switched from direct email-based mailing list to a to a web-based discussion group for Cart3D. We're hosting the group through google groups. Please visit  <http://groups.google.com/group/cart3d> for a look. You need to be a member to view the content and participate in discussions. Anyone can join, and you do not need an invitation. **Why google groups?** We're hosting this through google groups. Several factors weighed into this decision, universal accessibility, features, etc.. Frankly I liked the searching and presentation of archival posts. New posts typically show in a matter of seconds, are searchable within minutes, and show up in RSS feeds shortly thereafter. The environment also provides excellent tools for following, marking and archiving threads.  I wont say that I exhaustively checked all other options, but I did look at 3 or 4 alternatives and was most pleased with this particular environment. **Delivery options: e-mail, RSS, web?** Members can choose to receive e-mail notification of all new posts, posts on selected topics only, or no-email whatsoever. The default delivery option is "Email". You can access archived posts either through the web portal or via an RSS feed. In addition, the requirement of a google account intentionally requires you to actively opt-in to join the discussions and should reduce the traffic in off-topic posting. Nevertheless, be aware that you will be required to agree to google's privacy policy and your personal information will be guarded by this policy. I encourage all prospective members to review this policy before joining. **FAQ posts?** Search the discussion group for "FAQ" to see an always-up-to-date list of Frequently Asked Questions. The discussion group also has links to a growing list of "HowTo's" in response to certain frequent queries.     ---     *last update May 2018, [M. Aftosmis](https://www.nas.nasa.gov/about/staff/maftosmis.html)* |

---

<a id="page-betatest-html"></a>

## Getting Cart3D

*Original page: [betaTest.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/betaTest.html)*

Getting Cart3D  ---   **[The NASA Software Catalog](#page-betatest-html--closed)**    **[I want to fill out paperwork by hand](#page-betatest-html--govt)**  **[I am a current Cart3D user, just send me an update](#page-betatest-html--update)**   **[Contacting the NASA Ames Technology Transfer Office](#page-betatest-html--cto)**    --- **The NASA Software Catalog** The **[NASA Software Catalog](https://software.nasa.gov)** is the easiest way to get a copy of Cart3D. From the catalog's home page you can simply search for *cart3d* and it will take you to the [*C**art3D* download page](https://software.nasa.gov/software/ARC-14275-1). Those working for a **federal agency**  or anyone in industry or academia holding an **active federal government contract** can request a copy directly through the [Software Catalog](https://software.nasa.gov). If you're looking to get a copy for purely commercial purposes, it's best to go thought one of the licensees. Many government users also elect to go through a commercial licensee, since they offer support, training, GUIs, CAD integration and support a wider array of platforms. For direct-from-NASA licensing, or for purely educational use at an accredited educational institution see the [licensing](#page-licensing-html) page.   **[I want to fill out paperwork by hand](#page-betatest-html--govt)** If you love paperwork and work for the federal government or if you work for a non-NASA federal government agency, you can obtain the software through a Software Use Agreement. To get these forms, contact:  [hong.v.vong@nasa.gov](mailto:hong.v.vong@nasa.gov). Send the appropriate form back to the [Ames Technology Transfer Office](http://technology.arc.nasa.gov/).  If you have questions, contact information is [below](#page-betatest-html--cto). **I am a current Cart3D user covered by an SUA or FGTL, just sent me an update** Sure! If you downloaded from the NASA Software Catalog, just go there and get the latest copy. If you filled out your Software Usage Agreement by hand, [contact us](#page-cart3d-team-html) and we'll setup a download. Please include both    1. A description of platform you're running on (best is to send the output of "***% uname -a***) and     2. The current distribution that you're working from (found in *$CART3D/VERSIONS*) **Contacting the [NASA Ames Technology Transfer Office](https://www.nasa.gov/ames-technology-transfer-office/)**     <[ARC-TechTransfer@mail.nasa.gov](mailto:ARC-TechTransfer@mail.nasa.gov)> or       Program Specialist <[hong.v.vong@nasa.gov](mailto:hong.v.vong@nasa.gov)>      You can also contact the [NASA Ames Strategic Partnerships Office](https://www.nasa.gov/ames-strategic-partnerships-office/)    *([top](#page-betatest-html--top))*  ---    *last update December 2025, [M. Aftosmis](https://www.nas.nasa.gov/about/staff/maftosmis.html)* |

---

<a id="page-licensing-html"></a>

## Licensing

*Original page: [licensing.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/licensing.html)*

Licensing  ---   Cart3D can be licensed from NASA for commercial use, education, or redistribution. It can also be obtained commercially through one of NASA's current licensees. For information on direct licensing, please contact the [NASA Ames Technology Transfer Office](https://www.nasa.gov/ames-technology-transfer-office/) or the [NASA Ames Strategic Partnerships Office](https://www.nasa.gov/ames-strategic-partnerships-office/).    ---  *last update December 2025, [M. Aftosmis](mailto:michael.aftosmis@nasa.gov)* |

---

<a id="page-howto-samples-power-readme-html"></a>

## Boundary Conditions for Simulating Propulsive Effects — documentation and tutorials

*Original page: [howto/samples_power/README.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/howto/samples_power/README.html)*

<a id="page-power-back-pressure-for-subsonic-outflow"></a>

### Back Pressure for Subsonic Outflow

This boundary condition allows the user to set a constant pressure on a boundary face where subsonic outflow (leaving the domain) is desired. It is typically used inside inlets. The boundary condition implementation safeguards against reverse flow (flow going back into the domain) by enforcing a solid wall if this occurs. It also enforces the outflow normal Mach number to be no greater than sonic. A [sample case](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-1) is provided with the Cart3D distribution that exercises this outflow boundary condition.

To apply this boundary condition, the user specifies the tag ''`InletPressRatioBC`'' followed by the tag ID of the surface triangulation subset where flow should leave the domain and the back pressure value (normalized by the freestream pressure). These values are specified in the boundary conditions section of the [input.cntl](https:*www.nas.nasa.gov/publications/software/docs/cart3d/pages/input_cntl.html) file as shown in the example below.

__Example:__

```
$__Boundary_Conditions:
# Inlet boundary condition
InletPressRatioBC   2   1.35       # p/p_∞

```

User specifies only the back pressure normalized by the freestream pressure

<span style="padding-left:3em;content:''"></span>
$$p_{\text{back}}/p_\infty$$

Note that in a Cart3D solution

<span style="padding-left:3em;content:''"></span>
$$p_\infty=\dfrac{1}{\gamma}$$

<a id="page-power-boundary-conditions"></a>

### Boundary Conditions

Two subsonic inflow and two subsonic outflow boundary conditions are documented here. They are listed below with more details in their specific sections:

__General Inflow/Outflow Boundary Condition__

[Full State with Riemann Solver](#page-power-full-state-with-riemann-solver) - user specifies complete reference state

__Subsonic Outflow Boundary Conditions__

[Back Pressure for Subsonic Outflow](#page-power-back-pressure-for-subsonic-outflow) - user specifies a back pressure

[Normal Velocity for Subsonic Outflow](#page-power-normal-velocity-for-subsonic-outflow)  - user specifies an outgoing normal velocity

__Subsonic Inflow Boundary Conditions__

[Stagnation Properties for Subsonic Inflow](#page-power-stagnation-properties-for-subsonic-inflow) - user specifies total pressure and total temperature

[Mass Flow Rate and Total Temperature for Subsonic Inflow](#page-power-mass-flow-rate-and-total-temperature-for-subsonic-inflow) - user specifies total temperature and mass flow rate per unit area

<a id="page-power-cart3d-non-dimensionalization"></a>

### Cart3D Non-Dimensionalization

Flow state variables within ''`flowCart`'' are normalized as they are in most typical NASA CFD codes. With the default coordinate system ($$\hat{x}$$ out the back of the model, $$\hat{y}$$ is up, and $$\hat{z}$$   is spanwise), the speed of sound and primitive state variables are normalized as shown below. The gas constant and temperature are also shown for completeness. Variable with the infinity subscript ($$\infty$$) are freestream quantities. Variables with a tilde (~) signify dimensional variables while those without the tilde or the non-dimensional variables internal to ''`flowCart`''.

<span style="padding-left:3em;content:''">
$$c=\dfrac{\tilde{c}}{\tilde{c}_{\infty}} \qquad \qquad\,\, c_{\infty}=1$$
</span>

<span style="padding-left:3em;content:''">
$$\rho=\dfrac{\tilde{\rho}}{\tilde{\rho}_{\infty}} \qquad \qquad \:\! {\rho}_{\infty}=1$$
</span>

<span style="padding-left:3em;content:''">
$$u=\dfrac{\tilde{u}}{\tilde{c}_{\infty}} \qquad \qquad  \:\!u_{\infty}=M_{\infty}\cos\alpha \cos\beta$$
</span>

<span style="padding-left:3em;content:''">
$$v=\dfrac{\tilde{v}}{\tilde{c}_{\infty}} \qquad \qquad\, v_{\infty}=M_{\infty}\sin\alpha \cos\beta$$
</span>

<span style="padding-left:3em;content:''">
$$w=\dfrac{\tilde{w}}{\tilde{c}_{\infty}} \qquad \qquad \!\!w_{\infty}=M_{\infty}\sin\beta$$
</span>

<span style="padding-left:3em;content:''">
$$p=\dfrac{\tilde{p}}{\tilde{\rho}_{\infty}\tilde{c}^2_{\infty}} \qquad \quad\! p_{\infty}=\dfrac{1}{\gamma}$$
</span>

<span style="padding-left:3em;content:''">
$$R=\dfrac{1}{\gamma}$$
</span>

<span style="padding-left:3em;content:''">
$$T_{\infty}=1$$
</span>

<a id="page-power-ducted-fan-in-subsonic-flow"></a>

### Ducted Fan in Subsonic Flow

This sample case is included in the samples but no documentation is provided here. To run the case, simply source the `COMMANDS.txt` file. Feel free to change the inputs and try different boundary conditions. The geometry is shown below.

Below are the results from running the case as provided. Note, to keep the simulation time reasonably short, the solution is not sufficiently refined.

<a id="page-power-full-state-with-riemann-solver"></a>

### Full State with Riemann Solver

This versatile boundary condition can be used for inflow or outflow that is either subsonic or supersonic. It is the original power condition that was implemented in Cart3D with full documentation given in [AIAA 2004-4837](https:*www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2004_4837.pdf). The basic premise is that the user specifies a complete reference state (density, velocity components, and pressure) and the same Riemann solver used throughout the flow field is applied to the  boundary. Hence, the method is very robust and can be applied to all kinds of inflow and outflow. A [sample case](#page-power-power-cube-sample-case) is provided with the Cart3D distribution that exercises this boundary condition.

To apply this boundary condition, the user specifies the tag ''`SurfBC`'' followed by the tag ID of the surface triangulation subset where the boundary condition is to be applied. On the same line, the 5 primitive state values (density, x-velocity, y-velocity, z-velocity, and pressure) are provided. Note these values are not non-dimensionalized. These values are specified in the boundary conditions section of the [input.cntl](https:*www.nas.nasa.gov/publications/software/docs/cart3d/pages/input_cntl.html) file as shown in the example below.

__Example:__

```
$__Boundary_Conditions:
# Exit is component ID 2
SurfBC  2  2.0 3.0 0.0 0.0 5.0      # compID  rho  xvel  yvel  zvel  press
# Inlet is component ID 3
SurfBC  3  1.0 1.5 0.0 0.0 0.714285 # compID  rho  xvel  yvel  zvel  press

```
Please reference the [non-dimensionalizations of the Cart3D primitive variables](#page-power-cart3d-non-dimensionalization) within the solver itself.

<a id="page-power-introduction"></a>

### Introduction

Simulating propulsive effects in Cart3D can be accomplished by exercising the boundary conditions that control flow into and out of the computational domain. The actual propulsion system (propeller, turbofan, scramjet, etc.) is not modeled explicitly but instead a part of the triangulated model surface is tagged for application of a special inflow/outflow boundary. This documentation details how these boundary conditions are applied and describe the handful of example applications provided with the Cart3D distribution. More details on these boundary conditions including the formulation and implementation can be found in [AIAA 2004-4837](https:*www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2004_4837.pdf) and [AIAA-2018-0334](https:*www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2018-0334.pdf).
@@

<a id="page-power-mach-number-steering"></a>

### Mach Number Steering

Sometimes in simulating propulsion systems, it is desirable to specify the average Mach number through outflow (inlets). The [constant velocity outflow](#page-power-normal-velocity-for-subsonic-outflow) boundary condition allow users to do just that. The outflow boundary condition needs to be "steered" to obtain the desired average Mach number. The [Mach number steering sample case](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-4) demonstrates this capability.

The Mach number steering through an outflow boundary surface is controlled through a line in the Steering Information section of the [input.cntl](https:*www.nas.nasa.gov/publications/software/docs/cart3d/index.html) file as shown in the example below. The user specifies the tag ''`TargetMach`'' followed by  the tag ID of the boundary surface, the desired (target) average Mach number, an allowable tolerance for this Mach number, the multigrid cycle when steering would start, and the number of cycles in between boundary condition updates. Note that during a simulation, when the computed average Mach number is within the tolerance of the target value, the boundary condition state is not updated. The starting multigrid cycle for the steering allows the solution to converge a bit before major changes are made to the boundary condition. It also allows for steering to be delayed for more refined meshes when adjoint-driven mesh refinement is employed.

__Example:__

```
$__Steering_Information:
#         Comp   MFR target  tolerance   start  freq
#         (int)   (float)     (float)    (int)  (int)
TargetMach   3       0.5       0.001      50     20

```
User specifies the target average Mach number and an allowable tolerance on that rate. The starting cycle and the frequency for steering are also specified.

<a id="page-power-mass-flow-rate-and-total-temperature-for-subsonic-inflow"></a>

### Mass Flow Rate and Total Temperature for Subsonic Inflow

This boundary condition allows the user to set the mass flow rate through the entire boundary surface and total temperature of flow entering the computational domain. The flow is set to enter the domain normal to the area averaged normal of the boundary surface. It is typically used in nozzles and other flow outlets. The implementation safeguards against supersonic flow at the boundary. A [sample case](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-2) is provided with the Cart3D distribution that exercises this outflow boundary condition. The [mass flow rate steering sample case](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-3) also uses this boundary condition since it allows the user the explicitly set mass flow rate.

To apply this boundary condition, the user specifies the tag ''`PowerMFRBC`'' followed by the tag ID of the surface triangulation subset where flow is to enter the domain, the mass flow rate per unit area (technically normalized by the freestream density times the freestream speed of sound but that is unity) and the total temperature (normalized by the freestream value).  These values are specified in the boundary conditions section of the [input.cntl](https:*www.nas.nasa.gov/publications/software/docs/cart3d/pages/input_cntl.html) file as shown in the example below.

__Example:__

```
$__Boundary_Conditions:
# Nozzle boundary condition
PowerMFRBC          3   0.78 1.05   # m_dot  T_t/T_t∞
```
User specifies the mass flow rate normalized by the freestream density and speed of sound. Note that this is the total mass flow rate through the component, so the area is not normalized. The total temperature normalized by the freestream value.

<span style="padding-left:3em;content:''"></span>
$${\dot{m}_{\text{boundary}}}/\rho_∞c_∞ \qquad T_t/T_{t,\infty}$$

Note that in a Cart3D solution

<span style="padding-left:3em;content:''"></span>
$$\rho_∞=1 \qquad c_∞=1$$

<span style="padding-left:3em;content:''"></span>
$$T_{t,\infty}=1+\dfrac{\gamma-1}{2}M_∞^2$$

<a id="page-power-mass-flow-rate-steering"></a>

### Mass Flow Rate Steering

Often in simulating propulsion systems, it is desirable to specify a mass flow rate through outflow (inlets) and inflow (nozzles) surfaces. The [constant velocity outflow](#page-power-normal-velocity-for-subsonic-outflow) and [mass flow rate inflow](#page-power-mass-flow-rate-and-total-temperature-for-subsonic-inflow) boundary conditions allow users to do just that. The inflow boundary condition explicitly sets the mass flow rate. However, the outflow boundary condition needs to be "steered" to obtain the desired mass flow rate. The [mass flow rate steering sample case](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-3) demonstrates this capability.

The mass flow rate steering through an outflow boundary surface is controlled through a line in the Steering Information section of the [input.cntl](https:*www.nas.nasa.gov/publications/software/docs/cart3d/index.html) file as shown in the example below. The user specifies the tag ''`TargetMFR`'' followed by  the tag ID of the boundary surface, the desired (target) mass flow rate, an allowable tolerance for this mass flow rate, the multigrid cycle when steering would start, and the number of cycles in between boundary condition updates. Note that during a simulation, when the computed mass flow rate is within the tolerance of the target value, the boundary condition state is not updated. The starting multigrid cycle for the steering allows the solution to converge a bit before major changes are made to the boundary condition. It also allows for steering to be delayed for more refined meshes when adjoint-driven mesh refinement is employed.

__Example:__

```
$__Steering_Information:
#         Comp   MFR target  tolerance   start  freq
#         (int)   (float)     (float)    (int)  (int)
TargetMFR   3       0.8       0.0001      50     20

```
User specifies the target mass flow rate (normalized as discussed in the [inflow boundary condition](#page-power-mass-flow-rate-and-total-temperature-for-subsonic-inflow) documentation) and an allowable tolerance on that rate. The starting cycle and the frequency for steering are also specified.

<a id="page-power-normal-velocity-for-subsonic-outflow"></a>

### Normal Velocity for Subsonic Outflow

This boundary condition allows the user to set a constant normal outflow velocity on a boundary face. It is typically used inside inlets. The boundary condition implementation safeguards against an outflow normal Mach number that is greater than sonic. A [sample case](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-2) is provided with the Cart3D distribution that exercises this outflow boundary condition. The boundary condition can also be used to [steer mass flow rate](#page-power-mass-flow-rate-steering) through an outflow boundary, as shown in the [mass flow rate steering sample case](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-3).

To apply this boundary condition, the user specifies the tag ''`InletVelocityBC`'' followed by the tag ID of the surface triangulation subset where flow is leave the domain and the normal flow velocity (technically normalized by the freestream speed of sound though that value is unity). These values are specified in the boundary conditions section of the [input.cntl](https:*www.nas.nasa.gov/publications/software/docs/cart3d/pages/input_cntl.html) file as shown in the example below.

__Example:__

```
$__Boundary_Conditions:
# Inlet boundary condition
InletVelocityBC     2   0.6       # u_norm/c_∞
```
User specifies only the velocity parallel to the average normal of the boundary surface normalized by the freestream speed of sound

<span style="padding-left:3em;content:''"></span>
$$u_{\text{normal}}/c_\infty$$

Note that in a Cart3D solution, the speed of sound is unity

<span style="padding-left:3em;content:''"></span>
$$c_\infty=1$$

<a id="page-power-power-cube-sample-case"></a>

### Power Cube Sample Case

This example is designed as a simple introduction to working with power boundary conditions in Cart3D. It shows how to using the ''`SurfBC`'' tag to specify both inlet and exit boundary conditions on appropriately labeled portions of the surface of your geometry. The case files are located in `$CART3D/cases/samples/powerCube`.
@@

You can run this example automatically by typing:

```
% source ./COMMANDS.txt
```

@@float:right;padding-left:1.5em;caption:"geometry";@@

This example begins with a pre-made surface triangulation of a simple box. This box has been labeled with 3 different component ID's so that surface boundary conditions can be applied on various faces of the box. In this example,

*Component 1 is the body of the box
*Component 2 is the high-X face and
*Component 3 is the  low-X face

This box is pre-labeled, however, you could have labeled it yourself using the `breakTris` utility application (provided in the Cart3D distribution) by simply giving it a single component box as an input triangulation and seed triangles on the low & high-X faces.

The example considers supersonic flow coming from the left, so the low-X face (component 3) will be an "inlet" face. The high-X face will be an "exhaust" face. The inlet is supersonic so the flow will enter cleanly. The exit conditions are underexpanded so we expect a large plume. Note: This example is a tutorial on the mechanics of ''`SurfBC`'' flag in `input.cntl` for a deeper understanding of inlet and exit BCs and how to setup physically relevant states please see
[AIAA 2004-4837](https:*www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2004_4837.pdf). This publication provides validation of inlet and exit boundary conditions for a Cartesian method. It also provides a detailed discussion of the procedure for establishing inlet and exit states for both subsonic and supersonic conditions.

Below is the description of each step in the `COMMANDS.txt` script. Performing these manually is an excellent way to better understand the process.

!! 1. Use ''`autoInputs`'' to prepare input files for ''`cubes`'':
!! 

```
% autoInputs -t Components.i.tri -r 4 -maxR 9

```

* `-t <name of input triangulation>` – This option bases the mesh upon the cube triangulation named `Components.i.tri`.

* `-r 4` – This option places the far-field boundary 4 "radii" away. ''`autoInputs`'' uses the maximum dimension of the geometry as a length scale and then uses this radii as a base for mesh dimensions. The default is 30, but since the flow is completely supersonic in this case, the domain can be much more confined. @@color:mediumpurple;*This argument is optional.*@@

* `-maxR 9` – By default, autoInputs, will plan on cubes running 11 levels of subdivision. Since we're running a very small domain (`-r 4`) we tell ''`autoInputs`'' to setup the input files for only 9 levels of refinement. @@color:mediumpurple;*This argument is optional.*@@

The ''`autoInputs`'' tool generally does an excellent job of setting up both the outer the mesh dimensions and automatically creating  a collection of prespecified adaptation regions ("preSpec boxes"). It creates both `input.c3d` and `preSpec.c3d.cntl` files. At NASA Ames, we use ''`autoInputs`'' almost exclusively for setting up meshes and strongly recommend that users give it a try. The algorithm used in autoInputs follows a "symmetric but degeneracy adverse" philosophy. So the mesh settings are very good at avoiding geometric degeneracies which can cause problems downstream (see "A Few Technical Topics and Computational Geometry"). Even if you may have some "special" meshing requirements, the input files from autoInputs generally a very good place to start. autoInputs has a variety of flags for further tailoring your mesh, in particular, experiment with the `-symm{X,Y, or Z}` and `-halfBody` flags.

!! 2. Build the mesh:
!!

```
% cubes -pre preSpec.c3d.cntl -v -reorder

```

* `-pre preSpec.c3d.cntl` – This option allows for pre-specified adaptation regions (boxes). In this case, 3 overlapping boxes surround the cube, one box for each component. More about the preSpecs file is available [here](https:*www.nas.nasa.gov/publications/software/docs/cart3d/pages/preSpecNote.html). Experiment in 2D to see what you can do. Generally less complicated preSpec files are better.
* `-v` – This controls how verbose the tool is (try turning it off and see the difference). The opposite is `-quiet`. 
* `-reorder` – This option forces the mesh to be output in SFC order in the file `Mesh.R.c3d`, allowing you to skip running of the ''`reorder`'' utility. In practice, it is almost always worth using the `-reorder` flag, just to streamline the meshing. Otherwise, the unordered mesh file, `Mesh.c3d`, is output.

For this case, ''`cubes`'' is using a total of 3 input files:

# the geometry (`Components.i.tri`)
# the mesh input file (`input.c3d`)
# the preSpec file (`preSpec.c3d.cntl`)

It will build a mesh with around 560k cells where about 50k cells are actually cut against the surface. Since we're using ''`autoInputs`'', your mesh may be slightly different.

!! 3. Prepare the coarse mesh levels from the fine mesh:
!! 

```
% mgPrep -n 4
% \rm Mesh.R.c3d
```
* `-n 4 `''`mgPrep`'' – This option will build a multigrid mesh system with 4 total grids in the hierarchy. The original mesh plus 3 coarser meshes (1 + 3 = 4). This is a lot of multigrid for a coarse mesh like this, especially when running with power boundary conditions, but there is no harm in preparing 4 levels, you can always choose to use fewer at runtime. In cases with power, it is rare that using more than 3 levels is actually helpful.

After generating the coarser versions of the original mesh, you'll note that we're immediately executing a `\rm Mesh.R.c3d` which is the mesh that was built by ''`cubes`''. ''`mgPrep`'' takes as input  the `Mesh.R.c3d` mesh file from ''`cubes`'' and writes the entire mesh hierarchy as `Mesh.mg.c3d`. After you have the multigrid mesh system, the original mesh file is no longer needed, so we just remove it to save disk space.

!! 4. Run the solver ''`flowCart`'' using shared memory ([OpenMP](http:*www.openmp.org)):
!!

```
% setenv OMP_NUM_THREADS 2

```
This example uses the shared-memory version of ''`flowCart`'' running on multiple threads. The number of processing threads is controlled via the `OMP_NUM_THREADS` environment variable. *C-shell* requires a `setenv` command to set this, while *bash* users will use the `export` command. This case will run in under 1 minute using 2 cores of  a 2012-era laptop.  You can choose to run on any number up to the number of compute cores you have available.

The next command actually runs the flow simulation:

```
% flowCart -T -mg 3 -no_fmg -N 100 -fine -no_ckpt

```

* `-T` – This option will induce the simulation to generate a surface visualization file in *Tecplot*^®^ format called `Components.i.dat`.
* `-mg 3` – The simulation will use 3 levels of multigrid acceleration.
* `-N 100` – Run for a total of 30 multigrid cycles (using full-multigrid startup).
*`-fine` – The number of multigrid cycles specified above is for the fine mesh only, not including all the coarse mesh cycles during the full multigrid process.
* `-no_ckpt` – This case runs so fast, there's no point in wasting disk space keeping the restart file around, so we don't save any checkpoint info.

* @@color:red; Note:@@ Convergence monitoring files `history.dat`, `forces.dat`, and `moments.dat` are now produced by default, use `-no_his` to suppress them.

The ''`SurfBC`'' keyword is in `input.cntl` in the  `$__Boundary_Conditions` section of this file. As always tags in this file can appear in any order in the appropriate section of the control file. In this case, the inlet is CompID 3 and the exit is CompID 2. In this example, the ''`SurfBC`'' tags in `input.cntl` look like:
 

```
# Exit is component ID 2
SurfBC  2  2.0  3.0 0.0 0.0  5.0      # compID  rho xvel yvel zvel press
# Inlet is component ID 3
SurfBC  3  1.0  1.5 0.0 0.0  0.714285 # compID  rho xvel yvel zvel press
```

Each ''`SurfBC`'' tag is followed by the integer component ID that it gets applied to and 5 floats describing the state vector in primitive variables (density, u,v,w, & pressure). For a discussion of how to setup physically relevant BC states please see
[AIAA 2004-4837](https:*www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2004_4837.pdf). An additional example, specialized for rocket plumes, can be found in the ["HowTo"](https:*www.nas.nasa.gov/publications/software/docs/cart3d/pages/howto/howTo_RocketPlumes_08.11.pdf) section of the [Cart3D website](https:*www.nas.nasa.gov/publications/software/docs/cart3d/index.html).

Notice that even though we prepared 4 levels of grid, we're only using 3 on the command line ("-mg 3"). When running with a ''`SurfBC`'', especially with supersonic exits, it rarely pays to use more than 3 levels of multigrid. It is also worth noting that a ''`SurfBC`'' will generally play very nicely with polynomial multigrid, so if you are ever in need of more convergence acceleration, it may be worth exploring the `-pmg` option to ''`mgPrep`''. Finally, a ''`SurfBC`'' also can respond well to V-cycle multigrid (generally W-cycles are best). So when you're going for all-out speed, try timing a few V-cycles when using a ''`SurfBC`''.

```
% livePlot.pl

```
The ''`livePlot.pl`''  utility will allow you to examine the convergence history for this case. This utility uses the `xmgrace` plotting package (freeware). It will plot the convergence information in `forces.dat` and `history.dat` and showing something similar to the plot here at the right. While convergence of the residuals does not appear very deep, its mostly due to limiter chatter at the contact discontinuity between the exhaust jet and the external flow. Nevertheless, looking at the forces on the body (red line for example) you can see that the forces are converged to plotting accuracy very quickly in this supersonic flow. Compare your plot to that in `archive_converge.png` shown below.

!! 5. Post-process the solution
!!

```
% tecplot mach.lay

```

If you have *Tecplot^®^* available, you can plot the `cutPlanes.dat` and `Components.i.dat` easily using 'the layout file `mach.lay` that is included in the run directory. Compare your computed solution to that in `archive_machContours.png`. The mesh and solution contours are shown below.

```
% cat loadsCC.dat

```
Starting with Cart3D version v1.5, `loadsCC.dat` (produced on shutdown) now outputs computed mass flow rates for any ''`SurfBC`'' components (all inlets and exits). If you examine this file, you will note that the last 2 lines look something like:

```
SurfBC_2         Mass Flow Rate:        32914.045   Area:        16462.727
SurfBC_3         Mass Flow Rate:       -32925.455   Area:        16462.727
```

These results indicate that the exit (`SurfBC_2`) has mass entering (+) the computational domain, while the inlet (`SurfBC_3`) has mass leaving (-) the domain. Note that these two values are relatively close in value. This indicates that mass flow through the inlet and exit roughly balance.

In this case, `loadsCC.dat` includes data for component `entire`, this was requested in `input.cntl` in the `$__Force_Moment_Processing` section. The example did not initially have a <code>[Config.xml](https:*www.nas.nasa.gov/publications/software/docs/cart3d/pages/Config_xml.html)</code> file. However, this was detected at the startup of ''`flowCart`'' and one was automatically generated. If you don't provide your own `Config.xml`, ''`flowCart`'' only knows about the "entire" component, and any `SurfBC` gets named automatically using their component ID. If you provide `Config.xml` and name the components, ''`flowCart`'' will use the component names instead.

<a id="page-power-ramp-inlet-with-converging-diverging-nozzle"></a>

### Ramp Inlet with Converging-Diverging Nozzle

This sample case is included in the samples but no documentation is provided here. To run the case, simply source the `COMMANDS.txt` file. Feel free to change the inputs and try different boundary conditions. The geometry is shown below.

Below are the results from running the case as provided. Note, to keep the simulation time reasonably short, the solution is not sufficiently refined.

<a id="page-power-sample-cases"></a>

### Sample Cases

Currently two sample cases include full documentation:

[Power Cube Sample Case](#page-power-power-cube-sample-case)

[Two Stream Turbofan in Transonic Flow](#page-power-two-stream-turbofan-in-transonic-flow)

Other examples are provided and only briefly described but do not include tutorials. The tutorials for the turbofan should provide enough information to successfully run the other examples listed below:

[Ducted Fan in Subsonic Flow](#page-power-ducted-fan-in-subsonic-flow)

[Ramp Inlet with Converging-Diverging Nozzle](#page-power-ramp-inlet-with-converging-diverging-nozzle)

<a id="page-power-stagnation-properties-for-subsonic-inflow"></a>

### Stagnation Properties for Subsonic Inflow

This boundary condition allows the user to set the total pressure and total temperature of flow entering the computational domain. The flow is set to enter the domain normal to the area averaged normal of the boundary surface. It is typically used in nozzles and other flow outlets. The implementation safeguards against both reverse (outflow) and supersonic flow at the boundary. A [sample case](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-1) is provided with the Cart3D distribution that exercises this inflow boundary condition.

To apply this boundary condition, the user specifies the tag ''`PowerStagBC`'' followed by the tag ID of the surface triangulation subset where flow is to enter the domain, the stagnation pressure (normalized by the freestream total pressure), and the total temperature (normalized by the freestream total temperature). These values are specified in the boundary conditions section of the [input.cntl](https:*www.nas.nasa.gov/publications/software/docs/cart3d/pages/input_cntl.html) file as shown in the example below.

__Example:__

```
$__Boundary_Conditions:
# Nozzle boundary condition
PowerStagBC         3   1.4 1.05   # p_t/p_t∞  T_t/T_t∞
```
User specifies the total pressure and total temperature, both normalized by the freestream values.

<span style="padding-left:3em;content:''"></span>
$$p_t/p_{t,\infty} \qquad T_t/T_{t,\infty}$$

Note that in Cart3D, the freestream pressure and temperature are set to

<span style="padding-left:3em;content:''"></span>
$$p_{\infty}=\dfrac{1}{\gamma}$$

<span style="padding-left:3em;content:''"></span>
$$T_{\infty}=1$$

Thus, the freestream total pressure can be computed using

<span style="padding-left:3em;content:''"></span>
$$p_{t,\infty}=\dfrac{1}{\gamma} \bigg (1+\dfrac{\gamma-1}{2}M_∞^2 \biggr) ^\tfrac{\gamma}{\gamma-1}$$

and the freestream total temperature

<span style="padding-left:3em;content:''"></span>
$$T_{t,\infty}=1+\dfrac{\gamma-1}{2}M_∞^2$$

<a id="page-power-stylesheet"></a>

### stylesheet

pre code {
    color: #148F77  ;
}

.printButtonClass {
    background: lightgreen;
    float: right;
}
@media print {.printButtonClass{display: none;}}

.tc-subtitle {display:none;}

.justifiedText {
    text-align: justify
}

.tc-page-controls svg.tc-image-print-button {
    fill: #5eb95e;
}

.tc-tiddler-controls svg.tc-image-print-button {
    fill: #5eb95e;
}

<a id="page-power-tableofcontents"></a>

### TableOfContents



<a id="page-power-two-stream-turbofan-in-transonic-flow"></a>

### Two Stream Turbofan in Transonic Flow

This example shows how to run a case with a propulsion system using all of the subsonic inflow/outflow boundary conditions. There is also a case where mass flow rate steering is employed. The example considers an axisymmetric isolated nacelle with two streams of exhaust in transonic flow. Adjoint-based mesh refinement is employed, with the thrust coefficient (-C~D~) as the output of interest.

There are three sample cases that can be run with this example and are listed below:

[Sample Case #1](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-1) - demonstrates [back pressure](#page-power-back-pressure-for-subsonic-outflow) and [stagnation property](#page-power-stagnation-properties-for-subsonic-inflow) boundary conditions

[Sample Case #2](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-2) - demonstrates [normal velocity](#page-power-normal-velocity-for-subsonic-outflow) and [mass flow rate](#page-power-mass-flow-rate-and-total-temperature-for-subsonic-inflow) boundary conditions

[Sample Case #3](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-3) - demonstrates [mass flow rate steering](#page-power-mass-flow-rate-steering)

[Sample Case #4](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-4) - demonstrates [average Mach number steering](#page-power-mach-number-steering)

These cases are all run the same way but just use different boundary conditions and the third case uses mass flow rate steering. To run these cases, follow the procedures below.

__Running Sample Cases__

To begin the run, first the number of cores ([OpenMP](http:*www.openmp.org) threads) is set. This example is designed so the finest mesh contains just under 800K cells so one core is enough to run the solution in a few minutes. Select more cores for a faster run:

```
% setenv OMP_NUM_THREADS 1

```
Before running any case, it is always a good idea to check your inputs:

```
% c3d_checkInputs.py

```
To start the run, execute the included script:

```
% ./aero.csh

```
Alternatively, source `COMMANDS.txt` to execute these commands automatically:

```
% source COMMANDS.txt

```

<a id="page-power-two-stream-turbofan-in-transonic-flow-inputs"></a>

### Two Stream Turbofan in Transonic Flow, Inputs

The surface triangulation is in the `Components.i.tri` file and is shown below. Note the teal-meshed surface is the outflow surface while the red- and orange-meshed surfaces are inflow surfaces.

Note that to keep the file size of this example small, the triangulation is somewhat coarse. The `aero.csh` script requires that the triangulation file name is `Components.i.tri`; you may call your triangulation whatever you like but then you must create a symbolic link to `Components.i.tri`.

The mesh input file is `input.c3d`, which is generated using autoInputs:

```
% autoInputs -t Components.i.tri.

```

The input file for flowCart is `input.cntl`. The output functional is thrust (negative drag) coefficient, which is specified in the `$__Design_Info` section. Comparing the `aero.csh` script in this directory with the default in `$CART3D/bin`, there are several parameters that have been modified from their default settings:

# Requested 6 adapt cycles (instead of 9) to keep the final cell-count small.
# Reduced the number of `flowCart` cycles on most meshes for faster turn-around.
# Reduced number of `adjointCart` cycles for faster turn-around.
# Added flow and adjoint multigrid levels (4 instead of 3) to accelerate convergence.
# Used auto-growth of mesh to demonstrate this feature.

For more information, see the <code>[aero.csh](https:*www.nas.nasa.gov/publications/software/docs/cart3d/pages/adjoint/doc_adj_adapt.html)</code> documentation.

<a id="page-power-two-stream-turbofan-in-transonic-flow-sample-case-1"></a>

### Two Stream Turbofan in Transonic Flow, Sample Case #1

This case on the turbofan uses the [back pressure](#page-power-back-pressure-for-subsonic-outflow) boundary condition on the fan face and the [stagnation properties](#page-power-stagnation-properties-for-subsonic-inflow) condition on the inflow planes within the nozzle. The inputs for this case are the same as the other cases:

[Inputs](#page-power-two-stream-turbofan-in-transonic-flow-inputs)

The outputs for this case are more detailed than the other cases:

[Outputs](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-1-outputs)

<a id="page-power-two-stream-turbofan-in-transonic-flow-sample-case-1-outputs"></a>

### Two Stream Turbofan in Transonic Flow, Sample Case #1 Outputs

The primary output files are: `fun_con.dat` that contains the number of control volumes and the value of the thrust coefficient (C~T~ = -C~D~); `results.dat` that contains the corrected C~T~ and error estimates; `all_outputs.dat` that records mesh convergence of each component of the mesh adaptation functional (in this case C~T~ only); and `user_time.dat` that contains a summary of the CPU time used. This directory also contains `fun_con.archive.dat` and `results.archive.dat` to help check your run. If you're using one core, the numbers should be very close; otherwise, the numbers should approximately agree.

Below, we plot the primary outputs of the adaptation. Top-left figure shows mesh convergence of C~T~ with error bars indicating the level of discretization error (columns 2 and 3 from `fun_con.dat`, and column 9 from `results.dat` for error bars). Changes in C~T~ over the last three meshes are relatively small, indicating that while more adaptation cycles are required for tight convergence, the current most refined mesh has compute thrust reasonably well. Top-right figure shows convergence of several error estimates: the red line (squares) shows the reduction in discretization error after each adaptation, i.e., the magnitude of the error bars from the left plot, the black line (circles) is the magnitude of the adaptation error indicator (column 6 in results.dat), and the blue line is the change in C~T~ relative to the previous mesh. All three error measures decrease as the mesh is refined beyond the first couple of meshes. The current best value of C~T~ contains about 3% error. The bottom figure shows the iterative convergence of C~T~ for all adaptation cycles (plotted from `BEST/FLOW/functional.dat`). After each warm-start, the initial transient quickly dissipates and the value of C~T~ levels out. This is a good check to make sure that the solver is running a sufficient number of multigrid cycles on each mesh. For further details, see the Output Files documentation.

The composite plot above was generated with `aerograce.csh`:

```
% aerograce.csh
```
Note that this requires the `xmgrace` plotting package. Basic `xmgrace` commands for examining `fun_con.dat` and `results.dat` are:

```
% xmgrace -free -noask -block fun_con.dat -bxy 2:3 -block results.dat -bxy 2:5 -log x -par doc/xmgrace_fun_con.par

% xmgrace -free -noask -block results.dat -bxy 2:6 -log x -log y -par doc/xmgrace_results.par

```

You can obtain similar plots via `gnuplot`:

```
gnuplot% lo "doc/gnuplot_fun_con.par"

gnuplot% lo "doc/gnuplot_results.par"

```

Flow solutions (with meshes) can be viewed via *Tecplot*^®^, if you have it available:

```
% tec360 nacelle_inletView.lay

% tec360 nacelle_nozzleView.lay

```

The layout file points to a symbolic link called `BEST`, hence reloading the layout file after each flow solve should show the latest mesh and flow solution. Edit the layout file to see a specific adaptation cycle (directory). Here are two snapshots of the initial mesh (`adapt00` directory; 11,899 cells) with a cut-plane at the centerline of the nacelle and Mach number contours:

and here is the final mesh (`adapt06`; 791K cells):

To check the convergence of the flow solver and the aerodynamic coefficients, visit any of the adapt directories; for example it's handy to use the `BEST` link:

```
% cd BEST

```

and use the `livePlot.pl` command:

```
% livePlot.pl &

```

which will show you the iterative convergence history of `flowCart` in the current and all previous adaptation directories. Here is the plot from the finest mesh directory (adapt06):

Note that the forces shown in this convergence plot do not include momentum flux forces on the inflow/outflow boundaries. This means the forces do not match the functional convergence as shown from the output of `aerograce.csh` above.

Occasionally, it may be necessary to check the convergence of the adjoint solver (adjointCart). We show an example here using the adjoint solution in the adapt04 directory:

```
% cd adapt04

```

Use the `livePlot.pl` command again, but add the `-adj` option:

```
% livePlot.pl -adj &

```

This shows you the convergence history of `adjointCart` in the current directory by plotting the `historyADJ.dat` file:

The plot shows full-multigrid startup of the adjoint solver over four mesh levels and 120 iterations are performed on the finest mesh. Only residuals are shown, there are no forces or moments to report.

<a id="page-power-two-stream-turbofan-in-transonic-flow-sample-case-2"></a>

### Two Stream Turbofan in Transonic Flow, Sample Case #2

This case on the turbofan uses the [normal velocity](#page-power-normal-velocity-for-subsonic-outflow) boundary condition on the fan face and the [mass flow rate](#page-power-mass-flow-rate-and-total-temperature-for-subsonic-inflow) condition on the inflow planes within the nozzle. The inputs for this case are the same as the other cases:

[Inputs](#page-power-two-stream-turbofan-in-transonic-flow-inputs)

The outputs for this case are more detailed than the other cases:

[Outputs](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-2-outputs)

<a id="page-power-two-stream-turbofan-in-transonic-flow-sample-case-2-outputs"></a>

### Two Stream Turbofan in Transonic Flow, Sample Case #2 Outputs

The outputs are very similar to those of [Sample Case #1](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-1-outputs), and so many details are not repeated here. The results for this case are shown below, starting with the mesh convergence of functional which again is thrust (C~T~) which is the negative of drag (C~D~):

Note that in this case, the functional is still significantly changing even with the finest meshes. Clearly this case requires further adaptation, though that is not done here to keep the mesh size manageable. The user is encouraged to run more adaptation cycles, probably with more CPU cores for faster turnaround time.

Mach contour plots generated by the included *Tecplot*^®^ layouts are given below. The coarser mesh solutions are from `adapt04` while the next images are generated from the finest mesh (`adapt06`).

The convergence of the residuals and the pressure forces are given in the plot below generated by the `livePlot.pl` script. Each mesh appears to have converged sufficiently.

<a id="page-power-two-stream-turbofan-in-transonic-flow-sample-case-3"></a>

### Two Stream Turbofan in Transonic Flow, Sample Case #3

This case on the turbofan uses the [normal velocity](#page-power-normal-velocity-for-subsonic-outflow) boundary condition on the fan face and the [mass flow rate](#page-power-mass-flow-rate-and-total-temperature-for-subsonic-inflow) condition on the inflow planes within the nozzle. Additionally, it uses mass flow rate steering to force the mass flow rate through the int to be equal to the sum of the rates through the two exhaust streams. The inputs for this case are the same as the other cases:

[Inputs](#page-power-two-stream-turbofan-in-transonic-flow-inputs)

The outputs for this case are more detailed than the other cases:

[Outputs](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-3-outputs)

<a id="page-power-two-stream-turbofan-in-transonic-flow-sample-case-3-outputs"></a>

### Two Stream Turbofan in Transonic Flow, Sample Case #3 Outputs

This case is very similar to [Sample Case #2](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-2) except that [mass flow rate steering](#page-power-mass-flow-rate-steering) has been introduced. We start by using the identical boundary condition for the exhaust inflow surfaces as that in [Sample Case #2](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-2), which is shown below:

```
# Nozzle boundary condition
PowerMFRBC          3   0.78 1.05   # m_dot  T_t/T_t∞
PowerMFRBC          4   0.14 1.3    # m_dot  T_t/T_t∞
```

For this example, we would like to drive the mass flow rate through the outflow (fanface) surface to equal the sum of the inflow rates. To do so, we simply use the sum of the inflow rates as the input for the [mass flow rate steering](#page-power-mass-flow-rate-steering) block in the [input.cntl](https:*www.nas.nasa.gov/publications/software/docs/cart3d/index.html) file:

```
$__Steering_Information:
#         Comp   MFR target  tolerance   start  freq
#         (int)   (float)     (float)    (int)  (int)
TargetMFR   2      0.92        0.0002     500    10

```
The tolerance of 0.0002 says that when the mass flow rate through the outflow boundary is well within 3 digits of accuracy of the target value (0.92), the boundary condition velocity is not updated when the next steering cycle occurs. The starting iteration of 500 for this case allows the mesh to be at a reasonable refinement level so the mass flow rate is relatively accurate. Obviously the more refined the mesh is, the more accurate the mass flow rate will be. The frequency of 10 iterations is typical for these cases, though many values work. The starting iteration and update frequency usually don't make the process diverge but it can affect the efficiency of the steering process.

If the `aero.csh` script for this case is compared with that of [Sample Case #2](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-2), you'll note the number of iterations for each refinement level has been increased. This gives the mass flow rate steering a chance to reach the target value at each level to make the refinement process more efficient. It also allows the steering process to perform most of the work more cheaply in terms of computational resources when the mesh is still relatively small.

The convergence of the residuals and the pressure forces are given in the plot below generated by the `livePlot.pl` script. Each mesh appears to have converged sufficiently.

Note the X-force clearly shows the mass flow rate steering in action after iteration 500. However, around iteration 1060, the steering appears to have leveled off. This is evident in the standard output from `flowCart` which is in the `cart3d.out` file in `BEST/FLOW`:

```
Cycle 1054  Max and L1 Residual of density are: 1.68983801e+02  3.82529478e-03
Cycle 1055  Max and L1 Residual of density are: 1.65193446e+02  3.67534056e-03
    o  Steering: Current MFR and velocity through Component 2:     9.202447566116890e-01      5.354005650687548e-01
    o  Steering: Setting new outflow velocity =     5.353293650824136e-01
Cycle 1056  Max and L1 Residual of density are: 1.59524672e+02  3.58196694e-03
Cycle 1057  Max and L1 Residual of density are: 1.52600667e+02  3.44491816e-03
Cycle 1058  Max and L1 Residual of density are: 1.45357717e+02  3.32727864e-03
Cycle 1059  Max and L1 Residual of density are: 1.38793824e+02  3.22771892e-03
Cycle 1060  Max and L1 Residual of density are: 1.33490322e+02  3.11780042e-03
Cycle 1061  Max and L1 Residual of density are: 1.29419422e+02  2.99199370e-03
Cycle 1062  Max and L1 Residual of density are: 1.26136435e+02  2.87346114e-03
Cycle 1063  Max and L1 Residual of density are: 1.23022657e+02  2.76524282e-03
Cycle 1064  Max and L1 Residual of density are: 1.19499753e+02  2.65904315e-03
Cycle 1065  Max and L1 Residual of density are: 1.15241986e+02  2.57367268e-03
    o  Steering: Current MFR and velocity through Component 2:     9.201816493187803e-01      5.353293650824136e-01
Cycle 1066  Max and L1 Residual of density are: 1.10314685e+02  2.48810575e-03
Cycle 1067  Max and L1 Residual of density are: 1.05135419e+02  2.40741196e-03
```
Note that the last update to the outflow boundary condition occurs at cycle 1055 and no update occurs on cycle 1065 where the mass flow rate is within tolerance. The output from the loadsCC.dat file gives the mass flow rates through the inflow/outflow boundaries:

The mesh convergence of the functional is given in the output from `aerograce.csh` below. It is clear this case could use some more mesh refinement. For this sample case, the maximum refinement level is limited simply to keep the run time short. Feel free to run this case with more refinement levels.

Looking at the end of the loadsCC.dat file, we see what mass flow rates are computed in the finest mesh solution:

```
fanFace          Mass Flow Rate:      -0.92027559   Area:        1.4675018
fanExit          Mass Flow Rate:       0.77999779   Area:       0.72037306
turbineExit      Mass Flow Rate:       0.14000004   Area:       0.15695205
```

Here we see the mass flow rate we requested has been obtained within tolerance. Note that the computation of mass flow rate is a bit more accurate in the post-processing phase of a simulation than during the simulation itself. However, as the mesh is refined, the difference between the two mass flow rate computations should shrink. This means small tolerances in steering must be accompanied by more refined meshes.

<a id="page-power-two-stream-turbofan-in-transonic-flow-sample-case-4"></a>

### Two Stream Turbofan in Transonic Flow, Sample Case #4

This case on the turbofan uses the [normal velocity](#page-power-normal-velocity-for-subsonic-outflow) boundary condition on the fan face and the [mass flow rate](#page-power-mass-flow-rate-and-total-temperature-for-subsonic-inflow) condition on the inflow planes within the nozzle. Additionally, it uses average Mach number steering to force the Mach number of the flow through the fanface to a specified value. The inputs for this case are the same as the other cases:

[Inputs](#page-power-two-stream-turbofan-in-transonic-flow-inputs)

The outputs for this case are more detailed than the other cases:

[Outputs](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-4-outputs)

<a id="page-power-two-stream-turbofan-in-transonic-flow-sample-case-4-outputs"></a>

### Two Stream Turbofan in Transonic Flow, Sample Case #4 Outputs

This case is very similar to [Sample Case #2](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-2) except that [Mach number steering](#page-power-mach-number-steering) has been introduced. For this example, we would like to drive the average Mach number through the outflow (fanface) surface to a specific value (0.45). To do so, we use this value as the input for the [Mach number steering](#page-power-mach-number-steering) block in the [input.cntl](https:*www.nas.nasa.gov/publications/software/docs/cart3d/index.html) file:

```
$__Steering_Information:
#         Comp  Mach target  tolerance  start   freq
#         (int)   (float)     (float)    (int)  (int)
TargetMach  2       0.45       0.001      450    10
```
The tolerance of 0.001 says that when the average Mach number the flow going through the outflow boundary is nearly within 3 digits of accuracy of the target value (0.45), the boundary condition velocity is not updated when the next steering cycle occurs. The starting iteration of 450 for this case allows the mesh to be at a reasonable refinement level so the mass flow rate is relatively accurate. Obviously the more refined the mesh is, the more accurate the Mach number will be. The frequency of 10 iterations is typical for these cases, though many values work. The starting iteration and update frequency usually don't make the process diverge but it can affect the efficiency of the steering process.

If the `aero.csh` script for this case is compared with that of [Sample Case #2](#page-power-two-stream-turbofan-in-transonic-flow-sample-case-2), you'll note the number of iterations for each refinement level has been increased. This gives the Mach number steering a chance to reach the target value at each level to make the refinement process more efficient. It also allows the steering process to perform most of the work more cheaply in terms of computational resources when the mesh is still relatively small.

The convergence of the residuals and the pressure forces are given in the plot below generated by the `livePlot.pl` script. Each mesh appears to have converged sufficiently.

Note the X-force clearly shows the mass flow rate steering in action after iteration 450. Very quickly, at around iteration 1060, the steering appears to have leveled off. This is evident in the standard output from `flowCart` which is in the `cart3d.out` file in `adapt03/FLOW`:

```
 Cycle  509  Max and L1 Residual of density are: 3.93213291e+01  2.06098347e-02
 Cycle  510  Max and L1 Residual of density are: 3.43219651e+01  2.13190946e-02
    o  Steering: Current Mach number and velocity through Component 2:     4.517225053328570e-01      4.696934947239964e-01
    o  Steering: Setting new outflow velocity =     4.687979786273673e-01
 Cycle  511  Max and L1 Residual of density are: 2.77771517e+01  2.24168955e-02
 Cycle  512  Max and L1 Residual of density are: 2.18041262e+01  1.99984214e-02
 Cycle  513  Max and L1 Residual of density are: 1.79711857e+01  1.59969292e-02
 Cycle  514  Max and L1 Residual of density are: 1.67218593e+01  1.49264546e-02
 Cycle  515  Max and L1 Residual of density are: 1.73671009e+01  1.51319337e-02
 Cycle  516  Max and L1 Residual of density are: 1.87873950e+01  1.52373659e-02
 Cycle  517  Max and L1 Residual of density are: 1.99838813e+01  1.40246395e-02
 Cycle  518  Max and L1 Residual of density are: 2.02387307e+01  1.17656573e-02
 Cycle  519  Max and L1 Residual of density are: 1.91577174e+01  1.08372471e-02
 Cycle  520  Max and L1 Residual of density are: 1.67244916e+01  1.09495988e-02
    o  Steering: Current Mach number and velocity through Component 2:     4.508389998945563e-01      4.687979786273673e-01
 Cycle  521  Max and L1 Residual of density are: 1.32945225e+01  1.09832246e-02
 Cycle  522  Max and L1 Residual of density are: 9.43694933e+00  9.96077988e-03
```
Note that the last update to the outflow boundary condition occurs at cycle 510 and no update occurs on cycle 520 where the mass flow rate is within tolerance. Checking the flowCart output from all future adapt cycles also shows no change to the boundary condition. This suggests Mach number steering is extremely quick and stable.

The mesh convergence of the functional is given in the output from `aerograce.csh` below. It is clear this case could use some more mesh refinement. For this sample case, the maximum refinement level is limited simply to keep the run time short. Feel free to run this case with more refinement levels.

Below is a plot of Mach number contours on the symmetry plane of the final solution. Note that the Mach number at the fan face is indeed the requested value of 0.45.

---

<a id="page-adjoint-index-html"></a>

## Cart3D Adjoint-Based Mesh Refinement

*Original page: [adjoint/index.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/adjoint/index.html)*

---

#### Cart3D Adjoint-Based Mesh Refinement

#### User Guide

---

To run a simulation with adjoint-based adaptation, the first
step is to express the goals of the simulation explicitly in
terms of outputs of interest, for example, aerodynamic forces and
moments. Once specified, the execution is controlled by the
[*aero.csh*](#page-adjoint-aero-csh-html) script. The refinement
procedure generates a mesh that minimizes the influence of discretization
errors on the selected outputs. Details of the methodology can be found
in [AIAA-2007-4187](https://www.nas.nasa.gov/assets/nas/pdf/staff/Nemec_M_Adjoint_Error_Estimation_and_Adaptive_Refinement_for_Embedded-Boundary_Cartesian_Meshes.pdf),
[AIAA-2008-0725](https://www.nas.nasa.gov/assets/nas/pdf/staff/Nemec_M_Adjoint-Based_Adaptive_Mesh_Refinement_for_Complex_Geometries.pdf) and [NASA TM-2014-218386](http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20150000864.pdf); additional references are available
[here](#page-publications-publications-html).

To run a case, you need the basic Cart3D input files, namely,
*Components.i.tri*, *input.c3d* and
*input.cntl*, and the [*aero.csh*](#page-adjoint-aero-csh-html) script. The details are
explained below:

- [How do I specify output
  functionals for adaptation?](#page-adjoint-doc-adj-functionals-html)
- [What are the parameters that I
  need to adjust in aero.csh?](#page-adjoint-doc-adj-adapt-html)
- [What files are
  required by aero.csh and how do I run a case?](#page-adjoint-doc-adj-adapt-html--adj-adapt-inputs)
- [What are the
  output files of an adaptation run?](#page-adjoint-doc-adj-adapt-html--adj-adapt-output)

Please work through examples provided in
$CART3D/cases/samples\_adapt. You may like to try the
adaptive NACA0012 case first, followed by
the wing, cone-cylinder and multi-fun
examples.

### What's new in release 1.6.x?

- Robust re-implementation of farfield outputs (adjoint consistent
  with sBOOM version 2.9 or newer, [optSBOOM](#page-adjoint-doc-adj-functionals-html--adj-fun-ff)
  functionals)
- Examples featuring equivalent area and farfield outputs
  (see $CART3D/cases/samples\_adapt/Ae\_n0012
  and $CART3D/cases/samples\_adapt/aceSEL)

### What's new in release 1.5.x?

- Farfield outputs ([optSBOOM](#page-adjoint-doc-adj-functionals-html--adj-fun-ff) functionals)
- Mass-flow outputs ([optMassFlow](#page-adjoint-doc-adj-functionals-html--adj-fun-mfr) functionals)
- Full support of new inflow and outflow boundary conditions
  (InletPressRatioBC, InletVelocityBC, PowerStagBC, and PowerMFRBC in [*input.cntl*](#page-input-cntl-html--bcs))
- Error estimation for multiple outputs, see the $CART3D/cases/samples\_adapt/multi\_fun
  example and the [*Functionals.xml*](#page-adjoint-doc-adj-functionals-html--adj-fun-multi)
  file
- Many [*aero.csh*](#page-adjoint-aero-csh-html) improvements
  - Command line options: -h (-help or --help), -skipfinest,
    and -archive
  - [*auto\_growth*](#page-adjoint-doc-adj-adapt-html--param-auto-growth): automatically sets mesh growth
    factors
  - *mpi\_prefix*: allows you to run mpix\_flowCart
  - Straightforward [restarts](#page-adjoint-doc-adj-adapt-html--adj-adapt-usage) from any adapt directory
  - flowCart and adjointCart iterations (it\_fc, ws\_it, it\_ad)
    refer to fine-grid iterations (see -fine option in
    flowCart)

---

*[Marian Nemec](mailto:marian.nemec@nasa.gov), last update
November 2024*

---

<a id="page-trix-index-html"></a>

## TRIX News

*Original page: [trix/index.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/trix/index.html)*

---

#### New Tricks in *Trix*

---

### Symbolic Surface Deformations

The new command line argument *-sD NAME
"FUNCTION"* modifies the position of vertices of the input
triangulation by using user-specified functions. The argument
*NAME* is a string. The argument *FUNCTION* is a math expression. The value of the
*FUNCTION* is assigned to its *NAME*. The symbols *x*, *y* and *z* are reserved keywords.
They represent vertex coordinates of the input triangulation. For
example, the command

*% trix -sD x "x+2"
file.tri*

shifts all x-coordinates by two units to the right. Multiple
*-sD NAME "FUNCTION"* arguments may be
specified. For example,

*% trix -sD x "x+2" -sD y
"y+1" file.tri*

Multiple *-sD* arguments are parsed
left-to-right, which allows the user to define working (or
temporary) variables. For example, consider a simple nozzle
geometry consisting of three labeled components shown in the figure
below. Component "1" is the "plenum" face, component "2" is the
converging section of the nozzle and component "3" is the diverging
section. The command

*% trix -v -select 2 3
\
-sD deltaTR "-0.1" \
-sD throatX "8.3022118e-01" \
-sD theta "atan2(z,y)" \
-sD r "sqrt(z^2+y^2)" \
-sD tc "(x-xmin)/(throatX-xmin)" \
-sD td "1. - (x-throatX)/(xmax-throatX)" \
-sD Rc "r+deltaTR\*tc" \
-sD Rd "r+deltaTR\*td" \
-sD z "x-throatX > 0 ? Rd\*sin(theta) : Rc\*sin(theta)" \
-sD y "x-throatX > 0 ? Rd\*cos(theta) : Rc\*cos(theta)" \
nozzle.tri*

linearly deforms the nozzle surface to reduce the throat radius
by 0.1, as shown in the right frame of the figure. Use

*% trix -helpSymbolic
-helpVariable*

for definition of supported operators, math functions and all
reserved keywords, such as *xmin* and *xmax* used above.

### Symbolic Tagging

The new command line argument *-sT NAME
"FUNCTION"* is a sibling of the deformer option *-sD* described above. It allows manipulation of
triangle labels, e.g., Component ID's, through use of
user-specified functions. The primary keyword is 'tag', which
represents the Component ID of each triangle. For example,

*% trix -sT tag "(tag==3)
? 5 : tag" file.tri*

changes triangles with component label 3 to 5 and keeps all
others the same. Frequently, we wish to tag a region of the
geometry as a unique component. This can be accomplished by using a
command such as

*% trix -sT tag "(x>0
&& x<10) ? maxtag+1 : tag" file.tri*

which relabels all triangles with vertices that have
x-coordinates between 0 and 10 as a new component. Note the keyword
'maxtag', which holds the highest component tag of the input
triangulation. Another example of tagging a region is the
command

*% trix -sT tag
"(x^2+y^2+z^2 <= 1) ? maxtag+1 : tag" file.tri*

which tags a region of the input triangulation that is contained
within a sphere of unit radius as a new component.

#### New Commands from Previous Releases

### Tagging Surfaces within Rectangular Regions

*% trix -tagRegion XMIN
XMAX YMIN YMAX ZMIN ZMAX file.i.tri*

The command line argument *tagRegion*
assigns a new tag to the wetted surface contained within a
specified rectangular cuboid. This is illustrated in the frame on
right, where the tip region of the cone contained within the cube
would be tagged with a unique ID. The rectangular region is
specified via its min and max corners *XMIN XMAX
YMIN YMAX ZMIN ZMAX*. The new tag index is selected
automatically as the next highest available component in the input
triangulation.

### Rotating about Specified Axis

*% trix -cx XT -cy YT -cz
ZT -rg XH YH ZH ANG file.i.tri*

To rotate the geometry about an arbitrary axis, use the general
rotation argument *rg* in conjunction with the
center of rotation flags *cx cy cz*. The tail
of the rotation vector is specified by *XT YT
ZT*, the head by the first three arguments of *rg XH YH ZH*, and the rotation angle in degrees is the
fourth argument *ANG*.

### Selecting Components for Geometry Manipulation

*% trix -select 1 2 5 -x
2 file.i.tri*

Previous versions of *trix* applied geometry
manipulations, such as translations and rotations, to the entire
triangulation. The new *select* command line
argument lifts this restriction. When *select*
is specified, the geometry manipulations are applied to the
selected components only. For example, the command above moves
components 1, 2 and 5 by two units in the x-direction. A practical
example is manipulating control surfaces within a configuration, as
illustrated in the animation below. Here, rotations about specified
axes are used to move the control surfaces without needing to first
separate the components.

To see all options and more details:

*% trix -help*

or just

*% trix -*

[(top)](#page-trix-index-html--top)

---

*Questions? ... visit [Cart3D
Discuss](http://groups.google.com/group/cart3d)*

*Last update February
2024*

---

<a id="page-flowcart-html"></a>

## flowCart

*Original page: [flowCart.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/flowCart.html)*

Overview ---   **[What is flowCart?](#page-flowcart-html--flowcart)**  **[What kind of discretization does it use?](#page-flowcart-html--discretization)**   **[How is flowCart integrated into the Cart3D?](#page-flowcart-html--integration)**  **[How can I extract force and moment data? What's CLiC?](#page-flowcart-html--clic)**  **[Parallelization and Domain Decomposition](#page-flowcart-html--parallelization)**  **[Multigrid](#page-flowcart-html--multigrid)**    ---     **What is flowCart?** **flowCart** is a scalable, multilevel, solver for the Euler equations governing the inviscid flow of a compressible fluid. Meshes from cubes are treated as unstructured collections of Cartesian cells, and it takes advantage of the fact that cells are Cartesian wherever possible to reduce the operation count. Both the parallelization and multigrid are completely transparent to the user and are turned on by simple command line arguments to encourage their use. OpenMP and MPI versions of flowCart use the same command line arguments, and scale similarly.  *[(top)](#page-flowcart-html--top)* **What kind of discretization does it use?** **Spatial Discretization:** **flowCart** uses cell-centered, finite-volume, upwind differencing. There are several flux functions provided, and it is extensible, so that you can add your favorite to the growing list of available flux functions. The linear reconstruction scheme makes it formally second-order accurate, and great pains have been taken to ensure that this accuracy is preserved through mesh refinement boundaries and in the arbitrarily shaped cells that appear against geometric boundaries.  The stencil for the gradient reconstruction is central difference, and the flux is upwind. Implementationally, the gradient is computed using a least-squares reconstruction, and the least squares problem is solved via the *normal equations.* At cut-cells, and cells which neighbor refinement boundaries, the reconstruction is always performed using true cell and face centroids. This reconstruction is linearity preserving, even where the mesh is extremely non-uniform. This results in substantially improved accuracy at wall-boundaries and through refinement interfaces. Results with this solver are comparable with body-fitted structured and unstructured solvers for comparable numbers of cells. Most users with experience in other packages notice that flowCart has substantially improved wave propagation, and very low dissipation. **Temporal Discretization:** Steady-State runs: Advance to steady state is performed by an unstructured, [nested multigrid procedure](#page-flowcart-html--multigrid). A user selectable Runge-Kutta scheme provides inner smoothing to drive the multigrid for rapid convergence. Virtually any explicit Runge-Kutta scheme can be used by putting in your favorite coefficients ([here](#page-input-cntl-html)). Since it uses perfectly nested meshes, Multigrid performance is competitive with better schemes found in the literature and is *really easy to use*. Of course, if you just want to drive your solutions with plain old Runge-Kutta, you can do that too.    Time-Dependent runs: When running time-dependent simulations, the steady-state multigrid scheme becomes the inner-driver for an outer Backward Euler (BE1) or 2nd-Order Backward (BDF2) time-accurate method using a dual-time formulation. Within each time-step the multigrid converges modified residuals using pseudo-time iterations. BDF2 is unconditionally stable so you are free to choose as large a timestep as your physics will let you get away with. Unsteady restarts are second-order in time.    *[(top)](#page-flowcart-html--top)*  **How is flowCart integrated into the Cart3D package?** **flowCart** is tightly integrated into Cart3D. Pre/Post processing operations have been substantially reduced. Since it was written specifically as a module for Cart3D, file translation steps have been completely eliminated. The **cubes** mesh generator puts out Mesh.c3d files, and these can be used directly by flowCart, without translation. flowCart can be asked to extract Cp's and other flow quantities both on the body's surface and on cutting planes through the domain directly (***-T*** and ***-clic*** options). It also provides both residual and lift/drag information for convergence monitoring (the ***-his*** option). After a run, you can extract a variety of information using **clic.** Loads and moment information are all conservatively transferred back to the input surface triangulation, so that you can postprocess on the surface without having to load the entire discrete solution.  *[(top)](#page-flowcart-html--top)*    **How can I extract force and moment data? What's CLiC?**  - **[CLiC](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/clic/clic.html)** is a   Component based force and moment module developed   as a post-processor and data-extractor for Cart3D.   Its an extremely flexible and powerful package.   Using **clic**, you can extract Cp cuts on any   component or group of components in your   configuration, you can compute LDM (lift, drag,   moment) for components, component groups or   configurations, and extract the usual bevy of   point-moments, line-moments ("hinge moments")   etc.. If you want to see some of what it was   designed to do, take a look at the original ISO   software project plan ([here](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/clic-spp.pdf),   64kb acrobat format). Clic can be run as a   postprocessor, or called directly through an API.   The [clic home page](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/clic/clic.html)   will get you started with the package. - You can   also [automatically   extract component-based loads information](#page-postprocess-html--fomo)   from within flowCart. Additionally, you can also   instrument your domain with [point and line   sensors](#page-postprocess-html--sensors) and collect data along these sensors   wither on-the-fly or automatically at run's end.   In v1.5 we introduced the capability to take   on-the-fly temporal (or iterative) statistics.   This includes computing average loads information   (loadsCC.avg.dat along side loadsCC.dat)   over your monitoring window. You can also extract   the envelope   and averages of local surface quantities,   so you can pinpoint which parts of a surface   geometry are most affected by local unsteadiness.   *[(top)](#page-flowcart-html--top)* **Parallelization and Domain Decomposition** **flowCart** uses a domain decomposition approach to parallelization. As a result, scalability on large numbers of processors is quite good. Scalability is linear down to about 10k cells per core (2013-era hardware) even with multigrid.  In domain decomposition approaches, the computation proceeds on subdomains that are farmed off to the machine's processors. After a certain amount of work is done, each subdomain passes information to its neighboring subdomain in an explicitly coded communication step. **OpenMP**  **flowCart** was developed using the [OpenMP](http://www.openmp.org) standard. However, it is setup like a distributed memory code, in that it uses domain decomposition, and memory locality is carefully controlled.  Thus, although it uses explicit message passing and achieves excellent scalability, it is a shared memory code. For machines with physically distributed memory, use the distributed variant **mpix\_flowC****art**. MPI   mpix\_flowCart was completely re-worked in Cart3D v1.5. in v1.6 it remains 100% feature-complete and performs all the same tricks as the shared memory version. The code base has been reworked for maintainability and the MPI and OpenMP builds use as much of the same code as possible. Scalability of the MPI code is as good or better than the shared memory version. Here is a comparison. We include distributions using a variety of MPI backends, including OpenMPI and MPICH. For a full list of backends, see the release notes that come with ***Cart3D***, or look in ***$CART3D/bin/$CART3D\_ARCH/***  **Domain Decomposition** Our domain decomposition approach was designed to be transparent to the user. It relies on a space-filling-curve based reordering of the **cubes** mesh which is then partitioned on the fly when you startup the solver. All you need to know is that if you want to run in parallel you just have to (1) reorder the mesh with the **[reorder](#page-flowcart-reorder-html)** utility, and (2) set your **OMP\_NUM\_THREADS** environment variable to the desired number of processors, **flowCart** does everything else.  Here's how you'd do that for a 16 CPU case using OpenMP, (in csh):  |  |  | | --- | --- | | **1.% reorder** | *# reorder the mesh (using default file names)* | | **2.% setenv OMP\_NUM\_THREADS 16** | *# choose number of processors* | | **3.% flowCart** | *# start the run* |    With MPI, the procedure is basically the same, but you use  mpiexec  to start the solver:  |  |  | | --- | --- | | **1.% reorder** | *# reorder the mesh (using default file names)* | | **2.% mpiexec -np 16 mpix\_flowCart** | *# use mpirun/mpiexec to set num. of processors and run* | *[(top)](#page-flowcart-html--top)* **Multigrid** **flowCart** uses Full Approximation Storage (FAS) multigrid for convergence acceleration. You invoke it via the ***-mg %d*** command line option (e.g. for 4 levels of multigrid you'd type "*% flowCart -mg 4"*). By default its set up to do use a full multigrid startup procedure. This means that computation starts on the coarsest mesh, once that converges a bit, the next finer mesh gets into the act and it does two level multigrid. After a while the evolving solution gets transferred to the next finest mesh, and it does three level multigrid. This continues until we've reached the finest mesh, and then the entire mesh hierarchy is used to converge the solution on the finest mesh. When you run in parallel, each processor gets its own hierarchy of fine to coarse meshes to work with, here is what it would look like for 2 processors:   Multigrid obviously needs a sequence of coarse meshes to get going. These meshes are derived from the fine grid that you make with **cubes**. We use a special coarsening procedure which attempts to coarsen with an 8:1 coarsening ratio. Anytime a parent cell finds all its children at the same level of refinement, that parent cell gets inserted into the coarse mesh. Of course there are situations where one or more children has further subdivision, and coarsening gets "suspended" in that cell. As a result, coarsening ratios with this algorithm *approach* (but do not exceed) 8:1. In practice, finer meshes achieve coarsening ratios above 7:1, which is sufficient to ensure good smoothing. W-cycle multigrid requires at least 4:1 to maintain its (theoretical) constant time bound. Meshes at any level in the hierarchy fully cover the computational domain, and may contain cells at many levels of refinement.  The **[mgPrep](#page-flowcart-mgprep-html)** utility takes a *reordered* fine mesh (output from **[reorder](#page-flowcart-reorder-html)**) and creates a sequence of coarse meshes directly from this mesh. Since the fine mesh has already been *reordered,* the coarse meshes will automatically be ready for domain-decomposition, should you wish to run the final multigrid run in parallel. When a multigrid mesh is run in parallel, all meshes in the hierarchy get automatically partitioned using the exact same space-filling-curves, so coarse meshes in a given partition overlaps maximally with the finer grids that it supports.  Building coarse meshes with **mgPrep** is extremely easy. **mgPrep** takes in a reordered mesh (usually called *Mesh.R.c3d*) and outputs the input mesh, and the series of coarse meshes generated. The hierarchy of meshes is stored in a single file, usually called *Mesh.mg.c3d.* Here's how (starting from *Mesh.c3d* output from **cubes)**:   |  |  | | --- | --- | | **1.% reorder** | *# reorder the mesh (using default file names)* | | **2.% mgPrep -n 6** | *# generate coarser meshes (orig + 5 coarser)* | | **3.% setenv OMP\_NUM\_THREADS 16** | *# choose number of processors* | | **4 % flowCart -mg 6** | *# run on 16 CPU's with 6 level multigrid* |   ***Note:*** This example has some options that you could play with, for example, step 2 creates 5 levels of coarser meshes (with the original mesh this makes 6 meshes total). By default it creates *Mesh.mg.c3d.* **flowCart**'s [input control file](#page-input-cntl-html) now needs to point to this file as the input mesh. In step 3, the example uses all 6 meshes in the hierarchy, but you don't need to, you could use anywhere from 1 to 6 levels of mesh, so running on 3 meshes (for example) with "***% flowCart -mg 3"*** would be an equally valid command line. Even running single mesh "***% flowCart"*** will work fine (in this case, ***-mg 1***, is the default). *[(top)](#page-flowcart-html--top)*    ---   Questions?   Visit [Cart3D Discuss](https://groups.google.com/forum/#%21forum/cart3d),   or  [Contact Us](#page-cart3d-team-html)  last update December, 2024.   --- |

---

<a id="page-flowcart-files-html"></a>

## flowCart Files

*Original page: [flowCart_Files.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/flowCart_Files.html)*

Examples of Input and Output Files  ---  [Cart3D surface triangulation formats **\*.tri**,**\*.a.tri**,**\*.i.tri**](#page-cart3dtriangulations-html)   [**cubes** input file < input.c3d>](#page-input-c3d-html)   [**cubes** preSpec files for controlling adaptation < preSpec.c3d.cntl>](#page-prespec-c3d-cntl-html)         [> tell me more about Prespecified Adaptation Regions...](#page-prespecnote-html)   [**flowCart** input control files < input.cntl> (clickable)](#page-input-cntl-html)   **flowCart** [history.dat](#page-history-dat-html), [forces.dat](#page-forces-dat-html), or [moments.dat](#page-moments-dat-html) files (*-his* flag)    [Sample **Config.xml** GMP configuration file](#page-config-xml-html)     ---  [(back)](#page-flowcart-io-html) to **flowCart** I/O main page. |

---

<a id="page-cart3dtriangulations-html"></a>

## cart3dTriangulations.html

*Original page: [cart3dTriangulations.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/cart3dTriangulations.html)*

### **Surface Triangulation File Formats**

---

There are several Cart3D
file formats, which differ depending on the type of information that
they
include. The format is both extensible and hierarchical. Subsequent
formats
are built upon one another, so that additional information is always
"appended"
to the end of the more basic format. For backward compatibility All
ASCII
files are assumed to be written with FORTRAN free format
"WRITE(iUnit,\*)"
and all unformatted files are written with FORTRAN unformatted
"WRITE(\*)".

The hierarchy goes as
follows:

 

|  |  |  |  |
| --- | --- | --- | --- |
|  | **Level of Information** | **Contents** | **Filename Extension** |
| 1. | (least info) | [single component](#page-cart3dtriangulations-html--1-component-file-forma) | **\*.tri** or **\*.a.tri** (ascii) |
| 2. |  | [configuration](#page-cart3dtriangulations-html--2-configuration-file-format)    (multiple components) | **\*.tri** or **\*.a.tri** (ascii) |
| 3. |  | [wetted surface](#page-cart3dtriangulations-html--3-wetted-surface-triangulation-format)    (multiple components) | **\*.i.tri** |
| 4. | (most info) | [annotated triangulation](#page-cart3dtriangulations-html--4-annotated-triangulation)   (wetted surface + nodal quantities) | **\*.triq** |

**The
Good
News:** While the information contained in these files is
different its all based on the same file format. This gives backward
compatibility
down the hierarchy. For example a "**\*.triq**" file could be read as
a single component file, you just would stop reading before you hit the
flow variables.

**What's in
the files?**

> **1.
> Component file format :**: The file format for a component may be
> either
> (fortran) unformatted or ascii, and it is currently a "standard" finite
> element type data format. Triangles should be oriented such that the
> circuit
> v1,v2,v3 is COUNTERCLOCKWISE as viewed from an observer on the FLOW
> side
> of the surface.
>
>
> ---

```
       nVerts  nTri            <- integer  integer
       x_1  y_1  z_1           <- Geometry of Vertex 1 (float  float  float)
       x_2  y_2  z_2           <- Geometry of Vertex 2 (float  float  float)
       x_3  y_3  z_3           <- ...
       .
       .
       .
       x_nVerts  y_nVerts  z_nVerts <- geom of last unique vertex (nVerts)
       v1_t1  v2_t1  v3_t1          <- Vertices of triangle 1 (int  int  int)
       v1_t2  v2_t2  v3_t2          <- Vertices of triangle 2 (int  int  int)
       v1_t3  v2_t3  v3_t3          <- Vertices of triangle 3 (int  int  int)
       .
       .
       .
       v1_nTri  v2_nTri  v3_nTri    <-Vertices of last triangle (int  int  int)
```

> ---

> Example of F77
> code
> for reading a **\*.tri** file

```
           read(iUnit,*) nVerts, nTri
           read(iUnit,*) ( x(j), y(j), z(j)  ,j=1,nVerts) 
           read(iUnit,*) ( v1(j),v2(j),v3(j) ,j=1,nTri  )
```

> ---
>
>
> [(top)](#page-cart3dtriangulations-html--top)
>
> **2.
> Configuration file format** :: The file format for a configuration is
> an extension of that used for individual components. It may be either
> (fortran)
> unformatted or ascii. These files are exactly the same as component
> triangulations,
> but with component information for each triangle appended onto the
> triangle
> list. The triangles for all the components in a configuration are
> sequentially
> numbered into global triangle and vertex lists. Triangles should be
> oriented
> such that the circuit v1,v2,v3 is COUNTERCLOCKWISE as viewed from an
> observer
> on the FLOW side of the surface.
>
> ---
>
> ```
>        nVerts  nTri            <- total # of unique verts/tris  (int  int)
>        x_1  y_1  z_1           <- Geometry of Vertex 1 (float  float  float)
>        x_2  y_2  z_2           <- Geometry of Vertex 2 (float  float  float)
>        x_3  y_3  z_3           <- ...
>        .
>        .
>        .
>        x_nVerts  y_nVerts  z_nVerts <- geom of last unique vertex (nVerts)
>        v1_t1  v2_t1  v3_t1          <- Vertices of triangle 1 (int  int  int)
>        v1_t2  v2_t2  v3_t2          <- Vertices of triangle 2 (int  int  int)
>        v1_t3  v2_t3  v3_t3          <- Vertices of triangle 3 (int  int  int)
>        .
>        .
>        .
>        v1_nTri  v2_nTri  v3_nTri    <- Vertices of last triangle (int int int)
>        1 1 1 1 1 . . . 1 1 1 2 2 2 . . . 2 2 2 <-component numbers of all 
>                                                  triangles in the configuraiton
>                                                  sequentially starting at "1"
> ```
>
> ---
>
>  Example
> of F77 code for reading a **\*.tri** file
>
> ```
>            read(iUnit,*) nVerts, nTri
>            read(iUnit,*) ( x(j), y(j), z(j)  ,j=1,nVerts) 
>            read(iUnit,*) ( v1(j),v2(j),v3(j) ,j=1,nTri  ) 
>            read(iUnit,*) ( comp(j)           ,j=1,nTri  )
> ```
>
> ---
>
> [(top)](#page-cart3dtriangulations-html--top)
>
> **3.
> Wetted Surface Triangulation Format ::** This format is IDENTICAL to
> that for a CONFIGURATION file. So, why call it a different file? Well,
> a Configuration file, has multiple components, each of which has a
> unique
> set of vertices and triangles. So, in a "configuration file", a
> triangle
> on one component will never share a common vertex with a triangle from
> a different component. In other words, in a "configuration file" the
> index
> spaces of the different components are completely disjoint. In a
> "Wetted
> surface file" the index spaces of different components are free to
> overlap.
> So a triangle on one component may use one or more of the vertices from
> another component. Think of the intersection between two components,
> vertices
> along the line of intersection, will be used in the triangle lists of
> BOTH
> of the components. For this reason these triangulations are often
> called
> "Intersected Triangulations".
>
> ---
>
> Example
> of F77 code for reading a \*.i.tri file
>
> ```
>            read(iUnit,*) nVerts, nTri
>            read(iUnit,*) ( x(j), y(j), z(j)  ,j=1,nVerts) 
>            read(iUnit,*) ( v1(j),v2(j),v3(j) ,j=1,nTri  ) 
>            read(iUnit,*) ( comp(j)           ,j=1,nTri  )
> ```
>
> ---
>
> [(top)](#page-cart3dtriangulations-html--top)
>
> **4.
> Annotated Triangulation::** This is a triangulation that has been
> annotated
> by attaching a list of scalar quantities to each vertex. The filename
> "\*.triq"
> indicates that the "tri" file has been appended with the solution
> vector
> or "q" information. By convention the first scalar is always the
> pressure
> coefficient.
>
> ---
>
> ```
>      nVerts nTri nScal   --> # of vertices, # of triangles, # of scalars
>      x1 y1 z1            --> coordinates of the vertices
>      x2 y2 z2 
>      x3 y3 z3 
>      ..... 
>      ..... 
>      v11 v12 v13         --> triangles connectivity (3 integers)
>      v21 v22 v23 
>      ..... 
>      ..... 
>      ..... 
>      1 1 1 1 1 2 2 2 3 3 3 3 1 2... --> component number for each tri
>      ..... 
>      0.0000 0.0000 0.0000 .....-->scalars values (1,..,nScal) for each vertex 
>      ..... 
>      .....
> ```
>
> ---
>
> Example
> of F77 code for reading a \*.triq file
>
> ```
>            read(iUnit,*) nVerts, nTri, nScal 
>            read(iUnit,*) ( x(j), y(j), z(j)  ,j=1,nVerts) 
>            read(iUnit,*) ( v1(j),v2(j),v3(j) ,j=1,nTri  ) 
>            read(iUnit,*) ( comp(j)           ,j=1,nTri  ) 
>            read(iUnit,*) ((scalar(j,k),k=1,nScal),j=1,nVerts)
> ```
>
> [(top)](#page-cart3dtriangulations-html--top)
>
> ---
>
>
> *last update 7 Jun 00, M. Aftosmis*

---

<a id="page-cart3d-howtos-html"></a>

## How Tos

*Original page: [cart3d_howtos.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/cart3d_howtos.html)*

How To's  ---      *[Cart3D Command Summary and Quick Reference Guide](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/COMMAND_SUMMARY.pdf)*      *[How to setup power boundary conditions and inlets:](#page-howto-samples-power-readme-html)* Documentation and examples using new surface boundary conditions introduced in v1.5.5 for modeling inlets and propulsion systems.      *[How to Label Triangulations:](#page-howto-facelabels-index-html)* All about component IDs, face labels, GMP tags, and SurfBCs       [*Internal flow:*](#page-howto-internalflow-index-html) Approaches for meshing and running internal flow problems with Cart3D      [*New tricks with trix:*](#page-trix-index-html) New features in *trix* and examples of how to use them. Also see *trix* in the  [COMMAND\_SUMMARY.pdf](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/COMMAND_SUMMARY.pdf)       [*How to set up rocket nozzles:*](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/howto/howTo_RocketPlumes_08.11.pdf) Considerations for specifying power BCs for rocket plumes      [*How to estimate viscous drag:*](#page-howto-viscousdrag-index-html) Estimating friction drag with the viscousDrag tool      ---  Please post requests for new HowTo's to the [Cart3D Discussion Group](https://groups.google.com/forum/#%21forum/cart3d) before [emailing us](#page-cart3d-team-html) directly. |

---

<a id="page-bool-intersection-html"></a>

## Boolean Intersection Predicates

*Original page: [bool_intersection.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/bool_intersection.html)*

---

#### Intersection of Generally Positioned Polygons in *R*3

go back to [Cart3d](#page-cart3dhome-html),
goto The Cartesian
Origin mail to: *[michael.aftosmis@nasa.gov](mailto:michael.aftosmis@nasa.gov)*

---

One approach to understanding the utility
of boolean intersection predicates is by considering the specific example
of a tri-tri intersection in 3 space. Figure 1 shows a view of two intersecting
triangles as a model for discussion. Each intersecting tri-tri pair will
contribute one segment to the final polyhedra which comprise the wetted
surface of the configuration. The assumption of *general* as opposed
to *arbitrary* positioned data indicates that the intersection is
always non-degenerate. Triangles don't share vertices and edges of tri-tri
pairs do not intersect exactly. Thus, all intersections will be *proper*
(as opposed to *improper* or "degenerate"). This restriction will
be lifted in the [section
on tie-breaking algorithms](#page-degeneracy-html--virtualpurturbationsos).

**Figure 1:** An intersecting pair of
generally positioned triangles in three dimensions.

 Several approaches exist to compute
such intersections but a particularly attractive technique offers itself
as a Boolean test. This predicate has the advantage that it can be performed
robustly and quickly using only multiplication and addition, thus avoiding
the inaccuracy and robustness pitfalls associated with division using fixed
width representations of floating point numbers. It is useful to present
a rather comprehensive treatment of this intersection primitive not only
for illustrating the development of the basic geometric computations, but
also because the important topics of robustness, floating-point round-off-error
and tie-breaking will return to the expressions and assumptions exposed.

 For two triangles to properly intersect
in three dimensional space, the following conditions must exist:

 1. Two edges of one triangle must
span the plane of the other.

 2. If condition (1) exists, there
must be a total of two edges (of the six available) which pierce within
the boundaries of the triangles (*e.g.* edge *ab* pierces (0,1,2)
and edge *02* pierces (*a,b,c*) ).

---

#### *Segment-Triangle Intersection*

Observations (1) and (2) reveal that the tri-tri
intersection may be viewed a special arrangement of the more general problem
of a segment-triangle intersection. This fundamental problem is common
throughout the study of pologonal geometry. A variety of approaches to
this basic problem exist. Generally the first approach that comes to mind
is to directly compute the pierce points of the edges of one triangle in
the plane of the other. Pierce locations from one triangle's edges may
then be tested for containment within the boundary of the other triangle.
Unfortunately, this approach, while conceptually simple, is error prone
and problematic when implemented using finite precision mathematics. In
addition to demanding special effort to trap out zeros, the floating point
division required by this approach may result in numbers not representable
by finite width words, thus resulting in a loss of control over the accuracy
of results and leading to serious problems with robustness.

An alternative to this slope-pierce test
is to consider a Boolean check based on computation of a triple product
without division. A series of such logical checks have the attractive property
that they permit one to establish the
*existence* and *connectivity*
of the segments without relying on the problematic computation of the actual
pierce point locations. The final step of computing the locations of these
points may then be relegated to post-processing where they may be grouped
together and, since the connectivity is already established, floating point
errors will not have fatal consequences.

The Boolean primitive for the 3D intersection
of an edge and a triangle is based on the concept of the signed volume
of a tetrahedra. This signed volume is based on the well established relationship
for the computation of the volume of a simplex, *T*, in *n* dimensions
in determinate form (see for example the excellent discussion in Ref.[1]).
The signed volume *Vol*(*T*) of the simplex *T* with vertices
{v0,v1,v2,...,vn) in *n* dimensions is:

 

 

{1}

where *vk j* denotes the *j*th
coordinate of the
*k*th vertex with
and In 3 dimensions,
eq. {1} gives six times the signed volume of the tetrahedron
*Tabcd*.

 

 

{2}

This volume serves as the fundamental building
block of the geometry engine used in *intersect* and *cubes*.
It is positive when (*a,b,c*) forms a counterclockwise loop when viewed
from an observation point on the side of the plane defined by (*a,b,c*)
which is opposite from point *d*. Positive and negative volumes define
the two states of a Boolean test, while zero indicates that the four vertices
are exactly coplanar. If the vertices are indeed coplanar, then the situation
constitutes a "tie" which will be resolved with a general tie-breaking
algorithm. In applying this logical test to edge
*ab* and triangle
(*0,1,2*) in fig. 1, *ab* spans the plane if and only if (iff)
the signed volumes
*T012a* and *T012b*
have opposite signs. Figure 2 presents a graphical look at the application
of this test.

 

 

**Figure 2**: Boolean test to check if
edge *ab* spans the plane defined by triangle (*0,1,2*) through
computation of two signed volumes.

With *a* and *b* established
as spanning the plane (*0,1,2*) all that remains is to determine if
*ab*
pierces within the boundary of the triangle (*0,1,2*). This will be
the case iff the three tetrahedra formed by connecting the endpoints of
*ab* with the three vertices of the triangle (*0,1,2*) (taken
two at a time) all have the same sign, that is:

 

 

{3}

Figure 3 illustrates this test for the case
where the three volumes are all positive.

 

 

**Figure 3.** Boolean test for pierce of
a line segment *ab* within the boundary of a triangle (*0,1,2*).

After determining the existence of all
the segments which result from intersections between tri-tri pairs and
connecting a linked list of all such segments to the triangles that intersect
to produce them, all that remains is to actually compute the locations
of the pierce points. This is accomplished by using a parametric representation
of each intersected triangle and the edge which pierces it. The technique
is a straightforward three dimensional generalization of the 2D method
presented in [1].

 The signed volume computation of
eq.{2} is also used for performing inCircle/inSphere tests, for in/out
determination in ray-casting algorithms and for a variety of other topological
predicates. Due to its obvious importance, a great deal of research has
gone into developing rapid and [robust evaluations
of this determinant](#page-degeneracy-html). References [2] and [3] treat this topic in detail.

#### References

[1] [O'Rourke,
J.](http://cs.smith.edu/~orourke/), *Computational Geometry in C*. Cambridge Univ. Press, 1994.

 [2] [Shewchuk,
J.R.](http://www.cs.cmu.edu/%7Equake/triangle.html), *Robust Adaptive Floating-Point Geometric Predicates*, Proceedings
of the Twelfth Annual Symposium on Computational Geometry, pages 141-150,
ACM, May 1996.

 [3] Edelsbrunner, H., and Mücke,
E.P., "Simulation of Simplicity: A Technique to Cope with Degenerate
Cases in Geometric Algorithms." *ACM Transactions of Graphics*, **9**(1):66-104,1990.

---

|  |
| --- |
| For more information mail to: *[michael.aftosmis@nasa.gov](mailto:michael.aftosmis@nasa.gov)* |
| Return to ***[Cart3d](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/cart3d.html)***or [*Michael Aftosmis's* homepage](https://www.nas.nasa.gov/about/staff/maftosmis.html) |

---

*last update 25 Sep 96*

---

<a id="page-degeneracy-html"></a>

## Degeneracy, Tie-Breaking and Floating-Point Math

*Original page: [degeneracy.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/degeneracy.html)*

---

#### Tie-Breaking, Degeneracies, and Floating Point Arithmetic

go back to [Cart3d](#page-cart3dhome-html),
goto The Cartesian
Origin mail to: *[michael.aftosmis@nasa.gov](mailto:michael.aftosmis@nasa.gov)*

---

The discussion on [Boolean
predicates](#page-bool-intersection-html) argues that polygon intersection, re-triangulation and ray-casting
algorithms in *Rn* may all be cast in a unified framework
which requires only the evaluation of the determinant:
{1}
where *vk j* denotes the *j*th
coordinate of the
*k*th vertex with
and In 3 dimensions,
this equation yields six times the signed volume of the tetrahedron *Tabcd*.

 

{2}

As a result, the robustness of the overall
procedure equates to a robust implementation of this signed volume calculation.
Fortunately, accurate and rapid evaluation of this determinant has long
been the subject of study in computational geometry and computer science.

 Computing the sign of eq.{2} constitutes
a *topological primitive*, which is an operation that tests an input
and always yields one of a prespecified number of results. Of course, such
primitives can only classify, and new objects - like the the actual locations
of intersection points (see fig.1 [boolean
intersections](#page-bool-intersection-html)) - cannot be determined without further work. Such topological
primitives do, however, provide the intersections implicitly and this information
is all that is really needed to establish the connectivity of the segment
list describing the intersection.

#### *Exact Arithmetic*

The signed volume computation for arbitrarily
positioned geometry can return a result which is positive (return +1),
negative (return -1) or zero (return 0), where +/-1 are non-degenerate
cases and zero represents some geometric degeneracy. Distinguishing between
these cases on finite precision hardware, however, is not necessarily a
trivial task. Two approaches are common, the first may be thought of as
an *integer inflation* strategy (cf. [1]). In this approach all vertex
locations are preprocessed so that the data is inflated and adjusted to
span the maximum allowable integer space which is representable by the
hardware (±(231 - 1) with 32 bits or ±(263
- 1) with 64 bits). New data is then constrained to its nearest allowable
integer location. The second approach is to use exact (arbitrary precision)
arithmetic. Unfortunately, while much hardware development has gone into
rapid (round-off prone) floating point computation, few hardware architectures
are optimized for either the arbitrary precision or integer math alternatives
.

In an effort to perform as much of the
computation as possible on the floating-point hardware, the ***Cart3d***
codes first compute eq.{2} in floating point, and then make an *a-posteriori*
estimate of the round-off error (cf. [2],[3]) If the round-off error is
of the same order as the signed volume, then the case is considered indeterminate
and we look to exact arithmetic and introduce the tie-breaking algorithm
for truly degenerate cases. This approach may be characterized as a floating
point filter, since only cases that are "dangerous" are selected for further
processing. A brief discussion of these topics is available on the web
at the [Robust Predicates
Page](http://www.cs.cmu.edu/%7Equake/robust.html) along with substantial downloadable documentation. Since only
a very small fraction of the computations fall through the filter, the
speed penalty for using exact arithmetic on virtually all realistic examples
is essentially negligible. To further speed things up, the routines presented
in [2] have been optimized for **float** and
**double** data types
on IEEE compliant hardware. 

#### *"Simulating" Simplicity and Tie-Breaking*

The tie-breaking algorithm stems from work
done by Edelsbrunner and
M.cke[4]
and is known as Simulation of Simplicity (appropriately abbreviated as
"SoS"). The technique resolves geometric degeneracies by assuming a consistent
set of virtual perturbations sufficient to insure that geometry is *always*
in general position - in accordance with the assumptions outlined in the
discussion on [boolean intersections](#page-bool-intersection-html)).
The attraction of this approach is that the topological primitives never
return a zero (degenerate) result, thus alleviating the need to consider
specialized treatments for degenerate geometry. Moreover, since the perturbations
are "virtual" and "consistent", data is never altered and ties are always
resolved with the same result. Symbolic perturbation schemes like SoS are
attractive for many reasons. Primarily, however, whats important is that
they represent an *algorithmic* approach to tie-breaking. They do
not depend on the experience of the programmer to foresee all possible
diabolical cases, and virtually eliminate the need for special-case coding
to trap out degeneracies arising from objects or data in special position.

 *-- more details coming soon --*

#### *References*

[1] Knuth, D.E., The Art of Computer Programming:
Semi-numerical Algorithms Addison Wesley, 1973.

[2] [Shewchuk,
J.R.](http://www.cs.cmu.edu/%7Equake/triangle.html), *Robust Adaptive Floating-Point Geometric Predicates*, Proceedings
of the Twelfth Annual Symposium on Computational Geometry, pages 141-150,
ACM, May 1996.
Abstract

 [3] Priest, D.M., "Algorithms for
Arbitrary Precision Floating Point Arithmetic", *Tenth Symposium on Computer
Arithmetic*, pp. 132-143, IEEE Comp. Soc. Press, 1991.

 [4] Edelsbrunner, H., and M.cke,
E.P., "Simulation of Simplicity: A Technique to Cope with Degenerate
Cases in Geometric Algorithms." *ACM Transactions of Graphics*, **9**(1):66-104,1990.

---

---

|  |
| --- |
| For more information mail to: *[michael.aftosmis@nasa.gov](mailto:michael.aftosmis@nasa.gov)* |
| Return to ***[Cart3d](#page-cart3dhome-html)***or [*Michael Aftosmis's* homepage](https://www.nas.nasa.gov/about/staff/maftosmis.html) |

---

*last update 3 Sep 96*

---

<a id="page-degen-ex-html"></a>

## degenerate geometry examples

*Original page: [degen_ex.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/degen_ex.html)*

---

#### Examples of Degenerate Geometry

### (and what intersect does with them...)

**Degeneracies** arise in geometric data
due to the *special* position of two or more objects. Well-known examples
in three-dimensions include co-planar, co-linear, or co-located points,
and its even possible for two line segments to intersect exactly in three
dimensions. More diabolical cases include those where two line segments
in special position overlap without sharing an end point, or even having
the decency to be co-linear. While seemingly unlikely (mathematically)
degenerate data is common in the real world, and especially the somewhat
quantized world of computational geometry.

 In *Application Challenges to Computational
Geometry* the CG Task force writes:

*The effect of degeneracy is to vastly
increase the number of special cases. While a sorting algorithm must deal
only with the possibility of two keys being equal, a typical geometric
algorithm faces the possibility of dozens or hundreds of different special
cases[..]. The presence of numerical data, added to the inherent complexity
of geometric data types, makes geometric algorithms much harder to* [robustly]*implement correctly than combinatorial (say, graph-theoretical) ones. They
are also much harder than just purely numerical algorithms (such as those
addressed by numerical analysis) many of which consist of large chunks
of straight-line code. **Since the overall utility of an implementation
may depend upon the correct treatment of special cases, the handling of
special cases can permeate the implementation.***

Here are a few examples of intersecting
polyhedra and what *intersect* does with them while computing the
triangulation of the exposed surface.| Single point co-planar with a face | (resulting polyhedra) |
| Co-planar faces with no common points | (resulting polyhedra) |
| Four coplanar faces with no common points | (resulting polyhedra) |

---

|  |
| --- |
| For more information mail to: *[michael.aftosmis@nasa.gov](mailto:michael.aftosmis@nasa.gov)* |
| Return to ***[Cart3d](#page-cart3dhome-html)***or [*Michael Aftosmis's* homepage](https://www.nas.nasa.gov/about/staff/maftosmis.html) |

---

*last update 1 Apr 96*

---

<a id="page-prespecnote-html"></a>

## About preSpec files

*Original page: [preSpecNote.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/preSpecNote.html)*

---

Prespecified Adaptation Regions

---

**Prespecified
adaptation regions** give you some control over the
geometry-based adaptation that **cubes** does by default. Its
intended to give the user some useful control over the mesh
generation process, and to allow expert users to impart some
knowledge into the mesh generation process.  As explained in
the [sample preSpec.c3d.cntl](#page-prespec-c3d-cntl-html)
file, the *-pre preSpecFilename*command line flag directs **cubes** to parse
the preSpec file. The ["BBox:"](#page-prespec-c3d-cntl-html) tag is used to identify the
region.

Extra refinement passes for specific
components. preSpec files also allow you to specify any particular
components for which you want cubes to do an additional
refinement pass or two. This feature is mainly used to improve
resolution of SurfBC components being used in simulations with
power.

The BBox tag looks like this:

**#
BBox:**(int)  
(float)   (float)   (float) (float)
   (float) (float)
**BBox:** level   Xmin   Xmax   
Ymin  Ymax    Zmin 
Zmax

where *level*is the number of
subdivisions you want in the region defined by the minmax
box.

*#
XLev:'s are optional and mostly used to pre-refine SurfBCs for
simulations with power
#*   
(int) (int)  ...   
(int)
**XLev:** 2   2   3 
8  13   *# ... do 2 extra
refinement passes on comps 2, 3, 8 &
13***XLev:** 1   1   4 
       *# ... do 1 extra
refinement pass on comps 1 & 4*

See a sample
preSpec file [<here>](#page-prespec-c3d-cntl-html).

Cubes works by successively refining an
original background mesh some number of user specified times
(entry #6 in the [cubes input file,](#page-input-c3d-html)
or with the the *-maxR %d*  command line option). When a cell falls
into a region specified by a **BBox** preSpec tag, that cell
is automatically refined until it undergoes at least
*level* subdivisions.  If
the maximum number of refinement passes is less than
*level*, you will only get maxR
refinements in these regions. For example, if you set
*level* to 12, but only direct
cubes to do 9 refinement passes, then you will only get 9
refinement passes in the refinement regions.
You can specify any
number of refinement regions.

|  |  |
| --- | --- |
| ---   Questions?   Visit [Cart3D Discuss](https://groups.google.com/forum/#%21forum/cart3d),   or  [Contact Us](#page-cart3d-team-html)  last update Aug. 2015, M. Aftosmis   --- |  |

---

<a id="page-input-c3d-html"></a>

## input_c3d.html

*Original page: [input_c3d.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/input_c3d.html)*

---

input.c3d

---

**"  \_ \_
\_ \_ \_ \_ \_ \_
\_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_**

**/                                                                    
\**

**|                                                                    
|**

**|        
Cartesian Mesh Generation Input
Specifications             
|**

**|  see ["notes"](#page-input-c3d-html--notes)
at the bottom of the page for fileformat information   |**

**\\_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_
\_ \_ \_ \_
\_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_ \_/"**

 

**1. Surface Geometry File Name (Cart3d surface
triangulation file format):**

**oneraFine.i.tri**

*//                                      
...this
is just a comment, 'oneraFine.i.tri' follows the file naming*

*//                                         
conventions
set up for [cart3D
surface
triangulations](#page-cart3dtriangulations-html)*

*//                                         
for
notes about the format of entries in*

*//                                         
this
file, see the ["notes"](#page-input-c3d-html--notes) section at the
bottom...*

**2. Outer Cartesian Box Specs:**

**Xmin    
Xmax           
Ymin   
Ymax           
Zmin     Zmax**

 **-40.0   
40.07          
-40.0   
40.0001       
.00001   40.03**

*//    
...
note that we're careful not to use any "perfect numbers" to avoid
accidentally
//        
building
degeneracies into the mesh*

**3. Starting Mesh Dimensions (# of nodes in
each dimension, inclusive):**

**# verts in
X   
### verts in Y     # verts in Z**

 **8
             
8
            
4**

*//                      
....try
to keep these "background grid" dimensions*

*//                          
small,
so that you get good multigrid coarsening ratios...*

**4. Maximum Hex Cell Aspect Ratio (
Isotropic
= 1):**

**1**

*//                     
....flowCart
has limited support for anisotropic cell division.*

**5. Minimum Number of cell refinements on
body
surface (auto = -1):**

**-1**

**6. Maximum Number of cell refinements:**

     **7**

*//                    
...this
is the number of refinement sweeps that "cubes" will perform*

**7. Num of bits of resolution assigned to
integer
coordinates (maximum = 21):**

    **21**

*//                    
...dont
mess with this unless you really know what you're doing*

------------------------------------------------------------------------

***\*\* NOTES:***

*o   add comments
after the
":" terminating each entry but before next line.*

*Additional
(blank)
lines may be added with out messing up the parsing.*

*o   User entries
in this
example are shown in Black, comments in pink and*

*fixed
text in green. (Actually the parser just checks the input number
and the colen ":".)*

*#1: A [wetted surface triangulation](#page-cart3dtriangulations-html--3-wetted-surface-triangulation-format)  with no internal geometry:
(usually
output from "intersect")*

*#4: max difference in
number of
refinements
of 2 directions. cell AR=2^N*

*#7: Leave this at "21".*

------------------------------------------------------------------------

[(top)](#page-input-c3d-html--top)

---

<a id="page-prespec-c3d-cntl-html"></a>

## Sample preSpec.c3d.cntl

*Original page: [preSpec_c3d_cntl.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/preSpec_c3d_cntl.html)*

---

preSpec.c3d.cntl

---

 *#*
*#    This file contains samples showing how
to pre-specify regions for*
*#    adaptation at durring mesh generation,
and how to list components**#    on
which you want "[cubes](#page-meshgeneration-html--what-is)" to do additional
mesh refinement.*
*#*
*#    COMMENTS begin with '#' and can be added
anyplace*
*#*
*#    to get "cubes" to look at this file use
the -pre <filename> command line*
*#    option, (e.g. % cubes -pre
preSpec.c3d.cntl) If you dont use the
-pre*
*#    option, it will  assume no preSpec
file exists.*
*#*
*#   
----------------------------------------------------*
*#
FORMAT:*
*#    (mins and maxes are defined in the geometry coordinate
system)*
*#*

*# **BBox: level  
Xmin   Xmax     
Ymin   Ymax     
Zmin    Zmax***
*#      (int) 
(float) (float)   (float) (float)  (float) 
(float)
#*#
XLev: NumXtraLevels Comp 
Comp  Comp ...
### XLev:  (int)
       (int) (int) (int) ...
#
*#   
----------------------------------------------------*

**[$\_\_Prespecified\_Adaptation\_Regions:](#page-prespecnote-html)**
 
### <-Section
heading(required)
                                                 
#   This example was for a
transonic
                                                 
#         ONERAM6
wing
**BBox:** 7   0. 
1.     -1. 
1.      0.   1.4  *#  <- cover
the whole wing*
**BBox:** 8   0.  0.77   
0.  1.5     0.   0.78
*#  <- inboard upper surface*
**BBox:** 8   0.4 1.2    
0.  1.2     0.75 1.6 
*# 
<- outboard upper surface
#
### XLev:'s are optional and mostly used to pre-refine SurfBCs for
simulations with power
#*
**XLev:** 2   2   3 
8  13   *# ... do 2 extra
refinement passes on comps 2, 3, 8 &
13***XLev:** 1   1   4 
       *# ... do 1 extra
refinement pass on comps 1 & 4*

*#
===================================================*
*# 
**NOTES:***
*#   o  in this sample, there are 2 bounding
boxes (the 2 "BBox:" tags)*
*#      these must FOLLOW a the left
justified SECTION HEADING*
*#     
"$\_\_Prespecified\_Adaptation\_Regions:"*
*#      as shown
above.*
*#*
*#   o  **EMACS USERS:** emacs will
automatically highlight this file for you in*
*#      shell-script-mode. to get it
to do this automatically, add the*
*#      following line to your
.emacs file.*
*#      (setq
auto-mode-alist*
*#           
(append'(("\\.cntl$" . shell-script-mode) )
auto-mode-alist))* 

|  |  |
| --- | --- |
| ---   Questions?   Visit [Cart3D Discuss](https://groups.google.com/forum/#%21forum/cart3d),   or  [Contact Us](#page-cart3d-team-html)  last update December, 2024.   --- |  |

---

<a id="page-flowcart-reorder-html"></a>

## Reorder

*Original page: [flowCart_reorder.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/flowCart_reorder.html)*

Reorder ---  **[What is "reorder"?](#page-flowcart-reorder-html--what-is-reorder)**  **[Can I just do this automatically in "cubes"?](#page-flowcart-reorder-html--reorder-in-cubes)**  **[What does this order look like?](#page-flowcart-reorder-html--what-does-this-order-look-like)**  **[Mesh Partitioning with SFCs?](#page-flowcart-reorder-html--mesh-partitioning-with-sfcs)**  **[Running Reorder, usage](#page-flowcart-reorder-html--running-reorder-usage)**    ---  **What is "reorder"?** **Reorder** takes meshes (Mesh.c3d) output by cubes and reorders them using a space-filling-curve based ordering. You should look at it as a backend for cubes.  **flowCart** takes these re-ordered meshes and can partition them on-the-fly onto any number of processors.  Even if you're going to run on a single CPU (unpartitioned domain) its worth reordering, since reordered meshes have better locality and will execute faster on cache-based machines.  Reorder is the key for **flowCart's** domain-decomposition strategy  [top](#page-flowcart-reorder-html--top) **Can I reorder automatically in cubes?** Starting with release of Cart3d\_v1.3, you can reorder automatically from within cubes using the "-reorder" command line option to cubes. This option tells cubes to do the reordering upon output of the mesh, and  names the output mesh Mesh.R.c3d, just as if you ran reorder externally. Reordering from within cubes  streamlines the meshing-simulation cycle somewhat, but doesn't offer all the choices as the standalone reorder code does, it also increases cubes' memory requirements somewhat. Do either, your choice.  [top](#page-flowcart-reorder-html--top) **What does this order look like?** A "space-filling curve"  is a curve which provides a map from a 2-D plane or a 3-D space to a 1-D interval. They have many interesting mathematical characteristics and there are some excellent resources on the web for getting acquainted with space-filling-curves. We have a paper on the various uses of space-filling-curves in Cart3D. Otherwise, a good place to start is:  [http://www.dcs.napier.ac.uk/~andrew/hilbert.html](http://www.dcs.napier.ac.uk/%7Eandrew/hilbert.html) (with a windows screen saver :o)  [http://www.math.ohio-state.edu/~fiedorow/math655/Peano.html](http://www.math.ohio-state.edu/%7Efiedorow/math655/Peano.html) (with illustrations of the 3D version) **Reorder** uses these curves to re-order the cell and face lists that are output from **cubes**.  There are options for using either Morton or Peano-Hilbert SFC's to reorder your mesh. In general Peano-Hilbert produces slightly better partitionings and therefore its set  up as the default.  Here is a 2-D view of these two orderings.   - **Morton** or "N"   ordering: - **Peano-Hilbert** or   "U" ordering:  From these illustrations, its pretty clear to see how these curves can be used as partitioners but lets make it even clearer.   [top](#page-flowcart-reorder-html--top) **Mesh Partitioning with SFCs?** |  |  | | --- | --- | | Since the SFCs map the problem domain to a one- dimensional hyperspace along the curve, we can partition the problem space by partitioning the 1-D curve (e.g. collecting cells as we traverse the curve).  In the example a the right, there is an adapted mesh with 3 levels of cells, and its been partitioned into 4 subdomains.  Since this example is so small, the partition quality doesn't look particularly good. However, one characteristic of the SFC's that we use is that of *locality* this property guarantees that meshes will have asymptotically good partitioning. In fact, with both Peano-Hilbert and Morton orderings, the surface-to-volume ratio of the partitions will approach that of a cube.  Here are 2 examples of the meshes partitioned with the U-order.   - Small mesh around 3   teardrops (4   partitions) - Large   mesh around complete space shuttle launch   vehicle (16   partitions) |  |  [top](#page-flowcart-reorder-html--top) **Running Reorder, usage:** **Reorder** takes both the **Mesh.c3d** and the meta information in **Mesh.c3d.Info** output by **cubes**, and returns the reordered mesh to **Mesh.R.c3d**. It usually runs pretty quick, since its runtime is dominated by an internal call to qsort.  The usual unix command line is simply:  **% reorder**  Command line options can be used to specify either a different **Mesh.c3d.Info** file (**-m** ***infoFileName***)  or a different mesh file (**-i *MeshFileName***). By default, the output file name is "**Mesh.R.c3d**" to remind us that the mesh has been reordered and is ready for **flowCart** to do the domain-decomposition. You can also choose between Morton and Peano-Hilbert orderings, with Peano-Hilbert being used by default.   After **reorder** executes, you can delete the original **Mesh.c3d**, since you wont need it anymore.  Here is the full usage statement:  **% reorder -**   |  | | --- | | **Usage: reorder [ argument list ]**  **Options:**  **-i %s ... Input  mesh file name, default:<Mesh.c3d>**  **-o %s ... Output mesh file name, default:<Mesh.R.c3d>**  **-m %s ... Mesh info file,        default:<Mesh.c3d.Info>**  **-sfc %c.. sfc choice, H=peano-hilbert (default), M=morton  -s ...... Use perfect sort of face list (default is bin sort)** | [top](#page-flowcart-reorder-html--top)   ---   Questions?   Visit [Cart3D Discuss](https://groups.google.com/forum/#%21forum/cart3d),   or  [Contact Us](#page-cart3d-team-html)  last update Aug. 2015, M. Aftosmis   --- |

---

<a id="page-flowcart-mgprep-html"></a>

## mgPrep

*Original page: [flowCart_mgPrep.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/flowCart_mgPrep.html)*

mgPrep & mgTree ---   **[What are "mgPrep" & "mgTree"?](#page-flowcart-mgprep-html--what-is-mgprep)**   **[Input/Output files and Usage:](#page-flowcart-mgprep-html--input-output-files-and-useage)**   **[An example of coarse mesh generation with "mgPrep":](#page-flowcart-mgprep-html--an-example-of-coarse-mesh-generation-with-mgprep)**  **[Partitioning and coarse meshes:](#page-flowcart-mgprep-html--partitioning-and-coarse-meshes)**   **[Further Documentation:](#page-flowcart-mgprep-html--moredoc)**    --- **What are "mgPrep" and "mgTree"?** To use **flowCart's** multigrid convergence acceleration, you need to generate a series of coarse meshes in addition to the fine mesh upon which you actually want to see the final solution. **mgPrep** is the mesh coarsening module which creates these coarse grids from an initial input grid.    mgTree is a special version of mgPrep that uses a tree-based mesh decomposition and coarsening strategy, it is essentially a drop-in replacement for mgPrep that we use in 2D. See the NACA 0012, or either of the two unsteady samples that are included in $CART3D/cases/samples/ or the  example in $CART3D/cases/samples\_adapt/naca0012 **Input/Output files and Usage:** **mgPrep** takes reordered meshes (from **reorder**) usually named **Mesh.R.c3d** and produces hierarchies of meshes (usually named **Mesh.mg.c3d**). The ordering of the input meshes is preserved and is propagated to coarser meshes in the hierarchy. So if you pass it a mesh that you reordered with Peano-Hilbert, all the coarse meshes will be ordered by Peano-Hilbert too. Here is the full usage statement:  **% mgPrep -**   |  | | --- | | **Usage: mgPrep [ argument list ]**    **Options:    -n %d        Number of MultiGrid levels to prepare (n >= 2)    -i %s        Input  file name, default:<Mesh.R.c3d>    -o %s        Output  file name, default:<Mesh.mg.c3d>    -sfc %c      sfc init mesh order, H=peano-hilbert (default), M=morton    -v           verbose mode ON    -separate    dumps coarser meshes in separate files    -pmg         Write the finest mesh twice    -no\_stats    Suppress printing of coarsening statistics    -verifyInput Verify the input mesh before coarsening    -Xcut %d     Num of X=const cut planes <mgPlanes.dat>    -Ycut %d     Num of Y=const cut planes <mgPlanes.dat>    -Zcut %d     Num of Z=const cut planes <mgPlanes.dat>    -Dcut        Dump tecplottable file <cutcells.dat>**  **-mesh2d       Not recommended**, use "% mgTree -mesh2d " instead (see $CART3D/cases/samples/naca0012/HOWTO.html) |  [top](#page-flowcart-mgprep-html--top) **An example of coarse mesh generation with "mgPrep":** the command line:  **% mgPrep -n 5 -i Mesh.R.c3d**  will produce a sequence of 5 meshes from the original input **Mesh.R.c3d** that look something like this (click on any image for an enlargement). This hierarchy of meshes will get stored in a file called **Mesh.mg.c3d** by default, and is ready to be read into **cubes**. |  |  |  |  |  | | --- | --- | --- | --- | --- | | Input mesh | 1st coarse mesh | 2nd coarse mesh | 3rd coarse mesh | 4th coarse mesh | |  |  |  |  |  | | 4500000 cells  Coarsening Ratio: | 630000 cells  7.1:1 | 97000 cells  6.5:1 | 19000 cells  5.1:1 | 4500 cells  4.2:1 |  **Note:** The output Mesh hierarchy in this case can be used to run anywhere from single mesh (% flowCart - mg 1) to 5 levels of multigrid (% flowCart -mg 5)  or grid sequencing - you dont need to decide how many of levels of multigrid to use until runtime.  Also note that once you have the coarse mesh stack, the fine mesh is the first one stored in Mesh.mg.c3d, so once you have that, you're free to delete Mesh.R.c3d. % \rm Mesh.R.c3d  [top](#page-flowcart-mgprep-html--top) **Partitioning and coarse meshes:** As a happy consequence of some of the mathematical properties of the SFC partitioner, coarse grids will get partitioned in a manner similar to the way that their fine mesh got partitioned. This helps to reduce the amount of communication between subdomains in multi-processor runs. Here is an example showing the partitioning of a mesh hierarchy around an X-38 for a run on 2 processors. Note that the meshes in the blue and pink partitions remain nicely overlapped to reduce the amount of communication  with other processors durring multigrid prolongation and restriction. |  |  |  | | --- | --- | --- | | **Input mesh** | **1st coarse mesh** | **2nd coarse mesh** | |  |  |  |  ---   Questions?   Visit [Cart3D Discuss](https://groups.google.com/forum/#%21forum/cart3d),   or  [Contact Us](#page-cart3d-team-html)  last update Aug. 2015, M. Aftosmis   --- |

---

<a id="page-flowcart-run-html"></a>

## Running flowCart

*Original page: [flowCart_run.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/flowCart_run.html)*

Running flowCart ---   **[Input/Output files](#page-flowcart-run-html--usage)**    **[Notable new features](#page-flowcart-run-html--new)**  **[Command line options and usage](#page-flowcart-run-html--usage)**   **[Multi-processor running (shared memory and MPI)](#page-flowcart-run-html--multicpuruns)**   **[F](#page-flowcart-run-html--fluxfuns)****[lux Function Options:](#page-flowcart-run-html--fluxfuns)**  [Robust Mode](#page-flowcart-run-html--robustmode)  [Limiter Choice](#page-flowcart-run-html--limiters)  **[How do I restart a calculation?](#page-flowcart-run-html--restarting)**  **[More information on flowCart I/O](#page-flowcart-io-html--top)**    ---      **Input/Output files:** **flowCart** requires an [input.cntl](#page-input-cntl-html) file, a mesh, and a [Mesh.c3d.Info](#page-notes-html--note-1) file to get running. If you want to map the solution back to the surface triangulation, then you'll also need to provide the configuration triangulation file which you used to create the mesh (in **cubes**).  Meshes can be any one of the following:   |  |  |  | | --- | --- | --- | | **File type - default name** | **Contents** | **Platform and solution approach** | | **Mesh.c3d** | Unordered mesh from **cubes** | *Single* CPU, *no* multigrid or grid sequencing | | **Mesh.R.c3d** | Reordered mesh ready for partitioning | *Multiple* CPU, *no* multigrid or grid sequencing | | **Mesh.mg.c3d** | Reordered, multilevel mesh file | *Multiple* CPU, *with* multigrid or grid sequencing |   Most often, you're feeding flowCart a "Mesh.mg.c3d" mesh, and you can delete the other files. For more information on **flowCart** I/O command line options and files, click **[here](#page-flowcart-io-html)**.  When using the  "[$\_\_Force\_Moment\_Processing](#page-input-cntl-html--force-moment)" section in [input.cntl](#page-input-cntl-html) you will also have to have a [GMP-style Config.xml](#page-config-xml-html) file, but flowCart will automatically produce one of these if the only component you're interested in is "entire".  Checkpoint files  allow you to restart or reload simulations. When running steady-state they contain the current solution and are named something like check.##### where the current mg-cycle gets stamped into the #'s (e.g. check.00200). When you're running unsteady, the files get a "\*.td" suffix attached to indicate that they came from a time-dependent run, (e.g. check.000200.td). You'll notice that checkpoints from unsteady runs are generally about twice as large as steady runs. This is due to the fact that unsteady checkpoints actually store two time levels worth of data. Having two time levels permit restarts to advance with full second-order time accuracy, without any hiccup in convergence. Finally, note that flowCart automatically identifies steady state and time dependent checkpoints. You can initialize an unsteady simulation from a steady state run, or even restart a time dependent run in steady state mode. Feel free to mix-and-match. MPI and Shared Memory version of flowCart are 100% compatible.  **Notable new features:**  Here is a short list of some notable giving you control over how flowCart/mpix\_flowCart behaves and what you can make it do.   - ‐nSteps %d: You can now run unsteady   (time-dependent) simulations. When running unsteady, we use a   dual-time approach, and the ‐N %d flag tells flowCart the max   number of inner mg-cycles to use within a time   step. - ‐stats %d:   Extract statistical information like   averages, min's and max's, over a temporal   window of your run. "Stats" work for   steady-state simulations as well, where you'll   obviously get iterative averages and not   time-averages. In combination with ‐T, stats also produces a   distribution of min's, max's and average values   over the entire surface. If you've requested loadsCC.dat, ‐stats will also produce loadsCC.avg.dat with   time/iteration averaged loads. - ‐fine: Modifies   the meaning of ‐N   so that it requests a specified number of   mg-cycles on the finest mesh (coarse grid   mg-cycles "don't count"). The ‐fine flag also makes ‐N incremental, so it says "run   this many more   multigrid cycles" on restarts. - [*Simpler   and more flexible propulsion boundary   conditions*](#page-howto-samples-power-readme-html) along with   mass-flow steering and mass flow objective   functions for adaptation and design. - Mass flow monitoring:   flowCart   now tracks mass flow through inlets and exits   specified with the SurfBC tag in [input.cntl.](#page-input-cntl-html--bcs)   Mass flows through these components get reported   in loadsCC.dat. - Equivalent Area Sensors:   Similar to extracting a pressure distribution   with a lineSensor, [eaSensors](#page-input-cntl-html--postprocess)   extract equivalent area measured along a line in   the flow. - Buoyancy driven flow:   Run atmospheric simulations with buoyancy by   specifying the [Froude number](http://glossary.ametsoc.org/wiki/Froude_number) and a   gravity vector in [input.cntl.](#page-input-cntl-html)   **Command line options and Usage:**  **Usage:**  **% flowCart -**   |  | | --- | | % **flowCart -**  **Usage: flowCart [ OPTIONS ]**    **Options:** **-- Runtime Options--  -N %d** Max Number of multigrid cycles to advance **-fine** Do -N cycles on finest mesh (incremental if restart) <FALSE> **-mg %d** Number of multigrid levels **-gs %d** Use grid sequencing on # levels for startup (no mg)  **-cfl   %f** CFL number **-tm  %f** Cut-cell Grad mod (1stOrd=0. -> 1.0=2ndOrd) <def: 1.0> **-limiter %d** 0=None, 1=BJ, 2=VanLeer, 3=SinLim, 4=VanAlbada, 5=MinLim **-buffLim** Buffer limiters to neighbors default: <false> **-flux  %d** Flux function  <0> (0=VanLeer, 1=VanLeer-Hanel, 2=Colella, 3=HLLC) **-no\_fmg** No full multigrid startup(begin MG at finestLevel) **-subcell** Use subcell resolution on finest mesh **-gr** Use grads in MGrestrict (auto ON if gradEval all stages) **-choice %d** 1=do\_mg\_new, 2=jameson, 3=orig\_do\_mg\_cycle**,** <def: 3> **-y\_is\_spanwise** Default assumes z\_is\_spanwise direction **-stats %d** Compute running avgs over this many cycles or timesteps   **-- Unsteady Run Options --  -nSteps %d** Max number of unsteady time steps (triggers time-dependent run) **-dt %f** Non-dimensional size of physical timesteps Lref/ainf**-checkptTD %d**      TD checkpoint every "checkptTD" timesteps <end of run>**-checkptGrads**      Write out checkpoint files for gradients at end of run **-vizTD  %d**       Output viz files every "vizTD" timesteps <same as checkptTD> ****-autoDT %d****      Auto adjust physical timestep for roughly const. CFLphys <FALSE>  **-refWaveSpeed %F** Reference max wave speed (with -autoDT option) <UNSET>  -**track %d**Track error for adapt every "track" time steps <don't track> **-clean**Relax solution before time-stepping and reset nSteps  <FALSE>   **-- I/O Options --  -no\_ckpt** Suppress checkPointing **-restart** Restart into any # of partitions <Restart.file> **-v** Verbose mode ON **-T** Dump surf triangulation in Tecplot format <surfName.dat> **-clic** Dump surf triangulation in Clic format <surfName.triq> **-his** **Deprecated** Write [history.dat,forces.dat] **<now auto on>**;  **-no\_his** Turn off writing history files <history.dat,forces.dat> **-binaryIO** Write postprocessing data in binary (plotfiles etc.) <FALSE> **-i %s** Input  file name, default:<input.cntl> **-Xcut %d** Num of X=const cut planes <disjointCutPlanes.dat> **-Ycut %d** Num of Y=const cut planes <disjointCutPlanes.dat> **-Zcut %d** Num of Z=const cut planes <disjointCutPlanes.dat> **-Dcut** Dump tecplottable file <cutcells.dat> **-version** Dump version info and exit |  [top](#page-flowcart-run-html--top) **Multi-processor running:** |  |  | | --- | --- | | **flowCart** uses [OpenMP](http://www.openmp.org/) or  [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) for *explicit* communication between subdomains in multi-CPU/multi-core runs. This means that you can run it on any machine or cluster with either distributed or shared memory. Since it always explicitly decomposes the domain, both executables achieve extremely good scalability. To the right is a plot of showing parallel speedup (from Feb. 2014) for a typical case on about 57M cells.This case was run on NASA's Pleiades when it was the [11th fastest in the world.](http://www.top500.org/list/2015/06/)  Both MPI and Shared Memory (OpenMP) versions scaling linearly down to below 100k cells per core. Here is an older comparison that was done on NASA's famous Columbia system.     More  information on the programming paradigm and details of the parallelization are availible in: |  |  ‐ "[Using OpenMP: Portable Shared Memory Parallel Programming](http://www.amazon.com/Using-OpenMP-Programming-Engineering-Computation/dp/0262533022/ref=sr_1_1?ie=UTF8&s=books&qid=1228421788&sr=1-1)" by Chapman, Jost and van der Pas. Oct. 2007   ‐ ["Performance of a New CFD Flow Solver using a Hybrid Programming Paradigm". Jol Para. Dist Comp. 2004 (pdf).](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/JolParDistComp03.pdf)   - To run   on a multi-cpu or multi-core system with OpenMP (shared   memory) follow these steps (csh):   |  |  | | --- | --- | | **1% setenv OMP\_NUM\_THREADS nProc** | *#  ...nProc is the integer*  *#    number of CPU's you want* | | **2% flowCart -[OPTIONS]** | *# ...run the executable, flowCart will interrogate*  *#     the environment and set the number of subdomains*  *#     to nProc automatically* |   - MPI   executables are built using both the [OpenMPI](http://www.open-mpi.org)   and  [SGI-MPT](http://www.nas.nasa.gov/hecc/support/kb/sgi-mpt_89.html) Message Passing   libraries. To run on a multi-cpu or multi-core   system with MPI   do the following (csh):   |  |  | | --- | --- | | **1% mpiexec -np nProc mpix\_flowCart -[OPTIONS]** | *#  ...nProc is the integer*  *#    number of CPU's you want* |    Note: We frequently get questions about MPI running.  mpix\_flowCart is distributed using dynamically linked executables. There often turn out to be basic issues with the underlying MPI installation. If you're new to MPI, there are a couple of things you should start with before jumping in...   1. Run flowCart (shared memory) on    more than 1 thread using OpenMP. 2. Does "%mpiexec -np 2    /bin/date" return 2 copies of the date? 3. Can you run any (other) mpi code? 4. Can you run any of the basic MPI    demos installed on your system?    [top](#page-flowcart-run-html--top) **Flux Function Options:** **flowCart** is designed to allow user-selectable flux functions. Currently, van Leer, Colella'89 and HLLC (beta-test) are available. Of these van Leer ("-flux 0") and van Leer-Hänel ("-flux 1") are the most well exercised, and since they're positive flux functions, they're certain to be the most robust. Most users should choose "-flux 0" (van Leer) as their default. You can specify the flux function on the command line using "-flux 0" or by using the  **FluxFun** tag in the [**$\_Solver\_Control\_Information**](#page-input-cntl-html--solver) category of the [input control file](#page-input-cntl-html) and set it to "0". Flux functions are ordered roughly in order of decreasing smearing of contact discontinuities. HLLC gets an exact contact speed, while this is the classic Achilles heel of van Leer. Take a 2D airfoil (like the naca0012 in $CART3D/cases/samples/naca0012) and experiment.   [top](#page-flowcart-run-html--top)  **Robust Mode:**  Starting with versions v1.3, **flowCart** has a new "robust mode" of operation. In general, the code is very robust running with the standard Runge-Kutta (RK) scheme sketched in the sample [input.cntl](#page-input-cntl-html) file, This scheme works great most of the time, but its a compromise between robustness and speed. As is clear from the RK tags, typically you can get away with only evaluating the gradient and limiters at the first stage of the scheme, saving  this work on subsequent stages. In taking this shortcut, we give up some of the remarkable positivity property of some flux-vector-splittings. By being more careful - evaluating the gradient at every stage in the RK scheme, and re-evaluating it during multigrid restrictions, the scheme can be made extremely robust.  To invoke "robust mode" simply set up your RK scheme in the [input.cntl](#page-input-cntl-html) file to evaluate the gradient (and limiter) at every stage in the RK, like this:   Entries in [$\_\_Solver\_Control\_Information](#page-input-cntl-html--solver) block of "[input.cntl](#page-input-cntl-html)" to invoke "robust mode"  RK    0.0695      1  RK    0.1602      1  RK    0.2898      1  RK    0.5060      1   RK    1.0         1 Use "robust mode" sparingly. Nearly all calculations will run with the standard scheme (which is about 50% cheaper). Virtually all the simulations that you'll find on this website were run with the standard scheme.    [top](#page-flowcart-run-html--top)    **Limiter Choice:**  Starting with Cart3D\_v1.3 the limiters have been completely re-coded to be better behaved than in earlier distributions. They are now linearity preserving regardless of mesh stretching, and there is a choice of 5 of them. Having said that, you should almost always use limiter 2 (van Leer). This is the most aggressive smooth limiter and is not excessively dissipative - especially in the k-exact implementation in flowCart. Furthermore, its among the fastest to compute. The limiter choices are (in order of increasing dissipation): (0) no limiter, (1) Barth-Jespersen (2) van Leer, (3) sin limiter, (4) van Albada (5) Minmod. To understand the differences between them,  Look at the figure below. The plot on the left shows the limiter value (0-1) vs. the normalized acceleration or deceleration of the data. (sketch on the left). Phi = 1 means "no limiting". The parameter f is equivalently either del+/delC or del-/delC. The Barth-Jespersen (BJ) limiter essentially follows the monotonicity boundary for a 2nd order monotone scheme, so you can think of this as an effective "upper bound" for the limiter value. More aggressive, and you're not TVD. Below this curve, and you're limiting more than is strictly necessary. All the limiters go through Phi=0 at f=0.5. This says that they recommend no limiting when the data is linear, which is necessary for a 2nd order scheme.     Basically, we have BJ as the most aggressive, and minmod as the most dissipative. Everybody else falls in between. This graph helps you understand how much dissipation you're buying when you choose something other than "-limiter 1" on the command line, or in input.cntl. About the only time you should even consider something else is for high mach number cases since the more aggressive limiters may "over compress" the shocks leading to excessive "staircasing".  Van Albada is even more dissipative, and you can see that it gives up its slope substantially more quickly than sin(). BJ, or course, holds onto its slope as long as it can without overshoots, but the transitions are abrupt.   [top](#page-flowcart-run-html--top)   **How do I restart a calculation?** You can restart a calculation using the parameters in your input.cntl file, and your can override most of these, if you'd like, using command line options (preferred). Both are scanned upon restart. 1.  When you run a flowCart simulation, a checkpoint file is automatically produced (you can suppress this with the "‐no\_ckpt " command line option, if you know you don't want to restart).  This checkpoint file is automatically named using the scheme checkpoint\_file\_name = **check.**[*nCycles*], so if you ran 150 cycles the restart file would be named "**check.00150**". When you're running unsteady, the files get a "\*.td" suffix attached to indicate that they came from a time-dependent run, (e.g. check.000200.td). You can use these files to restart the calculation by soft-linking the special name "Restart.file" to the checkpoint file, and invoking the "‐restart " command line flag. e.g.  |  |  | | --- | --- | | **%ln -s check.00150 Restart.file** | *# create a softlink* | | **%flowCart -restart -N 200** | *# run flowCart an additional 50 cycles* | 2.  flowCart automatically identifies steady state and time dependent checkpoints. You can initialize an unsteady simulation from a steady state run, or even restart a time dependent run in steady state mode. Feel free to mix-and-match. MPI and Shared Memory version of flowCart are 100% compatible. [top](#page-flowcart-run-html--top)   ---     *last update December, 2024.* |

---

<a id="page-flowcart-io-html"></a>

## flowCart I/O

*Original page: [flowCart_io.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/flowCart_io.html)*

Input/Output files ---   [What input files are required by flowCart?](#page-flowcart-io-html--input)    [What output files are produced by flowCart?](#page-flowcart-io-html--output-files)    [How do I extract "pretty" cutting planes?](#page-flowcart-io-html--cutplanes)    [How do I restart a calculation?](#page-flowcart-io-html--restart)     --- **What input  files are required by flowCart?** **flowCart** requires 4 files to get underway.  1. an [Input    control file](#page-input-cntl-html) (default name <[input.cntl](#page-input-cntl-html)>)    which controls the behavior of the solver. 2. the [Mesh.c3d.Info](#page-notes-html--note-1)    file produced by cubes when you generated the mesh ([see](#page-notes-html--note-1)**Note    1**). 3. a Cart3D    [wetted    surface triangulation](#page-cart3dtriangulations-html) <Components.i.tri> which describes the    geometry    to the solver. This geometry is generally produced by the [intersect](#page-surfacemodeling-html--intersect)code (but any \*.i.tri file will do). This file is specified in the [Input    control file](#page-input-cntl-html), and *must* be the same as that used by [cubes](#page-meshgeneration-html--what-is)    to produce the mesh. 4. a Cart3D    mesh file (default name <[Mesh.c3d](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/fileformat.pdf)>    but most frequently this will be <Mesh.mg.c3d>) which is produced    by    [cubes](#page-meshgeneration-html--cubes),    [reorder](#page-flowcart-reorder-html),    or [mgPrep](#page-flowcart-mgprep-html). 5. If you're    using the new (Cart3D v1.4)  "[$\_\_Force\_Moment\_Processing](#page-input-cntl-html--force-moment)"    section in [input.cntl](#page-input-cntl-html)    you will also have to have a [GMP-style    Config.xml](#page-config-xml-html) file, but flowCart    will automatically produce one of these if the only component you're    interested in is "entire". **What output files are produced by flowCart?**     ([top](#page-flowcart-io-html--top))  **flowCart** produces 3 types of output files.  1. "Restart"    or "checkpoint" files: see ["How do I restart a    calculation?"](#page-flowcart-io-html--restart)    below for information about the uses of these files 2. Optionally,    flowCart can be asked to produce various types of postprocessing    files.     These files have filename extensions of **\*.triq** or **\*.dat** or **\*.plt.** These last 2 types may be read directly into the Tecplot or Paraview visualization software. ([see](#page-notes-html) Note 2) These files can contain snapshots    of    the final discrete solution displayed on cutting planes through the    mesh    <cutPlanes.dat> or mapped to the input wetted surface    triangulation    <inputSurfFileName.dat>.  The "\***.triq**" files conform    to    Cart3D's [annotated surface    triangulation](#page-cart3dtriangulations-html)    file format, and can be used directly by the integrated force and    moment    computation module ([clic](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/clic/clic.html)).  Using    clic,    you can extract any force, moment, and Cp profiles for any    configuration    or collections of components within a configuration. This table    summarizes    the flowCart command line arguments and/or input control file flags    used    to create postprocessing files.  By the way, **yes**, you can    always    restart a calculation for 0 (zero) additional iterations to extract    postprocessing    files.  |  |  |  | | --- | --- | --- | | **cmd line option** | **[input.cntl](#page-input-cntl-html) file section heading** | **file produced** | | -T | (none) | inputSurfFileName.dat ([Tecplottable](#page-notes-html--note-2)) | | -clic | (none) | inputSurfFileName.triq ([clic](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/clic/clic.html) input file) | | (none) | **$\_\_PostProcessing:** | cutPlanes.dat | | -{X,Y,Z}cut | (none) | disjointCutPlanes.dat (ugly [see](#page-notes-html--note-3) **Note 3**) | | -his <default> | (none) | forces.dat, moments.dat, history.dat | | (none) | **$\_\_Force\_Moment\_Processing:** | lineSensor.XXX.dat & pointSensor.XXX.dat | | (none) | **$\_\_Force\_Moment\_Processing:** | loadsCC.dat & loadsCC.avg.dat (ASCII files with final component loads data) | | (none) | **$\_\_Force\_Moment\_Processing:** | <GMPCompName>.dat (xy-plot file with loads history for selected component) |  3. Convergence    monitoring  files: FlowCart automatically writes [history.dat](#page-history-dat-html), [forces.dat](#page-forces-dat-html)  and [moments.dat](#page-moments-dat-html) for    recording convergence    information. These files are directly plottable with the free solftware    packages like gnuplot, xmgr or xmgrace available on most X-windows systems. Both of    these    files begin with a comment region containing meta information including    details of the run so that the run can be reproduced at a later time.    They    also include a copy of the exact command line just incase you forget    what    you did. One column in each of these files contains the elapsed    wall-clock    time between iterations, click on [history.dat](#page-history-dat-html),     [forces.dat](#page-forces-dat-html) or [moments.dat](#page-moments-dat-html) to see an example of each. The Force or Moment  tags in the [$\_\_Force\_Moment\_Processing:](#page-input-cntl-html--force-moment) section of [input.cntl](#page-input-cntl-html) also can be used for convergence monitoring. In this case it will give an iterative history of integrated loads on the requested [GMP component](#page-config-xml-html).   The pointSensor tag in the [$\_\_Convergence\_History\_reporting:](#page-input-cntl-html--converge) section of [input.cntl](#page-input-cntl-html) will monitor the values of the state vector and Cp at the point specified by the sensor.      When you're running an unsteady simulation, checkpoint files, line and point sensors, cutting planes and surface triangulations all get a "timestamp" inserted into their names. When you ask for "-stats" the mean and envelope surface loads file  gets written to Components.i.stats.dat, and if you're running unsteady, this file also gets a timestamp.   ([top](#page-flowcart-io-html--top))   **How do I extract "pretty" cutting planes?**  There are 2 ways of extracting cutting planes from flowCart simulations, In the table above, the Xcut, -Ycut, -Zcut options will do this, however, these will show the solution reconstructed on a cell-by-cell basis, and you'll get different "zones" for each partition in the domain decomposition. These were really only intended for use during development, and I don't recommend these options **Much better way** is to use the **$\_\_Post\_Processing** category in the [input.cntl](#page-input-cntl-html) file. In this category, you can request specific cutting plane lists to be output at the end of your run using the **Xslices**, **Yslices**, or **Zslices** tags. Here is an example (goes anyplace in your [input.cntl](#page-input-cntl-html) file): **$\_\_Post\_Processing:** *# Pretty printed cutting planes*   **Xslices -.5  0.1** *# X-stations for cutPlane extraction*   **Zslices 0.001 .297** *# Z-stations for cutPlane extraction*  ([top](#page-flowcart-io-html--top)) **How do I restart a calculation?**    Link the checkpoint file ("check.XXXXX") to the name "Restart.file" and press play.    See the page on running **flowCart** ([here](#page-flowcart-run-html--restarting)) and the samples in $CART3D/cases/samples  for full details.   [top](#page-flowcart-io-html--top) ---   Questions?   Visit [Cart3D Discuss](https://groups.google.com/forum/#%21forum/cart3d),   or  [Contact Us](#page-cart3d-team-html)  last update December, 2024.   --- |

---

<a id="page-notes-html"></a>

## notes.html

*Original page: [notes.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/notes.html)*

**Note 1:**  **This is the whole
point of the Mesh.c3d.Info files** .
> These files contain meta
> information about the mesh and the command line flags used in
> creating the meshes which can be accessed by downstream
> programs. These files are ascii, so that you can look into them
> to jog your memory about how you ran cubes, but you shoud NOT
> edit any Mesh.c3d.Info file.

**Note 2:**  **This should not be
considered an endorsement of Tecplot, Fieldview or of Paraview**
> We needed to choose some
> readily available visualization software to use for flow field
> visualization.  Most of the codes in Cart3D produce tecplot
> "\*.dat" output files for quick visualization, in most cases
> these can also be read into Paraview or Fieldview as well. You
> should note that Cart3D's extended tri file format is actually a
> subset of the VTK unstructured format, and is written in XML.
> Tecplot, Paraview and Fieldview can read these files without
> translation. Thery're also readable by any other vtk-compatible
> viewer. Dont know if your \*tri file is \*.trix or not? Just type
> %
> file myfile.tri. If it tells you that the file is XML,
> its trix formatted.

**Note 3:**  **You probably dont
want to use this option for serious plots.**
> The "-Xcut
> %d", "-Ycut
> %d" and "-Zcut
> %d" command line arguments
> are intended for quick on-the-fly plotting, but are not very pretty. They display
> views of the cell-by-cell reconstructed solution on a
> partition-by-partition basis.  This means that if you have
> 8 partitions, and ask for 3 cutting planes, you'll probably get
> 24 zones in your "cutPlanes.dat" file. In additon, there is no
> reason to expect contour lines to be continuous between a cell
> and its neighbor (since the reconstruction permits jumps between
> cells). To get pretty plots which display a node-based view of
> the discrete solution use the $\_\_PostProcessing category ID with
> the Xslices, Yslices or Zslices tokens in your input control
> file. e.g.
> **$\_\_Post\_Processing:**       
> *#
> <- category ID*
> **Xslices   
> 0.001 1.2 2.0**   *# <- get X-slices at x =
> 0.001, 1.2 and 2.0.*
>
> **Zslices   
> 7.    9.2 20.0**  *# <- get
> Z-slices at z = 7, 9.2 and 20.*

---

<a id="page-config-xml-html"></a>

## Config.xml basics

*Original page: [Config_xml.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/Config_xml.html)*

---

##

### *Config.xml*

Config.xml is a GMP
configuration file which describes a component hierarchy for your
configuration. This example shows one possible hierarchy for the bJet
example in $CART3D/cases/samples/bJet/bJet.a.tri
[AIAA 2003-1237](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA2003-1237.pdf) contains a full
description with many more examples.

Note:
"all"
and "entire" are special Component names
  
- "entire" refers the whole geometry as a single entity
  
- "all" refers to each individual named component in the
configuration

The [$\_\_Force\_Moment\_Processing:
section in flowCart's "input.cntl" file](#page-input-cntl-html--force-moment) requires Config.xml to exist, if one is not
found, flowCart will attempt to create it automatically. NASA's "OVERGRID"
code can also be used to create Config.xml

---

> ```
> <?xml version="1.0" encoding="ISO-8859-1"?>
> <!-- $Id: Config.xml,v 1.2 2008/11/28 20:06:34 aftosmis Exp $  -->
>
> <Configuration Name="bJet Sample" Source="bJet.a.tri">
>
> <!-- triangulated components -->
>  <Component Name="fuselage" Type="tri">
>    <Data> Face Label=1 </Data>
>  </Component>
>  <Component Name="wing"     Type="tri">
>    <Data> Face Label=2 </Data>
>  </Component>
>  <Component Name="htail"     Parent="empennage" Type="tri">
>    <Data> Face Label=3 </Data>
>  </Component>
>  <Component Name="vtail"     Parent="empennage" Type="tri">
>    <Data> Face Label=4 </Data>
>  </Component>
>  <Component Name="r_nacelle" Parent="engine"    Type="tri">
>    <Data> Face Label=5 </Data>
>  </Component>
>  <Component Name="l_nacelle" Parent="engine"    Type="tri">
>    <Data> Face Label=6 </Data>
>  </Component>
>  <Component Name="r_pylon"   Parent="engine"    Type="tri">
>    <Data> Face Label=7 </Data>
>  </Component>
>  <Component Name="l_pylon"   Parent="engine"    Type="tri">
>    <Data> Face Label=8 </Data>
>  </Component>
>
> <!-- Containers -->
>  <Component  Name="empennage" Type="container"> </Component>
>  <Component  Name="engines"   Type="container"> </Component>
>
> </Configuration>
> ```

---

<a id="page-input-cntl-html"></a>

## input.cntl - sample flowCart input file

*Original page: [input_cntl.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/input_cntl.html)*

---

#### input.cntl

- Comments are in pink text and start with a "#"
- Inputs
  in black text are mandatory
- Inputs
  in purple text are
  optional
- Inputs in grey text are mandatory, but generally
  not changed by users

##

---

#
#   
+--------------------------------------------------------+
#   
|       Input control & steering
file for **flowCart**       |
#   
|           3D
Cut-Cell Cartesian Flow
Solver           
|
#   
+--------------------------------------------------------+
# 
 
#     
NOTE:  o Start Comments in this file with the "#" character
#            
o Blocks can come in any order
#            
o info within blocks can come in any order
#
#             o  EMACS
USERS: emacs will syntactically highlight
#               
this file for you following shell-script-mode rules.
#               
to get it to do this automatically, add the following
#               
line to your .emacs file.
#               
(setq auto-mode-alist
#                  
(append'(("\\.cntl$" . shell-script-mode) ) auto-mode-alist))
#
 #
#
--------------------------------------------------------------

 $\_\_Case\_Information:     
### ...Specify Free Stream Quantities

Mach     0.54 
#  (double)
alpha    3.06 
#  (double) - angle of attack
beta     0.0  
#  (double) - sideslip Angle
gamma
   1.4
  #  (double) - ratio of specific heats (optional,
default = 1.4)

     
         #...Optional include
buoyancy
              
#   set Froude number and gravity vector
          
    #   Froude =
(Vref\*Vref)/(gref\*Lref)
 
            
#          = 9.1328
for std atm@30km w/ Lref = 1km
              
### Froude  (double) (double) (double) (double)
Froude 9.1328  0. 0. -1. #  Fr gx gy gz  # with |g|
= 1.0

 
$\_\_File\_Name\_Information:

MeshInfo          
Mesh.c3d.Info # Mesh information file (usually
Mesh.c3d.Info)
MeshFile          
Mesh.mg.c3d   # Mesh file

$\_\_Solver\_Control\_Information:

#  
Runge-Kutta Stage Coefficients
#
 #   stageCoef  GradEval  NOTE:
GradEval = 0 = no new evaluation at this stage,
#   
--------  -------        
GradEval = 1 = Yes, re-evaluate at this stage
RK    0.0695 
      1  #
RK   
0.1602       0  #
RK   
0.2898       0  #
->to run 1ST ORDER, set
GradEval to "0" in all stages
RK   
0.5060       0  #
->for  "ROBUST MODE" set GradEval to "1" in all
stages
RK    1.0
         0 
#
                        
# 
You
can use any Runge-Kutta scheme you like

CFL          
1.4 # CFL number typically between 0.9 and 1.4,
1.2 good default
Limiter      
1   # (int) default is 1, organized in
order of increasing
                 
#       dissipation.
                 
#         Limiter: 0 = no
Limiter
                 
#                 
1
= Barth-Jespersen
                 
#                 
2
= van Leer
                 
#                 
3
= sin limiter
                 
#                 
4
= van Albada
                 
#                 
5
= MinMod
                 
#
FluxFun      
0   # (int) - Flux
Function:   0 = van Leer
                  
#                         
1
= van Leer-Hanel
                  
#                         
2
= Colella 1998
                 
#                         
3
= HLLC (alpha test)
                               

Precon       
0   # (int) - Preconditioning: 0 = scalar
timestep
wallBCtype    0  
### Cut-Cell Boundary Condition type   0 = Agglomerated
Normals
                 
#                                   
1
= SubCell Resolution
nMGlev
       3
  
#
(int) - Number of MultiGrid levels (1 = single grid)
MG\_cycleType  2   # (int)
- MultiGrid cycletype: 1 = "V-cycle", 2 = "W-cycle"
                 
### 'sawtooth' cycle is: V-cycle with nPre = 1, nPost = 0
MG\_nPre      
1   # (int) - nunber of pre-smoothing 
passes in multigrid
MG\_nPost     
1   # (int) - number of post-smoothing passes
in multigrid

                 
#...Unsteady Run options (all are Optional)
                 
#   only active if "-nSteps %d" found on cmd line
TD\_method    
1   # {0,1} = {Bkwrd Euler "BE1", 2ndOrd Bkwrd
"BDF2"} <default=1>
TD\_dt\_phys   
0.1 # Non-dimensional physical time step  t\*=
L\_ref/a\_infd

 $\_\_Boundary\_Conditions:  
 
                 
    # BC types: 0 = FAR FIELD (Riemann)
                       
#          
1 = SYMMETRY
                       
#          
2 = INFLOW  (specify all)
                       
#          
3 = OUTFLOW (simple extrap)
                       
#         
18 = SBC\_ATMOSPHERE\_TOP (Riemann)
Dir\_Lo\_Hi    
0   0 0   # (int) {0,1,2}
direction  (int) Low BC   (int) Hi BC
Dir\_Lo\_Hi    
1   0 0   # (int) {0,1,2}
direction  (int) Low BC   (int) Hi BC
Dir\_Lo\_Hi    
2   1 0   # (int) {0,1,2}
direction  (int) Low BC   (int) Hi BC

   
                 
###   1. Generic INLET/EXHAUST BC specification
( 2004)
                     
###      Specify conditions upstream of a "power
face" (jet)
  
                  
###      or downstream of a "inlet"
(sink)
                    
###      Use CompIDs to identify
surfBC faces on the geometry,
                    
###      use any number of SurfBCs
      
             
###      see [AIAA Paper 2004-4837](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2004_4837.pdf)
                    
###            
(int)  (float) (float) (float) (float) (float)
SurfBC 2  2.0 3.0 0. 0. 5.0  
   # compID   Rho   
Xvel     Yvel   
Zvel   Press
SurfBC 
3  1.0 0.3 0. 0. 0.714285
### compID   Rho   
Xvel     Yvel   
Zvel   Press
#
#                   
2. Inlet boundary conditions ( see *[AIAA
Paper 2018-0334](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2018-0334.pdf)*)
#                                              
(int) (float)
 InletPressRatioBC 
3   1.27  #
InletPressRatioBC %d     %f  *# 
compID  P/P\_inf*
 InletVelocityBC   
4   0.73  # InletVelocityBC  
%d     %f  *#  compID 
Vnorm/c\_inf*
 #
#                   
3. Nozzle boundary conditions (,
see *[AIAA
Paper 2018-0334](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2018-0334.pdf)*)
#                                    
(int)(float)(float)
PowerStagBC 5   1.4 
1.1  # PowerStagBC %d   %f  
  %f *# compID Pt/Ptinf Tt/Ttinf*
PowerStagBC 6   1.6 
1.3  # PowerStagBC %d   %f     %f *#
compID Pt/Ptinf Tt/Ttinf*
PowerMFRBC  7  
0.5  1.2  # PowerMFRBC  %d   %f     %f *#**compID  MFR     Tt/Ttinf*

$\_\_Convergence\_History\_reporting:
iForce     1
  # (int) - Report force & mom. information
every iSkip cycles.
iHist     
1   # (int) - Update 'history.dat' every iHist
cycles.
nOrders   
8   # (int) - Num of orders of Magnitude
reduction in residual.

#
#                   
...PointSensors
for convergence monitoring: these allow
#                      
"live"
sampling of data at any number of specific locations
#                      
output
while case is running in "./pointSensor\_<Name>.dat"
#
pointSensor Name(%s) Xlocation(%g) Ylocation(%g) Zlocation(%g)
pointSensor pt0 
0.644  0.0717  0.965 # point sensor name x y z
for convrgnce monitor
pointSensor pt1 
1.1    0.0717  0.965 # point sensor
name x y z for convrgnce monitor
     
 $\_\_Partition\_Information:
 
 nPart  
1      # (int) - Number of
SubDomains to partition into:
type   
1      # (int) - Type of
partitioning: 1 = SpaceFillingCurve

$\_\_Post\_Processing:
#                
Extract "slices" (Pretty printed cutting planes)
#                                     
...general
format
#Xslices 
(float) (float) ...(float)     -- any number
of locations
#Yslices 
(float) (float) ...(float)     -- any number
of locations
#Zslices 
(float) (float) ...(float)     -- any number
of locations
#                        
...Example
 Zslices 0.0001 .30 .65  .97 1.19 1.34 1.4  # Choose
cut-plane locations

#                             
...Point and Line Sensors (output at end
#                                
of
run in "./lineSensor\_<Name>.dat"
#                                    
or
in "./pointSensors.dat"
#                          
 
NOTE: You can put in any number of line sensors
#lineSensor Name(%s) Xorig(%g) Yorig(%g) Zorig(%g)  
Xdest(%g) Ydest(%g) Zdest(%g)
#lineSensor (Name) (float) (float) (float)   (float)
(float) (float) 
 lineSensor
LINE1   0.  1.023  0.2331  3.1
1.023  0.2331
 lineSensor LINE2
  0.  1.023  0.5     3.1 1.023 
0.5
#
#                            
...Equivalent Area line sensor
#eaSensor Name(%s) Xorig Yorig Zorig 
      Xdest  Ydest  Zdest
   Radius
#eaSensor (Name) (float) (float) (float)   (float)
(float) (float) (float)
 eaSensor
  EA1     0.  1.023  0.5
         3.1 
1.023  0.5       3.0
#
#pointSensor Name(%s)
Xlocation(%g) Ylocation(%g) Zlocation(%g)
pointSensor point1   -2.1  2.3  4.0

#                  
        ...live steering this section is
optional
#           
        
         if it exists it
will get re-parsed every
#                             
iCLfreq
iterations
$\_\_Steering\_Information:

#              
       
  
...**Target Lift Coef. Steering**
#TargetCL 
(CL target value) (CL tolerance)  (iCLStart iter)
(iCLfrequency)
###             
(float)          
(float)        
(int)           
(int)
TargetCL  0.2 
0.01  60 5   # <- TargetCL TargetValue
tolerance iCLstart iCLfreq
### NominalAlphaStep  %f  # (OPTIONAL) choose Initial
alpha step, also max step size
NominalAlphaStep 0.2    # (float)
Degrees for delta Alpha, OPTIONAL, default <0.2>

#
#              
          
 ...**MassFlow
Steering** (see *[AIAA
Paper 2018-0334](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2018-0334.pdf)*)
#                            
May be used on compIDs with "InletVelocityBC"
#TargetMFR (Comp) (MFR target
value)  (tolerance)  (start)   (freq.)
 #         
(int)     
(float)         
(float)     (int)    
(int)
TargetMFR   
4        
0.90            
0.0001      
150       20

#              
          
...**Mach Number Steering**
#                            
May be used on compIDs with "InletVelocityBC"
#TargetMach (Comp) (Mach
target) (tolerance)  (start) (freq.)
#          
(int)   
(float)      (float)  
   (int)   (int)
TargetMach  
4       
0.45        
0.001        450  
  10

 $\_\_Force\_Moment\_Processing:
 #      
               
    [GMP
enabled](https://software.nasa.gov/software/ARC-15193-1) force reporting
#                          
requested
loads will be logged to
#                          
"CompName.dat"
during run, and final loads
#                          
will
be output in 'loadsCC.dat'
at termination
#          
NOTES:
#           
o This section uses components [defined in Config.xml](#page-config-xml-html)
 #
### ... Axis definitions (with respect to body axis directions
(Xb,Yb,Zb)
#                      
w/
usual stability and control orientation)
#
 Model\_X\_axis 
-Xb    # <<-- model's X axis is in
the (-X) stability axis direction
Model\_Y\_axis 
-Zb    #      see [this example for more info](#page-clic-html-clic-axis-html)
Model\_Z\_axis  -Yb

#                   
 ...Reference Area and
Length Specifications
#                      
Component
names and numbers refer to [those in
Config.xml](#page-config-xml-html)
 #                      
"all"
and "entire" are special Component names
#                      
"entire"
= the whole geometry
#                      
"all"   
= each individual named component
 # Reference\_Area (float)  compNumberList or
compNameList
Reference\_Area   
1.0  all

### Reference\_Length (float) compNumberList or compNameList
Reference\_Length 
1.  all

### ... Force and Moment Info (logged to 'CompName.dat'
#               
with final summary in 'loadsCC.dat'
)
#
### Force  compNumber or compName
Force   
entire  #
compNumber and compNames defined in [Config.xml](#page-config-xml-html)
Force    wing

 #           
      ...Compute moment of aero loads about
arbitrary point
 # Moment\_Point Xctr(%f) Yctr(%f) Zctr(%f) CompName or
CompNumber
 Moment\_Point 0.25
0. 0. entire
 Moment\_Point 1.25
0. 0. wing

#           
      ..."Moment\_Line" is useful for
extracting hinge-moments
 # Moment\_Line X0(%f) Y0(%f) Z0(%f) X1(%f)
Y1(%f) Z1(%f) CompName or CompNumber
 Moment\_Line 
0.25 0. 0.    0.25 0. 1. 
entire
 Moment\_Line 
0.5  0. 0.    0.7  0. 1. 
entire

$\_\_Design\_Info:  # OPTIONAL
 #                  
...this section pertains to adjoint-based mesh adaptation,
#                     
error-estimation
and aerodynamic shape optimization.
#                     
Listed below are just a subset of the available objective
###                
     functions. See complete  [documentation
on functionals here](#page-adjoint-doc-adj-functionals-html).
#                     
For
complete details please see the full documentation
#                     
distributed with these modules
#                     
in: 
[$CART3D/doc/adjoint/index.html](#page-adjoint-index-html)
 #
#
### Objective
Function: SUM of functionals (J)
### J = 0 -> W(P-T)^N
### J = 1 -> W(1-P/T)^N
#
### Bound: denotes the active boundary for functionals 0 and 1
### Bound = 0 -> J = (1-P/T)^2
### Bound = 1 -> J = (1-P/T)^2 iff P>T else Obj=0
### Bound =-1 -> J = (1-P/T)^2 iff P<T else Obj=0
#
### 1. Force Coefficients -- objective functions on force
coefs:
#   
Reference Frame = 0 Aerodynamic Frame
#         
          = 1 Aircraft
(Body) Frame
#
#    Force Codes: CD=0 Cy=1 CL=2 in
Aerodynamic Frame
#    Force Codes: CA=0 CY=1 CN=2 in Aircraft
(Body) Frame
#    Format:
#       
Name    Force  Frame  
J     N   Target
Weight Bound GMP\_Comp
#     
(String) (0,1,2) (0,1) (0,1) (int) (dble)
(dble)  (0)  (name)
----------------------------------------------------------------------
 optForce 
CD      
0      0    
0     1    
0.     1.    
0    entire
 optForce 
CL       2
     0    
0     1    
0.     0.5   
0    entire
#          
(multiple entries get weighted and summed)
 
### 2. Field Sensor objectives
#        
   Name     
J      N   
Target   Weight   Bound
#         
(String) (0,1,2) (int) (double) (double) (-1,0,1)
### --------------------------------------------------------
optSensor   LINE1  
   0     
2     
0.0     
1.0      0
optSensor  
point1 
   0     
2     
0.0      0.5
     0
optSensor  
EA1  
    
0     
2     
0.0     
1.0      0
#          
(multiple entries get weighted and summed)

### 3. Mass Flow Rate Objective
#    can be
used for any GMP\_comp w/ surface boundary condition defined
#    in $\_\_Boundary\_Conditions section.
#
#           
Name     J    
N    Target  Weight Bound GMP\_Comp
#          
(String) (0,1) (int) (dble)  (dble)  (0) 
(name)
#
------------------------------------------------------------
optMassFlow 
inlet    0    
1      0.   
1.0     0  fanFace
optMassFlow  exhaust  0    
1      0.   
1.0     0  turbineExit

---

last update December, 2024.

---

---

<a id="page-adjoint-aero-csh-html"></a>

## aero.csh

*Original page: [adjoint/aero_csh.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/adjoint/aero_csh.html)*

---

### *aero.csh* script

Tips are displayed if you "mouse-over" black keywords

---

> ```
> #!/bin/csh -f
>
> # $Id: aero_csh.html,v 1.19 2024/11/27 19:19:29 mnemec Exp $
>
> # AERO: Adjoint Error Optimization
> # Script to drive adjoint-based mesh refinement
>
> # ATTENTION: requires Cart3D release 1.5 or newer
>
> # M. Nemec, Marian.Nemec@nasa.gov
> # Oct 2006, last update: Oct 2021
>
> # Help:
> # ------
> # % ./aero.csh help
>
> # Read tips, hints and documentation in $CART3D/doc/adjoint
> # See examples in $CART3D/cases/samples_adapt
>
> # Set user specified options below, defaults are suggested
>
> # -------------
> # Basic options
> # -------------
>
> # Number of adaptation cycles, e.g. if you pick 8 then the run will terminate
> # after the flow solve in adapt08. Min value is 0, max value is 99.
> set n_adapt_cycles = 9
>
> # maxR for initial mesh (cubes)
> set maxR = 7
>
> # Spanwise orientation (-y_is_spanwise flag in flowCart)
> set y_is_spanwise = 0    # {Yes, No} = {1, 0}
>
> # Set mesh2d = 1 for 2D cases, mesh2d = 0 for 3D cases
> set mesh2d = 0
>
> # ----------------
> # Advanced options
> # ----------------
>
> # Uncomment next line to control thread affinity (improve parallel performance)
> # on linux machines
> #                     ...generally safer on hyperthreaded systems
> #setenv KMP_AFFINITY scatter
> #                     ...may be slightly faster when not hyperthreaded
> #setenv KMP_AFFINITY compact
>
> # ----- Flow and adjoint solver settings -----
>
> # Number of fine-grid flowCart iterations on initial mesh
> set it_fc = 150
> # Additional flowCart iterations on each new mesh
> # cycle         1   2   3   4   5   6   7   8   9  10
> set ws_it = ( 150 150 150 200 200 200 200 200 250 250 )
> # Number of fine-grid adjointCart iterations on each mesh
> set it_ad = 150
>
> # Number of flowCart multigrid levels (default=3)
> set mg_fc = 3
> # Number of adjointCart multigrid levels (usually same as flowCart)
> set mg_ad = 3
>
> # Limiter: default 1 (Barth-Jespersen) is the most accurate, 2 (van Leer) is
> # smoother and may offer deeper convergence. Limiter 5 (minmod) is most robust
> # and 0 means no limiter.
> set limiter = 1
>
> # Functional averaging window in terms of flowCart mg-cycles. This is useful
> # for cases that do not converge to steady-state. The averaged functional is
> # reported in the fourth column of fun_con.dat and in all_outputs.avg.dat
> # (default: avg_window = 1).
> set avg_window = 1
>
> # ----- Mesh adaptation settings ------
>
> # Maximum number of cells in the penultimate mesh, i.e. number of cells allowed
> # in the working mesh for the final embedding.  The error estimation step
> # requires ~6.9 GB per million cells.  You can use this to gauge the largest
> # mesh your memory resources allow. Default value is 9M, which assumes a
> # machine with 64 GB of memory.
> set max_cells2embed = 9000000
>
> # Specify mesh growth for each adaptation cycle. Mesh growth should be greater
> # than 1 and less than or equal to 8. Specifying 8 means that you allow
> # refinement of every cell in the mesh.  Recommended minimum growth is 1.1. We
> # found the sequence below to work well for many problems: if your initial mesh
> # has roughly 10,000 cells, then after 10 adaptations it will surpass 10
> # million cells. (For 2D cases, we recommend growth factors of 1.2 for first
> # two cycles and 1.4 for the rest.) If you enter 0 for any cycle, then the mesh
> # growth is selected automatically in that cycle.
> # cycle               0   1   2   3   4   5   6   7   8   9  10
> set mesh_growth = ( 1.5 1.5 2.0 2.0 2.0 2.0 2.0 2.5 2.5 2.5 2.5 )
>
> # Mesh growth can be selected automatically by enabling the auto_growth
> # option. This is a new feature, where aero.csh sets the refinement threshold
> # to the mean of the error distribution. This feature frequently yields better
> # results than the default mesh_growth array.
> set auto_growth = 0    # 0=no, 1=yes, default=0
>
> # Set apc: adapt or interface propagation cycle
> # a = adapt
> # p = propagate interfaces (adapt mesh without reducing finest cell size)
> # Use p-cycles sparingly. We recommend using one initial p-cycle to reduce
> # the bias of the initial mesh. P-cycles should also be used once the volume
> # mesh over-refines your surface triangulation.
> # cycle     0 1 2 3 4 5 6 7 8 9 10
> set apc = ( p a a a a a a a a a a )
>
> # ----- Customization of initial mesh -----
>
> # Use file name preSpec.c3d.cntl for preSpec regions (either BBoxes or XLevs
> # for cubes or ABoxes for adapt) 0=no, 1=yes, default=0
> set use_preSpec = 0
>
> # Initial mesh parameters (cubes)
> set cubes_a = 10    # angle criterion (-a)
> set cubes_b = 3     # buffer layers   (-b)
>
> # Remesh: use refMesh.{mg.c3d,c3d.Info} to guide cell density of initial mesh
> # (cubes -remesh option). If turned on, we recommend increasing mg_init
> # (initial number of multigrid levels, see flag below) to 3 or 4. Note that the
> # cubes_a flag is automatically set to 20 if it is less than 20.  0=no, 1=yes,
> # default=0
> set use_remesh = 0
>
> # Internal mesh (cubes): 0=no, 1=yes, default=0
> set Internal = 0
>
> # ----- Run control -----
>
> # Do error analysis on finest mesh? 0=no, 1=yes, default=0
> set error_on_finest = 0
>
> # Exit on reaching the finest mesh, do not flow solve there. This is useful if
> # you wish to run a different solver (e.g. the MPI version) on the finest mesh.
> # The final adapt?? directory holds the final mesh, all the input files, as
> # well as a FLOWCART file that contains the appropriate command line.
> set skip_finest = 0    # 0=no, 1=yes, default=0
>
> # Maximum level of refinement allowed in the mesh.  This controls the size of
> # the smallest cell in the mesh. Default value is 21, which is the maximum
> # supported by Cart3D. The run terminates when either max_ref_level or
> # n_adapt_cycles is satisfied, whichever is smaller.
> set max_ref_level = 21
>
> # Set extra refinement levels for the final mesh. This allows you to adapt the
> # mesh multiple times with the same error map in the last adapt cycle, thereby
> # bypassing the flow, adjoint, and error estimation steps. Use with caution:
> # the mesh should be fine enough so that the error estimate is decreasing -
> # preferably the solution should be in the Richardson region. This helps
> # circumvent the memory limitations of the error estimation code. Default value
> # is 0 and maximum allowed value is 3.
> set final_mesh_xref = 0
>
> # ---------------------------------------------------
> # EXPERT user options: flags below are rarely changed
> # ---------------------------------------------------
>
> # Cut-cell gradients: 0=best robustness (default), 1=best accuracy
> # If mesh2d=1, then we set tm=1 automatically
> set tm = 0
>
> # CFL number: usually ~1.1 but with power may be lower, i.e. 0.8
> set cfl    = 1.0
> # Minimum CFL number, used in case of convergence problems with flowCart
> set cflmin = 0.2
>
> # Adaptation error tolerance. Run terminates if the error estimate falls
> # below this value. At least 2 cycles will be run before terminating based on
> # this tolerance to avoid triggering based on an inappropriate initial mesh.
> set etol = 0.0000001
>
> # Grid sequencing (-gs) or multigrid (-mg) (-mg default)
> # flowCart
> set mg_gs_fc = '-mg'
> # adjointCart
> set mg_gs_ad = '-mg'
>
> # Full multigrid: default is to use full multigrid, except in cases with power
> # boundary conditions (automatic with warm starts)
> set fmg = 1    # 0=no, 1=yes, default=1
>
> # Polynomial multigrid, helps certain tough cases converge deeper
> set pmg = 0    # 0=no, 1=yes, default=0
>
> # Buffer limiter: improves stability for flows with strong off-body shocks
> set buffLim = 0    # 0=no, 1=yes, default=0
>
> # Number of multigrid levels for initial mesh (default 2, ramps up to mg_fc/ad)
> set mg_init = 2
>
> # Set names of executables (must be in path)
> set flowCart        = flowCart
> set xsensit         = xsensit
> set adjointCart     = adjointCart
> set adjointErrorEst = adjointErrorEst_quad
>
> # MPI prefix: uncomment next line and also remember to set the correct flowCart
> # executable above. If not running MPI, comment out next line.
> # set mpi_prefix = 'mpiexec -n 16'
>
> # Flow solver warm-starts: 0=no, 1=yes, default=1
> set use_warm_starts = 1
>
> # Subcell resolution: 0=no, 1=yes, default=0
> set subcell = 0
>
> # Run adjoint solver in 1st-order mode. In hard cases where aero.csh
> # consistently falls back on 1st-order mode after trying more accurate
> # settings, this setting will short-circuit the process and go directly to
> # 1st-order adjoints. The flow solution remains 2nd-order. The adjoints should
> # still provide a consistent set of error estimates, which is probably safe for
> # _relative_ errors and adaptive meshing. However, use caution: the error
> # estimates may be inaccurate with respect to a 2nd-order run.
> # default 0=auto, 1=force 1st order
> set adj_first_order = 0
>
> # In 3D cases with tm=1, error estimation is still done with tm=0 for
> # robustness. This flag forces aero.csh to use tm=1 in error estimation. This
> # is recommended only for simple (academic) cases that converge well. In
> # general, the default (0) setting is _strongly_ recommended.
> set err_TM1 = 0    # default 0=off, 1=force tm 1 for error estimation
>
> # In 3D cases with tm=1, adjoint solutions are still done with tm=0 for
> # robustness. This flag forces aero.csh to use tm=1 for the adjoint solves. In
> # general, the default (0) setting is _strongly_ recommended. Note that
> # adjoint convergence is monitored, so if divergence occurs then tm=0 is set
> # during runtime.
> set adj_TM1 = 0    # default 0=off, 1=force tm 1 for adjoint solutions
>
> # When running optimization with adaptive meshing, this flag allows different
> # levels of adjoint convergence for the mesh adaptation functional vs. the
> # design functionals. The iterations for the adjoint solutions used in gradient
> # computations are it_ad + delta_it_ad. The main idea is to use fewer
> # iterations when building the mesh (for speed) and go for deeper converge in
> # the gradient adjoints (for better accuracy).  Default value is 0.
> set delta_it_ad = 0
>
> # Keep final error map in EMBED/Restart.XX.file, useful for cubes -remesh
> set keep_error_maps = 0    # default 0=no, 1=yes
>
> # Refine all cells: useful for uniform mesh refinement studies. This overrides
> # the error map and forces adapt to refine all cells. The adjoint correction
> # term and error estimate are reported.
> set refine_all_cells = 0    # default 0=no, 1=yes
>
> # adapt buffers (default 1)
> set buf = 1
>
> # Set the number of multigrid levels when aero.csh drops down to pMG due to
> # convergence problems. Default value is 2, which means no geometric
> # multigrid. In subsonic cases, 3 multigrid levels may be better. Note that
> # this flag has no effect on the pmg multigrid levels when the pmg flag is
> # selected above.  It influences only the automatic run control of aero.csh.
> set mg_pmg_auto = 2
>
> # Fine tuning of mesh growth when performing extra refinements on the final
> # mesh, i.e. when $final_mesh_xref>0 and $mesh_growth are being used.
> # The mesh growth for each extra refinement is given by:
> # ($mesh_growth-1)*$xref_fraction+1
> # The main idea is that as extra refinement cycles are performed, the
> # adaptation focuses on only the highest error cells. This is where the error
> # map is most accurate and most adaptation is required. Each value should be
> # between 0.2 and 1, and at most three extra refinements are allowed.
> set xref_fraction = ( 1.0 1.0 0.8 )
>
> # Safety factor used in aero_getResults.pl to terminate the run if the error
> # indicator value increases by more than this factor in successive
> # cycles. Default value is 8.
> set error_safety_factor = 8
>
> # Adaptation restart: Alternative to command line argument 'restart'
> set adapt_restart = 0
>
> # Adaptation jumpstart from existing mesh: Alternative to command line
> # argument jumpstart
> set adapt_jumpstart = 0
>
> # Write Tecplot output files in binary format
> set binaryIO = 1    # default 1=yes, 0=no
>
> # Verbose mode for executables [flowCart/xsensit/adjointCart/adjointErrorEst]
> # 0=no, 1=yes, default=0
> set verb = 0
>
> # Adaptation threshold array: To set ath manually, unset mesh_growth
> # and set the ath array (uncomment following two lines):
> # cycle      0  1 2 3 4 5 6 7 8 9 10
>
> #set ath = ( 32 16 8 4 2 1 1 1 1 1  1 )
> #unset mesh_growth
>
> # Output cell-wise errors, useful for making histograms: 0=off, 1=yes,
> # default=0
> set histo = 0
>
> # ---------------------------------------------
> # STOP: no user specified parameters below here
> # ---------------------------------------------
>
> .
> .
> .
>
> echo 'Done aero.csh'
> exit 0
>
> ERROR:
>   exit 1
> ```

[(top)](#page-adjoint-aero-csh-html--top)

[(adjoint docs)](#page-adjoint-index-html)

---

<a id="page-adjoint-doc-adj-functionals-html"></a>

## Specifying Functionals

*Original page: [adjoint/doc_adj_functionals.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/adjoint/doc_adj_functionals.html)*

---

#### Specifying Functionals for Adjoint-Based Mesh Refinement

---

**Output functionals** are quantities that define
the primary purpose of your simulation. For example, you may like
to estimate lift and drag as accurately as possible for a given
configuration and flow conditions. Here's how you specify output
functionals when using adjoint-based mesh refinement in
Cart3D.

- [Quick Start: What functional should
  I use?](#page-adjoint-doc-adj-functionals-html--adj-fun-uni)
- [Prerequisites](#page-adjoint-doc-adj-functionals-html--adj-fun-prereq)
- [Forces](#page-adjoint-doc-adj-functionals-html--adj-fun-forces)
- [Moments](#page-adjoint-doc-adj-functionals-html--adj-fun-moments)
- [*L/D*](#page-adjoint-doc-adj-functionals-html--adj-fun-ld)
- [Point and Line Sensors](#page-adjoint-doc-adj-functionals-html--adj-fun-sensors)
- [Mass Flow Rate](#page-adjoint-doc-adj-functionals-html--adj-fun-mfr)
- [Multiple Functionals](#page-adjoint-doc-adj-functionals-html--adj-fun-multi)
- [Farfield
  Functionals ( *NEW* )](#page-adjoint-doc-adj-functionals-html--adj-fun-ff)
- [Output Convergence
  History](#page-adjoint-doc-adj-functionals-html--adj-fun-output)

---

### Quick Start: What functional should I use?

You may be interested in many outputs from your simulations. For
starters, you may need aerodynamic forces and moments of the entire
configuration. Afterwards, you may like to consider aerodynamic performance
of specific components in your configuration. While it is OK to have all
these outputs drive the adaptation, it may not be necessary. In practice, we
found that using the lift and drag functionals (or axial and normal force)
with different weights works well for cases that have no side-slip angle. Add
the lateral force coefficient when your cases involve side-slip. Loosely
speaking, this is sufficient because all our functionals are functions of
pressure. The outputs are added to *flowCart*'s [*input.cntl*](#page-input-cntl-html)
file in a section called [*$\_\_Design\_Info*](#page-input-cntl-html--design). Here's an
example of our favorite "universal" functional:

> ```
> $__Force_Moment_Processing:
> Model_X_axis -Xb
> Model_Y_axis -Zb
> Model_Z_axis -Yb
>
> Reference_Area 1.0 all
>
> Force entire
>
> $__Design_Info:
> # Force Format:
> #
> #          Name     Force   Frame    J      N    Target  Weight  Bound  GMP_Comp
> #        (String)  (0,1,2)  (0,1)  (0,1)  (int)  (dble)  (dble)   (0)
> # ------------------------------------------------------------------------------
> optForce    CD        0       0      0      1      0.      1.      0     entire
> optForce    CL        2       0      0      1      0.      0.2     0     entire
> ```

If you simply copy-and-paste the above into your [*input.cntl*](#page-input-cntl-html)
file, then you are ready to run. For more advanced users:
whenever possible try to take advantage of your GMP configuration
hierarchy. For example, your *entire* configuration may
involve wind tunnel hardware and other components that are not
part of the surfaces that define your outputs. In other words,
use the keyword *entire* sparingly.

Caution: the setup above is for geometries is in the standard Cart3D [orientation](#page-clic-html-clic-doc-html--frames) (*Y-is-up* and *Z-is-spanwise*).

Keep reading if you need the details, otherwise stop.

---

### Prerequisites

There are two ways to specify output functionals. The most
straightforward approach is to append a [*$\_\_Design\_Info*](#page-input-cntl-html--design) section to the
bottom of *flowCart*'s [*input.cntl*](#page-input-cntl-html)
file. A more general approach is to specify the outputs via a
[*Functionals.xml*](#page-adjoint-doc-adj-functionals-html--adj-fun-multi) file. This
approach allows error estimation for multiple outputs, but it
requires the solution of multiple adjoints. Both approaches use
the same functional syntax described below.

The functionals *(J)* use one of two general forms that
are specified by indices 0 and 1:

| Index | Funtional Form |
| --- | --- |
| 0 | J = W(P - T)^N |
| 1 | J = W(1 - P / T)^N |

where *P* is the performance parameter, *T* is
its target value, *W* is a weight and *N* is an
exponent to allow linear and higher-order functionals. Why is
there a target value? This is because these functionals can be
also used as objective functions in shape optimization problems.
In shape optimization, you may have design goals in mind that you
specify as target values, e.g., a desired value of drag. This
should not be misinterpreted as a requirement to specify the mesh
converged value of drag for the adaptation. The adjoint-based
mesh refinement method is not concerned with the value of the
functional, instead it focuses on minimizing the discretization
errors in the functional. In short, for common functionals you'll
most likely specify a target value of zero. For example, if you
wish to set drag as the output functional to drive the mesh
refinement, select equation (0) and set *P* to drag,
*T* to zero, *W* to one and *N* to one.

As performance parameters, you may select body forces and
moments, *L/D*, and point and line field sensors. For body
forces and moments, as well as *L/D*, you select either
the aerodynamic or aircraft (body) [reference frame](#page-clic-html-clic-doc-html--frames). Use the following indices to select the
desired frame:

| Index | Reference Frame |
| --- | --- |
| 0 | Aerodynamic |
| 1 | Aircraft (body) |

Prerequisites to the use of [*$\_\_Design\_Info*](#page-input-cntl-html--design) are the
definitions of the [*$\_\_Force\_Moment\_Processing*](#page-input-cntl-html--force-moment) and
[*$\_\_Post\_Processing*](#page-input-cntl-html--postprocess) sections. Only forces and moments
declared in the [*$\_\_Force\_Moment\_Processing*](#page-input-cntl-html--force-moment) section can be
used to define output functionals. Similarly, point and line
sensors must be first declared in the [*$\_\_Post\_Processing*](#page-input-cntl-html--postprocess) section.

### Forces

The keyword *optForce* is used to denote force
functionals. Use the following indices to select one of the three
force coefficients:

| Index | Force Coefficient |
| --- | --- |
| 0 | Drag coefficient (CD) in aerodynamic frame  Axial force coefficient (CA) in body frame |
| 1 | Lateral force coefficient (Cy) in aerodynamic frame  Side force coefficient (CY) in body frame |
| 2 | Lift coefficient (CL) in aerodynamic frame  Normal force coefficient (CN) in body frame |

Here is an example of choosing the drag coefficient as an
output functional for the component *entire*:

> ```
> $__Force_Moment_Processing:
> Model_X_axis -Xb
> Model_Y_axis -Zb
> Model_Z_axis -Yb
>
> Reference_Area 1.0 all
>
> Force entire
>
> $__Design_Info:
> # Force Format:
> #
> #          Name     Force   Frame    J      N    Target  Weight  Bound  GMP_Comp
> #        (String)  (0,1,2)  (0,1)  (0,1)  (int)  (dble)  (dble)   (0)
> # ------------------------------------------------------------------------------
> optForce    CD        0       0      0      1      0.      1.      0     entire
> ```

where *Name* can be any string and *Bound* is
always set to zero. The key statement in the [*$\_\_Force\_Moment\_Processing*](#page-input-cntl-html--force-moment) section is
*Force entire*. This allows us to bind the drag
coefficient to the component *entire*. You may choose any
component defined in your GMP [*Config.xml*](#page-config-xml-html)
file. If instead you are interested in the lift coefficient, then
you need to change the *Force* index to 2 and optionally
change the *Name* to CL. A sample case that uses the drag
coefficient is the $CART3D/cases/samples\_adapt/wing
example.

### Moments

The keyword *optMoment\_Point* is used to denote
point-moment functionals. Use the following indices to select one
of the three moments:

| Index | Moment Coefficient |
| --- | --- |
| 0 | CM\_ax in aerodynamic frame  Roll (Cl) in body frame |
| 1 | CM\_ay in aerodynamic frame  Pitch (Cm) in body frame |
| 2 | CM\_az in aerodynamic frame  Yaw (Cn) in body frame Lift |

Here is an example of choosing the pitching moment for the
component *wing*:

> ```
> $__Force_Moment_Processing:
> Model_X_axis  -Xb
> Model_Y_axis  -Zb
> Model_Z_axis  -Yb
> Reference_Area    1.  all
> Reference_Length  1.  all
>
> Moment_Point 0.25 0.0 0.0 wing
>
> $__Design_Info:
> # Point Moment Format:
> # Index: 0-based index relative to moments defined above per component
> #
> #                Name   Index  Moment  Frame   J     N   Target  Weight  Bound  GMP_Comp
> #              (String) (int) (0,1,2)  (0,1) (0,1) (int) (dble)  (dble)  (0)
> # --------------------------------------------------------------------------------------
> optMoment_Point  MP0      0      1       1     0     1     0.      1.     0     wing
> ```

where most fields have already been defined in the [Forces](#page-adjoint-doc-adj-functionals-html--adj-fun-forces) section. A new field is
*Index*. This field identifies which *Moment\_Point*
the functional binds to. For example, you may have several
moments listed for the component *wing* in the [*$\_\_Force\_Moment\_Processing*](#page-input-cntl-html--force-moment) section, each
with a different moment center. The *Index* field is a
0-based index that numbers the moments in the order they are
listed for each component.

### *L/D*

The keyword *optLD* is used to denote *L/D*.
Depending on the chosen reference frame, the functional will use
either CL and CD, or CN and
CA. The performance parameter *P* in
functionals of type [0](#page-adjoint-doc-adj-functionals-html--form-0) or [1](#page-adjoint-doc-adj-functionals-html--form-1) has the following form:

*SIGN(CL )\*ABS(CL )^A*
/ *(CD+Bias)*

where *Bias* is a user specified constant that may be
used to account for viscous drag. This formulation allows you to
consider endurance factor functionals in the adaptation. Here is
an example of choosing *L/D* for the component
*airplane*:

> ```
> $__Force_Moment_Processing:
> Model_X_axis -Xb
> Model_Y_axis -Zb
> Model_Z_axis -Yb
>
> Reference_Area 1.0 all
>
> Force airplane
>
> $__Design_Info:
> # L/D -> SIGN(CL)*ABS(CL)^A/(CD+Bias) in Aero Frame
> #     -> SIGN(CN)*ABS(CN)^A/(CA+Bias) in Body Frame
> # Format:
> #
> #      Name   Frame   J     N     A     Bias  Target  Weight  Bound  GMP_Comp
> #    (String) (0,1) (0,1) (int) (dble) (dble) (dble)  (dble)   (0)
> # ---------------------------------------------------------------------------
> optLD  LoD      0     0     1     1.    0.001    0.     1.      0    airplane
> ```

See [Forces](#page-adjoint-doc-adj-functionals-html--adj-fun-forces) for field details. A
sample case that shows the specification of the L/D functional is
the $CART3D/cases/samples\_adapt/naca0012
example.

### Point, Line and Equivalent Area Sensors

In this section, we choose a pressure sensor at an arbitrary
point or along an arbitrary line in the flowfield as the output
functional. For line sensors, you can also request an equivalent
area functional. The sensor may lie on the surface of the
geometry, however, the algorithm does not check if the sensor is
outside the computational domain, i.e., inside the geometry, so
be careful. To use a sensor, first define its location in the
[*$\_\_Post\_Processing*](#page-input-cntl-html--postprocess) section and
then specify the functional form in the [*$\_\_Design\_Info*](#page-input-cntl-html--design) section. Here
is an example:

> ```
> $__Post_Processing:
> #
> # Line Sensor
> #           Name   Origin X Y Z   Destination X Y Z
> lineSensor  LINE1  0. 1.02 0.23   3.1 1.02 0.23
> #
> # Equivalent Area Sensor
> #           Name   Origin X Y Z   Destination X Y Z  Radius
> eaSensor    EA1    0. 1.02 0.23   3.1 1.02 0.23      1.5
> #
> # Point Sensor
> #           Name   X   Y  Z
> pointSensor PT1    10. 0. 0.
>
> $__Design_Info:
> # Field Sensors
> #          Name     J     N   Target  Weight  Bound
> #        (String) (0,1) (int) (dble)  (dble)   (0)
> # -------------------------------------------------
> optSensor  PT1      0     2      0.    1.0      0
> optSensor  LINE1    0     2      0.    1.0      0
> optSensor  EA1      0     2      0.    1.0      0
> ```

Note that the *Name* field identifies which sensors
should be considered as output functionals for the adaptation.
You may specify many sensors in the [*$\_\_Post\_Processing*](#page-input-cntl-html--postprocess) section and use their
*names* to pick among them the functionals that drive the
adaptation. A sample case that shows how to specify a line sensor
is the $CART3D/cases/samples\_adapt/cone\_cylinder
pressure-signature example.

### Mass Flow Rate

The keyword *optMassFlow* is used to set mass flow rate through a
specified component as the output of interest. First, set up [*$\_\_Boundary\_Conditions*](#page-input-cntl-html--bcs) with the desired inflow and outflow boundary
conditions. Second, make sure the GMP component names in [*Config.xml*](#page-config-xml-html)
file match the numerical ID's of the [*$\_\_Boundary\_Conditions*](#page-input-cntl-html--bcs). Last, use the keyword *optMassFlow*
in the [*$\_\_Design\_Info*](#page-input-cntl-html--design) section. Here is an example:

> ```
> $__Boundary_Conditions:
>
> # Inlet boundary condition
> # Name           Comp  U_n/c_inf
> InletVelocityBC  2     0.5
>
> # Nozzle boundary condition
> # Name      Comp  m_dot  T_t/T_t_inf
> PowerMFRBC  3     0.07   1.01
>
> $__Design_Info:
>
> # Mass Flow Rate output
> #            Name     J     N   Target  Weight  Bound  GMP
> #           (String) (0,1) (int) (dble)  (dble)   (0)
> # ------------------------------------------------------------
> optMassFlow  inlet    0     1      0.    1.0      0    fanFace
> optMassFlow  nozzle   0     1      0.    1.0      0    fanExit
> ```

Note that the *Name* field can be any string and is used to label
the output functional.

### How do I specify multiple functionals?

If using the [*$\_\_Design\_Info*](#page-input-cntl-html--design) section in
*input.cntl*, then list all the functionals (with
appropriate declarations in the [*$\_\_Post\_Processing*](#page-input-cntl-html--postprocess) and [*$\_\_Force\_Moment\_Processing*](#page-input-cntl-html--force-moment) sections) in this
section, one-per-line. The adaptation functional becomes the
**sum** of the specified functionals - the sum is
implied by the list. Use the *Weight* field to specify
relative importance of the functionals. The [Point and Line Sensors](#page-adjoint-doc-adj-functionals-html--adj-fun-sensors) section (in the
gray box above) shows an example. Remember that the error
estimation and adaptation targets this composite functional and
not its individual components.

***Functionals.xml***

If error estimation in multiple outputs is desired, then use a
*Functionals.xml* file. For example:

> ```
> <?xml version="1.0" encoding="ISO-8859-1"?>
> <Functionals>
>   <!-- Adapt to 0.1*CL + CD -->
>   <AeroFun ID="LplusD" Options="Adapt">
>     # Force Codes: CA=0 CY=1 CN=2 in Aircraft (Body) Frame
>     #         Name    Force   Frame    J      N    Target   Weight  Bound   GMP
>     #        (String) (0,1,2) (0,1) (0,1,2) (int)  (dble)   (dble) (-1,0,1)
>     optForce   CL        2      0      0      1      0.      0.1      0     entire
>     optForce   CD        0      0      0      1      0.      1.0      0     entire
>   </AeroFun>
>
>   <!-- Monitor error in CL, up to and including penultimate mesh -->
>   <AeroFun ID="Lift" Options="Error">
>     # Force Codes: CD=0 Cy=1 CL=2 in Aerodynamic Frame
>     #         Name    Force   Frame    J      N    Target   Weight  Bound   GMP
>     #        (String) (0,1,2) (0,1) (0,1,2) (int)  (dble)   (dble) (-1,0,1)
>     optForce   CL        2      0      0      1      0.      1.       0     entire
>   </AeroFun>
>
>   <!-- Monitor error in CD, up to and including penultimate mesh -->
>   <AeroFun ID="Drag" Options="Error">
>     # Force Codes: CA=0 CY=1 CN=2 in Aircraft (Body) Frame
>     #         Name    Force   Frame    J      N    Target   Weight  Bound   GMP
>     #        (String) (0,1,2) (0,1) (0,1,2) (int)  (dble)   (dble) (-1,0,1)
>     optForce   CD        0      0      0      1      0.      1.       0     entire
>   </AeroFun>
>
>   <!-- Monitor mesh convergence of pitch, no adjoint or error estimate -->
>   <AeroFun ID="Pitch">
>     #              Name   Index Moment  Frame  J     N    Target Weight Bound GMP
>     #            (String) (int) (0,1,2) (0,1) (0,1) (int) (dble) (dble) (0)
>     optMoment_Point  MP0   0      1      1     0     1     0.      1.    0   entire
>   </AeroFun>
> </Functionals>
> ```

Each *AeroFun* element is equivalent
to a [*$\_\_Design\_Info*](#page-input-cntl-html--design) section. The
value of the *Options* attribute specifies which
functionals require error estimation and of these which one
should be used to drive the adaptation. The keywords are as
follows:

| Keyword | Meaning |
| --- | --- |
| Adapt | Functional that drives the adaptation. Implies error computation up to and including the penultimate mesh. Only one AeroFun element can be marked with Adapt. |
| Error | Error computation up to and including the penultimate mesh. No influence on mesh adaptation, but important for assessing if the mesh is good enough for the functional. Generates both *fun\_con.dat* and *results.dat* files. |
| ErrorAll | Similar to Error, but error computation is done on all meshes, including the finest mesh. Generates both *fun\_con.dat* and *results.dat* files. |

*Options*="*Adapt*, *ErrorAll*" is valid.
If Options are omitted, then functional convergence is reported
(*fun\_con\_ID.dat*), but errors are not computed.

### Farfield Functionals

For sonic-boom analysis problems where the farfield propagation
is performed
using [*sBOOM (ver. 2.9 or newer)*](https://software.nasa.gov/software/LAR-20310-1), noise metrics can be specified as
output functionals in coupled *Cart3D-sBOOM* simulations. The mesh
adaptation targets the farfield outputs, for example A-SEL, and the
computed error estimate reflects the level of discretization error
in these outputs. Technical details of the procedure are provided in
[AIAA-2022-4085](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2022-4085_c3d_sBOOM.pdf). The farfield outputs are specified
via the *Functionals.xml* file:

> ```
> <?xml version="1.0" encoding="ISO-8859-1"?>
> <Functionals>
>   <!-- Adapt to sBOOM noise metrics -->
>   <!-- Functional is specified in the sBOOM input file -->
>   <!-- Line-Sensor-Name is defined in input.cntl -->
>   <AeroFun ID="Any-Unique-String" Options="Adapt">
>     #         Line-Sensor-Name   sBOOM-input-file   Weight
>     optSBOOM      LINE1            presb.input       1.0
>   </AeroFun>
> </Functionals>
> ```

The syntax supports all features of *Functionals.xml* files,
such as the ability to "mix-and-match" functionals:

> ```
> <?xml version="1.0" encoding="ISO-8859-1"?>
> <Functionals>
>   <AeroFun ID="noise_and_aero" Options="Adapt">
>     #         Name    Force   Frame    J      N    Target   Weight  Bound  GMP
>     #        (String) (0,1,2) (0,1) (0,1,2) (int)  (dble)   (dble) (-1,0,1)
>     optForce   CL        2      0      0      1      0.      0.1      0    entire
>     optForce   CD        0      0      0      1      0.      1.0      0    entire
>     #         Line-Sensor-Name   sBOOM-input-file   Weight
>     optSBOOM      azi_0deg       presb_j4_0.input    5.0
>     optSBOOM      azi_10deg      presb_j4_10.input   5.0
>   </AeroFun>
> </Functionals>
> ```

and error estimation for multiple outputs:

> ```
> <?xml version="1.0" encoding="ISO-8859-1"?>
> <Functionals>
> <!-- Each output must be assigned to a different line sensor -->
> <!-- The line sensor can be colocated -->
>   <AeroFun ID="j4" Options="Adapt">
>     optSBOOM   line1      presb_j4.input     1.0
>   </AeroFun>
>   <AeroFun ID="j8" Options="Error">
>     optSBOOM   line2      presb_j8.input     1.0
>   </AeroFun>
>   <AeroFun ID="j5" Options="Error">
>     optSBOOM   line3      presb_j5.input     1.0
>   </AeroFun>
> </Functionals>
> ```

### Output Convergence History

When a [*$\_\_Design\_Info*](#page-input-cntl-html--design) section is
specified or a *Functionals.xml* file is present, the
convergence history of the functionals is recorded in an output
file called *functional.dat*. This file is a sibling of
the *forces.dat* and *history.dat* files, and shows
the value of the functionals at each iteration of
*flowCart*. In addition, there is output to
*flowCart*'s *stdout* stream that summarizes the
specified functionals.

[(top)](#page-adjoint-doc-adj-functionals-html--top)
[(adjoint docs)](#page-adjoint-index-html)

---

*[Marian Nemec](mailto:marian.nemec@nasa.gov), last update
November 2024*

---

<a id="page-adjoint-doc-adj-adapt-html"></a>

## Cart3D AERO

*Original page: [adjoint/doc_adj_adapt.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/adjoint/doc_adj_adapt.html)*

---

#### Adaptation Run Control: *aero.csh*

---

**Content**

- [Get Started](#page-adjoint-doc-adj-adapt-html--adj-adapt-start)
- - [Input Files](#page-adjoint-doc-adj-adapt-html--adj-adapt-inputs)
  - [*Aero.csh*
    Parameters](#page-adjoint-doc-adj-adapt-html--adj-adapt-params)
  - [Upgrading Older
    *aero.csh* Scripts](#page-adjoint-doc-adj-adapt-html--adj-adapt-upgrade)
  - [Usage](#page-adjoint-doc-adj-adapt-html--adj-adapt-usage)
  - [Output](#page-adjoint-doc-adj-adapt-html--adj-adapt-output)
- [Advanced Run
  Control](#page-adjoint-doc-adj-adapt-html--adj-adapt-control)

---

#### Get Started

### Input Files

The default version of the [*aero.csh*](#page-adjoint-aero-csh-html) script is installed in
$CART3D/bin. You should copy this template script into
your run directory. Alternatively, the scripts in
$CART3D/cases/samples\_adapt may be even better
templates. For example, if your case is 2D then a good starting
point is the
$CART3D/cases/samples\_adapt/naca0012/aero.csh
script.

The [*aero.csh*](#page-adjoint-aero-csh-html) script
requires three Cart3D input files:

- wetted surface [triangulation file](#page-cart3dtriangulations-html), *Components.i.tri* (must be
  watertight)
- *cubes*' input file [*input.c3d*](#page-input-c3d-html)
- *flowCart*'s input file [*input.cntl*](#page-input-cntl-html) augmented with a [*$\_\_Design\_Info*](#page-input-cntl-html--design) section (see
  [output functionals)](#page-adjoint-doc-adj-functionals-html)

Optionally, you may also include a [GMP](https://software.nasa.gov/software/ARC-15193-1)
configuration file [*Config.xml*](#page-config-xml-html). The surface triangulation file must be
called *Components.i.tri*. In practice, the surface
triangulation can be called anything - just create a symbolic
link to *Components.i.tri*. We recommend [autoInputs](#page-meshgeneration-html--autoinputs) to create *input.c3d*. You may also use the
[*preSpec.c3d.cntl*](#page-prespec-c3d-cntl-html) file to specify BBox and XLev
regions for cubes (this can be useful for cases with power
boundaries) and ABox regions for *adapt* (use cautiously,
not recommended in general).

### *Aero.csh* Parameters

The many user-adjustable parameters of [*aero.csh*](#page-adjoint-aero-csh-html) are divided into three
groups: [Basic](#page-adjoint-aero-csh-html--aero-basic), [Advanced](#page-adjoint-aero-csh-html--aero-advanced) and [Expert](#page-adjoint-aero-csh-html--aero-expert) sections. All options are
documented inline. Further documentation of options related to
cubes and flowCart is available [here](#page-flowcart-run-html).
To get started, the two most important flags are [mesh2d](#page-adjoint-aero-csh-html--aero-mesh2d) and [y\_is\_spanwise](#page-adjoint-aero-csh-html--aero-spanwise) in the [Basic](#page-adjoint-aero-csh-html--aero-basic) section. The default
settings assume a 3D case:

set mesh2d = 0

and that the geometry is in a *Y-is-up* and
*Z-is-spanwise* [orientation](#page-clic-html-clic-doc-html--frames):

set y\_is\_spanwise = 0

The default parameter values in $CART3D/bin/aero.csh
are conservative - we found them to work robustly in difficult
cases, e.g. cases with strong power boundary conditions. The
*aero.csh* scripts in $CART3D/cases/samples\_adapt
have settings more appropriate for typical cases, such as
transonic flow around transport configurations. These should give
better accuracy and faster turn-around. Frequently adjusted
parameters include:

- Number of refinement cycles [n\_adapt\_cycles](#page-adjoint-aero-csh-html--aero-cycles)
- Number of multigrid levels [mg\_fc](#page-adjoint-aero-csh-html--aero-mg-fc) and [mg\_ad](#page-adjoint-aero-csh-html--aero-mg-ad)

Important parameters that influence the efficiency and
accuracy of adaptive runs are the [mesh-growth](#page-adjoint-aero-csh-html--aero-mesh-growth) and [adapt-propagate](#page-adjoint-aero-csh-html--aero-apc) arrays, and the [auto\_growth](#page-adjoint-aero-csh-html--aero-auto) option. These parameters are described
in the [Advanced Run Control](#page-adjoint-doc-adj-adapt-html--adj-adapt-control)
section below, their defaults should be fine for getting
started.

### Upgrading Older *aero.csh* Scripts

Traditionally, it has been straightforward to migrate from
older versions of *aero.csh* to the latest release.
However, the input parameters have been significantly reorganized
and modified in the 1.5 release. In an effort to make the porting
easier, a utility script aero\_upgrade.py is provided to
automatically upgrade *aero.csh*. The script uses the
master copy in $CART3D/bin/aero.csh as a template to
port the user settings from the old version. Execute the script
in the directory containing the old *aero.csh*:

% $CART3D/bin/aero\_upgrade.py

The script leaves a backup of the original in
aero.csh.orig. Always check the new script against the
backup.

### Usage

To run the adaptation, simply execute [*aero.csh*](#page-adjoint-aero-csh-html):

% ./aero.csh

The script returns 0 on success and 1 if an error occurs.
(*Novice Linux users: the ./ means
to execute the aero.csh that is in the current
directory, not the one in $CART3D/bin.*)

The script accepts several command line arguments. To see a
usage statement use the *help* or *-h* options:

% ./aero.csh help

To restart an adaptation run, increase [n\_adapt\_cycles](#page-adjoint-aero-csh-html--aero-cycles) and use the
keyword *restart* as follows:

% ./aero.csh restart

The *restart* flag can be also used to recover from a
crashed run, e.g., machine crashes. To restart from a particular
adaptXX directory, delete all adapt??
directories after adaptXX and restart.

To start an adaptation run from an existing mesh, place
*Mesh.mg.c3d* and *Mesh.c3d.Info* files into the
run directory and use the keyword *jumpstart* as
follows:

% ./aero.csh jumpstart

Jumpstarting is useful for restarting an adaptation form some
previous adapt?? directory. Simply copy the desired mesh
files to a new run directory and continue adapting. When [*aero.csh*](#page-adjoint-aero-csh-html) executes, the mesh files
are moved to the adapt00 directory.

Advanced feature: at the start of each adaptation cycle, [*aero.csh*](#page-adjoint-aero-csh-html) checks for the presence of
*input.new.cntl* file. If this file is present, it will
replace the active flowCart *input.cntl* file. This feature
allows on-the-fly control of the flow solution as the adaptation
executes.

To stop an adaptation run after the computation of the next
error estimate, create a STOP file:

% touch STOP

To clean-up a run directory and start a new run on the same
case, remove the adapt?? directories:

% rm -rf adapt??

Setting environment variable debug\_verbose (% setenv
debug\_verbose) triggers more verbose output.

### Outputs

The file and directory tree of a typical adaptation run, with
[n\_adapt\_cycles](#page-adjoint-aero-csh-html--aero-cycles) = 2, is
shown below:

> [naca0012](#page-adjoint-doc-adj-adapt-html)
> |-- [ADAPT](#page-adjoint-doc-adj-adapt-html)
> | |-- [# Generation of adapted meshes](#page-adjoint-doc-adj-adapt-html)
> | `-- [adapt.\*.out](#page-adjoint-doc-adj-adapt-html)
> |-- [BEST@ -> adapt02](#page-adjoint-doc-adj-adapt-html)
> |-- [EMBED](#page-adjoint-doc-adj-adapt-html)
> | |-- [# Generation of embedded meshes](#page-adjoint-doc-adj-adapt-html)
> | |-- [Error\_in\_J](#page-adjoint-doc-adj-adapt-html)
> | | |-- [# Error estimation](#page-adjoint-doc-adj-adapt-html)
> | | |-- [adjointErrorEst.\*.out](#page-adjoint-doc-adj-adapt-html)
> | | `-- [cutPlanesErrEst.\*.plt](#page-adjoint-doc-adj-adapt-html)
> | `-- [adapt.\*.out](#page-adjoint-doc-adj-adapt-html)
> |-- [adapt00](#page-adjoint-doc-adj-adapt-html)
> | |-- [AD\_A\_J](#page-adjoint-doc-adj-adapt-html)
> | | |-- [# Adjoint solution](#page-adjoint-doc-adj-adapt-html)
> | | `-- [cart3d.out](#page-adjoint-doc-adj-adapt-html)
> | |-- [FLOW](#page-adjoint-doc-adj-adapt-html)
> | | |-- [# Flow solution](#page-adjoint-doc-adj-adapt-html)
> | | |-- [cart3d.out](#page-adjoint-doc-adj-adapt-html)
> | | |-- [entire.dat](#page-adjoint-doc-adj-adapt-html)
> | | |-- [forces.dat](#page-adjoint-doc-adj-adapt-html)
> | | |-- [functional.dat](#page-adjoint-doc-adj-adapt-html)
> | | |-- [history.dat](#page-adjoint-doc-adj-adapt-html)
> | | `-- [loadsCC.dat](#page-adjoint-doc-adj-adapt-html)
> | |-- [Mesh.c3d.Info](#page-adjoint-doc-adj-adapt-html)
> | |-- [Mesh.mg.c3d](#page-adjoint-doc-adj-adapt-html)
> | `-- [cart3d.out](#page-adjoint-doc-adj-adapt-html)
> |-- [adapt01](#page-adjoint-doc-adj-adapt-html)
> |-- [adapt02](#page-adjoint-doc-adj-adapt-html)
> |-- [aero.csh](#page-adjoint-doc-adj-adapt-html)\*
> |-- [all\_outputs.dat](#page-adjoint-doc-adj-adapt-html)
> |-- [Components.i.tri](#page-adjoint-doc-adj-adapt-html)
> |-- [fun\_con.dat](#page-adjoint-doc-adj-adapt-html)
> |-- [input.c3d](#page-adjoint-doc-adj-adapt-html)
> |-- [input.cntl](#page-adjoint-doc-adj-adapt-html)
> |-- [results.dat](#page-adjoint-doc-adj-adapt-html)
> `-- [user\_time.dat](#page-adjoint-doc-adj-adapt-html)

Red marks *input files*; not all
outputs are listed. Each adaptation cycle generates a new
adapt?? directory that holds the flow
and adjoint solutions of the current mesh in the subdirectories
FLOW and AD\_A\_J, respectively. The output
(*stdout* and *stderr*) of the various codes that
execute in these directories is directed to files called
*cart3d.out* in the FLOW and
AD\_A\_J subdirectories. This is a good
place to look if your run has crashed and you're trying to figure
out the problem. The script also maintains a link called
BEST.

#### The BEST Link

The BEST symbolic link always
points to an adapt directory that contains the best
solution obtained so far. Under normal circumstances, this link
points to the last adapt directory. Various conditions,
such as run termination due to a machine crash or queue time
expiration or solver divergence may cause an unexpected exit. The
script ensures that BEST always
points to a valid, best-so-far flow solution and is intended for
use with automatic results-harvesting scripts.

#### Functional Convergence

The script generates at least two summary files: [*fun\_con.dat*](#page-adjoint-fun-con-dat-html) and [*results.dat*](#page-adjoint-results-dat-html). The file
*fun\_con.dat* contains four columns: adaptation cycle,
number of cells in the mesh, functional value and average
functional value. The average functional value is useful for
cases where the functional does not reach a steady value but
instead becomes oscillatory. The **avg\_window**
variable in [*aero.csh*](#page-adjoint-aero-csh-html) can be
used to specify the number of flow solver iterations to be used
for functional averaging on each mesh.

The file [*results.dat*](#page-adjoint-results-dat-html)
contains nine columns:

1. Adaptation cycle
2. Number of cells in the mesh
3. Log of the number of cells (useful for formal convergence
   studies)
4. Functional value (should be same as in
   *fun\_con.dat*)
5. Corrected functional value
6. Adaptation error estimate
7. Adaptation error estimate normalized by etol
8. Value of etol
9. Functional error estimate

Consult [NASA TM-2014-218386](http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20150000864.pdf) for a precise definition of these terms.
The main idea is that as the mesh is refined, the error estimates
should decrease and eventually fall below **etol**.
This is one of the adaptation exit criteria. The run terminates
after the next flow solution. There are two benefits to delaying
the exit until after the flow solution. First, the approach is
conservative in terms of the error estimate value - the
functional on the next mesh should be well within the tolerance.
Second, we have done all the hard work of computing the mesh
refinement parameter so let's use it and compute the best
solution possible.

Here is an output example from
$CART3D/cases/samples\_adapt/wing. The top-left plot
shows the functional values (CD) as the mesh is
refined (column 2 versus 3 of [*fun\_con.dat*](#page-adjoint-fun-con-dat-html)) with error bars
representing the level of discretization error (column 9 of
[*results.dat*](#page-adjoint-results-dat-html)). Top-right
figure shows convergence of several error estimates: the red line
(squares) shows the reduction in discretization error after each
adaptation, i.e., the magnitude of the error bars from the left
plot, the black line (circles) is the magnitude of the adaptation
error indicator (column 6 in [*results.dat*](#page-adjoint-results-dat-html)), and the blue line
is the change in CD relative to the previous mesh. The
bottom plot shows the iterative convergence of CD for
all adaptation cycles. After each warm-start, the initial
transient quickly dissipates and the value of CD
levels out. This is a good check to make sure that the solver is
running a sufficient number of multigrid cycles on each
mesh.

We caution the user that the file-size and number of files
generated by *aero.csh* can be considerable, especially
when building databases with several thousand cases. This is done
to provide a full history and restart capability of the runs. We
provide an *aero\_archive.csh* script for archiving runs,
which significantly reduces storage requirements. Once executed,
the restart capability is lost, but all important output files
are kept. A possible implementation is to check the return code
of *aero.csh*, and if zero then call
*aero\_archive.csh*. Alternatively, this script can be called
from *aero.csh* by using the *archive* option:

% ./aero.csh archive

---

#### Advanced Run Control

### Mesh Growth

In each adapt cycle, the maximum number of cells added to the
mesh is controlled by the mesh-growth factor. There are two ways
to set the mesh growth:

1. Specify growth factors explicitly via the [mesh\_growth](#page-adjoint-aero-csh-html--aero-mesh-growth) array. The
   factor may range from 1.01 to 8, where specifying 8 means that
   you allow refinement of every cell in the mesh. The basic idea
   is to adhere to a strict cell budget, thereby fixing the cost
   of the adaptive run. The way this works is that the cells are
   ranked from highest to lowest error and a fraction of the
   worst-error cells is refined such that the target mesh-growth
   is satisfied.
2. Allow [aero.csh](#page-adjoint-aero-csh-html) to automatically determine
   growth factors by setting [auto\_growth=1](#page-adjoint-aero-csh-html--aero-auto).

Note that [*aero.csh*](#page-adjoint-aero-csh-html)
monitors [max\_cells2embed](#page-adjoint-aero-csh-html--aero-max) and
automatically adjusts the mesh growth so that this cell limit is never
exceeded. To request automatic growth rates only in specific cycles, set
the [mesh\_growth](#page-adjoint-aero-csh-html--aero-mesh-growth) to 0 (zero) in
those cycles and disable global auto-growth by setting [auto\_growth=0](#page-adjoint-aero-csh-html--aero-auto).

General tips:

- In the default [mesh\_growth](#page-adjoint-aero-csh-html--aero-mesh-growth) sequence, we
  start with smaller factors because the accuracy of flow and
  adjoint solutions is poor on coarse meshes. We go after cells
  with largest errors. The sequence is tuned to efficiently
  provide the "best" mesh possible for your cell budget.
- If the simulation contains regions of separated flow (poor
  solver convergence), you may try to keep mesh growth low to
  develop the mesh in the well-behaved regions first, and then
  increase it quickly to fill in the rest of the mesh.
- Cell-wise error distributions can be viewed by plotting the
  files *cutPlanesErrEst.\*.plt* found in the
  EMBED directory. The cell-wise errors are extracted
  for the same cut-planes as the flow solution, which are
  specified in the *$\_\_Post\_Processing* section of the [*input.cntl*](#page-input-cntl-html) file. The error cut-planes show the
  normalized cell errors - this means that we take the adjoint
  error estimate and divide it by the allowable error-per-cell.
  The allowable error-per-cell is simply etol divided by
  the number of cells in the mesh.
- Try the [auto\_growth](#page-adjoint-aero-csh-html--aero-auto)
  option, it generally works very well. In our production runs,
  we frequently specify a [mesh\_growth](#page-adjoint-aero-csh-html--aero-mesh-growth) of 0 in most
  adapt cycles, except in the final one or two cycles where we
  may specify a larger growth explicitly.

### A/P cycles

The [apc](#page-adjoint-aero-csh-html--aero-apc) array controls
the refinement level of the mesh for each adaptation. It is a
sequence of **a** and **p** characters
that represent the following:

|  |  |  |
| --- | --- | --- |
| **a** | Adaptation cycle | Allow the smallest cells to be refined by one level |
| **p** | Propagation cycle | Preserve the smallest cell size |

The purpose of a propagation cycle is to adjust
cell-refinement boundaries. This is especially important in
problems where the output's zone of dependence involves a large
portion of the volume mesh, e.g. sonic-boom problems. Figure
below shows the difference between an **a** and
**p** refinement:

Tips:

- In general, use **a** cycles.
- We use a single **p** cycle on the initial
  mesh to reduce the bias of this mesh on the solution.
- Pay attention to the refinement level of the volume mesh
  relative to the size of the triangles on the geometry. In
  regions of high surface curvature, each Cartesian cell should
  be supported by several triangles. Once the size of the
  Cartesian cells approaches the size of the triangles, the user
  should switch to **p** cycles to prevent
  over-refinement of the surface triangulation. This is the best
  mesh you can get for a given (fixed) surface
  triangulation.
- Finishing the adaptation with a **p** cycle
  usually works well. This strategy prevents over-refinement of
  the surface triangulation mentioned above and delivers smoother
  meshes.

### Extra Refinement Passes

The parameter [final\_mesh\_xref](#page-adjoint-aero-csh-html--aero-xref) (0/1/2) allows
extra refinement passes on the final mesh. When non-zero, the
final error map is extrapolated to the adapted mesh such that the
adaptation can be performed multiple times. This circumvents the
flow, adjoint and error estimation steps, which reduces memory
requirements and improves turn-around of the adaptive run. The
total mesh growth is the sum of the mesh growth on the last adapt
cycle plus the mesh growth in each *xref* iteration. This
can be fine tuned via the [xref\_fraction](#page-adjoint-aero-csh-html--aero-xref-frac) array. We take
the mesh growth specified for the final cycle and multiply it by
the fractions specified in [xref\_fraction](#page-adjoint-aero-csh-html--aero-xref-frac). The main idea
is to focus the refinement on the highest error cells. In other
words, refine regions only where the error map is most
accurate.

As the error map is no longer updated to reflect the flow on
the refined mesh, this is only appropriate for 1 or at most 2
refinement passes. The default value is 0 (no *xrefs*). In
general, we obtain excellent results with [final\_mesh\_xref=1](#page-adjoint-aero-csh-html--aero-xref). The figure below
shows a good example with [final\_mesh\_xref=2](#page-adjoint-aero-csh-html--aero-xref), where an error
map that is constructed on a mesh with 500k cells is used to
build a mesh with 22M cells.

[(top)](#page-adjoint-doc-adj-adapt-html--top)
[(adjoint docs)](#page-adjoint-index-html)

---

*[Marian Nemec](mailto:marian.nemec@nasa.gov), last update
November 2024*

---

<a id="page-history-dat-html"></a>

## history.dat

*Original page: [history_dat.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/history_dat.html)*

---

history.dat

---

### history.dat
### cmd: flowCart -mg 4 -N 50
-his -T -clic -v
#
### cmd: flowCart -mg 4 -N 50 -his -T -clic -v
### Compiled on Nov 24 2008, 08:35:45
#
### Surface Triangulation file: oneraM6.tri
#                 
Mesh file: Mesh.mg.c3d
#         Input control file:
input.cntl
#
### Mach No = 0.5400,     alpha = 3.060, beta = 0.0000
#     cfl = 1.40,     FluxFun =
0,   Limiter = 2,
#  nMGlev = 4,    MGcycleType = 2,  
MG\_nPre = 1, MG\_nPost = 1
#
#      RK  
0.06950         1   #
stageCoef GradEval
#      RK  
0.16020         0   #
stageCoef GradEval
#      RK  
0.28980         0   #
stageCoef GradEval
#      RK  
0.50600         0   #
stageCoef GradEval
#      RK  
1.00000         0   #
stageCoef GradEval
#
#   mgCycle CPUtime/proc maxResidual(rho) 
globalL1Residual(rho)
    
1        
0.012     
1.18392774e+03     1.28602717e-01
    
2        
0.019     
7.17125299e+02     1.10185422e-01
    
3        
0.032     
2.72056489e+02     8.54530100e-02
    
4        
0.048     
9.41534270e+01     7.38994709e-02
    
5        
0.055     
1.79641708e+02     8.00163953e-02
    
6        
0.073     
2.26845101e+02     8.15036862e-02
    
7        
0.080     
2.14703779e+02     7.74255547e-02
    
8        
0.092     
1.75544722e+02     7.27530849e-02
    
9        
0.108     
1.28679240e+02     6.76224769e-02
    10        
0.116     
8.44100892e+01     5.99799211e-02
    11        
0.130     
4.75758899e+01     5.01837004e-02
    12        
0.142     
3.33870866e+01     3.90737531e-02
    13        
0.152     
5.07119174e+01     3.03509915e-02
    14        
0.170     
5.58816474e+01     2.77056640e-02
    15        
0.176     
5.22684111e+01     2.53747385e-02
    16        
0.190     
4.34073051e+01     2.25264034e-02
    17        
0.206     
3.22528772e+01     1.90497224e-02
    18        
0.213     
2.09179665e+01     1.49666052e-02
    19        
0.292     
1.77680443e+03     6.55621403e-02
    20        
0.374     
8.23890807e+02     3.96019558e-02
    21        
0.451     
3.01295505e+02     3.03247838e-02
    22        
0.537     
1.17091367e+02     2.28809666e-02
    23        
0.611     
8.31273724e+01     1.66757705e-02
    24        
0.690     
7.35424888e+01     9.44572763e-03
    25        
0.773     
6.32429691e+01     5.72191073e-03
    26        
0.852     
4.29102608e+01     3.85586231e-03
    27        
0.946     
2.34426311e+01     2.53917360e-03
    28        
1.477     
2.90289021e+04     6.82956141e-02
    29        
1.976     
8.59877178e+03     2.94777547e-02
    30        
2.482     
4.18420434e+03     1.51778107e-02
    31        
2.971     
4.05465216e+03     9.96112792e-03
    32        
3.441     
3.43743282e+03     5.45610789e-03
    33        
3.922     
2.58752058e+03     3.62266874e-03
    34        
4.396     
1.58112089e+03     1.92112955e-03
    35        
4.870     
8.22441090e+02     9.94575257e-04
    36        
5.354     
3.39803695e+02     7.93107203e-04
    37       
12.775     
1.41929948e+05     1.46731436e-01
    38       
20.158     
2.19880846e+04     1.05015332e-01
    39       
27.535     
6.93373951e+04     7.39073400e-02
    40       
34.760     
9.57283992e+04     4.93420719e-02
    41       
41.340     
5.03065557e+04     3.44357469e-02
    42       
47.934     
3.64778354e+04     2.41561578e-02
    43       
54.635     
3.52976513e+04     1.74785205e-02
    44       
61.289     
2.75690380e+04     1.26766649e-02
    45       
67.957     
2.51628421e+04     9.87416701e-03
    46       
74.713     
2.64474765e+04     7.70489851e-03
    47       
82.197     
3.06277974e+04     6.10631908e-03
    48       
89.193     
3.25638002e+04     5.03401268e-03
    49       
96.001     
3.54124284e+04     4.08775284e-03
    50      
102.522     
3.05374264e+04     3.50980052e-03

---

<a id="page-forces-dat-html"></a>

## forces.dat

*Original page: [forces_dat.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/forces_dat.html)*

---

forces.dat

---

### forces.dat
### cmd: flowCart -mg 4 -N 50
-his -T -clic -v
#
### cmd: flowCart -mg 4 -N 50 -his -T -clic -v
### Compiled on Nov 24 2008, 08:35:45
#
### Surface Triangulation file: oneraM6.tri
#                 
Mesh file: Mesh.mg.c3d
#         Input control file:
input.cntl
#
### Mach No = 0.5400,     alpha = 3.060, beta = 0.0000
#     cfl = 1.40,     FluxFun =
0,   Limiter = 2,
#  nMGlev = 4,    MGcycleType = 2,  
MG\_nPre = 1, MG\_nPost = 1
#
#      RK  
0.06950         1   #
stageCoef GradEval
#      RK  
0.16020         0   #
stageCoef GradEval
#      RK  
0.28980         0   #
stageCoef GradEval
#      RK  
0.50600         0   #
stageCoef GradEval
#      RK  
1.00000         0   #
stageCoef GradEval
#
#  mgCycle
CPUtime/proc  Force[X]      
Force[Y]       Force[Z]

    
3        
0.444   2.22220874e-02  3.41840241e-02 
-6.20548018e-02

    
6        
0.870   2.11696997e-02  4.22919508e-02 
-5.99052902e-02

    
9        
1.314   1.55798774e-02  3.56286147e-02 
-5.58095641e-02

   
12        
1.684   1.14393696e-02  3.33706556e-02 
-5.33233914e-02

   
15        
2.029   9.67312568e-03  3.70731956e-02 
-5.27037118e-02

   
18        
2.437   9.45134979e-03  4.05594297e-02 
-5.30589957e-02

   
21        
2.778   9.73112694e-03  4.18531359e-02 
-5.35697038e-02

   
24        
3.276   1.00008656e-02  4.18119744e-02 
-5.38828619e-02

   
27        
3.671   1.01412282e-02  4.16258678e-02 
-5.39592881e-02

   
30        
4.924   1.06888452e-02  3.86344095e-02 
-5.48604380e-02

   
33        
7.793   6.90865263e-03  3.47993032e-02 
-5.21861806e-02

   
36       
10.650   5.91832515e-03  3.80871104e-02 
-5.12523093e-02

   
39       
13.433   5.97405001e-03  4.01704900e-02 
-5.11847599e-02

   
42       
32.571   5.35619037e-03  4.05962637e-02 
-5.11027805e-02

   
45       
51.088   3.39150640e-03  3.89910263e-02 
-4.98237295e-02

   
48       
69.921   3.05140762e-03  3.84195243e-02 
-4.95847408e-02

---

<a id="page-moments-dat-html"></a>

## moments.dat

*Original page: [moments_dat.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/moments_dat.html)*

---

moments.dat

---

### moments.dat
### cmd: flowCart -mg 4 -N 50
-his -T -clic -v
#
### cmd: flowCart -mg 4 -N 50 -his -T -clic -v
### Compiled on Nov 24 2008, 08:35:45
#
### Surface Triangulation file: oneraM6.tri
#                 
Mesh file: Mesh.mg.c3d
#         Input control file:
input.cntl
#
### Mach No = 0.5400,     alpha = 3.060, beta = 0.0000
#     cfl = 1.40,     FluxFun =
0,   Limiter = 2,
#  nMGlev = 4,    MGcycleType = 2,  
MG\_nPre = 1, MG\_nPost = 1
#
#      RK  
0.06950         1   #
stageCoef GradEval
#      RK  
0.16020         0   #
stageCoef GradEval
#      RK  
0.28980         0   #
stageCoef GradEval
#      RK  
0.50600         0   #
stageCoef GradEval
#      RK  
1.00000         0   #
stageCoef GradEval
#
#       Moment center is at
MomentCenter[x,y,z] = [0, 0, 0]
#       Moment log output is in model
coordinate sysem
#  mgCycle CPUtime/proc Moment\_model[X] Moment\_model[Y]
Moment\_model[Z]
    
3         0.050  
-2.48906010e-01  1.54178484e-01  3.11833992e-01
    
6         0.073  
-1.20362596e-01  6.44899618e-02  1.41595513e-01
    
9         0.108  
-9.73288514e-02  5.47200438e-02  1.09996626e-01
    12        
0.142   -8.26468966e-02  5.12646585e-02 
9.21598081e-02
    15        
0.176   -8.35622355e-02  5.05789087e-02 
9.35599974e-02
    18        
0.213   -8.90720000e-02  5.07734395e-02 
1.00691710e-01
    21        
0.451   -9.43596807e-02  3.24502831e-02 
8.80750336e-02
    24        
0.690   -8.68095053e-02  3.28399882e-02 
8.31384915e-02
    27        
0.946   -8.73625253e-02  3.28024741e-02 
8.37560623e-02
    30        
2.482   -8.15665478e-02  2.38628188e-02 
7.89556544e-02
    33        
3.922   -8.00323831e-02  2.21418698e-02 
7.66542291e-02
    36        
5.354   -7.98684200e-02  2.21361999e-02 
7.64408700e-02
    39       
27.535   -7.28831390e-02  -1.86995880e-03 
6.72155495e-02
    42       
47.934   -7.43492991e-02  -3.23945683e-03 
6.86250368e-02
    45       
67.957   -7.51943609e-02  -3.47952560e-03 
6.95607388e-02
    48       
89.193   -7.59388926e-02  -3.69007148e-03 
7.04454144e-02

---

<a id="page-howto-facelabels-index-html"></a>

## Cart3D HowTo Tri-Tags

*Original page: [howto/faceLabels/index.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/howto/faceLabels/index.html)*

---

### *How-To* Label Triangulations

#### ComponentID's, IntersectComponents, GMPtags and SurfBC's in Cart3D

---

- [Introduction](#page-howto-facelabels-index-html--t-setup)
- [Assigning Face Labels](#page-howto-facelabels-index-html--t-scribe)
- [Component Lables and GMPtags](#page-howto-facelabels-index-html--t-gmptags)

- [Tricks with *trix* and *comp2tri*](#page-howto-facelabels-index-html--t-trix)

- [Flow Analysis](#page-howto-facelabels-index-html--t-flow)
- [Quick Reference Guide](#page-howto-facelabels-index-html--t-worksheet)

---

#### Introduction

Component labels are a basic element
of [Cart3D
triangulations](#page-cart3dtriangulations-html), where they indicate which component of a configuration a
triangle belongs to. In practice, labels are simply integers assigned to the
triangles. A basic example is shown on the right where each triangle pair
belonging to the same face of a cube is assigned the same integer (denoted by
the same color). Such labels not only identify components, but are also useful
for specifying power boundary conditions and for defining aerodynamic
components, sub-components, or even arbitrary regions of the geometry for force
and moment integration. Recently new options have been added to provide greater
flexibility when working with labels. Specifically, the new options preserve
labeling within each watertight component and allow them to pass
transparently through the *intersect* procedure. Historically, labeling
an arbitrary region of the geometry was possible only after
the *intersect* step. Here we show how to label a region of a watertight
component before extracting the wetted surface with *intersect*. These
features are supported in Cart3D v1.4.5 and newer.

Let's assume you have been doing flow analyses on a wing-body-tail
configuration shown in the figure on the left. The geometry file is in the
standard [wetted
surface triangulation](#page-cart3dtriangulations-html--3-wetted-surface-triangulation-format) format, where the fuselage, wing and tail components
have been intersected and are labeled as indicated. For example, you may have
been using these labels to specify force and moment integrations via
the [Geometry
Manipulation Protocol (GMP)](#page-config-xml-html)
and [CLiC](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/clic/clic.html).
Frequently, we want to add additional labels to a configuration such as this
(maybe to specify surface boundary conditions) or to modify or remove existing
labels.

To demonstrate, let's add "engines" and "pylons" to this configuration. We'll
add a total of four components (2 engines + 2 pylons) as shown in the figure on
the right. Assume that we are given a single engine and pylon in separate files
and that each is a watertight, standard
[single component](#page-cart3dtriangulations-html--1-component-file-forma) (there are either no labels or all triangles have the
same label). We will mirror the parts later to create a symmetric
configuration. A translucent side-view of the engine is shown in the inset
frame. The diffuser terminates at a fan-face inlet and the nozzle begins at a
power (exhaust) face
where surface boundary conditions will be applied.

#### Assigning Face Labels

The first step is to label the fan and power faces uniquely. Then
the *SurfBC* token
in [input.cntl](#page-input-cntl-html)
can be used to specify appropriate boundary conditions for the simulation.
A convenient
approach is to use the utility code *breakTris*, which assigns a label
to a region of the triangulation surrounded by "sharp" edges. The user specifies
a seed triangle index, e.g. a triangle belonging to the fan face, and
optionally a sharpness threshold, and *breakTris* automatically labels
all triangles in the same region with a unique label. Multiple seeds can be
specified to generate multiple labels. We use the following command line to
label the fan and power faces:

% breakTris -seed 14435 13839 -i engine.tri -o engine\_tags.i.tri

Any triangle on the fan and power faces can be used as seed. *Overgrid*,
*Tecplot* and *Paraview* are all convenient tools for finding
seed indices (in this case triangles 14435 and 13839). In *Overgrid*
this is done by pressing the **f** key while hovering over a
triangle. By convention, we use the extension *.i.tri* for the output
file to denote that the file now contains a
watertight [wetted
surface triangulation](#page-cart3dtriangulations-html--3-wetted-surface-triangulation-format) with multiple components (just like the
wing-body-tail configuration above).

This "labeling step" adds new tags to the engine triangulation. A
cut-away view of the tagged engine
is shown on the left, where the nacelle (gray) is labeled as component 1, fan
face (blue) as component 2 and power face (red) as component 3. Notice that
for the integration of aerodynamic forces, you may wish to label the
diffuser and nozzle sections of the nacelle as separate components as well.

The engine is now ready for volume meshing and flow analysis. An excellent
tutorial on isolated engine simulations is provided in *$CART3D/doc/powerBCs/*
[which is also availible on-line.](#page-howto-samples-power-readme-html) For further information on using *trix*
or *breakTris* see the Cart3D Command Summary in *$CART3D/doc/COMMAND\_SUMMARY.pdf*
If you're working with CAD parts via [Engineering Sketch Pad (ESP) or OpenCSM,](https://acdl.mit.edu/ESP/) both the OpenCSM scripting
language and the *egads2cart* utility (included with ESP) can create face lables when the
triangulation is generated on the CAD part. The same is true if you're generating geometry
with Pointwise, or with [OpenVSP.](https://openvsp.org/) In addition, the
Cart3D utility code *triangulate* can also
assign face labels if given a component list file
( *% triangulate -C Components.list* ... ).

#### Component Labels and GMP Tags

To generate an intersected watertight configuration consisting of engines,
pylons and the wing-body-tail geometry, we need to differentiate between the
existing labels, e.g. the fan, power or fuselage labels, and the traditional
component labels that indicate watertight triangulations for processing
by *intersect*. In other words, the key is to realize that we are
intersecting three watertight triangulations — engine, pylon and
wing-body-tail — some of which contain many face labels. We introduce two
terms to clearly distinguish the label type:

- **IntersectComponents**: these labels denote components that
  should be intersected. These labels are usually assigned
  by *comp2tri* and are used by *intersect* to identify
  individual watertight components that should be checked for intersection.
  Simply put, these are the traditional component numbers (ComponentID's) as
  defined in the standard
  [configuration file format](#page-cart3dtriangulations-html--2-configuration-file-format).
- **GMPtags**: these labels are used to define GMP components
  and *SurfBC* labels. Common uses are for force and moment
  integration, or to assign boundary conditions, and are simply propagated
  through *intersect* transparently.

There are two codes that manipulate face labels: *trix*
and *comp2tri*. These codes offer many labeling options and we
demonstrate only the critical ones here. As with all codes in Cart3D, a
trailing dash "**-**" gives full usage statements, e.g. "*%
trix -*" or "*% comp2tri -*". There are three steps to obtain
a *cubes*-ready geometry.

### Step 1: Creating GMPtags

The first step is to promote the existing labels of the wing-body-tail
triangulation to **GMPtags** via *trix*:

% trix -v -comp2gmp -o wing\_body\_tail\_gmp.i wing\_body\_tail.i.tri

and similarly for the engine:

% trix -v -comp2gmp -o engine\_gmp.i engine\_tags.i.tri

Note that *trix* automatically appends the *.tri* extension to
output files. The output files are written in in an *extended triangulation
format* (called "*TRIX*") that allows arbitrary triangle and vertex
data. Syntactically, this format it is subset of
the [VTK Unstructured Grid XML
(VTU)](http://www.vtk.org/Wiki/VTK_XML_Formats) format. This compatibility means that vtk-based tools such as
as *Paraview* and *VisIt* can read these files
natively. Alternatively, they can be converted for visualization
in *Tecplot* using "*% trix -T ...*". Native support for *TRIX*
files is planned for an upcoming release of *Overgrid*.

### Step 2: Manipulating GMPtags to Avoid Collisions

If we think ahead to the final geometry we wish to analyze
in *flowCart*, we recognize that **GMPtags** from different
triangulations may not be unique. For example, the wing in the wing-body-tail geometry is
labeled as 2, which is the same as the fan face of the engine. The simulation
would not go well if inlet boundary conditions are assigned to both the fan
face and the wing. The user has the option to relabel manually or use the
flag *gmpTagOffset* in *comp2tri*, which
assigns unique labels automatically. We show the manual approach here and
outline the automatic approach in the next section.

Let's assume that we wish to keep the wing-body-tail labels unchanged (1
through 4), and shift the engine labels from 1-3 to 5-7. This is accomplished
via *trix*:

% trix -v -add2gmp 4 -o engine\_gmp\_shift4.i
engine\_gmp.i.tri

which increments all engine **GMPtags** by 4, resulting in the nacelle
being labeled as 5, fan face as 6 and power face as 7. Now we deal with the
technicality that we wish to build a symmetric configuration with engines and
pylons on both sides. We use *trix* with the
option *mirror Z* to create a new component
mirrored about the Z plane. If required, *trix* also includes options to
scale, translate and rotate the geometry.

% trix -v -mirror Z -o engine\_mirror.i
engine\_gmp\_shift4.i.tri

The same steps are repeated with the pylon, except that we use *"-add2gmp
7"* to shift its **GMPtag** past the engine power face
label. We are now ready to assemble and intersect the components.

### Step 3: *comp2tri* and *intersect*

Assembly of the [configuration triangulation](#page-cart3dtriangulations-html--2-configuration-file-format) is done with *comp2tri*:

% comp2tri -trix -v wing\_body\_tail\_gmp.i.tri \
       engine\_gmp\_shift4.i.tri  engine\_mirror.i.tri  pylon\_shift7.i.tri  pylon\_mirror.i.tri

which generates the familiar *Components.tri* file (written in extended
triangulation format). We finish with *intersect*:

% intersect -v -i Components.tri

 to obtain the
final wetted surface
in *Components.i.tri*
with both **IntersectComponents** (ComponentIDs) and **GMPtags**.
The **GMPtags** of the final configuration are shown below, where
the face labels of the original wing-body-tail geometry are preserved (1-4),
the engines use labels 5 through 7 and the pylons are tagged with 8. These
labels can be used in your
[Config.xml](#page-config-xml-html)
file, as well as in
[input.cntl](#page-input-cntl-html)
to specify *SurfBC* boundary conditions.

Recall that *Components.i.tri* also
contains **IntersectComponents**. The image below shows
the **IntersectComponent** labels assigned
by *comp2tri*. These labels identify the 5 individual watertight
component triangulations that were processed by *intersect*. Labels are
assigned in the same order as the components appear in the
*comp2tri* command line above.

[Earlier,](#page-howto-facelabels-index-html--t-trix) we mentioned that it was possible to
skip the manual labeling in **Step 2** and have
unique labels assigned automatically. This can be accomplished with the following
command:

% comp2tri -trix -gmpTagOffset 4
wing\_body\_tail\_gmp.i.tri \
      engine\_gmp.i.tri  engine\_mirror.i.tri
  pylon.a.tri  pylon\_mirror.a.tri

where *gmpTagOffset* offsets the face labels
found in each input file on the command line by a specified factor. In this
case we use 4 because this is the highest **GMPtag** in all the
triangulations, which causes the **GMPtags** found in the first
file on the command line to be left unchanged, adds 4 to
each **GMPtag** in the second file, adds 8 to
each **GMPtag** in the third file, and so on to avoid label
collisions. If an input file does not contain **GMPtags**,
e.g. a single component triangulation such as the pylon,
then *comp2tri* automatically assigns the next sequential label.

#### Flow Analysis

Note that *intersect* generates *Components.i.tri* in the
extended triangulation format. Many codes in the *Cart3D* package are
able to read and write these files,
including *autoInputs*, *cubes* and *aero.csh*. One of the
current exceptions is *mpi\_flowCart* (as of release v1.4.7). If your work-flow
uses *mpi\_flowCart*, then you can convert the triangulation file
back to the
standard [wetted
surface triangulation](#page-cart3dtriangulations-html--3-wetted-surface-triangulation-format) format via trix:

% trix -v -tri Components.i.tri

which preserves the **GMPtags** of the configuration.

---

#### Quick Reference Guide

---

*last update: May 2023*

---

<a id="page-howto-internalflow-index-html"></a>

## howto/internalFlow/index.html

*Original page: [howto/internalFlow/index.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/howto/internalFlow/index.html)*

### Setting up Internal flows with Cart3D:

- [1. Geometry Setup](#page-howto-internalflow-index-html--moztocid334223)

- [1.1
  Normal Vectors:](#page-howto-internalflow-index-html--moztocid34152)
- [1.2.
  Laying out the Cartesian Grid:](#page-howto-internalflow-index-html--moztocid858414)

- [2. An
  Example](#page-howto-internalflow-index-html--moztocid681932)

 

---

#### 1. geometry setup

 Lets
assume that we're talking about something that looks like a model in a
wind tunnel.

Notice that the flow comes in aligned with the tunnel, and the angle of
attack is specified by rotating the model with respect to the tunnel.

   

#### 1.1 Normal Vectors:

 Cubes
usually meshes a geometry using the normals pointing "into-the-flow",
However, for internal flows, cubes' "-internal" flag automatically
reverses the sence of the normal vectors, this means that the geometry
should have its normal vectors pointing "Away-from-the-flow" and when
you run cubes, be sure to use the "-internal" flag.

#### 1.2. Laying out the Cartesian Grid:

Carteisan
cells that cut solid geometry automatically get a slip boundary
condition (inviscid wall) applied to the cut-faces. If a Cartesian cell
does not cut geometry, you can apply either a far-field, or inviscid
wall - remember, a symmetry plane is the same as a planar inviscid
wall. (Actually, if your "wind tunnel" was a rectangular box, you could
do this simply by setting the Ymin, Ymax and Zmin, Zmax boundaries to
"symmetry/Inviscid wall" and Xmin&Xmax to "far field" --
without ever using a \*tri file to specify the tunnel geometry and just
let the boundary of the cartesian grid be the "tunnel walls". But
anyway... lets assume that you have geometry for your tunnel. The
following sketch shows how to lay it out.

####

#### 2. An Example

Here
is an example, showing a cutaway internal flow run at supersonic
speeds, since the flow is supersonic, you can see the shock reflections
off the tunnel walls.

Here is a picture of the cut-cells in the grid

---

<a id="page-howto-viscousdrag-index-html"></a>

## Cart3D HowTo Tri-Tags

*Original page: [howto/viscousDrag/index.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/howto/viscousDrag/index.html)*

---

### *How-To* estimate drag with "*viscousDrag*"

### Estimating friction drag with *$CART3D/bin/$CART3D\_ARCH/viscousDrag*

##

---

- [Introduction](#page-howto-viscousdrag-index-html--intro)
 - [Usage](#page-howto-viscousdrag-index-html--usage)
 - [Sample files](#page-howto-viscousdrag-index-html--sample)
- [References](#page-howto-viscousdrag-index-html--refs)

---

---

### Introduction

The ***viscousDrag***
module is found in *$CART3D/bin/$CART3D\_ARCH/viscousDrag*

the module does a single pass *a-posteriori* drag estimate
based on the inviscid solution provided by *flowCart* (via
a \*.triq file). This method of estimating the friction drag is
most accurate when there is only weak interaction between the
boundary-layer and the outer flow. The module solves the 2D
integral boundary layer equations along streamwise strips taking
the computed inviscid solution for boundary conditions at the
outer edge of the boundary-layer. The module includes an option
to use a flat-plate boundary layer profile instead of actually
driving it from the inviscid solution (useful for comparison).

### Usage

```
   Usage: viscousDrag [ argument list ]
 Options:
          -- Runtime Options--
-v                  verbose mode ON
-flatPlate          Use flat plate boundary layer profile
-mem                Report memory usage (auto on with -v)
-i %s               Input  file name, default:<input.cntl>
-clic %s            Use CLiC file named (e.g. modelName.triq)
-scale %F           Number of feet per-unit-model length def:<1.0>
-version            Dump version info and exit
```

### Sample files with additional documentation

###

[Here is a link](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/howto/viscousDrag/rm10_viscousDrag_ex_2007.zip) to a
zip file containing an example running the viscousDrag module on
the RM-10 supersonic research mode. To run viscousDrag on this
example use something like

  % 
viscousDrag -clic rm10.i.triq

This runs ***viscousDrag*** using the data contained in
rm10.i.triq which has results from a Mach 1.2 run of *flowCart*

**Files**

- [rm10\_viscousDrag\_ex\_2007.zip](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/howto/viscousDrag/rm10_viscousDrag_ex_2007.zip) 
  : *Complete example*
- [00\_README.txt](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/howto/viscousDrag/00_README.txt)
- [input.cntl](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/howto/viscousDrag/input.cntl) : *Sample input
  file*
- [Mesh.c3d.Info](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/howto/viscousDrag/Mesh.c3d.Info) : *See the* *"cubes"
  line* *t**o s**ee how it was meshed*
- [rm10.i.tri.gz](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/howto/viscousDrag/rm10.i.tri.gz) : *Cart3D
  component**-based* *geometry file with
  surface triangulation*
- [rm10.i.triq.gz](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/howto/viscousDrag/rm10.i.triq.gz) : *Annotated
  triangulation file* *from run at Mach 1.2 with
  flow data*
- [NOTES\_rm10.pdf](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/howto/viscousDrag/NOTES_rm10.pdf) : *N**otes
  on running and expected results*

### References

See pages 8-10 of *[AIAA
Paper 2006-0652.](http://www.nas.nasa.gov/publications/software/docs/cart3d/pages/publications/AIAA_2006-0652.pdf)*• Aftosmis, M.J., Berger, M.J., and Alonso, J.J.,
“Applications of a Cartesian Mesh Boundary-Layer Approach for
Complex Configurations,” *AIAA Paper 2006-0652*, Jan.
2006..

---

 *last update: Oct. 2016*

---

<a id="page-clic-html-clic-choices-html"></a>

## PART3D

*Original page: [clic/html/clic_choices.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/clic/html/clic_choices.html)*

### CLiC

---

- [Introduction](#page-clic-html-clic-intro-html)
- [User guide](#page-clic-html-clic-doc-html)
- [<\*.clic> file](#page-clic-html-clic-inputfile-html)
- [<\*.cntl> file](#page-clic-html-clic-cntl-html)
- [Download](#page-clic-html-clic-download-html)

 

---

Last modified: August 16, 1999

---

<a id="page-clic-html-clic-intro-html"></a>

## clic/html/clic_intro.html

*Original page: [clic/html/clic_intro.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/clic/html/clic_intro.html)*

### Welcome to the CLiC web page

     In analyzing the aerodynamics of a configuration,
it is generally necessary to back-out force and moment information to gain
insight into convergence behavior, and to feedback into the "Loads and
Lines" decision making process. Force and moment information may be required
for an entire configuration, or may be confined to a subset of the components
on a given configuration. While conceptually straighforward, post-processing
this information on a case-by-case basis typically requires substantial
expertise and can be extremely time-consuming.

    Examples:

- Global CL (Lift coefficient), CD (Drag coefficient),
  Cy (Lateral force coefficient),

           
CN (Normal force coefficient), CA (Axial force coefficient),
CY (Side force coefficient)- CL, CD, Cy, CN, CA,
  CY  for selected components
- CM of selected components with respect to an arbitrary moment
  center
- CM of selected components with respect to an arbitrary line
  (defined by 2 points)
- Cp profile of selected components at body intersection with
  a plane (defined by 3 points)

    The frequent requirement of using force and moment data
for convergence assessment implies that a comprehensive module should be
accessible as a client-side background executable as well as through a
traditional intuitive front-end.

    The **CLiC** module
meets these requirements. The code takes as input [an
annotated triangulation](#page-clic-html-clic-inputfile-html) of a configuration, with triangles assigned
to components on  a triangle-by-triangle basis, and load information
annotated to vertices in the triangulation. Output requests (load information)
are made with a file-based interface that can be modified via a standard
Unix editor (a *[clic-mode style](#page-clic-html-clic-emacs-html)* is
provided for the Emacs editor). A primary design goal is that the module
requires minimal memory and CPU so that it can be spawned remotely as a
"thin client" form a remote server. An API library is also provided with
a minimal set of routines callable from a C application code and which
provide the whole functionality of the standalone code.

 

 

### Links to other projects within this task

- [The Cart3D
  project by Mike Aftosmis](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/cart3d.html)

---

*Last modified: August 16, 1999*

---

<a id="page-clic-html-clic-axis-html"></a>

## clic/html/clic_axis.html

*Original page: [clic/html/clic_axis.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/clic/html/clic_axis.html)*

### Example of axis specification

In this example, we would specify the axis transformation as:

> > > > > > > - $\_Model\_X\_axis\_is -Xb
> > > > > > >
> > > > > > > - $\_Model\_Y\_axis\_is  -Zb
> > > > > > >
> > > > > > > - $\_Model\_Z\_axis\_is  -Yb

  

---

*Last modified: September 8, 1999*
Created by M. Delanaye, maintained by Michael Aftosmis
[michael.aftosmis@nasa.gov](mailto:michael.aftosmis@nasa.gov)

---

<a id="page-clic-html-clic-doc-html"></a>

## clic/html/clic_doc.html

*Original page: [clic/html/clic_doc.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/clic/html/clic_doc.html)*

### CLiC User Guide

**Introduction**

    The **CliC**module
computes forces, moments (with respect to points and lines) and pressure
distribution for a configuration represented by a triangulation (manifold
or not). Load information and pressure distribution can be performed for
a single component, a group of components or the entire geometry. By using
some simple keywords, the user can build a database of components, or group
of components and address them via names.

    The **CliC**
module takes as input file <\**.clic >* an annotated triangulation
whose triangles are associated a component number on a triangle-by-triangle
basis. The load information are requested through a control file <\**.cntl
>*. The commands are defined in this control file through the usage
of keywords. The syntax of this file has been left as simple as possible
to offer the greatest flexibility.

 

- **[Installation](#page-clic-html-clic-doc-html--installation)**
- **[Usage](#page-clic-html-clic-doc-html--usage)**
- **[Axis definitions](#page-clic-html-clic-doc-html--frames)**
- **[Forces and moments](#page-clic-html-clic-doc-html--forces-and-moments)**
- **[Input files](#page-clic-html-clic-doc-html--inout-files)**
- **[API](#page-clic-html-clic-doc-html--api)**
- **[sample <\*.clic>
  triangulation file (ascii)](#page-clic-html-clic-inputfile-html)**
- **[sample <\*.cntl> control file](#page-clic-html-clic-doc-html--api)**

**Installation**

    The code can be [downloaded](#page-clic-html-clic-download-html)
from the AIC branch web server as a gzipped tar file. Unzip and untar this
archive by:

gunzip -c clic.tar.gz | tar xovf -

Define the environment variable ARCH and 
in your .cshrc:
> > > > setenv ARCH IRIX     
> > > > (SGI 32 bit machines)
> > > >
> > > > setenv ARCH IRIX64  ( SGI 64 bit machines)
> > > >
> > > > setenv ARCH linux     
> > > > (PC running linux)

and customize the Makefile.$ARCH for your
particular machine. Then simpy type

make

to create an executable clic\_$ARCH and the API
library libclic.a.

NB: the command make
clean can be used to clean the distribution
directory of all object files and executables, while make
new will recompile the whole distribution.
make
parser calls Lex&Yacc to create the C
routines which parse the control file.

**Usage**

The **CliC**module
can be used in standalone mode with the executable clic\_$ARCH.
It can also be started as a separate thread from any remote machine by
issuing system calls from the remote running server code. Runtime options
of the **CliC**module
can be obtained by typing:
clic\_$ARCH -

which outputs the following list of options:

  Usage: clic [ argument list ]

 Options:

      -i %s          
control file name, def:<clic.ctnl>

      -outDir %s     
output directory, def:<.>

      -ascii         
Input  geometry file is ASCII

      -v             
verbose more ON

      -mem           
Report memory usage (auto on with -v)

      -parse         
verbose parsing ON

      -T             
Create tecplot file

      -html          
html output, NOT YET SUPPORTED !

- The "-i" option specifies the control file name (*.cntl
  file*), by default the code assumes the file *clic.ctnl*.
- The "-outDir" option lets you specifies
  a directory name for the output files. If this directory does not exist,
  it will be created at runtime.
- The "-ascii" option specifies that the input triangulation
  file (*.clic file*) is an ascii FORTRAN free formatted file (write(6,\*)).
  If omitted an unformatted file is assumed.
- The "-v" option turns ON the verbose mode which outputs
  to stdout information regarding the actions taken by **CLiC**
- The "-mem" option is similar to the option "-v" and
  is automatically ON when the latter is used. It outputs the memory allocated
  after every memory allocation or free made by **CLiC**
- The "-parse" option turns on the verbose mode for
  the parsing of the control file which is performed using Lex&Yacc.
  It is essentially a debugging tool to check the correctness of the control
  file syntax.
- The "-T" option allows the output of a tecplot file
  containing the triangulation generated by **CLiC**
  for each separate component.
- The "-html" option is not yet supported, but will
  be used to output the computed load information in html format.

**Axis definitions**

In order to avoid any ambiguity for the forces
and moments computed by **CLiC**,
we have decided to adopt two classical frames of reference. The body frame
**Fb**is
linked to the aircraft body. The origin is the point of reference of the
aircraft, which in general, is the center of mass. The fuselage axis **xb**,
oriented towards th front of the aircraft, belongs to the symmetrical plane
of the aircraft. Although its definition in the symmetrical plane is arbitrary,
it is usually chosen such that it is parallel to a generatrix if the fuselage
is cylindrical. The axis **zb** is in the symmetrical plane
and is directed downward relative to the aircraft. The axis **yb**
is perpendicular to the symmetrical plane and is oriented towards the right
"pilot's side" of the aircraft. The user must specify in the control file
the axis transformation necessary to transform the frame in which the geometry
is defined to the body frame **Fb**.

The aerodynamical frame **Fa**has
the same origin as the body frame. It is defined by:

- The axis **xa** carried and oriented
  by rhe aerodynamic velocity of the aircraft.
- Two rotation angles alpha and beta, respectively
  named angle of attach and sideslip angle, for transformation from the body
  frame **Fb** to the aerodynamic frame **Fa**.

 

Figure 1. Body frame **Fb**and
aerodynamical frame**Fa**

**Forces
and moments**

**Forces**

Forces coefficients in the body frame **Fb**are
define as:

CA is the so-called coefficient of the axial force. while
CN is the coefficient of the normal force. The coefficients
in the aerodynamic frame are denoted:

The Lift and drag coefficients are CL and CD respectively.
The relationship between the coefficients en the aerodynamic and body frames
is obtained using the angles alpha and beta:

The **CLiC**
module can output CL (Lift coefficient), CD (Drag
coefficient), Cy (Lateral force coefficient),  CN
(Normal force coefficient), CA (Axial force coefficient), CY
(Side force coefficient) for a single component, group of components or
the entire geometry.

NB: Forces coefficients require the definition of *surface of reference,*in
general the surface area of the wing. This data is a mandatory input of
the
**CLiC** module which must be present
in the control file.

**Moments**

Moment coefficients can be computed with respect to a point whose coordinates
are provided in the control file. This computation can be again performed
for a single component, group of components or the entire geometry. 
The moment is output as vectors in the aerodynamical and body frames. Similarly
a moment with respect to a line (defined by the coordinates of two points
provided in the control file) can be computed. In addition to a reference
surface, a reference length must also be provided.

**Input
files**

Data are passed to the **CLiC** module
by two files. The first file *[<\*.clic>](#page-clic-html-clic-inputfile-html)*must
contain the annotated triangulation, and pressure distribution (Cp),
as well as a component number for each triangle. This file can be written
as fortran free formatted file or an unformatted file. A sample <\*.clic>
file can be found [here](#page-clic-html-clic-inputfile-html).

The second file is a control file (*.cntl*) containing the different
commands for the forces and moments computation, as well as some mandatory
information such as reference area and length. The file names of the input
triangulation files [<\**.clic>*](#page-clic-html-clic-inputfile-html) which
contain the annotated triangulations defining the configuration are also
listed in the control file. Naming of components or group of components
can also be specified in this file.

Each command present in the control file starts by a keyword and is
followed by some context dependent information. This file is read by a
parsing routine created by Lex&Yacc and is therefore very flexible
regarding syntax. Hence, comments can be introduced anywhere in the file,
a comment is a chain of characters which starts by a # sign:

Example:      # this
is my first control file

The keywords *all* and *entire*have
specific meaning:

- *all* in a command means that it applies
  to every single component or group of components
- *entire* in a command
  means that it applies to the entire configuration, e.g. a moment calculation
  for the complete configuration

The following keywords can be used:
> - $\_Title: (optional)
>
>
> This keyword should be followed by a list of characters
> corresponding to the title of the problem, which is used as a header in
> the output files.
>
> Usage:  
> $\_Title   This is an
> example for the use of clic
>
> - $\_Directory (optional)
>
>
> This keyword should be followed by a list of strings
> corresponding to the directories where the clic files can be found.
>
> Usage:  
> $\_Directory   dir1
> dir2 dir3
>
> - $\_Ouput\_Directory (optional)
>
> This keyword should be followed by a string
> that specifies the output directory where **CLiC**will save its
>
> output files
>
> Usage:  
> $\_Output\_Directory  
> dir
>
> - $\_ClicFileName (optional)
>
>
> This keyword should be followed by a list of strings
> corresponding to the clic files (*.clic*) to be read.
>
> Usage:  
> $\_ClicFileName  file1.clic
> file2.clic file3.clic

> - $\_Model\_X\_axis\_is (optional)
>
>
> This keyword defines the Model X-axis with respect
> to the [body Frame](#page-clic-html-clic-doc-html--frames). It should be followed by one of
> the following: +Xb, -Xb,  Xb, +Yb, -Yb,  Yb, +Zb, -Zb, 
> Zb.
>
> Usage:  
> $\_Model\_X\_axis\_is  
> -Yb #the x-axis of the mesh corresponds to
> the
>
>                                   
> opposite y-axis of the body frame
>
> Click [here](#page-clic-html-clic-axis-html) for an example.
>
> - $\_Model\_Y\_axis\_is (optional)
>
>
> Same as previous except that it applies to the
> Y-axis
>
> Click [here](#page-clic-html-clic-axis-html) for an example.

> - $\_Model\_Z\_axis\_is (optional)
>
>
> Same as previous except that it applies to the
> Z-axis
>
> Click [here](#page-clic-html-clic-axis-html) for an example.

- $\_Incidence\_Angle (mandatory)

This keyword should be followed a real number
specifying the angle of attack as defined [here](#page-clic-html-clic-doc-html--angles).

Usage:  
$\_Incidence\_Angle  10.0

- $\_Sideslip\_Angle (mandatory)

This keyword should be followed  real number
specifying the sideslip angle as defined [here](#page-clic-html-clic-doc-html--angles).

Usage:  
$\_Sideslip\_Angle 3.0

- $\_ComponentName (optional)

This keyword attributes a name to a component
number. It should be followed by a number (the component number) and a
character chain (the name).

Usage:  
$\_ComponentName  2  
rudder

- $\_ComponentGroup (optional)

This keyword is used to create a group of components.
It should be followed by a the name given to the group and a list of component
numbers or names.

Usage:  
$\_ComponentGroup  wings
1 leftWing leftFlap 2

- $\_Reference\_Area (mandatory)

This keyword attributes a reference area for a
component, group of components given by their name or number.

Usage:  
$\_Reference\_Area 1.11 wings fuselage
10 11

- $\_Reference\_Length (mandatory)

This keyword attributes a reference length for
a component, group of components given by their name or number.

Usage:  
$\_Reference\_Length  1.11
wings fuselage 10 11 all

- $\_Force (optional)

This keyword requests the computation of the force
coefficients for a list of components of group of components.

Usage:  
$\_Force  fuselage wing flap
entire

- $\_Moment\_Point (optional)

This keyword requests the computation of the moment
with respect to a point. It should be followed by the coordinates of the
point and a list of components of group of components.

Usage:  
$\_Moment\_Point 1.0 1.5 -0.3 fuselage
wing flap

- $\_Moment\_Line (optional)

This keyword should requests the computation of
the moment with respect to a line. It should be followed by the coordinates
of the two points defining the line and a list of components of group of
components.

Usage:  
$\_Moment\_Line  1.0 0.0 0.9  
0.0 1.5 -0.3 fuselage wing flap

- $\_Distribution (optional)

This keyword should requests the computation a
Cp distribution obtained by cutting a component or a group of components
with a plane defined by three points whose coordinates are given after
the keyword.

Usage:  
$\_Distribution 1.0 0.0 0.9 
0.0 1.5 -0.3  0.0 0.0 1.0 fuselage wing flap

**API**

The **CliC**module
is also provided with an Application Programming Interface. The user can
link his C/C++ application to the library libclic.a
and
include the file clicapi.h
which
defines the prototypes of the API routines.

The following routines can be used:

tsClicErr cliApiInit(char \*controlFile,int
nVerts,int nTris, double \*vtxs,

                    
int \*tris,int \*cmp,void \*\* ClicDb)

This routine initializes the clic database by
passing the name of the control file and the triangulation with the list
of the vertices coordinates and triangles definition. To keep track of
the database, a void pointer to it is assigned.

- controlFile: name of the control file
- nVerts: number of vertices of the triangulation
- nTris: number of triangles of the triangulation
- \*vtxs: list of vertices coordinates, of
  length 3\*Nverts

       vtxs[3\*i+k]
is the kth coordinate of vertex i- \*tris: list of triangles, of length 3\*nTris

       tris[3\*i+k]
is the index of the kth vertex of triangle i- \*cmp: component number for each triangle,
  length nTris
- \*ClicDb: pointer to the clic database, CLICDEFAULT
  can be used if no pointer is

        
to be returned

tsClicErr clicApiCompute(float \*valuesVerts,int
nScal, void \*ClicDb)

This routine passes the scalar values at each
vertex of the triangulation and compute all the requested information.

- \*valuesVerts:
  array containing the scalar values at each vertex, size nVerts\*nScal

   Rem: valuesVerts[iVert\*nScal+iScal]
is  "iScal" value at vertex
"iVert"

       
iScal=0 always corresponds to Cp- nScal: number of scalars
- \*ClicDb: pointer to the clic database, CLICDEFAULT
  can be used

tsClicErr clicApiGetForce(double \*CD,double
\*Cy,double \*CL,

                         
double \*CA,double \*CY,double \*CN,

                         
int CmptIndex,char\* CmptName,void \*ClicDb)

Returns the force coefficients for a component
given either by its number or name.

- CmptIndex: component number,  CLICNOINDEX
  if
  not specified
- \*CmptName: component name,  CLICNONAME
  if
  not specified
- \*ClicDb: pointer to the clic database, CLICDEFAULT
  can be used

tsClicErr clicApiGetPointMoment(a\_tsClicDbPointMnt
\* a\_ClicDbPointMnts,

                              
int \*nClicDbPointMnts,

                              
int CmptIndex,char\* CmptName,void \*ClicDb)

Returns all point moment calculations performed
for a component given either by its number or name.

- CmptIndex: component number,  CLICNOINDEX
  if
  not specified
- \*CmptName: component name,  CLICNONAME
  if
  not specified
- \*ClicDb: pointer to the clic database, CLICDEFAULT
  can be used

tsClicErr clicApiGetLineMoment(a\_tsClicDbLineMnt
\* a\_ClicDbLineMnts,

                             
int \*nClicDbLineMnts,

                             
int CmptIndex,char\* CmptName,void \*ClicDb)

Returns all line moments calculations performed
for a component given either by its number or name.

- CmptIndex: component number,  CLICNOINDEX
  if
  not specified
- \*CmptName: component name,  CLICNONAME
  if
  not specified
- \*ClicDb: pointer to the clic database, CLICDEFAULT
  can be used

tsClicErr clicApiStreamObj(char \*\*stream,tsClicObjType
clicObjType,

                         
int CmptIndex,char\* CmptName,void \*ClicDb)

Output a formatted stream for a clic object present
in the database. A clic object can be:

- force object
- point moments object
- line moments object
- Cp distributions object

- clicObjType can be CLIC\_FORCE,CLIC\_POINT\_MOMENT,CLIC\_LINE\_MOMENT,CLIC\_CP
- CmptIndex: component number,  CLICNOINDEX
  if
  not specified
- \*CmptName: component name,  CLICNONAME
  if
  not specified
- \*ClicDb: pointer to the clic database, CLICDEFAULT
  can be used

tsClicErr clicApiWriteObj(char \*outputFileName,tsClicObjType
clicObjType,

                         
int CmptIndex,char\* CmptName,void \*ClicDb)

Write a clic object present in the database in
a file named  outputFileName. A clic object can be:

- force object
- point moments object
- line moments object
- Cp distributions object

- clicObjType can be CLIC\_FORCE,CLIC\_POINT\_MOMENT,CLIC\_LINE\_MOMENT,CLIC\_CP
- CmptIndex: component number,  CLICNOINDEX
  if
  not specified
- \*CmptName: component name,  CLICNONAME
  if
  not specified
- \*ClicDb: pointer to the clic database, CLICDEFAULT
  can be used

tsClicErr clicApiMemRelease(void \*ClicDb)

Destroy the clic database pointed by \*ClicDb
(CLICDEFAULT can be used) and free the memory.

tsClicErr clicApiStreamMemRelease(char \*\*stream)

Free the memory associated with a stream allocated by the **CLiC**
module.

Each routine returns an error code which can be:

- CLICERR\_FILENOTFOUND
- CLICERR\_CANNOTOPENFILE:
- CLICERR\_OUTOFMEMORY,
- CLICERR\_NODATABASE,
- CLICERR\_NOTACOMPONENT,
- CLICERR\_NOTCOMPUTED

---

*Last modified: September 8, 1999*

---

<a id="page-adjoint-fun-con-dat-html"></a>

## fun_con.dat

*Original page: [adjoint/fun_con_dat.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/adjoint/fun_con_dat.html)*

---

### *fun\_con.dat*

---

```
### Mesh convergence of output J
### Mon Aug 24 21:44:42 2015
### File generated by /u/wk/mnemec/cart3d/bin/aero_mkFunCon.pl
#
### (1)Cycle  (2)nCells      (3)Functional      (4)Avg. Fun. (window=1)
0            7171           4.4026759995e-02   4.4026759995e-02
1            10755          4.3915670336e-02   4.3915670336e-02
2            16047          4.2055060290e-02   4.2055060290e-02
3            31890          3.8095201070e-02   3.8095201070e-02
4            61949          3.4773217450e-02   3.4773217450e-02
5            121410         2.4242977116e-02   2.4242977116e-02
6            233353         1.7030655859e-02   1.7030655859e-02
7            454046         1.3520281096e-02   1.3520281096e-02
```

---

<a id="page-adjoint-results-dat-html"></a>

## results.dat

*Original page: [adjoint/results_dat.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/adjoint/results_dat.html)*

---

### *results.dat*

---

```
#
### Mon Aug 24 21:35:46 2015
### File generated by /u/wk/mnemec/cart3d/bin/aero_getResults.pl
#
### Adaptation results
#
### (1)Cycle (2)Cells (3)1/2*log10(Cells) (4)J_H  (5)J_cor  (6)Error_Indicator  (7)Error_Indicator/TOL  (8)TOL  (9)2|J_H-J_cor|
0            7171            1.9277899e+00  4.4026760e-02  4.1649112e-02  1.1685609e-02  1.1685609e+04  1.00e-06  4.7552958e-03
1            10755           2.0158052e+00  4.3915670e-02  4.1636917e-02  1.1552601e-02  1.1552601e+04  1.00e-06  4.5575057e-03
2            16047           2.1026969e+00  4.2055060e-02  3.8379709e-02  1.0113485e-02  1.0113485e+04  1.00e-06  7.3507022e-03
3            31890           2.2518273e+00  3.8095201e-02  3.4254747e-02  8.6918233e-03  8.6918233e+03  1.00e-06  7.6809077e-03
4            61949           2.3960172e+00  3.4773217e-02  3.0547198e-02  7.0525885e-03  7.0525885e+03  1.00e-06  8.4520393e-03
5            121410          2.5421272e+00  2.4242977e-02  2.0860444e-02  3.6291939e-03  3.6291939e+03  1.00e-06  6.7650653e-03
6            233353          2.6840067e+00  1.7030656e-02  1.4796647e-02  1.6855700e-03  1.6855700e+03  1.00e-06  4.4680179e-03
```

---

<a id="page-clic-html-clic-inputfile-html"></a>

## clic/html/clic_inputfile.html

*Original page: [clic/html/clic_inputfile.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/clic/html/clic_inputfile.html)*

### Annotated input triangulation file

The input file (*.clic file*) to the **CLiC**module
contains an annotated triangulation describing the entire configuration.
The triangles are assigned a component on a triangle-by-triangle basis,
and scalar values (like pressure) are provided at the vertices of the triangulation.
The following format is employed.

> nVerts nTri nScal     *number
> of vertices, number of triangles, number of scalars*
>
> x1 y1 z1        *coordinates
> of the vertices*
>
> x2 y2 z2
>
> x3 y3 z3
>
> .....
>
> .....
>
> v11 v12 v13     *triangles
> definition (3 integers)*
>
> v21 v22 v23
>
> .....
>
> .....
>
> .....
>
> 1 1 1 1 1 1 2 2 2 2  3 3 3 3 1 2 *component
> number for each triangle*
>
> .....
>
> 0.0000 0.0000 0.0000 ..... *scalars
> values  (1,..,nScal) for each vertex*
>
> ......
>
> NOTE: The file is assumed to be written in FORTRAN style (ASCII free
> format or
>
> unformatted).
>
>    *Example of reading code:*
>
>
>       read(iUnit,\*) nVerts, nTri, nScal
>
>       read(iUnit,\*) ( x(j), y(j), z(j) 
> ,j=1,nVerts)
>
>       read(iUnit,\*) ( v1(j),v2(j),v3(j)
> ,j=1,nTri  )
>
>       read(iUnit,\*) ( comp(j)          
> ,j=1,nTri  )
>
>       read(iUnit,\*) ((scalar(j,k),k=1,nScal),j=1,nVerts)
>
> Remark:   The first scalar (k=1)
> is always assumed to correspond to the pressure
>
>                  
> coefficient !

---

*Last modified: September 8, 1999*

---

<a id="page-clic-html-clic-cntl-html"></a>

## clic/html/clic_cntl.html

*Original page: [clic/html/clic_cntl.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/clic/html/clic_cntl.html)*

### <\*.cntl> control file

The control file <\*.cntl> containing the different commands for the
forces and

moments computation, as well as some mandatory information such as
reference area and

length. The file names of the input triangulation files <\*.clic>
which contain the annotated

triangulations defining the configuration are also listed in the control
file. Naming of

components or group of components can also be specified in this file.

Here is an example file:

### Example of a control file < citationX.cntl
>

#---------------------------------------------

### ... Title definition

$\_Title citationX

### ...  List of file names

$\_ClicFileName citationX.clic #(optional)
will use controlfile.clic

### ... Output directory (optional)

$\_Output\_Directory output     
### %s output directory

### ... Axis definitions

$\_Model\_X\_axis\_is +Xb

$\_Model\_Y\_axis\_is -Yb

$\_Model\_Z\_axis\_is -Zb

 

### ... Angles definiton

$\_Incidence\_Angle  0.     
### angle of attack

$\_Sideslip\_Angle   0.     
### sideslip angle

 

### ... Component name definition

$\_ComponentName 0 fuselage

$\_ComponentName 1 wings

$\_ComponentName 2 tail

$\_ComponentName 3 rudder

$\_ComponentName 4 Leftengine

$\_ComponentName 5 Rightengine

 

### ... reference area and length specifictions

$\_Reference\_Area   
1. all

$\_Reference\_Length  1. all

 

### ... Component Group

$\_ComponentGroup engines 4 5 6 7

$\_ComponentGroup aircraft 0 1 2 3
4 5 6 7

### ... Force Info

$\_Force aircraft  engines fuselage
entire

### ... Point Moment

$\_Moment\_Point 0. 0. 0. aircraft engines
fuselage

$\_Moment\_Point 1. 0. 0. aircraft engines
fuselage

 

### ... Line Moment

$\_Moment\_Line 0. 0. 0. 1. 0. 0. aircraft
engines fuselage

$\_Moment\_Line 1. 0. 0. 1. 0. 0. aircraft
engines fuselage

### ... Cp Distribution

$\_Distribution 0. 6. 0. 0. 6. 1. 1.
6. 1.     aircraft fuselage

---

*Last modified: September 8, 1999*

---

<a id="page-clic-html-clic-download-html"></a>

## clic/html/clic_download.html

*Original page: [clic/html/clic_download.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/clic/html/clic_download.html)*

### CLiC Download

*CLiC* is now bundled with the full
Cart3D distribution.
Component-wise loads breakdown are available directly form the
solver using the [***Config.xml***](#page-config-xml-html)and the **$\_\_Force\_Moment\_Processing**
section of [***input.cntl***](#page-input-cntl-html).
For standalone usage see  ***$CART3D/bin/$CART3D\_ARC/clic***and refer to the documentation in ***$CART3D/doc/COMMAND\_SUMMARY\_v\*.pdf***.

---

*Last modified: March, 2022*

---

<a id="page-clic-html-clic-emacs-html"></a>

## clic/html/clic_emacs.html

*Original page: [clic/html/clic_emacs.html](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/clic/html/clic_emacs.html)*

### Emacs *clic-mode* lisp file

The file *[clic.el](https://www.nas.nasa.gov/publications/software/docs/cart3d/pages/clic/html/clic.el)*  contains the lisp
code defining a Emacs clic-mode which can be used to format the edition
of the control file provided to the **CLiC** module.
Please save this file in your *elisp* directory.

 The following lines should be added to your
*.emacs* file at the required location.

;; clic-mode

       (hilit-set-mode-patterns

       
'clic-mode

       
'(("#.\*" nil comment)

         
("^$[a-zA-Z\\_]+" nil keyword)

       
))

       (global-set-key
[f3]       'hilit-rehighlight-buffer)

       (global-set-key
[f4]       'hilit-rehighlight-region)

Please also add these lines at the end of the
*.emacs* file

;; clic-mode definition

     (load-library "clic.el")

---

*Last modified: August 16, 1999*
