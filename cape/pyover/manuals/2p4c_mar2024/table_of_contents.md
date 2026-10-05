# **User's Manual for OVERFLOW 2.4**

**Version 2.4c March 2024**

**Robert H. Nichols University of Alabama, Birmingham Birmingham, AL 35294**

**Pieter G. Buning NASA Langley Research Center Hampton, VA 23681**

#### **Table of Contents**

#### **1. Introduction**

**1.1 Code History 1.2 OVERFLOW 2.4 Capabilities 1.3 Navier-Stokes Equation Implicit Solution Procedure 1.4 Low-Mach Number Preconditioning 1.5 Species Equations 1.6 Non-Dimensional Flow Variables References**

#### **2. Requesting and Compiling OVERFLOW 2.4**

**2.1 Obtaining OVERFLOW 2.4 2.2 Compiling OVERFLOW 2.4 2.3 Recommended Companion Software 2.4 OVERFLOW Mailing List References**

#### **3. OVERFLOW-Mode Operation**

**3.1 Required Input Files 3.2 Force/Moment Integration 3.3 NAMELIST Inputs 3.4 Turbulence Model Specifications 3.5 Variable and Multispecies 3.6 Boundary Condition Specification 3.7 Unsteady Flow Output Options 3.8 Initializing the Solution 3.9 Running the Code 3.10 Test Cases References**

#### **4. OVERFLOW-D-Mode Operation Without Grid Movement**

**4.1 Near-Body Grid Generation 4.2 Off-Body Grid Generation 4.3 X-Ray Specification 4.4 Grid Assembly using DCF 4.5 Grid Adaptation to Solution Error References**

## **5. OVERFLOW-D-Mode Operation With Grid Movement**

**5.1 Body Motion Non-Dimensionalization 5.2 Six-Degree-of-Freedom Motion Using &SIXINP 5.3 GMP XML Specification 5.4 GMP Prescribed Motion 5.5 GMP 6-DOF Motion 5.6 Using overgrid to Preview GMP Body Motion**

 **5.7 Simulating Body Collisions 5.8 Moving-Body Output Files 5.9 Grid Adaptation to Body Motion References**

## **6. Debugging Tools**

**6.1 Turbulence Model Diagnostics (DEBUG=1) 6.2 Time Step Diagnostics (DEBUG=2) 6.3 Residual Diagnostics (DEBUG=3) 6.4 Solution Adaption Information (DEBUG=4) 6.5 Reynolds Stress Turbulence Model Diagnostics (DEBUG=5)**

#### **7. Utility Codes**

**Appendix A. File Formats**

**Appendix B. NAMELIST Input**

**Appendix C. Boundary Condition Types**

**Appendix D. Viscous Region Specification**

**Appendix E. Rotorcraft Simulations (future)**