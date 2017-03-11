#!/usr/bin/env tclsh

global Ovr Par

# Location of executables
InitExecs

# Get the global configuration
source [GetIfile GlobalDefs.tcl]


# ------
# Wall spacing and stretching ratio
# ------
set Par(ds,wall) 0.001
set Par(sr,wall) 1.2
set Par(klayer) 3
# ------
# Surface stretching ratio
# ------
set Par(sr)      1.2
set Par(sr,slow) 1.1
# ------
# Main marching distance
# ------
set Par(md)              5.0
set Par(md,protub)       2.0
set Par(md,protub,small) 1.0
# ------
# Global spacing
# ------
set Par(ds,glb) [expr 1.0 * $GlobalScaleFactor]

# ------
# Reference length and area
# ------
set MIXSUR(refl) 1.0
set MIXSUR(refa) 3.14159
set MIXSUR(xmc) 0.0
set MIXSUR(ymc) 0.0
set MIXSUR(zmc) 0.0

# ------
# Default hypgen inputs
# ------
set Par(smu) 0.5
set default(zreg)  $Par(md)
set default(dz0)   $Par(ds,wall)
set default(dz1)   $Par(ds,glb)
set default(srmax) $Par(sr)
set default(ibcja) -10
set default(ibcjb) -10
set default(ibcka) -10
set default(ibckb) -10
set default(imeth) 2
set default(smu2)  $Par(smu)
set default(itsvol) 50
set default(timj)  0.0
set default(timk)  0.0
set default(epsss) 0.01
set default(iaxis) 1
set default(exaxis) 0.3
set default(volres) 0.3
set default(nsub) 2
set default(klayer) $Par(klayer)


# ------
# Volume grids created by other means
# ------
set bullet_body(nomakevol) 1

# ------
# Inputs for the OVERFLOW flow solver
# ------
set Ovr(incore) .T.
set Ovr(nsteps) 100
set Ovr(restrt) .F.
set Ovr(fmg)    .T.
set Ovr(fmgcyc) "1000, 1000, 0"
set Ovr(nglvl)  4
set Ovr(nfomo)  2
set Ovr(dtphys) 0.0
set Ovr(nitnwt) 0
set Ovr(walldist) 2

# Dummy freestream conditions
set Ovr(alpha)  0.0
set Ovr(beta)   0.0
set Ovr(fsmach) 0.8
set Ovr(rey) 10000.0
set Ovr(tinf) 450.0

# Specific inputs
if {$ovfi_inputs == "ssor"} {
    set Ovr(multig) .F.
    set Ovr(nqt) 205
    # Grid inputs
    set Ovr(default,iter)     1
    set Ovr(default,itert)    1
    set Ovr(default,iterc)    1
    set Ovr(default,itime)    4
    set Ovr(default,irhs)     5
    set Ovr(default,ilhs)     6
    set Ovr(default,ilimit)   3
    set Ovr(default,cflmin)   2.5
    set Ovr(default,cflmax)  10.0
    set Ovr(default,delta)    2.0
    set Ovr(default,fso)      3.0
    set Ovr(default,fsot)     3.0
    set Ovr(default,fsoc)     3.0
    set Ovr(default,visc)     .T.
    set Ovr(default,mut_limit) 0.0
    set Ovr(default,icc)      0
    set Ovr(default,itlhic)  10
    set Ovr(default,iupc)     2
}


# ------
# Xray spacing parameters
# ------
set Par(ds,xray,mini)  0.01
set Par(ds,xray,tiny)  0.02
set Par(ds,xray,vfine) 0.03
set Par(ds,xray,fine)  0.05
set Par(ds,xray,small) 0.1
set Par(ds,xray,big)   0.5

# Display group names, etc. in XRINFO blocks
set XRINFO(verbose) 1

# Hole-cutting parameters
set Ovr(dqual)   0.5
set Ovr(morfan)  1
set Ovr(norfan)  20
set Ovr(lfringe) 2


# ------
# MIXSUR components
# ------
set mixsurcomp(bullet_no_base) "
    bullet_cap
    bullet_body "



