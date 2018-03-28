-0.2, 400.0, 400.0, -1.0E+6, 0, -100.0    FSMACH,ALPHA,BETA,REY,GAMINF,TINF
1                       NREF
1.0, 3.14159, 0.0, 0.0, 0.0       REFL,REFA,XMC,YMC,ZMC
3                       NSURF

2, 1                 NRSUB, NREFS  (bullet_cap Family)
1 3 1 18 1 -1 1  1  NG,IBDIR,JS,JE,KS,KE,LS,LE,IFAM  bullet_body (1)
2 3 1 -1 1 -1 1  1  NG,IBDIR,JS,JE,KS,KE,LS,LE,IFAM  bullet_cap (2)
0         NPRI

1, 1                 NRSUB, NREFS  (bullet_body Family)
1 3 18 -13 1 -1 1  1  NG,IBDIR,JS,JE,KS,KE,LS,LE,IFAM  bullet_body (1)
0         NPRI

2, 1                 NRSUB, NREFS  (bullet_base Family)
1 3 -13 -1 1 -1 1  1  NG,IBDIR,JS,JE,KS,KE,LS,LE,IFAM  bullet_body (1)
3 3 1 -1 1 -1 1  1  NG,IBDIR,JS,JE,KS,KE,LS,LE,IFAM  bullet_base (2)
0         NPRI

5          NCOMP

ALL
3, 1         NIS, IREFC
1 2 3                     (surface numbers)

bullet_no_base
2, 1         NIS, IREFC
1 2                     (surface numbers)

bullet_cap
1, 1             NIS, IREFC
1              (surface numbers)

bullet_body
1, 1             NIS, IREFC
2              (surface numbers)

bullet_base
1, 1             NIS, IREFC
3              (surface numbers)

