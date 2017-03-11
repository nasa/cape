
SurfGrids = bullet_cap.srf \
            bullet_body.srf \
            bullet_base.srf \
            bullet.xry

all: $(SurfGrids)

clobber:
	/bin/rm -f \
	bullet_cap.srf \
	bullet_body.srf \
	bullet_base.srf
	
$(SurfGrids): BuildBullet.tcl localinputs.tcl
	./BuildBullet.tcl
