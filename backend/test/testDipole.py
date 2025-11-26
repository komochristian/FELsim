from ebeam import beam
from schematic import draw_beamline
from beamline import *
import numpy as np

ebeam = beam()
schem = draw_beamline()

I = 2.28
sec1 = driftLattice(0.01)
sec2 = dipole_wedge(length=0.01, angle=0, dipole_length=0.0889,dipole_angle=1.5)
sec3 = dipole(length=0.0889, angle=1.5)
sec4 = dipole_wedge(length=0.01, angle=1.5, dipole_length=0.0889,dipole_angle=1.5)
sec5 = driftLattice(1.0)
sec6 = dipole_wedge(length=0.01, angle=0.75, dipole_length=0.0889,dipole_angle=1.5)
sec7 = dipole(length=0.0889, angle=1.5)
sec8 = dipole_wedge(length=0.01, angle=0.75, dipole_length=0.0889,dipole_angle=1.5)
sec9 = driftLattice(1.0)
sec10 = dipole_wedge(length=0.01, angle=2.0, dipole_length=0.04064,dipole_angle=4.0)
sec11 = dipole(length=0.04064, angle=4.0)
sec12 = dipole_wedge(length=0.01, angle=2.0, dipole_length=0.04064,dipole_angle=4.0)
sec13 = driftLattice(1.0)
sec14 = dipole_wedge(length=0.01, angle=2.00, dipole_length=0.04064,dipole_angle=4.0)
sec15 = dipole(length=0.04064, angle=4.0)
sec16 = dipole_wedge(length=0.01, angle=2.0, dipole_length=0.04064,dipole_angle=4.0)
sec17 = driftLattice(0.01)

line = [sec1,sec2,sec3, sec4,sec5,sec6,sec7,sec8,sec9,sec10,sec11,sec12,sec13,sec14,sec15,sec16,sec17]

#line = [sec1,sec11,sec17]

#line = [sec1]

beam_dist = ebeam.gen_6d_gaussian(0,[1,1,1,1,0.1,100],10000)

schem.plotBeamPositionTransform(beam_dist, line, 0.01)
