from beamOptimizer import beamOptimizer
from ebeam import beam
from schematic import draw_beamline
from beamline import *
import numpy as np

ebeam = beam()
schem = draw_beamline()

Ia = 3.56
sec1 = driftLattice(0.5)
sec2 = qpfLattice(current = Ia)
sec3 = driftLattice(0.25)
sec4 = qpdLattice(current = Ia)
sec5 = driftLattice(0.25)
sec6 = qpfLattice(current = Ia)
sec7 = driftLattice(0.25)
sec8 = qpdLattice(current = Ia)
sec9 = driftLattice(0.50)
sec10 = dipole(length=0.0889, angle=1.5)
line = [sec1,sec2,sec3,sec4,sec5,sec6,sec7,sec8,sec9,sec10]

beamtype = beamline()
pBeam = beamtype.changeBeamType(line, "electron", 40)

beam_dist = ebeam.gen_6d_gaussian(0,[1,0.1,1,0.1,2.856,1],1000)
schem.plotBeamPositionTransform(beam_dist, pBeam, 0.01, spacing = 10)

vals = {
        1: ["I", "current", lambda num:num],
        3: ["I2", "current", lambda num:num],
        5: ["I", "current", lambda num:num],
        7: ["I2", "current", lambda num:num],
        8: ["L", "length", lambda num:num]
        }

starting = {"I": {"bounds": (0,10), "start": 3},
            "I2": {"bounds": (0,10), "start": 3}}

objectives = {9:[{"measure": ["y", "std"],"goal":1,"weight":1},
                 {"measure": ["x", "std"],"goal":1,"weight":1}]}

test = beamOptimizer(line, beam_dist)
for i in line: print(i)
result = test.calc("Nelder-Mead", vals, starting, objectives, plotProgress = True, plotBeam= True)
for i in line: print(i)







# test = beamOptimizer(line, (1,3,5),1,1, [A,B], "Nelder-Mead", matrixVariables)

