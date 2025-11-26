#   Authors:
from pathlib import Path

import numpy as np

from ebeam import beam
from beamline import lattice
from schematic import draw_beamline
from excelElements import ExcelElements
from beamline import *
import sys
import time
import pandas as pd
from beamOptimizer import *
from AlgebraicOptimization import AlgebraicOpti
import sympy.plotting as plot
import sympy as sp

'''
-- Beam Properties initial conditions --
Ref: PhysRevAccelBeams.22.040704 (2019)

Transverse emittance normalized: 8 pi.mm.mrad
Energy spread between 0.1 to 0.5 %
Bunch length 1 ps (1 deg) to 2 ps (2 deg) 
'''
Energy = 40  # Electron beam energy (MeV)
f = 2856 * (10 ** 6)  # Accelerator RF frequency (Hz)
bunch_spread = 2  # std in pico-second
energy_std_percent = 0.5  # Energy standard deviation in percent of the mean (%)
h = 5 * (10 ** 9)  # Energy chirp from the linac (1/s)

epsilon_n = 8  # Transverse emittance normalized (pi.mm.mrad) epsilon_n = beta * gamma * epsilon_geometric
x_std = 0.8  # (mm)
y_std = 0.8  # (mm)

nb_particles = 1000

# Transverse phase space Initial conditions as a function of the normalized emittance and beam size
relat = lattice(1,fringeType=None)
relat.setE(E=Energy)
norm = relat.gamma * relat.beta
epsilon = epsilon_n / norm
x_prime_std = epsilon / x_std  # (mrad)
y_prime_std = epsilon / y_std  # (mrad)

# Longitudinal phase space normalization and conversion
tof_std = bunch_spread * (10 ** -9) * f  # (10 ** -3) (dToF / T)
gen_tof = np.random.normal(0, tof_std, size=(nb_particles, 1))
energy_std = energy_std_percent * 10 # (10 ** -3) (dW / W)

ebeam = beam()
beam_dist = ebeam.gen_6d_gaussian(0,
                                  [x_std,x_prime_std,y_std,y_prime_std,tof_std,energy_std],
                                  nb_particles)
tof_dist = beam_dist[:,4] / f  # (10 ** -3) s
beam_dist[:,5] += h * tof_dist

'''
Import UH beamline lattice from an Excel file
Generate beamline() elements
'''

pd.set_option('display.max_rows', None)
# Create beamline from Excel file
path3 = r"/Users/***/Desktop/Documents/FELsim"
path2 = r"C:\Users\NielsB\cernbox\Hawaii University\Beam dynamics\UH_FELxBeamDyn"
path1 = r"C:\Users\User\Documents\FELsim"
directory = Path(path2)
file_path = directory / 'Beamline_elements.xlsx'
#file_path = directory / 'Beamline_elements_2.xlsx'
excel = ExcelElements(file_path)
df = excel.get_dataframe()
#beamline
beamlineUH = excel.create_beamline()

'''
replace all dipole wedge elements with drift elements
'''
# for i in range(len(beamlineUH)):
#     if (isinstance(beamlineUH[i], dipole_wedge)):
#         drif = driftLattice(beamlineUH[i].length)
#         beamlineUH[i] = drif

# if len(beamline) >= 5:
#     beamline = beamline[:-5]
schem = draw_beamline()
beamtype = beamline()
line_UH = relat.changeBeamType("electron", Energy, beamlineUH)

segments = 98
line = line_UH[:segments]
opti = beamOptimizer(line, beam_dist)
acceptance = {"shape":'circle', "radius":1, "origin":[0,0]}
schem.plotBeamPositionTransform(beam_dist, line,0.01, plot=True, showIndice=True, defineLim=False, saveFig = 10, matchScaling=False, shape=acceptance, scatter=False)

# schem.plotBeamPositionTransform(beam_dist, line,0.01)


# variables = {1: ["I", "current", lambda num:num],
#              3: ["I2", "current", lambda num:num]}
# startPoint = {"I": {"bounds": (0,10), "start": 1},
#            "I2": {"bounds": (0, 10), "start": 1}}
# objectives = {8: [{"measure": ["x", "alpha"], "goal": -5, "weight":1},]
#                #  {"measure": ["y", "beta"], "goal": 1, "weight":1}],
#                # 9:[{"measure": ["x", "alpha"], "goal": 0, "weight":1},
#                #  {"measure": ["y", "alpha"], "goal": 0, "weight":1}]
#              }
# result = opti.calc("Nelder-Mead", variables, startPoint, objectives, plotBeam= True, printResults=True)
line[1].current =  0.9989681933
line[3].current = 1.044851479

variables = {10: ["I", "current", lambda num:num]}
startPoint = {"I": {"bounds": (0,10), "start": 1}}

# objectives = {10: [{"measure": ["x", "alpha"], "goal": 5, "weight": 1},
#                    {"measure": ["y", "alpha"], "goal": -5, "weight": 20}]}
objectives = {15: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 1}]}


result = opti.calc("Nelder-Mead", variables, startPoint, objectives, plotBeam=False, printResults=False, plotProgress=False)
print(line[10].current)  # 4! 3.2

# variables = {16: ["I", "current", lambda num:num],
#              18: ["I2", "current", lambda num:num],
#              20: ["I", "current", lambda num:num]}
# startPoint = {"I": {"bounds": (0,10), "start": 2},
#               "I2": {"bounds": (0, 10), "start": 2}}
# 
# objectives = {
#             # 24: [{"measure": ["x", "envelope"], "goal": 0, "weight": 1},
#             #        {"measure": ["y", "alpha"], "goal": 5, "weight": 10}],
#               16: [{"measure": ["y", "envelope"], "goal": 0.5, "weight": 100}],
#               20: [{"measure": ["y", "envelope"], "goal": 0.5, "weight": 100}]}
# 
# result = opti.calc("Nelder-Mead", variables, startPoint, objectives, plotBeam= True, printResults=True, plotProgress=True)

line[16].current = 2.4
line[18].current = 5.108214683


# variables = {20: ["I", "current", lambda num:num],
#              18: ["I2", "current", lambda num:num]}
# startPoint = {"I": {"bounds": (0,4), "start": 2},
#               "I2": {"bounds": (0, 4), "start": 2}}

# objectives = {
#                # 32: [{"measure": ["y", "envelope"], "goal": 0.7, "weight": 1}],
#                25: [{"measure": ["x", "envelope"], "goal": 0, "weight": 1}],
#                26: [{"measure": ["y", "alpha"], "goal":3.29, "weight": 1}],
#             #    27: [{"measure": ["y", "alpha"], "goal": -2.5, "weight": 1}]
#               }

# result = opti.calc("Nelder-Mead", variables, startPoint, objectives, plotBeam= True, printResults=True, plotProgress=True)
line[20].current = 3.142089844


# variables = {27: ["I", "current", lambda num:num]}
# startPoint = {"I": {"bounds": (0,10), "start": 2}}

# objectives = {
#                # 32: [{"measure": ["y", "envelope"], "goal": 0.7, "weight": 1}],
#               #  28: [{"measure": ["x", "envelope"], "goal": 0, "weight": 1}, 
#               #       {"measure": ["x", "alpha"], "goal": 0, "weight": 1}],
#                32: [{"measure": ["y", "alpha"], "goal": 20, "weight": 1} ]
                
#             #    27: [{"measure": ["y", "alpha"], "goal": -2.5, "weight": 1}]
#               }

# result = opti.calc("Nelder-Mead", variables, startPoint, objectives, plotBeam= True, printResults=True, plotProgress=True)

variables = {27: ["I", "current", lambda num:num]}
startPoint = {"I": {"bounds": (0,10), "start": 1}}

objectives = {32: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 1}]}


result = opti.calc("Nelder-Mead", variables, startPoint, objectives, plotBeam=False, printResults=False, plotProgress=False)
print(line[27].current)  # 4.694135938
#line[27].current = 4.694135938

variables = {
             37: ["I", "current", lambda num:num],
             35: ["I2", "current", lambda num:num],
             33: ["I3", "current", lambda num:num],
            #  39: ["I", "current", lambda num:num],
            #  41: ["I2", "current", lambda num:num],
            #  43: ["I3", "current", lambda num:num],
             }
             
startPoint = {"I": {"bounds": (0,10), "start": 2},
              "I2": {"bounds": (0, 10), "start": 2},
              "I3": {"bounds": (0, 10), "start": 2},
                }              

objectives = {            
               37: [
                     {"measure": ["x", "alpha"], "goal": 0, "weight": 1},
                    {"measure": ["y", "alpha"], "goal": 0, "weight": 1},
                    {"measure": ["x", "envelope"], "goal": 2.0, "weight": 1},
                    {"measure": ["y", "envelope"], "goal": 2.0, "weight": 1}
                    ]
              }

result = opti.calc("Nelder-Mead", variables, startPoint, objectives, plotBeam= False, printResults=True, plotProgress=True)
#"COBYLA"

I1 = 2.391954557  # 2.391954557    1.5820359927989482
I2 = 3.483629052  # 3.483629052    3.443475696137673
I3 = 1.464965657  # 1.464965657    1.8909887406970303

factor1 = I2 / (I1 + I3)
factor0 = I2 / I1
factor2 = I3 / I2

#line[33].current = I1
#line[35].current = I2
#line[37].current = I3

line[43].current = line[33].current
line[41].current = line[35].current
line[39].current = line[37].current


# Chromacity quad optimization
variables = {50: ["I", "current", lambda num:num]}
startPoint = {"I": {"bounds": (0,10), "start": 1}}

objectives = {55: [{"measure": ["x", "dispersion"], "goal": 0, "weight": 1}]}


result = opti.calc("Nelder-Mead", variables, startPoint, objectives, plotBeam=False, printResults=False, plotProgress=False)
print(line[50].current)

schem.plotBeamPositionTransform(beam_dist, line,0.01, showIndice=True)



'''
ALGEBRAIC OPTIMIZATION WORKPLACE BELOW
'''
# testLine = line_UH[16:21]
# alg = AlgebraicOpti()
# xvar = {
#          0: {"current": "I2"},
#          2: {"current": "I"},
#          4: {"current": "I2"}
# }
# alg.BIVARIATE_SEARCH_RANGE = 5
# finm = alg.findSymmetricObjective(testLine, xvar, beam_dist, plotBeam= [2,3], plotEquation = True)
# print(finm[0,0])
