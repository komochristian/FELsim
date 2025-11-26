#   Authors:
from pathlib import Path
from ebeam import beam
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
Beam Properties initialization
Energy spread between 0.1 to 0.5 %
Bunch length 1 ps (1 deg) to 2 ps (2 deg) 
'''
ebeam = beam()
f = 2856 * (10 ** 6)  # Accelerator RF frequency (Hz)

nb_particles = int(1e3)

bunch_spread = 1  # std in pico-second
tof_std = bunch_spread * (10 ** -9) * f  # (10 ** -3) (dToF / T)
energy_std = 1  # (10 ** -3) (dW / W)

x_std = 1  # (mm)
x_prime_std = 0.1  # (mrad)
y_std = 1  # (mm)
y_prime_std = 0.1  # (mrad)

beam_dist = ebeam.gen_6d_gaussian(0,
                                  [x_std,x_prime_std,y_std,y_prime_std,tof_std,energy_std],
                                  nb_particles)

'''
Import UH beamline lattice from an Excel file
Generate beamline() elements
'''

pd.set_option('display.max_rows', None)
# Create beamline from Excel file
path2 = r"C:\Users\NielsB\cernbox\Hawaii University\Beam dynamics\FELsim"
path1 = r"C:\Users\User\Documents\FELsim"
directory = Path(path2)
# file_path = directory / 'Beamline_elements.xlsx'
file_path = directory / 'Beamline_elements.xlsx'
excel = ExcelElements(file_path)
df = excel.get_dataframe()
#beamline
beamlineUH = excel.create_beamline()
# if len(beamline) >= 5:
#     beamline = beamline[:-5]
schem = draw_beamline()
beamtype = beamline()
line_UH = beamtype.changeBeamType(beamlineUH, "electron", 40)

print('Beamline nb of Elements: ' + str(len(line_UH)))
'''
Use optimization tools on segments of the UH beamline to the lattice values
'''
n = 66

truncatedB_line_optimization = line_UH[:-92]

#Initial conditions
twiss_aggregated_df = schem.plotBeamPositionTransform(beam_dist, truncatedB_line_optimization, 1, plot=False)
alpha_x = twiss_aggregated_df.at[twiss_aggregated_df.index[0], twiss_aggregated_df.keys()[1]]
alpha_y = twiss_aggregated_df.at[twiss_aggregated_df.index[1], twiss_aggregated_df.keys()[1]]
beta_x = twiss_aggregated_df.at[twiss_aggregated_df.index[0], twiss_aggregated_df.keys()[2]]
beta_y = twiss_aggregated_df.at[twiss_aggregated_df.index[1], twiss_aggregated_df.keys()[2]]

print('alpha x:' + str(alpha_x[-1]))
print('alpha y:' + str(alpha_y[-1]))
print('beta x:' + str(beta_x[-1]))
print('beta y:' + str(beta_y[-1]))


truncatedA_line_optimization = line_UH[:-75]

for beam in truncatedA_line_optimization: print(beam)
xvar = {1: {"current": "I"},
        3: {"current": "I2"},
        }
from AlgebraicOptimization import *
alg = AlgebraicOpti()
alg.BIVARIATE_SEARCH_RANGE = 1
alg.SEARCHINTERVAL = 0.5
alg.findSymmetricObjective(truncatedA_line_optimization, xvar, startParticles=beam_dist,plotBeam=[0,0])
twiss_aggregated_df = schem.plotBeamPositionTransform(beam_dist, truncatedA_line_optimization, 0.033, plot=True)


truncated_line1 = line_UH[:-88]
truncated_line2 = line_UH[:-30]

truncated_line1.append(qpfLattice(current = 3.816604, length=0.0889/2))
schem.plotBeamPositionTransform(beam_dist, truncated_line1, 1, plot=True)

j=0
for element in truncated_line1:
    print(str(j) + str(truncated_line1[j]))
    j += 1

vals = {
        1: ["Ia", "current", lambda num:num],
        3: ["Ib", "current", lambda num:num],
        }

starting = {"Ia": {"bounds": (0,10), "start": 2.6},
            "Ib": {"bounds": (0,10), "start": 4.6}}

objectives = {10:[{"measure": ["x", "alpha"],"goal":1,"weight":1},
                 {"measure": ["y", "alpha"],"goal":0,"weight":1}]}

test = beamOptimizer(truncated_line1, beam_dist)

result = test.calc("Nelder-Mead", vals, starting, objectives, plotProgress = True, plotBeam= True, printResults=True)
'''
‘Nelder-Mead’, ‘Powell’, ‘CG’, ‘BFGS’, ‘Newton-CG’, 
‘L-BFGS-B’, ‘TNC’, ‘COBYLA’, ‘COBYQA’, ‘SLSQP’
‘trust-constr’, 'dogleg’, ‘trust-ncg’, ‘trust-exact’ ‘trust-krylov’ 
'''


# Custom line
I = 3.56
sec_qpf = qpfLattice(current = I)
sec_qpd = qpdLattice(current = I)
sec_drift = driftLattice(0.02)
sec_dipole = dipole(length=0.0889, angle=1.5)
sec_wedge = dipole_wedge(length=0.01, angle=5, dipole_length=0.0889,dipole_angle=5)
sec_wedge2 = dipole_wedge(length=0.01, angle=5, dipole_length=0.0889,dipole_angle=5)

sec_qpd = qpdLattice(current = 3.56)
line_test = [sec_wedge, sec_drift,sec_wedge2]

'''
Plotting results
'''


schem.plotBeamPositionTransform(beam_dist, truncated_line2, 0.1)
