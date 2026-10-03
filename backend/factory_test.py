from felsim.cosyAdapter import COSYAdapter
from felsim.rftrackAdapter import RFTrackAdapter
from felsim.felsimAdapter import FELsimAdapter
from felsim.simulatorFactory import SimulatorFactory
from felsim.ebeam import beam

ebeam = beam()
particles = ebeam.gen_6d_gaussian(0,[1,1,1,1,0.1,100], 1000)
fel = FELsimAdapter(lattice_path="var/UH_FEL_beamline.json", beam_energy=40)
RFTrackAdapter(lattice_path="var/UH_FEL_beamline.json", beam_energy=40).simulate(particles)
COSYAdapter(lattice_path="var/UH_FEL_beamline.json", excel_path="beam_excel/Beamline_elements_3.xlsx", mode='transfer_matrix').simulate()
