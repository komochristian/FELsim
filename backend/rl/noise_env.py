import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
from ebeam import beam
from beamline import *
import copy


class Noise_env(gym.Env):
    def __init__(self, beamline, monitor_indices, otr_locations, seed=None):
        super().__init__()
        self.beamline = beamline
        self.monitor_indices = monitor_indices
        self.otr_locations = otr_locations
        self.rng = np.random.default_rng(seed)
        self.ebeam = beam()
        self.local_particles = None

        if not isinstance(self.beamline, Beamline):
            self.beamline = Beamline(self.beamline)

        for loc in otr_locations:
            self.beamline.split_element(loc)



if __name__ == "__main__":
         
    dummy_beamline = [
            driftLattice(length = 0.5),
            qpdLattice(current = 1),
            driftLattice(length = 0.5),
            qpfLattice(current = 1),
            driftLattice(length = 0.1)
        ]
    
    env = Noise_env(dummy_beamline, [], [0.7, 1])

    new_beamline = env.beamline.beamline
    ebeam = beam()

    

    
        

