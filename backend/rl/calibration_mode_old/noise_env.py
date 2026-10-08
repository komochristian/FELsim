import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
from ebeam import beam
from beamline import *
import copy


class Noise_env(gym.Env):
    def __init__(self, beamline, otr_locations, seed=None):
        super().__init__()
        self.beamline = beamline
        self.otr_locations = otr_locations
        self.rng = np.random.default_rng(seed)
        self.ebeam = beam()
        self.local_particles = None

        if not isinstance(self.beamline, Beamline):
            self.beamline = Beamline(self.beamline)

        for loc in otr_locations:
            self.beamline.split_element(loc)

    def _get_obs(self):
        # obs space: list of current values, (mapped to) list OTR statistics of closest downstream otr, distance to downstream OTR, 
        # scalar OTR stats of closest upstream otr,distance to upstream otr, upstream otr current
        pass

    def step(self):
        # step: call script, for a each quad, get obs space for each quad, then guess noise for each quad
        pass



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

    

    
        

