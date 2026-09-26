import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from beamline import *


class QuadScanProcedure:
    def __init__(self, beamline: Beamline, otr_location, quad_indice, current_steps):
        self.beamline = beamline
        self.quad_indice = quad_indice
        self.otr_location = otr_location
        self.current_steps = current_steps
        self.otr_indice = -1

        for i, seg in enumerate(self.beamline.beamline):
            if seg.endPos == self.otr_location:
                self.otr_indice = i
                break

        self.precheck()



    def precheck(self):
        """Verify hardware availability and safety bounds before running."""
        # 1. Ensure monitor is downstream of the quadrupole

        if self.otr_location < self.quad_indice:
            raise ValueError(f"[PRECHECK FAILED] OTR location {self.otr_location} is not downstream of Quad {self.quad_indice}.")
            
        # 2. Check current safety limits (0 to 10 Amps)
        if np.any(self.current_steps < 0.0) or np.any(self.current_steps > 10.0):
            raise ValueError("[PRECHECK FAILED] Scan currents exceed safe physical limits [0, 10] A.")
