import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from beamline import *
from ebeam import beam
from otr_diagnostic_v3 import OTRDiagnostic
import torch

class QuadScanObject:
    def __init__(self, beamline: Beamline, otr_location, quad_indice, current_steps, bunch_charge_C=60e-12):
        self.beamline = beamline
        self.quad_indice = quad_indice
        self.otr_location = otr_location
        self.current_steps = current_steps
        self.otr_indice = -1
        self.bunch_charge_C=bunch_charge_C

        for i, seg in enumerate(self.beamline.beamline):
            if round(seg.endPos, 8) == self.otr_location:
                self.otr_indice = i
                break

        self._precheck()



    def _precheck(self):
        """Verify hardware availability and safety bounds before running."""
        # 1. Ensure monitor is downstream of the quadrupole
        if not isinstance(self.beamline.beamline[self.quad_indice], qpdLattice) or not isinstance(self.beamline.beamline[self.quad_indice], qpfLattice):
            raise ValueError(f"[PRECHECK FAILED] Quad indice {self.quad_indice} is not a quadrupole object")

        if self.otr_location < self.quad_indice:
            raise ValueError(f"[PRECHECK FAILED] OTR location {self.otr_location} is not downstream of Quad {self.quad_indice}.")
            
        # 2. Check current safety limits (0 to 10 Amps)
        if np.any(self.current_steps < 0.0) or np.any(self.current_steps > 10.0):
            raise ValueError("[PRECHECK FAILED] Scan currents exceed safe physical limits [0, 10] A.")

    def scan_range(self, local_particles = None):
        photon_image = []
        ebeam = beam()

        if local_particles is None:
            local_particles = ebeam.gen_6d_gaussian(0, [0.5e-3, 1e-4, 0.5e-3, 1e-4, 0.1, 1e-3], 1000)

        for current_val in self.current_steps:
            particle_copy = local_particles
            photon_image.append(self._scan_single(particle_copy, current_val))

        return photon_image

    def _scan_single(self, local_particles, new_current_val):
        otr = OTRDiagnostic()

        for i, seg in enumerate(self.beamline.beamline):
            if i == self.quad_indice:
                seg.current = new_current_val
            
            local_particles = np.array(seg.useMatrice(local_particles))

            if i == self.otr_indice:
                photon_image = otr(local_particles, bunch_charge_C=self.bunch_charge_C)
                centroid_s, centroid_p, std_s, std_p = self._get_image_moments(otr, photon_image)

                return centroid_s, centroid_p, std_s, std_p

        raise ValueError("[SCAN ERROR] Unable to fetch OTR photon image")

    def _get_image_moments(
        self, otr_model, photon_image: torch.Tensor, in_mm: bool = True
    ) -> tuple[float, float, float, float]:
        """Calculate centroid and RMS width (std) from an OTR photon image.

        Args:
            otr_model: The OTRDiagnostic instance.
            photon_image: Tensor of shape (n_y, n_x) containing photon counts per
            pixel.
            in_mm: If True, returns values in millimeters. If False, returns meters.

        Returns:
            tuple: (centroid_s, centroid_p, std_s, std_p)
        """
        # 1. Fetch physical camera axes S and P matching photon_image's device
        s_axis, p_axis = otr_model.sensor_axes(device=photon_image.device)

        # 2. Total photon count and 1D projections (marginals)
        total = photon_image.sum()
        if total == 0:
            return 0.0, 0.0, 0.0, 0.0

        marginal_s = photon_image.sum(dim=0)  # Sum down P rows -> S profile
        marginal_p = photon_image.sum(dim=1)  # Sum across S cols -> P profile

        # 3. Calculate centroids (1st spatial moment)
        centroid_s = (marginal_s * s_axis).sum() / total
        centroid_p = (marginal_p * p_axis).sum() / total

        # 4. Calculate standard deviations / RMS (2nd central moment)
        std_s = torch.sqrt((marginal_s * (s_axis - centroid_s) ** 2).sum() / total)
        std_p = torch.sqrt((marginal_p * (p_axis - centroid_p) ** 2).sum() / total)

        scale = 1e3 if in_mm else 1.0

        return (
            (centroid_s * scale).item(),
            (centroid_p * scale).item(),
            (std_s * scale).item(),
            (std_p * scale).item(),
        )