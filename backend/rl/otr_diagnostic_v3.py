"""Differentiable photon-level OTR diagnostic for the UH FELICIA geometry.

The public interface is intentionally small::

    otr = OTRDiagnostic()
    photon_image = otr(particles_6d, bunch_charge_C=50e-12)

``photon_image`` has shape ``(n_y, n_x)`` and units of photons incident on
each physical camera pixel per bunch.  Rows are the optical P coordinate and
columns are the optical S coordinate.  Camera quantum efficiency and all
electronic camera effects are deliberately outside this module.

Default particle layout
-----------------------
The default six columns are ``(x, x_prime, y, y_prime, z, delta)`` in SI
units.  ``delta`` is relative momentum deviation from the configured
reference kinetic energy.  The longitudinal coordinate is accepted but is
ignored by the present incoherent-OTR model.  The alternate notebook-like
layout ``(x, y, x_prime, y_prime, z, delta)`` is also supported.

Physics and numerical model
---------------------------
The implementation is a reusable form of the validated UH notebook.  It
retains the Aumeyr source, exact screen intersection and polarization mapping,
scaled Fresnel/Bluestein propagation, a circular pupil, Ginzburg-Frank
absolute photon normalization, the SiC s/p Fresnel response, the validated
first-pass 1/lambda broadband model, and physical sensor-pixel integration.

The normal forward path deposits equal-weight macroparticle image centers on
the oversampled camera grid, convolves them with the cached oversampled
single-electron PSF, and only then integrates onto physical camera pixels.
This preserves gradients with respect to particle coordinates almost
everywhere (bilinear deposition is piecewise differentiable).  The reference
response is rebuilt automatically for every new module
configuration/device/dtype.  The much slower particle-resolved optical
calculation is retained in ``particle_resolved_image`` for small validation
samples.

At the validated defaults, the spatial PSF is propagated at 550 nm.  SiC
Fresnel factors are applied to the absolute collection-yield calculation,
matching the reference notebook.  The broadband spatial shape is held fixed
while the photon spectrum is integrated as 1/lambda from 400 to 1000 nm.
The UH lens and camera line is fixed to the nominal 45-degree screen geometry;
changing ``screen_angle_deg`` rotates only the physical screen and its OTR.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "NumericalGrid",
    "OTRDiagnostic",
    "OTRScreen",
    "OpticalTransport",
    "SensorGeometry",
    "Spectrum",
]


@dataclass(frozen=True)
class OTRScreen:
    """Physical OTR-screen parameters; the optical line remains at 45 degrees."""

    screen_angle_deg: float = 45.0
    refractive_index_n: float = 2.6480
    extinction_coefficient_k: float = 0.0


@dataclass(frozen=True)
class Spectrum:
    """Reference wavelength and first-pass broadband integration range."""

    reference_wavelength_m: float = 550e-9
    wavelength_min_m: float = 400e-9
    wavelength_max_m: float = 1000e-9


@dataclass(frozen=True)
class OpticalTransport:
    """Thin-lens optical transport in the fixed UH (S, P) basis."""

    focal_length_m: float = 150e-3
    screen_to_lens_m: float = 183.33333333333334e-3
    lens_to_sensor_m: float = 825e-3
    pupil_radius_m: float = 25.4e-3
    optical_transmission: float = 0.90
    lens_offset_s_m: float = 0.0
    lens_offset_p_m: float = 0.0
    sensor_offset_s_m: float = 0.0
    sensor_offset_p_m: float = 0.0


@dataclass(frozen=True)
class SensorGeometry:
    """Physical camera-pixel geometry."""

    pixel_pitch_m: float = 5.86e-6
    n_x: int = 1920
    n_y: int = 1200


@dataclass(frozen=True)
class NumericalGrid:
    """Numerical settings, deliberately separate from physical parameters."""

    source_grid_size: int = 1024
    source_half_width_m: float = 343.797e-6
    lens_grid_size: int = 1024
    lens_half_width_m: float = 26.0e-3
    psf_oversampling: int = 4
    psf_oversampled_size: int = 512
    spectrum_samples: int = 601
    dtype: str = "float64"


def _coerce_config(cls: type[Any], value: Any) -> Any:
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls(**value)
    raise TypeError(f"{cls.__name__} must be a {cls.__name__} or a mapping")


class _ModifiedBesselK01(torch.autograd.Function):
    """K0/K1 wrapper with analytic first derivatives for autograd."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k0 = torch.special.modified_bessel_k0(x)
        k1 = torch.special.modified_bessel_k1(x)
        ctx.save_for_backward(x, k0, k1)
        return k0, k1

    @staticmethod
    def backward(
        ctx: Any,
        grad_k0: torch.Tensor | None,
        grad_k1: torch.Tensor | None,
    ) -> tuple[torch.Tensor]:
        x, k0, k1 = ctx.saved_tensors
        grad_x = torch.zeros_like(x)
        if grad_k0 is not None:
            grad_x = grad_x - grad_k0 * k1
        if grad_k1 is not None:
            grad_x = grad_x + grad_k1 * (-k0 - k1 / x)
        return (grad_x,)


def _modified_bessel_k01(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _ModifiedBesselK01.apply(x)


class OTRDiagnostic(nn.Module):
    """UH OTR diagnostic ending at photons incident on camera pixels.

    Parameters are grouped into physical dataclasses plus a separate numerical
    dataclass.  Each group may instead be supplied as a dictionary, e.g.::

        OTRDiagnostic(
            optical_transport={"pupil_radius_m": 20e-3},
            sensor_geometry={"n_x": 1024, "n_y": 1024},
        )

    Args:
        otr_screen: :class:`OTRScreen` or matching mapping.
        spectrum: :class:`Spectrum` or matching mapping.
        optical_transport: :class:`OpticalTransport` or matching mapping.
        sensor_geometry: :class:`SensorGeometry` or matching mapping.
        numerical: :class:`NumericalGrid` or matching mapping.
        reference_kinetic_energy_MeV: Reference kinetic energy used to convert
            input ``delta = Delta p / p0`` to particle gamma.  UH default:
            42.2 MeV.
        particle_layout: ``"x_xp_y_yp_z_delta"`` (default) or
            ``"x_y_xp_yp_z_delta"``.
    """

    electron_charge_C = 1.602176634e-19
    electron_rest_energy_MeV = 0.51099895
    fine_structure_constant = 1.0 / 137.035999084
    nominal_screen_angle_deg = 45.0

    def __init__(
        self,
        *,
        otr_screen: OTRScreen | Mapping[str, Any] | None = None,
        spectrum: Spectrum | Mapping[str, Any] | None = None,
        optical_transport: OpticalTransport | Mapping[str, Any] | None = None,
        sensor_geometry: SensorGeometry | Mapping[str, Any] | None = None,
        numerical: NumericalGrid | Mapping[str, Any] | None = None,
        reference_kinetic_energy_MeV: float = 42.2,
        particle_layout: str = "x_xp_y_yp_z_delta",
    ) -> None:
        super().__init__()
        self.otr_screen = _coerce_config(OTRScreen, otr_screen)
        self.spectrum = _coerce_config(Spectrum, spectrum)
        self.optical_transport = _coerce_config(OpticalTransport, optical_transport)
        self.sensor_geometry = _coerce_config(SensorGeometry, sensor_geometry)
        self.numerical = _coerce_config(NumericalGrid, numerical)
        self.reference_kinetic_energy_MeV = float(reference_kinetic_energy_MeV)
        self.particle_layout = particle_layout
        self._validate_configuration()

        self._response_cache: dict[tuple[str, torch.dtype], dict[str, Any]] = {}
        self._response_build_count = 0

    @classmethod
    def uh_defaults(cls) -> dict[str, Any]:
        """Return the grouped validated UH defaults as ordinary dictionaries."""
        return {
            "otr_screen": asdict(OTRScreen()),
            "spectrum": asdict(Spectrum()),
            "optical_transport": asdict(OpticalTransport()),
            "sensor_geometry": asdict(SensorGeometry()),
            "numerical": asdict(NumericalGrid()),
            "reference_kinetic_energy_MeV": 42.2,
            "particle_layout": "x_xp_y_yp_z_delta",
        }

    @property
    def magnification(self) -> float:
        """Derived signed geometrical magnification; never independently set."""
        return -(
            self.optical_transport.lens_to_sensor_m
            / self.optical_transport.screen_to_lens_m
        )

    @property
    def response_build_count(self) -> int:
        """Number of reference-response builds (useful in validation tests)."""
        return self._response_build_count

    def clear_response_cache(self) -> None:
        """Discard cached reference responses; the next call rebuilds them."""
        self._response_cache.clear()

    def _validate_configuration(self) -> None:
        s = self.otr_screen
        sp = self.spectrum
        o = self.optical_transport
        c = self.sensor_geometry
        n = self.numerical

        if not 0.0 < s.screen_angle_deg < 90.0:
            raise ValueError("screen_angle_deg must lie between 0 and 90 degrees")
        if s.refractive_index_n <= 0.0 or s.extinction_coefficient_k < 0.0:
            raise ValueError("SiC n must be positive and k must be nonnegative")
        if not (
            0.0
            < sp.wavelength_min_m
            <= sp.reference_wavelength_m
            <= sp.wavelength_max_m
        ):
            raise ValueError(
                "wavelength_min <= reference <= wavelength_max is required"
            )
        if (
            min(
                o.focal_length_m,
                o.screen_to_lens_m,
                o.lens_to_sensor_m,
                o.pupil_radius_m,
            )
            <= 0.0
        ):
            raise ValueError(
                "focal length, distances, and pupil radius must be positive"
            )
        if not 0.0 <= o.optical_transmission <= 1.0:
            raise ValueError("optical_transmission must lie in [0, 1]")
        if c.pixel_pitch_m <= 0.0 or c.n_x < 2 or c.n_y < 2:
            raise ValueError("sensor pitch must be positive and n_x,n_y must be >= 2")
        if self.reference_kinetic_energy_MeV <= 0.0:
            raise ValueError("reference kinetic energy must be positive")
        if self.particle_layout not in {
            "x_xp_y_yp_z_delta",
            "x_y_xp_yp_z_delta",
        }:
            raise ValueError("unsupported particle_layout")
        for name, value in {
            "source_grid_size": n.source_grid_size,
            "lens_grid_size": n.lens_grid_size,
            "psf_oversampling": n.psf_oversampling,
            "psf_oversampled_size": n.psf_oversampled_size,
            "spectrum_samples": n.spectrum_samples,
        }.items():
            if int(value) != value or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        if n.psf_oversampled_size % n.psf_oversampling:
            raise ValueError(
                "psf_oversampled_size must be divisible by psf_oversampling"
            )
        kernel_size = n.psf_oversampled_size // n.psf_oversampling
        if kernel_size > min(c.n_x, c.n_y):
            raise ValueError("the physical PSF kernel must fit inside the sensor")
        if n.dtype not in {"float32", "float64"}:
            raise ValueError("numerical.dtype must be 'float32' or 'float64'")

    @property
    def _real_dtype(self) -> torch.dtype:
        return torch.float64 if self.numerical.dtype == "float64" else torch.float32

    @property
    def _complex_dtype(self) -> torch.dtype:
        return (
            torch.complex128 if self._real_dtype == torch.float64 else torch.complex64
        )

    def _tensor(self, value: Any, device: torch.device) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=self._real_dtype)
        return torch.as_tensor(value, device=device, dtype=self._real_dtype)

    @staticmethod
    def _centered_axis_from_width(
        n: int,
        half_width: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spacing = torch.as_tensor(2.0 * half_width / n, device=device, dtype=dtype)
        axis = (torch.arange(n, device=device, dtype=dtype) - n / 2.0 + 0.5) * spacing
        return axis, spacing

    @staticmethod
    def _centered_axis_from_pitch(
        n: int,
        pitch: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return (torch.arange(n, device=device, dtype=dtype) - n / 2.0 + 0.5) * pitch

    @staticmethod
    def _axis_spacing(axis: torch.Tensor) -> torch.Tensor:
        if axis.numel() < 2:
            raise ValueError("axis must contain at least two points")
        return axis[1] - axis[0]

    @staticmethod
    def _next_power_of_two(n: int) -> int:
        return 1 << (int(n) - 1).bit_length()

    def _scaled_dft_1d(
        self,
        field: torch.Tensor,
        alpha: torch.Tensor,
        n_output: int,
        *,
        dim: int,
    ) -> torch.Tensor:
        original_dim = dim % field.ndim
        x = torch.movedim(field, original_dim, -1)
        if not torch.is_complex(x):
            x = x.to(self._complex_dtype)

        n_input = x.shape[-1]
        n_output = int(n_output)
        n = torch.arange(n_input, dtype=self._real_dtype, device=x.device)
        m = torch.arange(n_output, dtype=self._real_dtype, device=x.device)
        center_in = (n_input - 1.0) / 2.0
        center_out = (n_output - 1.0) / 2.0

        x = x * torch.exp(1j * 2.0 * math.pi * alpha * center_out * n)
        chirped_input = x * torch.exp(-1j * math.pi * alpha * n**2)

        n_fft = self._next_power_of_two(n_input + n_output - 1)
        input_padded = F.pad(chirped_input, (0, n_fft - n_input))
        positive_lags = torch.arange(n_output, dtype=self._real_dtype, device=x.device)
        negative_lags = torch.arange(
            -(n_input - 1), 0, dtype=self._real_dtype, device=x.device
        )
        kernel = torch.cat(
            [
                torch.exp(1j * math.pi * alpha * positive_lags**2),
                torch.zeros(
                    n_fft - (n_input + n_output - 1),
                    dtype=self._complex_dtype,
                    device=x.device,
                ),
                torch.exp(1j * math.pi * alpha * negative_lags**2),
            ]
        )
        convolution = torch.fft.ifft(
            torch.fft.fft(input_padded, dim=-1) * torch.fft.fft(kernel, dim=-1),
            dim=-1,
        )[..., :n_output]

        result = convolution * torch.exp(-1j * math.pi * alpha * m**2)
        result = (
            result
            * torch.exp(1j * 2.0 * math.pi * alpha * center_in * m)
            * torch.exp(-1j * 2.0 * math.pi * alpha * center_in * center_out)
        )
        return torch.movedim(result, -1, original_dim)

    def _scaled_fresnel(
        self,
        field: torch.Tensor,
        x_input: torch.Tensor,
        y_input: torch.Tensor,
        x_output: torch.Tensor,
        y_output: torch.Tensor,
        wavelength: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        k = 2.0 * math.pi / wavelength
        dx0 = self._axis_spacing(x_input)
        dy0 = self._axis_spacing(y_input)
        dx1 = self._axis_spacing(x_output)
        dy1 = self._axis_spacing(y_output)

        y0, x0 = torch.meshgrid(y_input, x_input, indexing="ij")
        transformed = field * torch.exp(1j * k * (x0**2 + y0**2) / (2.0 * z))
        transformed = self._scaled_dft_1d(
            transformed,
            dx0 * dx1 / (wavelength * z),
            x_output.numel(),
            dim=-1,
        )
        transformed = self._scaled_dft_1d(
            transformed,
            dy0 * dy1 / (wavelength * z),
            y_output.numel(),
            dim=-2,
        )
        transformed = transformed * dx0 * dy0

        y1, x1 = torch.meshgrid(y_output, x_output, indexing="ij")
        output_chirp = torch.exp(1j * k * (x1**2 + y1**2) / (2.0 * z))
        return output_chirp * transformed / (1j * wavelength * z)

    def _reference_gamma_beta(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma0 = self._tensor(
            1.0 + self.reference_kinetic_energy_MeV / self.electron_rest_energy_MeV,
            device,
        )
        p0_mc = torch.sqrt(gamma0**2 - 1.0)
        beta0 = p0_mc / gamma0
        return gamma0, beta0, p0_mc

    def _fixed_geometry(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Return the actual screen normal and fixed UH optical basis.

        ``screen_angle_deg`` describes only the physical screen.  The lens and
        camera remain on the nominal 45-degree reflected-light line, so their
        propagation basis must not follow a changed screen angle.
        """
        ex = self._tensor([1.0, 0.0, 0.0], device)
        ey = self._tensor([0.0, 1.0, 0.0], device)
        ez = self._tensor([0.0, 0.0, 1.0], device)

        screen_angle = self._tensor(
            math.radians(self.otr_screen.screen_angle_deg), device
        )
        normal = torch.sin(screen_angle) * ex + torch.cos(screen_angle) * ez

        nominal_angle = self._tensor(
            math.radians(self.nominal_screen_angle_deg), device
        )
        nominal_normal = (
            torch.sin(nominal_angle) * ex + torch.cos(nominal_angle) * ez
        )
        beta0_hat = ez
        k0_hat = (
            beta0_hat
            - 2.0 * torch.dot(beta0_hat, nominal_normal) * nominal_normal
        )
        k0_hat = k0_hat / torch.linalg.vector_norm(k0_hat)
        s0_hat = torch.linalg.cross(nominal_normal, beta0_hat, dim=0)
        s0_hat = s0_hat / torch.linalg.vector_norm(s0_hat)
        p0_hat = torch.linalg.cross(k0_hat, s0_hat, dim=0)
        p0_hat = p0_hat / torch.linalg.vector_norm(p0_hat)
        return {
            "ex": ex,
            "ey": ey,
            "ez": ez,
            "normal": normal,
            "k0_hat": k0_hat,
            "s0_hat": s0_hat,
            "p0_hat": p0_hat,
        }

    def _particle_gamma(
        self, delta: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        _, _, p0_mc = self._reference_gamma_beta(device)
        p_mc = p0_mc * (1.0 + delta)
        return torch.sqrt(1.0 + p_mc**2)

    def _unpack_particles(self, particles_6d: Any) -> tuple[torch.Tensor, ...]:
        if isinstance(particles_6d, torch.Tensor):
            device = particles_6d.device
        else:
            device = torch.device("cpu")
        particles = self._tensor(particles_6d, device)
        if particles.ndim != 2 or particles.shape[1] != 6:
            raise ValueError("particles_6d must have shape (n_macroparticles, 6)")
        if particles.shape[0] == 0:
            raise ValueError("particles_6d must contain at least one macroparticle")
        if self.particle_layout == "x_xp_y_yp_z_delta":
            x, xp, y, yp, longitudinal, delta = particles.unbind(dim=1)
        else:
            x, y, xp, yp, longitudinal, delta = particles.unbind(dim=1)
        return particles, x, y, xp, yp, longitudinal, delta

    def _particle_geometry(
        self, xp: torch.Tensor, yp: torch.Tensor, device: torch.device
    ) -> dict[str, torch.Tensor]:
        fixed = self._fixed_geometry(device)
        beta_hat = torch.stack((xp, yp, torch.ones_like(xp)))
        beta_hat = beta_hat / torch.linalg.vector_norm(beta_hat)
        normal = fixed["normal"]
        k_hat = beta_hat - 2.0 * torch.dot(beta_hat, normal) * normal
        k_hat = k_hat / torch.linalg.vector_norm(k_hat)
        s_hat = torch.linalg.cross(normal, beta_hat, dim=0)
        s_hat = s_hat / torch.linalg.vector_norm(s_hat)
        p_hat = torch.linalg.cross(k_hat, s_hat, dim=0)
        p_hat = p_hat / torch.linalg.vector_norm(p_hat)
        cos_2theta0 = torch.clamp(torch.dot(beta_hat, k_hat), -1.0, 1.0)
        theta0 = 0.5 * torch.acos(cos_2theta0)
        return {
            **fixed,
            "beta_hat": beta_hat,
            "k_hat": k_hat,
            "s_hat": s_hat,
            "p_hat": p_hat,
            "theta0": theta0,
            "u_s": torch.dot(k_hat, fixed["s0_hat"]),
            "u_p": torch.dot(k_hat, fixed["p0_hat"]),
        }

    def _screen_hit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        xp: torch.Tensor,
        yp: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fixed = self._fixed_geometry(device)
        normal = fixed["normal"]
        denominator = normal[0] * xp + normal[1] * yp + normal[2]
        z_hit = -(normal[0] * x + normal[1] * y) / denominator
        r_hit = torch.stack((x + xp * z_hit, y + yp * z_hit, z_hit))
        return (
            torch.dot(r_hit, fixed["s0_hat"]),
            torch.dot(r_hit, fixed["p0_hat"]),
        )

    def _screen_hits_vectorized(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        xp: torch.Tensor,
        yp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fixed = self._fixed_geometry(x.device)
        normal = fixed["normal"]
        denominator = normal[0] * xp + normal[1] * yp + normal[2]
        z_hit = -(normal[0] * x + normal[1] * y) / denominator
        r_hit = torch.stack((x + xp * z_hit, y + yp * z_hit, z_hit), dim=1)
        return (
            r_hit @ fixed["s0_hat"],
            r_hit @ fixed["p0_hat"],
        )

    def _aumeyr_source(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        xp: torch.Tensor,
        yp: torch.Tensor,
        gamma: torch.Tensor,
        s_source: torch.Tensor,
        p_source: torch.Tensor,
        rho_floor: torch.Tensor,
        wavelength: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = s_source.device
        beta = torch.sqrt(1.0 - gamma.reciprocal() ** 2)
        wave_number = 2.0 * math.pi / wavelength
        q = 2.0 * math.pi / (beta * gamma * wavelength)
        geometry = self._particle_geometry(xp, yp, device)
        s_center, p_center = self._screen_hit(x, y, xp, yp, device)

        p_grid, s_grid = torch.meshgrid(p_source, s_source, indexing="ij")
        ds = s_grid - s_center
        dp = p_grid - p_center
        rho = torch.sqrt(ds**2 + dp**2)
        rho_eff = torch.clamp(rho, min=rho_floor)
        k0, k1 = _modified_bessel_k01(q * rho_eff)

        intrinsic_phase = torch.exp(
            1j * wave_number * dp * torch.tan(geometry["theta0"]) * (1.0 - 1.0 / beta)
        )
        e_s_local = (ds / rho_eff) * k1 * intrinsic_phase
        e_p_local = (
            (dp / rho_eff) * k1 - 1j * k0 * torch.tan(geometry["theta0"]) / gamma
        ) * intrinsic_phase

        tilt_phase = torch.exp(
            1j * wave_number * (geometry["u_s"] * ds + geometry["u_p"] * dp)
        )
        e_s_local = e_s_local * tilt_phase
        e_p_local = e_p_local * tilt_phase

        s0_hat = geometry["s0_hat"]
        p0_hat = geometry["p0_hat"]
        ss = torch.dot(geometry["s_hat"], s0_hat)
        sp = torch.dot(geometry["s_hat"], p0_hat)
        ps = torch.dot(geometry["p_hat"], s0_hat)
        pp = torch.dot(geometry["p_hat"], p0_hat)
        e_s0 = ss * e_s_local + ps * e_p_local
        e_p0 = sp * e_s_local + pp * e_p_local

        gamma0, _, _ = self._reference_gamma_beta(device)
        relative_amplitude = gamma0 / gamma
        return relative_amplitude * e_s0, relative_amplitude * e_p0

    def _thin_lens_transfer(
        self,
        s_axis: torch.Tensor,
        p_axis: torch.Tensor,
        wavelength: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        o = self.optical_transport
        p_grid, s_grid = torch.meshgrid(p_axis, s_axis, indexing="ij")
        radius_squared = (s_grid - o.lens_offset_s_m) ** 2 + (
            p_grid - o.lens_offset_p_m
        ) ** 2
        pupil = (radius_squared <= o.pupil_radius_m**2).to(self._real_dtype)
        phase = torch.exp(
            -1j
            * (2.0 * math.pi / wavelength)
            * radius_squared
            / (2.0 * o.focal_length_m)
        )
        return pupil.to(self._complex_dtype) * phase, pupil

    def _fresnel_coefficients(
        self, incidence_angle: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        complex_index = torch.complex(
            self._tensor(self.otr_screen.refractive_index_n, incidence_angle.device),
            self._tensor(
                self.otr_screen.extinction_coefficient_k, incidence_angle.device
            ),
        )
        sin_i = torch.sin(incidence_angle).to(self._complex_dtype)
        cos_i = torch.cos(incidence_angle).to(self._complex_dtype)
        sin_t = sin_i / complex_index
        cos_t = torch.sqrt(1.0 - sin_t**2)
        rs = (cos_i - complex_index * cos_t) / (cos_i + complex_index * cos_t)
        rp = (complex_index * cos_i - cos_t) / (complex_index * cos_i + cos_t)
        return rs, rp

    def _image_moments(
        self,
        image: torch.Tensor,
        s_axis: torch.Tensor,
        p_axis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total = image.sum()
        marginal_s = image.sum(dim=0)
        marginal_p = image.sum(dim=1)
        mean_s = (marginal_s * s_axis).sum() / total
        mean_p = (marginal_p * p_axis).sum() / total
        rms_s = torch.sqrt((marginal_s * (s_axis - mean_s) ** 2).sum() / total)
        rms_p = torch.sqrt((marginal_p * (p_axis - mean_p) ** 2).sum() / total)
        return mean_s, mean_p, rms_s, rms_p

    def _fourier_shift(
        self,
        image: torch.Tensor,
        shift_s: torch.Tensor,
        shift_p: torch.Tensor,
        pitch: float,
    ) -> torch.Tensor:
        freq_s = torch.fft.fftfreq(
            image.shape[1], d=pitch, dtype=self._real_dtype, device=image.device
        )
        freq_p = torch.fft.fftfreq(
            image.shape[0], d=pitch, dtype=self._real_dtype, device=image.device
        )
        freq_p_grid, freq_s_grid = torch.meshgrid(freq_p, freq_s, indexing="ij")
        phase = torch.exp(
            -1j * 2.0 * math.pi * (freq_s_grid * shift_s + freq_p_grid * shift_p)
        )
        return torch.fft.ifft2(torch.fft.fft2(image) * phase).real

    def _build_reference_response(self, device: torch.device) -> dict[str, Any]:
        n = self.numerical
        sp = self.spectrum
        o = self.optical_transport
        c = self.sensor_geometry
        wavelength = self._tensor(sp.reference_wavelength_m, device)
        z1 = self._tensor(o.screen_to_lens_m, device)
        z2 = self._tensor(o.lens_to_sensor_m, device)

        s_source, ds_source = self._centered_axis_from_width(
            n.source_grid_size,
            n.source_half_width_m,
            device=device,
            dtype=self._real_dtype,
        )
        p_source, dp_source = self._centered_axis_from_width(
            n.source_grid_size,
            n.source_half_width_m,
            device=device,
            dtype=self._real_dtype,
        )
        s_lens, ds_lens = self._centered_axis_from_width(
            n.lens_grid_size,
            n.lens_half_width_m,
            device=device,
            dtype=self._real_dtype,
        )
        p_lens, dp_lens = self._centered_axis_from_width(
            n.lens_grid_size,
            n.lens_half_width_m,
            device=device,
            dtype=self._real_dtype,
        )
        rho_floor = 0.5 * torch.minimum(ds_source, dp_source)

        gamma0, beta0, _ = self._reference_gamma_beta(device)
        zero = self._tensor(0.0, device)
        e_s, e_p = self._aumeyr_source(
            x=zero,
            y=zero,
            xp=zero,
            yp=zero,
            gamma=gamma0,
            s_source=s_source,
            p_source=p_source,
            rho_floor=rho_floor,
            wavelength=wavelength,
        )
        source_fields = torch.stack((e_s, e_p), dim=0)
        lens_before = self._scaled_fresnel(
            source_fields,
            s_source,
            p_source,
            s_lens,
            p_lens,
            wavelength,
            z1,
        )
        lens_transfer, pupil = self._thin_lens_transfer(s_lens, p_lens, wavelength)

        oversampled_pitch = c.pixel_pitch_m / n.psf_oversampling
        s_psf = self._centered_axis_from_pitch(
            n.psf_oversampled_size,
            oversampled_pitch,
            device=device,
            dtype=self._real_dtype,
        )
        p_psf = self._centered_axis_from_pitch(
            n.psf_oversampled_size,
            oversampled_pitch,
            device=device,
            dtype=self._real_dtype,
        )
        camera_fields = self._scaled_fresnel(
            lens_before * lens_transfer,
            s_lens,
            p_lens,
            s_psf,
            p_psf,
            wavelength,
            z2,
        )
        psf_oversampled = torch.sum(torch.abs(camera_fields) ** 2, dim=0)

        # At the nominal screen angle, remove only the even-grid numerical
        # offset.  For a physically rotated screen the OTR cone moves relative
        # to the fixed lens/camera line, so its displacement must be retained.
        psf_mean_s, psf_mean_p, _, _ = self._image_moments(
            psf_oversampled, s_psf, p_psf
        )
        screen_is_nominal = math.isclose(
            self.otr_screen.screen_angle_deg,
            self.nominal_screen_angle_deg,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        if (
            screen_is_nominal
            and o.lens_offset_s_m == 0.0
            and o.lens_offset_p_m == 0.0
        ):
            psf_oversampled = self._fourier_shift(
                psf_oversampled,
                -psf_mean_s,
                -psf_mean_p,
                oversampled_pitch,
            )
            psf_oversampled = torch.clamp(psf_oversampled, min=0.0)
        psf_oversampled = psf_oversampled / psf_oversampled.sum()
        _, _, psf_rms_s, psf_rms_p = self._image_moments(psf_oversampled, s_psf, p_psf)

        physical_kernel_size = n.psf_oversampled_size // n.psf_oversampling
        psf_kernel = psf_oversampled.reshape(
            physical_kernel_size,
            n.psf_oversampling,
            physical_kernel_size,
            n.psf_oversampling,
        ).sum(dim=(1, 3))
        psf_kernel = psf_kernel / psf_kernel.sum()

        # Absolute Ginzburg-Frank yield through the actual pupil.  Preserve the
        # notebook convention: use the propagated Aumeyr field only for local
        # relative s/p weights; its arbitrary common amplitude cancels.
        p_lens_grid, s_lens_grid = torch.meshgrid(p_lens, s_lens, indexing="ij")
        fixed = self._fixed_geometry(device)
        r_lens = torch.sqrt(z1**2 + s_lens_grid**2 + p_lens_grid**2)
        ray_hat = (
            z1 * fixed["k0_hat"][:, None, None]
            + s_lens_grid * fixed["s0_hat"][:, None, None]
            + p_lens_grid * fixed["p0_hat"][:, None, None]
        ) / r_lens
        zero = self._tensor(0.0, device)
        actual_reference_geometry = self._particle_geometry(zero, zero, device)
        cos_theta = torch.clamp(
            torch.sum(
                ray_hat * actual_reference_geometry["k_hat"][:, None, None],
                dim=0,
            ),
            -1.0,
            1.0,
        )
        sin2_theta = 1.0 - cos_theta**2
        d_omega = z1 / r_lens**3 * ds_lens * dp_lens
        angular_factor = beta0**2 * sin2_theta / (1.0 - beta0**2 * cos_theta**2) ** 2
        d2n_dlambda_domega = (
            self.fine_structure_constant / (math.pi**2 * wavelength) * angular_factor
        )
        i_s = torch.abs(lens_before[0]) ** 2
        i_p = torch.abs(lens_before[1]) ** 2
        i_total = i_s + i_p
        w_s = torch.where(i_total > 0.0, i_s / i_total, torch.zeros_like(i_total))
        w_p = torch.where(i_total > 0.0, i_p / i_total, torch.zeros_like(i_total))
        incidence_angle = self._tensor(
            math.radians(self.otr_screen.screen_angle_deg), device
        )
        r_s, r_p = self._fresnel_coefficients(incidence_angle)
        reflectivity_effective = torch.abs(r_s) ** 2 * w_s + torch.abs(r_p) ** 2 * w_p
        dn_dnm_per_electron = (
            d2n_dlambda_domega * reflectivity_effective * d_omega * pupil
        ).sum() * 1e-9

        wavelength_nm = torch.linspace(
            sp.wavelength_min_m * 1e9,
            sp.wavelength_max_m * 1e9,
            n.spectrum_samples,
            dtype=self._real_dtype,
            device=device,
        )
        spectrum_per_nm_per_electron = dn_dnm_per_electron * (
            sp.reference_wavelength_m * 1e9 / wavelength_nm
        )
        photons_per_electron = (
            torch.trapz(spectrum_per_nm_per_electron, wavelength_nm)
            * o.optical_transmission
        )

        self._response_build_count += 1
        return {
            "psf_oversampled": psf_oversampled.detach(),
            "psf_kernel": psf_kernel.detach(),
            "psf_rms_s_m": psf_rms_s.detach(),
            "psf_rms_p_m": psf_rms_p.detach(),
            "dn_dnm_per_electron_at_reference": dn_dnm_per_electron.detach(),
            "photons_per_electron": photons_per_electron.detach(),
            "fresnel_r_s": r_s.detach(),
            "fresnel_r_p": r_p.detach(),
            "source_axes": (s_source, p_source, rho_floor),
            "lens_axes": (s_lens, p_lens),
        }

    def _response(self, device: torch.device) -> dict[str, Any]:
        key = (str(device), self._real_dtype)
        if key not in self._response_cache:
            with torch.no_grad():
                self._response_cache[key] = self._build_reference_response(device)
        return self._response_cache[key]

    def response_summary(
        self, *, device: str | torch.device = "cpu"
    ) -> dict[str, float | complex | int]:
        """Return absolute-yield and PSF metadata, building it if necessary."""
        response = self._response(torch.device(device))
        return {
            "response_build_count": self.response_build_count,
            "magnification": self.magnification,
            "dN_dnm_per_electron_at_reference": float(
                response["dn_dnm_per_electron_at_reference"].cpu()
            ),
            "photons_per_electron_camera": float(
                response["photons_per_electron"].cpu()
            ),
            "psf_rms_s_m": float(response["psf_rms_s_m"].cpu()),
            "psf_rms_p_m": float(response["psf_rms_p_m"].cpu()),
            "fresnel_r_s": complex(response["fresnel_r_s"].cpu()),
            "fresnel_r_p": complex(response["fresnel_r_p"].cpu()),
        }

    def sensor_axes(
        self,
        *,
        device: str | torch.device = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return global physical camera axes ``(S, P)`` in meters."""
        device = torch.device(device)
        c = self.sensor_geometry
        o = self.optical_transport
        s = (
            self._centered_axis_from_pitch(
                c.n_x, c.pixel_pitch_m, device=device, dtype=self._real_dtype
            )
            + o.sensor_offset_s_m
        )
        p = (
            self._centered_axis_from_pitch(
                c.n_y, c.pixel_pitch_m, device=device, dtype=self._real_dtype
            )
            + o.sensor_offset_p_m
        )
        return s, p

    def _deposit_bilinear(
        self,
        center_s_local: torch.Tensor,
        center_p_local: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        """Deposit particle centers on an oversampled, physical-pixel-aligned ROI."""
        c = self.sensor_geometry
        n = self.numerical
        oversampling = n.psf_oversampling
        pitch = c.pixel_pitch_m / oversampling
        full_n_x = c.n_x * oversampling
        full_n_y = c.n_y * oversampling

        continuous_s = center_s_local / pitch + (full_n_x - 1.0) / 2.0
        continuous_p = center_p_local / pitch + (full_n_y - 1.0) / 2.0
        index_s0 = torch.floor(continuous_s).to(torch.long)
        index_p0 = torch.floor(continuous_p).to(torch.long)
        fraction_s = continuous_s - index_s0.to(self._real_dtype)
        fraction_p = continuous_p - index_p0.to(self._real_dtype)

        contributes = (
            (index_s0 >= -1)
            & (index_s0 < full_n_x)
            & (index_p0 >= -1)
            & (index_p0 < full_n_y)
        )
        if not bool(contributes.detach().any().cpu()):
            return torch.zeros(
                (0, 0),
                dtype=self._real_dtype,
                device=center_s_local.device,
            ), 0, 0

        contributing_s0 = index_s0[contributes]
        contributing_p0 = index_p0[contributes]
        min_s = int(
            torch.clamp(contributing_s0.min(), 0, full_n_x - 1).detach().cpu()
        )
        max_s = int(
            torch.clamp((contributing_s0 + 1).max(), 0, full_n_x - 1)
            .detach()
            .cpu()
        )
        min_p = int(
            torch.clamp(contributing_p0.min(), 0, full_n_y - 1).detach().cpu()
        )
        max_p = int(
            torch.clamp((contributing_p0 + 1).max(), 0, full_n_y - 1)
            .detach()
            .cpu()
        )

        # Pad by half the PSF on every side so the centered FFT convolution is
        # linear over the occupied region.  Keep ROI edges on physical-pixel
        # boundaries so the final block sum is exact.
        halo = n.psf_oversampled_size // 2 + oversampling
        start_s = math.floor((min_s - halo) / oversampling) * oversampling
        stop_s = math.ceil((max_s + 1 + halo) / oversampling) * oversampling
        start_p = math.floor((min_p - halo) / oversampling) * oversampling
        stop_p = math.ceil((max_p + 1 + halo) / oversampling) * oversampling
        roi_n_x = stop_s - start_s
        roi_n_y = stop_p - start_p

        flat = torch.zeros(
            roi_n_y * roi_n_x,
            dtype=self._real_dtype,
            device=center_s_local.device,
        )
        for offset_p, weight_p in ((0, 1.0 - fraction_p), (1, fraction_p)):
            for offset_s, weight_s in ((0, 1.0 - fraction_s), (1, fraction_s)):
                index_s = index_s0 + offset_s
                index_p = index_p0 + offset_p
                valid = (
                    (index_s >= 0)
                    & (index_s < full_n_x)
                    & (index_p >= 0)
                    & (index_p < full_n_y)
                )
                local_s = torch.clamp(index_s - start_s, 0, roi_n_x - 1)
                local_p = torch.clamp(index_p - start_p, 0, roi_n_y - 1)
                flat_index = local_p * roi_n_x + local_s
                flat.scatter_add_(
                    0,
                    flat_index,
                    weight_p * weight_s * valid.to(self._real_dtype),
                )
        density = flat.reshape(roi_n_y, roi_n_x) / center_s_local.numel()
        return density, start_s, start_p

    def _convolve_on_sensor(
        self,
        density: torch.Tensor,
        psf_oversampled: torch.Tensor,
        start_s: int,
        start_p: int,
    ) -> torch.Tensor:
        """Convolve on the oversampled grid, then integrate physical pixels."""
        c = self.sensor_geometry
        n = self.numerical
        oversampling = n.psf_oversampling
        if density.numel() == 0:
            return torch.zeros(
                (c.n_y, c.n_x), dtype=self._real_dtype, device=density.device
            )

        kernel_y, kernel_x = psf_oversampled.shape
        kernel_canvas = torch.zeros_like(density)
        start_y = (density.shape[0] - kernel_y) // 2
        start_x = (density.shape[1] - kernel_x) // 2
        kernel_canvas[
            start_y : start_y + kernel_y,
            start_x : start_x + kernel_x,
        ] = psf_oversampled
        image = torch.fft.fftshift(
            torch.fft.ifft2(
                torch.fft.fft2(torch.fft.ifftshift(density))
                * torch.fft.fft2(torch.fft.ifftshift(kernel_canvas))
            ).real
        )
        # A centered even-length kernel lives halfway between two samples.
        # Correct the corresponding half-pixel circular-convolution offset,
        # exactly as the validated notebook recenters its oversampled result.
        oversampled_pitch = c.pixel_pitch_m / oversampling
        shift_s = self._tensor(
            0.5 * oversampled_pitch if kernel_x % 2 == 0 else 0.0,
            density.device,
        )
        shift_p = self._tensor(
            0.5 * oversampled_pitch if kernel_y % 2 == 0 else 0.0,
            density.device,
        )
        if kernel_x % 2 == 0 or kernel_y % 2 == 0:
            image = self._fourier_shift(image, shift_s, shift_p, oversampled_pitch)
        image = torch.clamp(image, min=0.0)

        # Crop the padded ROI to the finite sensor, then sum each
        # oversampling x oversampling block into one physical camera pixel.
        full_n_x = c.n_x * oversampling
        full_n_y = c.n_y * oversampling
        global_s0 = max(start_s, 0)
        global_s1 = min(start_s + image.shape[1], full_n_x)
        global_p0 = max(start_p, 0)
        global_p1 = min(start_p + image.shape[0], full_n_y)
        local_s0 = global_s0 - start_s
        local_s1 = global_s1 - start_s
        local_p0 = global_p0 - start_p
        local_p1 = global_p1 - start_p
        image = image[local_p0:local_p1, local_s0:local_s1]
        physical_roi = image.reshape(
            image.shape[0] // oversampling,
            oversampling,
            image.shape[1] // oversampling,
            oversampling,
        ).sum(dim=(1, 3))

        physical_s0 = global_s0 // oversampling
        physical_p0 = global_p0 // oversampling
        normalized_image = F.pad(
            physical_roi,
            (
                physical_s0,
                c.n_x - physical_s0 - physical_roi.shape[1],
                physical_p0,
                c.n_y - physical_p0 - physical_roi.shape[0],
            ),
        )

        # Preserve the deposited probability mass exactly after removal of
        # tiny negative FFT roundoff.  This also preserves genuine loss when
        # macroparticle centers fall outside the finite physical sensor.
        image_sum = normalized_image.sum()
        density_sum = density.sum()
        return torch.where(
            image_sum > 0.0,
            normalized_image * (density_sum / image_sum),
            normalized_image,
        )

    def forward(
        self,
        particles_6d: Any,
        *,
        bunch_charge_C: float | torch.Tensor,
    ) -> torch.Tensor:
        """Return photons per physical camera pixel per bunch.

        The input macroparticles are equal weight.  Their number affects only
        sampling of the normalized transverse distribution; the physical
        number of electrons is ``bunch_charge_C / electron_charge_C``.
        """
        particles, x, y, xp, yp, _longitudinal, _delta = self._unpack_particles(
            particles_6d
        )
        charge = self._tensor(bunch_charge_C, particles.device)
        if charge.ndim != 0:
            raise ValueError("bunch_charge_C must be a scalar charge magnitude")
        if bool((charge.detach() < 0.0).cpu()):
            raise ValueError("bunch_charge_C must be the nonnegative charge magnitude")

        # Exact hit coordinates preserve the notebook geometry.  In the
        # nominal 45-degree case and at zero slope: S_c=-y and P_c=-x.
        hit_s, hit_p = self._screen_hits_vectorized(x, y, xp, yp)
        center_s_global = self.magnification * hit_s
        center_p_global = self.magnification * hit_p
        center_s_local = center_s_global - self.optical_transport.sensor_offset_s_m
        center_p_local = center_p_global - self.optical_transport.sensor_offset_p_m

        response = self._response(particles.device)
        density, start_s, start_p = self._deposit_bilinear(
            center_s_local, center_p_local
        )
        normalized_image = self._convolve_on_sensor(
            density,
            response["psf_oversampled"],
            start_s,
            start_p,
        )
        n_electrons = charge / self.electron_charge_C
        return normalized_image * n_electrons * response["photons_per_electron"]

    def particle_resolved_image(
        self,
        particles_6d: Any,
        *,
        bunch_charge_C: float | torch.Tensor,
        max_particles: int = 16,
    ) -> torch.Tensor:
        """Slow particle-resolved reference calculation for validation.

        Every particle gets its own exact Aumeyr source, screen intersection,
        cone tilt, polarization projection, and two-step scaled-Fresnel
        propagation.  This is intentionally not the normal fast path.  The
        current absolute calibration assigns the validated reference broadband
        yield to each equal-weight electron and normalizes each resolved image
        on the finite sensor before summing.
        """
        particles, x, y, xp, yp, _longitudinal, delta = self._unpack_particles(
            particles_6d
        )
        if particles.shape[0] > max_particles:
            raise ValueError(
                f"particle_resolved_image is limited to {max_particles} particles; "
                "use forward() for bunches"
            )
        charge = self._tensor(bunch_charge_C, particles.device)
        if charge.ndim != 0 or bool((charge.detach() < 0.0).cpu()):
            raise ValueError("bunch_charge_C must be a nonnegative scalar magnitude")
        response = self._response(particles.device)
        s_source, p_source, rho_floor = response["source_axes"]
        s_lens, p_lens = response["lens_axes"]
        s_sensor, p_sensor = self.sensor_axes(device=particles.device)
        wavelength = self._tensor(
            self.spectrum.reference_wavelength_m, particles.device
        )
        z1 = self._tensor(self.optical_transport.screen_to_lens_m, particles.device)
        z2 = self._tensor(self.optical_transport.lens_to_sensor_m, particles.device)
        lens_transfer, _ = self._thin_lens_transfer(s_lens, p_lens, wavelength)
        image = torch.zeros(
            (self.sensor_geometry.n_y, self.sensor_geometry.n_x),
            dtype=self._real_dtype,
            device=particles.device,
        )
        gammas = self._particle_gamma(delta, particles.device)
        for index in range(particles.shape[0]):
            e_s, e_p = self._aumeyr_source(
                x=x[index],
                y=y[index],
                xp=xp[index],
                yp=yp[index],
                gamma=gammas[index],
                s_source=s_source,
                p_source=p_source,
                rho_floor=rho_floor,
                wavelength=wavelength,
            )
            lens_before = self._scaled_fresnel(
                torch.stack((e_s, e_p), dim=0),
                s_source,
                p_source,
                s_lens,
                p_lens,
                wavelength,
                z1,
            )
            camera_fields = self._scaled_fresnel(
                lens_before * lens_transfer,
                s_lens,
                p_lens,
                s_sensor,
                p_sensor,
                wavelength,
                z2,
            )
            particle_image = torch.sum(torch.abs(camera_fields) ** 2, dim=0)
            image = image + particle_image / particle_image.sum()

        image = image / particles.shape[0]
        n_electrons = charge / self.electron_charge_C
        return image * n_electrons * response["photons_per_electron"]

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import torch
from otr_diagnostic_v3 import OTRDiagnostic
from ebeam import beam

# 1. Initialize diagnostic and beam
otr = OTRDiagnostic()
ebeam = beam()

# 2. Generate beam with SI units (e.g., 0.5 mm spatial spread, 0.1 mrad divergence)
particles_6d = ebeam.gen_6d_gaussian(0, [0.5e-3, 1e-4, 0.5e-3, 1e-4, 0.1, 1e-3], 20000)

# 3. Forward pass
photon_image = otr(particles_6d, bunch_charge_C=50e-12)

# 4. Extract sensor axes in mm (S = Horizontal, P = Vertical)
s_axis, p_axis = otr.sensor_axes()
s_mm = s_axis.cpu().numpy() * 1e3
p_mm = p_axis.cpu().numpy() * 1e3

# 5. Convert photon tensor to NumPy array
image_data = photon_image.detach().cpu().numpy()

# 6. Plot the photon image
plt.figure(figsize=(9, 6))
extent = [s_mm[0], s_mm[-1], p_mm[0], p_mm[-1]]  # [S_min, S_max, P_min, P_max]

im = plt.imshow(
    image_data,
    origin="lower",
    extent=extent,
    cmap="viridis",
    aspect="equal",
)

plt.colorbar(im, label="Photons / Pixel / Bunch")
plt.xlabel("Camera S Coordinate [mm]")
plt.ylabel("Camera P Coordinate [mm]")
plt.title(f"OTR Diagnostic Capture (Total Photons: {image_data.sum():.2e})")
plt.tight_layout()
plt.show()