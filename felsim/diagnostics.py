"""Weighted population beam moments for Xsuite-like particle sets.

Coordinates are canonical (x, px, y, py, zeta, pzeta) in SI: x, y, zeta in m;
px = P_x/p0, py = P_y/p0; pzeta = ptau/beta0 = (E - E0)/(beta0 p0 c), the Xsuite
canonical conjugate of zeta = s - beta0 c t. The momentum deviation
delta = (p - p0)/p0 (equal to pzeta to first order) is used only for sigma_delta,
chirp and dispersion. Second moments are population moments (ddof = 0) weighted
by particle weight; particles with state <= 0 or weight <= 0 are excluded.
Normalized emittances use the reference beta0 gamma0 = p0c/mass0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.constants import c, e

METRIC_FIELDS = ('nemitt_x', 'nemitt_y', 'eps_z', 'sigma_delta', 'sigma_delta_uncorr',
                 'sigma_t', 'sigma_x', 'sigma_y', 'mean_delta', 'mean_energy_eV',
                 'chirp', 'eigen1_n', 'eigen2_n', 'volume6', 'corr_x_zeta', 'corr_px_zeta')


def particle_weights(parts):
    """One weight per particle; unit weights when the object has none."""
    n = np.asarray(parts.x).size
    return np.broadcast_to(np.asarray(getattr(parts, 'weight', np.ones(n)), dtype=float), (n,))


def active_mask(parts):
    """Particles alive (state > 0) with a positive finite weight."""
    weights = particle_weights(parts)
    state = np.asarray(getattr(parts, 'state', np.ones(len(weights))))
    return (state > 0) & (weights > 0) & np.isfinite(weights)


def coordinate_matrix(parts, canonical=True):
    """6 x N coordinates of all particles; last row pzeta, or delta if not canonical."""
    last = (np.asarray(parts.ptau, dtype=float)/np.asarray(parts.beta0, dtype=float)
            if canonical else np.asarray(parts.delta, dtype=float))
    return np.vstack([np.asarray(getattr(parts, key), dtype=float)
                      for key in ('x', 'px', 'y', 'py', 'zeta')]+[last])


def weighted_covariance(values, weights, *, min_count=2):
    """Weighted population covariance (ddof = 0) of coordinates x particles.

    Zero-weight columns are dropped first; negative or nonfinite weights, or fewer
    than ``min_count`` positive weights, give a NaN matrix.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 2 or weights.ndim != 1 or values.shape[1] != len(weights):
        raise ValueError('Expected coordinates by particles and one weight per particle')
    if int(min_count) != min_count or min_count < 1:
        raise ValueError('min_count must be a positive integer')
    invalid = np.full((values.shape[0], values.shape[0]), np.nan)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        return invalid
    selected = weights > 0
    if np.count_nonzero(selected) < min_count:
        return invalid
    values, weights = values[:, selected], weights[selected]
    wn = weights/weights.max()
    wn = wn/wn.sum()
    with np.errstate(invalid='ignore', over='ignore'):
        centered = values-values@wn[:, None]
        return (centered*wn)@centered.T


def covariance_emittance(cov):
    """sqrt(det cov): RMS emittance of a 2 x 2 block, RMS volume in higher dimension."""
    cov = np.asarray(cov, dtype=float)
    if not np.isfinite(cov).all():
        return np.nan
    return float(np.sqrt(max(float(np.linalg.det(cov)), 0)))


def eigenemittances(cov):
    """Eigenemittances, the moduli of the eigenvalues of J cov, ascending; pairs interleaved."""
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] % 2:
        raise ValueError('Expected a square covariance with interleaved canonical pairs')
    n = cov.shape[0]//2
    if not np.isfinite(cov).all():
        return np.full(n, np.nan)
    j = np.kron(np.eye(n), [[0., 1.], [-1., 0.]])
    return np.sort(np.abs(np.linalg.eigvals(j@cov).imag))[::2]


def dispersion_corrected_nemitt_x(parts, *, min_particles=7):
    """Normalized eps_x after removing x, px correlations with delta (R16, R26).

    The weighted least-squares slopes R16 = <x delta>/<delta^2>, R26 = <px delta>/<delta^2>
    equal the dispersion D, D' of ``ebeam.beam.cal_twiss``.
    """
    mask = active_mask(parts)
    weights = particle_weights(parts)[mask]
    if len(weights) < min_particles:
        return dict(nemitt_x_corrected=np.nan, r16_fit_m=np.nan, r26_fit=np.nan)
    xpd = np.vstack([np.asarray(getattr(parts, key), dtype=float)[mask]
                     for key in ('x', 'px', 'delta')])
    cov = weighted_covariance(xpd, weights)
    r = cov[:2, 2]/cov[2, 2]
    intrinsic = cov[:2, :2]-np.outer(r, r)*cov[2, 2]
    betagamma0 = float(np.asarray(parts.p0c).flat[0]/parts.mass0)
    return dict(nemitt_x_corrected=betagamma0*covariance_emittance(intrinsic),
                r16_fit_m=float(r[0]), r26_fit=float(r[1]))


def _count_metrics(weights, charge_per_weight, initial_charge):
    q = float(weights.sum()*charge_per_weight)
    return dict(n_alive=len(weights), charge=q,
                transmission=q/initial_charge if initial_charge else np.nan)


@dataclass(frozen=True)
class _MetricArrays:
    canonical: np.ndarray
    delta: np.ndarray
    energy_eV: np.ndarray
    weights: np.ndarray
    charge_per_weight: float
    betagamma0: float
    beta0: float


def _metric_arrays(parts, mask):
    ptau = np.asarray(parts.ptau)[mask]
    return _MetricArrays(coordinate_matrix(parts)[:, mask], np.asarray(parts.delta)[mask],
                         np.asarray(parts.energy0)[mask]+ptau*np.asarray(parts.p0c)[mask],
                         particle_weights(parts)[mask],
                         abs(float(np.asarray(getattr(parts, 'q0', 1.)).flat[0]))*e,
                         float(np.asarray(parts.p0c).flat[0]/parts.mass0),
                         float(np.asarray(parts.beta0).flat[0]))


def _metrics_from_arrays(data, selection=slice(None), *, initial_charge=None, min_particles=7):
    weights = data.weights[selection]
    out = _count_metrics(weights, data.charge_per_weight, initial_charge)
    if len(weights) < min_particles or out['charge'] <= 0:
        return out | dict.fromkeys(METRIC_FIELDS, np.nan)
    coords = data.canonical[:, selection]
    delta = data.delta[selection]
    cov = weighted_covariance(coords, weights)
    zd = weighted_covariance(np.vstack([coords[4], delta]), weights)
    wn = weights/weights.sum()
    bg = data.betagamma0
    eig = eigenemittances(cov[:4, :4])*bg
    chirp = zd[0, 1]/max(zd[0, 0], 1e-300)  # 1/m, d(delta)/d(zeta)
    out.update(nemitt_x=bg*covariance_emittance(cov[:2, :2]),
               nemitt_y=bg*covariance_emittance(cov[2:4, 2:4]),
               eps_z=covariance_emittance(cov[4:, 4:]),
               sigma_delta=np.sqrt(zd[1, 1]),
               sigma_delta_uncorr=np.sqrt(max(zd[1, 1]-chirp*zd[0, 1], 0)),
               sigma_t=np.sqrt(cov[4, 4])/(data.beta0*c),
               sigma_x=np.sqrt(cov[0, 0]), sigma_y=np.sqrt(cov[2, 2]),
               mean_delta=float(delta@wn), mean_energy_eV=float(data.energy_eV[selection]@wn),
               chirp=chirp, eigen1_n=eig[0], eigen2_n=eig[1], volume6=covariance_emittance(cov),
               corr_x_zeta=cov[0, 4]/max(np.sqrt(cov[0, 0]*cov[4, 4]), 1e-300),
               corr_px_zeta=cov[1, 4]/max(np.sqrt(cov[1, 1]*cov[4, 4]), 1e-300))
    return out


def metrics(parts, *, initial_charge=None, min_particles=7):
    """Active charge (C), transmission and SI beam moments; NaN moments below ``min_particles``.

    eps_z is geometric in (zeta, pzeta); nemitt_x, nemitt_y, eigen1_n, eigen2_n are normalized.
    """
    mask = active_mask(parts)
    if np.count_nonzero(mask) < min_particles:
        q = abs(float(np.asarray(getattr(parts, 'q0', 1.)).flat[0]))*e
        return _count_metrics(particle_weights(parts)[mask], q, initial_charge) | dict.fromkeys(
            METRIC_FIELDS, np.nan)
    return _metrics_from_arrays(_metric_arrays(parts, mask), initial_charge=initial_charge,
                                min_particles=min_particles)


def slice_metrics(parts, n_slices=12, *, min_particles=7, min_total=20):
    """``metrics`` in equal-width zeta slices over the active range; column ``zeta`` is the centre (m)."""
    if int(n_slices) != n_slices or n_slices < 1:
        raise ValueError('n_slices must be a positive integer')
    mask = active_mask(parts)
    if np.count_nonzero(mask) < min_total:
        return pd.DataFrame()
    data = _metric_arrays(parts, mask)
    z = data.canonical[4]
    if not np.isfinite(z).all():
        raise ValueError('Slice diagnostics require finite active zeta')
    edges = np.linspace(z.min()-1e-15, z.max()+1e-15, int(n_slices)+1)  # m, pad keeps end particles
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        row = _metrics_from_arrays(data, (z >= lo) & (z < hi), min_particles=min_particles)
        row['zeta'] = 0.5*(lo+hi)
        rows.append(row)
    return pd.DataFrame(rows)
