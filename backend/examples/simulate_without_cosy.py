"""Run a small electron-beam transport example using FELsim's Python engine.

From the repository root:
    .venv/bin/python backend/examples/simulate_without_cosy.py
    .venv/bin/python backend/examples/simulate_without_cosy.py --show

The example tracks a reproducible Gaussian beam through three drifts and a
focusing/defocusing quadrupole pair. It saves an envelope plot and CSV in
outputs/tmp/without_cosy/. No COSY executable, lattice file, or server is needed.

This is an illustrative 45 MeV beamline using the native transfer matrices,
not the full UH machine model; it does not include collective effects.
"""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
# Backend modules currently use local imports rather than a Python package.
sys.path.insert(0, str(REPO_ROOT / "backend"))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--show", action="store_true", help="Also open an interactive plot window."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "tmp" / "without_cosy",
        help="Directory for beam_envelope.png and beam_envelope.csv.",
    )
    args = parser.parse_args()

    # Select a headless backend before importing FELsim modules, which also
    # import pyplot. The default run works in a terminal or on a remote host.
    import matplotlib

    if not args.show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from felsimAdapter import FELsimAdapter
    from simulatorBase import BeamlineElement

    energy_mev = 45.0
    rng = np.random.default_rng(2026)
    # FELsim columns: x [mm], x' [mrad], y [mm], y' [mrad],
    # 1000 * delta(time of flight)/T_RF, 1000 * delta(kinetic energy)/K_0.
    # Thus the last RMS value, 0.1, means a relative energy spread of 0.01%.
    particles = rng.normal(
        loc=0.0,
        scale=[0.8, 0.15, 0.8, 0.15, 0.3, 0.1],
        size=(2000, 6),
    )
    elements = [
        BeamlineElement("DRIFT", length=0.30),
        BeamlineElement("QUAD_F", length=0.10, current=0.5),
        BeamlineElement("DRIFT", length=0.40),
        BeamlineElement("QUAD_D", length=0.10, current=0.5),
        BeamlineElement("DRIFT", length=0.50),
    ]
    simulator = FELsimAdapter()
    simulator.set_beam_energy(energy_mev)
    simulator.set_beamline(elements)
    evolution = simulator.collect_evolution(particles, interval=0.02)

    # Use the actual particle standard deviations, with units in each CSV
    # header. These are one-sigma beam sizes about the beam centroid.
    rows = []
    for s in evolution.s_positions:
        snapshot = evolution.particles[s]
        if not np.isfinite(snapshot).all():
            raise RuntimeError(f"Non-finite particle coordinates at s={s} m")
        rows.append(
            {
                "s_m": s,
                "sigma_x_mm": np.std(snapshot[:, 0], ddof=1),
                "sigma_y_mm": np.std(snapshot[:, 2], ddof=1),
                "sigma_xprime_mrad": np.std(snapshot[:, 1], ddof=1),
                "sigma_yprime_mrad": np.std(snapshot[:, 3], ddof=1),
            }
        )
    envelope = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.plot(envelope["s_m"], envelope["sigma_x_mm"], label=r"$\sigma_x$")
    ax.plot(envelope["s_m"], envelope["sigma_y_mm"], label=r"$\sigma_y$")
    s = 0.0
    for element in elements:
        if element.element_type in ("QUAD_F", "QUAD_D"):
            label = "QF" if element.element_type == "QUAD_F" else "QD"
            ax.axvspan(s, s + element.length, color="grey", alpha=0.15)
            ax.text(
                s + element.length / 2,
                0.97,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
            )
        s += element.length
    ax.set(
        xlabel="Distance along beamline [m]",
        ylabel="RMS beam size [mm]",
        title=f"FELsim Python engine: {energy_mev:g} MeV electron beam",
        xlim=(0, evolution.total_length),
        ylim=(0, 1.2 * envelope[["sigma_x_mm", "sigma_y_mm"]].to_numpy().max()),
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "beam_envelope.csv"
    png_path = output_dir / "beam_envelope.png"
    envelope.to_csv(csv_path, index=False)
    fig.savefig(png_path, dpi=160)

    final = envelope.iloc[-1]
    print(
        f"Tracked {len(particles)} electrons through {len(elements)} elements "
        f"over {evolution.total_length:.2f} m."
    )
    print(
        f"Final RMS beam sizes: x={final['sigma_x_mm']:.4f} mm, "
        f"y={final['sigma_y_mm']:.4f} mm"
    )
    print(f"Envelope data: {csv_path}")
    print(f"Envelope plot: {png_path}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
