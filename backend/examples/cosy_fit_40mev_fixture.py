"""Run the 11-stage COSY FIT reference of the 40 MeV fixture and judge it and the Nelder-Mead fixture on both objectives.

python backend/examples/cosy_fit_40mev_fixture.py --out outputs/ref/axis2_fixture_40MeV
"""
import argparse
import datetime
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from felsim import paths
from felsim.fixtures import (COSY_FIT_40MEV, HISTORICAL_40MEV, MIRROR, cosy_fit_40mev, cross_objectives,
                             historical_40mev, run_stages, set_quad_current)
from historical_40mev_fixture import C_START, C_X, C_Y, envelopes_along_s, git

C_FEL, C_COSY = C_X, C_Y


def cosy_version(dist=Path("~/COSY/10.2/UNIX").expanduser()):
    info = json.loads((dist / "installation.json").read_text())
    return {"version": info["cosy_version"], "dist_dir": str(dist), "archive_sha256": info["archive_sha256"],
            "machine": info["machine"]}


def delta_rms(p=HISTORICAL_40MEV, seed=0):
    """Measured rms relative energy deviation (%) of the seeded start beam (spread plus chirp)."""
    return float(historical_40mev(seed, p)["particles"][:, 5].std() / 10)


def plot(path, s_fel, s_cos, cosy_pts, quads, fel_I, cos_I, obj, spread):
    fig = plt.figure(figsize=(7.2, 8.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1, 1])
    ax1, ax2 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])
    ax3, ax4 = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])

    for (s, sx, sy), c, lab in ((s_fel, C_FEL, "FELsim NM fixture"), (s_cos, C_COSY, "COSY FIT")):
        ax1.plot(s, sx, color=c, lw=1.2, label=rf"$\sigma_x$, {lab} currents")
        ax1.plot(s, sy, color=c, lw=1.2, ls=(0, (4, 2)), label=rf"$\sigma_y$, {lab} currents")
    ax1.plot(cosy_pts[0], cosy_pts[1], "o", ms=4, mfc="none", mec="k", mew=0.8,
             label=r"COSY map $\sqrt{\epsilon\beta}$ at targets, COSY FIT currents")
    ax1.set_yscale("log")
    ax1.set_xlabel("s (m)")
    ax1.set_ylabel("rms size (mm)")
    ax1.set_xlim(0, s_fel[0][-1])
    ax1.set_ylim(1e-2, 1e2)
    ax1.legend(ncol=2, fontsize=6.5, frameon=False, loc="upper left")
    ax1.grid(alpha=0.25, lw=0.5)

    x, w = np.arange(len(quads)), 0.38
    ax2.bar(x - w / 2, [fel_I[q] for q in quads], w, color=C_FEL, label="FELsim NM fixture (seed 0)")
    ax2.bar(x + w / 2, [cos_I[q] for q in quads], w, color=C_COSY, label="COSY FIT (11 stages)")
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_ylim(top=6.5)
    ax2.set_xticks(x, [f"{q}{'*' if q in MIRROR else ''}" for q in quads], fontsize=6, rotation=90)
    ax2.set_xlabel("quadrupole element index (* mirrored); negative = reversed polarity")
    ax2.set_ylabel("current (A)")
    ax2.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    ax2.grid(axis="y", alpha=0.25, lw=0.5)

    floor = 1e-12
    for ax, key, title in ((ax3, "felsim_12", "FELsim 12-stage objective\n(FELsim tracking, seed 0)"),
                           (ax4, "cosy_11", "COSY 11-stage objective\n(COSY maps)")):
        f, c = np.maximum(obj[key]["fel"], floor), np.maximum(obj[key]["cosy"], floor)
        k = np.arange(1, len(f) + 1)
        ax.bar(k - 0.2, f, 0.4, color=C_FEL, label="FELsim NM currents")
        ax.bar(k + 0.2, c, 0.4, color=C_COSY, label="COSY FIT currents")
        if key == "felsim_12":
            ax.vlines(k - 0.2, np.maximum(spread["min"], floor), np.maximum(spread["max"], floor),
                      color="k", lw=1, label="NM re-optimized, seeds 0-4 range")
        ax.set_yscale("log")
        ax.set_ylim(floor / 3, 1e4)
        ax.set_xticks(k)
        ax.set_xlabel("stage")
        ax.set_title(title, fontsize=8)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax3.set_ylabel("stage objective")
    ax3.legend(fontsize=6, frameon=False, loc="upper left", ncol=1)
    fig.suptitle(r"UH beamline 40 MeV, $\epsilon_n$ = 8 mm mrad, $\sigma$ = 0.8 mm: FELsim Nelder-Mead vs COSY 10.2 FIT",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(paths.default_output_dir("axis2_fixture_40MeV")))
    ap.add_argument("--cosy-dir", default="outputs/tmp/axis2_cosy_fixture/cosy_run", help="COSY working directory")
    ap.add_argument("--figure", default=None, help="figure path without suffix")
    ap.add_argument("--seeds", type=int, default=5, help="Nelder-Mead fixture seeds 0..N-1")
    a = ap.parse_args()
    cosy_dir = paths.REPO_ROOT / a.cosy_dir

    cosy = cosy_fit_40mev(output_dir=cosy_dir)
    (cosy_dir / "cosy_fit_full.log").write_text(cosy["log"])

    nm = []
    for seed in range(a.seeds):
        fix = historical_40mev(seed)
        t0 = time.perf_counter()
        rec = run_stages(fix)
        nm.append({"seed": seed, "wall_s": time.perf_counter() - t0, "stage_fun": [r["fun"] for r in rec],
                   "total": float(sum(r["fun"] for r in rec)),
                   "currents_A": {i: float(el.current) for i, el in enumerate(fix["line"]) if hasattr(el, "current")}})
    quads = sorted(i for i in cosy["currents_A"])
    fel_I = {q: nm[0]["currents_A"][q] for q in quads}
    cos_I = cosy["currents_A"]

    at_fel = cross_objectives(fel_I, seed=0, output_dir=cosy_dir)
    at_cos = cross_objectives(cos_I, seed=0, output_dir=cosy_dir)
    cos_on_felsim_seeds = [float(np.sum(cross_objectives(cos_I, seed=s, cosy=False)["felsim_tracking"]["felsim_12"]))
                           for s in range(a.seeds)]
    stage_fun = np.array([r["stage_fun"] for r in nm])
    totals = np.array([r["total"] for r in nm])
    diff = np.array([cos_I[q] - fel_I[q] for q in quads])

    results = {
        "reference": "cosy_fit_40mev", "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--", "felsim", "backend")),
        "cosy": cosy_version(), "cosy_settings": COSY_FIT_40MEV,
        "start_beam": {**HISTORICAL_40MEV, "measured_delta_rms_percent_seed0": delta_rms(),
                       "cosy_geometric_emittance_mm_mrad": cosy["eps_mm_mrad"], "cosy_beta0_m": cosy["beta0_m"],
                       "note": "COSY sees only the linear Twiss (alpha 0, beta0 = sigma^2/eps); energy spread and chirp "
                               "enter the FELsim particles only, with chromatic tracking off"},
        "cosy_fit": {"currents_A": cos_I, "stage_objective_cosy_map": cosy["stage_loss"],
                     "timing_s": cosy["timing_s"],
                     "negative_currents": {q: v for q, v in cos_I.items() if v < 0}},
        "nelder_mead_seeds": nm,
        "nelder_mead_seed_spread": {"total_min": float(totals.min()), "total_max": float(totals.max()),
                                    "total_mean": float(totals.mean()), "total_std": float(totals.std(ddof=1)),
                                    "stage_min": stage_fun.min(0).tolist(), "stage_max": stage_fun.max(0).tolist()},
        "cross_objectives": {"at_felsim_nm_seed0_currents": at_fel, "at_cosy_fit_currents": at_cos,
                             "cosy_currents_felsim12_total_tracking_seeds": cos_on_felsim_seeds,
                             "evaluators": "felsim_tracking: seed-0 particles, negative current = reversed polarity; "
                                           "cosy_map: COSY 10.2 FIT blocks with Nmax 0 started at the currents"},
        "current_difference_cosy_minus_felsim_A": {"max_abs": float(np.abs(diff).max()),
                                                   "median_abs": float(np.median(np.abs(diff))),
                                                   "argmax_element": int(quads[int(np.abs(diff).argmax())])},
    }
    out = paths.REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rpath = out / "cosy_fit_results.json"
    rpath.write_text(json.dumps(results, indent=2, default=float))

    fig = paths.REPO_ROOT / a.figure if a.figure else out / "cosy_vs_felsim"
    fig.parent.mkdir(parents=True, exist_ok=True)
    fixf = historical_40mev(0)
    for q, v in fel_I.items():
        set_quad_current(fixf["line"], q, v)
    fixc = historical_40mev(0)
    for q, v in cos_I.items():
        set_quad_current(fixc["line"], q, v)
    s_end = np.cumsum([el.length for el in fixc["line"]])
    pts = sorted({(float(s_end[i]), v) for (i, _, m), v in cosy["moments"].items() if m == "envelope"})
    plot(fig, envelopes_along_s(fixf["line"], fixf["particles"]), envelopes_along_s(fixc["line"], fixc["particles"]),
         (np.array([p[0] for p in pts]), np.array([p[1] for p in pts])), quads, fel_I, cos_I,
         {"felsim_12": {"fel": np.array(at_fel["felsim_tracking"]["felsim_12"]),
                        "cosy": np.array(at_cos["felsim_tracking"]["felsim_12"])},
          "cosy_11": {"fel": np.array(at_fel["cosy_map"]["cosy_11"]), "cosy": np.array(at_cos["cosy_map"]["cosy_11"])}},
         {"min": stage_fun.min(0), "max": stage_fun.max(0)})

    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    manifest["cosy_fit_reference"] = {
        "code": "backend/examples/cosy_fit_40mev_fixture.py, felsim/fixtures.py",
        "command": " ".join(["python", *sys.argv]), "git_sha": results["git_sha"],
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "cosy": results["cosy"], "platform": platform.platform(), "seeds": list(range(a.seeds)),
        "results": rpath.name, "results_sha256": hashlib.sha256(rpath.read_bytes()).hexdigest(),
        "figure": [str(fig.with_suffix(".png")), str(fig.with_suffix(".pdf"))],
    }
    mpath.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
