"""Run the historical 40 MeV, 12-stage Nelder-Mead fixture and save results, manifest and figure.

python backend/examples/historical_40mev_fixture.py --seed 0 --out outputs/ref/axis2_fixture_40MeV
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy

from felsim import paths
from felsim.fixtures import (HISTORICAL_40MEV, MIRROR, historical_40mev, measure_name, run_stages, stage_loss,
                             target_moments)

C_X, C_Y, C_REF, C_START = "#2a78d6", "#eb6834", "#eb6834", "#a3a29c"


def felsim_s1_currents():
    """FELSIM_S1_CURRENTS from the COSY optimization example (A, 0-based element index)."""
    spec = importlib.util.spec_from_file_location(
        "uhm_cosy", paths.REPO_ROOT / "backend" / "test" / "UHM_beamline_opt_cosy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {int(k): float(v) for k, v in mod.FELSIM_S1_CURRENTS.items()}


def envelopes_along_s(line, particles, ds=0.01):
    """rms x and y sizes (mm) along s (m), slicing sliceable elements at ds."""
    s, sx, sy = [0.0], [particles[:, 0].std()], [particles[:, 2].std()]
    for el in line:
        rest = el.length
        steps = [ds] * int(rest // ds) if el.sliceable else []
        rest -= ds * len(steps)
        if rest > 1e-12 or not steps:
            steps.append(rest)
        for d in steps:
            particles = el.useMatrice(particles, length=d) if el.sliceable else el.useMatrice(particles)
            s.append(s[-1] + d)
            sx.append(particles[:, 0].std())
            sy.append(particles[:, 2].std())
    return np.array(s), np.array(sx), np.array(sy)


def git(*args, cwd=paths.REPO_ROOT):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def yi_provenance(clone, sha="3d0d674"):
    """Commit, author, date and touched files of Yi Lu's fork snapshot in a local clone, or None."""
    if clone is None or not git("cat-file", "-t", sha, cwd=clone) == "commit":
        return None
    full, author, date, subject = git("log", "-1", "--format=%H|%an <%ae>|%aI|%s", sha, cwd=clone).split("|", 3)
    base = git("merge-base", sha, git("rev-parse", "origin/main"), cwd=clone)
    branch_files = git("diff", "--name-only", base, sha, cwd=clone).splitlines()
    top = {}
    for f in branch_files:
        top[f.split("/")[0]] = top.get(f.split("/")[0], 0) + 1
    return {"repository": git("remote", "get-url", "origin", cwd=clone), "branch": "pytorch_transform",
            "commit": full, "author": author, "date": date, "subject": subject,
            "files_in_commit": git("show", "--name-only", "--format=", sha, cwd=clone).splitlines(),
            "branch_vs_upstream": {"merge_base": base, "files_changed": len(branch_files),
                                   "authors": sorted(set(git("log", "--format=%an", f"{base}..{sha}", cwd=clone).splitlines())),
                                   "files_by_top_dir": dict(sorted(top.items(), key=lambda kv: -kv[1]))},
            "merged": False}


def plot(path, before, after, quads, start, final, ref, beta_ym, seed, n):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.0), gridspec_kw={"height_ratios": [1.3, 1]})
    for (s, sx, sy), ls, lab in ((before, (0, (4, 2)), "start (Excel)"), (after, "-", "optimized")):
        ax1.plot(s, sx, color=C_X, ls=ls, lw=1.2, label=rf"$\sigma_x$, {lab}")
        ax1.plot(s, sy, color=C_Y, ls=ls, lw=1.2, label=rf"$\sigma_y$, {lab}")
    ax1.set_yscale("log")
    ax1.set_xlabel("s (m)")
    ax1.set_ylabel("rms size (mm)")
    ax1.set_xlim(0, after[0][-1])
    ax1.legend(ncol=2, fontsize=7, frameon=False, loc="upper left")
    ax1.grid(alpha=0.25, lw=0.5)
    x = np.arange(len(quads))
    w = 0.28
    ax2.bar(x - w, [start[q] for q in quads], w, color=C_START, label="start (Excel)")
    ax2.bar(x, [final[q] for q in quads], w, color=C_X, label=f"optimized (seed {seed}, N={n})")
    ax2.bar(x + w, [ref.get(q, np.nan) for q in quads], w, color=C_REF, label="FELSIM_S1_CURRENTS")
    ax2.set_xticks(x, [f"{q}{'*' if q in MIRROR else ''}" for q in quads], fontsize=6, rotation=90)
    ax2.set_xlabel("quadrupole element index (* mirrored)")
    ax2.set_ylabel("current (A)")
    ax2.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    ax2.set_ylim(0, 6.5)
    ax2.grid(axis="y", alpha=0.25, lw=0.5)
    fig.suptitle(rf"FELsim 40 MeV, $\epsilon_n$ = 8 mm mrad, 12-stage Nelder-Mead; undulator $\beta$ target {beta_ym:.3f} m",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stages", type=int, default=12, help="run stages 1..N only (reduced run)")
    ap.add_argument("--particles", type=int, default=1000)
    ap.add_argument("--out", default=str(paths.default_output_dir("axis2_fixture_40MeV")))
    ap.add_argument("--figure", default=None, help="figure path without suffix")
    ap.add_argument("--yi-clone", default=None, help="local clone of Louis-26/FELsim holding 3d0d674")
    a = ap.parse_args()

    p = {**HISTORICAL_40MEV, "n_particles": a.particles}
    fix = historical_40mev(seed=a.seed, p=p)
    line, stages = fix["line"], fix["stages"][:a.stages]
    quads = sorted(i for i, el in enumerate(line) if hasattr(el, "current"))
    start = {i: float(line[i].current) for i in quads}
    before = envelopes_along_s(line, fix["particles"])

    t0 = time.perf_counter()
    records = run_stages(fix, stages=stages)
    total = time.perf_counter() - t0

    final = {i: float(line[i].current) for i in quads}
    moments = target_moments(line, fix["particles"], stages)
    losses = [stage_loss(st, moments) for st in stages]
    for r, l, st in zip(records, losses, stages):
        r["loss_final_line"] = l
        r["targets"] = [{"index": i, "plane": g["measure"][0], "measure": m, "goal": g["goal"], "weight": g["weight"],
                         "achieved": moments[(i, g["measure"][0], m)]}
                        for i, goals in st["objectives"].items() for g in goals for m in [measure_name(g)]]
    ref = felsim_s1_currents()
    comparison = {i: {"optimized_A": final[i], "reference_A": ref[i],
                      "rel_diff": (final[i] - ref[i]) / ref[i]} for i in quads if i in ref}

    out = paths.REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    reduced = a.stages < 12 or a.particles != 1000
    results = {
        "fixture": "historical_40mev", "reduced": reduced, "seed": a.seed,
        "git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", "felsim", "backend")),
        "start_beam": p, "beta_ym_target_m": fix["beta_ym_m"], "method": "Nelder-Mead (scipy defaults, bounds 0-10 A)",
        "objective": "per stage: sum(weight*(measured-goal)^2)/n_goals (beamOptimizer._optiSpeed)",
        "objective_total": float(np.sum(losses)), "stage_fun_sum": float(sum(r["fun"] for r in records)),
        "timing_s": {"total": total, "per_stage": [r["wall_s"] for r in records]},
        "nfev_total": int(sum(r["nfev"] for r in records)),
        "currents_start_A": start, "currents_final_A": final, "mirror": MIRROR,
        "comparison_felsim_s1": comparison,
        "stages": records,
        "yi_lu_snapshot": yi_provenance(a.yi_clone),
    }
    rpath = out / "results.json"
    rpath.write_text(json.dumps(results, indent=2, default=float))

    fig = paths.REPO_ROOT / a.figure if a.figure else out / "historical_40mev_fixture"
    fig.parent.mkdir(parents=True, exist_ok=True)
    after = envelopes_along_s(line, fix["particles"])
    plot(fig, before, after, quads, start, final, ref, fix["beta_ym_m"], a.seed, a.particles)

    manifest = {
        "code": "backend/examples/historical_40mev_fixture.py, felsim/fixtures.py",
        "command": " ".join(["python", *sys.argv]),
        "git_sha": results["git_sha"], "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "pandas": pandas.__version__, "matplotlib": matplotlib.__version__},
        "platform": platform.platform(),
        "parameters": {"seed": a.seed, "stages": a.stages, "particles": a.particles,
                       "lattice": "beam_excel/Beamline_elements.xlsx[:118]",
                       "lattice_sha256": hashlib.sha256((paths.EXCEL_DIR / "Beamline_elements.xlsx").read_bytes()).hexdigest()},
        "figure": [str(fig.with_suffix(".png")), str(fig.with_suffix(".pdf"))],
        "results_sha256": hashlib.sha256(rpath.read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
