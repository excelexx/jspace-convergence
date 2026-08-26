"""Run every metric in the sweep, each into its own subfolder.

Resumable and fault-isolated: a metric whose results.json is already marked
complete is skipped, and a metric that crashes is recorded and does not stop
the remaining ones. Safe to re-run.

mutual_knn is run first on purpose: it is the paper's own metric, so its
text-text component means should reproduce the published 0.5577 / 0.4202 /
0.5057 and act as an end-to-end check that everything upstream of the metric
really is the paper's pipeline.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PY = os.path.join("..", "venv", "Scripts", "python.exe")

ORDER = [
    "mutual_knn",      # paper's metric -- validation baseline
    "cknna",
    "cka",
    "unbiased_cka",
    "cycle_knn",
    "edit_knn",
    "lcs_knn",
    "svcca",
]


def main():
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    status = {}
    t0 = time.time()
    for metric in ORDER:
        outdir = os.path.join(HERE, metric)
        os.makedirs(outdir, exist_ok=True)
        res = os.path.join(outdir, "results.json")
        if os.path.exists(res):
            try:
                with open(res) as f:
                    if json.load(f).get("complete"):
                        print(f"[skip] {metric} already complete", flush=True)
                        status[metric] = "cached"
                        continue
            except Exception:
                pass
        print(f"\n{'='*70}\n[run] {metric}  ({time.time()-t0:.0f}s elapsed)\n"
              f"{'='*70}", flush=True)
        cmd = [PY, "scorer.py", "--metric", metric, "--outdir", outdir,
               "--n_perm", str(n_perm)]
        r = subprocess.run(cmd)
        status[metric] = "ok" if r.returncode == 0 else f"FAILED rc={r.returncode}"
        print(f"[{metric}] {status[metric]}", flush=True)
        with open(os.path.join(HERE, "sweep_status.json"), "w") as f:
            json.dump(dict(status=status, elapsed=time.time() - t0), f, indent=1)

    print(f"\n{'='*70}\nsweep finished in {time.time()-t0:.0f}s")
    for m, s in status.items():
        print(f"  {m:14s} {s}")

    # build the summary table from whatever completed
    subprocess.run([PY, "make_summary.py"])


if __name__ == "__main__":
    main()
