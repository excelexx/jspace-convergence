"""Run the content-ablation / random-dictionary controls for all eight metrics.

Resumable, fault-isolated, and single-instance (a PID lock stops two drivers
racing for the GPU). mutual_knn runs first as the validation baseline: it is
the paper's own metric, so it must reproduce Table 1
(54.1% / 52.0% / 26.3% / 34.7% retention).
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PY = os.path.join("..", "venv", "Scripts", "python.exe")
LOCK = os.path.join(HERE, "ablation.lock")

ORDER = ["mutual_knn", "cknna", "cka", "unbiased_cka",
         "cycle_knn", "edit_knn", "lcs_knn", "svcca"]


def _alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:
        return False


def acquire():
    if os.path.exists(LOCK):
        try:
            other = int(open(LOCK).read().strip())
        except Exception:
            other = None
        if other and other != os.getpid() and _alive(other):
            print(f"another run_all is already running (pid {other}); exiting")
            return False
    open(LOCK, "w").write(str(os.getpid()))
    return True


def release():
    try:
        if os.path.exists(LOCK) and open(LOCK).read().strip() == str(os.getpid()):
            os.remove(LOCK)
    except Exception:
        pass


def main():
    if not acquire():
        return
    try:
        status, t0 = {}, time.time()
        for m in ORDER:
            outdir = os.path.join(HERE, m)
            os.makedirs(outdir, exist_ok=True)
            res = os.path.join(outdir, "results.json")
            if os.path.exists(res):
                try:
                    if json.load(open(res)).get("complete"):
                        print(f"[skip] {m}", flush=True)
                        status[m] = "cached"
                        continue
                except Exception:
                    pass
            print(f"\n{'='*68}\n[run] {m}  ({(time.time()-t0)/60:.1f} min)\n"
                  f"{'='*68}", flush=True)
            r = subprocess.run([PY, "scorer.py", "--metric", m,
                                "--outdir", outdir])
            status[m] = "ok" if r.returncode == 0 else f"FAILED rc={r.returncode}"
            print(f"[{m}] {status[m]}", flush=True)
            json.dump(dict(status=status, elapsed=time.time() - t0),
                      open("status.json", "w"), indent=1)
        print(f"\nablation sweep finished in {(time.time()-t0)/60:.1f} min")
        for k, v in status.items():
            print(f"  {k:14s} {v}")
        subprocess.run([PY, "make_summary.py"])
    finally:
        release()


if __name__ == "__main__":
    main()
