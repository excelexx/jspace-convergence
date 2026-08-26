"""Driver for the CKNNA neighbourhood-size sweep (appendix, locality paragraph).

CKNNA at topk = n-1 is exactly unbiased CKA (verify_metrics2.py checks the
identity), so sweeping topk continuously interpolates between the local metric
family that reproduces the paper's J-space claim and the global one that
rejects it. Every config runs the paper-identical decomposition, preprocessing,
layer band and grid-mean aggregation; only the neighbourhood size varies.

Resumable and fault-isolated: each config is a separate scorer2.py process and
a completed results.json is never recomputed.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PY = os.path.join("..", "venv", "Scripts", "python.exe")

# topk >= 2 per the reference implementation; 999 = n-1 is the full-kernel end
CONFIGS = [(f"t1_cknna_k{k}", dict(kind="cknna", topk=k))
           for k in [2, 3, 5, 10, 25, 50, 100, 250, 999]]

LOCK = os.path.join(HERE, "sweep2.lock")


def _pid_alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    except Exception:
        return False


def acquire_lock():
    """Refuse to start if another driver is already running.

    Two concurrent drivers contend for the GPU and silently recompute the same
    configs -- that happened once and made every metric ~7x slower."""
    if os.path.exists(LOCK):
        try:
            with open(LOCK) as f:
                other = int(f.read().strip())
        except Exception:
            other = None
        if other and other != os.getpid() and _pid_alive(other):
            print(f"another run_sweep2 is already running (pid {other}); "
                  f"exiting. Remove {LOCK} if that is wrong.", flush=True)
            return False
        print(f"clearing stale lock from pid {other}", flush=True)
    with open(LOCK, "w") as f:
        f.write(str(os.getpid()))
    return True


def release_lock():
    try:
        if os.path.exists(LOCK):
            with open(LOCK) as f:
                if f.read().strip() == str(os.getpid()):
                    os.remove(LOCK)
    except Exception:
        pass


def main():
    if not acquire_lock():
        return
    try:
        _main()
    finally:
        release_lock()


def _main():
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    status = {}
    t0 = time.time()
    for name, cfg in CONFIGS:
        status[name] = run_one(name, cfg, n_perm, t0, status)
    print(f"\n{'='*70}\nCKNNA sweep finished in {(time.time()-t0)/60:.1f} min")
    for k, v in status.items():
        print(f"  {k:26s} {v}")


def run_one(name, cfg, n_perm, t0, status):
    outdir = os.path.join(HERE, name)
    os.makedirs(outdir, exist_ok=True)
    res = os.path.join(outdir, "results.json")
    if os.path.exists(res):
        try:
            with open(res) as f:
                if json.load(f).get("complete"):
                    print(f"[skip] {name}", flush=True)
                    return "cached"
        except Exception:
            pass
    print(f"\n{'='*70}\n[run] {name}  {json.dumps(cfg)}  "
          f"({(time.time()-t0)/60:.1f} min elapsed)\n{'='*70}", flush=True)
    r = subprocess.run([PY, "scorer2.py", "--name", name,
                        "--config", json.dumps(cfg), "--outdir", outdir,
                        "--n_perm", str(n_perm)])
    st = "ok" if r.returncode == 0 else f"FAILED rc={r.returncode}"
    print(f"[{name}] {st}", flush=True)
    return st


if __name__ == "__main__":
    main()
