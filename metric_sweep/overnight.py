"""Unattended driver for the whole sweep. Safe to launch and walk away.

Sequence:
  1. wait out any build_features.py already running; if it dies or stalls,
     re-run it (it is idempotent, so this resumes rather than restarts)
  2. verify the metric ports against the reference implementations
  3. run every metric into its own subfolder (resumable, fault-isolated)
  4. build SUMMARY.md
  5. scaffold each metric folder into a standalone, re-runnable copy

Progress goes to overnight.log.
"""
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PY = os.path.join("..", "venv", "Scripts", "python.exe")
MANIFEST = os.path.join("_features", "manifest.json")
STALL_SECONDS = 600          # no new feature file for this long => build died
LOG = open("overnight.log", "a", encoding="utf-8", buffering=1)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")


def newest_feature_mtime():
    newest = 0.0
    for side in ("text", "cap", "img"):
        d = os.path.join("_features", side)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            newest = max(newest, os.path.getmtime(os.path.join(d, f)))
    return newest


def wait_for_features():
    """Block until the feature cache is complete, restarting a dead build."""
    while not os.path.exists(MANIFEST):
        last = newest_feature_mtime()
        age = time.time() - last if last else 1e9
        if age > STALL_SECONDS:
            log(f"feature build stalled ({age:.0f}s since last file) -> "
                f"running build_features.py to resume")
            r = subprocess.run([PY, "build_features.py"])
            if r.returncode != 0:
                log(f"build_features.py FAILED rc={r.returncode}; retrying once")
                r = subprocess.run([PY, "build_features.py"])
                if r.returncode != 0:
                    log("build_features.py failed twice, aborting")
                    return False
        else:
            time.sleep(60)
    log("feature cache complete")
    return True


def scaffold():
    """Make each metric folder standalone: its own code copy + README."""
    from run_all import ORDER
    for m in ORDER:
        d = os.path.join(HERE, m)
        if not os.path.isdir(d):
            continue
        for src in ("common.py", "scorer.py"):
            shutil.copy2(os.path.join(HERE, src), os.path.join(d, src))
        with open(os.path.join(d, "run.py"), "w", encoding="utf-8") as f:
            f.write(
                '"""Standalone re-run of this metric.\n\n'
                'Uses the shared, metric-independent feature cache in\n'
                '../_features (Gram matrices and SVCCA bases derived from the\n'
                "paper's own decomposition). Everything else is local.\n"
                '"""\n'
                "import os\nimport subprocess\nimport sys\n\n"
                "HERE = os.path.dirname(os.path.abspath(__file__))\n"
                f"METRIC = {m!r}\n\n"
                "if __name__ == '__main__':\n"
                "    sys.exit(subprocess.run([sys.executable,\n"
                "        os.path.join(HERE, 'scorer.py'),\n"
                "        '--metric', METRIC, '--outdir', HERE]).returncode)\n")
        with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as f:
            f.write(
                f"# {m}\n\n"
                f"Alignment under the `{m}` metric, over both of the paper's\n"
                "experiments (55 text-text pairs, 44 text-vision pairs), with\n"
                "every other step held identical to the paper.\n\n"
                "- `results.json` - per-pair alignment plus competence stats\n"
                "- `run.log` - the scoring log\n"
                "- `run.py` - standalone re-run: "
                "`../../venv/Scripts/python.exe run.py`\n\n"
                "See `../README.md` for the full method and `../SUMMARY.md`\n"
                "for the cross-metric comparison.\n")
    log(f"scaffolded {len(ORDER)} standalone metric folders")


def main():
    t0 = time.time()
    log("=== overnight sweep start ===")
    if not wait_for_features():
        return 1

    log("running verify_metrics.py")
    r = subprocess.run([PY, "verify_metrics.py"])
    if r.returncode != 0:
        log("METRIC VERIFICATION FAILED - aborting before the sweep")
        return 1
    log("metric ports verified")

    log("running the full metric sweep")
    subprocess.run([PY, "run_all.py"])

    log("building summary")
    subprocess.run([PY, "make_summary.py"])
    scaffold()

    if os.path.exists("sweep_status.json"):
        with open("sweep_status.json") as f:
            log("status: " + json.dumps(json.load(f)["status"]))
    log(f"=== overnight sweep done in {(time.time()-t0)/60:.1f} min ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
