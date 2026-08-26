"""Standalone re-run of this metric.

Uses the shared, metric-independent feature cache in
../_features (Gram matrices and SVCCA bases derived from the
paper's own decomposition). Everything else is local.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
METRIC = 'mutual_knn'

if __name__ == '__main__':
    sys.exit(subprocess.run([sys.executable,
        os.path.join(HERE, 'scorer.py'),
        '--metric', METRIC, '--outdir', HERE]).returncode)
