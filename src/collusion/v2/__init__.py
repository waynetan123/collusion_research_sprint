"""v2 experiment: the forked SALT-NLP harness plus our pool and launch tooling.

``collusion.v2.harness`` is upstream SALT-NLP/agent-collusion's ``experiments``
package at UPSTREAM_COMMIT (imported unmodified in 271d9ce, patched, then moved
here with only its imports rewritten). ``collusion.v2.analysis`` holds upstream's
judge scripts. The rest of this package prepares the harness's inputs from our
pool and launches it. Only the harness code comes from the paper: every task is
exported from our own pool into the gitignored data/ directory.
"""

from pathlib import Path

from collusion.pool.io import ROOT

UPSTREAM_REPO = "https://github.com/SALT-NLP/agent-collusion"
UPSTREAM_COMMIT = "5afb2c595aa3eff6b9b829a032f4f117b36ee54d"

# The harness resolves task paths against its working directory and refuses
# paths outside it, so it runs from the repo root and the tasks live under it.
HARNESS_ROOT: Path = ROOT
# Locations relative to HARNESS_ROOT (all under the gitignored data/).
POOL_TASK_DIR = "data/tasks"
POOL_MANIFEST = f"{POOL_TASK_DIR}/task_manifest.json"
SEQUENCE_DIR = "data/sequences"
