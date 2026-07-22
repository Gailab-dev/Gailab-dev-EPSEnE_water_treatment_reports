# -*- coding: utf-8 -*-
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from water_treatment.clustering import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(default_plant="yongyeon"))
