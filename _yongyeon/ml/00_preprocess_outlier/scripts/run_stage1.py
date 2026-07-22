"""Run stage-1 outlier correction for Yongyeon."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from water_treatment.stage1_outlier import cli


if __name__ == "__main__":
    raise SystemExit(cli(Path(__file__).resolve().parents[1] / "config" / "stage1.json"))
