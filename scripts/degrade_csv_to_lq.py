#!/usr/bin/env python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flashvsr_b1.data.degradation.offline_lq import main


if __name__ == "__main__":
    raise SystemExit(main())
