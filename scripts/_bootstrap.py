"""Put the repo and all three upstreams on sys.path."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "PXDesign", ROOT / "Protenix", ROOT / "fampnn"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
