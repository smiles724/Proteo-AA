"""Put the repo and all three upstreams on the path."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "PXDesign", ROOT / "Protenix", ROOT / "fampnn"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
