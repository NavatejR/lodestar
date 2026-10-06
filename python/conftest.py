"""Ensures `import lodestar` resolves to the local package when pytest runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
