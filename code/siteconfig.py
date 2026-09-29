"""Public-package site configuration.

All generator scripts resolve data and output locations through this module,
so the public package has NO dependency on the private workspace layout.
"""
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent
DATA = SITE / "data"
FIGURES = SITE / "figures"
