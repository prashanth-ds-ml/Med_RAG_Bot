from __future__ import annotations

"""
Pytest shared configuration for the repository.

Why this file exists:
- Ensures the project root is added to Python's import path during test runs.
- Allows imports like `from app.tracking.hash_utils import ...` to work
  consistently without requiring manual PYTHONPATH exports every time.

Why this matters:
- Keeps test execution reproducible across machines and shells.
- Avoids fragile "works only in my terminal" import behavior.
"""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))