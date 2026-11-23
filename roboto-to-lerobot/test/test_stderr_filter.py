"""Tests for the SVT-AV1 stderr filter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"

_SCRIPT = f"""
import sys
sys.path.insert(0, {str(_SRC)!r})
from roboto_to_lerobot.stderr_filter import install_svt_stderr_filter
install_svt_stderr_filter()
for i in range(20000):
    print(f"line {{i}}", file=sys.stderr)
raise ValueError("the run's last words")
"""


def test_traceback_of_an_uncaught_exception_reaches_stderr():
    """A run that fails with an uncaught exception logs the exception's message, even
    when stderr is still busy as the process exits."""
    result = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True, timeout=60)
    assert result.returncode == 1
    assert result.stderr.rstrip().endswith("ValueError: the run's last words")
