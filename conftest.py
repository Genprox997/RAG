"""
Pytest fixtures.

Note: the default `tmp_path` fixture relies on shutil.rmtree for cleanup, which
is intercepted by the sandbox safe-delete hook and intermittently breaks test
setup/teardown. `safe_tmp_path` avoids rmtree entirely: it only ever *creates*
a unique directory under .pytest_local (gitignored) and never deletes it, so no
bulk-delete hook can fire. Leftover dirs are harmless and cleaned manually if needed.
"""
import os
import tempfile

import pytest

_LOCAL_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pytest_local")


@pytest.fixture
def safe_tmp_path():
    os.makedirs(_LOCAL_BASE, exist_ok=True)
    d = tempfile.mkdtemp(prefix="t", dir=_LOCAL_BASE)
    yield d
