import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_notebook as cn  # noqa: E402


def write_notebook(path: Path, cells: list[dict], nbformat: int = 4) -> Path:
    path.write_text(json.dumps({"nbformat": nbformat, "cells": cells}), encoding="utf-8")
    return path


def code(source: str, outputs: list | None = None) -> dict:
    return {"cell_type": "code", "source": source, "outputs": outputs or []}


def test_the_repository_notebook_passes():
    assert cn.check_notebook(cn.DEFAULT_NOTEBOOK) == []
    assert cn.main([]) == 0


def test_magic_and_shell_lines_are_ignored(tmp_path):
    notebook = write_notebook(
        tmp_path / "ok.ipynb",
        [
            code("%%time\n!pip install torch\nx = 1\n%timeit x"),
            {"cell_type": "markdown", "source": "# hi"},
        ],
    )
    assert cn.check_notebook(notebook) == []


def test_syntax_error_is_reported_with_the_cell_number(tmp_path):
    notebook = write_notebook(
        tmp_path / "bad.ipynb", [code("x = 1"), code("def broken(:\n    pass")]
    )
    problems = cn.check_notebook(notebook)
    assert len(problems) == 1
    assert problems[0].startswith("cell 1: syntax error")
    assert cn.main([str(notebook)]) == 1


def test_saved_token_is_reported(tmp_path):
    token = "hf_" + "a" * 34
    notebook = write_notebook(
        tmp_path / "token.ipynb",
        [code("login()", outputs=[{"output_type": "stream", "text": [f"token: {token}"]}])],
    )
    assert any("token" in problem for problem in cn.check_notebook(notebook))


def test_wrong_format_empty_and_unreadable_notebooks_fail(tmp_path):
    assert cn.check_notebook(write_notebook(tmp_path / "v3.ipynb", [code("x = 1")], nbformat=3))
    assert cn.check_notebook(write_notebook(tmp_path / "empty.ipynb", []))
    broken = tmp_path / "broken.ipynb"
    broken.write_text("{not json", encoding="utf-8")
    assert cn.check_notebook(broken)
    assert cn.check_notebook(tmp_path / "missing.ipynb")
