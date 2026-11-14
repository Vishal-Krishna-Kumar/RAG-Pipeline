"""Static checks for the notebook in this repository.

Running the notebook needs an NVIDIA GPU and gated model weights, so CI cannot execute it.
This script checks what can be checked without running it:

1. the file is a valid Jupyter notebook (nbformat 4) with cells,
2. every code cell is syntactically valid Python (IPython ``!`` and ``%`` lines are skipped),
3. no Hugging Face access token was saved in a cell or in an output.

    python scripts/check_notebook.py
    python scripts/check_notebook.py path/to/other.ipynb

Standard library only.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

DEFAULT_NOTEBOOK = Path(__file__).resolve().parents[1] / "RetrievalAugumentationGenerative.ipynb"
TOKEN = re.compile(r"hf_[A-Za-z0-9]{30,}")


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def python_lines(source: str) -> str:
    """Drop IPython shell (``!``) and magic (``%``) lines, which are not Python."""
    return "\n".join(
        "" if line.lstrip().startswith(("!", "%")) else line for line in source.splitlines()
    )


def check_notebook(path: Path) -> list[str]:
    """Return a list of problems; an empty list means the notebook passed."""
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"cannot read {path.name} as JSON: {error}"]

    problems = []
    if notebook.get("nbformat") != 4:
        problems.append(f"expected nbformat 4, found {notebook.get('nbformat')!r}")
    cells = notebook.get("cells") or []
    if not cells:
        problems.append("the notebook has no cells")

    for number, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        try:
            ast.parse(python_lines(cell_source(cell)))
        except SyntaxError as error:
            problems.append(f"cell {number}: syntax error on line {error.lineno}: {error.msg}")

    if TOKEN.search(json.dumps(notebook)):
        problems.append("a Hugging Face token (hf_...) is stored in the notebook")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DEFAULT_NOTEBOOK
    problems = check_notebook(path)
    for problem in problems:
        print(f"FAIL  {problem}")
    if problems:
        return 1

    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    code = sum(cell["cell_type"] == "code" for cell in cells)
    print(f"OK    {path.name}: {len(cells)} cells ({code} code, {len(cells) - code} markdown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
