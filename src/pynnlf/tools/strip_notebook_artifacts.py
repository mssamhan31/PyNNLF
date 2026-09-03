"""Developer utility: strip Jupyter artefacts from exported Python files.

Inputs:  Python files beneath the src directory, relative to the current working directory.
Outputs: the same files rewritten in place with notebook-only lines removed.
Key steps: drop get_ipython() calls, percent magics, exclamation-mark shell lines and
           "# In[..]" cell markers.
"""

from pathlib import Path
import re

PATTERNS = [
    r"^\s*get_ipython\(",        # ipython magic runner
    r"^\s*%(\w+)",               # %pip, %run, etc
    r"^\s*!",                    # shell escapes
    r"^\s*#\s*In\[\d*\]:\s*$",   # cell markers
]

def strip_file(p: Path) -> int:
    """Rewrite one file in place, removing Jupyter-only lines.

    Returns:
        bool: True if the file was modified.
    """
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    out, removed = [], 0
    for line in lines:
        if any(re.search(pat, line) for pat in PATTERNS):
            removed += 1
            continue
        out.append(line)
    if removed:
        p.write_text("".join(out), encoding="utf-8")
    return removed

def main():
    """Strip Jupyter artefacts from every Python file under the src directory.

    Returns:
        None.
    """
    root = Path("src")
    py_files = list(root.rglob("*.py"))
    total = 0
    for f in py_files:
        total += strip_file(f)
    print(f"Removed {total} notebook artifact lines across {len(py_files)} files.")

if __name__ == "__main__":
    main()