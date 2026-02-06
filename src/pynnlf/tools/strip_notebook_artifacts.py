from pathlib import Path
import re

PATTERNS = [
    r"^\s*get_ipython\(",        # ipython magic runner
    r"^\s*%(\w+)",               # %pip, %run, etc
    r"^\s*!",                    # shell escapes
    r"^\s*#\s*In\[\d*\]:\s*$",   # cell markers
]

def strip_file(p: Path) -> int:
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
    root = Path("src")
    py_files = list(root.rglob("*.py"))
    total = 0
    for f in py_files:
        total += strip_file(f)
    print(f"Removed {total} notebook artifact lines across {len(py_files)} files.")

if __name__ == "__main__":
    main()