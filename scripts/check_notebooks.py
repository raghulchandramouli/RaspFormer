"""Execute notebook source with the installed IPython kernel and record failures."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

from IPython.terminal.interactiveshell import TerminalInteractiveShell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".context/notebook-checks"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    os.chdir(root / "notebooks")
    summary = []
    shell = TerminalInteractiveShell.instance()
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        shell.reset(new_session=False)
        notebook = json.loads(path.read_text())
        cells = [(i + 1, "".join(cell["source"])) for i, cell in enumerate(notebook["cells"]) if cell["cell_type"] == "code"]
        with (output / f"{path.stem}.log").open("w") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                for index, source in cells:
                    result = shell.run_cell(source, store_history=False)
                    if result.error_before_exec or result.error_in_exec:
                        raise RuntimeError(f"{path.name} cell {index} failed; see {log.name}.")
        summary.append({"notebook": path.name, "code_cells": len(cells), "status": "passed"})
        print(f"{path.name}: {len(cells)} code cells passed", flush=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
