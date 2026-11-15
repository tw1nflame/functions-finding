"""Execute the real-data notebook with this Python interpreter and retain outputs."""

import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    os.environ["MPLCONFIGDIR"] = str(root / ".mplconfig")
    os.environ["IPYTHONDIR"] = str(root / "artifacts" / "ipython")
    os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
    path = root / "examples" / "real_bacterial_growth.ipynb"
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=900,
        resources={"metadata": {"path": str(root / "examples")}},
    )
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f"Executed {len(code_cells)} code cells, no errors: {path}")


if __name__ == "__main__":
    main()
