"""Create a user workspace by copying the bundled scaffold.

Inputs:  a target workspace directory, and optionally the dataset identifiers to download.
Outputs: a workspace containing models/, data/, specs/, a README and an output directory.
Key steps: copy the packaged scaffold into the target, read the workspace configuration,
           create the output directory, then optionally download the requested datasets
           over HTTP.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
import shutil
import urllib.request
import yaml

def init(
    workspace_dir: str | Path,
    *,
    download_data: bool = False,
    all_data: bool = False,
    datasets: list[str] | None = None,
    base_url_override: str | None = None,
) -> Path:
    """
    Create a user workspace by copying bundled scaffold.
    Optionally download datasets from GitHub raw (no checksum, just download).

    Args:
        workspace_dir (str | Path): target directory to create the workspace
        download_data (bool): whether to download datasets from online source
        all_data (bool): if True, download all datasets except ds0
        datasets (list[str] | None): list of dataset IDs to download (e.g., ["ds15"])
        base_url_override (str | None): override base_url in config if provided

    Returns:
        Path: workspace directory path
    """
    ws = Path(workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)

    scaffold = resources.files("pynnlf").joinpath("scaffold")

    for item in ("models", "data", "specs", "README_WORKSPACE.md"):
        src = scaffold.joinpath(item)
        dst = ws / item
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    cfg = yaml.safe_load((ws / "specs" / "pynnlf_config.yaml").read_text(encoding="utf-8"))  # safe_load avoids executing arbitrary tags from the workspace configuration
    out_dir = ws / cfg["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if download_data:
        base_url = base_url_override or cfg["dataset_download"]["base_url"]
        ds_map: dict[str, str] = cfg["datasets"]

        to_get = [k for k in ds_map.keys() if k != "ds0"] if all_data else (datasets or [])
        for ds_id in to_get:
            if ds_id not in ds_map:
                raise KeyError(f"Unknown dataset id: {ds_id}")
            fname = ds_map[ds_id]
            url = base_url.rstrip("/") + "/" + fname
            dest = ws / cfg["paths"]["data_dir"] / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, dest)

    return ws