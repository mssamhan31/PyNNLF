from __future__ import annotations

from importlib import resources
from pathlib import Path
import json
import shutil
import urllib.request


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
    """
    ws = Path(workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)

    scaffold = resources.files("pynnlf").joinpath("scaffold")

    # Copy scaffold contents into workspace (updated structure)
    # - models/ contains model library + hyperparameters.json
    # - specs/ contains pynnlf_config.json + experiment.json + batch.json
    for item in ("models", "data", "specs", "README_WORKSPACE.md"):
        src = scaffold.joinpath(item)
        dst = ws / item
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # Ensure output directory exists
    (ws / "experiment_result").mkdir(parents=True, exist_ok=True)

    if download_data:
        cfg = json.loads((ws / "specs" / "pynnlf_config.json").read_text(encoding="utf-8"))
        base_url = base_url_override or cfg["dataset_download"]["base_url"]
        ds_map: dict[str, str] = cfg["datasets"]

        if all_data:
            to_get = [k for k in ds_map.keys() if k != "ds0"]  # ds0 bundled
        else:
            to_get = datasets or []

        for ds_id in to_get:
            if ds_id not in ds_map:
                raise KeyError(f"Unknown dataset id: {ds_id}")
            fname = ds_map[ds_id]
            url = base_url.rstrip("/") + "/" + fname
            dest = ws / "data" / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, dest)

    return ws