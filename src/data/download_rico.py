"""
Download and prepare Rico UI hierarchy JSON files for this project.

The official Rico "UI Screenshots and View Hierarchies" archive stores files in
one combined directory. This script extracts JSON hierarchy files into the
layout expected by the preprocessing code:

    data/raw/<app_package>/<screen_id>.json

Usage:
  python -m src.data.download_rico --out_dir data/raw
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import tarfile
import urllib.request
from pathlib import Path

RICO_UNIQUE_UIS_URL = (
    "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/"
    "rico_dataset_v0.1/unique_uis.tar.gz"
)

logger = logging.getLogger(__name__)


def _download(url: str, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if archive_path.exists() and archive_path.stat().st_size > 0:
        logger.info("Using existing archive: %s", archive_path)
        return

    logger.info("Downloading %s to %s", url, archive_path)
    with urllib.request.urlopen(url) as response, archive_path.open("wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        next_report = 0

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            if total:
                pct = int(downloaded * 100 / total)
                if pct >= next_report:
                    logger.info("Download progress: %d%%", pct)
                    next_report += 5
            elif downloaded // (1024 * 1024 * 512) > next_report:
                next_report += 1
                logger.info("Downloaded %.1f GB", downloaded / (1024**3))


def _safe_name(value: str) -> str:
    value = value.strip() or "unknown_app"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _app_id_from_rico_json(data: dict) -> str:
    for key in ("package_name", "package", "app_id"):
        if data.get(key):
            return _safe_name(str(data[key]))

    activity_name = str(data.get("activity_name", ""))
    if "/" in activity_name:
        return _safe_name(activity_name.split("/", 1)[0])
    if activity_name:
        return _safe_name(activity_name.rsplit(".", 1)[0])
    return "unknown_app"


def extract_hierarchies(
    archive_path: Path,
    out_dir: Path,
    max_screens: int | None = None,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0
    failed = 0

    logger.info("Extracting hierarchy JSON files to %s", out_dir)
    with tarfile.open(archive_path, mode="r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".json"):
                continue

            if max_screens is not None and extracted >= max_screens:
                break

            screen_id = Path(member.name).stem
            fileobj = tar.extractfile(member)
            if fileobj is None:
                failed += 1
                continue

            try:
                raw = fileobj.read()
                data = json.loads(raw.decode("utf-8"))
                app_id = _app_id_from_rico_json(data)
                app_dir = out_dir / app_id
                app_dir.mkdir(parents=True, exist_ok=True)
                target = app_dir / f"{screen_id}.json"
                if target.exists():
                    skipped += 1
                    continue
                target.write_bytes(raw)
                extracted += 1
                if extracted % 1000 == 0:
                    logger.info("Extracted %d JSON files", extracted)
            except Exception as exc:
                failed += 1
                logger.warning("Failed to extract %s: %s", member.name, exc)

    return {"extracted": extracted, "skipped": skipped, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare Rico hierarchy JSON files.")
    parser.add_argument("--out_dir", default="data/raw", help="Output directory for app-grouped JSON files.")
    parser.add_argument(
        "--archive_path",
        default="data/downloads/unique_uis.tar.gz",
        help="Where to store or read the Rico archive.",
    )
    parser.add_argument("--url", default=RICO_UNIQUE_UIS_URL, help="Rico archive URL.")
    parser.add_argument(
        "--max_screens",
        type=int,
        default=None,
        help="Optional cap for a small local/GPU smoke-test subset.",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Use an existing archive_path without trying to download it.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    archive_path = Path(args.archive_path)
    if not args.skip_download:
        _download(args.url, archive_path)
    elif not archive_path.exists():
        raise FileNotFoundError(f"--skip_download was set but archive does not exist: {archive_path}")

    stats = extract_hierarchies(
        archive_path=archive_path,
        out_dir=Path(args.out_dir),
        max_screens=args.max_screens,
    )
    logger.info(
        "Done. Extracted=%d skipped=%d failed=%d",
        stats["extracted"],
        stats["skipped"],
        stats["failed"],
    )


if __name__ == "__main__":
    main()
