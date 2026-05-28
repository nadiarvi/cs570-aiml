"""
Downloads the Rico view-hierarchy dataset from HuggingFace and saves each
screen's JSON hierarchy to data/raw/<app_package>/<screen_id>.json.
"""
import json
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = Path("data/raw")
CONFIG = "ui-screenshots-and-view-hierarchies"


def main():
    print(f"Loading Rico ({CONFIG}) ...")
    ds = load_dataset("creative-graphic-design/Rico", CONFIG, split="train")

    print("Columns:", ds.column_names)
    print("Sample keys:", list(ds[0].keys()))
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = 0

    for i, example in enumerate(tqdm(ds, desc="Saving hierarchies")):
        # Locate hierarchy field
        hierarchy = (
            example.get("ui_obj_hierarchy")
            or example.get("hierarchy")
            or example.get("view_hierarchy")
            or example.get("activity")
        )
        if hierarchy is None:
            # Fall back: dump the whole example minus any image field
            hierarchy = {k: v for k, v in example.items() if "image" not in k.lower()}

        # Locate app ID
        app_id = (
            example.get("app_package_name")
            or example.get("package_name")
            or example.get("app_id")
            or f"unknown_app"
        )

        # Locate screen ID
        screen_id = (
            example.get("screen_id")
            or example.get("id")
            or str(i)
        )

        app_dir = OUT_DIR / str(app_id)
        app_dir.mkdir(exist_ok=True)
        out_path = app_dir / f"{screen_id}.json"

        if out_path.exists():
            skipped += 1
            continue

        with open(out_path, "w") as f:
            if isinstance(hierarchy, str):
                f.write(hierarchy)
            else:
                json.dump(hierarchy, f)
        saved += 1

    print(f"\nDone. Saved {saved} screens, skipped {skipped} already present.")
    print(f"Output: {OUT_DIR}/")


if __name__ == "__main__":
    main()
