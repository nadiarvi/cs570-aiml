"""
Run on server to inspect what raw JSON files actually contain.
  python inspect_raw.py
"""
import json, glob
from pathlib import Path

files = sorted(glob.glob("data/raw/**/*.json", recursive=True))
print(f"Total raw JSON files: {len(files)}")
if not files:
    print("NO JSON FILES FOUND in data/raw/")
else:
    for path in files[:3]:
        with open(path) as f:
            data = json.load(f)
        print(f"\nFile: {path}")
        print(f"  Top-level keys: {list(data.keys())[:15]}")
        if "activity" in data:
            act = data["activity"]
            print(f"  activity keys: {list(act.keys())[:10]}")
            if "root" in act:
                root = act["root"]
                print(f"  root keys: {list(root.keys())[:10]}")
                print(f"  root class: {root.get('class', '(missing)')}")
                print(f"  root children count: {len(root.get('children', []))}")
        else:
            # Show first few key-value pairs to understand format
            for k, v in list(data.items())[:5]:
                snippet = str(v)[:80]
                print(f"  {k!r}: {snippet}")
