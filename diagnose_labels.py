"""
Run on the server to diagnose why all heuristic labels are class 2.
  python diagnose_labels.py
"""
import glob
import collections
import torch

pt_files = sorted(glob.glob("data/processed/**/*.pt", recursive=True))[:10]

total_nodes = 0
class_counter = collections.Counter()
depth_counter = collections.Counter()
label_counter = collections.Counter()
has_text_count = 0
has_rid_count = 0

print(f"Inspecting {len(pt_files)} files...\n")

for path in pt_files:
    g = torch.load(path, weights_only=False)
    nodes = g["nodes"]           # compact node dicts stored at build time
    labels = g["y_heuristic"].tolist()
    total_nodes += len(nodes)

    for node, label in zip(nodes, labels):
        label_counter[label] += 1
        raw_cls = node.get("class", "")
        short = raw_cls.split(".")[-1].lower() if raw_cls else ""
        class_counter[short] += 1
        depth_counter[node.get("depth", "?")] += 1
        if node.get("text"):
            has_text_count += 1
        if node.get("resource-id"):
            has_rid_count += 1

print(f"Total nodes across {len(pt_files)} files: {total_nodes}")
print(f"Nodes with non-empty text:        {has_text_count}")
print(f"Nodes with non-empty resource-id: {has_rid_count}")
print()
print("Label distribution (y_heuristic):")
for k, v in sorted(label_counter.items()):
    print(f"  label {k}: {v} nodes ({100*v/total_nodes:.1f}%)")
print()
print("Top 15 short class names:")
for cls, cnt in class_counter.most_common(15):
    print(f"  {cls!r:30s} {cnt}")
print()
print("Depth distribution:")
for d, cnt in sorted(depth_counter.items()):
    print(f"  depth {d}: {cnt}")
print()

# Show 5 sample nodes that SHOULD be translatable but aren't
print("Sample nodes (first 5 per file, raw class + depth + text + label):")
for path in pt_files[:3]:
    g = torch.load(path, weights_only=False)
    nodes, labels = g["nodes"], g["y_heuristic"].tolist()
    print(f"\n  {path}  ({len(nodes)} nodes)")
    for node, label in list(zip(nodes, labels))[:8]:
        print(f"    class={node.get('class','')!r}  depth={node.get('depth','?')}  "
              f"text={node.get('text','')!r[:30]}  rid={node.get('resource-id','')!r[:20]}  "
              f"label={label}")
