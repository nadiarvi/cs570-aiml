# CLAUDE.md — GCN-Based Modifiability Prediction on UI Graphs
**CS570 Team 14 | Coding agent instructions**

**Task:** Given a Rico UI screen as a hierarchy graph G = (V, E) with per-node features x_v, learn f_θ : V → {0=canonical, 1=translatable, 2=open} for every node v ∈ V.

---

## Repository layout

```
ui-modifiability/
├── data/raw/          # Rico JSON hierarchy files
├── data/processed/    # Preprocessed .pt graph objects
├── data/splits/       # App-level split indices (JSON)
├── src/
│   ├── data/
│   │   ├── rico_loader.py      # Rico JSON → flat node/edge lists
│   │   ├── features.py         # Per-node feature extraction (d=420)
│   │   ├── labeler_llm.py      # Gemini labeling (primary)
│   │   ├── labeler.py          # Heuristic labeling (comparison)
│   │   ├── graph_builder.py    # Assemble PyTorch graph tensors
│   │   ├── dataset.py          # RicoGraphDataset + collate_graphs
│   │   ├── splits.py           # App-level 70/15/15 split logic
│   │   └── preprocess.py       # End-to-end preprocessing script
│   ├── models/
│   │   ├── mlp.py              # MLP baseline
│   │   ├── gcn.py              # 2-layer GCN (mean aggregation, scratch)
│   │   └── gat.py              # GAT with multi-head attention (scratch)
│   ├── train.py                # Shared training loop
│   ├── evaluate.py             # All metrics + figure generation
│   ├── ablation.py             # Ablation runner
│   └── tests/smoke_test.py     # Local dummy-tensor test
├── experiments/configs/        # JSON config per run
├── results/
│   ├── checkpoints/
│   ├── logs/
│   └── figures/
├── scripts/                    # Bash scripts (see section below)
├── .env                        # GEMINI_API_KEYS=key1,key2 (never commit)
└── requirements.txt
```

---

## Bash Scripts

All scripts live in `scripts/`. Run from repo root: `bash scripts/<name>.sh`.

### `scripts/setup.sh`
```bash
#!/usr/bin/env bash
set -e
conda create -n ui-gcn python=3.10 -y
conda activate ui-gcn
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y  # edit CUDA version if needed
pip install sentence-transformers scikit-learn tqdm pandas matplotlib seaborn datasets Pillow google-generativeai python-dotenv
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

### `scripts/download_data.sh`
```bash
#!/usr/bin/env bash
set -e
conda activate ui-gcn
python -c "from datasets import load_dataset; load_dataset('creative-graphic-design/Rico', 'ui_layout_vectors'); print('Done.')"
```

### `scripts/preprocess_pilot.sh`
```bash
#!/usr/bin/env bash
set -e
conda activate ui-gcn
python src/data/preprocess.py --rico_dir data/raw --out_dir data/processed --workers 4 --max_screens 500
echo "Pilot done. Check data/processed/ and label distribution."
```

### `scripts/preprocess_full.sh`
```bash
#!/usr/bin/env bash
set -e
conda activate ui-gcn
tmux new -s preprocess -d
tmux send-keys -t preprocess "conda activate ui-gcn && python src/data/preprocess.py --rico_dir data/raw --out_dir data/processed --workers 8" Enter
echo "Running in tmux 'preprocess'. Attach: tmux attach -t preprocess"
```

### `scripts/add_llm_labels.sh`
```bash
#!/usr/bin/env bash
set -e
# Run AFTER preprocess_full.sh. Patches y_llm into existing .pt files.
# Safe to stop and resume — cache skips already-labeled nodes.
[ ! -f .env ] && echo "Error: .env not found. Create: echo 'GEMINI_API_KEYS=key' > .env" && exit 1
conda activate ui-gcn
python src/data/preprocess.py --processed_dir data/processed --add_llm --max_screens 5000
echo "LLM labeling done. Cache: data/llm_label_cache.json"
```

### `scripts/train_mlp.sh`
```bash
#!/usr/bin/env bash
set -e
conda activate ui-gcn
tmux new -s train_mlp -d
tmux send-keys -t train_mlp "conda activate ui-gcn && python src/train.py --config experiments/configs/mlp_baseline.json" Enter
echo "Running in tmux 'train_mlp'."
```

### `scripts/train_gcn.sh`
```bash
#!/usr/bin/env bash
set -e
conda activate ui-gcn
tmux new -s train_gcn -d
tmux send-keys -t train_gcn "conda activate ui-gcn && python src/train.py --config experiments/configs/gcn_baseline.json" Enter
echo "Running in tmux 'train_gcn'."
```

### `scripts/train_gat.sh`
```bash
#!/usr/bin/env bash
set -e
conda activate ui-gcn
tmux new -s train_gat -d
tmux send-keys -t train_gat "conda activate ui-gcn && python src/train.py --config experiments/configs/gat_baseline.json" Enter
echo "Running in tmux 'train_gat'."
```

### `scripts/run_ablations.sh`
```bash
#!/usr/bin/env bash
set -e
conda activate ui-gcn
tmux new -s ablations -d
tmux send-keys -t ablations "conda activate ui-gcn && python src/ablation.py --output results/ablation_results.csv" Enter
echo "Running in tmux 'ablations'."
```

### `scripts/generate_figures.sh`
```bash
#!/usr/bin/env bash
set -e
conda activate ui-gcn
python src/evaluate.py --results_csv results/ablation_results.csv --logs_dir results/logs --figures_dir results/figures
ls results/figures/
```

---

## MAJOR COMPONENT 1 — Data Pipeline
**Goal: Construct the labeled, graph-structured Rico dataset.**

### 1.1 `src/data/rico_loader.py`

```python
def load_hierarchy(json_path: str) -> dict:
    """Load raw Rico JSON. Return root node dict."""

def flatten_hierarchy(root: dict) -> tuple[list[dict], list[tuple], list[tuple]]:
    """
    BFS/DFS traversal. Returns (nodes, containment_edges, sibling_edges).
    Per-node: node["depth"], node["sibling_count"], node["child_count"], node["ancestors"]
    Containment edges: (parent_idx, child_idx)
    Sibling edges: all C(k,2) undirected pairs for each parent with k children.
    Handle missing "text", "bounds", "resource-id" gracefully.
    """

def get_app_id(json_path: str) -> str:
    """Extract app package name from path — used for app-level splits."""
```

### 1.2 `src/data/features.py` — feature vector d = 420

**Visual (12):** from `bounds = [x1,y1,x2,y2]` on 1080×1920 canvas
```python
x_norm, y_norm = x1/1080, y1/1920
w_norm, h_norm = (x2-x1)/1080, (y2-y1)/1920
area_norm = w_norm * h_norm
aspect_ratio = w_norm / (h_norm + 1e-6)
cx_norm, cy_norm = ((x1+x2)/2)/1080, ((y1+y2)/2)/1920
# + 4 dims relative to parent bounds (dx1,dy1,dx2,dy2); zeros if no parent
```

**Structural (4):** `depth/max_depth`, `log(1+sibling_count)`, `log(1+child_count)`, `is_leaf`

**Type one-hot (20):** map `node["class"]` to:
```
TextView, ImageView, Button, EditText, LinearLayout, RelativeLayout,
FrameLayout, ScrollView, ListView, RecyclerView, ViewPager, CheckBox,
RadioButton, Switch, ImageButton, WebView, ProgressBar, SeekBar, Other, Unknown
```

**Text embedding (384):** `all-MiniLM-L6-v2` on `node["text"]` + `node["content-desc"]`; zero vector if both empty. **Cache embeddings to disk by text string — this is the bottleneck.**

```python
def extract_features(nodes: list[dict], feature_groups: list[str] = None) -> torch.FloatTensor:
    # feature_groups: None=all or subset ['visual','structural','type','text'] for ablation
    # for node in tqdm(nodes, desc="Extracting features", leave=False): ...
```

### 1.3 `src/data/labeler_llm.py` — Gemini labeling (primary)

**Setup:** `pip install google-generativeai python-dotenv`

Create `.env` in repo root (add to `.gitignore`):
```
GEMINI_API_KEYS=key1,key2,key3
```
Get free keys at https://aistudio.google.com/app/apikey — 1,500 requests/day per account.

```python
import os, google.generativeai as genai, json, hashlib, time
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

load_dotenv()
_api_keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS","").split(",") if k.strip()]
_current_key_idx = 0

def get_model():
    genai.configure(api_key=_api_keys[_current_key_idx])
    return genai.GenerativeModel("gemini-1.5-flash")  # free tier: 15 RPM, 1500 req/day

def rotate_key() -> bool:
    global _current_key_idx
    if _current_key_idx + 1 < len(_api_keys):
        _current_key_idx += 1
        print(f"Rotated to key {_current_key_idx+1}/{len(_api_keys)}")
        return True
    return False

CACHE_PATH = Path("data/llm_label_cache.json")
RATE_LIMIT_DELAY = 4.1  # seconds — stays under 15 RPM

def load_cache() -> dict:
    return json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}

def save_cache(cache: dict):
    CACHE_PATH.write_text(json.dumps(cache))
```

**Prompt:**
```python
SYSTEM_PROMPT = """You are a UI modifiability classifier for Android screens.
Classify a UI element as one of:
  0 = canonical   — Must NOT be modified. Identity-critical: prices, account IDs,
                    auth fields, legal text, CAPTCHAs, order numbers.
  1 = translatable — Form may change, meaning preserved. Nav labels, headers,
                     button text, section copy.
  2 = open         — Freely modifiable. Decorative images, banners, carousels,
                     promotional content.

Key rule: ancestor chain is your strongest signal. "$24.99" inside CheckoutFlow
is canonical; "$24.99" inside MarketingBanner is open. When uncertain, prefer canonical.
Respond ONLY with valid JSON: {"label": <0|1|2>, "reason": "<one sentence>"}"""

def build_node_prompt(node: dict, ancestors: list[dict]) -> str:
    chain = " → ".join(
        f"{a.get('class','?').split('.')[-1]}[{a.get('resource-id','')}]"
        for a in ancestors
    ) or "none"
    return (f"Class: {node.get('class','?').split('.')[-1]}\n"
            f"Text: \"{node.get('text','')}\"\n"
            f"Content-desc: \"{node.get('content-desc','')}\"\n"
            f"Resource-ID: \"{node.get('resource-id','')}\"\n"
            f"Depth: {node.get('depth',0)} | Children: {node.get('child_count',0)} "
            f"| Siblings: {node.get('sibling_count',0)}\n"
            f"Ancestors: {chain}")
```

**API call with caching, rate limiting, key rotation:**
```python
def get_llm_label(node: dict, ancestors: list[dict], cache: dict, retries: int = 3) -> int:
    prompt = build_node_prompt(node, ancestors)
    key = hashlib.sha256(prompt.encode()).hexdigest()
    if key in cache:
        return cache[key]  # no API call needed
    for attempt in range(retries):
        try:
            time.sleep(RATE_LIMIT_DELAY)
            response = get_model().generate_content(
                [{"role": "user", "parts": [SYSTEM_PROMPT + "\n\n" + prompt]}],
                generation_config={"temperature": 0.0, "max_output_tokens": 64},
            )
            label = int(json.loads(response.text.strip())["label"])
            assert label in (0, 1, 2)
            cache[key] = label; save_cache(cache)
            return label
        except Exception as e:
            err = str(e).lower()
            if "quota" in err or "resource_exhausted" in err or "429" in err:
                if rotate_key(): continue
                save_cache(cache)
                raise RuntimeError("All API keys exhausted. Resume tomorrow.")
            time.sleep(2 ** attempt * RATE_LIMIT_DELAY)
    return -1

def label_nodes_llm(nodes: list[dict], cache: dict) -> list[int]:
    return [get_llm_label(n, n["ancestors"], cache)
            for n in tqdm(nodes, desc="LLM labeling", leave=False)]
```

**Free tier capacity:**
| Screens | Est. API calls | Time at 15 RPM |
|---------|---------------|----------------|
| 500 (pilot) | ~6K | ~7 hrs |
| 5K (recommended) | ~50K | ~2.3 days |

Use `--max_screens 5000`. Resume anytime — cache saves after every label.

### 1.3b `src/data/labeler.py` — Heuristic labeling (comparison baseline)

```python
def assign_label(node: dict, ancestors: list[dict]) -> int:
    """
    First match wins.
    CANONICAL (0): currency regex, price text + checkout ancestor, resource-id contains
      price/amount/total/account/id/uid/balance, EditText + auth ancestor,
      email regex, CAPTCHA class/id, content-desc with account/profile/order/transaction.
    TRANSLATABLE (1): TextView depth<=3, Button + nav ancestor, resource-id contains
      title/header/label/section/heading, TextView with siblings at depth 2-5.
    OPEN (2): ImageView, Banner/Carousel/Ad class, promo resource-id, all remaining nodes.
    Return None if missing bounds AND text AND resource-id → y=-1 in graph builder.
    """
```
Expected distribution: ~70% open, ~20% translatable, ~10% canonical.

### 1.4 `src/data/graph_builder.py`

```python
def build_graph(nodes, containment_edges, sibling_edges, features,
                labels_heuristic, labels_llm=None, include_sibling_edges=True) -> dict:
    """
    Returns: x [N,d], y_heuristic [N], y_llm [N], edge_index [2,E],
             containment_edge_index [2,Ec], sibling_edge_index [2,Es], num_nodes.
    - Containment: add both directions (parent→child AND child→parent).
    - No self-loops here — GCN layer adds them internally.
    - edge_index dtype MUST be torch.long.
    """
```

### 1.5 `src/data/splits.py`

```python
def make_splits(all_json_paths, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    CRITICAL: split at APP level, not screen level.
    Screen-level splitting inflates val/test F1 by ~5-15% (apps reuse layouts).
    Steps: group by get_app_id() → shuffle app IDs → assign whole apps → flatten → save to
    data/splits/split_seed42.json. Returns (train_paths, val_paths, test_paths).
    """
```

### 1.6 `src/data/preprocess.py`

Two independent passes:
```bash
# Pass 1 — heuristic labels only (fast). Run first.
python src/data/preprocess.py --rico_dir data/raw --out_dir data/processed --workers 8

# Pass 2 — add LLM labels to existing .pt files. Run in background after Pass 1.
python src/data/preprocess.py --processed_dir data/processed --add_llm --max_screens 5000
```

```python
# Pass 1
for json_path in tqdm(all_json_paths, desc="Preprocessing"):
    nodes, c_edges, s_edges = flatten_hierarchy(load_hierarchy(json_path))
    features = extract_features(nodes)
    labels_h = [assign_label(n, n["ancestors"])
                for n in tqdm(nodes, desc="  Heuristic", leave=False)]
    graph = build_graph(nodes, c_edges, s_edges, features, labels_h)
    torch.save(graph, out_path)

# Pass 2 — patches y_llm without touching y_heuristic
cache = load_cache()
for pt_path in tqdm(all_pt_paths, desc="Adding LLM labels"):
    graph = torch.load(pt_path)
    labels_l = label_nodes_llm(graph["nodes"], cache)
    graph["y_llm"] = torch.tensor([-1 if l is None else l for l in labels_l], dtype=torch.long)
    torch.save(graph, pt_path)
```

---

## MAJOR COMPONENT 2 — Model Architectures
**Goal: MLP, GCN, GAT from scratch. No PyTorch Geometric. All share `forward(x, edge_index) → [N, 3] logits`.**

### 2.1 `src/models/mlp.py`
```python
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes=3, num_layers=3, dropout=0.3):
        # Linear → ReLU → Dropout → [repeat] → Linear

    def forward(self, x, edge_index=None):  # edge_index ignored
        ...
```

### 2.2 `src/models/gcn.py`
```python
class GCNLayer(nn.Module):
    def forward(self, x, edge_index):
        # 1. Add self-loops  2. Compute degree  3. Scatter-add → divide by degree (mean)
        # 4. Linear transform  5. ReLU (except last layer)
        src, dst = edge_index
        agg = torch.zeros(N, in_dim, device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, in_dim), x[src])
        deg = torch.bincount(dst, minlength=N).float().clamp(min=1)
        agg = agg / deg.unsqueeze(1)
        return agg @ self.weight.T + self.bias

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes=3, num_layers=2, dropout=0.3):
        # Stack GCNLayers. ReLU+Dropout after each except last. Final layer = logits.
```

### 2.3 `src/models/gat.py`
```python
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.3):
        # Per head k: W_k [in_dim → out_dim/heads], attention a_k [2*out_dim/heads]

    def forward(self, x, edge_index):
        # Per head: linear → e_ij = LeakyReLU(a^T [h_i||h_j]) → scatter softmax → weighted agg
        # Softmax: subtract max per dst for stability, then exp/sum, then normalize
        # Concat heads → [N, out_dim]

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes=3, num_layers=2, num_heads=4, dropout=0.3):
        # Stack GATLayers. Final layer: single head, no activation.
```

---

## MAJOR COMPONENT 3 — Training Infrastructure

### 3.1 `src/data/dataset.py`
```python
class RicoGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_paths, include_sibling_edges=True,
                 feature_groups=None, label_source="llm"):  # "llm" | "heuristic"
        # Lazy load — torch.load in __getitem__, NOT __init__

    def __getitem__(self, idx):
        graph = torch.load(self.graph_paths[idx])
        graph["y"] = graph[f"y_{self.label_source}"]
        return graph

def collate_graphs(batch) -> dict:
    # Concat N graphs into one disconnected graph.
    # Offset edge_index for each graph by cumulative node count.
    # Return x, y, edge_index, batch_mask [sum(N_i)].
```

### 3.2 `src/train.py`

**Config keys:** `model_type`, `in_dim`, `hidden_dim`, `num_layers`, `num_heads`, `dropout`, `lr`, `weight_decay`, `epochs`, `patience`, `batch_size`, `include_sibling_edges`, `feature_groups`, `label_source`, `device`, `save_dir`

**Loss:**
```python
counts = torch.bincount(all_train_labels)
weights = (1.0 / counts.float()); weights /= weights.sum()
criterion = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=-1)
```

**Loop:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

epoch_bar = tqdm(range(config["epochs"]), desc="Epochs")
for epoch in epoch_bar:
    model.train()
    batch_bar = tqdm(train_loader, desc=f"  Epoch {epoch+1}", leave=False)
    for batch in batch_bar:
        x, edge_index, y = batch["x"].to(device), batch["edge_index"].to(device), batch["y"].to(device)
        loss = criterion(model(x, edge_index), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        batch_bar.set_postfix(loss=f"{loss.item():.4f}")
    val_f1 = evaluate(model, val_loader, device)["macro_f1"]
    scheduler.step(val_f1)
    epoch_bar.set_postfix(val_f1=f"{val_f1:.4f}")
    # early stopping on val macro_f1; save best checkpoint
```

**Default hyperparameters:**
| Param | MLP | GCN | GAT |
|-------|-----|-----|-----|
| hidden_dim | 256 | 256 | 256 |
| num_layers | 3 | 2 | 2 |
| num_heads | — | — | 4 |
| dropout/lr/wd | 0.3/1e-3/1e-4 | same | same |
| epochs/patience/batch | 100/15/32 | same | same |

---

## MAJOR COMPONENT 4 — Experiments, Evaluation & Ablations

### 4.1 `src/evaluate.py`
```python
def evaluate(model, loader, device) -> dict:
    # Returns: macro_f1, per_class_f1 {0,1,2}, per_class_precision {0,1,2},
    #          per_class_recall {0,1,2}, accuracy, confusion_matrix [3,3]
    # Use sklearn: f1_score(average='macro'), precision_recall_fscore_support, confusion_matrix
    # Exclude nodes where y == -1

def evaluate_app_holdout(model, test_loader, device) -> dict:
    # Run evaluate() on test split ONLY. This is the ONLY split reported in the paper.

def plot_confusion_matrix(cm, run_name, save_path):
    # Seaborn heatmap, labels=['canonical','translatable','open'], normalize by row
    # → results/figures/<run_name>_confusion.png

def plot_training_curves(log_csv, run_name, save_path):
    # Two subplots: train loss + val Macro-F1 over epochs
    # → results/figures/<run_name>_curves.png

def plot_macro_f1_bar(results_csv, save_path):   # → results/figures/model_comparison.png
def plot_ablation_bar(results_csv, save_path):   # → results/figures/ablation_comparison.png
```

**Results table** (all numbers from app-level test holdout only):

| Run | Label Source | Macro-F1 | F1-can | F1-trans | F1-open | P-can | R-can | Acc |
|-----|-------------|----------|--------|----------|---------|-------|-------|-----|

### 4.2 `src/ablation.py`
```python
ABLATION_CONFIGS = [
    # Track A — heuristic labels
    {"name": "mlp_all",        "model": "mlp", "edges": "all",         "features": "all",     "layers": 3, "label_source": "heuristic"},
    {"name": "gcn_2l_all",     "model": "gcn", "edges": "all",         "features": "all",     "layers": 2, "label_source": "heuristic"},
    {"name": "gat_2l_all",     "model": "gat", "edges": "all",         "features": "all",     "layers": 2, "label_source": "heuristic"},
    {"name": "gcn_2l_contain", "model": "gcn", "edges": "containment", "features": "all",     "layers": 2, "label_source": "heuristic"},
    {"name": "gcn_2l_notext",  "model": "gcn", "edges": "all",         "features": "no_text", "layers": 2, "label_source": "heuristic"},
    {"name": "gcn_1l_all",     "model": "gcn", "edges": "all",         "features": "all",     "layers": 1, "label_source": "heuristic"},
    {"name": "gcn_3l_all",     "model": "gcn", "edges": "all",         "features": "all",     "layers": 3, "label_source": "heuristic"},
    # Track B — LLM labels (run after add_llm_labels.sh completes)
    {"name": "mlp_all_llm",    "model": "mlp", "edges": "all",         "features": "all",     "layers": 3, "label_source": "llm"},
    {"name": "gcn_2l_all_llm", "model": "gcn", "edges": "all",         "features": "all",     "layers": 2, "label_source": "llm"},
    {"name": "gat_2l_all_llm", "model": "gat", "edges": "all",         "features": "all",     "layers": 2, "label_source": "llm"},
]
# CSV columns: name, model, edges, features, layers, label_source,
#   macro_f1, f1_canonical, f1_translatable, f1_open,
#   p_canonical, r_canonical, p_translatable, r_translatable, p_open, r_open, per_node_accuracy
```

### 4.3 Required figures

| Figure | Path |
|--------|------|
| Macro-F1 bar: MLP vs GCN vs GAT | `results/figures/model_comparison.png` |
| Confusion matrix — MLP | `results/figures/mlp_all_confusion.png` |
| Confusion matrix — best GCN | `results/figures/gcn_2l_all_confusion.png` |
| Ablation bar chart | `results/figures/ablation_comparison.png` |
| Training curves (per run) | `results/figures/<run>_curves.png` |

### 4.4 App-level holdout checklist
- [ ] `splits.py` groups by `get_app_id()` — whole apps assigned to splits
- [ ] Split indices saved to `data/splits/split_seed42.json` before preprocessing
- [ ] `evaluate_app_holdout()` runs only on test split paths
- [ ] No test-split app screen appears in train or val

---

## Experiment configs

```json
// experiments/configs/mlp_baseline.json
{"model_type":"mlp","in_dim":420,"hidden_dim":256,"num_layers":3,"dropout":0.3,
 "lr":1e-3,"weight_decay":1e-4,"epochs":100,"patience":15,"batch_size":32,
 "include_sibling_edges":false,"feature_groups":null,"label_source":"heuristic",
 "device":"cuda","save_dir":"results/checkpoints/mlp_baseline"}

// experiments/configs/gcn_baseline.json
{"model_type":"gcn","in_dim":420,"hidden_dim":256,"num_layers":2,"dropout":0.3,
 "lr":1e-3,"weight_decay":1e-4,"epochs":100,"patience":15,"batch_size":32,
 "include_sibling_edges":true,"feature_groups":null,"label_source":"heuristic",
 "device":"cuda","save_dir":"results/checkpoints/gcn_2l_all"}

// experiments/configs/gat_baseline.json
{"model_type":"gat","in_dim":420,"hidden_dim":256,"num_layers":2,"num_heads":4,"dropout":0.3,
 "lr":1e-3,"weight_decay":1e-4,"epochs":100,"patience":15,"batch_size":32,
 "include_sibling_edges":true,"feature_groups":null,"label_source":"heuristic",
 "device":"cuda","save_dir":"results/checkpoints/gat_2l_all"}
```

---

## Development Workflow — local → GitHub → GPU server

**Local:** write all code, test with dummy tensors only (no dataset needed).

```python
# src/tests/smoke_test.py — run before every push
import torch
from src.models.mlp import MLP; from src.models.gcn import GCN; from src.models.gat import GAT
N = 50; x = torch.randn(N, 420); edge_index = torch.randint(0,N,(2,100)).long()
y = torch.randint(0, 3, (N,)).long()
for Cls, kw in [(MLP,{}),(GCN,{}),(GAT,{"num_heads":4})]:
    model = Cls(in_dim=420, hidden_dim=256, num_classes=3, **kw)
    logits = model(x, edge_index)
    assert logits.shape == (N, 3)
    torch.nn.CrossEntropyLoss()(logits, y).backward()
    print(f"{Cls.__name__} ✓")
```
```bash
python src/tests/smoke_test.py   # < 5 seconds on CPU
```

**GPU server** (after pushing to GitHub):
```bash
git clone https://github.com/<your-repo>/ui-modifiability.git && cd ui-modifiability
bash scripts/setup.sh
bash scripts/download_data.sh         # download Rico directly here — never transfer from local
echo "GEMINI_API_KEYS=your_key" > .env
```

---

## Experiment Guide

Two parallel tracks — Track A (heuristic, fast) gets you results immediately; Track B (LLM) runs in the background.

```
Track A — heuristic labels                     Track B — LLM labels
──────────────────────────                     ─────────────────────
Step 1: setup.sh + download_data.sh            │
Step 2: preprocess_pilot.sh                    │
Step 3: preprocess_full.sh          ───────────┤→ Step B1: add_llm_labels.sh (background)
Step 4: train_mlp/gcn/gat.sh                   │
Step 5: run_ablations.sh                       │
Step 6: generate_figures.sh         ←──────────┘  Step B2: run_ablations.sh (*_llm configs)
                                                   Step B3: compare → pick better method
```

| Step | Script | Goal | Expected output |
|------|--------|------|-----------------|
| 1 | `setup.sh` | GPU confirmed | `GPU available: True` |
| 2 | `download_data.sh` + `preprocess_pilot.sh` | Pipeline runs end-to-end on 500 screens; verify label distribution (~70% open) | ~500 `.pt` files |
| 3 | `preprocess_full.sh` | All ~72K screens preprocessed with heuristic labels | ~72K `.pt` files + `split_seed42.json` |
| B1 | `add_llm_labels.sh` (tmux) | `y_llm` patched into `.pt` files in background | Cache at `data/llm_label_cache.json` |
| 4 | `train_mlp/gcn/gat.sh` (3 tmux sessions) | MLP/GCN/GAT trained on heuristic labels | 3 checkpoints + 3 log CSVs |
| 5 | `run_ablations.sh` | All ablation rows populated | `results/ablation_results.csv` |
| 6 | `generate_figures.sh` | All presentation figures | PNG files in `results/figures/` |
| B2 | `run_ablations.sh` | Re-run with `*_llm` configs after Track B done | LLM rows in CSV |
| B3 | — | Compare Macro-F1 across label sources; pick better for report | — |

---

## Evaluation Spec

| Metric | How computed | Note |
|--------|-------------|------|
| **Macro-F1** (primary) | `f1_score(average='macro')` | Weights all 3 classes equally |
| Per-class F1/P/R | `precision_recall_fscore_support(average=None)` | Diagnoses per-class failures |
| Per-node Accuracy | `accuracy_score` (y ≠ -1 only) | Sanity check only |
| App-level holdout | Test split (15%) only | Prevents layout-template leakage |
| Confusion matrix | Normalized by row | Shows canonical ↔ open confusion |

Good result: large MLP → GCN Macro-F1 gap confirms relational hypothesis. If gap is small, report it honestly — flat heuristics sufficing is also a valid finding. If macro_f1 ≈ 0.33, model is collapsing to "open" → check weighted loss.

---

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Screen-level split | Always split by `get_app_id()` |
| No embedding cache | Cache text→embedding to disk; without it, ~6–8 hrs for full dataset |
| Loading all graphs at init | Lazy load in `__getitem__` — loading 72K graphs at once OOMs |
| No weighted loss | `nn.CrossEntropyLoss(weight=...)` — without it, model predicts "open" always |
| `edge_index` as float | Must be `torch.long` — float causes silent CUDA errors |
| Reporting val F1 as final | Report ONLY test split numbers |
| `.env` missing | `echo 'GEMINI_API_KEYS=key' > .env` — keys not in git |
| Daily quota hit | Script auto-rotates keys; if all exhausted, re-run tomorrow — cache resumes |

---

## Timeline

| Date | Deliverable |
|------|-------------|
| **May 25** | MLP has a real Macro-F1 number. GCN + GAT training without diverging. |
| **May 29** | All ablation configs complete. `ablation_results.csv` populated. No new experiments. |
| **May 30** | All figures generated. Slide draft assembled. |
| **May 31** | 5-min rehearsal. Slide deck finalized. |
| **June 1 @ 11:59 PM** | PPT submitted via Google Form. |
| **June 7** | All report sections drafted. |
| **June 12** | Full report assembled + internal review. |
| **June 21 @ 11:59 PM** | Final report submitted via Google Form. |
