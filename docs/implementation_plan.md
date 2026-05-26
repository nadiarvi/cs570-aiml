# Implementation Plan — GCN-Based Modifiability Prediction on UI Graphs
**CS570 Team 14 | For coding agent use**

---

## Project Overview

**Task:** Per-node multi-class classification on UI hierarchy graphs.  
**Input:** A UI screen rendered as a hierarchy graph G = (V, E) with per-node features x_v.  
**Output:** Label y_v ∈ {0=canonical, 1=translatable, 2=open} for every node v ∈ V.  
**Dataset:** Rico (~72K Android UI screens with view hierarchies).  
**Models:** MLP baseline → GCN → GAT (all implemented from scratch in PyTorch, no PyG).  
**Environment:** GPU server, Python 3.10+, PyTorch with CUDA.

---

## Repository Structure

```
ui-modifiability/
├── data/
│   ├── raw/                    # Rico json hierarchy files (download here)
│   ├── processed/              # Preprocessed graph objects (.pt files)
│   └── splits/                 # Train/val/test app-level split indices
├── src/
│   ├── data/
│   │   ├── rico_loader.py      # Rico JSON → Python dicts
│   │   ├── graph_builder.py    # Dict → PyTorch graph tensors
│   │   ├── labeler.py          # Heuristic labeling rules
│   │   ├── features.py         # Feature extraction per node
│   │   └── splits.py           # App-level 70/15/15 split logic
│   ├── models/
│   │   ├── mlp.py              # MLP baseline (no graph structure)
│   │   ├── gcn.py              # 2-layer GCN (mean aggregation)
│   │   └── gat.py              # GAT with learned attention weights
│   ├── train.py                # Training loop (shared across models)
│   ├── evaluate.py             # Macro-F1, confusion matrix, per-class stats
│   └── ablation.py             # Runs ablation matrix, logs to CSV
├── experiments/
│   └── configs/                # JSON configs for each run
├── results/
│   ├── checkpoints/            # Saved model weights
│   ├── logs/                   # Training logs (loss, F1 per epoch)
│   └── figures/                # Plots and confusion matrices
├── requirements.txt
└── README.md
```

---

## Environment Setup

```bash
# requirements.txt
torch>=2.1.0
torchvision
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
sentence-transformers   # for text embeddings (MiniLM)
Pillow
```

```bash
# On GPU server
pip install -r requirements.txt

# Confirm CUDA is available
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Phase 1 — Data Pipeline

### 1.1 Rico Dataset Download

Rico is publicly available. Download the UI screenshots + view hierarchy JSONs.

```
Source: http://interactionmining.org/rico
Files needed:
  - ui_layout_vectors.zip  (view hierarchy JSONs)
  - unique_uis.tar.gz      (optional: screenshots, needed only for visual features)

Target directory: data/raw/
Expected structure:
  data/raw/<app_id>/<screen_id>.json   # ~72,000 JSON files
```

Each JSON file contains a nested dict representing the view hierarchy of one screen. Key fields per node:
- `"class"` — widget type (e.g., `android.widget.TextView`)
- `"bounds"` — `[x1, y1, x2, y2]` bounding box in pixels
- `"text"` — visible text string (may be empty)
- `"children"` — list of child node dicts (recursive)
- `"resource-id"` — Android resource ID string
- `"content-desc"` — accessibility label

---

### 1.2 Rico Loader (`src/data/rico_loader.py`)

**Purpose:** Load a single Rico JSON file and flatten the nested hierarchy into a list of nodes and edges.

**Function signatures to implement:**

```python
def load_hierarchy(json_path: str) -> dict:
    """Load raw Rico JSON. Return the root node dict."""

def flatten_hierarchy(root: dict) -> tuple[list[dict], list[tuple[int,int]], list[tuple[int,int]]]:
    """
    Flatten a nested Rico hierarchy into:
      - nodes: list of node attribute dicts, one per UI element
      - containment_edges: list of (parent_idx, child_idx) tuples
      - sibling_edges: list of (child_i_idx, child_j_idx) for all pairs sharing a parent
    
    Uses BFS or DFS traversal. Assign each node an integer index in traversal order.
    Returns (nodes, containment_edges, sibling_edges).
    """

def get_app_id(json_path: str) -> str:
    """Extract app package name from file path (used for app-level splits)."""
```

**Implementation notes:**
- Handle missing fields gracefully (some nodes lack `"text"` or `"bounds"`).
- Sibling edges: for every parent with k children, add all C(k,2) undirected pairs as sibling edges.
- Mark each node's `depth` during traversal (root = 0).
- Mark each node's `sibling_count` = number of siblings (children of the same parent).

---

### 1.3 Feature Extraction (`src/data/features.py`)

Each node gets a feature vector x_v ∈ ℝ^d composed of the following groups:

#### Visual features (12 dims)
```python
# From "bounds": [x1, y1, x2, y2] on a 1080×1920 canvas
x_norm   = x1 / 1080
y_norm   = y1 / 1920
w_norm   = (x2 - x1) / 1080
h_norm   = (y2 - y1) / 1920
area_norm = w_norm * h_norm
aspect_ratio = w_norm / (h_norm + 1e-6)
cx_norm  = ((x1 + x2) / 2) / 1080   # center x
cy_norm  = ((y1 + y2) / 2) / 1920   # center y
# Relative position within parent bounds (4 dims: dx1, dy1, dx2, dy2)
# If no parent, use 0s
```

#### Structural features (4 dims)
```python
depth              # int, normalized by max depth in tree (float)
sibling_count      # int, normalized by log(1 + count)
child_count        # int, normalized by log(1 + count)
is_leaf            # 1 if no children, else 0
```

#### Type one-hot (20 dims)
Map the `"class"` field to a fixed 20-category vocabulary. Suggested categories:
```
TextView, ImageView, Button, EditText, LinearLayout, RelativeLayout,
FrameLayout, ScrollView, ListView, RecyclerView, ViewPager, CheckBox,
RadioButton, Switch, ImageButton, WebView, ProgressBar, SeekBar,
Other, Unknown
```
One-hot encode. Anything not in the list → "Other".

#### Text embedding (384 dims)
```python
# Use sentence-transformers: all-MiniLM-L6-v2
# Input: node["text"] or node["content-desc"] (whichever is non-empty; concat both if both exist)
# If both are empty, use a zero vector of dim 384
# Cache embeddings to disk (embedding computation is the bottleneck)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

**Total feature dim d = 12 + 4 + 20 + 384 = 420.**

Store features as a float32 tensor of shape `[num_nodes, 420]`.

**Ablation note:** The feature ablation experiment needs to run with subsets. Add a `feature_groups` argument to feature extraction that controls which groups are included (default: all).

---

### 1.4 Heuristic Labeling Rules (`src/data/labeler.py`)

Labels are assigned per-node. The heuristic rules apply high-precision patterns grounded in BrowserArena failure analysis. Rules are applied in priority order — first match wins.

#### Label definitions:
- `0 = canonical` — must not be altered (identity-critical)
- `1 = translatable` — form may change, meaning preserved
- `2 = open` — freely modifiable

#### Rule set (implement as ordered if-elif chain):

```python
def assign_label(node: dict, ancestors: list[dict]) -> int:
    """
    node: attribute dict for this node
    ancestors: list of ancestor node dicts from root → parent (in order)
    Returns: 0 (canonical), 1 (translatable), or 2 (open)
    """
```

**Canonical rules (label = 0) — highest priority:**
1. Node text matches currency pattern: `r'\$[\d,]+(\.\d{2})?'`
2. Node text matches price-like pattern and ancestor class contains "Checkout", "Cart", "Payment", "Order", or "Price"
3. Node resource-id contains: "price", "amount", "total", "account", "id", "uid", "balance"
4. Node class is `EditText` AND ancestor resource-id contains "account", "email", "username", "password", "phone"
5. Node text matches email pattern: `r'[\w.]+@[\w.]+\.\w+'`
6. Node class contains "CAPTCHA" or resource-id contains "captcha"
7. Node content-desc contains "account", "profile", "order id", "transaction"
8. Ancestor class chain contains "CheckoutFlow", "PaymentView", "AccountView" (from resource-id heuristics)

**Translatable rules (label = 1) — medium priority:**
1. Node class is `TextView` AND depth ≤ 3 (likely a header/section title)
2. Node class is `Button` AND ancestor class is a navigation component (resource-id contains "nav", "menu", "tab")
3. Node resource-id contains "title", "header", "label", "section", "heading"
4. Node class is `TextView` AND sibling_count ≥ 1 AND child_count == 0 AND depth is between 2-5

**Open rules (label = 2) — default / lowest priority:**
1. Node class is `ImageView` AND not matched by canonical rules
2. Node class contains "Banner", "Carousel", "Recommendation", "Ad"
3. Node resource-id contains "banner", "promo", "ads", "recommendation", "hero"
4. All remaining nodes → label 2 (open) as the default fallback

**Implementation notes:**
- Build `ancestors` list during BFS/DFS traversal in `rico_loader.py` — pass it alongside each node.
- Log label distribution per screen; expect heavy class imbalance (open >> translatable >> canonical).
- Return `None` for nodes with insufficient features (e.g., missing bounds AND missing text AND no resource-id) — these will be excluded from training.

---

### 1.5 Graph Builder (`src/data/graph_builder.py`)

Assemble the per-screen graph into PyTorch tensors.

```python
def build_graph(
    nodes: list[dict],
    containment_edges: list[tuple],
    sibling_edges: list[tuple],
    features: torch.Tensor,      # shape [N, 420]
    labels: list[int],           # length N, -1 for unlabeled
    include_sibling_edges: bool = True,   # ablation flag
) -> dict:
    """
    Returns a dict:
      {
        "x": torch.FloatTensor [N, 420],
        "y": torch.LongTensor [N],            # -1 = exclude from loss
        "edge_index": torch.LongTensor [2, E], # both containment + sibling edges, undirected
        "containment_edge_index": torch.LongTensor [2, E_c],
        "sibling_edge_index": torch.LongTensor [2, E_s],
        "num_nodes": int,
      }
    """
```

**Edge construction:**
- Containment edges: add both directions (parent→child AND child→parent) to make the graph undirected.
- Sibling edges: already undirected.
- For ablation: `include_sibling_edges=False` omits sibling edges from `edge_index`.
- Self-loops: DO NOT add self-loops here. The GCN layer will handle them internally.

---

### 1.6 App-Level Dataset Splits (`src/data/splits.py`)

**Critical:** Split at the app level, not the screen level. Screens from the same app share layouts — screen-level splitting would leak structural patterns.

```python
def make_splits(
    all_json_paths: list[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Groups paths by app_id, shuffles apps (not screens), then assigns
    whole apps to train/val/test.
    Returns (train_paths, val_paths, test_paths).
    """
```

**Implementation:**
1. Group all JSON paths by `get_app_id()` → `{app_id: [path1, path2, ...]}`
2. Shuffle app IDs with `random.seed(seed)`
3. Assign app IDs to splits by ratio
4. Flatten app IDs back to screen paths
5. Save split indices to `data/splits/split_seed42.json`

---

### 1.7 Preprocessing Pipeline (`src/data/preprocess.py`)

End-to-end script that processes all Rico JSONs and saves `.pt` files.

```python
# Usage: python src/data/preprocess.py --rico_dir data/raw --out_dir data/processed --workers 8

for json_path in tqdm(all_json_paths):
    nodes, containment_edges, sibling_edges = flatten_hierarchy(load_hierarchy(json_path))
    features = extract_features(nodes)           # [N, 420]
    labels = [assign_label(n, ancestors[i]) for i, n in enumerate(nodes)]
    graph = build_graph(nodes, containment_edges, sibling_edges, features, labels)
    torch.save(graph, out_path)
```

**Important:** Use `multiprocessing.Pool` with `--workers` for parallelism. Text embedding is the bottleneck — cache embeddings keyed by text string to avoid recomputing.

Expected output: ~72K `.pt` files, ~50–200 KB each.

---

## Phase 2 — Models

All models take per-node features as input and output per-node class logits.

### 2.1 MLP Baseline (`src/models/mlp.py`)

No graph structure. Processes each node independently.

```python
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 3, dropout: float = 0.3):
        """
        Architecture: Linear → ReLU → Dropout → [repeat] → Linear (output)
        in_dim: 420 (full features) or subset for ablation
        num_classes: 3
        """

    def forward(self, x: torch.Tensor, edge_index=None) -> torch.Tensor:
        """
        x: [N, in_dim]
        edge_index: ignored (signature kept for compatibility with training loop)
        Returns: [N, num_classes] logits
        """
```

---

### 2.2 GCN (`src/models/gcn.py`)

Mean-aggregation GCN implemented from scratch. No PyTorch Geometric.

**Core GCN layer:**

```python
class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Single GCN layer: h_v^(l+1) = σ( W · MEAN_{u ∈ N(v) ∪ {v}} h_u^(l) )"""

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        x: [N, in_dim] node features
        edge_index: [2, E] edge list (source, destination), already includes self-loops
        
        Steps:
        1. Add self-loops to edge_index (torch.arange(N) stacked twice).
        2. Compute degree D[v] = number of edges incident to v (including self-loop).
        3. For each node v, aggregate: h_agg[v] = SUM_{u in N(v) ∪ {v}} x[u] / D[v]
           (mean aggregation = sum / degree)
        4. Apply linear transform: out = h_agg @ W.T
        5. Apply activation (ReLU) — except on the last layer.
        Implementation hint: use torch.zeros(N, out_dim).scatter_add_ for the aggregation step.
        """
```

**Full GCN model:**

```python
class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2, dropout: float = 0.3):
        """
        Stack num_layers GCNLayers.
        After each layer (except last): ReLU + Dropout.
        Final layer outputs logits (no activation).
        """

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """Returns [N, num_classes] logits."""
```

**Aggregation implementation detail (efficient scatter_add approach):**
```python
# edge_index shape: [2, E], edge_index[0] = sources, edge_index[1] = destinations
# For each destination node v, sum all source features:
src, dst = edge_index
agg = torch.zeros(N, in_dim, device=x.device)
agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, in_dim), x[src])
deg = torch.bincount(dst, minlength=N).float().clamp(min=1)
agg = agg / deg.unsqueeze(1)   # mean
out = agg @ self.weight.T + self.bias
```

---

### 2.3 GAT (`src/models/gat.py`)

Graph Attention Network with single-head attention (can extend to multi-head).

```python
class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.3):
        """
        For each head k: learn W_k [in_dim → out_dim/num_heads] and attention vector a_k [2 * out_dim/num_heads]
        Concat heads at the end.
        """

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        Steps per head:
        1. Linear transform: h = x @ W.T  → [N, head_dim]
        2. For each edge (i,j): e_ij = LeakyReLU(a^T [h_i || h_j])   (concat)
        3. Softmax over incoming edges per destination node: α_ij = softmax_j(e_ij)
        4. Aggregate: h_v = σ( SUM_j α_vj * h_j )
        Implementation hint: use torch_sparse or manual scatter_softmax.
        For softmax over neighbors: subtract max before exp for numerical stability.
        """

class GAT(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.3):
        """Stack GATLayers. Final layer: single head, no activation."""
```

**Scatter softmax implementation (no external library):**
```python
# For edges (src, dst) with attention scores e:
# 1. Compute max per destination for stability
max_e = torch.full((N,), float('-inf'), device=x.device)
max_e.scatter_reduce_(0, dst, e, reduce='amax', include_self=True)
e_stable = e - max_e[dst]
exp_e = torch.exp(e_stable)
# 2. Sum exp per destination
sum_exp = torch.zeros(N, device=x.device)
sum_exp.scatter_add_(0, dst, exp_e)
# 3. Normalize
alpha = exp_e / (sum_exp[dst] + 1e-9)
```

---

## Phase 3 — Training

### 3.1 Dataset and DataLoader (`src/data/dataset.py`)

```python
class RicoGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_paths: list[str], include_sibling_edges: bool = True,
                 feature_groups: list[str] = None):
        """
        Loads preprocessed .pt graph files.
        include_sibling_edges: controls edge_index composition (ablation)
        feature_groups: subset of ['visual','structural','type','text'] — None means all
        """

    def __getitem__(self, idx) -> dict:
        """Returns graph dict for one screen."""

    def __len__(self) -> int: ...
```

**DataLoader note:** Each screen is one graph (variable node count). Use `batch_size=1` with a custom collate, or implement mini-batching by concatenating node feature matrices and offsetting edge indices. Mini-batching is recommended for GPU efficiency:

```python
def collate_graphs(batch: list[dict]) -> dict:
    """
    Concatenate N graphs into one large disconnected graph:
      - x: torch.cat([g["x"] for g in batch], dim=0)  → [sum(N_i), D]
      - y: torch.cat([g["y"] for g in batch], dim=0)
      - edge_index: cat and offset each graph's edges by cumulative node count
      - batch_mask: [sum(N_i)] int tensor — which graph each node belongs to (for pooling if needed)
    """
```

---

### 3.2 Training Loop (`src/train.py`)

```python
def train(config: dict) -> None:
    """
    config keys:
      model_type: "mlp" | "gcn" | "gat"
      in_dim, hidden_dim, num_layers, dropout
      lr, weight_decay, epochs, patience (early stopping)
      include_sibling_edges, feature_groups
      device: "cuda" or "cpu"
      save_dir: path to save checkpoints and logs
    """
```

**Loss function:** Weighted cross-entropy to handle class imbalance.
```python
# Compute class weights from training label distribution
counts = torch.bincount(all_train_labels)   # [3]
weights = 1.0 / counts.float()
weights = weights / weights.sum()           # normalize
criterion = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=-1)
# ignore_index=-1 skips nodes with no label
```

**Optimizer:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
```

**Training loop structure:**
```python
for epoch in range(config["epochs"]):
    model.train()
    for batch in train_loader:
        x, edge_index, y = batch["x"].to(device), batch["edge_index"].to(device), batch["y"].to(device)
        logits = model(x, edge_index)       # [N, 3]
        loss = criterion(logits, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    val_f1 = evaluate(model, val_loader, device)
    scheduler.step(val_f1)
    
    # Early stopping: stop if val_f1 hasn't improved for `patience` epochs
    # Save best checkpoint by val Macro-F1
    
    log(epoch, train_loss, val_f1)
```

**Hyperparameter defaults:**
```
lr: 1e-3
weight_decay: 1e-4
hidden_dim: 256
num_layers: 2 (GCN/GAT), 3 (MLP)
dropout: 0.3
epochs: 100
patience: 15
batch_size: 32 (graphs per batch)
```

---

### 3.3 Evaluation (`src/evaluate.py`)

```python
def evaluate(model, loader, device) -> dict:
    """
    Returns:
      macro_f1: float (primary metric)
      per_class_f1: dict {0: f1_canonical, 1: f1_translatable, 2: f1_open}
      accuracy: float
      confusion_matrix: np.ndarray [3,3]
    """

def plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    """Seaborn heatmap, class names: canonical / translatable / open"""

def plot_training_curves(log_path: str, save_path: str) -> None:
    """Plot train loss and val Macro-F1 over epochs"""
```

**Primary metric:** Macro-F1. Canonical is rarest and highest-stakes — macro averaging weights all classes equally, preventing the metric from being dominated by the easy "open" class.

---

## Phase 4 — Ablation Experiments

### 4.1 Ablation Matrix

Run every combination of the following:

| Factor | Variants |
|--------|---------|
| Model | MLP, GCN, GAT |
| Edge type | containment-only, containment+sibling |
| Feature group | visual+structural+type (no text), all (+ text embedding) |
| GCN/GAT depth | 1 layer, 2 layers, 3 layers |

**Core comparison** (highest priority — must run before presentation):
1. MLP (all features, N/A edges)
2. GCN 2-layer (containment+sibling, all features)
3. GAT 2-layer (containment+sibling, all features)

**Edge ablation** (GCN only):
4. GCN 2-layer (containment only, all features)

**Feature ablation** (GCN only):
5. GCN 2-layer (containment+sibling, no text embedding)

**Depth ablation** (GCN only):
6. GCN 1-layer (containment+sibling, all features)
7. GCN 3-layer (containment+sibling, all features)

### 4.2 Ablation Runner (`src/ablation.py`)

```python
# python src/ablation.py --output results/ablation_results.csv

ABLATION_CONFIGS = [
    {"name": "mlp_all",           "model": "mlp", "edges": "all",          "features": "all", "layers": 3},
    {"name": "gcn_2l_all",        "model": "gcn", "edges": "all",          "features": "all", "layers": 2},
    {"name": "gat_2l_all",        "model": "gat", "edges": "all",          "features": "all", "layers": 2},
    {"name": "gcn_2l_contain",    "model": "gcn", "edges": "containment",  "features": "all", "layers": 2},
    {"name": "gcn_2l_notext",     "model": "gcn", "edges": "all",          "features": "no_text", "layers": 2},
    {"name": "gcn_1l_all",        "model": "gcn", "edges": "all",          "features": "all", "layers": 1},
    {"name": "gcn_3l_all",        "model": "gcn", "edges": "all",          "features": "all", "layers": 3},
]

# For each config: train → evaluate on test set → save to CSV
# CSV columns: name, model, edges, features, layers, macro_f1, f1_canonical, f1_translatable, f1_open, accuracy
```

---

## Phase 5 — Results & Visualization

### 5.1 Results Table

Generate a Markdown/LaTeX table from `results/ablation_results.csv`. Primary sort by Macro-F1 descending.

### 5.2 Required Figures

1. **Bar chart:** Macro-F1 for MLP vs GCN vs GAT (the central comparison)
2. **Confusion matrices:** One 3×3 heatmap each for MLP and best GCN
3. **Ablation bar chart:** GCN variants grouped by ablation factor (edge type, features, depth)
4. **Training curves:** Loss and Val-F1 over epochs for MLP, GCN, GAT on same axes

### 5.3 Expected Results Interpretation

A large Macro-F1 gap (GCN >> MLP) confirms the central hypothesis: modifiability is relational, not derivable from per-node features alone. A small or null gap is also informative — it would mean flat heuristics suffice and the graph structure adds nothing. Either outcome is a valid finding to report.

---

## Critical Path & Ordering

```
[Rico download] → [Preprocessing pipeline] → [Splits]
                                                  |
                              ┌───────────────────┼──────────────────┐
                              ↓                   ↓                  ↓
                         [MLP train]         [GCN train]        [GAT train]
                              |                   |                  |
                              └───────────────────┼──────────────────┘
                                                  ↓
                                          [Ablation runs]
                                                  ↓
                                     [Evaluation + Figures]
```

**Minimum viable run** (to get a number before the May 29 checkpoint):
1. Process 500 screens from a single app category
2. Run labeling heuristics, inspect label distribution
3. Train MLP for 20 epochs, report Macro-F1 → this is the first real number

---

## Key Design Decisions & Rationale

| Decision | Choice | Why |
|----------|--------|-----|
| Primary metric | Macro-F1 | Canonical is rare; accuracy would mask systematic failure on it |
| Split granularity | App-level | Screen-level splits leak — apps reuse layouts across screens |
| Text embedding | MiniLM-L6 | Fast, 384-dim, good semantic quality; BERT-large would be overkill |
| Loss function | Weighted cross-entropy | Compensates for class imbalance without oversampling |
| GCN aggregation | Mean | Simple, interpretable baseline; no normalization ambiguity |
| Self-loops | Added inside GCN layer | Keeps graph construction agnostic of model internals |
| Sibling edges | Separate from containment | Allows edge-type ablation without rebuilding graphs |

---

## Common Pitfalls to Avoid

1. **Memory:** Loading all 72K graphs at once will OOM. Use `Dataset.__getitem__` with lazy loading.
2. **Embedding cache:** Computing MiniLM embeddings per-node for 72K screens is ~6–8 hours without caching. Cache by text string.
3. **Label imbalance:** Expect ~70% "open", ~20% "translatable", ~10% "canonical". Without weighted loss, the model will predict "open" for everything and get high accuracy but F1 ≈ 0.33.
4. **App-level leakage:** If you split at screen level, val/test F1 will be inflated by ~5–15% due to repeated layout templates.
5. **Disconnected graphs:** Some Rico screens have very flat hierarchies (depth 1–2). A 3-layer GCN on a depth-2 tree aggregates the whole graph — this is fine but worth logging.
6. **Edge index dtype:** `edge_index` must be `torch.long` (int64). Float edges will cause silent errors.
