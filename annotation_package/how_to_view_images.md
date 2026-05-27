# How to View the Screenshots You're Annotating

## The short answer

The screenshots are **not a folder you can just open on your laptop**. They come from the Rico dataset (a ~6 GB archive) that lives on our shared Google Drive. You access them through a Google Colab notebook that mounts Drive and extracts the images.

**You do not need to download anything to your laptop.** Everything runs in Colab (Google's free browser-based compute environment).

---

## Step-by-step setup

### 1. Open the annotation helper notebook

Open this link in your browser (ask Nadia for the shared Colab link, or open `annotation_helper.ipynb` from the shared Drive folder):

```
My Drive > cs570-project > annotation_helper.ipynb
```

Or upload `annotation_helper.ipynb` (included in this package) to your Google Drive and open it with Google Colab.

### 2. Run the setup cells (top to bottom)

The notebook has three setup cells. Run them once at the start of every Colab session:

| Cell | What it does | Time |
|------|-------------|------|
| Mount Drive | Connects Colab to your Google Drive | ~10 seconds |
| Extract Rico | Unzips screenshots from Drive to local Colab storage | ~5–10 minutes (first time each session) |
| Verify | Confirms images are accessible | ~5 seconds |

After the extract step, every screen's screenshot is available at:
```
/content/rico_raw/combined/<screen_id>.png
```

### 3. Enter your screen list and annotate

The notebook has an annotation loop. You paste in your screen IDs (from `task_allocation.md`) and it shows you:
- The screenshot
- The list of nodes to label (node_id, element type, text content, depth)

You fill in your CSV as you go.

---

## What the images look like

Each screen is a full Android phone screenshot (~1440×2560 px). The notebook displays them at 40% size so they fit on screen. You can zoom in your browser if you need to see small elements clearly.

The screenshots are real screenshots from real apps — product listings, banking screens, checkout flows, etc. They are from the Rico dataset collected by Carnegie Mellon researchers (publicly available for research).

---

## Frequently asked questions

**Do I need a GPU?**
No. The annotation helper notebook does not train any model. The free Colab CPU tier is fine.

**What if the Rico archive isn't on Drive yet?**
The main project setup downloads it automatically. If it's missing, ask Nadia — she can share the Drive folder directly.

**What if a screenshot looks blank or corrupted?**
Skip it and add a note in your CSV (`notes: screenshot blank`). Let Nadia know the screen ID.

**Can I annotate offline?**
No — the images are on Google Drive and accessed through Colab. You need an internet connection.

**What if a node ID in my template doesn't match what the notebook shows?**
The `.pt` graph files (preprocessed by our pipeline) define which node IDs to annotate. If the preprocessing hasn't run yet for a screen, the notebook falls back to reading the raw JSON hierarchy. The node IDs might differ slightly. Flag any mismatch in your notes and message Nadia.

**How long does one screen take?**
A screen with ~30 nodes takes about 10–20 minutes once you're comfortable with the labels. Budget ~3–5 hours total for your 30 screens.

---

## If something doesn't work

Message Nadia at nadia.arvi@gmail.com with:
1. The screen ID you were trying to view
2. A screenshot of the error message
