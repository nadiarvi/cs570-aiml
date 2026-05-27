# CS570 Team 14 — Gold Annotation Package

This folder contains everything you need to annotate the gold test set for our UI Modifiability project.

## What's in this package

| File / Folder | Purpose |
|---|---|
| `project_background.md` | **Start here if you're new** — what the project is, why we need labels, what you're doing |
| `annotation_guidelines.md` | Full labeling rules, examples, edge cases, decision flowchart |
| `quick_reference.md` | One-page cheat sheet — keep this open while annotating |
| `annotation_helper.ipynb` | **Colab notebook to view screenshots and node lists** — your main annotation tool |
| `how_to_view_images.md` | Explains where the screenshots live and how to access them |
| `screen_list.csv` | All 50 screens, who annotates which, overlap flag |
| `task_allocation.md` | Per-person breakdown with counts and deadlines |
| `templates/annotations_A.csv` | Pre-seeded template for Annotator A |
| `templates/annotations_B.csv` | Pre-seeded template for Annotator B |

---

## Quick-start (5 minutes)

**Step 1 — Read `project_background.md`**
If you have no context on the project, read this first. It explains what you're labeling and why in plain language. Takes 5 minutes.

**Step 2 — Read `annotation_guidelines.md`**
The full labeling rules. Pay special attention to the Decision Flowchart and the Tricky Cases section. Keep `quick_reference.md` open as a cheat sheet while you work.

**Step 3 — Open `annotation_helper.ipynb` in Google Colab**
Upload it to your Google Drive and open with Colab. This notebook handles everything: it loads the screenshots, shows you node lists, and guides you screen by screen. See `how_to_view_images.md` if you get stuck.

> The screenshots are **not local files** — they come from the Rico dataset (~6 GB) on our shared Google Drive. The notebook extracts them automatically. You do not need to download anything to your laptop.

**Step 4 — Fill in your CSV template**
Open `templates/annotations_A.csv` or `templates/annotations_B.csv` (ask Nadia which one is yours). Add one row per node as you go:
```
screen_id,node_id,label,annotator_id,notes
20353,0,open,na,root container
20353,2,canonical,na,price in checkout
```

**Step 5 — Submit**
Name your file `annotations_<your_initials>.csv` and send it to the team lead (Nadia).

---

## Rules to remember

- **Overlap screens** (flagged in `screen_list.csv`) must be annotated **independently** — do not discuss labels until both of you submit.
- Label each node as exactly one of: `canonical` (0), `translatable` (1), or `open` (2).
- Use the `notes` column whenever you are uncertain — it helps during disagreement resolution.
- When in doubt between Canonical and Translatable, **pick Canonical**.
- Skip layout containers with no text — do not add a row for them.

---

## Contact

Team lead: Nadia — nadia.arvi@gmail.com
