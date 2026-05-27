# Task Allocation — Gold Annotation
## CS570 Team 14 | 2 Annotators

---

## Summary

| | Annotator A (Nadia) | Annotator B (Teammate) |
|---|---|---|
| Initials for CSV | `na` | `tb` (update to your initials) |
| Overlap screens | 10 | 10 |
| Unique screens | 20 | 20 |
| **Total screens** | **30** | **30** |
| Est. nodes (overlap) | ~390 | ~390 |
| Est. nodes (unique) | ~600–700 | ~600–700 |
| **Est. total nodes** | **~1,000** | **~1,000** |

Total across both annotators: ~50 screens, ~1,500–2,000 node labels.

---

## Overlap Screens (10 screens — both annotators)

These 10 screens are labeled by **both** annotators independently. Results are compared to measure inter-annotator agreement (Cohen's kappa). Do not discuss your labels until both submissions are in.

| # | screen_id | Notes |
|---|-----------|-------|
| 1 | 20353 | Checkout flow — many Canonical nodes expected |
| 2 | 21089 | Bank dashboard — balances and account info |
| 3 | 23456 | Product listing with prices |
| 4 | 25871 | Booking confirmation screen |
| 5 | 30124 | Restaurant order summary |
| 6 | 33567 | Investment portfolio screen |
| 7 | 40892 | Cart and checkout summary |
| 8 | 45231 | Account settings with credentials |
| 9 | 51678 | Profile screen with personal info |
| 10 | 58903 | Payment transfer confirmation |

---

## Annotator A — Unique Screens (20 screens)

| # | screen_id | App domain | Est. nodes |
|---|-----------|-----------|-----------|
| 1 | 61234 | News | 31 |
| 2 | 62891 | Music player | 27 |
| 3 | 64523 | Fitness tracking | 44 |
| 4 | 67012 | Maps / navigation | 19 |
| 5 | 70345 | Productivity / tasks | 39 |
| 6 | 72689 | Weather | 24 |
| 7 | 74021 | Camera gallery | 18 |
| 8 | 76543 | Messaging / chat | 55 |
| 9 | 78901 | Notes editor | 32 |
| 10 | 80234 | Calendar | 29 |
| 11 | 82567 | Health / vitals | 43 |
| 12 | 84090 | Games / leaderboard | 21 |
| 13 | 85432 | Browser search | 37 |
| 14 | 86789 | Podcast list | 26 |
| 15 | 87654 | E-reader bookshelf | 40 |
| 16 | 88321 | Recipe detail | 33 |
| 17 | 89012 | Ride-hailing booking | 47 |
| 18 | 89543 | Hotel search | 51 |
| 19 | 90124 | Flight booking | 44 |
| 20 | 90678 | Crypto wallet | 38 |

**Annotator A total: 30 screens (~990 estimated nodes)**

---

## Annotator B — Unique Screens (20 screens)

| # | screen_id | App domain | Est. nodes |
|---|-----------|-----------|-----------|
| 1 | 11023 | Grocery cart | 36 |
| 2 | 12456 | Pharmacy / prescription | 29 |
| 3 | 13789 | Insurance policy | 42 |
| 4 | 14321 | Online course | 35 |
| 5 | 15678 | Video streaming home | 28 |
| 6 | 16543 | Auction listing | 47 |
| 7 | 17890 | Real estate listing | 53 |
| 8 | 18234 | Car rental checkout | 31 |
| 9 | 19012 | Dating app profile | 24 |
| 10 | 19567 | Event ticket purchase | 39 |
| 11 | 22345 | Sports scores | 33 |
| 12 | 24678 | Job application form | 45 |
| 13 | 26901 | Pet care products | 27 |
| 14 | 28234 | Children's app home | 22 |
| 15 | 29567 | Clothing product detail | 49 |
| 16 | 31890 | Home appliance listing | 41 |
| 17 | 34123 | Hardware store | 18 |
| 18 | 36456 | Cosmetics product page | 37 |
| 19 | 38789 | Sports equipment checkout | 30 |
| 20 | 41023 | Donation confirmation | 25 |

**Annotator B total: 30 screens (~1,001 estimated nodes)**

---

## Workflow

```
Week 1
├── Both annotators: read guidelines, complete 5 overlap screens
└── Check in: share any edge cases you encountered (not labels)

Week 2
├── Both annotators: complete remaining 5 overlap screens
├── Annotator A: complete all 20 unique screens
└── Annotator B: complete all 20 unique screens

After both submit
├── Team lead computes Cohen's kappa on overlap
├── Disagreement report generated automatically
└── Team resolves disagreements together → gold_test_labels.csv
```

---

## Submission checklist

- [ ] All overlap screens annotated
- [ ] All unique screens annotated
- [ ] CSV named `annotations_<initials>.csv`
- [ ] `annotator_id` column is filled consistently with your initials
- [ ] Notes added for any uncertain decisions
- [ ] File sent to Nadia (nadia.arvi@gmail.com)

---

## Notes on screen IDs

The screen IDs in this document are placeholder values representing the Rico dataset screen identifiers. Before annotating, confirm with the team lead that the correct `.pt` graph files for these IDs are available on the Colab/Drive at:

```
/content/drive/MyDrive/cs570-project/data/processed/train/contextual/<app_id>/<screen_id>.pt
```

and that the corresponding screenshots exist at:

```
/content/rico_raw/combined/<screen_id>.png
```
