# Gold Label Annotation Guidelines
## CS570 Team 14 — UI Modifiability Project

---

## What You're Helping With

We are building a machine learning model that looks at Android app screens and predicts how "modifiable" each element on the screen is. For example, a price tag like `$24.99` in a checkout screen is critical and should never be changed arbitrarily — but a banner headline like `"Summer Sale!"` is freely editable.

Your job is to look at Android app screenshots and their underlying UI element trees, and assign one of three labels to each element. Your annotations will be the **gold standard test set** — the final authority on whether our model is correct.

We are annotating approximately **50 app screens**, each containing roughly **20–60 labeled elements**. Total task: approximately **1,500–2,000 individual element labels** across the team.

---

## What You're Looking At

Each Android app screen is represented two ways:

1. **A screenshot (PNG)** — what the user sees on their phone
2. **A UI hierarchy (JSON)** — a tree of every element on the screen, with metadata like class type, text content, position (bounds), and a unique ID

You will annotate at the **element level**, not the screen level. Each element in the tree gets one label.

Not every element needs a label — we skip things like invisible containers with no text. You will be given a list of element IDs that need labels for each screen.

---

## The Three Labels

Every element gets exactly one of these labels:

---

### 🔴 Label 0 — Canonical

> **Definition:** This element contains critical, identity-specific, or financially sensitive information that must NOT be changed. Altering it could deceive the user, cause financial harm, or break trust.

Think of it as: **"This must stay exactly as it is."**

**Label as Canonical if the element contains:**

- Prices, totals, amounts, fees, balances
  - ✅ `$24.99`, `Total: $103.50`, `Cashback: $2.00`
- Account credentials or identifiers
  - ✅ Email address fields, username fields, password fields
  - ✅ Displayed email like `user@gmail.com`
  - ✅ Account number, phone number, member ID
- Payment or financial information
  - ✅ Credit card number fields, CVV, billing address
  - ✅ Bank routing number, IBAN, transaction ID
- Order or transaction records
  - ✅ Order number `#A12345678`, transaction reference
- Legal, compliance, or verification elements
  - ✅ CAPTCHA elements
  - ✅ Terms & conditions checkboxes with legal text
  - ✅ "I agree to the [Terms of Service]" text
- Security-critical actions in a financial/account context
  - ✅ "Confirm Payment" button inside a payment flow
  - ✅ "Send $50" button in a money transfer screen

**Examples in context:**

| Screen | Element | Label | Reason |
|--------|---------|-------|--------|
| Checkout screen | `$24.99` (item price) | **Canonical** | Price is financially critical |
| Login screen | Email input field | **Canonical** | Account credential |
| Bank app | `Balance: $1,204.50` | **Canonical** | Financial data |
| Order history | `Order #A938271` | **Canonical** | Transaction identifier |
| Payment screen | "Pay Now" button | **Canonical** | Critical financial action |

---

### 🟡 Label 1 — Translatable

> **Definition:** This element contains meaningful text that communicates something to the user, but it is not financially or identity-critical. The text could be reworded, localized, or adapted — as long as the meaning is preserved.

Think of it as: **"The words can change, but the meaning must stay the same."**

**Label as Translatable if the element contains:**

- Navigation labels and menu items
  - ✅ "Home", "Settings", "Profile", "Search", "Back"
- Page titles and section headers
  - ✅ "My Orders", "Account Settings", "Notifications"
- Button labels for non-critical actions
  - ✅ "Add to Cart", "View Details", "Sign In", "Cancel"
- Descriptive labels and field hints
  - ✅ "Enter your email", "First Name", "Shipping Address"
  - ✅ Placeholder text inside input fields
- Informational text at moderate detail level
  - ✅ "Your order has been placed", "No results found"
  - ✅ App feature descriptions, onboarding text
- Rating or review labels
  - ✅ "4.5 stars", "1,203 reviews"
- Category labels
  - ✅ "Electronics", "Women's Clothing", "Fast Food"

**Examples in context:**

| Screen | Element | Label | Reason |
|--------|---------|-------|--------|
| Any screen | "Settings" (nav tab) | **Translatable** | Navigation label, meaning must be preserved |
| Product screen | "Add to Cart" button | **Translatable** | Action label, non-financial |
| Sign-up screen | "First Name" (field label) | **Translatable** | Descriptive label |
| Order screen | "Your order is confirmed" | **Translatable** | Status message, meaning matters |
| Search screen | "Search results for..." | **Translatable** | Informational text |

---

### 🟢 Label 2 — Open

> **Definition:** This element is decorative, promotional, or low-stakes content that can be freely changed or replaced without deceiving the user or breaking app functionality.

Think of it as: **"This can be freely changed — swapping it out wouldn't hurt anyone."**

**Label as Open if the element is:**

- Non-critical images
  - ✅ Product photos, app illustrations, background images
  - ✅ Profile picture placeholders
  - ✅ App logo / icon images (not text)
- Promotional or marketing content
  - ✅ Banner ads: "50% OFF this weekend!"
  - ✅ Hero images or carousels on a homepage
  - ✅ "Featured for you", "Recommended", "Trending"
- Decorative or filler content
  - ✅ Star rating graphic (the visual stars, not the number text)
  - ✅ Divider lines, spacing elements
  - ✅ Social sharing buttons (Facebook, Twitter icons)
- Low-stakes or generic text that doesn't direct the user to do anything important
  - ✅ App taglines: "Shop smarter, live better"
  - ✅ Generic loading text: "Hang tight..."
  - ✅ Testimonial quotes from unknown users

**Examples in context:**

| Screen | Element | Label | Reason |
|--------|---------|-------|--------|
| Home screen | Hero banner image | **Open** | Purely decorative/promotional |
| Product screen | Product photo | **Open** | Visual, freely replaceable |
| Home screen | "Summer Sale — 40% off!" | **Open** | Promotional, not a critical instruction |
| Any screen | App illustration/icon | **Open** | Decorative |
| Profile screen | Background cover photo | **Open** | Decorative |

---

## The Decision Flowchart

When you're unsure, work through these questions in order:

```
1. Does this element contain a price, balance, account number,
   payment info, or personal credential?
        YES → Label 0 (Canonical)
        NO  → continue

2. Is this element in a payment, checkout, or financial transaction
   context AND does it contain any numeric or identity data?
        YES → Label 0 (Canonical)
        NO  → continue

3. Does this element contain meaningful text that tells the user
   something important (navigation, instructions, labels, status)?
        YES → Label 1 (Translatable)
        NO  → continue

4. Is this element an image, banner, decoration, or promotional content?
        YES → Label 2 (Open)
        NO  → continue

5. If none of the above fit clearly, use your best judgment.
   When in doubt between Canonical and Translatable, pick Canonical.
   When in doubt between Translatable and Open, pick Translatable.
   Add a note in the `notes` column explaining your uncertainty.
```

---

## Tricky Cases and Edge Cases

### Same text, different context → different label

The element `$24.99` can have different labels depending on where it appears:

| Location | Label | Reason |
|----------|-------|--------|
| Checkout total | **Canonical** | User is about to pay this amount |
| Promotional banner "Items from $24.99" | **Open** | Marketing copy, not a binding transaction |
| Product listing price | **Canonical** | Financial information the user relies on |

**Rule:** When price text appears in a payment/checkout/cart flow, it is always Canonical. When it appears in a promotional or discovery context (homepage, ads), it may be Open.

---

### Buttons: it depends on what they do

| Button text | Context | Label |
|-------------|---------|-------|
| "Pay Now" | Checkout screen | **Canonical** |
| "Confirm Transfer" | Bank transfer | **Canonical** |
| "Add to Cart" | Product page | **Translatable** |
| "Continue Shopping" | Cart page | **Translatable** |
| "Share on Facebook" | Any screen | **Open** |

---

### Images

Almost all images are **Open** unless they contain embedded critical text (e.g., a screenshot of a receipt shown as an image). If an `ImageView` has no text and is not a CAPTCHA, label it Open.

---

### Input field labels vs. input field content

- The label above a field ("Email address") → **Translatable**
- The hint text inside the field ("Enter your email") → **Translatable**
- The actual entered value (a real email address displayed) → **Canonical**

---

### "Sign In" / "Log In" buttons

These are **Translatable** — they are navigation/action labels, not financial. The security-critical part is the credential you enter, not the button label itself.

---

### Elements with no visible text or description

If an element has no text, no content description, and is just a layout container (like a `LinearLayout` with no labels), **skip it** — do not include it in the CSV. We only label elements that have something meaningful to assess.

---

## How to Annotate: Step by Step

### Step 1 — Get the list of screens you are assigned

Your team lead will give you a list of screen IDs (e.g., `screen_id: 20353`). Each screen ID maps to:
- A PNG screenshot in the Rico dataset
- A `.pt` graph file (from preprocessing) that contains the list of `node_id`s to annotate

### Step 2 — View the screenshot

The PNG screenshots are in the extracted Rico folder under:
```
/content/rico_raw/combined/<screen_id>.png
```

Display it in the notebook with:
```python
from IPython.display import Image
Image("/content/rico_raw/combined/20353.png", width=400)
```

### Step 3 — View the node list for that screen

Run this in the Colab notebook inspect cell (Section 9):
```python
SCREEN_PATH = "/content/drive/MyDrive/cs570-project/data/processed/train/contextual/<app_id>/<screen_id>.pt"
```

This will print a table of `node_id`, heuristic label, and depth for every node on that screen.

### Step 4 — Assign your labels

For each node in the list:
1. Find the corresponding element in the screenshot by its position/bounds
2. Assign label 0, 1, or 2
3. Record it in the CSV

### Step 5 — Fill in the CSV

Create or update your annotation CSV with one row per annotated node:

```csv
screen_id,node_id,label,annotator_id,notes
20353,0,2,alice,root container - open
20353,1,1,alice,section header
20353,2,0,alice,price in checkout context
20353,3,1,alice,
```

Field descriptions:

| Field | Required | What to put |
|-------|----------|-------------|
| `screen_id` | ✅ | The screen ID (e.g., `20353`) |
| `node_id` | ✅ | The node ID from the inspect table (e.g., `0`, `1`, `2`) |
| `label` | ✅ | One of: `canonical`, `translatable`, `open` (or `0`, `1`, `2`) |
| `annotator_id` | ✅ | Your name or initials — use the same string every time |
| `notes` | optional | Any uncertainty, ambiguity, or rationale |

**Important:** Every `(screen_id, node_id)` pair must be unique. Do not add the same node twice.

---

## Overlap Subset — For Agreement Measurement

**10 screens** will be annotated by at least two people. This is intentional — we use it to measure how consistent the annotations are (inter-annotator agreement).

If you are assigned overlap screens:
- Annotate independently — do not discuss your labels with the other annotator until both of you have finished
- Do annotate every node on that screen, not just the ones you feel confident about
- Use the `notes` field to flag anything you found ambiguous

After both annotators submit, we will compare and resolve disagreements together.

---

## Submission

Save your annotations as a CSV file named:
```
annotations_<your_initials>.csv
```
For example: `annotations_na.csv`, `annotations_jk.csv`

Send the file to the team lead, who will merge all annotations into `gold_test_labels.csv`.

---

## Quality Reminders

- **Be consistent.** If you label `$9.99` as Canonical in screen A, do the same in screen B.
- **Use notes.** If you're genuinely unsure, write a short note — it helps during disagreement resolution.
- **Don't skip silently.** If you think a node should not be labeled at all (e.g., it's clearly a layout container), write `skip` in the notes column and omit it from the CSV, or ask the team lead.
- **Context matters more than content.** The same word can have different labels in different screens. Always consider what the screen is for.
- **When in doubt, Canonical wins.** We would rather over-protect a data element than under-protect it.
- **Do not annotate from memory.** Always look at the actual screenshot and node info for each label decision.

---

## Quick Reference Card

| You see... | Label |
|------------|-------|
| Price / total / balance / fee | 🔴 Canonical |
| Account number / email / phone (displayed value) | 🔴 Canonical |
| Payment confirmation button | 🔴 Canonical |
| CAPTCHA | 🔴 Canonical |
| Page title / section header | 🟡 Translatable |
| Navigation tab / menu item | 🟡 Translatable |
| "Add to Cart" / "Sign In" button | 🟡 Translatable |
| Field label ("Email", "Name") | 🟡 Translatable |
| Status message ("Order confirmed") | 🟡 Translatable |
| Product photo / app illustration | 🟢 Open |
| Promotional banner / sale text | 🟢 Open |
| App tagline / marketing copy | 🟢 Open |
| Social share buttons | 🟢 Open |
| Background / decorative image | 🟢 Open |

---

## Questions?

If you encounter a case not covered in this guideline, flag it with a note in the CSV and bring it up with the team lead. Do not guess silently — ambiguous cases should be resolved as a team so the labels are consistent.
