# Quick Reference Card — UI Modifiability Labels

## The Three Labels

| Label | Code | Meaning | Key test |
|-------|------|---------|----------|
| Canonical | 0 | Must not change | Would altering this deceive or financially harm the user? |
| Translatable | 1 | Meaning must be preserved | Does this communicate something the user needs to act on? |
| Open | 2 | Freely replaceable | Decorative, promotional, or low-stakes? |

---

## Decision Flowchart

```
1. Price / balance / account number / payment info / credential?
        YES → Canonical (0)
        NO  → ↓

2. In payment/checkout/financial context AND has numeric or identity data?
        YES → Canonical (0)
        NO  → ↓

3. Meaningful text (navigation, instructions, field labels, status messages)?
        YES → Translatable (1)
        NO  → ↓

4. Image / banner / decoration / promotional content?
        YES → Open (2)

5. Still unsure?
        → Canonical beats Translatable
        → Translatable beats Open
        → Write a note in the notes column
```

---

## Common Examples

| What you see | Label |
|---|---|
| `$24.99` in checkout / cart total | Canonical |
| `user@gmail.com` displayed on screen | Canonical |
| `Order #A938271` | Canonical |
| `Balance: $1,204.50` | Canonical |
| `Pay Now` / `Confirm Transfer` button | Canonical |
| CAPTCHA element | Canonical |
| `Settings` / `Home` / `Profile` nav tab | Translatable |
| `Add to Cart` / `Sign In` button | Translatable |
| `My Orders` page title | Translatable |
| `First Name` field label | Translatable |
| `Your order has been placed` status | Translatable |
| `4.5 stars`, `1,203 reviews` | Translatable |
| Product photo | Open |
| `Summer Sale — 40% off!` banner | Open |
| App tagline / marketing copy | Open |
| Background image / illustration | Open |
| Social share buttons | Open |

---

## Tricky Cases

**Same price, different context**
- `$24.99` in checkout → **Canonical**
- `"Items from $24.99"` in a promo banner → **Open**

**Buttons depend on what they do**
- `Pay Now` → **Canonical**
- `Add to Cart` → **Translatable**
- `Share on Facebook` → **Open**

**Input field parts**
- The label above a field (`Email address`) → **Translatable**
- The hint inside the field (`Enter your email`) → **Translatable**
- A real email address displayed on screen → **Canonical**

**Images: almost always Open** unless they contain embedded critical text (e.g. a receipt image).

---

## CSV Format

```
screen_id,node_id,label,annotator_id,notes
20353,0,open,na,root container
20353,2,canonical,na,price shown at checkout
20353,5,translatable,na,section header
```

- `label`: use `canonical`, `translatable`, `open` (or `0`, `1`, `2`)
- `annotator_id`: your initials, same every time
- `notes`: optional but encouraged for uncertain decisions

## File naming
Save as: `annotations_<your_initials>.csv`  
Send to: **nadia.arvi@gmail.com**
