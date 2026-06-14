"""
Dakshin Delights — Fast restaurant assistant.
"""

from __future__ import annotations

import os
import re
import sys
import time
import json
import math
import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from rapidfuzz import process

    def fuzzy_best(query, choices, cutoff=72):
        m = process.extractOne(query, choices, score_cutoff=cutoff)
        return m[0] if m else None
except ImportError:
    from difflib import get_close_matches

    def fuzzy_best(query, choices, cutoff=72):
        m = get_close_matches(query, choices, n=1, cutoff=cutoff / 100)
        return m[0] if m else None

load_dotenv()

# ==========================================================================
# 1. Menu data + categories
# ==========================================================================
MENU_DETAILS = {
    "idli (2 pcs)": {"price": 45, "tags": ["vegan", "vegetarian", "jain", "gluten-free"]},
    "idli (4 pcs)": {"price": 82, "tags": ["vegan", "vegetarian", "jain", "gluten-free"]},
    "masala idli (2 pcs)": {"price": 81, "tags": ["vegetarian"]},
    "idli with nene chutney (2 pcs)": {"price": 71, "tags": ["vegan", "vegetarian", "jain", "gluten-free"]},
    "nuchina unde": {"price": 38, "tags": ["vegan", "vegetarian", "jain", "spicy"]},
    "button idli": {"price": 52, "tags": ["vegetarian", "jain"]},
    "butter idli": {"price": 71, "tags": ["vegetarian"]},
    "ghee idli": {"price": 71, "tags": ["vegetarian", "jain"]},
    "plain idli": {"price": 45, "tags": ["vegan", "vegetarian", "jain", "gluten-free"]},
    "ghee podi idli": {"price": 90, "tags": ["vegetarian", "jain", "spicy"]},
    "ghee sambar button idli": {"price": 86, "tags": ["vegetarian"]},
    "rava idli (2 pcs)": {"price": 90, "tags": ["vegetarian"]},
    "idiyappam": {"price": 71, "tags": ["vegan", "vegetarian", "jain", "gluten-free"]},
    "ghee plain dosa": {"price": 105, "tags": ["vegetarian"]},
    "ghee ragi dosa (2 pcs)": {"price": 105, "tags": ["vegetarian", "gluten-free"]},
    "ghee pudina dosa": {"price": 124, "tags": ["vegetarian", "spicy"]},
    "ghee pudi masala dosa": {"price": 138, "tags": ["vegetarian", "spicy"]},
    "garlic roast dosa": {"price": 138, "tags": ["vegetarian", "spicy"]},
    "open butter masala dosa": {"price": 143, "tags": ["vegetarian"]},
    "ghee onion dosa (2 pcs)": {"price": 129, "tags": ["vegetarian"]},
    "ghee khali dosa (2 pcs)": {"price": 129, "tags": ["vegetarian"]},
    "multi grain dosa (2 pcs)": {"price": 114, "tags": ["vegetarian"]},
    "neer dosa (3 pcs)": {"price": 95, "tags": ["vegan", "vegetarian", "gluten-free"]},
    "pudina roast dosa": {"price": 138, "tags": ["vegetarian", "spicy"]},
    "butter masala dosa": {"price": 143, "tags": ["vegetarian"]},
    "butter podi dosa": {"price": 129, "tags": ["vegetarian", "spicy"]},
    "rava masala dosa": {"price": 162, "tags": ["vegetarian"]},
    "jain masala dosa (with jain sambar)": {"price": 143, "tags": ["vegetarian", "jain"]},
    "podi masala dosa": {"price": 171, "tags": ["vegetarian", "spicy"]},
    "mushroom biryani": {"price": 138, "tags": ["vegetarian", "spicy", "gluten-free"]},
    "ven pongal": {"price": 90, "tags": ["vegetarian", "jain", "gluten-free"]},
    "vangi bath": {"price": 76, "tags": ["vegetarian", "spicy"]},
    "bisibelebath with raita": {"price": 90, "tags": ["vegetarian", "spicy", "gluten-free"]},
    "tomato avalakki bath": {"price": 114, "tags": ["vegetarian", "spicy"]},
    "lemon idli": {"price": 95, "tags": ["vegetarian", "jain"]},
    "ghee pudi rice": {"price": 133, "tags": ["vegetarian", "spicy"]},
    "lemon ottu shavige": {"price": 105, "tags": ["vegan", "vegetarian", "jain", "gluten-free"]},
    "mix veg avalakki": {"price": 114, "tags": ["vegetarian"]},
    "coconut ottu shavige": {"price": 86, "tags": ["vegan", "vegetarian", "jain", "gluten-free"]},
    "coconut rice": {"price": 76, "tags": ["vegan", "vegetarian", "jain"]},
    "tomato ottu shavige": {"price": 71, "tags": ["vegan", "vegetarian", "jain", "gluten-free"]},
    "tomato rice": {"price": 105, "tags": ["vegetarian", "spicy"]},
    "lemon rice": {"price": 71, "tags": ["vegan", "vegetarian", "jain", "gluten-free"]},
    "poori": {"price": 60, "tags": ["vegetarian"]},
    "custard apple with pulp": {"price": 180, "tags": ["vegetarian", "gluten-free"]},
    "special falooda": {"price": 155, "tags": ["vegetarian"]},
    "gud bud": {"price": 150, "tags": ["vegetarian"]},
    "badam milk": {"price": 50, "tags": ["vegetarian", "gluten-free"]},
    "gulab jamun": {"price": 40, "tags": ["vegetarian"]},
}

IDLI_ITEMS = [
    "idli (2 pcs)", "idli (4 pcs)", "masala idli (2 pcs)", "idli with nene chutney (2 pcs)",
    "nuchina unde", "button idli", "butter idli", "ghee idli", "plain idli", "ghee podi idli",
    "ghee sambar button idli", "rava idli (2 pcs)", "idiyappam",
]
DOSA_ITEMS = [
    "ghee plain dosa", "ghee ragi dosa (2 pcs)", "ghee pudina dosa", "ghee pudi masala dosa",
    "garlic roast dosa", "open butter masala dosa", "ghee onion dosa (2 pcs)",
    "ghee khali dosa (2 pcs)", "multi grain dosa (2 pcs)", "neer dosa (3 pcs)",
    "pudina roast dosa", "butter masala dosa", "butter podi dosa", "rava masala dosa",
    "jain masala dosa (with jain sambar)", "podi masala dosa",
]
RICE_ITEMS = ["mushroom biryani", "ven pongal", "vangi bath", "bisibelebath with raita"]
LIGHT_ITEMS = [
    "tomato avalakki bath", "lemon idli", "ghee pudi rice", "lemon ottu shavige",
    "mix veg avalakki", "coconut ottu shavige", "coconut rice", "tomato ottu shavige",
    "tomato rice", "lemon rice",
]
POORI_ITEMS = ["poori"]
DESSERT_ITEMS = ["custard apple with pulp", "special falooda", "gud bud", "gulab jamun"]
DRINK_ITEMS = ["badam milk", "special falooda"]

# Ordered groups for the full-menu printout (deduped at render time)
MENU_GROUPS = [
    ("Idli & Steamed", IDLI_ITEMS),
    ("Dosa", DOSA_ITEMS),
    ("Rice", RICE_ITEMS),
    ("Light Meals & Tiffin", LIGHT_ITEMS),
    ("Poori", POORI_ITEMS),
    ("Desserts", DESSERT_ITEMS),
    ("Drinks", DRINK_ITEMS),
]

CATEGORY_ITEMS = {
    "idli": IDLI_ITEMS, "dosa": DOSA_ITEMS, "rice": RICE_ITEMS,
    "light": LIGHT_ITEMS, "poori": POORI_ITEMS,
    "desserts": DESSERT_ITEMS, "drinks": DRINK_ITEMS,
}

# word -> canonical category
CATEGORY_SYNONYMS = {
    "idli": "idli", "idly": "idli", "idlis": "idli", "steamed": "idli",
    "dosa": "dosa", "dosas": "dosa", "dose": "dosa", "crepe": "dosa",
    "rice": "rice", "biryani": "rice", "biriyani": "rice", "pongal": "rice", "pulao": "rice",
    "light": "light", "tiffin": "light", "snack": "light", "snacks": "light",
    "avalakki": "light", "shavige": "light", "poha": "light",
    "poori": "poori", "puri": "poori",
    "dessert": "desserts", "desserts": "desserts", "sweet": "desserts",
    "sweets": "desserts", "icecream": "desserts", "falooda": "desserts",
    "drink": "drinks", "drinks": "drinks", "beverage": "drinks",
    "beverages": "drinks", "juice": "drinks", "shake": "drinks", "milk": "drinks",
}
FULL_MENU_WORDS = ("full", "all", "everything", "entire", "complete", "whole")
POPULAR_ITEMS = ["mushroom biryani", "podi masala dosa", "ghee podi idli",
                 "special falooda", "gulab jamun", "badam milk"]
CATEGORY_TITLES = {
    "idli": "Idli & Steamed", "dosa": "Dosa", "rice": "Rice",
    "light": "Light Meals & Tiffin", "poori": "Poori",
    "desserts": "Desserts", "drinks": "Drinks",
}

AVAILABILITY_FILE = "item_availability.json"
ANALYTICS_FILE = "analytics_dashboard.json"
_AVAIL_WARNED = False          # so a bad file warns once, not every call


# ==========================================================================
# 2. Availability / specials
# ==========================================================================
def load_availability_and_specials() -> dict:
    global _AVAIL_WARNED
    default = {
        "todays_specials": [
            "Ghee Onion Dosa (2 pcs)", "Ghee Khali Dosa (2 pcs)",
            "Multi Grain Dosa (2 pcs)", "Neer Dosa (3 pcs)",
        ],
        "availability": {item: True for item in MENU_DETAILS},
    }
    if not os.path.exists(AVAILABILITY_FILE):
        try:
            with open(AVAILABILITY_FILE, "w") as f:
                json.dump(default, f, indent=4)
        except Exception:
            pass
        return default
    try:
        with open(AVAILABILITY_FILE, "r") as f:
            data = json.load(f)
        data.setdefault("availability", {})
        data.setdefault("todays_specials", [])
        for item in MENU_DETAILS:
            data["availability"].setdefault(item, True)
        _AVAIL_WARNED = False          # file is healthy again
        return data
    except Exception as e:
        if not _AVAIL_WARNED:           # warn ONCE, then stay quiet
            print(f"⚠️  '{AVAILABILITY_FILE}' has a formatting error ({e}). "
                  "Using all-items-available until the JSON is fixed.")
            _AVAIL_WARNED = True
        return default


# ==========================================================================
# 3. Lookup / category helpers
# ==========================================================================
def _norm(s: str) -> str:
    """Normalise for matching. Keeps piece counts ('idli (4 pcs)' -> 'idli 4 pcs')
    so variants stay distinct, and unifies common spellings."""
    s = (s or "").lower()
    s = s.replace("idly", "idli").replace("dosai", "dosa").replace("vadai", "vada")
    s = s.replace("pudi", "podi")                       # podi == pudi (spice powder)
    s = s.replace("pieces", "pcs").replace("piece", "pcs")
    s = re.sub(r"(\d+)\s*pc\b", r"\1 pcs", s)           # '4pc' / '4 pc' -> '4 pcs'
    s = re.sub(r"[()]", " ", s)                          # keep contents, drop the brackets
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _base(key: str) -> str:
    """Item name without piece-count tokens, e.g. 'idli 4 pcs' -> 'idli'."""
    return " ".join(tok for tok in _norm(key).split()
                    if tok != "pcs" and not tok.isdigit())


# normalized form -> canonical menu key (keep FIRST so 'idli' -> 'idli (2 pcs)')
_NORM_MENU: Dict[str, str] = {}
for _k in MENU_DETAILS:
    _NORM_MENU.setdefault(_norm(_k), _k)


def match_candidates(query: str) -> List[str]:
    """Return ALL menu keys a query could mean (for disambiguation).
    [] = nothing, [x] = unambiguous, [x, y, ...] = ambiguous."""
    if not query:
        return []
    raw = query.lower().strip()
    if raw in MENU_DETAILS:
        return [raw]
    s = _norm(query)
    if not s:
        return []
    if s in _NORM_MENU:
        return [_NORM_MENU[s]]
    qt = set(s.split())
    cands = [k for nk, k in _NORM_MENU.items() if qt and qt <= set(nk.split())]
    if not cands:
        cands = [k for nk, k in _NORM_MENU.items() if s in nk]
    if not cands:
        best = fuzzy_best(s, list(_NORM_MENU.keys()), 72)
        cands = [_NORM_MENU[best]] if best else []
    if len(cands) > 1:                       # if the query IS a base, keep only its variants
        base_eq = [c for c in cands if _base(c) == s]
        if base_eq:
            cands = base_eq
    return sorted(set(cands), key=lambda c: (len(c), c))


def find_item(user_input: str) -> Optional[str]:
    """Single best menu key (or None). For ambiguity-aware flows use match_candidates."""
    cands = match_candidates(user_input)
    return cands[0] if cands else None


def suggest_items(user_input: str, n: int = 3) -> List[str]:
    s = (user_input or "").lower().replace("(", " ").replace(")", " ")
    toks = [t for t in s.split() if len(t) > 2]
    scored = []
    for k in MENU_DETAILS:
        overlap = sum(1 for t in toks if t in k)
        if overlap:
            scored.append((overlap, k))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [k for _, k in scored[:n]]


# --- preference understanding ------------------------------------------------
# We have no per-item ingredient list, so onion/garlic avoidance is mapped to
# the 'jain' tag (Jain food is prepared with NO onion & NO garlic) — the only
# guaranteed-safe subset — plus a name check so e.g. 'Ghee Onion Dosa' is
# always excluded.
_NO_ONION = ["no onion", "without onion", "onion free", "onion-free", "avoid onion",
             "skip onion", "hold the onion", "not eat onion", "dont eat onion",
             "don't eat onion", "donot eat onion", "do not eat onion",
             "can't eat onion", "cant eat onion", "allergic to onion", "no onions"]
_NO_GARLIC = ["no garlic", "without garlic", "garlic free", "garlic-free", "avoid garlic",
              "skip garlic", "not eat garlic", "dont eat garlic", "don't eat garlic",
              "do not eat garlic", "allergic to garlic"]
_NO_SPICE = ["not spicy", "no spice", "no spicy", "less spicy", "mild", "non spicy"]
# negation/dislike cue appearing somewhere before the ingredient
_DISLIKE = r"(?:no|not|n't|dont|don't|donot|do not|without|avoid|hate|dislike|allergic|" \
           r"skip|hold|less|free of|minus|never|don't like|dont like)\b[^.?!]*\b"


def parse_preference(text: str) -> dict:
    """Turn free text ('no onion', "i don't like garlic", 'vegan', 'mild') into rules."""
    t = f" {(text or '').lower()} "
    require, exclude_tags, exclude_words, notes = set(), set(), set(), []

    onion = any(p in t for p in _NO_ONION) or bool(re.search(_DISLIKE + "onion", t))
    garlic = any(p in t for p in _NO_GARLIC) or bool(re.search(_DISLIKE + "garlic", t))
    if "jain" in t or onion or garlic:
        require.add("jain")                 # jain => no onion & no garlic
        if onion:
            exclude_words.add("onion")
        if garlic:
            exclude_words.add("garlic")
        notes.append("Jain dishes — no onion or garlic")
    if "vegan" in t:
        require.add("vegan")
    if "gluten" in t:
        require.add("gluten-free")
    if ("vegetarian" in t or " veg " in t) and "vegan" not in t \
            and "non veg" not in t and "non-veg" not in t and "nonveg" not in t:
        require.add("vegetarian")
    if any(p in t for p in _NO_SPICE):
        exclude_tags.add("spicy")
    elif "spicy" in t or "spice" in t:
        require.add("spicy")

    return {"require": require, "exclude_tags": exclude_tags,
            "exclude_words": exclude_words, "notes": notes,
            "active": bool(require or exclude_tags or exclude_words)}


def canonical_pref(parsed: dict) -> str:
    """Short label to store/display, e.g. 'no garlic, no onion' or 'vegan'."""
    parts = [f"no {w}" for w in sorted(parsed["exclude_words"])]
    for r in sorted(parsed["require"]):
        if r == "jain" and parsed["exclude_words"]:
            continue                        # implied by 'no onion/garlic'
        parts.append(r)
    if "spicy" in parsed["exclude_tags"]:
        parts.append("mild")
    return ", ".join(parts)


def item_matches(item: str, parsed: dict) -> bool:
    if item not in MENU_DETAILS:
        return False
    tags = set(MENU_DETAILS[item]["tags"])
    name = item.lower()
    if not parsed["require"] <= tags:
        return False
    if parsed["exclude_tags"] & tags:
        return False
    if any(w in name for w in parsed["exclude_words"]):
        return False
    return True


def resolve_category(text: str) -> Optional[str]:
    """Map free text like 'desserts menu' / 'show me dosas' to a category key."""
    t = (text or "").lower()
    # prefer longer synonym words first to avoid accidental short matches
    for word in sorted(CATEGORY_SYNONYMS, key=len, reverse=True):
        if word in t:
            return CATEGORY_SYNONYMS[word]
    return None


def is_full_menu_request(text: str) -> bool:
    t = (text or "").lower().strip()
    if any(w in t for w in FULL_MENU_WORDS):
        return True
    return t in {"menu", "the menu", "show menu", "show me the menu",
                 "menu please", "card", "your menu"}


def _line(item: str, avail: dict) -> str:
    status = "" if avail.get(item, True) else "   ❌ sold out"
    return f"  • {item.title()} — ₹{MENU_DETAILS[item]['price']}{status}"


def render_full_menu() -> str:
    avail = load_availability_and_specials()["availability"]
    out, seen = ["🍽️  Welcome to the Dakshin Delights menu!"], set()
    for header, items in MENU_GROUPS:
        rows = [_line(i, avail) for i in items if i not in seen and i in MENU_DETAILS]
        seen.update(items)
        if rows:
            out.append(f"\n── {header} ──")
            out.extend(rows)
    out.append("\nJust tell me what you'd like, or ask me for a recommendation. 😊")
    return "\n".join(out)


def render_category(cat: str) -> str:
    avail = load_availability_and_specials()["availability"]
    items = [i for i in CATEGORY_ITEMS.get(cat, []) if i in MENU_DETAILS]
    if not items:
        return ""
    header = CATEGORY_TITLES.get(cat, cat.title())
    rows = [_line(i, avail) for i in items]
    text = f"🍽️  Here's our {header} selection:\n\n" + "\n".join(rows)
    # If some are sold out, point to the in-stock ones so the guest isn't stuck.
    in_stock = [i for i in items if avail.get(i, True)]
    sold_out = [i for i in items if not avail.get(i, True)]
    if sold_out and in_stock:
        text += ("\n\n👉 In stock right now: "
                 + ", ".join(x.title() for x in in_stock[:4]))
    text += "\n\nWant me to add any of these to your order? 😊"
    return text


def show_menu_text(category_text: str = "") -> str:
    """Full menu, a specific category, or a free-text fallback search."""
    t = (category_text or "").strip()
    if not t or is_full_menu_request(t):
        return render_full_menu()
    cat = resolve_category(t)
    if cat:
        return render_category(cat)
    return "🔎  Here's what I found:\n" + menu_search(query=t)


def menu_search(query="", max_price=None, preference_tag=None) -> str:
    """Search by free text / category / price / dietary preference, with availability."""
    avail = load_availability_and_specials()["availability"]
    q = (query or "").lower().strip()
    parsed = parse_preference(preference_tag) if preference_tag else None

    # If the query is really a category or 'full menu', expand it.
    pool = list(MENU_DETAILS.keys())
    name_filter = q
    if q:
        if is_full_menu_request(q):
            name_filter = ""
        else:
            cat = resolve_category(q)
            if cat:
                pool = CATEGORY_ITEMS[cat]
                name_filter = ""

    rows = []
    for name in pool:
        d = MENU_DETAILS[name]
        if name_filter and name_filter not in name:
            continue
        if max_price is not None and d["price"] > max_price:
            continue
        if parsed and parsed["active"] and not item_matches(name, parsed):
            continue
        status = "Available" if avail.get(name, True) else "Not Available"
        rows.append(f"- {name.title()}: ₹{d['price']} [{status}] (Dietary: {', '.join(d['tags'])})")
    return "\n".join(rows) if rows else "No items match your criteria. Try adjusting your constraints."


def similar_available(item_name: str) -> List[str]:
    avail = load_availability_and_specials()["availability"]
    item_name = item_name.lower().strip()
    if item_name in DOSA_ITEMS:
        cat = DOSA_ITEMS
    elif item_name in IDLI_ITEMS:
        cat = IDLI_ITEMS
    elif item_name in DESSERT_ITEMS or item_name in DRINK_ITEMS:
        cat = DESSERT_ITEMS + DRINK_ITEMS
    elif item_name in RICE_ITEMS:
        cat = RICE_ITEMS
    else:
        cat = LIGHT_ITEMS
    return [x for x in cat if x != item_name and avail.get(x, True)][:3]


# ==========================================================================
# 4. Recommendations (deterministic, real, available)
# ==========================================================================
# Short, appetizing blurbs (keyed by item base name) used in recommendations.
DESCRIPTIONS = {
    "mushroom biryani": "fragrant & hearty", "podi masala dosa": "crispy with spicy podi",
    "ghee podi idli": "soft idlis in ghee & podi", "special falooda": "our chilled signature sweet",
    "gulab jamun": "warm & syrupy", "badam milk": "rich, chilled almond milk",
    "neer dosa": "soft, lacy coastal dosa", "ghee plain dosa": "golden & crisp",
    "rava masala dosa": "crunchy semolina dosa", "ven pongal": "comforting & buttery",
    "custard apple with pulp": "creamy seasonal treat", "gud bud": "loaded ice-cream sundae",
    "ghee onion dosa": "crisp dosa with caramelised onion", "ghee khali dosa": "plain ghee-roasted dosa",
    "multi grain dosa": "wholesome multi-grain dosa", "butter masala dosa": "buttery classic masala dosa",
    "open butter masala dosa": "open-style buttery masala dosa", "lemon rice": "tangy & zesty",
    "bisibelebath with raita": "spicy lentil-rice with raita", "garlic roast dosa": "crisp, garlicky roast",
}


def _named(k: str, desc: bool = True) -> str:
    d = DESCRIPTIONS.get(_base(k), "") if desc else ""
    return f"{k.title()} (₹{MENU_DETAILS[k]['price']})" + (f" — {d}" if d else "")


def todays_specials_text(s: Session) -> str:
    """Dedicated, friendly 'today's specials' answer (preference-aware)."""
    data = load_availability_and_specials()
    avail = data["availability"]
    parsed = parse_preference(s.preference)
    keys = [find_item(x) for x in data["todays_specials"]]
    ok = [k for k in keys if k and avail.get(k, True) and item_matches(k, parsed)]
    if not ok:
        return ("We don't have specials matching your preference today, "
                "but I'd love to recommend something — just ask! 😊")
    lines = [f"  • {_named(k)}" for k in ok]
    note = f"\n({'; '.join(parsed['notes'])}.)" if parsed["notes"] else ""
    return ("🌟  Today's Specials at Dakshin Delights:\n\n" + "\n".join(lines) + note
            + "\n\nShall I add any of these for you? 😊")


def dietary_listing(text: str, s: Session) -> Optional[str]:
    """List menu items matching a dietary filter ('show all jain dishes')."""
    parsed = parse_preference(text)
    if not parsed["active"]:
        return None
    avail = load_availability_and_specials()["availability"]
    blocks = []
    for header, items in MENU_GROUPS:
        rows = [f"  • {_named(k, desc=False)}" for k in items
                if k in MENU_DETAILS and avail.get(k, True) and item_matches(k, parsed)]
        if rows:
            blocks.append(f"── {header} ──\n" + "\n".join(rows))
    if "jain" in parsed["require"]:
        label = "Jain"
    elif parsed["require"]:
        label = ", ".join(t.replace("gluten-free", "Gluten-free").title()
                          if t != "gluten-free" else "Gluten-free"
                          for t in sorted(parsed["require"]))
    else:
        label = "matching"
    if not blocks:
        return f"🤔  We don't have any {label} dishes available right now."
    return (f"🥗  Our {label} dishes:\n\n" + "\n\n".join(blocks)
            + "\n\nShall I add any of these for you? 😊")


# --- FAQ / general restaurant questions --------------------------------------
FAQ = [
    (["open", "hours", "timing", "what time", "close", "closing"],
     "We're open every day from 7:00 AM to 10:30 PM. 🕖"),
    (["location", "address", "located", "where are you", "where is the", "how to reach"],
     "You'll find us at Dakshin Delights, 12 MG Road, Bengaluru. 📍"),
    (["deliver", "home delivery", "takeaway", "take away", "parcel"],
     "Yes! We do takeaway and deliver within 5 km via our app and partner platforms. 🛵"),
    (["payment", "pay by", "card", "upi", "cash", "gpay", "paytm"],
     "We accept cash, UPI, and all major credit/debit cards. 💳"),
    (["pure veg", "non veg", "non-veg", "vegetarian only", "do you serve meat", "eggless"],
     "We're a 100% pure-vegetarian South Indian kitchen. 🌱"),
    (["parking", "park my"],
     "Free parking for cars and two-wheelers is available right outside. 🚗"),
    (["wifi", "wi-fi", "internet"],
     "Complimentary Wi-Fi is available for all our guests. 📶"),
    (["reservation", "book a table", "table booking", "reserve a table"],
     "Happy to help — for table reservations, call us at +91-98000-00001. 📞"),
    (["contact", "phone number", "your number", "call you"],
     "You can reach us at +91-98000-00001. 📞"),
    (["gst", "tax", "service charge"],
     "Bills include 5% GST, shown clearly on your receipt — no hidden service charge. 🧾"),
    (["how spicy", "spice level", "make it spicy", "less spicy", "mild"],
     "Most dishes can be made mild or spicy — just tell me your preference! 🌶️"),
    (["who are you", "what are you", "your name"],
     "I'm Dakshin, your friendly assistant at Dakshin Delights. I can show the menu, "
     "recommend dishes, take your order, and answer questions! 😊"),
    (["thank", "thanks", "thank you"],
     "You're most welcome! 😊 Anything else I can get for you?"),
]


def faq_answer(t: str) -> Optional[str]:
    for keys, ans in FAQ:
        if any(k in t for k in keys):
            return ans
    return None


def build_recommendations(budget=None, group_size=None, preference="",
                          cart=None, saved_pref="") -> str:
    data = load_availability_and_specials()
    avail = data["availability"]
    specials = data["todays_specials"]
    parsed = parse_preference(preference or saved_pref)
    cart = cart or {}

    def usable(items):
        return [i for i in items if i in MENU_DETAILS and avail.get(i, True)
                and item_matches(i, parsed)]

    note = f"\n\n({'; '.join(parsed['notes'])}.)" if parsed["notes"] else ""

    # --- Budget combo: a main + an accompaniment (+ a sweet/drink) within budget ---
    if budget:
        budget = float(budget)
        pool = sorted(usable(list(MENU_DETAILS.keys())), key=lambda i: MENU_DETAILS[i]["price"])
        combo, total = [], 0.0
        mains = [i for i in pool if i in DOSA_ITEMS + RICE_ITEMS and MENU_DETAILS[i]["price"] <= budget]
        if mains:
            combo.append(mains[-1]); total += MENU_DETAILS[mains[-1]]["price"]
        for grp in (DRINK_ITEMS + DESSERT_ITEMS, LIGHT_ITEMS + IDLI_ITEMS):
            for e in pool:
                if e in grp and e not in combo and total + MENU_DETAILS[e]["price"] <= budget:
                    combo.append(e); total += MENU_DETAILS[e]["price"]; break
        for i in pool:
            if i not in combo and total + MENU_DETAILS[i]["price"] <= budget and len(combo) < 3:
                combo.append(i); total += MENU_DETAILS[i]["price"]
        if not combo:
            return f"🤔  Nothing fits under ₹{int(budget)} right now — our cheapest is Nuchina Unde (₹38)."
        body = " + ".join(_named(c, desc=False) for c in combo)
        left = int(budget - total)
        tail = f"  (₹{left} to spare!)" if left > 0 else ""
        return (f"✨  Here's a tasty combo within ₹{int(budget)}:\n\n  {body}\n  = ₹{int(total)}{tail}"
                + note + "\n\nWant me to add this combo? 😊")

    # --- Group bundle ---
    if group_size:
        n = max(1, int(group_size))
        base_dosa = usable(DOSA_ITEMS)
        base_idli = usable(IDLI_ITEMS)
        sweet = usable(DESSERT_ITEMS)
        drink = usable(DRINK_ITEMS)
        parts = []
        if base_dosa:
            parts.append(f"  • {n}x {base_dosa[0].title()}")
        if base_idli:
            parts.append(f"  • {n}x {base_idli[0].title()}")
        if sweet:
            parts.append(f"  • {max(1, n // 2)}x {sweet[0].title()}")
        if drink:
            parts.append(f"  • {n}x {drink[0].title()}")
        if not parts:
            return "🤔  I couldn't build a group bundle for that preference — happy to suggest individually!"
        return (f"✨  A shareable spread for {n}:\n\n" + "\n".join(parts) + note
                + "\n\nShall I add this bundle? 😊")

    # --- Default: today's specials + diverse chef's picks + a pairing ---
    lines: List[str] = []
    sp_ok = [k for k in (find_item(x) for x in specials)
             if k and avail.get(k, True) and item_matches(k, parsed)]
    if sp_ok:
        lines.append("🌟  Today's specials: " + ", ".join(_named(k, desc=False) for k in sp_ok[:3]))

    picks, used = [], set(sp_ok) | set(cart)
    for group in (DOSA_ITEMS, IDLI_ITEMS + RICE_ITEMS + LIGHT_ITEMS, DESSERT_ITEMS + DRINK_ITEMS):
        cand = [i for i in usable(group) if i not in used and i not in picks]
        cand.sort(key=lambda i: (i not in POPULAR_ITEMS, MENU_DETAILS[i]["price"]))
        if cand:
            picks.append(cand[0]); used.add(cand[0])
    if picks:
        lines.append("👨‍🍳  Chef's picks for you:")
        lines += [f"   • {_named(p)}" for p in picks]

    if cart:
        extra = [i for i in usable(DESSERT_ITEMS + DRINK_ITEMS) if i not in cart][:2]
        if extra:
            lines.append("🍮  Goes great with your order: " + " and ".join(_named(e, desc=False) for e in extra))

    if not lines:
        return ("🤔  I couldn't find anything matching that preference right now. "
                "Would you like to relax it, or see the full menu?")
    return "✨  Here are my recommendations:\n\n" + "\n".join(lines) + note + "\n\nWant me to add any of these? 😊"


def save_order(items: Dict[str, int], total: float):
    orders = []
    if os.path.exists(ANALYTICS_FILE):
        try:
            with open(ANALYTICS_FILE, "r") as f:
                orders = json.load(f)
        except Exception:
            pass
    orders.append({
        "order_id": len(orders) + 101,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "items": items,
        "total": float(total),
    })
    with open(ANALYTICS_FILE, "w") as f:
        json.dump(orders, f, indent=4)


# ==========================================================================
# 5. Session state
# ==========================================================================
@dataclass
class Session:
    cart: Dict[str, int] = field(default_factory=dict)
    preference: str = ""
    history: List[str] = field(default_factory=list)
    awaiting_confirmation: bool = False
    pending: Optional[dict] = None          # disambiguation / swap awaiting a reply
    messages: List[BaseMessage] = field(default_factory=list)

    def cart_summary(self) -> str:
        return ", ".join(f"{q}x {i.title()}" for i, q in self.cart.items()) if self.cart else "Empty"


# ==========================================================================
# 6. Tool schemas (binding) + local handlers (execution)
# ==========================================================================
@tool
def show_menu(category: str = "") -> str:
    """Show the menu. Leave category empty (or 'full'/'all') for the WHOLE menu, or pass a
    category like 'desserts', 'drinks', 'dosa', 'idli', 'rice', 'light meals', 'poori'."""


@tool
def search_menu(query: str = "", max_price: Optional[float] = None,
                preference_tag: Optional[str] = None) -> str:
    """Search/filter the menu by free text, a category, a max price (e.g. 200), and/or a
    dietary tag (vegan, jain, spicy, gluten-free, vegetarian)."""


@tool
def recommend(budget: Optional[float] = None, group_size: Optional[int] = None,
              preference: str = "") -> str:
    """Suggest dishes: pass budget for a combo under a price, group_size for a shareable
    bundle, and/or a preference (vegan/jain/spicy...). With none, returns specials + favourites
    (and pairings if the cart isn't empty)."""


@tool
def add_item(item_name: str, quantity: int = 1) -> str:
    """Add a menu item (with quantity) to the customer's cart."""


@tool
def remove_item(item_name: str, quantity: int = 1) -> str:
    """Remove a menu item (with quantity) from the customer's cart."""


@tool
def set_preference(preference: str) -> str:
    """Save the customer's dietary preference, e.g. 'vegan', 'jain', 'spicy', 'gluten-free'."""


@tool
def show_bill() -> str:
    """Produce the draft bill / receipt with subtotal, 5% GST and total. Call before confirming."""


@tool
def confirm_order(confirm: bool = True) -> str:
    """Confirm (True) and send the order to the kitchen, or cancel the draft (False)."""


@tool
def view_analytics() -> str:
    """Show business analytics: total orders, revenue, average order value, most popular item."""


TOOL_SCHEMAS = [show_menu, search_menu, recommend, add_item, remove_item,
                set_preference, show_bill, confirm_order, view_analytics]

# Outputs printed verbatim (never paraphrased / refused by the model)
VERBATIM_TOOLS = {"show_menu", "recommend", "show_bill", "view_analytics"}


def handle_tool(name: str, args: dict, s: Session) -> str:
    avail = load_availability_and_specials()["availability"]

    if name == "show_menu":
        return show_menu_text(args.get("category", ""))

    if name == "search_menu":
        res = menu_search(args.get("query", ""), args.get("max_price"), args.get("preference_tag"))
        return "🔎  Here's what I found:\n\n" + res

    if name == "recommend":
        return build_recommendations(
            budget=args.get("budget"), group_size=args.get("group_size"),
            preference=args.get("preference", ""), cart=s.cart, saved_pref=s.preference,
        )

    if name == "set_preference":
        s.preference = (args.get("preference", "") or "").lower().strip()
        parsed = parse_preference(s.preference)
        if parsed["notes"]:
            return f"👍  Got it — noted '{s.preference}'. I'll only suggest {'; '.join(parsed['notes'])}."
        return f"👍  Noted your '{s.preference}' preference — I'll keep it in mind!"

    if name == "add_item":
        item = find_item(args.get("item_name", ""))
        qty = max(1, int(args.get("quantity", 1) or 1))
        raw = args.get("item_name", "")
        if not item:
            sugg = suggest_items(raw)
            tip = ("  Did you mean: " + ", ".join(x.title() for x in sugg) + "?") if sugg else ""
            return f"🤔  I couldn't find '{raw}' on our menu.{tip}"
        if not avail.get(item, True):
            alts = similar_available(item)
            if alts:
                alt_str = ", ".join(f"{a.title()} (₹{MENU_DETAILS[a]['price']})" for a in alts)
                return (f"😕  Sorry, {item.title()} is out of stock right now.\n"
                        f"   You might love these instead: {alt_str}")
            return f"😕  Sorry, {item.title()} is out of stock right now."
        if item == "poori" and datetime.now().hour < 19:
            return (f"⏰  Poori is served only after 7:00 PM "
                    f"(it's {datetime.now().strftime('%I:%M %p')} now). Can I get you something else?")
        s.cart[item] = s.cart.get(item, 0) + qty
        s.history.append(f"Added {qty} x {item.title()}")
        s.awaiting_confirmation = False
        return f"✅  Added {qty}x {item.title()} to your order!"

    if name == "remove_item":
        item = find_item(args.get("item_name", ""))
        qty = max(1, int(args.get("quantity", 1) or 1))
        if not item or item not in s.cart:
            return f"🤔  '{args.get('item_name', '')}' isn't in your cart, so nothing to remove."
        if s.cart[item] <= qty:
            del s.cart[item]
            s.history.append(f"Removed all {item.title()}")
            return f"🗑️  Removed all {item.title()} from your order."
        s.cart[item] -= qty
        s.history.append(f"Removed {qty} x {item.title()}")
        s.awaiting_confirmation = False
        return f"🗑️  Removed {qty}x {item.title()} from your order."

    if name == "show_bill":
        s.cart = {i: q for i, q in s.cart.items() if i in MENU_DETAILS}
        if not s.cart:
            return "🛒  Your cart is empty — add a few tasty items first!"
        lines, subtotal = [], 0.0
        for i, q in s.cart.items():
            line = MENU_DETAILS[i]["price"] * q
            subtotal += line
            lines.append(f"  • {q}x {i.title()} (₹{MENU_DETAILS[i]['price']} each) → ₹{line:.2f}")
        gst = subtotal * 0.05
        total = subtotal + gst
        s.awaiting_confirmation = True
        return (
            "🧾  Here's your bill:\n\n"
            "╔══════════════════════════════════╗\n"
            "        DAKSHIN DELIGHTS\n"
            "╚══════════════════════════════════╝\n"
            + "\n".join(lines) +
            "\n────────────────────────────────────\n"
            f"  Subtotal:     ₹{subtotal:.2f}\n"
            f"  GST (5%):     ₹{gst:.2f}\n"
            f"  Grand Total:  ₹{total:.2f}\n"
            f"  Rounded Bill: ₹{math.floor(total)}\n"
            "────────────────────────────────────\n"
            "Shall I confirm this order? Reply YES to place it, or NO to keep editing. 😊"
        )

    if name == "confirm_order":
        if not args.get("confirm", True):
            s.awaiting_confirmation = False
            return "No problem — order not placed. Take your time, I'm here when you're ready. 😊"
        s.cart = {i: q for i, q in s.cart.items() if i in MENU_DETAILS}
        if not s.cart:
            return "🛒  Your cart is empty — nothing to confirm yet!"
        subtotal = sum(MENU_DETAILS[i]["price"] * q for i, q in s.cart.items())
        total = math.floor(subtotal + subtotal * 0.05)
        save_order(dict(s.cart), total)
        s.cart.clear()
        s.history.clear()
        s.awaiting_confirmation = False
        return (f"🎉  Order confirmed and sent to the kitchen! Your total is ₹{total}. "
                "Thank you for choosing Dakshin Delights — enjoy your meal! 😋")

    if name == "view_analytics":
        if not os.path.exists(ANALYTICS_FILE):
            return "No orders yet — place one first to see analytics."
        try:
            with open(ANALYTICS_FILE, "r") as f:
                orders = json.load(f)
            if not orders:
                return "No orders yet — place one first to see analytics."
            revenue = sum(o["total"] for o in orders)
            counts: Dict[str, int] = {}
            for o in orders:
                for it, q in o["items"].items():
                    counts[it] = counts.get(it, 0) + q
            top = max(counts, key=counts.get) if counts else "None"
            return (
                "📊  DAKSHIN DELIGHTS — Business Snapshot\n"
                f"• Total Orders:        {len(orders)}\n"
                f"• Total Revenue:       ₹{revenue:.2f}\n"
                f"• Average Order Value: ₹{revenue / len(orders):.2f}\n"
                f"• Most Popular Item:   {top.title()} ({counts.get(top, 0)} sold)"
            )
        except Exception as e:
            return f"Error compiling analytics: {e}"

    return f"Unknown tool '{name}'."


# ==========================================================================
# 7. The single agent
# ==========================================================================
# Use a fast model. The deterministic command parser below handles add/remove/
# bill/confirm/menu/recommend without the LLM, so tool-routing quirks of any
# model no longer affect the core flows — the model only handles chit-chat.
MODEL_NAME = "gemini-3.5-flash"
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
llm_with_tools = llm.bind_tools(TOOL_SCHEMAS)


def system_prompt(s: Session) -> SystemMessage:
    data = load_availability_and_specials()
    specials = ", ".join(data["todays_specials"]) or "None"
    oos = [i.title() for i, ok in data["availability"].items() if not ok]
    return SystemMessage(content=(
        "You are 'Dakshin', the warm, upbeat host of 'Dakshin Delights', a South Indian "
        "restaurant. You greet guests by name when they share it, sound friendly and helpful, "
        "and answer EVERY request. You never say you 'cannot' show something — you use a tool.\n\n"
        f"LIVE CART (ground truth): {s.cart_summary()}\n"
        f"Saved dietary preference: {s.preference or 'None'}\n"
        f"Awaiting bill confirmation: {s.awaiting_confirmation}\n"
        f"Today's specials: {specials}\n"
        f"Out of stock: {', '.join(oos) or 'None'}\n\n"
        "How to respond:\n"
        "• The SYSTEM prints the actual menu/recommendation/bill/cart from your tool calls. "
        "Your own message should be just ONE short, warm lead-in sentence (a greeting or "
        "'Here you go!') with NO prices and NO item lists — never re-type what a tool returns.\n"
        "Rules:\n"
        "1. MENU requests: call show_menu (empty/'full' = whole menu; else the category: "
        "'desserts', 'drinks', 'dosa', 'idli', 'rice', 'light meals', 'poori').\n"
        "2. RECOMMENDATIONS / combos / group / budget picks: call recommend (pass budget, "
        "group_size, and/or preference when stated).\n"
        "3. PREFERENCES (vegan, jain, 'no onion', 'no garlic', mild, spicy...): call set_preference. "
        "Never name a dish that violates it — let the tools filter.\n"
        "4. CART: you MUST call add_item / remove_item to change it. Never claim a change you didn't "
        "make with a tool. The system shows the real cart; treat the LIVE CART above as truth.\n"
        "5. BILLING: 'bill'/'checkout' -> show_bill. Then YES -> confirm_order(true); NO -> "
        "confirm_order(false).\n"
        "6. Only ever suggest AVAILABLE items. Never invent prices or totals.\n"
        "7. GENERAL / OFF-TOPIC questions: if someone asks something unrelated to the "
        "restaurant (trivia, advice, chit-chat), answer briefly and warmly if you can, then "
        "gently steer back — e.g. 'Happy to help! Now, can I get you anything from our menu?' "
        "Never be rude or refuse curtly; stay the gracious host."
    ))


def extract_text(msg) -> str:
    """Keep only human-readable text; drop Gemini thinking/tool_use/signature blocks."""
    content = msg.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") in (None, "text") \
                    and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "".join(parts).strip()
    return ""


# --- Deterministic command parsing (no LLM) ---------------------------------
# A clear order like "add one lemon rice and masala idli" must ALWAYS work,
# regardless of how the model feels about tool-calling. We parse these directly.
NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "couple": 2, "pair": 2, "single": 1,
}
ADD_VERBS = ["add", "want", "get me", "gimme", "give me", "i'll have", "ill have",
             "i will have", "i'll take", "ill take", "order", "include", "i'd like",
             "id like", "i would like"]
REMOVE_VERBS = ["remove", "delete", "cancel", "take out", "drop", "take off"]
NEG_CUES = {"no", "not", "don't", "dont", "donot", "without", "avoid", "skip", "allergic",
            "hold", "less", "non", "free", "cannot", "can't", "cant", "never", "dislike", "hate"}
QUESTION_MARKERS = ["?", "do you", "what", "which", "how", "why", "where", "price",
                    "cost", "available", "is there", "are there", "menu", "recommend",
                    "suggest", "show", "tell me", "list"]
YES_WORDS = ["yes", "yeah", "yep", "confirm", "sure", "ok", "okay", "place", "proceed",
             "go ahead", "do it", "yup"]
NO_WORDS = ["no", "nope", "cancel", "not yet", "wait", "stop", "don't", "dont"]
BILL_PHRASES = {"bill", "my bill", "the bill", "checkout", "check out", "total",
                "give me the bill", "give me bill", "show bill", "show me the bill"}
RECOMMEND_MARKERS = ["recommend", "suggest", "what's good", "whats good", "what is good",
                     "any suggestion", "what should i", "best dish", "best seller",
                     "most popular", "surprise me", "what's nice", "whats nice",
                     "what do you recommend", "specials"]
MENU_PHRASES = ["what do you have", "what's available", "whats available",
                "what can i order", "your options", "what do you offer"]
DONE_PHRASES = ["that's it", "thats it", "that is it", "i'm done", "im done",
                "i am done", "done ordering", "that's all", "thats all", "that is all",
                "finalize", "wrap up", "place order", "place the order", "ready to pay"]
MENU_SHOW_WORDS = ["show", "available", "avaliable", "list", "options", "see",
                   "display", "what", "which", "have"]


def has_quantity(t: str) -> bool:
    return bool(re.search(r"\b\d+\b", t)) or any(w in NUM_WORDS for w in t.split())


def order_followup(s: Session) -> str:
    """A short, friendly nudge after items are added (gentle, preference-aware upsell)."""
    if not s.cart:
        return ""
    avail = load_availability_and_specials()["availability"]
    parsed = parse_preference(s.preference)
    has_extra = any(i in DESSERT_ITEMS or i in DRINK_ITEMS for i in s.cart)
    if not has_extra:
        for cand in ["badam milk", "gulab jamun", "special falooda"]:
            if avail.get(cand, True) and item_matches(cand, parsed) and cand not in s.cart:
                return (f"Would you like to add {cand.title()} (₹{MENU_DETAILS[cand]['price']}) "
                        "to go with that? 😊  Or just say 'bill' whenever you're ready.")
    return "Anything else for you, or shall I get your bill? 😊"


def detect_budget(t: str) -> Optional[float]:
    m = re.search(r"(?:under|below|less than|within|upto|up to|max(?:imum)?)\s*₹?\s*(\d+)", t)
    return float(m.group(1)) if m else None


def detect_group(t: str) -> Optional[int]:
    m = re.search(r"(?:group of|for|party of|serves?|feed)\s*(\d+)", t)
    return int(m.group(1)) if m else None


def parse_items(text: str):
    """Return (matched [(key,qty)], unmatched [raw], ambiguous [(raw,[keys],qty)]).
    Negation/preference chunks ('no onion') are ignored. 'with' is NOT a separator
    (so 'idli with 4 pcs' stays one chunk)."""
    t = text.lower().strip()
    for v in sorted(ADD_VERBS + REMOVE_VERBS, key=len, reverse=True):
        if t.startswith(v + " "):
            t = t[len(v):].strip()
            break
    chunks = re.split(r"\band\b|,|&|\bplus\b", t)
    matched: List[Tuple[str, int]] = []
    unmatched: List[str] = []
    ambiguous: List[Tuple[str, List[str], int]] = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        if {tok.strip(".!") for tok in ch.split()} & NEG_CUES:
            continue
        qty = 1
        m = re.match(r"^(\d+)\s+(.*)", ch)
        if m:
            qty, rest = int(m.group(1)), m.group(2)
        else:
            parts = ch.split()
            if parts and parts[0] in NUM_WORDS:
                qty, rest = NUM_WORDS[parts[0]], " ".join(parts[1:])
            else:
                rest = ch
        rest = re.sub(r"^(of|plate of|plates of|order of|the|some)\s+", "", rest).strip()
        rest = rest.replace("please", "").strip()
        if not rest:
            continue
        cands = match_candidates(rest)
        if not cands:
            if len(rest) >= 3:
                unmatched.append(rest)
        elif len(cands) == 1:
            matched.append((cands[0], max(1, qty)))
        else:
            ambiguous.append((rest, cands[:6], max(1, qty)))
    return matched, unmatched, ambiguous


def _miss_lines(unmatched: List[str]) -> List[str]:
    out = []
    for raw in unmatched:
        sugg = suggest_items(raw)
        tip = ("  Did you mean: " + ", ".join(x.title() for x in sugg) + "?") if sugg else ""
        out.append(f"🤔  I couldn't find '{raw}' on our menu.{tip}")
    return out


def _choice_prompt(raw: str, cands: List[str]) -> str:
    opts = "  ".join(f"{i}) {c.title()}" for i, c in enumerate(cands, 1))
    return (f"🤔  We have a few '{raw}' options — which would you like?\n{opts}\n"
            "(reply with the number or the name)")


def _add_block(matched, unmatched, ambiguous, s: Session) -> str:
    """Run the adds, queue ONE disambiguation if needed, and build the reply."""
    outs = [handle_tool("add_item", {"item_name": k, "quantity": q}, s) for k, q in matched]
    outs += _miss_lines(unmatched)
    reply = "\n".join(outs).strip()
    if ambiguous:
        raw, cands, qty = ambiguous[0]
        s.pending = {"type": "choose", "candidates": cands, "qty": qty}
        line = _choice_prompt(raw, cands)
        reply = (reply + "\n\n" + line).strip() if reply else line
        return reply + f"\n\n🛒  Your cart: {s.cart_summary()}"
    return (reply + f"\n\n🛒  Your cart: {s.cart_summary()}\n\n" + order_followup(s)).strip()


CORRECTION_CUES = ["i said", "i meant", "i wanted", "i asked", "instead of", "instead",
                   "rather than", "actually", "make it", "change it", "change to",
                   "correction", "i ordered"]
SPECIAL_INTENT = ["special today", "today's special", "todays special", "today special",
                  "what's special", "whats special", "what is the special", "what is special",
                  "any special", "specials", "special menu", "special of the day",
                  "chef special", "chef's special", "todays specials", "today's specials"]
DIET_TAG_WORDS = ["jain", "vegan", "vegetarian", "gluten", "spicy"]
DIET_LIST_WORDS = ["show", "list", "all", "which", "what", "options", "dish", "dishes",
                   "items", "food", "available", "have", "any"]


def resolve_pending(text: str, s: Session) -> Optional[str]:
    """Resolve a queued disambiguation ('choose') or swap confirmation ('swap')."""
    p = s.pending
    t = text.lower().strip()
    if p["type"] == "choose":
        cands, qty = p["candidates"], p["qty"]
        if any(w in t for w in ["cancel", "none", "never mind", "nvm", "forget", "neither"]):
            s.pending = None
            return "No problem — cancelled that. What else can I get you? 😊"
        # by number / ordinal
        idx = None
        mnum = re.search(r"\b([1-9])\b", t)
        if mnum:
            idx = int(mnum.group(1))
        else:
            for w, n in {"first": 1, "1st": 1, "second": 2, "2nd": 2,
                         "third": 3, "3rd": 3, "fourth": 4}.items():
                if re.search(rf"\b{w}\b", t):
                    idx = n
                    break
        choice = cands[idx - 1] if idx and 1 <= idx <= len(cands) else None
        # by name / keyword
        if not choice:
            qtoks = set(_norm(t).split())
            scored = sorted(((len(qtoks & set(_norm(c).split())), c) for c in cands),
                            reverse=True)
            if scored and scored[0][0] > 0:
                choice = scored[0][1]
        if choice:
            s.pending = None
            out = handle_tool("add_item", {"item_name": choice, "quantity": qty}, s)
            return (out + f"\n\n🛒  Your cart: {s.cart_summary()}\n\n" + order_followup(s)).strip()
        return "Sorry, I didn't catch which one. " + _choice_prompt(p.get("raw", "item"), cands)

    if p["type"] == "swap":
        old, new, qty = p["old"], p["new"], p["qty"]
        if any(re.search(rf"\b{re.escape(w)}\b", t) for w in YES_WORDS):
            s.pending = None
            if old in s.cart:
                handle_tool("remove_item", {"item_name": old, "quantity": 999}, s)
            handle_tool("add_item", {"item_name": new, "quantity": qty}, s)
            return (f"✅  Swapped {qty}x {old.title()} for {qty}x {new.title()}!"
                    f"\n\n🛒  Your cart: {s.cart_summary()}")
        if any(re.search(rf"\b{re.escape(w)}\b", t) for w in NO_WORDS):
            s.pending = None
            return "Okay, I'll leave your order as it is. 😊"
        return f"Shall I swap {old.title()} for {new.title()}? (yes / no)"
    s.pending = None
    return None


def handle_correction(t: str, s: Session) -> Optional[str]:
    """Handle 'I said X not Y' / 'change to X' — swap a cart variant when possible."""
    parts = re.split(r"\bnot\b|\binstead of\b|\brather than\b", t, maxsplit=1)
    wanted_text = parts[0]
    for f in ["i said", "i meant", "i wanted", "i asked for", "i asked", "i ordered",
              "actually", "please", "i want", "make it", "change it to", "change to",
              "change", " with "]:
        wanted_text = wanted_text.replace(f, " ")
    wanted_text = wanted_text.strip()
    cands = match_candidates(wanted_text)
    if not cands:
        return None                     # let the model handle a vaguer correction
    wanted = cands[0]
    # find a sibling variant already in the cart (same base, different variant)
    sibling = next((k for k in s.cart if _base(k) == _base(wanted) and k != wanted), None)
    if len(cands) > 1 and not sibling:
        s.pending = {"type": "choose", "candidates": cands[:6], "qty": 1, "raw": wanted_text}
        return _choice_prompt(wanted_text, cands[:6])
    if sibling:
        qty = s.cart.get(sibling, 1)
        s.pending = {"type": "swap", "old": sibling, "new": wanted, "qty": qty}
        return (f"Got it — you'd like {qty}x {wanted.title()} instead of "
                f"{qty}x {sibling.title()}. Shall I swap them? (yes / no)")
    out = handle_tool("add_item", {"item_name": wanted, "quantity": 1}, s)
    return (out + f"\n\n🛒  Your cart: {s.cart_summary()}").strip()


def interpret_command(text: str, s: Session) -> Optional[str]:
    """Handle clear commands WITHOUT the LLM. Returns the reply if handled,
    else None (defer to the model for greetings / small talk / anything unclear)."""
    t = text.lower().strip()

    # 0) Resolve a queued disambiguation / swap first
    if s.pending:
        res = resolve_pending(text, s)
        if res is not None:
            return res

    # 1) Confirm / cancel a pending bill
    if s.awaiting_confirmation:
        if any(re.search(rf"\b{re.escape(w)}\b", t) for w in YES_WORDS):
            return handle_tool("confirm_order", {"confirm": True}, s)
        if any(re.search(rf"\b{re.escape(w)}\b", t) for w in NO_WORDS):
            return handle_tool("confirm_order", {"confirm": False}, s)

    has_add = any(t.startswith(v) or f" {v} " in f" {t} " for v in ADD_VERBS)
    has_remove = any(v in t for v in REMOVE_VERBS)
    cat = resolve_category(t)

    # 2) Bill / "that's it, the bill please"
    if re.search(r"\bbill\b", t) or "checkout" in t or "check out" in t \
            or any(p in t for p in DONE_PHRASES):
        return handle_tool("show_bill", {}, s)

    # 3) Correction ("i said X not Y", "change to X") -> swap / clarify
    if any(c in t for c in CORRECTION_CUES):
        res = handle_correction(t, s)
        if res is not None:
            return res

    # 3b) Today's specials  ("what's the special today", "special menu", "any specials")
    if not has_add and not has_remove and any(p in t for p in SPECIAL_INTENT):
        return todays_specials_text(s)

    # 3c) Dietary listing  ("show all jain dishes", "vegan options", "what's spicy")
    if not has_add and not has_remove \
            and not any(m in t for m in RECOMMEND_MARKERS) \
            and any(d in t for d in DIET_TAG_WORDS) \
            and any(w in t for w in DIET_LIST_WORDS):
        res = dietary_listing(t, s)
        if res is not None:
            return res

    # 3d) Preference / dislike statement ("i don't like onions, give me such foods",
    #     "i'm vegan", "no garlic") -> save preference AND recommend matching dishes.
    #     Runs before add-parsing so 'give me' doesn't hijack it; skipped when a
    #     specific dish is named (e.g. 'add jain dosa').
    pp = parse_preference(t)
    if pp["active"]:
        chk_items, _, chk_amb = parse_items(t)
        is_q = any(q in t for q in QUESTION_MARKERS)
        give_cue = any(c in t for c in ["give me", "want", "i'd like", "such food",
                                        "recommend", "suggest", "prefer", "i like",
                                        "don't like", "dont like", "donot like", "i'm ",
                                        "i am ", "looking for"])
        statement = bool(pp["exclude_words"] or pp["exclude_tags"]) or give_cue \
            or len(t.split()) <= 2
        if not chk_items and not chk_amb and statement and (not is_q or give_cue):
            label = canonical_pref(pp) or t
            ack = handle_tool("set_preference", {"preference": label}, s)
            return ack + "\n\n" + build_recommendations(saved_pref=s.preference)

    # 4) Recommendations (also when the ask is clearly group- or budget-based)
    group_intent = bool(re.search(r"\b(group|party|friends|people|family)\b", t)) and detect_group(t)
    budget_intent = any(w in t for w in ["under", "below", "within", "cheap", "budget", "spend"]) \
        and detect_budget(t)
    if any(m in t for m in RECOMMEND_MARKERS) or group_intent or budget_intent:
        pp = parse_preference(t)
        return handle_tool("recommend", {
            "budget": detect_budget(t), "group_size": detect_group(t),
            "preference": t if pp["active"] else "",
        }, s)

    # 5) Menu  (BEFORE add; a category word like "dosa menu" beats the word "all")
    explicit_menu = ("menu" in t) or any(p in t for p in MENU_PHRASES)
    menu_intent = explicit_menu or (
        not has_add and not has_remove and (
            is_full_menu_request(t)
            or t in CATEGORY_SYNONYMS
            or (cat and any(w in t for w in MENU_SHOW_WORDS) and not has_quantity(t))
        )
    )
    if menu_intent:
        return handle_tool("show_menu", {"category": cat or ""}, s)

    # 6) Add / remove (with disambiguation when a name matches several dishes)
    items, misses, ambiguous = parse_items(t)
    if has_remove and (items or misses):
        outs = [handle_tool("remove_item", {"item_name": k, "quantity": q}, s) for k, q in items]
        outs += _miss_lines(misses)
        return "\n".join(outs) + f"\n\n🛒  Your cart: {s.cart_summary()}"
    if has_add and (items or misses or ambiguous):
        return _add_block(items, misses, ambiguous, s)

    # 7) Preference statement with no item ("i'm vegan", "no onion", "mild")
    is_question = any(q in t for q in QUESTION_MARKERS)
    if not has_add and not has_remove and not items and not ambiguous and not is_question:
        pp = parse_preference(t)
        if pp["active"]:
            ack = handle_tool("set_preference", {"preference": t}, s)
            return ack + "\n\n" + build_recommendations(saved_pref=s.preference)

    # 8) Bare item list -> add ("one masala idli and 2 lemon rice", "podi dosa")
    if (items or ambiguous) and not is_question:
        return _add_block(items, misses, ambiguous, s)

    # 9) Common restaurant FAQ (hours, location, delivery, payment, etc.)
    ans = faq_answer(t)
    if ans is not None:
        return ans

    return None     # everything else (greetings, general/out-of-scope) -> the model


# --- streaming output --------------------------------------------------------
def stream_out(text: str, word_delay: float = 0.018):
    """Print the reply token-by-token for a live, streaming feel."""
    for tok in re.findall(r"\S+|\s+", text):
        sys.stdout.write(tok)
        sys.stdout.flush()
        if tok.strip():
            time.sleep(word_delay)
    sys.stdout.write("\n")
    sys.stdout.flush()


def run_turn(s: Session, user_text: str, max_steps: int = 4) -> str:
    """One turn. Tool outputs are shown verbatim (so nothing gets dropped); the
    model only supplies a short warm lead-in (greeting/intro)."""
    s.messages.append(HumanMessage(content=user_text))
    tool_outputs: List[str] = []
    cart_changed = False
    lead_in = ""

    def assemble() -> str:
        body = "\n\n".join(tool_outputs).strip()
        out = f"{lead_in}\n\n{body}".strip() if (lead_in and body) else (body or lead_in)
        if cart_changed or any(w in out.lower() for w in ("added", "removed")):
            out += f"\n\n🛒  Your cart: {s.cart_summary()}"
        return out

    for _ in range(max_steps):
        ai: AIMessage = llm_with_tools.invoke([system_prompt(s)] + s.messages)
        s.messages.append(ai)

        if not getattr(ai, "tool_calls", None):
            if tool_outputs:
                return assemble()
            text = extract_text(ai)
            return text or "How can I help you today? 😊"

        # Capture a short, price-free lead-in (a greeting/intro) from the model.
        if not lead_in:
            t = extract_text(ai)
            if t and "₹" not in t and len(t) <= 220:
                lead_in = t

        for tc in ai.tool_calls:
            result = handle_tool(tc["name"], tc.get("args", {}), s)
            tool_outputs.append(result)
            if tc["name"] in ("add_item", "remove_item"):
                cart_changed = True
            s.messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    return assemble() or "Sorry, could you rephrase that? 😊"


# ==========================================================================
# 8. Interactive loop
# ==========================================================================
def main():
    data = load_availability_and_specials()
    specials = ", ".join(data["todays_specials"]) or "None today"

    print("=" * 60)
    print("Welcome to Dakshin Delights! ( Your Assistant)")
    print("=" * 60)
    print(f"Today: {datetime.now().strftime('%A, %B %d, %Y')}")
    print(f"🌟 TODAY'S SPECIALS: {specials} 🌟")
    print("=" * 60)
    print("Commands:  'menu' · 'history' · 'preferences' · 'cart' · 'exit'\n")

    s = Session()
    _ = uuid.uuid4().hex

    while True:
        try:
            user = input("\nYou: ").strip()
            if not user:
                continue
            low = user.lower()

            if low in ("exit", "quit"):
                print("Thank you for choosing Dakshin Delights! Goodbye! 👋")
                break
            if low == "menu":                       # instant full menu, no LLM
                sys.stdout.write("\nWaiter:\n")
                stream_out(render_full_menu())
                continue
            if low == "history":
                print("\n[History]")
                print("  (none yet)" if not s.history
                      else "\n".join(f"  {i}. {h}" for i, h in enumerate(s.history, 1)))
                continue
            if low == "preferences":
                print(f"\n[Preference]: {(s.preference or 'None set').title()}")
                continue
            if low == "cart":
                print(f"\n[Cart]: {s.cart_summary()}")
                continue

            # 1) Deterministic fast-path for clear add/remove/bill/confirm — no LLM,
            #    no mis-routing. 2) Otherwise let the model handle it.
            reply = interpret_command(user, s)
            if reply is None:
                reply = run_turn(s, user)

            sys.stdout.write("\nWaiter: ")
            sys.stdout.flush()
            stream_out(reply)

        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()