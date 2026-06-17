"""
Dakshin Delights — Fast restaurant assistant using LangGraph.
"""

from __future__ import annotations

import os
import re
import sys
import time
import json
import math
import uuid
import threading
import itertools
from typing import Dict, List, Optional, Tuple, Annotated
from typing_extensions import TypedDict
from datetime import datetime
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

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
# 1. Session state
# ==========================================================================
@dataclass
class Session:
    cart: Dict[str, int] = field(default_factory=dict)
    preference: str = ""
    history: List[str] = field(default_factory=list)
    awaiting_confirmation: bool = False
    messages: List[BaseMessage] = field(default_factory=list)

    def cart_summary(self) -> str:
        return ", ".join(f"{q}x {i.title()}" for i, q in self.cart.items()) if self.cart else "Empty"

# Global active session pointer for tool callbacks
active_session: Optional[Session] = None

# ==========================================================================
# 2. Menu data + categories
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

CATEGORY_SYNONYMS = {
    "idli": "idli", "idly": "idli", "idlis": "idli", "steamed": "idli",
    "dosa": "dosa", "dosas": "dosa", "dose": "dosa", "crepe": "dosa",
    "rice": "rice", "biryani": "rice", "biriyani": "rice", "pongal": "rice", "pulao": "rice",
    "light": "light", "tiffin": "light", "snack": "light", "snacks": "light",
    "avalakki": "light", "shavige": "light", "poha": "light",
    "poori": "poori", "puri": "poori",
    "dessert": "desserts", "desserts": "desserts", "sweet": "desserts",
    "sweets": "desserts", "icecream": "desserts", "ice cream": "desserts", "ice-cream": "desserts", "falooda": "desserts",
    "drink": "drinks", "drinks": "drinks", "beverage": "drinks",
    "beverages": "drinks", "juice": "drinks", "shake": "drinks", "milk": "drinks",
    "coke": "drinks", "coca cola": "drinks", "soda": "drinks", "cold drink": "drinks", "cold drinks": "drinks",
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
_AVAIL_WARNED = False

# ==========================================================================
# 3. Availability / specials
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
        _AVAIL_WARNED = False
        return data
    except Exception as e:
        if not _AVAIL_WARNED:
            print(f"⚠️  '{AVAILABILITY_FILE}' has a formatting error ({e}). "
                  "Using all-items-available until fixed.")
            _AVAIL_WARNED = True
        return default

# ==========================================================================
# 4. Lookup / category helpers
# ==========================================================================
def _norm(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("idly", "idli").replace("dosai", "dosa").replace("vadai", "vada")
    s = s.replace("pudi", "podi")
    s = s.replace("maslaa", "masala").replace("masla", "masala")
    s = s.replace("pieces", "pcs").replace("piece", "pcs")
    s = re.sub(r"(\d+)\s*pc\b", r"\1 pcs", s)
    s = re.sub(r"[()]", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _base(key: str) -> str:
    return " ".join(tok for tok in _norm(key).split()
                    if tok != "pcs" and not tok.isdigit())

_NORM_MENU: Dict[str, str] = {}
for _k in MENU_DETAILS:
    _NORM_MENU.setdefault(_norm(_k), _k)

def match_candidates(query: str) -> List[str]:
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
    
    # Word boundary matching
    qt = s.split()
    cands = []
    for nk, k in _NORM_MENU.items():
        nk_words = nk.split()
        if all(any(re.search(rf"\b{re.escape(qw)}\b", nw) for nw in nk_words) for qw in qt):
            cands.append(k)
            
    if not cands:
        # Fallback to substring matching if no word-boundary match is found
        cands = [k for nk, k in _NORM_MENU.items() if s in nk]
    if not cands:
        best = fuzzy_best(s, list(_NORM_MENU.keys()), 72)
        cands = [_NORM_MENU[best]] if best else []
    if len(cands) > 1:
        base_eq = [c for c in cands if _base(c) == s]
        if base_eq:
            cands = base_eq
    return sorted(set(cands), key=lambda c: (len(c), c))

def find_item(user_input: str) -> Optional[str]:
    cands = match_candidates(user_input)
    return cands[0] if cands else None

def suggest_items(user_input: str, n: int = 3) -> List[str]:
    s = (user_input or "").lower().strip()
    s_clean = re.sub(r"[()&,.-]", " ", s)
    toks = [t for t in s_clean.split() if len(t) > 2]
    
    # 1. Check category match first
    cat = resolve_category(s)
    if cat and cat in CATEGORY_ITEMS:
        avail = load_availability_and_specials()["availability"]
        cat_items = [i for i in CATEGORY_ITEMS[cat] if avail.get(i, True)]
        if cat_items:
            popular_in_cat = [i for i in POPULAR_ITEMS if i in cat_items]
            other_in_cat = [i for i in cat_items if i not in popular_in_cat]
            return (popular_in_cat + other_in_cat)[:n]

    # 2. Match actual word boundaries to avoid e.g. "ice" matching "rice"
    scored = []
    avail = load_availability_and_specials()["availability"]
    for k in MENU_DETAILS:
        if not avail.get(k, True):
            continue
        k_clean = re.sub(r"[()&,.-]", " ", k).lower()
        k_words = k_clean.split()
        overlap = sum(1 for t in toks if any(re.search(rf"\b{re.escape(t)}\b", kw) for kw in k_words))
        if overlap:
            scored.append((overlap, k))
            
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return [k for _, k in scored[:n]]

    # 3. Fallback: spelling similarity
    from difflib import get_close_matches
    available_items = [k for k in MENU_DETAILS if avail.get(k, True)]
    matches = get_close_matches(s, available_items, n=n, cutoff=0.3)
    if matches:
        return matches

    # 4. Final fallback: popular available items
    return [k for k in POPULAR_ITEMS if avail.get(k, True)][:n]

# --- preference understanding ------------------------------------------------
_NO_ONION = ["no onion", "without onion", "onion free", "onion-free", "avoid onion",
             "skip onion", "hold the onion", "not eat onion", "dont eat onion",
             "don't eat onion", "donot eat onion", "do not eat onion",
             "can't eat onion", "cant eat onion", "allergic to onion", "no onions"]
_NO_GARLIC = ["no garlic", "without garlic", "garlic free", "garlic-free", "avoid garlic",
              "skip garlic", "not eat garlic", "dont eat garlic", "don't eat garlic",
              "do not eat garlic", "allergic to garlic"]
_NO_SPICE = ["not spicy", "no spice", "no spicy", "less spicy", "mild", "non spicy"]
_DISLIKE = r"(?:no|not|n't|dont|don't|donot|do not|without|avoid|hate|dislike|allergic|" \
           r"skip|hold|less|free of|minus|never|don't like|dont like)\b[^.?!]*\b"

def parse_preference(text: str) -> dict:
    t = f" {(text or '').lower()} "
    require, exclude_tags, exclude_words, notes = set(), set(), set(), []

    onion = any(p in t for p in _NO_ONION) or bool(re.search(_DISLIKE + "onion", t))
    garlic = any(p in t for p in _NO_GARLIC) or bool(re.search(_DISLIKE + "garlic", t))
    if "jain" in t or onion or garlic:
        require.add("jain")
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
    parts = [f"no {w}" for w in sorted(parsed["exclude_words"])]
    for r in sorted(parsed["require"]):
        if r == "jain" and parsed["exclude_words"]:
            continue
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
    t = (text or "").lower()
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
    text += "\n\nWant me to add any of these to your order? 😊"
    return text

def show_menu_text(category_text: str = "") -> str:
    t = (category_text or "").strip()
    if not t or is_full_menu_request(t):
        return render_full_menu()
    cat = resolve_category(t)
    if cat:
        if t in CATEGORY_SYNONYMS or t.replace("menu", "").strip() in CATEGORY_SYNONYMS:
            return render_category(cat)
    return "🔎  Here's what I found:\n" + menu_search(query=t)

def menu_search(query="", max_price=None, preference_tag=None) -> str:
    avail = load_availability_and_specials()["availability"]
    q = _norm(query)
    parsed = parse_preference(preference_tag) if preference_tag else None

    pool = list(MENU_DETAILS.keys())
    name_filter = q
    if q:
        if is_full_menu_request(q):
            name_filter = ""
        else:
            cat = resolve_category(q)
            if cat:
                pool = CATEGORY_ITEMS[cat]
                if q in CATEGORY_SYNONYMS or q.replace("menu", "").strip() in CATEGORY_SYNONYMS:
                    name_filter = ""
                else:
                    name_filter = q

    rows = []
    for name in pool:
        d = MENU_DETAILS[name]
        if name_filter:
            filter_words = name_filter.split()
            name_clean = name.lower()
            if not all(w in name_clean for w in filter_words):
                continue
        if max_price is not None and d["price"] > max_price:
            continue
        if parsed and parsed["active"] and not item_matches(name, parsed):
            continue
        status = "Available" if avail.get(name, True) else "Not Available"
        rows.append(f"- {name.title()}: ₹{d['price']} [{status}] (Dietary: {', '.join(d['tags'])})")
    
    if not rows:
        if q and q not in FULL_MENU_WORDS:
            sugg = suggest_items(q)
            sugg_str = ", ".join(x.title() for x in sugg) if sugg else "other items on our menu"
            return f"We do not have '{q.title()}' in our menu, else you can try these: {sugg_str}"
        return "No items match your criteria. Try adjusting your constraints."
    return "\n".join(rows)

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
     "table reservations are first-come, first-served. 📞"),
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
        "items": items, "total": float(total),
    })
    try:
        with open(ANALYTICS_FILE, "w") as f:
            json.dump(orders, f, indent=4)
    except Exception:
        pass

# ==========================================================================
# 5. Tools implementations
# ==========================================================================
@tool
def show_menu(category: str = "") -> str:
    """Show the restaurant menu. Leave category empty (or 'full'/'all') for the WHOLE menu,
    or pass a specific category like 'desserts', 'drinks', 'dosa', 'idli', 'rice', 'light meals', 'poori'."""
    return show_menu_text(category)

@tool
def search_menu(query: str = "", max_price: Optional[float] = None,
                preference_tag: Optional[str] = None) -> str:
    """Search/filter the menu by free text keyword, a category, a max price, and/or a
    dietary tag (vegan, jain, spicy, gluten-free, vegetarian)."""
    res = menu_search(query, max_price, preference_tag)
    return "🔎  Here's what I found:\n\n" + res

@tool
def recommend(budget: Optional[float] = None, group_size: Optional[int] = None,
              preference: str = "") -> str:
    """Suggest dishes: pass budget for a combo under a price, group_size for a shareable
    bundle, and/or a preference (vegan/jain/spicy...). With none, returns specials + favourites
    (and pairings if the cart isn't empty)."""
    if active_session is None:
        return "No active session."
    return build_recommendations(
        budget=budget, group_size=group_size,
        preference=preference, cart=active_session.cart, saved_pref=active_session.preference
    )

@tool
def add_item(item_name: str, quantity: int = 1) -> str:
    """Add a menu item (with quantity) to the customer's cart."""
    if active_session is None:
        return "No active session."
    
    cands = match_candidates(item_name)
    if not cands:
        sugg = suggest_items(item_name)
        sugg_str = ", ".join(x.title() for x in sugg) if sugg else "other items on our menu"
        return f"We do not have '{item_name.title()}' in our menu, else you can try these: {sugg_str}"
        
    if len(cands) > 1:
        opts = "  ".join(f"• {c.title()}" for c in cands)
        return (f"🤔  We have various options in '{item_name.title()}' — which one would you like?\n"
                f"{opts}\n"
                f"(Please specify the name of the one you want)")
                
    item = cands[0]
    avail = load_availability_and_specials()["availability"]
    qty = max(1, int(quantity or 1))
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
    active_session.cart[item] = active_session.cart.get(item, 0) + qty
    active_session.history.append(f"Added {qty} x {item.title()}")
    active_session.awaiting_confirmation = False
    return f"✅  Added {qty}x {item.title()} to your order!"

@tool
def remove_item(item_name: str, quantity: int = 1) -> str:
    """Remove a menu item (with quantity) from the customer's cart."""
    if active_session is None:
        return "No active session."
    
    cands = match_candidates(item_name)
    if not cands:
        return f"🤔  '{item_name}' isn't on our menu, so nothing to remove."
        
    if len(cands) > 1:
        in_cart = [c for c in cands if c in active_session.cart]
        if not in_cart:
            return f"🤔  None of the '{item_name.title()}' options are in your cart."
        if len(in_cart) == 1:
            item = in_cart[0]
        else:
            opts = "  ".join(f"• {c.title()}" for c in in_cart)
            return (f"🤔  Which '{item_name.title()}' would you like to remove?\n"
                    f"{opts}\n"
                    f"(Please specify the name of the one you want to remove)")
    else:
        item = cands[0]
        
    qty = max(1, int(quantity or 1))
    if item not in active_session.cart:
        return f"🤔  '{item.title()}' isn't in your cart, so nothing to remove."
    if active_session.cart[item] <= qty:
        del active_session.cart[item]
        active_session.history.append(f"Removed all {item.title()}")
        return f"🗑️  Removed all {item.title()} from your order."
    active_session.cart[item] -= qty
    active_session.history.append(f"Removed {qty} x {item.title()}")
    active_session.awaiting_confirmation = False
    return f"🗑️  Removed {qty}x {item.title()} from your order."

@tool
def set_preference(preference: str) -> str:
    """Save the customer's dietary preference, e.g. 'vegan', 'jain', 'spicy', 'gluten-free'."""
    if active_session is None:
        return "No active session."
    active_session.preference = (preference or "").lower().strip()
    parsed = parse_preference(active_session.preference)
    if parsed["notes"]:
        return f"👍  Got it — noted '{active_session.preference}'. I'll only suggest {'; '.join(parsed['notes'])}."
    return f"👍  Noted your '{active_session.preference}' preference — I'll keep it in mind!"

@tool
def show_bill() -> str:
    """Produce the draft bill / receipt with subtotal, 5% GST and total. Call before confirming."""
    if active_session is None:
        return "No active session."
    active_session.cart = {i: q for i, q in active_session.cart.items() if i in MENU_DETAILS}
    if not active_session.cart:
        return "🛒  Your cart is empty — add a few tasty items first!"
    lines, subtotal = [], 0.0
    for i, q in active_session.cart.items():
        line = MENU_DETAILS[i]["price"] * q
        subtotal += line
        lines.append(f"  • {q}x {i.title()} (₹{MENU_DETAILS[i]['price']} each) → ₹{line:.2f}")
    gst = subtotal * 0.05
    total = subtotal + gst
    active_session.awaiting_confirmation = True
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

@tool
def confirm_order(confirm: bool = True) -> str:
    """Confirm (True) and send the order to the kitchen, or cancel the draft (False)."""
    if active_session is None:
        return "No active session."
    if not confirm:
        active_session.awaiting_confirmation = False
        return "No problem — order not placed. Take your time, I'm here when you're ready. 😊"
    active_session.cart = {i: q for i, q in active_session.cart.items() if i in MENU_DETAILS}
    if not active_session.cart:
        return "🛒  Your cart is empty — nothing to confirm yet!"
    subtotal = sum(MENU_DETAILS[i]["price"] * q for i, q in active_session.cart.items())
    total = math.floor(subtotal + subtotal * 0.05)
    save_order(dict(active_session.cart), total)
    active_session.cart.clear()
    active_session.history.clear()
    active_session.awaiting_confirmation = False
    return (f"🎉  Order confirmed and sent to the kitchen! Your total is ₹{total}. "
            "Thank you for choosing Dakshin Delights — enjoy your meal! 😋")

@tool
def view_analytics() -> str:
    """Show business analytics: total orders, revenue, average order value, most popular item."""
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

TOOL_SCHEMAS = [show_menu, search_menu, recommend, add_item, remove_item,
                set_preference, show_bill, confirm_order, view_analytics]

# ==========================================================================
# 6. Model Setup
# ==========================================================================
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "gemini").lower()
if MODEL_PROVIDER == "llama":
    llama_base = os.environ.get("LLAMA_API_BASE", "").strip()
    if llama_base and not llama_base.endswith("/v1"):
        llama_base = f"{llama_base.rstrip('/')}/v1"
    llm = ChatOpenAI(
        model="gemma-4-e4b",
        temperature=0,
        base_url=llama_base or None,
        api_key="none"
    )
else:
    MODEL_NAME = "gemini-3.5-flash"
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)

llm_with_tools = llm.bind_tools(TOOL_SCHEMAS)

# ==========================================================================
# 7. LangGraph Definition
# ==========================================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    trimmed = state["messages"][-10:]
    while trimmed and not isinstance(trimmed[0], HumanMessage):
        trimmed = trimmed[1:]
    
    messages = [system_prompt(active_session)] + trimmed
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(TOOL_SCHEMAS))

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile()

def menu_for_model() -> str:
    data = load_availability_and_specials()
    avail = data["availability"]
    lines = []
    for name, d in MENU_DETAILS.items():
        stock = "" if avail.get(name, True) else " | OUT OF STOCK"
        lines.append(f"- {name.title()} | ₹{d['price']} | {', '.join(d['tags'])}{stock}")
    return "\n".join(lines)

def system_prompt(s: Session) -> SystemMessage:
    data = load_availability_and_specials()
    specials = ", ".join(data["todays_specials"]) or "None"
    oos = [i.title() for i, ok in data["availability"].items() if not ok]
    return SystemMessage(content=(
        "You are 'Dakshin', the warm, upbeat host of 'Dakshin Delights', a South Indian "
        "restaurant. You greet guests by name when they share it, sound friendly and helpful, "
        "and answer EVERY request. You never say you 'cannot' show something — you use a tool.\n\n"
        "THE MENU BELOW IS YOUR ONLY SOURCE OF TRUTH:\n"
        f"{menu_for_model()}\n\n"
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
        "4. CART additions/removals: for any request by the customer to order, add, remove, or change items "
        "(e.g. 'one masala dosa', 'add idli', 'remove pongal'), you MUST call add_item / remove_item. "
        "Always pass the EXACT item name as requested by the customer (e.g., if they say 'podi dosa', pass "
        "'podi dosa'; if they say 'masala dosa', pass 'masala dosa'). Never translate, guess, or resolve "
        "generic or ambiguous item names to a specific menu item on your own. Always let the tool handle "
        "resolution, ambiguity, and stock status.\n"
        "5. BILLING: 'bill'/'checkout' -> show_bill. Then YES -> confirm_order(true); NO -> "
        "confirm_order(false).\n"
        "6. Only ever recommend or suggest AVAILABLE items when giving recommendations or answering menu queries. "
        "Never invent prices or totals.\n"
        "7. GENERAL / OFF-TOPIC questions: if someone asks something unrelated to the "
        "restaurant (trivia, advice, chit-chat), answer briefly and warmly if you can, then "
        "gently steer back — e.g. 'Happy to help! Now, can I get you anything from our menu?' "
        "Never be rude or refuse curtly; stay the gracious host."
    ))

def extract_text(msg) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") in (None, "text") and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "".join(parts).strip()
    return ""

def get_generic_prompt(text: str) -> Optional[Tuple[str, str]]:
    low = text.lower().strip()
    if any(w in low for w in ["menu", "bill", "receipt", "checkout", "hours", "location", "wifi", "parking", "payment"]):
        return None

    target = None
    if "masala dosa" in low or "maslaa dosa" in low or "masla dosa" in low:
        target = "masala dosa"
    elif "masala idli" in low or "maslaa idli" in low or "masla idli" in low:
        target = "masala idli"
    elif "podi dosa" in low or "pudi dosa" in low:
        target = "podi dosa"
    elif "podi idli" in low or "pudi idli" in low:
        target = "ghee podi idli"
    elif "podi" in low or "pudi" in low:
        target = "podi"
    elif "masala" in low or "maslaa" in low or "masla" in low:
        target = "masala"
    elif "dosa" in low or "dosas" in low or "dose" in low:
        target = "dosa"
    elif "idli" in low or "idly" in low or "idlis" in low:
        target = "idli"

    if target:
        cands = match_candidates(target)
        if len(cands) > 1:
            opts = "  ".join(f"• {c.title()}" for c in cands)
            prompt = (f"We have various options in '{target.title()}' — which one would you like?\n"
                      f"{opts}\n"
                      f"(Please specify the name of the one you want)")
            return target, prompt
    return None

# --- run turn ---
def run_turn(s: Session, user_text: str) -> str:
    global active_session
    active_session = s

    input_state = {"messages": s.messages + [HumanMessage(content=user_text)]}
    output_state = graph.invoke(input_state)

    new_messages = output_state["messages"][len(s.messages):]
    s.messages.extend(new_messages)

    tool_outputs = []
    cart_changed = False
    for msg in new_messages:
        if isinstance(msg, ToolMessage):
            tool_outputs.append(msg.content)
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            if any(tc["name"] in ("add_item", "remove_item") for tc in msg.tool_calls):
                cart_changed = True

    lead_in = ""
    final_msg = new_messages[-1] if new_messages else None
    if isinstance(final_msg, AIMessage):
        lead_in = extract_text(final_msg)
        if lead_in and ("₹" in lead_in or len(lead_in) > 220):
            lead_in = ""

    body = "\n\n".join(o for o in tool_outputs if o).strip()
    out = f"{lead_in}\n\n{body}".strip() if (lead_in and body) else (body or lead_in)

    # Automatically append generic options if user request was generic and options are not in output
    generic_res = get_generic_prompt(user_text)
    if generic_res:
        target, generic_prompt = generic_res
        cands = match_candidates(target)
        already_printed = f"various options in '{target}'".lower() in out.lower() or any(c.lower() in out.lower() for c in cands)
        if cands and not already_printed:
            out = f"{out.strip()}\n\n🤔  {generic_prompt}"

    if cart_changed:
        out += f"\n\n🛒  Your cart: {s.cart_summary()}"

    return out

# --- interactive loop ---
def stream_out(text: str, word_delay: float = 0.018):
    for tok in re.findall(r"\S+|\s+", text):
        sys.stdout.write(tok)
        sys.stdout.flush()
        if tok.strip():
            time.sleep(word_delay)
    sys.stdout.write("\n")
    sys.stdout.flush()

class TypingIndicator:
    def __init__(self, label="Waiter is typing"):
        self.label = label
        self._stop = threading.Event()
        self._t: Optional[threading.Thread] = None

    def _run(self):
        for dots in itertools.cycle([".", "..", "...", "   "]):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r💭  {self.label}{dots}   ")
            sys.stdout.flush()
            self._stop.wait(0.4)

    def __enter__(self):
        if sys.stdout.isatty():
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=1)
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()

def main():
    data = load_availability_and_specials()
    specials = ", ".join(data["todays_specials"]) or "None today"

    print("=" * 60)
    print("Welcome to Dakshin Delights! (AI Waiter via LangGraph)")
    print("=" * 60)
    print(f"Today: {datetime.now().strftime('%A, %B %d, %Y')}")
    print(f"🌟 TODAY'S SPECIALS: {specials} 🌟")
    print("=" * 60)
    print("Commands:  'menu' · 'history' · 'preferences' · 'cart' · 'exit'\n")

    s = Session()

    while True:
        try:
            user = input("\nYou: ").strip()
            if not user:
                continue
            low = user.lower()

            if low in ("exit", "quit"):
                print("Thank you for choosing Dakshin Delights! Goodbye! 👋")
                break
            if low == "menu":
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

            with TypingIndicator():
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
