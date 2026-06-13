"""
Dakshin Delights — Fast single-agent restaurant assistant.

ONE ReAct-style agent (no multi-agent graph). All tools are bound to a single
Gemini model and run in a tight loop; all deterministic work (menu, categories,
cart, GST, receipts, recommendations) is pure Python with no LLM round-trips.

This version fixes "no category / full menu" dead-ends and adds proper, curated
recommendations so every query is answerable.
"""

import os
import json
import math
import uuid
from typing import Dict, List, Optional
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


# ==========================================================================
# 2. Availability / specials
# ==========================================================================
def load_availability_and_specials() -> dict:
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
        except Exception as e:
            print(f"Error creating availability file: {e}")
        return default
    try:
        with open(AVAILABILITY_FILE, "r") as f:
            data = json.load(f)
        data.setdefault("availability", {})
        data.setdefault("todays_specials", [])
        for item in MENU_DETAILS:
            data["availability"].setdefault(item, True)
        return data
    except Exception as e:
        print(f"Error reading availability file, using default: {e}")
        return default


# ==========================================================================
# 3. Lookup / category helpers
# ==========================================================================
def find_item(user_input: str) -> Optional[str]:
    """Resolve free text to an exact menu key: exact -> substring -> fuzzy."""
    if not user_input:
        return None
    s = user_input.lower().strip()
    if s in MENU_DETAILS:
        return s
    for k in MENU_DETAILS:
        if s == k or s in k:
            return k
    return fuzzy_best(s, list(MENU_DETAILS.keys()), 72)


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


def parse_preference(text: str) -> dict:
    """Turn free text ('no onion', 'vegan', 'mild') into concrete filter rules."""
    t = f" {(text or '').lower()} "
    require, exclude_tags, exclude_words, notes = set(), set(), set(), []

    onion = any(p in t for p in _NO_ONION)
    garlic = any(p in t for p in _NO_GARLIC)
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
    if ("vegetarian" in t or " veg " in t) and "vegan" not in t:
        require.add("vegetarian")
    if any(p in t for p in _NO_SPICE):
        exclude_tags.add("spicy")
    elif "spicy" in t or "spice" in t:
        require.add("spicy")

    return {"require": require, "exclude_tags": exclude_tags,
            "exclude_words": exclude_words, "notes": notes,
            "active": bool(require or exclude_tags or exclude_words)}


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

    lines: List[str] = []

    # Budget combo
    if budget:
        budget = float(budget)
        pool = sorted(usable(list(MENU_DETAILS.keys())), key=lambda i: MENU_DETAILS[i]["price"])
        combo, total = [], 0.0
        mains = [i for i in pool if i in DOSA_ITEMS + RICE_ITEMS and MENU_DETAILS[i]["price"] <= budget]
        if mains:
            m = mains[-1]
            combo.append(m); total += MENU_DETAILS[m]["price"]
        for e in [i for i in pool if i in DESSERT_ITEMS + DRINK_ITEMS]:
            if total + MENU_DETAILS[e]["price"] <= budget:
                combo.append(e); total += MENU_DETAILS[e]["price"]; break
        for i in pool:
            if i in combo:
                continue
            if total + MENU_DETAILS[i]["price"] <= budget and len(combo) < 3:
                combo.append(i); total += MENU_DETAILS[i]["price"]
        if combo:
            lines.append(f"Within ₹{int(budget)}: "
                         + " + ".join(f"{c.title()} (₹{MENU_DETAILS[c]['price']})" for c in combo)
                         + f"  =  ₹{int(total)}")

    # Group bundle
    if group_size:
        n = max(1, int(group_size))
        base = usable(DOSA_ITEMS) or usable(IDLI_ITEMS)
        sweet = usable(DESSERT_ITEMS)
        drink = usable(DRINK_ITEMS)
        parts = []
        if base:
            parts.append(f"{n} x {base[0].title()}")
        if sweet:
            parts.append(f"{max(1, n // 2)} x {sweet[0].title()}")
        if drink:
            parts.append(f"{n} x {drink[0].title()}")
        if parts:
            lines.append(f"For a group of {n}: " + ", ".join(parts))

    # Pairing for the current cart
    if cart and not budget and not group_size:
        sweet = [i for i in usable(DESSERT_ITEMS) if i not in cart]
        drink = [i for i in usable(DRINK_ITEMS) if i not in cart]
        pair = []
        if drink:
            pair.append(f"{drink[0].title()} (₹{MENU_DETAILS[drink[0]]['price']})")
        if sweet:
            pair.append(f"{sweet[0].title()} (₹{MENU_DETAILS[sweet[0]]['price']})")
        if pair:
            lines.append("Pairs perfectly with your order: " + " and ".join(pair))

    # Default: specials + favourites + a few more — ALL preference-filtered
    if not lines:
        sp_keys = [find_item(x) for x in specials]
        sp_ok = [k for k in sp_keys if k and avail.get(k, True) and item_matches(k, parsed)]
        if sp_ok:
            lines.append("Today's specials: "
                         + ", ".join(f"{k.title()} (₹{MENU_DETAILS[k]['price']})" for k in sp_ok))
        fav = usable(POPULAR_ITEMS)[:3]
        if fav:
            lines.append("Customer favourites: "
                         + ", ".join(f"{p.title()} (₹{MENU_DETAILS[p]['price']})" for p in fav))
        chosen = set(sp_ok) | set(fav)
        more = [i for i in usable(list(MENU_DETAILS.keys())) if i not in chosen][:4]
        if more and len(chosen) < 3:
            lines.append("You might also like: "
                         + ", ".join(f"{m.title()} (₹{MENU_DETAILS[m]['price']})" for m in more))

    if not lines:
        return ("🤔  I couldn't find anything matching that preference right now. "
                "Would you like to relax it, or see the full menu?")

    if parsed["notes"]:
        lines.append(f"({'; '.join(parsed['notes'])}.)")
    return "✨  Here are my picks for you:\n\n" + "\n".join(lines)


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
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
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
        "6. Only ever suggest AVAILABLE items. Never invent prices or totals."
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
    print("Welcome to Dakshin Delights! (Fast Single-Agent Assistant)")
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
                print("Thank you for choosing Dakshin Delights! Goodbye!")
                break
            if low == "menu":                       # instant full menu, no LLM
                print("\nWaiter:\n" + render_full_menu())
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

            reply = run_turn(s, user)
            print(f"\nWaiter: {reply}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()