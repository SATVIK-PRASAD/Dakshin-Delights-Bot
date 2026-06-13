"""
Dakshin Delights — Fast single-agent restaurant assistant.

Approach: ONE ReAct-style agent (not a multi-agent graph). All tools are bound
to a single Gemini model and executed in a tight loop. Deterministic work
(cart, GST, receipts, availability) runs in pure Python with no LLM round-trips.
A normal turn = 1 LLM call; a tool turn = 1 call to choose the tool + 1 short
call to reply. No triage call, no graph routing -> much lower latency.
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

# Fuzzy matching: rapidfuzz if available, else difflib fallback
try:
    from rapidfuzz import process

    def fuzzy_best(query, choices, cutoff=70):
        m = process.extractOne(query, choices, score_cutoff=cutoff)
        return m[0] if m else None
except ImportError:
    from difflib import get_close_matches

    def fuzzy_best(query, choices, cutoff=70):
        m = get_close_matches(query, choices, n=1, cutoff=cutoff / 100)
        return m[0] if m else None

load_dotenv()

# ==========================================================================
# 1. Menu data
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

IDLI_ITEMS = [k for k in MENU_DETAILS if ("idli" in k or "idiyappam" in k) and "dosa" not in k]
DOSA_ITEMS = [k for k in MENU_DETAILS if "dosa" in k]
DESSERT_DRINK_ITEMS = ["custard apple with pulp", "special falooda", "gud bud", "badam milk", "gulab jamun"]

AVAILABILITY_FILE = "item_availability.json"
ANALYTICS_FILE = "analytics_dashboard.json"

# ==========================================================================
# 2. Availability / specials / search helpers
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


def category_for(item_name: str) -> List[str]:
    item_name = item_name.lower().strip()
    if "dosa" in item_name:
        return DOSA_ITEMS
    if "idli" in item_name or "idiyappam" in item_name:
        return IDLI_ITEMS
    if item_name in DESSERT_DRINK_ITEMS:
        return DESSERT_DRINK_ITEMS
    return [k for k in MENU_DETAILS
            if k not in IDLI_ITEMS and k not in DOSA_ITEMS and k not in DESSERT_DRINK_ITEMS]


def similar_available(item_name: str) -> List[str]:
    avail = load_availability_and_specials()["availability"]
    return [x for x in category_for(item_name)
            if x != item_name.lower().strip() and avail.get(x, True)][:3]


def find_item(user_input: str) -> Optional[str]:
    """Resolve free text to an exact menu key: exact -> substring -> fuzzy."""
    if not user_input:
        return None
    s = user_input.lower().strip()
    if s in MENU_DETAILS:
        return s
    # exact substring either direction (e.g. "lemon rice" -> "lemon rice")
    for k in MENU_DETAILS:
        if s == k or s in k:
            return k
    return fuzzy_best(s, list(MENU_DETAILS.keys()), 72)


def suggest_items(user_input: str, n: int = 3) -> List[str]:
    """Best-effort 'did you mean' list based on shared words."""
    s = (user_input or "").lower().replace("(", " ").replace(")", " ")
    toks = [t for t in s.split() if len(t) > 2]
    scored = []
    for k in MENU_DETAILS:
        overlap = sum(1 for t in toks if t in k)
        if overlap:
            scored.append((overlap, k))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [k for _, k in scored[:n]]


def menu_search(query="", max_price=None, preference_tag=None) -> str:
    avail = load_availability_and_specials()["availability"]
    q = (query or "").lower().strip()
    pref = (preference_tag or "").lower().strip()
    rows = []
    for name, d in MENU_DETAILS.items():
        if q and q not in name:
            continue
        if max_price is not None and d["price"] > max_price:
            continue
        if pref and pref not in d["tags"]:
            continue
        status = "Available" if avail.get(name, True) else "Not Available"
        rows.append(f"- {name.title()}: ₹{d['price']} [{status}] (Dietary: {', '.join(d['tags'])})")
    return "\n".join(rows) if rows else "No items match your criteria. Try adjusting your constraints."


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
# 3. Session state (plain Python — no graph needed)
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
# 4. Tool SCHEMAS (for binding) + local HANDLERS (for execution)
# ==========================================================================
@tool
def search_menu(query: str = "", max_price: Optional[float] = None,
                preference_tag: Optional[str] = None) -> str:
    """Search the menu. Filter by a text query, a max price (e.g. 200), and/or a dietary tag (vegan, jain, spicy, gluten-free, vegetarian)."""


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
    """Confirm (True) and send the order to the kitchen, or cancel the draft (False). Only after show_bill."""


@tool
def view_analytics() -> str:
    """Show business analytics: total orders, revenue, average order value, most popular item."""


TOOL_SCHEMAS = [search_menu, add_item, remove_item, set_preference, show_bill, confirm_order, view_analytics]

# Outputs from these tools are the deliverable -> printed verbatim (never paraphrased).
VERBATIM_TOOLS = {"show_bill", "view_analytics"}


def handle_tool(name: str, args: dict, s: Session) -> str:
    avail = load_availability_and_specials()["availability"]

    if name == "search_menu":
        return menu_search(args.get("query", ""), args.get("max_price"), args.get("preference_tag"))

    if name == "set_preference":
        s.preference = (args.get("preference", "") or "").lower().strip()
        return f"Preference saved: '{s.preference}'."

    if name == "add_item":
        item = find_item(args.get("item_name", ""))
        qty = max(1, int(args.get("quantity", 1) or 1))
        if not item:
            sugg = suggest_items(args.get("item_name", ""))
            tip = ("\nDid you mean: " + ", ".join(x.title() for x in sugg) + "?") if sugg else ""
            return (f"NOT ADDED: '{args.get('item_name', '')}' is not on our menu.{tip}")
        if not avail.get(item, True):
            alts = similar_available(item)
            tip = ("\nTry instead:\n" + "\n".join(f"- {a.title()}" for a in alts)) if alts else ""
            return f"NOT ADDED: '{item.title()}' is currently unavailable.{tip}"
        if item == "poori" and datetime.now().hour < 19:
            return f"NOT ADDED: Poori is only served after 7:00 PM (now {datetime.now().strftime('%I:%M %p')})."
        s.cart[item] = s.cart.get(item, 0) + qty
        s.history.append(f"Added {qty} x {item.title()}")
        s.awaiting_confirmation = False
        return f"ADDED {qty} x {item.title()}. Cart is now: {s.cart_summary()}."

    if name == "remove_item":
        item = find_item(args.get("item_name", ""))
        qty = max(1, int(args.get("quantity", 1) or 1))
        if not item or item not in s.cart:
            return f"Error: '{args.get('item_name', '')}' is not in your cart."
        if s.cart[item] <= qty:
            del s.cart[item]
            s.history.append(f"Removed all {item.title()}")
            msg = f"Removed all {item.title()}."
        else:
            s.cart[item] -= qty
            s.history.append(f"Removed {qty} x {item.title()}")
            msg = f"Removed {qty} x {item.title()}."
        s.awaiting_confirmation = False
        return f"{msg} Cart: {s.cart_summary()}."

    if name == "show_bill":
        # Defensive: only keep real, priceable items (never silently lose a valid one)
        s.cart = {i: q for i, q in s.cart.items() if i in MENU_DETAILS}
        if not s.cart:
            return "Your cart is empty — add some items first."
        lines, subtotal = [], 0.0
        for i, q in s.cart.items():
            line = MENU_DETAILS[i]["price"] * q
            subtotal += line
            lines.append(f"  • {q}x {i.title()} (₹{MENU_DETAILS[i]['price']} each) -> ₹{line:.2f}")
        gst = subtotal * 0.05
        total = subtotal + gst
        s.awaiting_confirmation = True
        return (
            "====================================\n"
            "         DRAFT ORDER RECEIPT        \n"
            "====================================\n"
            + "\n".join(lines) +
            "\n------------------------------------\n"
            f"  Subtotal:     ₹{subtotal:.2f}\n"
            f"  GST (5%):     ₹{gst:.2f}\n"
            f"  Grand Total:  ₹{total:.2f}\n"
            f"  Rounded Bill: ₹{math.floor(total)}\n"
            "====================================\n"
            "Reply YES to confirm, or NO to keep editing."
        )

    if name == "confirm_order":
        if not args.get("confirm", True):
            s.awaiting_confirmation = False
            return "Order cancelled. You can keep editing your cart."
        s.cart = {i: q for i, q in s.cart.items() if i in MENU_DETAILS}
        if not s.cart:
            return "Error: cannot confirm an empty order."
        subtotal = sum(MENU_DETAILS[i]["price"] * q for i, q in s.cart.items())
        total = math.floor(subtotal + subtotal * 0.05)
        save_order(dict(s.cart), total)
        s.cart.clear()
        s.history.clear()
        s.awaiting_confirmation = False
        return f"SUCCESS: Order confirmed and sent to the kitchen! Total ₹{total}. Thank you!"

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
                "=== DAKSHIN DELIGHTS ANALYTICS ===\n"
                f"• Total Orders:        {len(orders)}\n"
                f"• Total Revenue:       ₹{revenue:.2f}\n"
                f"• Average Order Value: ₹{revenue / len(orders):.2f}\n"
                f"• Most Popular Item:   {top.title()} ({counts.get(top, 0)} sold)\n"
                "=================================="
            )
        except Exception as e:
            return f"Error compiling analytics: {e}"

    return f"Unknown tool '{name}'."


# ==========================================================================
# 5. The single agent
# ==========================================================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_with_tools = llm.bind_tools(TOOL_SCHEMAS)


def system_prompt(s: Session) -> SystemMessage:
    data = load_availability_and_specials()
    specials = ", ".join(data["todays_specials"]) or "None"
    oos = [i.title() for i, ok in data["availability"].items() if not ok]
    return SystemMessage(content=(
        "You are the host & order assistant for 'Dakshin Delights', a South Indian restaurant. "
        "You handle greetings, menu questions, recommendations, cart edits, billing and order confirmation.\n\n"
        f"LIVE CART (ground truth): {s.cart_summary()}\n"
        f"Saved dietary preference: {s.preference or 'None'}\n"
        f"Awaiting bill confirmation: {s.awaiting_confirmation}\n"
        f"Today's specials (promote these): {specials}\n"
        f"Out of stock (refuse + suggest same-category alternatives): {', '.join(oos) or 'None'}\n\n"
        "Rules:\n"
        "1. Use search_menu for any lookup, price, budget ('under ₹200'), or dietary filter; honour the saved preference.\n"
        "2. To change the cart you MUST call add_item / remove_item. NEVER say an item was added, removed, "
        "or that a quantity changed unless you called the tool THIS turn and its result starts with 'ADDED'/'Removed'. "
        "If a tool result starts with 'NOT ADDED', tell the customer it was not added and why.\n"
        "3. NEVER list or invent the full cart contents yourself — the system appends the real cart automatically. "
        "Only confirm the specific change you just made. Treat the LIVE CART above as the single source of truth.\n"
        "4. For checkout/'bill', call show_bill. When the customer then says YES, call confirm_order(true); "
        "if NO, call confirm_order(false).\n"
        "5. Recommend pairings (Badam Milk ₹50, Gulab Jamun ₹40) and only ever suggest AVAILABLE items.\n"
        "6. Be warm, concise and accurate. Never invent prices or totals — rely on tool output."
    ))


def extract_text(msg) -> str:
    """Pull ONLY the human-readable text out of a Gemini reply.

    Gemini's content can be a plain string OR a list of content blocks (a text
    part plus thinking/tool_use parts that carry a 'signature'/'extras' field).
    Doing str(content) on the list dumps that signature metadata into the chat,
    so we keep text blocks only and discard everything else.
    """
    content = msg.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                # Keep real text; skip 'thinking', 'tool_use', signatures, extras.
                if p.get("type") in (None, "text") and isinstance(p.get("text"), str):
                    parts.append(p["text"])
        return "".join(parts).strip()
    return ""


def run_turn(s: Session, user_text: str, max_steps: int = 4) -> str:
    """One conversational turn. Returns the assistant's reply text."""
    s.messages.append(HumanMessage(content=user_text))
    verbatim_chunks: List[str] = []
    cart_changed = False  # did add_item / remove_item run this turn?

    for _ in range(max_steps):
        ai: AIMessage = llm_with_tools.invoke([system_prompt(s)] + s.messages)
        s.messages.append(ai)

        if not getattr(ai, "tool_calls", None):
            text = extract_text(ai)
            reply = "\n\n".join(verbatim_chunks + ([text] if text else [])).strip()
            if not reply:
                reply = verbatim_chunks[-1] if verbatim_chunks else "How can I help?"
            # Authoritative cart line so the model can never misreport contents.
            claims_cart = any(w in reply.lower()
                              for w in ("cart", "added", "removed", "updated quantity"))
            if cart_changed or claims_cart:
                reply += f"\n\n🛒 Your cart: {s.cart_summary()}"
            return reply

        for tc in ai.tool_calls:
            result = handle_tool(tc["name"], tc.get("args", {}), s)
            if tc["name"] in ("add_item", "remove_item"):
                cart_changed = True
            if tc["name"] in VERBATIM_TOOLS:
                verbatim_chunks.append(result)
            s.messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    # Safety net if the model keeps looping
    tail = f"\n\n🛒 Your cart: {s.cart_summary()}" if cart_changed else ""
    return ("\n\n".join(verbatim_chunks)).strip() + tail or "Sorry, could you rephrase that?"


# ==========================================================================
# 6. Interactive loop
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
    print("Commands:  'history' · 'preferences' · 'cart' · 'exit'\n")

    s = Session()
    _ = uuid.uuid4().hex  # session id (kept for parity/logging if needed)

    while True:
        try:
            user = input("\nYou: ").strip()
            if not user:
                continue
            low = user.lower()

            # Pure-local commands — zero LLM calls
            if low in ("exit", "quit"):
                print("Thank you for choosing Dakshin Delights! Goodbye!")
                break
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