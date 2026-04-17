"""In-memory shop state backing the synthetic 30-tool e-commerce suite.

A single `ShopState` holds products, the user, cart, orders, etc. All tools
mutate or read this object — no database, no network, deterministic.

Fixtures are seeded from a fixed seed so runs are reproducible across
mutation pairs; scenarios can opt into a different seed via the environment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


# -- Core records ------------------------------------------------------------


@dataclass
class Product:
    id: str
    name: str
    category: str
    price: float
    stock: int
    description: str = ""

    def view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "price": self.price,
            "in_stock": self.stock > 0,
            "description": self.description,
        }


@dataclass
class Address:
    id: str
    line1: str
    city: str
    state: str
    zip: str
    country: str = "US"

    def view(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class PaymentMethod:
    id: str
    brand: str
    last4: str
    expiry: str  # "MM/YY"

    def view(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class CartItem:
    product_id: str
    quantity: int


@dataclass
class Cart:
    items: dict[str, CartItem] = field(default_factory=dict)
    promo_code: str | None = None

    def view(self, state: "ShopState") -> dict[str, Any]:
        rows = []
        subtotal = 0.0
        for item in self.items.values():
            product = state.products.get(item.product_id)
            if product is None:
                continue
            line_total = product.price * item.quantity
            subtotal += line_total
            rows.append({
                "product_id": product.id,
                "name": product.name,
                "unit_price": product.price,
                "quantity": item.quantity,
                "line_total": round(line_total, 2),
            })
        discount = state.discount_for(self.promo_code, subtotal)
        total = round(subtotal - discount, 2)
        return {
            "items": rows,
            "subtotal": round(subtotal, 2),
            "promo_code": self.promo_code,
            "discount": round(discount, 2),
            "total": total,
        }


@dataclass
class Order:
    id: str
    items: list[CartItem]
    address_id: str
    payment_method_id: str
    status: str            # "placed", "shipped", "delivered", "cancelled", "returned"
    total: float
    tracking_number: str | None = None
    return_reason: str | None = None

    def view(self, state: "ShopState") -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "total": self.total,
            "address_id": self.address_id,
            "payment_method_id": self.payment_method_id,
            "tracking_number": self.tracking_number,
            "items": [
                {
                    "product_id": it.product_id,
                    "quantity": it.quantity,
                    "name": state.products[it.product_id].name
                    if it.product_id in state.products
                    else None,
                }
                for it in self.items
            ],
        }


@dataclass
class Review:
    product_id: str
    rating: int
    text: str
    author: str


@dataclass
class Ticket:
    id: str
    subject: str
    body: str
    status: str = "open"   # "open", "closed"


@dataclass
class UserProfile:
    user_id: str
    name: str
    email: str
    phone: str = ""


# -- Aggregate state ---------------------------------------------------------


@dataclass
class ShopState:
    user: UserProfile
    products: dict[str, Product] = field(default_factory=dict)
    categories: list[str] = field(default_factory=list)
    addresses: dict[str, Address] = field(default_factory=dict)
    payment_methods: dict[str, PaymentMethod] = field(default_factory=dict)
    cart: Cart = field(default_factory=Cart)
    orders: dict[str, Order] = field(default_factory=dict)
    reviews: list[Review] = field(default_factory=list)
    tickets: dict[str, Ticket] = field(default_factory=dict)
    promo_codes: dict[str, float] = field(default_factory=dict)  # code -> fraction off
    help_articles: dict[str, str] = field(default_factory=dict)

    # -- Helpers used by tools ---------------------------------------------

    def discount_for(self, code: str | None, subtotal: float) -> float:
        if not code:
            return 0.0
        frac = self.promo_codes.get(code, 0.0)
        return round(subtotal * frac, 2)

    def next_order_id(self) -> str:
        return f"O{1000 + len(self.orders)}"

    def next_ticket_id(self) -> str:
        return f"T{100 + len(self.tickets)}"

    def next_address_id(self) -> str:
        return f"A{10 + len(self.addresses)}"

    def next_payment_id(self) -> str:
        return f"P{10 + len(self.payment_methods)}"


# -- Fixture -----------------------------------------------------------------


def default_fixture(seed: int = 0) -> ShopState:
    """Build a deterministic shop with ~20 products across 5 categories."""
    rng = random.Random(seed)

    state = ShopState(
        user=UserProfile(
            user_id="U1",
            name="Alex Morgan",
            email="alex.morgan@example.com",
            phone="+1-555-0100",
        )
    )

    categories = ["electronics", "books", "home", "clothing", "kitchen"]
    state.categories = list(categories)

    products_by_cat = {
        "electronics": [
            ("USB-C Cable 2m", 12.99),
            ("Wireless Mouse", 24.50),
            ("Bluetooth Headphones", 79.00),
            ("Portable SSD 1TB", 109.99),
        ],
        "books": [
            ("The Pragmatic Programmer", 32.00),
            ("Designing Data-Intensive Applications", 45.00),
            ("Clean Code", 28.50),
            ("Structure and Interpretation of Computer Programs", 55.00),
        ],
        "home": [
            ("LED Desk Lamp", 35.00),
            ("Air Purifier", 129.00),
            ("Throw Blanket", 39.99),
        ],
        "clothing": [
            ("Cotton T-Shirt", 18.00),
            ("Denim Jacket", 85.00),
            ("Running Socks 3-pack", 14.50),
        ],
        "kitchen": [
            ("French Press", 27.00),
            ("Chef's Knife", 65.00),
            ("Measuring Cups", 12.00),
            ("Cast Iron Skillet", 48.00),
            ("Silicone Spatula", 8.50),
            ("Electric Kettle", 39.00),
        ],
    }

    idx = 1
    for cat, items in products_by_cat.items():
        for name, price in items:
            pid = f"P{idx:03d}"
            idx += 1
            stock = rng.randint(0, 20)
            state.products[pid] = Product(
                id=pid,
                name=name,
                category=cat,
                price=price,
                stock=stock,
                description=f"{name} ({cat}).",
            )

    state.addresses["A1"] = Address(
        id="A1",
        line1="100 Main St",
        city="Seattle",
        state="WA",
        zip="98101",
    )
    state.payment_methods["P1"] = PaymentMethod(
        id="P1",
        brand="visa",
        last4="4242",
        expiry="12/28",
    )

    state.promo_codes = {"SAVE10": 0.10, "WELCOME": 0.15}

    state.reviews = [
        Review(product_id="P001", rating=5, text="Works great!", author="customer1"),
        Review(product_id="P003", rating=4, text="Good sound for the price.", author="customer2"),
        Review(product_id="P009", rating=5, text="Lamp is sturdy.", author="customer3"),
    ]

    state.help_articles = {
        "return_policy": "Items can be returned within 30 days of delivery.",
        "shipping_times": "Standard shipping arrives in 3-5 business days.",
        "refunds": "Refunds are issued to the original payment method within 5-7 days.",
    }

    return state
