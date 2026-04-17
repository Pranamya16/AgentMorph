"""Payment tools: list, add, checkout."""

from __future__ import annotations

from typing import Any

from agentmorph.tools.base import Tool
from agentmorph.tools.ecommerce.state import Order, PaymentMethod, ShopState


def build(state: ShopState) -> list[Tool]:
    def list_payment_methods() -> list[dict[str, Any]]:
        return [pm.view() for pm in state.payment_methods.values()]

    def add_payment_method(brand: str, last4: str, expiry: str) -> dict[str, Any]:
        if len(last4) != 4 or not last4.isdigit():
            raise ValueError("last4 must be exactly 4 digits")
        pid = state.next_payment_id()
        pm = PaymentMethod(id=pid, brand=brand, last4=last4, expiry=expiry)
        state.payment_methods[pid] = pm
        return pm.view()

    def checkout(address_id: str, payment_method_id: str) -> dict[str, Any]:
        if address_id not in state.addresses:
            raise KeyError(f"unknown address {address_id}")
        if payment_method_id not in state.payment_methods:
            raise KeyError(f"unknown payment method {payment_method_id}")
        if not state.cart.items:
            raise ValueError("cart is empty")

        snapshot = state.cart.view(state)
        order_id = state.next_order_id()
        order = Order(
            id=order_id,
            items=list(state.cart.items.values()),
            address_id=address_id,
            payment_method_id=payment_method_id,
            status="placed",
            total=snapshot["total"],
            tracking_number=None,
        )
        state.orders[order_id] = order

        # Deduct stock — not strictly required but makes tools more realistic.
        for it in order.items:
            p = state.products.get(it.product_id)
            if p is not None:
                p.stock = max(0, p.stock - it.quantity)

        # Clear the cart.
        state.cart.items.clear()
        state.cart.promo_code = None

        return order.view(state)

    return [
        Tool(
            name="list_payment_methods",
            description="List the saved payment methods on the user's account.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            func=list_payment_methods,
            read_only=True,
            category="payments",
        ),
        Tool(
            name="add_payment_method",
            description="Add a new card to the user's account.",
            parameters={
                "type": "object",
                "properties": {
                    "brand": {"type": "string"},
                    "last4": {"type": "string", "minLength": 4, "maxLength": 4},
                    "expiry": {"type": "string", "description": "MM/YY"},
                },
                "required": ["brand", "last4", "expiry"],
                "additionalProperties": False,
            },
            func=add_payment_method,
            read_only=False,
            category="payments",
        ),
        Tool(
            name="checkout",
            description=(
                "Place an order for everything currently in the cart, shipping to "
                "`address_id` and charging `payment_method_id`. Clears the cart."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "address_id": {"type": "string"},
                    "payment_method_id": {"type": "string"},
                },
                "required": ["address_id", "payment_method_id"],
                "additionalProperties": False,
            },
            func=checkout,
            read_only=False,
            category="payments",
        ),
    ]
