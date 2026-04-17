"""Cart tools: view, add, update, remove, apply promo code."""

from __future__ import annotations

from typing import Any

from agentmorph.tools.base import Tool
from agentmorph.tools.ecommerce.state import CartItem, ShopState


def build(state: ShopState) -> list[Tool]:
    def view_cart() -> dict[str, Any]:
        return state.cart.view(state)

    def add_to_cart(product_id: str, quantity: int = 1) -> dict[str, Any]:
        if product_id not in state.products:
            raise KeyError(f"no product {product_id}")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        existing = state.cart.items.get(product_id)
        if existing:
            existing.quantity += quantity
        else:
            state.cart.items[product_id] = CartItem(product_id=product_id, quantity=quantity)
        return state.cart.view(state)

    def update_cart_item(product_id: str, quantity: int) -> dict[str, Any]:
        if product_id not in state.cart.items:
            raise KeyError(f"{product_id} not in cart")
        if quantity <= 0:
            state.cart.items.pop(product_id)
        else:
            state.cart.items[product_id].quantity = quantity
        return state.cart.view(state)

    def remove_from_cart(product_id: str) -> dict[str, Any]:
        state.cart.items.pop(product_id, None)
        return state.cart.view(state)

    def apply_promo_code(code: str) -> dict[str, Any]:
        if code not in state.promo_codes:
            raise ValueError(f"invalid promo code: {code}")
        state.cart.promo_code = code
        return state.cart.view(state)

    return [
        Tool(
            name="view_cart",
            description="Show the current contents of the user's cart, with subtotal, discount, and total.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            func=view_cart,
            read_only=True,
            category="cart",
        ),
        Tool(
            name="add_to_cart",
            description="Add `quantity` units of `product_id` to the cart. Stacks with existing quantity.",
            parameters={
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "quantity": {"type": "integer", "minimum": 1, "default": 1},
                },
                "required": ["product_id"],
                "additionalProperties": False,
            },
            func=add_to_cart,
            read_only=False,
            category="cart",
        ),
        Tool(
            name="update_cart_item",
            description="Set the quantity of a cart line. A quantity of 0 removes the line.",
            parameters={
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "quantity": {"type": "integer", "minimum": 0},
                },
                "required": ["product_id", "quantity"],
                "additionalProperties": False,
            },
            func=update_cart_item,
            read_only=False,
            category="cart",
        ),
        Tool(
            name="remove_from_cart",
            description="Remove a product entirely from the cart. Idempotent.",
            parameters={
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
                "additionalProperties": False,
            },
            func=remove_from_cart,
            read_only=False,
            category="cart",
        ),
        Tool(
            name="apply_promo_code",
            description="Apply a promo code to the cart. Returns the updated totals.",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
            func=apply_promo_code,
            read_only=False,
            category="cart",
        ),
    ]
