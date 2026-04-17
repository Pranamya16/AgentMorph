"""Order tools: list, detail, cancel, return, reorder."""

from __future__ import annotations

from typing import Any

from agentmorph.tools.base import Tool
from agentmorph.tools.ecommerce.state import CartItem, Order, ShopState


def build(state: ShopState) -> list[Tool]:
    def list_orders(status: str | None = None) -> list[dict[str, Any]]:
        out = []
        for order in state.orders.values():
            if status and order.status != status:
                continue
            out.append(order.view(state))
        return out

    def get_order(order_id: str) -> dict[str, Any]:
        order = state.orders.get(order_id)
        if order is None:
            raise KeyError(f"no order {order_id}")
        return order.view(state)

    def cancel_order(order_id: str) -> dict[str, Any]:
        order = state.orders.get(order_id)
        if order is None:
            raise KeyError(f"no order {order_id}")
        if order.status in ("delivered", "returned"):
            raise ValueError(f"cannot cancel an order in state {order.status!r}")
        order.status = "cancelled"
        return order.view(state)

    def request_return(order_id: str, reason: str) -> dict[str, Any]:
        order = state.orders.get(order_id)
        if order is None:
            raise KeyError(f"no order {order_id}")
        if order.status != "delivered":
            raise ValueError("can only return delivered orders")
        order.status = "returned"
        order.return_reason = reason
        return order.view(state)

    def reorder(order_id: str) -> dict[str, Any]:
        order = state.orders.get(order_id)
        if order is None:
            raise KeyError(f"no order {order_id}")
        # Append order items back into the cart.
        for it in order.items:
            existing = state.cart.items.get(it.product_id)
            if existing:
                existing.quantity += it.quantity
            else:
                state.cart.items[it.product_id] = CartItem(
                    product_id=it.product_id, quantity=it.quantity
                )
        return state.cart.view(state)

    return [
        Tool(
            name="list_orders",
            description="List the user's past orders, optionally filtered by status.",
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["placed", "shipped", "delivered", "cancelled", "returned"],
                    }
                },
                "additionalProperties": False,
            },
            func=list_orders,
            read_only=True,
            category="orders",
        ),
        Tool(
            name="get_order",
            description="Fetch full details for one order by id.",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
            func=get_order,
            read_only=True,
            category="orders",
        ),
        Tool(
            name="cancel_order",
            description="Cancel an order that has not yet been delivered.",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
            func=cancel_order,
            read_only=False,
            category="orders",
        ),
        Tool(
            name="request_return",
            description="Request a return for a delivered order, with a free-text reason.",
            parameters={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id", "reason"],
                "additionalProperties": False,
            },
            func=request_return,
            read_only=False,
            category="orders",
        ),
        Tool(
            name="reorder",
            description="Copy a past order's items into the current cart.",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
            func=reorder,
            read_only=False,
            category="orders",
        ),
    ]
