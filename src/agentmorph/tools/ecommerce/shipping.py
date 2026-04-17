"""Shipping tools: tracking, estimates, options."""

from __future__ import annotations

from typing import Any

from agentmorph.tools.base import Tool
from agentmorph.tools.ecommerce.state import ShopState


def build(state: ShopState) -> list[Tool]:
    def track_shipment(order_id: str) -> dict[str, Any]:
        order = state.orders.get(order_id)
        if order is None:
            raise KeyError(f"no order {order_id}")
        if order.tracking_number is None:
            # Deterministic tracking number for newly-placed orders.
            order.tracking_number = f"TRK-{order.id}-01"
            if order.status == "placed":
                order.status = "shipped"
        return {
            "order_id": order.id,
            "status": order.status,
            "tracking_number": order.tracking_number,
            "eta_days": 3 if order.status == "shipped" else 0,
        }

    def estimate_shipping(address_id: str, product_id: str) -> dict[str, Any]:
        if address_id not in state.addresses:
            raise KeyError(f"unknown address {address_id}")
        if product_id not in state.products:
            raise KeyError(f"no product {product_id}")
        # Simple deterministic rule: price scales with product weight, mocked as
        # 5% of product price, clamped into a reasonable range.
        price = state.products[product_id].price
        fee = max(3.99, min(19.99, round(price * 0.05, 2)))
        return {"address_id": address_id, "product_id": product_id, "fee": fee, "eta_days": 4}

    def list_shipping_options(address_id: str) -> list[dict[str, Any]]:
        if address_id not in state.addresses:
            raise KeyError(f"unknown address {address_id}")
        return [
            {"option": "standard", "fee": 4.99, "eta_days": 5},
            {"option": "express", "fee": 12.99, "eta_days": 2},
            {"option": "overnight", "fee": 24.99, "eta_days": 1},
        ]

    return [
        Tool(
            name="track_shipment",
            description="Return the tracking number and status of an order.",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
            func=track_shipment,
            read_only=True,
            category="shipping",
        ),
        Tool(
            name="estimate_shipping",
            description="Estimate shipping cost for one product to one address.",
            parameters={
                "type": "object",
                "properties": {
                    "address_id": {"type": "string"},
                    "product_id": {"type": "string"},
                },
                "required": ["address_id", "product_id"],
                "additionalProperties": False,
            },
            func=estimate_shipping,
            read_only=True,
            category="shipping",
        ),
        Tool(
            name="list_shipping_options",
            description="List the standard / express / overnight shipping options.",
            parameters={
                "type": "object",
                "properties": {"address_id": {"type": "string"}},
                "required": ["address_id"],
                "additionalProperties": False,
            },
            func=list_shipping_options,
            read_only=True,
            category="shipping",
        ),
    ]
