"""Tests for the synthetic 30-tool e-commerce suite.

These must pass with zero framework deps installed — they exercise only the
pure-Python tool implementations.
"""

from __future__ import annotations

import pytest

from agentmorph.tools.ecommerce import (
    ECOMMERCE_TOOL_NAMES,
    build_ecommerce_registry,
)


def test_exactly_thirty_tools() -> None:
    assert len(ECOMMERCE_TOOL_NAMES) == 30
    state, reg = build_ecommerce_registry()
    assert tuple(reg.names()) == ECOMMERCE_TOOL_NAMES
    assert len(reg) == 30


def test_all_tools_have_object_schema() -> None:
    _state, reg = build_ecommerce_registry()
    for tool in reg:
        assert tool.parameters.get("type") == "object", tool.name


def test_read_only_flags_are_reasonable() -> None:
    _state, reg = build_ecommerce_registry()
    expected_read_only = {
        "search_products", "get_product", "list_categories", "check_stock",
        "get_recommendations", "get_product_reviews", "view_cart",
        "get_user_profile", "list_addresses", "list_orders", "get_order",
        "list_payment_methods", "track_shipment", "estimate_shipping",
        "list_shipping_options", "search_help", "get_ticket",
    }
    for tool in reg:
        assert (tool.name in expected_read_only) == tool.read_only, tool.name


def test_shopping_flow_end_to_end() -> None:
    state, reg = build_ecommerce_registry()

    # 1. Search kitchen, add a product, view cart, apply promo, checkout.
    hits = reg.call("search_products", {"query": "kettle", "category": "kitchen"})
    assert hits.ok and hits.output
    kettle_id = hits.output[0]["id"]

    assert reg.call("add_to_cart", {"product_id": kettle_id, "quantity": 2}).ok

    cart = reg.call("view_cart", {}).output
    assert cart["items"][0]["quantity"] == 2
    assert cart["subtotal"] > 0

    promo = reg.call("apply_promo_code", {"code": "SAVE10"})
    assert promo.ok and promo.output["discount"] > 0

    checkout = reg.call(
        "checkout",
        {"address_id": "A1", "payment_method_id": "P1"},
    )
    assert checkout.ok
    order_id = checkout.output["id"]

    # 2. Cart is empty, order exists.
    after = reg.call("view_cart", {}).output
    assert after["items"] == []
    orders = reg.call("list_orders", {}).output
    assert any(o["id"] == order_id for o in orders)

    # 3. Tracking assigns a tracking number and flips status to shipped.
    track = reg.call("track_shipment", {"order_id": order_id}).output
    assert track["tracking_number"] is not None


def test_checkout_rejects_empty_cart() -> None:
    _state, reg = build_ecommerce_registry()
    res = reg.call("checkout", {"address_id": "A1", "payment_method_id": "P1"})
    assert not res.ok and "empty" in res.error


def test_promo_validation() -> None:
    _state, reg = build_ecommerce_registry()
    res = reg.call("apply_promo_code", {"code": "NOT_REAL"})
    assert not res.ok and "invalid" in res.error.lower()


def test_cancel_delivered_order_refused() -> None:
    state, reg = build_ecommerce_registry()
    # Synthesize a delivered order directly and try to cancel.
    from agentmorph.tools.ecommerce.state import CartItem, Order
    oid = state.next_order_id()
    state.orders[oid] = Order(
        id=oid,
        items=[CartItem(product_id=next(iter(state.products)), quantity=1)],
        address_id="A1",
        payment_method_id="P1",
        status="delivered",
        total=10.0,
    )
    res = reg.call("cancel_order", {"order_id": oid})
    assert not res.ok


def test_idempotent_remove_from_cart() -> None:
    state, reg = build_ecommerce_registry()
    pid = next(iter(state.products))
    reg.call("add_to_cart", {"product_id": pid, "quantity": 1})
    # Removing once empties the cart; removing again is a no-op (idempotent).
    r1 = reg.call("remove_from_cart", {"product_id": pid})
    r2 = reg.call("remove_from_cart", {"product_id": pid})
    assert r1.ok and r2.ok
    assert r1.output == r2.output


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_fixture_is_deterministic(seed: int) -> None:
    s1, _ = build_ecommerce_registry(seed=seed)
    s2, _ = build_ecommerce_registry(seed=seed)
    assert {pid: p.stock for pid, p in s1.products.items()} == {
        pid: p.stock for pid, p in s2.products.items()
    }
