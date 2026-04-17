"""Synthetic 30-tool e-commerce suite.

Call `build_ecommerce_registry()` to get a fresh `ShopState` + `ToolRegistry`
containing exactly 30 tools spread across eight domain modules:

    catalog    (5)   search_products, get_product, list_categories,
                     check_stock, get_recommendations
    reviews    (2)   get_product_reviews, submit_review
    cart       (5)   view_cart, add_to_cart, update_cart_item,
                     remove_from_cart, apply_promo_code
    user       (4)   get_user_profile, update_user_profile,
                     list_addresses, add_address
    orders     (5)   list_orders, get_order, cancel_order,
                     request_return, reorder
    payments   (3)   list_payment_methods, add_payment_method, checkout
    shipping   (3)   track_shipment, estimate_shipping, list_shipping_options
    support    (3)   search_help, create_ticket, get_ticket

The tool-schema design is intentionally kept light so Claude (chat) can refine
specific parameter descriptions against the MAST taxonomy in Stage 2 without
invalidating any trajectories we record in Stage 1.
"""

from __future__ import annotations

from agentmorph.tools.base import ToolRegistry
from agentmorph.tools.ecommerce import (
    cart,
    catalog,
    orders,
    payments,
    reviews,
    shipping,
    support,
    user,
)
from agentmorph.tools.ecommerce.state import ShopState, default_fixture


#: Authoritative, ordered list of the 30 tool names — treat as an API.
ECOMMERCE_TOOL_NAMES: tuple[str, ...] = (
    # catalog
    "search_products",
    "get_product",
    "list_categories",
    "check_stock",
    "get_recommendations",
    # reviews
    "get_product_reviews",
    "submit_review",
    # cart
    "view_cart",
    "add_to_cart",
    "update_cart_item",
    "remove_from_cart",
    "apply_promo_code",
    # user
    "get_user_profile",
    "update_user_profile",
    "list_addresses",
    "add_address",
    # orders
    "list_orders",
    "get_order",
    "cancel_order",
    "request_return",
    "reorder",
    # payments
    "list_payment_methods",
    "add_payment_method",
    "checkout",
    # shipping
    "track_shipment",
    "estimate_shipping",
    "list_shipping_options",
    # support
    "search_help",
    "create_ticket",
    "get_ticket",
)
assert len(ECOMMERCE_TOOL_NAMES) == 30, "e-commerce suite must expose exactly 30 tools"


def build_ecommerce_registry(
    state: ShopState | None = None, *, seed: int = 0
) -> tuple[ShopState, ToolRegistry]:
    """Build a fresh shop state + tool registry.

    Parameters
    ----------
    state:
        Existing `ShopState` to bind tools to. If `None`, a fresh default
        fixture is built with `seed`.
    seed:
        Seed for the default fixture (ignored if `state` is provided).
    """
    if state is None:
        state = default_fixture(seed=seed)

    registry = ToolRegistry()
    for module in (catalog, reviews, cart, user, orders, payments, shipping, support):
        registry.extend(module.build(state))

    # Sanity-check that the concrete registry matches the declared API.
    actual = tuple(registry.names())
    if actual != ECOMMERCE_TOOL_NAMES:
        missing = set(ECOMMERCE_TOOL_NAMES) - set(actual)
        extra = set(actual) - set(ECOMMERCE_TOOL_NAMES)
        raise RuntimeError(
            f"ecommerce registry drifted from ECOMMERCE_TOOL_NAMES; "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    return state, registry


__all__ = ["build_ecommerce_registry", "ECOMMERCE_TOOL_NAMES", "ShopState"]
