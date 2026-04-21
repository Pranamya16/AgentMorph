"""Rule 6 — `tool-name-insensitivity` name mapping.

Deterministic rename for each of the 30 e-commerce tools. Choices aim for:
  * semantic plausibility (the new name still describes what the tool does)
  * distinct tokenisation (not a trivial synonym — forces the model to
    re-interpret rather than pattern-match the original name)

Do NOT rename a tool to another real tool's name — that would make the
mapping ambiguous. Do NOT use names that a model is likely to hallucinate
(`get_cart`, `buy_product`) — those collide with model priors and hide
the test signal.

The map is the source of truth for rule 6's `MUTATOR.metadata["name_map"]`.
"""

# Original name -> renamed name.
NAME_MAP: dict[str, str] = {
    # catalog
    "search_products": "find_items",
    "get_product": "fetch_item_details",
    "list_categories": "enumerate_sections",
    "check_stock": "query_inventory_level",
    "get_recommendations": "fetch_related_items",
    # reviews
    "get_product_reviews": "list_item_feedback",
    "submit_review": "post_item_feedback",
    # cart
    "view_cart": "inspect_basket",
    "add_to_cart": "put_in_basket",
    "update_cart_item": "modify_basket_line",
    "remove_from_cart": "drop_from_basket",
    "apply_promo_code": "redeem_voucher",
    # user
    "get_user_profile": "fetch_account_info",
    "update_user_profile": "edit_account_info",
    "list_addresses": "enumerate_shipping_locations",
    "add_address": "register_shipping_location",
    # orders
    "list_orders": "enumerate_purchases",
    "get_order": "fetch_purchase_details",
    "cancel_order": "rescind_purchase",
    "request_return": "initiate_refund",
    "reorder": "replay_purchase",
    # payments
    "list_payment_methods": "enumerate_cards",
    "add_payment_method": "register_card",
    "checkout": "complete_purchase",
    # shipping
    "track_shipment": "query_delivery_status",
    "estimate_shipping": "compute_shipping_fee",
    "list_shipping_options": "enumerate_delivery_tiers",
    # support
    "search_help": "query_knowledge_base",
    "create_ticket": "open_support_case",
    "get_ticket": "fetch_support_case",
}


# Inverse map, for rule 6's checker to translate renamed tool-call events
# in the mutated trajectory back to their original names before comparing.
INVERSE_NAME_MAP: dict[str, str] = {new: old for old, new in NAME_MAP.items()}


# Sanity: each side of the mapping is a bijection on the 30-tool set.
assert len(NAME_MAP) == 30
assert len(set(NAME_MAP.values())) == 30
assert set(NAME_MAP.keys()) & set(NAME_MAP.values()) == set(), (
    "rename targets must not collide with any original tool name"
)
