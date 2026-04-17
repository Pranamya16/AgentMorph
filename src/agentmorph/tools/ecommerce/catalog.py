"""Catalog tools: search, product detail, categories, stock, recommendations."""

from __future__ import annotations

from typing import Any

from agentmorph.tools.base import Tool
from agentmorph.tools.ecommerce.state import ShopState


def build(state: ShopState) -> list[Tool]:
    def search_products(
        query: str,
        category: str | None = None,
        min_price: float | None = None,
        max_price: float | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        q = query.lower().strip()
        out: list[dict[str, Any]] = []
        for p in state.products.values():
            if category and p.category != category:
                continue
            if min_price is not None and p.price < min_price:
                continue
            if max_price is not None and p.price > max_price:
                continue
            if q and q not in p.name.lower() and q not in p.description.lower():
                continue
            out.append(p.view())
            if len(out) >= limit:
                break
        return out

    def get_product(product_id: str) -> dict[str, Any]:
        p = state.products.get(product_id)
        if p is None:
            raise KeyError(f"no product {product_id}")
        return p.view()

    def list_categories() -> list[str]:
        return list(state.categories)

    def check_stock(product_id: str) -> dict[str, Any]:
        p = state.products.get(product_id)
        if p is None:
            raise KeyError(f"no product {product_id}")
        return {"product_id": p.id, "stock": p.stock, "in_stock": p.stock > 0}

    def get_recommendations(product_id: str, limit: int = 3) -> list[dict[str, Any]]:
        p = state.products.get(product_id)
        if p is None:
            raise KeyError(f"no product {product_id}")
        # Simple deterministic rule: same-category siblings by price proximity.
        siblings = [q for q in state.products.values() if q.category == p.category and q.id != p.id]
        siblings.sort(key=lambda q: abs(q.price - p.price))
        return [q.view() for q in siblings[:limit]]

    return [
        Tool(
            name="search_products",
            description=(
                "Search the product catalog. Returns up to `limit` products matching "
                "`query` in name/description, optionally filtered by category and price."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text query."},
                    "category": {"type": "string", "description": "Restrict to one category."},
                    "min_price": {"type": "number"},
                    "max_price": {"type": "number"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            func=search_products,
            read_only=True,
            category="catalog",
        ),
        Tool(
            name="get_product",
            description="Fetch the full detail record for a product by id.",
            parameters={
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
                "additionalProperties": False,
            },
            func=get_product,
            read_only=True,
            category="catalog",
        ),
        Tool(
            name="list_categories",
            description="List all product categories available in the catalog.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            func=list_categories,
            read_only=True,
            category="catalog",
        ),
        Tool(
            name="check_stock",
            description="Return the current stock level of a product.",
            parameters={
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
                "additionalProperties": False,
            },
            func=check_stock,
            read_only=True,
            category="catalog",
        ),
        Tool(
            name="get_recommendations",
            description="Return up to `limit` products related to the given product.",
            parameters={
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                },
                "required": ["product_id"],
                "additionalProperties": False,
            },
            func=get_recommendations,
            read_only=True,
            category="catalog",
        ),
    ]
