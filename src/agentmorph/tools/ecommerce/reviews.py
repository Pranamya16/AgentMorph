"""Review tools: read reviews, submit a review."""

from __future__ import annotations

from typing import Any

from agentmorph.tools.base import Tool
from agentmorph.tools.ecommerce.state import Review, ShopState


def build(state: ShopState) -> list[Tool]:
    def get_product_reviews(product_id: str) -> list[dict[str, Any]]:
        return [r.__dict__ for r in state.reviews if r.product_id == product_id]

    def submit_review(product_id: str, rating: int, text: str) -> dict[str, Any]:
        if product_id not in state.products:
            raise KeyError(f"no product {product_id}")
        if not 1 <= rating <= 5:
            raise ValueError("rating must be 1..5")
        state.reviews.append(
            Review(product_id=product_id, rating=rating, text=text, author=state.user.user_id)
        )
        return {"ok": True, "product_id": product_id}

    return [
        Tool(
            name="get_product_reviews",
            description="List all reviews for a product.",
            parameters={
                "type": "object",
                "properties": {"product_id": {"type": "string"}},
                "required": ["product_id"],
                "additionalProperties": False,
            },
            func=get_product_reviews,
            read_only=True,
            category="reviews",
        ),
        Tool(
            name="submit_review",
            description="Submit a star rating (1..5) and text review for a product.",
            parameters={
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "rating": {"type": "integer", "minimum": 1, "maximum": 5},
                    "text": {"type": "string"},
                },
                "required": ["product_id", "rating", "text"],
                "additionalProperties": False,
            },
            func=submit_review,
            read_only=False,
            category="reviews",
        ),
    ]
