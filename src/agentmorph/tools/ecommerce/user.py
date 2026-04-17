"""User tools: profile + addresses."""

from __future__ import annotations

from typing import Any

from agentmorph.tools.base import Tool
from agentmorph.tools.ecommerce.state import Address, ShopState


_ALLOWED_PROFILE_FIELDS = {"name", "email", "phone"}


def build(state: ShopState) -> list[Tool]:
    def get_user_profile() -> dict[str, Any]:
        return state.user.__dict__.copy()

    def update_user_profile(field: str, value: str) -> dict[str, Any]:
        if field not in _ALLOWED_PROFILE_FIELDS:
            raise ValueError(
                f"field must be one of {sorted(_ALLOWED_PROFILE_FIELDS)}"
            )
        setattr(state.user, field, value)
        return state.user.__dict__.copy()

    def list_addresses() -> list[dict[str, Any]]:
        return [a.view() for a in state.addresses.values()]

    def add_address(
        line1: str, city: str, state_: str, zip: str, country: str = "US"
    ) -> dict[str, Any]:
        aid = state.next_address_id()
        addr = Address(id=aid, line1=line1, city=city, state=state_, zip=zip, country=country)
        state.addresses[aid] = addr
        return addr.view()

    return [
        Tool(
            name="get_user_profile",
            description="Fetch the current user's profile (name, email, phone).",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            func=get_user_profile,
            read_only=True,
            category="user",
        ),
        Tool(
            name="update_user_profile",
            description="Update one field of the user profile. Allowed fields: name, email, phone.",
            parameters={
                "type": "object",
                "properties": {
                    "field": {"type": "string", "enum": sorted(_ALLOWED_PROFILE_FIELDS)},
                    "value": {"type": "string"},
                },
                "required": ["field", "value"],
                "additionalProperties": False,
            },
            func=update_user_profile,
            read_only=False,
            category="user",
        ),
        Tool(
            name="list_addresses",
            description="List all shipping addresses saved on the user's account.",
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            func=list_addresses,
            read_only=True,
            category="user",
        ),
        Tool(
            name="add_address",
            description="Add a new shipping address. Note the parameter is `state_` to avoid shadowing built-ins.",
            parameters={
                "type": "object",
                "properties": {
                    "line1": {"type": "string"},
                    "city": {"type": "string"},
                    "state_": {"type": "string", "description": "US state (2-letter)."},
                    "zip": {"type": "string"},
                    "country": {"type": "string", "default": "US"},
                },
                "required": ["line1", "city", "state_", "zip"],
                "additionalProperties": False,
            },
            func=add_address,
            read_only=False,
            category="user",
        ),
    ]
