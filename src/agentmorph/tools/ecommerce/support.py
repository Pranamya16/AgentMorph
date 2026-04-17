"""Support tools: help search, ticket creation, ticket detail."""

from __future__ import annotations

from typing import Any

from agentmorph.tools.base import Tool
from agentmorph.tools.ecommerce.state import ShopState, Ticket


def build(state: ShopState) -> list[Tool]:
    def search_help(query: str) -> list[dict[str, Any]]:
        q = query.lower().strip()
        out = []
        for topic, body in state.help_articles.items():
            if q in topic.lower() or q in body.lower() or not q:
                out.append({"topic": topic, "body": body})
        return out

    def create_ticket(subject: str, body: str) -> dict[str, Any]:
        tid = state.next_ticket_id()
        ticket = Ticket(id=tid, subject=subject, body=body, status="open")
        state.tickets[tid] = ticket
        return ticket.__dict__.copy()

    def get_ticket(ticket_id: str) -> dict[str, Any]:
        t = state.tickets.get(ticket_id)
        if t is None:
            raise KeyError(f"no ticket {ticket_id}")
        return t.__dict__.copy()

    return [
        Tool(
            name="search_help",
            description="Search the help-center knowledge base for articles matching `query`.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            func=search_help,
            read_only=True,
            category="support",
        ),
        Tool(
            name="create_ticket",
            description="Open a new support ticket with a subject and body.",
            parameters={
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["subject", "body"],
                "additionalProperties": False,
            },
            func=create_ticket,
            read_only=False,
            category="support",
        ),
        Tool(
            name="get_ticket",
            description="Fetch a support ticket by id.",
            parameters={
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"],
                "additionalProperties": False,
            },
            func=get_ticket,
            read_only=True,
            category="support",
        ),
    ]
