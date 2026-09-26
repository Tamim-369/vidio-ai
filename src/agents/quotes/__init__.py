"""Picks the quote(s) to narrate."""
from src.agents.quotes.agent import Quote, generate_quotes
from src.agents.quotes.json_parse import _loads_json

__all__ = ["Quote", "generate_quotes", "_loads_json"]
