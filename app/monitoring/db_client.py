from __future__ import annotations

"""
db_client.py — MongoDB connection for Med360 observability.

Design:
  - Reads MONGODB_URI from environment (set in .env).
  - Returns None gracefully if MongoDB is unavailable — chat keeps working.
  - Synchronous pymongo (not async motor) — CLI is sync; motor is for FastAPI Phase 8.
  - Single module-level client created on first call to get_db().
"""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_URI = "mongodb://localhost:27017"
_DEFAULT_DB  = "med360"

_client = None  # pymongo.MongoClient, lazily created


def get_db():
    """
    Return the med360 MongoDB database, or None if unavailable.

    Safe to call at any time — returns None if Mongo is not running
    so all callers can do:  db = get_db(); if db is None: return
    """
    global _client

    uri = os.environ.get("MONGODB_URI", _DEFAULT_URI)
    db_name = os.environ.get("MONGODB_DB", _DEFAULT_DB)

    if _client is None:
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

            _client = MongoClient(uri, serverSelectionTimeoutMS=2000)
            # Trigger a real connection check
            _client.admin.command("ping")
        except Exception:
            _client = None
            return None

    try:
        return _client[db_name]
    except Exception:
        return None


def ping_db() -> bool:
    """Return True if MongoDB is reachable, False otherwise."""
    return get_db() is not None


def close_db() -> None:
    """Close the client connection (call on process exit if desired)."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None
