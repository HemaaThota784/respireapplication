"""MongoDB connection (collections: users, assessments)."""
from pymongo import ASCENDING, DESCENDING, MongoClient

import config

_client = None


def get_db():
    global _client
    if _client is None:
        _client = MongoClient(config.MONGODB_URI, tz_aware=True, serverSelectionTimeoutMS=5000)
    return _client[config.MONGODB_DB]


def init_db():
    """Create indexes (safe to run on every start)."""
    db = get_db()
    db.users.create_index("email", unique=True)
    db.users.create_index("google_id", unique=True, sparse=True)
    db.assessments.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    return db.command("ping")
