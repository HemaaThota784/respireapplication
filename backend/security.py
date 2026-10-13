"""Password hashing (bcrypt) and JWT helpers."""
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt

import config


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:      # e.g. password longer than bcrypt's 72-byte limit
        return False


# Used to keep login timing similar when the e-mail does not exist.
DUMMY_HASH = hash_password("not-a-real-password")


def create_access_token(user_id: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": user_id, "iat": now,
               "exp": now + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)}
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[str]:
    """Returns the user id, or None if the token is invalid/expired."""
    try:
        data = jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM],
                          options={"require": ["exp", "sub"]})
        return str(data["sub"])
    except jwt.PyJWTError:
        return None
