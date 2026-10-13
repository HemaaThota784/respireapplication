"""Authentication: e-mail/password (JWT in an httpOnly cookie) + Google OAuth."""
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote

from authlib.integrations.base_client.errors import OAuthError
from authlib.integrations.starlette_client import OAuth
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, Field, field_validator
from pymongo.errors import DuplicateKeyError
from starlette.concurrency import run_in_threadpool

import config
from db import get_db
from security import (DUMMY_HASH, create_access_token, decode_access_token,
                      hash_password, verify_password)

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ---------------------------------------------------------------- helpers
def user_public(u: dict) -> dict:
    return {"id": str(u["_id"]), "name": u.get("name", ""), "email": u["email"],
            "picture": u.get("picture"), "provider": u.get("provider", "local"),
            "created_at": u["created_at"].isoformat()}


def set_auth_cookie(response: Response, user_id: str) -> None:
    response.set_cookie(
        key=config.COOKIE_NAME, value=create_access_token(user_id),
        max_age=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        httponly=True, secure=config.COOKIE_SECURE, samesite="lax", path="/")


def get_current_user(request: Request) -> dict:
    """FastAPI dependency: reads the JWT from the cookie (or 'Authorization: Bearer ...')."""
    token = request.cookies.get(config.COOKIE_NAME)
    if not token:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
    user_id = decode_access_token(token) if token else None
    if not user_id:
        raise HTTPException(401, "Not authenticated.")
    try:
        user = get_db().users.find_one({"_id": ObjectId(user_id)})
    except InvalidId:
        user = None
    if not user:
        raise HTTPException(401, "Not authenticated.")
    return user


# ---------------------------------------------------------------- schemas
class RegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)

    @field_validator("name")
    @classmethod
    def clean_name(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Name is required.")
        return v

    @field_validator("password")
    @classmethod
    def password_bytes(cls, v):
        if len(v.encode("utf-8")) > 72:
            raise ValueError("Password is too long (max 72 bytes).")
        return v


class LoginIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


# ---------------------------------------------------------------- e-mail / password
@router.post("/register", status_code=201)
def register(body: RegisterIn, response: Response):
    users = get_db().users
    email = body.email.lower()
    if users.find_one({"email": email}):
        raise HTTPException(409, "An account with this email already exists.")
    now = datetime.now(timezone.utc)
    doc = {"name": body.name, "email": email, "password_hash": hash_password(body.password),
           "provider": "local", "created_at": now, "last_login": now}
    try:
        doc["_id"] = users.insert_one(doc).inserted_id
    except DuplicateKeyError:
        raise HTTPException(409, "An account with this email already exists.")
    set_auth_cookie(response, str(doc["_id"]))
    return user_public(doc)


@router.post("/login")
def login(body: LoginIn, response: Response):
    users = get_db().users
    user = users.find_one({"email": body.email.lower()})
    if user is None:
        verify_password(body.password, DUMMY_HASH)           # constant-ish timing
        raise HTTPException(401, "Incorrect email or password.")
    if not user.get("password_hash"):
        raise HTTPException(400, "This account uses Google sign-in. Please use 'Continue with Google'.")
    if not verify_password(body.password, user["password_hash"]):
        raise HTTPException(401, "Incorrect email or password.")
    users.update_one({"_id": user["_id"]}, {"$set": {"last_login": datetime.now(timezone.utc)}})
    set_auth_cookie(response, str(user["_id"]))
    return user_public(user)


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(config.COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/me")
def me(user: dict = Depends(get_current_user)):
    return user_public(user)


# ---------------------------------------------------------------- Google OAuth
oauth = OAuth()
GOOGLE_ENABLED = bool(config.GOOGLE_CLIENT_ID and config.GOOGLE_CLIENT_SECRET)
if GOOGLE_ENABLED:
    oauth.register(
        name="google",
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_id=config.GOOGLE_CLIENT_ID,
        client_secret=config.GOOGLE_CLIENT_SECRET,
        client_kwargs={"scope": "openid email profile"},
    )


@router.get("/providers")
def providers():
    return {"google": GOOGLE_ENABLED}


def _login_error(message: str) -> RedirectResponse:
    return RedirectResponse(f"{config.FRONTEND_URL}/?auth_error={quote(message)}")


@router.get("/google/login")
async def google_login(request: Request):
    if not GOOGLE_ENABLED:
        raise HTTPException(404, "Google sign-in is not configured.")
    return await oauth.google.authorize_redirect(request, config.GOOGLE_REDIRECT_URI)


def _upsert_google_user(info: dict) -> dict:
    users = get_db().users
    now = datetime.now(timezone.utc)
    email = info["email"].lower()
    google_id = str(info["sub"])
    user = users.find_one({"google_id": google_id}) or users.find_one({"email": email})
    if user:   # existing account (Google or same verified e-mail) -> link and sign in
        users.update_one({"_id": user["_id"]}, {"$set": {
            "google_id": google_id, "last_login": now,
            "picture": info.get("picture") or user.get("picture"),
            "name": user.get("name") or info.get("name") or email.split("@")[0]}})
        return users.find_one({"_id": user["_id"]})
    doc = {"name": info.get("name") or email.split("@")[0], "email": email, "google_id": google_id,
           "picture": info.get("picture"), "provider": "google", "created_at": now, "last_login": now}
    doc["_id"] = users.insert_one(doc).inserted_id
    return doc


@router.get("/google/callback")
async def google_callback(request: Request):
    if not GOOGLE_ENABLED:
        raise HTTPException(404, "Google sign-in is not configured.")
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError:
        return _login_error("Google sign-in failed. Please try again.")
    info = token.get("userinfo") or {}
    if not info.get("email") or not info.get("email_verified"):
        return _login_error("Your Google account email is not verified.")
    user = await run_in_threadpool(_upsert_google_user, info)
    resp = RedirectResponse(f"{config.FRONTEND_URL}/")
    set_auth_cookie(resp, str(user["_id"]))
    return resp
