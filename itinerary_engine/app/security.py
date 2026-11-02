"""Who may call what.

Three separate concerns, deliberately kept apart because they protect different things:

* **API keys** guard machine-to-machine calls and anything holding customer data. A missing
  key configuration fails closed -- a setting nobody remembered to add must never mean
  "let everyone in".
* **A UI session** guards the browser planner. It cannot use an API key, because a key
  embedded in a page is a published key. Instead a password is exchanged once for a signed,
  expiring cookie.
* **Rate limiting** guards the machine. Planning is CPU-bound and holds the dataset in
  memory, so an open planning endpoint is a cheap way to take the service down.

The UI password is the one place this does not fail closed, and for a reason: with no
password set there is no way to log in either, so refusing would lock out the owner rather
than an attacker. Instead the state is made loud -- a warning at startup, a flag on
``/health``, and a banner in the page itself -- and the deployment template sets one.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from collections import deque

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

API_KEYS_ENV = "ITINERARY_API_KEYS"
ALLOW_ANONYMOUS_ENV = "ITINERARY_API_ALLOW_ANONYMOUS"
UI_PASSWORD_ENV = "ITINERARY_UI_PASSWORD"
SESSION_SECRET_ENV = "ITINERARY_SESSION_SECRET"

SESSION_COOKIE = "itinerary_session"
SESSION_HOURS = 12

#: Requests per minute per client for the expensive endpoints. Planning a trip costs about
#: a second of CPU, so this is generous for a person and tight for a script.
PLANNING_PER_MINUTE = int(os.environ.get("ITINERARY_RATE_LIMIT", "30"))

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Your API key. Set ITINERARY_API_KEYS on the server to issue keys.",
)


# --- API keys ---------------------------------------------------------------------------------
def configured_keys() -> set[str]:
    raw = os.environ.get(API_KEYS_ENV, "")
    return {key.strip() for key in raw.split(",") if key.strip()}


def anonymous_allowed() -> bool:
    return os.environ.get(ALLOW_ANONYMOUS_ENV, "").strip().lower() in {"1", "true", "yes"}


def require_api_key(key: str | None = Depends(api_key_header)) -> str:
    """Reject anything without a valid key, and fail closed when none are configured."""
    keys = configured_keys()
    if not keys:
        if anonymous_allowed():
            return "anonymous"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"This API has no keys configured, so it cannot authenticate anyone. Set "
                f"{API_KEYS_ENV} to one or more comma-separated keys and restart, or set "
                f"{ALLOW_ANONYMOUS_ENV}=1 for local development."
            ),
        )
    if not key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Missing X-API-Key header.")
    # Constant-time, so a wrong key cannot be found one character at a time.
    if not any(hmac.compare_digest(key, valid) for valid in keys):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")
    return key


# --- the browser session ----------------------------------------------------------------------
def ui_password() -> str:
    return os.environ.get(UI_PASSWORD_ENV, "").strip()


def ui_is_protected() -> bool:
    return bool(ui_password())


def _secret() -> bytes:
    """The key sessions are signed with.

    A configured secret keeps sessions valid across restarts and between instances. Without
    one a random secret is generated per process, which is safe but logs everyone out on
    redeploy -- fine for one instance, wrong for several.
    """
    configured = os.environ.get(SESSION_SECRET_ENV, "").strip()
    if configured:
        return configured.encode("utf-8")
    global _EPHEMERAL_SECRET
    if _EPHEMERAL_SECRET is None:
        _EPHEMERAL_SECRET = secrets.token_bytes(32)
        if ui_is_protected():
            logger.warning("%s is not set, so sessions end at restart and are not shared "
                           "between instances.", SESSION_SECRET_ENV)
    return _EPHEMERAL_SECRET


_EPHEMERAL_SECRET: bytes | None = None


def issue_session(hours: int = SESSION_HOURS) -> str:
    """A signed token saying "this browser logged in, and until when"."""
    expires = int(time.time() + hours * 3600)
    body = str(expires).encode("ascii")
    signature = hmac.new(_secret(), body, hashlib.sha256).digest()[:16]
    return f"{expires}.{base64.urlsafe_b64encode(signature).decode('ascii').rstrip('=')}"


def session_is_valid(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    stamp, _, signature = token.partition(".")
    try:
        expires = int(stamp)
    except ValueError:
        return False
    if expires < time.time():
        return False
    expected = hmac.new(_secret(), stamp.encode("ascii"), hashlib.sha256).digest()[:16]
    expected_text = base64.urlsafe_b64encode(expected).decode("ascii").rstrip("=")
    return hmac.compare_digest(signature, expected_text)


def password_matches(given: str) -> bool:
    password = ui_password()
    return bool(password) and hmac.compare_digest(str(given or ""), password)


def require_ui_session(request: Request) -> None:
    """Let the browser through when it has logged in -- or when no password is set.

    An API key is accepted too, so a script can drive the same endpoints without pretending
    to be a browser.
    """
    if not ui_is_protected():
        return
    if session_is_valid(request.cookies.get(SESSION_COOKIE)):
        return
    key = request.headers.get("X-API-Key")
    if key and any(hmac.compare_digest(key, valid) for valid in configured_keys()):
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Sign in to use the planner.")


# --- rate limiting -------------------------------------------------------------------------------
class RateLimiter:
    """A sliding window per client. In-process, so it is per instance -- which is the right
    shape for protecting *this* machine's CPU, and not a substitute for an edge limiter."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._seen: dict[str, deque[float]] = {}

    def check(self, client: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        window = self._seen.setdefault(client, deque())
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.per_minute:
            return False
        window.append(now)
        if len(self._seen) > 4096:  # keep the dict from growing without bound
            for key in [k for k, v in self._seen.items() if not v][:1024]:
                self._seen.pop(key, None)
        return True


_planning_limiter = RateLimiter(PLANNING_PER_MINUTE)


def limit_planning(request: Request) -> None:
    client = request.headers.get("x-forwarded-for", "").split(",")[0].strip() \
        or (request.client.host if request.client else "unknown")
    if not _planning_limiter.check(client):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many requests. The limit is {PLANNING_PER_MINUTE} a minute.",
            headers={"Retry-After": "60"},
        )


def reset_rate_limit() -> None:
    """Used by the tests, which would otherwise exhaust the window for each other."""
    _planning_limiter._seen.clear()


def warn_if_open() -> None:
    """Say clearly, at startup, what is not protected."""
    if not ui_is_protected():
        logger.warning(
            "%s is not set: the planner UI and its endpoints are open to anyone who can "
            "reach this server. Set it before exposing this deployment.", UI_PASSWORD_ENV)
    if not configured_keys() and not anonymous_allowed():
        logger.warning("%s is not set: the public API and the customer records will refuse "
                       "every request until it is.", API_KEYS_ENV)
