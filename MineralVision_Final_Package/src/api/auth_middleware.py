"""
MineralVision JWT Authentication Middleware

Provides JWT token validation (PyJWT) and bcrypt password hashing for the API.

Security contracts (REMEDIATION_SPEC C2):
- JWT_SECRET env is REQUIRED when ENV=production (app refuses to start without it).
  In development an ephemeral random secret is generated with a loud warning.
  There is NO hardcoded fallback secret.
- Passwords are hashed with bcrypt directly (work factor 12). No passlib, no sha256.
- JWTMiddleware enforces authentication globally; only explicit public paths
  are reachable without a valid token.
"""

import os
import logging
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import bcrypt

from .security.oidc import OIDCConfigurationError, OIDCIdentity, OIDCTokenError, OIDCValidator

logger = logging.getLogger(__name__)

# Configuration
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
BCRYPT_WORK_FACTOR = 12

ENV = os.getenv("ENV", os.getenv("ENVIRONMENT", "development")).lower()
AUTH_MODE = os.getenv("AUTH_MODE", "local").lower()
if AUTH_MODE not in {"local", "oidc"}:
    raise RuntimeError("AUTH_MODE must be either 'local' or 'oidc'")
if ENV == "production" and AUTH_MODE != "oidc":
    raise RuntimeError("AUTH_MODE=oidc is required in production; local symmetric JWTs are not a production identity provider")
OIDC_VALIDATOR: Optional[OIDCValidator] = OIDCValidator() if AUTH_MODE == "oidc" else None


def _load_jwt_secret() -> str:
    """Load the JWT signing secret according to the security contract."""
    secret = os.getenv("JWT_SECRET")
    if secret:
        return secret
    if ENV == "production":
        raise RuntimeError(
            "JWT_SECRET environment variable is REQUIRED when ENV=production. "
            "Refusing to start without a signing secret."
        )
    generated = secrets.token_urlsafe(48)
    logger.warning(
        "JWT_SECRET is not set — generated an ephemeral random secret for "
        "DEVELOPMENT only. All tokens are invalidated on restart. "
        "Set JWT_SECRET for any shared or production deployment."
    )
    return generated


# Local signing remains available only for development/test compatibility. OIDC mode
# validates asymmetric Keycloak tokens and therefore never requires this secret.
JWT_SECRET = _load_jwt_secret() if AUTH_MODE == "local" else ""

security = HTTPBearer(auto_error=False)


class TokenPayload:
    """JWT token payload structure."""

    def __init__(self, user_id: str, username: str, email: str, role: str, exp: datetime,
                 jti: Optional[str] = None, roles: Optional[list[str]] = None,
                 project_ids: Optional[list[str]] = None, mfa_verified: bool = False):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.role = role
        self.roles = roles or ([role] if role else [])
        self.project_ids = project_ids or []
        self.mfa_verified = mfa_verified
        self.exp = exp
        self.jti = jti


def create_access_token(user_data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        user_data: Dictionary containing user information (id, username, email, role)
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    if AUTH_MODE == "oidc":
        raise RuntimeError("Local access-token issuance is disabled when AUTH_MODE=oidc; use Keycloak OIDC tokens")
    if expires_delta is None:
        expires_delta = timedelta(hours=JWT_EXPIRATION_HOURS)

    expire = datetime.utcnow() + expires_delta

    payload = {
        "sub": user_data.get("id"),
        "username": user_data.get("username"),
        "email": user_data.get("email"),
        "role": user_data.get("role", "user"),
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": str(uuid.uuid4())
    }

    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[TokenPayload]:
    """Decode a configured OIDC or development-only local bearer token."""
    if AUTH_MODE == "oidc":
        try:
            if OIDC_VALIDATOR is None:
                raise OIDCConfigurationError("OIDC validator is unavailable")
            identity: OIDCIdentity = OIDC_VALIDATOR.validate(token)
            return TokenPayload(
                user_id=identity.subject,
                username=identity.username,
                email=identity.email,
                role=identity.role,
                roles=identity.roles,
                project_ids=identity.project_ids,
                mfa_verified=identity.mfa_verified,
                exp=datetime.fromtimestamp(identity.expires_at),
                jti=identity.token_id,
            )
        except (OIDCConfigurationError, OIDCTokenError):
            return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        jti = payload.get("jti")
        if jti and is_token_blacklisted(jti):
            return None
        role = payload.get("role", "user")
        return TokenPayload(
            user_id=payload.get("sub"),
            username=payload.get("username"),
            email=payload.get("email"),
            role=role,
            roles=[role],
            project_ids=[],
            mfa_verified=False,
            exp=datetime.fromtimestamp(payload.get("exp")),
            jti=jti,
        )
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )
    except (ValueError, TypeError):
        return False


def hash_password(password: str) -> str:
    """Hash a password using bcrypt (work factor 12)."""
    return bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_WORK_FACTOR)
    ).decode("utf-8")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[TokenPayload]:
    """
    FastAPI dependency to get the current authenticated user.

    Returns None if no credentials are provided (for optional auth).
    Raises HTTPException for invalid tokens.
    """
    if credentials is None:
        return None

    payload = decode_token(credentials.credentials)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return payload


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> TokenPayload:
    """
    FastAPI dependency that requires authentication.

    Raises HTTPException if no valid token is provided.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )

    payload = decode_token(credentials.credentials)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return payload


def require_role(allowed_roles: list):
    """
    Dependency factory to require specific roles.

    Args:
        allowed_roles: List of role names that are allowed access

    Returns:
        Dependency function that validates user role
    """
    async def role_checker(user: TokenPayload = Depends(require_auth)) -> TokenPayload:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {allowed_roles}"
            )
        return user

    return role_checker


class JWTMiddleware:
    """
    Pure-ASGI middleware enforcing JWT authentication globally.

    Every request requires a valid Bearer token except requests to the
    explicit public paths (login/register, health, API docs). Authenticated
    user info is added to the request scope state.
    """

    PUBLIC_PATHS = {
        "/auth/login",
        "/auth/register",
        "/api/auth/login",
        "/api/auth/register",
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
    }

    # Prefixes for token-bearing public flows (invitation validate/accept,
    # password reset) — authorized by possession of the secret token itself.
    PUBLIC_PATH_PREFIXES = (
        "/innovations/onboarding/invitations/",
        "/innovations/onboarding/password-reset/",
    )

    def __init__(self, app, enforce: bool = True):
        self.app = app
        self.enforce = enforce

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")

        # Always allow CORS preflight and public paths
        if method == "OPTIONS" or path in self.PUBLIC_PATHS or \
                any(path.startswith(p) for p in self.PUBLIC_PATH_PREFIXES):
            await self.app(scope, receive, send)
            return

        if not self.enforce:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode()

        payload = None
        if auth_header.startswith("Bearer "):
            payload = decode_token(auth_header[7:])

        if payload is None:
            detail = "Authentication required" if not auth_header else "Invalid or expired token"
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b"Bearer"),
                ],
            })
            await send({
                "type": "http.response.body",
                "body": ('{"detail": "%s"}' % detail).encode()
            })
            return

        # Add user info to scope state
        scope.setdefault("state", {})
        scope["state"]["user"] = {
            "id": payload.user_id,
            "username": payload.username,
            "email": payload.email,
            "role": payload.role,
            "roles": payload.roles,
            "project_ids": payload.project_ids,
            "mfa_verified": payload.mfa_verified,
        }

        await self.app(scope, receive, send)


# Token blacklist for logout functionality
_token_blacklist: set = set()


def blacklist_token(jti: str):
    """Add a token ID to the blacklist."""
    if jti:
        _token_blacklist.add(jti)


def is_token_blacklisted(jti: str) -> bool:
    """Check if a token ID is blacklisted."""
    return jti in _token_blacklist
