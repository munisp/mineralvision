"""
MineralVision JWT Authentication Middleware

Provides JWT token validation and user authentication for API endpoints.
"""

import os
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from functools import wraps

from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

# Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "mineralvision-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

security = HTTPBearer(auto_error=False)


class TokenPayload:
    """JWT token payload structure."""
    
    def __init__(self, user_id: str, username: str, email: str, role: str, exp: datetime):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.role = role
        self.exp = exp


def create_access_token(user_data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        user_data: Dictionary containing user information (id, username, email, role)
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT token string
    """
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
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        TokenPayload if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return TokenPayload(
            user_id=payload.get("sub"),
            username=payload.get("username"),
            email=payload.get("email"),
            role=payload.get("role", "user"),
            exp=datetime.fromtimestamp(payload.get("exp"))
        )
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password


def hash_password(password: str) -> str:
    """Hash a password using SHA256."""
    return hashlib.sha256(password.encode()).hexdigest()


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[TokenPayload]:
    """
    FastAPI dependency to get the current authenticated user.
    
    Returns None if no valid token is provided (for optional auth).
    Raises HTTPException for invalid tokens.
    """
    if credentials is None:
        return None
    
    token = credentials.credentials
    payload = decode_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return payload


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
) -> TokenPayload:
    """
    FastAPI dependency that requires authentication.
    
    Raises HTTPException if no valid token is provided.
    """
    token = credentials.credentials
    payload = decode_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return payload


def require_role(allowed_roles: list):
    """
    Decorator factory to require specific roles.
    
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
    Middleware for JWT token validation.
    
    Validates tokens on protected routes and adds user info to request state.
    """
    
    PROTECTED_PREFIXES = [
        "/api/projects",
        "/api/drillholes",
        "/api/samples",
        "/api/qaqc",
        "/api/geostatistics",
        "/api/visualization",
        "/api/reports",
        "/api/users",
        "/api/upload"
    ]
    
    EXCLUDED_PATHS = [
        "/api/auth/login",
        "/api/auth/register",
        "/api/status",
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json"
    ]
    
    def __init__(self, app, enforce: bool = False):
        """
        Initialize JWT middleware.
        
        Args:
            app: FastAPI application
            enforce: If True, reject requests without valid tokens on protected routes
        """
        self.app = app
        self.enforce = enforce
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        path = scope.get("path", "")
        
        # Skip excluded paths
        if any(path.startswith(excluded) for excluded in self.EXCLUDED_PATHS):
            await self.app(scope, receive, send)
            return
        
        # Check if path is protected
        is_protected = any(path.startswith(prefix) for prefix in self.PROTECTED_PREFIXES)
        
        if is_protected and self.enforce:
            # Extract token from headers
            headers = dict(scope.get("headers", []))
            auth_header = headers.get(b"authorization", b"").decode()
            
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                payload = decode_token(token)
                
                if payload is None:
                    # Send 401 response
                    response = {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [(b"content-type", b"application/json")]
                    }
                    await send(response)
                    await send({
                        "type": "http.response.body",
                        "body": b'{"detail": "Invalid or expired token"}'
                    })
                    return
                
                # Add user info to scope
                scope["state"] = scope.get("state", {})
                scope["state"]["user"] = {
                    "id": payload.user_id,
                    "username": payload.username,
                    "email": payload.email,
                    "role": payload.role
                }
            elif self.enforce:
                # No token provided on protected route
                response = {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json")]
                }
                await send(response)
                await send({
                    "type": "http.response.body",
                    "body": b'{"detail": "Authentication required"}'
                })
                return
        
        await self.app(scope, receive, send)


# Token blacklist for logout functionality
_token_blacklist: set = set()


def blacklist_token(jti: str):
    """Add a token ID to the blacklist."""
    _token_blacklist.add(jti)


def is_token_blacklisted(jti: str) -> bool:
    """Check if a token ID is blacklisted."""
    return jti in _token_blacklist
