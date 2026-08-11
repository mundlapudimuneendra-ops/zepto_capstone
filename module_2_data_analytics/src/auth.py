"""
auth.py
-------
Cryptographic authentication and user management service.
Uses PBKDF2-HMAC-SHA256 password hashing and HMAC-SHA256 session token verification.
"""

from __future__ import annotations

import base64
import hmac
import hashlib
import json
import re
import secrets
import time
from typing import Any, Dict, Optional, Tuple

from src.database import DatabaseManager, DEFAULT_DB_PATH

# Default secret key for token signing (can be overridden via environment or parameter)
SECRET_KEY = secrets.token_hex(32)


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hashes a password using PBKDF2-HMAC-SHA256 with 100,000 iterations.
    Returns (hex_hash, hex_salt).
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    salt_bytes = bytes.fromhex(salt)
    hash_bytes = hashlib.pbkdf2_hmac(
        hash_name="sha256",
        password=password.encode("utf-8"),
        salt=salt_bytes,
        iterations=100000,
    )
    return hash_bytes.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verifies a plain password against stored hash and salt using constant-time comparison."""
    computed_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(computed_hash, stored_hash)


def validate_email(email: str) -> bool:
    """Validates email structure."""
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email.strip()))


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """
    Enforces password strength rules:
    - Minimum length: 8 characters
    - Must contain at least one letter
    - Must contain at least one number
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Za-z]", password):
        return False, "Password must contain at least one letter."
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number."
    return True, "Password is valid."


def generate_token(username: str, secret_key: str = SECRET_KEY, ttl_seconds: int = 86400) -> str:
    """Generates a cryptographically signed HMAC token containing payload and expiration timestamp."""
    payload = {
        "username": username,
        "exp": int(time.time()) + ttl_seconds,
    }
    payload_json = json.dumps(payload, sort_keys=True).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_json).decode("utf-8").rstrip("=")
    
    signature = hmac.new(
        secret_key.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    
    return f"{payload_b64}.{signature}"


def verify_token(token: str, secret_key: str = SECRET_KEY) -> Tuple[bool, Optional[str], str]:
    """
    Verifies a cryptographically signed token.
    Returns (is_valid, username, message).
    """
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return False, None, "Malformed token format."
        
        payload_b64, signature = parts
        
        # Verify signature
        expected_sig = hmac.new(
            secret_key.encode("utf-8"),
            payload_b64.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(signature, expected_sig):
            return False, None, "Invalid token signature."
        
        # Decode payload
        padding = "=" * (-len(payload_b64) % 4)
        payload_json = base64.urlsafe_b64decode(payload_b64 + padding).decode("utf-8")
        payload = json.loads(payload_json)
        
        # Check expiration
        if time.time() > payload.get("exp", 0):
            return False, None, "Token has expired."
        
        return True, payload.get("username"), "Token is valid."
    except Exception as e:
        return False, None, f"Token validation error: {str(e)}"


class AuthManager:
    """High-level service class handling user authentication workflows."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None, secret_key: str = SECRET_KEY):
        self.db = db_manager if db_manager is not None else DatabaseManager(DEFAULT_DB_PATH)
        self.secret_key = secret_key

    def register_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """
        Validates input and registers a new user.
        Returns user info dictionary on success or raises ValueError on validation error.
        """
        username = username.strip()
        email = email.strip()

        if not username:
            raise ValueError("Username cannot be empty.")
        if len(username) < 3:
            raise ValueError("Username must be at least 3 characters long.")
        if not validate_email(email):
            raise ValueError("Invalid email format.")
        
        is_strong, msg = validate_password_strength(password)
        if not is_strong:
            raise ValueError(msg)

        password_hash, salt = hash_password(password)
        user_data = self.db.create_user(username, email, password_hash, salt)
        return user_data

    def login_user(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticates a user and generates a session token.
        Returns token and user profile info on success or raises ValueError.
        """
        username = username.strip()
        user = self.db.get_user_by_username(username)
        if not user:
            # Also check by email if user passed email instead of username
            user = self.db.get_user_by_email(username)
            if not user:
                raise ValueError("Invalid username/email or password.")

        if not verify_password(password, user["password_hash"], user["salt"]):
            raise ValueError("Invalid username/email or password.")

        token = generate_token(user["username"], self.secret_key)
        return {
            "token": token,
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "created_at": user["created_at"],
            },
        }

    def authenticate_token(self, token: str) -> Dict[str, Any]:
        """
        Verifies authorization token and returns user details.
        Raises ValueError if invalid/expired.
        """
        is_valid, username, msg = verify_token(token, self.secret_key)
        if not is_valid or not username:
            raise ValueError(msg)
        
        user = self.db.get_user_by_username(username)
        if not user:
            raise ValueError("User associated with token no longer exists.")
        
        return {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "created_at": user["created_at"],
        }
