"""
test_auth.py
------------
Unit tests for authentication, password hashing, database operations, and token verification.
"""

import time
import unittest
from src.database import DatabaseManager
from src.auth import (
    AuthManager,
    hash_password,
    verify_password,
    validate_email,
    validate_password_strength,
    generate_token,
    verify_token,
)


class TestAuthComponents(unittest.TestCase):
    """Tests core cryptographic and helper functions."""

    def test_hash_and_verify_password(self):
        password = "SecurePassword123"
        hashed, salt = hash_password(password)
        
        self.assertTrue(verify_password(password, hashed, salt))
        self.assertFalse(verify_password("WrongPassword123", hashed, salt))
        self.assertNotEqual(password, hashed)

    def test_validate_email(self):
        self.assertTrue(validate_email("user@example.com"))
        self.assertTrue(validate_email("john.doe+test@domain.co.in"))
        self.assertFalse(validate_email("invalid-email"))
        self.assertFalse(validate_email("user@domain"))

    def test_validate_password_strength(self):
        # Valid password
        valid, msg = validate_password_strength("Pass1234")
        self.assertTrue(valid)

        # Too short
        valid, msg = validate_password_strength("P123")
        self.assertFalse(valid)
        self.assertIn("at least 8 characters", msg)

        # No digits
        valid, msg = validate_password_strength("PasswordOnly")
        self.assertFalse(valid)
        self.assertIn("at least one number", msg)

        # No letters
        valid, msg = validate_password_strength("123456789")
        self.assertFalse(valid)
        self.assertIn("at least one letter", msg)

    def test_token_generation_and_verification(self):
        secret = "test_secret_key"
        token = generate_token("testuser", secret_key=secret, ttl_seconds=3600)
        
        valid, username, msg = verify_token(token, secret_key=secret)
        self.assertTrue(valid)
        self.assertEqual(username, "testuser")

        # Verify failure with wrong secret
        valid, username, msg = verify_token(token, secret_key="wrong_secret")
        self.assertFalse(valid)
        self.assertIn("signature", msg)

        # Verify failure with expired token
        expired_token = generate_token("testuser", secret_key=secret, ttl_seconds=-10)
        valid, username, msg = verify_token(expired_token, secret_key=secret)
        self.assertFalse(valid)
        self.assertIn("expired", msg)


class TestDatabaseManager(unittest.TestCase):
    """Tests SQLite database user persistence."""

    def setUp(self):
        self.db = DatabaseManager(":memory:")

    def test_create_and_get_user(self):
        hashed, salt = hash_password("Secret123")
        user = self.db.create_user("alice", "alice@example.com", hashed, salt)
        
        self.assertEqual(user["username"], "alice")
        self.assertEqual(user["email"], "alice@example.com")

        fetched = self.db.get_user_by_username("alice")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["email"], "alice@example.com")

        fetched_email = self.db.get_user_by_email("alice@example.com")
        self.assertIsNotNone(fetched_email)
        self.assertEqual(fetched_email["username"], "alice")

    def test_duplicate_user_rejection(self):
        hashed, salt = hash_password("Secret123")
        self.db.create_user("bob", "bob@example.com", hashed, salt)

        # Duplicate username
        with self.assertRaises(ValueError) as ctx:
            self.db.create_user("bob", "other@example.com", hashed, salt)
        self.assertIn("Username", str(ctx.exception))

        # Duplicate email
        with self.assertRaises(ValueError) as ctx:
            self.db.create_user("charlie", "bob@example.com", hashed, salt)
        self.assertIn("Email", str(ctx.exception))


class TestAuthManager(unittest.TestCase):
    """Tests high-level registration, login, and authentication workflow."""

    def setUp(self):
        self.db = DatabaseManager(":memory:")
        self.auth = AuthManager(self.db, secret_key="test_auth_secret")

    def test_register_and_login_success(self):
        reg = self.auth.register_user("devuser", "dev@example.com", "DevPassword1")
        self.assertEqual(reg["username"], "devuser")

        login_res = self.auth.login_user("devuser", "DevPassword1")
        self.assertIn("token", login_res)
        self.assertEqual(login_res["user"]["username"], "devuser")

        # Test login via email
        login_email = self.auth.login_user("dev@example.com", "DevPassword1")
        self.assertIn("token", login_email)

    def test_login_invalid_credentials(self):
        self.auth.register_user("testuser", "test@example.com", "DevPassword1")

        # Wrong password
        with self.assertRaises(ValueError):
            self.auth.login_user("testuser", "WrongPass1")

        # Non-existent user
        with self.assertRaises(ValueError):
            self.auth.login_user("nonexistent", "DevPassword1")

    def test_authenticate_token(self):
        self.auth.register_user("tokenuser", "token@example.com", "DevPassword1")
        login_res = self.auth.login_user("tokenuser", "DevPassword1")
        token = login_res["token"]

        user_info = self.auth.authenticate_token(token)
        self.assertEqual(user_info["username"], "tokenuser")

        # Invalid token
        with self.assertRaises(ValueError):
            self.auth.authenticate_token("invalid.token.str")


if __name__ == "__main__":
    unittest.main()
