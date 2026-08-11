"""
test_api.py
-----------
Integration tests for AppService endpoints, authentication headers, and protected ML predictions.
"""

import os
import unittest
from src.app import AppService
from src.database import DatabaseManager

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "best_classifier.joblib")
REGRESSOR_PATH = os.path.join(MODELS_DIR, "fare_regressor.joblib")


class TestAppService(unittest.TestCase):
    """Tests service workflows for API endpoints."""

    def setUp(self):
        self.service = AppService(
            db_path=":memory:",
            classifier_path=CLASSIFIER_PATH,
            regressor_path=REGRESSOR_PATH,
        )

    def test_registration_and_login_flow(self):
        # Register user
        reg_payload = {
            "username": "apiuser",
            "email": "apiuser@example.com",
            "password": "Password123",
        }
        reg_res = self.service.register(reg_payload)
        self.assertEqual(reg_res["status"], "success")
        self.assertEqual(reg_res["user"]["username"], "apiuser")

        # Login user
        login_payload = {
            "username": "apiuser",
            "password": "Password123",
        }
        login_res = self.service.login(login_payload)
        self.assertEqual(login_res["status"], "success")
        token = login_res["data"]["token"]
        self.assertTrue(token)

        # Get profile with valid Bearer token
        auth_header = f"Bearer {token}"
        profile_res = self.service.get_profile(auth_header)
        self.assertEqual(profile_res["status"], "success")
        self.assertEqual(profile_res["user"]["username"], "apiuser")

    def test_protected_endpoint_without_token(self):
        # Attempt profile access without auth header
        with self.assertRaises(ValueError):
            self.service.get_profile(None)

        # Attempt prediction without auth header
        passenger = {
            "ticket_class": 1,
            "sex": "female",
            "age": 29.0,
            "sublings": 0,
            "parch": 0,
            "fare": 211.33,
            "embarked": "S",
        }
        with self.assertRaises(ValueError):
            self.service.predict_survival(None, passenger)

    def test_protected_ml_prediction_with_valid_token(self):
        # Register and login
        self.service.register({
            "username": "mluser",
            "email": "mluser@example.com",
            "password": "Password123",
        })
        login_res = self.service.login({
            "username": "mluser",
            "password": "Password123",
        })
        token = login_res["data"]["token"]
        auth_header = f"Bearer {token}"

        # Sample passenger data matching schema
        passenger = {
            "ticket_class": 1,
            "sex": "female",
            "age": 29.0,
            "sublings": 0,
            "parch": 0,
            "fare": 211.33,
            "embarked": "S",
        }

        # Test Survival Classifier prediction
        if os.path.exists(CLASSIFIER_PATH):
            res = self.service.predict_survival(auth_header, passenger)
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["authenticated_as"], "mluser")
            self.assertIn("survived", res["prediction"])
            self.assertIn("survival_probability", res["prediction"])
            self.assertIn(res["prediction"]["survived"], [0, 1])

        # Test Fare Regressor prediction
        if os.path.exists(REGRESSOR_PATH):
            fare_res = self.service.predict_fare(auth_header, passenger)
            self.assertEqual(fare_res["status"], "success")
            self.assertEqual(fare_res["authenticated_as"], "mluser")
            self.assertIn("predicted_fare", fare_res["prediction"])
            self.assertGreaterEqual(fare_res["prediction"]["predicted_fare"], 0.0)


if __name__ == "__main__":
    unittest.main()
