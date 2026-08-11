"""
app.py
------
Service interface and REST API layer combining Authentication with ML Model inference.
Provides HTTP endpoints for /api/register, /api/login, /api/profile, and protected /api/predict.
"""

from __future__ import annotations

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import joblib
import pandas as pd

from src.auth import AuthManager
from src.database import DatabaseManager, DEFAULT_DB_PATH

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
CLASSIFIER_PATH = os.path.join(DEFAULT_MODELS_DIR, "best_classifier.joblib")
REGRESSOR_PATH = os.path.join(DEFAULT_MODELS_DIR, "fare_regressor.joblib")


class AppService:
    """Service wrapper integrating AuthManager and ML Model predictions."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        classifier_path: str = CLASSIFIER_PATH,
        regressor_path: str = REGRESSOR_PATH,
    ):
        self.db = DatabaseManager(db_path)
        self.auth = AuthManager(self.db)
        self.classifier_path = classifier_path
        self.regressor_path = regressor_path
        self.classifier = None
        self.regressor = None
        self._load_models()

    def _load_models(self) -> None:
        """Loads serialized joblib ML models if they exist."""
        if os.path.exists(self.classifier_path):
            try:
                self.classifier = joblib.load(self.classifier_path)
            except Exception as e:
                print(f"[Warning] Failed to load classifier: {e}")
        
        if os.path.exists(self.regressor_path):
            try:
                self.regressor = joblib.load(self.regressor_path)
            except Exception as e:
                print(f"[Warning] Failed to load regressor: {e}")

    def register(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Registers a user from payload containing username, email, password."""
        username = payload.get("username", "")
        email = payload.get("email", "")
        password = payload.get("password", "")
        user_info = self.auth.register_user(username, email, password)
        return {"status": "success", "message": "User registered successfully.", "user": user_info}

    def login(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Logs in a user from payload containing username and password."""
        username = payload.get("username", "")
        password = payload.get("password", "")
        auth_data = self.auth.login_user(username, password)
        return {"status": "success", "message": "Login successful.", "data": auth_data}

    def get_profile(self, auth_header: Optional[str]) -> Dict[str, Any]:
        """Returns authenticated user profile."""
        token = self._extract_token(auth_header)
        user = self.auth.authenticate_token(token)
        return {"status": "success", "user": user}

    def _prepare_input_df(self, passenger_data: Dict[str, Any], model: Any) -> pd.DataFrame:
        """Normalizes input payload keys to match trained model feature names."""
        data = dict(passenger_data)
        # Normalize column aliases
        if "ticket_class" in data and "pclass" not in data:
            data["pclass"] = data["ticket_class"]
        if "sublings" in data and "sibsp" not in data:
            data["sibsp"] = data["sublings"]
        
        df = pd.DataFrame([data])
        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
            # Select expected columns
            df = df[expected_cols]
        return df

    def predict_survival(self, auth_header: Optional[str], passenger_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Protected Endpoint: Runs survival prediction using trained Random Forest model.
        """
        token = self._extract_token(auth_header)
        user = self.auth.authenticate_token(token)

        if self.classifier is None:
            raise RuntimeError("Classifier model is not available.")

        df = self._prepare_input_df(passenger_data, self.classifier)
        prediction = int(self.classifier.predict(df)[0])
        probability = float(self.classifier.predict_proba(df)[0][1])

        return {
            "status": "success",
            "authenticated_as": user["username"],
            "prediction": {
                "survived": prediction,
                "survival_label": "Survived" if prediction == 1 else "Did Not Survive",
                "survival_probability": round(probability, 4),
            },
        }

    def predict_fare(self, auth_header: Optional[str], passenger_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Protected Endpoint: Runs fare regression prediction using trained LinearRegression model.
        """
        token = self._extract_token(auth_header)
        user = self.auth.authenticate_token(token)

        if self.regressor is None:
            raise RuntimeError("Regressor model is not available.")

        df = self._prepare_input_df(passenger_data, self.regressor)
        predicted_fare = float(self.regressor.predict(df)[0])

        return {
            "status": "success",
            "authenticated_as": user["username"],
            "prediction": {
                "predicted_fare": round(max(0.0, predicted_fare), 2),
            },
        }

    def _extract_token(self, auth_header: Optional[str]) -> str:
        """Extracts Bearer token from HTTP Authorization header."""
        if not auth_header:
            raise ValueError("Authorization header missing.")
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
        elif len(parts) == 1:
            return parts[0]
        raise ValueError("Invalid Authorization header format. Expected 'Bearer <token>'.")


class AuthHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler exposing authentication & prediction REST endpoints."""

    app_service: Optional[AppService] = None

    def _send_json_response(self, status_code: int, data: Dict[str, Any]) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self) -> None:
        self._send_json_response(200, {"status": "ok"})

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        service = self.app_service or AppService()

        if parsed_url.path == "/api/profile":
            auth_header = self.headers.get("Authorization")
            try:
                result = service.get_profile(auth_header)
                self._send_json_response(200, result)
            except ValueError as e:
                self._send_json_response(401, {"status": "error", "message": str(e)})
            except Exception as e:
                self._send_json_response(500, {"status": "error", "message": str(e)})
        else:
            self._send_json_response(404, {"status": "error", "message": "Endpoint not found."})

    def do_POST(self) -> None:
        parsed_url = urlparse(self.path)
        service = self.app_service or AppService()

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(post_data.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json_response(400, {"status": "error", "message": "Invalid JSON body."})
            return

        auth_header = self.headers.get("Authorization")

        try:
            if parsed_url.path == "/api/register":
                result = service.register(payload)
                self._send_json_response(201, result)

            elif parsed_url.path == "/api/login":
                result = service.login(payload)
                self._send_json_response(200, result)

            elif parsed_url.path == "/api/predict/survival":
                result = service.predict_survival(auth_header, payload)
                self._send_json_response(200, result)

            elif parsed_url.path == "/api/predict/fare":
                result = service.predict_fare(auth_header, payload)
                self._send_json_response(200, result)

            else:
                self._send_json_response(404, {"status": "error", "message": "Endpoint not found."})

        except ValueError as e:
            self._send_json_response(400 if "Validation" in str(e) or "Username" in str(e) or "Password" in str(e) else 401, {"status": "error", "message": str(e)})
        except RuntimeError as e:
            self._send_json_response(503, {"status": "error", "message": str(e)})
        except Exception as e:
            self._send_json_response(500, {"status": "error", "message": str(e)})


def run_server(host: str = "127.0.0.1", port: int = 8000, db_path: str = DEFAULT_DB_PATH) -> None:
    """Runs the REST API server."""
    AuthHTTPRequestHandler.app_service = AppService(db_path=db_path)
    server_address = (host, port)
    httpd = HTTPServer(server_address, AuthHTTPRequestHandler)
    print(f"Serving Auth & ML API on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
