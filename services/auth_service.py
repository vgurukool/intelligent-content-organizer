import os
import time
import base64
import hashlib
import secrets
import logging
from typing import Optional, Dict, Any, Tuple
import httpx
import jwt

logger = logging.getLogger(__name__)

class KeycloakAuthService:
    def __init__(self):
        self.keycloak_public_url = os.getenv("KEYCLOAK_PUBLIC_URL", "https://platform.vgurukool.com/keycloak").rstrip("/")
        self.keycloak_internal_url = os.getenv("KEYCLOAK_INTERNAL_URL", "http://keycloak.keycloak.svc.cluster.local:80/keycloak").rstrip("/")
        self.realm = os.getenv("KEYCLOAK_REALM", "cnoe")
        self.client_id = os.getenv("KEYCLOAK_CLIENT_ID", "vgurukool-apps")
        self.redirect_uri = os.getenv("KEYCLOAK_REDIRECT_URI", "https://incorg.vgurukool.com/auth/callback")
        self.session_secret = os.getenv("SESSION_SECRET", "vgurukool-incorg-session-token-secret-2026")
        self.cookie_name = "incorg_session"

        # Endpoints
        self.auth_endpoint = f"{self.keycloak_public_url}/realms/{self.realm}/protocol/openid-connect/auth"
        self.token_endpoint = f"{self.keycloak_internal_url}/realms/{self.realm}/protocol/openid-connect/token"
        self.token_endpoint_public = f"{self.keycloak_public_url}/realms/{self.realm}/protocol/openid-connect/token"
        self.logout_endpoint = f"{self.keycloak_public_url}/realms/{self.realm}/protocol/openid-connect/logout"

    def _generate_pkce(self) -> Tuple[str, str]:
        """Generate PKCE code_verifier and code_challenge."""
        verifier = secrets.token_urlsafe(48)
        digest = hashlib.sha256(verifier.encode("utf-8")).digest()
        challenge = base64.urlsafe_b64encode(digest).decode("utf-8").replace("=", "")
        return verifier, challenge

    def get_login_url(self) -> Tuple[str, str, str]:
        """
        Returns (authorization_url, code_verifier, state)
        """
        verifier, challenge = self._generate_pkce()
        state = secrets.token_urlsafe(16)
        
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "scope": "openid profile email roles",
            "redirect_uri": self.redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256"
        }
        query_str = "&".join(f"{k}={v}" for k, v in params.items())
        login_url = f"{self.auth_endpoint}?{query_str}"
        return login_url, verifier, state

    async def exchange_code_for_tokens(self, code: str, verifier: str) -> Optional[Dict[str, Any]]:
        """Exchange authorization code with Keycloak using PKCE verifier."""
        payload = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "code": code,
            "redirect_uri": self.redirect_uri,
            "code_verifier": verifier
        }
        
        # Try internal endpoint first, fallback to public endpoint
        endpoints = [self.token_endpoint, self.token_endpoint_public]
        for ep in endpoints:
            try:
                async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
                    resp = await client.post(ep, data=payload)
                    if resp.status_code == 200:
                        return resp.json()
                    else:
                        logger.warning(f"Keycloak token exchange at {ep} failed with status {resp.status_code}: {resp.text}")
            except Exception as e:
                logger.warning(f"Keycloak token exchange error at {ep}: {e}")

        return None

    def create_session_jwt(self, user_info: Dict[str, Any]) -> str:
        """Create signed JWT session token valid for 7 days."""
        now = int(time.time())
        claims = {
            "sub": user_info.get("sub", "unknown"),
            "preferred_username": user_info.get("preferred_username", "anonymous"),
            "name": user_info.get("name") or user_info.get("preferred_username", "User"),
            "email": user_info.get("email", ""),
            "roles": user_info.get("roles", ["user"]),
            "iat": now,
            "exp": now + (7 * 24 * 3600)  # 7 days
        }
        return jwt.encode(claims, self.session_secret, algorithm="HS256")

    def verify_session_jwt(self, token_str: str) -> Optional[Dict[str, Any]]:
        """Verify and decode session JWT."""
        try:
            return jwt.decode(token_str, self.session_secret, algorithms=["HS256"])
        except Exception:
            return None

    def parse_user_from_token(self, token_response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user profile and roles from Keycloak access_token or id_token."""
        access_token = token_response.get("access_token")
        id_token = token_response.get("id_token")

        user_info = {
            "preferred_username": "user",
            "name": "User",
            "email": "",
            "roles": ["user"]
        }

        # Try to parse id_token first for profile info
        if id_token:
            try:
                decoded_id = jwt.decode(id_token, options={"verify_signature": False})
                user_info["sub"] = decoded_id.get("sub")
                user_info["preferred_username"] = decoded_id.get("preferred_username") or user_info["preferred_username"]
                user_info["name"] = decoded_id.get("name") or user_info["preferred_username"]
                user_info["email"] = decoded_id.get("email", "")
            except Exception as e:
                logger.debug(f"Error decoding id_token: {e}")

        # Parse access_token for realm_access.roles
        if access_token:
            try:
                decoded_access = jwt.decode(access_token, options={"verify_signature": False})
                realm_access = decoded_access.get("realm_access", {})
                roles = realm_access.get("roles", [])
                if roles:
                    user_info["roles"] = roles
                if not user_info.get("preferred_username") or user_info["preferred_username"] == "user":
                    user_info["preferred_username"] = decoded_access.get("preferred_username", "user")
                    user_info["name"] = decoded_access.get("name") or user_info["preferred_username"]
            except Exception as e:
                logger.debug(f"Error decoding access_token: {e}")

        return user_info

    def get_logout_url(self) -> str:
        """Keycloak end_session_endpoint URL with redirect to incorg."""
        return f"{self.logout_endpoint}?client_id={self.client_id}&post_logout_redirect_uri=https://incorg.vgurukool.com/"

# Global singleton instance
auth_service = KeycloakAuthService()
