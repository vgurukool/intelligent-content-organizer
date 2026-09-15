import logging
from typing import Optional
from fastapi import APIRouter, Request, Response, HTTPException, status
from fastapi.responses import RedirectResponse, JSONResponse
from services.auth_service import auth_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Authentication"])

@router.get("/login")
def login(request: Request, response: Response):
    """Initiates Keycloak OIDC authorization code flow with PKCE."""
    try:
        login_url, verifier, state = auth_service.get_login_url()
        redirect_resp = RedirectResponse(url=login_url, status_code=status.HTTP_302_FOUND)
        
        # Store PKCE verifier and state in temporary short-lived cookies
        redirect_resp.set_cookie(
            key="pkce_verifier",
            value=verifier,
            httponly=True,
            max_age=300,  # 5 minutes
            samesite="lax",
            secure=False
        )
        redirect_resp.set_cookie(
            key="pkce_state",
            value=state,
            httponly=True,
            max_age=300,
            samesite="lax",
            secure=False
        )
        return redirect_resp
    except Exception as e:
        logger.error(f"Failed to generate login URL: {e}")
        raise HTTPException(status_code=500, detail="SSO authorization initiation failed")

@router.get("/callback")
async def auth_callback(request: Request, code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    """Handles Keycloak redirect callback and token exchange."""
    if error:
        logger.warning(f"Keycloak returned auth error: {error}")
        return RedirectResponse(url="/?auth_error=" + error)

    if not code:
        return RedirectResponse(url="/?auth_error=missing_code")

    saved_verifier = request.cookies.get("pkce_verifier")
    saved_state = request.cookies.get("pkce_state")

    if not saved_verifier:
        logger.warning("Missing PKCE verifier cookie during callback")

    try:
        token_response = await auth_service.exchange_code_for_tokens(
            code=code,
            verifier=saved_verifier or ""
        )

        if not token_response or "access_token" not in token_response:
            logger.error(f"Failed to exchange code for tokens: {token_response}")
            return RedirectResponse(url="/?auth_error=token_exchange_failed")

        user_info = auth_service.parse_user_from_token(token_response)
        session_jwt = auth_service.create_session_jwt(user_info)

        redirect_resp = RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
        
        # Set persistent secure session cookie
        redirect_resp.set_cookie(
            key=auth_service.cookie_name,
            value=session_jwt,
            httponly=True,
            max_age=7 * 24 * 3600,  # 7 days
            samesite="lax",
            secure=False
        )
        # Clear temporary PKCE cookies
        redirect_resp.delete_cookie("pkce_verifier")
        redirect_resp.delete_cookie("pkce_state")

        logger.info(f"User '{user_info.get('preferred_username')}' successfully authenticated via Keycloak SSO")
        return redirect_resp

    except Exception as e:
        logger.error(f"Exception during Keycloak callback handling: {e}")
        return RedirectResponse(url="/?auth_error=callback_exception")

@router.get("/logout")
def logout(request: Request, response: Response):
    """Logs user out of incorg and redirects to Keycloak end session endpoint."""
    logout_url = auth_service.get_logout_url()
    redirect_resp = RedirectResponse(url=logout_url, status_code=status.HTTP_302_FOUND)
    redirect_resp.delete_cookie(auth_service.cookie_name)
    redirect_resp.delete_cookie("pkce_verifier")
    redirect_resp.delete_cookie("pkce_state")
    return redirect_resp

@router.get("/api/v1/auth/me")
def get_current_user(request: Request):
    """Returns profile of currently logged-in user, or unauthenticated."""
    cookie_token = request.cookies.get(auth_service.cookie_name)
    if cookie_token:
        user_data = auth_service.verify_session_jwt(cookie_token)
        if user_data:
            return {
                "authenticated": True,
                "username": user_data.get("preferred_username"),
                "name": user_data.get("name"),
                "email": user_data.get("email"),
                "roles": user_data.get("roles", [])
            }
    return {
        "authenticated": False,
        "username": None,
        "name": "Guest",
        "roles": []
    }
