from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Annotated, Dict, Any
from firebase_admin import auth as firebase_auth
from .firebase import initialize_firebase_admin

bearer_scheme = HTTPBearer(auto_error=False)

async def get_current_user(
    token_credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)]
) -> Dict[str, Any]:
    try:
        initialize_firebase_admin()
    except Exception as e:
        print(f"[auth] Firebase init error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server config error")

    if not token_credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Bearer token missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    id_token = token_credentials.credentials
    try:
        decoded = firebase_auth.verify_id_token(id_token)
        return decoded
    except Exception as e:
        print(f"[auth] token verify error: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token.")
