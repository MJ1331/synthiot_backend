from fastapi import APIRouter, HTTPException, status, Depends
from ..schemas import SignupRequest, SignupResponse, UserDetailsRequest, UserDetailsResponse
from ..firebase import fs_client
from firebase_admin import auth as firebase_auth
from datetime import datetime
from ..auth import get_current_user

router = APIRouter()

def user_details_collection():
    return fs_client().collection("user_details")

@router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
async def signup(payload: SignupRequest):
    try:
        user_rec = firebase_auth.create_user(
            email=payload.email,
            password=payload.password,
            display_name=payload.display_name or None
        )
    except Exception as e:
        msg = str(e)
        print(f"[user.signup] auth create error: {msg}")
        if "EMAIL_EXISTS" in msg or "email already exists" in msg.lower():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already in use")
        if "Password should be at least" in msg or "WEAK_PASSWORD" in msg:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Weak password")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create user")

    uid = user_rec.uid
    try:
        user_details_collection().document(uid).set({
            "uid": uid,
            "email": payload.email,
            "display_name": payload.display_name or "",
            "created_at": datetime.utcnow().isoformat() + "Z"
        }, merge=True)
    except Exception as e:
        print(f"[user.signup] firestore write failed: {e} - rolling back auth user")
        try:
            firebase_auth.delete_user(uid)
        except Exception:
            pass
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create user record")

    return SignupResponse(uid=uid, email=payload.email, display_name=payload.display_name, created_at=datetime.utcnow().isoformat() + "Z")

@router.post("/user_details", response_model=UserDetailsResponse)
async def set_user_details(payload: UserDetailsRequest, user_claims: dict = Depends(get_current_user)):
    uid = user_claims.get("uid")
    email = user_claims.get("email")
    if not uid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token")
    try:
        user_details_collection().document(uid).set({
            "uid": uid,
            "email": email or "",
            "display_name": payload.display_name,
            "created_at": datetime.utcnow().isoformat() + "Z"
        }, merge=True)
        return UserDetailsResponse(uid=uid, email=email, display_name=payload.display_name, updated_at=datetime.utcnow().isoformat() + "Z")
    except Exception as e:
        print(f"[user] set_user_details error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to store user details")

@router.get("/user_details", response_model=UserDetailsResponse)
async def get_user_details(user_claims: dict = Depends(get_current_user)):
    uid = user_claims.get("uid")
    email = user_claims.get("email")
    if not uid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token")
    try:
        doc = user_details_collection().document(uid).get()
        data = doc.to_dict() if doc.exists else {}
        return UserDetailsResponse(uid=uid, email=data.get("email", email), display_name=data.get("display_name") or None, updated_at=data.get("created_at", datetime.utcnow().isoformat() + "Z"))
    except Exception as e:
        print(f"[user] get_user_details error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to read user details")
