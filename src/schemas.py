from pydantic import BaseModel, EmailStr, validator
from typing import Optional, Dict, Any

class CreateProjectRequest(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectResponse(BaseModel):
    id: str
    owner_uid: str
    name: str
    description: Optional[str] = None
    created_at: str

class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    display_name: Optional[str] = None

    @validator("password")
    def min_password(cls, v):
        if not v or len(v) < 6:
            raise ValueError("Password must be at least 6 characters.")
        return v

class SignupResponse(BaseModel):
    uid: str
    email: EmailStr
    display_name: Optional[str] = None
    created_at: str

class UserDetailsRequest(BaseModel):
    display_name: str

class UserDetailsResponse(BaseModel):
    uid: str
    email: Optional[EmailStr] = None
    display_name: Optional[str] = None
    updated_at: str

class ChatRequest(BaseModel):
    prompt: str
    rows: Optional[int] = None
    freq_seconds: Optional[int] = None

class ChatResponse(BaseModel):
    generation_id: str
    status: str

class GenerationRecord(BaseModel):
    generation_id: str
    prompt: str
    params: Dict[str, Any]
    created_at: str
    status: str
