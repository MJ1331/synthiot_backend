from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from ..schemas import CreateProjectRequest, ProjectResponse
from ..auth import get_current_user
from ..firebase import fs_client
import uuid
from datetime import datetime

router = APIRouter()

def projects_collection():
    return fs_client().collection("project_details")

@router.get("/", response_model=List[ProjectResponse])
async def get_projects(user_claims: dict = Depends(get_current_user)):
    uid = user_claims.get("uid")
    if not uid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token")
    try:
        docs = projects_collection().where("owner_uid", "==", uid).stream()
        projects = []
        for d in docs:
            data = d.to_dict() or {}
            projects.append({
                "id": d.id,
                "owner_uid": data.get("owner_uid", ""),
                "name": data.get("name", ""),
                "description": data.get("description", ""),
                "created_at": data.get("created_at", ""),
            })
        projects.sort(key=lambda p: p.get("created_at") or "", reverse=True)
        return projects
    except Exception as e:
        print(f"[projects] read error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to read projects")

@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(payload: CreateProjectRequest, user_claims: dict = Depends(get_current_user)):
    uid = user_claims.get("uid")
    if not uid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token")
    name = (payload.name or "").strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Project name required")
    proj_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat() + "Z"
    try:
        projects_collection().document(proj_id).set({
            "owner_uid": uid,
            "name": name,
            "description": payload.description or "",
            "created_at": created_at,
        })
        return {"id": proj_id, "owner_uid": uid, "name": name, "description": payload.description or "", "created_at": created_at}
    except Exception as e:
        print(f"[projects] create error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create project")
