# backend/main.py

import os
import json
import math
import random
import uuid
from typing import Annotated, Dict, Any, Optional, List, Iterator

from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr, validator
from dateutil import parser as dateparser

# Firebase Admin
import firebase_admin
from firebase_admin import credentials, auth, firestore
from firebase_admin.exceptions import FirebaseError

# Groq (LLM)
from groq import Groq

# Numerical helpers
import numpy as np

# I/O
import io, csv

# ---------------- CONFIG ----------------
SERVICE_ACCOUNT_KEY_PATH = "firebase_service_account.json"  
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]

# Groq API key (prefer environment variable)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # MUST set this before starting server
# Example model name - change if needed or available in your account
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---------------- Firebase init (Auth + Firestore) ----------------
def initialize_firebase_admin():
    try:
        firebase_admin.get_app()
        return
    except ValueError:
        pass
    except Exception as e:
        raise RuntimeError(f"FATAL ERROR during Firebase check: {e}")

    try:
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
        firebase_admin.initialize_app(cred)
        # warm Firestore client
        firestore.client()
        print("--- Firebase Admin SDK (Auth + Firestore) initialized ---")
    except FileNotFoundError:
        raise RuntimeError(f"FATAL ERROR: Service account key not found at {SERVICE_ACCOUNT_KEY_PATH}")
    except Exception as e:
        raise RuntimeError(f"FATAL ERROR initializing Firebase: {e}")

# ---------------- Auth dependency ----------------
bearer_scheme = HTTPBearer(auto_error=False)

async def get_current_user(
    token_credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)]
) -> Dict[str, Any]:
    try:
        initialize_firebase_admin()
    except RuntimeError as e:
        print(f"FATAL CONFIGURATION ERROR: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server config error")

    if not token_credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Bearer token missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    id_token = token_credentials.credentials
    try:
        decoded = auth.verify_id_token(id_token)
        return decoded
    except FirebaseError as e:
        print(f"Firebase Authentication Error: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token.")
    except Exception as e:
        print(f"Unexpected Token Verification Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Token verification failed.")

# ---------------- FastAPI app + CORS ----------------
app = FastAPI(
    title="SynthIoT API (Firebase + Groq + On-demand CSV)",
    version="1.0.0",
    description="Protected API demonstrating Firebase ID Token verification + synthetic temp generation.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Pydantic models ----------------
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
    rows: Optional[int] = None           # optional override
    freq_seconds: Optional[int] = None  # optional override (seconds)

class ChatResponse(BaseModel):
    generation_id: str
    status: str

class GenerationRecord(BaseModel):
    generation_id: str
    prompt: str
    params: Dict[str, Any]
    created_at: str
    status: str

# ---------------- Firestore helpers ----------------
def _fs_client():
    initialize_firebase_admin()
    return firestore.client()

def _project_ref(project_id: str):
    return _fs_client().collection("project_details").document(project_id)

def _generation_ref(project_id: str, gen_id: str):
    return _project_ref(project_id).collection("generations").document(gen_id)

def projects_collection():
    return _fs_client().collection("project_details")

def user_details_collection():
    return _fs_client().collection("user_details")

def assert_user_owns_project(uid: str, project_id: str):
    doc = _project_ref(project_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Project not found")
    data = doc.to_dict() or {}
    if data.get("owner_uid") != uid:
        raise HTTPException(status_code=403, detail="Not authorized for this project")
    return data

def create_project_for_user(uid: str, name: str, description: Optional[str]) -> Dict[str, Any]:
    coll = projects_collection()
    proj_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat() + "Z"
    doc = coll.document(proj_id)
    doc.set({
        "owner_uid": uid,
        "name": name,
        "description": description or "",
        "created_at": created_at,
    })
    return {"id": proj_id, "owner_uid": uid, "name": name, "description": description or "", "created_at": created_at}

def list_projects_for_user(uid: str) -> List[Dict[str, Any]]:
    coll = projects_collection()
    try:
        docs = coll.where("owner_uid", "==", uid).stream()
    except Exception as e:
        print(f"[Firestore] query error for uid={uid}: {e}")
        raise
    projects: List[Dict[str, Any]] = []
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

def create_user_record_in_firestore(uid: str, email: str, display_name: Optional[str]) -> None:
    coll = user_details_collection()
    now = datetime.utcnow().isoformat() + "Z"
    try:
        coll.document(uid).set({
            "uid": uid,
            "email": email,
            "display_name": display_name or "",
            "created_at": now,
        }, merge=True)
    except Exception as e:
        print(f"[Firestore] error writing user_details for uid={uid}: {e}")
        raise

def get_user_details_from_firestore(uid: str) -> Dict[str, Any]:
    doc = user_details_collection().document(uid).get()
    if not doc.exists:
        return {}
    return doc.to_dict() or {}

# ---------------- Groq client & parsing ----------------
def build_groq_client():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable not set.")
    return Groq(api_key=GROQ_API_KEY)

def parse_prompt_with_groq(prompt: str, rows_hint: Optional[int]=None, freq_hint: Optional[int]=None) -> Dict[str, Any]:
    """
    Call Groq to parse prompt into structured params JSON.
    Fallback to local heuristics if Groq fails.
    """
    try:
        client = build_groq_client()
        system_msg = {
            "role": "system",
            "content": (
                "You are a precise JSON-output assistant. When given a prompt specifying synthetic temperature generation, "
                "output only valid JSON (no extra text) with fields: location (string), start_iso, end_iso, rows (int), freq_seconds (int), "
                "weather (clear|rain|snow|normal), temp_range (array [min,max] optional). Omit fields not specified."
            )
        }
        user_msg = {"role": "user", "content": f'Parse this prompt into JSON: "{prompt}". Hints: rows={rows_hint}, freq_seconds={freq_hint}.'}
        resp = client.chat.completions.create(messages=[system_msg, user_msg], model=GROQ_MODEL, temperature=0.0, max_tokens=512)
        text = resp.choices[0].message.content
        try:
            parsed = json.loads(text)
        except Exception:
            start = text.find('{'); end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                parsed = json.loads(text[start:end+1])
            else:
                raise ValueError("Groq returned non-JSON")
    except Exception as e:
        print(f"[Groq parse error] {e} -- falling back to local parser")
        parsed = parse_prompt_to_params(prompt, rows_hint=rows_hint, freq_hint=freq_hint)

    # set defaults
    parsed.setdefault('rows', parsed.get('rows', 24))
    parsed.setdefault('freq_seconds', parsed.get('freq_seconds', 3600))
    if 'start_iso' not in parsed or 'end_iso' not in parsed:
        rows = int(parsed['rows'])
        freq_s = int(parsed['freq_seconds'])
        end = datetime.utcnow()
        start = end - timedelta(seconds=(rows - 1) * freq_s)
        parsed.setdefault('start_iso', start.isoformat() + "Z")
        parsed.setdefault('end_iso', end.isoformat() + "Z")
    parsed.setdefault('seed', random.randint(0, 2**31 - 1))
    parsed.setdefault('weather', parsed.get('weather', 'normal'))
    return parsed

def estimate_temp_range_with_groq(params: Dict[str, Any]) -> List[float]:
    """
    Ask Groq to estimate a reasonable temperature range given location/time/weather.
    Fallback heuristics if Groq is unavailable.
    """
    try:
        client = build_groq_client()
        desc = []
        if params.get('location'):
            desc.append(f"location: {params['location']}")
        if params.get('start_iso') and params.get('end_iso'):
            desc.append(f"time window: {params['start_iso']} to {params['end_iso']}")
        desc.append(f"weather: {params.get('weather','normal')}")
        system_msg = {"role": "system", "content": "Return only a JSON array [min,max] representing a reasonable temperature range in °C."}
        user_msg = {"role": "user", "content": "Provide a temperature range for: " + "; ".join(desc)}
        resp = client.chat.completions.create(messages=[system_msg, user_msg], model=GROQ_MODEL, temperature=0.2, max_tokens=80)
        text = resp.choices[0].message.content.strip()
        try:
            arr = json.loads(text)
        except Exception:
            s = text.find('['); e = text.rfind(']')
            if s != -1 and e != -1 and e > s:
                arr = json.loads(text[s:e+1])
            else:
                raise ValueError("Groq range parse failed")
        if isinstance(arr, list) and len(arr) >= 2:
            tmin = float(arr[0]); tmax = float(arr[1])
            if tmin > tmax:
                tmin, tmax = tmax, tmin
            return [tmin, tmax]
    except Exception as e:
        print(f"[Groq range error] {e} -- falling back to heuristics")

    w = params.get('weather', 'normal')
    if w == 'clear':
        return [18.0, 32.0]
    if w == 'rain':
        return [12.0, 24.0]
    return [15.0, 28.0]

# ---------------- Prompt parsing fallback (your existing heuristic parser) ----------------
def parse_prompt_to_params(prompt: str, rows_hint: Optional[int] = None, freq_hint: Optional[int] = None) -> Dict[str, Any]:
    p = prompt.lower()
    params: Dict[str, Any] = {}
    import re
    m = re.search(r'(\d+)\s*(rows|samples|points)', prompt, re.I)
    if m:
        params['rows'] = int(m.group(1))
    elif rows_hint:
        params['rows'] = int(rows_hint)
    mdate = re.search(r'(\d{4}-\d{2}-\d{2})', prompt)
    if mdate:
        day = mdate.group(1)
        params['start_iso'] = f"{day}T00:00:00Z"
        params['end_iso'] = f"{day}T23:59:59Z"
        params.setdefault('rows', 24)
        params.setdefault('freq_seconds', 3600)
    if 'hour' in p or 'hourly' in p:
        params.setdefault('freq_seconds', 3600)
    mmin = re.search(r'every\s+(\d+)\s*(minute|minutes|min)', p)
    if mmin:
        params['freq_seconds'] = int(mmin.group(1)) * 60
    if freq_hint:
        params.setdefault('freq_seconds', int(freq_hint))
    mloc = re.search(r'(in|for)\s+([A-Za-z][A-Za-z ,.]+)', prompt, re.I)
    if mloc:
        location = mloc.group(2).strip().split(' for ')[0].split(' in ')[0].strip()
        params['location'] = location
    if 'rain' in p or 'storm' in p or 'snow' in p:
        params['weather'] = 'rain'
    elif 'clear' in p or 'sunny' in p:
        params['weather'] = 'clear'
    else:
        params['weather'] = 'normal'
    mrange = re.search(r'(-?\d+(\.\d+)?)\s*(?:to|-)\s*(-?\d+(\.\d+)?)\s*(?:°?c|c|deg|degrees)?', prompt, re.I)
    if mrange:
        tmin = float(mrange.group(1)); tmax = float(mrange.group(3))
        params['temp_range'] = [min(tmin, tmax), max(tmin, tmax)]
    params.setdefault('rows', params.get('rows', 24))
    params.setdefault('freq_seconds', params.get('freq_seconds', 3600))
    params.setdefault('start_iso', params.get('start_iso', (datetime.utcnow() - timedelta(hours=params['rows'] * (params['freq_seconds'] / 3600))).isoformat() + "Z"))
    params.setdefault('end_iso', params.get('end_iso', datetime.utcnow().isoformat() + "Z"))
    params.setdefault('seed', random.randint(0, 2**31 - 1))
    return params

# ---------------- Temperature generator (iterator) ----------------
def build_timestamps(params: Dict[str, Any]) -> List[datetime]:
    try:
        start = dateparser.parse(params['start_iso'])
        end = dateparser.parse(params['end_iso'])
        freq = timedelta(seconds=int(params.get('freq_seconds', 3600)))
        timestamps = []
        t = start
        max_points = 200000
        while t <= end and len(timestamps) < max_points:
            timestamps.append(t)
            t += freq
        return timestamps
    except Exception:
        rows = int(params.get('rows', 24))
        freq_s = int(params.get('freq_seconds', 3600))
        start = datetime.utcnow()
        return [start + timedelta(seconds=i * freq_s) for i in range(rows)]

def diurnal_baseline(ts: datetime, params: Dict[str, Any]) -> float:
    if 'temp_range' in params:
        tmin, tmax = params['temp_range']
    else:
        w = params.get('weather', 'normal')
        if w == 'clear':
            tmin, tmax = 18.0, 32.0
        elif w == 'rain':
            tmin, tmax = 12.0, 24.0
        else:
            tmin, tmax = 15.0, 28.0
    mean = (tmin + tmax) / 2.0
    amplitude = (tmax - tmin) / 2.0
    hour = ts.hour + ts.minute / 60.0
    phase = 15.0
    return mean + amplitude * math.sin(2 * math.pi * (hour - phase) / 24.0)

def generate_temperature_rows_iter(params: Dict[str, Any]) -> Iterator[List[Any]]:
    timestamps = build_timestamps(params)
    if not timestamps:
        return
    if 'temp_range' in params and isinstance(params['temp_range'], (list, tuple)) and len(params['temp_range']) >= 2:
        tmin, tmax = float(params['temp_range'][0]), float(params['temp_range'][1])
    else:
        tmin, tmax = estimate_temp_range_with_groq(params)
    seed = int(params.get('seed', 0))
    rnd = np.random.default_rng(seed)
    phi = 0.88
    sigma_proc = 0.25
    sigma_sensor = 0.45
    prev_noise = rnd.normal(0, sigma_proc)
    p_spike_start = 0.003
    p_spike_end = 0.2
    in_spike = False
    spike_effect = 0.0
    for t in timestamps:
        baseline = diurnal_baseline(t, params)
        if in_spike:
            if rnd.random() < p_spike_end:
                in_spike = False
                spike_effect = 0.0
        else:
            if rnd.random() < p_spike_start:
                in_spike = True
                spike_effect = rnd.choice([rnd.uniform(2.0, 6.0), -rnd.uniform(2.0, 6.0)])
        proc_noise = phi * prev_noise + rnd.normal(0, sigma_proc)
        prev_noise = proc_noise
        sensor_noise = rnd.normal(0, sigma_sensor)
        temp = baseline + proc_noise + sensor_noise
        if in_spike:
            temp += spike_effect
        temp = round(float(temp), 1)
        yield [t.isoformat() + "Z", temp]

# ---------------- CSV streaming utility ----------------
def csv_stream_generator(row_iterator: Iterator[List[Any]]):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["timestamp", "temperature"])
    yield buf.getvalue()
    buf.seek(0); buf.truncate(0)
    for row in row_iterator:
        writer.writerow(row)
        yield buf.getvalue()
        buf.seek(0); buf.truncate(0)

# ---------------- Endpoints: auth, signup, projects, user_details (keep your existing behavior) ----------------
@app.get("/")
async def root():
    return {"message": "API is running."}

@app.get("/protected_data")
async def protected_data(user_claims: Annotated[Dict[str, Any], Depends(get_current_user)]):
    return {"message": "ok", "uid": user_claims.get("uid"), "email": user_claims.get("email")}

@app.get("/projects", response_model=List[ProjectResponse])
async def get_projects(user_claims: Annotated[Dict[str, Any], Depends(get_current_user)]):
    uid = user_claims.get("uid")
    print(f"[INFO] GET /projects by uid={uid}")
    if not uid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token")
    try:
        projects = list_projects_for_user(uid)
        print(f"[INFO] found {len(projects)} projects for uid={uid}")
        return projects
    except Exception as e:
        print(f"Error reading projects: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to read projects")

@app.post("/projects", status_code=status.HTTP_201_CREATED, response_model=ProjectResponse)
async def post_project(payload: CreateProjectRequest, user_claims: Annotated[Dict[str, Any], Depends(get_current_user)]):
    uid = user_claims.get("uid")
    if not uid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token")
    name = (payload.name or "").strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Project name required")
    project = create_project_for_user(uid, name, payload.description)
    return project

@app.post("/signup", status_code=status.HTTP_201_CREATED, response_model=SignupResponse)
async def signup(payload: SignupRequest):
    try:
        initialize_firebase_admin()
    except RuntimeError as e:
        print(f"FATAL: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server config error")
    try:
        user_rec = auth.create_user(
            email=payload.email,
            password=payload.password,
            display_name=payload.display_name or None
        )
    except Exception as e:
        msg = str(e)
        print(f"[Auth] create_user error: {msg}")
        if "EMAIL_EXISTS" in msg or "email already exists" in msg.lower():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already in use")
        if "Password should be at least" in msg or "WEAK_PASSWORD" in msg:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Weak password")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create user")
    uid = user_rec.uid
    try:
        create_user_record_in_firestore(uid=uid, email=payload.email, display_name=payload.display_name)
    except Exception as e:
        print(f"[Firestore] failed to write user_details for uid={uid}: {e} - rolling back auth user")
        try:
            auth.delete_user(uid)
        except Exception as e2:
            print(f"[Auth] failed rollback delete for uid={uid}: {e2}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create user record")
    created_at = datetime.utcnow().isoformat() + "Z"
    return SignupResponse(uid=uid, email=payload.email, display_name=payload.display_name, created_at=created_at)

@app.post("/user_details", response_model=UserDetailsResponse)
async def set_user_details(payload: UserDetailsRequest, user_claims: Annotated[Dict[str, Any], Depends(get_current_user)]):
    uid = user_claims.get("uid")
    email = user_claims.get("email")
    if not uid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token")
    try:
        create_user_record_in_firestore(uid=uid, email=email or "", display_name=payload.display_name)
        updated_at = datetime.utcnow().isoformat() + "Z"
        return UserDetailsResponse(uid=uid, email=email, display_name=payload.display_name, updated_at=updated_at)
    except Exception as e:
        print(f"[Firestore] set_user_details error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to store user details")

@app.get("/user_details", response_model=UserDetailsResponse)
async def get_user_details(user_claims: Annotated[Dict[str, Any], Depends(get_current_user)]):
    uid = user_claims.get("uid")
    email = user_claims.get("email")
    if not uid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token")
    try:
        data = get_user_details_from_firestore(uid)
        return UserDetailsResponse(
            uid=uid,
            email=data.get("email", email),
            display_name=data.get("display_name") or None,
            updated_at=data.get("created_at", datetime.utcnow().isoformat() + "Z"),
        )
    except Exception as e:
        print(f"[Firestore] get_user_details error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to read user details")

# ---------------- Chat generation + download endpoints (Groq + generator) ----------------
@app.post("/projects/{project_id}/chat", response_model=ChatResponse)
async def project_chat(project_id: str, payload: ChatRequest, user_claims: Annotated[Dict[str, Any], Depends(get_current_user)]):
    uid = user_claims.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="Invalid token")
    assert_user_owns_project(uid, project_id)
    prompt = payload.prompt
    # parse with Groq (structured params)
    try:
        params = parse_prompt_with_groq(prompt, rows_hint=payload.rows, freq_hint=payload.freq_seconds)
    except Exception as e:
        print(f"[parse error] {e}")
        params = parse_prompt_to_params(prompt, rows_hint=payload.rows, freq_hint=payload.freq_seconds)
    # if no explicit temp_range, ask Groq to estimate
    if 'temp_range' not in params:
        try:
            params['temp_range'] = estimate_temp_range_with_groq(params)
        except Exception as e:
            print(f"[range estimate error] {e}")
            params['temp_range'] = params.get('temp_range', [15.0, 28.0])
    gen_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat() + "Z"
    gen_doc = {
        "generation_id": gen_id,
        "project_id": project_id,
        "owner_uid": uid,
        "prompt": prompt,
        "params": params,
        "created_at": created_at,
        "status": "ready",
    }
    try:
        _generation_ref(project_id, gen_id).set(gen_doc)
    except Exception as e:
        print(f"[Firestore] failed to write generation doc: {e}")
        raise HTTPException(status_code=500, detail="Failed to persist generation metadata")
    return ChatResponse(generation_id=gen_id, status="ready")

@app.get("/projects/{project_id}/download")
async def download_generation_csv(project_id: str, gen: str = Query(...), user_claims: Annotated[Dict[str, Any], Depends(get_current_user)] = None):
    uid = user_claims.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="Invalid token")
    assert_user_owns_project(uid, project_id)
    gen_doc = _generation_ref(project_id, gen).get()
    if not gen_doc.exists:
        raise HTTPException(status_code=404, detail="Generation not found")
    data = gen_doc.to_dict() or {}
    if data.get("owner_uid") != uid:
        raise HTTPException(status_code=403, detail="Not authorized to download this generation")
    params = data.get("params", {})
    # enforce caps
    try:
        ts = build_timestamps(params)
        max_rows = 200_000
        if len(ts) > max_rows:
            raise HTTPException(status_code=400, detail=f"Requested dataset too large (max {max_rows} rows).")
    except HTTPException:
        raise
    except Exception:
        pass
    rows_iter = generate_temperature_rows_iter(params)
    filename = f"{project_id}_{gen}.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(csv_stream_generator(rows_iter), media_type="text/csv", headers=headers)

@app.get("/projects/{project_id}/generations/{gen_id}", response_model=GenerationRecord)
async def get_generation_record(project_id: str, gen_id: str, user_claims: Annotated[Dict[str, Any], Depends(get_current_user)] = None):
    uid = user_claims.get("uid")
    assert_user_owns_project(uid, project_id)
    gen_doc = _generation_ref(project_id, gen_id).get()
    if not gen_doc.exists:
        raise HTTPException(status_code=404, detail="Generation not found")
    data = gen_doc.to_dict() or {}
    return GenerationRecord(
        generation_id=data.get("generation_id"),
        prompt=data.get("prompt"),
        params=data.get("params", {}),
        created_at=data.get("created_at"),
        status=data.get("status", "ready")
    )
