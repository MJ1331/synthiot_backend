# src/routes/generations.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from typing import Dict
from ..schemas import ChatRequest, ChatResponse, GenerationRecord
from ..auth import get_current_user
from ..firebase import fs_client
from ..utils import parse_prompt_to_params
from ..model import CTGAN_MODEL, sample_from_model
from ..generator import generate_temperature_rows_iter, csv_stream_generator_from_iterator, build_timestamps
from ..config import MAX_GENERATION_ROWS
from datetime import datetime
import pandas as pd
import uuid

router = APIRouter()

def _project_ref(project_id: str):
    return fs_client().collection("project_details").document(project_id)

def _generation_ref(project_id: str, gen_id: str):
    return _project_ref(project_id).collection("generations").document(gen_id)

def assert_user_owns_project(uid: str, project_id: str):
    doc = _project_ref(project_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Project not found")
    data = doc.to_dict() or {}
    if data.get("owner_uid") != uid:
        raise HTTPException(status_code=403, detail="Not authorized for this project")
    return data

@router.post("/{project_id}/chat", response_model=ChatResponse)
async def project_chat(project_id: str, payload: ChatRequest, user_claims: dict = Depends(get_current_user)):
    uid = user_claims.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="Invalid token")
    assert_user_owns_project(uid, project_id)
    prompt = payload.prompt or ""
    params = parse_prompt_to_params(prompt, rows_hint=payload.rows, freq_hint=payload.freq_seconds)
    params['prompt'] = prompt
    # enforce caps BEFORE persisting
    try:
        timestamps = build_timestamps(params, MAX_GENERATION_ROWS)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid time range / params")
    estimated_rows = len(timestamps)
    if estimated_rows > MAX_GENERATION_ROWS:
        raise HTTPException(status_code=400, detail=f"Requested dataset too large (max {MAX_GENERATION_ROWS} rows).")
    params['_estimated_row_count'] = estimated_rows
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
        "estimated_rows": estimated_rows,
    }
    try:
        _generation_ref(project_id, gen_id).set(gen_doc)
    except Exception as e:
        print(f"[generations] failed to persist generation doc: {e}")
        raise HTTPException(status_code=500, detail="Failed to persist generation metadata")
    return ChatResponse(generation_id=gen_id, status="ready")

@router.get("/{project_id}/download")
async def download_generation_csv(project_id: str, gen: str = Query(...), user_claims: dict = Depends(get_current_user)):
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
    try:
        ts = build_timestamps(params, MAX_GENERATION_ROWS)
        if len(ts) > MAX_GENERATION_ROWS:
            raise HTTPException(status_code=400, detail=f"Requested dataset too large (max {MAX_GENERATION_ROWS} rows).")
    except HTTPException:
        raise
    except Exception:
        pass

    estimated_rows = int(params.get("_estimated_row_count", params.get("rows", len(ts))))
    filename = f"{project_id}_{gen}.csv"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    model = CTGAN_MODEL
    if model:
        try:
            df_chunks = sample_from_model(model, params, estimated_rows)
            timestamps = build_timestamps(params, MAX_GENERATION_ROWS)
            def model_csv_stream():
                first = True
                yielded_rows = 0
                for df in df_chunks:
                    if not isinstance(df, pd.DataFrame):
                        df = pd.DataFrame(df)
                    if yielded_rows + len(df) > estimated_rows:
                        df = df.iloc[: max(0, estimated_rows - yielded_rows)]
                    # attach timestamps if absent
                    if "timestamp" not in df.columns:
                        start = yielded_rows
                        end = yielded_rows + len(df)
                        df.insert(0, "timestamp", [t.isoformat() + "Z" for t in timestamps[start:end]])
                    if first:
                        yield df.to_csv(index=False, header=True)
                        first = False
                    else:
                        yield df.to_csv(index=False, header=False)
                    yielded_rows += len(df)
                    if yielded_rows >= estimated_rows:
                        break
                if first:
                    # nothing produced
                    yield "timestamp,temperature\n"
            return StreamingResponse(model_csv_stream(), media_type="text/csv", headers=headers)
        except Exception as e:
            print(f"[generations] model sampling failed: {e} - falling back to built-in generator")

    # fallback
    rows_iter = generate_temperature_rows_iter(params, MAX_GENERATION_ROWS)
    return StreamingResponse(csv_stream_generator_from_iterator(["timestamp","temperature"], rows_iter), media_type="text/csv", headers=headers)

@router.get("/{project_id}/generations/{gen_id}", response_model=GenerationRecord)
async def get_generation_record(project_id: str, gen_id: str, user_claims: dict = Depends(get_current_user)):
    uid = user_claims.get("uid")
    if not uid:
        raise HTTPException(status_code=400, detail="Invalid token")
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
