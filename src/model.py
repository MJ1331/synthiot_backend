import os
import pickle
import pandas as pd
from typing import Iterator, Dict, Any
from .config import MODEL_PATH, CSV_CHUNK_SIZE

CTGAN_MODEL = None

def load_model(path: str = None):
    global CTGAN_MODEL
    p = path or MODEL_PATH
    if not p or not os.path.exists(p):
        print(f"[model] model file not found at '{p}' -> CTGAN disabled (fallback mode).")
        CTGAN_MODEL = None
        return None
    try:
        with open(p, "rb") as f:
            CTGAN_MODEL = pickle.load(f)
        print(f"[model] Loaded model from {p}: {type(CTGAN_MODEL)}")
        return CTGAN_MODEL
    except Exception as e:
        print(f"[model] Failed to load model from {p}: {e}")
        CTGAN_MODEL = None
        return None

def sample_from_model(model, params: Dict[str, Any], n: int) -> Iterator[pd.DataFrame]:
    try:
        # 1) model.sample(n) or model.sample(num_rows=n) or model.sample(n, conditions=params)
        if hasattr(model, "sample"):
            try:
                res = model.sample(n)
            except TypeError:
                try:
                    res = model.sample(num_rows=n)
                except TypeError:
                    try:
                        res = model.sample(n, conditions=params)
                    except Exception:
                        res = None
            if res is not None:
                if isinstance(res, pd.DataFrame):
                    for i in range(0, len(res), CSV_CHUNK_SIZE):
                        yield res.iloc[i:i+CSV_CHUNK_SIZE]
                    return
                if isinstance(res, (list, tuple)):
                    df = pd.DataFrame(res)
                    for i in range(0, len(df), CSV_CHUNK_SIZE):
                        yield df.iloc[i:i+CSV_CHUNK_SIZE]
                    return

        # 2) model.generate(prompt, n) or model.generate(n, prompt)
        if hasattr(model, "generate"):
            try:
                res = model.generate(params.get("prompt", ""), n)
            except TypeError:
                try:
                    res = model.generate(n, params.get("prompt", ""))
                except Exception:
                    res = None
            if res is not None:
                if isinstance(res, pd.DataFrame):
                    for i in range(0, len(res), CSV_CHUNK_SIZE):
                        yield res.iloc[i:i+CSV_CHUNK_SIZE]
                    return
                if isinstance(res, (list, tuple)):
                    df = pd.DataFrame(res)
                    for i in range(0, len(df), CSV_CHUNK_SIZE):
                        yield df.iloc[i:i+CSV_CHUNK_SIZE]
                    return

        # 3) model.predict(prompt, n)
        if hasattr(model, "predict"):
            try:
                res = model.predict(params.get("prompt", ""), n)
                if isinstance(res, pd.DataFrame):
                    for i in range(0, len(res), CSV_CHUNK_SIZE):
                        yield res.iloc[i:i+CSV_CHUNK_SIZE]
                    return
                if isinstance(res, (list, tuple)):
                    df = pd.DataFrame(res)
                    for i in range(0, len(df), CSV_CHUNK_SIZE):
                        yield df.iloc[i:i+CSV_CHUNK_SIZE]
                    return
            except Exception:
                pass

        # 4) other candidate method names
        for name in ("sample_dataframe", "sample_rows", "sample_n"):
            if hasattr(model, name):
                fn = getattr(model, name)
                try:
                    res = fn(n)
                except Exception:
                    res = None
                if res is not None:
                    if isinstance(res, pd.DataFrame):
                        for i in range(0, len(res), CSV_CHUNK_SIZE):
                            yield res.iloc[i:i+CSV_CHUNK_SIZE]
                        return
                    if isinstance(res, (list, tuple)):
                        df = pd.DataFrame(res)
                        for i in range(0, len(df), CSV_CHUNK_SIZE):
                            yield df.iloc[i:i+CSV_CHUNK_SIZE]
                        return
    except Exception as e:
        print(f"[model] sampling attempt raised: {e}")

    raise RuntimeError("Model does not support known sampling API or sampling failed.")
