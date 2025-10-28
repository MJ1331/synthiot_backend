import os
import json
from dotenv import load_dotenv

load_dotenv()

SERVICE_ACCOUNT_KEY_PATH = os.getenv("SERVICE_ACCOUNT_KEY_PATH", "firebase_service_account.json")
ALLOWED_ORIGINS = json.loads(os.getenv("ALLOWED_ORIGINS_JSON", '["http://localhost:5173"]'))
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
MAX_GENERATION_ROWS = int(os.getenv("MAX_GENERATION_ROWS", "200000"))

CSV_CHUNK_SIZE = int(os.getenv("CSV_CHUNK_SIZE", "5000"))
