from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import ALLOWED_ORIGINS, MAX_GENERATION_ROWS
from .firebase import initialize_firebase_admin
from .model import load_model
from .routes import projects, users, generations

app = FastAPI(title="SynthIoT Structured API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routers
app.include_router(projects.router, prefix="/projects", tags=["projects"])
app.include_router(generations.router, prefix="/projects", tags=["generations"])
app.include_router(users.router, prefix="", tags=["users"])

@app.on_event("startup")
def startup_event():
    # Try to initialize Firebase (optional; errors printed)
    try:
        initialize_firebase_admin()
    except Exception as e:
        print(f"[startup] Firebase init warning: {e}")
    # Load CTGAN model (optional)
    load_model()
    print(f"[startup] MAX_GENERATION_ROWS={MAX_GENERATION_ROWS}")

@app.get("/")
async def root():
    return {"message": "API running"}
