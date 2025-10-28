import firebase_admin
from firebase_admin import credentials, firestore
from .config import SERVICE_ACCOUNT_KEY_PATH

def initialize_firebase_admin():
    try:
        firebase_admin.get_app()
        return
    except ValueError:
        # not initialized
        pass

    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
    firebase_admin.initialize_app(cred)
    # warm Firestore client
    firestore.client()
    print("--- Firebase Admin SDK initialized ---")

def fs_client():
    initialize_firebase_admin()
    return firestore.client()
