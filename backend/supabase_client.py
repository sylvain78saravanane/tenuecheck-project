from __future__ import annotations

import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

_client: Client = None


def get_supabase() -> Client:
    """Retourne le client Supabase (singleton)."""
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise EnvironmentError(
                "SUPABASE_URL et SUPABASE_SERVICE_KEY doivent être définis dans .env"
            )
        _client = create_client(url, key)
    return _client