import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from supabase import create_client, Client


@lru_cache()
def get_supabase() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set as environment "
            "variables on the backend service."
        )
    return create_client(url, key)


def save_project(user_id: str, name: str, image_url: Optional[str],
                  total_floors: int) -> Dict[str, Any]:
    sb = get_supabase()
    result = sb.table("projects").insert({
        "user_id": user_id,
        "name": name,
        "image_url": image_url,
        "total_floors": total_floors,
    }).execute()
    return result.data[0]


def save_rooms(project_id: str, rooms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sb = get_supabase()
    rows = [{**room, "project_id": project_id} for room in rooms]
    result = sb.table("project_rooms").insert(rows).execute()
    return result.data
