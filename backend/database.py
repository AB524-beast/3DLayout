import os
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

from supabase import create_client, Client

logger = logging.getLogger(__name__)


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
    if not result.data:
        raise RuntimeError("Failed to insert project: no data returned")
    return result.data[0]


def save_rooms(project_id: str, rooms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rooms:
        return []
    sb = get_supabase()
    rows = [{**room, "project_id": project_id} for room in rooms]
    result = sb.table("project_rooms").insert(rows).execute()
    return result.data or []


def get_project(project_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    sb = get_supabase()
    result = sb.table("projects").select("*").eq("id", project_id).eq("user_id", user_id).execute()
    return result.data[0] if result.data else None


def get_user_projects(user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    sb = get_supabase()
    result = (
        sb.table("projects")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    return result.data or []


def delete_project(project_id: str, user_id: str) -> bool:
    sb = get_supabase()
    sb.table("project_rooms").delete().eq("project_id", project_id).execute()
    result = sb.table("projects").delete().eq("id", project_id).eq("user_id", user_id).execute()
    return True


def get_project_rooms(project_id: str) -> List[Dict[str, Any]]:
    sb = get_supabase()
    result = sb.table("project_rooms").select("*").eq("project_id", project_id).execute()
    return result.data or []
