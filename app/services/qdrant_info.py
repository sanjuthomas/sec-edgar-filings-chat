from __future__ import annotations

import httpx


def get_collection_vector_size(base_url: str, collection: str) -> int | None:
    try:
        with httpx.Client(base_url=base_url.rstrip("/"), timeout=5.0) as client:
            response = client.get(f"/collections/{collection}")
            response.raise_for_status()
            payload = response.json()
        vectors = payload.get("result", {}).get("config", {}).get("params", {}).get("vectors")
        if isinstance(vectors, dict) and "size" in vectors:
            return int(vectors["size"])
        if isinstance(vectors, dict):
            for value in vectors.values():
                if isinstance(value, dict) and "size" in value:
                    return int(value["size"])
    except Exception:
        return None
    return None
