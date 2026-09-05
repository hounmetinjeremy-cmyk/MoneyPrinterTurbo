"""Retrieval side of the movement-clip library.

scripts/build_clip_library.py indexes "pure movement" stock-footage clips
daily: each clip is stored in Supabase (Storage for the file, the
mpt.movement_clips table for its metadata) together with a text embedding
of the search terms used to find it. This module is the other half — given
a day's own search terms, embed them the same way, rank indexed clips by
cosine similarity, and download enough of the best matches to (ideally)
cover the video's required duration.

Deliberately a soft layer, not a replacement: app.services.material's
download_videos() calls fetch_movement_clips_for_subject() first (only
when config.app.use_movement_library is enabled) and falls back to its
existing Pexels/other-source path for whatever duration this couldn't
cover. The library starts empty and grows one cron run at a time, so
treating it as the only source would break generation outright whenever
coverage is thin — which, early on, is most days.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import requests
from loguru import logger

from app.utils import utils

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768  # matches mpt.movement_clips.embedding's vector(768)
STORAGE_BUCKET = "movement-clips"

# Ranking pulls every candidate's embedding over the REST API and scores it
# in Python rather than via a Postgres RPC using the table's HNSW index —
# simpler and dependency-free while the library stays in the hundreds/low
# thousands of rows (this runs once per generation, not latency-critical).
# Revisit with an RPC (mirroring Supabase's documented match_documents
# pattern) once fetching every row on each generation becomes wasteful.
MAX_CANDIDATE_CLIPS = 500


@dataclass
class MovementClip:
    storage_path: str
    duration_seconds: float
    keywords: List[str]
    source_asset_id: str


@dataclass
class RetrievedMaterial:
    local_path: str
    duration_seconds: float
    keywords: List[str]
    source_asset_id: str


def embed_text(text: str, *, gemini_api_key: str) -> List[float]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=gemini_api_key)
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS),
    )
    return list(response.embeddings[0].values)


def _parse_embedding(raw) -> Optional[List[float]]:
    """
    PostgREST has no native-JSON-array cast for pgvector's "vector" type —
    it serializes the column using the type's own text output format
    ("[0.1,0.2,...]"), a JSON *string*, not an array. Handle both, in case
    that ever changes upstream.
    """
    if isinstance(raw, list):
        try:
            return [float(v) for v in raw]
        except (TypeError, ValueError):
            return None
    if isinstance(raw, str):
        try:
            return [float(v) for v in raw.strip("[]").split(",") if v.strip()]
        except ValueError:
            return None
    return None


def _fetch_candidate_clips(*, supabase_url: str, service_role_key: str) -> List[dict]:
    r = requests.get(
        f"{supabase_url}/rest/v1/movement_clips",
        headers={
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
            "Accept-Profile": "mpt",
        },
        params={
            "select": "storage_path,duration_seconds,keywords,source_asset_id,embedding",
            "order": "created_at.desc",
            "limit": str(MAX_CANDIDATE_CLIPS),
        },
        timeout=(30, 60),
    )
    if r.status_code >= 300:
        raise RuntimeError(f"failed to list movement_clips: {r.status_code} {r.text}")
    return r.json()


def find_similar_clips(
    *,
    supabase_url: str,
    service_role_key: str,
    query_embedding: List[float],
    limit: int,
) -> List[MovementClip]:
    """Rank indexed clips by cosine similarity to query_embedding, best first."""
    rows = _fetch_candidate_clips(
        supabase_url=supabase_url, service_role_key=service_role_key
    )

    query = np.array(query_embedding, dtype=np.float64)
    query_norm = np.linalg.norm(query)
    scored: List[tuple] = []
    for row in rows:
        embedding = _parse_embedding(row.get("embedding"))
        if not embedding or len(embedding) != len(query_embedding):
            continue
        vector = np.array(embedding, dtype=np.float64)
        denom = np.linalg.norm(vector) * query_norm
        similarity = float(np.dot(vector, query) / denom) if denom else 0.0
        scored.append((similarity, row))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        MovementClip(
            storage_path=row["storage_path"],
            duration_seconds=float(row["duration_seconds"]),
            keywords=list(row.get("keywords") or []),
            source_asset_id=str(row.get("source_asset_id") or ""),
        )
        for _, row in scored[:limit]
    ]


def download_clip(
    *,
    supabase_url: str,
    service_role_key: str,
    storage_path: str,
    output_path: str,
) -> None:
    r = requests.get(
        f"{supabase_url}/storage/v1/object/{STORAGE_BUCKET}/{storage_path}",
        headers={
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
        },
        timeout=(30, 120),
    )
    if r.status_code >= 300:
        raise RuntimeError(
            f"failed to download {storage_path} from Storage: {r.status_code} {r.text}"
        )
    with open(output_path, "wb") as f:
        f.write(r.content)


def fetch_movement_clips_for_subject(
    *,
    search_terms: List[str],
    required_duration: float,
    max_clip_duration: float,
    material_directory: str,
    gemini_api_key: str,
    supabase_url: str,
    service_role_key: str,
) -> List[RetrievedMaterial]:
    """
    Embed search_terms the same way scripts/build_clip_library.py embedded
    them at index time, retrieve the best-matching indexed clips, and
    download clips (stopping once required_duration is covered, same
    stopping convention as material.download_videos's own loop) into
    material_directory — a plain cache dir, same convention as save_video,
    keyed by storage_path so a clip already downloaded for an earlier task
    isn't re-fetched.

    Never raises: a retrieval failure should fall back to the caller's
    existing material source, not abort generation. Returns whatever it
    managed to fetch (possibly an empty list) either way.
    """
    if required_duration <= 0 or not search_terms:
        return []

    save_dir = material_directory or utils.storage_dir("cache_videos", create=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        query_embedding = embed_text(
            ", ".join(search_terms), gemini_api_key=gemini_api_key
        )
    except Exception as e:
        logger.error(f"movement library: failed to embed search terms: {e}")
        return []

    # Ask for generously more candidates than duration alone suggests —
    # clips are short (a few seconds each) and some downloads can fail.
    candidate_limit = max(20, int(required_duration / max(max_clip_duration, 1)) * 3)
    try:
        candidates = find_similar_clips(
            supabase_url=supabase_url,
            service_role_key=service_role_key,
            query_embedding=query_embedding,
            limit=candidate_limit,
        )
    except Exception as e:
        logger.error(f"movement library: failed to rank candidate clips: {e}")
        return []

    results: List[RetrievedMaterial] = []
    covered_duration = 0.0
    for clip in candidates:
        if covered_duration >= required_duration:
            break

        path_hash = utils.md5(clip.storage_path)
        output_path = os.path.join(save_dir, f"movement-{path_hash}.mp4")
        if not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
            try:
                download_clip(
                    supabase_url=supabase_url,
                    service_role_key=service_role_key,
                    storage_path=clip.storage_path,
                    output_path=output_path,
                )
            except Exception as e:
                logger.error(
                    f"movement library: failed to download {clip.storage_path}: {e}"
                )
                continue

        results.append(
            RetrievedMaterial(
                local_path=output_path,
                duration_seconds=clip.duration_seconds,
                keywords=clip.keywords,
                source_asset_id=clip.source_asset_id,
            )
        )
        covered_duration += min(max_clip_duration, clip.duration_seconds)

    logger.info(
        f"movement library: covered {covered_duration:.1f}s of "
        f"{required_duration:.1f}s needed with {len(results)} clips"
    )
    return results
