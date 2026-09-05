#!/usr/bin/env python3
"""Daily cron: build the movement-clip library.

Downloads stock footage for a rotating slice of control/subjects.txt, keeps
only "pure movement" segments (no visible face, real motion — see
app/services/clip_library.py for why), and stores the survivors in Supabase:
the clip files in the private "movement-clips" Storage bucket, their
metadata + a text embedding of the search terms in the mpt.movement_clips
table (pgvector). This is what daily video generation will later search by
topic similarity instead of hitting Pexels live for every run — that
retrieval wiring is a separate follow-up step, not done by this script.

Independent from cli.py's own material-fetching path on purpose: this is an
offline library-building job, not part of the generation critical path, so
it can afford to be slower and more selective.

Required environment:
    GEMINI_API_KEY, PEXELS_API_KEY  — same secrets daily-short.yml already
        uses (this script embeds search terms with the same Gemini account
        and searches Pexels the same way generation does).
    SUPABASE_DB_URL  — Postgres connection string for the "center" Supabase
        project (Settings -> Database -> Connection string). Used to insert
        rows directly into mpt.movement_clips.
    SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY  — used to upload clip files to
        the private "movement-clips" Storage bucket via its REST API.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
from time import perf_counter
from typing import List

import psycopg2
import requests
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.schema import VideoAspect  # noqa: E402
from app.services import clip_library  # noqa: E402
from app.services.llm import generate_terms  # noqa: E402
from app.services.material import save_video, search_videos_pexels  # noqa: E402

SUBJECTS_FILE = "control/subjects.txt"
STORAGE_BUCKET = "movement-clips"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768  # matches mpt.movement_clips.embedding's vector(768)


# Kept small on purpose: this runs once a day and is not on the generation
# critical path, but a hosted GitHub Actions job still has a wall-clock
# budget. Growing the library is a marathon, not a sprint — a few subjects
# a day, every day, adds up.
#
# Overridable via environment for a one-off manually-triggered scale test
# (see .github/workflows/build-clip-library.yml's workflow_dispatch inputs)
# without changing the daily cron's own pace.
def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


SUBJECTS_PER_RUN = _int_env("CLIP_LIBRARY_SUBJECTS_PER_RUN", 2)
SEARCH_TERMS_PER_SUBJECT = _int_env("CLIP_LIBRARY_SEARCH_TERMS_PER_SUBJECT", 3)
VIDEOS_PER_SEARCH_TERM = _int_env("CLIP_LIBRARY_VIDEOS_PER_SEARCH_TERM", 2)
SEGMENT_DURATION_SECONDS = 6.0
MIN_MOTION_SCORE = 0.02


def _require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        print(f"{name} is not set", file=sys.stderr)
        raise SystemExit(1)
    return value


def pick_subjects(path: str, count: int) -> List[str]:
    if not os.path.exists(path):
        logger.warning(f"{path} does not exist, nothing to index today")
        return []
    with open(path, encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
    if not lines:
        return []
    random.shuffle(lines)
    return lines[:count]


def embed_text(client, text: str) -> List[float]:
    from google.genai import types

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSIONS),
    )
    return list(response.embeddings[0].values)


def upload_clip(
    *, supabase_url: str, service_role_key: str, storage_path: str, local_path: str
) -> None:
    with open(local_path, "rb") as f:
        data = f.read()
    url = f"{supabase_url}/storage/v1/object/{STORAGE_BUCKET}/{storage_path}"
    r = requests.put(
        url,
        headers={
            "Authorization": f"Bearer {service_role_key}",
            "apikey": service_role_key,
            "Content-Type": "video/mp4",
            "x-upsert": "true",
        },
        data=data,
        timeout=(30, 120),
    )
    if r.status_code >= 300:
        raise RuntimeError(
            f"failed to upload {local_path} to Storage: {r.status_code} {r.text}"
        )


def insert_movement_clip(
    conn,
    *,
    source_asset_id: str,
    storage_path: str,
    duration_seconds: float,
    width: int,
    height: int,
    motion_score: float,
    keywords: List[str],
    embedding: List[float],
) -> None:
    # psycopg2 has no built-in adapter for pgvector's "vector" type — handing
    # it a plain Python list would try to bind it as a native array, which
    # the column's declared type rejects. Send it as vector's own text input
    # format ("[0.1,0.2,...]") and cast explicitly instead of pulling in the
    # separate pgvector-python dependency for a single insert path.
    embedding_literal = "[" + ",".join(repr(float(v)) for v in embedding) + "]"
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into mpt.movement_clips (
                source_provider, source_asset_id, storage_path,
                duration_seconds, width, height, has_visible_face,
                motion_score, keywords, embedding
            ) values (%s, %s, %s, %s, %s, %s, false, %s, %s, %s::vector)
            on conflict (source_provider, source_asset_id, storage_path)
            do nothing
            """,
            (
                "pexels",
                source_asset_id,
                storage_path,
                duration_seconds,
                width,
                height,
                motion_score,
                keywords,
                embedding_literal,
            ),
        )
    conn.commit()


def process_subject(
    subject: str,
    *,
    genai_client,
    supabase_url: str,
    service_role_key: str,
    db_conn,
    tmp_dir: str,
) -> int:
    logger.info(f"indexing subject: {subject}")
    search_terms = generate_terms(
        video_subject=subject, video_script=subject, amount=SEARCH_TERMS_PER_SUBJECT
    )
    if not search_terms:
        logger.warning(f"no search terms generated for subject, skipping: {subject}")
        return 0

    embedding = embed_text(genai_client, ", ".join(search_terms))
    stored_count = 0

    for term in search_terms:
        try:
            items = search_videos_pexels(
                term,
                minimum_duration=int(SEGMENT_DURATION_SECONDS),
                video_aspect=VideoAspect.portrait,
            )
        except Exception as e:
            logger.error(f"pexels search failed for term {term!r}: {e}")
            continue

        for item in items[:VIDEOS_PER_SEARCH_TERM]:
            source_info = item.source_info or {}
            asset_id = str(source_info.get("asset_id") or item.url)
            video_started_at = perf_counter()
            try:
                local_video_path = save_video(video_url=item.url, save_dir=tmp_dir)
            except Exception as e:
                logger.error(f"failed to download material video: {e}")
                continue
            if not local_video_path:
                continue
            download_elapsed = perf_counter() - video_started_at

            try:
                analysis_started_at = perf_counter()
                segments = clip_library.analyze_video_segments(
                    local_video_path, segment_duration=SEGMENT_DURATION_SECONDS
                )
            except Exception as e:
                logger.error(f"failed to analyze video {local_video_path}: {e}")
                continue

            analysis_elapsed = perf_counter() - analysis_started_at
            clean_segments = clip_library.select_clean_segments(
                segments, min_motion_score=MIN_MOTION_SCORE
            )
            logger.info(
                f"term={term!r} asset={asset_id}: "
                f"{len(clean_segments)}/{len(segments)} segments are clean "
                f"(download {download_elapsed:.1f}s, analysis {analysis_elapsed:.1f}s)"
            )

            store_started_at = perf_counter()
            for index, segment in enumerate(clean_segments):
                clip_path = os.path.join(tmp_dir, f"{asset_id}-{index}.mp4")
                try:
                    clip_library.cut_segment_to_file(
                        local_video_path,
                        segment.start_time,
                        segment.end_time,
                        clip_path,
                    )
                    storage_path = f"{asset_id}/{index}.mp4"
                    upload_clip(
                        supabase_url=supabase_url,
                        service_role_key=service_role_key,
                        storage_path=storage_path,
                        local_path=clip_path,
                    )
                    insert_movement_clip(
                        db_conn,
                        source_asset_id=f"{asset_id}-{index}",
                        storage_path=storage_path,
                        duration_seconds=segment.end_time - segment.start_time,
                        width=source_info.get("rendition", {}).get("width") or 0,
                        height=source_info.get("rendition", {}).get("height") or 0,
                        motion_score=segment.motion_score,
                        keywords=search_terms,
                        embedding=embedding,
                    )
                    stored_count += 1
                except Exception as e:
                    logger.error(f"failed to store segment {index} of {asset_id}: {e}")
                finally:
                    if os.path.exists(clip_path):
                        os.remove(clip_path)

            if os.path.exists(local_video_path):
                os.remove(local_video_path)

            store_elapsed = perf_counter() - store_started_at
            logger.info(
                f"term={term!r} asset={asset_id}: cut+upload+insert took "
                f"{store_elapsed:.1f}s for {len(clean_segments)} segments"
            )

    return stored_count


def main() -> None:
    gemini_api_key = _require_env("GEMINI_API_KEY")
    _require_env("PEXELS_API_KEY")  # read via app.config by search_videos_pexels
    supabase_db_url = _require_env("SUPABASE_DB_URL")
    supabase_url = _require_env("SUPABASE_URL")
    service_role_key = _require_env("SUPABASE_SERVICE_ROLE_KEY")

    from google import genai

    genai_client = genai.Client(api_key=gemini_api_key)

    subjects = pick_subjects(SUBJECTS_FILE, SUBJECTS_PER_RUN)
    if not subjects:
        logger.warning("no subjects to index today, exiting")
        return

    logger.info(
        f"config: subjects_per_run={SUBJECTS_PER_RUN}, "
        f"search_terms_per_subject={SEARCH_TERMS_PER_SUBJECT}, "
        f"videos_per_search_term={VIDEOS_PER_SEARCH_TERM} "
        f"(up to {len(subjects) * SEARCH_TERMS_PER_SUBJECT * VIDEOS_PER_SEARCH_TERM} "
        "candidate video downloads this run)"
    )

    db_conn = psycopg2.connect(supabase_db_url)
    total_stored = 0
    run_started_at = perf_counter()
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for subject in subjects:
                total_stored += process_subject(
                    subject,
                    genai_client=genai_client,
                    supabase_url=supabase_url,
                    service_role_key=service_role_key,
                    db_conn=db_conn,
                    tmp_dir=tmp_dir,
                )
    finally:
        db_conn.close()

    run_elapsed = perf_counter() - run_started_at
    logger.success(
        f"stored {total_stored} new clean movement clips today "
        f"in {run_elapsed:.1f}s ({run_elapsed / 60:.1f} min)"
    )


if __name__ == "__main__":
    main()
