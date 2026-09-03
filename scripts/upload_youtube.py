#!/usr/bin/env python3
"""Upload one generated video to YouTube using a pre-authorized refresh token.

The OAuth consent screen (opening a browser) only ever happens once, on a
local machine, ahead of time — see docs/youtube_oauth_setup.md. This script
only refreshes that token and calls the Data API, so it runs unattended
in CI with no browser and no human present.
"""
from __future__ import annotations

import argparse
import os
import sys

import google.auth.transport.requests
import google.oauth2.credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
# "People & Blogs" — a safe default for short-form narrated content; change
# if a more specific YouTube category fits the channel better.
DEFAULT_CATEGORY_ID = "22"


def build_youtube_client():
    client_id = os.environ["YT_CLIENT_ID"]
    client_secret = os.environ["YT_CLIENT_SECRET"]
    refresh_token = os.environ["YT_REFRESH_TOKEN"]

    credentials = google.oauth2.credentials.Credentials(
        None,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        token_uri="https://oauth2.googleapis.com/token",
        scopes=SCOPES,
    )
    credentials.refresh(google.auth.transport.requests.Request())
    return build("youtube", "v3", credentials=credentials)


def upload(video_path: str, title: str, description: str) -> str:
    youtube = build_youtube_client()
    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:4900],
            "categoryId": DEFAULT_CATEGORY_ID,
        },
        "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    request = youtube.videos().insert(
        part="snippet,status", body=body, media_body=media
    )
    response = request.execute()
    return response["id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", help="Path to the rendered video file")
    parser.add_argument("title", help="Video title / subject, used as the YouTube title")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"::error::video file not found: {args.video_path}", file=sys.stderr)
        raise SystemExit(1)

    video_id = upload(args.video_path, args.title, args.title)
    print(f"Uploaded: https://www.youtube.com/watch?v={video_id}")


if __name__ == "__main__":
    main()
