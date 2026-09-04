#!/usr/bin/env python3
"""One-time, local-only script: obtain a YouTube OAuth refresh token.

Run this ONCE on your own computer (not in CI — it opens a browser).
It prints a refresh token you paste into the YT_REFRESH_TOKEN GitHub
secret; scripts/upload_youtube.py then reuses it unattended forever
(refresh tokens for this scope do not expire from normal use).

Usage:
    pip install google-auth-oauthlib
    python scripts/get_youtube_refresh_token.py --client-secrets client_secrets.json
"""
from __future__ import annotations

import argparse

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--client-secrets",
        default="client_secrets.json",
        help="OAuth client JSON downloaded from Google Cloud Console (Desktop app type)",
    )
    args = parser.parse_args()

    flow = InstalledAppFlow.from_client_secrets_file(args.client_secrets, SCOPES)
    credentials = flow.run_local_server(port=0)

    print("\nSuccess. Add these three values as GitHub repo secrets:\n")
    print(f"YT_CLIENT_ID={credentials.client_id}")
    print(f"YT_CLIENT_SECRET={credentials.client_secret}")
    print(f"YT_REFRESH_TOKEN={credentials.refresh_token}")


if __name__ == "__main__":
    main()
