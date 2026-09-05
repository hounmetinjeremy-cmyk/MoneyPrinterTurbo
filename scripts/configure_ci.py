#!/usr/bin/env python3
"""Materialize config.toml for CI runs from config.example.toml + secrets.

The CLI (cli.py) covers most per-run settings as flags, but a handful of
values have no flag and only ever come from config.toml: the LLM
provider/key and the character overlay toggle. This script patches just
those lines onto a fresh copy of config.example.toml, so every other
setting keeps the example file's shipped default.

Fails loudly (instead of silently misconfiguring a run) if an expected
line is missing, e.g. after config.example.toml is edited upstream.
"""

from __future__ import annotations

import os
import shutil
import sys

EXAMPLE = "config.example.toml"
TARGET = "config.toml"


def replace_line(text: str, old: str, new: str) -> str:
    if old not in text:
        raise SystemExit(
            f"expected line not found in {EXAMPLE}, config.example.toml may have "
            f"changed and scripts/configure_ci.py needs updating: {old!r}"
        )
    return text.replace(old, new, 1)


def main() -> None:
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        print("GEMINI_API_KEY is not set", file=sys.stderr)
        raise SystemExit(1)

    pexels_key = os.environ.get("PEXELS_API_KEY", "")
    if not pexels_key:
        print("PEXELS_API_KEY is not set", file=sys.stderr)
        raise SystemExit(1)

    shutil.copyfile(EXAMPLE, TARGET)
    with open(TARGET, "r", encoding="utf-8") as f:
        text = f.read()

    text = replace_line(text, 'llm_provider = "moonshot"', 'llm_provider = "gemini"')
    text = replace_line(text, 'gemini_api_key = ""', f'gemini_api_key = "{gemini_key}"')
    # Leaving gemini_model_name empty lets the app fall back to its Pro-tier
    # registry default, which Google removed from the free tier in April
    # 2026 (429 RESOURCE_EXHAUSTED, limit: 0). Pin an actual free-tier
    # Flash model instead.
    text = replace_line(
        text, 'gemini_model_name = ""', 'gemini_model_name = "gemini-3.5-flash"'
    )
    # video_source stays "pexels" (config.example.toml's shipped default) — the
    # character overlay is an independent corner picture-in-picture, it does
    # not require the background footage itself to be local.
    text = replace_line(
        text, "pexels_api_keys = []", f'pexels_api_keys = ["{pexels_key}"]'
    )
    text = replace_line(
        text,
        "# character_overlay_enabled = false",
        "character_overlay_enabled = true",
    )
    text = replace_line(
        text, "# max_duration_seconds = 60", "max_duration_seconds = 60"
    )

    # Purely additive and self-disabling when unconfigured (see
    # config.example.toml's Movement Clip Library section), so it's safe to
    # turn on here whenever the Supabase secrets scripts/build_clip_library.py
    # already uses are present — no separate opt-in step needed once they're
    # set up.
    if os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_ROLE_KEY"):
        text = replace_line(
            text,
            "# use_movement_library = false",
            "use_movement_library = true",
        )

    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
