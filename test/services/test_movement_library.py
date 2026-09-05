import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import movement_library


class TestEmbedText(unittest.TestCase):
    def test_embed_text_returns_values_from_response(self):
        fake_embedding = MagicMock()
        fake_embedding.values = [0.1, 0.2, 0.3]
        fake_response = MagicMock()
        fake_response.embeddings = [fake_embedding]
        fake_client = MagicMock()
        fake_client.models.embed_content.return_value = fake_response

        with patch("google.genai.Client", return_value=fake_client) as client_cls:
            result = movement_library.embed_text("hammer, wood", gemini_api_key="key")

        client_cls.assert_called_once_with(api_key="key")
        self.assertEqual(result, [0.1, 0.2, 0.3])
        _, kwargs = fake_client.models.embed_content.call_args
        self.assertEqual(kwargs["model"], movement_library.EMBEDDING_MODEL)
        self.assertEqual(kwargs["contents"], "hammer, wood")


class TestParseEmbedding(unittest.TestCase):
    def test_parses_pgvector_text_format(self):
        self.assertEqual(
            movement_library._parse_embedding("[0.1,0.2,0.3]"), [0.1, 0.2, 0.3]
        )

    def test_accepts_a_native_list(self):
        self.assertEqual(movement_library._parse_embedding([0.1, 0.2]), [0.1, 0.2])

    def test_rejects_garbage(self):
        self.assertIsNone(movement_library._parse_embedding("not a vector"))
        self.assertIsNone(movement_library._parse_embedding(None))
        self.assertIsNone(movement_library._parse_embedding(42))


class TestFindSimilarClips(unittest.TestCase):
    def test_ranks_by_cosine_similarity_best_first(self):
        rows = [
            {
                "storage_path": "a/0.mp4",
                "duration_seconds": 6.0,
                "keywords": ["unrelated"],
                "source_asset_id": "a-0",
                # Orthogonal to the query — similarity 0.
                "embedding": [0.0, 1.0],
            },
            {
                "storage_path": "b/0.mp4",
                "duration_seconds": 5.0,
                "keywords": ["hammer"],
                "source_asset_id": "b-0",
                # Same direction as the query — similarity 1.
                "embedding": [1.0, 0.0],
            },
        ]
        with patch.object(
            movement_library, "_fetch_candidate_clips", return_value=rows
        ):
            result = movement_library.find_similar_clips(
                supabase_url="https://x.supabase.co",
                service_role_key="key",
                query_embedding=[1.0, 0.0],
                limit=10,
            )

        self.assertEqual([c.storage_path for c in result], ["b/0.mp4", "a/0.mp4"])

    def test_skips_rows_with_unparseable_or_mismatched_embeddings(self):
        rows = [
            {
                "storage_path": "bad/0.mp4",
                "duration_seconds": 6.0,
                "keywords": [],
                "source_asset_id": "bad-0",
                "embedding": "garbage",
            },
            {
                "storage_path": "wrong-dims/0.mp4",
                "duration_seconds": 6.0,
                "keywords": [],
                "source_asset_id": "wrong-0",
                "embedding": [1.0],  # query is 2-dim below
            },
            {
                "storage_path": "good/0.mp4",
                "duration_seconds": 6.0,
                "keywords": [],
                "source_asset_id": "good-0",
                "embedding": [1.0, 0.0],
            },
        ]
        with patch.object(
            movement_library, "_fetch_candidate_clips", return_value=rows
        ):
            result = movement_library.find_similar_clips(
                supabase_url="https://x.supabase.co",
                service_role_key="key",
                query_embedding=[1.0, 0.0],
                limit=10,
            )

        self.assertEqual([c.storage_path for c in result], ["good/0.mp4"])

    def test_limit_caps_the_number_of_results(self):
        rows = [
            {
                "storage_path": f"{i}/0.mp4",
                "duration_seconds": 6.0,
                "keywords": [],
                "source_asset_id": f"{i}-0",
                "embedding": [1.0, 0.0],
            }
            for i in range(5)
        ]
        with patch.object(
            movement_library, "_fetch_candidate_clips", return_value=rows
        ):
            result = movement_library.find_similar_clips(
                supabase_url="https://x.supabase.co",
                service_role_key="key",
                query_embedding=[1.0, 0.0],
                limit=2,
            )
        self.assertEqual(len(result), 2)


class TestDownloadClip(unittest.TestCase):
    def test_writes_response_content_to_output_path(self):
        fake_response = MagicMock(status_code=200, content=b"video-bytes")
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "clip.mp4")
            with patch.object(
                movement_library.requests, "get", return_value=fake_response
            ) as get:
                movement_library.download_clip(
                    supabase_url="https://x.supabase.co",
                    service_role_key="key",
                    storage_path="abc/0.mp4",
                    output_path=output_path,
                )
            with open(output_path, "rb") as f:
                self.assertEqual(f.read(), b"video-bytes")
            url = get.call_args[0][0]
            self.assertIn("abc/0.mp4", url)
            self.assertIn(movement_library.STORAGE_BUCKET, url)

    def test_raises_on_http_error(self):
        fake_response = MagicMock(status_code=404, text="not found")
        with patch.object(movement_library.requests, "get", return_value=fake_response):
            with self.assertRaises(RuntimeError):
                movement_library.download_clip(
                    supabase_url="https://x.supabase.co",
                    service_role_key="key",
                    storage_path="missing/0.mp4",
                    output_path="/tmp/should-not-be-written.mp4",
                )


class TestFetchMovementClipsForSubject(unittest.TestCase):
    def _clip(self, path, duration):
        return movement_library.MovementClip(
            storage_path=path,
            duration_seconds=duration,
            keywords=["hammer"],
            source_asset_id=path,
        )

    def test_returns_empty_when_no_duration_needed(self):
        result = movement_library.fetch_movement_clips_for_subject(
            search_terms=["hammer"],
            required_duration=0,
            max_clip_duration=6,
            material_directory="",
            gemini_api_key="key",
            supabase_url="https://x.supabase.co",
            service_role_key="key",
        )
        self.assertEqual(result, [])

    def test_returns_empty_when_no_search_terms(self):
        result = movement_library.fetch_movement_clips_for_subject(
            search_terms=[],
            required_duration=30,
            max_clip_duration=6,
            material_directory="",
            gemini_api_key="key",
            supabase_url="https://x.supabase.co",
            service_role_key="key",
        )
        self.assertEqual(result, [])

    def test_stops_once_required_duration_is_covered(self):
        clips = [self._clip(f"c{i}/0.mp4", 6.0) for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch.object(movement_library, "embed_text", return_value=[1.0, 0.0]),
                patch.object(
                    movement_library, "find_similar_clips", return_value=clips
                ),
                patch.object(movement_library, "download_clip") as download,
            ):
                result = movement_library.fetch_movement_clips_for_subject(
                    search_terms=["hammer"],
                    required_duration=13,  # needs 3 six-second clips, not 5
                    max_clip_duration=6,
                    material_directory=tmp_dir,
                    gemini_api_key="key",
                    supabase_url="https://x.supabase.co",
                    service_role_key="key",
                )

        self.assertEqual(len(result), 3)
        self.assertEqual(download.call_count, 3)

    def test_skips_clips_already_cached_on_disk(self):
        clip = self._clip("cached/0.mp4", 6.0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            cached_path = os.path.join(
                tmp_dir, f"movement-{movement_library.utils.md5(clip.storage_path)}.mp4"
            )
            with open(cached_path, "wb") as f:
                f.write(b"already-downloaded")

            with (
                patch.object(movement_library, "embed_text", return_value=[1.0, 0.0]),
                patch.object(
                    movement_library, "find_similar_clips", return_value=[clip]
                ),
                patch.object(movement_library, "download_clip") as download,
            ):
                result = movement_library.fetch_movement_clips_for_subject(
                    search_terms=["hammer"],
                    required_duration=6,
                    max_clip_duration=6,
                    material_directory=tmp_dir,
                    gemini_api_key="key",
                    supabase_url="https://x.supabase.co",
                    service_role_key="key",
                )

        download.assert_not_called()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].local_path, cached_path)

    def test_a_single_failed_download_does_not_abort_the_rest(self):
        clips = [self._clip("bad/0.mp4", 6.0), self._clip("good/0.mp4", 6.0)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch.object(movement_library, "embed_text", return_value=[1.0, 0.0]),
                patch.object(
                    movement_library, "find_similar_clips", return_value=clips
                ),
                patch.object(
                    movement_library,
                    "download_clip",
                    side_effect=[RuntimeError("boom"), None],
                ),
            ):
                result = movement_library.fetch_movement_clips_for_subject(
                    search_terms=["hammer"],
                    required_duration=6,
                    max_clip_duration=6,
                    material_directory=tmp_dir,
                    gemini_api_key="key",
                    supabase_url="https://x.supabase.co",
                    service_role_key="key",
                )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].source_asset_id, "good/0.mp4")

    def test_never_raises_when_embedding_fails(self):
        with patch.object(
            movement_library, "embed_text", side_effect=RuntimeError("gemini down")
        ):
            result = movement_library.fetch_movement_clips_for_subject(
                search_terms=["hammer"],
                required_duration=30,
                max_clip_duration=6,
                material_directory="",
                gemini_api_key="key",
                supabase_url="https://x.supabase.co",
                service_role_key="key",
            )
        self.assertEqual(result, [])

    def test_never_raises_when_similarity_search_fails(self):
        with (
            patch.object(movement_library, "embed_text", return_value=[1.0, 0.0]),
            patch.object(
                movement_library,
                "find_similar_clips",
                side_effect=RuntimeError("supabase down"),
            ),
        ):
            result = movement_library.fetch_movement_clips_for_subject(
                search_terms=["hammer"],
                required_duration=30,
                max_clip_duration=6,
                material_directory="",
                gemini_api_key="key",
                supabase_url="https://x.supabase.co",
                service_role_key="key",
            )
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
