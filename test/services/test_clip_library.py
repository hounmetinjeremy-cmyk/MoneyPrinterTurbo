import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import clip_library
from app.utils import utils


def _make_synthetic_video(path: str, num_seconds: int = 6, fps: int = 10) -> None:
    """
    Build a tiny test video: solid color frames that change over time (so
    there is always real motion to detect) and never contain a face.
    """
    result = subprocess.run(
        [
            utils.get_ffmpeg_binary(),
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={num_seconds}:size=64x64:rate={fps}",
            "-pix_fmt",
            "yuv420p",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to build synthetic test video: {result.stderr}")


class TestMotionAndFaceScoring(unittest.TestCase):
    def test_identical_frames_score_zero_motion(self):
        frame = np.full((32, 32, 3), 100, dtype=np.uint8)
        self.assertEqual(clip_library.compute_motion_score(frame, frame), 0.0)

    def test_very_different_frames_score_high_motion(self):
        black = np.zeros((32, 32, 3), dtype=np.uint8)
        white = np.full((32, 32, 3), 255, dtype=np.uint8)
        score = clip_library.compute_motion_score(black, white)
        self.assertGreater(score, 0.9)

    def test_frame_has_visible_face_uses_cascade_detection(self):
        """
        Detection quality itself belongs to OpenCV's own test suite — this
        only verifies frame_has_visible_face reports whatever the cascades
        found, so the rest of the pipeline's face-filtering logic can be
        tested deterministically without needing a real photographed face.
        """
        frame = np.zeros((48, 48, 3), dtype=np.uint8)
        with (
            patch.object(clip_library, "_face_cascade") as mock_frontal_getter,
            patch.object(clip_library, "_profile_cascade") as mock_profile_getter,
        ):
            mock_frontal_getter.return_value.detectMultiScale.return_value = []
            mock_profile_getter.return_value.detectMultiScale.return_value = []
            self.assertFalse(clip_library.frame_has_visible_face(frame))

            mock_frontal_getter.return_value.detectMultiScale.return_value = [
                (5, 5, 20, 20)
            ]
            self.assertTrue(clip_library.frame_has_visible_face(frame))

    def test_frame_has_visible_face_catches_a_profile_the_frontal_cascade_misses(self):
        """
        A frontal-only cascade misses someone turned to the side or looking
        down at their work — exactly the poses common in "hands using tools"
        footage. The profile cascade must be consulted too, not just frontal.
        """
        frame = np.zeros((48, 48, 3), dtype=np.uint8)
        with (
            patch.object(clip_library, "_face_cascade") as mock_frontal_getter,
            patch.object(clip_library, "_profile_cascade") as mock_profile_getter,
        ):
            mock_frontal_getter.return_value.detectMultiScale.return_value = []
            mock_profile_getter.return_value.detectMultiScale.return_value = [
                (5, 5, 20, 20)
            ]
            self.assertTrue(clip_library.frame_has_visible_face(frame))

    def test_frame_has_visible_face_checks_the_flipped_frame_too(self):
        """
        haarcascade_profileface.xml only reliably matches one facing
        direction — a profile turned the other way needs the frame flipped
        before re-checking, or it's missed entirely.
        """
        frame = np.zeros((48, 48, 3), dtype=np.uint8)
        with (
            patch.object(clip_library, "_face_cascade") as mock_frontal_getter,
            patch.object(clip_library, "_profile_cascade") as mock_profile_getter,
        ):
            mock_frontal_getter.return_value.detectMultiScale.return_value = []
            # Not found on the frame as-is, but found once flipped.
            mock_profile_getter.return_value.detectMultiScale.side_effect = [
                [],
                [(5, 5, 20, 20)],
            ]
            self.assertTrue(clip_library.frame_has_visible_face(frame))


class TestSelectCleanSegments(unittest.TestCase):
    def test_drops_segments_with_a_visible_face(self):
        segments = [
            clip_library.SegmentAnalysis(0, 6, has_visible_face=True, motion_score=0.5),
            clip_library.SegmentAnalysis(
                6, 12, has_visible_face=False, motion_score=0.5
            ),
        ]
        result = clip_library.select_clean_segments(segments)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].start_time, 6)

    def test_drops_near_static_segments(self):
        segments = [
            clip_library.SegmentAnalysis(
                0, 6, has_visible_face=False, motion_score=0.001
            ),
            clip_library.SegmentAnalysis(
                6, 12, has_visible_face=False, motion_score=0.5
            ),
        ]
        result = clip_library.select_clean_segments(segments, min_motion_score=0.02)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].start_time, 6)

    def test_keeps_nothing_when_all_segments_are_disqualified(self):
        segments = [
            clip_library.SegmentAnalysis(0, 6, has_visible_face=True, motion_score=0.5),
            clip_library.SegmentAnalysis(
                6, 12, has_visible_face=False, motion_score=0.0
            ),
        ]
        self.assertEqual(clip_library.select_clean_segments(segments), [])


class TestAnalyzeAndCutRealVideo(unittest.TestCase):
    """
    Exercises analyze_video_segments and cut_segment_to_file against a real
    (synthetic) video file end-to-end, rather than mocking cv2.VideoCapture's
    internal frame-by-frame state — that state (current position, sequential
    read()) is exactly what's under test here.
    """

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.video_path = os.path.join(self.tmp_dir.name, "synthetic.mp4")
        _make_synthetic_video(self.video_path, num_seconds=6, fps=10)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_analyze_video_segments_reports_motion_and_no_face(self):
        segments = clip_library.analyze_video_segments(
            self.video_path, segment_duration=3.0, sample_stride_frames=2
        )
        self.assertEqual(len(segments), 2)
        for segment in segments:
            self.assertFalse(segment.has_visible_face)
            self.assertGreater(segment.motion_score, 0.0)

    def test_cut_segment_to_file_produces_a_playable_clip(self):
        output_path = os.path.join(self.tmp_dir.name, "clip.mp4")
        clip_library.cut_segment_to_file(self.video_path, 1.0, 3.0, output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_cut_segment_to_file_raises_on_ffmpeg_failure(self):
        with self.assertRaises(RuntimeError):
            clip_library.cut_segment_to_file(
                os.path.join(self.tmp_dir.name, "does-not-exist.mp4"),
                0,
                1,
                os.path.join(self.tmp_dir.name, "out.mp4"),
            )


if __name__ == "__main__":
    unittest.main()
