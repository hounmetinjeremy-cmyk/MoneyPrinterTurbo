"""Movement-clip library: cleans stock footage down to "pure movement" clips
(no visible faces, real motion) and stores them for later retrieval by topic.

This is deliberately independent from app/services/video.py's generation-time
material handling — it runs offline, once a day, building up a reusable
library rather than fetching fresh stock footage on every single video. See
scripts/build_clip_library.py for the cron entrypoint that wires this
together with Pexels search, Supabase Storage, and the movement_clips table.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from app.utils import utils

# Loaded lazily (see _face_cascade()) so importing this module — and running
# the pure-logic unit tests below — never requires an OpenCV/GUI-capable
# environment or the Haar cascade data file to be present.
_FACE_CASCADE = None

# A frame is judged to contain a visible face at this minimum size, expressed
# as a fraction of frame width — this filters out tiny incidental faces (a
# person far in the background) while still catching an on-camera subject.
_MIN_FACE_SIZE_FRACTION = 0.08


def _face_cascade():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        import cv2

        _FACE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _FACE_CASCADE


def frame_has_visible_face(frame: np.ndarray) -> bool:
    """
    Detect whether a single BGR frame contains a visible human face.

    Used to keep the movement-clip library to "pure movement" content (hands,
    tools, objects in motion) with no on-camera person — sidestepping the
    "why doesn't this person's mouth move" problem entirely rather than
    trying to solve lip-sync on footage of people who have nothing to do
    with the day's script.
    """
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    min_size = int(frame.shape[1] * _MIN_FACE_SIZE_FRACTION)
    faces = _face_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)
    )
    return len(faces) > 0


def compute_motion_score(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
    """
    Score how much visible motion happened between two consecutive frames.

    Normalized mean absolute pixel difference, 0.0 (identical frames — a
    static shot, a title card, a freeze frame) to 1.0 (maximum possible
    difference). Frames are compared in grayscale so color grading/exposure
    shifts between shots don't get mistaken for real motion.
    """
    import cv2

    if prev_frame.shape != curr_frame.shape:
        curr_frame = cv2.resize(curr_frame, (prev_frame.shape[1], prev_frame.shape[0]))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(np.abs(curr_gray - prev_gray)) / 255.0)


@dataclass
class SegmentAnalysis:
    start_time: float
    end_time: float
    has_visible_face: bool
    motion_score: float


def analyze_video_segments(
    video_path: str,
    segment_duration: float = 6.0,
    sample_stride_frames: int = 5,
) -> List[SegmentAnalysis]:
    """
    Split a video into fixed-length segments and score each one for face
    presence and motion, without decoding every single frame — sampling
    every `sample_stride_frames`-th frame keeps this fast enough to run over
    a day's worth of stock footage in a scheduled job.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video for analysis: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0

        segments: List[SegmentAnalysis] = []
        start_time = 0.0
        while start_time < duration:
            end_time = min(start_time + segment_duration, duration)
            has_face = False
            motion_scores: List[float] = []
            prev_sampled_frame: Optional[np.ndarray] = None

            first_frame_index = int(start_time * fps)
            last_frame_index = int(end_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_index)

            frame_index = first_frame_index
            while frame_index < last_frame_index:
                ok, frame = cap.read()
                if not ok:
                    break
                if (frame_index - first_frame_index) % sample_stride_frames == 0:
                    if not has_face and frame_has_visible_face(frame):
                        has_face = True
                    if prev_sampled_frame is not None:
                        motion_scores.append(
                            compute_motion_score(prev_sampled_frame, frame)
                        )
                    prev_sampled_frame = frame
                frame_index += 1

            segments.append(
                SegmentAnalysis(
                    start_time=start_time,
                    end_time=end_time,
                    has_visible_face=has_face,
                    motion_score=float(np.mean(motion_scores))
                    if motion_scores
                    else 0.0,
                )
            )
            start_time = end_time

        return segments
    finally:
        cap.release()


def select_clean_segments(
    segments: List[SegmentAnalysis],
    min_motion_score: float = 0.02,
) -> List[SegmentAnalysis]:
    """
    Keep only segments with no visible face and enough real motion to be
    worth adding to the library — discards both "someone on camera" shots
    and near-static filler (title cards, frozen establishing shots).
    """
    return [
        segment
        for segment in segments
        if not segment.has_visible_face and segment.motion_score >= min_motion_score
    ]


def cut_segment_to_file(
    video_path: str, start_time: float, end_time: float, output_path: str
) -> None:
    """Extract [start_time, end_time) from video_path into output_path via ffmpeg."""
    result = subprocess.run(
        [
            utils.get_ffmpeg_binary(),
            "-y",
            "-ss",
            str(start_time),
            "-to",
            str(end_time),
            "-i",
            video_path,
            "-an",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to cut segment [{start_time}, {end_time}) "
            f"from {video_path}: {result.stderr.strip()}"
        )
