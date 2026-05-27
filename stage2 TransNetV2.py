from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from transnetv2 import TransNetV2


VIDEO_PATH = Path(__file__).with_name("10THAU.mp4")
OUTPUT_PATH = Path(__file__).with_name("stage2_TransNetV2_scenes.json")


def _format_timecode(seconds: float) -> str:
    total_milliseconds = int(round(seconds * 1000.0))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


# -----------------------------
# model loading
# -----------------------------
def _load_model() -> Optional[TransNetV2]:
    try:
        return TransNetV2()
    except Exception:
        return None


def _read_video_frames(video_path: Path, size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        command = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel", "error",
            "-err_detect", "ignore_err",
            "-fflags", "+discardcorrupt",
            "-i", str(video_path),
            "-vf", f"scale={width}:{height}",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        completed = subprocess.run(command, capture_output=True, check=False)
        if completed.stdout:
            frame_size = width * height * 3
            usable_bytes = len(completed.stdout) // frame_size * frame_size
            if usable_bytes > 0:
                return np.frombuffer(completed.stdout[:usable_bytes], dtype=np.uint8).reshape([-1, height, width, 3])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()

    if not frames:
        return np.empty((0, height, width, 3), dtype=np.uint8)
    return np.asarray(frames, dtype=np.uint8)


# -----------------------------
# chunk inference (sliding window simulation)
# -----------------------------
def _predict_in_chunks(model: TransNetV2, frames: np.ndarray, chunk_size: int = 1000):
    """
    Simulates sliding window inference.
    Avoids long-sequence bias in full-video forward pass.
    """
    all_preds = []

    for start in range(0, len(frames), chunk_size):
        end = min(start + chunk_size, len(frames))
        chunk = frames[start:end]

        _, preds = model.predict_frames(chunk)
        all_preds.append(preds)

    return np.concatenate(all_preds, axis=0)


# -----------------------------
# TransNetV2 detection
# -----------------------------
def _detect_shots_transnetv2(video_path: Path) -> Optional[List[Tuple[float, float]]]:
    model = _load_model()
    if model is None:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    frames = _read_video_frames(video_path, size=(48, 27))

    if len(frames) == 0:
        return None

    # -----------------------------
    # ✅ chunk inference (关键改动)
    # -----------------------------
    frame_predictions = _predict_in_chunks(model, frames, chunk_size=800)

    # -----------------------------
    # smoothing
    # -----------------------------
    frame_predictions = gaussian_filter1d(frame_predictions, sigma=1.0)

    # -----------------------------
    # lower threshold (movie-safe)
    # -----------------------------
    scene_list = model.predictions_to_scenes(
        frame_predictions,
        threshold=0.3   # ✔ 改回合理范围（0.1–0.3更电影友好）
    )

    shots: List[Tuple[float, float]] = []
    for start_f, end_f in scene_list:
        shots.append((start_f / fps, (end_f + 1) / fps))

    return shots


# -----------------------------
# fallback
# -----------------------------
def _detect_shots_fallback(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    frames = _read_video_frames(video_path, size=(96, 54))

    prev = None
    diffs = []
    times = []

    if len(frames) == 0:
        return [(0.0, 0.0)]

    sample_step = max(1, int(round(fps)))
    for i, frame in enumerate(frames):
        if i % sample_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            if prev is not None:
                diffs.append(cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA))
                times.append(i / fps)

            prev = hist

    if not diffs:
        return [(0.0, len(frames) / fps)]

    diffs = np.array(diffs)
    threshold = np.mean(diffs) + 1.2 * np.std(diffs)

    cuts = np.where(diffs > threshold)[0]

    boundaries = [0.0] + [times[c] for c in cuts] + [len(frames) / fps]
    boundaries = sorted(set(boundaries))

    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


# -----------------------------
# main
# -----------------------------
def main():
    shots = _detect_shots_transnetv2(VIDEO_PATH)

    backend = "transnetv2"
    if shots is None:
        backend = "opencv-fallback"
        shots = _detect_shots_fallback(VIDEO_PATH)

    scenes = []
    for index, (start_seconds, end_seconds) in enumerate(shots, start=1):
        scenes.append(
            {
                "index": index,
                "start_seconds": round(float(start_seconds), 3),
                "end_seconds": round(float(end_seconds), 3),
                "start_timecode": _format_timecode(float(start_seconds)),
                "end_timecode": _format_timecode(float(end_seconds)),
                "duration_seconds": round(float(end_seconds) - float(start_seconds), 3),
            }
        )

    OUTPUT_PATH.write_text(
        json.dumps(
            {"video": VIDEO_PATH.name, "backend": backend, "scenes": scenes},
            indent=2
        ),
        encoding="utf-8"
    )

    print("Backend:", backend)
    print("Shot count:", len(shots))
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()