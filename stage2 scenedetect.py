from __future__ import annotations

import json
from pathlib import Path

import cv2
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from scenedetect.detectors import AdaptiveDetector


VIDEO_PATH = Path(__file__).with_name("output_0.5x.mp4")
OUTPUT_DIR = Path(__file__).with_name("scene_validation")

THRESHOLD = 50.0
MIN_SCENE_LEN = 24  # ⭐关键：约 1 秒（24fps）
TOLERANCE_SECONDS = 1.0 / 24.0


def get_seconds(tc) -> float:
    """
    Robust compatibility helper across PySceneDetect versions.
    """
    if hasattr(tc, "get_seconds"):
        return float(tc.get_seconds())
    if hasattr(tc, "seconds"):
        return float(tc.seconds)
    return float(tc)


def format_timecode(seconds: float) -> str:
    total_milliseconds = int(round(seconds * 1000.0))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def capture_frame(video_path: Path, timestamp_seconds: float, output_path: Path) -> bool:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return False

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_index = max(0, int(round(timestamp_seconds * fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    success, frame = capture.read()
    capture.release()

    if not success or frame is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(output_path), frame)


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1. Load video
    # -------------------------
    video = open_video(str(VIDEO_PATH))

    # -------------------------
    # 2. Scene detection
    # -------------------------
    scene_manager = SceneManager()

    scene_manager.add_detector(
        AdaptiveDetector(
            #threshold=THRESHOLD,
            min_scene_len=MIN_SCENE_LEN,
        )
    )

    scene_manager.detect_scenes(video=video, show_progress=True)

    # -------------------------
    # 3. Get scenes (correct API)
    # -------------------------
    scenes = scene_manager.get_scene_list(video)

    video_fps = float(getattr(video, "frame_rate", 0.0) or 0.0)
    video_duration = float(getattr(video, "duration", 0.0) or 0.0)

    summary = []
    continuity_issues = []

    previous_end = None

    # -------------------------
    # 4. Process scenes
    # -------------------------
    for index, scene in enumerate(scenes, start=1):

        start_seconds = get_seconds(scene[0])
        end_seconds = get_seconds(scene[1])

        summary.append({
            "index": index,
            "start_seconds": round(start_seconds, 3),
            "end_seconds": round(end_seconds, 3),
            "start_timecode": format_timecode(start_seconds),
            "end_timecode": format_timecode(end_seconds),
            "duration_seconds": round(end_seconds - start_seconds, 3),
        })

        # continuity check
        if previous_end is not None and abs(start_seconds - previous_end) > TOLERANCE_SECONDS:
            continuity_issues.append({
                "scene_index": index,
                "previous_end": round(previous_end, 3),
                "current_start": round(start_seconds, 3),
                "gap_seconds": round(start_seconds - previous_end, 3),
            })

        previous_end = end_seconds

        # save sample frames
        if index <= 50:
            capture_frame(
                VIDEO_PATH,
                start_seconds,
                OUTPUT_DIR / f"scene_{index:03d}_start.jpg",
            )

            if index > 1:
                capture_frame(
                    VIDEO_PATH,
                    max(0.0, start_seconds - 0.5),
                    OUTPUT_DIR / f"scene_{index:03d}_pre.jpg",
                )

    # -------------------------
    # 5. Output JSON
    # -------------------------
    validation = {
        "video": VIDEO_PATH.name,
        "scene_count": len(scenes),
        "video_fps": round(video_fps, 3),
        "video_duration_seconds": round(video_duration, 3),
        "threshold": THRESHOLD,
        "min_scene_len": MIN_SCENE_LEN,
        "tolerance_seconds": TOLERANCE_SECONDS,
        "continuous": len(continuity_issues) == 0,
        "continuity_issues": continuity_issues,
        "scenes": summary,
    }

    (OUTPUT_DIR / "scenes.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # -------------------------
    # 6. print summary
    # -------------------------
    print(f"Scene count: {len(scenes)}")
    print(f"Video duration: {video_duration:.3f}s")
    print(f"Continuous: {len(continuity_issues) == 0}")

    if continuity_issues:
        print("Continuity issues:")
        for issue in continuity_issues[:10]:
            print(
                f"  scene {issue['scene_index']}: gap {issue['gap_seconds']:+.3f}s "
                f"(prev end {issue['previous_end']:.3f}s → start {issue['current_start']:.3f}s)"
            )

    print(f"Saved validation data to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()