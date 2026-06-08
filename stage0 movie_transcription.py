import os
import subprocess
import argparse
import torch
from faster_whisper import WhisperModel

# --- PATH CONFIGURATION ---
INPUT_DIR = r"D:\videos\Video"
OUTPUT_SRT_DIR = r"D:\videos\subtitles"
TEMP_AUDIO_DIR = r"D:\videos\temp_audio"

os.makedirs(OUTPUT_SRT_DIR, exist_ok=True)
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)


def run_ffmpeg(video_path, wav_path):
    """Extract stable PCM WAV for Whisper input."""
    if os.path.exists(wav_path):
        return wav_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-avoid_negative_ts", "make_zero",
        "-loglevel", "error",
        wav_path
    ]

    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return wav_path


def resolve_device(requested="auto"):
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def format_time(seconds):
    ms = int((seconds % 1) * 1000)
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    s = s % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            f.write(f"{i+1}\n")
            f.write(f"{format_time(seg.start)} --> {format_time(seg.end)}\n")
            f.write(seg.text.strip() + "\n\n")


def transcribe_file(file_name, model_size, device="auto"):
    file_path = os.path.join(INPUT_DIR, file_name)
    base_name = os.path.splitext(file_name)[0]

    wav_path = os.path.join(TEMP_AUDIO_DIR, f"{base_name}.wav")
    output_srt = os.path.join(OUTPUT_SRT_DIR, f"{base_name}.srt")

    print(f"\n[INFO] Processing: {file_name}")

    device = resolve_device(device)
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"[INFO] device={device}, compute_type={compute_type}")

    # 1. Always use ffmpeg output (NO librosa)
    ext = os.path.splitext(file_name)[1].lower()

    if ext == ".wav":
        input_audio = file_path
    else:
        input_audio = run_ffmpeg(file_path, wav_path)

    # 2. Load model ONCE per file (stable version)
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type
    )

    # 3. Transcribe (NO VAD, NO skipping, FULL COVERAGE)
    segments, info = model.transcribe(
        input_audio,

        beam_size=1,              # ⚡ stability > accuracy
        vad_filter=False,         # ❌ disable all skipping
        condition_on_previous_text=False,  # ⚡ prevents drift/skip bug
        word_timestamps=False     # ⚡ avoids timestamp instability
    )

    segments = list(segments)

    # 4. Safety check (VERY important)
    if len(segments) == 0:
        print(f"[WARN] No segments detected in {file_name}")

    # 5. Write SRT
    write_srt(segments, output_srt)

    print(f"[OK] {output_srt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="large-v3")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    VALID_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".wav", ".mp3", ".m4a")

    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(VALID_EXTS)
    ]

    print(f"[INFO] Found {len(files)} files")

    for f in files:
        try:
            transcribe_file(
                f,
                args.model,
                args.device
            )
        except Exception as e:
            print(f"[FAILED] {f}: {e}")


if __name__ == "__main__":
    main()