#!/usr/bin/env python3
"""
Stage 1 — Script-Video Temporal Alignment

Goal:
- Align script text segments with video segments (two modalities only).

Inputs:
- video file
- script file (.txt/.json/.csv/.srt)

Output:
- zt_bins.json containing script-aligned temporal bins.

Notes:
- Uses CLIP for both text and image embeddings in the same feature space.
- If script has explicit timestamps, those timestamps are used directly.
- If script has no timestamps, script segments are aligned to fixed video windows.
"""

import argparse
import csv
import importlib
import json
import math
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    import torch
    from transformers import AutoProcessor, CLIPModel
    from sentence_transformers import SentenceTransformer
except Exception:
    torch = None
    AutoProcessor = None
    CLIPModel = None
    SentenceTransformer = None

try:
    import librosa
except Exception:
    librosa = None

def parse_srt(srt_path: str) -> List[Tuple[float, float, str]]:
    """Small SRT parser returning list of (start_sec, end_sec, text)."""
    items = []
    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    parts = re.split(r"\n\s*\n", content)
    time_re = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})")

    def to_sec(t: str) -> float:
        hh, mm, rest = t.split(":")
        ss, ms = rest.split(",")
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0

    for p in parts:
        m = time_re.search(p)
        if not m:
            continue

        start = to_sec(m.group(1))
        end = to_sec(m.group(2))

        lines = p.splitlines()
        text_lines = []
        for i, line in enumerate(lines):
            if time_re.search(line):
                text_lines = lines[i + 1 :]
                break

        text = " ".join([ln.strip() for ln in text_lines if ln.strip()])
        items.append((start, end, text))

    return items


def parse_time_to_sec(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    m = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})([\.,](\d{1,3}))?", s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = int(m.group(3))
    ms = int((m.group(5) or "0").ljust(3, "0"))
    return hh * 3600 + mm * 60 + ss + ms / 1000.0


def parse_script(script_path: str) -> List[Dict[str, Any]]:
    """Parse script into [{'text': str, 'start': float|None, 'end': float|None}, ...]."""
    lower = script_path.lower()

    if lower.endswith(".srt"):
        srt_segments = parse_srt(script_path)
        return [{"text": t, "start": s, "end": e} for (s, e, t) in srt_segments]

    if lower.endswith(".json"):
        with open(script_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("segments", data.get("bins", []))
        if not isinstance(data, list):
            raise ValueError("JSON script must be a list or have 'segments' key")

        out = []
        for item in data:
            if isinstance(item, str):
                txt = item.strip()
                if txt:
                    out.append({"text": txt, "start": None, "end": None})
                continue
            if not isinstance(item, dict):
                continue

            txt = str(item.get("text", item.get("dialogue", ""))).strip()
            if not txt:
                continue

            out.append(
                {
                    "text": txt,
                    "start": parse_time_to_sec(item.get("start")),
                    "end": parse_time_to_sec(item.get("end")),
                }
            )
        return out

    if lower.endswith(".csv"):
        out = []
        with open(script_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (row.get("text") or row.get("dialogue") or row.get("line") or "").strip()
                if not text:
                    continue
                out.append(
                    {
                        "text": text,
                        "start": parse_time_to_sec(row.get("start") or row.get("start_sec")),
                        "end": parse_time_to_sec(row.get("end") or row.get("end_sec")),
                    }
                )
        return out

    if lower.endswith(".pdf"):
        try:
            pypdf = importlib.import_module("pypdf")
        except Exception as exc:
            raise ImportError("PDF parsing requires pypdf. Please install with: pip install pypdf") from exc

        reader = pypdf.PdfReader(script_path)
        boilerplate_re = re.compile(
            r"^(?:page\s+\d+\s+of\s+\d+|"
            r"source:\s*imsdb|"
            r"https?://|"
            r"formatted for maxqda auto-coding:?|"
            r"scene headings\s*—\s*bold uppercase\s*\|\s*character names\s*—\s*uppercase, indented\s*\|\s*transitions\s*—\s*right-aligned)$",
            re.IGNORECASE,
        )

        def normalize_line(raw_line: str) -> str:
            return re.sub(r"\s+", " ", raw_line).strip()

        def is_scene_heading(line: str) -> bool:
            upper = line.upper()
            if not line or len(line) > 90:
                return False
            if re.match(r"^(?:INT|EXT|I/E|INT/EXT|EST)\b", upper):
                return True
            if not line.isupper():
                return False
            keywords = (
                " - DAY",
                " - NIGHT",
                " - CONTINUOUS",
                " - LATER",
                "MONTAGE",
                "CONTINUOUS",
                "DAY",
                "NIGHT",
                "FADE IN",
                "FADE OUT",
            )
            return any(token in upper for token in keywords) and len(upper.split()) <= 12

        def is_character_cue(line: str) -> bool:
            if not line or len(line) > 45:
                return False
            if line.startswith("(") or line.endswith(":"):
                return False
            if not line.isupper():
                return False
            words = line.split()
            return 1 <= len(words) <= 4

        def is_parenthetical(line: str) -> bool:
            return line.startswith("(") and line.endswith(")")

        segments = []
        current_lines = []
        current_type = None
        story_started = False

        def flush_current() -> None:
            nonlocal current_lines, current_type
            if current_lines:
                text = " ".join(current_lines)
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    segments.append(text)
            current_lines = []
            current_type = None

        for page in reader.pages:
            txt = page.extract_text() or ""
            for raw_line in txt.split("\n"):
                if not raw_line.strip():
                    continue

                line = normalize_line(raw_line)
                if len(line) < 3 or boilerplate_re.match(line):
                    continue

                if is_scene_heading(line):
                    story_started = True
                    flush_current()
                    current_lines = [line]
                    current_type = "scene"
                    continue

                if not story_started:
                    continue

                if is_character_cue(line):
                    flush_current()
                    current_lines = [line]
                    current_type = "cue"
                    continue

                if is_parenthetical(line):
                    if current_type in {"cue", "dialogue"}:
                        current_lines.append(line)
                        current_type = "dialogue"
                    else:
                        flush_current()
                        current_lines = [line]
                        current_type = "dialogue"
                    continue

                if current_type is None:
                    current_lines = [line]
                    current_type = "action"
                    continue

                if current_type == "scene":
                    current_lines.append(line)
                    current_type = "action"
                    continue

                if current_type == "cue":
                    current_lines.append(line)
                    current_type = "dialogue"
                    continue

                if current_type == "dialogue":
                    if is_scene_heading(line) or is_character_cue(line):
                        flush_current()
                        current_lines = [line]
                        current_type = "scene" if is_scene_heading(line) else "cue"
                    else:
                        current_lines.append(line)
                    continue

                current_lines.append(line)

        flush_current()

        if not segments:
            return []

        return [{"text": seg, "start": None, "end": None} for seg in segments]

    with open(script_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    blocks = [b.strip() for b in re.split(r"\n\s*\n", content) if b.strip()]
    return [{"text": b.replace("\n", " "), "start": None, "end": None} for b in blocks]


def sample_frames_for_timed_segments(
    video_path: str,
    segments: List[Tuple[float, float]],
    sample_fps: float = 1.0,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Video FPS: {video_fps:.2f}")

    schedule = []
    frames = {(s, e): [] for (s, e) in segments}
    for (s, e) in segments:
        t = s
        step = 1.0 / sample_fps if sample_fps > 0 else 0.0
        while sample_fps > 0 and t < e:
            schedule.append((t, (s, e)))
            t += step

    schedule.sort(key=lambda x: x[0])
    if not schedule:
        cap.release()
        return frames

    idx = 0
    target_t, target_seg = schedule[idx]
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_t = frame_id / video_fps
        while current_t >= target_t:
            frames[target_seg].append((current_t, frame.copy()))
            idx += 1
            if idx >= len(schedule):
                cap.release()
                return frames
            target_t, target_seg = schedule[idx]

        frame_id += 1

    cap.release()
    return frames





def sample_video_windows(video_path: str, window_sec: float = 4.0, sample_fps: float = 1.0, stride_sec: Optional[float] = None):
    # sliding-window wrapper: stride_sec controls overlap; defaults to window_sec (no overlap)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    print(f"[INFO] Video: {fps:.2f} FPS, {frame_count} frames, {duration:.2f} sec")

    if stride_sec is None:
        stride_sec = window_sec

    windows = []
    t = 0.0
    while t < duration:
        windows.append({"start": t, "end": min(t + window_sec, duration), "frames": []})
        t += stride_sec
    print(f"[INFO] Created {len(windows)} sliding windows with stride={stride_sec:.2f}s, window={window_sec:.2f}s")

    if sample_fps <= 0 or not windows:
        return windows

    # fill frames per sample time
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot re-open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    schedule = []
    step = 1.0 / sample_fps
    t = 0.0
    while t < duration:
        schedule.append(t)
        t += step
    print(f"[INFO] Sampling {len(schedule)} frames at {sample_fps:.2f} FPS")

    idx = 0
    target_t = schedule[idx] if schedule else None
    frame_id = 0
    frame_count_read = 0

    while target_t is not None:
        try:
            ret, frame = cap.read()
        except Exception as e:
            print(f"[ERROR] Failed to read frame {frame_id}: {e}")
            break
        frame_count_read += 1
        if not ret:
            print(f"[INFO] End of video at frame {frame_id}")
            break

        current_t = frame_id / video_fps
        while target_t is not None and current_t >= target_t:
            # assign to any window that covers target_t
            for w in windows:
                if w["start"] - 1e-6 <= target_t <= w["end"] + 1e-6:
                    w["frames"].append((target_t, frame.copy()))
                    break
            idx += 1
            if idx >= len(schedule):
                target_t = None
            else:
                target_t = schedule[idx]

        frame_id += 1

    cap.release()
    print(f"[INFO] Read {frame_count_read} frames, assigned to windows")
    return windows


def compute_frame_and_window_embeddings(windows, processor, model, device="cpu", batch_size=16):
    """Return (frame_embs_per_window, window_avg_embs).
    frame_embs_per_window: list of lists of (t, vec)
    window_avg_embs: list of averaged vectors (or None)
    """
    if processor is None or model is None or torch is None:
        warnings.warn("CLIP not available; visual embeddings are empty")
        return [[ ] for _ in windows], [None for _ in windows]

    frame_embs_per_window = []
    window_avg_embs = []

    for w in tqdm(windows, desc="Visual embeddings"):
        frames = w.get("frames", [])
        if not frames:
            frame_embs_per_window.append([])
            window_avg_embs.append(None)
            continue

        imgs = [cv2.cvtColor(f[1], cv2.COLOR_BGR2RGB) for f in frames]
        times = [f[0] for f in frames]
        all_feats = []
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i : i + batch_size]
            inputs = processor(images=batch, return_tensors="pt")
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            all_feats.append(feats.cpu().numpy())

        arr = np.concatenate(all_feats, axis=0)
        per_frame = [(times[i], arr[i]) for i in range(arr.shape[0])]
        frame_embs_per_window.append(per_frame)
        avg = arr.mean(axis=0)
        avg = _l2_normalize(avg)
        window_avg_embs.append(avg.tolist())

    return frame_embs_per_window, window_avg_embs


def load_clip(device: str):
    if AutoProcessor is None or CLIPModel is None or torch is None:
        return None, None
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    model: Any = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()
    return processor, model


def load_text_encoder(model_name: str, device: str):
    if SentenceTransformer is None:
        warnings.warn("sentence-transformers is not available; falling back to CLIP text encoder")
        return None
    try:
        encoder = SentenceTransformer(model_name, device=device)
        return encoder
    except Exception as exc:
        warnings.warn(f"Could not load text encoder '{model_name}': {exc}; falling back to CLIP text encoder")
        return None


def build_contextual_texts(texts: List[str], radius: int = 1, max_chars: int = 600) -> List[str]:
    contextual = []
    n = len(texts)
    for i in range(n):
        start = max(0, i - radius)
        end = min(n, i + radius + 1)
        parts = []
        for j in range(start, end):
            txt = re.sub(r"\s+", " ", str(texts[j])).strip()
            if txt:
                parts.append(txt)
        merged = " [SEP] ".join(parts).strip()
        if len(merged) > max_chars:
            merged = merged[:max_chars]
        contextual.append(merged)
    return contextual


def encode_texts_with_sentence_transformer(
    texts: List[str],
    encoder,
    batch_size: int = 32,
    context_radius: int = 1,
    max_chars: int = 600,
) -> Optional[np.ndarray]:
    if encoder is None:
        return None
    contextual_texts = build_contextual_texts(texts, radius=context_radius, max_chars=max_chars)
    embeddings = encoder.encode(
        contextual_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=float)


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(arr)
    if nrm < 1e-12:
        return arr
    return arr / nrm


def compute_visual_embeddings(windows, processor, model, device="cpu", batch_size=16):
    out = []
    if processor is None or model is None or torch is None:
        warnings.warn("CLIP not available; visual embeddings are empty")
        return [None for _ in windows]

    for w in tqdm(windows, desc="Visual embeddings"):
        frames = w.get("frames", [])
        if not frames:
            out.append(None)
            continue

        imgs = [cv2.cvtColor(f[1], cv2.COLOR_BGR2RGB) for f in frames]
        all_feats = []
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i : i + batch_size]
            inputs = processor(images=batch, return_tensors="pt")
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            all_feats.append(feats.cpu().numpy())

        feat = np.concatenate(all_feats, axis=0).mean(axis=0)
        feat = _l2_normalize(feat)
        out.append(feat.tolist())

    return out


def compute_script_text_embeddings(script_segments, processor, model, device="cpu", batch_size=32):
    if processor is None or model is None or torch is None:
        warnings.warn("CLIP not available; text embeddings are empty")
        return [None for _ in script_segments]

    texts = [s["text"] for s in script_segments]
    out = [None for _ in script_segments]

    for i in tqdm(range(0, len(texts), batch_size), desc="Text embeddings"):
        batch = texts[i : i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        arr = feats.cpu().numpy()
        for j, vec in enumerate(arr):
            out[i + j] = vec.tolist()

    return out


def compute_audio_embeddings_for_bins(
    video_path: str,
    bins: List[Dict[str, Any]],
    sr: int = 16000,
    n_mfcc: int = 40,
) -> List[Optional[List[float]]]:
    """Compute simple MFCC-based audio embeddings for each bin time interval.

    Returns one normalized vector per bin, or None when unavailable.
    """
    if librosa is None:
        warnings.warn("librosa is not available; audio_emb will remain empty")
        return [None] * len(bins)

    try:
        y, _ = librosa.load(video_path, sr=sr, mono=True)
    except Exception as exc:
        warnings.warn(f"Could not load audio from video: {exc}; audio_emb will remain empty")
        return [None] * len(bins)

    out: List[Optional[List[float]]] = []
    n_samples = len(y)
    for b in bins:
        start = float(b.get("start", 0.0) or 0.0)
        end = float(b.get("end", start) or start)
        if end < start:
            end = start

        ist = max(0, int(start * sr))
        ied = min(n_samples, int(end * sr))
        if ied <= ist:
            out.append(None)
            continue

        seg = y[ist:ied]
        if seg.size < max(64, n_mfcc):
            out.append(None)
            continue

        try:
            mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=n_mfcc)
            vec = np.mean(mfcc, axis=1).astype(float)
            vec = _l2_normalize(vec)
            out.append(vec.tolist())
        except Exception:
            out.append(None)

    return out


def cosine(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom < 1e-12:
        return None
    return float(np.dot(va, vb) / denom)


def compute_similarity_matrix(text_embs: List[Any], window_embs: List[Any]) -> np.ndarray:
    T = len(text_embs)
    W = len(window_embs)
    S = np.full((T, W), -np.inf, dtype=float)
    for i in range(T):
        for j in range(W):
            if text_embs[i] is None or window_embs[j] is None:
                continue
            s = cosine(text_embs[i], window_embs[j])
            if s is None:
                continue
            S[i, j] = s
    return S


def select_top_k_candidates(S: np.ndarray, top_k: int) -> List[np.ndarray]:
    candidates = []
    for i in range(S.shape[0]):
        row = S[i]
        finite_idx = np.where(np.isfinite(row))[0]
        if finite_idx.size == 0:
            candidates.append(np.array([], dtype=int))
            continue
        ordered = finite_idx[np.argsort(-row[finite_idx])]
        candidates.append(ordered[: max(1, top_k)])
    return candidates


def smooth_confidence_series(values: List[float]) -> List[float]:
    if not values:
        return values
    out = []
    for i, value in enumerate(values):
        prev_value = values[i - 1] if i > 0 else value
        next_value = values[i + 1] if i + 1 < len(values) else value
        out.append(float(0.25 * prev_value + 0.5 * value + 0.25 * next_value))
    return out


def fill_missing_timestamps(script_segments: List[Dict[str, Any]], default_duration: float = 1.0, video_duration: Optional[float] = None) -> List[Dict[str, Any]]:
    filled = [dict(seg) for seg in script_segments]
    n = len(filled)
    starts = [seg.get("start") for seg in filled]
    ends = [seg.get("end") for seg in filled]

    anchor_indices = [i for i, seg in enumerate(filled) if seg.get("start") is not None and seg.get("end") is not None]
    if not anchor_indices:
        t = 0.0
        max_allowed = video_duration if video_duration is not None else float('inf')
        for seg in filled:
            seg["start"] = float(t)
            seg["end"] = float(min(t + default_duration, max_allowed))
            seg["aligned_to_transcription"] = False
            t += default_duration
        return filled

    first_anchor = anchor_indices[0]
    t = float(starts[first_anchor] or 0.0)
    for i in range(first_anchor - 1, -1, -1):
        seg = filled[i]
        seg["end"] = float(t)
        seg["start"] = float(max(0.0, t - default_duration))
        t = seg["start"]
        seg["aligned_to_transcription"] = False

    for i in range(first_anchor, n):
        seg = filled[i]
        if seg.get("start") is not None and seg.get("end") is not None:
            t = float(seg["end"])
            continue
        seg["start"] = float(t)
        seg["end"] = float(t + default_duration)
        t = seg["end"]
        seg["aligned_to_transcription"] = False

    # Reconcile gaps between anchored regions by linear interpolation.
    i = 0
    while i < n:
        if filled[i].get("start") is not None and filled[i].get("end") is not None:
            i += 1
            continue
        gap_start = i
        while i < n and (filled[i].get("start") is None or filled[i].get("end") is None):
            i += 1
        gap_end = i
        left = gap_start - 1
        right = gap_end if gap_end < n else None
        if left >= 0 and right is not None and filled[left].get("end") is not None and filled[right].get("start") is not None:
            left_t = float(filled[left]["end"])
            right_t = float(filled[right]["start"])
            span = max(right - left, 1)
            for k, idx in enumerate(range(gap_start, gap_end), start=1):
                alpha = k / (span + 0.0)
                start = left_t + alpha * (right_t - left_t)
                end = left_t + ((k + 1) / (span + 1.0)) * (right_t - left_t)
                filled[idx]["start"] = float(min(start, end))
                filled[idx]["end"] = float(max(start, end))
                filled[idx]["aligned_to_transcription"] = False

    # Final monotonic clamp for safety, with video_duration cap if provided.
    last_end = 0.0
    max_allowed = video_duration if video_duration is not None else float('inf')
    for seg in filled:
        if seg.get("start") is None or seg.get("end") is None:
            seg["start"] = float(last_end)
            seg["end"] = float(min(last_end + default_duration, max_allowed))
        if seg["start"] < last_end:
            shift = last_end - seg["start"]
            seg["start"] += shift
            seg["end"] += shift
        if seg["end"] < seg["start"]:
            seg["end"] = float(min(seg["start"] + default_duration, max_allowed))
        # Cap at video duration if provided
        if video_duration is not None:
            seg["start"] = float(min(seg["start"], max_allowed))
            seg["end"] = float(min(seg["end"], max_allowed))
        last_end = float(seg["end"])

    return filled


def mutual_nn_mask(S: np.ndarray, top_k_text: int = 5, top_k_window: int = 5) -> np.ndarray:
    T, W = S.shape
    if T == 0 or W == 0:
        return np.zeros_like(S, dtype=bool)
    mask_text = np.zeros_like(S, dtype=bool)
    mask_window = np.zeros_like(S, dtype=bool)
    # for each text, mark top_k_text windows
    for i in range(T):
        idx = np.argsort(-S[i])[:max(0, top_k_text)]
        mask_text[i, idx] = True
    # for each window, mark top_k_window texts
    for j in range(W):
        idx = np.argsort(-S[:, j])[:max(0, top_k_window)]
        mask_window[idx, j] = True
    return mask_text & mask_window


def dp_monotonic(S: np.ndarray) -> Tuple[List[Optional[int]], List[Optional[float]]]:
    """Dynamic programming to find monotonic one-to-one alignment maximizing sum of S entries.
    Returns (path_indices_per_text, scores_per_text) where indices are window indices or None.
    """
    T, W = S.shape
    if T == 0 or W == 0:
        return [None] * T, [None] * T

    # If any row has no finite entries, we'll keep them None but still attempt DP on finite parts.
    neg_inf = -1e9
    dp = np.full((T, W), neg_inf, dtype=float)
    ptr = np.full((T, W), -1, dtype=int)

    dp[0] = np.where(np.isfinite(S[0]), S[0], neg_inf)

    for i in range(1, T):
        prev = dp[i - 1]
        # prefix maximum and argmax
        prefix_max = np.maximum.accumulate(prev)
        argmax_prefix = np.zeros(W, dtype=int)
        best = 0
        for j in range(W):
            if prev[j] > prev[best]:
                best = j
            argmax_prefix[j] = best

        # dp value: choose best previous up to j
        vals = np.where(np.isfinite(S[i]), S[i] + prefix_max, neg_inf)
        dp[i] = vals
        for j in range(W):
            ptr[i, j] = int(argmax_prefix[j])

    # if no finite in last row -> nothing
    if not np.isfinite(dp[-1]).any():
        return [None] * T, [None] * T

    # backtrack
    path = [None] * T
    scores = [None] * T
    j = int(np.argmax(dp[-1]))
    for i in range(T - 1, -1, -1):
        if not np.isfinite(S[i, j]):
            path[i] = None
            scores[i] = None
        else:
            path[i] = int(j)
            scores[i] = float(S[i, j])
        if i > 0:
            j = int(ptr[i, j])

    return path, scores


def align_script_to_transcription_dp(
    script_segments,
    trans_tuples,
    text_encoder,
    device="cpu",
    context_radius: int = 1,
    top_k_candidates: int = 8,
    null_penalty: float = -0.12,
    jump_penalty: float = 0.03,
    time_penalty: float = 0.02,
    global_time_penalty: float = 0.02,
    score_threshold: float = 0.30,
    debug_indices: Optional[List[int]] = None,
    video_path: Optional[str] = None,
):
    """
    Align script to transcription with a semantic encoder + contextual smoothing.

    Uses a soft candidate lattice (top-k per script segment) and null-aware DP.
    Returns script segments with timestamps, confidence, and alignment metadata.
    """
    if text_encoder is None:
        warnings.warn("No sentence encoder available; skipping transcription alignment")
        return script_segments
    
    # Extract video duration if provided to cap end timestamps
    video_duration = None
    if video_path:
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if fps > 0 and frame_count > 0:
                    video_duration = frame_count / fps
                    print(f"[INFO] Video duration: {video_duration:.2f}s ({frame_count} frames @ {fps:.2f} FPS)")
            cap.release()
        except Exception as e:
            warnings.warn(f"Could not extract video duration: {e}")

    script_texts = [s["text"] for s in script_segments]
    trans_texts = [t[2] for t in trans_tuples]
    trans_starts = np.array([float(t[0]) for t in trans_tuples], dtype=float)
    trans_ends = np.array([float(t[1]) for t in trans_tuples], dtype=float)
    trans_durations = np.maximum(trans_ends - trans_starts, 1e-3)
    movie_start = float(trans_starts[0]) if len(trans_starts) else 0.0
    movie_end = float(trans_ends[-1]) if len(trans_ends) else 0.0
    movie_span = max(movie_end - movie_start, float(np.sum(trans_durations)), 1.0)
    expected_gap = float(np.median(np.diff(trans_starts))) if len(trans_starts) > 1 else float(np.median(trans_durations))

    print(f"[INFO] Encoding transcription texts with {type(text_encoder).__name__}...")
    trans_embs = encode_texts_with_sentence_transformer(
        trans_texts,
        text_encoder,
        batch_size=32,
        context_radius=context_radius,
        max_chars=600,
    )

    print(f"[INFO] Encoding script texts with {type(text_encoder).__name__}...")
    script_embs = encode_texts_with_sentence_transformer(
        script_texts,
        text_encoder,
        batch_size=32,
        context_radius=context_radius,
        max_chars=600,
    )

    if trans_embs is None or script_embs is None:
        warnings.warn("Sentence encoder failed; skipping transcription alignment")
        return script_segments

    scores_mat = np.asarray(script_embs, dtype=float) @ np.asarray(trans_embs, dtype=float).T
    T, W = scores_mat.shape
    candidate_js = select_top_k_candidates(scores_mat, top_k_candidates)
    # Precompute midpoints for transcript segments for local-window filtering
    trans_mids = 0.5 * (trans_starts + trans_ends)
    # local window (seconds) used to hard-filter candidates to nearby transcript region
    # Prefer a modest fixed window to avoid allowing very-far matches on long movies.
    local_window = max(5.0 * expected_gap, 30.0)

    # Null-aware DP path inference.
    dp_prev = {None: 0.0}
    backpointers: List[Dict[Optional[int], Optional[int]]] = []
    row_scores: List[Dict[Optional[int], float]] = []

    # Null-aware DP path inference.
    debug_set = set(debug_indices) if debug_indices else set()
    dp_prev = {None: 0.0}
    backpointers: List[Dict[Optional[int], Optional[int]]] = []
    row_scores: List[Dict[Optional[int], float]] = []

    for i in range(T):
        curr_scores: Dict[Optional[int], float] = {}
        curr_back: Dict[Optional[int], Optional[int]] = {}

        # Global time prior: map script progress to an expected point on the transcription timeline.
        if T > 1 and W > 1:
            target_time = movie_start + (movie_span * (i / float(T - 1)))
            # Expected transcription index based on linear progress
            expected_j = int((i / float(T - 1)) * float(W - 1)) if T > 1 else 0
        else:
            target_time = movie_start
            expected_j = 0

        # Enforce a temporal feasibility bound: candidates must be within ~1.2x the expected progress
        # This prevents far-ahead matches while allowing some flexibility for variable pacing
        # min_feasible: allow some backward jumps (e.g., 0.8x expected)
        # max_feasible: allow forward skips (e.g., 1.2x expected)
        min_feasible_j = max(0, int(expected_j * 0.8))
        max_feasible_j = min(W - 1, int(expected_j * 1.2 + 5))  # +5 to allow small gaps
        
        # Precompute a compact top view of the score row for debugging
        if i in debug_set:
            topk_idx = list(np.argsort(-scores_mat[i])[: min(10, scores_mat.shape[1])])
            topk_vals = [(int(jj), float(scores_mat[i, jj])) for jj in topk_idx]
            print(f"[DEBUG][row {i}] target_time={target_time:.3f}, expected_j={expected_j}, feasible_range=[{min_feasible_j}, {max_feasible_j}], top_scores={topk_vals}")

        # Filter candidate set to local window around target_time to avoid far-away jumps.
        orig_candidates = candidate_js[i]
        if len(orig_candidates) > 0:
            # First apply temporal feasibility bound
            feasible_candidates = [j for j in orig_candidates if min_feasible_j <= int(j) <= max_feasible_j]
            if len(feasible_candidates) == 0:
                feasible_candidates = [int(np.clip(expected_j, min_feasible_j, max_feasible_j))]
            
            # Then apply local window filter
            local_candidates = [j for j in feasible_candidates if abs(trans_mids[j] - target_time) <= local_window]
            if local_candidates:
                candidate_js[i] = np.array(local_candidates, dtype=int)
            else:
                # if none of the top-k are local, include nearest transcript index + top-2 original candidates
                # to avoid semantic false matches while preserving some semantic options
                fallback_list = []
                nearest_j = int(np.argmin(np.abs(trans_mids - target_time))) if len(trans_mids) > 0 else None
                if nearest_j is not None and min_feasible_j <= nearest_j <= max_feasible_j:
                    fallback_list.append(nearest_j)
                # add up to 2 top semantic candidates that are feasible to balance temporal locality with semantic relevance
                for jj in feasible_candidates[:2]:
                    if int(jj) not in fallback_list:
                        fallback_list.append(int(jj))
                candidate_js[i] = np.array(fallback_list, dtype=int) if fallback_list else np.array(feasible_candidates[:1], dtype=int)

        if i in debug_set:
            print(f"[DEBUG][row {i}] orig_candidates={list(orig_candidates)}, postfilter_candidates={list(candidate_js[i])}, local_window={local_window}")
            print(f"[DEBUG][row {i}] dp_prev_summary: best_prev_state={max(dp_prev.items(), key=lambda kv: kv[1])[0]}, num_prev_states={len(dp_prev)}")

        best_prev_state = max(dp_prev.items(), key=lambda kv: kv[1])[0]
        best_prev_score = dp_prev[best_prev_state]
        curr_scores[None] = best_prev_score + null_penalty
        curr_back[None] = best_prev_state

        for j in candidate_js[i]:
            best_score = -1e18
            best_state: Optional[int] = None
            for prev_state, prev_score in dp_prev.items():
                if prev_state is not None and j < prev_state:
                    continue

                penalty = 0.0
                if prev_state is not None:
                    gap = max(0, j - prev_state - 1)
                    penalty += jump_penalty * float(gap)
                    time_gap = float(trans_starts[j] - trans_starts[prev_state])
                    penalty += time_penalty * abs(time_gap - expected_gap)

                # Keep alignments near the expected absolute position on the transcript timeline.
                trans_mid = float(0.5 * (trans_starts[j] + trans_ends[j]))
                norm_deviation = abs(trans_mid - target_time) / movie_span
                penalty += global_time_penalty * norm_deviation

                candidate_score = prev_score + float(scores_mat[i, j]) - penalty
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_state = prev_state

            curr_scores[int(j)] = best_score
            curr_back[int(j)] = best_state

        backpointers.append(curr_back)
        row_scores.append(curr_scores)

        if i in debug_set:
            # print the candidate row scores we just computed
            items = sorted([(k, v) for k, v in curr_scores.items()], key=lambda kv: (-kv[1], kv[0] if kv[0] is not None else -1))
            print(f"[DEBUG][row {i}] curr_scores (top 12) = {items[:12]}")

        dp_prev = curr_scores

    final_state = max(dp_prev.items(), key=lambda kv: kv[1])[0]
    path: List[Optional[int]] = [None] * T
    state = final_state
    for i in range(T - 1, -1, -1):
        path[i] = state
        state = backpointers[i].get(state)

    # Debug: show path before clamping
    if debug_set:
        for i in sorted(debug_set):
            if 0 <= i < T:
                print(f"[DEBUG][path_before row {i}] {path[i]}")

    # Enforce monotonic non-decreasing path: clamp any decrease to the nearest allowed candidate >= previous
    last_assigned = None
    for i in range(T):
        if path[i] is None:
            continue
        if last_assigned is not None and path[i] < last_assigned:
            # try to pick a candidate at this row that is >= last_assigned
            candidates_here = list(candidate_js[i]) if i < len(candidate_js) else []
            ge_candidates = [int(c) for c in candidates_here if int(c) >= int(last_assigned)]
            if ge_candidates:
                new_choice = int(min(ge_candidates))
                if i in debug_set:
                    print(f"[DEBUG][path_fix row {i}] path was {path[i]}, clamped to candidate {new_choice} >= last_assigned {last_assigned}")
                path[i] = new_choice
            else:
                # no suitable candidate: clamp to last_assigned
                if i in debug_set:
                    print(f"[DEBUG][path_fix row {i}] path was {path[i]}, clamped to last_assigned {last_assigned} (no ge candidate)")
                path[i] = int(last_assigned)
        if path[i] is not None:
            last_assigned = int(path[i])

    # Debug: show path after clamping
    if debug_set:
        for i in sorted(debug_set):
            if 0 <= i < T:
                print(f"[DEBUG][path_after row {i}] {path[i]}")

    raw_confidence: List[float] = []
    matched = 0
    output = []
    for i, script_seg in enumerate(script_segments):
        state_scores = row_scores[i]
        best_state = path[i]
        best_score = state_scores.get(best_state, -1e18)
        alt_scores = [v for k, v in state_scores.items() if k != best_state and k is not None]
        second_score = max(alt_scores) if alt_scores else best_score
        margin = float(best_score - second_score)

        if best_state is None or float(scores_mat[i, best_state]) < score_threshold:
            output.append(
                {
                    "text": script_seg["text"],
                    "start": None,
                    "end": None,
                    "match_score": float(scores_mat[i, best_state]) if best_state is not None else None,
                    "score_margin": margin,
                    "confidence": 0.0,
                    "transcription_index": None,
                    "aligned_to_transcription": False,
                }
            )
            raw_confidence.append(0.0)
            continue

        start, end, _ = trans_tuples[int(best_state)]
        matched += 1
        confidence = 1.0 / (1.0 + math.exp(-6.0 * (float(scores_mat[i, best_state]) - null_penalty)))
        confidence = float(np.clip(confidence + min(0.15, max(0.0, margin)), 0.0, 1.0))
        output.append(
            {
                "text": script_seg["text"],
                "start": float(start),
                "end": float(end),
                "match_score": float(scores_mat[i, best_state]),
                "score_margin": margin,
                "confidence": confidence,
                "transcription_index": int(best_state),
                "aligned_to_transcription": True,
            }
        )
        raw_confidence.append(confidence)

    smoothed_confidence = smooth_confidence_series(raw_confidence)
    for i, conf in enumerate(smoothed_confidence):
        output[i]["confidence"] = float(conf)

    filled = fill_missing_timestamps(
        output, 
        default_duration=float(np.median(trans_durations)) if len(trans_durations) else 1.0,
        video_duration=video_duration
    )
    print(f"[INFO] Transcription alignment: {matched}/{T} matched (score >= {score_threshold:.2f})")
    return filled


def monotonic_align(script_embs, window_embs):
    """Greedy monotonic alignment: each script segment maps to a non-decreasing window index."""
    alignments = []
    last_j = 0
    n_windows = len(window_embs)

    for i, s_emb in enumerate(script_embs):
        best_j = None
        best_score = -1e9

        for j in range(last_j, n_windows):
            score = cosine(s_emb, window_embs[j])
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best_j = j

        if best_j is None:
            alignments.append({"script_index": i, "window_index": None, "score": None})
            continue

        last_j = best_j
        alignments.append(
            {
                "script_index": i,
                "window_index": best_j,
                "score": float(best_score),
            }
        )

    return alignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--script", default=None, help="Script path (.txt/.json/.csv/.srt/.pdf)")
    parser.add_argument("--srt", default=None, help="Deprecated alias of --script")
    parser.add_argument("--out", default="zt_bins.json")
    parser.add_argument("--sample_fps", type=float, default=1.0)
    parser.add_argument("--window_sec", type=float, default=4.0)
    parser.add_argument("--stride_sec", type=float, default=None, help="Sliding window stride in seconds; defaults to window_sec (no overlap)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--frame_rerank", action="store_true", help="Run frame-level reranking inside matched windows for finer timestamps")
    parser.add_argument("--top_k", type=int, default=1, help="Return top_k candidate windows per script segment")
    parser.add_argument("--score_threshold", type=float, default=None, help="Minimum cosine score to accept alignment; else window_index set to None")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--transcription", default=None, help="SRT transcription path for text-to-text timeline alignment")
    parser.add_argument("--text_encoder", default="sentence-transformers/all-mpnet-base-v2", help="Sentence embedding model for script/transcription matching")
    parser.add_argument("--context_radius", type=int, default=1, help="Neighbor radius for contextual text embeddings")
    parser.add_argument("--candidate_top_k", type=int, default=8, help="Top-k transcription candidates per script segment")
    parser.add_argument("--null_penalty", type=float, default=-0.12, help="Penalty for aligning a script segment to NULL")
    parser.add_argument("--jump_penalty", type=float, default=0.03, help="Penalty for large forward jumps between transcript matches")
    parser.add_argument("--time_penalty", type=float, default=0.02, help="Penalty for transcript time gaps that deviate from the local prior")
    parser.add_argument("--global_time_penalty", type=float, default=0.02, help="Penalty for deviating from the expected absolute transcript position")
    parser.add_argument("--debug_indices", type=str, default=None, help="Comma-separated script indices to print debug diagnostics for, e.g. '1,3,5'")
    parser.add_argument("--extract_audio_emb", action="store_true", help="Extract MFCC audio embeddings for output bins")
    parser.add_argument("--audio_sr", type=int, default=16000, help="Sample rate for audio embedding extraction")
    parser.add_argument("--audio_n_mfcc", type=int, default=40, help="MFCC dimension for audio embeddings")
    args = parser.parse_args()

    script_path = args.script or args.srt
    if not script_path:
        raise ValueError("Please provide --script (or --srt as alias)")

    if torch is not None and args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "auto":
        args.device = "cpu"

    script_segments = parse_script(script_path)
    if not script_segments:
        raise ValueError("No script segments found")

    print(f"[INFO] Parsed {len(script_segments)} script segments")

    processor = None
    model = None

    # Optional: use transcription to infer timestamps for script segments first.
    if args.transcription:
        trans_tuples = parse_srt(args.transcription)
        if not trans_tuples:
            raise ValueError(f"No transcription segments found in {args.transcription}")
        print(f"[INFO] Parsed {len(trans_tuples)} transcription segments")
        text_encoder = load_text_encoder(args.text_encoder, args.device)
        processor, model = load_clip(args.device)
        # parse debug indices if requested
        debug_idx_list = None
        if getattr(args, 'debug_indices', None):
            try:
                debug_idx_list = [int(x) for x in args.debug_indices.split(",") if x.strip()]
            except Exception:
                debug_idx_list = None

        script_segments = align_script_to_transcription_dp(
            script_segments,
            trans_tuples,
            text_encoder,
            device=args.device,
            context_radius=args.context_radius,
            top_k_candidates=args.candidate_top_k,
            null_penalty=args.null_penalty,
            jump_penalty=args.jump_penalty,
            time_penalty=args.time_penalty,
            global_time_penalty=args.global_time_penalty,
            score_threshold=args.score_threshold if args.score_threshold is not None else 0.30,
            debug_indices=debug_idx_list,
            video_path=args.video,
        )

    has_timestamps = all(s.get("start") is not None and s.get("end") is not None for s in script_segments)

    if processor is None or model is None:
        processor, model = load_clip(args.device)

    if has_timestamps:
        timed = []
        for seg in script_segments:
            start = seg.get("start")
            end = seg.get("end")
            if start is None or end is None:
                continue
            timed.append((float(start), float(end)))

        frames_by_seg = sample_frames_for_timed_segments(args.video, timed, sample_fps=args.sample_fps)
        windows = [
            {
                "start": s,
                "end": e,
                "frames": frames_by_seg[(s, e)],
            }
            for (s, e) in timed
        ]
        visual_embs = compute_visual_embeddings(windows, processor, model, device=args.device, batch_size=args.batch_size)
        text_embs = compute_script_text_embeddings(script_segments, processor, model, device=args.device, batch_size=max(8, args.batch_size))

        bins = []
        for i, seg in enumerate(script_segments):
            score = cosine(text_embs[i], visual_embs[i])
            seg_start = seg.get("start")
            seg_end = seg.get("end")
            bins.append(
                {
                    "start": float(seg_start) if seg_start is not None else 0.0,
                    "end": float(seg_end) if seg_end is not None else 0.0,
                    "text": seg["text"],
                    "visual_emb": visual_embs[i],
                    "audio_emb": None,
                    "text_emb": text_embs[i],
                    "script_index": i,
                    "alignment_score": score,
                }
            )
    else:
        windows = sample_video_windows(args.video, window_sec=args.window_sec, sample_fps=args.sample_fps, stride_sec=args.stride_sec)
        print(f"[INFO] Built {len(windows)} video windows")

        # compute visual features per window and per-frame if needed
        frame_embs_per_window, visual_embs = compute_frame_and_window_embeddings(windows, processor, model, device=args.device, batch_size=args.batch_size)
        text_embs = compute_script_text_embeddings(script_segments, processor, model, device=args.device, batch_size=max(8, args.batch_size))

        # build similarity matrix and apply mutual-NN filtering if requested
        S = compute_similarity_matrix(text_embs, visual_embs)
        if args.top_k and args.top_k > 0:
            mask = mutual_nn_mask(S, top_k_text=args.top_k, top_k_window=args.top_k)
            S_masked = np.where(mask, S, -np.inf)
        else:
            S_masked = S

        # DP monotonic alignment on masked similarity matrix
        path, scores = dp_monotonic(S_masked)

        bins = []
        matched_count = 0
        for i, win_idx in enumerate(path):
            fallback_used = False
            if win_idx is None or (win_idx is not None and not np.isfinite(S[i, win_idx])):
                row = S[i]
                if np.isfinite(row).any():
                    best_j = int(np.argmax(row))
                    if np.isfinite(row[best_j]):
                        win_idx = best_j
                        fallback_used = True

            if win_idx is None:
                bins.append({
                    "start": 0.0,
                    "end": 0.0,
                    "text": script_segments[i]["text"],
                    "visual_emb": None,
                    "audio_emb": None,
                    "text_emb": text_embs[i],
                    "script_index": i,
                    "alignment_score": None,
                })
                continue

            # apply threshold
            score = scores[i] if not fallback_used else float(S[i, win_idx])
            if args.score_threshold is not None and (score is None or score < args.score_threshold):
                bins.append({
                    "start": 0.0,
                    "end": 0.0,
                    "text": script_segments[i]["text"],
                    "visual_emb": None,
                    "audio_emb": None,
                    "text_emb": text_embs[i],
                    "script_index": i,
                    "alignment_score": score,
                })
                continue

            # refine inside window
            if args.frame_rerank and frame_embs_per_window[win_idx]:
                best_score = -1e9
                best_t = None
                best_vec = None
                for (t, vec) in frame_embs_per_window[win_idx]:
                    sc = cosine(text_embs[i], vec)
                    if sc is None:
                        continue
                    if sc > best_score:
                        best_score = sc
                        best_t = t
                        best_vec = vec
                if best_t is not None:
                    start = float(best_t)
                    end = float(best_t)
                    vemb = best_vec
                    score = best_score
                else:
                    start = float(windows[win_idx]["start"])
                    end = float(windows[win_idx]["end"])
                    vemb = visual_embs[win_idx]
            else:
                start = float(windows[win_idx]["start"])
                end = float(windows[win_idx]["end"])
                vemb = visual_embs[win_idx]

            matched_count += 1

            bins.append({
                "start": start,
                "end": end,
                "text": script_segments[i]["text"],
                "visual_emb": vemb,
                "audio_emb": None,
                "text_emb": text_embs[i],
                "script_index": i,
                "alignment_score": float(score) if score is not None else None,
            })

            print(f"[INFO] Matched {matched_count}/{len(script_segments)} script segments to a window")

    if args.extract_audio_emb:
        print("[INFO] Extracting audio embeddings for bins...")
        audio_embs = compute_audio_embeddings_for_bins(
            args.video,
            bins,
            sr=int(args.audio_sr),
            n_mfcc=int(args.audio_n_mfcc),
        )
        for i in range(len(bins)):
            bins[i]["audio_emb"] = audio_embs[i]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"video": args.video, "script": script_path, "bins": bins}, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Wrote {args.out} with {len(bins)} aligned bins")


if __name__ == "__main__":
    main()