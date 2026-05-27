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
- Script alignment now runs through ASR timestamps, then refines locally with video embeddings.
"""

import argparse
import csv
import importlib
import json
import math
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import AutoProcessor, CLIPModel
except Exception:
    AutoProcessor = None
    CLIPModel = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import librosa
except Exception:
    librosa = None

def compute_local_coherence(video_embs):
    """
    Neighbor Consistency (NCC)
    measures temporal smoothness of video representation
    """

    if len(video_embs) < 2:
        return 0.0

    sims = []

    for i in range(len(video_embs) - 1):
        sim = cosine(video_embs[i], video_embs[i + 1])
        if sim is not None:
            sims.append(sim)

    if len(sims) == 0:
        return 0.0

    return float(np.mean(sims))

def compute_retrieval_metrics(script_embs, video_embs, alignment=None, topk=(1, 5, 10)):
    T = len(script_embs)
    W = len(video_embs)

    S = np.full((T, W), -np.inf)

    for i in range(T):
        for j in range(W):
            s = cosine(script_embs[i], video_embs[j])
            if s is not None:
                S[i, j] = s

    recall = {k: 0.0 for k in topk}
    mrr = 0.0
    valid = 0

    for i in range(T):
        row = S[i]
        if not np.isfinite(row).any():
            continue

        ranking = np.argsort(-row)

        if alignment is not None:
            gt_j = alignment[i]
        else:
            gt_j = ranking[0]

        if gt_j is None or not np.isfinite(row[gt_j]):
            continue

        valid += 1

        rank_pos = np.where(ranking == gt_j)[0][0] + 1
        mrr += 1.0 / rank_pos

        for k in topk:
            if gt_j in ranking[:k]:
                recall[k] += 1

    mrr /= max(valid, 1)

    for k in topk:
        recall[k] /= max(valid, 1)

    return {
        "Recall@1": recall[1],
        "Recall@5": recall[5],
        "Recall@10": recall[10],
        "MRR": mrr
    }

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


def _parse_script_srt(script_path: str) -> List[Dict[str, Any]]:
    return [{"text": text, "start": start, "end": end} for (start, end, text) in parse_srt(script_path) if text]


def _parse_script_json(script_path: str) -> List[Dict[str, Any]]:
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


def _parse_script_csv(script_path: str) -> List[Dict[str, Any]]:
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


def _parse_script_pdf(script_path: str) -> List[Dict[str, Any]]:
    try:
        pypdf = importlib.import_module("pypdf")
    except ImportError as exc:
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


def _parse_script_plain(script_path: str) -> List[Dict[str, Any]]:
    with open(script_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    blocks = [b.strip() for b in re.split(r"\n\s*\n", content) if b.strip()]
    return [{"text": b.replace("\n", " "), "start": None, "end": None} for b in blocks]


def parse_script(script_path: str) -> List[Dict[str, Any]]:
    """Parse script into [{'text': str, 'start': float|None, 'end': float|None}, ...]."""
    lower = script_path.lower()

    if lower.endswith(".srt"):
        return _parse_script_srt(script_path)

    if lower.endswith(".json"):
        return _parse_script_json(script_path)

    if lower.endswith(".csv"):
        return _parse_script_csv(script_path)

    if lower.endswith(".pdf"):
        return _parse_script_pdf(script_path)

    return _parse_script_plain(script_path)


def sample_frames_for_timed_segments(
    video_path: str,
    segments: List[Tuple[float, float]],
    sample_fps: float = 1.0,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
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
    consecutive_failures = 0

    while True:
        try:
            ret, frame = cap.read()
        except Exception as e:
            print(f"[WARN] Failed to read frame {frame_id}: {e}; skipping to next frame")
            ret = False

        if not ret:
            consecutive_failures += 1
            if frame_count > 0 and frame_id >= frame_count - 1:
                break
            if consecutive_failures > 100:
                print(f"[WARN] Too many consecutive decode failures near frame {frame_id}; stopping frame sampling")
                break
            frame_id += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            continue

        consecutive_failures = 0

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
        windows.append({"start": t, "end": min(t + window_sec, duration)})
        t += stride_sec
    print(f"[INFO] Created {len(windows)} sliding windows with stride={stride_sec:.2f}s, window={window_sec:.2f}s")

    if sample_fps <= 0 or not windows:
        return windows
    return windows


def compute_frame_and_window_embeddings(
    video_path: str,
    windows,
    processor,
    model,
    device="cpu",
    image_batch_size: int = 16,
    sample_fps: float = 1.0,
    store_frame_embeddings: bool = False,
):
    """Return (frame_embs_per_window, window_avg_embs).
    frame_embs_per_window: list of lists of (t, vec)
    window_avg_embs: list of averaged vectors (or None)
    """
    if processor is None or model is None or torch is None:
        warnings.warn("CLIP not available; visual embeddings are empty")
        return ([[] for _ in windows] if store_frame_embeddings else [[] for _ in windows]), [None for _ in windows]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0
    schedule: List[float] = []
    step = 1.0 / sample_fps if sample_fps > 0 else 0.0
    t = 0.0
    while sample_fps > 0 and t < duration:
        schedule.append(t)
        t += step
    print(f"[INFO] Sampling {len(schedule)} frames at {sample_fps:.2f} FPS for embeddings")

    frame_embs_per_window = [[] for _ in windows]
    window_sums = [None for _ in windows]
    window_counts = [0 for _ in windows]
    window_avg_embs = [None for _ in windows]

    if not schedule:
        cap.release()
        return frame_embs_per_window, window_avg_embs

    batch_imgs: List[Any] = []
    batch_times: List[float] = []

    def flush_batch() -> None:
        if not batch_imgs:
            return
        inputs = processor(images=batch_imgs, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        arr = feats.detach().cpu().numpy()
        for local_idx, vec in enumerate(arr):
            ts = float(batch_times[local_idx])
            for w_idx, w in enumerate(windows):
                if w["start"] - 1e-6 <= ts <= w["end"] + 1e-6:
                    if store_frame_embeddings:
                        frame_embs_per_window[w_idx].append((ts, vec.copy()))
                    if window_sums[w_idx] is None:
                        window_sums[w_idx] = np.array(vec, dtype=float)
                    else:
                        window_sums[w_idx] += np.array(vec, dtype=float)
                    window_counts[w_idx] += 1
        batch_imgs.clear()
        batch_times.clear()

    idx = 0
    target_t = schedule[idx]
    frame_id = 0
    consecutive_failures = 0

    while target_t is not None:
        try:
            ret, frame = cap.read()
        except Exception as e:
            print(f"[WARN] Failed to read frame {frame_id}: {e}; skipping to next frame")
            ret = False

        if not ret:
            consecutive_failures += 1
            if frame_count > 0 and frame_id >= frame_count - 1:
                break
            if consecutive_failures > 100:
                print(f"[WARN] Too many consecutive decode failures near frame {frame_id}; stopping embedding sampling")
                break
            frame_id += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            continue

        consecutive_failures = 0
        current_t = frame_id / fps
        while target_t is not None and current_t >= target_t:
            batch_imgs.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            batch_times.append(target_t)
            if len(batch_imgs) >= image_batch_size:
                flush_batch()
            idx += 1
            if idx >= len(schedule):
                target_t = None
            else:
                target_t = schedule[idx]
        frame_id += 1

    flush_batch()
    cap.release()

    for w_idx, count in enumerate(window_counts):
        if count <= 0 or window_sums[w_idx] is None:
            window_avg_embs[w_idx] = None
            continue
        avg = window_sums[w_idx] / float(count)
        avg = _l2_normalize(avg)
        window_avg_embs[w_idx] = avg.tolist()

    return frame_embs_per_window, window_avg_embs


def load_clip(device: str, model_name: Optional[str] = None):
    """Load CLIP processor and model. If model_name is provided, use it; else fallback to openai/clip-vit-base-patch32."""
    if AutoProcessor is None or CLIPModel is None or torch is None:
        return None, None
    # Prefer a video-aware CLIP id if provided; otherwise try the user-specified model_name.
    preferred = model_name or "MCG-NJU/video-clip"
    hf_id = preferred
    processor = None
    model = None
    # Try preferred VideoCLIP id first, then a stronger image CLIP (ViT-H), then fallback to openai/clip-vit-base-patch32
    tried_ids = []
    for candidate in [hf_id, "laion/CLIP-ViT-H-14", "openai/clip-vit-base-patch32"]:
        if candidate in tried_ids:
            continue
        tried_ids.append(candidate)
        try:
            processor = AutoProcessor.from_pretrained(candidate, use_fast=True)
            try:
                model = CLIPModel.from_pretrained(candidate, use_safetensors=True)
            except (OSError, ValueError, RuntimeError):
                model = CLIPModel.from_pretrained(candidate)
            hf_id = candidate
            break
        except Exception:
            processor = None
            model = None
            continue
    if processor is None or model is None:
        return None, None
    model = model.to(device)
    model.eval()
    return processor, model


def load_text_encoder(model_name: str, device: str):
    if SentenceTransformer is None:
        warnings.warn("sentence-transformers is not available; CLIP text encoder will be used for transcription alignment")
        return None
    try:
        encoder = SentenceTransformer(model_name, device=device)
        return encoder
    except (OSError, ValueError, RuntimeError) as exc:
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


def encode_texts_with_clip(
    texts: List[str],
    processor,
    model,
    device: str = "cpu",
    batch_size: int = 32,
    context_radius: int = 1,
    max_chars: int = 600,
) -> Optional[np.ndarray]:
    if processor is None or model is None or torch is None:
        return None
    contextual_texts = build_contextual_texts(texts, radius=context_radius, max_chars=max_chars)
    out = []
    for i in tqdm(range(0, len(contextual_texts), batch_size), desc="Text embeddings"):
        batch = contextual_texts[i : i + batch_size]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        out.append(feats.cpu().numpy())
    return np.asarray(np.concatenate(out, axis=0), dtype=float) if out else None


def encode_texts_for_transcription(
    texts: List[str],
    text_encoder,
    clip_processor,
    clip_model,
    device: str = "cpu",
    batch_size: int = 32,
    context_radius: int = 1,
    max_chars: int = 600,
) -> Optional[np.ndarray]:
    if text_encoder is not None:
        return encode_texts_with_sentence_transformer(
            texts,
            text_encoder,
            batch_size=batch_size,
            context_radius=context_radius,
            max_chars=max_chars,
        )
    return encode_texts_with_clip(
        texts,
        clip_processor,
        clip_model,
        device=device,
        batch_size=batch_size,
        context_radius=context_radius,
        max_chars=max_chars,
    )


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(arr)
    if nrm < 1e-12:
        return arr
    return arr / nrm


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

    out: List[Optional[List[float]]] = []
    for b in bins:
        start = float(b.get("start", 0.0) or 0.0)
        end = float(b.get("end", start) or start)
        if end < start:
            end = start

        duration = max(0.0, end - start)
        if duration <= 0:
            out.append(None)
            continue

        try:
            y, _ = librosa.load(video_path, sr=sr, mono=True, offset=start, duration=duration)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            warnings.warn(f"Could not load audio segment [{start:.2f}, {end:.2f}] from video: {exc}")
            out.append(None)
            continue

        if y.size < max(64, n_mfcc):
            out.append(None)
            continue

        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            vec = np.mean(mfcc, axis=1).astype(float)
            vec = _l2_normalize(vec)
            out.append(vec.tolist())
        except (ValueError, RuntimeError):
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

    text_valid = [i for i, vec in enumerate(text_embs) if vec is not None]
    window_valid = [j for j, vec in enumerate(window_embs) if vec is not None]
    if not text_valid or not window_valid:
        return S

    text_stack = np.asarray([np.asarray(text_embs[i], dtype=float) for i in text_valid], dtype=float)
    window_stack = np.asarray([np.asarray(window_embs[j], dtype=float) for j in window_valid], dtype=float)

    if text_stack.shape[1] != window_stack.shape[1]:
        raise ValueError("Text and video embeddings must share the same dimensionality")

    S[np.ix_(text_valid, window_valid)] = text_stack @ window_stack.T
    return S


def build_local_script_profile(S: np.ndarray, script_idx: int, window_idx: Optional[int], radius: int = 4):
    if window_idx is None or window_idx < 0 or window_idx >= S.shape[1]:
        return [], None, None, None

    T = S.shape[0]
    left = max(0, int(script_idx) - int(radius))
    right = min(T, int(script_idx) + int(radius) + 1)

    profile = []
    finite_scores = []
    for j in range(left, right):
        score = S[j, int(window_idx)]
        if not np.isfinite(score):
            continue
        score_f = float(score)
        profile.append({
            "script_index": int(j),
            "offset": int(j - script_idx),
            "score": score_f,
        })
        finite_scores.append((j, score_f))

    if not finite_scores:
        return profile, None, None, None

    finite_scores.sort(key=lambda item: item[1], reverse=True)
    peak_j, peak_score = finite_scores[0]
    second_score = finite_scores[1][1] if len(finite_scores) > 1 else peak_score
    margin = float(peak_score - second_score)
    return profile, int(peak_j), float(peak_score), margin


def refine_video_from_asr(
    script_segments: List[Dict[str, Any]],
    windows: List[Dict[str, Any]],
    frame_embs_per_window: List[List[Tuple[float, Any]]],
    visual_embs: List[Any],
    text_embs: List[Any],
    local_radius_sec: float = 8.0,
    local_time_penalty: float = 0.02,
    continuity_weight: float = 0.12,
    transition_penalty: float = 0.02,
    frame_rerank: bool = False,
):
    if not windows:
        return [], {
            "video_monotonicity_rate": 1.0,
            "video_jump_size_mean": 0.0,
            "video_mean_asr_offset_sec": None,
            "video_mean_alignment_score": None,
        }

    window_centers = [0.5 * (float(w["start"]) + float(w["end"])) for w in windows]
    windows_np = np.asarray(window_centers, dtype=float)

    candidate_lists: List[List[Dict[str, Any]]] = []
    for i, seg in enumerate(script_segments):
        asr_start = seg.get("start")
        asr_end = seg.get("end")
        asr_confidence = seg.get("confidence")
        asr_center = None if asr_start is None or asr_end is None else 0.5 * (float(asr_start) + float(asr_end))

        if asr_center is None:
            candidate_lists.append([])
            continue

        candidate_js = [j for j, center in enumerate(window_centers) if abs(center - asr_center) <= float(local_radius_sec)]
        if not candidate_js:
            nearest_j = int(np.argmin(np.abs(windows_np - asr_center)))
            candidate_js = sorted(set([max(0, nearest_j - 1), nearest_j, min(len(windows) - 1, nearest_j + 1)]))

        candidates = []
        for j in candidate_js:
            semantic_score = cosine(text_embs[i], visual_embs[j])
            if semantic_score is None:
                continue
            adjusted_score = float(semantic_score) - float(local_time_penalty) * abs(window_centers[j] - asr_center)
            candidates.append(
                {
                    "window_index": int(j),
                    "window_start": float(windows[j]["start"]),
                    "window_end": float(windows[j]["end"]),
                    "window_center": float(window_centers[j]),
                    "offset_sec": float(window_centers[j] - asr_center),
                    "semantic_score": float(semantic_score),
                    "adjusted_score": float(adjusted_score),
                    "asr_center": float(asr_center),
                    "asr_start": float(asr_start),
                    "asr_end": float(asr_end),
                    "asr_confidence": asr_confidence,
                }
            )
        candidate_lists.append(sorted(candidates, key=lambda item: item["adjusted_score"], reverse=True))

    # Monotonic DP over local candidate windows.
    dp_prev: Dict[int, float] = {}
    backpointers: List[Dict[int, Optional[int]]] = []
    row_scores: List[Dict[int, float]] = []

    for i, candidates in enumerate(candidate_lists):
        curr_scores: Dict[int, float] = {}
        curr_back: Dict[int, Optional[int]] = {}
        if not candidates:
            backpointers.append(curr_back)
            row_scores.append(curr_scores)
            dp_prev = curr_scores
            continue

        prev_candidates = candidate_lists[i - 1] if i > 0 else []
        prev_asr_center = prev_candidates[0]["asr_center"] if prev_candidates else None

        for cand in candidates:
            j = int(cand["window_index"])
            emission = float(cand["adjusted_score"])
            if i == 0 or not dp_prev:
                curr_scores[j] = emission
                curr_back[j] = None
                continue

            best_score = -1e18
            best_prev = None
            for prev_j, prev_score in dp_prev.items():
                if prev_j > j:
                    continue
                prev_cand = next((p for p in prev_candidates if int(p["window_index"]) == int(prev_j)), None)
                # Continuity: penalize abrupt changes in text-video matching score (smoothness),
                # rather than raw visual similarity which often reflects scene continuity.
                continuity_penalty = 0.0
                if prev_cand is not None:
                    prev_emission = float(prev_cand.get("adjusted_score", 0.0))
                    continuity_penalty = float(continuity_weight) * abs(emission - prev_emission)

                expected_step = 0.0
                if prev_cand is not None:
                    expected_step = float(cand["asr_center"]) - float(prev_cand["asr_center"])
                actual_step = float(cand["window_center"]) - float(prev_cand["window_center"]) if prev_cand is not None else 0.0
                transition_cost = float(transition_penalty) * abs(actual_step - expected_step)
                score = float(prev_score) + emission - continuity_penalty - transition_cost
                if score > best_score:
                    best_score = score
                    best_prev = int(prev_j)

            if best_prev is None:
                best_prev = max(dp_prev.items(), key=lambda kv: kv[1])[0]
                best_score = float(dp_prev[best_prev]) + emission

            curr_scores[j] = best_score
            curr_back[j] = best_prev

        backpointers.append(curr_back)
        row_scores.append(curr_scores)
        dp_prev = curr_scores

    # Backtrack best monotone path.
    chosen_windows: List[Optional[int]] = [None] * len(candidate_lists)
    if dp_prev:
        state = max(dp_prev.items(), key=lambda kv: kv[1])[0]
        for i in range(len(candidate_lists) - 1, -1, -1):
            chosen_windows[i] = int(state) if state is not None else None
            state = backpointers[i].get(state) if state is not None else None

    bins: List[Dict[str, Any]] = []
    selected_window_indices: List[Optional[int]] = []
    selected_video_centers: List[Optional[float]] = []
    selected_scores: List[Optional[float]] = []
    asr_offsets: List[float] = []
    continuity_scores: List[Optional[float]] = []

    for i, seg in enumerate(script_segments):
        candidates = candidate_lists[i]
        chosen_j = chosen_windows[i]
        if not candidates:
            bins.append(
                {
                    "start": 0.0,
                    "end": 0.0,
                    "asr_start": seg.get("start"),
                    "asr_end": seg.get("end"),
                    "asr_confidence": seg.get("confidence"),
                    "text": seg["text"],
                    "visual_emb": None,
                    "audio_emb": None,
                    "text_emb": text_embs[i],
                    "script_index": i,
                    "asr_index": seg.get("transcription_index"),
                    "alignment_score": None,
                    "refinement_confidence": 0.0,
                    "video_window_index": None,
                    "video_candidate_profile": [],
                    "video_peak_window_index": None,
                    "video_peak_score": None,
                    "video_peak_margin": None,
                    "video_local_consistency": None,
                    "video_asr_offset_sec": None,
                    "refinement_source": "asr-only",
                }
            )
            selected_window_indices.append(None)
            selected_video_centers.append(None)
            selected_scores.append(None)
            asr_offsets.append(0.0)
            continuity_scores.append(None)
            continue

        selected = next((c for c in candidates if int(c["window_index"]) == int(chosen_j)), candidates[0])
        profile = candidates[: min(len(candidates), 8)]
        peak = profile[0] if profile else selected
        peak_margin = None
        if len(profile) > 1:
            peak_margin = float(profile[0]["adjusted_score"] - profile[1]["adjusted_score"])
        elif profile:
            peak_margin = 0.0

        j = int(selected["window_index"])
        start = float(windows[j]["start"])
        end = float(windows[j]["end"])
        vemb = visual_embs[j]
        local_consistency = None
        if i > 0 and selected_window_indices[-1] is not None:
            prev_j = int(selected_window_indices[-1])
            local_consistency = cosine(visual_embs[prev_j], visual_embs[j])

        if frame_rerank and frame_embs_per_window[j]:
            # Instead of picking a single max-response frame, do gaussian-weighted average
            # over frames within +/-1.0s to reduce susceptibility to noisy frames.
            times = np.array([t for (t, _) in frame_embs_per_window[j]], dtype=float)
            vecs = [vec for (_, vec) in frame_embs_per_window[j]]
            scores = np.array([float(cosine(text_embs[i], v) or 0.0) for v in vecs], dtype=float)
            if len(times) > 0:
                center_idx = int(np.argmax(scores))
                center_t = float(times[center_idx])
                window_mask = np.abs(times - center_t) <= 1.0
                selected_times = times[window_mask]
                selected_vecs = [vec for k, vec in enumerate(vecs) if window_mask[k]]
                if len(selected_vecs) > 0:
                    sigma = 0.6
                    weights = np.exp(-0.5 * ((selected_times - center_t) / sigma) ** 2)
                    weights = weights.astype(float)
                    # weighted sum of vectors
                    agg = None
                    for w, v in zip(weights, selected_vecs):
                        if agg is None:
                            agg = w * np.asarray(v, dtype=float)
                        else:
                            agg += w * np.asarray(v, dtype=float)
                    agg = agg / (np.sum(weights) + 1e-12)
                    agg = _l2_normalize(agg)
                    vemb = agg.tolist()
                    # set start/end to weighted mean time
                    start = float(np.sum(weights * selected_times) / (np.sum(weights) + 1e-12))
                    end = float(start)

        asr_center = float(0.5 * (float(seg["start"]) + float(seg["end"]))) if seg.get("start") is not None and seg.get("end") is not None else None
        video_center = float(0.5 * (start + end))
        offset_sec = float(video_center - asr_center) if asr_center is not None else None
        refinement_score = float(selected["adjusted_score"])
        selected_window_indices.append(j)
        selected_video_centers.append(video_center)
        selected_scores.append(refinement_score)
        asr_offsets.append(float(offset_sec) if offset_sec is not None else 0.0)
        continuity_scores.append(float(local_consistency) if local_consistency is not None else None)

        bins.append(
            {
                "start": start,
                "end": end,
                "asr_start": float(seg["start"]) if seg.get("start") is not None else None,
                "asr_end": float(seg["end"]) if seg.get("end") is not None else None,
                "asr_confidence": seg.get("confidence"),
                "text": seg["text"],
                "visual_emb": vemb,
                "audio_emb": None,
                "text_emb": text_embs[i],
                "script_index": i,
                "asr_index": seg.get("transcription_index"),
                "alignment_score": float(selected["semantic_score"]),
                "refinement_confidence": float(max(0.0, min(1.0, 0.5 * (refinement_score + 1.0)))),
                "video_window_index": j,
                "video_candidate_profile": profile,
                "video_peak_window_index": int(peak["window_index"]),
                "video_peak_score": float(peak["adjusted_score"]),
                "video_peak_margin": float(peak_margin) if peak_margin is not None else None,
                "video_local_consistency": float(local_consistency) if local_consistency is not None else None,
                "video_asr_offset_sec": float(offset_sec) if offset_sec is not None else None,
                "refinement_source": "asr-backbone+monotonic-video-dp",
            }
        )

    valid_indices = [j for j in selected_window_indices if j is not None]
    quality = {
        "video_monotonicity_rate": monotonicity_rate(valid_indices) if valid_indices else 1.0,
        "video_jump_size_mean": jump_size_mean(valid_indices) if valid_indices else 0.0,
        "video_mean_asr_offset_sec": float(np.mean([x for x in asr_offsets if x is not None])) if asr_offsets else None,
        "video_mean_alignment_score": float(np.mean([s for s in selected_scores if s is not None])) if selected_scores else None,
        "video_mean_local_consistency": float(np.mean([c for c in continuity_scores if c is not None])) if any(c is not None for c in continuity_scores) else None,
    }

    return bins, quality


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
    max_allowed = video_duration if video_duration is not None else float("inf")

    if not anchor_indices:
        t = 0.0
        for seg in filled:
            seg["start"] = float(min(t, max_allowed))
            seg["end"] = float(min(t + default_duration, max_allowed))
            seg["aligned_to_transcription"] = False
            t = seg["end"]
        return filled

    # Fill gaps by interpolating between anchor endpoints.
    first_anchor = anchor_indices[0]
    first_start = float(filled[first_anchor]["start"])
    prev_end = max(0.0, first_start)

    for i in range(first_anchor - 1, -1, -1):
        seg = filled[i]
        seg["end"] = float(prev_end)
        seg["start"] = float(max(0.0, prev_end - default_duration))
        prev_end = seg["start"]
        seg["aligned_to_transcription"] = False

    i = 0
    while i < n:
        if filled[i].get("start") is not None and filled[i].get("end") is not None:
            i += 1
            continue
        gap_start = i
        while i < n and (filled[i].get("start") is None or filled[i].get("end") is None):
            i += 1
        gap_end = i

        left_idx = gap_start - 1
        right_idx = gap_end if gap_end < n else None
        left_end = float(filled[left_idx]["end"]) if left_idx >= 0 and filled[left_idx].get("end") is not None else prev_end
        right_start = float(filled[right_idx]["start"]) if right_idx is not None and filled[right_idx].get("start") is not None else None

        gap_len = gap_end - gap_start
        if right_start is None:
            cursor = left_end
            for idx in range(gap_start, gap_end):
                filled[idx]["start"] = float(cursor)
                filled[idx]["end"] = float(min(cursor + default_duration, max_allowed))
                filled[idx]["aligned_to_transcription"] = False
                cursor = filled[idx]["end"]
            continue

        span = max(right_start - left_end, 0.0)
        step = span / float(gap_len + 1)
        cursor = left_end
        for offset, idx in enumerate(range(gap_start, gap_end), start=1):
            start = left_end + step * offset
            end = left_end + step * (offset + 1)
            if idx == gap_end - 1:
                end = right_start
            if start < cursor:
                start = cursor
            if end < start:
                end = start + default_duration
            filled[idx]["start"] = float(min(start, max_allowed))
            filled[idx]["end"] = float(min(end, max_allowed))
            filled[idx]["aligned_to_transcription"] = False
            cursor = filled[idx]["end"]

    # Final monotonic clamp and boundary reconciliation.
    last_end = 0.0
    for seg in filled:
        start = seg.get("start")
        end = seg.get("end")
        if start is None or end is None:
            start = last_end
            end = min(last_end + default_duration, max_allowed)
        start = max(float(start), float(last_end))
        end = max(float(end), start)
        if video_duration is not None:
            start = min(start, max_allowed)
            end = min(end, max_allowed)
        seg["start"] = float(start)
        seg["end"] = float(end)
        last_end = float(end)

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
    clip_processor=None,
    clip_model=None,
    alignment_mode: str = "nw",
    device="cpu",
    context_radius: int = 1,
    top_k_candidates: int = 8,
    null_penalty: float = -0.12,
    jump_penalty: float = 0.03,
    time_penalty: float = 0.02,
    global_time_penalty: float = 0.02,
    score_threshold: float = 0.30,
    local_window_sec: Optional[float] = None,
    local_window_ratio: float = 0.05,
    debug_indices: Optional[List[int]] = None,
    video_path: Optional[str] = None,
):
    """
    Align script to transcription with a semantic encoder + contextual smoothing.

    Uses a soft candidate lattice (top-k per script segment) and null-aware DP.
    Returns script segments with timestamps, confidence, and alignment metadata.
    """
    if text_encoder is None and (clip_processor is None or clip_model is None):
        warnings.warn("No transcription text encoder available; skipping transcription alignment")
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

    encoder_name = type(text_encoder).__name__ if text_encoder is not None else "CLIP"
    print(f"[INFO] Encoding transcription texts with {encoder_name}...")
    trans_embs = encode_texts_for_transcription(
        trans_texts,
        text_encoder,
        clip_processor,
        clip_model,
        device=device,
        batch_size=32,
        context_radius=context_radius,
        max_chars=600,
    )

    print(f"[INFO] Encoding script texts with {encoder_name}...")
    script_embs = encode_texts_for_transcription(
        script_texts,
        text_encoder,
        clip_processor,
        clip_model,
        device=device,
        batch_size=32,
        context_radius=context_radius,
        max_chars=600,
    )

    if trans_embs is None or script_embs is None:
        warnings.warn("Transcription text encoding failed; skipping transcription alignment")
        return script_segments

    scores_mat = np.asarray(script_embs, dtype=float) @ np.asarray(trans_embs, dtype=float).T
    T, W = scores_mat.shape

    # Adaptive score normalization: z-score across all pairwise scores
    score_mean = float(np.mean(scores_mat)) if scores_mat.size else 0.0
    score_std = float(np.std(scores_mat)) if scores_mat.size else 1.0
    if score_std < 1e-6:
        score_std = 1.0
    scores_z = (scores_mat - score_mean) / score_std

    # Scale penalty hyperparameters to the score std to make them robust across movies
    null_penalty = float(null_penalty) / score_std
    jump_penalty = float(jump_penalty) / score_std
    time_penalty = float(time_penalty) / score_std
    global_time_penalty = float(global_time_penalty) / score_std

    # Use z-scored similarity for downstream decisions
    scores_used = scores_z
    candidate_js = select_top_k_candidates(scores_used, top_k_candidates)
    # Precompute midpoints for transcript segments for local-window filtering
    trans_mids = 0.5 * (trans_starts + trans_ends)
    local_window = float(local_window_sec) if local_window_sec is not None else max(local_window_ratio * movie_span, 30.0)

    dp_prev = {None: 0.0}
    backpointers: List[Dict[Optional[int], Optional[int]]] = []
    row_scores: List[Dict[Optional[int], float]] = []

    debug_set = set(debug_indices) if debug_indices else set()
    dp_prev = {None: 0.0}
    backpointers: List[Dict[Optional[int], Optional[int]]] = []
    row_scores: List[Dict[Optional[int], float]] = []

    for i in range(T):
        curr_scores: Dict[Optional[int], float] = {}
        curr_back: Dict[Optional[int], Optional[int]] = {}

        if T > 1 and W > 1:
            target_time = movie_start + (movie_span * (i / float(T - 1)))
            expected_j = int((i / float(T - 1)) * float(W - 1)) if T > 1 else 0
        else:
            target_time = movie_start
            expected_j = 0

        min_feasible_j = max(0, int(expected_j * 0.8))
        max_feasible_j = min(W - 1, int(expected_j * 1.2 + 5))
        
        if i in debug_set:
            topk_idx = list(np.argsort(-scores_mat[i])[: min(10, scores_mat.shape[1])])
            topk_vals = [(int(jj), float(scores_mat[i, jj])) for jj in topk_idx]
            print(f"[DEBUG][row {i}] target_time={target_time:.3f}, expected_j={expected_j}, feasible_range=[{min_feasible_j}, {max_feasible_j}], top_scores={topk_vals}")

        orig_candidates = candidate_js[i]
        if len(orig_candidates) > 0:
            feasible_candidates = [j for j in orig_candidates if min_feasible_j <= int(j) <= max_feasible_j]
            if len(feasible_candidates) == 0:
                feasible_candidates = [int(np.clip(expected_j, min_feasible_j, max_feasible_j))]
            
            local_candidates = [j for j in feasible_candidates if abs(trans_mids[j] - target_time) <= local_window]
            if local_candidates:
                candidate_js[i] = np.array(local_candidates, dtype=int)
            else:
                fallback_list = []
                nearest_j = int(np.argmin(np.abs(trans_mids - target_time))) if len(trans_mids) > 0 else None
                if nearest_j is not None and min_feasible_j <= nearest_j <= max_feasible_j:
                    fallback_list.append(nearest_j)
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
            items = sorted([(k, v) for k, v in curr_scores.items()], key=lambda kv: (-kv[1], kv[0] if kv[0] is not None else -1))
            print(f"[DEBUG][row {i}] curr_scores (top 12) = {items[:12]}")

        dp_prev = curr_scores

    # If using Needleman-Wunsch alignment, compute global sequence alignment
    mapping_list: List[List[int]] = [[] for _ in range(T)]
    if alignment_mode == "nw":
        # Needleman-Wunsch global alignment with gap penalty = null_penalty
        gap = float(null_penalty)
        F = np.full((T + 1, W + 1), -1e18, dtype=float)
        F[0, 0] = 0.0
        for i in range(1, T + 1):
            F[i, 0] = F[i - 1, 0] + gap
        for j in range(1, W + 1):
            F[0, j] = F[0, j - 1] + gap
        for i in range(1, T + 1):
            for j in range(1, W + 1):
                match = F[i - 1, j - 1] + float(scores_used[i - 1, j - 1])
                delete = F[i - 1, j] + gap  # gap in transcription
                insert = F[i, j - 1] + gap  # gap in script
                F[i, j] = max(match, delete, insert)
        # Traceback
        i, j = T, W
        align_pairs = []
        while i > 0 or j > 0:
            if i > 0 and j > 0 and abs(F[i, j] - (F[i - 1, j - 1] + float(scores_used[i - 1, j - 1]))) < 1e-6:
                align_pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif i > 0 and abs(F[i, j] - (F[i - 1, j] + gap)) < 1e-6:
                # aligned script i-1 to gap
                i -= 1
            else:
                # aligned transcription j-1 to gap
                j -= 1
        align_pairs.reverse()
        # Build mapping: allow multiple trans indices per script index
        for si, tj in align_pairs:
            mapping_list[si].append(tj)
    else:
        final_state = max(dp_prev.items(), key=lambda kv: kv[1])[0]
        path: List[Optional[int]] = [None] * T
        state = final_state
        for i in range(T - 1, -1, -1):
            path[i] = state
            state = backpointers[i].get(state)

        # Enforce non-decreasing and clamp as before
        last_assigned = None
        for i in range(T):
            if path[i] is None:
                continue
            if last_assigned is not None and path[i] < last_assigned:
                candidates_here = list(candidate_js[i]) if i < len(candidate_js) else []
                ge_candidates = [int(c) for c in candidates_here if int(c) >= int(last_assigned)]
                if ge_candidates:
                    new_choice = int(min(ge_candidates))
                    if i in debug_set:
                        print(f"[DEBUG][path_fix row {i}] path was {path[i]}, clamped to candidate {new_choice} >= last_assigned {last_assigned}")
                    path[i] = new_choice
                else:
                    if i in debug_set:
                        print(f"[DEBUG][path_fix row {i}] path was {path[i]}, clamped to last_assigned {last_assigned} (no ge candidate)")
                    path[i] = int(last_assigned)
            if path[i] is not None:
                last_assigned = int(path[i])

        # Convert path to mapping_list (one-to-one)
        for i in range(T):
            if path[i] is not None:
                mapping_list[i].append(int(path[i]))

    # path-based debug/clamping handled inside DP branch; mapping_list now used downstream

    raw_confidence: List[float] = []
    matched = 0
    output = []
    # Convert provided threshold (in original score units) to z-scored threshold
    score_threshold = float(score_threshold)
    score_threshold_z = (score_threshold - score_mean) / score_std
    for i, script_seg in enumerate(script_segments):
        mapped_js = mapping_list[i]
        if not mapped_js:
            output.append(
                {
                    "text": script_seg["text"],
                    "start": None,
                    "end": None,
                    "match_score": None,
                    "score_margin": None,
                    "confidence": 0.0,
                    "transcription_index": None,
                    "aligned_to_transcription": False,
                }
            )
            raw_confidence.append(0.0)
            continue

        # Choose best mapped transcription index by the used (z-scored) similarity
        best_j = int(max(mapped_js, key=lambda jj: float(scores_used[i, jj]) if 0 <= jj < W else -1e18))
        best_score_z = float(scores_used[i, best_j])
        # second best among other candidate trans indices
        other_js = [j for j in range(W) if j not in mapped_js]
        second_score_z = max([float(scores_used[i, j]) for j in other_js]) if other_js else best_score_z
        margin = float(best_score_z - second_score_z)

        if best_score_z < score_threshold_z:
            output.append(
                {
                    "text": script_seg["text"],
                    "start": None,
                    "end": None,
                    "match_score": float(scores_mat[i, best_j]) if 0 <= best_j < W else None,
                    "score_margin": margin,
                    "confidence": 0.0,
                    "transcription_index": None,
                    "aligned_to_transcription": False,
                }
            )
            raw_confidence.append(0.0)
            continue

        # Aggregate mapped transcription range to form span
        mapped_js_sorted = sorted(set(mapped_js))
        start = trans_tuples[mapped_js_sorted[0]][0]
        end = trans_tuples[mapped_js_sorted[-1]][1]
        matched += 1
        # use original (unnormalized) score for reporting
        match_score_orig = float(scores_mat[i, best_j]) if 0 <= best_j < W else None
        # confidence: sigmoid over z-score plus margin
        confidence = 1.0 / (1.0 + math.exp(-6.0 * (best_score_z - 0.0)))
        confidence = float(np.clip(confidence + min(0.15, max(0.0, margin)), 0.0, 1.0))
        output.append(
            {
                "text": script_seg["text"],
                "start": float(start),
                "end": float(end),
                "match_score": match_score_orig,
                "score_margin": margin,
                "confidence": confidence,
                "transcription_index": int(best_j),
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

def alignment_mae(pred_timestamps, gt_timestamps):
    """
    Mean Absolute Error between predicted and GT timestamps.
    
    pred_timestamps: List[float] (DTW aligned time or index->time mapping)
    gt_timestamps: List[float]
    """
    pred = np.array(pred_timestamps, dtype=float)
    gt = np.array(gt_timestamps, dtype=float)

    n = min(len(pred), len(gt))
    if n == 0:
        return None

    return float(np.mean(np.abs(pred[:n] - gt[:n])))

def asr_deviation(dtw_path, asr_path):
    """
    Mean absolute deviation between DTW alignment and ASR alignment.

    dtw_path: List[int] (script -> ASR index)
    asr_path: List[int] (ground truth ASR alignment per script)
    """
    dtw = np.array(dtw_path, dtype=float)
    gt = np.array(asr_path, dtype=float)

    n = min(len(dtw), len(gt))
    if n == 0:
        return None

    return float(np.mean(np.abs(dtw[:n] - gt[:n])))

def monotonicity_rate(path):
    """
    Fraction of monotonic transitions: j_{i+1} >= j_i

    path: List[int]
    """
    path = np.array(path, dtype=float)
    if len(path) < 2:
        return 1.0

    violations = np.sum(path[1:] < path[:-1])
    return float(1.0 - violations / (len(path) - 1))

def jump_size_mean(path):
    """
    Mean absolute step size between consecutive alignment indices.

    path: List[int]
    """
    path = np.array(path, dtype=float)
    if len(path) < 2:
        return 0.0

    jumps = np.abs(path[1:] - path[:-1])
    return float(np.mean(jumps))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--script", required=True, help="Script path (.txt/.json/.csv/.srt/.pdf)")
    parser.add_argument("--out", default="zt_bins.json")
    parser.add_argument("--sample_fps", type=float, default=2.0)
    parser.add_argument("--window_sec", type=float, default=2.0)
    parser.add_argument("--stride_sec", type=float, default=None, help="Sliding window stride in seconds; defaults to window_sec (no overlap)")
    parser.add_argument("--image_batch_size", type=int, default=16)
    parser.add_argument("--text_batch_size", type=int, default=32)
    parser.add_argument("--frame_rerank", action="store_true", help="Run frame-level reranking inside matched windows for finer timestamps")
    parser.add_argument("--score_threshold", type=float, default=None, help="Minimum cosine score to accept alignment; else window_index set to None")
    parser.add_argument("--neighbor_script_radius", type=int, default=4, help="How many neighboring script segments to include when exporting local score profiles")
    parser.add_argument("--video_local_radius_sec", type=float, default=8.0, help="Local ASR-centered radius in seconds for video refinement")
    parser.add_argument("--video_time_penalty", type=float, default=0.02, help="Penalty for video windows that are far from the ASR time anchor")
    parser.add_argument("--video_continuity_weight", type=float, default=0.12, help="Weight for local video embedding continuity")
    parser.add_argument("--video_transition_penalty", type=float, default=0.02, help="Penalty for jump mismatch between consecutive video windows")
    parser.add_argument("--asr_local_window_sec", type=float, default=None, help="Optional hard cap for ASR DP local search window in seconds")
    parser.add_argument("--asr_local_window_ratio", type=float, default=0.05, help="Fallback ASR DP local window ratio of movie duration")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--transcription", default=None, help="SRT transcription path for text-to-text timeline alignment")
    parser.add_argument("--text_encoder", default="sentence-transformers/all-mpnet-base-v2", help="Sentence embedding model for script/transcription matching")
    parser.add_argument("--clip_model", default="MCG-NJU/video-clip", help="Optional HuggingFace CLIP model id to use for visual+text encoding (default tries a VideoCLIP id, falls back to stronger CLIP then openai/clip)")
    parser.add_argument("--alignment_mode", default="nw", choices=["dp","nw"], help="Alignment algorithm: 'dp' (monotonic DP) or 'nw' (Needleman-Wunsch allowing many-to-many). Default: nw (one-to-many)")
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

    script_path = args.script

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
    use_asr_backbone = False

    if args.transcription:
        trans_tuples = parse_srt(args.transcription)
        if not trans_tuples:
            raise ValueError(f"No transcription segments found in {args.transcription}")
        print(f"[INFO] Parsed {len(trans_tuples)} transcription segments")
        text_encoder = load_text_encoder(args.text_encoder, args.device)
        processor, model = load_clip(args.device, model_name=args.clip_model)
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
            clip_processor=processor,
            clip_model=model,
            alignment_mode=args.alignment_mode,
            device=args.device,
            context_radius=args.context_radius,
            top_k_candidates=args.candidate_top_k,
            null_penalty=args.null_penalty,
            jump_penalty=args.jump_penalty,
            time_penalty=args.time_penalty,
            global_time_penalty=args.global_time_penalty,
            score_threshold=args.score_threshold if args.score_threshold is not None else 0.30,
            local_window_sec=args.asr_local_window_sec,
            local_window_ratio=args.asr_local_window_ratio,
            debug_indices=debug_idx_list,
            video_path=args.video,
        )
        use_asr_backbone = True

    if not use_asr_backbone:
        raise ValueError("This pipeline now requires --transcription; global video DTW has been removed.")

    if processor is None or model is None:
        processor, model = load_clip(args.device)

    windows = sample_video_windows(args.video, window_sec=args.window_sec, sample_fps=args.sample_fps, stride_sec=args.stride_sec)
    print(f"[INFO] Built {len(windows)} video windows")

    frame_embs_per_window, visual_embs = compute_frame_and_window_embeddings(
        args.video,
        windows,
        processor,
        model,
        device=args.device,
        image_batch_size=args.image_batch_size,
        sample_fps=args.sample_fps,
        store_frame_embeddings=args.frame_rerank,
    )
    text_embs = compute_script_text_embeddings(script_segments, processor, model, device=args.device, batch_size=args.text_batch_size)

    bins, quality = refine_video_from_asr(
        script_segments,
        windows,
        frame_embs_per_window,
        visual_embs,
        text_embs,
        local_radius_sec=args.video_local_radius_sec,
        local_time_penalty=args.video_time_penalty,
        continuity_weight=args.video_continuity_weight,
        transition_penalty=args.video_transition_penalty,
        frame_rerank=args.frame_rerank,
    )

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

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"video": args.video, "script": script_path, "bins": bins, "quality": quality}, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Wrote {args.out} with {len(bins)} aligned bins")

    print("[EVAL] Quality:", quality)
    print("[EVAL] NCC:", compute_local_coherence(visual_embs))


if __name__ == "__main__":
    main()