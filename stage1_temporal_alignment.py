#!/usr/bin/env python3
"""
Stage 1 — Temporal Alignment

Inputs:
 - video_path: path to video file
 - subtitles_path: path to .srt subtitle file

Produces `zt_bins.json` containing a list of temporal bins anchored on subtitles
Each bin contains: start, end, text, visual_emb (optional), audio_emb (optional), text_emb (optional)

This script uses CLIP vision encoder (transformers) for visual features if available,
and librosa for audio features if available. SentenceTransformer is optional for text embeddings.
"""
import argparse
import json
import os
import re
import warnings
from typing import List, Tuple

import cv2
import numpy as np

try:
    import librosa
except Exception:
    librosa = None

try:
    from transformers import AutoImageProcessor, CLIPVisionModel
except Exception:
    AutoImageProcessor = None
    CLIPVisionModel = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def parse_srt(srt_path: str) -> List[Tuple[float, float, str]]:
    """Very small .srt parser returning list of (start_sec, end_sec, text)"""
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
        # text is lines after the time line
        lines = p.splitlines()
        # find time line index
        text_lines = []
        for i, line in enumerate(lines):
            if time_re.search(line):
                text_lines = lines[i + 1 :]
                break
        text = " ".join([ln.strip() for ln in text_lines if ln.strip()])
        items.append((start, end, text))

    return items


def sample_frames_for_segments(video_path: str, segments: List[Tuple[float, float, str]], sample_fps: float = 1.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = {}
    for (s, e, _) in segments:
        # sample at sample_fps inside [s,e)
        if sample_fps <= 0:
            continue
        step = 1.0 / sample_fps
        t = s
        seg_frames = []
        while t < e:
            frame_idx = int(round(t * video_fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            seg_frames.append((t, frame.copy()))
            t += step
        frames[(s, e)] = seg_frames
    cap.release()
    return frames


def compute_visual_embeddings(frames_by_segment, device="cpu"):
    if AutoImageProcessor is None or CLIPVisionModel is None:
        warnings.warn("transformers CLIP not available; visual embeddings will be empty")
        return {k: None for k in frames_by_segment}

    processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    out = {}
    for seg, frames in frames_by_segment.items():
        if not frames:
            out[seg] = None
            continue
        imgs = [cv2.cvtColor(f[1], cv2.COLOR_BGR2RGB) for f in frames]
        imgs = [cv2.resize(im, (224, 224)) for im in imgs]
        processed = processor(images=imgs, return_tensors="pt")
        # move to device
        for k, v in processed.items():
            processed[k] = v.to(device)
        with torch.no_grad():
            feats = model(**processed).last_hidden_state[:, 0, :].mean(dim=0).cpu().numpy()
        # normalize
        if feats is not None:
            feats = feats / (np.linalg.norm(feats) + 1e-12)
        out[seg] = feats.tolist() if feats is not None else None
    return out


def compute_audio_embeddings(video_path: str, segments, sr=16000):
    if librosa is None:
        warnings.warn("librosa not available; audio embeddings will be empty")
        return {k: None for k in segments}

    y, orig_sr = librosa.load(video_path, sr=sr, mono=True)
    out = {}
    for (s, e, _) in segments:
        ist = int(max(0, s * sr))
        ied = int(min(len(y), e * sr))
        if ied <= ist:
            out[(s, e)] = None
            continue
        seg = y[ist:ied]
        mel = librosa.feature.melspectrogram(seg, sr=sr, n_mels=64)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        vec = np.mean(log_mel, axis=1)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        out[(s, e)] = vec.tolist()
    return out


def compute_text_embeddings(segments, model_name="all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        warnings.warn("sentence-transformers not available; text embeddings will be empty")
        return {k: None for k in segments}
    model = SentenceTransformer(model_name)
    out = {}
    texts = [t for (_, _, t) in segments]
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    for i, seg in enumerate(segments):
        vec = emb[i]
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        out[(seg[0], seg[1])] = vec.tolist()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--srt", required=True)
    parser.add_argument("--out", default="zt_bins.json")
    parser.add_argument("--sample_fps", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    segments = parse_srt(args.srt)
    print(f"Parsed {len(segments)} subtitle segments")
    frames_by_seg = sample_frames_for_segments(args.video, segments, sample_fps=args.sample_fps)

    vis = compute_visual_embeddings(frames_by_seg, device=args.device)
    aud = compute_audio_embeddings(args.video, segments)
    txt = compute_text_embeddings(segments) if SentenceTransformer is not None else { (s,e): None for (s,e,_) in segments }

    bins = []
    for (s, e, text) in segments:
        key = (s, e)
        bins.append({
            "start": float(s),
            "end": float(e),
            "text": text,
            "visual_emb": vis.get(key),
            "audio_emb": aud.get(key),
            "text_emb": txt.get(key),
        })

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"video": args.video, "bins": bins}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out} with {len(bins)} bins")


if __name__ == "__main__":
    main()
