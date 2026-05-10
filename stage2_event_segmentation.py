#!/usr/bin/env python3
"""
Stage 2 — Event Segmentation

Inputs:
 - zt_bins.json produced by stage1_temporal_alignment.py

Produces `events.json` with event boundaries and aggregated multimodal features.

Uses ruptures if available for change point detection; otherwise falls back to a simple threshold on fused distances.
"""
import argparse
import json
import math
import os
import statistics
import warnings
from typing import List

import numpy as np

try:
    import ruptures as rpt
except Exception:
    rpt = None


def load_bins(path: str):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j.get("bins", [])


def fused_vector(b):
    parts = []
    if b.get("visual_emb") is not None:
        parts.append(np.array(b["visual_emb"], dtype=float))
    if b.get("audio_emb") is not None:
        parts.append(np.array(b["audio_emb"], dtype=float))
    if b.get("text_emb") is not None:
        parts.append(np.array(b["text_emb"], dtype=float))
    if not parts:
        return None
    # pad to same dimension by trunc/pad (simple approach)
    maxd = max(p.shape[0] for p in parts)
    vecs = []
    for p in parts:
        if p.shape[0] < maxd:
            p = np.pad(p, (0, maxd - p.shape[0]))
        elif p.shape[0] > maxd:
            p = p[:maxd]
        vecs.append(p)
    cat = np.concatenate(vecs)
    nrm = np.linalg.norm(cat)
    if nrm < 1e-12:
        return None
    return cat / nrm


def compute_distance_series(bins):
    vecs = [fused_vector(b) for b in bins]
    # replace None with zeros
    dims = max((v.shape[0] for v in vecs if v is not None), default=0)
    for i, v in enumerate(vecs):
        if v is None:
            vecs[i] = np.zeros(dims)
    X = np.stack(vecs) if vecs else np.zeros((0, dims))
    # distance between consecutive bins
    if X.shape[0] < 2:
        return np.array([]), X
    d = np.linalg.norm(np.diff(X, axis=0), axis=1)
    return d, X


def detect_change_points_with_ruptures(d, n_bkps=10):
    # use Pelt on l2 metric
    algo = rpt.Pelt(model="rbf").fit(d.reshape(-1, 1))
    bkps = algo.predict(pen=1.0)
    # convert bkps to indices in bins space
    return sorted(set(bkps))


def simple_threshold_change_points(d, thresh=None):
    if len(d) == 0:
        return []
    if thresh is None:
        # median + 1.5 * mad
        med = np.median(d)
        mad = np.median(np.abs(d - med))
        thresh = med + 1.5 * (mad if mad > 1e-12 else 1e-3)
    idx = [i + 1 for i, val in enumerate(d) if val >= thresh]
    return idx


def aggregate_bins_to_events(bins, change_indices):
    events = []
    start_idx = 0
    for cut in change_indices:
        ev_bins = bins[start_idx:cut]
        if not ev_bins:
            start_idx = cut
            continue
        ts = ev_bins[0]["start"]
        te = ev_bins[-1]["end"]
        # aggregate vector by averaging available embeddings
        vis = [b["visual_emb"] for b in ev_bins if b.get("visual_emb") is not None]
        aud = [b["audio_emb"] for b in ev_bins if b.get("audio_emb") is not None]
        txt = [b["text_emb"] for b in ev_bins if b.get("text_emb") is not None]
        def avg(lst):
            if not lst:
                return None
            arr = np.array(lst, dtype=float)
            vec = np.mean(arr, axis=0)
            if np.linalg.norm(vec) < 1e-12:
                return None
            return (vec / np.linalg.norm(vec)).tolist()

        events.append({
            "start": float(ts),
            "end": float(te),
            "n_bins": len(ev_bins),
            "visual_emb": avg(vis),
            "audio_emb": avg(aud),
            "text_emb": avg(txt),
            "texts": [b.get("text", "") for b in ev_bins],
        })
        start_idx = cut
    # last segment
    if start_idx < len(bins):
        ev_bins = bins[start_idx:]
        ts = ev_bins[0]["start"]
        te = ev_bins[-1]["end"]
        vis = [b["visual_emb"] for b in ev_bins if b.get("visual_emb") is not None]
        aud = [b["audio_emb"] for b in ev_bins if b.get("audio_emb") is not None]
        txt = [b["text_emb"] for b in ev_bins if b.get("text_emb") is not None]
        def avg(lst):
            if not lst:
                return None
            arr = np.array(lst, dtype=float)
            vec = np.mean(arr, axis=0)
            if np.linalg.norm(vec) < 1e-12:
                return None
            return (vec / np.linalg.norm(vec)).tolist()
        events.append({
            "start": float(ts),
            "end": float(te),
            "n_bins": len(ev_bins),
            "visual_emb": avg(vis),
            "audio_emb": avg(aud),
            "text_emb": avg(txt),
            "texts": [b.get("text", "") for b in ev_bins],
        })
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zt", required=True, help="zt_bins.json produced by stage1")
    parser.add_argument("--out", default="events.json")
    parser.add_argument("--use_ruptures", action="store_true")
    args = parser.parse_args()

    bins = load_bins(args.zt)
    d, X = compute_distance_series(bins)
    if d.size == 0:
        # trivial single event
        events = [{
            "start": bins[0]["start"] if bins else 0.0,
            "end": bins[-1]["end"] if bins else 0.0,
            "n_bins": len(bins),
            "visual_emb": bins[0].get("visual_emb"),
            "audio_emb": bins[0].get("audio_emb"),
            "text_emb": bins[0].get("text_emb"),
            "texts": [b.get("text", "") for b in bins],
        }]
    else:
        if args.use_ruptures and rpt is not None:
            try:
                cuts = detect_change_points_with_ruptures(X)
            except Exception:
                cuts = simple_threshold_change_points(d)
        else:
            cuts = simple_threshold_change_points(d)

        events = aggregate_bins_to_events(bins, cuts)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"events": events}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out} with {len(events)} events")


if __name__ == "__main__":
    main()
