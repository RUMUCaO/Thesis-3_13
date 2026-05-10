#!/usr/bin/env python3
"""
Structural Consensus Evaluation (SCE) utilities

Provides functions to compute:
 - Kendall's tau ordering agreement between two event sequences
 - Align(Si, Sj) semantic alignment score between two event lists
 - Consensus across K views

Each event is expected to have an embedding under key `embedding` (list or numpy array)
and a `start` timestamp.
"""
import argparse
import json
import math
from typing import List

import numpy as np
from scipy.stats import kendalltau


def kendall_tau_ordering(events_a: List[dict], events_b: List[dict]):
    # Build ranked order of events by start time
    a_starts = [e["start"] for e in events_a]
    b_starts = [e["start"] for e in events_b]
    # Create ordering indices based on start times
    a_order = np.argsort(a_starts)
    b_order = np.argsort(b_starts)
    # If lengths differ, we compute tau on pairwise relative order of min length prefix
    n = min(len(a_order), len(b_order))
    if n < 2:
        return 1.0
    # Use indices as ranks
    tau, p = kendalltau(a_order[:n], b_order[:n])
    if math.isnan(tau):
        return 0.0
    return float(tau)


def cosine(a, b):
    if a is None or b is None:
        return 0.0
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < 1e-12 or bn < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def align_score(events_i: List[dict], events_j: List[dict], emb_key: str = "embedding"):
    if not events_i:
        return 0.0
    total = 0.0
    for e in events_i:
        best = 0.0
        ei = e.get(emb_key)
        for ej in events_j:
            score = cosine(ei, ej.get(emb_key))
            if score > best:
                best = score
        total += best
    return total / len(events_i)


def consensus_score(views: List[List[dict]], emb_key: str = "embedding"):
    K = len(views)
    if K < 2:
        return 1.0
    s = 0.0
    count = 0
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            s += align_score(views[i], views[j], emb_key=emb_key)
            count += 1
    return s / count if count > 0 else 0.0


def load_events(path: str, emb_field_candidates=None):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    events = j.get("events", [])
    if emb_field_candidates is None:
        emb_field_candidates = ["text_emb", "visual_emb", "audio_emb", "embedding"]
    # ensure each event has a unified `embedding` field for evaluation; prefer text_emb then visual_emb
    for ev in events:
        for k in emb_field_candidates:
            if ev.get(k) is not None:
                ev["embedding"] = ev.get(k)
                break
        if ev.get("embedding") is None:
            ev["embedding"] = None
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--views", nargs='+', help="paths to event json files (events.json) from different views")
    args = parser.parse_args()
    views = [load_events(p) for p in args.views]
    # compute pairwise kendall between first two (as example)
    if len(views) >= 2:
        tau = kendall_tau_ordering(views[0], views[1])
        print(f"Kendall tau (view0 vs view1): {tau:.4f}")
    cons = consensus_score(views)
    print(f"Consensus score over {len(views)} views: {cons:.4f}")


if __name__ == "__main__":
    main()
