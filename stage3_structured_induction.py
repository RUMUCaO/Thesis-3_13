#!/usr/bin/env python3
"""
Stage 3 — Structured Representation Induction

Inputs:
 - events.json produced by stage2_event_segmentation.py

Produces `structure.json` containing nodes and sparse edges computed by a compatibility function:
  wij = alpha * semantic_cos + beta * temporal_kernel + gamma * entity_score

This baseline implementation uses event embeddings (text_emb or visual_emb) for semantic similarity
and temporal distance for temporal kernel. entity_score is left as 0 unless provided.
"""
import argparse
import json
import math
from typing import List

import numpy as np


def load_events(path: str):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j.get("events", [])


def get_embedding(ev):
    # prefer text, then visual, then audio
    if ev.get("text_emb") is not None:
        return np.array(ev["text_emb"], dtype=float)
    if ev.get("visual_emb") is not None:
        return np.array(ev["visual_emb"], dtype=float)
    if ev.get("audio_emb") is not None:
        return np.array(ev["audio_emb"], dtype=float)
    return None


def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an < 1e-12 or bn < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def temporal_kernel(a_start, b_start, sigma=30.0):
    # gaussian kernel on start time difference (seconds)
    d = abs(a_start - b_start)
    return math.exp(- (d * d) / (2 * sigma * sigma))


def build_structure(events, alpha=0.6, beta=0.3, gamma=0.1, top_k=3):
    n = len(events)
    embs = [get_embedding(ev) for ev in events]
    starts = [ev["start"] for ev in events]
    nodes = [{"id": i, "start": events[i]["start"], "end": events[i]["end"], "n_bins": events[i].get("n_bins", 1)} for i in range(n)]
    edges = []
    for i in range(n):
        scores = []
        for j in range(n):
            if i == j:
                continue
            s_sem = cosine_sim(embs[i], embs[j])
            s_tmp = temporal_kernel(starts[i], starts[j])
            s_ent = 0.0  # placeholder for entity consistency (if available)
            w = alpha * s_sem + beta * s_tmp + gamma * s_ent
            scores.append((j, w))
        # keep top_k neighbors
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        for j, w in scores:
            if w <= 0:
                continue
            edges.append({"source": i, "target": j, "weight": float(w)})
    return {"nodes": nodes, "edges": edges}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", default="structure.json")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    events = load_events(args.events)
    struct = build_structure(events, top_k=args.top_k)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(struct, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out} with {len(struct['nodes'])} nodes and {len(struct['edges'])} edges")


if __name__ == "__main__":
    main()
