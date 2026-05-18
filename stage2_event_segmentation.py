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
from typing import Dict, List

import numpy as np

try:
    import ruptures as rpt
except Exception:
    rpt = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def load_bins(path: str):
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j.get("bins", [])


def _normalize_vec(vec: np.ndarray):
    nrm = np.linalg.norm(vec)
    if nrm < 1e-12:
        return None
    return vec / nrm


def _pad_or_truncate(vec: np.ndarray, target_dim: int):
    if vec.shape[0] < target_dim:
        return np.pad(vec, (0, target_dim - vec.shape[0]))
    if vec.shape[0] > target_dim:
        return vec[:target_dim]
    return vec


def infer_modality_dims(bins) -> Dict[str, int]:
    dims = {"visual": 0, "audio": 0, "text": 0, "subtitle": 0}
    for b in bins:
        if b.get("visual_emb") is not None:
            dims["visual"] = max(dims["visual"], int(np.asarray(b["visual_emb"]).shape[0]))
        if b.get("audio_emb") is not None:
            dims["audio"] = max(dims["audio"], int(np.asarray(b["audio_emb"]).shape[0]))
        if b.get("text_emb") is not None:
            dims["text"] = max(dims["text"], int(np.asarray(b["text_emb"]).shape[0]))
        # subtitle_emb is expected to be provided by stage1 or precomputed
        if b.get("subtitle_emb") is not None:
            dims["subtitle"] = max(dims["subtitle"], int(np.asarray(b["subtitle_emb"]).shape[0]))
    return dims


def fused_vector(
    b,
    modality_dims: Dict[str, int],
    base_weights: Dict[str, float],
    use_modality_confidence: bool = True,
    subtitle_vec=None,
):
    """Build weighted multimodal vector with per-modality normalization and dynamic masking.

    Key behaviors:
    - Normalize each modality before fusion.
    - Apply modality weights (wv, wa, wt) before concatenation.
    - Renormalize active weights for available/confident modalities only.
    - Avoid fake geometry from hard zero-padding absent modalities.
    """
    normalized_vectors: Dict[str, np.ndarray] = {}
    active_weights: Dict[str, float] = {}

    for key, emb_key in (("visual", "visual_emb"), ("audio", "audio_emb"), ("text", "text_emb")):
        target_dim = modality_dims.get(key, 0)
        if target_dim <= 0:
            continue

        raw = b.get(emb_key)
        if raw is None:
            continue

        vec = np.asarray(raw, dtype=float)
        vec = _pad_or_truncate(vec, target_dim)
        vec = _normalize_vec(vec)
        if vec is None:
            continue

        confidence = 1.0
        if use_modality_confidence and key == "text":
            confidence = float(b.get("confidence", 1.0))
            confidence = max(0.0, min(1.0, confidence))

        weight = float(base_weights.get(key, 1.0)) * confidence
        if weight <= 0.0:
            continue

        normalized_vectors[key] = vec
        active_weights[key] = weight

    # handle subtitle separately (may be aggregated over a window before passing in)
    sub_dim = modality_dims.get("subtitle", 0)
    if sub_dim > 0 and subtitle_vec is not None:
        svec = np.asarray(subtitle_vec, dtype=float)
        svec = _pad_or_truncate(svec, sub_dim)
        svec = _normalize_vec(svec)
        if svec is not None:
            # avoid duplicate counting: if subtitle and text are nearly identical, drop subtitle
            if "text" in normalized_vectors:
                tvec = normalized_vectors["text"]
                common_dim = min(int(tvec.shape[0]), int(svec.shape[0]))
                if common_dim > 0:
                    tcmp = tvec[:common_dim]
                    scmp = svec[:common_dim]
                    sim = float(np.dot(tcmp, scmp)) if (np.linalg.norm(tcmp) > 0 and np.linalg.norm(scmp) > 0) else 0.0
                else:
                    sim = 0.0
                if sim >= 0.995:
                    # treat subtitle as duplicate of text; skip subtitle
                    pass
                else:
                    sw = float(base_weights.get("subtitle", 1.0))
                    if sw > 0.0:
                        normalized_vectors["subtitle"] = svec
                        active_weights["subtitle"] = sw
            else:
                sw = float(base_weights.get("subtitle", 1.0))
                if sw > 0.0:
                    normalized_vectors["subtitle"] = svec
                    active_weights["subtitle"] = sw

    if not normalized_vectors:
        return None

    weight_sum = float(sum(active_weights.values()))
    if weight_sum <= 1e-12:
        return None

    weighted_parts = []
    for key in ("visual", "audio", "subtitle", "text"):
        target_dim = modality_dims.get(key, 0)
        if target_dim <= 0:
            continue
        if key in normalized_vectors:
            nw = active_weights[key] / weight_sum
            weighted_parts.append(nw * normalized_vectors[key])
        else:
            weighted_parts.append(np.zeros(target_dim, dtype=float))

    cat = np.concatenate(weighted_parts)
    cat = _normalize_vec(cat)
    return cat


def smooth_features(X: np.ndarray, window: int = 3):
    if X.ndim != 2 or X.shape[0] == 0 or window <= 1:
        return X
    window = max(1, int(window))
    radius = window // 2
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        left = max(0, i - radius)
        right = min(X.shape[0], i + radius + 1)
        out[i] = np.mean(X[left:right], axis=0)
    return out


def aggregate_subtitle_embeddings(bins, window_secs: float = 10.0):
    """Aggregate subtitle embeddings per bin using a time window (seconds).

    For each bin, average subtitle_embs from bins whose center time is within
    +/- window_secs/2. This reduces jitter from per-utterance subtitles.
    Returns a list (len=bins) of aggregated subtitle vectors or None.
    """
    if not bins:
        return []
    if window_secs is None or window_secs <= 0:
        # no aggregation, return raw subtitle_emb if present
        return [b.get("subtitle_emb") for b in bins]

    centers = [0.5 * (b.get("start", 0.0) + b.get("end", 0.0)) for b in bins]
    half = float(window_secs) / 2.0
    aggr = []
    for i, c in enumerate(centers):
        parts = []
        for j, cj in enumerate(centers):
            if abs(cj - c) <= half:
                s = bins[j].get("subtitle_emb")
                if s is not None:
                    parts.append(np.asarray(s, dtype=float))
        if not parts:
            aggr.append(None)
        else:
            arr = np.mean(np.stack(parts, axis=0), axis=0)
            vec = _normalize_vec(arr)
            aggr.append(None if vec is None else vec)
    return aggr


def parse_srt(path: str):
    """Minimal SRT parser returning list of {'start': seconds, 'end': seconds, 'text': str}.

    Not robust to all SRT variants but sufficient for typical files.
    """
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    # split by blank lines
    parts = [p.strip() for p in content.replace('\r\n', '\n').split('\n\n') if p.strip()]
    for p in parts:
        lines = [l.strip() for l in p.split('\n') if l.strip()]
        if len(lines) < 2:
            continue
        # second line should be timecode
        timecode = lines[1]
        if "-->" not in timecode:
            # sometimes the timecode is on first line if index missing
            timecode = lines[0]
            text_lines = lines[1:]
        else:
            text_lines = lines[2:]
        try:
            left, right = [t.strip() for t in timecode.split("-->")]
            def to_sec(ts: str):
                # format: HH:MM:SS,ms or H:MM:SS.ms
                ts = ts.replace(',', '.')
                parts = ts.split(':')
                if len(parts) == 3:
                    h, m, s = parts
                    return float(h) * 3600.0 + float(m) * 60.0 + float(s)
                return float(ts)
            s = to_sec(left)
            e = to_sec(right)
        except Exception:
            continue
        txt = " ".join(text_lines).strip()
        out.append({"start": s, "end": e, "text": txt})
    return out


def attach_subtitle_embeddings_to_bins(bins, srt_entries, srt_embs, assign_window: float = None):
    """Attach subtitle embedding(s) to bins. For each subtitle entry, map to a bin by midpoint.

    If no bin contains the midpoint, assign to nearest bin within assign_window/2 if provided.
    The function sets `bin['subtitle_emb']` to the mean of assigned subtitle embeddings (or None).
    """
    if not bins:
        return
    # prepare container lists
    for b in bins:
        b.setdefault("_subtitle_parts", [])

    bin_centers = [0.5 * (b.get("start", 0.0) + b.get("end", 0.0)) for b in bins]
    for entry, emb in zip(srt_entries, srt_embs):
        mid = 0.5 * (entry["start"] + entry["end"])
        assigned = False
        for i, b in enumerate(bins):
            if b.get("start", 0.0) <= mid < b.get("end", 0.0):
                b["_subtitle_parts"].append(emb)
                assigned = True
                break
        if not assigned and assign_window is not None and assign_window > 0:
            # find nearest bin center
            dists = [abs(mc - mid) for mc in bin_centers]
            idx = int(np.argmin(dists))
            if dists[idx] <= (assign_window / 2.0):
                bins[idx]["_subtitle_parts"].append(emb)

    # finalize: compute mean per bin and attach as 'subtitle_emb'
    for b in bins:
        parts = b.get("_subtitle_parts", [])
        if not parts:
            b["subtitle_emb"] = None
        else:
            arr = np.mean(np.stack([np.asarray(p, dtype=float) for p in parts], axis=0), axis=0)
            vec = _normalize_vec(arr)
            b["subtitle_emb"] = None if vec is None else vec.tolist()
        # cleanup
        if "_subtitle_parts" in b:
            del b["_subtitle_parts"]



def compute_distance_series(
    bins,
    visual_weight: float = 1.0,
    audio_weight: float = 1.0,
    text_weight: float = 1.0,
    subtitle_weight: float = 0.0,
    subtitle_window: float = 10.0,
    smooth_window: int = 3,
    use_modality_confidence: bool = True,
):
    modality_dims = infer_modality_dims(bins)
    weights = {
        "visual": float(visual_weight),
        "audio": float(audio_weight),
        "text": float(text_weight),
        "subtitle": float(subtitle_weight),
    }

    # precompute aggregated subtitle vectors per bin to reduce jitter
    subtitle_aggr = aggregate_subtitle_embeddings(bins, window_secs=subtitle_window)

    vecs = []
    for i, b in enumerate(bins):
        svec = subtitle_aggr[i] if i < len(subtitle_aggr) else None
        v = fused_vector(
            b,
            modality_dims=modality_dims,
            base_weights=weights,
            use_modality_confidence=use_modality_confidence,
            subtitle_vec=svec,
        )
        vecs.append(v)

    total_dim = int(sum(modality_dims.values()))
    for i, v in enumerate(vecs):
        vecs[i] = np.zeros(total_dim) if v is None else v

    X = np.stack(vecs) if vecs else np.zeros((0, total_dim))
    X = smooth_features(X, window=smooth_window)
    # distance between consecutive bins
    if X.shape[0] < 2:
        return np.array([]), X
    d = np.linalg.norm(np.diff(X, axis=0), axis=1)
    return d, X


def detect_change_points_with_ruptures(X, penalty: float = 1.0):
    # Segment directly on embedding trajectory (not on diff signal).
    algo = rpt.Pelt(model="rbf").fit(X)
    bkps = algo.predict(pen=penalty)
    # convert bkps to indices in bins space
    n = X.shape[0]
    return sorted(set(int(c) for c in bkps if 0 < int(c) < n))


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


def enforce_min_event_bins(change_indices, n_bins: int, min_event_bins: int):
    if min_event_bins <= 1 or n_bins <= 1:
        return sorted(set(int(c) for c in change_indices if 0 < int(c) < n_bins))

    cuts = sorted(set(int(c) for c in change_indices if 0 < int(c) < n_bins))
    changed = True
    while changed:
        changed = False
        boundaries = [0] + cuts + [n_bins]
        seg_lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]

        for seg_idx, seg_len in enumerate(seg_lengths):
            if seg_len >= min_event_bins:
                continue

            # Remove one adjacent boundary to merge this short segment.
            left_cut_idx = seg_idx - 1
            right_cut_idx = seg_idx

            if left_cut_idx < 0 and right_cut_idx >= len(cuts):
                # Single segment only.
                cuts = []
            elif left_cut_idx < 0:
                cuts.pop(right_cut_idx)
            elif right_cut_idx >= len(cuts):
                cuts.pop(left_cut_idx)
            else:
                left_neighbor_len = seg_lengths[seg_idx - 1]
                right_neighbor_len = seg_lengths[seg_idx + 1]
                # Prefer merging into the shorter neighbor to reduce over-merge.
                if left_neighbor_len <= right_neighbor_len:
                    cuts.pop(left_cut_idx)
                else:
                    cuts.pop(right_cut_idx)

            changed = True
            break

    return cuts


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
        subs = [b["subtitle_emb"] for b in ev_bins if b.get("subtitle_emb") is not None]
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
            "subtitle_emb": avg(subs),
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
        subs = [b["subtitle_emb"] for b in ev_bins if b.get("subtitle_emb") is not None]
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
            "subtitle_emb": avg(subs),
            "texts": [b.get("text", "") for b in ev_bins],
        })
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zt", required=True, help="zt_bins.json produced by stage1")
    parser.add_argument("--out", default="events.json")
    parser.add_argument("--use_ruptures", action="store_true")
    parser.add_argument("--visual_weight", type=float, default=1.0, help="Weight for visual modality in fusion")
    parser.add_argument("--audio_weight", type=float, default=1.0, help="Weight for audio modality in fusion")
    parser.add_argument("--text_weight", type=float, default=1.0, help="Weight for text modality in fusion")
    parser.add_argument("--subtitle_weight", type=float, default=0.4, help="Weight for subtitle modality (high-frequency text) in fusion")
    parser.add_argument("--subtitle_window", type=float, default=10.0, help="Subtitle aggregation window in seconds (use 0 to disable)")
    parser.add_argument("--subtitles", type=str, default=None, help="Path to separate subtitles SRT file (optional)")
    parser.add_argument("--subtitle_model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model for subtitle embeddings")
    parser.add_argument("--subtitle_batch", type=int, default=64, help="Batch size when encoding subtitles")
    parser.add_argument("--smooth_window", type=int, default=3, help="Temporal smoothing window on fused trajectory")
    parser.add_argument("--min_event_bins", type=int, default=3, help="Minimum number of bins per event")
    parser.add_argument("--ruptures_penalty", type=float, default=1.0, help="Penalty used by ruptures PELT")
    parser.add_argument(
        "--disable_modality_confidence",
        action="store_true",
        help="Disable confidence-aware dynamic weighting (text confidence from Stage 1)",
    )
    args = parser.parse_args()

    bins = load_bins(args.zt)

    # if supplied, compute subtitle embeddings from SRT and attach to bins
    if args.subtitles:
        if SentenceTransformer is None:
            print("[WARN] sentence-transformers not installed; cannot compute subtitle embeddings.")
        else:
            srt_entries = parse_srt(args.subtitles)
            if not srt_entries:
                print(f"[WARN] no subtitle entries parsed from {args.subtitles}")
            else:
                model = SentenceTransformer(args.subtitle_model)
                texts = [e["text"] for e in srt_entries]
                emb_list = []
                batch = args.subtitle_batch if args.subtitle_batch and args.subtitle_batch > 0 else 64
                for i in range(0, len(texts), batch):
                    chunk = texts[i : i + batch]
                    embs = model.encode(chunk, convert_to_numpy=True, show_progress_bar=False)
                    # normalize
                    for v in embs:
                        vn = _normalize_vec(np.asarray(v, dtype=float))
                        emb_list.append(None if vn is None else vn)

                # attach embeddings to bins by midpoint mapping within subtitle_window
                assign_window = args.subtitle_window if args.subtitle_window and args.subtitle_window > 0 else None
                attach_subtitle_embeddings_to_bins(bins, srt_entries, emb_list, assign_window=assign_window)
    d, X = compute_distance_series(
        bins,
        visual_weight=args.visual_weight,
        audio_weight=args.audio_weight,
        text_weight=args.text_weight,
        subtitle_weight=args.subtitle_weight,
        subtitle_window=args.subtitle_window,
        smooth_window=args.smooth_window,
        use_modality_confidence=not args.disable_modality_confidence,
    )
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
                cuts = detect_change_points_with_ruptures(X, penalty=args.ruptures_penalty)
            except Exception:
                cuts = simple_threshold_change_points(d)
        else:
            cuts = simple_threshold_change_points(d)

        cuts = enforce_min_event_bins(cuts, n_bins=len(bins), min_event_bins=args.min_event_bins)

        events = aggregate_bins_to_events(bins, cuts)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"events": events}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out} with {len(events)} events")


if __name__ == "__main__":
    main()
