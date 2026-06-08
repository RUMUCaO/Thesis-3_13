#!/usr/bin/env python3
"""Select stable reference face clusters from whisperx results.

Usage:
  python select_reference_clusters.py whisperx_results.json --video input.mp4 --out candidates.json --top_k 20
  python select_reference_clusters.py whisperx_results.json --video input.mp4 --out candidates.json --aggregate --num_characters 10

Output: JSON list of candidate clusters with metadata and optional extracted frames in ./candidates/
With --aggregate: auto-merge similar embeddings into 10-15 character-level clusters (no manual labeling needed)
"""
import sys
import json
import argparse
import os
import subprocess
from collections import defaultdict
import numpy as np


def overlap(a0,a1,b0,b1):
    return max(0.0, min(a1,b1)-max(a0,b0))


def active_speakers_in_interval(diar, s0, s1, min_overlap=0.5):
    active = defaultdict(float)
    for seg in diar:
        o = overlap(s0,s1, seg['start'], seg['end'])
        if o >= min_overlap:
            active[seg['speaker']] += o
    return active


def select_candidates(
    results,
    min_duration=2.0,
    max_duration=180.0,
    min_samples=3,
    max_overlaps=2,
    max_active_speakers=3,
    min_dominant_share=0.25,
    min_purity=0.55,
    top_k=20,
):
    faces = results.get('faces') or []
    diar = results.get('diarization') or []

    # Precompute overlaps between face clusters
    face_list = []
    for f in faces:
        s,e = f['start_time'], f['end_time']
        dur = e-s
        samples = f.get('num_samples') or len(f.get('frame_indices', [])) if ('frame_indices' in f) else f.get('num_samples',0)
        face_list.append({'cluster':int(f['cluster']), 'start':s, 'end':e, 'num_samples':samples, 'duration':dur, 'raw':f})

    # compute overlaps count for each face
    for i,fi in enumerate(face_list):
        overlaps = 0
        for j,fj in enumerate(face_list):
            if i==j: continue
            if overlap(fi['start'],fi['end'], fj['start'], fj['end']) > 1.0:
                overlaps += 1
        fi['overlap_count'] = overlaps
        active = active_speakers_in_interval(diar, fi['start'], fi['end'])
        fi['active_speakers'] = len(active)
        fi['speaker_overlaps'] = dict(active)

        overlap_values = sorted(active.values(), reverse=True)
        fi['total_speaker_overlap'] = float(sum(overlap_values))
        fi['dominant_speaker_overlap'] = float(overlap_values[0]) if overlap_values else 0.0
        fi['second_speaker_overlap'] = float(overlap_values[1]) if len(overlap_values) > 1 else 0.0
        fi['dominant_share'] = fi['dominant_speaker_overlap'] / fi['duration'] if fi['duration'] > 0 else 0.0
        fi['purity'] = (
            fi['dominant_speaker_overlap'] / fi['total_speaker_overlap']
            if fi['total_speaker_overlap'] > 0
            else 0.0
        )
        fi['dominance_margin'] = fi['dominant_speaker_overlap'] - fi['second_speaker_overlap']

        # score: prefer clusters that are stable and speaker-pure, not merely long.
        duration_cap = min(fi['duration'], max_duration)
        sample_term = 1.0 + min(fi['num_samples'], 5000) / 100.0
        purity_term = 0.5 + fi['purity']
        dominant_term = 0.5 + fi['dominant_share']
        overlap_term = 1.0 / (1.0 + fi['overlap_count'] / 10.0)
        speaker_term = 1.0 / (1.0 + max(0, fi['active_speakers'] - 1) * 0.35)
        fi['score'] = duration_cap * sample_term * purity_term * dominant_term * overlap_term * speaker_term

    # Prefer strict candidates, but always provide a ranked shortlist.
    strict = []
    relaxed = []
    for f in face_list:
        passes = (
            f['duration'] >= min_duration and
            f['duration'] <= max_duration and
            f['num_samples'] >= min_samples and
            f['active_speakers'] <= max_active_speakers and
            f['dominant_share'] >= min_dominant_share and
            f['purity'] >= min_purity
        )
        f['passes_thresholds'] = passes

        # Penalize near-misses but keep them in the pool so the shortlist is not empty.
        penalty = 1.0
        if f['duration'] < min_duration:
            penalty *= 0.6
        if f['duration'] > max_duration:
            penalty *= 0.2
        if f['num_samples'] < min_samples:
            penalty *= 0.7
        if f['overlap_count'] > max_overlaps:
            penalty *= max(0.2, 1.0 / (1.0 + f['overlap_count'] / 20.0))
        if f['active_speakers'] > max_active_speakers:
            penalty *= 0.4
        if f['dominant_share'] < min_dominant_share:
            penalty *= 0.3
        if f['purity'] < min_purity:
            penalty *= 0.3

        f['rank_score'] = f['score'] * penalty
        if passes:
            strict.append(f)
        else:
            relaxed.append(f)

    strict_sorted = sorted(strict, key=lambda x: x['rank_score'], reverse=True)
    relaxed_sorted = sorted(relaxed, key=lambda x: x['rank_score'], reverse=True)

    shortlist = strict_sorted[:top_k]
    if len(shortlist) < top_k:
        need = top_k - len(shortlist)
        shortlist.extend(relaxed_sorted[:need])

    return shortlist


def aggregate_candidates_by_embedding(candidates, num_characters=10, embedding_eps=0.4):
    """
    Aggregate candidates by face embedding similarity using DBSCAN.
    Each group represents a potential character.
    
    Returns: list of mega-clusters, each containing:
    - character_id: assigned character number (1, 2, ...)
    - representative_cluster: best cluster in this group (highest rank_score)
    - member_clusters: all cluster IDs in this group
    - mean_embedding: average embedding of this group
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        print("Warning: sklearn not available, skipping embedding aggregation")
        # fallback: just return each candidate as its own character
        mega = []
        for i, c in enumerate(candidates, 1):
            mega.append({
                'character_id': i,
                'representative_cluster': c['cluster'],
                'member_clusters': [c['cluster']],
                'representative_item': c,
            })
        return mega

    # Extract embeddings
    embeddings = []
    valid_indices = []
    for i, c in enumerate(candidates):
        emb = c['raw'].get('embedding')
        if emb is not None:
            try:
                # Handle float16 compression
                if isinstance(emb, list):
                    emb = np.array(emb, dtype=np.float32)
                else:
                    emb = np.array(emb, dtype=np.float32)
                # Normalize
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                embeddings.append(emb)
                valid_indices.append(i)
            except Exception:
                pass

    if len(embeddings) == 0:
        print("No valid embeddings found, returning ungrouped candidates")
        mega = []
        for i, c in enumerate(candidates, 1):
            mega.append({
                'character_id': i,
                'representative_cluster': c['cluster'],
                'member_clusters': [c['cluster']],
                'representative_item': c,
            })
        return mega

    embeddings = np.vstack(embeddings)
    
    # DBSCAN clustering on embeddings
    # Use cosine distance via metric='cosine' and eps=0.4 (default: similar to merge_eps threshold)
    clustering = DBSCAN(eps=embedding_eps, min_samples=1, metric='cosine').fit(embeddings)
    labels = clustering.labels_

    # Group by cluster label
    groups = defaultdict(list)
    for label, idx in zip(labels, valid_indices):
        groups[label].append(idx)

    # Build mega-clusters
    mega = []
    for char_id, indices in enumerate(sorted(groups.keys()), 1):
        indices_list = groups[indices]
        members = [candidates[i]['cluster'] for i in indices_list]
        
        # Pick representative (highest rank_score among members)
        best_idx = max(indices_list, key=lambda i: candidates[i]['rank_score'])
        rep_candidate = candidates[best_idx]
        
        # Compute mean embedding
        mean_emb = np.mean([embeddings[valid_indices.index(i)] for i in indices_list], axis=0)
        
        mega.append({
            'character_id': char_id,
            'representative_cluster': rep_candidate['cluster'],
            'member_clusters': members,
            'representative_item': rep_candidate,
            'num_members': len(members),
            'mean_embedding': mean_emb.tolist(),
        })

    # Sort by representative rank_score to prioritize better candidates
    mega = sorted(mega, key=lambda x: x['representative_item']['rank_score'], reverse=True)
    
    # Re-assign character IDs (1..N) in priority order
    for char_id, item in enumerate(mega, 1):
        item['character_id'] = char_id

    # Limit to requested number
    if num_characters and len(mega) > num_characters:
        print(f"Limiting {len(mega)} mega-clusters to top {num_characters} by rank")
        mega = mega[:num_characters]

    return mega

def extract_frame(video_path, time_s, out_path):
    # use ffmpeg to extract a single frame at given time
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-ss', f'{time_s:.3f}', '-i', video_path,
        '-frames:v', '1', '-q:v', '2', out_path
    ]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def main(
    results: str = "whisperx_results.json",
    video: str = None,
    out: str = "candidates.json",
    top_k: int = 20,
    aggregate: bool = True,
    num_characters: int = 10,
    embedding_eps: float = 0.4,
    min_duration: float = 2.0,
    max_duration: float = 180.0,
    min_samples: int = 3,
    max_overlaps: int = 2,
    max_active_speakers: int = 3,
    min_dominant_share: float = 0.25,
    min_purity: float = 0.55,
):
    with open(results, 'r', encoding='utf8') as f:
        results_data = json.load(f)

    candidates = select_candidates(
        results_data,
        min_duration=min_duration,
        max_duration=max_duration,
        min_samples=min_samples,
        max_overlaps=max_overlaps,
        max_active_speakers=max_active_speakers,
        min_dominant_share=min_dominant_share,
        min_purity=min_purity,
        top_k=top_k,
    )

    # Optionally aggregate by embedding
    if aggregate:
        mega_clusters = aggregate_candidates_by_embedding(
            candidates,
            num_characters=num_characters,
            embedding_eps=embedding_eps,
        )
        print(f"Aggregated {len(candidates)} candidates into {len(mega_clusters)} character-level clusters")
        
        # prepare output with mega-clusters
        out_list = []
        base_dir = 'candidates'
        if video:
            os.makedirs(base_dir, exist_ok=True)

        for mega in mega_clusters:
            c = mega['representative_item']
            rep_time = (c['start'] + c['end']) / 2.0
            item = {
                'character_id': mega['character_id'],  # Auto-assigned character ID
                'cluster': mega['representative_cluster'],
                'member_clusters': mega['member_clusters'],
                'start': c['start'],
                'end': c['end'],
                'duration': c['duration'],
                'num_samples': c['num_samples'],
                'overlap_count': c['overlap_count'],
                'active_speakers': c['active_speakers'],
                'score': c['score'],
                'rank_score': c['rank_score'],
                'passes_thresholds': c['passes_thresholds'],
                'dominant_share': c['dominant_share'],
                'purity': c['purity'],
                'dominant_speaker_overlap': c['dominant_speaker_overlap'],
                'second_speaker_overlap': c['second_speaker_overlap'],
                'speaker_overlaps': c['speaker_overlaps'],
            }
            if video:
                out_img = os.path.join(base_dir, f'character_{mega["character_id"]}_cluster_{mega["representative_cluster"]}.jpg')
                ok = extract_frame(video, rep_time, out_img)
                if ok:
                    item['representative_frame'] = out_img
                else:
                    item['representative_frame'] = None
            out_list.append(item)

        with open(out, 'w', encoding='utf8') as f:
            json.dump(out_list, f, ensure_ascii=False, indent=2)

        print(f'Wrote {len(out_list)} character clusters to {out}')
        if video:
            print(f'Frames (when extracted) saved in {base_dir}')

    else:
        # Original mode: output individual candidates without aggregation
        out_list = []
        base_dir = 'candidates'
        if video:
            os.makedirs(base_dir, exist_ok=True)

        for c in candidates:
            rep_time = (c['start'] + c['end']) / 2.0
            item = {
                'cluster': c['cluster'],
                'start': c['start'],
                'end': c['end'],
                'duration': c['duration'],
                'num_samples': c['num_samples'],
                'overlap_count': c['overlap_count'],
                'active_speakers': c['active_speakers'],
                'score': c['score'],
                'rank_score': c['rank_score'],
                'passes_thresholds': c['passes_thresholds'],
                'dominant_share': c['dominant_share'],
                'purity': c['purity'],
                'dominant_speaker_overlap': c['dominant_speaker_overlap'],
                'second_speaker_overlap': c['second_speaker_overlap'],
                'speaker_overlaps': c['speaker_overlaps'],
            }
            if video:
                out_img = os.path.join(base_dir, f'cluster_{c["cluster"]}.jpg')
                ok = extract_frame(video, rep_time, out_img)
                if ok:
                    item['representative_frame'] = out_img
                else:
                    item['representative_frame'] = None
            out_list.append(item)

        with open(out, 'w', encoding='utf8') as f:
            json.dump(out_list, f, ensure_ascii=False, indent=2)

        print(f'Wrote {len(out_list)} candidates to {out}')
        if video:
            print(f'Frames (when extracted) saved in {base_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select and optionally aggregate face clusters into character-level references")
    parser.add_argument("results", nargs='?', default="whisperx_results.json", help="whisperx_results.json path")
    parser.add_argument("--video", help="input video path for frame extraction")
    parser.add_argument("--out", default="candidates.json", help="output JSON path")
    parser.add_argument("--top_k", type=int, default=20, help="top K candidates to consider")
    parser.add_argument("--aggregate", action="store_true", help="aggregate candidates by embedding into character-level clusters")
    parser.add_argument("--num_characters", type=int, default=10, help="target number of character clusters (used with --aggregate)")
    parser.add_argument("--embedding_eps", type=float, default=0.4, help="DBSCAN eps for embedding clustering (used with --aggregate)")
    parser.add_argument("--min_duration", type=float, default=2.0, help="minimum cluster duration (seconds)")
    parser.add_argument("--max_duration", type=float, default=180.0, help="maximum cluster duration (seconds)")
    parser.add_argument("--min_samples", type=int, default=3, help="minimum number of samples in cluster")
    parser.add_argument("--max_overlaps", type=int, default=2, help="maximum number of overlapping face clusters")
    parser.add_argument("--max_active_speakers", type=int, default=3, help="maximum active speakers in cluster")
    parser.add_argument("--min_dominant_share", type=float, default=0.25, help="minimum dominant speaker share")
    parser.add_argument("--min_purity", type=float, default=0.55, help="minimum speaker purity score")
    
    args = parser.parse_args()
    
    main(
        results=args.results,
        video=args.video,
        out=args.out,
        top_k=args.top_k,
        aggregate=args.aggregate,
        num_characters=args.num_characters,
        embedding_eps=args.embedding_eps,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_samples=args.min_samples,
        max_overlaps=args.max_overlaps,
        max_active_speakers=args.max_active_speakers,
        min_dominant_share=args.min_dominant_share,
        min_purity=args.min_purity,
    )