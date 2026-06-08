#!/usr/bin/env python3
"""Analyze overlap between diarization speakers and detected face clusters.

Usage:
  python analyze_speaker_face_overlap.py [path/to/whisperx_results.json]

Output: prints per-speaker overlap times with face clusters and highlights ambiguous cases.
"""
import sys
import json
from collections import defaultdict


def load(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def overlap_seconds(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def main(path='whisperx_results.json'):
    data = load(path)
    diar = data.get('diarization')
    faces = data.get('face_segments')

    if diar is None:
        print('No diarization found in', path)
        return
    if faces is None:
        print('No face_segments found in', path)
        return

    # Normalize types
    face_map = {int(f['cluster']): (f['start'], f['end']) for f in faces}

    speakers = sorted({s['speaker'] for s in diar})
    clusters = sorted(face_map.keys())

    # compute overlap seconds per speaker x cluster
    overlap = {sp: defaultdict(float) for sp in speakers}

    for seg in diar:
        s0, s1, sp = seg['start'], seg['end'], seg['speaker']
        for cl, (f0, f1) in face_map.items():
            o = overlap_seconds(s0, s1, f0, f1)
            if o > 0:
                overlap[sp][cl] += o

    # Print summary
    print(f'Speakers: {len(speakers)}, face clusters: {len(clusters)}')
    print('\nPer-speaker top face-cluster overlaps (seconds):')
    for sp in speakers:
        items = sorted(overlap[sp].items(), key=lambda kv: kv[1], reverse=True)
        total = sum(v for _, v in items)
        print(f'  {sp}: total_overlap={total:.2f}s, matches={len(items)}')
        for cl, sec in items[:5]:
            print(f'    cluster {cl}: {sec:.2f}s')
        if len(items) > 5:
            print(f'    ... {len(items)-5} more')

    # Identify ambiguous speakers (no dominant cluster or multiple similar matches)
    print('\nPossible issues:')
    for sp in speakers:
        items = sorted(overlap[sp].items(), key=lambda kv: kv[1], reverse=True)
        if not items:
            print(f'  {sp}: no temporal overlap with any face cluster')
            continue
        top_sec = items[0][1]
        rest = sum(v for _, v in items[1:])
        if top_sec < 0.5:
            print(f'  {sp}: top overlap small ({top_sec:.2f}s) — diarization segments may be short or misaligned')
        if rest > top_sec * 0.5:
            print(f'  {sp}: ambiguous — secondary clusters total {rest:.2f}s (~{rest/top_sec:.2f}x of top)')

    print('\nYou can inspect speaker/cluster time ranges to debug further.')


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'whisperx_results.json'
    main(path)
