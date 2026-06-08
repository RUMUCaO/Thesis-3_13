#!/usr/bin/env python3
"""
Lightweight implementation of the RoleNet pipeline from
Weng et al., "RoleNet: Movie Analysis from the Perspective of Social Networks".

This script reproduces the paper's processing flow given a simple
input: a JSON file listing scenes, each with the set of role names
that appear in that scene. It builds a RoleNet (co-occurrence),
computes centrality, auto-detects leading roles, identifies macro
and micro communities, and runs a context-based story segmentation
(`storyshed`). Results are written to an output JSON.

The event extraction path is intentionally RoleNet-style:
scene -> role graph -> boundary detection -> contiguous event spans ->
LLM/extractive event labels.

Usage example:
  python RoleNet_pipeline.py --input scenes.json --out rolenet_out.json

Input format (JSON): list of scenes; each scene is a dict:
  [{"scene_id":0, "roles":["Alice","Bob"]}, ...]

This is a reproducible, minimal reference implementation — it
does not include face detection or automatic scene detection.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

try:
    import networkx as nx
except Exception:
    nx = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

try:
    import hdbscan
except Exception:
    hdbscan = None

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph
except Exception:
    AgglomerativeClustering = None
    kneighbors_graph = None

try:
    import openai
except Exception:
    openai = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None

def convert_identity_to_scenes(data: dict) -> List[Dict]:
    """Convert identity_reconciliation.json format to scene list for RoleNet."""
    unified = data.get("unified_characters", [])
    if not unified:
        # not identity format, return as-is (maybe it's already scene list)
        return None

    scene_roles = defaultdict(list)
    for char in unified:
        char_id = char.get("character_id")
        if not char_id:
            continue
        for sid in char.get("scene_ids", []):
            scene_roles[sid].append(char_id)

    # Build scene list sorted by scene_id
    scenes = []
    for sid in sorted(scene_roles.keys()):
        scenes.append({
            "scene_id": sid,
            "roles": scene_roles[sid]
        })
    return scenes

def embed_texts(texts):
    # return L2-normalized numpy array embeddings
    prepared = [t if isinstance(t, str) and t.strip() else " " for t in texts]
    use_sentence_transformer = os.environ.get('USE_SENTENCE_TRANSFORMER', '').strip().lower() in {'1', 'true', 'yes'}
    if use_sentence_transformer and SentenceTransformer is not None:
        try:
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embs = model.encode(prepared, convert_to_numpy=True, normalize_embeddings=True)
            return np.asarray(embs, dtype=np.float32)
        except Exception:
            pass
    # default: TF-IDF dense vectors for fast, reproducible boundary detection
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=2048)
        mat = np.asarray(vec.fit_transform(prepared).todense(), dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        mat = mat / norms
        return mat
    except Exception:
        # last resort: bag-of-roles placeholder
        mat = np.zeros((len(prepared), 16), dtype=np.float32)
        return mat


def scene_text_for_event(scene: Dict, include_summary: bool = True) -> str:
    parts = []
    keys = ('description', 'text', 'action')
    if include_summary:
        keys = ('summary',) + keys
    for key in keys:
        value = scene.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item)
        elif isinstance(value, dict):
            parts.extend(str(item) for item in value.values() if item)
        elif value:
            parts.append(str(value))

    dialogue = scene.get('dialogue')
    if isinstance(dialogue, list):
        for item in dialogue:
            if isinstance(item, dict):
                text = item.get('text')
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))

    cast = scene.get('cast')
    if isinstance(cast, list):
        parts.extend(str(item) for item in cast if item)

    roles = scene.get('roles')
    if isinstance(roles, list):
        parts.extend(str(item) for item in roles if item)

    return ' '.join(part for part in parts if part).strip()


DEFAULT_EVENT_SUMMARY_TEMPLATE = {
    "system": (
        "You are a narrative compression engine. Summarize the input into a concise event summary. "
        "Preserve the main characters, the causal actions, and the event progression. Output 1-3 sentences."
    ),
    "chunk_user": (
        "Summarize this chunk in 1-2 sentences, keeping the main characters and actions. "
        "Do not invent new characters or actions.\n\n{chunk}"
    ),
    "reduce_user": (
        "Produce a concise event summary in 1-3 sentences from the following chunk summaries. "
        "Keep role names consistent, preserve causal order, and do not introduce unsupported entities.\n\n{chunk_summaries}"
    ),
    "role_constraint": (
        "Role consistency constraints: only mention roles that appear in the provided role list unless the input explicitly supports others. "
        "If a role is mentioned with a different alias in the text, normalize it to the provided role name when possible."
    ),
}


def build_event_summary_prompts(template=None, characters=None):
    cfg = dict(DEFAULT_EVENT_SUMMARY_TEMPLATE)
    if isinstance(template, dict):
        cfg.update({k: v for k, v in template.items() if isinstance(v, str) and v.strip()})
    role_text = ', '.join(characters or []) or 'none'
    return {
        'system': cfg['system'],
        'chunk_user': cfg['chunk_user'],
        'reduce_user': cfg['reduce_user'],
        'role_constraint': cfg['role_constraint'].format(roles=role_text),
    }


def cluster_scenes_to_events(scenes, threshold=0.65):
    # Robust clustering with optional HDBSCAN, then temporal agglomerative fallback.
    texts = []
    for s in scenes:
        t = scene_text_for_event(s, include_summary=False)
        texts.append(t)
    embs = embed_texts(texts)
    n = len(embs)
    if n == 0:
        return []

    labels = None
    if hdbscan is not None:
        try:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, n // 20), metric='euclidean')
            labels = clusterer.fit_predict(embs)
        except Exception:
            labels = None

    if labels is None and AgglomerativeClustering is not None:
        try:
            connectivity = None
            if kneighbors_graph is not None:
                try:
                    connectivity = kneighbors_graph(embs, n_neighbors=min(2, max(1, n - 1)), include_self=False)
                except Exception:
                    connectivity = None
            est_k = max(2, int(max(2, np.sqrt(n))))
            agg = AgglomerativeClustering(n_clusters=est_k, connectivity=connectivity)
            labels = agg.fit_predict(embs)
        except Exception:
            labels = None

    if labels is None:
        labels = np.zeros(n, dtype=int)
        cur = 0
        labels[0] = cur
        for i in range(1, n):
            sim = float(np.dot(embs[i - 1], embs[i]))
            if sim >= threshold:
                labels[i] = cur
            else:
                cur += 1
                labels[i] = cur

    # Convert clustering labels into contiguous scene runs so event boundaries remain chronological.
    event_objs = []
    start = 0
    cur_lab = labels[0]
    e_id = 1
    for i in range(1, n):
        if labels[i] != cur_lab:
            idx_list = list(range(start, i))
            roles = []
            for j in idx_list:
                roles.extend(scenes[j].get('roles', []))
            chunk_texts = [texts[j] for j in idx_list]
            chunk_embs = embs[idx_list]
            centroid = np.mean(chunk_embs, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
                confidences = np.dot(chunk_embs, centroid)
                confidence = float(np.clip(np.mean(confidences), 0.0, 1.0))
            else:
                confidence = 0.0
            event_objs.append({
                'event_id': e_id,
                'scene_indices': idx_list,
                'start_scene': idx_list[0],
                'end_scene': idx_list[-1],
                'event_start_time': scenes[idx_list[0]].get('start_time', scenes[idx_list[0]].get('start', idx_list[0])),
                'event_end_time': scenes[idx_list[-1]].get('end_time', scenes[idx_list[-1]].get('end', idx_list[-1])),
                'confidence': confidence,
                'raw_summaries': chunk_texts,
                'characters': sorted(list(set(roles)))
            })
            e_id += 1
            start = i
            cur_lab = labels[i]
    idx_list = list(range(start, n))
    roles = []
    for j in idx_list:
        roles.extend(scenes[j].get('roles', []))
    chunk_embs = embs[idx_list]
    centroid = np.mean(chunk_embs, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm
        confidences = np.dot(chunk_embs, centroid)
        confidence = float(np.clip(np.mean(confidences), 0.0, 1.0))
    else:
        confidence = 0.0
    event_objs.append({
        'event_id': e_id,
        'scene_indices': idx_list,
        'start_scene': idx_list[0],
        'end_scene': idx_list[-1],
        'event_start_time': scenes[idx_list[0]].get('start_time', scenes[idx_list[0]].get('start', idx_list[0])),
        'event_end_time': scenes[idx_list[-1]].get('end_time', scenes[idx_list[-1]].get('end', idx_list[-1])),
        'confidence': confidence,
        'raw_summaries': [texts[j] for j in idx_list],
        'characters': sorted(list(set(roles)))
    })
    return event_objs


def summarize_event(event, scenes, max_sentences=2, prompt_template=None):
    # MapReduce summarization with constrained LLM-first logic and extractive fallback.
    prompts = build_event_summary_prompts(prompt_template, event.get('characters', []))

    def llm_summarize(text, max_tokens=200, system_prompt=None):
        if openai is not None and os.environ.get('OPENAI_API_KEY'):
            try:
                resp = openai.ChatCompletion.create(
                    model=os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'),
                    messages=[
                        {'role': 'system', 'content': system_prompt or prompts['system']},
                        {'role': 'user', 'content': text},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.2,
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                pass
        use_transformers_summarizer = os.environ.get('USE_TRANSFORMERS_SUMMARIZER', '').strip().lower() in {'1', 'true', 'yes'}
        if use_transformers_summarizer and pipeline is not None:
            try:
                summarizer = pipeline('summarization')
                out = summarizer(text, max_length=max(64, min(400, max_tokens)), truncation=True)
                return out[0]['summary_text'].strip()
            except Exception:
                pass
        return None

    parts = []
    for idx in event['scene_indices']:
        s = scenes[idx]
        t = scene_text_for_event(s, include_summary=False)
        parts.append(t)
    combined = ' '.join(parts)
    if not combined.strip():
        return {'event_summary': combined, 'entities': event['characters'], 'duration': [event['start_scene'], event['end_scene']]}

    # Map stage: summarize chunks of scenes.
    chunks = []
    current_chunk = ''
    max_chunk_chars = 2000
    for part in parts:
        if current_chunk and len(current_chunk) + len(part) + 1 > max_chunk_chars:
            chunks.append(current_chunk)
            current_chunk = part
        else:
            current_chunk = (current_chunk + ' ' + part).strip()
    if current_chunk:
        chunks.append(current_chunk)

    chunk_summaries = []
    for chunk in chunks:
        prompt = prompts['chunk_user'].format(chunk=chunk)
        summary = llm_summarize(
            prompts['role_constraint'] + '\n\n' + prompt,
            max_tokens=180,
            system_prompt=prompts['system'],
        )
        if summary is None:
            cand_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk) if s.strip()]
            if cand_sentences:
                emb_sent = embed_texts(cand_sentences)
                centroid = np.mean(emb_sent, axis=0)
                scores = np.dot(emb_sent, centroid)
                top_idx = list(np.argsort(-scores)[:max(1, min(max_sentences, len(cand_sentences)))])
                summary = ' '.join([cand_sentences[i] for i in top_idx])
            else:
                summary = chunk[:300]
        chunk_summaries.append(summary)

    # Reduce stage: compress chunk summaries into a final event summary.
    reduce_text = ' '.join(chunk_summaries)
    final = llm_summarize(
        prompts['role_constraint'] + '\n\n' + prompts['reduce_user'].format(chunk_summaries=reduce_text),
        max_tokens=220,
        system_prompt=prompts['system'],
    )
    if final is None:
        cand_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', reduce_text) if s.strip()]
        if cand_sentences:
            emb_sent = embed_texts(cand_sentences)
            centroid = np.mean(emb_sent, axis=0)
            scores = np.dot(emb_sent, centroid)
            top_idx = list(np.argsort(-scores)[:max(1, min(3, len(cand_sentences)))])
            final = ' '.join([cand_sentences[i] for i in top_idx])
        else:
            final = reduce_text[:300]

    arc = 'middle'
    txt = final.lower()
    if any(k in txt for k in ['introduce', 'meet', 'arrive', 'introduces']):
        arc = 'setup'
    if any(k in txt for k in ['conflict', 'fight', 'argue', 'tension', 'threat', 'attack']):
        arc = 'conflict'
    if any(k in txt for k in ['reveal', 'discover', 'realize', 'revelation']):
        arc = 'revelation'
    if any(k in txt for k in ['resolve', 'reconcile', 'solve', 'end', 'finally']):
        arc = 'resolution'

    return {
        'event_summary': final,
        'entities': event['characters'],
        'arc_state': arc,
        'duration': [event['start_scene'], event['end_scene']],
        'event_start_time': event.get('event_start_time'),
        'event_end_time': event.get('event_end_time'),
        'confidence': event.get('confidence', 0.0),
    }


def build_beats_from_events(events, min_beats=3, max_beats=12):
    if not events:
        return []
    texts = [e.get('event_summary', ' ') for e in events]
    embs = embed_texts(texts)
    n_events = len(events)
    # choose k based on number of events
    k = max(min_beats, min(max_beats, int(np.clip(n_events / max(1, n_events//3), min_beats, max_beats))))
    if KMeans is not None and n_events >= k:
        try:
            km = KMeans(n_clusters=k, random_state=0).fit(embs)
            labels = km.labels_
        except Exception:
            labels = np.zeros(n_events, dtype=int)
    else:
        # simple grouping by sequence
        labels = np.zeros(n_events, dtype=int)
        step = max(1, n_events // k)
        for i in range(n_events):
            labels[i] = min(k-1, i // step)

    beats = []
    for b in range(int(labels.max())+1):
        idxs = [i for i, lab in enumerate(labels) if lab == b]
        if not idxs:
            continue
        beat_events = [events[i] for i in idxs]
        summary = ' '.join([e.get('event_summary','') for e in beat_events])
        chars = sorted(list({c for e in beat_events for c in e.get('entities', [])}))
        # heuristic function mapping
        arc_states = [e.get('arc_state','middle') for e in beat_events]
        # choose dominant arc
        func = Counter(arc_states).most_common(1)[0][0]
        beats.append({
            'beat_id': b+1,
            'function': func,
            'events': [e['event_id'] for e in beat_events],
            'summary': summary,
            'characters': chars,
        })
    return beats


def decode_global_script(beats):
    # Template-based renderer producing 1 sentence per beat
    script_lines = []
    for beat in beats:
        chars = ', '.join(beat.get('characters', [])[:3]) or 'various characters'
        func = beat.get('function', 'middle')
        summary = beat.get('summary', '')[:300]
        line = f"Beat {beat['beat_id']} ({func}): {summary} — Main characters: {chars}."
        script_lines.append(line)
    return script_lines


def extract_scene_roles(scene: Dict) -> List[str]:
    roles = []
    seen = set()

    def add_many(values):
        for value in values or []:
            name = str(value).strip()
            if not name or name in seen:
                continue
            seen.add(name)
            roles.append(name)

    if isinstance(scene.get("roles"), list):
        add_many(scene.get("roles"))

    if not roles and isinstance(scene.get("cast"), list):
        add_many(scene.get("cast"))

    if not roles:
        add_many(scene.get("unified_character_ids"))
        add_many(scene.get("face_character_ids"))
        add_many(scene.get("speaker_character_ids"))

    if not roles:
        add_many(scene.get("characters"))

    return roles


def build_role_index(scenes: List[Dict]) -> Dict[str, int]:
    roles = {}
    for s in scenes:
        for r in extract_scene_roles(s):
            if r not in roles:
                roles[r] = len(roles)
    return roles


def build_occurrence_matrix(scenes: List[Dict], role_idx: Dict[str, int]) -> np.ndarray:
    S = np.zeros((len(scenes), len(role_idx)), dtype=float)
    for i, s in enumerate(scenes):
        for r in extract_scene_roles(s):
            j = role_idx[r]
            S[i, j] = 1.0
    return S


def build_rolenet_from_occurrence(S: np.ndarray) -> np.ndarray:
    # Role co-occurrence: R = S^T * S, but zero diagonal
    R = S.T @ S
    np.fill_diagonal(R, 0.0)
    return R


def compute_centrality(R: np.ndarray) -> np.ndarray:
    # centrality = sum of edge weights incident to node
    return np.sum(R, axis=1)


def detect_leading_roles_by_gap(centrality: np.ndarray, max_leaders: int = 5) -> List[int]:
    if centrality.size == 0:
        return []
    idx = np.argsort(-centrality)
    sorted_vals = centrality[idx]
    if len(sorted_vals) == 1:
        return [int(idx[0])]
    diffs = sorted_vals[:-1] - sorted_vals[1:]
    # pick the largest gap
    gap_idx = int(np.argmax(diffs))
    num_leaders = gap_idx + 1
    num_leaders = min(num_leaders, max_leaders)
    return [int(i) for i in idx[:num_leaders]]


def macro_communities_min_cut(R: np.ndarray, leaders: List[int], role_names: List[str]) -> Dict[int, List[str]]:
    if nx is None:
        raise RuntimeError("networkx is required for min-cut macro community assignment")
    G = nx.Graph()
    n = R.shape[0]
    for i in range(n):
        G.add_node(i)
    # add weighted edges
    for i in range(n):
        for j in range(i + 1, n):
            w = float(R[i, j])
            if w > 0:
                G.add_edge(i, j, weight=w)

    # For bilateral-like case we assume two leaders; if more, assign by max weight
    if len(leaders) == 0:
        return {}
    if len(leaders) == 1:
        return {leaders[0]: [role_names[i] for i in range(len(role_names))]}
    if len(leaders) == 2:
        s, t = leaders[0], leaders[1]
        cut_value, (S, T) = nx.minimum_cut(G, s, t, capacity='weight')
        out = {s: [role_names[i] for i in S], t: [role_names[i] for i in T]}
        return out

    # general case: assign each non-leader to the leader with which it has max weight
    out = {l: [] for l in leaders}
    for i in range(R.shape[0]):
        if i in leaders:
            out[i].append(role_names[i])
            continue
        best = None
        best_w = -1.0
        for l in leaders:
            if R[i, l] > best_w:
                best_w = R[i, l]
                best = l
        out[best].append(role_names[i])
    return out


def micro_communities_agglomerative(R: np.ndarray, role_names: List[str]) -> List[List[str]]:
    # Agglomerative merging by descending edge weight, recording dendrogram levels
    n = R.shape[0]
    if n == 0:
        return []
    # initialize each node as its own community
    communities = [{i} for i in range(n)]
    # create list of edges sorted by weight desc
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((float(R[i, j]), i, j))
    edges.sort(reverse=True)

    # record community assignments at each merge level and compute inter-community avg weight
    history = []  # list of (communities_copy, inter_avg)

    def inter_community_avg(comms: List[Set[int]]) -> float:
        # compute average weight of edges connecting nodes in different communities
        tot = 0.0
        cnt = 0
        for a in range(len(comms)):
            for b in range(a + 1, len(comms)):
                for i in comms[a]:
                    for j in comms[b]:
                        tot += R[i, j]
                        cnt += 1
        return float(tot / cnt) if cnt > 0 else 0.0

    history.append(([set(c) for c in communities], inter_community_avg(communities)))

    for w, i, j in edges:
        # find communities containing i and j
        ci = cj = None
        for idx, c in enumerate(communities):
            if i in c:
                ci = idx
            if j in c:
                cj = idx
            if ci is not None and cj is not None:
                break
        if ci == cj:
            continue
        # merge smaller into larger
        newc = communities[ci] | communities[cj]
        new_communities = [communities[k] for k in range(len(communities)) if k not in (ci, cj)] + [newc]
        communities = new_communities
        history.append(([set(c) for c in communities], inter_community_avg(communities)))

    # pick level with minimal inter-community avg (paper chooses minimal I)
    best_level = min(range(len(history)), key=lambda k: history[k][1])
    best_comms = history[best_level][0]
    return [[role_names[i] for i in sorted(list(c))] for c in best_comms]


def build_profile_vectors(R: np.ndarray) -> np.ndarray:
    # each column is profile vector for a role: normalize columns
    if R.size == 0:
        return R
    cols = R.T.copy()
    norms = np.linalg.norm(cols, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    cols = cols / norms
    return cols  # shape [num_roles, num_roles]


def scene_context_matrix(scene_roles: List[str], role_idx: Dict[str, int], profiles: np.ndarray) -> np.ndarray:
    # rows: roles in scene, cols: role profile vector dimension (num_roles)
    rows = []
    for r in scene_roles:
        if r in role_idx:
            rows.append(profiles[role_idx[r]])
    if not rows:
        return np.zeros((0, profiles.shape[1]), dtype=float)
    return np.stack(rows, axis=0)


def context_difference(A: np.ndarray, B: np.ndarray) -> float:
    # A: m x d, B: n x d. compute average dot product between rows, normalized into [0,1]
    if A.size == 0 and B.size == 0:
        return 0.0
    if A.size == 0 or B.size == 0:
        return 1.0
    sims = A @ B.T
    # normalize by number of comparisons
    avg_sim = float(np.mean(sims))
    # clamp to [-1,1]
    avg_sim = max(-1.0, min(1.0, avg_sim))
    return 1.0 - ((avg_sim + 1.0) / 2.0)  # map similarity to difference in [0,1]


def scene_role_jaccard(scene_a: List[str], scene_b: List[str]) -> float:
    set_a = {role for role in scene_a if role}
    set_b = {role for role in scene_b if role}
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def scene_role_importance(scene_roles: List[str], role_idx: Dict[str, int], centrality: np.ndarray) -> float:
    weights = [float(centrality[role_idx[role]]) for role in scene_roles if role in role_idx]
    if not weights:
        return 0.0
    return float(np.mean(weights))


def role_net_transition_scores(scenes: List[Dict], role_idx: Dict[str, int], profiles: np.ndarray, centrality: np.ndarray) -> List[float]:
    scores: List[float] = []
    for i in range(len(scenes) - 1):
        current_roles = extract_scene_roles(scenes[i])
        next_roles = extract_scene_roles(scenes[i + 1])
        current_text = scene_text_for_event(scenes[i], include_summary=False)
        next_text = scene_text_for_event(scenes[i + 1], include_summary=False)

        current_context = scene_context_matrix(current_roles, role_idx, profiles)
        next_context = scene_context_matrix(next_roles, role_idx, profiles)

        context_shift = context_difference(current_context, next_context)
        role_shift = 1.0 - scene_role_jaccard(current_roles, next_roles)
        text_embs = embed_texts([current_text, next_text])
        text_shift = 1.0 - float(np.dot(text_embs[0], text_embs[1])) if len(text_embs) >= 2 else 0.0

        current_importance = scene_role_importance(current_roles, role_idx, centrality)
        next_importance = scene_role_importance(next_roles, role_idx, centrality)
        importance_shift = abs(current_importance - next_importance)
        max_importance = float(np.max(centrality)) if centrality.size else 1.0
        if max_importance > 0:
            importance_shift = min(1.0, importance_shift / max_importance)

        score = (0.35 * context_shift) + (0.25 * role_shift) + (0.25 * text_shift) + (0.15 * importance_shift)
        scores.append(float(np.clip(score, 0.0, 1.0)))
    return scores


def detect_event_boundaries(diffs: List[float]) -> List[int]:
    if not diffs:
        return []
    values = np.asarray(diffs, dtype=np.float32)
    adaptive_thr = float(np.mean(values) + 0.35 * np.std(values))
    valley_based = set(storyshed(diffs))
    score_based = {idx for idx, value in enumerate(diffs) if value >= adaptive_thr}
    boundaries = sorted(valley_based | score_based)
    return boundaries


def build_event_spans(scenes: List[Dict], boundaries: List[int]) -> List[Dict]:
    if not scenes:
        return []

    event_spans: List[Dict] = []
    start_idx = 0
    event_id = 1
    boundary_set = set(boundaries)

    for idx in range(len(scenes) - 1):
        if idx not in boundary_set:
            continue
        scene_indices = list(range(start_idx, idx + 1))
        if scene_indices:
            roles = []
            for scene_idx in scene_indices:
                roles.extend(extract_scene_roles(scenes[scene_idx]))
            span_scores = []
            for left, right in zip(scene_indices[:-1], scene_indices[1:]):
                left_text = scene_text_for_event(scenes[left])
                right_text = scene_text_for_event(scenes[right])
                emb = embed_texts([left_text, right_text])
                if len(emb) >= 2:
                    span_scores.append(float(np.dot(emb[0], emb[1])))
            event_spans.append({
                'event_id': event_id,
                'scene_indices': scene_indices,
                'start_scene': scene_indices[0],
                'end_scene': scene_indices[-1],
                'event_start_time': scenes[scene_indices[0]].get('start_time', scenes[scene_indices[0]].get('start', scene_indices[0])),
                'event_end_time': scenes[scene_indices[-1]].get('end_time', scenes[scene_indices[-1]].get('end', scene_indices[-1])),
                'confidence': float(np.clip(np.mean(span_scores), 0.0, 1.0)) if span_scores else 1.0,
                'raw_summaries': [scene_text_for_event(scenes[scene_idx], include_summary=False) for scene_idx in scene_indices],
                'characters': sorted(list(set(roles))),
            })
            event_id += 1
        start_idx = idx + 1

    tail_indices = list(range(start_idx, len(scenes)))
    if tail_indices:
        roles = []
        for scene_idx in tail_indices:
            roles.extend(extract_scene_roles(scenes[scene_idx]))
        event_spans.append({
            'event_id': event_id,
            'scene_indices': tail_indices,
            'start_scene': tail_indices[0],
            'end_scene': tail_indices[-1],
            'event_start_time': scenes[tail_indices[0]].get('start_time', scenes[tail_indices[0]].get('start', tail_indices[0])),
            'event_end_time': scenes[tail_indices[-1]].get('end_time', scenes[tail_indices[-1]].get('end', tail_indices[-1])),
            'confidence': 1.0,
            'raw_summaries': [scene_text_for_event(scenes[scene_idx], include_summary=False) for scene_idx in tail_indices],
            'characters': sorted(list(set(roles))),
        })

    return event_spans


def label_event_spans(event_spans: List[Dict], scenes: List[Dict], prompt_template=None) -> List[Dict]:
    labeled_events = []
    for event in event_spans:
        labeled_event = dict(event)
        labeled_event.update(summarize_event(event, scenes, max_sentences=2, prompt_template=prompt_template))
        labeled_events.append(labeled_event)
    return labeled_events


def storyshed(diffs: List[float]) -> List[int]:
    # diffs: difference curve between adjacent scenes, length = num_scenes-1
    n = len(diffs)
    if n == 0:
        return []
    # find peaks and valleys
    peaks = set()
    valleys = set()
    for i in range(n):
        left = diffs[i - 1] if i > 0 else diffs[i]
        right = diffs[i + 1] if i + 1 < n else diffs[i]
        if diffs[i] >= left and diffs[i] >= right and diffs[i] > 0:
            peaks.add(i)
        if diffs[i] <= left and diffs[i] <= right:
            valleys.add(i)

    boundaries = set()
    peak_vals = [diffs[i] for i in peaks] if peaks else [0.0]
    global_thr = float(np.mean(peak_vals))

    for v in valleys:
        # find nearest left peak
        left_peaks = [p for p in peaks if p < v]
        right_peaks = [p for p in peaks if p > v]
        if not left_peaks or not right_peaks:
            continue
        lp = max(left_peaks)
        rp = min(right_peaks)
        # water fill level = min(max(diffs[lp], diffs[rp]), global_thr)
        level = min(max(diffs[lp], diffs[rp]), global_thr)
        for i in range(lp + 1, rp + 1):
            if diffs[i] >= level:
                boundaries.add(i)

    # also apply global threshold: any diff >= global_thr is boundary
    for i, val in enumerate(diffs):
        if val >= global_thr:
            boundaries.add(i)

    return sorted(boundaries)


def run_pipeline(scenes: List[Dict], out_path: str):
    if isinstance(scenes, dict):
        scenes = scenes.get('scenes', scenes if isinstance(scenes.get('scenes', None), list) else [])

    role_idx = build_role_index(scenes)
    role_names = [None] * len(role_idx)
    for k, v in role_idx.items():
        role_names[v] = k

    S = build_occurrence_matrix(scenes, role_idx)
    R = build_rolenet_from_occurrence(S)
    centrality = compute_centrality(R)
    leaders = detect_leading_roles_by_gap(centrality)

    macro = macro_communities_min_cut(R, leaders, role_names) if len(leaders) > 0 and nx is not None else {}
    micro = micro_communities_agglomerative(R, role_names)

    profiles = build_profile_vectors(R)
    # compute diffs between consecutive scenes using RoleNet structure shifts
    diffs = role_net_transition_scores(scenes, role_idx, profiles, centrality)
    boundaries = detect_event_boundaries(diffs)

    # ---------- RoleNet event extraction ----------
    for s in scenes:
        if 'summary' not in s:
            s['summary'] = scene_text_for_event(s, include_summary=False) or (' '.join(extract_scene_roles(s)))

    event_spans = build_event_spans(scenes, boundaries)
    event_summaries = label_event_spans(event_spans, scenes)

    # build beats from events (optional higher-level compression)
    beats = build_beats_from_events(event_summaries)

    # decode global script
    global_script = decode_global_script(beats)

    out = {
        "roles": role_names,
        "R": R.tolist(),
        "centrality": centrality.tolist(),
        "leading_role_indices": leaders,
        "macro_communities": {str(k): v for k, v in macro.items()},
        "micro_communities": micro,
        "story_diffs": diffs,
        "story_boundaries": boundaries,
        "event_spans": event_spans,
        "events": event_summaries,
        "beats": beats,
        "global_script": global_script,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="identity_reconciliation.json")
    parser.add_argument("--out", default="rolenet_out.json")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Try to convert if it's identity_reconciliation format
    scenes = convert_identity_to_scenes(data)
    if scenes is None:
        # assume it's already a scene list
        scenes = data
        if isinstance(scenes, dict) and "scenes" in scenes:
            scenes = scenes["scenes"]

    run_pipeline(scenes, args.out)
    print(f"Wrote results to {args.out}")


if __name__ == "__main__":
    main()
