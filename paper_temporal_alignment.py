#!/usr/bin/env python3
"""Paper-inspired movie/script alignment.

This script extracts a sequential script-name stream and a sequential face-cluster
stream, then aligns them with a monotonic HMM-style dynamic program inspired by
"Aligning Movies with Scripts by Exploiting Temporal Ordering Constraints".

The implementation is intentionally self-contained and practical for this repo:
- script chunks are built from a structured screenplay JSON or a plain-text script
- movie bins are built from face clusters in a WhisperX JSON export
- optional diarization is consumed as speaker diagnostics and speaker/face overlap
- the alignment is monotonic and can be refined with a lightweight EM loop
- both an IBM Model 1 style emission model and an HMM baseline are exported

Usage examples:
  python paper_temporal_alignment.py --script script_structured.json --video-json whisperx_results.json --out paper_alignment.json
  python paper_temporal_alignment.py --script generated_scripts.json --video-json whisperx_results.json --bin-seconds 2.0 --iters 8
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf8") as handle:
        return json.load(handle)


def dump_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def canonical_name(name: Any) -> str:
    raw = normalize_text(name)
    if not raw:
        return ""
    raw = raw.replace(" :", ":").strip()
    raw = re.sub(r"\s*\(.*?\)\s*$", "", raw)
    return raw.upper()


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


@dataclass
class ScriptChunk:
    index: int
    names: List[str]
    text: str
    source: str


@dataclass
class MovieBin:
    index: int
    start: float
    end: float
    clusters: List[int]
    speakers: List[str]


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        cleaned = canonical_name(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered


def parse_structured_script(data: Dict[str, Any]) -> List[ScriptChunk]:
    scenes = data.get("scenes", [])
    chunks: List[ScriptChunk] = []
    for scene_index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            continue

        names: List[str] = []
        for block in scene.get("dialogue_blocks", []) or []:
            if isinstance(block, dict):
                names.append(canonical_name(block.get("speaker")))
        for name in scene.get("characters", []) or []:
            names.append(canonical_name(name))
        names = unique_preserve_order(names)

        text_parts: List[str] = []
        heading = normalize_text(scene.get("heading"))
        if heading:
            text_parts.append(heading)
        for block in scene.get("dialogue_blocks", []) or []:
            if not isinstance(block, dict):
                continue
            speaker = canonical_name(block.get("speaker"))
            line = normalize_text(block.get("text"))
            if speaker and line:
                text_parts.append(f"{speaker}: {line}")
            elif line:
                text_parts.append(line)
        for action in scene.get("action_blocks", []) or []:
            action_text = normalize_text(action)
            if action_text:
                text_parts.append(action_text)

        chunks.append(
            ScriptChunk(
                index=scene_index,
                names=names,
                text=" ".join(text_parts).strip(),
                source="structured_scene",
            )
        )

    if chunks:
        return chunks

    root_names = [canonical_name(name) for name in data.get("characters", []) or []]
    if root_names:
        return [ScriptChunk(index=i, names=[name], text=name, source="root_character_list") for i, name in enumerate(root_names)]

    return []


def parse_plaintext_script(text: str) -> List[ScriptChunk]:
    chunks: List[ScriptChunk] = []
    current_names: List[str] = []
    current_lines: List[str] = []
    line_index = 0

    def flush() -> None:
        nonlocal current_names, current_lines, line_index
        if not current_names and not current_lines:
            return
        chunks.append(
            ScriptChunk(
                index=len(chunks),
                names=unique_preserve_order(current_names),
                text=" ".join(current_lines).strip(),
                source=f"line_{line_index}",
            )
        )
        current_names = []
        current_lines = []

    for raw_line in text.splitlines():
        line_index += 1
        line = raw_line.strip()
        if not line:
            flush()
            continue

        speaker = canonical_name(line)
        if speaker and len(speaker) <= 40 and speaker == speaker.replace(" ", "") and speaker.isupper():
            flush()
            current_names.append(speaker)
            continue

        current_lines.append(line)

    flush()
    return chunks


def load_script_chunks(path: str | Path) -> List[ScriptChunk]:
    data = load_json(path)
    if isinstance(data, dict):
        chunks = parse_structured_script(data)
        if chunks:
            return chunks
        if "text" in data:
            return parse_plaintext_script(normalize_text(data.get("text")))
    if isinstance(data, list):
        chunks: List[ScriptChunk] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            names = unique_preserve_order(item.get("names") or item.get("characters") or [])
            text = normalize_text(item.get("text") or item.get("dialogue") or item.get("action"))
            chunks.append(
                ScriptChunk(
                    index=idx,
                    names=names,
                    text=text,
                    source=item.get("source", "list_item"),
                )
            )
        if chunks:
            return chunks
    return []


def load_face_segments(path: str | Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("video-json must be a dictionary produced by whisperx_character_id.py")

    faces = data.get("face_segments", []) or []
    diar = data.get("diarization", []) or []
    face_segments: List[Dict[str, Any]] = []
    for item in faces:
        if not isinstance(item, dict):
            continue
        cluster = item.get("cluster")
        try:
            cluster_id = int(cluster)
        except Exception:
            continue
        face_segments.append(
            {
                "cluster": cluster_id,
                "start": float(item.get("start", 0.0)),
                "end": float(item.get("end", 0.0)),
            }
        )

    diar_segments: List[Dict[str, Any]] = []
    for item in diar:
        if not isinstance(item, dict):
            continue
        speaker = normalize_text(item.get("speaker"))
        if not speaker:
            continue
        diar_segments.append(
            {
                "speaker": speaker,
                "start": float(item.get("start", 0.0)),
                "end": float(item.get("end", 0.0)),
            }
        )

    return face_segments, diar_segments


def build_movie_bins(face_segments: List[Dict[str, Any]], bin_seconds: float = 2.0) -> List[MovieBin]:
    if not face_segments:
        return []

    max_end = max(float(seg["end"]) for seg in face_segments)
    if max_end <= 0:
        return []

    bins: List[MovieBin] = []
    num_bins = max(1, int(math.ceil(max_end / float(bin_seconds))))
    for idx in range(num_bins):
        start = idx * float(bin_seconds)
        end = min(max_end, (idx + 1) * float(bin_seconds))
        clusters = []
        for seg in face_segments:
            if overlap_seconds(start, end, float(seg["start"]), float(seg["end"])) > 0:
                clusters.append(int(seg["cluster"]))
        bins.append(MovieBin(index=idx, start=start, end=end, clusters=sorted(set(clusters)), speakers=[]))
    return bins


def build_speaker_bins(diar_segments: List[Dict[str, Any]], bin_seconds: float, total_end: float) -> List[List[str]]:
    if not diar_segments or total_end <= 0:
        return []
    num_bins = max(1, int(math.ceil(total_end / float(bin_seconds))))
    bins: List[List[str]] = [[] for _ in range(num_bins)]
    for idx in range(num_bins):
        start = idx * float(bin_seconds)
        end = min(total_end, (idx + 1) * float(bin_seconds))
        speakers = []
        for seg in diar_segments:
            if overlap_seconds(start, end, float(seg["start"]), float(seg["end"])) > 0:
                speakers.append(normalize_text(seg["speaker"]))
        bins[idx] = sorted(set([sp for sp in speakers if sp]))
    return bins


def build_initial_emission_model(script_chunks: List[ScriptChunk], movie_bins: List[MovieBin]) -> Dict[str, Dict[int, float]]:
    names = sorted({name for chunk in script_chunks for name in chunk.names})
    clusters = sorted({cluster for bin_ in movie_bins for cluster in bin_.clusters})
    if not names or not clusters:
        return {}

    model: Dict[str, Dict[int, float]] = {}
    uniform = 1.0 / float(len(clusters))
    for name in names:
        model[name] = {cluster: uniform for cluster in clusters}
    return model


def emission_logprob(script_names: Sequence[str], face_clusters: Sequence[int], model: Dict[str, Dict[int, float]], null_penalty: float = -2.0) -> float:
    names = [name for name in script_names if name]
    clusters = [int(cluster) for cluster in face_clusters]

    if not names and not clusters:
        return float(null_penalty)
    if not names:
        return float(null_penalty) - 0.35 * float(len(clusters))
    if not clusters:
        return float(null_penalty) - 0.35 * float(len(names))

    score = 0.0
    for cluster in clusters:
        p = 0.0
        for name in names:
            p += float(model.get(name, {}).get(cluster, 1e-12))
        p /= max(1, len(names))
        score += math.log(max(p, 1e-12))

    score += 0.1 * min(len(names), len(clusters))
    return score


def greedy_monotonic_emission_align(script_chunks: List[ScriptChunk], movie_bins: List[MovieBin], model: Dict[str, Dict[int, float]]) -> List[int]:
    if not script_chunks or not movie_bins:
        return []

    path: List[int] = []
    last_index = 0
    for movie_bin in movie_bins:
        best_index = None
        best_score = -1e18
        for script_index in range(last_index, len(script_chunks)):
            score = emission_logprob(script_chunks[script_index].names, movie_bin.clusters, model)
            if score > best_score:
                best_score = score
                best_index = script_index
        if best_index is None:
            best_index = last_index
        path.append(int(best_index))
        last_index = int(best_index)
    return path


def viterbi_hmm_align(script_chunks: List[ScriptChunk], movie_bins: List[MovieBin], model: Dict[str, Dict[int, float]], stay_bonus: float = 0.0, advance_penalty: float = 0.25) -> List[int]:
    if not script_chunks or not movie_bins:
        return []

    m_count = len(script_chunks)
    n_count = len(movie_bins)
    neg_inf = -1e18

    dp = [[neg_inf for _ in range(m_count)] for _ in range(n_count)]
    back = [[0 for _ in range(m_count)] for _ in range(n_count)]

    for m in range(m_count):
        start_penalty = -0.35 * float(m)
        dp[0][m] = start_penalty + emission_logprob(script_chunks[m].names, movie_bins[0].clusters, model)
        back[0][m] = m

    for n in range(1, n_count):
        for m in range(m_count):
            stay_score = dp[n - 1][m] + stay_bonus
            best_prev = m
            best_score = stay_score

            if m > 0:
                adv_score = dp[n - 1][m - 1] - advance_penalty
                if adv_score > best_score:
                    best_score = adv_score
                    best_prev = m - 1

            dp[n][m] = best_score + emission_logprob(script_chunks[m].names, movie_bins[n].clusters, model)
            back[n][m] = best_prev

    last_m = max(range(m_count), key=lambda idx: dp[-1][idx])
    path = [last_m]
    for n in range(n_count - 1, 0, -1):
        last_m = back[n][last_m]
        path.append(last_m)
    path.reverse()
    return path


def update_emission_model_from_path(script_chunks: List[ScriptChunk], movie_bins: List[MovieBin], path: List[int], smoothing: float = 0.5) -> Dict[str, Dict[int, float]]:
    names = sorted({name for chunk in script_chunks for name in chunk.names})
    clusters = sorted({cluster for bin_ in movie_bins for cluster in bin_.clusters})
    if not names or not clusters:
        return {}

    counts: Dict[str, Counter[int]] = {name: Counter() for name in names}
    for movie_bin, script_index in zip(movie_bins, path):
        if script_index < 0 or script_index >= len(script_chunks):
            continue
        script_names = script_chunks[script_index].names
        face_clusters = movie_bin.clusters
        if not script_names or not face_clusters:
            continue
        weight = 1.0 / float(len(script_names) * len(face_clusters))
        for name in script_names:
            for cluster in face_clusters:
                counts[name][int(cluster)] += weight

    model: Dict[str, Dict[int, float]] = {}
    for name in names:
        total = sum(counts[name].values()) + smoothing * float(len(clusters))
        model[name] = {}
        for cluster in clusters:
            model[name][cluster] = float((counts[name][cluster] + smoothing) / max(total, 1e-12))
    return model


def fit_ibm1_alignment(script_chunks: List[ScriptChunk], movie_bins: List[MovieBin], iterations: int = 8, smoothing: float = 0.5) -> Tuple[List[int], Dict[str, Dict[int, float]]]:
    model = build_initial_emission_model(script_chunks, movie_bins)
    if not model:
        return [], {}

    path: List[int] = []
    for _ in range(max(1, iterations)):
        path = greedy_monotonic_emission_align(script_chunks, movie_bins, model)
        model = update_emission_model_from_path(script_chunks, movie_bins, path, smoothing=smoothing)
        if not model:
            break
    return path, model


def fit_hmm_alignment(script_chunks: List[ScriptChunk], movie_bins: List[MovieBin], model: Dict[str, Dict[int, float]]) -> List[int]:
    return viterbi_hmm_align(script_chunks, movie_bins, model)


def summarize_alignment(script_chunks: List[ScriptChunk], movie_bins: List[MovieBin], path: List[int], model: Dict[str, Dict[int, float]]) -> Dict[str, Any]:
    cluster_counts: DefaultDict[str, Counter[int]] = defaultdict(Counter)
    for movie_bin, script_index in zip(movie_bins, path):
        if script_index < 0 or script_index >= len(script_chunks):
            continue
        names = script_chunks[script_index].names
        for name in names:
            for cluster in movie_bin.clusters:
                cluster_counts[name][cluster] += 1

    name_to_cluster: Dict[str, Optional[int]] = {}
    cluster_to_name: Dict[str, Optional[str]] = {}
    for name, counter in cluster_counts.items():
        best_cluster = counter.most_common(1)
        name_to_cluster[name] = int(best_cluster[0][0]) if best_cluster else None

    inverse: DefaultDict[int, Counter[str]] = defaultdict(Counter)
    for name, cluster in name_to_cluster.items():
        if cluster is None:
            continue
        inverse[int(cluster)][name] += 1
    for cluster, counter in inverse.items():
        best_name = counter.most_common(1)
        cluster_to_name[str(cluster)] = best_name[0][0] if best_name else None

    alignment_rows: List[Dict[str, Any]] = []
    for movie_bin, script_index in zip(movie_bins, path):
        script_chunk = script_chunks[script_index] if 0 <= script_index < len(script_chunks) else None
        alignment_rows.append(
            {
                "movie_bin": movie_bin.index,
                "movie_start": float(movie_bin.start),
                "movie_end": float(movie_bin.end),
                "script_index": int(script_index),
                "script_names": script_chunk.names if script_chunk else [],
                "movie_clusters": list(movie_bin.clusters),
                "score": emission_logprob(script_chunk.names if script_chunk else [], movie_bin.clusters, model) if script_chunk else None,
            }
        )

    return {
        "alignment": alignment_rows,
        "name_to_face_cluster": name_to_cluster,
        "face_cluster_to_name": cluster_to_name,
        "cluster_counts": {name: dict(counter) for name, counter in cluster_counts.items()},
    }


def _html_escape(text: Any) -> str:
        raw = normalize_text(text)
        return (
                raw.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
        )


def _chip_list(values: Sequence[Any], kind: str) -> str:
        if not values:
                return '<span class="muted">none</span>'
        chips = []
        for value in values:
                chips.append(f'<span class="chip {kind}">{_html_escape(value)}</span>')
        return "".join(chips)


def _render_alignment_table(title: str, summary: Dict[str, Any], script_chunks: List[ScriptChunk], total_bins: int) -> str:
        rows = summary.get("alignment", []) or []
        if not rows:
                return f'<section class="panel"><h2>{_html_escape(title)}</h2><p class="muted">No alignment rows available.</p></section>'

        max_script_index = max((int(row.get("script_index", 0)) for row in rows), default=0)
        body_rows: List[str] = []
        for row in rows:
                script_index = int(row.get("script_index", 0))
                script_chunk = script_chunks[script_index] if 0 <= script_index < len(script_chunks) else None
                movie_index = int(row.get("movie_bin", 0))
                script_progress = (script_index + 1) / float(max(1, len(script_chunks)))
                movie_progress = (movie_index + 1) / float(max(1, total_bins))
                bar_left = movie_progress * 100.0
                script_left = script_progress * 100.0
                movie_clusters = row.get("movie_clusters", []) or []
                script_names = row.get("script_names", []) or []
                score = row.get("score")
                score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "-"

                body_rows.append(
                        "<tr>"
                        f'<td class="mono">{movie_index}</td>'
                        f'<td class="mono">{_html_escape(row.get("movie_start", 0.0))}s - {_html_escape(row.get("movie_end", 0.0))}s</td>'
                        f'<td class="mono">{script_index}</td>'
                        f'<td>{_chip_list(script_names, "script")}</td>'
                        f'<td>{_chip_list(movie_clusters, "movie")}</td>'
                        f'<td class="mono">{score_text}</td>'
                        "</tr>"
                        "<tr class=\"mini-row\">"
                        f'<td colspan="6">'
                        f'<div class="mini-track"><span class="mini-label">movie</span><span class="mini-bar"><span class="mini-cursor movie-cursor" style="left:{bar_left:.2f}%"></span></span><span class="mini-label">script</span><span class="mini-bar"><span class="mini-cursor script-cursor" style="left:{script_left:.2f}%"></span></span></div>'
                        f'<div class="mini-note">{_html_escape(script_chunk.text[:220] if script_chunk else "")}</div>'
                        "</td>"
                        "</tr>"
                )

        return (
                f'<section class="panel">'
                f'<h2>{_html_escape(title)}</h2>'
                f'<div class="mini-meta">{len(rows)} movie bins, {len(script_chunks)} script chunks, {max_script_index + 1 if rows else 0} max script index in path</div>'
                '<table class="alignment-table">'
                '<thead><tr><th>Movie Bin</th><th>Movie Time</th><th>Script Chunk</th><th>Script Names</th><th>Face Clusters</th><th>Score</th></tr></thead>'
                f'<tbody>{"".join(body_rows)}</tbody>'
                '</table>'
                '</section>'
        )


def render_alignment_html(payload: Dict[str, Any]) -> str:
        script_chunks = payload.get("script_chunks", []) or []
        movie_bins = payload.get("movie_bins", []) or []
        metadata = payload.get("metadata", {}) or {}
        speaker_to_face = payload.get("speaker_to_face_cluster", {}) or {}
        speaker_bins = payload.get("speaker_bins", []) or []
        ibm1_summary = payload.get("ibm1_alignment_summary", {}) or {}
        hmm_summary = payload.get("hmm_alignment_summary", {}) or {}

        speaker_rows: List[str] = []
        for speaker, cluster in sorted(speaker_to_face.items(), key=lambda kv: str(kv[0])):
                speaker_rows.append(f'<tr><td class="mono">{_html_escape(speaker)}</td><td class="mono">{_html_escape(cluster)}</td></tr>')

        bin_rows: List[str] = []
        for idx, speakers in enumerate(speaker_bins[: min(len(speaker_bins), 80)]):
                bin_rows.append(
                        f'<tr><td class="mono">{idx}</td><td>{_chip_list(speakers, "movie")}</td></tr>'
                )

        script_preview_rows: List[str] = []
        for chunk in script_chunks[: min(len(script_chunks), 80)]:
                script_preview_rows.append(
                        f'<tr><td class="mono">{chunk.get("index", 0)}</td><td>{_chip_list(chunk.get("names", []), "script")}</td><td>{_html_escape(chunk.get("source", ""))}</td><td>{_html_escape(chunk.get("text", "")[:120])}</td></tr>'
                )

        total_bins = len(movie_bins)
        html = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Paper-inspired alignment report</title>
    <style>
        :root {{
            --bg: #0f1115;
            --panel: #171b22;
            --panel-alt: #1c212a;
            --text: #e7edf5;
            --muted: #93a4b7;
            --line: #2a3240;
            --script: #64d2ff;
            --movie: #f7c948;
            --accent: #7dd3fc;
            --good: #6ee7b7;
        }}
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; background: radial-gradient(circle at top, #17202d 0, var(--bg) 42%); color: var(--text); font-family: Segoe UI, Arial, sans-serif; }}
        header {{ padding: 28px 32px 10px; border-bottom: 1px solid var(--line); }}
        h1 {{ margin: 0 0 8px; font-size: 28px; }}
        h2 {{ margin: 0 0 10px; font-size: 20px; }}
        .sub {{ color: var(--muted); font-size: 13px; line-height: 1.5; }}
        .wrap {{ padding: 24px 32px 40px; display: grid; gap: 18px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
        .stat {{ background: linear-gradient(180deg, var(--panel), var(--panel-alt)); border: 1px solid var(--line); border-radius: 16px; padding: 14px 16px; }}
        .stat .k {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
        .stat .v {{ font-size: 24px; margin-top: 6px; }}
        .panel {{ background: linear-gradient(180deg, var(--panel), var(--panel-alt)); border: 1px solid var(--line); border-radius: 18px; padding: 16px; box-shadow: 0 12px 30px rgba(0,0,0,.2); }}
        .panel-grid {{ display: grid; grid-template-columns: 1.4fr .8fr; gap: 16px; }}
        .alignment-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        .alignment-table th, .alignment-table td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; vertical-align: top; }}
        .alignment-table th {{ text-align: left; color: var(--muted); font-weight: 600; background: rgba(255,255,255,.02); }}
        .mini-row td {{ padding-top: 0; padding-bottom: 14px; background: rgba(255,255,255,.015); }}
        .mini-track {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
        .mini-label {{ color: var(--muted); font-size: 11px; width: 42px; text-transform: uppercase; letter-spacing: .08em; }}
        .mini-bar {{ position: relative; flex: 1; height: 10px; border-radius: 999px; background: rgba(255,255,255,.06); overflow: hidden; border: 1px solid rgba(255,255,255,.06); }}
        .mini-cursor {{ position: absolute; top: -2px; width: 4px; height: 14px; border-radius: 999px; }}
        .movie-cursor {{ background: var(--movie); }}
        .script-cursor {{ background: var(--script); }}
        .mini-note {{ color: var(--muted); font-size: 12px; line-height: 1.45; max-height: 44px; overflow: hidden; text-overflow: ellipsis; }}
        .chip {{ display: inline-block; padding: 3px 8px; margin: 0 6px 4px 0; border-radius: 999px; font-size: 12px; border: 1px solid transparent; }}
        .chip.script {{ background: rgba(100,210,255,.14); color: #b7ecff; border-color: rgba(100,210,255,.25); }}
        .chip.movie {{ background: rgba(247,201,72,.14); color: #ffe7a6; border-color: rgba(247,201,72,.25); }}
        .muted {{ color: var(--muted); }}
        .mono {{ font-family: Consolas, monospace; }}
        table.small {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        table.small th, table.small td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; vertical-align: top; }}
        .section-title {{ margin: 0 0 10px; }}
        .stack {{ display: grid; gap: 12px; }}
        @media (max-width: 1100px) {{
            .panel-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>Paper-inspired alignment report</h1>
        <div class="sub">This page shows the monotonic relation between script chunks and movie bins. IBM1 and HMM use the same input, so you can compare how the path changes under emission learning vs. pure sequence decoding.</div>
    </header>
    <main class="wrap">
        <section class="stats">
            <div class="stat"><div class="k">Script chunks</div><div class="v">{metadata.get("script_chunks", 0)}</div></div>
            <div class="stat"><div class="k">Movie bins</div><div class="v">{metadata.get("movie_bins", 0)}</div></div>
            <div class="stat"><div class="k">Face segments</div><div class="v">{metadata.get("face_segments", 0)}</div></div>
            <div class="stat"><div class="k">Diarization segments</div><div class="v">{metadata.get("diar_segments", 0)}</div></div>
        </section>

        <section class="panel-grid">
            <section class="panel">
                <h2 class="section-title">IBM1 alignment</h2>
                {_render_alignment_table("IBM1 path", ibm1_summary, [ScriptChunk(**chunk) for chunk in script_chunks], total_bins)}
            </section>
            <section class="stack">
                <section class="panel">
                    <h2 class="section-title">HMM baseline</h2>
                    {_render_alignment_table("HMM path", hmm_summary, [ScriptChunk(**chunk) for chunk in script_chunks], total_bins)}
                </section>
                <section class="panel">
                    <h2 class="section-title">Speaker to face cluster</h2>
                    <table class="small">
                        <thead><tr><th>Speaker</th><th>Cluster</th></tr></thead>
                        <tbody>{''.join(speaker_rows) if speaker_rows else '<tr><td colspan="2" class="muted">No speaker mapping available.</td></tr>'}</tbody>
                    </table>
                </section>
            </section>
        </section>

        <section class="panel-grid">
            <section class="panel">
                <h2 class="section-title">Script chunks</h2>
                <table class="small">
                    <thead><tr><th>#</th><th>Names</th><th>Source</th><th>Text preview</th></tr></thead>
                    <tbody>{''.join(script_preview_rows) if script_preview_rows else '<tr><td colspan="4" class="muted">No script chunks available.</td></tr>'}</tbody>
                </table>
            </section>
            <section class="panel">
                <h2 class="section-title">Movie speaker bins</h2>
                <table class="small">
                    <thead><tr><th>#</th><th>Speakers in bin</th></tr></thead>
                    <tbody>{''.join(bin_rows) if bin_rows else '<tr><td colspan="2" class="muted">No diarization bins available.</td></tr>'}</tbody>
                </table>
            </section>
        </section>
    </main>
</body>
</html>"""
        return html


def assign_speakers_to_faces(diar_segments: List[Dict[str, Any]], face_segments: List[Dict[str, Any]]) -> Dict[str, int]:
    assignments: Dict[str, Counter[int]] = defaultdict(Counter)
    for seg in diar_segments:
        speaker = normalize_text(seg.get("speaker"))
        if not speaker:
            continue
        best_cluster = None
        best_overlap = 0.0
        for face in face_segments:
            overlap = overlap_seconds(float(seg["start"]), float(seg["end"]), float(face["start"]), float(face["end"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_cluster = int(face["cluster"])
        if best_cluster is not None:
            assignments[speaker][best_cluster] += best_overlap

    final: Dict[str, int] = {}
    for speaker, counter in assignments.items():
        if counter:
            final[speaker] = int(counter.most_common(1)[0][0])
    return final


def compute_actor_presence(movie_bins: List[MovieBin], face_segments: List[Dict[str, Any]]) -> Tuple[List[Dict[int, float]], List[int]]:
    """Compute Actor Presence (AP) per movie bin.
    Returns list of dicts (cluster -> fraction_of_actor_frames) and list of unique cluster ids.
    """
    clusters = sorted({int(s.get("cluster")) for s in face_segments}) if face_segments else []
    if not movie_bins or not clusters:
        return [], clusters

    ap_list: List[Dict[int, float]] = []
    for bin_ in movie_bins:
        counts: Dict[int, float] = {}
        total = 0.0
        for face in face_segments:
            cl = int(face.get("cluster"))
            ov = overlap_seconds(bin_.start, bin_.end, float(face.get("start", 0.0)), float(face.get("end", 0.0)))
            if ov > 0:
                # optionally use visual_speech flag to boost (if present)
                boost = 1.0
                if face.get("visual_speech"):
                    boost = 1.5
                counts[cl] = counts.get(cl, 0.0) + ov * boost
                total += ov * boost

        if total > 0:
            for cl in list(counts.keys()):
                counts[cl] = float(counts[cl] / total)
        ap_list.append(counts)

    return ap_list, clusters


def build_ap_similarity_matrix(ap_list: List[Dict[int, float]], clusters: List[int]) -> List[List[float]]:
    """Build A = P P^T similarity matrix from AP list (sparse dicts)."""
    n = len(ap_list)
    if n == 0:
        return []
    mat: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        pi = ap_list[i]
        if not isinstance(pi, dict):
            pi = {}
        for j in range(i, n):
            pj = ap_list[j]
            if not isinstance(pj, dict):
                pj = {}
            s = 0.0
            # dot product over cluster keys
            for cl, vi in pi.items():
                vj = pj.get(cl, 0.0)
                s += vi * vj
            mat[i][j] = s
            mat[j][i] = s
    return mat


def detect_visual_speech_from_video(video_path: str | Path, face_segments: List[Dict[str, Any]], fps_sample: float = 2.0) -> List[Dict[str, Any]]:
    """Attempt a lightweight visual-speech detector that marks face_segments with 'visual_speech' boolean.
    Requirements: face_segments must include 'frame_indices' or 'bbox' and video must be readable by OpenCV.
    This is a heuristic: compute mean optical-flow magnitude inside face bbox across consecutive keyframes and mark segments
    with average motion above a threshold as visual speech. If dependencies or frame indices absent, returns face_segments unchanged.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return face_segments

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return face_segments

    # build a map frame_index -> frame
    frame_cache: Dict[int, Any] = {}
    needed_frames = set()
    for f in face_segments:
        fi = f.get("frame_indices")
        if fi:
            for idx in fi:
                needed_frames.add(int(idx))

    if not needed_frames:
        cap.release()
        return face_segments

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    max_frame = max(needed_frames)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx in needed_frames:
            frame_cache[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if idx > max_frame:
            break
    cap.release()

    # compute simple optical flow per face segment
    out_segments = []
    for f in face_segments:
        fi = f.get("frame_indices")
        if not fi or len(fi) < 2:
            f["visual_speech"] = False
            out_segments.append(f)
            continue
        mags = []
        prev = None
        for frame_idx in sorted(set(int(x) for x in fi)):
            gray = frame_cache.get(frame_idx)
            if gray is None:
                continue
            if prev is None:
                prev = gray
                continue
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mags.append(float(mag.mean()))
            prev = gray

        avg_mag = float(sum(mags) / max(1, len(mags))) if mags else 0.0
        # threshold heuristic: motion mean > 0.5 indicates possible mouth motion / visible speech
        f["visual_speech"] = avg_mag > 0.5
        out_segments.append(f)
    return out_segments


def load_audio_similarity(path: str | Path) -> Optional[List[List[float]]]:
    """Load a precomputed audio similarity matrix from JSON (list of lists)."""
    try:
        data = load_json(path)
        if isinstance(data, list) and all(isinstance(row, list) for row in data):
            return [[float(x) for x in row] for row in data]
    except Exception:
        return None
    return None


def combine_similarity_matrices(audio_sim: List[List[float]], ap_sim: List[List[float]], alpha: float = 0.8) -> List[List[float]]:
    """Combine audio similarity S and actor-presence similarity A into F = S + alpha * A (with clipping)."""
    n = len(audio_sim)
    if n == 0:
        return ap_sim
    if not ap_sim:
        return audio_sim
    F: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = float(audio_sim[i][j]) if i < len(audio_sim) and j < len(audio_sim[i]) else 0.0
            a = float(ap_sim[i][j]) if i < len(ap_sim) and j < len(ap_sim[i]) else 0.0
            F[i][j] = min(1.0, s + float(alpha) * a)
    return F


def spectral_cluster_from_similarity(sim_mat: List[List[float]], n_clusters: Optional[int] = None) -> List[int]:
    """Perform spectral clustering on similarity matrix. Falls back to Agglomerative if spectral not available.
    Returns labels per bin index.
    """
    try:
        import numpy as np
        from sklearn.cluster import SpectralClustering
        arr = np.array(sim_mat)
        if n_clusters is None:
            # heuristic: choose small number based on eigen-gap or fixed default
            n_clusters = max(2, min(12, max(2, arr.shape[0] // 100)))
        model = SpectralClustering(n_clusters=int(n_clusters), affinity='precomputed')
        labels = model.fit_predict(arr)
        return [int(x) for x in labels]
    except Exception:
        # fallback to previous cluster_bins_with_ap behavior using threshold
        return cluster_bins_with_ap(sim_mat, threshold=0.02)


def cluster_bins_with_ap(sim_mat: List[List[float]], threshold: float = 0.02) -> List[int]:
    """Simple clustering of bins using AP similarity matrix.
    If sklearn is available, use AgglomerativeClustering (n_clusters auto via threshold).
    Otherwise, build graph by thresholding sim_mat and return connected components as cluster ids.
    Returns cluster_id per bin index.
    """
    n = len(sim_mat)
    if n == 0:
        return []
    try:
        from sklearn.cluster import AgglomerativeClustering
        import numpy as np

        # convert similarity to distance
        arr = np.array(sim_mat)
        # distance = 1 - similarity (clip)
        dist = np.clip(1.0 - arr, 0.0, 1.0)
        # heuristic: try to find small number of clusters via threshold on distance
        model = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', distance_threshold=1.0 - threshold)
        labels = model.fit_predict(dist)
        return [int(x) for x in labels]
    except Exception:
        # fallback: threshold graph -> connected components
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                if sim_mat[i][j] >= threshold:
                    union(i, j)

        labels = [find(i) for i in range(n)]
        # compress labels
        comp: Dict[int, int] = {}
        next_id = 0
        out = []
        for l in labels:
            if l not in comp:
                comp[l] = next_id
                next_id += 1
            out.append(comp[l])
        return out


def aggregate_speaker_clusters_to_face(movie_bins: List[MovieBin], bin_cluster_labels: List[int]) -> Dict[int, int]:
    """Given clustering of bins (e.g., AP-based), aggregate which face cluster corresponds to each bin-cluster.
    Return mapping: bin_cluster_id -> dominant face_cluster (int).
    """
    agg: DefaultDict[int, Counter[int]] = defaultdict(Counter)
    for bin_, lbl in zip(movie_bins, bin_cluster_labels):
        for cl in bin_.clusters:
            agg[int(lbl)][int(cl)] += 1

    result: Dict[int, int] = {}
    for lbl, counter in agg.items():
        if not counter:
            continue
        best = counter.most_common(1)[0][0]
        result[int(lbl)] = int(best)
    return result


def run_pipeline(
    script_path: str | Path,
    video_json_path: str | Path,
    bin_seconds: float = 2.0,
    iterations: int = 8,
    smoothing: float = 0.5,
    use_ap: bool = False,
    ap_threshold: float = 0.02,
    audio_sim_path: Optional[str] = None,
    alpha: float = 0.8,
    use_spectral: bool = False,
    n_clusters: Optional[int] = None,
) -> Dict[str, Any]:
    script_chunks = load_script_chunks(script_path)
    if not script_chunks:
        raise ValueError(f"No script chunks could be parsed from {script_path}")

    face_segments, diar_segments = load_face_segments(video_json_path)
    movie_bins = build_movie_bins(face_segments, bin_seconds=bin_seconds)
    # Compute Actor Presence (AP) per bin and optionally cluster bins by AP
    ap_list, ap_clusters = compute_actor_presence(movie_bins, face_segments)
    ap_similarity = build_ap_similarity_matrix(ap_list, ap_clusters) if ap_list else []

    # Optionally detect visual speech (heuristic) to boost AP weights
    if use_ap and face_segments:
        try:
            face_segments = detect_visual_speech_from_video(video_json_path, face_segments)
        except Exception:
            pass

    # Load optional audio similarity matrix S
    audio_sim = load_audio_similarity(audio_sim_path) if audio_sim_path else None

    bin_cluster_labels: List[int] = []
    bincluster_to_face: Dict[int, int] = {}

    # If audio similarity is provided and AP exists, combine S and A into F and cluster
    if audio_sim is not None and ap_similarity:
        fused = combine_similarity_matrices(audio_sim, ap_similarity, alpha=alpha)
        if use_spectral:
            bin_cluster_labels = spectral_cluster_from_similarity(fused, n_clusters=n_clusters)
        else:
            # fallback: threshold graph on fused
            bin_cluster_labels = cluster_bins_with_ap(fused, threshold=ap_threshold)
        bincluster_to_face = aggregate_speaker_clusters_to_face(movie_bins, bin_cluster_labels)
        for bin_, lbl in zip(movie_bins, bin_cluster_labels):
            mapped = bincluster_to_face.get(int(lbl))
            if mapped is not None and int(mapped) not in bin_.clusters:
                bin_.clusters.append(int(mapped))
    elif use_ap and ap_similarity:
        # AP-only clustering (previous behavior)
        bin_cluster_labels = cluster_bins_with_ap(ap_similarity, threshold=ap_threshold)
        bincluster_to_face = aggregate_speaker_clusters_to_face(movie_bins, bin_cluster_labels)
        for bin_, lbl in zip(movie_bins, bin_cluster_labels):
            mapped = bincluster_to_face.get(int(lbl))
            if mapped is not None and int(mapped) not in bin_.clusters:
                bin_.clusters.append(int(mapped))
    ibm1_path, ibm1_model = fit_ibm1_alignment(script_chunks, movie_bins, iterations=iterations, smoothing=smoothing)
    hmm_path = fit_hmm_alignment(script_chunks, movie_bins, ibm1_model)
    ibm1_summary = summarize_alignment(script_chunks, movie_bins, ibm1_path, ibm1_model)
    hmm_summary = summarize_alignment(script_chunks, movie_bins, hmm_path, ibm1_model)

    total_end = max((float(seg["end"]) for seg in face_segments), default=0.0)
    speaker_bins = build_speaker_bins(diar_segments, bin_seconds, total_end) if diar_segments else []
    # compute speaker->face mapping (naive overlap) and also try to map spoken clusters via AP-derived bin clusters
    speaker_to_face = assign_speakers_to_faces(diar_segments, face_segments) if diar_segments and face_segments else {}
    # If we have bin cluster labels and aggregation, map diar speakers via the bins they overlap
    if diar_segments and bin_cluster_labels:
        # mapping from bin index to bin_cluster_label
        bin_to_label = {idx: lbl for idx, lbl in enumerate(bin_cluster_labels)}
        # for each speaker, collect labels of bins they overlap and vote
        speaker_label_counts: DefaultDict[str, Counter[int]] = defaultdict(Counter)
        for idx, bin_ in enumerate(movie_bins):
            label = bin_to_label.get(idx)
            if label is None:
                continue
            # find diar segments overlapping this bin
            for seg in diar_segments:
                if overlap_seconds(bin_.start, bin_.end, float(seg.get("start", 0.0)), float(seg.get("end", 0.0))) > 0:
                    sp = normalize_text(seg.get("speaker"))
                    if sp:
                        speaker_label_counts[sp][int(label)] += 1
        for sp, counter in speaker_label_counts.items():
            best = counter.most_common(1)
            if best:
                lbl = best[0][0]
                mapped_face = bincluster_to_face.get(int(lbl))
                if mapped_face is not None:
                    speaker_to_face[sp] = int(mapped_face)

    return {
        "metadata": {
            "script_chunks": len(script_chunks),
            "movie_bins": len(movie_bins),
            "face_segments": len(face_segments),
            "diar_segments": len(diar_segments),
            "bin_seconds": float(bin_seconds),
            "iterations": int(iterations),
            "smoothing": float(smoothing),
        },
        "script_chunks": [chunk.__dict__ for chunk in script_chunks],
        "movie_bins": [bin_.__dict__ for bin_ in movie_bins],
        "speaker_bins": speaker_bins,
        "speaker_to_face_cluster": speaker_to_face,
        "ibm1_alignment_summary": ibm1_summary,
        "hmm_alignment_summary": hmm_summary,
        "ibm1_alignment_path": ibm1_path,
        "hmm_alignment_path": hmm_path,
        "emission_model": {name: {str(cluster): prob for cluster, prob in cluster_probs.items()} for name, cluster_probs in ibm1_model.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-inspired movie/script alignment with monotonic sequence constraints")
    parser.add_argument("--script", default="script_structured.json", help="script JSON or plaintext script path")
    parser.add_argument("--video-json", default="whisperx_results.json", help="WhisperX output JSON path")
    parser.add_argument("--out", default="paper_alignment.json", help="output JSON path")
    parser.add_argument("--viz-out", default="paper_alignment.html", help="optional HTML visualization report path")
    parser.add_argument("--bin-seconds", type=float, default=2.0, help="movie bin size in seconds")
    parser.add_argument("--iters", type=int, default=8, help="EM-style refinement iterations")
    parser.add_argument("--smoothing", type=float, default=0.5, help="Laplace smoothing strength")
    parser.add_argument("--use-ap", action="store_true", help="compute Actor Presence (AP) and use AP-based bin clustering to improve speaker->face mapping")
    parser.add_argument("--ap-threshold", type=float, default=0.02, help="similarity threshold for AP clustering fallback or Agglomerative distance threshold complement")
    parser.add_argument("--audio-sim", help="optional JSON path to precomputed audio similarity matrix (list of lists)")
    parser.add_argument("--alpha", type=float, default=0.8, help="weight for AP when combining with audio similarity: F = S + alpha * A")
    parser.add_argument("--use-spectral", action="store_true", help="use spectral clustering on combined similarity matrix instead of threshold/agglo fallback")
    parser.add_argument("--n-clusters", type=int, default=0, help="(optional) number of clusters for spectral clustering; 0 = heuristic")
    args = parser.parse_args()

    payload = run_pipeline(
        script_path=args.script,
        video_json_path=args.video_json,
        bin_seconds=args.bin_seconds,
        iterations=args.iters,
        smoothing=args.smoothing,
        use_ap=bool(args.use_ap),
        ap_threshold=float(args.ap_threshold),
        # audio sim / fusion options
        audio_sim_path=args.audio_sim,
        alpha=float(args.alpha),
        use_spectral=bool(args.use_spectral),
        n_clusters=int(args.n_clusters) if args.n_clusters and int(args.n_clusters) > 0 else None,
    )
    dump_json(args.out, payload)
    if args.viz_out:
        viz_path = Path(args.viz_out)
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        viz_path.write_text(render_alignment_html(payload), encoding="utf8")
        print(f"Saved visualization report to {args.viz_out}")
    print(f"Saved paper-inspired alignment to {args.out}")


if __name__ == "__main__":
    main()