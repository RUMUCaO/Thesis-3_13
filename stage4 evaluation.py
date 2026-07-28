"""Structural Consensus Evaluation (SCE) helper script.

This file provides a lightweight, dependency-tolerant implementation of the
metrics described in the SCE section:

- Temporal Ordering Agreement
- Cross-View Event Alignment
- Multi-view Structural Agreement

The script is designed to work on the JSON artifacts already present in this
workspace. When a heavy external backend is unavailable, it falls back to
deterministic heuristics so the evaluation remains runnable.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.optimize import linear_sum_assignment
 
# Global model (loaded only once)
_SBERT_MODEL = None

def get_sbert_model():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        _SBERT_MODEL = SentenceTransformer('all-mpnet-base-v2')
    return _SBERT_MODEL

def sbert_encode(text: str) -> np.ndarray:
    """Returns the normalized embedding vectors (already normalized, the dot product is the cosine similarity)"""
    model = get_sbert_model()
    return model.encode(text, normalize_embeddings=True)

def sbert_similarity(text_a: str, text_b: str) -> float:
    """Use Sentence-BERT to calculate the cosine similarity between two texts"""
    model = get_sbert_model()
    emb_a = model.encode(text_a, normalize_embeddings=True)   # Normalized, dot product equals cosine
    emb_b = model.encode(text_b, normalize_embeddings=True)
    return float(np.dot(emb_a, emb_b))

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
HYPHEN_RE = re.compile(r"\s*[-–—]\s*")
WHITESPACE_RE = re.compile(r"\s+")
LOCATION_RE = re.compile(
	r"\b(?:INT|EXT|INT\./EXT|I/E)\.?(?:\s+)?([A-Z0-9][A-Z0-9 '\-/,]*)?(?:\s*[-–—]\s*(DAY|NIGHT|MORNING|AFTERNOON|EVENING|CONTINUOUS))?",
	re.IGNORECASE,
)
TIME_RE = re.compile(r"\b(DAY|NIGHT|MORNING|AFTERNOON|EVENING|DAWN|DUSK|CONTINUOUS|LATER|EARLY|LATE)\b", re.IGNORECASE)


@dataclass
class SceneRecord:
	scene_id: int
	start: Optional[float]
	end: Optional[float]
	heading: str
	text: str
	characters: List[str]
	dialogue: List[str]
	action: List[str]
	source: str = "unknown"


def load_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def save_json(path: Path, data: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(data, handle, ensure_ascii=False, indent=2)


def as_list(value: Any) -> List[Any]:
	if value is None:
		return []
	if isinstance(value, list):
		return value
	return [value]


def normalize_whitespace(text: str) -> str:
	return WHITESPACE_RE.sub(" ", text).strip()


def normalize_text(text: str) -> str:
	text = normalize_whitespace(text.lower())
	text = HYPHEN_RE.sub(" ", text)
	return text


def normalize_name(name: str) -> str:
	name = normalize_whitespace(str(name)).upper()
	name = re.sub(r"\b(?:MR|MRS|MS|MISS|DR|SPEAKER)\.?\b", "", name)
	name = re.sub(r"[^A-Z0-9]+", " ", name)
	return normalize_whitespace(name)


def tokenize(text: str) -> List[str]:
	return [token.lower() for token in TOKEN_RE.findall(text)]


def scene_heading(scene: Dict[str, Any]) -> str:
	for key in ("heading", "location", "scene_heading"):
		value = scene.get(key)
		if value:
			return str(value)
	return ""


def scene_text(scene: Dict[str, Any]) -> str:
    # Prefer using llm_extraction to build the complete text
    llm_extraction = scene.get("llm_extraction")
    if isinstance(llm_extraction, dict):
        pieces = []
        if llm_extraction.get("heading"):
            pieces.append(str(llm_extraction["heading"]))
        for block in as_list(llm_extraction.get("action_blocks")):
            pieces.append(str(block))
        for block in as_list(llm_extraction.get("dialogue_blocks")):
            if isinstance(block, dict):
                speaker = block.get("speaker")
                text = block.get("text")
                if speaker:
                    pieces.append(f"{speaker}: {text}" if text else speaker)
                elif text:
                    pieces.append(text)
        if pieces:
            return normalize_whitespace(" ".join(pieces))
    
    # Revert to the original field
    for key in ("text", "raw_text_span", "transcript_text"):
        value = scene.get(key)
        if value:
            return normalize_whitespace(str(value))
    return ""


def extract_characters(scene: Dict[str, Any]) -> List[str]:
	candidates: List[str] = []
	for key in ("characters", "cast", "unified_character_ids", "speaker_character_ids", "face_character_ids"):
		value = scene.get(key)
		if isinstance(value, list):
			candidates.extend(str(item) for item in value if item not in (None, ""))

	llm_extraction = scene.get("llm_extraction")
	if isinstance(llm_extraction, dict):
		candidates.extend(str(item) for item in as_list(llm_extraction.get("characters")) if item)

	dialogue_blocks = scene.get("dialogue_blocks") or []
	for block in dialogue_blocks:
		if isinstance(block, dict) and block.get("speaker"):
			candidates.append(str(block["speaker"]))

	speakers = scene.get("speakers") or []
	candidates.extend(str(item) for item in speakers if item)

	normalized: List[str] = []
	for item in candidates:
		name = normalize_name(item)
		if name and name not in normalized:
			normalized.append(name)
	return normalized


def extract_dialogue(scene: Dict[str, Any]) -> List[str]:
	dialogue: List[str] = []
	for block in scene.get("dialogue_blocks", []) or []:
		if isinstance(block, dict):
			text = block.get("text")
			if text:
				dialogue.append(normalize_whitespace(str(text)))
			speaker = block.get("speaker")
			if speaker:
				dialogue.append(normalize_name(str(speaker)))
		elif block:
			dialogue.append(normalize_whitespace(str(block)))

	for block in as_list(scene.get("dialogue")):
		if isinstance(block, dict):
			text = block.get("text")
			speaker = block.get("speaker")
			if speaker:
				dialogue.append(normalize_name(str(speaker)))
			if text:
				dialogue.append(normalize_whitespace(str(text)))
		elif block:
			dialogue.append(normalize_whitespace(str(block)))

	return [item for item in dialogue if item]


def extract_action(scene: Dict[str, Any]) -> List[str]:
	action: List[str] = []
	for key in ("action_blocks", "action", "summary", "transcript_text"):
		value = scene.get(key)
		if isinstance(value, list):
			for item in value:
				if item:
					action.append(normalize_whitespace(str(item)))
		elif value:
			action.append(normalize_whitespace(str(value)))

	llm_extraction = scene.get("llm_extraction")
	if isinstance(llm_extraction, dict):
		for item in as_list(llm_extraction.get("action_blocks")):
			if item:
				action.append(normalize_whitespace(str(item)))

	return [item for item in action if item]


def extract_location_and_time(scene: Dict[str, Any]) -> Tuple[str, str]:
	heading = scene_heading(scene)
	if not heading:
		return "", ""

	heading_norm = normalize_whitespace(heading)
	location = ""
	time_of_day = ""
	match = LOCATION_RE.search(heading_norm)
	if match:
		location = normalize_whitespace(match.group(1) or "")
		time_of_day = normalize_whitespace(match.group(2) or "")
	else:
		time_match = TIME_RE.search(heading_norm)
		if time_match:
			time_of_day = normalize_whitespace(time_match.group(1))
		location = heading_norm
	return location.upper(), time_of_day.upper()


def scene_records_from_payload(payload: Any, source: str) -> List[SceneRecord]:
	if isinstance(payload, dict) and "scenes" in payload:
		scenes = payload["scenes"]
	elif isinstance(payload, list):
		scenes = payload
	else:
		raise ValueError(f"Unsupported payload for {source}: expected list or object with 'scenes'.")

	records: List[SceneRecord] = []
	for index, scene in enumerate(scenes):
		if not isinstance(scene, dict):
			continue
		scene_id = scene.get("scene_id", scene.get("id", index))
		try:
			scene_id = int(scene_id)
		except Exception:
			scene_id = index
		start = scene.get("start")
		end = scene.get("end")
		try:
			start = float(start) if start is not None else None
		except Exception:
			start = None
		try:
			end = float(end) if end is not None else None
		except Exception:
			end = None
		records.append(
			SceneRecord(
				scene_id=scene_id,
				start=start,
				end=end,
				heading=scene_heading(scene),
				text=scene_text(scene),
				characters=extract_characters(scene),
				dialogue=extract_dialogue(scene),
				action=extract_action(scene),
				source=source,
			)
		)
	return records

def combined_text(scene: SceneRecord) -> str:
    # Only concatenate the heading and text (the text already includes the character, dialogue, and actions)
    parts = [scene.heading, scene.text]
    return normalize_whitespace(" ".join(part for part in parts if part))

def positional_index_map(records: Sequence[SceneRecord]) -> Dict[int, float]:
	ordered = sorted(records, key=lambda item: (item.start is None, item.start if item.start is not None else item.scene_id, item.scene_id))
	total = max(len(ordered) - 1, 1)
	return {record.scene_id: index / total for index, record in enumerate(ordered)}


def match_scene_pairs_greedy(
	generated: Sequence[SceneRecord],
	reference: Sequence[SceneRecord],
	similarity_fn: Callable[[SceneRecord, SceneRecord], float],
	threshold: float = 0.0,
	one_to_one: bool = True,
) -> List[Tuple[SceneRecord, SceneRecord, float]]:
	if not generated or not reference:
		return []

	candidates: List[Tuple[float, int, int]] = []
	for g_index, g_scene in enumerate(generated):
		for r_index, r_scene in enumerate(reference):
			score = similarity_fn(g_scene, r_scene)
			if score >= threshold:
				candidates.append((score, g_index, r_index))

	candidates.sort(reverse=True)
	used_generated: set[int] = set()
	used_reference: set[int] = set()
	matches: List[Tuple[SceneRecord, SceneRecord, float]] = []

	for score, g_index, r_index in candidates:
		if g_index in used_generated:
			continue
		if one_to_one and r_index in used_reference:
			continue
		used_generated.add(g_index)
		if one_to_one:
			used_reference.add(r_index)
		matches.append((generated[g_index], reference[r_index], score))

	matches.sort(key=lambda item: item[0].scene_id)
	return matches


def match_scene_pairs(
    generated: Sequence[SceneRecord],
    reference: Sequence[SceneRecord],
    similarity_fn: Callable[[SceneRecord, SceneRecord], float],
    threshold: float = 0.0,
    one_to_one: bool = True,
) -> List[Tuple[SceneRecord, SceneRecord, float]]:
    if not generated or not reference:
        return []

    n_gen = len(generated)
    n_ref = len(reference)

# Use the Hungarian algorithm for optimal one-to-one matching (only if scipy is available and one_to_one is True)
    if one_to_one and linear_sum_assignment is not None:
# Constructing a similarity matrix
        sim_matrix = np.zeros((n_gen, n_ref))
        for i, g_scene in enumerate(generated):
            for j, r_scene in enumerate(reference):
                sim_matrix[i, j] = similarity_fn(g_scene, r_scene)

# The Hungarian algorithm finds the minimum cost, where cost = -similarity (maximizing total similarity).
        cost_matrix = -sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        for i, j in zip(row_ind, col_ind):
            score = sim_matrix[i, j]
            if score >= threshold:
                matches.append((generated[i], reference[j], score))
        # Sort by generated scene ID to ensure stable output
        matches.sort(key=lambda m: m[0].scene_id)
        print("Hungarian algorithm used for optimal matching.")
        return matches
    else:
        print("Using greedy matching as fallback.")
# Fallback Solution: Greedy Matching (Maintaining Original Logic)
        candidates: List[Tuple[float, int, int]] = []
        for g_index, g_scene in enumerate(generated):
            for r_index, r_scene in enumerate(reference):
                score = similarity_fn(g_scene, r_scene)
                if score >= threshold:
                    candidates.append((score, g_index, r_index))

        candidates.sort(reverse=True)
        used_generated: set[int] = set()
        used_reference: set[int] = set()
        matches: List[Tuple[SceneRecord, SceneRecord, float]] = []

        for score, g_index, r_index in candidates:
            if g_index in used_generated:
                continue
            if one_to_one and r_index in used_reference:
                continue
            used_generated.add(g_index)
            if one_to_one:
                used_reference.add(r_index)
            matches.append((generated[g_index], reference[r_index], score))

        matches.sort(key=lambda item: item[0].scene_id)
    return matches

def scene_similarity(g_scene: SceneRecord, r_scene: SceneRecord, _idf=None) -> float:
    """
    Use Sentence-BERT to calculate the semantic similarity between combined texts.
    The _idf parameter is kept for compatibility with existing calls, but not actually used.
    """
    text_g = combined_text(g_scene)
    text_r = combined_text(r_scene)
    return sbert_similarity(text_g, text_r)
 
def dtw_distance(seq_a: Sequence[SceneRecord], seq_b: Sequence[SceneRecord]) -> float:
    """
    DTW over SBERT scene embeddings using cosine distance.
    Lower is better.
    """
    if not seq_a or not seq_b:
        return float("inf")

    n, m = len(seq_a), len(seq_b)

    # Pre-computed embeddings (combined_text for each scene)
    a_vecs = [sbert_encode(combined_text(s)) for s in seq_a]
    b_vecs = [sbert_encode(combined_text(s)) for s in seq_b]

    def cost(i: int, j: int) -> float:
        # Cosine distance = 1 - Cosine similarity
        return 1.0 - float(np.dot(a_vecs[i], b_vecs[j]))

    dp = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = cost(i - 1, j - 1)
            dp[i][j] = c + min(
                dp[i - 1][j],     # insertion
                dp[i][j - 1],     # deletion
                dp[i - 1][j - 1]  # match
            )

    return dp[n][m] / (n + m)


def extract_event_tuple(scene: SceneRecord) -> Dict[str, List[str]]:
	location, time_of_day = extract_location_and_time({"heading": scene.heading})
	content = tokenize(scene.text or combined_text(scene))
	predicate = [token for token in content if token not in {"the", "a", "an", "and", "or", "to", "of", "in", "on", "with", "for", "is", "are"}][:4]
	if not predicate:
		predicate = content[:2]
	return {
		"subject": scene.characters[:4],
		"predicate": predicate,
		"object": [token for token in content if token not in predicate][:6],
		"location": [location] if location else [],
		"time": [time_of_day] if time_of_day else [],
	}

def align_views(left: Sequence[SceneRecord], right: Sequence[SceneRecord]) -> float:
    """
    Calculate the average maximum similarity between two views (scene sequences).
    Each scene within a view is represented by its SBERT embedding.
    """
    if not left or not right:
        return 0.0
    
    # Pre-computed embeddings
    left_emb = [sbert_encode(combined_text(s)) for s in left]
    right_emb = [sbert_encode(combined_text(s)) for s in right]
    
    scores = []
    for lv in left_emb:
        # Calculate the maximum cosine similarity between the current left view scene and all right view scenes
        best = max(np.dot(lv, rv) for rv in right_emb)
        scores.append(best)
    return float(np.mean(scores)) if scores else 0.0

def consensus_score(views: Sequence[Sequence[SceneRecord]]) -> Dict[str, float]:
	if len(views) < 2:
		return {"consensus_score": 0.0}
	pair_scores: List[float] = []
	for left_index in range(len(views)):
		for right_index in range(len(views)):
			if left_index == right_index:
				continue
			pair_scores.append(align_views(views[left_index], views[right_index]))
	return {"consensus_score": mean(pair_scores) if pair_scores else 0.0}

def compute_np_metric(
    generated: Sequence[SceneRecord],
    reference: Sequence[SceneRecord],
    matches: List[Tuple[SceneRecord, SceneRecord, float]],
    k: int = 5,
) -> float:
    """
    Neighborhood Preservation (NP) metric using SBERT embeddings.

    For each matched pair (g, r), compute the Jaccard overlap between
    the top-k nearest neighbors of g in the generated scene set and
    the top-k nearest neighbors of r in the reference scene set.
    """
    if not matches:
        return 0.0

    # Encode all scenes using SBERT (normalized embeddings)
    gen_texts = [combined_text(s) for s in generated]
    ref_texts = [combined_text(s) for s in reference]
    
    # Batch encode for efficiency
    all_texts = gen_texts + ref_texts
    all_embeddings = get_sbert_model().encode(all_texts, normalize_embeddings=True)
    
    n_gen = len(gen_texts)
    gen_vecs = all_embeddings[:n_gen]        # (n_gen, dim)
    ref_vecs = all_embeddings[n_gen:]        # (n_ref, dim)

    # Compute cosine similarity matrices (dot product since vectors are normalized)
    gen_sim = np.dot(gen_vecs, gen_vecs.T)   # (n_gen, n_gen)
    ref_sim = np.dot(ref_vecs, ref_vecs.T)   # (n_ref, n_ref)

    # Get top-k neighbors (excluding self) for each scene
    def topk_neighbors(sim_matrix: np.ndarray, k: int) -> List[Set[int]]:
        n = sim_matrix.shape[0]
        neighbors = []
        for i in range(n):
            # argsort descending, skip self (similarity[i,i] = 1.0)
            indices = np.argsort(-sim_matrix[i])
            topk = [int(idx) for idx in indices if idx != i][:k]
            neighbors.append(set(topk))
        return neighbors

    gen_neighbors = topk_neighbors(gen_sim, k)
    ref_neighbors = topk_neighbors(ref_sim, k)

    # Map scene_id -> index in the respective list
    gen_index_map = {s.scene_id: i for i, s in enumerate(generated)}
    ref_index_map = {s.scene_id: i for i, s in enumerate(reference)}

    g2r: Dict[int, int] = {}
    for g_scene, r_scene, _ in matches:
        g_idx = gen_index_map.get(g_scene.scene_id)
        r_idx = ref_index_map.get(r_scene.scene_id)
        if g_idx is not None and r_idx is not None:
            g2r[g_idx] = r_idx

    jaccards = []
    for g_scene, r_scene, _ in matches:
        g_idx = gen_index_map.get(g_scene.scene_id)
        r_idx = ref_index_map.get(r_scene.scene_id)
        if g_idx is None or r_idx is None:
            continue
        # Project g's top-k neighbors into the reference index space;
        # neighbors without a matched counterpart are dropped (they cannot
        # be verified), which also prevents unmatched scenes from inflating
        # the union.
        set_g = {g2r[n] for n in gen_neighbors[g_idx] if n in g2r}
        set_r = ref_neighbors[r_idx]
        inter = len(set_g & set_r)
        union = len(set_g | set_r)
        # If none of g's neighbors are matched, the neighborhood is
        # considered not preserved (score 0 for this pair).
        jac = inter / union if union > 0 else 0.0
        jaccards.append(jac)

    return float(np.mean(jaccards)) if jaccards else 0.0

def order_correlation(matches, gen_seq, ref_seq):
    # Matches is a list of (gen_scene, ref_scene, score).
    gen_ranks = {}
    ref_ranks = {}
    for idx, scene in enumerate(gen_seq):
        gen_ranks[scene.scene_id] = idx
    for idx, scene in enumerate(ref_seq):
        ref_ranks[scene.scene_id] = idx
    
    # Only consider matching scenarios
    gen_positions = []
    ref_positions = []
    for g_scene, r_scene, _ in matches:
        gen_positions.append(gen_ranks[g_scene.scene_id])
        ref_positions.append(ref_ranks[r_scene.scene_id])
    
    # Calculate the Spearman correlation coefficient
    from scipy.stats import spearmanr
    corr, _ = spearmanr(gen_positions, ref_positions)
    return corr

def evaluate_sce(
    generated_payload: Any,
    reference_payload: Any,
    identity_payload: Optional[Any] = None,
    threshold: float = 0.7,
) -> Dict[str, Any]:
    generated = scene_records_from_payload(generated_payload, source="generated")
    reference = scene_records_from_payload(reference_payload, source="reference")

    similarity_fn = lambda g, r: scene_similarity(g, r)
    matches = match_scene_pairs(generated, reference, similarity_fn, threshold=threshold, one_to_one=True)
    np_score = compute_np_metric(generated, reference, matches, k=5)
    dtw_score = dtw_distance(generated, reference)
    order_corr = order_correlation(matches, generated, reference)

    results = {
        "threshold": threshold,
        "counts": {
            "generated_scenes": len(generated),
            "reference_scenes": len(reference),
            "matched_scenes": len(matches),
        },
        "Multimodal Semantic Similarity": {
            "Multimodal Semantic Similarity": mean(score for _, _, score in matches) if matches else 0.0,
        },
        "Weak Event Alignment": {
            "consensus_score": consensus_score([generated, reference])["consensus_score"],
        },
        "dtw_distance": dtw_score,
        "np_metric": np_score,
        "matched_pairs": [
            {
                "generated_scene_id": g_scene.scene_id,
                "reference_scene_id": r_scene.scene_id,
                "similarity": score,
            }
            for g_scene, r_scene, score in matches[:20]
        ],
        "order_correlation": order_corr,
    }
    return results

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Structural Consensus Evaluation metrics.")
    parser.add_argument("--generated", type=Path, default=Path("generated_scripts.json"), help="Path to the generated scene JSON.")
    parser.add_argument("--reference", type=Path, default=Path("script_structured.json"), help="Path to the reference scene JSON.")
    parser.add_argument("--identity", type=Path, default=Path("identity_reconciliation.json"), help="Optional identity reconciliation JSON.")
    parser.add_argument("--corpus", type=Path, default=None, help="Optional corpus JSON for perplexity estimation.")
    parser.add_argument("--output", type=Path, default=Path("report/stage4_evaluation_report.json"), help="Output JSON report path.")
    parser.add_argument("--print", dest="print_report", action="store_true", help="Print the report to stdout.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold for greedy matching (default: 0.6)")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    generated_payload = load_json(args.generated)
    reference_payload = load_json(args.reference)
    identity_payload = load_json(args.identity) if args.identity.exists() else None

    report = evaluate_sce(
        generated_payload,
        reference_payload,
        identity_payload,
        threshold=args.threshold,
    )

    # The output file will automatically include a threshold suffix (if the threshold is not the default value of 0.6, the suffix will be added; it can also always be added).
    output_path = args.output
    # Optional: Add a suffix if the threshold is not 0.6; or always add a suffix.
    if args.threshold != 0.6:
        stem = output_path.stem
        suffix = output_path.suffix
        output_path = output_path.parent / f"{stem}_th{args.threshold}{suffix}"
    save_json(output_path, report)

    if args.print_report:
        print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
