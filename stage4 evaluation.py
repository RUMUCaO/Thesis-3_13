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

try:
	from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - optional dependency
	linear_sum_assignment = None


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
	for key in ("text", "raw_text", "transcript_text"):
		value = scene.get(key)
		if value:
			return normalize_whitespace(str(value))

	pieces: List[str] = []
	heading = scene_heading(scene)
	if heading:
		pieces.append(heading)

	for key in ("action", "summary", "description"):
		value = scene.get(key)
		if isinstance(value, dict):
			pieces.extend(str(v) for v in value.values() if v)
		elif value:
			pieces.append(str(value))

	for key in ("dialogue", "dialogue_blocks", "action_blocks", "cast"):
		value = scene.get(key)
		if isinstance(value, list):
			for item in value:
				if isinstance(item, dict):
					for inner_key in ("speaker", "text", "value"):
						if item.get(inner_key):
							pieces.append(str(item[inner_key]))
				elif item:
					pieces.append(str(item))

	llm_extraction = scene.get("llm_extraction")
	if isinstance(llm_extraction, dict):
		if llm_extraction.get("heading"):
			pieces.append(str(llm_extraction["heading"]))
		for key in ("characters", "action_blocks"):
			for item in as_list(llm_extraction.get(key)):
				if item:
					pieces.append(str(item))
		for block in as_list(llm_extraction.get("dialogue_blocks")):
			if isinstance(block, dict):
				speaker = block.get("speaker")
				text = block.get("text")
				if speaker:
					pieces.append(str(speaker))
				if text:
					pieces.append(str(text))

	return normalize_whitespace(" ".join(pieces))


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
	parts = [scene.heading, scene.text, " ".join(scene.characters), " ".join(scene.dialogue), " ".join(scene.action)]
	return normalize_whitespace(" ".join(part for part in parts if part))


def build_idf(corpus: Sequence[str]) -> Dict[str, float]:
	doc_count = len(corpus)
	if doc_count == 0:
		return {}
	df: Counter[str] = Counter()
	for text in corpus:
		df.update(set(tokenize(text)))
	return {token: math.log((1.0 + doc_count) / (1.0 + count)) + 1.0 for token, count in df.items()}


def vectorize(text: str, idf: Dict[str, float]) -> Dict[str, float]:
	counts = Counter(tokenize(text))
	total = sum(counts.values()) or 1
	return {token: (count / total) * idf.get(token, 1.0) for token, count in counts.items()}


def cosine_similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
	if not left or not right:
		return 0.0
	common = set(left) & set(right)
	numerator = sum(left[token] * right[token] for token in common)
	left_norm = math.sqrt(sum(value * value for value in left.values()))
	right_norm = math.sqrt(sum(value * value for value in right.values()))
	if left_norm == 0.0 or right_norm == 0.0:
		return 0.0
	return numerator / (left_norm * right_norm)


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
	left_set = {item for item in left if item}
	right_set = {item for item in right if item}
	if not left_set and not right_set:
		return 1.0
	union = left_set | right_set
	if not union:
		return 0.0
	return len(left_set & right_set) / len(union)


def overlap_ratio(left: Iterable[str], right: Iterable[str]) -> float:
	left_tokens = set(tokenize(" ".join(left)))
	right_tokens = set(tokenize(" ".join(right)))
	if not left_tokens and not right_tokens:
		return 1.0
	union = left_tokens | right_tokens
	if not union:
		return 0.0
	return len(left_tokens & right_tokens) / len(union)


def positional_index_map(records: Sequence[SceneRecord]) -> Dict[int, float]:
	ordered = sorted(records, key=lambda item: (item.start is None, item.start if item.start is not None else item.scene_id, item.scene_id))
	total = max(len(ordered) - 1, 1)
	return {record.scene_id: index / total for index, record in enumerate(ordered)}


def match_scene_pairs(
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


def scene_similarity(g_scene: SceneRecord, r_scene: SceneRecord, idf: Dict[str, float]) -> float:
	text_score = cosine_similarity(vectorize(combined_text(g_scene), idf), vectorize(combined_text(r_scene), idf))
	char_score = jaccard(g_scene.characters, r_scene.characters)
	dialog_score = overlap_ratio(g_scene.dialogue, r_scene.dialogue)
	action_score = overlap_ratio(g_scene.action, r_scene.action)
	return 0.5 * text_score + 0.2 * char_score + 0.15 * dialog_score + 0.15 * action_score


def compute_temporal_metrics(
	generated: Sequence[SceneRecord],
	reference: Sequence[SceneRecord],
	matches: Sequence[Tuple[SceneRecord, SceneRecord, float]],
) -> Dict[str, float]:
	if not matches:
		return {"alignment_retrieval_map": 0.0, "multimodal_semantic_similarity": 0.0}

	gen_pos = positional_index_map(generated)
	ref_pos = positional_index_map(reference)
	window = 0.5 / max(len(generated), len(reference), 1)

	pair_positions: List[Tuple[float, float]] = []
	distances: List[float] = []
	semantic_scores: List[float] = []

	for g_scene, r_scene, score in matches:
		g_pos = gen_pos.get(g_scene.scene_id, 0.0)
		r_pos = ref_pos.get(r_scene.scene_id, 0.0)
		pair_positions.append((g_pos, r_pos))
		distances.append(abs(g_pos - r_pos))
		semantic_scores.append(score)

	concordant = 0
	discordant = 0
	for left_index in range(len(pair_positions)):
		for right_index in range(left_index + 1, len(pair_positions)):
			left_g, left_r = pair_positions[left_index]
			right_g, right_r = pair_positions[right_index]
			order_g = left_g - right_g
			order_r = left_r - right_r
			if order_g == 0 or order_r == 0:
				continue
			if order_g * order_r > 0:
				concordant += 1
			else:
				discordant += 1

	shared_idf = build_idf([combined_text(scene) for scene in list(generated) + list(reference)])
	reference_vectors = [vectorize(combined_text(scene), shared_idf) for scene in reference]
	generated_vectors = [vectorize(combined_text(scene), shared_idf) for scene in generated]
	reference_index_by_id = {scene.scene_id: index for index, scene in enumerate(reference)}
	average_precisions: List[float] = []
	for g_scene, r_scene, _score in matches:
		g_vector = generated_vectors[[scene.scene_id for scene in generated].index(g_scene.scene_id)]
		ranked = sorted(
			enumerate(reference_vectors),
			key=lambda item: cosine_similarity(g_vector, item[1]),
			reverse=True,
		)
		target_index = reference_index_by_id.get(r_scene.scene_id)
		if target_index is None:
			continue
		for rank, (candidate_index, _) in enumerate(ranked, start=1):
			if candidate_index == target_index:
				average_precisions.append(1.0 / rank)
				break

	return {
		"alignment_retrieval_map": mean(average_precisions) if average_precisions else 0.0,
		"multimodal_semantic_similarity": mean(semantic_scores) if semantic_scores else 0.0,
	}
 
def dtw_distance(seq_a: Sequence[SceneRecord], seq_b: Sequence[SceneRecord], idf: Dict[str, float]) -> float:
	"""
	Classic DTW over scene embeddings using cosine distance.
	Lower is better.
	"""
	if not seq_a or not seq_b:
		return float("inf")

	n, m = len(seq_a), len(seq_b)

	# precompute embeddings
	a_vecs = [vectorize(combined_text(s), idf) for s in seq_a]
	b_vecs = [vectorize(combined_text(s), idf) for s in seq_b]

	def cost(i: int, j: int) -> float:
		return 1.0 - cosine_similarity(a_vecs[i], b_vecs[j])

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

def event_embedding(scene: SceneRecord, idf: Dict[str, float]) -> Dict[str, float]:
	return vectorize(combined_text(scene), idf)

def align_views(left: Sequence[SceneRecord], right: Sequence[SceneRecord]) -> float:
	if not left or not right:
		return 0.0
	shared_idf = build_idf([combined_text(scene) for scene in list(left) + list(right)])
	left_embeddings = [event_embedding(scene, shared_idf) for scene in left]
	right_embeddings = [event_embedding(scene, shared_idf) for scene in right]
	scores = []
	for embedding in left_embeddings:
		scores.append(max(cosine_similarity(embedding, candidate) for candidate in right_embeddings))
	return mean(scores) if scores else 0.0

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
    Neighborhood Preservation (NP) metric.

    For each matched pair (g, r), compute the Jaccard overlap between
    the top-k nearest neighbors of g in the generated scene set and
    the top-k nearest neighbors of r in the reference scene set.
    """
    if not matches:
        return 0.0

    # Build shared IDF and vectorize all scenes
    all_scenes = list(generated) + list(reference)
    idf = build_idf([combined_text(s) for s in all_scenes])

    gen_vecs = [vectorize(combined_text(s), idf) for s in generated]
    ref_vecs = [vectorize(combined_text(s), idf) for s in reference]

    # Precompute similarity matrices for fast neighbor retrieval
    n_gen = len(gen_vecs)
    n_ref = len(ref_vecs)

    gen_sim = np.zeros((n_gen, n_gen), dtype=np.float32)
    for i in range(n_gen):
        for j in range(i, n_gen):
            sim = cosine_similarity(gen_vecs[i], gen_vecs[j])
            gen_sim[i, j] = sim
            gen_sim[j, i] = sim

    ref_sim = np.zeros((n_ref, n_ref), dtype=np.float32)
    for i in range(n_ref):
        for j in range(i, n_ref):
            sim = cosine_similarity(ref_vecs[i], ref_vecs[j])
            ref_sim[i, j] = sim
            ref_sim[j, i] = sim

    # Get top-k neighbors (excluding self)
    def topk_neighbors(sim_matrix: np.ndarray, k: int) -> List[Set[int]]:
        n = sim_matrix.shape[0]
        neighbors = []
        for i in range(n):
            # Sort indices by similarity descending, skip self
            indices = np.argsort(-sim_matrix[i])
            # remove self (similarity 1.0)
            topk = [int(idx) for idx in indices if idx != i][:k]
            neighbors.append(set(topk))
        return neighbors

    gen_neighbors = topk_neighbors(gen_sim, k)
    ref_neighbors = topk_neighbors(ref_sim, k)

    # Map scene objects back to their index in the lists
    gen_index_map = {s.scene_id: i for i, s in enumerate(generated)}
    ref_index_map = {s.scene_id: i for i, s in enumerate(reference)}

    jaccards = []
    for g_scene, r_scene, _ in matches:
        g_idx = gen_index_map.get(g_scene.scene_id)
        r_idx = ref_index_map.get(r_scene.scene_id)
        if g_idx is None or r_idx is None:
            continue
        set_g = gen_neighbors[g_idx]
        set_r = ref_neighbors[r_idx]
        union = len(set_g | set_r)
        if union == 0:
            jac = 0.0
        else:
            jac = len(set_g & set_r) / union
        jaccards.append(jac)

    return float(np.mean(jaccards)) if jaccards else 0.0

def evaluate_sce(
	generated_payload: Any,
	reference_payload: Any,
	identity_payload: Optional[Any] = None,
) -> Dict[str, Any]:
	generated = scene_records_from_payload(generated_payload, source="generated")
	reference = scene_records_from_payload(reference_payload, source="reference")
	identity = identity_payload if isinstance(identity_payload, dict) else None

	shared_idf = build_idf([combined_text(scene) for scene in list(generated) + list(reference)])
	similarity_fn = lambda left, right: scene_similarity(left, right, shared_idf)
	matches = match_scene_pairs(generated, reference, similarity_fn, threshold=0.05, one_to_one=True)
	np_score = compute_np_metric(generated, reference, matches, k=5)
	dtw_score = dtw_distance(generated, reference, shared_idf)

	results: Dict[str, Any] = {
		"counts": {
			"generated_scenes": len(generated),
			"reference_scenes": len(reference),
			"matched_scenes": len(matches),
		},
		"temporal_ordering_agreement": compute_temporal_metrics(generated, reference, matches),
		"cross_view_event_alignment": {
			"align_score": mean(score for _, _, score in matches) if matches else 0.0,
		},
		"consensus": consensus_score([generated, reference]),
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
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	generated_payload = load_json(args.generated)
	reference_payload = load_json(args.reference)
	identity_payload = load_json(args.identity) if args.identity.exists() else None

	report = evaluate_sce(generated_payload, reference_payload, identity_payload)

	save_json(args.output, report)
	if args.print_report:
		print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
	main()
