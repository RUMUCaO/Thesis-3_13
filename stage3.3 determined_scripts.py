#!/usr/bin/env python3
"""Generate a coherent screenplay-like script from movie-derived scene outputs.

The generator keeps three constraints explicit:
1. Temporal structure: scenes are emitted in time order and each block is timestamped.
2. Character consistency: speaker labels are canonicalized through identity reconciliation when available.
3. Narrative flow: VLM scene summaries are combined with aligned WhisperX dialogue into one continuous script.

Input defaults assume the stage3 outputs from this repository:
- scene_level_results.json
- whisperx_results.json
- identity_reconciliation.json

The output is both a human-readable text script and a structured JSON document.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def load_json(path: str | Path) -> Any:
	with Path(path).open("r", encoding="utf8") as handle:
		return json.load(handle)


def dump_json(path: str | Path, data: Any) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf8") as handle:
		json.dump(data, handle, ensure_ascii=False, indent=2)


def load_scene_results(path: str | Path, selected_only: bool = False) -> List[Dict[str, Any]]:
	data = load_json(path)
	if not isinstance(data, list):
		raise ValueError("scene results must be a list")

	scenes = [scene for scene in data if isinstance(scene, dict)]
	scenes.sort(key=lambda scene: (float(scene.get("start", 0.0)), int(scene.get("scene_id", 0))))

	if selected_only:
		scenes = [scene for scene in scenes if bool(scene.get("selection_tags"))]

	return scenes


def load_identity_maps(path: str | Path | None) -> Dict[str, Any]:
	if not path:
		return {}
	try:
		data = load_json(path)
	except FileNotFoundError:
		return {}
	if isinstance(data, dict):
		return data
	return {}


def load_whisperx_alignment(path: str | Path | None) -> Dict[str, Any]:
	if not path:
		return {}
	try:
		data = load_json(path)
	except FileNotFoundError:
		return {}
	if isinstance(data, dict):
		return data
	return {}


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
	return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def normalize_whitespace(text: str) -> str:
	return re.sub(r"\s+", " ", text or "").strip()


def canonical_character_name(name: str, identity_map: Dict[str, Any]) -> str:
	raw = normalize_whitespace(str(name))
	if not raw:
		return ""

	alias_map: Dict[str, str] = {}
	speaker_to_character = identity_map.get("speaker_to_character", {})
	if isinstance(speaker_to_character, dict):
		alias_map.update({str(k): str(v) for k, v in speaker_to_character.items()})

	unified_characters = identity_map.get("unified_characters", [])
	if isinstance(unified_characters, list):
		for item in unified_characters:
			if not isinstance(item, dict):
				continue
			character_id = str(item.get("character_id", "")).strip()
			if not character_id:
				continue
			for alias in item.get("aliases", []) or []:
				alias_map.setdefault(str(alias).strip(), character_id)
			display_name = str(item.get("display_name", "")).strip()
			if display_name:
				alias_map.setdefault(display_name, character_id)

	return alias_map.get(raw, raw)


def infer_scene_heading(scene: Dict[str, Any]) -> str:
	scene_id = scene.get("scene_id", 0)
	start = float(scene.get("start", 0.0))
	end = float(scene.get("end", 0.0))
	return f"SCENE {scene_id} | {start:0.3f}s - {end:0.3f}s"


def scene_cast(scene: Dict[str, Any], identity_map: Dict[str, Any]) -> List[str]:
	ordered: List[str] = []
	seen = set()

	def add_many(values: Sequence[Any]) -> None:
		for value in values or []:
			name = canonical_character_name(str(value), identity_map)
			if not name or name in seen:
				continue
			seen.add(name)
			ordered.append(name)

	add_many(scene.get("unified_character_ids", []) or [])
	add_many(scene.get("face_character_ids", []) or [])
	add_many(scene.get("speaker_character_ids", []) or [])

	if not ordered:
		raw_speakers = scene.get("speakers", []) or []
		add_many(raw_speakers)

	if not ordered:
		add_many(scene.get("characters", []) or [])

	return ordered


def scene_action_text(scene: Dict[str, Any], include_script_hints: bool = False) -> str:
	description = scene.get("description", {}) if isinstance(scene.get("description", {}), dict) else {}
	parts: List[str] = []

	summary = normalize_whitespace(str(description.get("summary", "")))
	if summary:
		parts.append(summary)

	actions = description.get("actions", [])
	if isinstance(actions, list) and actions:
		unique_actions: List[str] = []
		seen_actions = set()
		for action in actions:
			cleaned = normalize_whitespace(str(action))
			if not cleaned or cleaned in seen_actions:
				continue
			seen_actions.add(cleaned)
			unique_actions.append(cleaned)
		action_text = "; ".join(unique_actions)
		if action_text:
			parts.append(action_text)

	relation_state = normalize_whitespace(str(description.get("relation_state", "")))
	if relation_state and relation_state.lower() not in {"no relation.", "no relation", "unknown"} and relation_state not in parts:
		parts.append(f"Relation: {relation_state}")

	transcript = normalize_whitespace(str(scene.get("transcript_text", "")))
	if transcript:
		parts.append(f"Dialogue trace: {transcript}")

	if include_script_hints:
		script_hint = normalize_whitespace(str(scene.get("script_hint", "")))
		if script_hint:
			parts.append(f"Hint: {script_hint}")

	return " ".join(parts).strip()


def extract_whisperx_words(whisperx_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从 WhisperX 输出中提取句子/片段级别的文本（text），而非单词。
    支持以下几种输入结构：
      1. 顶层 "segments" 列表（标准 WhisperX 输出）
      2. 顶层 "result" 列表（类似你提供的示例，包含 speaker、character_cluster）
      3. 向后兼容旧结构 "transcription_aligned.segments"
    返回的每个字典包含：start, end, text, speaker(可选), character_cluster(可选)
    """
    items: List[Dict[str, Any]] = []
    seen = set()

    def add_item(start: Any, end: Any, text: str, speaker: Any = None, character_cluster: Any = None) -> None:
        if start is None or end is None or not text:
            return
        # 转为浮点数并四舍五入到3位小数，便于去重
        start_f = round(float(start), 3)
        end_f = round(float(end), 3)
        key = (start_f, end_f, text)
        if key in seen:
            return
        seen.add(key)
        item = {"start": start_f, "end": end_f, "text": text}
        if speaker is not None:
            item["speaker"] = str(speaker)
        if character_cluster is not None:
            # 参考你的示例，将 character_cluster 转为整数或 None
            try:
                item["character_cluster"] = int(character_cluster) if character_cluster is not None else None
            except (ValueError, TypeError):
                item["character_cluster"] = None
        items.append(item)

    # 1) 优先处理 "result" 结构（你提供的示例格式）
    result = whisperx_data.get("result")
    if isinstance(result, list):
        for seg in result:
            if not isinstance(seg, dict):
                continue
            start = seg.get("start")
            end = seg.get("end")
            text = normalize_whitespace(str(seg.get("text", "")))
            speaker = seg.get("speaker")
            # 兼容 "character_cluster" 或 "character" 字段
            character_cluster = seg.get("character_cluster") or seg.get("character")
            add_item(start, end, text, speaker, character_cluster)
        # 如果 result 存在且非空，直接返回（避免重复解析其他字段）
        if items:
            items.sort(key=lambda x: (x["start"], x["end"]))
            return items

    # 2) 处理顶层 "segments"（标准 WhisperX 输出）
    top_segments = whisperx_data.get("segments")
    if isinstance(top_segments, list):
        for seg in top_segments:
            if not isinstance(seg, dict):
                continue
            start = seg.get("start")
            end = seg.get("end")
            text = normalize_whitespace(str(seg.get("text", "")))
            # 标准 segments 可能带有 speaker，也可提取
            speaker = seg.get("speaker")
            add_item(start, end, text, speaker)

    # 3) 向后兼容：transcription_aligned.segments
    aligned = whisperx_data.get("transcription_aligned", {})
    if isinstance(aligned, dict):
        aligned_segments = aligned.get("segments")
        if isinstance(aligned_segments, list):
            for seg in aligned_segments:
                if not isinstance(seg, dict):
                    continue
                start = seg.get("start")
                end = seg.get("end")
                text = normalize_whitespace(str(seg.get("text", "")))
                speaker = seg.get("speaker")
                add_item(start, end, text, speaker)

    items.sort(key=lambda x: (x["start"], x["end"]))
    return items


def extract_diarization_spans(whisperx_data: Dict[str, Any]) -> List[Dict[str, Any]]:
	spans: List[Dict[str, Any]] = []
	diarization = whisperx_data.get("diarization", [])
	if not isinstance(diarization, list):
		return spans

	for item in diarization:
		if not isinstance(item, dict):
			continue
		start = item.get("start")
		end = item.get("end")
		speaker = str(item.get("speaker", "")).strip()
		if start is None or end is None or not speaker:
			continue
		spans.append({"start": float(start), "end": float(end), "speaker": speaker})

	spans.sort(key=lambda item: (item["start"], item["end"]))
	return spans


def assign_word_speaker(word: Dict[str, Any], diarization_spans: List[Dict[str, Any]]) -> Optional[str]:
	best_speaker = None
	best_overlap = 0.0
	for span in diarization_spans:
		overlap = overlap_seconds(word["start"], word["end"], span["start"], span["end"])
		if overlap > best_overlap:
			best_overlap = overlap
			best_speaker = span["speaker"]
	return best_speaker


def attach_speakers_to_words(words: List[Dict[str, Any]], diarization_spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	annotated: List[Dict[str, Any]] = []
	for word in words:
		item = dict(word)
		item["speaker"] = assign_word_speaker(item, diarization_spans)
		annotated.append(item)
	return annotated


def scene_index_for_time(time_s: float, scene_starts: List[float]) -> int:
	idx = bisect.bisect_right(scene_starts, time_s) - 1
	return max(0, min(idx, len(scene_starts) - 1))


def attach_words_to_scenes(words: List[Dict[str, Any]], scenes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
	scene_starts = [float(scene.get("start", 0.0)) for scene in scenes]
	scene_words: List[List[Dict[str, Any]]] = [[] for _ in scenes]

	for word in words:
		midpoint = (float(word["start"]) + float(word["end"])) / 2.0
		idx = scene_index_for_time(midpoint, scene_starts)
		scene = scenes[idx]
		if float(word["end"]) < float(scene.get("start", 0.0)):
			continue
		if float(word["start"]) > float(scene.get("end", 0.0)):
			continue
		scene_words[idx].append(word)

	return scene_words


def group_words_into_turns(words: List[Dict[str, Any]], max_gap: float = 0.75) -> List[Dict[str, Any]]:
	if not words:
		return []

	words = sorted(words, key=lambda item: (item["start"], item["end"]))
	turns: List[Dict[str, Any]] = []
	current = {
		"speaker": words[0].get("speaker"),
		"start": words[0]["start"],
		"end": words[0]["end"],
		"tokens": [words[0]["text"]],
	}

	for word in words[1:]:
		same_speaker = word.get("speaker") == current["speaker"]
		close_enough = float(word["start"]) - float(current["end"]) <= max_gap
		if same_speaker and close_enough:
			current["end"] = max(current["end"], word["end"])
			current["tokens"].append(word["text"])
			continue

		turns.append({
			"speaker": current["speaker"],
			"start": current["start"],
			"end": current["end"],
			"text": normalize_whitespace(" ".join(current["tokens"])),
		})
		current = {
			"speaker": word.get("speaker"),
			"start": word["start"],
			"end": word["end"],
			"tokens": [word["text"]],
		}

	turns.append({
		"speaker": current["speaker"],
		"start": current["start"],
		"end": current["end"],
		"text": normalize_whitespace(" ".join(current["tokens"])),
	})
	return turns


def canonical_turn_speaker(turn: Dict[str, Any], identity_map: Dict[str, Any], scene_cast_names: List[str]) -> str:
	speaker = turn.get("speaker")
	if speaker:
		mapped = canonical_character_name(str(speaker), identity_map)
		if mapped and mapped != str(speaker):
			return mapped
		return str(speaker)
	if len(scene_cast_names) == 1:
		return scene_cast_names[0]
	return "NARRATOR"


def build_scene_turns(scene_words: List[Dict[str, Any]], scene: Dict[str, Any], identity_map: Dict[str, Any]) -> List[Dict[str, Any]]:
	turns = group_words_into_turns(scene_words)
	cast_names = scene_cast(scene, identity_map)

	if not turns and normalize_whitespace(str(scene.get("transcript_text", ""))):
		turns = [{
			"speaker": cast_names[0] if len(cast_names) == 1 else None,
			"start": float(scene.get("start", 0.0)),
			"end": float(scene.get("end", 0.0)),
			"text": normalize_whitespace(str(scene.get("transcript_text", ""))),
		}]

	formatted: List[Dict[str, Any]] = []
	for turn in turns:
		speaker_name = canonical_turn_speaker(turn, identity_map, cast_names)
		formatted.append({
			"speaker": speaker_name,
			"start": turn.get("start"),
			"end": turn.get("end"),
			"text": normalize_whitespace(str(turn.get("text", ""))),
		})
	return formatted


def format_scene_block(
	scene: Dict[str, Any],
	turns: List[Dict[str, Any]],
	identity_map: Optional[Dict[str, Any]] = None,
	include_script_hints: bool = False,
) -> Dict[str, Any]:
	heading = infer_scene_heading(scene)
	cast = scene_cast(scene, identity_map or {})
	action_text = scene_action_text(scene, include_script_hints=include_script_hints)

	lines: List[str] = [heading]
	if cast:
		lines.append(f"CAST: {', '.join(cast)}")
	if action_text:
		lines.append(f"ACTION: {action_text}")

	for turn in turns:
		speaker = normalize_whitespace(str(turn.get("speaker", "NARRATOR"))) or "NARRATOR"
		text = normalize_whitespace(str(turn.get("text", "")))
		if not text:
			continue
		lines.append(f"{speaker.upper()}: {text}")

	if len(lines) == 1:
		lines.append("ACTION: [No transcript or scene description available]")

	return {
		"scene_id": scene.get("scene_id"),
		"start": scene.get("start"),
		"end": scene.get("end"),
		"heading": heading,
		"cast": cast,
		"action": action_text,
		"dialogue": turns,
		"text": "\n".join(lines),
	}


def generate_script_document(
	scenes: List[Dict[str, Any]],
	whisperx_data: Optional[Dict[str, Any]] = None,
	identity_map: Optional[Dict[str, Any]] = None,
	include_script_hints: bool = False,
) -> Dict[str, Any]:
	identity_map = identity_map or {}
	whisperx_data = whisperx_data or {}

	words = attach_speakers_to_words(extract_whisperx_words(whisperx_data), extract_diarization_spans(whisperx_data))
	scene_words = attach_words_to_scenes(words, scenes)

	scene_blocks: List[Dict[str, Any]] = []
	for scene, words_in_scene in zip(scenes, scene_words):
		turns = build_scene_turns(words_in_scene, scene, identity_map)
		scene_block = format_scene_block(scene, turns, identity_map=identity_map, include_script_hints=include_script_hints)
		scene_blocks.append(scene_block)

	script_text = "\n\n".join(block["text"] for block in scene_blocks)

	all_characters = []
	seen = set()
	for scene in scenes:
		for name in scene_cast(scene, identity_map):
			if name not in seen:
				seen.add(name)
				all_characters.append(name)

	return {
		"metadata": {
			"scene_count": len(scenes),
			"character_count": len(all_characters),
			"use_script_hints": include_script_hints,
		},
		"characters": all_characters,
		"scenes": scene_blocks,
		"script_text": script_text,
	}


def main(
    scenes: str = "scene_level_results.json",
    whisperx: str = "whisperx_results.json",
    identity: str = "identity_reconciliation.json",
    out: str = "generated_scripts.json",
    txt: str = "generated_scripts.txt",
    selected_only: bool = False,
    include_script_hints: bool = False,
) -> None:
    scenes_data = load_scene_results(scenes, selected_only=selected_only)
    whisperx_data = load_whisperx_alignment(whisperx)
    identity_map = load_identity_maps(identity)

    document = generate_script_document(
        scenes=scenes_data,
        whisperx_data=whisperx_data,
        identity_map=identity_map,
        include_script_hints=include_script_hints,
    )

    dump_json(out, document)
    txt_path = Path(txt)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(document["script_text"], encoding="utf8")

    print(f"Wrote structured script to {out}")
    print(f"Wrote text script to {txt}")
    print(f"Scenes: {document['metadata']['scene_count']}, characters: {document['metadata']['character_count']}")

if __name__ == "__main__":
    # Example: call with defaults or provide explicit arguments
    main()