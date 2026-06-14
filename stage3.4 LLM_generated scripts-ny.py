"""Helpers to finalize identity mappings and prepare LLM payloads.

Usage examples:
  python "LLM_generated scripts.py" --prepare-llm --scenes scene_level_results.json --whisperx whisperx_results.json --identity identity_reconciliation.json --out llm_payload.json

The script is intentionally conservative: it does not guess character ids automatically, but
it generates a CSV template and can apply a filled mapping back into the JSON used by your
pipeline. It also can run the stage3 extractor (using the same Python interpreter) to
regenerate `identity_reconciliation.json` after mappings are applied.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Any, Optional
import builtins
from openai import OpenAI
from typing import Optional


def load_json(path: str) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def save_json(obj: Any, path: str) -> None:
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, ensure_ascii=False, indent=2)

def extract_whisperx_sentences(whisperx_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从 WhisperX 输出中提取句子/片段级别的文本（text），支持：
      1. 顶层 "result" 列表（包含 speaker、character_cluster）
      2. 顶层 "segments" 列表（标准 WhisperX 输出）
      3. 向后兼容旧结构 "transcription_aligned.segments"
    返回每个句子包含 start, end, text, speaker(可选), character_cluster(可选)
    """
    items: List[Dict[str, Any]] = []
    seen = set()

    def add_item(start: Any, end: Any, text: str, speaker: Any = None, character_cluster: Any = None) -> None:
        if start is None or end is None or not text:
            return
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
            character_cluster = seg.get("character_cluster") or seg.get("character")
            add_item(start, end, text, speaker, character_cluster)
        if items:
            items.sort(key=lambda x: (x["start"], x["end"]))
            return items

    # 2) 处理顶层 "segments"（标准 WhisperX）
    top_segments = whisperx_data.get("segments")
    if isinstance(top_segments, list):
        for seg in top_segments:
            if not isinstance(seg, dict):
                continue
            start = seg.get("start")
            end = seg.get("end")
            text = normalize_whitespace(str(seg.get("text", "")))
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


def normalize_whitespace(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text or "").strip()


def canonical_character_name(raw_speaker: str, identity: Dict[str, Any]) -> str:
    """
    根据 identity_reconciliation.json 的内容，将原始说话人标签映射为规范角色名。
    identity 结构示例:
    {
      "speaker_to_character": {"0": "KAT", "1": "BIANCA", ...},
      "unified_characters": [
        {"character_id": "KAT", "display_name": "Kat Stratford", "aliases": ["KAT", "Kat", "Katherine"]},
        ...
      ]
    }
    """
    if not raw_speaker or raw_speaker == "UNKNOWN":
        return "UNKNOWN"
    
    raw = normalize_whitespace(str(raw_speaker))
    
    # 1) 优先使用 speaker_to_character 直接映射
    speaker_to_char = identity.get("speaker_to_character", {})
    if raw in speaker_to_char:
        return str(speaker_to_char[raw])
    
    # 2) 检查是否在 unified_characters 的别名中
    unified = identity.get("unified_characters", [])
    for char in unified:
        display = str(char.get("display_name", ""))
        char_id = str(char.get("character_id", ""))
        aliases = char.get("aliases", [])
        # 把 display_name 也视为别名
        check_names = [display] + [str(a) for a in aliases]
        if raw in check_names:
            return char_id if char_id else display
        # 不区分大小写匹配（可选）
        raw_lower = raw.lower()
        for name in check_names:
            if raw_lower == name.lower():
                return char_id if char_id else display
    
    # 3) 如果都找不到，保留原始标签
    return raw


def group_sentences_into_turns(sentences: List[Dict[str, Any]], max_gap: float = 0.75) -> List[Dict[str, Any]]:
    """将连续的句子按说话人聚合成对话轮次，gap > max_gap 时切分"""
    if not sentences:
        return []
    sentences = sorted(sentences, key=lambda x: (x["start"], x["end"]))
    turns: List[Dict[str, Any]] = []
    current = {
        "speaker": sentences[0].get("speaker"),
        "start": sentences[0]["start"],
        "end": sentences[0]["end"],
        "text": sentences[0]["text"],
    }
    for s in sentences[1:]:
        same_speaker = s.get("speaker") == current["speaker"]
        close_enough = s["start"] - current["end"] <= max_gap
        if same_speaker and close_enough:
            current["end"] = max(current["end"], s["end"])
            current["text"] += " " + s["text"]
        else:
            turns.append(current)
            current = {
                "speaker": s.get("speaker"),
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
            }
    turns.append(current)
    # 规范化文本空格
    for turn in turns:
        turn["text"] = normalize_whitespace(turn["text"])
    return turns


def prepare_llm_payload(scenes_path: str, whisperx_path: str, identity_path: Optional[str], out_path: str, selected_only: bool = False) -> None:
    scenes = load_json(scenes_path)
    whisperx = load_json(whisperx_path)
    identity = load_json(identity_path) if identity_path and os.path.exists(identity_path) else {}

    # 提取句子级别的对话（优先使用 result 结构）
    sentences = extract_whisperx_sentences(whisperx)

    # 加载场景列表并支持 selected_only 过滤
    if isinstance(scenes, dict):
        if scenes.get("scenes") and isinstance(scenes.get("scenes"), list):
            scenes_items = scenes.get("scenes")
        else:
            found = None
            for v in scenes.values():
                if isinstance(v, list):
                    found = v
                    break
            scenes_items = found or []
    elif isinstance(scenes, list):
        scenes_items = scenes
    else:
        scenes_items = []

    # 按时间排序场景
    scenes_items = sorted(scenes_items, key=lambda s: float(s.get("start", s.get("start_time", 0))))

    if selected_only:
        scenes_items = [s for s in scenes_items if s.get("selection_tags")]

    # 构建场景起始时间列表用于二分查找
    scene_starts = [float(s.get("start", s.get("start_time", 0))) for s in scenes_items]

    import bisect
    def scene_index_for_time(time_s: float) -> int:
        idx = bisect.bisect_right(scene_starts, time_s) - 1
        return max(0, min(idx, len(scene_starts)-1))

    # 将句子分配到场景
    scene_sentences: List[List[Dict[str, Any]]] = [[] for _ in scenes_items]
    for sent in sentences:
        mid = (sent["start"] + sent["end"]) / 2.0
        idx = scene_index_for_time(mid)
        scene = scenes_items[idx]
        # 确保句子完全在场景时间窗内（允许轻微误差）
        if sent["end"] < float(scene.get("start", 0)) or sent["start"] > float(scene.get("end", 0)):
            continue
        scene_sentences[idx].append(sent)

    # 构建每个场景的 payload
    scenes_list: List[Dict[str, Any]] = []
    for idx, scene in enumerate(scenes_items):
        s_start = float(scene.get("start", scene.get("start_time", 0)))
        s_end = float(scene.get("end", scene.get("end_time", s_start)))
        scene_id = scene.get("scene_id") or scene.get("id") or scene.get("name") or f"scene_{idx+1}"
        vlm_summary = scene.get("summary") or scene.get("vlm_summary") or scene.get("description") or scene.get("text") or ""

        # 将场景内的句子聚合成对话轮次
        raw_turns = group_sentences_into_turns(scene_sentences[idx])

        # 应用身份映射规范说话人
        turns = []
        for turn in raw_turns:
            raw_speaker = turn.get("speaker")
            if raw_speaker:
                canon = canonical_character_name(raw_speaker, identity)
                speaker = canon if canon else raw_speaker
            else:
                # 如果场景中只有一个角色，可以默认使用该角色（可选逻辑）
                speaker = "UNKNOWN"
            turns.append({
                "speaker": speaker,
                "start": turn["start"],
                "end": turn["end"],
                "text": turn["text"]
            })

        scene_payload = {
            "scene_id": scene_id,
            "start": s_start,
            "end": s_end,
            "vlm_summary": vlm_summary,
            "turns": turns,
            "raw_scene_record": scene,
        }
        scenes_list.append(scene_payload)

    payload = {"scenes": scenes_list, "identity": identity}
    save_json(payload, out_path)
    print(f"Wrote LLM payload to {out_path} ({len(scenes_list)} scenes)")
    
def generate_script_scene_by_scene(
    payload_path: str,
    output_path: str,
    api_key: str,
    model: str = "deepseek-v4-pro",
    temperature: float = 0.7,
    max_tokens_per_scene: int = 2000,
    system_prompt: Optional[str] = None
) -> str:
    """
    逐场景调用 LLM 生成剧本（每个场景一次 API 请求）。
    
    Args:
        payload_path: llm_payload.json 文件路径（由 prepare_llm_payload 生成）
        output_path: 最终剧本保存路径
        api_key: DeepSeek API 密钥
        model: 模型名称，默认 "deepseek-chat"
        temperature: 生成温度
        max_tokens_per_scene: 每个场景允许的最大输出 token 数
        system_prompt: 自定义系统提示（若为 None 则使用默认）
    
    Returns:
        完整剧本文本
    """
    # 读取 payload
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    
    scenes = payload.get("scenes", [])
    identity = payload.get("identity", {})
    
    if not scenes:
        print("警告: payload 中没有 scenes 字段或为空")
        return ""
    
    # 默认系统提示（只针对单个场景）
    if system_prompt is None:
        system_prompt = (
            "You are a professional screenwriter. Given a single scene's visual summary "
            "and dialogue turns (with speaker labels), generate the script for that scene only.\n\n"
            "Format requirements:\n"
            "SCENE X | STARTs - ENDs\n"
            "[Scene description: expand the visual summary with environmental and action details]\n"
            "CHARACTER: dialogue line\n"
            "(action brackets) where appropriate\n\n"
            "Do NOT invent new dialogue that is not present in the input turns. "
            "You may slightly polish the wording for readability, but keep the meaning and key phrases. "
            "Use the canonical character names provided in the speaker field after mapping."
        )
    
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    full_script_parts = []
    
    for idx, scene in enumerate(scenes, start=1):
        scene_id = scene.get("scene_id", f"scene_{idx}")
        start = scene.get("start", 0.0)
        end = scene.get("end", 0.0)
        
        # 处理 vlm_summary（可能是字符串或字典）
        vlm_summary_raw = scene.get("vlm_summary", "")
        if isinstance(vlm_summary_raw, dict):
            summary_text = vlm_summary_raw.get("summary", "")
            if not summary_text:
                summary_text = str(vlm_summary_raw)
        else:
            summary_text = str(vlm_summary_raw)
        
        # 处理对话轮次：规范化说话人名称
        turns = scene.get("turns", [])
        processed_turns = []
        for turn in turns:
            raw_speaker = turn.get("speaker", "UNKNOWN")
            # 利用 identity 映射规范角色名
            mapped_speaker = canonical_character_name(raw_speaker, identity)
            # 如果 still UNKNOWN 且 raw_scene_record 里有 unified_character_ids，可以尝试取第一个
            if mapped_speaker == "UNKNOWN":
                raw_record = scene.get("raw_scene_record", {})
                unified_ids = raw_record.get("unified_character_ids", [])
                if unified_ids and len(unified_ids) > 0:
                    mapped_speaker = str(unified_ids[0])
            text = turn.get("text", "")
            processed_turns.append((mapped_speaker, text))
        
        # 构建该场景的用户输入
        turns_text_lines = [f"{speaker}: {text}" for speaker, text in processed_turns if text.strip()]
        turns_block = "\n".join(turns_text_lines) if turns_text_lines else "(no dialogue)"
        
        user_content = f"""Scene ID: {scene_id}
Time range: {start:.1f}s - {end:.1f}s
Visual summary: {summary_text}

Dialogue turns (speaker → canonical name):
{turns_block}

Generate the script for this single scene. Start with the scene header "SCENE {idx} | {start:.1f}s - {end:.1f}s", then a descriptive paragraph, then the dialogue lines. Do not include any scenes outside this one."""
        
        # 调用 API
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=max_tokens_per_scene
            )
            scene_script = response.choices[0].message.content.strip()
            full_script_parts.append(scene_script)
            print(f"✓ 已生成场景 {idx}/{len(scenes)}: {scene_id}")
        except Exception as e:
            print(f"✗ 场景 {idx} ({scene_id}) 生成失败: {e}")
            full_script_parts.append(f"SCENE {idx} | {start:.1f}s - {end:.1f}s\n[生成失败: {e}]")
    
    # 合并所有场景
    full_script = "\n\n".join(full_script_parts)
    
    # 保存到文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_script)
    
    print(f"\n✅ 完整剧本已保存至: {output_path}")
    return full_script

if __name__ == "__main__":
    import os
    API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    prepare_llm_payload(
		scenes_path="scene_level_results.json",
		whisperx_path="whisperx_results.json",
		identity_path="identity_reconciliation.json",
		out_path="llm_payload.json",
		selected_only=False
	)

    generate_script_scene_by_scene(
        payload_path="llm_payload.json",
        output_path="deepseek_script.txt",
        api_key=API_KEY,
        model="deepseek-v4-pro",
        temperature=0.7
	)
