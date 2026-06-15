#!/usr/bin/env python3
"""
Extract top entities from a script sequence using BERT-based NER.
No coreference, but very stable and fast.
"""

import json
import sys
from collections import Counter
from typing import List, Dict, Any

import torch
from transformers import pipeline


def load_scenes(input_file=None) -> List[Dict[str, Any]]:
    if input_file:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)
    if isinstance(data, dict) and "scenes" in data:
        return data["scenes"]
    else:
        raise ValueError("JSON must have a 'scenes' key.")


def extract_all_dialogue_text(scenes: List[Dict[str, Any]]) -> str:
    texts = []
    for scene in scenes:
        for dialogue in scene.get("dialogue", []):
            text = dialogue.get("text", "").strip()
            if text:
                texts.append(text)
    return " ".join(texts)


def extract_top_entities_ner(text: str, top_n: int = 5) -> List[str]:
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device==0 else 'CPU'}", flush=True)
    
    # 加载 BERT-based NER 模型（自动缓存，首次下载约 1.1GB）
    ner = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        device=device,
        aggregation_strategy="simple"  # 合并子词
    )
    
    entities = ner(text)
    # 提取实体文本并转为小写
    entity_texts = [ent['word'].lower() for ent in entities if len(ent['word']) > 2]
    
    # 过滤常见停用词
    stop_set = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}
    filtered = [e for e in entity_texts if e not in stop_set]
    
    counter = Counter(filtered)
    return [ent for ent, _ in counter.most_common(top_n)]


def main():
    if len(sys.argv) > 1:
        scenes = load_scenes(sys.argv[1])
    else:
        scenes = load_scenes("script_structured.json")
    
    all_text = extract_all_dialogue_text(scenes)
    if not all_text.strip():
        print("No dialogue text found.", file=sys.stderr)
        sys.exit(1)
    
    top = extract_top_entities_ner(all_text, top_n=15)
    print(json.dumps(top, indent=2))


if __name__ == "__main__":
    main()